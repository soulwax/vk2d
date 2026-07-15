//! Offscreen render targets + a Nearest upscale blit. The world is drawn into a
//! fixed logical-resolution target, then blitted to the swapchain so pixel art
//! stays crisp at any window size. The target size is a parameter — there is no
//! baked-in game resolution.

use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Color,
    ColorTargetState, ColorWrites, CommandEncoder, Device, Extent3d, FilterMode, FragmentState,
    LoadOp, MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, RenderPass, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};

/// The offscreen scene texture is stored linear (non-sRGB) so post-process math
/// sees raw values; the blit target is whatever the surface uses.
pub(crate) const SCENE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// Fullscreen-triangle blit shader; positions/uvs are generated from the vertex
/// index (no vertex buffer). Generic — nothing game-specific.
const BLIT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    out.position = vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(scene_texture, scene_sampler, in.uv);
}
"#;

/// One offscreen render target and the pipeline that upscales it to a swapchain
/// frame with Nearest filtering.
pub(crate) struct SceneTarget {
    texture: Texture,
    view: TextureView,
    #[allow(dead_code)]
    sampler: Sampler,
    #[allow(dead_code)]
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
    blit_pipeline: RenderPipeline,
    /// The target's pixel size — needed to source it as a sprite (UV math
    /// against its own dimensions, same as any other texture).
    width: u32,
    height: u32,
}

impl SceneTarget {
    /// Create a `width`x`height` offscreen target whose blit pipeline outputs
    /// to `surface_format`, sampled with `filter` (`Nearest` keeps pixel art
    /// crisp at native resolution; `Linear` is needed when this target is
    /// supersampled and will be downsampled back down — a `Nearest`
    /// downsample of a supersampled buffer drops every other texel row/
    /// column, visible as scanline stripes; `Linear` lets the hardware
    /// average texels during the shrink instead).
    pub(crate) fn new(
        device: &Device,
        width: u32,
        height: u32,
        surface_format: TextureFormat,
        filter: FilterMode,
    ) -> Self {
        let (texture, view) = create_scene_texture(device, width, height);
        let sampler = create_sampler(device, filter);
        let bind_group_layout = create_bind_group_layout(device);
        let bind_group = create_bind_group(device, &bind_group_layout, &view, &sampler);
        let blit_pipeline = create_blit_pipeline(device, surface_format, &bind_group_layout);
        Self {
            texture,
            view,
            sampler,
            bind_group_layout,
            bind_group,
            blit_pipeline,
            width: width.max(1),
            height: height.max(1),
        }
    }

    /// The target's pixel size (`width.max(1), height.max(1)` — matches the
    /// texture actually allocated).
    pub(crate) fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Begin the scene pass: clears the target to `clear` and returns a render
    /// pass to draw the world into. The pass ends when dropped.
    pub(crate) fn begin_scene_pass<'pass>(
        &'pass self,
        encoder: &'pass mut CommandEncoder,
        clear: Color,
    ) -> RenderPass<'pass> {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("vk2d.scene.pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(clear),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        })
    }

    /// The target's rendered color view — used to read a finished target back
    /// as a material texture input (e.g. binding a scene or bloom pass into a
    /// composite material). Sampleable because the scene texture already
    /// carries `TEXTURE_BINDING` usage alongside `RENDER_ATTACHMENT`.
    pub(crate) fn color_view(&self) -> &TextureView {
        &self.view
    }

    /// The target's underlying color texture — used by the doc-hidden pixel
    /// readback helper on [`crate::Context`] so integration tests can assert a
    /// finished target's actual texels. Not part of the drawing path.
    #[doc(hidden)]
    pub(crate) fn color_texture(&self) -> &Texture {
        &self.texture
    }

    /// The Nearest sampler paired with this target's color texture. Reused so
    /// binding a target as a material input does not need a fresh sampler per
    /// bind; `TextureView`/`Sampler` are cheap reference-counted handles, so
    /// cloning does not duplicate GPU memory.
    pub(crate) fn color_sampler(&self) -> &Sampler {
        &self.sampler
    }

    /// Blit the finished target onto `surface_view` with the Nearest sampler,
    /// upscaling the logical image to the window.
    pub(crate) fn blit_to(&self, encoder: &mut CommandEncoder, surface_view: &TextureView) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("vk2d.blit.pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: surface_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn create_scene_texture(device: &Device, width: u32, height: u32) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("vk2d.scene.texture"),
        size: Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SCENE_FORMAT,
        // COPY_SRC lets the doc-hidden pixel-readback helper on `Context`
        // (used by integration tests to assert a finished target's texels)
        // copy the target texture into a mappable buffer. It adds no runtime
        // cost to the draw path and keeps the readback pointed at the real
        // scene texture rather than a test-only copy.
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn create_sampler(device: &Device, filter: FilterMode) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some("vk2d.scene.sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: filter,
        min_filter: filter,
        ..Default::default()
    })
}

fn create_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vk2d.scene.bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    // `filterable: true` so this layout accepts either a
                    // Nearest or a Linear sampler bound against it — Linear
                    // is required when the target is supersampled and needs
                    // a smooth downsample (see `SceneTarget::new` doc
                    // comment); `Filtering` samplers are a superset that
                    // also accept Nearest, so this is not a behavior change
                    // for existing Nearest-filtered targets.
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    view: &TextureView,
    sampler: &Sampler,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vk2d.scene.bind_group"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_blit_pipeline(
    device: &Device,
    surface_format: TextureFormat,
    bind_group_layout: &BindGroupLayout,
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vk2d.blit.shader"),
        source: ShaderSource::Wgsl(BLIT_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vk2d.blit.pipeline_layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("vk2d.blit.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
