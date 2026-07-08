//! Sprite batching: many textured quads in one draw call. Game-agnostic — the
//! app supplies decoded RGBA bytes and arbitrary source rects; there is no
//! atlas file loading or fixed frame size baked in.

// Staged: the batch's upload/draw and the SpriteInstance producer live in the
// Frame API (Task 9); until then only `load_texture_rgba` + the pipeline are
// wired. Scoped allow keeps the crate green under `-D warnings`.

use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferAddress, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device, Extent3d,
    FilterMode, FragmentState, FrontFace, IndexFormat, MultisampleState,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

use crate::blend::BlendMode;

/// Texture sampling filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// Nearest-neighbour: crisp pixels, no smoothing (pixel-art default).
    Nearest,
    /// Bilinear: smooth interpolation between texels.
    Linear,
}

impl Filter {
    fn wgpu(self) -> FilterMode {
        match self {
            Filter::Nearest => FilterMode::Nearest,
            Filter::Linear => FilterMode::Linear,
        }
    }
}

/// Bytes per vertex: position (2×f32) + uv (2×f32) + tint (4×f32).
const VERTEX_STRIDE: BufferAddress = 32;
const VERTS_PER_SPRITE: usize = 4;
const INDICES_PER_SPRITE: usize = 6;

/// One sprite to draw, in logical-pixel space (origin top-left). Produced by the
/// frame API from a `TextureId` + `SpriteParams`.
#[derive(Clone, Copy)]
pub(crate) struct SpriteInstance {
    /// Center of the sprite in logical pixels.
    pub center: [f32; 2],
    /// On-screen size in logical pixels.
    pub size: [f32; 2],
    /// Source rect in the texture, in pixels: `[x, y, w, h]`.
    pub source_px: [f32; 4],
    /// RGBA tint multiplier.
    pub tint: [f32; 4],
}

const SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(input.position, 0.0, 1.0);
    out.uv = input.uv;
    out.tint = input.tint;
    return out;
}
@group(0) @binding(0) var atlas_texture: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(atlas_texture, atlas_sampler, in.uv) * in.tint;
}
"#;

/// A GPU texture in the registry: its bind group (for the sprite pipeline),
/// pixel dimensions (for uv computation), and the raw view + sampler so other
/// consumers (materials, see `crate::material`) can rebind it into their own
/// bind-group layouts without recreating the texture.
pub(crate) struct GpuTexture {
    pub bind_group: BindGroup,
    pub width: f32,
    pub height: f32,
    /// The texture's view, reused by material texture-slot binding.
    pub view: TextureView,
    /// The texture's sampler, reused by material texture-slot binding.
    pub sampler: Sampler,
}

/// The shared sprite pipeline + per-frame geometry buffers. One instance drives
/// all sprite draws; each draw binds a texture's `GpuTexture::bind_group`.
pub(crate) struct SpriteBatch {
    pipeline: RenderPipeline,
    pub(crate) bind_group_layout: BindGroupLayout,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    /// Capacity in sprites; buffers grow when a frame needs more.
    capacity: usize,
}

/// A contiguous slice of the uploaded index buffer belonging to one texture,
/// produced by [`SpriteBatch::stage`] and consumed by [`SpriteBatch::draw_run`].
#[derive(Clone, Copy)]
pub(crate) struct SpriteRun {
    pub index_start: u32,
    pub index_count: u32,
}

impl SpriteBatch {
    pub(crate) fn new(device: &Device, target_format: TextureFormat) -> Self {
        let bind_group_layout = create_bind_group_layout(device);
        let pipeline = create_pipeline(device, target_format, &bind_group_layout);
        let capacity = 256;
        let vertex_buffer = create_empty_buffer(
            device,
            "vk2d.sprite.vertices",
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            (capacity * VERTS_PER_SPRITE) as BufferAddress * VERTEX_STRIDE,
        );
        let index_buffer = create_empty_buffer(
            device,
            "vk2d.sprite.indices",
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            (capacity * INDICES_PER_SPRITE * std::mem::size_of::<u16>()) as BufferAddress,
        );
        Self {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            index_buffer,
            capacity,
        }
    }

    /// Pack all `runs` (each a slice of instances sharing one atlas's
    /// dimensions) into ONE vertex/index buffer, uploaded once, and return a
    /// per-run index slice for drawing. Ordering is preserved, so drawing the
    /// runs in order layers correctly. Grows the buffers on demand.
    pub(crate) fn stage(
        &mut self,
        device: &Device,
        queue: &Queue,
        runs: &[(&[SpriteInstance], (f32, f32))],
        logical_size: (u32, u32),
    ) -> Vec<SpriteRun> {
        let total: usize = runs.iter().map(|(s, _)| s.len()).sum();
        if total == 0 {
            return Vec::new();
        }
        if total > self.capacity {
            self.grow(device, total);
        }
        let mut vertices: Vec<u8> = Vec::with_capacity(total * VERTS_PER_SPRITE * 32);
        let mut indices: Vec<u8> = Vec::with_capacity(total * INDICES_PER_SPRITE * 2);
        let mut slices = Vec::with_capacity(runs.len());
        let mut vertex_base = 0u16;
        let mut index_cursor = 0u32;
        for (sprites, atlas_wh) in runs {
            let index_start = index_cursor;
            for sprite in sprites.iter() {
                push_sprite_vertices(&mut vertices, sprite, *atlas_wh, logical_size);
                for local in [0u16, 1, 2, 0, 2, 3] {
                    indices.extend_from_slice(&(vertex_base + local).to_le_bytes());
                }
                vertex_base += VERTS_PER_SPRITE as u16;
                index_cursor += INDICES_PER_SPRITE as u32;
            }
            slices.push(SpriteRun {
                index_start,
                index_count: index_cursor - index_start,
            });
        }
        queue.write_buffer(&self.vertex_buffer, 0, &vertices);
        queue.write_buffer(&self.index_buffer, 0, &indices);
        slices
    }

    /// Draw one staged run, binding `atlas`'s texture.
    pub(crate) fn draw_run<'pass>(
        &'pass self,
        pass: &mut RenderPass<'pass>,
        run: SpriteRun,
        atlas: &'pass GpuTexture,
    ) {
        if run.index_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &atlas.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        pass.draw_indexed(run.index_start..run.index_start + run.index_count, 0, 0..1);
    }

    fn grow(&mut self, device: &Device, needed: usize) {
        let mut capacity = self.capacity.max(1);
        while capacity < needed {
            capacity *= 2;
        }
        self.vertex_buffer = create_empty_buffer(
            device,
            "vk2d.sprite.vertices",
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            (capacity * VERTS_PER_SPRITE) as BufferAddress * VERTEX_STRIDE,
        );
        self.index_buffer = create_empty_buffer(
            device,
            "vk2d.sprite.indices",
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            (capacity * INDICES_PER_SPRITE * std::mem::size_of::<u16>()) as BufferAddress,
        );
        self.capacity = capacity;
    }
}

/// Upload decoded RGBA bytes as a texture and build its bind group. The app has
/// already decoded the image; the library never loads files.
pub(crate) fn create_gpu_texture(
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
    bytes: &[u8],
    width: u32,
    height: u32,
    filter: Filter,
) -> GpuTexture {
    let size = Extent3d {
        width: width.max(1),
        height: height.max(1),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("vk2d.texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Default::default(),
            aspect: TextureAspect::All,
        },
        bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        size,
    );
    let view = texture.create_view(&TextureViewDescriptor::default());
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("vk2d.texture.sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: filter.wgpu(),
        min_filter: filter.wgpu(),
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("vk2d.texture.bind_group"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ],
    });
    GpuTexture {
        bind_group,
        width: width as f32,
        height: height as f32,
        view,
        sampler,
    }
}

fn push_sprite_vertices(
    out: &mut Vec<u8>,
    sprite: &SpriteInstance,
    atlas_wh: (f32, f32),
    logical_size: (u32, u32),
) {
    let (aw, ah) = atlas_wh;
    let hw = sprite.size[0] * 0.5;
    let hh = sprite.size[1] * 0.5;
    let cx = sprite.center[0];
    let cy = sprite.center[1];

    let u0 = (sprite.source_px[0] / aw).clamp(0.0, 1.0);
    let v0 = (sprite.source_px[1] / ah).clamp(0.0, 1.0);
    let u1 = ((sprite.source_px[0] + sprite.source_px[2]) / aw).clamp(0.0, 1.0);
    let v1 = ((sprite.source_px[1] + sprite.source_px[3]) / ah).clamp(0.0, 1.0);

    let corners = [
        (cx - hw, cy - hh, u0, v0),
        (cx + hw, cy - hh, u1, v0),
        (cx + hw, cy + hh, u1, v1),
        (cx - hw, cy + hh, u0, v1),
    ];
    for (px, py, u, v) in corners {
        let (ndc_x, ndc_y) = logical_to_clip(px, py, logical_size);
        out.extend_from_slice(&ndc_x.to_le_bytes());
        out.extend_from_slice(&ndc_y.to_le_bytes());
        out.extend_from_slice(&u.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
        for channel in sprite.tint {
            out.extend_from_slice(&channel.to_le_bytes());
        }
    }
}

/// Logical pixel (top-left origin) to clip space (-1..1, Y up).
pub(crate) fn logical_to_clip(px: f32, py: f32, logical_size: (u32, u32)) -> (f32, f32) {
    let (w, h) = logical_size;
    let x = px / w.max(1) as f32 * 2.0 - 1.0;
    let y = 1.0 - py / h.max(1) as f32 * 2.0;
    (x, y)
}

fn create_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vk2d.sprite.bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    })
}

fn create_pipeline(
    device: &Device,
    target_format: TextureFormat,
    bind_group_layout: &BindGroupLayout,
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vk2d.sprite.shader"),
        source: ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vk2d.sprite.pipeline_layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("vk2d.sprite.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[VertexBufferLayout {
                array_stride: VERTEX_STRIDE,
                step_mode: VertexStepMode::Vertex,
                attributes: &[
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 8,
                        shader_location: 1,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 2,
                    },
                ],
            }],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: target_format,
                blend: BlendMode::Alpha.blend_state(),
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_empty_buffer(
    device: &Device,
    label: &'static str,
    usage: BufferUsages,
    size: BufferAddress,
) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage,
        mapped_at_creation: false,
    })
}
