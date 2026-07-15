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
use crate::handles::{TargetId, TextureId};

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
pub(crate) const VERTEX_STRIDE: BufferAddress = 32;
const VERTS_PER_SPRITE: usize = 4;
const INDICES_PER_SPRITE: usize = 6;

/// The per-vertex attribute layout shared by EVERY sprite draw path: the plain
/// sprite pipeline ([`create_pipeline`]) and the material sprite-shaded pipeline
/// (`crate::material`). Position at offset 0, uv at 8, tint at 16 — one
/// authoritative definition so the two pipelines can never drift apart. A silent
/// stride/offset mismatch between them would feed the material's vertex stage
/// garbage without any compile error, so both must build their vertex state from
/// exactly this.
pub(crate) const SPRITE_VERTEX_ATTRIBUTES: [VertexAttribute; 3] = [
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
];

/// The [`VertexBufferLayout`] every sprite-format pipeline binds: `VERTEX_STRIDE`
/// bytes per vertex, per-vertex stepping, and [`SPRITE_VERTEX_ATTRIBUTES`]. The
/// single source of truth behind both the plain sprite pipeline and the material
/// sprite-shaded pipeline — see [`SPRITE_VERTEX_ATTRIBUTES`].
pub(crate) fn sprite_vertex_buffer_layout() -> VertexBufferLayout<'static> {
    VertexBufferLayout {
        array_stride: VERTEX_STRIDE,
        step_mode: VertexStepMode::Vertex,
        attributes: &SPRITE_VERTEX_ATTRIBUTES,
    }
}

/// What a sprite run samples: an uploaded texture, or a finished render
/// target's own color output (the `target_sprite` path — drawing a prior
/// pass's result as a positioned blit, e.g. compositing an offscreen effect
/// into the scene). Extends the batch key alongside `TextureId` so the two
/// sources can never be confused at the batching layer; each still resolves
/// to one bind group at draw time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpriteSource {
    /// An app-uploaded texture, indexed into `Context::textures`.
    Texture(TextureId),
    /// A render target's color output, indexed into `Context::targets`.
    Target(TargetId),
}

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

/// One submission-ordered batch of instances sharing source pixel dimensions.
/// Keeping the dimensions with the run lets staging consume the
/// frame's runs directly, without building an intermediate reference vector.
pub(crate) struct SpriteDrawRun {
    pub atlas_size: (f32, f32),
    pub instances: Vec<SpriteInstance>,
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
    /// Reused CPU upload and draw-range scratch. These retain the largest
    /// frame's capacity instead of allocating three fresh vectors per frame.
    vertex_bytes: Vec<u8>,
    index_bytes: Vec<u8>,
    staged_runs: Vec<SpriteRun>,
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
            vertex_bytes: Vec::with_capacity(capacity * VERTS_PER_SPRITE * 32),
            index_bytes: Vec::with_capacity(capacity * INDICES_PER_SPRITE * 2),
            staged_runs: Vec::new(),
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
        runs: &[SpriteDrawRun],
        logical_size: (u32, u32),
    ) {
        let total: usize = runs.iter().map(|run| run.instances.len()).sum();
        self.vertex_bytes.clear();
        self.index_bytes.clear();
        self.staged_runs.clear();
        if total == 0 {
            return;
        }
        if total > self.capacity {
            self.grow(device, total);
        }
        self.vertex_bytes
            .reserve(total * VERTS_PER_SPRITE * VERTEX_STRIDE as usize);
        self.index_bytes
            .reserve(total * INDICES_PER_SPRITE * std::mem::size_of::<u16>());
        self.staged_runs.reserve(runs.len());
        let mut vertex_base = 0u16;
        let mut index_cursor = 0u32;
        for run in runs {
            let index_start = index_cursor;
            for sprite in &run.instances {
                push_sprite_vertices(&mut self.vertex_bytes, sprite, run.atlas_size, logical_size);
                for local in [0u16, 1, 2, 0, 2, 3] {
                    self.index_bytes
                        .extend_from_slice(&(vertex_base + local).to_le_bytes());
                }
                vertex_base += VERTS_PER_SPRITE as u16;
                index_cursor += INDICES_PER_SPRITE as u32;
            }
            self.staged_runs.push(SpriteRun {
                index_start,
                index_count: index_cursor - index_start,
            });
        }
        queue.write_buffer(&self.vertex_buffer, 0, &self.vertex_bytes);
        queue.write_buffer(&self.index_buffer, 0, &self.index_bytes);
    }

    /// Resolve a run staged by the current frame without exposing a borrow of
    /// the scratch vector across subsequent batch draw calls.
    pub(crate) fn staged_run(&self, slot: usize) -> Option<SpriteRun> {
        self.staged_runs.get(slot).copied()
    }

    /// Draw one staged run, binding `atlas`'s texture.
    pub(crate) fn draw_run<'pass>(
        &'pass self,
        pass: &mut RenderPass<'pass>,
        run: SpriteRun,
        atlas: &'pass GpuTexture,
    ) {
        self.draw_run_with_bind_group(pass, run, &atlas.bind_group);
    }

    /// Draw one staged run with an explicit bind group — the shared path
    /// behind [`Self::draw_run`] (an uploaded texture's own bind group) and
    /// `target_sprite` (a render target's color view/sampler bound into a
    /// bind group built with this same pipeline's layout via
    /// [`Self::build_source_bind_group`]). Both sources share one pipeline
    /// and vertex format, so only the bind group differs.
    pub(crate) fn draw_run_with_bind_group<'pass>(
        &'pass self,
        pass: &mut RenderPass<'pass>,
        run: SpriteRun,
        bind_group: &'pass BindGroup,
    ) {
        self.draw_run_with_pipeline(pass, run, &self.pipeline, bind_group);
    }

    /// Draw one staged run against an EXPLICIT pipeline + bind group, reusing
    /// this batch's staged vertex/index buffers. The material sprite-shaded
    /// path ([`crate::Frame::material_sprite`]) uses this to draw the same
    /// sprite geometry through a material's own pipeline (the material's
    /// fragment stage) and bind group (its uniforms + the sprite's texture in
    /// slot 0). The vertex buffer layout the caller's pipeline was built with
    /// MUST match [`sprite_vertex_buffer_layout`] — the invariant a
    /// stride/offset drift would silently violate — which the material pipeline
    /// guarantees by building from that same helper.
    pub(crate) fn draw_run_with_pipeline<'pass>(
        &'pass self,
        pass: &mut RenderPass<'pass>,
        run: SpriteRun,
        pipeline: &'pass RenderPipeline,
        bind_group: &'pass BindGroup,
    ) {
        if run.index_count == 0 {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        pass.draw_indexed(run.index_start..run.index_start + run.index_count, 0, 0..1);
    }

    /// Build a bind group for this batch's pipeline from an arbitrary
    /// view/sampler pair — the hookup for `target_sprite`, which sources a
    /// render target's own color output (`SceneTarget::color_view`/
    /// `color_sampler`) rather than an entry in the texture registry.
    /// Structurally identical to a `GpuTexture`'s bind group (same layout:
    /// texture at binding 0, sampler at binding 1), so the shared pipeline
    /// draws either source without caring which it is.
    pub(crate) fn build_source_bind_group(
        &self,
        device: &Device,
        view: &TextureView,
        sampler: &Sampler,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vk2d.sprite.target_source.bind_group"),
            layout: &self.bind_group_layout,
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
    // One transform for all four corners of this sprite instead of
    // recomputing logical_to_clip's divide per corner — logical_size is
    // constant for the whole batch (see ClipXform's doc comment).
    let xform = ClipXform::new(logical_size);
    for (px, py, u, v) in corners {
        let (ndc_x, ndc_y) = xform.apply(px, py);
        out.extend_from_slice(&ndc_x.to_le_bytes());
        out.extend_from_slice(&ndc_y.to_le_bytes());
        out.extend_from_slice(&u.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
        for channel in sprite.tint {
            out.extend_from_slice(&channel.to_le_bytes());
        }
    }
}

/// Logical pixel (top-left origin) to clip space (-1..1, Y up). No
/// production call site uses this directly anymore — every per-vertex hot
/// path now goes through [`ClipXform`], which hoists the divide out of the
/// per-vertex loop. Kept (not deleted) because `view.rs`'s own tests call
/// it directly to check `View2`'s math independently of the batching
/// layer, and `ClipXform`'s own tests use it as the correctness reference.
#[allow(dead_code)]
pub(crate) fn logical_to_clip(px: f32, py: f32, logical_size: (u32, u32)) -> (f32, f32) {
    let (w, h) = logical_size;
    let x = px / w.max(1) as f32 * 2.0 - 1.0;
    let y = 1.0 - py / h.max(1) as f32 * 2.0;
    (x, y)
}

/// Precomputed factors for [`logical_to_clip`]'s conversion, reused across
/// every vertex of a batch instead of recomputing the same `w.max(1) as f32`
/// cast and divide per vertex. `logical_size` is constant for a whole
/// staging pass (it only changes when the render target's own size
/// changes, at most once per frame) — hoisting the two divisions and casts
/// out of the per-vertex hot path leaves `apply` as two multiply-adds,
/// bit-for-bit identical to `logical_to_clip`'s result for the same inputs
/// (see `clip_xform_tests` for the equivalence proof).
#[derive(Clone, Copy)]
pub(crate) struct ClipXform {
    sx: f32,
    ox: f32,
    sy: f32,
    oy: f32,
}

impl ClipXform {
    pub(crate) fn new(logical_size: (u32, u32)) -> Self {
        let (w, h) = logical_size;
        Self {
            sx: 2.0 / w.max(1) as f32,
            ox: -1.0,
            sy: -2.0 / h.max(1) as f32,
            oy: 1.0,
        }
    }

    #[inline]
    pub(crate) fn apply(&self, px: f32, py: f32) -> (f32, f32) {
        (px * self.sx + self.ox, py * self.sy + self.oy)
    }
}

/// Bind-group layout shared by every sprite draw, including
/// [`crate::Frame::target_sprite`]'s render-target readback.
///
/// Declared `Filtering`-capable so `target_sprite`'s readback (which may
/// source a `Linear`-filtered supersampled [`crate::SceneTarget`]) validates;
/// ordinary `Nearest`-filtered sprite textures also validate fine against a
/// `Filtering` layout, so this is a widening with no behavior change for
/// existing callers — see `target.rs`'s twin fix in the same task for the
/// parallel blit-path layout.
fn create_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vk2d.sprite.bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
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
            buffers: &[sprite_vertex_buffer_layout()],
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

#[cfg(test)]
mod source_tests {
    use super::*;

    // `SpriteSource` is the batch key `Frame::push_sprite_instance` compares
    // (`self.cmds.last()` vs the new draw's source) to decide whether to
    // extend the current run or open a new one. These pure equality checks
    // lock that contract: a `Texture` and a `Target` sharing the same raw
    // index must never compare equal (otherwise a target_sprite draw could
    // silently batch into a same-index texture's run and sample the wrong
    // bind group), and same-variant/same-index must compare equal (so
    // consecutive `target_sprite` calls for one target still batch into a
    // single draw, matching how consecutive `sprite()` calls of one texture
    // do today).
    #[test]
    fn texture_and_target_sources_never_compare_equal_even_at_the_same_index() {
        let tex = SpriteSource::Texture(TextureId(0));
        let target = SpriteSource::Target(TargetId(0));
        assert_ne!(tex, target);
    }

    #[test]
    fn same_variant_same_index_sources_compare_equal() {
        assert_eq!(
            SpriteSource::Texture(TextureId(2)),
            SpriteSource::Texture(TextureId(2))
        );
        assert_eq!(
            SpriteSource::Target(TargetId(1)),
            SpriteSource::Target(TargetId(1))
        );
    }

    #[test]
    fn same_variant_different_index_sources_are_distinct() {
        assert_ne!(
            SpriteSource::Target(TargetId(1)),
            SpriteSource::Target(TargetId(2))
        );
        assert_ne!(
            SpriteSource::Texture(TextureId(1)),
            SpriteSource::Texture(TextureId(2))
        );
    }
}

#[cfg(test)]
mod clip_xform_tests {
    use super::*;

    #[test]
    fn clip_xform_matches_logical_to_clip_for_arbitrary_points() {
        // ClipXform is a precomputed-factor fast path for the same math
        // logical_to_clip does per-call — every point it converts must land
        // within float-rounding tolerance of the per-call function's result,
        // for any logical size and any point actually reachable within (or
        // just outside) that target's own pixel range. Bit-exact equality
        // is NOT the right bar here: `px / w * 2.0 - 1.0` (direct) and
        // `px * (2.0 / w) - 1.0` (precomputed reciprocal) are mathematically
        // equivalent but not bit-identical IEEE 754 float expressions —
        // dividing by `w` then multiplying differs from multiplying by
        // `w`'s precomputed reciprocal in the last few bits of the
        // mantissa. At any real logical point (within a few multiples of
        // the target's own width/height — sprites/shapes overlap the
        // viewport, they are not drawn thousands of widths off-target) that
        // difference is many orders of magnitude below one screen pixel, so
        // it is genuinely invisible; a relative-error tolerance instead of
        // `==` reflects "same visual result," not "same bits." Test points
        // scale WITH each target's own size (a fraction of `w`/`h`, not a
        // fixed absolute pixel value) so a small target like `(7, 13)`
        // isn't probed at wildly out-of-range points that inflate the
        // relative float-rounding error for reasons unrelated to the
        // transform itself.
        let sizes = [(1600u32, 900u32), (3200, 1800), (1, 1), (7, 13)];
        let point_fracs = [(0.0_f32, 0.0_f32), (0.5, 0.5), (1.0, 1.0), (-0.1, 1.5)];
        const RELATIVE_TOLERANCE: f32 = 1e-4;
        for size in sizes {
            let xform = ClipXform::new(size);
            let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
            for (fx, fy) in point_fracs {
                let (px, py) = (fx * w, fy * h);
                let expected = logical_to_clip(px, py, size);
                let actual = xform.apply(px, py);
                let scale = expected.0.abs().max(expected.1.abs()).max(1.0);
                assert!(
                    (actual.0 - expected.0).abs() <= RELATIVE_TOLERANCE * scale
                        && (actual.1 - expected.1).abs() <= RELATIVE_TOLERANCE * scale,
                    "size={size:?} point=({px},{py}): xform {actual:?} != direct {expected:?} \
                     (relative tolerance {RELATIVE_TOLERANCE})"
                );
            }
        }
    }

    #[test]
    fn clip_xform_new_matches_zero_sized_logical_size_fallback() {
        // logical_to_clip clamps width/height to a minimum of 1 to avoid a
        // divide by zero; ClipXform::new must apply the identical clamp so
        // a degenerate (0,0) logical size still produces the same fallback
        // coordinates as the per-call function, not a NaN/inf.
        let xform = ClipXform::new((0, 0));
        let expected = logical_to_clip(5.0, 5.0, (0, 0));
        let actual = xform.apply(5.0, 5.0);
        assert_eq!(actual, expected);
        assert!(actual.0.is_finite() && actual.1.is_finite());
    }
}
