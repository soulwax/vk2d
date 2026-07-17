//! Generic WGSL material: one pipeline type for every shader. Shaders are
//! authored as WGSL text and compiled to SPIR-V at load time (via naga); the
//! app pushes uniforms by name. There is no per-shader Rust — an effect is a
//! `.wgsl` file plus a `[(name, UniformType)]` declaration.

use std::collections::HashMap;

use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device,
    Extent3d, FilterMode, FragmentState, FrontFace, MultisampleState, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};

use crate::blend::BlendMode;
use crate::sprite::sprite_vertex_buffer_layout;
use crate::{TargetId, TextureId, Vk2dError};

/// The sprite-shaded vertex stage vk2d appends to a material's WGSL before
/// building its sprite-shaded pipeline (see [`Material::sprite_pipeline`]). It
/// consumes the exact per-vertex layout `sprite.rs` produces
/// ([`crate::sprite::sprite_vertex_buffer_layout`]): position/uv/tint at
/// locations 0/1/2. The position arrives already in clip space (the record path
/// converts logical pixels → clip against the render target's own size); this
/// stage forwards uv/tint to the material's `fs_main` at
/// `@location(0)`/`@location(1)`.
///
/// Clip-space Y note: `sprite.rs`'s plain vertex stage is compiled from WGSL
/// (`ShaderSource::Wgsl`) and passes the clip position through unchanged. This
/// stage, however, is part of a material module compiled to SPIR-V through
/// [`compile_wgsl_to_spirv`], whose naga options set `ADJUST_COORDINATE_SPACE`
/// (a clip-Y negation the game's fullscreen materials depend on). To land the
/// SAME on-screen pixels as the plain sprite path, this stage PRE-NEGATES Y so
/// the adjustment cancels it back to `sprite.rs`'s convention — without that,
/// a `material_sprite` renders vertically mirrored relative to `sprite()`. The
/// global compile flag is left untouched so existing fullscreen materials stay
/// correct; the compensation is local to the sprite path only.
///
/// A sprite-shaded material's `fs_main` therefore receives an interpolated
/// `@location(0) uv: vec2<f32>` and `@location(1) tint: vec4<f32>`; it samples
/// the sprite's texture (bound into the material's texture slot 0,
/// `@binding(1)`/`@binding(2)`) at `uv` and controls the final pixel colour. The
/// entry point is name-spaced (`vk2d_sprite_vs_main`) so it never collides with
/// the author's own `vs_main` (the fullscreen pipeline's vertex stage), letting
/// ONE compiled module back BOTH pipelines.
const SPRITE_VS: &str = r#"
struct Vk2dSpriteVsIn {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
};
struct Vk2dSpriteVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};
@vertex
fn vk2d_sprite_vs_main(in: Vk2dSpriteVsIn) -> Vk2dSpriteVsOut {
    var out: Vk2dSpriteVsOut;
    // Pre-negate Y so the material module's ADJUST_COORDINATE_SPACE flip cancels
    // it, matching sprite.rs's (WGSL, un-adjusted) clip convention.
    out.position = vec4<f32>(in.position.x, -in.position.y, 0.0, 1.0);
    out.uv = in.uv;
    out.tint = in.tint;
    return out;
}
"#;

/// The kind of a named uniform, used to compute its size and byte offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniformType {
    /// A single `f32` (WGSL `f32`).
    Float,
    /// Two `f32`s (WGSL `vec2<f32>`).
    Vec2,
    /// Three `f32`s (WGSL `vec3<f32>`).
    Vec3,
    /// Four `f32`s (WGSL `vec4<f32>`).
    Vec4,
}

/// A value pushed to a named uniform. Matches the four `UniformType` shapes.
#[derive(Debug, Clone, Copy)]
pub enum UniformValue {
    /// A single `f32`.
    Float(f32),
    /// Two `f32`s.
    Vec2(f32, f32),
    /// Three `f32`s.
    Vec3(f32, f32, f32),
    /// Four `f32`s.
    Vec4(f32, f32, f32, f32),
}

impl UniformValue {
    /// Encode the occupied component bytes into a stack buffer.
    ///
    /// Uniform updates are a hot per-frame path. Returning a fixed buffer plus
    /// its used length avoids the two heap allocations the former `Vec<f32>`
    /// -> `Vec<u8>` conversion performed for every update.
    fn to_le_bytes(self) -> ([u8; 16], usize) {
        let (components, len) = match self {
            UniformValue::Float(x) => ([x, 0.0, 0.0, 0.0], 4),
            UniformValue::Vec2(x, y) => ([x, y, 0.0, 0.0], 8),
            UniformValue::Vec3(x, y, z) => ([x, y, z, 0.0], 12),
            UniformValue::Vec4(x, y, z, w) => ([x, y, z, w], 16),
        };
        let mut bytes = [0; 16];
        for (chunk, component) in bytes.chunks_exact_mut(4).zip(components) {
            chunk.copy_from_slice(&component.to_le_bytes());
        }
        (bytes, len)
    }
}

/// Everything needed to build one material.
#[derive(Default)]
pub struct MaterialDesc<'a> {
    /// WGSL source. Must define `vs_main` + `fs_main` and a uniform block at
    /// `@group(0) @binding(0)` matching `uniforms`.
    pub wgsl: &'a str,
    /// How the material composites.
    pub blend: BlendMode,
    /// The uniform block's fields, in declaration order. Each field is placed in
    /// its own 16-byte-aligned slot (std140-friendly), so the app pushes values
    /// by name without hand-writing offsets.
    pub uniforms: &'a [(&'a str, UniformType)],
    /// Optional shared prelude prepended before compilation (e.g. a helper
    /// library). `None` compiles `wgsl` unchanged. The body may reference any
    /// item the prelude declares.
    pub prelude: Option<&'a str>,
    /// Names of sampled textures this material declares beyond its uniform
    /// block, in binding order (see [`crate::Frame::bind_material_texture`]).
    /// Empty for uniform-only materials.
    ///
    /// Binding contract: the uniform block stays at `@group(0) @binding(0)`.
    /// For the i-th name in `textures`, the WGSL must declare a
    /// `texture_2d<f32>` at `@group(0) @binding(1 + 2*i)` and a `sampler` at
    /// `@group(0) @binding(2 + 2*i)`. An unbound slot samples a shared 1x1
    /// white fallback texture until the app calls
    /// [`crate::Frame::bind_material_texture`].
    pub textures: &'a [&'a str],
}

/// Assign each declared uniform its own 16-byte-aligned slot and return a
/// `name -> byte offset` map. Each entry occupies one 16-byte slot regardless of
/// its type (the simplest std140-safe layout: a `vec4` slot per field).
pub(crate) fn uniform_offsets(uniforms: &[(&str, UniformType)]) -> HashMap<String, u32> {
    let mut map = HashMap::new();
    for (i, (name, _ty)) in uniforms.iter().enumerate() {
        map.insert((*name).to_string(), (i * 16) as u32);
    }
    map
}

/// A texture + sampler pair a material can bind into one of its declared
/// texture slots. Owned copies (not references) so the material can rebuild
/// its bind group at any time without borrowing the texture registry.
struct BoundTexture {
    source: TextureSource,
    view: TextureView,
    sampler: Sampler,
}

/// Stable registry identity for an already-bound texture resource. wgpu
/// handles do not expose value equality, while vk2d's opaque ids do.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureSource {
    Texture(TextureId),
    Target(TargetId),
}

/// A compiled material: its pipeline, uniform buffer, name->offset map, and
/// (if it declares any) its texture slots.
pub(crate) struct Material {
    /// The fullscreen pipeline (author's `vs_main` + `fs_main`), built lazily
    /// the first time [`crate::Frame::material_fullscreen`] records this
    /// material. Deferred because a material authored ONLY for
    /// [`crate::Frame::material_sprite`] gives its `fs_main` the sprite vertex
    /// stage's `@location(0) uv`/`@location(1) tint` inputs, which the bufferless
    /// fullscreen `vs_main` (bare `@builtin(position)`) cannot provide — building
    /// the fullscreen pipeline eagerly would fail wgpu's stage-matching for such
    /// a material even though it is never drawn fullscreen. The WGSL is still
    /// compiled to SPIR-V at load (syntax/validation errors surface there); only
    /// this pipeline object's construction — and its per-pipeline stage linking —
    /// waits for first fullscreen use. Symmetric with `sprite_pipeline`.
    pipeline: Option<RenderPipeline>,
    pub bind_group: BindGroup,
    pub uniform_buffer: Buffer,
    pub offsets: HashMap<String, u32>,
    /// Declared texture names from `MaterialDesc::textures`, in binding order
    /// (slot i binds at `@binding(1+2i)`/`@binding(2+2i)`).
    pub(crate) texture_names: Vec<String>,
    /// The bind-group layout backing `bind_group`, kept so the bind group can
    /// be rebuilt when a texture slot changes.
    bind_group_layout: BindGroupLayout,
    /// Per-slot bound texture, in the same order as `texture_names`. `None`
    /// until the app calls `set_texture`; the fallback is substituted when
    /// rebuilding the bind group.
    bound_textures: Vec<Option<BoundTexture>>,
    /// The compiled shader module backing BOTH pipelines. It carries the
    /// author's `vs_main`/`fs_main` AND the appended sprite vertex stage
    /// (`vk2d_sprite_vs_main`, see [`SPRITE_VS`]), so the sprite-shaded pipeline
    /// can be built from it on demand without recompiling.
    module: wgpu::ShaderModule,
    /// The scene format both pipelines render into — kept so the sprite-shaded
    /// pipeline can be built lazily with the same target as the fullscreen one.
    target_format: TextureFormat,
    /// The material's blend mode — reused by the sprite-shaded pipeline so a
    /// material composites identically whichever draw verb records it.
    blend: BlendMode,
    /// The sprite-shaded pipeline (the sprite vertex stage + this material's
    /// `fs_main`), built lazily the first time [`crate::Frame::material_sprite`]
    /// records this material — zero cost for materials only ever drawn
    /// fullscreen. See [`Material::sprite_pipeline`].
    sprite_pipeline: Option<RenderPipeline>,
}

impl Material {
    /// Compile the WGSL to SPIR-V and build the pipeline + uniform buffer.
    /// `fallback` is the shared 1x1 white view/sampler substituted into any
    /// declared texture slot that hasn't been bound yet.
    pub(crate) fn new(
        device: &Device,
        desc: &MaterialDesc,
        target_format: TextureFormat,
        fallback: (&TextureView, &Sampler),
    ) -> Result<Self, Vk2dError> {
        // Append the sprite vertex stage so ONE compiled module backs both the
        // fullscreen pipeline (author's `vs_main`) and the sprite-shaded
        // pipeline (`vk2d_sprite_vs_main` + author's `fs_main`). A material that
        // is only ever drawn fullscreen still carries the extra (dead) vertex
        // entry point; it is a handful of instructions and never dispatched.
        let author = match desc.prelude {
            Some(prelude) => format!("{prelude}\n{}", desc.wgsl),
            None => desc.wgsl.to_string(),
        };
        let source = format!("{author}\n{SPRITE_VS}");
        let spirv = compile_wgsl_to_spirv(&source)?;
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("vk2d.material.shader"),
            source: ShaderSource::SpirV(std::borrow::Cow::Owned(spirv)),
        });

        let offsets = uniform_offsets(desc.uniforms);
        let uniform_size = ((desc.uniforms.len().max(1)) * 16) as u64;
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("vk2d.material.uniforms"),
            size: uniform_size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_names: Vec<String> = desc.textures.iter().map(|s| (*s).to_string()).collect();
        let bind_group_layout = create_bind_group_layout(device, texture_names.len());
        let bound_textures: Vec<Option<BoundTexture>> =
            (0..texture_names.len()).map(|_| None).collect();
        let bind_group = build_bind_group(
            device,
            &bind_group_layout,
            &uniform_buffer,
            &bound_textures,
            fallback,
        );

        Ok(Self {
            pipeline: None,
            bind_group,
            uniform_buffer,
            offsets,
            texture_names,
            bind_group_layout,
            bound_textures,
            module,
            target_format,
            blend: desc.blend,
            sprite_pipeline: None,
        })
    }

    /// The fullscreen pipeline (author's `vs_main` + `fs_main`, no vertex
    /// buffer), built on first use and cached thereafter. See the `pipeline`
    /// field for why it is lazy. Returns the built pipeline for the draw pass to
    /// bind.
    pub(crate) fn fullscreen_pipeline(&mut self, device: &Device) -> &RenderPipeline {
        if self.pipeline.is_none() {
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("vk2d.material.pipeline_layout"),
                bind_group_layouts: &[Some(&self.bind_group_layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("vk2d.material.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &self.module,
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
                    module: &self.module,
                    entry_point: Some("fs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: self.target_format,
                        blend: self.blend.blend_state(),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            self.pipeline = Some(pipeline);
        }
        self.pipeline.as_ref().expect("just built above")
    }

    /// The already-built fullscreen pipeline, or `None` if it has not been built
    /// yet — the read-only accessor the draw pass uses (it holds `&Context` and
    /// cannot call the `&mut` [`Self::fullscreen_pipeline`] builder). The
    /// pre-warm phase builds it first for any material with a `material_fullscreen`
    /// command this frame.
    pub(crate) fn fullscreen_pipeline_cached(&self) -> Option<&RenderPipeline> {
        self.pipeline.as_ref()
    }

    /// The sprite-shaded pipeline for this material, built on first use and
    /// cached thereafter (binding-layout Option (b): a lazy per-material
    /// variant, so a material only ever drawn fullscreen never pays for it).
    ///
    /// It pairs `sprite.rs`'s vertex layout + the appended `vk2d_sprite_vs_main`
    /// vertex stage with this material's own `fs_main`, and reuses the
    /// material's EXISTING bind-group layout unchanged — the sprite's texture is
    /// bound into the material's texture slot 0 (`@binding(1)`/`@binding(2)`)
    /// through the ordinary [`Material::set_texture`] path, so no extra layout
    /// entry is needed. The vertex buffer layout comes from the shared
    /// [`sprite_vertex_buffer_layout`], guaranteeing a byte-exact match with the
    /// plain sprite pipeline (a stride/offset drift would feed the vertex stage
    /// garbage with no compile error).
    ///
    /// Returns `None` if the material declares no texture slot — a sprite-shaded
    /// material must have at least slot 0 for the sprite's source texture; the
    /// caller ([`crate::Frame::material_sprite`]) treats that as a no-op.
    pub(crate) fn sprite_pipeline(&mut self, device: &Device) -> Option<&RenderPipeline> {
        if self.texture_names.is_empty() {
            return None;
        }
        if self.sprite_pipeline.is_none() {
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("vk2d.material.sprite_pipeline_layout"),
                bind_group_layouts: &[Some(&self.bind_group_layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("vk2d.material.sprite_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &self.module,
                    entry_point: Some("vk2d_sprite_vs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[sprite_vertex_buffer_layout()],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    // The sprite batch emits CCW quads (see `sprite.rs`); match
                    // its winding and cull nothing, exactly like the plain
                    // sprite pipeline, so a material-shaded sprite rasterizes
                    // identically.
                    front_face: FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: self.target_format,
                        blend: self.blend.blend_state(),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            self.sprite_pipeline = Some(pipeline);
        }
        self.sprite_pipeline.as_ref()
    }

    /// The already-built sprite-shaded pipeline, or `None` if it has not been
    /// built yet. A read-only accessor for the draw pass (which holds `&Context`
    /// and cannot call the `&mut` [`Self::sprite_pipeline`] builder); the
    /// pre-warm phase builds it first, so this returns `Some` for any material
    /// that had a `material_sprite` run this frame.
    pub(crate) fn sprite_pipeline_cached(&self) -> Option<&RenderPipeline> {
        self.sprite_pipeline.as_ref()
    }

    /// Whether this material declares at least one texture slot — the
    /// prerequisite for [`crate::Frame::material_sprite`], whose sprite source
    /// binds into slot 0. A uniform-only material returns `false` and
    /// `material_sprite` no-ops.
    pub(crate) fn has_texture_slot(&self) -> bool {
        !self.texture_names.is_empty()
    }

    /// Write a named uniform's value at its mapped offset. Unknown names are
    /// ignored (the no-panic contract).
    pub(crate) fn set_uniform(&self, queue: &Queue, name: &str, value: UniformValue) {
        if let Some(&offset) = self.offsets.get(name) {
            let (bytes, len) = value.to_le_bytes();
            queue.write_buffer(&self.uniform_buffer, offset as u64, &bytes[..len]);
        }
    }

    /// Build a fresh bind group for a single [`crate::Frame::material_sprite`]
    /// run: this material's uniform buffer, the sprite's `view`/`sampler` in
    /// texture slot 0 (`@binding(1)`/`@binding(2)`), and the shared `fallback`
    /// in every other declared slot. Unlike [`Self::set_texture`], this does NOT
    /// mutate the material's persistent `bound_textures`/`bind_group` — each
    /// material-sprite run gets its own bind group, so two runs that draw
    /// different textures through the SAME material in one frame stay correct
    /// (a single shared bind group would leave only the last texture bound).
    pub(crate) fn build_sprite_bind_group(
        &self,
        device: &Device,
        view: &TextureView,
        sampler: &Sampler,
        fallback: (&TextureView, &Sampler),
    ) -> BindGroup {
        let (fallback_view, fallback_sampler) = fallback;
        let mut entries = Vec::with_capacity(1 + self.texture_names.len() * 2);
        entries.push(BindGroupEntry {
            binding: 0,
            resource: self.uniform_buffer.as_entire_binding(),
        });
        for i in 0..self.texture_names.len() {
            // Slot 0 gets the sprite's source; any further declared slots fall
            // back to the shared 1x1 white (a sprite-shaded material typically
            // declares only slot 0, but extra slots must still be populated).
            let (v, s) = if i == 0 {
                (view, sampler)
            } else {
                (fallback_view, fallback_sampler)
            };
            entries.push(BindGroupEntry {
                binding: 1 + 2 * i as u32,
                resource: BindingResource::TextureView(v),
            });
            entries.push(BindGroupEntry {
                binding: 2 + 2 * i as u32,
                resource: BindingResource::Sampler(s),
            });
        }
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("vk2d.material.sprite_bind_group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        })
    }

    /// Bind `view`/`sampler` to the declared texture slot named `name` and
    /// rebuild the bind group. Unknown slot names are ignored (the no-panic
    /// contract) — the caller (`Context::set_material_texture`) is expected to
    /// have already resolved the `TextureId`.
    pub(crate) fn set_texture(
        &mut self,
        device: &Device,
        name: &str,
        source: TextureSource,
        view: TextureView,
        sampler: Sampler,
        fallback: (&TextureView, &Sampler),
    ) {
        let Some(slot) = self.texture_names.iter().position(|n| n == name) else {
            return;
        };
        if self.bound_textures[slot]
            .as_ref()
            .is_some_and(|bound| bound.source == source)
        {
            return;
        }
        self.bound_textures[slot] = Some(BoundTexture {
            source,
            view,
            sampler,
        });
        self.bind_group = build_bind_group(
            device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.bound_textures,
            fallback,
        );
    }
}

/// Build (or rebuild) the material's bind group: the uniform buffer at
/// binding 0, then each declared texture slot's view/sampler pair (or the
/// shared fallback for an unbound slot) at `1+2i`/`2+2i`.
fn build_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform_buffer: &Buffer,
    bound_textures: &[Option<BoundTexture>],
    fallback: (&TextureView, &Sampler),
) -> BindGroup {
    let (fallback_view, fallback_sampler) = fallback;
    let mut entries = Vec::with_capacity(1 + bound_textures.len() * 2);
    entries.push(BindGroupEntry {
        binding: 0,
        resource: uniform_buffer.as_entire_binding(),
    });
    for (i, bound) in bound_textures.iter().enumerate() {
        let (view, sampler) = match bound {
            Some(b) => (&b.view, &b.sampler),
            None => (fallback_view, fallback_sampler),
        };
        entries.push(BindGroupEntry {
            binding: 1 + 2 * i as u32,
            resource: BindingResource::TextureView(view),
        });
        entries.push(BindGroupEntry {
            binding: 2 + 2 * i as u32,
            resource: BindingResource::Sampler(sampler),
        });
    }
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vk2d.material.bind_group"),
        layout,
        entries: &entries,
    })
}

/// Build the material's bind-group layout: the uniform block at binding 0,
/// then for each of `texture_count` declared textures, a filterable
/// `texture_2d<f32>` at `1+2i` and a filtering `sampler` at `2+2i` — the
/// contract documented on `MaterialDesc::textures`.
fn create_bind_group_layout(device: &Device, texture_count: usize) -> BindGroupLayout {
    let mut entries = Vec::with_capacity(1 + texture_count * 2);
    entries.push(BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    for i in 0..texture_count {
        entries.push(BindGroupLayoutEntry {
            binding: 1 + 2 * i as u32,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(BindGroupLayoutEntry {
            binding: 2 + 2 * i as u32,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        });
    }
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vk2d.material.bind_group_layout"),
        entries: &entries,
    })
}

/// Build a 1x1 opaque-white texture view + sampler, used as the fallback for
/// any declared material texture slot that hasn't been bound yet (so an
/// un-bound `textureSample` reads white rather than leaving the slot
/// unpopulated). Created once, lazily, by `Context`.
pub(crate) fn create_fallback_texture(device: &Device, queue: &Queue) -> (TextureView, Sampler) {
    let size = Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("vk2d.material.fallback_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        // Linear, matching `SCENE_FORMAT` and `sprite.rs::create_gpu_texture`
        // — the same "no gamma anywhere" convention. An opaque-white texel is
        // sRGB/linear-encoding-invariant, so this specific value never showed
        // the bug, but a mismatched fallback format still risked drift if a
        // future fallback color were ever not pure white/black.
        format: crate::target::SCENE_FORMAT,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Default::default(),
            aspect: TextureAspect::All,
        },
        &[255u8, 255, 255, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        size,
    );
    let view = texture.create_view(&TextureViewDescriptor::default());
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("vk2d.material.fallback_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        ..Default::default()
    });
    (view, sampler)
}

/// Parse + validate WGSL and emit SPIR-V words. `Err(ShaderCompile)` on any
/// failure, with a message formatted from naga's diagnostic (including source
/// location where naga provides one).
///
/// This is the startup path the whole "author WGSL, run SPIR-V" story rests on:
/// call it once per material at load; the resulting words feed
/// `wgpu::ShaderSource::SpirV`.
///
/// Exposed (via `crate::testing`) only for the crate's integration tests; not a
/// stable public API.
pub fn compile_wgsl_to_spirv(wgsl: &str) -> Result<Vec<u32>, Vk2dError> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| Vk2dError::ShaderCompile {
        message: e.emit_to_string(wgsl),
    })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| Vk2dError::ShaderCompile {
        message: format!("{e:?}"),
    })?;
    // NOTE on clip-space Y: naga's default SPIR-V options set
    // `ADJUST_COORDINATE_SPACE`, which flips clip-space Y. Every EXISTING vk2d
    // material (the game's fullscreen post/effect shaders) was authored against
    // that convention — they emit their own `out.uv = (p.x, 1 - p.y)` and rely
    // on the adjustment — so this flag MUST stay set or every shipping
    // fullscreen material would render vertically mirrored. The sprite-shaded
    // path instead compensates inside its injected vertex stage (see
    // `SPRITE_VS`), keeping this global compile untouched.
    let spv =
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .map_err(|e| Vk2dError::ShaderCompile {
                message: format!("{e:?}"),
            })?;
    Ok(spv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_offsets_are_16_byte_aligned_slots() {
        let map = uniform_offsets(&[
            ("u_time", UniformType::Float),
            ("u_from", UniformType::Vec2),
            ("u_color", UniformType::Vec4),
        ]);
        assert_eq!(map.get("u_time"), Some(&0));
        assert_eq!(map.get("u_from"), Some(&16));
        assert_eq!(map.get("u_color"), Some(&32));
    }

    #[test]
    fn uniform_value_bytes_have_right_length() {
        assert_eq!(UniformValue::Float(1.0).to_le_bytes().1, 4);
        assert_eq!(UniformValue::Vec4(1.0, 2.0, 3.0, 4.0).to_le_bytes().1, 16);
    }

    /// A sprite-shaded material's `fs_main` consumes the interpolated
    /// `@location(0) uv` / `@location(1) tint` the appended sprite vertex stage
    /// produces. The material source concatenated with `SPRITE_VS` (the exact
    /// concatenation `Material::new` performs before compiling) must compile:
    /// the two vertex entry points (`vs_main` + `vk2d_sprite_vs_main`) coexist
    /// in one module, and `fs_main`'s inputs line up with the sprite stage's
    /// outputs by `@location`. This is the compile-side half of the pipeline
    /// construction invariant — the live GPU pipeline build is exercised by the
    /// `material_sprite_pipeline` integration test.
    #[test]
    fn sprite_shaded_material_source_compiles_with_appended_vertex_stage() {
        // A representative sprite-shaded material: a uniform block, one texture
        // slot (slot 0 = the sprite source), a fullscreen `vs_main`, and an
        // `fs_main` that samples slot 0 at the interpolated uv and multiplies by
        // both the vertex tint and a uniform — the passthrough-tint shape.
        const SPRITE_MAT: &str = r#"
struct U { tint: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t0: texture_2d<f32>;
@group(0) @binding(2) var s0: sampler;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
}
struct FsIn {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};
@fragment fn fs_main(in: FsIn) -> @location(0) vec4<f32> {
    return textureSample(t0, s0, in.uv) * in.tint * u.tint;
}
"#;
        let combined = format!("{SPRITE_MAT}\n{SPRITE_VS}");
        assert!(
            compile_wgsl_to_spirv(&combined).is_ok(),
            "sprite-shaded material + appended vertex stage must compile"
        );
        // And the appended stage alone is well-formed WGSL (guards a typo in
        // SPRITE_VS from only surfacing once spliced into a full material).
        assert!(
            compile_wgsl_to_spirv(&format!(
                "@fragment fn fs_main() -> @location(0) vec4<f32> {{ return vec4<f32>(1.0); }}\n{SPRITE_VS}"
            ))
            .is_ok(),
            "the appended sprite vertex stage must be valid WGSL"
        );
    }

    /// The load-bearing invariant: the vertex buffer layout the material
    /// sprite-shaded pipeline builds from is byte-for-byte the one `sprite.rs`'s
    /// plain sprite pipeline uses. A silent stride/offset drift would feed the
    /// material's vertex stage garbage with no compile error, so pin the shared
    /// layout's stride and every attribute's offset/format/location here.
    #[test]
    fn sprite_shaded_pipeline_uses_the_exact_sprite_vertex_layout() {
        use crate::sprite::{SPRITE_VERTEX_ATTRIBUTES, VERTEX_STRIDE, sprite_vertex_buffer_layout};

        let layout = sprite_vertex_buffer_layout();
        // Stride: position(8) + uv(8) + tint(16) = 32 bytes.
        assert_eq!(layout.array_stride, VERTEX_STRIDE);
        assert_eq!(layout.array_stride, 32);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
        // The layout draws its attributes from the single shared source of
        // truth, so the material pipeline (which also builds from it) cannot
        // drift from the sprite pipeline.
        assert_eq!(layout.attributes, &SPRITE_VERTEX_ATTRIBUTES);

        // Explicit offsets/formats/locations — position @0, uv @8, tint @16.
        let expect = [
            (wgpu::VertexFormat::Float32x2, 0u64, 0u32),
            (wgpu::VertexFormat::Float32x2, 8, 1),
            (wgpu::VertexFormat::Float32x4, 16, 2),
        ];
        assert_eq!(SPRITE_VERTEX_ATTRIBUTES.len(), expect.len());
        for (attr, (format, offset, location)) in SPRITE_VERTEX_ATTRIBUTES.iter().zip(expect) {
            assert_eq!(attr.format, format);
            assert_eq!(attr.offset, offset);
            assert_eq!(attr.shader_location, location);
        }
    }
}
