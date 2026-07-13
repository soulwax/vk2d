//! Generic WGSL material: one pipeline type for every shader. Shaders are
//! authored as WGSL text and compiled to SPIR-V at load time (via naga); the
//! app pushes uniforms by name. There is no per-shader Rust — an effect is a
//! `.wgsl` file plus a `[(name, UniformType)]` declaration.

use std::collections::HashMap;

use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device,
    Extent3d, FilterMode, FragmentState, MultisampleState, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};

use crate::blend::BlendMode;
use crate::{TargetId, TextureId, Vk2dError};

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
    pub pipeline: RenderPipeline,
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
        let source = match desc.prelude {
            Some(prelude) => std::borrow::Cow::Owned(format!("{prelude}\n{}", desc.wgsl)),
            None => std::borrow::Cow::Borrowed(desc.wgsl),
        };
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

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("vk2d.material.pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("vk2d.material.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
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
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: target_format,
                    blend: desc.blend.blend_state(),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group,
            uniform_buffer,
            offsets,
            texture_names,
            bind_group_layout,
            bound_textures,
        })
    }

    /// Write a named uniform's value at its mapped offset. Unknown names are
    /// ignored (the no-panic contract).
    pub(crate) fn set_uniform(&self, queue: &Queue, name: &str, value: UniformValue) {
        if let Some(&offset) = self.offsets.get(name) {
            let (bytes, len) = value.to_le_bytes();
            queue.write_buffer(&self.uniform_buffer, offset as u64, &bytes[..len]);
        }
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
        format: TextureFormat::Rgba8UnormSrgb,
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
}
