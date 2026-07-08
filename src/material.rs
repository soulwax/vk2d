//! Generic WGSL material: one pipeline type for every shader. Shaders are
//! authored as WGSL text and compiled to SPIR-V at load time (via naga); the
//! app pushes uniforms by name. There is no per-shader Rust — an effect is a
//! `.wgsl` file plus a `[(name, UniformType)]` declaration.

use std::collections::HashMap;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    ColorTargetState, ColorWrites, Device, FragmentState, MultisampleState,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    TextureFormat, VertexState,
};

use crate::Vk2dError;
use crate::blend::BlendMode;

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
    /// The value's components as little-endian bytes (1–4 f32).
    fn to_le_bytes(self) -> Vec<u8> {
        let floats: Vec<f32> = match self {
            UniformValue::Float(x) => vec![x],
            UniformValue::Vec2(x, y) => vec![x, y],
            UniformValue::Vec3(x, y, z) => vec![x, y, z],
            UniformValue::Vec4(x, y, z, w) => vec![x, y, z, w],
        };
        floats.iter().flat_map(|f| f.to_le_bytes()).collect()
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

/// A compiled material: its pipeline, uniform buffer, and name->offset map.
pub(crate) struct Material {
    pub pipeline: RenderPipeline,
    pub bind_group: BindGroup,
    pub uniform_buffer: Buffer,
    pub offsets: HashMap<String, u32>,
    /// Declared texture names from `MaterialDesc::textures`, in binding order.
    /// Not yet consumed for binding/rendering — that lands with texture
    /// materials (Task A2).
    #[allow(dead_code)] // consumed by texture materials (Task A2)
    pub(crate) texture_names: Vec<String>,
}

impl Material {
    /// Compile the WGSL to SPIR-V and build the pipeline + uniform buffer.
    pub(crate) fn new(
        device: &Device,
        desc: &MaterialDesc,
        target_format: TextureFormat,
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

        let bind_group_layout = create_uniform_layout(device);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("vk2d.material.bind_group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

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

        let texture_names = desc.textures.iter().map(|s| (*s).to_string()).collect();

        Ok(Self {
            pipeline,
            bind_group,
            uniform_buffer,
            offsets,
            texture_names,
        })
    }

    /// Write a named uniform's value at its mapped offset. Unknown names are
    /// ignored (the no-panic contract).
    pub(crate) fn set_uniform(&self, queue: &Queue, name: &str, value: UniformValue) {
        if let Some(&offset) = self.offsets.get(name) {
            queue.write_buffer(&self.uniform_buffer, offset as u64, &value.to_le_bytes());
        }
    }
}

fn create_uniform_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vk2d.material.uniform_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX_FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
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
        assert_eq!(UniformValue::Float(1.0).to_le_bytes().len(), 4);
        assert_eq!(
            UniformValue::Vec4(1.0, 2.0, 3.0, 4.0).to_le_bytes().len(),
            16
        );
    }
}
