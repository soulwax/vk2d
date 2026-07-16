//! Vector-primitive batch: filled rects, outlines, lines, circles, and
//! triangles as coloured (untextured) triangles, drawn in one call per frame.
//! This is the renderer's answer to cheap 2D primitives — the gap the sprite
//! and material pipelines do not cover.

use std::sync::LazyLock;

use wgpu::{
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device,
    FragmentState, FrontFace, IndexFormat, MultisampleState, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, TextureFormat, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

use crate::blend::BlendMode;
use crate::color::Color;
use crate::sprite::ClipXform;

/// Bytes per vertex: position (2×f32) + colour (4×f32).
const VERTEX_STRIDE: BufferAddress = 24;
/// Triangle-fan segment count for a filled circle.
const CIRCLE_SEGMENTS: usize = 48;

/// Unit-circle points shared by every circle/ring. Trigonometry is paid once
/// on first use instead of 96 transcendental calls per primitive per frame.
static UNIT_CIRCLE: LazyLock<[(f32, f32); CIRCLE_SEGMENTS + 1]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let angle = (i as f32 / CIRCLE_SEGMENTS as f32) * std::f32::consts::TAU;
        angle.sin_cos()
    })
});

const SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(input.position, 0.0, 1.0);
    out.color = input.color;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// Accumulates coloured triangles across a frame, then draws them all at once.
/// Callers author in logical pixels; the batch converts to clip space on upload.
pub(crate) struct ShapeBatch {
    pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    capacity_verts: usize,
    /// CPU-side accumulation for this frame (cleared after upload).
    verts: Vec<f32>,
    indices: Vec<u16>,
    /// Reused upload scratch for the index buffer's alignment padding (see
    /// `upload`'s doc comment) — keeps peak capacity across frames. The
    /// vertex buffer no longer needs an equivalent scratch: `verts` itself
    /// is byte-cast directly via `bytemuck::cast_slice` at upload time.
    index_bytes: Vec<u8>,
}

impl ShapeBatch {
    pub(crate) fn new(device: &Device, target_format: TextureFormat) -> Self {
        let pipeline = create_pipeline(device, target_format);
        let capacity_verts = 2048;
        let vertex_buffer = create_empty_buffer(
            device,
            "vk2d.shapes.vertices",
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            capacity_verts as BufferAddress * VERTEX_STRIDE,
        );
        let index_buffer = create_empty_buffer(
            device,
            "vk2d.shapes.indices",
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            (capacity_verts * 3 * std::mem::size_of::<u16>()) as BufferAddress,
        );
        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            capacity_verts,
            verts: Vec::new(),
            indices: Vec::new(),
            index_bytes: Vec::new(),
        }
    }

    /// Clear the frame's accumulation. Called by the frame at begin.
    pub(crate) fn begin_frame(&mut self) {
        self.verts.clear();
        self.indices.clear();
    }

    fn push_vertex(&mut self, px: f32, py: f32, color: Color, xform: ClipXform) {
        let (x, y) = xform.apply(px, py);
        self.verts
            .extend_from_slice(&[x, y, color.r, color.g, color.b, color.a]);
    }

    /// Add a filled triangle (logical pixels). Builds the clip-space
    /// transform once per triangle (three vertices) instead of once per
    /// vertex — `logical_size` is constant for the whole batch, so the
    /// per-vertex divide `logical_to_clip` used to redo is redundant work
    /// `ClipXform::new` hoists out to a single cheap construction.
    fn push_triangle_px(
        &mut self,
        a: (f32, f32),
        b: (f32, f32),
        c: (f32, f32),
        color: Color,
        logical_size: (u32, u32),
    ) {
        let xform = ClipXform::new(logical_size);
        let base = (self.verts.len() / 6) as u16;
        self.push_vertex(a.0, a.1, color, xform);
        self.push_vertex(b.0, b.1, color, xform);
        self.push_vertex(c.0, c.1, color, xform);
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Filled rectangle.
    pub(crate) fn fill_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: Color,
        logical_size: (u32, u32),
    ) {
        let (tl, tr, br, bl) = ((x, y), (x + w, y), (x + w, y + h), (x, y + h));
        self.push_triangle_px(tl, tr, br, color, logical_size);
        self.push_triangle_px(tl, br, bl, color, logical_size);
    }

    /// Rectangle outline of the given thickness (drawn as four filled edges).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rect_outline(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        t: f32,
        color: Color,
        logical_size: (u32, u32),
    ) {
        let t = t.min(w * 0.5).min(h * 0.5).max(0.0);
        self.fill_rect(x, y, w, t, color, logical_size); // top
        self.fill_rect(x, y + h - t, w, t, color, logical_size); // bottom
        self.fill_rect(x, y + t, t, h - 2.0 * t, color, logical_size); // left
        self.fill_rect(x + w - t, y + t, t, h - 2.0 * t, color, logical_size); // right
    }

    /// A line segment of the given thickness (a rotated quad).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn line(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        thickness: f32,
        color: Color,
        logical_size: (u32, u32),
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt().max(1e-4);
        // Perpendicular unit vector scaled to half thickness.
        let (nx, ny) = (-dy / len * thickness * 0.5, dx / len * thickness * 0.5);
        let p0 = (x0 + nx, y0 + ny);
        let p1 = (x1 + nx, y1 + ny);
        let p2 = (x1 - nx, y1 - ny);
        let p3 = (x0 - nx, y0 - ny);
        self.push_triangle_px(p0, p1, p2, color, logical_size);
        self.push_triangle_px(p0, p2, p3, color, logical_size);
    }

    /// Filled circle (triangle fan).
    pub(crate) fn circle(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        color: Color,
        logical_size: (u32, u32),
    ) {
        for pair in UNIT_CIRCLE.windows(2) {
            let p0 = (cx + pair[0].1 * radius, cy + pair[0].0 * radius);
            let p1 = (cx + pair[1].1 * radius, cy + pair[1].0 * radius);
            self.push_triangle_px((cx, cy), p0, p1, color, logical_size);
        }
    }

    /// Circle outline (a ring, drawn as short line segments).
    pub(crate) fn circle_outline(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        thickness: f32,
        color: Color,
        logical_size: (u32, u32),
    ) {
        for pair in UNIT_CIRCLE.windows(2) {
            let x0 = cx + pair[0].1 * radius;
            let y0 = cy + pair[0].0 * radius;
            let x1 = cx + pair[1].1 * radius;
            let y1 = cy + pair[1].0 * radius;
            self.line(x0, y0, x1, y1, thickness, color, logical_size);
        }
    }

    /// Filled ellipse (triangle fan). `rx`/`ry` are independent horizontal/
    /// vertical semi-axes; `rotation` is radians about the centre, applied to
    /// each fan point before translation — the same triangle-fan approach as
    /// `circle`, generalized to non-uniform scale + rotation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ellipse(
        &mut self,
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        rotation: f32,
        color: Color,
        logical_size: (u32, u32),
    ) {
        let (sin_r, cos_r) = rotation.sin_cos();
        let point_at = |unit: (f32, f32)| -> (f32, f32) {
            // unit is (sin(angle), cos(angle)) per UNIT_CIRCLE's storage
            // order (see circle()'s use of pair.1 for x, pair.0 for y).
            let (ux, uy) = (unit.1 * rx, unit.0 * ry);
            let x = ux * cos_r - uy * sin_r;
            let y = ux * sin_r + uy * cos_r;
            (cx + x, cy + y)
        };
        for pair in UNIT_CIRCLE.windows(2) {
            let p0 = point_at(pair[0]);
            let p1 = point_at(pair[1]);
            self.push_triangle_px((cx, cy), p0, p1, color, logical_size);
        }
    }

    /// Filled triangle (public verb).
    pub(crate) fn triangle(
        &mut self,
        a: (f32, f32),
        b: (f32, f32),
        c: (f32, f32),
        color: Color,
        logical_size: (u32, u32),
    ) {
        self.push_triangle_px(a, b, c, color, logical_size);
    }

    /// Upload the frame's accumulated geometry to the GPU. Call once before
    /// `draw`, after all shape calls.
    pub(crate) fn upload(&mut self, device: &Device, queue: &Queue) {
        if self.indices.is_empty() {
            return;
        }
        let vert_count = self.verts.len() / 6;
        if vert_count > self.capacity_verts {
            self.grow(device, vert_count);
        }
        // `bytemuck::cast_slice` reinterprets the already-contiguous
        // Vec<f32>/Vec<u16> as &[u8] directly — no per-element
        // to_le_bytes()/extend_from_slice copy loop, and no scratch
        // vertex_bytes buffer needed at all (index_bytes is still needed
        // below, for the alignment padding cast_slice can't add). This
        // assumes a little-endian target, which the crate already did
        // implicitly (the code this replaces called `to_le_bytes()`
        // unconditionally); every platform vk2d ships to today is
        // little-endian.
        let vertex_bytes: &[u8] = bytemuck::cast_slice(&self.verts);
        self.index_bytes.clear();
        self.index_bytes.reserve(
            self.indices
                .len()
                .saturating_mul(std::mem::size_of::<u16>()),
        );
        self.index_bytes
            .extend_from_slice(bytemuck::cast_slice(&self.indices));
        // `write_buffer` requires the source length to be a multiple of
        // COPY_BUFFER_ALIGNMENT (4). u16 indices are 2 bytes each, so an ODD
        // index count yields a 2-mod-4 length and wgpu rejects the copy. Pad to
        // the next multiple of 4 (the extra bytes sit past `indices.len()`, so
        // draw_indexed never reads them). The index buffer is allocated with the
        // same rounding, so the padded write stays in bounds.
        while !self.index_bytes.len().is_multiple_of(4) {
            self.index_bytes.push(0);
        }
        queue.write_buffer(&self.vertex_buffer, 0, vertex_bytes);
        queue.write_buffer(&self.index_buffer, 0, &self.index_bytes);
    }

    /// Draw the uploaded shapes.
    pub(crate) fn draw<'pass>(&'pass self, pass: &mut RenderPass<'pass>) {
        if self.indices.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        pass.draw_indexed(0..self.indices.len() as u32, 0, 0..1);
    }

    fn grow(&mut self, device: &Device, needed_verts: usize) {
        let mut cap = self.capacity_verts.max(1);
        while cap < needed_verts {
            cap *= 2;
        }
        self.vertex_buffer = create_empty_buffer(
            device,
            "vk2d.shapes.vertices",
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            cap as BufferAddress * VERTEX_STRIDE,
        );
        self.index_buffer = create_empty_buffer(
            device,
            "vk2d.shapes.indices",
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            (cap * 3 * std::mem::size_of::<u16>()) as BufferAddress,
        );
        self.capacity_verts = cap;
    }
}

fn create_pipeline(device: &Device, target_format: TextureFormat) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vk2d.shapes.shader"),
        source: ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vk2d.shapes.pipeline_layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("vk2d.shapes.pipeline"),
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
                        format: VertexFormat::Float32x4,
                        offset: 8,
                        shader_location: 1,
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

// Geometry accumulation is verified visually by the `hello_sprite` example
// (which draws a rect + line + circle) and the probe; a headless unit test of
// `ShapeBatch` would need a GPU `Device` (the batch owns a pipeline). The pure
// clip-space mapping it relies on (`sprite::logical_to_clip`) is covered where
// it is defined.

#[cfg(test)]
mod upload_byte_cast_tests {
    // `upload()` itself needs a real GPU Device/Queue and so cannot run
    // headless (see the comment above), but the property it depends on —
    // that bytemuck::cast_slice(&verts_or_indices) produces the exact same
    // bytes the old per-element to_le_bytes()/extend_from_slice loop did —
    // is a pure, GPU-free claim. These tests pin that equivalence directly,
    // so a future bytemuck upgrade or refactor can't silently reintroduce a
    // byte-order/layout mismatch between the fast path and what wgpu
    // actually expects (little-endian, tightly packed, no padding between
    // elements — exactly what to_le_bytes()/extend_from_slice produced).

    fn to_le_bytes_loop_f32(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn to_le_bytes_loop_u16(values: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[test]
    fn cast_slice_matches_to_le_bytes_loop_for_f32_vertices() {
        let verts: Vec<f32> = vec![
            -1.0,
            1.0,
            0.0,
            0.5,
            1.0,
            0.25, // one vertex's worth of fields
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            f32::MIN,
            f32::MAX,
            -0.0,
            0.0,
            12345.678,
            -98_765.43,
        ];
        let expected = to_le_bytes_loop_f32(&verts);
        let actual: &[u8] = bytemuck::cast_slice(&verts);
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn cast_slice_matches_to_le_bytes_loop_for_u16_indices() {
        let indices: Vec<u16> = vec![0, 1, 2, 0, 2, 3, u16::MAX, u16::MIN, 12345];
        let expected = to_le_bytes_loop_u16(&indices);
        let actual: &[u8] = bytemuck::cast_slice(&indices);
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn cast_slice_of_empty_slice_is_empty() {
        let empty_f32: &[f32] = &[];
        let empty_u16: &[u16] = &[];
        let cast_f32: &[u8] = bytemuck::cast_slice(empty_f32);
        let cast_u16: &[u8] = bytemuck::cast_slice(empty_u16);
        assert!(cast_f32.is_empty());
        assert!(cast_u16.is_empty());
    }
}
