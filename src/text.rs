//! Text rendering via a fontdue glyph atlas. Game-agnostic — the app supplies
//! decoded TTF bytes; the library never reads font files. Printable ASCII is
//! baked into one coverage atlas at load; each string draws as textured quads
//! through an alpha pipeline. Layout is a simple advance walk with per-glyph
//! metrics (enough for HUD/labels; full shaping is out of scope).

use std::collections::HashMap;

use fontdue::{Font, FontSettings};
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferAddress, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Device, Extent3d,
    FilterMode, FragmentState, FrontFace, IndexFormat, MultisampleState,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

use crate::Vk2dError;
use crate::blend::BlendMode;
use crate::sprite::logical_to_clip;

const FIRST_CHAR: u32 = 0x20; // space
const LAST_CHAR: u32 = 0x7E; // tilde
const ATLAS_PADDING: u32 = 1;
const VERTEX_STRIDE: BufferAddress = 32;
const VERTS_PER_GLYPH: usize = 4;
const INDICES_PER_GLYPH: usize = 6;

#[derive(Clone, Copy)]
struct Glyph {
    atlas_px: [f32; 4],
    offset: [f32; 2],
    advance: f32,
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
@group(0) @binding(0) var glyph_atlas: texture_2d<f32>;
@group(0) @binding(1) var glyph_sampler: sampler;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coverage = textureSample(glyph_atlas, glyph_sampler, in.uv).r;
    return vec4<f32>(in.tint.rgb, in.tint.a * coverage);
}
"#;

/// A fontdue-backed text renderer: one baked atlas + a batched-quad pipeline.
/// Glyph geometry accumulates across a frame and draws in one call.
pub(crate) struct TextRenderer {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    atlas_width: f32,
    atlas_height: f32,
    glyphs: HashMap<char, Glyph>,
    baked_px: f32,
    ascent: f32,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    capacity_glyphs: usize,
    /// CPU accumulation for the frame.
    verts: Vec<u8>,
    indices: Vec<u8>,
    glyph_count: usize,
}

impl TextRenderer {
    /// Bake `ttf_bytes` at `baked_px` and build the text pipeline for
    /// `target_format`. `Err` (never a panic) if the font is unparsable.
    pub(crate) fn new(
        device: &Device,
        queue: &Queue,
        target_format: TextureFormat,
        ttf_bytes: &[u8],
        baked_px: f32,
        max_glyphs: usize,
    ) -> Result<Self, Vk2dError> {
        let font = Font::from_bytes(ttf_bytes, FontSettings::default()).map_err(|e| {
            Vk2dError::ShaderCompile {
                message: format!("font parse failed: {e}"),
            }
        })?;
        let bake = bake_atlas(&font, baked_px);
        let bind_group_layout = create_bind_group_layout(device);
        let bind_group =
            create_atlas_bind_group(device, queue, &bind_group_layout, &bake.atlas, bake.size);
        let pipeline = create_pipeline(device, target_format, &bind_group_layout);
        let capacity_glyphs = max_glyphs.max(1);
        let vertex_buffer = create_empty_buffer(
            device,
            "vk2d.text.vertices",
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            (capacity_glyphs * VERTS_PER_GLYPH) as BufferAddress * VERTEX_STRIDE,
        );
        let index_buffer = create_empty_buffer(
            device,
            "vk2d.text.indices",
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            (capacity_glyphs * INDICES_PER_GLYPH * std::mem::size_of::<u16>()) as BufferAddress,
        );
        Ok(Self {
            pipeline,
            bind_group,
            atlas_width: bake.size[0] as f32,
            atlas_height: bake.size[1] as f32,
            glyphs: bake.glyphs,
            baked_px,
            ascent: bake.ascent,
            vertex_buffer,
            index_buffer,
            capacity_glyphs,
            verts: Vec::new(),
            indices: Vec::new(),
            glyph_count: 0,
        })
    }

    /// Measure `text` at `px` height: `(width, height)` in logical pixels. The
    /// width is the pen advance; the height is the baked line height scaled.
    pub(crate) fn measure(&self, text: &str, px: f32) -> (f32, f32) {
        let scale = text_scale(px, self.baked_px);
        let mut width = 0.0;
        for ch in text.chars() {
            match self.glyphs.get(&ch) {
                Some(g) => width += g.advance * scale,
                None => width += self.baked_px * 0.3 * scale,
            }
        }
        (width, px)
    }

    /// Distance from a line's top (the `origin.y` passed to [`Self::queue_text`])
    /// down to its baseline, in logical pixels, at `px` height. Delegates to
    /// [`baseline_offset_from`], the exact same expression
    /// [`Self::queue_text`] adds to `origin.y` before rounding — so
    /// measurement and drawing can never disagree about where the baseline
    /// lands.
    pub(crate) fn baseline_offset(&self, px: f32) -> f32 {
        baseline_offset_from(self.ascent, px, self.baked_px)
    }

    /// Clear the frame's glyph accumulation.
    pub(crate) fn begin_frame(&mut self) {
        self.verts.clear();
        self.indices.clear();
        self.glyph_count = 0;
    }

    /// Append `text` starting at logical-pixel `origin` (top-left of the first
    /// line), `px` height, `color` tint. Snapped to whole pixels for crispness.
    pub(crate) fn queue_text(
        &mut self,
        text: &str,
        origin: [f32; 2],
        px: f32,
        color: [f32; 4],
        logical_size: (u32, u32),
    ) {
        let scale = text_scale(px, self.baked_px);
        let mut pen_x = origin[0].round();
        let baseline = origin[1].round() + baseline_offset_from(self.ascent, px, self.baked_px);
        for ch in text.chars() {
            let Some(glyph) = self.glyphs.get(&ch).copied() else {
                pen_x += self.baked_px * 0.3 * scale;
                continue;
            };
            if glyph.atlas_px[2] > 0.0 && glyph.atlas_px[3] > 0.0 {
                let base = (self.glyph_count * VERTS_PER_GLYPH) as u16;
                push_glyph_quad(
                    &mut self.verts,
                    &glyph,
                    pen_x,
                    baseline,
                    scale,
                    color,
                    self.atlas_width,
                    self.atlas_height,
                    logical_size,
                );
                for local in [0u16, 1, 2, 0, 2, 3] {
                    self.indices
                        .extend_from_slice(&(base + local).to_le_bytes());
                }
                self.glyph_count += 1;
            }
            pen_x += glyph.advance * scale;
        }
    }

    /// Upload the frame's accumulated glyphs. Call before `draw`.
    pub(crate) fn upload(&mut self, device: &Device, queue: &Queue) {
        if self.glyph_count == 0 {
            return;
        }
        if self.glyph_count > self.capacity_glyphs {
            self.grow(device, self.glyph_count);
        }
        queue.write_buffer(&self.vertex_buffer, 0, &self.verts);
        queue.write_buffer(&self.index_buffer, 0, &self.indices);
    }

    /// Draw the uploaded glyphs in one indexed draw call.
    pub(crate) fn draw<'pass>(&'pass self, pass: &mut RenderPass<'pass>) {
        if self.glyph_count == 0 {
            return;
        }
        let index_count = (self.glyph_count * INDICES_PER_GLYPH) as u32;
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        pass.draw_indexed(0..index_count, 0, 0..1);
    }

    fn grow(&mut self, device: &Device, needed: usize) {
        let mut capacity = self.capacity_glyphs.max(1);
        while capacity < needed {
            capacity *= 2;
        }
        self.vertex_buffer = create_empty_buffer(
            device,
            "vk2d.text.vertices",
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            (capacity * VERTS_PER_GLYPH) as BufferAddress * VERTEX_STRIDE,
        );
        self.index_buffer = create_empty_buffer(
            device,
            "vk2d.text.indices",
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            (capacity * INDICES_PER_GLYPH * std::mem::size_of::<u16>()) as BufferAddress,
        );
        self.capacity_glyphs = capacity;
    }
}

/// The single scale expression shared by drawing ([`TextRenderer::queue_text`])
/// and measurement ([`TextRenderer::measure`], [`TextRenderer::baseline_offset`]):
/// requested pixel height over the size the atlas was baked at. Every caller
/// that needs "how big does a baked glyph render" goes through this function
/// so draw and measure can never compute a different scale for the same
/// inputs.
fn text_scale(px: f32, baked_px: f32) -> f32 {
    px / baked_px
}

/// Distance from a line's top down to its baseline at `px` height, given the
/// font's baked `ascent` (in baked-atlas pixels) and the size it was baked at.
/// [`TextRenderer::queue_text`] adds this to `origin.y` (before rounding);
/// [`TextRenderer::baseline_offset`] — and therefore `measure_text_ext` —
/// returns the identical value. One function, one place either could diverge.
fn baseline_offset_from(ascent: f32, px: f32, baked_px: f32) -> f32 {
    ascent * text_scale(px, baked_px)
}

struct AtlasBake {
    atlas: Vec<u8>,
    size: [u32; 2],
    glyphs: HashMap<char, Glyph>,
    ascent: f32,
}

fn bake_atlas(font: &Font, baked_px: f32) -> AtlasBake {
    let mut rasterized: Vec<(char, fontdue::Metrics, Vec<u8>)> = Vec::new();
    let mut cell_w = 1u32;
    let mut cell_h = 1u32;
    for code in FIRST_CHAR..=LAST_CHAR {
        let ch = char::from_u32(code).unwrap_or(' ');
        let (metrics, bitmap) = font.rasterize(ch, baked_px);
        cell_w = cell_w.max(metrics.width as u32);
        cell_h = cell_h.max(metrics.height as u32);
        rasterized.push((ch, metrics, bitmap));
    }
    cell_w += ATLAS_PADDING;
    cell_h += ATLAS_PADDING;
    let glyph_count = rasterized.len() as u32;
    let cols = (glyph_count as f32).sqrt().ceil() as u32;
    let rows = glyph_count.div_ceil(cols);
    let atlas_w = (cols * cell_w).max(1);
    let atlas_h = (rows * cell_h).max(1);
    let mut atlas = vec![0u8; (atlas_w * atlas_h) as usize];
    let mut glyphs = HashMap::new();
    for (index, (ch, metrics, bitmap)) in rasterized.into_iter().enumerate() {
        let col = index as u32 % cols;
        let row = index as u32 / cols;
        let x0 = col * cell_w;
        let y0 = row * cell_h;
        let gw = metrics.width as u32;
        let gh = metrics.height as u32;
        for gy in 0..gh {
            for gx in 0..gw {
                let src = (gy * gw + gx) as usize;
                let dst = ((y0 + gy) * atlas_w + (x0 + gx)) as usize;
                atlas[dst] = bitmap[src];
            }
        }
        glyphs.insert(
            ch,
            Glyph {
                atlas_px: [x0 as f32, y0 as f32, gw as f32, gh as f32],
                offset: [metrics.xmin as f32, -(metrics.ymin as f32 + gh as f32)],
                advance: metrics.advance_width,
            },
        );
    }
    let ascent = font
        .horizontal_line_metrics(baked_px)
        .map(|m| m.ascent)
        .unwrap_or(baked_px * 0.8);
    AtlasBake {
        atlas,
        size: [atlas_w, atlas_h],
        glyphs,
        ascent,
    }
}

#[allow(clippy::too_many_arguments)]
fn push_glyph_quad(
    out: &mut Vec<u8>,
    glyph: &Glyph,
    pen_x: f32,
    baseline: f32,
    scale: f32,
    tint: [f32; 4],
    atlas_width: f32,
    atlas_height: f32,
    logical_size: (u32, u32),
) {
    let x = pen_x + glyph.offset[0] * scale;
    let y = baseline + glyph.offset[1] * scale;
    let w = glyph.atlas_px[2] * scale;
    let h = glyph.atlas_px[3] * scale;
    let u0 = glyph.atlas_px[0] / atlas_width;
    let v0 = glyph.atlas_px[1] / atlas_height;
    let u1 = (glyph.atlas_px[0] + glyph.atlas_px[2]) / atlas_width;
    let v1 = (glyph.atlas_px[1] + glyph.atlas_px[3]) / atlas_height;
    let corners = [
        (x, y, u0, v0),
        (x + w, y, u1, v0),
        (x + w, y + h, u1, v1),
        (x, y + h, u0, v1),
    ];
    for (px, py, u, v) in corners {
        let (ndc_x, ndc_y) = logical_to_clip(px, py, logical_size);
        out.extend_from_slice(&ndc_x.to_le_bytes());
        out.extend_from_slice(&ndc_y.to_le_bytes());
        out.extend_from_slice(&u.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
        for channel in tint {
            out.extend_from_slice(&channel.to_le_bytes());
        }
    }
}

fn create_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vk2d.text.bind_group_layout"),
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

fn create_atlas_bind_group(
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
    atlas: &[u8],
    size: [u32; 2],
) -> BindGroup {
    let texture_size = Extent3d {
        width: size[0],
        height: size[1],
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("vk2d.text.atlas"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R8Unorm,
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
        atlas,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size[0]),
            rows_per_image: Some(size[1]),
        },
        texture_size,
    );
    let view = texture.create_view(&TextureViewDescriptor::default());
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("vk2d.text.sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        ..Default::default()
    });
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("vk2d.text.bind_group"),
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
    })
}

fn create_pipeline(
    device: &Device,
    target_format: TextureFormat,
    bind_group_layout: &BindGroupLayout,
) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vk2d.text.shader"),
        source: ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vk2d.text.pipeline_layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("vk2d.text.pipeline"),
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

#[cfg(test)]
mod tests {
    use super::*;

    // Pure arithmetic tests: no GPU device and no real font needed. vk2d never
    // vendors font files (the app supplies TTF bytes), so these exercise the
    // exact scale/baseline expressions `queue_text` and `measure`/
    // `baseline_offset` share, using synthetic ascent/baked_px values instead
    // of a rasterized font.

    #[test]
    fn text_scale_is_requested_over_baked_px() {
        assert_eq!(text_scale(32.0, 16.0), 2.0);
        assert_eq!(text_scale(16.0, 16.0), 1.0);
        assert_eq!(text_scale(8.0, 16.0), 0.5);
    }

    #[test]
    fn baseline_offset_from_scales_ascent_by_text_scale() {
        // A font baked at 16px with a 12.8px ascent (0.8 ratio, fontdue's own
        // fallback ratio when a font lacks line metrics), requested at 32px:
        // the baseline should sit at ascent * (32/16) = 25.6px below the top.
        let offset = baseline_offset_from(12.8, 32.0, 16.0);
        assert!((offset - 25.6).abs() < 1e-4);
    }

    #[test]
    fn baseline_offset_from_matches_queue_text_expression() {
        // Same expression `queue_text` uses inline: origin.y.round() + this
        // term. Confirm the two independent-looking call sites reduce to
        // identical floats for a range of sizes, so `measure_text_ext`
        // (which calls `baseline_offset`) can never silently drift from what
        // gets drawn.
        let ascent = 20.0;
        let baked_px = 24.0;
        for px in [8.0, 16.0, 24.0, 48.0, 100.0] {
            let via_helper = baseline_offset_from(ascent, px, baked_px);
            let inline_equivalent = ascent * (px / baked_px);
            assert_eq!(via_helper, inline_equivalent);
        }
    }

    #[test]
    fn baseline_offset_is_positive_for_positive_ascent_and_size() {
        assert!(baseline_offset_from(10.0, 24.0, 16.0) > 0.0);
    }
}
