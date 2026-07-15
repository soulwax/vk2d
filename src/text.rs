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

/// Round a requested pixel height to its glyph-atlas cache key. Distinct
/// sizes bake distinct atlases; this is the single place that decides which
/// requested sizes share a bake (whole-pixel buckets — sub-pixel size
/// requests are rare in practice and would not visibly benefit from their
/// own atlas).
fn bucket_key(px: f32) -> u32 {
    px.round().max(1.0) as u32
}

/// One glyph atlas baked at a specific pixel size: the bind group vk2d's
/// text pipeline samples, plus the per-glyph metrics `queue_text`/`measure`
/// need to lay out that size's text. Every field here previously lived flat
/// on `TextRenderer` itself, back when there was only ever one baked size.
struct SizedAtlas {
    bind_group: BindGroup,
    glyphs: HashMap<char, Glyph>,
    atlas_width: f32,
    atlas_height: f32,
    ascent: f32,
    baked_px: f32,
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

/// A fontdue-backed text renderer: a lazily-grown per-size glyph atlas cache
/// plus a batched-quad pipeline shared across every size. Glyph geometry
/// accumulates across a frame, partitioned by which size baked it, and each
/// active size draws in its own indexed call (same pipeline, different bind
/// group; see `draw`).
pub(crate) struct TextRenderer {
    pipeline: RenderPipeline,
    bind_group_layout: BindGroupLayout,
    font: Font,
    /// Interior mutability: `measure`/`baseline_offset` must stay `&self`
    /// (they sit behind `Renderer2d::measure_text(&self, ...)`, a frozen
    /// trait method — see `renderer_vk.rs`'s `measure_text` impl), but a
    /// cache miss needs to bake a brand-new atlas texture + bind group,
    /// which mutates this cache. `RefCell` makes that legal without
    /// widening the trait signature. Single-threaded by construction (vk2d
    /// has no cross-thread `Context` access anywhere today), so borrow
    /// panics would only ever indicate a bug in this file, not real
    /// contention.
    atlases: std::cell::RefCell<HashMap<u32, SizedAtlas>>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    capacity_glyphs: usize,
    /// CPU accumulation for the frame, partitioned by bucket key so `draw`
    /// can issue one indexed call per active size. Each entry is
    /// `(bucket_key, index_range_start, index_range_len)`; ranges are
    /// contiguous slices of `indices` in the order buckets were first
    /// touched this frame.
    verts: Vec<u8>,
    indices: Vec<u8>,
    glyph_count: usize,
    active_buckets: Vec<(u32, u32, u32)>,
}

impl TextRenderer {
    /// Parse `ttf_bytes` and build the (initially empty) text pipeline for
    /// `target_format`. `Err` (never a panic) if the font is unparsable. No
    /// atlas is baked here — the first `measure`/`queue_text` call at a
    /// given size bakes that size lazily (see `get_or_bake`).
    pub(crate) fn new(
        device: &Device,
        _queue: &Queue,
        target_format: TextureFormat,
        ttf_bytes: &[u8],
        _baked_px: f32,
        max_glyphs: usize,
    ) -> Result<Self, Vk2dError> {
        let font = Font::from_bytes(ttf_bytes, FontSettings::default()).map_err(|e| {
            Vk2dError::ShaderCompile {
                message: format!("font parse failed: {e}"),
            }
        })?;
        let bind_group_layout = create_bind_group_layout(device);
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
            bind_group_layout,
            font,
            atlases: std::cell::RefCell::new(HashMap::new()),
            vertex_buffer,
            index_buffer,
            capacity_glyphs,
            verts: Vec::new(),
            indices: Vec::new(),
            glyph_count: 0,
            active_buckets: Vec::new(),
        })
    }

    /// Return the atlas baked at `px`'s bucket, baking it now on a cache
    /// miss. Takes `&self` (not `&mut self`) via the `RefCell`, so
    /// `measure`/`baseline_offset` — which sit behind the frozen
    /// `Renderer2d::measure_text(&self, ...)` — can trigger a bake without
    /// widening their own signatures.
    fn get_or_bake(
        &self,
        device: &Device,
        queue: &Queue,
        px: f32,
    ) -> std::cell::Ref<'_, HashMap<u32, SizedAtlas>> {
        let key = bucket_key(px);
        if !self.atlases.borrow().contains_key(&key) {
            let bake = bake_atlas(&self.font, key as f32);
            let bind_group = create_atlas_bind_group(
                device,
                queue,
                &self.bind_group_layout,
                &bake.atlas,
                bake.size,
            );
            self.atlases.borrow_mut().insert(
                key,
                SizedAtlas {
                    bind_group,
                    glyphs: bake.glyphs,
                    atlas_width: bake.size[0] as f32,
                    atlas_height: bake.size[1] as f32,
                    ascent: bake.ascent,
                    baked_px: key as f32,
                },
            );
        }
        self.atlases.borrow()
    }

    /// Measure `text` at `px` height: `(width, height)` in logical pixels.
    /// Bakes the `px` atlas now if it has never been requested before.
    pub(crate) fn measure(
        &self,
        device: &Device,
        queue: &Queue,
        text: &str,
        px: f32,
    ) -> (f32, f32) {
        let atlases = self.get_or_bake(device, queue, px);
        let atlas = &atlases[&bucket_key(px)];
        let mut width = 0.0;
        for ch in text.chars() {
            match atlas.glyphs.get(&ch) {
                Some(g) => width += g.advance,
                None => width += atlas.baked_px * 0.3,
            }
        }
        (width, px)
    }

    /// Distance from a line's top (the `origin.y` passed to
    /// [`Self::queue_text`]) down to its baseline, in logical pixels, at `px`
    /// height. Delegates to [`baseline_offset_from`], the exact same
    /// expression [`Self::queue_text`] uses — so measurement and drawing can
    /// never disagree about where the baseline lands. Bakes the `px` atlas
    /// now if it has never been requested before.
    pub(crate) fn baseline_offset(&self, device: &Device, queue: &Queue, px: f32) -> f32 {
        let atlases = self.get_or_bake(device, queue, px);
        let atlas = &atlases[&bucket_key(px)];
        baseline_offset_from(atlas.ascent)
    }

    /// Clear the frame's glyph accumulation.
    pub(crate) fn begin_frame(&mut self) {
        self.verts.clear();
        self.indices.clear();
        self.glyph_count = 0;
        self.active_buckets.clear();
    }

    /// Append `text` starting at logical-pixel `origin` (top-left of the
    /// first line), `px` height, `color` tint. Snapped to whole pixels for
    /// crispness. Bakes the `px` atlas now if it has never been requested
    /// before.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn queue_text(
        &mut self,
        device: &Device,
        queue: &Queue,
        text: &str,
        origin: [f32; 2],
        px: f32,
        color: [f32; 4],
        logical_size: (u32, u32),
    ) {
        let key = bucket_key(px);
        let range_start = self.indices.len() as u32;
        // Extract atlas data in a scope so the borrow is dropped before mutating self
        let (glyphs, ascent, atlas_width, atlas_height, baked_px) = {
            let atlases = self.get_or_bake(device, queue, px);
            let atlas = &atlases[&key];
            (
                atlas.glyphs.clone(),
                atlas.ascent,
                atlas.atlas_width,
                atlas.atlas_height,
                atlas.baked_px,
            )
        };
        let mut pen_x = origin[0].round();
        let baseline = origin[1].round() + baseline_offset_from(ascent);
        for ch in text.chars() {
            let Some(glyph) = glyphs.get(&ch).copied() else {
                pen_x += baked_px * 0.3;
                continue;
            };
            if glyph.atlas_px[2] > 0.0 && glyph.atlas_px[3] > 0.0 {
                let base = (self.glyph_count * VERTS_PER_GLYPH) as u16;
                push_glyph_quad(
                    &mut self.verts,
                    &glyph,
                    pen_x,
                    baseline,
                    color,
                    atlas_width,
                    atlas_height,
                    logical_size,
                );
                for local in [0u16, 1, 2, 0, 2, 3] {
                    self.indices
                        .extend_from_slice(&(base + local).to_le_bytes());
                }
                self.glyph_count += 1;
            }
            pen_x += glyph.advance;
        }
        let range_len = self.indices.len() as u32 - range_start;
        if range_len == 0 {
            return;
        }
        if let Some(existing) = self.active_buckets.iter_mut().find(|(k, _, _)| *k == key) {
            existing.2 += range_len;
        } else {
            self.active_buckets.push((key, range_start, range_len));
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

    /// Draw the uploaded glyphs: one indexed draw call per distinct size
    /// bucket that had text queued this frame (typically a handful — HUD,
    /// dialogue body, kill notices, speaker names — not one per glyph or
    /// per `queue_text` call). The pipeline stays bound throughout; only the
    /// bind group (which atlas texture) changes between calls.
    pub(crate) fn draw<'pass>(&'pass self, pass: &mut RenderPass<'pass>) {
        if self.active_buckets.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        let atlases = self.atlases.borrow();
        for (key, start, len) in &self.active_buckets {
            let Some(atlas) = atlases.get(key) else {
                continue;
            };
            pass.set_bind_group(0, &atlas.bind_group, &[]);
            pass.draw_indexed(*start..(*start + *len), 0, 0..1);
        }
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

/// Distance from a line's top down to its baseline, given the font's baked
/// `ascent` at the exact size it was baked at. With a per-size atlas cache,
/// the atlas that serves a `px` request is always baked at exactly `px`
/// (see `bucket_key`), so no scale multiplication is needed — this used to
/// scale `ascent` by `requested_px / baked_px`; now that ratio is always 1.0
/// by construction, so this simply returns `ascent` unchanged. Kept as its
/// own named function (rather than inlined) because `queue_text` and
/// `TextRenderer::baseline_offset` both need the identical value, and a
/// pinned shared expression is what stops the two from silently diverging.
fn baseline_offset_from(ascent: f32) -> f32 {
    ascent
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
    tint: [f32; 4],
    atlas_width: f32,
    atlas_height: f32,
    logical_size: (u32, u32),
) {
    let x = pen_x + glyph.offset[0];
    let y = baseline + glyph.offset[1];
    let w = glyph.atlas_px[2];
    let h = glyph.atlas_px[3];
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
    fn bucket_key_rounds_half_pixel_sizes_to_nearest_integer() {
        assert_eq!(bucket_key(31.6), 32);
        assert_eq!(bucket_key(32.4), 32);
        assert_eq!(bucket_key(31.4), 31);
        assert_eq!(bucket_key(32.5), 33); // f32::round rounds half-away-from-zero
    }

    #[test]
    fn bucket_key_is_stable_for_exact_integers() {
        assert_eq!(bucket_key(16.0), 16);
        assert_eq!(bucket_key(48.0), 48);
    }

    #[test]
    fn baseline_offset_from_returns_ascent_unchanged() {
        assert_eq!(baseline_offset_from(12.8), 12.8);
        assert_eq!(baseline_offset_from(0.0), 0.0);
    }

    #[test]
    fn baseline_offset_is_positive_for_positive_ascent() {
        assert!(baseline_offset_from(10.0) > 0.0);
    }
}
