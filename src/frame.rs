//! The immediate-mode frame API. `Context::begin_frame` returns a `Frame`; the
//! app issues draw calls (sprites, shapes, text, materials) in any order, then
//! `present()` records the passes and shows the result. The library batches
//! internally, so callers think in draw calls, not pipelines.

use wgpu::{CommandEncoderDescriptor, TextureViewDescriptor};

use crate::color::{Color, Point, Rect2, SpriteParams, TextStyle};
use crate::sprite::SpriteInstance;
use crate::target::SceneTarget;
use crate::{Context, FontId, MaterialId, TextureId, UniformValue, Vk2dError};

fn wgpu_color(c: Color) -> wgpu::Color {
    wgpu::Color {
        r: c.r as f64,
        g: c.g as f64,
        b: c.b as f64,
        a: c.a as f64,
    }
}

/// One accumulated draw, kept in submission order so overlapping draws layer
/// correctly (painter's algorithm).
enum DrawCmd {
    /// A sprite run: `(texture, run_slot)` where `run_slot` indexes the frame's
    /// ordered `runs` list (each entry is instances + that texture id).
    Sprites(TextureId, usize),
    /// The shape batch (all accumulated shapes; drawn once).
    Shapes,
    /// A material drawn fullscreen into the scene.
    Material(MaterialId),
    /// A font's accumulated text (drawn once per font).
    Text(FontId),
}

/// A frame in progress. Draw into it, then call [`Frame::present`].
pub struct Frame<'ctx> {
    ctx: &'ctx mut Context,
    surface_texture: wgpu::SurfaceTexture,
    /// Background clear colour for the scene pass.
    clear: Color,
    /// Sprite runs in submission order: each is a texture + its instances. A run
    /// grows while consecutive `sprite()` calls share a texture; a different
    /// texture (or an interleaved shape/material draw) starts a new run.
    runs: Vec<(TextureId, Vec<SpriteInstance>)>,
    /// Draw commands in submission order.
    cmds: Vec<DrawCmd>,
}

impl Context {
    /// Ensure the default scene target exists (sized to the logical size), and
    /// return its `TargetId` index.
    fn ensure_scene(&mut self) -> usize {
        if self.targets.is_empty() {
            let (w, h) = self.logical_size;
            self.create_target(w, h);
        }
        0
    }

    /// Begin a frame. Acquires the swapchain texture; returns a [`Frame`] to
    /// draw into. `Err(SurfaceLost)` means the caller should reconfigure and
    /// retry next tick.
    pub fn begin_frame(&mut self, clear: Color) -> Result<Frame<'_>, Vk2dError> {
        let surface_texture = self.acquire()?;
        self.ensure_scene();
        self.shapes.begin_frame();
        for font in &mut self.fonts {
            font.begin_frame();
        }
        Ok(Frame {
            ctx: self,
            surface_texture,
            clear,
            runs: Vec::new(),
            cmds: Vec::new(),
        })
    }
}

impl<'ctx> Frame<'ctx> {
    /// Draw a texture at `pos` (top-left) with the given params.
    pub fn sprite(&mut self, texture: TextureId, pos: Point, params: SpriteParams) {
        let Some(tex) = self.ctx.textures.get(texture.0 as usize) else {
            return;
        };
        // Source rect defaults to the whole texture.
        let source_px = params
            .source_px
            .map(|r| [r.x, r.y, r.w, r.h])
            .unwrap_or([0.0, 0.0, tex.width, tex.height]);
        let size = params
            .dest_size
            .map(|d| [d.x, d.y])
            .unwrap_or([source_px[2], source_px[3]]);
        let tint = params.tint;
        // Convert top-left pos to a centre (the batch draws centred quads).
        let center = [pos.x + size[0] * 0.5, pos.y + size[1] * 0.5];
        let instance = SpriteInstance {
            center,
            size,
            source_px,
            tint: [tint.r, tint.g, tint.b, tint.a],
        };
        // Extend the current run if the last draw was sprites of this texture;
        // otherwise open a new run (preserves painter ordering across textures
        // and interleaved shape/material draws).
        if let Some(DrawCmd::Sprites(t, slot)) = self.cmds.last()
            && t.0 == texture.0
        {
            self.runs[*slot].1.push(instance);
            return;
        }
        let slot = self.runs.len();
        self.runs.push((texture, vec![instance]));
        self.cmds.push(DrawCmd::Sprites(texture, slot));
    }

    fn ensure_shape_cmd(&mut self) {
        // All shapes share one accumulated batch drawn in a single call, so we
        // record the Shapes command exactly once (at the first shape draw). v0.1
        // simplification: shapes composite as one layer at their first
        // appearance rather than interleaving per-call with sprites/materials.
        if !self.cmds.iter().any(|c| matches!(c, DrawCmd::Shapes)) {
            self.cmds.push(DrawCmd::Shapes);
        }
    }

    /// Filled rectangle.
    pub fn fill_rect(&mut self, rect: Rect2, color: Color) {
        let ls = self.ctx.logical_size;
        self.ctx
            .shapes
            .fill_rect(rect.x, rect.y, rect.w, rect.h, color, ls);
        self.ensure_shape_cmd();
    }

    /// Rectangle outline.
    pub fn rect_outline(&mut self, rect: Rect2, thickness: f32, color: Color) {
        let ls = self.ctx.logical_size;
        self.ctx
            .shapes
            .rect_outline(rect.x, rect.y, rect.w, rect.h, thickness, color, ls);
        self.ensure_shape_cmd();
    }

    /// Line segment.
    pub fn line(&mut self, from: Point, to: Point, thickness: f32, color: Color) {
        let ls = self.ctx.logical_size;
        self.ctx
            .shapes
            .line(from.x, from.y, to.x, to.y, thickness, color, ls);
        self.ensure_shape_cmd();
    }

    /// Filled circle.
    pub fn circle(&mut self, center: Point, radius: f32, color: Color) {
        let ls = self.ctx.logical_size;
        self.ctx
            .shapes
            .circle(center.x, center.y, radius, color, ls);
        self.ensure_shape_cmd();
    }

    /// Circle outline.
    pub fn circle_outline(&mut self, center: Point, radius: f32, thickness: f32, color: Color) {
        let ls = self.ctx.logical_size;
        self.ctx
            .shapes
            .circle_outline(center.x, center.y, radius, thickness, color, ls);
        self.ensure_shape_cmd();
    }

    /// Filled triangle.
    pub fn triangle(&mut self, a: Point, b: Point, c: Point, color: Color) {
        let ls = self.ctx.logical_size;
        self.ctx
            .shapes
            .triangle((a.x, a.y), (b.x, b.y), (c.x, c.y), color, ls);
        self.ensure_shape_cmd();
    }

    /// Draw a string in `font` at `pos` (top-left of the first line).
    pub fn text(&mut self, font: FontId, text: &str, pos: Point, style: TextStyle) {
        let ls = self.ctx.logical_size;
        let color = [style.color.r, style.color.g, style.color.b, style.color.a];
        let Some(renderer) = self.ctx.fonts.get_mut(font.0 as usize) else {
            return;
        };
        renderer.queue_text(text, [pos.x, pos.y], style.size, color, ls);
        // Record one Text command per font (its glyphs all draw in one call).
        if !self
            .cmds
            .iter()
            .any(|c| matches!(c, DrawCmd::Text(f) if f.0 == font.0))
        {
            self.cmds.push(DrawCmd::Text(font));
        }
    }

    /// Push a named uniform to a material for this frame.
    pub fn set_uniform(&mut self, material: MaterialId, name: &str, value: UniformValue) {
        self.ctx.set_material_uniform(material, name, value);
    }

    /// Bind a loaded texture to a named sampler slot of `material` for this
    /// frame. Unknown material or slot name is a no-op (no panic).
    pub fn bind_material_texture(&mut self, material: MaterialId, slot: &str, texture: TextureId) {
        self.ctx.set_material_texture(material, slot, texture);
    }

    /// Draw a material fullscreen into the scene (the effect shader controls its
    /// own coverage — e.g. a quad vignette). Draws in submission order.
    pub fn material_fullscreen(&mut self, material: MaterialId) {
        if self.ctx.materials.get(material.0 as usize).is_some() {
            self.cmds.push(DrawCmd::Material(material));
        }
    }

    /// Finish the frame: record the scene pass (draws in submission order),
    /// blit the scene to the swapchain with a Nearest upscale, and present.
    pub fn present(self) {
        let (_ctx, surface_texture) = self.render_scene();
        surface_texture.present();
    }

    /// Finish the frame like [`Frame::present`], but paint an egui overlay onto
    /// the swapchain (over the scene) before presenting. The scene and overlay
    /// share one surface texture and are presented together. `build` constructs
    /// the UI. (Feature `egui`.)
    #[cfg(feature = "egui")]
    pub fn present_with_egui(
        self,
        overlay: &mut crate::EguiOverlay,
        window: &std::sync::Arc<winit::window::Window>,
        build: impl FnMut(&egui::Context),
    ) {
        // Record + submit the scene exactly as `present`, but keep the surface
        // texture un-presented so the overlay can paint onto it.
        let surface_view = self.render_scene();
        let (ctx, surface_texture) = surface_view;
        let view = surface_texture
            .texture
            .create_view(&TextureViewDescriptor::default());
        let size = (ctx.config.width, ctx.config.height);
        overlay.paint(
            &ctx.device,
            &ctx.queue,
            window,
            &view,
            [size.0, size.1],
            build,
        );
        surface_texture.present();
    }

    /// Record + submit the scene pass and blit (no present); return the context
    /// and the un-presented surface texture. Shared by `present` and
    /// `present_with_egui`.
    fn render_scene(self) -> (&'ctx mut Context, wgpu::SurfaceTexture) {
        let Frame {
            ctx,
            surface_texture,
            clear,
            runs,
            cmds,
        } = self;
        let ls = ctx.logical_size;

        ctx.shapes.upload(&ctx.device, &ctx.queue);
        for font in &mut ctx.fonts {
            font.upload(&ctx.device, &ctx.queue);
        }
        let staged: Vec<(&[SpriteInstance], (f32, f32))> = runs
            .iter()
            .map(|(tex, instances)| {
                let wh = ctx
                    .textures
                    .get(tex.0 as usize)
                    .map(|g| (g.width, g.height))
                    .unwrap_or((1.0, 1.0));
                (instances.as_slice(), wh)
            })
            .collect();
        let slots = ctx.sprites.stage(&ctx.device, &ctx.queue, &staged, ls);

        let surface_view = surface_texture
            .texture
            .create_view(&TextureViewDescriptor::default());
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("vk2d.frame.encoder"),
            });
        {
            let scene: &SceneTarget = &ctx.targets[0];
            let mut pass = scene.begin_scene_pass(&mut encoder, wgpu_color(clear));
            for cmd in &cmds {
                match cmd {
                    DrawCmd::Shapes => ctx.shapes.draw(&mut pass),
                    DrawCmd::Sprites(tex, slot) => {
                        if let (Some(run), Some(gpu)) =
                            (slots.get(*slot), ctx.textures.get(tex.0 as usize))
                        {
                            ctx.sprites.draw_run(&mut pass, *run, gpu);
                        }
                    }
                    DrawCmd::Material(mat) => {
                        if let Some(m) = ctx.materials.get(mat.0 as usize) {
                            pass.set_pipeline(&m.pipeline);
                            pass.set_bind_group(0, &m.bind_group, &[]);
                            pass.draw(0..3, 0..1);
                        }
                    }
                    DrawCmd::Text(font) => {
                        if let Some(f) = ctx.fonts.get(font.0 as usize) {
                            f.draw(&mut pass);
                        }
                    }
                }
            }
        }
        ctx.targets[0].blit_to(&mut encoder, &surface_view);
        ctx.queue.submit(std::iter::once(encoder.finish()));
        (ctx, surface_texture)
    }
}
