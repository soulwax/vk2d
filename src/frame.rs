//! The immediate-mode frame API. `Context::begin_frame` returns a `Frame`; the
//! app issues draw calls (sprites, shapes, text, materials) in any order, then
//! `present()` records the passes and shows the result. The library batches
//! internally, so callers think in draw calls, not pipelines.

use wgpu::{CommandEncoderDescriptor, TextureViewDescriptor};

use crate::color::{Color, Point, Rect2, SpriteParams, TextStyle};
use crate::sprite::SpriteInstance;
use crate::target::SceneTarget;
use crate::{Context, FontId, MaterialId, TargetId, TextureId, UniformValue, Vk2dError};

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

/// Where a [`Frame`] renders its scene pass to, and how it finishes.
///
/// A frame either targets the swapchain (the normal `begin_frame` path: the
/// scene target is blitted to an acquired surface texture, which `present()`
/// then shows) or a chosen offscreen [`TargetId`] (`begin_target_frame`: the
/// scene pass renders straight into that target and the frame finishes by
/// submitting the encoder — there is no swapchain acquire, blit, or present).
enum FrameDest {
    /// Render into `targets[0]` and blit to `surface_texture` on finish.
    Swapchain {
        surface_texture: wgpu::SurfaceTexture,
    },
    /// Render straight into `targets[index]`; finishing only submits.
    Offscreen { index: usize },
}

/// A frame in progress. Draw into it, then call [`Frame::present`] (swapchain
/// frames) or [`Frame::finish`] (offscreen target frames — does not present).
pub struct Frame<'ctx> {
    ctx: &'ctx mut Context,
    dest: FrameDest,
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
            dest: FrameDest::Swapchain { surface_texture },
            clear,
            runs: Vec::new(),
            cmds: Vec::new(),
        })
    }

    /// Begin a frame that renders into the offscreen `target` instead of the
    /// swapchain — the multi-pass entry point (bloom ping-pong, scene-reading
    /// post-process). `Err(Vk2dError::UnknownTarget)` if `target` was never
    /// created via [`Context::create_target`]; never panics.
    ///
    /// The returned [`Frame`] does **not** present to the window: finish it
    /// with [`Frame::finish`] (or drop it, which finishes implicitly), then
    /// read the target back via [`Frame::bind_material_target`] in a later
    /// pass. Calling [`Frame::present`] on a target frame is also safe — it
    /// finishes the same way and simply has nothing to show.
    pub fn begin_target_frame(
        &mut self,
        target: TargetId,
        clear: Color,
    ) -> Result<Frame<'_>, Vk2dError> {
        let index = target.0 as usize;
        if index >= self.targets.len() {
            return Err(Vk2dError::UnknownTarget);
        }
        self.shapes.begin_frame();
        for font in &mut self.fonts {
            font.begin_frame();
        }
        Ok(Frame {
            ctx: self,
            dest: FrameDest::Offscreen { index },
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

    /// Bind a render target's finished color texture to a named sampler slot
    /// of `material` for this frame — reads back a prior offscreen pass (e.g.
    /// a bloom target) as this frame's material input. Unknown material,
    /// slot name, or target id is a no-op (no panic). The target must already
    /// have been rendered and finished in an earlier frame (or an earlier
    /// pass within the same frame sequence) — binding does not itself
    /// schedule any rendering.
    pub fn bind_material_target(&mut self, material: MaterialId, slot: &str, target: TargetId) {
        self.ctx.set_material_target(material, slot, target);
    }

    /// Draw a material fullscreen into the scene (the effect shader controls its
    /// own coverage — e.g. a quad vignette). Draws in submission order.
    pub fn material_fullscreen(&mut self, material: MaterialId) {
        if self.ctx.materials.get(material.0 as usize).is_some() {
            self.cmds.push(DrawCmd::Material(material));
        }
    }

    /// Finish the frame: record the scene pass (draws in submission order).
    ///
    /// - Swapchain frames (from [`Context::begin_frame`]): blit the scene to
    ///   the swapchain with a Nearest upscale, then present.
    /// - Offscreen target frames (from [`Context::begin_target_frame`]):
    ///   submit only — there is no swapchain to blit or present. Calling
    ///   `present` on a target frame is safe (matches the type's normal
    ///   finishing verb) but shows nothing; see [`Frame::finish`] for the
    ///   offscreen-flavoured name.
    pub fn present(self) {
        let (_ctx, surface_texture) = self.render_scene();
        if let Some(surface_texture) = surface_texture {
            surface_texture.present();
        }
    }

    /// Finish an offscreen target frame (from [`Context::begin_target_frame`]):
    /// records the scene pass into its target and submits. An alias for
    /// [`Frame::present`] with a name that reads correctly for target frames,
    /// which never touch the swapchain.
    pub fn finish(self) {
        self.present();
    }

    /// Finish the frame like [`Frame::present`], but paint an egui overlay onto
    /// the swapchain (over the scene) before presenting. The scene and overlay
    /// share one surface texture and are presented together. `build` constructs
    /// the UI. (Feature `egui`.) Only meaningful for swapchain frames; called on
    /// an offscreen target frame it finishes the same way and skips painting
    /// (there is no surface to paint onto).
    #[cfg(feature = "egui")]
    pub fn present_with_egui(
        self,
        overlay: &mut crate::EguiOverlay,
        window: &std::sync::Arc<winit::window::Window>,
        build: impl FnMut(&egui::Context),
    ) {
        // Record + submit the scene exactly as `present`, but keep the surface
        // texture un-presented so the overlay can paint onto it.
        let (ctx, surface_texture) = self.render_scene();
        let Some(surface_texture) = surface_texture else {
            // Offscreen target frame: nothing to paint an overlay onto.
            return;
        };
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

    /// Record + submit the scene pass (and, for swapchain frames, the blit);
    /// return the context and the un-presented surface texture (`None` for an
    /// offscreen target frame — there is nothing to present). Shared by
    /// `present` and `present_with_egui`.
    fn render_scene(self) -> (&'ctx mut Context, Option<wgpu::SurfaceTexture>) {
        let Frame {
            ctx,
            dest,
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

        let target_index = match &dest {
            FrameDest::Swapchain { .. } => 0,
            FrameDest::Offscreen { index } => *index,
        };

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("vk2d.frame.encoder"),
            });
        {
            let scene: &SceneTarget = &ctx.targets[target_index];
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

        let surface_texture = match dest {
            FrameDest::Swapchain { surface_texture } => {
                let surface_view = surface_texture
                    .texture
                    .create_view(&TextureViewDescriptor::default());
                ctx.targets[target_index].blit_to(&mut encoder, &surface_view);
                Some(surface_texture)
            }
            FrameDest::Offscreen { .. } => None,
        };

        ctx.queue.submit(std::iter::once(encoder.finish()));
        (ctx, surface_texture)
    }
}
