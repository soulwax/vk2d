//! The immediate-mode frame API. `Context::begin_frame` returns a `Frame`; the
//! app issues draw calls (sprites, shapes, text, materials) in any order, then
//! `present()` records the passes and shows the result. The library batches
//! internally, so callers think in draw calls, not pipelines.

use wgpu::{CommandEncoderDescriptor, TextureViewDescriptor};

use crate::color::{Color, Point, Rect2, SpriteParams, TextStyle};
use crate::sprite::{SpriteDrawRun, SpriteInstance, SpriteSource};
use crate::target::SceneTarget;
use crate::view::View2;
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
    /// A sprite run: `(source, run_slot)` where `run_slot` indexes the frame's
    /// ordered `runs` list (each entry is instances + that source). `source`
    /// is either an uploaded texture ([`Frame::sprite`]) or a finished render
    /// target's own color output ([`Frame::target_sprite`]).
    Sprites(SpriteSource, usize),
    /// The shape batch (all accumulated shapes; drawn once).
    Shapes,
    /// A material drawn fullscreen into the scene.
    Material(MaterialId),
    /// A font's accumulated text (drawn once per font).
    Text(FontId),
}

/// Recycled CPU-side frame recording storage owned by [`Context`] between
/// frames. The vectors keep their high-water capacities, including one
/// instance vector per concurrently recorded sprite run.
#[derive(Default)]
pub(crate) struct FrameScratch {
    runs: Vec<SpriteDrawRun>,
    cmds: Vec<DrawCmd>,
    spare_instances: Vec<Vec<SpriteInstance>>,
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

impl FrameDest {
    /// The `targets` index this frame's scene pass renders into: always `0`
    /// for `Swapchain` (the reserved scene target — see
    /// [`Context::ensure_scene`]), or the chosen index for `Offscreen`.
    /// [`Frame::target_sprite`]'s self-sample guard compares a requested
    /// target against this.
    fn render_index(&self) -> usize {
        match self {
            FrameDest::Swapchain { .. } => 0,
            FrameDest::Offscreen { index } => *index,
        }
    }
}

/// Pure predicate behind `target_sprite`'s self-sample guard: would drawing
/// `requested_index` sample the same target this frame is currently rendering
/// into (`render_index`)? Extracted so the guard's logic — the crux of the
/// contract, since getting it wrong is a wgpu COLOR_TARGET/RESOURCE aliasing
/// crash rather than a graceful no-op — is unit-testable without a GPU
/// device, a window, or a live `Frame`.
fn is_self_sampling(render_index: usize, requested_index: usize) -> bool {
    render_index == requested_index
}

/// A frame in progress. Draw into it, then call [`Frame::present`] (swapchain
/// frames) or [`Frame::finish`] (offscreen target frames — does not present).
pub struct Frame<'ctx> {
    ctx: &'ctx mut Context,
    dest: FrameDest,
    /// Background clear colour for the scene pass.
    clear: Color,
    /// Sprite runs in submission order: each is a source + its instances. A run
    /// grows while consecutive `sprite()`/`target_sprite()` calls share a
    /// source; a different source (or an interleaved shape/material draw)
    /// starts a new run.
    runs: Vec<SpriteDrawRun>,
    /// Draw commands in submission order.
    cmds: Vec<DrawCmd>,
    /// Instance buffers left over from prior frame runs, reused when a new run
    /// starts instead of allocating a fresh one-element vector.
    spare_instances: Vec<Vec<SpriteInstance>>,
    /// Avoid an O(command_count) scan on every shape primitive.
    shape_cmd_recorded: bool,
    /// The current 2D view: a CPU-side affine applied to every draw call's
    /// coordinates at record time, before the pixel→clip conversion. Defaults
    /// to [`View2::identity`] (no transform).
    view: View2,
    /// The pixel size of the render target this frame draws into — the
    /// reference size for EVERY logical→clip conversion (`logical_to_clip`),
    /// used by the shape/sprite/text record + stage paths INSTEAD of
    /// [`Context::logical_size`].
    ///
    /// For a swapchain frame this is `targets[0]` (the reserved scene target,
    /// created at `logical_size` by [`Context::ensure_scene`]), so it EQUALS
    /// `logical_size` and nothing changes. For an offscreen
    /// [`Context::begin_target_frame`] whose target was created at a size
    /// OTHER than `logical_size` (an app's supersampled/render-scaled scene
    /// buffer, a half-size bloom ping-pong target), this is that target's own
    /// dimensions — so a caller that windows a world view onto the target's
    /// real pixel extent (`View2::window(.., out = target_size, ..)`) has its
    /// coordinates converted to clip against the SAME extent, instead of being
    /// doubled/halved by a `logical_size` that no longer matches the
    /// attachment. Without this, drawing into a 2× scene target placed every
    /// off-centre sprite/shape at 2× NDC (off-screen); centre content survived
    /// only because `0 * 2 == 0`.
    output_size: (u32, u32),
}

impl Context {
    /// Ensure the default scene target (`targets[0]`) exists, sized to the
    /// logical size, and return its index (always 0).
    ///
    /// This reserves index 0 for the scene the swapchain path renders into and
    /// blits. It is called both at the start of a frame AND by the public
    /// [`Context::create_target`], so an app that creates its own target BEFORE
    /// its first frame cannot accidentally claim index 0 and end up rendering
    /// the frame into the very target it also samples (which wgpu rejects as a
    /// COLOR_TARGET/RESOURCE usage conflict).
    pub(crate) fn ensure_scene(&mut self) -> usize {
        if self.targets.is_empty() {
            let (w, h) = self.logical_size;
            self.push_target(w, h);
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
        // The swapchain frame renders into the reserved scene target (index 0),
        // created at `logical_size` — so its output size EQUALS `logical_size`
        // and the clip conversion is unchanged. Reading it from the target
        // (not hardcoding `logical_size`) keeps a single source of truth with
        // the offscreen path below.
        let output_size = self.targets[0].size();
        let scratch = std::mem::take(&mut self.frame_scratch);
        Ok(Frame {
            ctx: self,
            dest: FrameDest::Swapchain { surface_texture },
            clear,
            runs: scratch.runs,
            cmds: scratch.cmds,
            spare_instances: scratch.spare_instances,
            shape_cmd_recorded: false,
            view: View2::identity(),
            output_size,
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
        // Clip conversions must reference THIS target's real pixel size, not
        // `logical_size` — an app-created render-scaled/supersampled target is
        // sized independently of `logical_size`, and mapping world coordinates
        // against the wrong extent misplaces every off-centre draw (the SSAA
        // scene-target sprite-invisibility bug — see `Frame::output_size`).
        let output_size = self.targets[index].size();
        let scratch = std::mem::take(&mut self.frame_scratch);
        Ok(Frame {
            ctx: self,
            dest: FrameDest::Offscreen { index },
            clear,
            runs: scratch.runs,
            cmds: scratch.cmds,
            spare_instances: scratch.spare_instances,
            shape_cmd_recorded: false,
            view: View2::identity(),
            output_size,
        })
    }
}

impl<'ctx> Frame<'ctx> {
    /// Replace the current view (see [`View2`]). Affects every draw call
    /// recorded after this point; draws already recorded keep the
    /// coordinates they were transformed with at record time.
    pub fn set_view(&mut self, view: View2) {
        self.view = view;
    }

    /// Reset the view back to [`View2::identity`] (no transform).
    pub fn reset_view(&mut self) {
        self.view = View2::identity();
    }

    /// Draw a texture at `pos` (top-left) with the given params.
    pub fn sprite(&mut self, texture: TextureId, pos: Point, params: SpriteParams) {
        let Some(tex) = self.ctx.textures.get(texture.0 as usize) else {
            return;
        };
        let whole_source = [0.0, 0.0, tex.width, tex.height];
        let instance = self.build_sprite_instance(pos, params, whole_source);
        self.push_sprite_instance(
            SpriteSource::Texture(texture),
            (tex.width, tex.height),
            instance,
        );
    }

    /// Draw a finished render target's own color output as a positioned
    /// sprite at `pos` (top-left) with the given params — compositing a prior
    /// offscreen pass into this frame's scene (e.g. a blur or bloom target).
    ///
    /// `target` must have already been rendered and finished in an earlier
    /// frame or an earlier pass in the same sequence; this call does not
    /// itself schedule any rendering (mirrors [`Frame::bind_material_target`]).
    /// Unknown target id is a no-op (no panic).
    ///
    /// **Self-sample guard:** drawing the target THIS frame is currently
    /// rendering into is a caller error, not a wgpu validation crash waiting
    /// to happen (sampling a texture that's simultaneously bound as the
    /// active color attachment is a COLOR_TARGET/RESOURCE usage conflict wgpu
    /// rejects at submit time). Both frame flavours are guarded: an offscreen
    /// frame ([`Context::begin_target_frame`]) checks `target` against its own
    /// `FrameDest::Offscreen { index }`, and a swapchain frame
    /// ([`Context::begin_frame`]) checks it against the reserved scene target
    /// (index 0 — see [`Context::ensure_scene`]), since that's the target a
    /// swapchain frame renders into. Either match logs at debug level and
    /// returns without queuing a draw.
    pub fn target_sprite(&mut self, target: TargetId, pos: Point, params: SpriteParams) {
        let requested_index = target.0 as usize;
        if is_self_sampling(self.dest.render_index(), requested_index) {
            #[cfg(debug_assertions)]
            eprintln!(
                "vk2d: target_sprite({requested_index}) ignored — that target is the one this frame is currently rendering into (self-sample guard)"
            );
            return;
        }
        let Some(scene) = self.ctx.targets.get(requested_index) else {
            return;
        };
        let (w, h) = scene.size();
        let whole_source = [0.0, 0.0, w as f32, h as f32];
        let instance = self.build_sprite_instance(pos, params, whole_source);
        self.push_sprite_instance(SpriteSource::Target(target), (w as f32, h as f32), instance);
    }

    /// Build a [`SpriteInstance`] from `pos` (top-left) + `params`, applying
    /// the current [`View2`] exactly like [`Frame::sprite`] — shared by
    /// `sprite` and `target_sprite` so a target-as-texture blit transforms
    /// identically to an ordinary texture sprite. `whole_source` is the
    /// source's full-size rect (`[0, 0, w, h]`), substituted when
    /// `params.source_px` is `None`.
    fn build_sprite_instance(
        &self,
        pos: Point,
        params: SpriteParams,
        whole_source: [f32; 4],
    ) -> SpriteInstance {
        // Source rect defaults to the whole source.
        let mut source_px = params
            .source_px
            .map(|r| [r.x, r.y, r.w, r.h])
            .unwrap_or(whole_source);
        let size = params
            .dest_size
            .map(|d| [d.x, d.y])
            .unwrap_or([source_px[2], source_px[3]]);
        let tint = params.tint;
        // A Y-up view (negative scale.y) flips the source vertically on
        // screen; XOR that correction into flip_y so the texture stays
        // upright, matching how a Y-up camera samples a normally-oriented
        // texture atlas. Both flip_x and the (possibly corrected) flip_y are
        // realized by mirroring the source rect on that axis (swap its min
        // and max edge), which the vertex builder turns into flipped UVs
        // without needing a dedicated flip field on `SpriteInstance`.
        let flip_x = params.flip_x;
        let flip_y = params.flip_y ^ (self.view.scale.y < 0.0);
        if flip_x {
            source_px[0] += source_px[2];
            source_px[2] = -source_px[2];
        }
        if flip_y {
            source_px[1] += source_px[3];
            source_px[3] = -source_px[3];
        }
        // Convert top-left pos to a centre (the batch draws centred quads),
        // then transform the centre through the view; scale the size by the
        // view's per-axis scale magnitude so it stays correctly sized (the
        // flip itself is carried by the mirrored source rect above, not by a
        // negative size).
        let center = [pos.x + size[0] * 0.5, pos.y + size[1] * 0.5];
        let center = self.view.apply(Point {
            x: center[0],
            y: center[1],
        });
        let size = [
            size[0] * self.view.scale.x.abs(),
            size[1] * self.view.scale.y.abs(),
        ];
        SpriteInstance {
            center: [center.x, center.y],
            size,
            source_px,
            tint: [tint.r, tint.g, tint.b, tint.a],
        }
    }

    /// Append `instance` to the run for `source`: extend the current run if
    /// the last draw shared this exact source; otherwise open a new run
    /// (preserves painter ordering across sources and interleaved
    /// shape/material draws).
    fn push_sprite_instance(
        &mut self,
        source: SpriteSource,
        atlas_size: (f32, f32),
        instance: SpriteInstance,
    ) {
        if let Some(DrawCmd::Sprites(s, slot)) = self.cmds.last()
            && *s == source
        {
            self.runs[*slot].instances.push(instance);
            return;
        }
        let slot = self.runs.len();
        let mut instances = self.spare_instances.pop().unwrap_or_default();
        instances.push(instance);
        self.runs.push(SpriteDrawRun {
            atlas_size,
            instances,
        });
        self.cmds.push(DrawCmd::Sprites(source, slot));
    }

    fn ensure_shape_cmd(&mut self) {
        // All shapes share one accumulated batch drawn in a single call, so we
        // record the Shapes command exactly once (at the first shape draw). v0.1
        // simplification: shapes composite as one layer at their first
        // appearance rather than interleaving per-call with sprites/materials.
        if !self.shape_cmd_recorded {
            self.cmds.push(DrawCmd::Shapes);
            self.shape_cmd_recorded = true;
        }
    }

    /// Transform a [`Rect2`] (given as top-left + size) through the current
    /// view: map the top-left corner and scale the size by the view's
    /// per-axis scale magnitude, then slide the origin back by the negative
    /// half when an axis flips — so a view with negative scale on either axis
    /// still yields a rect with positive width/height. Width/height are
    /// derived from `rect.w`/`rect.h` directly (never by re-differencing two
    /// transformed corners), so an identity view reproduces the exact input
    /// rect with no floating-point round-trip.
    fn view_rect(&self, rect: Rect2) -> Rect2 {
        let top_left = self.view.apply(Point {
            x: rect.x,
            y: rect.y,
        });
        let w = rect.w * self.view.scale.x.abs();
        let h = rect.h * self.view.scale.y.abs();
        // If the axis flips, the transformed top-left corner is actually the
        // rect's right/bottom edge on screen — slide back by the size so
        // (x, y) stays the min corner.
        let x = if self.view.scale.x < 0.0 {
            top_left.x - w
        } else {
            top_left.x
        };
        let y = if self.view.scale.y < 0.0 {
            top_left.y - h
        } else {
            top_left.y
        };
        Rect2 { x, y, w, h }
    }

    /// Filled rectangle.
    pub fn fill_rect(&mut self, rect: Rect2, color: Color) {
        let ls = self.output_size;
        let r = self.view_rect(rect);
        self.ctx.shapes.fill_rect(r.x, r.y, r.w, r.h, color, ls);
        self.ensure_shape_cmd();
    }

    /// Rectangle outline.
    pub fn rect_outline(&mut self, rect: Rect2, thickness: f32, color: Color) {
        let ls = self.output_size;
        let r = self.view_rect(rect);
        let thickness = thickness * self.view.length_scale();
        self.ctx
            .shapes
            .rect_outline(r.x, r.y, r.w, r.h, thickness, color, ls);
        self.ensure_shape_cmd();
    }

    /// Line segment.
    pub fn line(&mut self, from: Point, to: Point, thickness: f32, color: Color) {
        let ls = self.output_size;
        let from = self.view.apply(from);
        let to = self.view.apply(to);
        let thickness = thickness * self.view.length_scale();
        self.ctx
            .shapes
            .line(from.x, from.y, to.x, to.y, thickness, color, ls);
        self.ensure_shape_cmd();
    }

    /// Filled circle.
    pub fn circle(&mut self, center: Point, radius: f32, color: Color) {
        let ls = self.output_size;
        let center = self.view.apply(center);
        let radius = radius * self.view.length_scale();
        self.ctx
            .shapes
            .circle(center.x, center.y, radius, color, ls);
        self.ensure_shape_cmd();
    }

    /// Circle outline.
    pub fn circle_outline(&mut self, center: Point, radius: f32, thickness: f32, color: Color) {
        let ls = self.output_size;
        let center = self.view.apply(center);
        let radius = radius * self.view.length_scale();
        let thickness = thickness * self.view.length_scale();
        self.ctx
            .shapes
            .circle_outline(center.x, center.y, radius, thickness, color, ls);
        self.ensure_shape_cmd();
    }

    /// Filled triangle.
    pub fn triangle(&mut self, a: Point, b: Point, c: Point, color: Color) {
        let ls = self.output_size;
        let a = self.view.apply(a);
        let b = self.view.apply(b);
        let c = self.view.apply(c);
        self.ctx
            .shapes
            .triangle((a.x, a.y), (b.x, b.y), (c.x, c.y), color, ls);
        self.ensure_shape_cmd();
    }

    /// Draw a string in `font` at `pos` (top-left of the first line).
    pub fn text(&mut self, font: FontId, text: &str, pos: Point, style: TextStyle) {
        let ls = self.output_size;
        let color = [style.color.r, style.color.g, style.color.b, style.color.a];
        let pos = self.view.apply(pos);
        let style = TextStyle {
            size: style.size * self.view.length_scale(),
            ..style
        };
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
            mut spare_instances,
            shape_cmd_recorded: _,
            view: _,
            output_size,
        } = self;
        // Stage sprite geometry against the same output extent the record-path
        // shape/text conversions used (`Frame::output_size`), NOT
        // `ctx.logical_size` — for a render-scaled offscreen target the two
        // differ, and staging sprites against `logical_size` while shapes used
        // the target size (or vice versa) would misplace one relative to the
        // other. Both now reference the target's real pixel size.
        let ls = output_size;

        ctx.shapes.upload(&ctx.device, &ctx.queue);
        for font in &mut ctx.fonts {
            font.upload(&ctx.device, &ctx.queue);
        }
        let target_index = match &dest {
            FrameDest::Swapchain { .. } => 0,
            FrameDest::Offscreen { index } => *index,
        };

        // Pre-warm every target-sprite bind group this frame's `cmds` need,
        // BEFORE the scene render pass borrows `ctx.targets` (and therefore
        // `ctx`) immutably for its lifetime. `target_sprite_bind_group` takes
        // `&mut Context`, which the borrow checker cannot interleave with a
        // live `RenderPass` borrowed from `&ctx.targets[target_index]`.
        for cmd in &cmds {
            if let DrawCmd::Sprites(SpriteSource::Target(target), _) = cmd {
                let index = target.0 as usize;
                if index < ctx.targets.len() {
                    ctx.target_sprite_bind_group(index);
                }
            }
        }

        ctx.sprites.stage(&ctx.device, &ctx.queue, &runs, ls);

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
                    DrawCmd::Sprites(source, slot) => {
                        let Some(run) = ctx.sprites.staged_run(*slot) else {
                            continue;
                        };
                        match source {
                            SpriteSource::Texture(tex) => {
                                if let Some(gpu) = ctx.textures.get(tex.0 as usize) {
                                    ctx.sprites.draw_run(&mut pass, run, gpu);
                                }
                            }
                            SpriteSource::Target(target) => {
                                let index = target.0 as usize;
                                // Already built by the pre-warm pass above (it
                                // ran before `pass` — which borrows
                                // `ctx.targets` — existed), so this is a plain
                                // read: no `&mut ctx` call can happen while
                                // `pass` is alive.
                                if let Some(Some(bind_group)) =
                                    ctx.target_sprite_bind_groups.get(index)
                                {
                                    ctx.sprites
                                        .draw_run_with_bind_group(&mut pass, run, bind_group);
                                }
                            }
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
        let mut runs = runs;
        for mut run in runs.drain(..) {
            run.instances.clear();
            spare_instances.push(run.instances);
        }
        let mut cmds = cmds;
        cmds.clear();
        ctx.frame_scratch = FrameScratch {
            runs,
            cmds,
            spare_instances,
        };
        (ctx, surface_texture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `target_sprite`'s self-sample guard is the crux of its contract:
    // drawing the target a frame is currently rendering INTO must no-op
    // (never reach wgpu, which would reject it as a COLOR_TARGET/RESOURCE
    // usage conflict at submit time). vk2d has no headless `wgpu::Device`
    // test harness (`Context::new` requires a real `winit::Window`, and
    // `FrameDest::Swapchain` holds a real `wgpu::SurfaceTexture` that can't be
    // constructed off-GPU), so per the task brief's fallback these test the
    // extracted pure predicate directly rather than a live `Frame` — both
    // frame flavours it stands in for (`Offscreen` and the reserved-scene
    // `Swapchain` case) are exercised via `render_index`'s two arms.

    #[test]
    fn offscreen_frame_guards_its_own_target() {
        // Frame::begin_target_frame(target) — render_index() is that target's
        // index. Drawing target 3 while rendering into target 3 must guard.
        let render_index = FrameDest::Offscreen { index: 3 }.render_index();
        assert!(is_self_sampling(render_index, 3));
        // A different target is unaffected.
        assert!(!is_self_sampling(render_index, 0));
        assert!(!is_self_sampling(render_index, 4));
    }

    #[test]
    fn swapchain_frame_guards_the_reserved_scene_target() {
        // A swapchain frame (Context::begin_frame) always renders into the
        // reserved scene target, index 0 (Context::ensure_scene). We can't
        // construct a real `FrameDest::Swapchain` off-GPU (it owns a
        // `wgpu::SurfaceTexture`), so this asserts the documented invariant
        // `render_index() == 0` directly against the predicate a swapchain
        // frame's `target_sprite` call would evaluate.
        let render_index = 0usize; // FrameDest::Swapchain::render_index()
        assert!(is_self_sampling(render_index, 0));
        // Any other target is a legitimate, un-guarded draw.
        assert!(!is_self_sampling(render_index, 1));
        assert!(!is_self_sampling(render_index, 7));
    }

    #[test]
    fn is_self_sampling_is_pure_index_equality() {
        for (render_index, requested, expected) in
            [(0, 0, true), (1, 1, true), (2, 3, false), (5, 2, false)]
        {
            assert_eq!(is_self_sampling(render_index, requested), expected);
        }
    }
}
