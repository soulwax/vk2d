//! Diagnostic repro for the `target_sprite` black-readback bug: render a bright
//! shape into a target under a NON-identity (Y-up) `View2`, then read the target
//! back — first DIRECTLY (its own texels) and then through `target_sprite` into
//! a second target — and print the probed pixels. Compares against an
//! identity-view control so the difference is unambiguous.
//!
//! Run: `cargo run -p vk2d --example target_view_readback -- --frames 1`
//! (exits 0). Prints PASS/FAIL lines; not a gate, a diagnosis harness.

use std::sync::Arc;

use vk2d::{Backend, Color, Context, ContextConfig, Point, Rect2, SpriteParams, TargetId, View2};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const LOGICAL: (u32, u32) = (320, 180);
// A bright, unambiguous fill colour drawn into the middle of the target.
const FILL: Color = Color {
    r: 0.1,
    g: 0.9,
    b: 0.2,
    a: 1.0,
};

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App {
        window: None,
        ctx: None,
        done: false,
    };
    event_loop.run_app(&mut app).expect("run");
}

struct App {
    window: Option<Arc<Window>>,
    ctx: Option<Context>,
    done: bool,
}

/// A Y-up world view that windows a `(w,h)` world rect centred on the target's
/// own centre onto the target — the shape `set_world_view` produces in the
/// consuming game (see `renderer_vk::view2_from_world`).
fn y_up_view(w: f32, h: f32) -> View2 {
    View2::window(
        Point::new(0.0, h), // top-left in Y-up source (top edge = max y)
        Point::new(w, h),
        Point::new(w, h),
        true,
    )
}

fn probe(tag: &str, ctx: &Context, target: TargetId, x: u32, y: u32) -> [u8; 4] {
    let p = ctx.read_target_pixel(target, x, y).expect("pixel in range");
    let black = p[0] < 8 && p[1] < 8 && p[2] < 8;
    println!(
        "{tag}: pixel({x},{y}) = {p:?}  [{}]",
        if black { "BLACK" } else { "has colour" }
    );
    p
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.ctx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("vk2d — target_view_readback")
            .with_inner_size(LogicalSize::new(LOGICAL.0 as f64, LOGICAL.1 as f64));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        self.window = Some(window.clone());
        let ctx = Context::new(
            window,
            ContextConfig {
                logical_size: LOGICAL,
                prefer_backend: Backend::Vulkan,
            },
        )
        .expect("context");
        println!("vk2d {} on {}", vk2d::version(), ctx.adapter_info());
        self.ctx = Some(ctx);
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let WindowEvent::CloseRequested = event {
            self.done = true;
        }
        if !matches!(event, WindowEvent::RedrawRequested) || self.done {
            return;
        }
        let Some(ctx) = self.ctx.as_mut() else { return };
        let (w, h) = (LOGICAL.0 as f32, LOGICAL.1 as f32);
        let (cx, cy) = (LOGICAL.0 / 2, LOGICAL.1 / 2);

        // Two source targets: one drawn under identity, one under a Y-up view.
        let src_ident: TargetId = ctx.create_target(LOGICAL.0, LOGICAL.1);
        let src_view: TargetId = ctx.create_target(LOGICAL.0, LOGICAL.1);
        // Two readback destinations for the target_sprite composite.
        let dst_ident: TargetId = ctx.create_target(LOGICAL.0, LOGICAL.1);
        let dst_view: TargetId = ctx.create_target(LOGICAL.0, LOGICAL.1);

        // Draw a SMALL marker at a known source position into each target. A
        // full-cover fill would mask any positioning bug, so use a small rect
        // whose landing spot differs between identity and the Y-up view — this
        // is what reveals whether the write side puts content where expected.
        //
        // Identity: rect top-left at source (cx, 20) → lands at (cx, 20).
        // Y-up view centred on the target: the SAME source coords map through
        // the flip; we probe both the identity landing spot and the mirrored
        // one to see where the y-up content actually ends up.
        let marker = Rect2::new(cx as f32 - 20.0, 20.0, 40.0, 40.0);
        {
            let mut f = ctx.begin_target_frame(src_ident, Color::BLACK).unwrap();
            f.fill_rect(marker, FILL);
            f.finish();
        }
        {
            let mut f = ctx.begin_target_frame(src_view, Color::BLACK).unwrap();
            f.set_view(y_up_view(w, h));
            f.fill_rect(marker, FILL);
            f.finish();
        }

        // Read the source targets DIRECTLY (bypassing target_sprite): where did
        // the content actually land?
        println!("--- direct target texels (write side) ---");
        probe("src_ident @ (cx,40) top   ", ctx, src_ident, cx, 40);
        probe("src_ident @ (cx,140) bot  ", ctx, src_ident, cx, 140);
        probe("src_view  @ (cx,40) top   ", ctx, src_view, cx, 40);
        probe("src_view  @ (cx,140) bot  ", ctx, src_view, cx, 140);

        // Now composite each source into a destination via target_sprite (the
        // read side under test), under identity view, then read the dest back.
        for (src, dst) in [(src_ident, dst_ident), (src_view, dst_view)] {
            let mut f = ctx.begin_target_frame(dst, Color::BLACK).unwrap();
            f.target_sprite(
                src,
                Point::new(0.0, 0.0),
                SpriteParams {
                    dest_size: Some(Point::new(w, h)),
                    ..Default::default()
                },
            );
            f.finish();
        }
        println!("--- target_sprite composite (read side) ---");
        probe("dst_ident @ (cx,40) top   ", ctx, dst_ident, cx, 40);
        probe("dst_ident @ (cx,140) bot  ", ctx, dst_ident, cx, 140);
        probe("dst_view  @ (cx,40) top   ", ctx, dst_view, cx, 40);
        probe("dst_view  @ (cx,140) bot  ", ctx, dst_view, cx, 140);

        // SSAA case: a source target 2x the dest size, drawn under a Y-up view
        // sized to ITS OWN texels, composited DOWN to dest size via
        // target_sprite (Nearest). Mirrors the game's supersampled scene → screen
        // composite. Probe the target centre (full-cover fill so any content is
        // visible — here we test the downscale sampling path, not positioning).
        let ss = ctx.create_target(LOGICAL.0 * 2, LOGICAL.1 * 2);
        let ss_dst = ctx.create_target(LOGICAL.0, LOGICAL.1);
        {
            let mut f = ctx.begin_target_frame(ss, Color::BLACK).unwrap();
            f.set_view(y_up_view(w * 2.0, h * 2.0));
            f.fill_rect(Rect2::new(0.0, 0.0, w * 2.0, h * 2.0), FILL);
            f.finish();
        }
        {
            let mut f = ctx.begin_target_frame(ss_dst, Color::BLACK).unwrap();
            f.target_sprite(
                ss,
                Point::new(0.0, 0.0),
                SpriteParams {
                    dest_size: Some(Point::new(w, h)),
                    ..Default::default()
                },
            );
            f.finish();
        }
        println!("--- SSAA (2x source) target_sprite downscale ---");
        probe("ss_dst @ centre           ", ctx, ss_dst, cx, cy);

        // Swapchain composite: the game's real screen pass is a swapchain
        // begin_frame that target_sprites the (Y-up-drawn) scene back. A
        // swapchain frame renders into the reserved scene target (index 0) and
        // then blits it to the surface, so read index 0 back after present to
        // see what the on-screen composite produced. `TargetId(0)` is that
        // reserved target (created by the first create_target/ensure_scene).
        {
            let mut f = ctx.begin_frame(Color::BLACK).unwrap();
            f.target_sprite(
                src_view,
                Point::new(0.0, 0.0),
                SpriteParams {
                    dest_size: Some(Point::new(w, h)),
                    ..Default::default()
                },
            );
            f.present();
        }
        println!("--- swapchain composite (scene target index 0) ---");
        probe("screen @ (cx,140) bot     ", ctx, TargetId(0), cx, 140);
        probe("screen @ (cx,40) top      ", ctx, TargetId(0), cx, 40);

        self.done = true;
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            event_loop.exit();
            return;
        }
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}
