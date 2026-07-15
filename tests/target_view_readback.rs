//! Regression: `Frame::target_sprite` reads back a render target correctly
//! regardless of the `View2` under which that target's content was drawn.
//!
//! Motivation: a consumer (EchoWarrior's vk shell) reported that compositing
//! the scene target back to the screen via `target_sprite` rendered BLACK when
//! that target had been drawn under a non-identity (Y-up) world view. This test
//! reproduces the exact shape of that path inside vk2d — draw a marker into a
//! target under a Y-up `View2::window`, then `target_sprite` it into a second
//! target — and asserts the composite is faithful (content at the mirrored
//! position, black elsewhere), pinning that vk2d's read-back is view-agnostic.
//! It also covers the identity-view control and the supersampled downscale
//! path (a source target larger than its composite destination).
//!
//! Live-GPU test: needs a real adapter + device, which `Context::new` gets only
//! through a window/surface. It is therefore `#[ignore]`d from the default
//! `cargo test` gate (the crate's convention for GPU-touching checks — the rest
//! of the suite is device-free) and Windows-gated, because building a winit
//! event loop off the harness's main thread needs the Windows-only
//! `any_thread` escape hatch. Run it explicitly with:
//!
//! ```text
//! cargo test -p vk2d --test target_view_readback -- --ignored
//! ```

#![cfg(target_os = "windows")]

use std::sync::Arc;

use vk2d::{Backend, Color, Context, ContextConfig, Point, Rect2, SpriteParams, View2};
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowId};

const LOGICAL: (u32, u32) = (320, 180);
/// A bright, unambiguous fill colour: any black read-back is a real failure,
/// not a dim-tint artefact.
const FILL: Color = Color {
    r: 0.1,
    g: 0.9,
    b: 0.2,
    a: 1.0,
};

/// A Y-up world view windowing a `(w,h)` world rect onto a `(w,h)` target — the
/// shape `set_world_view` produces in the consuming game (negative `scale.y`).
fn y_up_view(w: f32, h: f32) -> View2 {
    View2::window(Point::new(0.0, h), Point::new(w, h), Point::new(w, h), true)
}

fn is_black(p: [u8; 4]) -> bool {
    p[0] < 8 && p[1] < 8 && p[2] < 8
}

fn has_fill(p: [u8; 4]) -> bool {
    // The green channel dominates FILL; a faithful read-back keeps it high.
    p[1] > 180 && p[0] < 80 && p[2] < 120
}

/// The actual assertions, run once a `Context` exists. Panics on failure so the
/// winit `run_app` driver surfaces it as a test failure.
fn run_assertions(ctx: &mut Context) {
    let (w, h) = (LOGICAL.0 as f32, LOGICAL.1 as f32);
    let (cx, cy) = (LOGICAL.0 / 2, LOGICAL.1 / 2);

    let src_ident = ctx.create_target(LOGICAL.0, LOGICAL.1, wgpu::FilterMode::Nearest);
    let src_view = ctx.create_target(LOGICAL.0, LOGICAL.1, wgpu::FilterMode::Nearest);
    let dst_ident = ctx.create_target(LOGICAL.0, LOGICAL.1, wgpu::FilterMode::Nearest);
    let dst_view = ctx.create_target(LOGICAL.0, LOGICAL.1, wgpu::FilterMode::Nearest);

    // A small marker near the top of source space. Under identity it stays at
    // the top (y≈40); under the Y-up flip it lands near the bottom (y≈140).
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

    // Write side: content landed where each view maps it.
    assert!(
        has_fill(ctx.read_target_pixel(src_ident, cx, 40).unwrap()),
        "identity-drawn marker should be at the TOP of its target"
    );
    assert!(
        is_black(ctx.read_target_pixel(src_ident, cx, 140).unwrap()),
        "identity-drawn target should be empty at the bottom"
    );
    assert!(
        has_fill(ctx.read_target_pixel(src_view, cx, 140).unwrap()),
        "Y-up-drawn marker should be MIRRORED to the bottom of its target"
    );
    assert!(
        is_black(ctx.read_target_pixel(src_view, cx, 40).unwrap()),
        "Y-up-drawn target should be empty at the top"
    );

    // Read side: target_sprite composites each source (under identity view)
    // into a fresh target. The whole point of the regression — the Y-up source
    // must NOT come back black.
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
    assert!(
        has_fill(ctx.read_target_pixel(dst_ident, cx, 40).unwrap()),
        "identity-view target must composite faithfully (top marker present)"
    );
    assert!(
        has_fill(ctx.read_target_pixel(dst_view, cx, 140).unwrap()),
        "REGRESSION: Y-up-view target composited via target_sprite came back \
         WITHOUT its content — the black-readback bug"
    );
    assert!(
        is_black(ctx.read_target_pixel(dst_view, cx, 40).unwrap()),
        "Y-up composite must stay empty where the source was empty (no smear)"
    );

    // Supersampled downscale: a 2x source drawn Y-up, composited DOWN to the
    // dest size via target_sprite (Nearest) — the game's SSAA scene→screen
    // shape. A full-cover fill so the centre probe tests the sampling path.
    let ss = ctx.create_target(LOGICAL.0 * 2, LOGICAL.1 * 2, wgpu::FilterMode::Nearest);
    let ss_dst = ctx.create_target(LOGICAL.0, LOGICAL.1, wgpu::FilterMode::Nearest);
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
    assert!(
        has_fill(ctx.read_target_pixel(ss_dst, cx, cy).unwrap()),
        "SSAA-sized Y-up source must downscale-composite faithfully, not black"
    );
}

struct Driver {
    window: Option<Arc<Window>>,
    ran: bool,
}

impl ApplicationHandler for Driver {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.ran {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("vk2d target_view_readback test")
            .with_visible(false)
            .with_inner_size(winit::dpi::LogicalSize::new(
                LOGICAL.0 as f64,
                LOGICAL.1 as f64,
            ));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        self.window = Some(window.clone());
        let mut ctx = Context::new(
            window,
            ContextConfig {
                logical_size: LOGICAL,
                prefer_backend: Backend::Vulkan,
            },
        )
        .expect("context");
        run_assertions(&mut ctx);
        self.ran = true;
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _e: &ActiveEventLoop,
        _id: WindowId,
        _ev: winit::event::WindowEvent,
    ) {
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.ran {
            event_loop.exit();
        }
    }
}

#[test]
#[ignore = "live GPU: needs a Vulkan device + window; run with -- --ignored"]
fn target_sprite_reads_back_non_identity_view_target() {
    let event_loop = EventLoop::builder()
        .with_any_thread(true)
        .build()
        .expect("event loop");
    let mut driver = Driver {
        window: None,
        ran: false,
    };
    event_loop.run_app(&mut driver).expect("run");
    assert!(
        driver.ran,
        "assertions never ran (window/context never came up)"
    );
}
