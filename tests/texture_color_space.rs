//! Regression: an uploaded texture's sampled color must match the bytes the
//! app supplied, byte-for-byte (within blend/interpolation rounding) — not
//! darkened by an implicit sRGB->linear decode.
//!
//! Motivation: `create_gpu_texture` (sprite.rs) and `create_fallback_texture`
//! (material.rs) independently hardcoded `TextureFormat::Rgba8UnormSrgb`,
//! while `SCENE_FORMAT` (target.rs) — the format every render target and
//! every fragment shader's OUTPUT already assumes — is linear
//! (`Rgba8Unorm`). Every fragment shader in this crate treats a sampled
//! texel as an already-final, display-ready color; sampling an
//! sRGB-formatted texture asks the GPU to decode sRGB->linear on read, with
//! no compensating re-encode anywhere downstream. The fix makes uploaded
//! textures share `SCENE_FORMAT`; this test locks it in by uploading a known,
//! deliberately non-gray, non-extreme color (avoiding 0/255, where sRGB and
//! linear encodings coincide and would mask the bug) and asserting a
//! read-back sprite draw reproduces it.
//!
//! Live-GPU test: needs a real adapter + device, which `Context::new` gets
//! only through a window/surface. `#[ignore]`d from the default `cargo test`
//! gate (the crate's convention — see `target_view_readback.rs`) and
//! Windows-gated for the same `any_thread` reason. Run explicitly with:
//!
//! ```text
//! cargo test -p vk2d --test texture_color_space -- --ignored
//! ```

#![cfg(target_os = "windows")]

use std::sync::Arc;

use vk2d::{Backend, Color, Context, ContextConfig, Filter, Point, SpriteParams};
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowId};

const LOGICAL: (u32, u32) = (64, 64);
/// A deliberately mid-range, non-gray, non-extreme RGBA byte value. sRGB and
/// linear encodings agree at 0 and 255 (and are close near them), so a bug
/// probed only at those corners would pass by accident; mid-range channels
/// (here 128/64/200) make the ~15-40% per-channel darkening an unwanted
/// sRGB decode introduces impossible to miss.
const SOURCE_RGBA: [u8; 4] = [128, 64, 200, 255];
/// Byte tolerance for the sprite pipeline's own float round-trip (upload as
/// u8 -> shader math in f32 -> quantize back to u8 on write). An sRGB decode
/// at these channel values would be off by 30-90 (well outside this band).
const TOLERANCE: i32 = 6;

fn channel_close(actual: u8, expected: u8) -> bool {
    (actual as i32 - expected as i32).abs() <= TOLERANCE
}

fn matches_source(p: [u8; 4]) -> bool {
    channel_close(p[0], SOURCE_RGBA[0])
        && channel_close(p[1], SOURCE_RGBA[1])
        && channel_close(p[2], SOURCE_RGBA[2])
        && channel_close(p[3], SOURCE_RGBA[3])
}

/// The actual assertions, run once a `Context` exists. Panics on failure so
/// the winit `run_app` driver surfaces it as a test failure.
fn run_assertions(ctx: &mut Context) {
    // A single solid-color 2x2 texture — small, but big enough that Nearest
    // sampling at the probe point is unambiguous.
    let pixels: Vec<u8> = SOURCE_RGBA.repeat(4);
    let tex = ctx.load_texture_rgba(&pixels, 2, 2, Filter::Nearest);

    let dst = ctx.create_target(LOGICAL.0, LOGICAL.1, wgpu::FilterMode::Nearest);
    {
        let mut f = ctx.begin_target_frame(dst, Color::BLACK).unwrap();
        f.sprite(
            tex,
            Point::new(0.0, 0.0),
            SpriteParams {
                dest_size: Some(Point::new(LOGICAL.0 as f32, LOGICAL.1 as f32)),
                ..Default::default()
            },
        );
        f.finish();
    }

    let center = (LOGICAL.0 / 2, LOGICAL.1 / 2);
    let readback = ctx
        .read_target_pixel(dst, center.0, center.1)
        .expect("target pixel readback");
    assert!(
        matches_source(readback),
        "REGRESSION: sampled sprite color {readback:?} does not match the \
         uploaded source {SOURCE_RGBA:?} within tolerance {TOLERANCE} — an \
         implicit sRGB<->linear conversion is happening somewhere in the \
         upload/sample/write path"
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
            .with_title("vk2d texture_color_space test")
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
fn uploaded_texture_color_survives_sample_and_readback_unchanged() {
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
