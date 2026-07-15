//! Live-GPU proof that `Frame::material_sprite` draws a textured sprite through
//! a material's own fragment stage and lands the shaded pixels where the sprite
//! is positioned.
//!
//! It draws a solid-red texture as a sprite through a passthrough material
//! (`fs_main` returns `textureSample(...) * vertex_tint * u.tint`) into a
//! target, then reads the target back: the sprite's footprint must carry the
//! shaded colour (red, un-tinted here) and the area outside it must stay the
//! clear colour. A second draw applies a green uniform tint to prove the
//! material's `fs_main` — not the default sprite pipeline — produced the pixel
//! (the default `sprite()` path has no uniform tint knob).
//!
//! Same harness constraints as `target_view_readback.rs`: needs a real Vulkan
//! adapter + device (reached only through a window/surface), so it is
//! `#[ignore]`d from the default gate and Windows-gated for the `any_thread`
//! event-loop escape hatch. Run explicitly with:
//!
//! ```text
//! cargo test -p vk2d --test material_sprite_readback -- --ignored
//! ```

#![cfg(target_os = "windows")]

use std::sync::Arc;

use vk2d::{
    Backend, Color, Context, ContextConfig, Filter, MaterialDesc, Point, SpriteParams, UniformType,
    UniformValue,
};
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowId};

const LOGICAL: (u32, u32) = (64, 64);

/// A passthrough sprite-shaded material: it samples its slot-0 texture at the
/// interpolated uv and multiplies by both the per-vertex tint (`params.tint`)
/// and a uniform tint. `fs_main` consumes the sprite vertex stage's outputs
/// (`@location(0) uv`, `@location(1) tint`), the contract `material_sprite`
/// documents.
const PASSTHROUGH: &str = r#"
struct U { tint: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t0: texture_2d<f32>;
@group(0) @binding(2) var s0: sampler;
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
};
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
}
@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t0, s0, in.uv) * in.tint * u.tint;
}
"#;

fn is_red(p: [u8; 4]) -> bool {
    p[0] > 180 && p[1] < 80 && p[2] < 80
}

fn is_green(p: [u8; 4]) -> bool {
    p[1] > 180 && p[0] < 80 && p[2] < 80
}

fn is_clear_blue(p: [u8; 4]) -> bool {
    // The clear colour below is a dim blue; the sprite does not cover it.
    p[2] > 80 && p[0] < 80 && p[1] < 80
}

fn run_assertions(ctx: &mut Context) {
    // A 2x2 solid opaque-red texture.
    let red = [
        255u8, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
    ];
    let tex = ctx.load_texture_rgba(&red, 2, 2, Filter::Nearest);

    let mat = ctx
        .load_material(MaterialDesc {
            wgsl: PASSTHROUGH,
            uniforms: &[("tint", UniformType::Vec4)],
            textures: &["t0"],
            ..Default::default()
        })
        .expect("passthrough material compiles");

    let dst = ctx.create_target(LOGICAL.0, LOGICAL.1, wgpu::FilterMode::Nearest);
    let clear = Color {
        r: 0.05,
        g: 0.05,
        b: 0.5,
        a: 1.0,
    };

    // Draw the red sprite as a 20x20 block near the top-left, through the
    // material with a WHITE uniform tint (identity) — the shaded pixel must be
    // the texture's red. The uniform is pushed before the frame begins (the
    // frame mutably borrows `ctx`).
    ctx.set_material_uniform(mat, "tint", UniformValue::Vec4(1.0, 1.0, 1.0, 1.0));
    {
        let mut f = ctx.begin_target_frame(dst, clear).unwrap();
        f.material_sprite(
            mat,
            tex,
            Point::new(8.0, 8.0),
            SpriteParams {
                dest_size: Some(Point::new(20.0, 20.0)),
                ..Default::default()
            },
        );
        f.finish();
    }
    // Inside the sprite footprint (top-left block, centre ~ (18,18)) → red.
    assert!(
        is_red(ctx.read_target_pixel(dst, 18, 18).unwrap()),
        "material_sprite must shade the sprite footprint with the sampled texture colour (red)"
    );
    // Outside the footprint (bottom-right corner) → the clear colour survives —
    // the material only paints where the sprite quad is, not fullscreen.
    assert!(
        is_clear_blue(ctx.read_target_pixel(dst, 50, 50).unwrap()),
        "material_sprite must not paint outside the sprite's footprint"
    );

    // Now prove the material's UNIFORM tint reaches the pixel — something the
    // default sprite pipeline (no uniform tint knob) could never do. Sample a
    // WHITE texture through a GREEN uniform tint: white * green = green, so a
    // green readback can only have come from the material's `fs_main`.
    let white = [255u8; 16]; // 2x2 opaque white
    let white_tex = ctx.load_texture_rgba(&white, 2, 2, Filter::Nearest);
    let dst2 = ctx.create_target(LOGICAL.0, LOGICAL.1, wgpu::FilterMode::Nearest);
    {
        ctx.set_material_uniform(mat, "tint", UniformValue::Vec4(0.0, 1.0, 0.0, 1.0));
        let mut f = ctx.begin_target_frame(dst2, clear).unwrap();
        f.material_sprite(
            mat,
            white_tex,
            Point::new(8.0, 8.0),
            SpriteParams {
                dest_size: Some(Point::new(20.0, 20.0)),
                ..Default::default()
            },
        );
        f.finish();
    }
    assert!(
        is_green(ctx.read_target_pixel(dst2, 18, 18).unwrap()),
        "REGRESSION: the material's uniform tint did not apply — the fragment stage \
         drawing this sprite is not the material's fs_main"
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
            .with_title("vk2d material_sprite_readback test")
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
fn material_sprite_shades_footprint_and_applies_material_tint() {
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
