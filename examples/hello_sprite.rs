//! Minimal vk2d example: open a window, draw a procedurally-generated sprite, a
//! filled rectangle, a line, a circle, and a time-driven WGSL material — all
//! through the immediate-mode API. No asset files: the texture is generated in
//! code and the shader is inline WGSL.
//!
//! Run: `cargo run -p vk2d --example hello_sprite`
//! Smoke: `cargo run -p vk2d --example hello_sprite -- --frames 3` (exits 0).

use std::sync::Arc;
use std::time::Instant;

use vk2d::{
    Backend, Color, Context, ContextConfig, Filter, MaterialDesc, MaterialId, Point, Rect2,
    SpriteParams, TextureId, UniformType, UniformValue,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const LOGICAL: (u32, u32) = (1280, 720);

/// A time-driven gradient material (fullscreen; fades in with a soft radial
/// pulse). Demonstrates WGSL authored as data + a named uniform.
const GRADIENT_WGSL: &str = r#"
struct Uniforms { time: vec4<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    out.position = vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = u.time.x;
    let d = distance(in.uv, vec2<f32>(0.5, 0.5));
    let pulse = 0.5 + 0.5 * sin(t * 2.0 - d * 12.0);
    let glow = smoothstep(0.5, 0.0, d) * pulse;
    return vec4<f32>(0.2 * glow, 0.6 * glow, 0.9 * glow, glow * 0.5);
}
"#;

/// Build an 8×8 RGBA checkerboard in code (no asset file).
fn checker_texture() -> (Vec<u8>, u32, u32) {
    let size = 8u32;
    let mut bytes = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let on = (x + y) % 2 == 0;
            let (r, g, b) = if on { (230, 210, 120) } else { (60, 50, 90) };
            bytes.extend_from_slice(&[r, g, b, 255]);
        }
    }
    (bytes, size, size)
}

fn main() {
    let max_frames = std::env::args()
        .skip(1)
        .filter_map(|a| a.strip_prefix("--frames=").map(str::to_owned))
        .next()
        .or_else(|| {
            let mut it = std::env::args().skip_while(|a| a != "--frames");
            it.next();
            it.next()
        })
        .and_then(|v| v.parse::<u32>().ok());

    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App {
        window: None,
        ctx: None,
        tex: TextureId(0),
        mat: MaterialId(0),
        start: Instant::now(),
        frames: 0,
        max_frames,
        close: false,
    };
    event_loop.run_app(&mut app).expect("run");
}

struct App {
    window: Option<Arc<Window>>,
    ctx: Option<Context>,
    tex: TextureId,
    mat: MaterialId,
    start: Instant,
    frames: u32,
    max_frames: Option<u32>,
    close: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.ctx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("vk2d — hello_sprite")
            .with_inner_size(LogicalSize::new(LOGICAL.0 as f64, LOGICAL.1 as f64));
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
        println!("vk2d {} on {}", vk2d::version(), ctx.adapter_info());

        let (bytes, w, h) = checker_texture();
        self.tex = ctx.load_texture_rgba(&bytes, w, h, Filter::Nearest);
        self.mat = ctx
            .load_material(MaterialDesc {
                wgsl: GRADIENT_WGSL,
                blend: vk2d::BlendMode::Additive,
                uniforms: &[("time", UniformType::Vec4)],
                prelude: None,
                textures: &[],
            })
            .expect("gradient material compiles");
        self.ctx = Some(ctx);
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.close = true,
            WindowEvent::Resized(size) => {
                if let Some(ctx) = self.ctx.as_mut() {
                    ctx.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let t = self.start.elapsed().as_secs_f32();
                if let Some(ctx) = self.ctx.as_mut() {
                    let mut frame = match ctx.begin_frame(Color::rgb(0.05, 0.06, 0.10)) {
                        Ok(f) => f,
                        Err(_) => return,
                    };
                    // A big checker sprite (nearest-filtered, scaled up).
                    frame.sprite(
                        self.tex,
                        Point::new(120.0, 120.0),
                        SpriteParams {
                            dest_size: Some(Point::new(256.0, 256.0)),
                            ..Default::default()
                        },
                    );
                    // A filled rect, a line, and a circle (vector primitives).
                    frame.fill_rect(
                        Rect2::new(500.0, 150.0, 220.0, 140.0),
                        Color::rgb(0.8, 0.3, 0.4),
                    );
                    frame.line(
                        Point::new(120.0, 460.0),
                        Point::new(700.0, 620.0),
                        6.0,
                        Color::rgb(0.4, 0.9, 0.6),
                    );
                    frame.circle(Point::new(950.0, 400.0), 90.0, Color::rgb(0.5, 0.7, 1.0));
                    // The time-driven WGSL material, over the top.
                    frame.set_uniform(self.mat, "time", UniformValue::Vec4(t, 0.0, 0.0, 0.0));
                    frame.material_fullscreen(self.mat);
                    frame.present();

                    self.frames += 1;
                    if self.max_frames.is_some_and(|m| self.frames >= m) {
                        self.close = true;
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.close {
            event_loop.exit();
            return;
        }
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}
