//! Shader gallery: the manual parity-check surface for WGSL shader ports.
//!
//! Walks `<repo>/Assets/Shaders/wgsl/` recursively (skipping `lib/`, which
//! holds shared prelude source, not standalone materials), loads every
//! `*.wgsl` file found as a material with the shared `arcane.wgsl` prelude
//! (if present) and a permissive uniform superset, then renders the selected
//! one fullscreen each frame, driven by elapsed time. Cycle with Left/Right
//! arrows. A shader whose WGSL fails to compile still shows up in the list —
//! selecting it draws a magenta error tile instead of crashing, so a bad port
//! is visible without taking down the whole gallery.
//!
//! This example intentionally tolerates an incomplete shader tree: at the
//! time this gallery was written only `Assets/Shaders/wgsl/healing_ray.wgsl`
//! and `vfx_pulse.wgsl` existed (not yet even under a tier subfolder), and the
//! `effects/`/`post/`/`ability/` split arrives in later tasks. Missing
//! folders, an empty shader set, and shaders with no tier subfolder (grouped
//! under `"root"`) are all handled without panicking.
//!
//! Run: `cargo run -p vk2d --example shader_gallery`
//! Smoke: `cargo run -p vk2d --example shader_gallery -- --frames 3` (exits 0).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "winit-input")]
use vk2d::InputState;
use vk2d::{
    Backend, Color, Context, ContextConfig, FontId, MaterialDesc, MaterialId, Point, Rect2,
    TargetId, TextStyle, UniformType, UniformValue, Vk2dError,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const LOGICAL: (u32, u32) = (1600, 900);

/// The uniform superset every gallery material is loaded with. Real shaders
/// only reference the subset they declare in their own `Uniforms` struct —
/// vk2d ignores unknown names when a value is pushed for one, so offering the
/// same permissive list to every material is harmless.
const UNIFORM_SUPERSET: &[(&str, UniformType)] = &[
    ("u_time", UniformType::Vec4),
    ("u_color", UniformType::Vec4),
    ("u_progress", UniformType::Vec4),
    ("u_seed", UniformType::Vec4),
    ("u_from", UniformType::Vec4),
    ("u_to", UniformType::Vec4),
];

/// A gallery entry: one `.wgsl` file found under the shader tree, its tier
/// (the top-level subfolder name, or `"root"` if it has none), and either a
/// loaded `MaterialId` or the compile error message.
struct GalleryEntry {
    /// File stem (e.g. `healing_ray`), used as the on-screen shader name.
    name: String,
    /// Top-level folder under `wgsl/` the file lives in (e.g. `ability`), or
    /// `"root"` for a `.wgsl` file directly under `wgsl/`.
    tier: String,
    /// `Some(material)` if the WGSL compiled; `None` (with the message kept in
    /// `error`) if `load_material` returned `Err(Vk2dError::ShaderCompile)`.
    material: Option<MaterialId>,
    /// The compile error, if loading failed.
    error: Option<String>,
    /// Whether this shader's WGSL declares a `scene` texture slot (the
    /// `post/` tier's contract) — when true, a demo scene target is bound
    /// before drawing it fullscreen.
    wants_scene: bool,
}

/// Resolve the repo root from `CARGO_MANIFEST_DIR` (this example lives in
/// `crates/vk2d`, so the repo root is two levels up).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Recursively collect every `*.wgsl` file under `root`, skipping any `lib`
/// directory (shared prelude source, not a standalone material) at any
/// depth. Returns `(path, tier)` pairs where `tier` is the path component
/// directly under `root` (or `"root"` if the file sits directly in `root`).
/// Never panics: a missing or unreadable directory yields an empty list.
fn walk_wgsl_tree(root: &Path) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let Ok(top) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in top.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dir_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if dir_name.eq_ignore_ascii_case("lib") {
                continue; // shared prelude source, not a material tier
            }
            collect_wgsl_recursive(&path, &dir_name, &mut out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("wgsl"))
        {
            out.push((path, "root".to_string()));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Walk `dir` (a tier folder, e.g. `ability/`) recursively, tagging every
/// `.wgsl` file found with `tier` regardless of nesting depth within it.
fn collect_wgsl_recursive(dir: &Path, tier: &str, out: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_wgsl_recursive(&path, tier, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("wgsl"))
        {
            out.push((path, tier.to_string()));
        }
    }
}

/// Read `<repo>/Assets/Shaders/wgsl/lib/arcane.wgsl` if it exists; an absent
/// prelude degrades to an empty string rather than failing the gallery.
fn read_arcane_prelude(wgsl_root: &Path) -> String {
    let path = wgsl_root.join("lib").join("arcane.wgsl");
    std::fs::read_to_string(&path).unwrap_or_default()
}

/// Load every discovered `.wgsl` file as a material. Compile failures are
/// recorded on the entry (not propagated) so one bad shader doesn't stop the
/// rest of the gallery from loading.
fn load_gallery(ctx: &mut Context, wgsl_root: &Path, arcane: &str) -> Vec<GalleryEntry> {
    let files = walk_wgsl_tree(wgsl_root);
    let mut entries = Vec::with_capacity(files.len());
    for (path, tier) in files {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string());
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                entries.push(GalleryEntry {
                    name,
                    tier,
                    material: None,
                    error: Some(format!("read failed: {e}")),
                    wants_scene: false,
                });
                continue;
            }
        };
        let wants_scene = source.contains("\"scene\"") || source.contains("scene_texture");
        let prelude = if arcane.is_empty() {
            None
        } else {
            Some(arcane)
        };
        let textures: &[&str] = if wants_scene { &["scene"] } else { &[] };
        let desc = MaterialDesc {
            wgsl: &source,
            blend: vk2d::BlendMode::Additive,
            uniforms: UNIFORM_SUPERSET,
            prelude,
            textures,
        };
        match ctx.load_material(desc) {
            Ok(mat) => entries.push(GalleryEntry {
                name,
                tier,
                material: Some(mat),
                error: None,
                wants_scene,
            }),
            Err(Vk2dError::ShaderCompile { message }) => entries.push(GalleryEntry {
                name,
                tier,
                material: None,
                error: Some(message),
                wants_scene,
            }),
            Err(other) => entries.push(GalleryEntry {
                name,
                tier,
                material: None,
                error: Some(format!("{other:?}")),
                wants_scene,
            }),
        }
    }
    entries
}

/// Find a TTF the gallery can borrow for on-screen labels. Reuses whichever
/// asset the rest of the project already ships; if none of these paths exist
/// (a stripped-down checkout of just the `vk2d` crate), the gallery falls
/// back to printing shader names to stdout instead of drawing text.
fn find_label_font(repo: &Path) -> Option<PathBuf> {
    const CANDIDATES: &[&str] = &[
        "Assets/Fonts/arcadeclassic/ARCADECLASSIC.TTF",
        "Assets/Fonts/compass-gold-v1/CompassGold.ttf",
        "Assets/Fonts/ruler-gold-v1/RulerGold.ttf",
    ];
    CANDIDATES
        .iter()
        .map(|rel| repo.join(rel))
        .find(|p| p.is_file())
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
        entries: Vec::new(),
        selected: 0,
        font: None,
        demo_target: None,
        start: Instant::now(),
        frames: 0,
        max_frames,
        close: false,
        #[cfg(feature = "winit-input")]
        input: InputState::new(),
    };
    event_loop.run_app(&mut app).expect("run");
}

struct App {
    window: Option<Arc<Window>>,
    ctx: Option<Context>,
    entries: Vec<GalleryEntry>,
    selected: usize,
    font: Option<FontId>,
    /// A small offscreen target holding a demo scene, bound as the `scene`
    /// texture for any selected material that declares one (the `post/`
    /// tier's contract). Created lazily the first time it's needed.
    demo_target: Option<TargetId>,
    start: Instant,
    frames: u32,
    max_frames: Option<u32>,
    close: bool,
    #[cfg(feature = "winit-input")]
    input: InputState,
}

impl App {
    /// Render the small demo scene into `demo_target` (a couple of filled
    /// rects), used as the `scene` input for post-process-style materials.
    /// Only invoked when the selected entry actually declares a scene slot.
    fn render_demo_scene(&mut self, target: TargetId) {
        let Some(ctx) = self.ctx.as_mut() else {
            return;
        };
        let Ok(mut frame) = ctx.begin_target_frame(target, Color::rgb(0.08, 0.08, 0.12)) else {
            return;
        };
        frame.fill_rect(
            Rect2::new(60.0, 60.0, 220.0, 160.0),
            Color::rgb(0.8, 0.3, 0.3),
        );
        frame.fill_rect(
            Rect2::new(340.0, 220.0, 180.0, 180.0),
            Color::rgb(0.3, 0.6, 0.9),
        );
        frame.circle(Point::new(480.0, 90.0), 60.0, Color::rgb(0.9, 0.8, 0.3));
        frame.finish();
    }

    #[cfg(feature = "winit-input")]
    fn select_delta(&mut self, delta: i32) {
        if self.entries.is_empty() {
            return;
        }
        let len = self.entries.len() as i32;
        let next = (self.selected as i32 + delta).rem_euclid(len);
        self.selected = next as usize;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.ctx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("vk2d — shader_gallery")
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

        let repo = repo_root();
        let wgsl_root = repo.join("Assets").join("Shaders").join("wgsl");
        let arcane = read_arcane_prelude(&wgsl_root);
        self.entries = load_gallery(&mut ctx, &wgsl_root, &arcane);

        if self.entries.is_empty() {
            println!(
                "shader_gallery: no shaders found under {}",
                wgsl_root.display()
            );
        } else {
            let loaded = self.entries.iter().filter(|e| e.material.is_some()).count();
            println!(
                "shader_gallery: loaded {loaded}/{} shader(s) from {}",
                self.entries.len(),
                wgsl_root.display()
            );
            for entry in &self.entries {
                match &entry.error {
                    Some(msg) => println!("  [{}] {} — FAILED: {msg}", entry.tier, entry.name),
                    None => println!("  [{}] {}", entry.tier, entry.name),
                }
            }
        }

        if let Some(font_path) = find_label_font(&repo) {
            match std::fs::read(&font_path) {
                Ok(bytes) => match ctx.load_font(&bytes, 24.0) {
                    Ok(font) => self.font = Some(font),
                    Err(e) => {
                        eprintln!("shader_gallery: font load failed ({e:?}), using stdout labels")
                    }
                },
                Err(e) => eprintln!(
                    "shader_gallery: could not read font {} ({e}), using stdout labels",
                    font_path.display()
                ),
            }
        } else {
            eprintln!("shader_gallery: no bundled font found, using stdout labels");
        }

        self.ctx = Some(ctx);
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        #[cfg(feature = "winit-input")]
        self.input.feed(&event);

        match event {
            WindowEvent::CloseRequested => self.close = true,
            WindowEvent::Resized(size) => {
                if let Some(ctx) = self.ctx.as_mut() {
                    ctx.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                #[cfg(feature = "winit-input")]
                {
                    if self
                        .input
                        .is_named_pressed(winit::keyboard::NamedKey::ArrowRight)
                    {
                        self.select_delta(1);
                    }
                    if self
                        .input
                        .is_named_pressed(winit::keyboard::NamedKey::ArrowLeft)
                    {
                        self.select_delta(-1);
                    }
                }

                let t = self.start.elapsed().as_secs_f32();
                let current_wants_scene = self
                    .entries
                    .get(self.selected)
                    .is_some_and(|e| e.wants_scene && e.material.is_some());

                if current_wants_scene {
                    let target = *self.demo_target.get_or_insert_with(|| {
                        self.ctx
                            .as_mut()
                            .expect("context initialized before first redraw")
                            .create_target(600, 400)
                    });
                    self.render_demo_scene(target);
                }

                if let Some(ctx) = self.ctx.as_mut() {
                    let mut frame = match ctx.begin_frame(Color::rgb(0.03, 0.03, 0.05)) {
                        Ok(f) => f,
                        Err(_) => return,
                    };

                    if self.entries.is_empty() {
                        if let Some(font) = self.font {
                            frame.text(
                                font,
                                "no shaders found under Assets/Shaders/wgsl/",
                                Point::new(48.0, 48.0),
                                TextStyle {
                                    size: 22.0,
                                    color: Color::rgb(0.9, 0.9, 0.9),
                                },
                            );
                        }
                    } else if let Some(entry) = self.entries.get(self.selected) {
                        match entry.material {
                            Some(mat) => {
                                frame.set_uniform(
                                    mat,
                                    "u_time",
                                    UniformValue::Vec4(t, 0.0, 0.0, 0.0),
                                );
                                frame.set_uniform(
                                    mat,
                                    "u_color",
                                    UniformValue::Vec4(1.0, 1.0, 1.0, 1.0),
                                );
                                frame.set_uniform(
                                    mat,
                                    "u_progress",
                                    UniformValue::Vec4((t * 0.3).fract(), 0.0, 0.0, 0.0),
                                );
                                frame.set_uniform(
                                    mat,
                                    "u_seed",
                                    UniformValue::Vec4(0.0, 0.0, 0.0, 0.0),
                                );
                                frame.set_uniform(
                                    mat,
                                    "u_from",
                                    UniformValue::Vec4(0.2, 0.5, 0.0, 0.0),
                                );
                                frame.set_uniform(
                                    mat,
                                    "u_to",
                                    UniformValue::Vec4(0.8, 0.5, 0.0, 0.0),
                                );
                                if entry.wants_scene
                                    && let Some(target) = self.demo_target
                                {
                                    frame.bind_material_target(mat, "scene", target);
                                }
                                frame.material_fullscreen(mat);
                            }
                            None => {
                                // Error tile: magenta clear (already the frame
                                // background would need a separate rect since
                                // begin_frame's clear already ran) + the name.
                                frame.fill_rect(
                                    Rect2::new(0.0, 0.0, LOGICAL.0 as f32, LOGICAL.1 as f32),
                                    Color::rgb(0.8, 0.0, 0.8),
                                );
                            }
                        }

                        if let Some(font) = self.font {
                            let label = match &entry.error {
                                Some(err) => format!(
                                    "[{}] {} — COMPILE FAILED: {}",
                                    entry.tier, entry.name, err
                                ),
                                None => format!(
                                    "[{}/{}] [{}] {}",
                                    self.selected + 1,
                                    self.entries.len(),
                                    entry.tier,
                                    entry.name
                                ),
                            };
                            frame.text(
                                font,
                                &label,
                                Point::new(24.0, 24.0),
                                TextStyle {
                                    size: 20.0,
                                    color: Color::rgb(1.0, 1.0, 1.0),
                                },
                            );
                        }
                    }

                    frame.present();

                    self.frames += 1;
                    if self.max_frames.is_some_and(|m| self.frames >= m) {
                        self.close = true;
                    }
                }

                #[cfg(feature = "winit-input")]
                self.input.end_frame();
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
