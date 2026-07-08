# vk2d

A standalone, immediate-mode 2D renderer on [wgpu] / Vulkan, with shaders
authored in WGSL and compiled to SPIR-V at startup.

vk2d is **game-agnostic** by design: the application supplies raw texture bytes,
its own logical resolution, arbitrary sprite source rects, and WGSL shader text.
The library owns rendering; it never assumes a specific game's resolution,
assets, or effects. It was extracted from a game's renderer to be reusable on
its own.

[wgpu]: https://github.com/gfx-rs/wgpu

## Design contracts

- **Immediate mode.** You think in draw calls, not pipelines. Each frame:
  `begin_frame → sprite/fill_rect/line/circle/text/material_fullscreen → present`.
  The library batches internally.
- **Shaders are data, not code.** There is one generic material type. An effect
  is a `.wgsl` file plus a `[(name, UniformType)]` uniform declaration — no
  per-shader Rust. WGSL is compiled to SPIR-V at load (via [naga]), so shader
  errors surface at startup with source locations, not at first draw.
- **No panics in the public API.** Setup and resource-loading calls return
  `Result<_, Vk2dError>`; per-frame draw calls are infallible and silently skip
  bad handles. A malformed shader or a zero-sized window is an `Err`, never a
  crash.
- **No backend types across the public API.** You trade neutral value types
  (`Color`, `Point`, `Rect2`) and opaque handles (`TextureId`, `MaterialId`,
  `FontId`, `TargetId`). No `wgpu` type leaks out of the default surface.
- **Fixed logical size + upscale.** The scene is drawn into an offscreen target
  at your chosen logical resolution and blitted to the window with a Nearest
  upscale — crisp pixel-art scaling for free.

[naga]: https://github.com/gfx-rs/wgpu/tree/trunk/naga

## Quickstart

```rust,no_run
use std::sync::Arc;
use vk2d::{Backend, Color, Context, ContextConfig, Filter, Point, Rect2, SpriteParams};
use winit::window::Window;

// `window: Arc<Window>` from your winit event loop.
fn draw(window: Arc<Window>) -> Result<(), vk2d::Vk2dError> {
    let mut ctx = Context::new(
        window,
        ContextConfig { logical_size: (1280, 720), prefer_backend: Backend::Vulkan },
    )?;

    // Upload a texture from decoded RGBA bytes (you decode the image; vk2d
    // never touches files).
    let pixels: Vec<u8> = vec![255; 4 * 4 * 4]; // a 4x4 white square
    let tex = ctx.load_texture_rgba(&pixels, 4, 4, Filter::Nearest);

    // Draw one frame.
    let mut frame = ctx.begin_frame(Color::rgb(0.05, 0.06, 0.10))?;
    frame.sprite(
        tex,
        Point::new(100.0, 100.0),
        SpriteParams { dest_size: Some(Point::new(256.0, 256.0)), ..Default::default() },
    );
    frame.fill_rect(Rect2::new(500.0, 150.0, 220.0, 140.0), Color::rgb(0.8, 0.3, 0.4));
    frame.present();
    Ok(())
}
```

A WGSL material (fullscreen, one `vec4` uniform):

```rust,no_run
# use vk2d::{Context, MaterialDesc, UniformType, UniformValue, BlendMode};
# fn go(ctx: &mut Context, mat_wgsl: &str) -> Result<(), vk2d::Vk2dError> {
let mat = ctx.load_material(MaterialDesc {
    wgsl: mat_wgsl,                      // must define vs_main + fs_main and a
    blend: BlendMode::Additive,         // uniform block at @group(0) @binding(0)
    uniforms: &[("time", UniformType::Vec4)],
})?;
# let mut frame = ctx.begin_frame(vk2d::Color::BLACK)?;
frame.set_uniform(mat, "time", UniformValue::Vec4(1.5, 0.0, 0.0, 0.0));
frame.material_fullscreen(mat);
# frame.present();
# Ok(())
# }
```

See [`examples/hello_sprite.rs`](examples/hello_sprite.rs) for a complete,
runnable window with a sprite, vector primitives, and a time-driven material.

```console
cargo run -p vk2d --example hello_sprite
cargo run -p vk2d --example hello_sprite -- --frames 3   # smoke run, exits 0
```

## Features

vk2d's core has a small dependency surface. Optional integrations are behind
feature flags (all **off** by default):

| Feature | Adds | Notes |
| --- | --- | --- |
| `egui` | `EguiOverlay` + `Frame::present_with_egui` | Paints an [egui] overlay onto the swapchain over the scene, presented together. Re-exports `egui` so consumers build UI against the exact version vk2d links. |
| `winit-input` | `InputState` | A small keyboard/mouse state tracker fed from winit `WindowEvent`s (`is_key_down`, `is_key_pressed`, `mouse_position`, …). |

[egui]: https://github.com/emilk/egui

## Requirements

- A Vulkan-capable adapter (`Backend::Vulkan`), or `Backend::Auto` to let wgpu
  pick any available backend (Vulkan/Metal/DX12/GL).
- A [winit] `Window` (0.30) to render into.

[winit]: https://github.com/rust-windowing/winit

## Status

Pre-1.0. The public API is small and stable in shape but may still change. The
crate currently lives in-repo alongside its first consumer and will be extracted
to its own repository once mature.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
