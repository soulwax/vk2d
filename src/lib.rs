//! vk2d — a standalone immediate-mode 2D renderer on wgpu/Vulkan.
//!
//! Game-agnostic by design: the application supplies raw texture bytes, an
//! app-chosen logical size, arbitrary sprite source rects, and WGSL shader
//! text. The library owns rendering; it never assumes a specific game's
//! resolution, assets, or effects.
//!
//! Shaders are authored in WGSL and compiled to SPIR-V at load time (via naga)
//! so shader compilation is front-loaded to startup and WGSL errors surface
//! with source locations. See the crate README for a quickstart.

#![warn(missing_docs)]

mod blend;
mod color;
mod context;
#[cfg(feature = "egui")]
mod egui_overlay;
mod error;
mod frame;
mod handles;
#[cfg(feature = "winit-input")]
mod input;
mod material;
mod shapes;
mod sprite;
mod target;
mod text;
mod view;

/// Re-export egui so consumers build UI against the exact version vk2d links.
#[cfg(feature = "egui")]
pub use egui;
#[cfg(feature = "egui")]
pub use egui_overlay::EguiOverlay;
#[cfg(feature = "winit-input")]
pub use input::InputState;

pub use blend::BlendMode;
pub use color::{Color, Point, Rect2, SpriteParams, TextStyle};
pub use context::{Backend, Context, ContextConfig, TextMetricsExt};
pub use error::Vk2dError;
pub use frame::Frame;
pub use handles::{FontId, MaterialId, TargetId, TextureId};
pub use material::{MaterialDesc, UniformType, UniformValue};
pub use sprite::Filter;
pub use view::View2;

/// Internal helpers exposed for the crate's own integration tests. Not a
/// stable public API — do not depend on it from applications.
#[doc(hidden)]
pub mod testing {
    pub use crate::material::compile_wgsl_to_spirv;
}

/// Crate version string (used by smoke tests and diagnostics).
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
