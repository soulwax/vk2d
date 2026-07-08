//! Opaque resource handles. Callers receive these from `Context` loaders and
//! pass them back to draw calls; the backend resolves them internally, so no
//! wgpu type ever crosses the public API.

/// Opaque texture handle (one per loaded texture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

/// Opaque material/shader handle (one per loaded WGSL material).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

/// Opaque font handle (one per loaded TTF at an atlas size).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontId(pub u32);

/// Opaque render-target handle (one per offscreen buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TargetId(pub u32);
