//! Neutral value types traded across the public API. No backend types here.

/// Straight (non-premultiplied) RGBA colour, channels in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    /// Red channel, `[0, 1]`.
    pub r: f32,
    /// Green channel, `[0, 1]`.
    pub g: f32,
    /// Blue channel, `[0, 1]`.
    pub b: f32,
    /// Alpha (opacity), `[0, 1]`.
    pub a: f32,
}

impl Color {
    /// A colour from explicit red, green, blue, and alpha channels.
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// An opaque colour (`a = 1.0`) from red, green, and blue channels.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Same colour with a replaced alpha — the common "fade this" operation.
    pub const fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    /// Opaque white.
    pub const WHITE: Self = Self::rgb(1.0, 1.0, 1.0);
    /// Opaque black.
    pub const BLACK: Self = Self::rgb(0.0, 0.0, 0.0);
    /// Fully transparent (all channels zero).
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);
}

/// A point / size in logical pixels (top-left origin). Kept minimal on
/// purpose — the renderer does not need a full vector-math type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Horizontal coordinate (or width), logical pixels.
    pub x: f32,
    /// Vertical coordinate (or height), logical pixels.
    pub y: f32,
}

impl Point {
    /// A point / size from `x` and `y` (logical pixels).
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// An axis-aligned rectangle in logical pixels (top-left origin).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect2 {
    /// Left edge (logical pixels).
    pub x: f32,
    /// Top edge (logical pixels).
    pub y: f32,
    /// Width (logical pixels).
    pub w: f32,
    /// Height (logical pixels).
    pub h: f32,
}

impl Rect2 {
    /// A rectangle from its top-left corner `(x, y)` and size `(w, h)`.
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// The right edge (`x + w`).
    pub fn right(&self) -> f32 {
        self.x + self.w
    }
    /// The bottom edge (`y + h`).
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }
    /// The horizontal centre (`x + w/2`).
    pub fn center_x(&self) -> f32 {
        self.x + self.w * 0.5
    }
    /// The vertical centre (`y + h/2`).
    pub fn center_y(&self) -> f32 {
        self.y + self.h * 0.5
    }
}

/// How a sprite samples its source region and orients on screen.
#[derive(Debug, Clone, Copy)]
pub struct SpriteParams {
    /// Source rect in the texture, in pixels. `None` draws the whole texture.
    pub source_px: Option<Rect2>,
    /// Destination size in logical pixels. `None` uses the source size.
    pub dest_size: Option<Point>,
    /// Rotation in radians about the sprite centre.
    pub rotation: f32,
    /// Flip the source horizontally (facing left/right).
    pub flip_x: bool,
    /// Flip the source vertically (e.g. a render-target Y-flip).
    pub flip_y: bool,
    /// Multiplied into every texel (tint / fade).
    pub tint: Color,
}

impl Default for SpriteParams {
    fn default() -> Self {
        Self {
            source_px: None,
            dest_size: None,
            rotation: 0.0,
            flip_x: false,
            flip_y: false,
            tint: Color::WHITE,
        }
    }
}

/// Text style inputs for `Frame::text`.
#[derive(Debug, Clone, Copy)]
pub struct TextStyle {
    /// Font size in logical pixels (whole-pixel for crisp text).
    pub size: f32,
    /// Text colour (multiplied into the glyph coverage).
    pub color: Color,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_and_sprite_defaults() {
        assert_eq!(Color::WHITE.a, 1.0);
        assert_eq!(Color::rgb(0.1, 0.2, 0.3).with_alpha(0.5).a, 0.5);
        let p = SpriteParams::default();
        assert!(p.source_px.is_none() && !p.flip_x && p.tint == Color::WHITE);
        let r = Rect2::new(2.0, 4.0, 6.0, 8.0);
        assert_eq!((r.right(), r.bottom()), (8.0, 12.0));
    }
}
