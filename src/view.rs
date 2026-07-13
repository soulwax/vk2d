//! A 2D view: a CPU-side affine applied to logical-pixel coordinates before
//! they reach the pixel→clip conversion at batch build time. This lets a
//! consumer window an arbitrary (optionally Y-up) world rect onto the output
//! without vk2d knowing anything about cameras, worlds, or games.

use crate::Point;

/// A 2D view: logical-pixel coordinates are mapped through
/// `p' = (p - origin) * scale` before rasterization. `scale.y < 0` gives a
/// Y-up source space. Identity = no transform (the default).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View2 {
    /// The source-space point that maps to output `(0, 0)`.
    pub origin: Point,
    /// Per-axis scale from source space to output pixels. A negative `y`
    /// flips vertically (Y-up source presented on a Y-down output).
    pub scale: Point,
}

impl View2 {
    /// The identity view: `apply` is a passthrough and `length_scale` is `1.0`.
    pub fn identity() -> Self {
        Self {
            origin: Point { x: 0.0, y: 0.0 },
            scale: Point { x: 1.0, y: 1.0 },
        }
    }

    /// Map a source window (`top_left` + `size`) onto an output of `out_size`
    /// pixels. When `y_up` is true, `top_left` is the window's top edge in a
    /// Y-up source space (source Y grows upward, so the top edge is the
    /// window's MAXIMUM y) — this matches how macroquad's
    /// `Camera2D::from_display_rect` presents a Y-up world. The resulting view
    /// maps that top-left corner to output `(0, 0)` and increasing source Y
    /// (further up in the world) to smaller output Y (higher on screen).
    pub fn window(top_left: Point, size: Point, out_size: Point, y_up: bool) -> Self {
        let sx = out_size.x / size.x.max(1e-6);
        let sy = out_size.y / size.y.max(1e-6);
        if y_up {
            Self {
                origin: Point {
                    x: top_left.x,
                    y: top_left.y,
                },
                scale: Point { x: sx, y: -sy },
            }
        } else {
            Self {
                origin: top_left,
                scale: Point { x: sx, y: sy },
            }
        }
    }

    /// Apply the view to a source-space point, producing an output-pixel point.
    pub fn apply(&self, p: Point) -> Point {
        Point {
            x: (p.x - self.origin.x) * self.scale.x,
            y: (p.y - self.origin.y) * self.scale.y,
        }
    }

    /// Uniform length scale (`|scale.x|`) for radii/thickness/lengths that have
    /// no independent axis to transform.
    pub fn length_scale(&self) -> f32 {
        self.scale.x.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;

    #[test]
    fn y_up_window_maps_world_to_output_pixels() {
        // World window: top-left (100, 900), size 800x450 (Y-up), output 1600x900.
        let v = View2::window(
            Point { x: 100.0, y: 900.0 },
            Point { x: 800.0, y: 450.0 },
            Point {
                x: 1600.0,
                y: 900.0,
            },
            true,
        );
        // World point at the window's top-left corner → output (0, 0).
        let tl = v.apply(Point { x: 100.0, y: 900.0 });
        assert!((tl.x - 0.0).abs() < 1e-4 && (tl.y - 0.0).abs() < 1e-4);
        // World centre of the window → output centre.
        let c = v.apply(Point { x: 500.0, y: 675.0 });
        assert!((c.x - 800.0).abs() < 1e-4 && (c.y - 450.0).abs() < 1e-4);
        // Higher world Y renders HIGHER on screen (smaller output y).
        let hi = v.apply(Point {
            x: 100.0,
            y: 1000.0,
        });
        assert!(hi.y < tl.y + 1e-4);
        assert!((v.length_scale() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn identity_is_a_passthrough() {
        let v = View2::identity();
        let p = Point { x: 12.5, y: -3.0 };
        let q = v.apply(p);
        assert_eq!((q.x, q.y), (p.x, p.y));
        assert_eq!(v.length_scale(), 1.0);
    }
}
