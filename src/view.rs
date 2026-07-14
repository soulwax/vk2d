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

    // ── The render-scaled-target clip-mapping contract ───────────────────────
    //
    // `View2::window(.., out_size, ..)` maps world coordinates into `out_size`
    // OUTPUT PIXELS, and the batch's `logical_to_clip(px, py, size)` divides
    // by `size` to reach NDC. These two must reference the SAME size, or every
    // off-centre draw lands at the wrong NDC. The bug this pins: a frame
    // rendering into a render-scaled (e.g. 2× SSAA) offscreen target built its
    // view with `out = target_size` (correct) but converted to clip with
    // `logical_size` (the swapchain's 1× size), doubling the NDC of anything
    // off the window centre and flinging sprites/shapes off-screen. Centre
    // content survived only because `0 * 2 == 0` — exactly why parallax motes
    // near the camera showed while the off-centre training dummy vanished.

    /// A world point at any window fraction must land at the SAME NDC whether
    /// the target is 1× or 2× the window — provided the clip conversion uses
    /// the target's own size (the fix), not a fixed logical size (the bug).
    #[test]
    fn window_view_maps_to_same_ndc_regardless_of_target_scale() {
        // World window: centre (1000, 1000), 1600×900 wide (Y-up).
        let top_left = Point {
            x: 200.0,
            y: 1450.0,
        };
        let size = Point {
            x: 1600.0,
            y: 900.0,
        };
        // A world point 3/4 across and 1/4 up the window (deliberately
        // off-centre, where the doubling bug is visible).
        let world = Point {
            x: 200.0 + 1600.0 * 0.75,
            y: 1450.0 - 900.0 * 0.75,
        };

        // 1× target: view out = clip size = (1600, 900).
        let out_1x = Point {
            x: 1600.0,
            y: 900.0,
        };
        let v1 = View2::window(top_left, size, out_1x, true);
        let p1 = v1.apply(world);
        let ndc1 = crate::sprite::logical_to_clip(p1.x, p1.y, (1600, 900));

        // 2× SSAA target: view out = clip size = (3200, 1800). The FIX makes
        // both use the target's size; the ndc must be identical to the 1×
        // case. (Under the bug, clip used (1600, 900) here → doubled ndc.)
        let out_2x = Point {
            x: 3200.0,
            y: 1800.0,
        };
        let v2 = View2::window(top_left, size, out_2x, true);
        let p2 = v2.apply(world);
        let ndc2 = crate::sprite::logical_to_clip(p2.x, p2.y, (3200, 1800));

        assert!(
            (ndc1.0 - ndc2.0).abs() < 1e-4 && (ndc1.1 - ndc2.1).abs() < 1e-4,
            "1× ndc {ndc1:?} != 2× ndc {ndc2:?} — clip size must match the view's out size"
        );
        // And that shared NDC must be on-screen (|x|,|y| ≤ 1), not the >1
        // off-screen value the mismatch produced.
        assert!(
            ndc1.0.abs() <= 1.0 && ndc1.1.abs() <= 1.0,
            "ndc {ndc1:?} off-screen"
        );
    }

    /// The concrete failure mode, pinned directly: converting a 2×-target view
    /// point to clip with the 1× `logical_size` (the bug) pushes an off-centre
    /// world point PAST the clip volume (|ndc| > 1), while converting with the
    /// matching 2× size (the fix) keeps it on-screen. Guards against a
    /// regression that reintroduces a fixed `logical_size` in the clip path.
    #[test]
    fn mismatched_clip_size_pushes_offcentre_content_off_screen() {
        let top_left = Point {
            x: 200.0,
            y: 1450.0,
        };
        let size = Point {
            x: 1600.0,
            y: 900.0,
        };
        let world = Point {
            x: 200.0 + 1600.0 * 0.9,
            y: 1450.0 - 900.0 * 0.5,
        };
        let out_2x = Point {
            x: 3200.0,
            y: 1800.0,
        };
        let v = View2::window(top_left, size, out_2x, true);
        let p = v.apply(world);
        // Bug: clip with logical_size (1600, 900) — off-screen.
        let bug = crate::sprite::logical_to_clip(p.x, p.y, (1600, 900));
        assert!(bug.0 > 1.0, "expected the bug to push x past clip: {bug:?}");
        // Fix: clip with the target's own size (3200, 1800) — on-screen.
        let fixed = crate::sprite::logical_to_clip(p.x, p.y, (3200, 1800));
        assert!(
            fixed.0.abs() <= 1.0,
            "fix should keep x on-screen: {fixed:?}"
        );
    }
}
