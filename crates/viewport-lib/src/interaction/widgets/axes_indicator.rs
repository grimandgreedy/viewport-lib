//! Axes orientation indicator : a small XYZ gizmo in the bottom-left corner of
//! the viewport.
//!
//! Provides:
//! - `build_axes_overlays`: emits screen-space `OverlayShapeItem`s (axis lines,
//!   circles, rings, and letter glyphs) into the shared 2D overlay pass, so the
//!   indicator needs no dedicated pipeline or vertex buffers of its own.
//! - `hit_test`: given a click position in pixels, returns the target
//!   (yaw, pitch) if an axis circle was hit.

use crate::{OverlayFill, OverlayShape, OverlayShapeItem};

// ---------------------------------------------------------------------------
// Axis view targets
// ---------------------------------------------------------------------------

/// Result of a hit test on the axes indicator.
#[derive(Debug, Clone, Copy)]
pub struct AxisView {
    /// Target camera orientation for the clicked axis view.
    pub orientation: glam::Quat,
    /// Which axis was hit: 0 = X, 1 = Y, 2 = Z.
    pub axis_index: usize,
}

// Axis colours in linear light (the overlay pass writes to the sRGB target, which
// encodes on write). Kept deliberately dark and muted so the indicator sits back
// against the scene rather than glowing.
const X_COLOUR: [f32; 4] = [0.75, 0.06, 0.07, 1.0]; // red
const Y_COLOUR: [f32; 4] = [0.18, 0.55, 0.03, 1.0]; // green
const Z_COLOUR: [f32; 4] = [0.05, 0.27, 0.72, 1.0]; // blue

// Layout parameters, in logical pixels (the overlay pass scales by DPI).
//
// Stroke widths are heavier than the raw geometry the earlier bespoke pipeline
// drew: the shared overlay shapes are drawn with a 1px SDF anti-alias band at
// every edge, which softens thin strokes, so the weights are bumped to keep the
// same visual "ink" as the old hard-edged triangles.
const ORIGIN_OFFSET: f32 = 54.0;
const LINE_LENGTH: f32 = 40.0;
const LINE_THICKNESS: f32 = 4.75;
const CIRCLE_RADIUS: f32 = 12.0;
/// Half-width / half-height of a letter glyph.
const GLYPH_HALF: f32 = 4.5;
/// Stroke thickness of a letter glyph.
const GLYPH_THICKNESS: f32 = 2.65;
/// z_order spacing between the three axes so a nearer axis stacks over a farther
/// one; the four sub-parts of one axis occupy the slots in between.
const AXIS_Z_STRIDE: i32 = 10;

// ---------------------------------------------------------------------------
// Overlay geometry
// ---------------------------------------------------------------------------

/// Emit the axes indicator as screen-space overlay shapes into `shapes`.
///
/// `viewport_w` / `viewport_h` are the logical viewport size; `orientation` is
/// the current camera orientation. The indicator sits in the bottom-left corner.
/// Shapes carry per-axis `z_order`s so the axis nearest the camera draws on top.
pub(crate) fn build_axes_overlays(
    viewport_w: f32,
    viewport_h: f32,
    orientation: glam::Quat,
    shapes: &mut Vec<OverlayShapeItem>,
) {
    // Origin in overlay space (top-left origin, Y down): bottom-left corner.
    let origin = glam::vec2(ORIGIN_OFFSET, viewport_h - ORIGIN_OFFSET);
    let _ = viewport_w; // origin is measured from the left edge; width is unused.

    // Derive view axes from the orientation quaternion.
    let view_right = orientation * glam::Vec3::X;
    let view_up = orientation * glam::Vec3::Y;
    let view_fwd = orientation * glam::Vec3::Z; // from centre toward eye

    let axes_world = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
    let colours = [X_COLOUR, Y_COLOUR, Z_COLOUR];

    // Screen offset (logical px) of an axis tip from the origin, Y flipped to the
    // overlay's downward Y.
    let tip_of = |axis: glam::Vec3| -> glam::Vec2 {
        let sx = axis.dot(view_right);
        let sy = axis.dot(view_up);
        glam::vec2(origin.x + sx * LINE_LENGTH, origin.y - sy * LINE_LENGTH)
    };

    // Back-to-front by depth (view_fwd dot axis): the axis pointing most toward
    // the eye is drawn last and gets the highest z_order.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        let da = axes_world[a].dot(view_fwd);
        let db = axes_world[b].dot(view_fwd);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, &i) in order.iter().enumerate() {
        let base_z = rank as i32 * AXIS_Z_STRIDE;
        let tip = tip_of(axes_world[i]);
        let colour = colours[i];

        // Axis line from the origin to the tip.
        shapes.push(segment(origin, tip, LINE_THICKNESS, colour, base_z));

        // Filled circle background: the solid axis colour, so the disc reads as a
        // clean coloured dot rather than a dark hole inside the ring.
        shapes.push(disc(tip, CIRCLE_RADIUS, OverlayShape::Circle, colour, base_z + 1));

        // Letter glyph in a dark tint of the axis colour, so it reads against the
        // solid disc.
        push_letter(shapes, i, tip, darken(colour, 0.75), base_z + 2);
    }
}

/// A straight stroke from `a` to `b` as a rotated flat rectangle.
fn segment(a: glam::Vec2, b: glam::Vec2, thickness: f32, colour: [f32; 4], z: i32) -> OverlayShapeItem {
    let mid = (a + b) * 0.5;
    let d = b - a;
    let len = d.length().max(0.001);
    // Screen-space angle (Y down). The overlay rotates the shape around its centre.
    let angle = d.y.atan2(d.x);
    OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [mid.x - len * 0.5, mid.y - thickness * 0.5],
        [len, thickness],
    )
    .with_fill(OverlayFill::Solid(colour))
    .with_rotation(angle)
    .with_z_order(z)
}

/// Mix an RGB colour toward black by `t` (0 = unchanged, 1 = black), leaving
/// alpha intact. Used to darken the letter glyph so it reads against the solid
/// disc fill.
fn darken(c: [f32; 4], t: f32) -> [f32; 4] {
    let k = 1.0 - t;
    [c[0] * k, c[1] * k, c[2] * k, c[3]]
}

/// A circle or ring centred on `centre` with the given outer `radius`.
fn disc(
    centre: glam::Vec2,
    radius: f32,
    shape: OverlayShape,
    colour: [f32; 4],
    z: i32,
) -> OverlayShapeItem {
    OverlayShapeItem::new(
        shape,
        [centre.x - radius, centre.y - radius],
        [radius * 2.0, radius * 2.0],
    )
    .with_fill(OverlayFill::Solid(colour))
    .with_z_order(z)
}

/// Push the strokes of the letter for axis `i` (0=X, 1=Y, 2=Z), centred on
/// `centre`. Glyph coordinates use the overlay's downward Y.
fn push_letter(
    shapes: &mut Vec<OverlayShapeItem>,
    i: usize,
    centre: glam::Vec2,
    colour: [f32; 4],
    z: i32,
) {
    let hw = GLYPH_HALF;
    let hh = GLYPH_HALF;
    let top = centre.y - hh;
    let bottom = centre.y + hh;
    let left = centre.x - hw;
    let right = centre.x + hw;
    let cx = centre.x;
    let mut stroke = |a: glam::Vec2, b: glam::Vec2| {
        shapes.push(segment(a, b, GLYPH_THICKNESS, colour, z));
    };
    match i {
        // X: two crossing diagonals.
        0 => {
            stroke(glam::vec2(left, top), glam::vec2(right, bottom));
            stroke(glam::vec2(left, bottom), glam::vec2(right, top));
        }
        // Y: two strokes from the top meeting at the centre, one down to the bottom.
        1 => {
            stroke(glam::vec2(left, top), centre);
            stroke(glam::vec2(right, top), centre);
            stroke(centre, glam::vec2(cx, bottom));
        }
        // Z: top horizontal, diagonal, bottom horizontal.
        2 => {
            stroke(glam::vec2(left, top), glam::vec2(right, top));
            stroke(glam::vec2(right, top), glam::vec2(left, bottom));
            stroke(glam::vec2(left, bottom), glam::vec2(right, bottom));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

/// Test if a click at `screen_pos` (pixels, origin top-left) hits an axis circle.
/// Returns the target camera orientation if hit.
///
/// `viewport_rect`: (x, y, width, height) in pixels (the viewport panel rect).
pub fn hit_test(
    screen_pos: [f32; 2],
    viewport_rect: [f32; 4],
    orientation: glam::Quat,
) -> Option<AxisView> {
    let vp_x = viewport_rect[0];
    let vp_y = viewport_rect[1];
    let vp_h = viewport_rect[3];

    // Click position relative to viewport, Y increasing upward.
    let rel_x = screen_pos[0] - vp_x;
    let rel_y = vp_h - (screen_pos[1] - vp_y); // flip Y

    // Origin in pixels (bottom-left).
    let ox = ORIGIN_OFFSET;
    let oy = ORIGIN_OFFSET;

    // Derive view axes from orientation quaternion.
    let view_right = orientation * glam::Vec3::X;
    let view_up = orientation * glam::Vec3::Y;
    let view_fwd = orientation * glam::Vec3::Z; // from center toward eye

    let project = |world_axis: glam::Vec3| -> (f32, f32) {
        let sx = world_axis.dot(view_right);
        let sy = world_axis.dot(view_up);
        (ox + sx * LINE_LENGTH, oy + sy * LINE_LENGTH)
    };

    let axes = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
    // Snap targets: eye lands on each world axis respectively (Z-up convention).
    // X click -> Right view: eye at +X, up = Z.
    // Y click -> Front view: eye at +Y, up = Z.
    // Z click -> Top view:   eye at +Z, up = Y (identity).
    let frac_1_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
    let front = glam::Quat::from_xyzw(0.0, frac_1_sqrt_2, frac_1_sqrt_2, 0.0);
    let targets = [
        AxisView {
            orientation: glam::Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2) * front,
            axis_index: 0,
        }, // X -> Right view
        AxisView {
            orientation: front,
            axis_index: 1,
        }, // Y -> Front view
        AxisView {
            orientation: glam::Quat::IDENTITY,
            axis_index: 2,
        }, // Z -> Top view
    ];

    // Check front-to-back (reverse depth order) so frontmost wins.
    let mut order: [usize; 3] = [0, 1, 2];
    order.sort_by(|&a, &b| {
        let da = axes[a].dot(view_fwd);
        let db = axes[b].dot(view_fwd);
        db.partial_cmp(&da).unwrap() // front first
    });

    for &i in &order {
        let (tx, ty) = project(axes[i]);
        let dx = rel_x - tx;
        let dy = rel_y - ty;
        if dx * dx + dy * dy <= CIRCLE_RADIUS * CIRCLE_RADIUS {
            return Some(targets[i]);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlays_have_expected_shape_count() {
        let mut shapes = Vec::new();
        build_axes_overlays(800.0, 600.0, glam::Quat::IDENTITY, &mut shapes);
        // Per axis: 1 line + 1 circle + letter strokes (X=2, Y=3, Z=3).
        assert_eq!(shapes.len(), 3 + 3 + (2 + 3 + 3));
    }

    #[test]
    fn origin_sits_in_bottom_left() {
        let mut shapes = Vec::new();
        build_axes_overlays(800.0, 600.0, glam::Quat::IDENTITY, &mut shapes);
        // Every shape's centre is near the bottom-left origin region.
        for s in &shapes {
            let cx = s.position[0] + s.size[0] * 0.5;
            let cy = s.position[1] + s.size[1] * 0.5;
            assert!(cx < 120.0, "shape too far right: {cx}");
            assert!(cy > 480.0, "shape too far up: {cy}");
        }
    }

    #[test]
    fn hit_test_centre_misses() {
        // A click in the middle of the viewport hits no axis circle.
        let hit = hit_test([400.0, 300.0], [0.0, 0.0, 800.0, 600.0], glam::Quat::IDENTITY);
        assert!(hit.is_none());
    }
}
