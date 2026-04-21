//! Axes orientation indicator — a small XYZ gizmo rendered in the bottom-left
//! corner of the viewport as screen-space GPU geometry.
//!
//! Provides:
//! - `build_axes_geometry`: generates triangle-list vertices for axis lines,
//!   circles, and letter glyphs, all in NDC coordinates.
//! - `hit_test`: given a click position in pixels, returns the target
//!   (yaw, pitch) if an axis circle was hit.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Vertex type (matches axes_overlay.wgsl)
// ---------------------------------------------------------------------------

/// A 2D vertex for the axes indicator overlay (position in NDC + RGBA color).
///
/// Matches the layout expected by `axes_overlay.wgsl` at shader locations 0 and 1.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AxesVertex {
    /// 2D position in Normalized Device Coordinates (X right, Y up, range -1..1).
    pub(crate) position: [f32; 2],
    /// RGBA color in linear 0..1.
    pub(crate) color: [f32; 4],
}

impl AxesVertex {
    /// wgpu vertex buffer layout matching shader locations 0 (position) and 1 (color).
    pub(crate) fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<AxesVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

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

// Colors (same as the original egui version).
const X_COLOR: [f32; 4] = [0.878, 0.322, 0.322, 1.0]; // #e05252
const Y_COLOR: [f32; 4] = [0.361, 0.722, 0.361, 1.0]; // #5cb85c
const Z_COLOR: [f32; 4] = [0.290, 0.620, 1.000, 1.0]; // #4a9eff

// Layout parameters (pixels, converted to NDC during generation).
const ORIGIN_OFFSET: f32 = 40.0;
const LINE_LENGTH: f32 = 30.0;
const LINE_HALF_WIDTH: f32 = 1.0;
const CIRCLE_RADIUS: f32 = 9.0;
const CIRCLE_SEGMENTS: usize = 24;

// ---------------------------------------------------------------------------
// Geometry generation
// ---------------------------------------------------------------------------

/// Build all axes indicator triangles in NDC coordinates.
///
/// `viewport_size`: (width, height) in physical pixels.
/// `orientation`: current camera orientation quaternion.
///
/// Returns a `Vec<AxesVertex>` for a TriangleList draw call (no index buffer).
pub(crate) fn build_axes_geometry(
    viewport_w: f32,
    viewport_h: f32,
    orientation: glam::Quat,
) -> Vec<AxesVertex> {
    let mut verts = Vec::with_capacity(1024);

    // Pixel -> NDC conversion helpers.
    let px_to_ndc_x = |px: f32| -> f32 { px * 2.0 / viewport_w };
    let px_to_ndc_y = |py: f32| -> f32 { py * 2.0 / viewport_h };

    // Origin in NDC (bottom-left corner).
    let ox = -1.0 + px_to_ndc_x(ORIGIN_OFFSET);
    let oy = -1.0 + px_to_ndc_y(ORIGIN_OFFSET);

    // Derive view axes from orientation quaternion.
    let view_right = orientation * glam::Vec3::X;
    let view_up = orientation * glam::Vec3::Y;
    let view_fwd = orientation * glam::Vec3::Z; // from center toward eye

    let project = |world_axis: glam::Vec3| -> (f32, f32) {
        let sx = world_axis.dot(view_right);
        let sy = world_axis.dot(view_up); // NDC Y is up, no flip needed
        (sx * px_to_ndc_x(LINE_LENGTH), sy * px_to_ndc_y(LINE_LENGTH))
    };

    let axes_world = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z];
    let colors = [X_COLOR, Y_COLOR, Z_COLOR];
    let offsets: [(f32, f32); 3] = [
        project(glam::Vec3::X),
        project(glam::Vec3::Y),
        project(glam::Vec3::Z),
    ];

    // Sort back-to-front by depth (view_fwd dot axis).
    let mut order: [usize; 3] = [0, 1, 2];
    order.sort_by(|&a, &b| {
        let da = axes_world[a].dot(view_fwd);
        let db = axes_world[b].dot(view_fwd);
        da.partial_cmp(&db).unwrap()
    });

    for &i in &order {
        let (dx, dy) = offsets[i];
        let color = colors[i];
        let tip_x = ox + dx;
        let tip_y = oy + dy;

        // --- Axis line (thin quad) ---
        let lw_x = px_to_ndc_x(LINE_HALF_WIDTH);
        let lw_y = px_to_ndc_y(LINE_HALF_WIDTH);
        // Perpendicular direction to the line.
        let len = (dx * dx + dy * dy).sqrt().max(0.001);
        let perp_x = -dy / len;
        let perp_y = dx / len;
        let px_ = perp_x * lw_x;
        let py_ = perp_y * lw_y;

        push_quad(
            &mut verts,
            [ox + px_, oy + py_],
            [ox - px_, oy - py_],
            [tip_x - px_, tip_y - py_],
            [tip_x + px_, tip_y + py_],
            color,
        );

        // --- Circle background (filled) ---
        let bg_color = [color[0] * 0.33, color[1] * 0.33, color[2] * 0.33, 0.7];
        let cr_x = px_to_ndc_x(CIRCLE_RADIUS);
        let cr_y = px_to_ndc_y(CIRCLE_RADIUS);
        push_circle_filled(&mut verts, tip_x, tip_y, cr_x, cr_y, bg_color);

        // --- Circle outline (ring) ---
        let ring_inner = 0.82; // inner radius as fraction of outer
        push_circle_ring(&mut verts, tip_x, tip_y, cr_x, cr_y, ring_inner, color);

        // --- Letter glyph ---
        let glyph_hw = px_to_ndc_x(4.5); // half-width of letter
        let glyph_hh = px_to_ndc_y(4.5); // half-height of letter
        let glw_x = px_to_ndc_x(0.8); // glyph line half-width
        let glw_y = px_to_ndc_y(0.8);
        match i {
            0 => push_letter_x(
                &mut verts, tip_x, tip_y, glyph_hw, glyph_hh, glw_x, glw_y, color,
            ),
            1 => push_letter_y(
                &mut verts, tip_x, tip_y, glyph_hw, glyph_hh, glw_x, glw_y, color,
            ),
            2 => push_letter_z(
                &mut verts, tip_x, tip_y, glyph_hw, glyph_hh, glw_x, glw_y, color,
            ),
            _ => {}
        }
    }

    verts
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
        }, // X → Right view
        AxisView {
            orientation: front,
            axis_index: 1,
        }, // Y → Front view
        AxisView {
            orientation: glam::Quat::IDENTITY,
            axis_index: 2,
        }, // Z → Top view
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

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn push_quad(
    verts: &mut Vec<AxesVertex>,
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    d: [f32; 2],
    color: [f32; 4],
) {
    // Two triangles: ABC, ACD
    for &pos in &[a, b, c, a, c, d] {
        verts.push(AxesVertex {
            position: pos,
            color,
        });
    }
}

fn push_circle_filled(
    verts: &mut Vec<AxesVertex>,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    color: [f32; 4],
) {
    let step = 2.0 * PI / CIRCLE_SEGMENTS as f32;
    for i in 0..CIRCLE_SEGMENTS {
        let a0 = step * i as f32;
        let a1 = step * (i + 1) as f32;
        verts.push(AxesVertex {
            position: [cx, cy],
            color,
        });
        verts.push(AxesVertex {
            position: [cx + rx * a0.cos(), cy + ry * a0.sin()],
            color,
        });
        verts.push(AxesVertex {
            position: [cx + rx * a1.cos(), cy + ry * a1.sin()],
            color,
        });
    }
}

fn push_circle_ring(
    verts: &mut Vec<AxesVertex>,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    inner_frac: f32,
    color: [f32; 4],
) {
    let step = 2.0 * PI / CIRCLE_SEGMENTS as f32;
    let irx = rx * inner_frac;
    let iry = ry * inner_frac;
    for i in 0..CIRCLE_SEGMENTS {
        let a0 = step * i as f32;
        let a1 = step * (i + 1) as f32;
        let (c0, s0) = (a0.cos(), a0.sin());
        let (c1, s1) = (a1.cos(), a1.sin());
        let o0 = [cx + rx * c0, cy + ry * s0];
        let o1 = [cx + rx * c1, cy + ry * s1];
        let i0 = [cx + irx * c0, cy + iry * s0];
        let i1 = [cx + irx * c1, cy + iry * s1];
        // Two triangles per ring segment.
        for &pos in &[o0, i0, o1, o1, i0, i1] {
            verts.push(AxesVertex {
                position: pos,
                color,
            });
        }
    }
}

/// Draw letter "X" as two crossing diagonal strokes.
fn push_letter_x(
    verts: &mut Vec<AxesVertex>,
    cx: f32,
    cy: f32,
    hw: f32,
    hh: f32,
    lw_x: f32,
    lw_y: f32,
    color: [f32; 4],
) {
    // Diagonal \: top-left to bottom-right
    push_line_segment(verts, cx - hw, cy + hh, cx + hw, cy - hh, lw_x, lw_y, color);
    // Diagonal /: bottom-left to top-right
    push_line_segment(verts, cx - hw, cy - hh, cx + hw, cy + hh, lw_x, lw_y, color);
}

/// Draw letter "Y": two strokes from top meeting at center, one vertical down.
fn push_letter_y(
    verts: &mut Vec<AxesVertex>,
    cx: f32,
    cy: f32,
    hw: f32,
    hh: f32,
    lw_x: f32,
    lw_y: f32,
    color: [f32; 4],
) {
    // Top-left to center.
    push_line_segment(verts, cx - hw, cy + hh, cx, cy, lw_x, lw_y, color);
    // Top-right to center.
    push_line_segment(verts, cx + hw, cy + hh, cx, cy, lw_x, lw_y, color);
    // Center to bottom.
    push_line_segment(verts, cx, cy, cx, cy - hh, lw_x, lw_y, color);
}

/// Draw letter "Z": top horizontal, diagonal, bottom horizontal.
fn push_letter_z(
    verts: &mut Vec<AxesVertex>,
    cx: f32,
    cy: f32,
    hw: f32,
    hh: f32,
    lw_x: f32,
    lw_y: f32,
    color: [f32; 4],
) {
    // Top horizontal.
    push_line_segment(verts, cx - hw, cy + hh, cx + hw, cy + hh, lw_x, lw_y, color);
    // Diagonal: top-right to bottom-left.
    push_line_segment(verts, cx + hw, cy + hh, cx - hw, cy - hh, lw_x, lw_y, color);
    // Bottom horizontal.
    push_line_segment(verts, cx - hw, cy - hh, cx + hw, cy - hh, lw_x, lw_y, color);
}

/// Push a line segment as a thin quad (2 triangles, 6 vertices).
fn push_line_segment(
    verts: &mut Vec<AxesVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    lw_x: f32,
    lw_y: f32,
    color: [f32; 4],
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt().max(0.0001);
    // Perpendicular in NDC (accounts for aspect via separate lw_x/lw_y).
    let px = -(dy / len) * lw_x;
    let py = (dx / len) * lw_y;

    push_quad(
        verts,
        [x0 + px, y0 + py],
        [x0 - px, y0 - py],
        [x1 - px, y1 - py],
        [x1 + px, y1 + py],
        color,
    );
}
