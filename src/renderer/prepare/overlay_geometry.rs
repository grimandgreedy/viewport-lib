//! CPU tessellation and fill helpers for overlay rects, shapes, and polylines.

use super::*;

pub(super) fn gradient_lerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
        a[3] * (1.0 - t) + b[3] * t,
    ]
}

pub(super) fn sample_stops(stops: &[crate::renderer::types::GradientStop], t: f32) -> [f32; 4] {
    let mut colours = [[0.0; 4]; 4];
    let mut positions = [0.0, 1.0, 1.0, 1.0];
    let count = pack_stops(stops, &mut colours, &mut positions) as usize;
    let t = t.clamp(0.0, 1.0);
    if count <= 1 || t <= positions[0] {
        return colours[0];
    }
    for i in 0..(count - 1) {
        if t <= positions[i + 1] {
            let span = (positions[i + 1] - positions[i]).max(1e-6);
            let local = (t - positions[i]) / span;
            return gradient_lerp(colours[i], colours[i + 1], local);
        }
    }
    colours[count - 1]
}

pub(super) fn sample_overlay_fill(
    fill: &crate::renderer::types::OverlayFill,
    p: [f32; 2],
    min: [f32; 2],
    max: [f32; 2],
) -> [f32; 4] {
    let size = [(max[0] - min[0]).max(1e-6), (max[1] - min[1]).max(1e-6)];
    let centre = [min[0] + size[0] * 0.5, min[1] + size[1] * 0.5];
    match fill {
        crate::renderer::types::OverlayFill::Solid(c) => *c,
        crate::renderer::types::OverlayFill::LinearGradient {
            start_colour,
            end_colour,
            angle,
        } => {
            let dir = [angle.cos(), angle.sin()];
            let rel = [p[0] - centre[0], p[1] - centre[1]];
            let extent = (dir[0].abs() * size[0] + dir[1].abs() * size[1]).max(1e-6);
            let t = (rel[0] * dir[0] + rel[1] * dir[1]) / extent + 0.5;
            gradient_lerp(*start_colour, *end_colour, t)
        }
        crate::renderer::types::OverlayFill::RadialGradient {
            centre_colour,
            edge_colour,
        } => {
            let dx = p[0] - centre[0];
            let dy = p[1] - centre[1];
            let radius = ((size[0] * 0.5).powi(2) + (size[1] * 0.5).powi(2))
                .sqrt()
                .max(1e-6);
            gradient_lerp(
                *centre_colour,
                *edge_colour,
                (dx * dx + dy * dy).sqrt() / radius,
            )
        }
        crate::renderer::types::OverlayFill::ConicalGradient {
            start_colour,
            end_colour,
            offset_angle,
        } => {
            let a = (p[1] - centre[1]).atan2(p[0] - centre[0]) - offset_angle;
            let t = ((a / std::f32::consts::TAU) % 1.0 + 1.0) % 1.0;
            gradient_lerp(*start_colour, *end_colour, t)
        }
        crate::renderer::types::OverlayFill::LinearGradientMulti { stops, angle } => {
            let dir = [angle.cos(), angle.sin()];
            let rel = [p[0] - centre[0], p[1] - centre[1]];
            let extent = (dir[0].abs() * size[0] + dir[1].abs() * size[1]).max(1e-6);
            sample_stops(stops, (rel[0] * dir[0] + rel[1] * dir[1]) / extent + 0.5)
        }
        crate::renderer::types::OverlayFill::RadialGradientMulti { stops } => {
            let dx = p[0] - centre[0];
            let dy = p[1] - centre[1];
            let radius = ((size[0] * 0.5).powi(2) + (size[1] * 0.5).powi(2))
                .sqrt()
                .max(1e-6);
            sample_stops(stops, (dx * dx + dy * dy).sqrt() / radius)
        }
        crate::renderer::types::OverlayFill::ConicalGradientMulti {
            stops,
            offset_angle,
        } => {
            let a = (p[1] - centre[1]).atan2(p[0] - centre[0]) - offset_angle;
            sample_stops(stops, ((a / std::f32::consts::TAU) % 1.0 + 1.0) % 1.0)
        }
    }
}

pub(super) fn polygon_area(points: &[[f32; 2]]) -> f32 {
    let mut area = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area * 0.5
}

pub(super) fn point_in_tri(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let cross = |u: [f32; 2], v: [f32; 2], w: [f32; 2]| {
        (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    };
    let ab = cross(a, b, p);
    let bc = cross(b, c, p);
    let ca = cross(c, a, p);
    (ab >= -1e-5 && bc >= -1e-5 && ca >= -1e-5) || (ab <= 1e-5 && bc <= 1e-5 && ca <= 1e-5)
}

pub(super) fn triangulate_polygon(points: &[[f32; 2]]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 || polygon_area(points).abs() < 1e-5 {
        return Vec::new();
    }
    let ccw = polygon_area(points) > 0.0;
    let mut idx: Vec<usize> = if ccw {
        (0..n).collect()
    } else {
        (0..n).rev().collect()
    };
    let mut tris = Vec::with_capacity(n.saturating_sub(2));
    let mut guard = 0usize;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let mut clipped = false;
        for i in 0..idx.len() {
            let ia = idx[(i + idx.len() - 1) % idx.len()];
            let ib = idx[i];
            let ic = idx[(i + 1) % idx.len()];
            let a = points[ia];
            let b = points[ib];
            let c = points[ic];
            let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if cross <= 1e-5 {
                continue;
            }
            let contains = idx
                .iter()
                .any(|&j| j != ia && j != ib && j != ic && point_in_tri(points[j], a, b, c));
            if contains {
                continue;
            }
            tris.push([ia, ib, ic]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    }
    tris
}

pub(super) fn emit_filled_polyline(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    points: &[[f32; 2]],
    fill: &crate::renderer::types::OverlayFill,
    opacity: f32,
    vp_w: f32,
    vp_h: f32,
) {
    let Some((min, max)) = polyline_bounds(points) else {
        return;
    };
    for tri in triangulate_polygon(points) {
        for idx in tri {
            let p = points[idx];
            let mut colour = sample_overlay_fill(fill, p, min, max);
            colour[3] *= opacity;
            verts.push(crate::resources::OverlayTextVertex {
                position: px_to_ndc(p[0], p[1], vp_w, vp_h),
                uv: [0.0, 0.0],
                colour,
                use_texture: 0.0,
                clip_index: -1.0,
                clip_rect: [0.0; 4],
            });
        }
    }
}

/// Tessellate a polyline into a triangle list of `OverlayTextVertex`s.
///
/// At each interior point the stroke is extruded along the bisector of the
/// two adjacent segment normals, scaled by `1 / cos(half_angle)` so the
/// outer edge stays at constant `thickness / 2` from the centreline. When
/// that scale exceeds `mitre_limit`, or when `join` is `Bevel`, the joint
/// emits two separate ribs and a connecting triangle.
///
/// Open polylines end with the requested cap style (butt, square, or round).
/// Closed polylines wrap the last point back to the first and ignore `cap`.
#[allow(clippy::too_many_arguments)]
pub(super) fn tessellate_polyline(
    points: &[[f32; 2]],
    thickness: f32,
    closed: bool,
    join: crate::renderer::types::LineJoin,
    mitre_limit: f32,
    cap: crate::renderer::types::PolylineCap,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) -> Vec<crate::resources::OverlayTextVertex> {
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }
    let half_t = thickness * 0.5;

    let get = |i: i32| -> Option<[f32; 2]> {
        if closed {
            let idx = i.rem_euclid(n as i32);
            Some(points[idx as usize])
        } else if i >= 0 && (i as usize) < n {
            Some(points[i as usize])
        } else {
            None
        }
    };

    let normalize = |v: [f32; 2]| -> [f32; 2] {
        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
        if len < 1e-6 {
            [0.0, 0.0]
        } else {
            [v[0] / len, v[1] / len]
        }
    };
    let sub = |a: [f32; 2], b: [f32; 2]| -> [f32; 2] { [a[0] - b[0], a[1] - b[1]] };
    let add = |a: [f32; 2], b: [f32; 2]| -> [f32; 2] { [a[0] + b[0], a[1] + b[1]] };
    let scale = |a: [f32; 2], k: f32| -> [f32; 2] { [a[0] * k, a[1] * k] };
    let perp_left = |t: [f32; 2]| -> [f32; 2] { [-t[1], t[0]] };
    let dot = |a: [f32; 2], b: [f32; 2]| -> f32 { a[0] * b[0] + a[1] * b[1] };

    // Build "ribs": for each logical join, one or two (left, right) pairs.
    let mut ribs: Vec<([f32; 2], [f32; 2])> = Vec::with_capacity(n + 4);

    let iter_count = if closed { n + 1 } else { n };
    for i in 0..iter_count {
        let cur_idx = if closed { i % n } else { i };
        let cur = points[cur_idx];
        let prev = get(cur_idx as i32 - 1);
        let next = if closed {
            Some(points[(cur_idx + 1) % n])
        } else if cur_idx + 1 < n {
            Some(points[cur_idx + 1])
        } else {
            None
        };

        let in_t = prev.map(|p| normalize(sub(cur, p)));
        let out_t = next.map(|q| normalize(sub(q, cur)));

        match (in_t, out_t) {
            (Some(t1), Some(t2)) => {
                let n1 = perp_left(t1);
                let n2 = perp_left(t2);
                let bisect = normalize(add(n1, n2));
                let cos_half = dot(n1, bisect).abs().max(1e-4);
                let mitre_scale = 1.0 / cos_half;
                let use_mitre = match join {
                    crate::renderer::types::LineJoin::Mitre => mitre_scale <= mitre_limit,
                    crate::renderer::types::LineJoin::Bevel => false,
                };
                if use_mitre {
                    let off = scale(bisect, half_t * mitre_scale);
                    ribs.push((add(cur, off), sub(cur, off)));
                } else {
                    // Bevel: two ribs at this point, one per adjacent segment.
                    let off1 = scale(n1, half_t);
                    let off2 = scale(n2, half_t);
                    ribs.push((add(cur, off1), sub(cur, off1)));
                    ribs.push((add(cur, off2), sub(cur, off2)));
                }
            }
            (Some(t1), None) => {
                let n1 = perp_left(t1);
                let off = scale(n1, half_t);
                ribs.push((add(cur, off), sub(cur, off)));
            }
            (None, Some(t2)) => {
                let n2 = perp_left(t2);
                let off = scale(n2, half_t);
                ribs.push((add(cur, off), sub(cur, off)));
            }
            (None, None) => {}
        }
    }

    if ribs.len() < 2 {
        return Vec::new();
    }

    // End tangents for cap placement (zero when the end segment is degenerate).
    let t_start = normalize(sub(points[1], points[0]));
    let t_end = normalize(sub(points[n - 1], points[n - 2]));

    // Square caps: push the endpoint ribs out along the tangent so the
    // stroke rectangle extends `thickness / 2` beyond each endpoint.
    if !closed && cap == crate::renderer::types::PolylineCap::Square {
        if let Some(first) = ribs.first_mut() {
            first.0 = sub(first.0, scale(t_start, half_t));
            first.1 = sub(first.1, scale(t_start, half_t));
        }
        if let Some(last) = ribs.last_mut() {
            last.0 = add(last.0, scale(t_end, half_t));
            last.1 = add(last.1, scale(t_end, half_t));
        }
    }

    // Convert rib pairs to triangle list. Each pair of consecutive ribs
    // forms a quad split into two triangles.
    let mut verts: Vec<crate::resources::OverlayTextVertex> =
        Vec::with_capacity((ribs.len() - 1) * 6);
    let mut emit = |px: [f32; 2]| {
        verts.push(crate::resources::OverlayTextVertex {
            position: px_to_ndc(px[0], px[1], vp_w, vp_h),
            uv: [0.0, 0.0],
            colour,
            use_texture: 0.0,
            clip_index: -1.0,
            clip_rect: [0.0; 4],
        });
    };
    for w in ribs.windows(2) {
        let (l0, r0) = w[0];
        let (l1, r1) = w[1];
        emit(l0);
        emit(r0);
        emit(l1);
        emit(r0);
        emit(r1);
        emit(l1);
    }

    // Round caps: a semicircle fan on the outside half of each endpoint.
    if !closed && cap == crate::renderer::types::PolylineCap::Round {
        let segs = 8;
        for (centre, outward) in [(points[0], scale(t_start, -1.0)), (points[n - 1], t_end)] {
            if outward == [0.0, 0.0] {
                continue;
            }
            let normal = perp_left(outward);
            for i in 0..segs {
                let a0 = std::f32::consts::PI * i as f32 / segs as f32;
                let a1 = std::f32::consts::PI * (i + 1) as f32 / segs as f32;
                let d0 = add(scale(normal, a0.cos()), scale(outward, a0.sin()));
                let d1 = add(scale(normal, a1.cos()), scale(outward, a1.sin()));
                emit(centre);
                emit(add(centre, scale(d0, half_t)));
                emit(add(centre, scale(d1, half_t)));
            }
        }
    }
    verts
}

/// Working point list and cumulative arc-length table for a polyline. For
/// closed paths the first point is appended again so the table covers the
/// closing segment.
pub(super) fn polyline_arc_table(points: &[[f32; 2]], closed: bool) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut pts: Vec<[f32; 2]> = points.to_vec();
    if closed && points.len() >= 2 {
        pts.push(points[0]);
    }
    let mut cum = Vec::with_capacity(pts.len());
    cum.push(0.0);
    let mut s = 0.0f32;
    for w in pts.windows(2) {
        let dx = w[1][0] - w[0][0];
        let dy = w[1][1] - w[0][1];
        s += (dx * dx + dy * dy).sqrt();
        cum.push(s);
    }
    (pts, cum)
}

/// Interpolate the point at arc length `s` along the path (clamped to the
/// path range).
pub(super) fn point_at_arc(pts: &[[f32; 2]], cum: &[f32], s: f32) -> [f32; 2] {
    let total = *cum.last().unwrap_or(&0.0);
    let s = s.clamp(0.0, total);
    let i = cum
        .partition_point(|&c| c < s)
        .saturating_sub(1)
        .min(pts.len().saturating_sub(2));
    let seg = (cum[i + 1] - cum[i]).max(1e-6);
    let t = ((s - cum[i]) / seg).clamp(0.0, 1.0);
    [
        pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
        pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t,
    ]
}

/// Split a path into the visible sub-paths of a dash pattern. Each sub-path
/// is an open point list covering one dash, including the interpolated dash
/// endpoints and any original waypoints in between. For closed paths, a dash
/// spanning the final-to-first seam is merged into a single sub-path.
pub(super) fn dash_subpaths(
    pts: &[[f32; 2]],
    cum: &[f32],
    dash: f32,
    gap: f32,
    offset: f32,
    closed: bool,
) -> Vec<Vec<[f32; 2]>> {
    let total = *cum.last().unwrap_or(&0.0);
    if total <= 1e-4 {
        return Vec::new();
    }
    let period = dash + gap;

    // Visible intervals are [k * period - offset, k * period - offset + dash).
    let k_min = ((offset - dash) / period).floor() as i64;
    let k_max = ((total + offset) / period).ceil() as i64;
    let mut intervals: Vec<(f32, f32)> = Vec::new();
    for k in k_min..=k_max {
        let a = (k as f32 * period - offset).max(0.0);
        let b = (k as f32 * period - offset + dash).min(total);
        if b - a > 1e-4 {
            intervals.push((a, b));
        }
    }
    if intervals.is_empty() {
        return Vec::new();
    }

    // A dash crossing the closed-path seam shows up as one interval ending at
    // `total` and another starting at 0; stitch them into one sub-path so the
    // seam gets a joint instead of two caps.
    let merge_seam = closed
        && intervals.len() >= 2
        && intervals.first().unwrap().0 <= 1e-4
        && total - intervals.last().unwrap().1 <= 1e-4;

    let extract = |a: f32, b: f32| -> Vec<[f32; 2]> {
        let mut sub = Vec::new();
        sub.push(point_at_arc(pts, cum, a));
        for (i, &c) in cum.iter().enumerate() {
            if c > a + 1e-4 && c < b - 1e-4 {
                sub.push(pts[i]);
            }
        }
        sub.push(point_at_arc(pts, cum, b));
        sub
    };

    let mut subs: Vec<Vec<[f32; 2]>> = intervals.iter().map(|&(a, b)| extract(a, b)).collect();
    if merge_seam {
        let first = subs.remove(0);
        let last = subs.last_mut().unwrap();
        last.pop(); // ends at `total`, same point as the first sub-path's start
        last.extend(first);
    }
    subs
}

/// Emit a filled disc (triangle fan) in screen pixel coordinates.
pub(super) fn emit_disc(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    centre: [f32; 2],
    radius: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let segs = 10;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: px_to_ndc(pos[0], pos[1], vp_w, vp_h),
        uv: [0.0, 0.0],
        colour,
        use_texture: 0.0,
        clip_index: -1.0,
        clip_rect: [0.0; 4],
    };
    for i in 0..segs {
        let a0 = std::f32::consts::TAU * i as f32 / segs as f32;
        let a1 = std::f32::consts::TAU * (i + 1) as f32 / segs as f32;
        verts.push(v(centre));
        verts.push(v([
            centre[0] + a0.cos() * radius,
            centre[1] + a0.sin() * radius,
        ]));
        verts.push(v([
            centre[0] + a1.cos() * radius,
            centre[1] + a1.sin() * radius,
        ]));
    }
}

/// Emit the stroke geometry for a polyline item, honouring its cap style and
/// stroke pattern. `colour` must already have opacity applied.
pub(super) fn emit_polyline_stroke(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    poly: &crate::renderer::types::OverlayPolylineItem,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    use crate::renderer::types::StrokePattern;
    match poly.stroke_pattern {
        StrokePattern::Solid => {
            verts.extend(tessellate_polyline(
                &poly.points,
                poly.thickness,
                poly.closed,
                poly.join,
                poly.mitre_limit,
                poly.cap,
                colour,
                vp_w,
                vp_h,
            ));
        }
        StrokePattern::Dashed {
            dash_length,
            gap_length,
            offset,
        } => {
            // Degenerate patterns collapse to a solid stroke rather than
            // disappearing or looping forever. The 0.25 px floor bounds the
            // sub-path count on long paths.
            if dash_length <= 0.0 || gap_length <= 0.0 {
                let solid = crate::renderer::types::OverlayPolylineItem {
                    stroke_pattern: StrokePattern::Solid,
                    ..poly.clone()
                };
                emit_polyline_stroke(verts, &solid, colour, vp_w, vp_h);
                return;
            }
            let dash = dash_length.max(0.25);
            let gap = gap_length.max(0.25);
            let (pts, cum) = polyline_arc_table(&poly.points, poly.closed);
            for sub in dash_subpaths(&pts, &cum, dash, gap, offset, poly.closed) {
                verts.extend(tessellate_polyline(
                    &sub,
                    poly.thickness,
                    false,
                    poly.join,
                    poly.mitre_limit,
                    poly.cap,
                    colour,
                    vp_w,
                    vp_h,
                ));
            }
        }
        StrokePattern::Dotted { spacing, offset } => {
            let (pts, cum) = polyline_arc_table(&poly.points, poly.closed);
            if pts.len() < 2 {
                return;
            }
            let total = *cum.last().unwrap();
            let spacing = spacing.max(0.5);
            let mut s = offset.rem_euclid(spacing);
            // On closed paths the point at `total` is the point at 0; skip it
            // so the seam doesn't get a doubled dot.
            let limit = if poly.closed {
                total - 1e-3
            } else {
                total + 1e-3
            };
            while s < limit {
                let c = point_at_arc(&pts, &cum, s);
                emit_disc(verts, c, poly.thickness * 0.5, colour, vp_w, vp_h);
                s += spacing;
            }
        }
    }
}

/// Encode `TileMode` as the float flag the texture shader consumes.
pub(super) fn tile_mode_to_f(m: crate::renderer::types::TileMode) -> f32 {
    match m {
        crate::renderer::types::TileMode::Stretch => 0.0,
        crate::renderer::types::TileMode::Tile => 1.0,
        crate::renderer::types::TileMode::Mirror => 2.0,
    }
}

/// Pack a multi-stop gradient stop list into the fixed-cap vertex arrays.
/// Returns the active stop count (clamped to [2, OVERLAY_MAX_GRADIENT_STOPS]).
/// Stops are sorted by position; remaining slots are filled with the last
/// stop so out-of-range loop iterations are a no-op.
pub(super) fn pack_stops(
    stops: &[crate::renderer::types::GradientStop],
    colours: &mut [[f32; 4]; 4],
    positions: &mut [f32; 4],
) -> f32 {
    let cap = crate::renderer::types::OVERLAY_MAX_GRADIENT_STOPS;
    let mut buf: Vec<crate::renderer::types::GradientStop> = stops.iter().copied().collect();
    buf.sort_by(|a, b| {
        a.position
            .partial_cmp(&b.position)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if buf.is_empty() {
        // Empty stops list: degrade to transparent black.
        let s = crate::renderer::types::GradientStop::new(0.0, [0.0; 4]);
        buf = vec![s, s];
    } else if buf.len() == 1 {
        buf.push(buf[0]);
    }
    let n = buf.len().min(cap);
    for i in 0..n {
        colours[i] = buf[i].colour;
        positions[i] = buf[i].position.clamp(0.0, 1.0);
    }
    // Fill remaining slots with the last stop so iteration past `count`
    // returns the same colour and produces a no-op interpolation.
    for i in n..cap {
        colours[i] = buf[n - 1].colour;
        positions[i] = positions[n - 1];
    }
    n as f32
}

/// Emit a solid-colour quad (6 vertices) in screen pixel coordinates.
pub(super) fn emit_solid_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let tl = px_to_ndc(x0, y0, vp_w, vp_h);
    let tr = px_to_ndc(x1, y0, vp_w, vp_h);
    let bl = px_to_ndc(x0, y1, vp_w, vp_h);
    let br = px_to_ndc(x1, y1, vp_w, vp_h);
    let uv = [0.0, 0.0];
    let tex = 0.0;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        clip_index: -1.0,
        clip_rect: [0.0; 4],
    };
    verts.extend_from_slice(&[v(tl), v(bl), v(tr), v(tr), v(bl), v(br)]);
}

/// Emit a textured quad (6 vertices) for a glyph in screen pixel coordinates.
pub(super) fn emit_textured_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let tl = px_to_ndc(x0, y0, vp_w, vp_h);
    let tr = px_to_ndc(x1, y0, vp_w, vp_h);
    let bl = px_to_ndc(x0, y1, vp_w, vp_h);
    let br = px_to_ndc(x1, y1, vp_w, vp_h);
    let tex = 1.0;
    let v = |pos: [f32; 2], uv: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        clip_index: -1.0,
        clip_rect: [0.0; 4],
    };
    // UV layout: top-left = uv_min, bottom-right = uv_max.
    verts.extend_from_slice(&[
        v(tl, uv_min),
        v(bl, [uv_min[0], uv_max[1]]),
        v(tr, [uv_max[0], uv_min[1]]),
        v(tr, [uv_max[0], uv_min[1]]),
        v(bl, [uv_min[0], uv_max[1]]),
        v(br, uv_max),
    ]);
}

/// Emit one textured quad per glyph in a laid-out run.
///
/// Shared by the label and glyph-run overlay paths: both produce a slice of
/// [`GlyphQuad`]s (positions and atlas UVs in logical pixels) and place them at
/// `(origin_x, origin_y)` with a single tint `colour`.
pub(super) fn emit_glyph_quads(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    quads: &[crate::resources::overlay::font::GlyphQuad],
    origin_x: f32,
    origin_y: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    for gq in quads {
        let gx = origin_x + gq.pos[0];
        let gy = origin_y + gq.pos[1];
        emit_textured_quad(
            verts,
            gx,
            gy,
            gx + gq.size[0],
            gy + gq.size[1],
            gq.uv_min,
            gq.uv_max,
            colour,
            vp_w,
            vp_h,
        );
    }
}

/// Emit one textured quad per glyph, each with its own tint colour.
///
/// The glyph-run path uses this so a single run can carry per-glyph colours. Each
/// entry pairs a laid-out [`GlyphQuad`] with the colour to draw it in.
pub(super) fn emit_glyph_quads_colored(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    quads: &[(crate::resources::overlay::font::GlyphQuad, [f32; 4])],
    origin_x: f32,
    origin_y: f32,
    vp_w: f32,
    vp_h: f32,
) {
    for (gq, colour) in quads {
        let gx = origin_x + gq.pos[0];
        let gy = origin_y + gq.pos[1];
        emit_textured_quad(
            verts,
            gx,
            gy,
            gx + gq.size[0],
            gy + gq.size[1],
            gq.uv_min,
            gq.uv_max,
            *colour,
            vp_w,
            vp_h,
        );
    }
}

/// Emit a thin screen-space line as a quad (6 vertices).
pub(super) fn emit_line_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 {
        return;
    }
    let half = thickness * 0.5;
    let nx = -dy / len * half;
    let ny = dx / len * half;

    let p0 = px_to_ndc(x0 + nx, y0 + ny, vp_w, vp_h);
    let p1 = px_to_ndc(x0 - nx, y0 - ny, vp_w, vp_h);
    let p2 = px_to_ndc(x1 + nx, y1 + ny, vp_w, vp_h);
    let p3 = px_to_ndc(x1 - nx, y1 - ny, vp_w, vp_h);
    let uv = [0.0, 0.0];
    let tex = 0.0;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        clip_index: -1.0,
        clip_rect: [0.0; 4],
    };
    verts.extend_from_slice(&[v(p0), v(p1), v(p2), v(p2), v(p1), v(p3)]);
}

/// Apply an opacity multiplier to a colour's alpha channel.
#[inline]
pub(super) fn apply_opacity(colour: [f32; 4], opacity: f32) -> [f32; 4] {
    [colour[0], colour[1], colour[2], colour[3] * opacity]
}

#[cfg(test)]
mod stroke_tests {
    use super::*;
    use crate::renderer::types::{LineJoin, OverlayPolylineItem, PolylineCap, StrokePattern};

    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    fn line_item(pattern: StrokePattern, cap: PolylineCap) -> OverlayPolylineItem {
        OverlayPolylineItem {
            points: vec![[0.0, 0.0], [100.0, 0.0]],
            thickness: 4.0,
            stroke_pattern: pattern,
            cap,
            ..Default::default()
        }
    }

    #[test]
    fn point_at_arc_interpolates_across_segments() {
        let (pts, cum) = polyline_arc_table(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], false);
        assert_eq!(*cum.last().unwrap(), 20.0);
        assert_eq!(point_at_arc(&pts, &cum, 5.0), [5.0, 0.0]);
        assert_eq!(point_at_arc(&pts, &cum, 15.0), [10.0, 5.0]);
        // Clamped past the end.
        assert_eq!(point_at_arc(&pts, &cum, 99.0), [10.0, 10.0]);
    }

    #[test]
    fn dash_subpaths_open_line() {
        let (pts, cum) = polyline_arc_table(&[[0.0, 0.0], [100.0, 0.0]], false);
        let subs = dash_subpaths(&pts, &cum, 10.0, 10.0, 0.0, false);
        assert_eq!(subs.len(), 5);
        assert_eq!(subs[0], vec![[0.0, 0.0], [10.0, 0.0]]);
        assert_eq!(subs[4], vec![[80.0, 0.0], [90.0, 0.0]]);
    }

    #[test]
    fn dash_subpaths_merges_closed_seam() {
        // 10x10 square, perimeter 40. dash 15 / gap 5 / offset 5 gives
        // visible intervals [0, 10], [15, 30], [35, 40]: the last wraps into
        // the first across the seam.
        let square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let (pts, cum) = polyline_arc_table(&square, true);
        let subs = dash_subpaths(&pts, &cum, 15.0, 5.0, 5.0, true);
        assert_eq!(subs.len(), 2);
        let seam = subs.last().unwrap();
        assert_eq!(seam.first().unwrap(), &[0.0, 5.0]);
        assert!(seam.contains(&[0.0, 0.0]));
        assert_eq!(seam.last().unwrap(), &[10.0, 0.0]);
    }

    #[test]
    fn caps_extend_the_stroke() {
        let tess = |cap| {
            tessellate_polyline(
                &[[10.0, 50.0], [90.0, 50.0]],
                4.0,
                false,
                LineJoin::Mitre,
                4.0,
                cap,
                WHITE,
                100.0,
                100.0,
            )
        };
        let butt = tess(PolylineCap::Butt);
        let square = tess(PolylineCap::Square);
        let round = tess(PolylineCap::Round);
        assert_eq!(butt.len(), 6);
        assert_eq!(square.len(), 6);
        // Round caps add two 8-segment fans.
        assert_eq!(round.len(), 6 + 2 * 8 * 3);
        let min_x = |v: &[crate::resources::OverlayTextVertex]| {
            v.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min)
        };
        assert!(square.iter().all(|v| v.position[0].is_finite()));
        assert!(min_x(&square) < min_x(&butt));
        assert!(min_x(&round) < min_x(&butt));
    }

    #[test]
    fn dotted_stroke_emits_discs() {
        let mut verts = Vec::new();
        let item = line_item(
            StrokePattern::Dotted {
                spacing: 10.0,
                offset: 0.0,
            },
            PolylineCap::Butt,
        );
        emit_polyline_stroke(&mut verts, &item, WHITE, 100.0, 100.0);
        // Dots at 0, 10, .., 100 = 11 discs of 10 fan triangles each.
        assert_eq!(verts.len(), 11 * 10 * 3);
    }

    #[test]
    fn degenerate_dash_pattern_falls_back_to_solid() {
        let mut dashed = Vec::new();
        emit_polyline_stroke(
            &mut dashed,
            &line_item(
                StrokePattern::Dashed {
                    dash_length: 10.0,
                    gap_length: 0.0,
                    offset: 0.0,
                },
                PolylineCap::Butt,
            ),
            WHITE,
            100.0,
            100.0,
        );
        let mut solid = Vec::new();
        emit_polyline_stroke(
            &mut solid,
            &line_item(StrokePattern::Solid, PolylineCap::Butt),
            WHITE,
            100.0,
            100.0,
        );
        assert_eq!(dashed.len(), solid.len());
    }
}

/// Emit a rounded rectangle as solid quads: one center rect + four edge rects +
/// four corner fans.  This is a CPU tessellation approach that avoids shader
/// changes.
pub(super) fn emit_rounded_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    radius: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let w = x1 - x0;
    let h = y1 - y0;
    let r = radius.min(w * 0.5).min(h * 0.5).max(0.0);

    if r < 0.5 {
        emit_solid_quad(verts, x0, y0, x1, y1, colour, vp_w, vp_h);
        return;
    }

    // Center cross (two rects that cover everything except the corners).
    // Horizontal bar (full width, inset top/bottom by r).
    emit_solid_quad(verts, x0, y0 + r, x1, y1 - r, colour, vp_w, vp_h);
    // Top bar (inset left/right by r, top edge).
    emit_solid_quad(verts, x0 + r, y0, x1 - r, y0 + r, colour, vp_w, vp_h);
    // Bottom bar.
    emit_solid_quad(verts, x0 + r, y1 - r, x1 - r, y1, colour, vp_w, vp_h);

    // Four corner fans.
    let corners = [
        (
            x0 + r,
            y0 + r,
            std::f32::consts::PI,
            std::f32::consts::FRAC_PI_2 * 3.0,
        ), // top-left
        (
            x1 - r,
            y0 + r,
            std::f32::consts::FRAC_PI_2 * 3.0,
            std::f32::consts::TAU,
        ), // top-right
        (x1 - r, y1 - r, 0.0, std::f32::consts::FRAC_PI_2), // bottom-right
        (
            x0 + r,
            y1 - r,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
        ), // bottom-left
    ];
    let segments = 6;
    let uv = [0.0, 0.0];
    let tex = 0.0;
    let v = |pos: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        clip_index: -1.0,
        clip_rect: [0.0; 4],
    };
    for (cx, cy, start, end) in corners {
        let center = px_to_ndc(cx, cy, vp_w, vp_h);
        for i in 0..segments {
            let a0 = start + (end - start) * i as f32 / segments as f32;
            let a1 = start + (end - start) * (i + 1) as f32 / segments as f32;
            let p0 = px_to_ndc(cx + a0.cos() * r, cy + a0.sin() * r, vp_w, vp_h);
            let p1 = px_to_ndc(cx + a1.cos() * r, cy + a1.sin() * r, vp_w, vp_h);
            verts.extend_from_slice(&[v(center), v(p0), v(p1)]);
        }
    }
}
