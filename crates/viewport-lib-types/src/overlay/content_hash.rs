//! Deterministic content hashing for overlay items.

use std::hash::Hasher;

use crate::colour::Colour;
use crate::overlay::Alignment;
use crate::overlay::{
    AnchorX, AnchorY, AnimTrack, BackdropEffects, FillRule, FontHandle, GlyphRunItem, GradientStop,
    LabelItem, LineCap, LineJoin, NineSlice, OverlayAnchoring, OverlayAnimations, OverlayClip,
    OverlayEasing, OverlayFill, OverlayGeometryId, OverlayOrigin, OverlayPolylineItem,
    OverlayShape, OverlayShapeItem, OverlayStroke, OverlayStyle, OverlayTextureId,
    OverlayTransform, PathSegment, PolylineCap, PositionedGlyph, RepeatMode, RetainedOverlay,
    ShadowLayer, StrokePattern, SubPath, TextStyle, TextureTransform, TileMode, TriangleDirection,
};

/// Feeds everything that affects how an overlay value draws into a hasher, so
/// a consumer that retains overlay geometry can tell when a group needs
/// recompiling, or find a group it has compiled before.
///
/// Equal values hash equal: floats are hashed by their bits with `-0.0` folded
/// into `0.0`. Collections hash their length, then their elements in order.
///
/// The bytes fed to the hasher are the same on every target and toolchain:
/// enum variants are explicit tags, lengths are `u64`, and integers and float
/// bits are written little-endian. So with the same hasher and seed, a value
/// hashes the same in a native and a wasm build, and across runs. It is tied
/// to the viewport-lib version, though: a new field or variant changes every
/// hash of its type, so a key kept across an upgrade will not match.
///
/// [`content_hash_at`](Self::content_hash_at) hashes screen positions relative
/// to `origin`, so content that only moved keeps its hash and a retained group
/// can follow it with `RetainedOverlay::transform`. The screen positions are
/// the corners of `clip.rect`, plus each value's own placement: `points` on a
/// polyline, and `transform.translate` on everything else. A polyline's
/// `translate` is a nudge on top of its points, so it is hashed as it is, along
/// with a world anchor and a transform's `pivot`, which is local to the item. The
/// relative positions are computed in `f32`, so a move by an amount that is
/// not exactly representable can round differently and change the hash.
///
/// What the hash cannot see:
/// - texture and font contents: `OverlayTextureId` and `FontHandle` hash by
///   id, so re-uploading under the same handle keeps the hash;
/// - the `pixels_per_point` a group is compiled at, which a cache key needs
///   alongside the hash.
///
/// An item's own `animations` are hashed even though a compiled group ignores
/// them, because they change what the item draws when it is drawn directly.
pub trait OverlayContentHash {
    /// Hash this value with screen positions measured from `origin`.
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H);

    /// Hash this value with screen positions as they are.
    fn content_hash<H: Hasher>(&self, state: &mut H) {
        self.content_hash_at([0.0, 0.0], state);
    }
}

fn put_u16<H: Hasher>(state: &mut H, v: u16) {
    state.write(&v.to_le_bytes());
}

fn put_u32<H: Hasher>(state: &mut H, v: u32) {
    state.write(&v.to_le_bytes());
}

fn put_u64<H: Hasher>(state: &mut H, v: u64) {
    state.write(&v.to_le_bytes());
}

fn put_i32<H: Hasher>(state: &mut H, v: i32) {
    state.write(&v.to_le_bytes());
}

fn hash_f32<H: Hasher>(state: &mut H, v: f32) {
    // `0.0 == -0.0`, so they must hash the same.
    let v = if v == 0.0 { 0.0 } else { v };
    put_u32(state, v.to_bits());
}

fn hash_f64<H: Hasher>(state: &mut H, v: f64) {
    let v = if v == 0.0 { 0.0 } else { v };
    put_u64(state, v.to_bits());
}

fn hash_floats<H: Hasher>(state: &mut H, v: &[f32]) {
    for &x in v {
        hash_f32(state, x);
    }
}

fn hash_point<H: Hasher>(state: &mut H, p: [f32; 2], origin: [f32; 2]) {
    hash_f32(state, p[0] - origin[0]);
    hash_f32(state, p[1] - origin[1]);
}

fn hash_len<H: Hasher>(state: &mut H, n: usize) {
    put_u64(state, n as u64);
}

fn hash_bool<H: Hasher>(state: &mut H, b: bool) {
    state.write_u8(b as u8);
}

fn hash_opt_u32<H: Hasher>(state: &mut H, v: Option<u32>) {
    match v {
        None => state.write_u8(0),
        Some(v) => {
            state.write_u8(1);
            put_u32(state, v);
        }
    }
}

impl<T: OverlayContentHash + ?Sized> OverlayContentHash for Box<T> {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        (**self).content_hash_at(origin, state);
    }
}

impl<T: OverlayContentHash> OverlayContentHash for Option<T> {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        match self {
            None => state.write_u8(0),
            Some(v) => {
                state.write_u8(1);
                v.content_hash_at(origin, state);
            }
        }
    }
}

impl<T: OverlayContentHash> OverlayContentHash for [T] {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        hash_len(state, self.len());
        for v in self {
            v.content_hash_at(origin, state);
        }
    }
}

impl<T: OverlayContentHash> OverlayContentHash for Vec<T> {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        self.as_slice().content_hash_at(origin, state);
    }
}

impl OverlayContentHash for Colour {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        hash_floats(state, &self.to_linear_rgba());
    }
}

impl OverlayContentHash for OverlayTextureId {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        put_u64(state, self.0);
    }
}

impl OverlayContentHash for FontHandle {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        put_u64(state, self.0 as u64);
    }
}

impl OverlayContentHash for OverlayGeometryId {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        put_u32(state, self.index);
        put_u32(state, self.generation);
    }
}

impl OverlayContentHash for AnchorX {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            AnchorX::Left => 0,
            AnchorX::Middle => 1,
            AnchorX::Right => 2,
        });
    }
}

impl OverlayContentHash for AnchorY {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            AnchorY::Top => 0,
            AnchorY::Middle => 1,
            AnchorY::Bottom => 2,
            AnchorY::Baseline => 3,
        });
    }
}

impl OverlayContentHash for Alignment {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self { x, y } = self;
        x.content_hash_at(origin, state);
        y.content_hash_at(origin, state);
    }
}

impl OverlayContentHash for OverlayOrigin {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        match self {
            OverlayOrigin::Viewport(a) => {
                state.write_u8(0);
                a.content_hash_at(origin, state);
            }
            OverlayOrigin::World(p) => {
                state.write_u8(1);
                hash_floats(state, p);
            }
        }
    }
}

impl OverlayContentHash for OverlayAnchoring {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            origin: anchor,
            align,
        } = self;
        anchor.content_hash_at(origin, state);
        align.content_hash_at(origin, state);
    }
}

impl OverlayContentHash for OverlayTransform {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            translate,
            rotation,
            pivot,
            scale,
        } = self;
        hash_point(state, *translate, origin);
        hash_f32(state, *rotation);
        hash_floats(state, pivot);
        hash_f32(state, *scale);
    }
}

impl OverlayContentHash for OverlayClip {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self { rect, mask } = self;
        match rect {
            None => state.write_u8(0),
            Some([x0, y0, x1, y1]) => {
                state.write_u8(1);
                hash_point(state, [*x0, *y0], origin);
                hash_point(state, [*x1, *y1], origin);
            }
        }
        hash_opt_u32(state, *mask);
    }
}

impl OverlayContentHash for TextStyle {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self { font, size } = self;
        font.content_hash_at(origin, state);
        hash_f32(state, *size);
    }
}

impl OverlayContentHash for GradientStop {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self { position, colour } = self;
        hash_f32(state, *position);
        colour.content_hash_at(origin, state);
    }
}

impl OverlayContentHash for TileMode {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            TileMode::Stretch => 0,
            TileMode::Tile => 1,
            TileMode::Mirror => 2,
        });
    }
}

impl OverlayContentHash for TextureTransform {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            offset,
            scale,
            rotation,
            tile_mode,
            flip_x,
            flip_y,
        } = self;
        hash_floats(state, offset);
        hash_floats(state, scale);
        hash_f32(state, *rotation);
        tile_mode.content_hash_at(origin, state);
        hash_bool(state, *flip_x);
        hash_bool(state, *flip_y);
    }
}

impl OverlayContentHash for NineSlice {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            insets_px,
            centre_mode,
            edge_mode,
        } = self;
        hash_floats(state, insets_px);
        centre_mode.content_hash_at(origin, state);
        edge_mode.content_hash_at(origin, state);
    }
}

impl OverlayContentHash for OverlayFill {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        match self {
            OverlayFill::Solid(c) => {
                state.write_u8(0);
                c.content_hash_at(origin, state);
            }
            OverlayFill::LinearGradient {
                start_colour,
                end_colour,
                angle,
            } => {
                state.write_u8(1);
                start_colour.content_hash_at(origin, state);
                end_colour.content_hash_at(origin, state);
                hash_f32(state, *angle);
            }
            OverlayFill::RadialGradient {
                centre_colour,
                edge_colour,
            } => {
                state.write_u8(2);
                centre_colour.content_hash_at(origin, state);
                edge_colour.content_hash_at(origin, state);
            }
            OverlayFill::ConicalGradient {
                start_colour,
                end_colour,
                offset_angle,
            } => {
                state.write_u8(3);
                start_colour.content_hash_at(origin, state);
                end_colour.content_hash_at(origin, state);
                hash_f32(state, *offset_angle);
            }
            OverlayFill::LinearGradientMulti { stops, angle } => {
                state.write_u8(4);
                stops.content_hash_at(origin, state);
                hash_f32(state, *angle);
            }
            OverlayFill::RadialGradientMulti { stops } => {
                state.write_u8(5);
                stops.content_hash_at(origin, state);
            }
            OverlayFill::ConicalGradientMulti {
                stops,
                offset_angle,
            } => {
                state.write_u8(6);
                stops.content_hash_at(origin, state);
                hash_f32(state, *offset_angle);
            }
            OverlayFill::Texture {
                id,
                transform,
                nine_slice,
                tint,
            } => {
                state.write_u8(7);
                id.content_hash_at(origin, state);
                transform.content_hash_at(origin, state);
                nine_slice.content_hash_at(origin, state);
                tint.content_hash_at(origin, state);
            }
        }
    }
}

impl OverlayContentHash for ShadowLayer {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            colour,
            blur,
            offset,
            spread,
            falloff,
        } = self;
        colour.content_hash_at(origin, state);
        hash_f32(state, *blur);
        hash_floats(state, offset);
        hash_f32(state, *spread);
        hash_f32(state, *falloff);
    }
}

impl OverlayContentHash for BackdropEffects {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        let Self {
            blur,
            saturation,
            brightness,
            hue_shift,
        } = self;
        hash_floats(state, &[*blur, *saturation, *brightness, *hue_shift]);
    }
}

impl OverlayContentHash for OverlayStyle {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            fill,
            shadows,
            inner_shadows,
            backdrop,
            tint,
            opacity,
        } = self;
        fill.content_hash_at(origin, state);
        shadows.content_hash_at(origin, state);
        inner_shadows.content_hash_at(origin, state);
        backdrop.content_hash_at(origin, state);
        hash_floats(state, tint);
        hash_f32(state, *opacity);
    }
}

impl OverlayContentHash for OverlayEasing {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        match self {
            OverlayEasing::Linear => state.write_u8(0),
            OverlayEasing::EaseIn => state.write_u8(1),
            OverlayEasing::EaseOut => state.write_u8(2),
            OverlayEasing::EaseInOut => state.write_u8(3),
            OverlayEasing::Pulse => state.write_u8(4),
            OverlayEasing::Back => state.write_u8(5),
            OverlayEasing::Bounce => state.write_u8(6),
            OverlayEasing::Elastic => state.write_u8(7),
            OverlayEasing::CubicBezier { x1, y1, x2, y2 } => {
                state.write_u8(8);
                hash_floats(state, &[*x1, *y1, *x2, *y2]);
            }
        }
    }
}

impl OverlayContentHash for RepeatMode {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            RepeatMode::Once => 0,
            RepeatMode::Loop => 1,
            RepeatMode::PingPong => 2,
        });
    }
}

// One impl per channel type the animation block uses; the track's `from` and
// `to` are plain values, not screen positions, so they are not offset.
macro_rules! anim_track_hash {
    ($t:ty, |$state:ident, $v:ident| $hash:expr) => {
        impl OverlayContentHash for AnimTrack<$t> {
            fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
                let Self {
                    start_time,
                    duration,
                    from,
                    to,
                    easing,
                    repeat,
                } = self;
                hash_f64(state, *start_time);
                hash_f32(state, *duration);
                for $v in [from, to] {
                    let $state = &mut *state;
                    $hash;
                }
                easing.content_hash_at(origin, state);
                repeat.content_hash_at(origin, state);
            }
        }
    };
}

anim_track_hash!(f32, |s, v| hash_f32(s, *v));
anim_track_hash!([f32; 2], |s, v| hash_floats(s, v));
anim_track_hash!([f32; 4], |s, v| hash_floats(s, v));

impl OverlayContentHash for OverlayAnimations {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            epoch,
            opacity,
            translate,
            rotation,
            scale,
            tint,
        } = self;
        hash_f64(state, *epoch);
        opacity.content_hash_at(origin, state);
        translate.content_hash_at(origin, state);
        rotation.content_hash_at(origin, state);
        scale.content_hash_at(origin, state);
        tint.content_hash_at(origin, state);
    }
}

impl OverlayContentHash for PathSegment {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        match self {
            PathSegment::Line { to } => {
                state.write_u8(0);
                hash_floats(state, to);
            }
            PathSegment::Quad { ctrl, to } => {
                state.write_u8(1);
                hash_floats(state, ctrl);
                hash_floats(state, to);
            }
            PathSegment::Cubic { ctrl1, ctrl2, to } => {
                state.write_u8(2);
                hash_floats(state, ctrl1);
                hash_floats(state, ctrl2);
                hash_floats(state, to);
            }
        }
    }
}

impl OverlayContentHash for SubPath {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        // Path coordinates are local to the shape, so `origin` does not apply.
        let Self {
            start,
            segments,
            closed,
        } = self;
        hash_floats(state, start);
        segments.content_hash_at(origin, state);
        hash_bool(state, *closed);
    }
}

impl OverlayContentHash for FillRule {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            FillRule::NonZero => 0,
            FillRule::EvenOdd => 1,
        });
    }
}

impl OverlayContentHash for TriangleDirection {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            TriangleDirection::Up => 0,
            TriangleDirection::Down => 1,
            TriangleDirection::Left => 2,
            TriangleDirection::Right => 3,
        });
    }
}

impl OverlayContentHash for LineCap {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            LineCap::Round => 0,
            LineCap::Square => 1,
        });
    }
}

impl OverlayContentHash for OverlayShape {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        match self {
            OverlayShape::Rect { corner_radius } => {
                state.write_u8(0);
                hash_f32(state, *corner_radius);
            }
            OverlayShape::RoundedRect { radii } => {
                state.write_u8(1);
                hash_floats(state, radii);
            }
            OverlayShape::Circle => state.write_u8(2),
            OverlayShape::Ellipse => state.write_u8(3),
            OverlayShape::Capsule => state.write_u8(4),
            OverlayShape::Ring { inner_radius_frac } => {
                state.write_u8(5);
                hash_f32(state, *inner_radius_frac);
            }
            OverlayShape::Arc {
                inner_radius_frac,
                start_angle,
                end_angle,
            } => {
                state.write_u8(6);
                hash_floats(state, &[*inner_radius_frac, *start_angle, *end_angle]);
            }
            OverlayShape::Triangle { direction } => {
                state.write_u8(7);
                direction.content_hash_at(origin, state);
            }
            OverlayShape::Line { thickness, cap } => {
                state.write_u8(8);
                hash_f32(state, *thickness);
                cap.content_hash_at(origin, state);
            }
            OverlayShape::Star {
                points,
                inner_radius_frac,
            } => {
                state.write_u8(9);
                put_u32(state, *points);
                hash_f32(state, *inner_radius_frac);
            }
            OverlayShape::RegularPolygon { sides } => {
                state.write_u8(10);
                put_u32(state, *sides);
            }
            OverlayShape::Cross { arm_width_frac } => {
                state.write_u8(11);
                hash_f32(state, *arm_width_frac);
            }
            OverlayShape::Vector {
                subpaths,
                fill_rule,
            } => {
                state.write_u8(12);
                subpaths.content_hash_at(origin, state);
                fill_rule.content_hash_at(origin, state);
            }
        }
    }
}

impl OverlayContentHash for OverlayShapeItem {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            shape,
            size,
            anchoring,
            transform,
            style,
            clip,
            z_order,
            provides_mask,
            animations,
        } = self;
        shape.content_hash_at(origin, state);
        hash_floats(state, size);
        anchoring.content_hash_at(origin, state);
        transform.content_hash_at(origin, state);
        style.content_hash_at(origin, state);
        clip.content_hash_at(origin, state);
        put_i32(state, *z_order);
        hash_opt_u32(state, *provides_mask);
        animations.content_hash_at(origin, state);
    }
}

impl OverlayContentHash for StrokePattern {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        match self {
            StrokePattern::Solid => state.write_u8(0),
            StrokePattern::Dashed {
                dash_length,
                gap_length,
                offset,
            } => {
                state.write_u8(1);
                hash_floats(state, &[*dash_length, *gap_length, *offset]);
            }
            StrokePattern::Dotted { spacing, offset } => {
                state.write_u8(2);
                hash_floats(state, &[*spacing, *offset]);
            }
        }
    }
}

impl OverlayContentHash for LineJoin {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            LineJoin::Mitre => 0,
            LineJoin::Bevel => 1,
        });
    }
}

impl OverlayContentHash for PolylineCap {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        state.write_u8(match self {
            PolylineCap::Butt => 0,
            PolylineCap::Square => 1,
            PolylineCap::Round => 2,
        });
    }
}

impl OverlayContentHash for OverlayStroke {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            width,
            colour,
            pattern,
            join,
            mitre_limit,
            cap,
        } = self;
        hash_f32(state, *width);
        colour.content_hash_at(origin, state);
        pattern.content_hash_at(origin, state);
        join.content_hash_at(origin, state);
        hash_f32(state, *mitre_limit);
        cap.content_hash_at(origin, state);
    }
}

impl OverlayContentHash for OverlayPolylineItem {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            points,
            anchoring,
            transform,
            style,
            animations,
            clip,
            stroke,
            closed,
            uvs,
            z_order,
        } = self;
        hash_len(state, points.len());
        for p in points {
            hash_point(state, *p, origin);
        }
        anchoring.content_hash_at(origin, state);
        // The points are the placement; `translate` only nudges them.
        transform.content_hash_at([0.0, 0.0], state);
        style.content_hash_at(origin, state);
        animations.content_hash_at(origin, state);
        clip.content_hash_at(origin, state);
        stroke.content_hash_at(origin, state);
        hash_bool(state, *closed);
        match uvs {
            None => state.write_u8(0),
            Some(uvs) => {
                state.write_u8(1);
                hash_len(state, uvs.len());
                for uv in uvs {
                    hash_floats(state, uv);
                }
            }
        }
        put_i32(state, *z_order);
    }
}

impl OverlayContentHash for LabelItem {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            anchoring,
            transform,
            style,
            animations,
            clip,
            text,
            text_style,
            max_width,
            z_order,
        } = self;
        anchoring.content_hash_at(origin, state);
        transform.content_hash_at(origin, state);
        style.content_hash_at(origin, state);
        animations.content_hash_at(origin, state);
        clip.content_hash_at(origin, state);
        hash_len(state, text.len());
        state.write(text.as_bytes());
        text_style.content_hash_at(origin, state);
        match max_width {
            None => state.write_u8(0),
            Some(w) => {
                state.write_u8(1);
                hash_f32(state, *w);
            }
        }
        put_i32(state, *z_order);
    }
}

impl OverlayContentHash for PositionedGlyph {
    fn content_hash_at<H: Hasher>(&self, _origin: [f32; 2], state: &mut H) {
        // Glyph positions are relative to the run, so `origin` does not apply.
        let Self { glyph_id, x, y } = self;
        put_u16(state, *glyph_id);
        hash_floats(state, &[*x, *y]);
    }
}

impl OverlayContentHash for GlyphRunItem {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            text_style,
            anchoring,
            transform,
            style,
            animations,
            clip,
            glyphs,
            glyph_tints,
            z_order,
        } = self;
        text_style.content_hash_at(origin, state);
        anchoring.content_hash_at(origin, state);
        transform.content_hash_at(origin, state);
        style.content_hash_at(origin, state);
        animations.content_hash_at(origin, state);
        clip.content_hash_at(origin, state);
        glyphs.content_hash_at(origin, state);
        hash_len(state, glyph_tints.len());
        for t in glyph_tints {
            hash_floats(state, t);
        }
        put_i32(state, *z_order);
    }
}

impl OverlayContentHash for RetainedOverlay {
    fn content_hash_at<H: Hasher>(&self, origin: [f32; 2], state: &mut H) {
        let Self {
            id,
            transform,
            opacity,
            z_order,
            clip,
            tint,
            anchoring,
            animations,
        } = self;
        id.content_hash_at(origin, state);
        transform.content_hash_at(origin, state);
        hash_f32(state, *opacity);
        put_i32(state, *z_order);
        clip.content_hash_at(origin, state);
        hash_floats(state, tint);
        anchoring.content_hash_at(origin, state);
        animations.content_hash_at(origin, state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    // `DefaultHasher::new()` uses fixed keys, so it is stable within a build.
    fn hash<T: OverlayContentHash>(v: &T) -> u64 {
        let mut h = DefaultHasher::new();
        v.content_hash(&mut h);
        h.finish()
    }

    fn hash_at<T: OverlayContentHash>(v: &T, origin: [f32; 2]) -> u64 {
        let mut h = DefaultHasher::new();
        v.content_hash_at(origin, &mut h);
        h.finish()
    }

    fn shape() -> OverlayShapeItem {
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 4.0 },
            [10.0, 20.0],
            [30.0, 40.0],
        )
        .with_fill(OverlayFill::Solid(Colour::linear(1.0, 0.5, 0.25, 1.0)))
        .with_shadow(ShadowLayer::default())
    }

    fn polyline() -> OverlayPolylineItem {
        let mut p = OverlayPolylineItem::new(vec![[0.0, 0.0], [10.0, 5.0], [20.0, 0.0]]);
        p.clip = OverlayClip::rect([0.0, 0.0, 50.0, 50.0]);
        p
    }

    fn label() -> LabelItem {
        LabelItem::new("hello").with_screen_anchor([5.0, 6.0])
    }

    fn glyph_run() -> GlyphRunItem {
        GlyphRunItem::new(vec![PositionedGlyph::new(3, 0.0, 10.0)])
    }

    #[test]
    fn equal_items_hash_equal() {
        assert_eq!(hash(&shape()), hash(&shape().clone()));
        assert_eq!(hash(&polyline()), hash(&polyline().clone()));
        assert_eq!(hash(&label()), hash(&label().clone()));
        assert_eq!(hash(&glyph_run()), hash(&glyph_run().clone()));
    }

    #[test]
    fn negative_zero_hashes_like_zero() {
        let a = label().with_position([0.0, 0.0]);
        let b = label().with_position([-0.0, -0.0]);
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));
    }

    #[test]
    fn changing_a_field_changes_the_hash() {
        let base = hash(&shape());
        let mut s = shape();
        s.size[0] += 1.0;
        assert_ne!(hash(&s), base);
        let mut s = shape();
        s.style.opacity = 0.5;
        assert_ne!(hash(&s), base);
        let mut s = shape();
        s.z_order = 3;
        assert_ne!(hash(&s), base);
        let mut s = shape();
        s.animations = Some(Box::new(
            OverlayAnimations::default().with_opacity(AnimTrack::new(0.0, 1.0, 0.0, 1.0)),
        ));
        assert_ne!(hash(&s), base);

        let base = hash(&polyline());
        let mut p = polyline();
        p.points[1][1] = 6.0;
        assert_ne!(hash(&p), base);
        let mut p = polyline();
        p.closed = true;
        assert_ne!(hash(&p), base);

        let base = hash(&label());
        let mut l = label();
        l.text.push('!');
        assert_ne!(hash(&l), base);
        let mut l = label();
        l.anchoring.align.y = AnchorY::Baseline;
        assert_ne!(hash(&l), base);

        let base = hash(&glyph_run());
        let mut g = glyph_run();
        g.glyphs[0].glyph_id = 4;
        assert_ne!(hash(&g), base);
    }

    /// Moving every screen position and the origin by the same amount keeps
    /// the hash, which is what lets a retained group follow moved content.
    #[test]
    fn a_pure_move_keeps_the_hash_at_a_moved_origin() {
        let d = [12.0, -7.0];
        let moved = |p: [f32; 2]| [p[0] + d[0], p[1] + d[1]];

        let s = shape();
        let mut s2 = s.clone();
        s2.transform.translate = moved(s.transform.translate);
        assert_ne!(hash(&s), hash(&s2));
        assert_eq!(hash_at(&s, [0.0, 0.0]), hash_at(&s2, d));

        let p = polyline();
        let mut p2 = p.clone();
        p2.points = p.points.iter().map(|&q| moved(q)).collect();
        let r = p.clip.rect.unwrap();
        let (a, b) = (moved([r[0], r[1]]), moved([r[2], r[3]]));
        p2.clip.rect = Some([a[0], a[1], b[0], b[1]]);
        assert_eq!(hash_at(&p, [0.0, 0.0]), hash_at(&p2, d));
    }

    /// Records the bytes it is fed, so a test can pin the exact layout.
    #[derive(Default)]
    struct Bytes(Vec<u8>);

    impl Hasher for Bytes {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, bytes: &[u8]) {
            self.0.extend_from_slice(bytes);
        }
    }

    /// The byte stream does not depend on the target's pointer width or byte
    /// order: a tag byte, then fixed-width little-endian values.
    #[test]
    fn the_bytes_fed_to_the_hasher_are_fixed() {
        let mut h = Bytes::default();
        OverlayShape::Star {
            points: 5,
            inner_radius_frac: -0.0,
        }
        .content_hash(&mut h);
        assert_eq!(h.0, [9, 5, 0, 0, 0, 0, 0, 0, 0]);

        let mut h = Bytes::default();
        vec![PathSegment::Line { to: [1.0, 0.0] }].content_hash(&mut h);
        let mut want = vec![1, 0, 0, 0, 0, 0, 0, 0, 0];
        want.extend_from_slice(&1.0f32.to_le_bytes());
        want.extend_from_slice(&0.0f32.to_le_bytes());
        assert_eq!(h.0, want);
    }

    #[test]
    fn collections_are_order_sensitive() {
        let a = OverlayPolylineItem::new(vec![[0.0, 0.0], [1.0, 1.0]]);
        let b = OverlayPolylineItem::new(vec![[1.0, 1.0], [0.0, 0.0]]);
        assert_ne!(hash(&a), hash(&b));
    }
}
