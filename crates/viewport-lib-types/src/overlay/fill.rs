//! Fill styles for overlay items: solid colour, gradients, textures, and
//! registered fill functions.

use crate::colour::Colour;
use crate::material::TextureFilter;
use crate::overlay::{NineSlice, OverlayTextureId, TextureTransform};

/// What an overlay item is filled with.
///
/// `Solid` is a single flat colour. The gradient variants interpolate a list of
/// [`GradientStop`]s across the item's bounding box, each along its own axis:
/// a direction for `LinearGradient`, the distance from the centre for
/// `RadialGradient`, and the angle about a centre for `ConicalGradient` and
/// `SweepGradient`. A stop list may hold any number of stops. `Texture` samples
/// an uploaded image instead.
///
/// Every overlay family evaluates a fill the same way, per pixel, so a fill
/// means the same thing on a shape, a vector path, a polyline and a run of
/// text. The two-stop constructors ([`OverlayFill::linear_gradient`] and the
/// others) cover the common case without building a stop list.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OverlayFill {
    /// Uniform solid colour in linear RGBA float format.
    Solid(crate::colour::Colour),
    /// Linear gradient along a direction across the bounding box.
    ///
    /// `angle = 0.0` runs left-to-right (the first stop on the left, the last
    /// on the right). Positive angles rotate the direction counter-clockwise
    /// in maths coordinates; because screen Y points downward, `angle = PI/2`
    /// runs top-to-bottom.
    LinearGradient {
        /// Stops along the axis, position `0` at the start and `1` at the end.
        /// Need not be sorted; the renderer sorts them.
        stops: Vec<GradientStop>,
        /// Gradient direction in radians. `0.0` = left-to-right.
        angle: f32,
    },
    /// Radial gradient from the box centre outward.
    ///
    /// Position `0` sits at the centre and position `1` at the larger
    /// half-extent of the box, `length(p) / max(half_width, half_height)`.
    RadialGradient {
        /// Stops from the centre (`0`) to the edge (`1`).
        stops: Vec<GradientStop>,
    },
    /// Conical gradient: one full turn about the box centre, like a colour
    /// wheel.
    ///
    /// The last stop meets the first at the seam, which `offset_angle`
    /// rotates; `0.0` places it to the right of the centre. The angle
    /// increases towards +Y, which is clockwise on screen.
    ConicalGradient {
        /// Stops along the turn, `0` at the seam and `1` one turn later.
        stops: Vec<GradientStop>,
        /// Rotation of the seam in radians. `0.0` places it to the right.
        offset_angle: f32,
    },
    /// Sweep gradient: the stops spread over an arc of angles about a centre
    /// anywhere in the box.
    ///
    /// Position `0` sits at `start_angle` and position `1` at `end_angle`,
    /// both in radians from +X and increasing towards +Y (clockwise on
    /// screen, the same convention as `LinearGradient::angle`). Outside the arc
    /// the end stops extend: the last stop past `end_angle` and the first just
    /// before `start_angle`, where the seam sits. An arc of one full turn is a
    /// `ConicalGradient` about `centre`; an arc that is not positive is read as
    /// one full turn.
    SweepGradient {
        /// Stops along the arc, `0` at `start_angle` and `1` at `end_angle`.
        stops: Vec<GradientStop>,
        /// The centre of the sweep as a fraction of the box: `[0, 0]` is the
        /// top-left corner, `[0.5, 0.5]` the middle, `[1, 1]` the bottom-right.
        centre: [f32; 2],
        /// The angle of position `0`, in radians.
        start_angle: f32,
        /// The angle of position `1`, in radians.
        end_angle: f32,
    },
    /// An uploaded image, sampled across the item's coverage.
    ///
    /// The image is uploaded with
    /// `DeviceResources::upload_overlay_texture` and named by `id`.
    Texture {
        /// The uploaded image to sample.
        id: OverlayTextureId,
        /// Affine transform applied to the sample before lookup: pan, scale,
        /// rotate, tile, and flip independently of the item it fills.
        transform: TextureTransform,
        /// Nine-patch parameters, which keep the corners of a resizable panel
        /// at their authored size, across whatever box the fill spans.
        nine_slice: Option<NineSlice>,
        /// Multiplied into every sample, white by default. This is where a
        /// textured item's colour comes from: a mid-grey tint darkens the
        /// image, and an alpha below one fades it.
        tint: crate::colour::Colour,
        /// How the image is sampled: `Linear` (the default) blends
        /// neighbouring texels, `Nearest` takes the texel under the pixel, so
        /// a pixel-art image or nine-slice frame drawn at an integer scale
        /// keeps hard texel edges.
        filter: TextureFilter,
    },
    /// A fill function the consumer registered with the renderer, evaluated
    /// per pixel with `params`.
    ///
    /// The function is WGSL registered through the renderer's plugin API,
    /// which returns the `kind`. It receives the pixel's offset from the
    /// centre of the box the fill spans, the box half-extents and `params`,
    /// and returns linear RGBA, which the item's opacity and tint then scale.
    /// A `kind` the renderer does not know draws transparent.
    Custom {
        /// The registered fill function.
        kind: CustomFillId,
        /// The function's parameters, as it defines them.
        params: [f32; CUSTOM_FILL_PARAMS],
    },
}

/// How many `f32` parameters an [`OverlayFill::Custom`] carries.
pub const CUSTOM_FILL_PARAMS: usize = 16;

/// A fill function registered with the renderer, as an [`OverlayFill::Custom`]
/// names it. The renderer hands these out when the function is registered;
/// they mean nothing to another renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CustomFillId(pub u32);

/// Which kind of fill an [`OverlayFill`] is, for asking a family what it can
/// draw without naming a specific gradient or image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverlayFillKind {
    /// A flat colour.
    Solid,
    /// Any of the gradient variants.
    Gradient,
    /// An uploaded image.
    Texture,
    /// A registered fill function.
    Custom,
}

impl Default for OverlayFill {
    fn default() -> Self {
        OverlayFill::Solid(Colour::linear(0.0, 0.0, 0.0, 0.55))
    }
}

/// A single colour stop in a gradient.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct GradientStop {
    /// Position along the gradient axis, in `[0, 1]`. Stops outside the
    /// range are clamped at evaluation time.
    pub position: f32,
    /// Linear RGBA colour at this stop.
    pub colour: crate::colour::Colour,
}

impl GradientStop {
    /// Construct a stop at the given position and colour.
    pub fn new(position: f32, colour: impl Into<crate::colour::Colour>) -> Self {
        Self {
            position,
            colour: colour.into(),
        }
    }

    /// Two stops: `start` at position `0` and `end` at position `1`.
    pub fn pair(
        start: impl Into<crate::colour::Colour>,
        end: impl Into<crate::colour::Colour>,
    ) -> Vec<GradientStop> {
        vec![GradientStop::new(0.0, start), GradientStop::new(1.0, end)]
    }
}

/// Where an outline band sits relative to the item's edge.
///
/// The third argument to `with_outline`, which lowers the band to shadow
/// layers: `Inset` is one inner layer, `Outer` one outer layer, and `Centre`
/// one of each at half the width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OutlineMode {
    /// Inside the edge (default): the band eats into the fill.
    #[default]
    Inset,
    /// Outside the edge: the fill is untouched and the band extends outward.
    Outer,
    /// Centred on the edge, half inside and half outside.
    Centre,
}

impl OverlayFill {
    /// A two-stop linear gradient from `start` to `end` along `angle`.
    pub fn linear_gradient(
        start: impl Into<crate::colour::Colour>,
        end: impl Into<crate::colour::Colour>,
        angle: f32,
    ) -> Self {
        OverlayFill::LinearGradient {
            stops: GradientStop::pair(start, end),
            angle,
        }
    }

    /// A two-stop radial gradient from `centre` at the middle to `edge`.
    pub fn radial_gradient(
        centre: impl Into<crate::colour::Colour>,
        edge: impl Into<crate::colour::Colour>,
    ) -> Self {
        OverlayFill::RadialGradient {
            stops: GradientStop::pair(centre, edge),
        }
    }

    /// A two-stop conical gradient, `start` at the seam turning to `end`.
    pub fn conical_gradient(
        start: impl Into<crate::colour::Colour>,
        end: impl Into<crate::colour::Colour>,
        offset_angle: f32,
    ) -> Self {
        OverlayFill::ConicalGradient {
            stops: GradientStop::pair(start, end),
            offset_angle,
        }
    }

    /// A two-stop sweep about the box centre, `start` at `start_angle` and
    /// `end` at `end_angle`. Set `centre` on the variant to move the centre.
    pub fn sweep_gradient(
        start: impl Into<crate::colour::Colour>,
        end: impl Into<crate::colour::Colour>,
        start_angle: f32,
        end_angle: f32,
    ) -> Self {
        OverlayFill::SweepGradient {
            stops: GradientStop::pair(start, end),
            centre: [0.5, 0.5],
            start_angle,
            end_angle,
        }
    }

    /// A custom fill: the registered function `kind` with `params`.
    pub fn custom(kind: CustomFillId, params: [f32; CUSTOM_FILL_PARAMS]) -> Self {
        OverlayFill::Custom { kind, params }
    }

    /// A texture fill with the default transform, no nine-patch, and no tint.
    pub fn texture(id: OverlayTextureId) -> Self {
        OverlayFill::Texture {
            id,
            transform: TextureTransform::default(),
            nine_slice: None,
            tint: crate::colour::Colour::linear(1.0, 1.0, 1.0, 1.0),
            filter: TextureFilter::Linear,
        }
    }

    /// A fully transparent solid, which is what "no fill" means once it
    /// reaches a shader. The default for an item that has not asked for one.
    pub fn none() -> Self {
        OverlayFill::Solid(crate::colour::Colour::linear(0.0, 0.0, 0.0, 0.0))
    }

    /// Whether this fill draws anything. A fully transparent solid, which is
    /// what [`OverlayFill::none`] is and what an item defaults to, draws
    /// nothing and reports `false`.
    pub fn is_set(&self) -> bool {
        *self != OverlayFill::none()
    }

    /// Which kind of fill this is.
    pub fn kind(&self) -> OverlayFillKind {
        match self {
            OverlayFill::Solid(_) => OverlayFillKind::Solid,
            OverlayFill::Texture { .. } => OverlayFillKind::Texture,
            OverlayFill::Custom { .. } => OverlayFillKind::Custom,
            _ => OverlayFillKind::Gradient,
        }
    }

    /// The stops of a gradient fill, or `None` for a solid or texture fill.
    pub fn stops(&self) -> Option<&[GradientStop]> {
        match self {
            OverlayFill::LinearGradient { stops, .. }
            | OverlayFill::RadialGradient { stops }
            | OverlayFill::ConicalGradient { stops, .. }
            | OverlayFill::SweepGradient { stops, .. } => Some(stops),
            OverlayFill::Solid(_) | OverlayFill::Texture { .. } | OverlayFill::Custom { .. } => {
                None
            }
        }
    }

    /// The image this fill samples, or `None` for a colour or gradient fill.
    pub fn texture_id(&self) -> Option<OverlayTextureId> {
        match self {
            OverlayFill::Texture { id, .. } => Some(*id),
            _ => None,
        }
    }

    /// Set the sampling transform on a texture fill. Does nothing to any other
    /// fill, which has nothing to sample.
    pub fn with_texture_transform(mut self, tt: TextureTransform) -> Self {
        if let OverlayFill::Texture { transform, .. } = &mut self {
            *transform = tt;
        }
        self
    }

    /// Set the nine-patch parameters on a texture fill.
    pub fn with_nine_slice(mut self, ns: NineSlice) -> Self {
        if let OverlayFill::Texture { nine_slice, .. } = &mut self {
            *nine_slice = Some(ns);
        }
        self
    }

    /// Set how a texture fill samples its image. Does nothing to any other
    /// fill.
    pub fn with_texture_filter(mut self, f: TextureFilter) -> Self {
        if let OverlayFill::Texture { filter, .. } = &mut self {
            *filter = f;
        }
        self
    }

    /// Set the tint multiplied into every sample of a texture fill.
    pub fn with_tint(mut self, colour: impl Into<crate::colour::Colour>) -> Self {
        if let OverlayFill::Texture { tint, .. } = &mut self {
            *tint = colour.into();
        }
        self
    }

    /// This fill with `f` applied to every colour it holds: the solid colour,
    /// every gradient stop, and a texture fill's tint. Positions, angles and
    /// the texture are kept.
    ///
    /// For deriving one style from another, such as a darker hover state from
    /// a gradient, without matching on every variant.
    pub fn map_colours(&self, f: impl Fn(Colour) -> Colour) -> OverlayFill {
        let stops = |stops: &[GradientStop]| -> Vec<GradientStop> {
            stops
                .iter()
                .map(|s| GradientStop {
                    position: s.position,
                    colour: f(s.colour),
                })
                .collect()
        };
        // No wildcard arm: a new variant must decide here which of its fields
        // are colours.
        match self {
            OverlayFill::Solid(c) => OverlayFill::Solid(f(*c)),
            OverlayFill::LinearGradient { stops: s, angle } => OverlayFill::LinearGradient {
                stops: stops(s),
                angle: *angle,
            },
            OverlayFill::RadialGradient { stops: s } => {
                OverlayFill::RadialGradient { stops: stops(s) }
            }
            OverlayFill::ConicalGradient {
                stops: s,
                offset_angle,
            } => OverlayFill::ConicalGradient {
                stops: stops(s),
                offset_angle: *offset_angle,
            },
            OverlayFill::SweepGradient {
                stops: s,
                centre,
                start_angle,
                end_angle,
            } => OverlayFill::SweepGradient {
                stops: stops(s),
                centre: *centre,
                start_angle: *start_angle,
                end_angle: *end_angle,
            },
            OverlayFill::Texture {
                id,
                transform,
                nine_slice,
                tint,
                filter,
            } => OverlayFill::Texture {
                id: *id,
                transform: *transform,
                nine_slice: *nine_slice,
                tint: f(*tint),
                filter: *filter,
            },
            // The function's colours are its own.
            OverlayFill::Custom { kind, params } => OverlayFill::Custom {
                kind: *kind,
                params: *params,
            },
        }
    }

    /// Sample the fill at `p`, given in logical pixels relative to the centre
    /// of a box of half-extents `half_size`. Returns linear RGBA.
    ///
    /// This is the CPU statement of the maths the overlay shaders run per
    /// pixel, for a consumer that needs the colour at a point (a hit test, a
    /// colour picker over a gradient). A texture fill returns its tint, since
    /// the image itself lives on the GPU, and a custom fill returns white, since
    /// its function runs only there.
    pub fn sample_in_box(&self, p: [f32; 2], half_size: [f32; 2]) -> [f32; 4] {
        match self {
            OverlayFill::Solid(c) => c.to_linear_rgba(),
            OverlayFill::Texture { tint, .. } => tint.to_linear_rgba(),
            OverlayFill::Custom { .. } => [1.0; 4],
            OverlayFill::LinearGradient { stops, angle } => {
                sample_sorted_stops(stops, linear_t(p, half_size, *angle))
            }
            OverlayFill::RadialGradient { stops } => {
                sample_sorted_stops(stops, radial_t(p, half_size))
            }
            OverlayFill::ConicalGradient {
                stops,
                offset_angle,
            } => sample_sorted_stops(
                stops,
                sweep_t(p, [0.0, 0.0], *offset_angle, std::f32::consts::TAU),
            ),
            OverlayFill::SweepGradient {
                stops,
                centre,
                start_angle,
                end_angle,
            } => {
                let c = [
                    (centre[0] * 2.0 - 1.0) * half_size[0],
                    (centre[1] * 2.0 - 1.0) * half_size[1],
                ];
                sample_sorted_stops(stops, sweep_t(p, c, *start_angle, end_angle - start_angle))
            }
        }
    }
}

/// The linear-gradient parameter at `p` in a box of half-extents `hs`.
fn linear_t(p: [f32; 2], hs: [f32; 2], angle: f32) -> f32 {
    let dir = [angle.cos(), angle.sin()];
    let max_proj = (hs[0] * dir[0]).abs() + (hs[1] * dir[1]).abs();
    ((p[0] * dir[0] + p[1] * dir[1]) / max_proj.max(0.001) * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// The radial-gradient parameter at `p` in a box of half-extents `hs`.
fn radial_t(p: [f32; 2], hs: [f32; 2]) -> f32 {
    let max_half = hs[0].max(hs[1]);
    ((p[0] * p[0] + p[1] * p[1]).sqrt() / max_half.max(0.001)).clamp(0.0, 1.0)
}

/// The sweep parameter at `p` about `centre`: the angle past `start` over
/// `span`, clamped. A span that is not positive is one full turn.
fn sweep_t(p: [f32; 2], centre: [f32; 2], start: f32, span: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let span = if span > 0.0 { span } else { tau };
    let a = (p[1] - centre[1]).atan2(p[0] - centre[0]) - start;
    let turned = (a / tau + 1.0).fract() * tau;
    (turned / span).clamp(0.0, 1.0)
}

/// Interpolate a stop list at `t`, sorting it first and clamping outside the
/// end stops. An empty list is transparent.
fn sample_sorted_stops(stops: &[GradientStop], t: f32) -> [f32; 4] {
    fn lerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]
    }
    let mut sorted: Vec<&GradientStop> = stops.iter().collect();
    sorted.sort_by(|a, b| {
        a.position
            .partial_cmp(&b.position)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(first) = sorted.first() else {
        return [0.0; 4];
    };
    if t <= first.position {
        return first.colour.to_linear_rgba();
    }
    for pair in sorted.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if t <= b.position {
            let span = (b.position - a.position).max(1e-6);
            return lerp(
                a.colour.to_linear_rgba(),
                b.colour.to_linear_rgba(),
                ((t - a.position) / span).clamp(0.0, 1.0),
            );
        }
    }
    sorted[sorted.len() - 1].colour.to_linear_rgba()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_colours_reaches_every_colour() {
        let a = Colour::linear(0.2, 0.4, 0.6, 1.0);
        let b = Colour::linear(0.8, 0.6, 0.4, 0.5);
        let double = |c: Colour| {
            let [r, g, b, a] = c.to_linear_rgba();
            Colour::linear(r * 2.0, g * 2.0, b * 2.0, a)
        };
        let stops = vec![
            GradientStop::new(0.0, a),
            GradientStop::new(0.3, b),
            GradientStop::new(0.7, a),
        ];
        let fills = [
            OverlayFill::Solid(a),
            OverlayFill::linear_gradient(a, b, 0.3),
            OverlayFill::radial_gradient(a, b),
            OverlayFill::conical_gradient(a, b, 0.3),
            OverlayFill::sweep_gradient(a, b, 0.2, 2.0),
            OverlayFill::LinearGradient {
                stops: stops.clone(),
                angle: 0.3,
            },
            OverlayFill::RadialGradient {
                stops: stops.clone(),
            },
            OverlayFill::SweepGradient {
                stops: stops.clone(),
                centre: [0.2, 0.7],
                start_angle: -1.0,
                end_angle: 1.5,
            },
            OverlayFill::texture(OverlayTextureId(3)).with_tint(a),
        ];
        let p = [3.0, -2.0];
        let half = [10.0, 10.0];
        for fill in fills {
            let mapped = fill.map_colours(double);
            assert_eq!(mapped.kind(), fill.kind());
            assert_eq!(mapped.texture_id(), fill.texture_id());
            if let OverlayFill::Texture { tint, .. } = &mapped {
                assert_eq!(*tint, double(a));
                continue;
            }
            // Doubling every colour doubles the sampled RGB wherever the
            // fill is sampled, so no colour was missed and nothing moved.
            let before = fill.sample_in_box(p, half);
            let after = mapped.sample_in_box(p, half);
            for i in 0..3 {
                assert!((after[i] - before[i] * 2.0).abs() < 1e-5, "{fill:?}");
            }
            assert!((after[3] - before[3]).abs() < 1e-5, "{fill:?}");
        }
    }

    #[test]
    fn a_full_turn_sweep_is_a_conical_gradient() {
        let a = Colour::linear(1.0, 0.0, 0.0, 1.0);
        let b = Colour::linear(0.0, 0.0, 1.0, 1.0);
        let conical = OverlayFill::conical_gradient(a, b, 0.4);
        let sweep = OverlayFill::sweep_gradient(a, b, 0.4, 0.4 + std::f32::consts::TAU);
        for p in [[3.0, -2.0], [-5.0, 1.0], [0.5, 4.0], [-2.0, -6.0]] {
            let c = conical.sample_in_box(p, [10.0, 10.0]);
            let s = sweep.sample_in_box(p, [10.0, 10.0]);
            for i in 0..4 {
                assert!((c[i] - s[i]).abs() < 1e-4, "{p:?}");
            }
        }
    }

    #[test]
    fn a_partial_sweep_holds_its_end_stops_outside_the_arc() {
        let a = Colour::linear(1.0, 0.0, 0.0, 1.0);
        let b = Colour::linear(0.0, 0.0, 1.0, 1.0);
        // A quarter turn from +X towards +Y.
        let sweep = OverlayFill::sweep_gradient(a, b, 0.0, std::f32::consts::FRAC_PI_2);
        let half = [10.0, 10.0];
        // Halfway round the arc.
        let mid = sweep.sample_in_box([1.0, 1.0], half);
        assert!((mid[0] - 0.5).abs() < 1e-4 && (mid[2] - 0.5).abs() < 1e-4);
        // Past the end of the arc: the last stop.
        assert_eq!(sweep.sample_in_box([-1.0, 1.0], half), b.to_linear_rgba());
    }

    #[test]
    fn stops_have_no_cap() {
        let stops: Vec<GradientStop> = (0..12)
            .map(|i| {
                GradientStop::new(
                    i as f32 / 11.0,
                    Colour::linear(i as f32 / 11.0, 0.0, 0.0, 1.0),
                )
            })
            .collect();
        let fill = OverlayFill::LinearGradient { stops, angle: 0.0 };
        // The last stops are honoured, not truncated.
        let right = fill.sample_in_box([10.0, 0.0], [10.0, 10.0]);
        assert!((right[0] - 1.0).abs() < 1e-5);
        let near = fill.sample_in_box([9.0, 0.0], [10.0, 10.0]);
        assert!((near[0] - 0.95).abs() < 1e-4);
    }
}
