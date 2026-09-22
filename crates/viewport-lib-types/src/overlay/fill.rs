//! Fill styles for overlay items: solid colour, gradients, and textures.

use crate::overlay::{NineSlice, OverlayTextureId, TextureTransform};

/// What an overlay item is filled with.
///
/// `Solid` is a single flat colour. `LinearGradient`, `RadialGradient`, and
/// `ConicalGradient` interpolate between two colours across the item's
/// bounding box, and the `Multi` variants do the same with three or more
/// stops. `Texture` samples an uploaded image instead.
///
/// One fill at a time, which is what the renderer can draw: the textured
/// pipeline has no gradient code and the gradient pipeline samples no texture.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OverlayFill {
    /// Uniform solid colour in linear RGBA float format.
    Solid(crate::colour::Colour),
    /// Linear gradient between two colours.
    ///
    /// The gradient runs along `angle` across the bounding box. `angle = 0.0`
    /// goes left-to-right (`start_colour` on the left, `end_colour` on the
    /// right). Positive angles rotate the direction counter-clockwise in math
    /// coordinates; because screen Y points downward, `angle = PI/2` produces
    /// a top-to-bottom gradient (start at top, end at bottom).
    LinearGradient {
        /// RGBA colour at the start of the gradient (left when angle is 0).
        start_colour: crate::colour::Colour,
        /// RGBA colour at the end of the gradient (right when angle is 0).
        end_colour: crate::colour::Colour,
        /// Gradient direction in radians. `0.0` = left-to-right.
        angle: f32,
    },
    /// Radial gradient running from the shape centre to its bounding-box edge.
    ///
    /// `centre_colour` sits at the shape origin; `edge_colour` sits at the
    /// farthest bounding-box corner. The transition follows
    /// `length(local_pos) / max_half_size`.
    RadialGradient {
        /// RGBA colour at the centre of the shape.
        centre_colour: crate::colour::Colour,
        /// RGBA colour at the bounding-box edge.
        edge_colour: crate::colour::Colour,
    },
    /// Conical (sweep) gradient rotating around the shape centre.
    ///
    /// The hue wraps once around the origin like a colour wheel.
    /// `offset_angle` rotates the seam (where `end_colour` meets
    /// `start_colour`) counter-clockwise in math coordinates.
    ConicalGradient {
        /// RGBA colour at the sweep start.
        start_colour: crate::colour::Colour,
        /// RGBA colour at the sweep end (wraps back to start).
        end_colour: crate::colour::Colour,
        /// Rotation offset in radians. `0.0` places the seam to the right.
        offset_angle: f32,
    },
    /// Linear gradient with three or more colour stops at arbitrary
    /// positions along the gradient axis. Use when a two-stop ramp is too
    /// flat; designers commonly stack 3-5 stops for polished surfaces.
    /// Stops outside `[0, 1]` are clamped; more than
    /// [`OVERLAY_MAX_GRADIENT_STOPS`] entries are truncated.
    LinearGradientMulti {
        /// Stops in source order. Need not be pre-sorted by position; the
        /// renderer sorts them at prepare time.
        stops: Vec<GradientStop>,
        /// Gradient direction in radians. `0.0` = left-to-right.
        angle: f32,
    },
    /// Radial gradient with three or more colour stops between the shape
    /// centre and its bounding-box edge.
    RadialGradientMulti {
        /// Stops along the centre-to-edge axis.
        stops: Vec<GradientStop>,
    },
    /// Conical gradient with three or more colour stops along the sweep.
    ConicalGradientMulti {
        /// Stops along the `[0, 1]` sweep parameter.
        stops: Vec<GradientStop>,
        /// Rotation offset in radians.
        offset_angle: f32,
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
        /// at their authored size. Honoured on an analytic overlay shape; a
        /// tessellated item samples the image straight through.
        nine_slice: Option<NineSlice>,
        /// Multiplied into every sample, white by default. This is where a
        /// textured item's colour comes from: a mid-grey tint darkens the
        /// image, and an alpha below one fades it.
        tint: crate::colour::Colour,
    },
}

/// Which kind of fill an [`OverlayFill`] is, for asking a family what it can
/// draw without naming a specific gradient or image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverlayFillKind {
    /// A flat colour.
    Solid,
    /// Any of the gradient variants, two-stop or multi-stop.
    Gradient,
    /// An uploaded image.
    Texture,
}

impl Default for OverlayFill {
    fn default() -> Self {
        OverlayFill::Solid([0.0, 0.0, 0.0, 0.55].into())
    }
}

/// A single colour stop in a multi-stop gradient.
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
}

/// Maximum number of stops carried in a single multi-stop gradient. Stops
/// beyond this cap are truncated at prepare time. Covers the vast majority
/// of UI gradient use cases; can be raised by widening the vertex layout
/// if a consumer needs more.
pub const OVERLAY_MAX_GRADIENT_STOPS: usize = 4;

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
    /// A texture fill with the default transform, no nine-patch, and no tint.
    pub fn texture(id: OverlayTextureId) -> Self {
        OverlayFill::Texture {
            id,
            transform: TextureTransform::default(),
            nine_slice: None,
            tint: crate::colour::Colour::linear(1.0, 1.0, 1.0, 1.0),
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
            _ => OverlayFillKind::Gradient,
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

    /// Set the tint multiplied into every sample of a texture fill.
    pub fn with_tint(mut self, colour: impl Into<crate::colour::Colour>) -> Self {
        if let OverlayFill::Texture { tint, .. } = &mut self {
            *tint = colour.into();
        }
        self
    }

    /// Sample the fill at `p`, given in logical pixels relative to the centre
    /// of a box of half-extents `half_size`. Returns linear RGBA.
    ///
    /// This mirrors the gradient maths the SDF shape shader runs per fragment
    /// (`overlay_shape.wgsl`), so a gradient means the same thing wherever it
    /// is evaluated. The glyph families use it per vertex: a linear gradient
    /// interpolated across a quad is exact, and a radial or conical one is
    /// piecewise-linear per glyph, which is close at text sizes and visibly
    /// faceted on very large display text.
    pub fn sample_in_box(&self, p: [f32; 2], half_size: [f32; 2]) -> [f32; 4] {
        fn lerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                a[3] + (b[3] - a[3]) * t,
            ]
        }
        fn linear_t(p: [f32; 2], hs: [f32; 2], angle: f32) -> f32 {
            let dir = [angle.cos(), angle.sin()];
            let max_proj = (hs[0] * dir[0]).abs() + (hs[1] * dir[1]).abs();
            ((p[0] * dir[0] + p[1] * dir[1]) / max_proj.max(0.001) * 0.5 + 0.5).clamp(0.0, 1.0)
        }
        fn radial_t(p: [f32; 2], hs: [f32; 2]) -> f32 {
            let max_half = hs[0].max(hs[1]);
            ((p[0] * p[0] + p[1] * p[1]).sqrt() / max_half.max(0.001)).clamp(0.0, 1.0)
        }
        fn conical_t(p: [f32; 2], offset: f32) -> f32 {
            let a = p[1].atan2(p[0]) - offset;
            (a / std::f32::consts::TAU + 1.0).fract()
        }
        /// Interpolate a sorted stop list at `t`, clamping outside the ends.
        fn sample_stops(stops: &[GradientStop], t: f32) -> [f32; 4] {
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

        match self {
            OverlayFill::Solid(c) => c.to_linear_rgba(),
            // A texture is not sampled here: the image lookup happens in the
            // shader, and the tint is what the rest of the pipeline multiplies
            // into it.
            OverlayFill::Texture { tint, .. } => tint.to_linear_rgba(),
            OverlayFill::LinearGradient {
                start_colour,
                end_colour,
                angle,
            } => lerp(
                start_colour.to_linear_rgba(),
                end_colour.to_linear_rgba(),
                linear_t(p, half_size, *angle),
            ),
            OverlayFill::RadialGradient {
                centre_colour,
                edge_colour,
            } => lerp(
                centre_colour.to_linear_rgba(),
                edge_colour.to_linear_rgba(),
                radial_t(p, half_size),
            ),
            OverlayFill::ConicalGradient {
                start_colour,
                end_colour,
                offset_angle,
            } => lerp(
                start_colour.to_linear_rgba(),
                end_colour.to_linear_rgba(),
                conical_t(p, *offset_angle),
            ),
            OverlayFill::LinearGradientMulti { stops, angle } => {
                sample_stops(stops, linear_t(p, half_size, *angle))
            }
            OverlayFill::RadialGradientMulti { stops } => {
                sample_stops(stops, radial_t(p, half_size))
            }
            OverlayFill::ConicalGradientMulti {
                stops,
                offset_angle,
            } => sample_stops(stops, conical_t(p, *offset_angle)),
        }
    }
}
