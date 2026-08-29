/// Fill style for an [`OverlayShapeItem`].
///
/// `Solid` is the default and matches the previous single-colour behaviour.
/// `LinearGradient`, `RadialGradient`, and `ConicalGradient` interpolate
/// between two colours across the shape's bounding box.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OverlayFill {
    /// Uniform solid colour in linear RGBA float format.
    Solid([f32; 4]),
    /// Linear gradient between two colours.
    ///
    /// The gradient runs along `angle` across the bounding box. `angle = 0.0`
    /// goes left-to-right (`start_colour` on the left, `end_colour` on the
    /// right). Positive angles rotate the direction counter-clockwise in math
    /// coordinates; because screen Y points downward, `angle = PI/2` produces
    /// a top-to-bottom gradient (start at top, end at bottom).
    LinearGradient {
        /// RGBA colour at the start of the gradient (left when angle is 0).
        start_colour: [f32; 4],
        /// RGBA colour at the end of the gradient (right when angle is 0).
        end_colour: [f32; 4],
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
        centre_colour: [f32; 4],
        /// RGBA colour at the bounding-box edge.
        edge_colour: [f32; 4],
    },
    /// Conical (sweep) gradient rotating around the shape centre.
    ///
    /// The hue wraps once around the origin like a colour wheel.
    /// `offset_angle` rotates the seam (where `end_colour` meets
    /// `start_colour`) counter-clockwise in math coordinates.
    ConicalGradient {
        /// RGBA colour at the sweep start.
        start_colour: [f32; 4],
        /// RGBA colour at the sweep end (wraps back to start).
        end_colour: [f32; 4],
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
}

impl Default for OverlayFill {
    fn default() -> Self {
        OverlayFill::Solid([0.0, 0.0, 0.0, 0.55])
    }
}

/// A single colour stop in a multi-stop gradient.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradientStop {
    /// Position along the gradient axis, in `[0, 1]`. Stops outside the
    /// range are clamped at evaluation time.
    pub position: f32,
    /// Linear RGBA colour at this stop.
    pub colour: [f32; 4],
}

impl GradientStop {
    /// Construct a stop at the given position and colour.
    pub const fn new(position: f32, colour: [f32; 4]) -> Self {
        Self { position, colour }
    }
}

/// Maximum number of stops carried in a single multi-stop gradient. Stops
/// beyond this cap are truncated at prepare time. Covers the vast majority
/// of UI gradient use cases; can be raised by widening the vertex layout
/// if a consumer needs more.
pub const OVERLAY_MAX_GRADIENT_STOPS: usize = 4;

/// Border placement relative to the shape edge.
///
/// Controls whether the border band sits inside, outside, or centred on the
/// SDF zero-crossing. `Inset` matches the default behaviour from earlier
/// phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BorderMode {
    /// Border sits inside the fill edge (default). The fill area shrinks by
    /// `border_width`.
    #[default]
    Inset,
    /// Border sits outside the fill edge. The fill area is unaffected; the
    /// border extends outward.
    Outer,
    /// Border is centred on the fill edge (half inside, half outside).
    Center,
}
