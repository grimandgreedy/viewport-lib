//! The baked appearance every overlay item shares, and what each family can
//! actually draw of it.

use crate::overlay::{OverlayFill, OverlayFillKind, ShadowLayer};

/// Colour filters applied to the blurred scene behind a shape.
///
/// Only active where the renderer owns the command encoder (`render`,
/// `render_viewport`); in the `paint` / `paint_to` paths a blurred item falls
/// back to a regular solid fill.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BackdropEffects {
    /// Blur radius in logical pixels. `0.0` (the default) disables the whole
    /// backdrop pass, and the other fields with it.
    pub blur: f32,
    /// Saturation multiplier. `1.0` leaves saturation unchanged, `0.0`
    /// produces greyscale.
    pub saturation: f32,
    /// Brightness multiplier. `1.0` is unchanged.
    pub brightness: f32,
    /// Hue rotation in radians. `0.0` is unchanged.
    pub hue_shift: f32,
}

impl Default for BackdropEffects {
    fn default() -> Self {
        Self {
            blur: 0.0,
            saturation: 1.0,
            brightness: 1.0,
            hue_shift: 0.0,
        }
    }
}

impl BackdropEffects {
    /// Whether the backdrop pass runs at all.
    pub fn is_active(&self) -> bool {
        self.blur > 0.0
    }

    /// Set the blur radius in logical pixels.
    pub fn with_blur(mut self, blur: f32) -> Self {
        self.blur = blur;
        self
    }

    /// Set the saturation, brightness, and hue-rotation filters.
    pub fn with_filters(mut self, saturation: f32, brightness: f32, hue_shift: f32) -> Self {
        self.saturation = saturation;
        self.brightness = brightness;
        self.hue_shift = hue_shift;
        self
    }
}

/// Baked appearance shared by the four overlay item types: what the item is
/// filled and layered with, as opposed to where it sits.
///
/// Changing any of this on a retained group means re-compiling it, which is the
/// line between this struct and
/// [`OverlayTransform`](crate::overlay::OverlayTransform): a transform, an
/// opacity, or a tint can change from frame to frame without touching the
/// compiled buffers, and a style cannot.
///
/// # No `serde`
///
/// This is the one overlay type that cannot be serialised: an
/// [`OverlayFill::Texture`] names an uploaded image by a runtime slot handle,
/// which means nothing in a file or across a process. Authored content that
/// wants a style stores the fill's parameters and resolves the image itself.
///
/// # Not every family draws every field
///
/// The three coverage backends differ in what they can express, so a field can
/// be present and inert. Ask [`OverlayStyleSupport`] rather than guessing, and
/// see its docs for the per-family table. In debug builds the renderer logs
/// once when a non-default value lands in a cell that family does not draw.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OverlayStyle {
    /// What the item is filled with: a colour, a gradient, or a texture.
    ///
    /// Defaults to a fully transparent solid, which draws no fill and leaves
    /// the shadow layers, the stroke, or the glyphs on their own.
    pub fill: OverlayFill,
    /// Stacked drop shadows and contours drawn behind the item, first entry
    /// furthest back. Up to
    /// [`OVERLAY_MAX_SHADOW_LAYERS`](crate::overlay::OVERLAY_MAX_SHADOW_LAYERS)
    /// are honoured.
    pub shadows: Vec<ShadowLayer>,
    /// Stacked inset shadows drawn on top of the fill, eroding inward from the
    /// item's boundary. Up to
    /// [`OVERLAY_MAX_SHADOW_LAYERS`](crate::overlay::OVERLAY_MAX_SHADOW_LAYERS)
    /// are honoured.
    pub inner_shadows: Vec<ShadowLayer>,
    /// Blur and colour filters applied to the scene behind the item.
    pub backdrop: BackdropEffects,
    /// Colour multiplier applied to the whole item, identity `[1, 1, 1, 1]`.
    ///
    /// Composes multiplicatively with the item's own colours and with the tint
    /// of a retained group containing it. **A tint never reaches a shadow
    /// layer**, on any path: a compiled group's shadow colours are baked, so
    /// honouring it there would make the same content look different on the
    /// two paths.
    pub tint: [f32; 4],
    /// Overall opacity multiplier in `[0, 1]`, applied to the fill and to every
    /// shadow layer.
    ///
    /// Unlike `tint`, this does reach the shadow layers: fading an item out has
    /// to take its shadow with it or the shadow outlives the thing casting it.
    pub opacity: f32,
}

impl Default for OverlayStyle {
    fn default() -> Self {
        Self {
            fill: OverlayFill::none(),
            shadows: Vec::new(),
            inner_shadows: Vec::new(),
            backdrop: BackdropEffects::default(),
            tint: [1.0, 1.0, 1.0, 1.0],
            opacity: 1.0,
        }
    }
}

impl OverlayStyle {
    /// A solid fill and nothing else.
    pub fn solid(colour: impl Into<crate::colour::Colour>) -> Self {
        Self {
            fill: OverlayFill::Solid(colour.into()),
            ..Default::default()
        }
    }

    /// Set the fill.
    pub fn with_fill(mut self, fill: OverlayFill) -> Self {
        self.fill = fill;
        self
    }

    /// Set the stacked outer shadow layers.
    pub fn with_shadows(mut self, shadows: Vec<ShadowLayer>) -> Self {
        self.shadows = shadows;
        self
    }

    /// Set the stacked inner (inset) shadow layers.
    pub fn with_inner_shadows(mut self, shadows: Vec<ShadowLayer>) -> Self {
        self.inner_shadows = shadows;
        self
    }

    /// Set the backdrop blur and filters.
    pub fn with_backdrop(mut self, backdrop: BackdropEffects) -> Self {
        self.backdrop = backdrop;
        self
    }

    /// Set the colour multiplier (identity `[1, 1, 1, 1]`).
    pub fn with_tint(mut self, tint: [f32; 4]) -> Self {
        self.tint = tint;
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }
}

/// Which [`OverlayStyle`] fields a given item actually draws.
///
/// The axis that decides this is the coverage backend, not the item type: an
/// analytic shape has a distance field, a vector path and a polyline are
/// tessellated triangles, and a label or glyph run is a bitmap atlas. A field
/// is cheap to share only where all three can express it.
///
/// Ask through [`OverlayStyleSupport::for_shape`] and the other constructors.
/// A lowering layer that builds overlay items from a widget tree should branch
/// on this and emit a fallback, rather than setting a field that silently does
/// nothing.
///
/// # The fields are the contract; the curve is not
///
/// A `ShadowLayer` means the same thing everywhere, so `shadows` and
/// `inner_shadows` are not variables here: an outer layer dilates what the
/// item covers and is clipped to outside it, an inner layer erodes it inward
/// from the boundary, and both draw on every family. What does differ is the
/// falloff: the SDF path smoothsteps the distance field, the glyph path runs
/// two box passes over the coverage bitmap, and the tessellated path draws five
/// banded steps. Identical values are close but not identical across families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct OverlayStyleSupport {
    /// The [`OverlayFill`] kinds this family draws. A fill of a kind not
    /// listed here is inert.
    pub fill: &'static [OverlayFillKind],
    /// `backdrop` is composited.
    pub backdrop: bool,
}

/// Colours and gradients, but no image.
const COLOUR_FILLS: &[OverlayFillKind] = &[OverlayFillKind::Solid, OverlayFillKind::Gradient];
/// Every fill kind.
const ALL_FILLS: &[OverlayFillKind] = &[
    OverlayFillKind::Solid,
    OverlayFillKind::Gradient,
    OverlayFillKind::Texture,
];

impl OverlayStyleSupport {
    /// Everything drawn.
    pub const ALL: Self = Self {
        fill: ALL_FILLS,
        backdrop: true,
    };

    /// Nothing drawn.
    pub const NONE: Self = Self {
        fill: &[],
        backdrop: false,
    };

    /// What an [`OverlayShapeItem`] draws, which depends on its variant:
    /// analytic variants have a distance field and draw everything, while
    /// `OverlayShape::Vector` is tessellated triangles, with no image sampling
    /// and no distance field to run a backdrop mask off.
    ///
    /// This is why the query takes the shape rather than being a constant per
    /// type: the variant is known where the item is built, which is where a
    /// lowering layer needs the answer.
    ///
    /// [`OverlayShapeItem`]: crate::overlay::OverlayShapeItem
    pub const fn for_shape(shape: &crate::overlay::OverlayShape) -> Self {
        match shape {
            crate::overlay::OverlayShape::Vector { .. } => Self {
                fill: COLOUR_FILLS,
                backdrop: false,
            },
            _ => Self::ALL,
        }
    }

    /// What an `OverlayPolylineItem` draws. Only a closed polyline has an
    /// interior to fill, so an open one draws no fill at all: pass the item's
    /// `closed` flag.
    pub const fn for_polyline(closed: bool) -> Self {
        Self {
            fill: if closed { ALL_FILLS } else { &[] },
            backdrop: false,
        }
    }

    /// What a `LabelItem` or a `GlyphRunItem` draws. Both rasterise through the
    /// glyph atlas, where a fill is a per-vertex tint over the coverage.
    ///
    /// A texture fill is the one gap left: the text pass binds the glyph atlas
    /// and issues a single batched draw, so sampling a second image means
    /// grouping the text by texture the way the shape pass does. Until then a
    /// textured glyph fill reports as unsupported rather than silently
    /// dropping.
    pub const fn for_glyphs() -> Self {
        Self {
            fill: COLOUR_FILLS,
            backdrop: false,
        }
    }

    /// Whether this family draws `fill`.
    pub fn draws_fill(&self, fill: &OverlayFill) -> bool {
        let kind = fill.kind();
        self.fill.iter().any(|k| *k == kind)
    }

    /// The style fields that are set to something other than their default and
    /// are not drawn by this family, as a list of field names.
    ///
    /// Empty when the style is fully honoured. The renderer calls this in debug
    /// builds to log the mismatch once; a consumer can call it in a test to
    /// assert that a lowering layer never emits an inert field.
    pub fn inert_fields(&self, style: &OverlayStyle) -> Vec<&'static str> {
        let mut out = Vec::new();
        // An untouched fill is nothing to report: it draws the same as not
        // being drawn.
        if style.fill != OverlayFill::none() && !self.draws_fill(&style.fill) {
            out.push("fill");
        }
        if !self.backdrop && style.backdrop.is_active() {
            out.push("backdrop");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::{OverlayShape, ShadowLayer};

    #[test]
    fn vector_shapes_report_the_cells_they_cannot_draw() {
        let style = OverlayStyle::solid([1.0, 0.0, 0.0, 1.0])
            .with_inner_shadows(vec![ShadowLayer::default()])
            .with_backdrop(BackdropEffects::default().with_blur(4.0));

        let analytic = OverlayStyleSupport::for_shape(&OverlayShape::Circle);
        assert!(analytic.inert_fields(&style).is_empty());

        let vector = OverlayStyleSupport::for_shape(&OverlayShape::Vector {
            subpaths: Vec::new(),
            fill_rule: crate::overlay::FillRule::NonZero,
        });
        assert_eq!(vector.inert_fields(&style), ["backdrop"]);
    }

    #[test]
    fn a_default_style_is_never_inert() {
        let style = OverlayStyle::default();
        assert!(
            OverlayStyleSupport::for_glyphs()
                .inert_fields(&style)
                .is_empty()
        );
        assert!(OverlayStyleSupport::NONE.inert_fields(&style).is_empty());
    }
}
