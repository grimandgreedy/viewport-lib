//! The baked appearance every overlay item shares, and what each family can
//! actually draw of it.

use crate::overlay::{OverlayFill, OverlayFillKind, ShadowLayer};

/// A blur and colour filters applied to what is drawn behind a shape or a
/// layer: the scene and every overlay item under it in paint order.
///
/// The backdrop is read at the item's place in the overlay draw order, so a
/// frosted card over a page background drawn as an overlay rect blurs that
/// background. Reading what the overlay pass has drawn so far needs the
/// renderer to own the command encoder (`render`, `render_viewport`) and to be
/// able to copy the target (`COPY_SRC`, which the offscreen and LDR
/// intermediate targets have). Where it cannot copy, the backdrop is the scene
/// alone; in the `paint` / `paint_to` paths, which draw into a pass the host
/// owns, there is nothing to blur and a shape with a backdrop is not drawn.
/// Inside a layer, a shape's backdrop is dropped and the shape draws its fill;
/// give the layer the backdrop instead.
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

/// How an overlay item's colour combines with what is already drawn under it.
///
/// The Porter-Duff modes and [`Additive`](Self::Additive),
/// [`Multiply`](Self::Multiply) and [`Screen`](Self::Screen) are fixed-function
/// blend states, so changing between them costs a pipeline switch and nothing
/// more. The Porter-Duff modes treat the item and the destination as
/// premultiplied-alpha colour.
///
/// The separable and non-separable blend modes from [`Overlay`](Self::Overlay)
/// to [`Luminosity`](Self::Luminosity) read the destination
/// ([`reads_destination`](Self::reads_destination)), which a blend state
/// cannot. They draw through a layer: the item, or the layer's members, are
/// drawn into a scratch target and composited with the destination as a
/// texture, using the W3C compositing formulas on linear colour. A layer with
/// [`Multiply`](Self::Multiply) that needs a scratch target anyway composites
/// the same way, which makes it exact over a non-opaque destination. Where the
/// destination cannot be read (a pass the host owns, or a target without
/// `COPY_SRC`), a destination-reading mode draws as
/// [`SourceOver`](Self::SourceOver).
///
/// [`SourceOver`](Self::SourceOver), [`Additive`](Self::Additive),
/// [`Multiply`](Self::Multiply), [`Screen`](Self::Screen),
/// [`DestinationOver`](Self::DestinationOver),
/// [`DestinationOut`](Self::DestinationOut), [`SourceAtop`](Self::SourceAtop),
/// [`Xor`](Self::Xor), [`Plus`](Self::Plus) and
/// [`Destination`](Self::Destination) leave the destination alone wherever the
/// item has no coverage. The rest ([`Clear`](Self::Clear),
/// [`Source`](Self::Source), [`SourceIn`](Self::SourceIn),
/// [`DestinationIn`](Self::DestinationIn), [`SourceOut`](Self::SourceOut) and
/// [`DestinationAtop`](Self::DestinationAtop)) change the destination where the
/// item is transparent too, so they act over the whole area the item's geometry
/// covers: for an analytic shape that is its quad, anti-aliasing and shadow
/// margin included, not just its outline.
///
/// A compiled overlay group draws every item [`SourceOver`](Self::SourceOver).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum OverlayBlend {
    /// The item over the destination: ordinary alpha compositing.
    #[default]
    SourceOver,
    /// Adds the item's colour, weighted by its alpha, to the destination and
    /// leaves the destination's alpha unchanged. Brightens; never darkens.
    Additive,
    /// Multiplies the destination by the item's colour where the item covers
    /// it. Darkens; white leaves the destination unchanged. Exact over an
    /// opaque destination.
    Multiply,
    /// The inverse of multiplying the inverses: brightens, black leaves the
    /// destination unchanged.
    Screen,
    /// Clears the destination to transparent.
    Clear,
    /// Replaces the destination with the item.
    Source,
    /// Leaves the destination as it is: the item draws nothing.
    Destination,
    /// The destination over the item: the item shows only where the
    /// destination is transparent.
    DestinationOver,
    /// The item, kept only where the destination has coverage.
    SourceIn,
    /// The destination, kept only where the item has coverage.
    DestinationIn,
    /// The item, kept only where the destination is transparent.
    SourceOut,
    /// The destination, removed where the item has coverage: the item punches
    /// a hole.
    DestinationOut,
    /// The item over the destination, kept only where the destination has
    /// coverage.
    SourceAtop,
    /// The destination over the item, kept only where the item has coverage.
    DestinationAtop,
    /// The item and the destination, each kept only where the other is
    /// transparent.
    Xor,
    /// The item and the destination added, colour and alpha both.
    Plus,
    /// Multiply or screen depending on the destination: darks darken, lights
    /// lighten, contrast rises. Reads the destination.
    Overlay,
    /// The darker of the item and the destination, per channel. Reads the
    /// destination.
    Darken,
    /// The lighter of the item and the destination, per channel. Reads the
    /// destination.
    Lighten,
    /// Brightens the destination to reflect the item. Reads the destination.
    ColourDodge,
    /// Darkens the destination to reflect the item. Reads the destination.
    ColourBurn,
    /// Multiply or screen depending on the item: [`Overlay`](Self::Overlay)
    /// with the two swapped. Reads the destination.
    HardLight,
    /// A softer [`HardLight`](Self::HardLight). Reads the destination.
    SoftLight,
    /// The absolute difference of the item and the destination. Reads the
    /// destination.
    Difference,
    /// Like [`Difference`](Self::Difference) with lower contrast. Reads the
    /// destination.
    Exclusion,
    /// The item's hue with the destination's saturation and luminosity. Reads
    /// the destination.
    Hue,
    /// The item's saturation with the destination's hue and luminosity. Reads
    /// the destination.
    Saturation,
    /// The item's hue and saturation with the destination's luminosity. Reads
    /// the destination.
    Colour,
    /// The item's luminosity with the destination's hue and saturation. Reads
    /// the destination.
    Luminosity,
}

impl OverlayBlend {
    /// Whether the mode needs the destination as a texture rather than a
    /// blend state: the modes from [`Overlay`](Self::Overlay) on.
    pub fn reads_destination(&self) -> bool {
        matches!(
            self,
            Self::Overlay
                | Self::Darken
                | Self::Lighten
                | Self::ColourDodge
                | Self::ColourBurn
                | Self::HardLight
                | Self::SoftLight
                | Self::Difference
                | Self::Exclusion
                | Self::Hue
                | Self::Saturation
                | Self::Colour
                | Self::Luminosity
        )
    }
}

/// How an overlay item's outline meets the pixels around it.
///
/// [`AntiAliased`](Self::AntiAliased) ramps coverage across a one-pixel band
/// inside the outline. Two items that touch each lay half-covered pixels along
/// the shared edge, and the background shows through between them.
/// [`Crisp`](Self::Crisp) covers a pixel fully when its centre is inside the
/// outline and not at all otherwise, so a grid of touching cells (a heatmap, a
/// table, a bar chart) covers every pixel in its area exactly once, at any
/// fractional position. A pixel centre that falls exactly on an edge two
/// axis-aligned rects share belongs to the rect on its right (or below).
///
/// Only an axis-aligned rect is exact: a curved, rotated or sheared outline
/// drawn [`Crisp`](Self::Crisp) shows stair-stepped edges. Ask
/// [`OverlayStyleSupport::crisp_edges`] which families draw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum OverlayEdge {
    /// A one-pixel coverage ramp inside the outline.
    #[default]
    AntiAliased,
    /// Full coverage at pixel centres inside the outline, none outside. Shadow
    /// layers with no blur (outlines and contours) step the same way; blurred
    /// ones keep their fade.
    Crisp,
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
/// The coverage backends differ in what they can express, so a field can
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
    /// How the item combines with what is drawn under it. Defaults to
    /// [`OverlayBlend::SourceOver`]. A mode that reads the destination draws
    /// the item as a layer of its own.
    pub blend: OverlayBlend,
    /// A colour filter applied to the item's fill colour, after its tint and
    /// opacity. Like `tint`, it does not reach the shadow layers.
    pub colour_matrix: Option<crate::overlay::ColourMatrix>,
    /// How the outline meets the pixels around it. Defaults to
    /// [`OverlayEdge::AntiAliased`]; [`OverlayEdge::Crisp`] draws touching
    /// cells without a seam between them.
    pub edge: OverlayEdge,
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
            blend: OverlayBlend::SourceOver,
            colour_matrix: None,
            edge: OverlayEdge::AntiAliased,
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

    /// Set how the item combines with what is drawn under it.
    pub fn with_blend(mut self, blend: OverlayBlend) -> Self {
        self.blend = blend;
        self
    }

    /// Set the colour filter applied to the item's fill.
    pub fn with_colour_matrix(mut self, matrix: crate::overlay::ColourMatrix) -> Self {
        self.colour_matrix = Some(matrix);
        self
    }

    /// Set how the outline meets the pixels around it.
    pub fn with_edge(mut self, edge: OverlayEdge) -> Self {
        self.edge = edge;
        self
    }
}

/// Which [`OverlayStyle`] fields a given item actually draws.
///
/// The axis that decides this is the coverage backend, not the item type: an
/// analytic shape has a distance field, a vector path and a polyline are
/// tessellated triangles, a mesh is the consumer's own triangles, and a label
/// or glyph run is a bitmap atlas. A field
/// is cheap to share only where all three can express it.
///
/// Ask through [`OverlayStyleSupport::for_shape`] and the other constructors.
/// A lowering layer that builds overlay items from a widget tree should branch
/// on this and emit a fallback, rather than setting a field that silently does
/// nothing.
///
/// # The fields are the contract; the curve is not
///
/// A `ShadowLayer` means the same thing everywhere it draws: an outer layer
/// dilates what the item covers and is clipped to outside it, an inner layer
/// erodes it inward from the boundary, and both draw on every family but the
/// mesh, which has no edge of its own to grow them from. What does differ is the
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
    /// `shadows` and `inner_shadows` are drawn.
    pub shadows: bool,
    /// An `edge` of [`OverlayEdge::Crisp`] draws hard edges. True for an
    /// analytic shape, and for a mesh, whose triangles have no anti-aliasing
    /// band to remove and are always crisp.
    pub crisp_edges: bool,
}

/// Every fill kind.
const ALL_FILLS: &[OverlayFillKind] = &[
    OverlayFillKind::Solid,
    OverlayFillKind::Gradient,
    OverlayFillKind::Texture,
    OverlayFillKind::Custom,
];

impl OverlayStyleSupport {
    /// Everything drawn.
    pub const ALL: Self = Self {
        fill: ALL_FILLS,
        backdrop: true,
        shadows: true,
        crisp_edges: true,
    };

    /// Nothing drawn.
    pub const NONE: Self = Self {
        fill: &[],
        backdrop: false,
        shadows: false,
        crisp_edges: false,
    };

    /// What an [`OverlayShapeItem`] draws. Every variant draws every fill kind
    /// and the backdrop, `OverlayShape::Vector` included. Every variant but
    /// `OverlayShape::Vector`, which is tessellated, draws a crisp edge.
    ///
    /// The query takes the shape so a lowering layer can ask where the item is
    /// built, and so a variant that cannot draw a field can say so.
    ///
    /// [`OverlayShapeItem`]: crate::overlay::OverlayShapeItem
    pub const fn for_shape(shape: &crate::overlay::OverlayShape) -> Self {
        Self {
            crisp_edges: !matches!(shape, crate::overlay::OverlayShape::Vector { .. }),
            ..Self::ALL
        }
    }

    /// What an `OverlayPolylineItem` draws. Only a closed polyline has an
    /// interior to fill, so an open one draws no fill at all: pass the item's
    /// `closed` flag.
    pub const fn for_polyline(closed: bool) -> Self {
        Self {
            fill: if closed { ALL_FILLS } else { &[] },
            backdrop: false,
            shadows: true,
            crisp_edges: false,
        }
    }

    /// What a `LabelItem` or a `GlyphRunItem` draws: every fill kind, evaluated
    /// per pixel over the glyph coverage, but no backdrop and no crisp edge.
    pub const fn for_glyphs() -> Self {
        Self {
            fill: ALL_FILLS,
            backdrop: false,
            shadows: true,
            crisp_edges: false,
        }
    }

    /// What an `OverlayMeshItem` draws: every fill kind, multiplied over its
    /// vertex colours, but no backdrop. A mesh has no edge to grow a shadow
    /// layer from, so its `shadows` and `inner_shadows` are not drawn either.
    /// Its edges are always crisp, so `edge` changes nothing.
    pub const fn for_mesh() -> Self {
        Self {
            fill: ALL_FILLS,
            backdrop: false,
            shadows: false,
            crisp_edges: true,
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
        if !self.shadows && !style.shadows.is_empty() {
            out.push("shadows");
        }
        if !self.shadows && !style.inner_shadows.is_empty() {
            out.push("inner_shadows");
        }
        if !self.crisp_edges && style.edge == OverlayEdge::Crisp {
            out.push("edge");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::Colour;
    use crate::overlay::{OverlayShape, ShadowLayer};

    #[test]
    fn vector_shapes_draw_every_cell() {
        let style = OverlayStyle::solid(Colour::linear(1.0, 0.0, 0.0, 1.0))
            .with_inner_shadows(vec![ShadowLayer::default()])
            .with_backdrop(BackdropEffects::default().with_blur(4.0));

        let analytic = OverlayStyleSupport::for_shape(&OverlayShape::Circle);
        assert!(analytic.inert_fields(&style).is_empty());

        let vector = OverlayStyleSupport::for_shape(&OverlayShape::Vector {
            subpaths: Vec::new(),
            fill_rule: crate::overlay::FillRule::NonZero,
            stroke: None,
            inverse: false,
        });
        assert!(vector.inert_fields(&style).is_empty());
        assert!(vector.draws_fill(&OverlayFill::texture(crate::overlay::OverlayTextureId(0))));
    }

    #[test]
    fn glyphs_draw_every_fill_kind() {
        let glyphs = OverlayStyleSupport::for_glyphs();
        assert!(glyphs.draws_fill(&OverlayFill::texture(crate::overlay::OverlayTextureId(0))));
        assert!(glyphs.draws_fill(&OverlayFill::sweep_gradient(
            Colour::WHITE,
            Colour::BLACK,
            0.0,
            1.0
        )));
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

    #[test]
    fn crisp_edges_are_reported_where_they_are_drawn() {
        let style = OverlayStyle::solid(Colour::WHITE).with_edge(OverlayEdge::Crisp);
        let rect = OverlayShape::Rect { corner_radius: 0.0 };
        assert!(
            OverlayStyleSupport::for_shape(&rect)
                .inert_fields(&style)
                .is_empty()
        );
        let vector = OverlayShape::Vector {
            subpaths: Vec::new(),
            fill_rule: crate::overlay::FillRule::NonZero,
            stroke: None,
            inverse: false,
        };
        assert_eq!(
            OverlayStyleSupport::for_shape(&vector).inert_fields(&style),
            vec!["edge"]
        );
        assert_eq!(
            OverlayStyleSupport::for_glyphs().inert_fields(&style),
            vec!["edge"]
        );
        assert!(
            OverlayStyleSupport::for_mesh()
                .inert_fields(&style)
                .is_empty()
        );
    }

    #[test]
    fn a_mesh_draws_no_shadow_layers() {
        let style = OverlayStyle::solid(Colour::WHITE).with_shadows(vec![ShadowLayer::default()]);
        assert_eq!(
            OverlayStyleSupport::for_mesh().inert_fields(&style),
            vec!["shadows"]
        );
        assert!(
            OverlayStyleSupport::for_mesh().draws_fill(&OverlayFill::custom(
                crate::overlay::CustomFillId(0),
                [0.0; 16]
            ))
        );
    }
}
