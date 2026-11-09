//! Layers: a run of overlay items drawn into a scratch target and composited
//! as one.

use std::ops::Range;

use crate::overlay::{BackdropEffects, ColourMatrix, OverlayBlend, OverlayClip, ShadowLayer};

/// How a layer's composited result combines with what is under it.
///
/// A layer draws its members into a scratch target on their own, then
/// composites that target as one quad: blurred, shadowed and filtered, faded by
/// `opacity` and blended by `blend`. Because the members are combined first, a
/// faded layer with overlapping members shows the top member at the overlap,
/// not the two blended through each other.
///
/// The common case costs nothing extra: a layer with nothing to do as a whole
/// (opacity `1.0`, no colour matrix, no blur, shadow or backdrop, and
/// [`OverlayBlend::SourceOver`]) draws its members as they are, and a layer
/// with a single immediate member and no blur, shadow, backdrop or
/// destination-reading blend draws that member with the layer's opacity,
/// blend and colour matrix applied to it. Neither allocates a scratch target.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OverlayLayerStyle {
    /// Opacity multiplier in `[0, 1]` applied to the composited layer.
    pub opacity: f32,
    /// How the composited layer combines with what is under it. Every mode is
    /// available, the destination-reading ones included.
    pub blend: OverlayBlend,
    /// A colour filter applied to the composited layer before `opacity`.
    pub colour_matrix: Option<ColourMatrix>,
    /// Gaussian blur applied to the layer's content: the standard deviation
    /// in logical pixels, as CSS `filter: blur()` takes it. `0.0` (the
    /// default) draws it sharp.
    pub blur: f32,
    /// A drop shadow cast by the layer's content as a whole: the union of its
    /// members' coverage, faded over `blur` the way an item's shadow layer is
    /// (a Gaussian of half that standard deviation, as CSS `box-shadow`), moved
    /// by `offset`, in `colour`. `spread` and `falloff` are not applied to a
    /// layer's shadow.
    pub shadow: Option<ShadowLayer>,
    /// A blurred, filtered copy of what is under the layer, drawn under its
    /// content and inside its clip (or its bounds when it has none).
    pub backdrop: BackdropEffects,
}

impl Default for OverlayLayerStyle {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            blend: OverlayBlend::SourceOver,
            colour_matrix: None,
            blur: 0.0,
            shadow: None,
            backdrop: BackdropEffects::default(),
        }
    }
}

impl OverlayLayerStyle {
    /// Set the opacity of the composited layer.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set how the layer combines with what is under it.
    pub fn with_blend(mut self, blend: OverlayBlend) -> Self {
        self.blend = blend;
        self
    }

    /// Set the colour filter applied to the composited layer.
    pub fn with_colour_matrix(mut self, matrix: ColourMatrix) -> Self {
        self.colour_matrix = Some(matrix);
        self
    }

    /// Blur the layer's content: a Gaussian of standard deviation `radius`
    /// logical pixels.
    pub fn with_blur(mut self, radius: f32) -> Self {
        self.blur = radius;
        self
    }

    /// Cast one drop shadow under the layer's content as a whole.
    pub fn with_shadow(mut self, shadow: ShadowLayer) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Draw a blurred copy of what is under the layer behind its content.
    pub fn with_backdrop(mut self, backdrop: BackdropEffects) -> Self {
        self.backdrop = backdrop;
        self
    }

    /// Whether the layer has an effect that only a scratch target can draw:
    /// a blur, a shadow, a backdrop, or a blend that reads the destination.
    pub fn needs_target(&self) -> bool {
        self.blur > 0.0
            || self.shadow.is_some_and(|s| s.colour.alpha() > 0.0)
            || self.backdrop.is_active()
            || self.blend.reads_destination()
    }

    /// Whether the layer changes nothing about how its members draw.
    pub fn is_passthrough(&self) -> bool {
        self.opacity >= 1.0
            && self.blend == OverlayBlend::SourceOver
            && self.colour_matrix.is_none_or(|m| m.is_identity())
            && !self.needs_target()
    }
}

/// The items a layer covers, as a contiguous range of each of the frame's
/// item lists, in submission order.
///
/// Ranges are indices into [`OverlayFrame`](crate::overlay::OverlayFrame)'s
/// `shapes`, `labels`, `glyph_runs`, `polylines`, `meshes` and `retained`. A layer built
/// with [`OverlayFrame::layer_mark`](crate::overlay::OverlayFrame::layer_mark)
/// and [`OverlayFrame::push_layer`](crate::overlay::OverlayFrame::push_layer)
/// takes everything pushed between the two calls.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct OverlayLayerMembers {
    /// Indices into `OverlayFrame::shapes`.
    pub shapes: Range<usize>,
    /// Indices into `OverlayFrame::labels`.
    pub labels: Range<usize>,
    /// Indices into `OverlayFrame::glyph_runs`.
    pub glyph_runs: Range<usize>,
    /// Indices into `OverlayFrame::polylines`.
    pub polylines: Range<usize>,
    /// Indices into `OverlayFrame::meshes`.
    pub meshes: Range<usize>,
    /// Indices into `OverlayFrame::retained`.
    pub retained: Range<usize>,
}

impl OverlayLayerMembers {
    /// Name `range` of the frame's shapes.
    pub fn with_shapes(mut self, range: Range<usize>) -> Self {
        self.shapes = range;
        self
    }

    /// Name `range` of the frame's labels.
    pub fn with_labels(mut self, range: Range<usize>) -> Self {
        self.labels = range;
        self
    }

    /// Name `range` of the frame's glyph runs.
    pub fn with_glyph_runs(mut self, range: Range<usize>) -> Self {
        self.glyph_runs = range;
        self
    }

    /// Name `range` of the frame's polylines.
    pub fn with_polylines(mut self, range: Range<usize>) -> Self {
        self.polylines = range;
        self
    }

    /// Name `range` of the frame's meshes.
    pub fn with_meshes(mut self, range: Range<usize>) -> Self {
        self.meshes = range;
        self
    }

    /// Name `range` of the frame's retained groups.
    pub fn with_retained(mut self, range: Range<usize>) -> Self {
        self.retained = range;
        self
    }

    /// The items between two marks taken from the same frame.
    pub fn between(start: OverlayLayerMark, end: OverlayLayerMark) -> Self {
        Self {
            shapes: start.shapes..end.shapes,
            labels: start.labels..end.labels,
            glyph_runs: start.glyph_runs..end.glyph_runs,
            polylines: start.polylines..end.polylines,
            meshes: start.meshes..end.meshes,
            retained: start.retained..end.retained,
        }
    }

    /// How many items the ranges name.
    pub fn len(&self) -> usize {
        self.shapes.len()
            + self.labels.len()
            + self.glyph_runs.len()
            + self.polylines.len()
            + self.meshes.len()
            + self.retained.len()
    }

    /// Whether the ranges name no item.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The length of each of a frame's item lists at one moment, so the items
/// pushed after it can be named as a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OverlayLayerMark {
    shapes: usize,
    labels: usize,
    glyph_runs: usize,
    polylines: usize,
    meshes: usize,
    retained: usize,
}

/// A layer: overlay items drawn together and composited as one.
///
/// Members keep their own `z_order` among themselves; the layer as a whole
/// sits at `z_order` in the frame's paint order and, at an equal `z_order`,
/// over the shapes and the text stream. An item belongs to the first layer
/// whose ranges name it; layers do not nest.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct OverlayLayer {
    /// The items the layer covers.
    pub members: OverlayLayerMembers,
    /// Where the composited layer sits in the frame's paint order.
    pub z_order: i32,
    /// Clips the composited layer, its backdrop included. The members keep
    /// their own clips as well. The default clips nothing.
    pub clip: OverlayClip,
    /// How the composited layer is drawn.
    pub style: OverlayLayerStyle,
}

impl OverlayLayer {
    /// A layer over `members` with `style`, at `z_order` 0, unclipped.
    pub fn new(members: OverlayLayerMembers, style: OverlayLayerStyle) -> Self {
        Self {
            members,
            z_order: 0,
            clip: OverlayClip::default(),
            style,
        }
    }

    /// Set where the layer sits in the paint order.
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Set what the composited layer is clipped to.
    pub fn with_clip(mut self, clip: OverlayClip) -> Self {
        self.clip = clip;
        self
    }
}

impl crate::overlay::OverlayFrame {
    /// The current length of every item list, to pass to
    /// [`push_layer`](Self::push_layer) once the layer's items are pushed.
    pub fn layer_mark(&self) -> OverlayLayerMark {
        OverlayLayerMark {
            shapes: self.shapes.len(),
            labels: self.labels.len(),
            glyph_runs: self.glyph_runs.len(),
            polylines: self.polylines.len(),
            meshes: self.meshes.len(),
            retained: self.retained.len(),
        }
    }

    /// Make the items pushed since `start` a layer drawn with `style`, and
    /// return it for further settings. Its `z_order` is the lowest of its
    /// members', so it paints where its first member would have.
    pub fn push_layer(
        &mut self,
        start: OverlayLayerMark,
        style: OverlayLayerStyle,
    ) -> &mut OverlayLayer {
        let members = OverlayLayerMembers::between(start, self.layer_mark());
        let z = self.shapes[members.shapes.clone()]
            .iter()
            .map(|i| i.z_order)
            .chain(
                self.labels[members.labels.clone()]
                    .iter()
                    .map(|i| i.z_order),
            )
            .chain(
                self.glyph_runs[members.glyph_runs.clone()]
                    .iter()
                    .map(|i| i.z_order),
            )
            .chain(
                self.polylines[members.polylines.clone()]
                    .iter()
                    .map(|i| i.z_order),
            )
            .chain(
                self.meshes[members.meshes.clone()]
                    .iter()
                    .map(|i| i.z_order),
            )
            .chain(
                self.retained[members.retained.clone()]
                    .iter()
                    .map(|i| i.z_order),
            )
            .min()
            .unwrap_or(0);
        self.layers
            .push(OverlayLayer::new(members, style).with_z_order(z));
        self.layers.last_mut().expect("pushed above")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::{OverlayFrame, OverlayShape, OverlayShapeItem};

    #[test]
    fn marks_name_the_items_pushed_between_them() {
        let mut frame = OverlayFrame::default();
        let rect = || {
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 0.0 },
                [0.0; 2],
                [1.0; 2],
            )
        };
        frame.shapes.push(rect());
        let mark = frame.layer_mark();
        frame.shapes.push(rect().with_z_order(5));
        frame.shapes.push(rect().with_z_order(3));
        let layer = frame.push_layer(mark, OverlayLayerStyle::default().with_opacity(0.5));
        assert_eq!(layer.members.shapes, 1..3);
        assert_eq!(layer.members.len(), 2);
        assert_eq!(layer.z_order, 3);
    }

    #[test]
    fn passthrough_and_target_rules() {
        assert!(OverlayLayerStyle::default().is_passthrough());
        assert!(
            !OverlayLayerStyle::default()
                .with_opacity(0.5)
                .is_passthrough()
        );
        assert!(
            !OverlayLayerStyle::default()
                .with_opacity(0.5)
                .needs_target()
        );
        assert!(
            OverlayLayerStyle::default()
                .with_blend(OverlayBlend::Difference)
                .needs_target()
        );
        assert!(OverlayLayerStyle::default().with_blur(2.0).needs_target());
    }
}
