//! The 2D placement every overlay item and every retained group shares.

/// Translate, rotate, and scale, in logical pixels and radians.
///
/// One vocabulary used in two roles: on an overlay item, and on a
/// [`RetainedOverlay`] standing for a whole compiled group. Both roles hold the
/// same struct, and the two compose.
///
/// # Composition
///
/// The final position of a vertex is
///
/// ```text
/// group_T . item_T . v
/// ```
///
/// where each `T` is `translate . rotate_about(pivot) . scale`.
///
/// That is the definition: two nested affine maps, applied outside-in. Under
/// it, rotation angles add and uniform scales multiply, but those are
/// *consequences*, not the rule. Translation in particular does **not** compose
/// additively: an item at `translate = [10, 0]` inside a group rotated a
/// quarter turn is displaced along the group's rotated axis, not along screen
/// x. Composing the channels one at a time instead makes a rotated scroll panel
/// shear its own contents apart.
///
/// `scale` is uniform by design. A uniform scale commutes with rotation, which
/// is what keeps `group_T . item_T` collapsible to a single affine map with a
/// scalar scale and an angle. A `[f32; 2]` scale does not commute with
/// rotation, so a rotated group containing a non-uniformly scaled item would
/// need the full matrix on both levels and in the instance buffer.
///
/// # What does not live here
///
/// `opacity`, `tint`, `z_order`, `anchor`, `align_x` / `align_y`, `clip_id` and
/// `clip_rect` stay flat on the item and on the group, because their meaning is
/// role-dependent: `z_order` is a draw-order key rather than a transform,
/// anchoring resolves an origin before any of this applies, and a clip is
/// evaluated against the screen rather than carried through the composition.
///
/// [`RetainedOverlay`]: crate::overlay::RetainedOverlay
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct OverlayTransform {
    /// Offset in logical pixels, applied last. On an item this is the nudge
    /// from the resolved `anchor` origin; on a retained group it moves the
    /// whole group.
    pub translate: [f32; 2],
    /// Rotation in radians about `pivot`. Positive rotates counter-clockwise in
    /// maths coordinates, which reads as clockwise on screen because the Y axis
    /// points down.
    pub rotation: f32,
    /// Centre of rotation and scaling, in logical pixels from the item's or
    /// group's own centre. `[0, 0]` turns about the centre.
    pub pivot: [f32; 2],
    /// Uniform scale about `pivot`, applied first. `1.0` is identity. See the
    /// composition note above for why this is a scalar.
    pub scale: f32,
}

impl Default for OverlayTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl OverlayTransform {
    /// No translation, no rotation, unit scale, pivot at the centre.
    pub const IDENTITY: Self = Self {
        translate: [0.0, 0.0],
        rotation: 0.0,
        pivot: [0.0, 0.0],
        scale: 1.0,
    };

    /// Identity except for `translate`.
    pub fn at(translate: [f32; 2]) -> Self {
        Self {
            translate,
            ..Self::IDENTITY
        }
    }

    /// Set the translation in logical pixels.
    pub fn with_translate(mut self, translate: [f32; 2]) -> Self {
        self.translate = translate;
        self
    }

    /// Set the rotation in radians.
    pub fn with_rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set the pivot, in logical pixels from the centre.
    pub fn with_pivot(mut self, pivot: [f32; 2]) -> Self {
        self.pivot = pivot;
        self
    }

    /// Set the uniform scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Whether this is the identity map, so a caller can skip the work.
    pub fn is_identity(&self) -> bool {
        self.translate == [0.0, 0.0]
            && self.rotation == 0.0
            && self.pivot == [0.0, 0.0]
            && self.scale == 1.0
    }

    /// Apply this transform to a point given relative to the transform's own
    /// centre, returning the point in the same space.
    ///
    /// This is one level of `group_T . item_T . v`: `scale` first about
    /// `pivot`, then `rotation` about `pivot`, then `translate`.
    pub fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let dx = (p[0] - self.pivot[0]) * self.scale;
        let dy = (p[1] - self.pivot[1]) * self.scale;
        let (s, c) = self.rotation.sin_cos();
        [
            self.pivot[0] + dx * c - dy * s + self.translate[0],
            self.pivot[1] + dx * s + dy * c + self.translate[1],
        ]
    }

    /// Compose two transforms into one, so that
    /// `outer.compose(inner).apply(v) == outer.apply(inner.apply(v))`.
    ///
    /// Exact because both levels carry a scalar scale, which commutes with
    /// rotation. The composed pivot is the outer pivot: the inner pivot has
    /// already been folded into the composed translation.
    pub fn compose(&self, inner: &Self) -> Self {
        let inner_origin = self.apply(inner.apply([0.0, 0.0]));
        let mut out = Self {
            translate: [0.0, 0.0],
            rotation: self.rotation + inner.rotation,
            pivot: [0.0, 0.0],
            scale: self.scale * inner.scale,
        };
        // Solve for the translation that puts the composed origin where the
        // two-step application puts it.
        let moved = out.apply([0.0, 0.0]);
        out.translate = [inner_origin[0] - moved[0], inner_origin[1] - moved[1]];
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3
    }

    /// The contract, numerically: an item transform inside a rotated,
    /// translated, scaled group resolves to the same point whether the two are
    /// applied in turn or composed first.
    #[test]
    fn composition_is_nested_affine_not_per_channel() {
        let group = OverlayTransform {
            translate: [100.0, 40.0],
            rotation: std::f32::consts::FRAC_PI_2,
            pivot: [0.0, 0.0],
            scale: 2.0,
        };
        let item = OverlayTransform {
            translate: [10.0, 0.0],
            rotation: 0.25,
            pivot: [5.0, -5.0],
            scale: 0.5,
        };
        for v in [[0.0, 0.0], [7.0, 3.0], [-12.0, 20.0]] {
            let stepwise = group.apply(item.apply(v));
            let composed = group.compose(&item).apply(v);
            assert!(
                close(stepwise, composed),
                "{stepwise:?} != {composed:?} for {v:?}"
            );
        }

        // The per-channel shortcut disagrees, which is the whole point of
        // writing the rule down: an item offset 10 px along screen x inside a
        // quarter-turn group must move along the group's rotated axis.
        let per_channel = [
            group.translate[0] + item.translate[0],
            group.translate[1] + item.translate[1],
        ];
        let correct = group.apply(item.apply([0.0, 0.0]));
        assert!(
            !close(per_channel, correct),
            "the two rules agree here, so this scene does not test the contract"
        );
    }

    #[test]
    fn identity_composes_to_nothing() {
        let t = OverlayTransform {
            translate: [3.0, -4.0],
            rotation: 0.3,
            pivot: [1.0, 2.0],
            scale: 1.5,
        };
        for v in [[0.0, 0.0], [10.0, -6.0]] {
            assert!(close(
                OverlayTransform::IDENTITY.compose(&t).apply(v),
                t.apply(v)
            ));
            assert!(close(
                t.compose(&OverlayTransform::IDENTITY).apply(v),
                t.apply(v)
            ));
        }
    }
}
