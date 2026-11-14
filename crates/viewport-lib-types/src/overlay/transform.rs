//! The 2D placement every overlay item and every retained group shares.

/// Translate, rotate, scale, stretch and skew, in logical pixels and radians.
///
/// One vocabulary used in two roles: on an overlay item, and on a
/// [`RetainedOverlay`] standing for a whole compiled group. Both roles hold the
/// same struct, and the two compose.
///
/// # The map
///
/// A transform is one 2D affine map, written as a rotation about `pivot` of a
/// stretch and shear in the item's own frame, then a translation:
///
/// ```text
/// T(p) = pivot + R(rotation) . S . (p - pivot) + translate
/// S    = scale . [ stretch.x   skew      ]
///                [ 0           stretch.y ]
/// ```
///
/// So `stretch` scales each axis of the item before it turns, `skew` shears x
/// by y in that same frame (`x' = x + skew * y`), and `scale` is the uniform
/// factor on top of both. The common case sets `scale` and `rotation` only;
/// `stretch` and `skew` stay at their identity values. Any 2D linear map can
/// be written this way, which is what [`Self::from_linear`] does.
///
/// # Composition
///
/// The final position of a vertex is
///
/// ```text
/// group_T . item_T . v
/// ```
///
/// two nested affine maps, applied outside-in. Under it, rotation angles add
/// and uniform scales multiply when neither level stretches or shears, but
/// those are *consequences*, not the rule. Translation in particular does
/// **not** compose additively: an item at `translate = [10, 0]` inside a group
/// rotated a quarter turn is displaced along the group's rotated axis, not
/// along screen x. Composing the channels one at a time instead makes a
/// rotated scroll panel shear its own contents apart. [`Self::compose`] does
/// it exactly, for any pair.
///
/// # What does not live here
///
/// `opacity`, `tint`, `z_order`, `anchoring` and `clip`
/// stay flat on the item and on the group, because their meaning is
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
    /// Centre of rotation, scaling and shear, in logical pixels from the
    /// item's or group's own centre. `[0, 0]` turns about the centre.
    pub pivot: [f32; 2],
    /// Uniform scale about `pivot`, applied with `stretch` before `rotation`.
    /// `1.0` is identity.
    pub scale: f32,
    /// Per-axis scale in the item's own frame, multiplied with `scale` and
    /// applied before `rotation`. `[1, 1]` is identity; a negative component
    /// mirrors that axis.
    #[cfg_attr(feature = "serde", serde(default = "unit_stretch"))]
    pub stretch: [f32; 2],
    /// Shear of x by y in the item's own frame, applied with `stretch` before
    /// `rotation`: `x' = x + skew * y`. `0.0` is identity.
    #[cfg_attr(feature = "serde", serde(default))]
    pub skew: f32,
}

#[cfg(feature = "serde")]
fn unit_stretch() -> [f32; 2] {
    [1.0, 1.0]
}

impl Default for OverlayTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl OverlayTransform {
    /// No translation, no rotation, unit scale, no stretch or skew, pivot at the
    /// centre.
    pub const IDENTITY: Self = Self {
        translate: [0.0, 0.0],
        rotation: 0.0,
        pivot: [0.0, 0.0],
        scale: 1.0,
        stretch: [1.0, 1.0],
        skew: 0.0,
    };

    /// Identity except for `translate`.
    pub fn at(translate: [f32; 2]) -> Self {
        Self {
            translate,
            ..Self::IDENTITY
        }
    }

    /// The transform whose linear part is `m` (row-major: `x' = m[0][0] x +
    /// m[0][1] y`, `y' = m[1][0] x + m[1][1] y`), with no translation and the
    /// pivot at the centre. Any 2D linear map decomposes into a rotation of a
    /// stretch and shear, so this is exact; `scale` comes back as `1.0` with
    /// the magnitudes in `stretch`.
    ///
    /// A map that collapses x (its first column is zero) has no rotation to
    /// read; it comes back unrotated with `stretch.x` zero, which draws nothing
    /// along that axis just as the map does.
    pub fn from_linear(m: [[f32; 2]; 2]) -> Self {
        // Gram-Schmidt on the columns: the first column fixes the rotation,
        // the second gives the shear (its component along the first) and the
        // y stretch (its component along the perpendicular, signed so a
        // reflection survives).
        let c0 = [m[0][0], m[1][0]];
        let c1 = [m[0][1], m[1][1]];
        let len0 = (c0[0] * c0[0] + c0[1] * c0[1]).sqrt();
        if !(len0 > f32::EPSILON) {
            return Self {
                stretch: [0.0, m[1][1]],
                ..Self::IDENTITY
            };
        }
        let q0 = [c0[0] / len0, c0[1] / len0];
        let q1 = [-q0[1], q0[0]];
        let skew = q0[0] * c1[0] + q0[1] * c1[1];
        let sy = q1[0] * c1[0] + q1[1] * c1[1];
        Self {
            rotation: q0[1].atan2(q0[0]),
            stretch: [len0, sy],
            skew,
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

    /// Set the per-axis stretch applied in the item's own frame.
    pub fn with_stretch(mut self, stretch: [f32; 2]) -> Self {
        self.stretch = stretch;
        self
    }

    /// Set the shear of x by y in the item's own frame.
    pub fn with_skew(mut self, skew: f32) -> Self {
        self.skew = skew;
        self
    }

    /// Replace the rotation, scale, stretch and skew with the decomposition of
    /// `m` (see [`Self::from_linear`]), keeping `translate` and `pivot`.
    pub fn with_linear(mut self, m: [[f32; 2]; 2]) -> Self {
        let lin = Self::from_linear(m);
        self.rotation = lin.rotation;
        self.scale = lin.scale;
        self.stretch = lin.stretch;
        self.skew = lin.skew;
        self
    }

    /// Whether this is the identity map, so a caller can skip the work.
    pub fn is_identity(&self) -> bool {
        self.translate == [0.0, 0.0] && self.pivot == [0.0, 0.0] && self.linear_is_identity()
    }

    /// Whether the rotation, scale, stretch and skew together leave points
    /// where they are, so only `translate` moves anything.
    pub fn linear_is_identity(&self) -> bool {
        self.rotation == 0.0 && self.scale == 1.0 && self.stretch == [1.0, 1.0] && self.skew == 0.0
    }

    /// The linear part of the map, row-major: `x' = m[0][0] x + m[0][1] y`,
    /// `y' = m[1][0] x + m[1][1] y`. This is `R(rotation) . S` from the type's
    /// description, with the pivot and translation left out.
    pub fn linear(&self) -> [[f32; 2]; 2] {
        let (s, c) = self.rotation.sin_cos();
        let sx = self.scale * self.stretch[0];
        let sy = self.scale * self.stretch[1];
        let k = self.scale * self.skew;
        // R . S with S = [[sx, k], [0, sy]].
        [[c * sx, c * k - s * sy], [s * sx, s * k + c * sy]]
    }

    /// The inverse of [`Self::linear`], or `None` when the map collapses an
    /// axis and has no inverse.
    pub fn inverse_linear(&self) -> Option<[[f32; 2]; 2]> {
        let m = self.linear();
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if !(det.is_finite() && det.abs() > f32::EPSILON * f32::EPSILON) {
            return None;
        }
        let inv = 1.0 / det;
        Some([
            [m[1][1] * inv, -m[0][1] * inv],
            [-m[1][0] * inv, m[0][0] * inv],
        ])
    }

    /// The factor by which the linear part scales area, square-rooted: the
    /// length a unit distance in the item's frame covers on screen, averaged
    /// over directions. Equal to `scale` when nothing stretches or shears.
    pub fn uniform_scale_factor(&self) -> f32 {
        let m = self.linear();
        (m[0][0] * m[1][1] - m[0][1] * m[1][0]).abs().sqrt()
    }

    /// Apply this transform to a point given relative to the transform's own
    /// centre, returning the point in the same space.
    ///
    /// This is one level of `group_T . item_T . v`: `scale`, `stretch` and
    /// `skew` first about `pivot`, then `rotation` about it, then `translate`.
    pub fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let m = self.linear();
        let dx = p[0] - self.pivot[0];
        let dy = p[1] - self.pivot[1];
        [
            self.pivot[0] + m[0][0] * dx + m[0][1] * dy + self.translate[0],
            self.pivot[1] + m[1][0] * dx + m[1][1] * dy + self.translate[1],
        ]
    }

    /// Compose two transforms into one, so that
    /// `outer.compose(inner).apply(v) == outer.apply(inner.apply(v))`.
    ///
    /// Exact for any pair: the two linear parts multiply and the product is
    /// decomposed again with [`Self::from_linear`]. The composed pivot is the
    /// centre: both pivots have been folded into the composed translation.
    pub fn compose(&self, inner: &Self) -> Self {
        let a = self.linear();
        let b = inner.linear();
        let m = [
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ];
        let mut out = Self::from_linear(m);
        // Solve for the translation that puts the composed origin where the
        // two-step application puts it.
        let origin = self.apply(inner.apply([0.0, 0.0]));
        let moved = out.apply([0.0, 0.0]);
        out.translate = [origin[0] - moved[0], origin[1] - moved[1]];
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
            ..OverlayTransform::IDENTITY
        };
        let item = OverlayTransform {
            translate: [10.0, 0.0],
            rotation: 0.25,
            pivot: [5.0, -5.0],
            scale: 0.5,
            ..OverlayTransform::IDENTITY
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

    /// Composition stays exact when both levels stretch and shear, which is
    /// where a scalar-only rule would have broken.
    #[test]
    fn composition_is_exact_under_stretch_and_skew() {
        let group = OverlayTransform::IDENTITY
            .with_translate([30.0, -8.0])
            .with_rotation(0.9)
            .with_stretch([2.0, 0.5])
            .with_pivot([4.0, 4.0]);
        let item = OverlayTransform::IDENTITY
            .with_rotation(-0.4)
            .with_skew(0.7)
            .with_stretch([1.0, -1.5])
            .with_scale(1.2)
            .with_translate([3.0, 9.0]);
        for v in [[0.0, 0.0], [7.0, 3.0], [-12.0, 20.0], [100.0, -40.0]] {
            let stepwise = group.apply(item.apply(v));
            let composed = group.compose(&item).apply(v);
            assert!(
                close(stepwise, composed),
                "{stepwise:?} != {composed:?} for {v:?}"
            );
        }
    }

    #[test]
    fn identity_composes_to_nothing() {
        let t = OverlayTransform {
            translate: [3.0, -4.0],
            rotation: 0.3,
            pivot: [1.0, 2.0],
            scale: 1.5,
            ..OverlayTransform::IDENTITY
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

    /// `from_linear` round-trips any invertible matrix, reflections included.
    #[test]
    fn linear_round_trips() {
        for m in [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, -1.0], [1.0, 0.0]],
            [[2.0, 0.5], [-0.3, 1.5]],
            [[-1.0, 0.0], [0.0, 1.0]],
            [[0.7, 0.7], [0.7, -0.7]],
        ] {
            let t = OverlayTransform::from_linear(m);
            let back = t.linear();
            for r in 0..2 {
                for c in 0..2 {
                    assert!(
                        (back[r][c] - m[r][c]).abs() < 1e-4,
                        "{m:?} -> {t:?} -> {back:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_undoes_the_linear_part() {
        let t = OverlayTransform::IDENTITY
            .with_rotation(0.6)
            .with_stretch([1.5, 0.75])
            .with_skew(0.3)
            .with_scale(2.0);
        let m = t.linear();
        let inv = t.inverse_linear().unwrap();
        for v in [[1.0, 0.0], [0.0, 1.0], [3.0, -2.0]] {
            let f = [
                m[0][0] * v[0] + m[0][1] * v[1],
                m[1][0] * v[0] + m[1][1] * v[1],
            ];
            let b = [
                inv[0][0] * f[0] + inv[0][1] * f[1],
                inv[1][0] * f[0] + inv[1][1] * f[1],
            ];
            assert!(close(b, v));
        }
        assert!(
            OverlayTransform::IDENTITY
                .with_stretch([0.0, 1.0])
                .inverse_linear()
                .is_none()
        );
    }
}
