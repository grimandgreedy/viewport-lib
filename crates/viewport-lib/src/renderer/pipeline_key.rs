//! Shared mesh-pipeline variant selection.
//!
//! `PipelineKey` names the axes a mesh draw pipeline varies on (facedness,
//! alpha-cutout, and the opaque scene pass's discard-free early-Z fast path).
//! The `select_*` functions below are the single place each pass family maps a
//! key onto its existing named pipeline fields, replacing the `match
//! (no_discard, two_sided)` / `if is_two_sided()` sites that used to be
//! hand-duplicated per call site in `hdr_path.rs`, `ldr_path.rs`,
//! `per_object.rs`, and `shadow_pass.rs`.
//!
//! This does not change pipeline construction: a family that has not built a
//! variant the key names (the per-object shadow pass has no cutout variant;
//! material-plugin opaque pipelines have no discard-free twin) falls back to
//! the nearest pipeline it does have and reports the gap through
//! `missing_variant`, so it shows up in `FrameStats` instead of silently
//! drawing the wrong thing. `missing_variant` is a *runtime* signal on
//! families still using named `Option<Pipeline>` fields; it stays load-bearing
//! until every family migrates to `PipelineVariantSet` below, at which point
//! it becomes unreachable and can be deleted (phase 4 of the plan).
//!
//! A family that *has* migrated to `PipelineVariantSet` gets a stronger,
//! compile-time form of the same guarantee: `PipelineVariantSet::build` takes
//! a closure returning a concrete `RenderPipeline`, not an `Option`, so there
//! is no code path where a key goes unbuilt. No startup assertion is needed
//! for those families -- the type system already rules out a missing variant.
//! What it does *not* rule out is a `build` closure that quietly maps two
//! keys that should differ onto the same pipeline (an axis it forgot to
//! branch on); that is a correctness bug, not a completeness one, and is
//! still the job of the pixel-comparison tests in
//! `tests/headless_pipeline_variant_matrix.rs`.

/// The axes that vary a mesh-family render pipeline. Computed once per item
/// (or per instanced batch) and passed to a `select_*` function to pick the
/// concrete pipeline. Building pipelines from this key eagerly, for every
/// reachable combination, is phase 2 of the pipeline-variant-specialization
/// plan; this type only unifies selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub(crate) struct PipelineKey {
    /// `BackfacePolicy::Cull` (false) vs any two-sided policy (true; the
    /// pipeline uses `cull_mode: None` and, for the styled policies, shades
    /// the flipped-normal back face -- a shading detail the pipeline choice
    /// itself does not distinguish).
    pub two_sided: bool,
    /// `AlphaMode::Mask`: the fragment stage must sample and cutoff-test
    /// alpha. Not every family needs a distinct pipeline for this (the opaque
    /// scene pass branches on a per-object uniform instead), so a `select_*`
    /// that does not take cutout pipelines simply ignores this field.
    pub cutout: bool,
    /// Eligible for the discard-free early-Z fast path: opaque alpha, no
    /// material plugin, no scalar/NaN-discard attribute, no per-submesh
    /// materials. Frame-level conditions that gate the fast path everywhere
    /// (active clip geometry, the `force_po_discard` debug knob) are folded
    /// into this by the caller before the key is built, since they are not a
    /// property of the item itself.
    pub no_discard_eligible: bool,
}

impl PipelineKey {
    /// A key carrying only facedness, for families with no cutout or
    /// no-discard axis (OIT, the LDR opaque/transparent draws, per-submesh
    /// material ranges).
    pub fn two_sided(two_sided: bool) -> Self {
        Self {
            two_sided,
            ..Self::default()
        }
    }

    /// Every axis combination, for eager cross-product construction
    /// (`PipelineVariantSet::build`). A pass that does not vary on every axis
    /// still sees all 8 during construction; its build closure just returns
    /// the same pipeline for the axis it ignores.
    pub fn all() -> impl Iterator<Item = PipelineKey> {
        (0u8..8).map(|bits| PipelineKey {
            two_sided: bits & 1 != 0,
            cutout: bits & 2 != 0,
            no_discard_eligible: bits & 4 != 0,
        })
    }

    /// Dense index in `0..8`, stable across calls, for the hash-free array
    /// lookup `PipelineVariantSet` uses.
    fn slot(self) -> usize {
        (self.two_sided as usize)
            | (self.cutout as usize) << 1
            | (self.no_discard_eligible as usize) << 2
    }
}

/// A pipeline built for every reachable [`PipelineKey`], indexed for a
/// hash-free draw-time lookup (`get`). Construction is eager: `build` runs
/// once per key up front (typically from the same lazy first-HDR-use or
/// deform-registration-changed trigger a family already rebuilds from), not
/// per draw call.
///
/// `RenderPipeline` is a cheap reference-counted GPU handle, so a `build`
/// closure that ignores an axis (e.g. OIT ignoring cutout, or shadow ignoring
/// no-discard) can just return a clone of the same pipeline for both of that
/// axis's values -- `PipelineVariantSet` does not force compiling 8 distinct
/// GPU pipelines when a pass only has 2 or 4 real variants.
pub(crate) struct PipelineVariantSet {
    variants: [crate::gpu::RenderPipeline; 8],
}

impl PipelineVariantSet {
    pub fn build(mut build: impl FnMut(PipelineKey) -> crate::gpu::RenderPipeline) -> Self {
        let mut variants: Vec<crate::gpu::RenderPipeline> = Vec::with_capacity(8);
        for key in PipelineKey::all() {
            variants.push(build(key));
        }
        Self {
            variants: variants
                .try_into()
                .unwrap_or_else(|_| unreachable!("PipelineKey::all() yields exactly 8 keys")),
        }
    }

    pub fn get(&self, key: PipelineKey) -> &crate::gpu::RenderPipeline {
        &self.variants[key.slot()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `PipelineVariantSet::build`'s closure returns a concrete
    /// `RenderPipeline`, not an `Option`, so a migrated family cannot skip a
    /// key -- completeness is a compile-time property, not something a
    /// startup check needs to verify. What *can* still silently break is the
    /// enumeration itself: if a future phase adds a fourth axis to
    /// `PipelineKey` without widening `all()`/`slot()` past 3 bits, two
    /// distinct keys would collide on the same array slot and `build` would
    /// silently drop one of them. This test pins `all()` and `slot()` in
    /// sync so that class of bug fails immediately instead of showing up as
    /// a wrong pipeline at draw time.
    #[test]
    fn all_keys_are_distinct_and_densely_slotted() {
        let keys: Vec<PipelineKey> = PipelineKey::all().collect();
        assert_eq!(keys.len(), 8, "PipelineKey has 3 bool axes: 2^3 = 8 keys");

        let mut seen_keys = std::collections::HashSet::new();
        let mut seen_slots = std::collections::HashSet::new();
        for key in keys {
            assert!(
                seen_keys.insert(key),
                "all() yielded {key:?} more than once"
            );
            let slot = key.slot();
            assert!(slot < 8, "{key:?} slotted out of range: {slot}");
            assert!(
                seen_slots.insert(slot),
                "{key:?} collided with another key at slot {slot}"
            );
        }
    }
}

/// Two-way facedness select, shared by every family whose only axis is
/// `Cull` vs a two-sided policy: OIT, the LDR opaque and inline-transparent
/// draws, per-submesh material ranges, and shadow casting once cutout is not
/// in play.
pub(crate) fn select_two_sided<'p>(
    key: PipelineKey,
    one_sided: &'p crate::gpu::RenderPipeline,
    two_sided: &'p crate::gpu::RenderPipeline,
) -> &'p crate::gpu::RenderPipeline {
    if key.two_sided { two_sided } else { one_sided }
}

/// The opaque scene pass's four-way select: facedness x the discard-free
/// early-Z twin. `nodiscard` / `nodiscard_two_sided` are `None` when the twin
/// was not built for this pass (a legitimate capability fallback -- some
/// backends skip it under storage-buffer pressure -- not a variant gap, so
/// this never touches `missing_variant`).
pub(crate) fn select_opaque_solid<'p>(
    key: PipelineKey,
    solid: &'p crate::gpu::RenderPipeline,
    solid_two_sided: &'p crate::gpu::RenderPipeline,
    nodiscard: Option<&'p crate::gpu::RenderPipeline>,
    nodiscard_two_sided: Option<&'p crate::gpu::RenderPipeline>,
) -> &'p crate::gpu::RenderPipeline {
    if key.no_discard_eligible {
        if let (Some(nd), Some(nd_two_sided)) = (nodiscard, nodiscard_two_sided) {
            return if key.two_sided { nd_two_sided } else { nd };
        }
    }
    select_two_sided(key, solid, solid_two_sided)
}
