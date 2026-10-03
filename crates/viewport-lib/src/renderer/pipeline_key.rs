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
//! it becomes unreachable and can be deleted.
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
/// concrete pipeline. This type only unifies selection; building a pipeline
/// eagerly for every reachable combination is what `PipelineVariantSet` does
/// with it.
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

    /// Every axis combination, for a test that wants to resolve each key a
    /// family offers.
    #[cfg(test)]
    pub fn all() -> impl Iterator<Item = PipelineKey> {
        (0u8..8).map(|bits| PipelineKey {
            two_sided: bits & 1 != 0,
            cutout: bits & 2 != 0,
            no_discard_eligible: bits & 4 != 0,
        })
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
