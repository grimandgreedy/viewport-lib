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
//! drawing the wrong thing.

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

/// Shadow-caster select: facedness x cutout. `cutout` / `cutout_two_sided`
/// are `None` for a family that has not built the alpha-cutout variant (today
/// only the instanced shadow path has); a cutout-material item drawn through
/// such a family falls back to the plain depth-only pipeline (casting a full
/// silhouette instead of a punched one) and bumps `*missing_variant`, the
/// bridge this phase adds so that gap shows up in `FrameStats` instead of
/// silently under-punching the shadow.
pub(crate) fn select_shadow_caster<'p>(
    key: PipelineKey,
    plain: &'p crate::gpu::RenderPipeline,
    plain_two_sided: &'p crate::gpu::RenderPipeline,
    cutout: Option<&'p crate::gpu::RenderPipeline>,
    cutout_two_sided: Option<&'p crate::gpu::RenderPipeline>,
    missing_variant: &mut u32,
) -> &'p crate::gpu::RenderPipeline {
    if key.cutout {
        if let (Some(c), Some(c_two_sided)) = (cutout, cutout_two_sided) {
            return if key.two_sided { c_two_sided } else { c };
        }
        *missing_variant += 1;
    }
    select_two_sided(key, plain, plain_two_sided)
}

/// Material-plugin opaque select: facedness only today. A plugin material
/// that would otherwise be eligible for the discard-free early-Z twin bumps
/// `*missing_variant` instead of getting one, since `MaterialPluginPipelines`
/// has no nodiscard field -- the second gap this phase's bridge surfaces.
pub(crate) fn select_plugin_opaque<'p>(
    key: PipelineKey,
    solid: &'p crate::gpu::RenderPipeline,
    solid_two_sided: &'p crate::gpu::RenderPipeline,
    missing_variant: &std::sync::atomic::AtomicU32,
) -> &'p crate::gpu::RenderPipeline {
    if key.no_discard_eligible {
        missing_variant.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    select_two_sided(key, solid, solid_two_sided)
}
