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

/// Which optional shader features a mesh pipeline compiles in. Each bit is a
/// WGSL `override` constant in `mesh.wgsl` and `mesh_oit.wgsl`; a block under
/// a false constant is compiled out. Every block also tests its per-draw flag,
/// so the universal set (every bit on) draws any item, and a set with a bit
/// off draws the same image for every item that does not use the feature.
///
/// The set is a lazily built axis beside [`PipelineKey`]: the eager members of
/// a family are built universal, and a draw whose set is not built yet draws
/// the universal member while it compiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub(crate) struct FeatureSet(u16);

impl FeatureSet {
    /// The mesh's sidecar record: the override windows and every stream read.
    pub const RECORD: u16 = 1;
    /// The scalar or face-colour stream, the LUT and NaN colour, face colours.
    pub const COLOUR_STREAM: u16 = 2;
    /// The warp stream.
    pub const WARP: u16 = 4;
    /// The position and normal override buffers.
    pub const OVERRIDES: u16 = 8;
    /// Lightmap UVs and sampling, the shadowmask, the subtractive mode.
    pub const LIGHTMAP: u16 = 16;
    /// Matcap shading.
    pub const MATCAP: u16 = 32;
    /// The deformer hooks.
    pub const DEFORM: u16 = 64;
    /// The emissive map: its slot lookup and sample. A material without one
    /// adds its emissive colour and custom data only.
    pub const EMISSIVE_MAP: u16 = 128;
    /// The second UV set: the uv1 stream read and the per-slot choice between
    /// the two sets. Off, every slot samples uv0.
    pub const UV1: u16 = 1 << 8;
    /// The per-slot UV offset, scale and rotation. Off, every slot samples
    /// the mesh UV as it is.
    pub const UV_TRANSFORM: u16 = 1 << 9;

    /// Every feature compiled out.
    pub const NONE: Self = Self(0);
    /// Every feature compiled in: the eager pipelines.
    pub const UNIVERSAL: Self = {
        let mut all = 0;
        let mut i = 0;
        while i < Self::CONSTANTS.len() {
            all |= Self::CONSTANTS[i].0;
            i += 1;
        }
        Self(all)
    };

    /// The WGSL constant each bit sets.
    const CONSTANTS: &'static [(u16, &'static str)] = &[
        (Self::RECORD, "FEAT_RECORD"),
        (Self::COLOUR_STREAM, "FEAT_COLOUR_STREAM"),
        (Self::WARP, "FEAT_WARP"),
        (Self::OVERRIDES, "FEAT_OVERRIDES"),
        (Self::LIGHTMAP, "FEAT_LIGHTMAP"),
        (Self::MATCAP, "FEAT_MATCAP"),
        (Self::DEFORM, "FEAT_DEFORM"),
        (Self::EMISSIVE_MAP, "FEAT_EMISSIVE_MAP"),
        (Self::UV1, "FEAT_UV1"),
        (Self::UV_TRANSFORM, "FEAT_UV_TRANSFORM"),
    ];

    pub fn bits(self) -> u16 {
        self.0
    }

    pub fn has(self, bit: u16) -> bool {
        self.0 & bit != 0
    }

    pub fn with(self, bit: u16, on: bool) -> Self {
        if on { Self(self.0 | bit) } else { self }
    }

    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The features an instance record and its material block use. A batch's
    /// set is the union over its records.
    pub fn of_record(
        inst: &crate::resources::mesh::instancing::InstanceData,
        material_flags: [u32; 4],
    ) -> Self {
        use crate::resources::material_gpu::item_flags as m;
        use crate::resources::mesh::instancing::instance_flags as f;
        let flag = |bit: u32| inst.flags & bit != 0;
        Self::NONE
            .with(Self::RECORD, flag(f::HAS_SIDECAR))
            .with(
                Self::COLOUR_STREAM,
                flag(f::HAS_ATTRIBUTE) || flag(f::USE_FACE_COLOUR),
            )
            .with(Self::WARP, flag(f::HAS_WARP))
            .with(
                Self::OVERRIDES,
                flag(f::HAS_POSITION_OVERRIDE) || flag(f::HAS_NORMAL_OVERRIDE),
            )
            .with(
                Self::LIGHTMAP,
                inst.lightmap_stream != 0
                    || material_flags[1] != 0
                    || material_flags[0] & (m::LIGHTMAP_DIRECTIONAL | m::HAS_SHADOWMASK) != 0,
            )
            .with(Self::MATCAP, material_flags[0] & m::USE_MATCAP != 0)
            .with(Self::DEFORM, inst.deform_flags != 0)
            .with(
                Self::EMISSIVE_MAP,
                material_flags[0] & m::HAS_EMISSIVE_TEX != 0,
            )
            .with(Self::UV1, material_flags[0] & m::USES_UV1 != 0)
            .with(Self::UV_TRANSFORM, material_flags[0] & m::UV_TRANSFORM != 0)
    }

    /// The pipeline constants for this set: the features it leaves out set
    /// false, the rest left at their `true` default.
    pub fn constants(self) -> Vec<(&'static str, f64)> {
        Self::CONSTANTS
            .iter()
            .filter(|(bit, _)| !self.has(*bit))
            .map(|(_, name)| (*name, 0.0))
            .collect()
    }

    /// A label suffix naming the set, for the build log.
    pub fn label(self, base: &str) -> String {
        format!("{base}_feat{:02x}", self.0)
    }
}
