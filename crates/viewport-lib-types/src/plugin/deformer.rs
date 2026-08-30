//! Deformer descriptor vocabulary.
//!
//! The pure data a consumer supplies to define a per-vertex deformer. The
//! registration that consumes it (shader composition + validation, pipeline
//! rebuild) stays in `viewport-lib`; this is just the description.

/// Where in the deformation chain a deformer executes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeformStage {
    /// Before the model transform. Positions are object space.
    ObjectSpace,
    /// After the model transform. Positions are world space.
    WorldSpace,
}

/// Description of a deformer to register against the mesh shader family.
///
/// `wgsl_body` defines:
///
/// ```text
/// fn <name>_deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex { ... }
/// ```
///
/// The composer prefixes every top-level identifier it finds inside
/// `wgsl_body` with `<name>__` to keep multiple deformers independent. The
/// body may read `deform_data` / `deform_instance_data` via the
/// `deform_read_*` helpers in `deform.wgsl`, and read its slot's parameter
/// region via `deform_header.slot_params[ctx.slot * 4 + k]`.
///
/// # Hook contract
///
/// The shipped `deform.wgsl` declares:
///
/// ```text
/// struct DeformVertex {
///     position: vec3<f32>,
///     normal: vec3<f32>,
///     vertex_index: u32,
/// };
///
/// struct DeformContext {
///     model: mat4x4<f32>,
///     object_origin: vec3<f32>,
///     time_seconds: f32,
///     flags: u32,
///     slot: u32,
/// };
/// ```
///
/// These signatures are stable: additive changes only. Existing fields will
/// not be renamed, retyped, or removed. New fields may be appended; bodies
/// compiled against an older view continue to work because the WGSL struct
/// init they don't touch is filled by the caller.
///
/// `position` is object-space on entry to an `ObjectSpace` body and
/// world-space on entry to a `WorldSpace` body. `vertex_index` is the
/// `@builtin(vertex_index)` from the vertex stage. `ctx.slot` is the slot
/// the registry assigned to this deformer; use it to address `deform_data`
/// and `deform_instance_data` rather than baking in a literal.
///
/// # Composition order
///
/// Deformers run in two stages: every [`DeformStage::ObjectSpace`] body
/// first, then every [`DeformStage::WorldSpace`] body. Within a stage,
/// bodies execute in ascending `priority` (lower runs first); ties break by
/// `name`. The in-crate skinning deformer registers at
/// `DEFORM_PRIORITY_SKINNING = -1000` so morph-class deformers registered
/// at the default priority of `0` run after it. This ordering is part of
/// the contract: a body that wants to run before skinning must register
/// with a priority strictly less than `-1000`.
///
/// # Pass scope
///
/// Every registered deformer runs in every mesh-family pass: the main solid
/// and transparent passes, the OIT accumulate pass, the instanced and
/// instanced-OIT pipelines, the shadow pass, and the outline-mask pass. A
/// deformer cannot opt out of a single pass; the composed pipeline is
/// shared across them.
///
/// In particular: shadows always deform. A skinned or wind-swept mesh
/// casts a shadow that matches its deformed silhouette. The outline mask
/// follows the deformed silhouette for the same reason. GPU picking, which
/// uses the same mesh family, picks the deformed mesh.
///
/// Bodies that have nothing to do for a draw (no slot data attached,
/// flag bit clear) are gated off by the per-slot flag branch the composer
/// wraps the call in, so the cost on undeformed draws is the branch
/// predicate, not the body.
#[derive(Clone, Debug)]
pub struct DeformerDesc {
    /// Deformer name. Must be a valid WGSL identifier and unique across the
    /// registry. The composer uses it as the prefix for every top-level
    /// declaration spliced from `wgsl_body`.
    pub name: &'static str,
    /// Which side of the model transform the body runs on.
    pub stage: DeformStage,
    /// Execution order within the stage; lower runs first. Ties break by name.
    pub priority: i32,
    /// WGSL body defining `fn <name>_deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex`
    /// plus any helper declarations. All top-level identifiers in this body
    /// are prefixed with `<name>__` at composition time.
    pub wgsl_body: String,
    /// Bytes per vertex in the slot storage buffer (validated on attach).
    pub per_vertex_stride: u32,
}
