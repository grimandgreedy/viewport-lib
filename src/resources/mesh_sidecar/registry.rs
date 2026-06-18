//! Deformer registry: composes registered `wgsl_body` strings into the
//! mesh-family shader sources and validates the result before the renderer
//! swaps pipelines.
//!
//! At registration time the descriptor's body is identifier-prefixed by the
//! deformer name, spliced above the shader's entry points, and called from
//! the appropriate stage hook (`viewport_deform_object_space` or
//! `viewport_deform_world_space`) inside a per-slot flag branch. Each
//! affected shader is validated by creating a throwaway shader module under
//! an error scope; failures roll back the registration with the naga
//! message returned to the caller.

use crate::error::{ViewportError, ViewportResult};
use crate::renderer::shader_hashes::SHADERS;

/// Where in the deformation chain a deformer executes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeformStage {
    /// Before the model transform. Positions are object space.
    ObjectSpace,
    /// After the model transform. Positions are world space.
    WorldSpace,
}

/// Hardware-side maximum slot count, mirrored from
/// [`crate::resources::mesh_sidecar::deform::DEFORM_SLOT_COUNT`] so the
/// registry can validate against it without pulling the sidecar module into
/// the public API.
#[allow(dead_code)]
pub(crate) const REGISTRY_SLOT_COUNT: usize =
    crate::resources::mesh_sidecar::deform::DEFORM_SLOT_COUNT;

/// First slot index reserved for internal in-crate deformers. Slots in
/// `[0, DEFORM_SLOT_COUNT_PUB)` are allocated to host registrations through
/// `register_deformer`; slots in `[DEFORM_SLOT_COUNT_PUB, REGISTRY_SLOT_COUNT)`
/// are reserved for shipped-in-crate deformers (skinning, displacement) that
/// register at renderer construction.
pub(crate) const REGISTRY_INTERNAL_SLOT_START: usize =
    crate::resources::mesh_sidecar::deform::DEFORM_INTERNAL_SLOT_START;

/// Maximum number of deformer slots the registry exposes to hosts.
pub const DEFORM_SLOT_COUNT_PUB: usize = REGISTRY_INTERNAL_SLOT_START;

/// Number of `vec4<f32>` parameter words per slot in the shared deformation
/// header uniform.
pub const DEFORM_PARAMS_PER_SLOT_PUB: usize =
    crate::resources::mesh_sidecar::deform::DEFORM_PARAMS_PER_SLOT;

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

/// Handle to a registered deformer. The inner index is the slot the registry
/// assigned; only used internally to address the storage buffer and flag bit.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct DeformerId(pub(crate) usize);

impl DeformerId {
    /// Returns the slot index the registry assigned at registration.
    pub fn slot(&self) -> usize {
        self.0
    }
}

/// Stored registration: keeps the descriptor (for re-composition on device
/// reset) plus the slot index that owns this deformer.
#[derive(Clone, Debug)]
pub(crate) struct StoredDeformer {
    pub desc: DeformerDesc,
    pub slot: usize,
}

/// Mesh-family base shaders that carry the deformation hook contract.
pub(crate) const MESH_FAMILY_SHADERS: &[&str] = &[
    "mesh.wgsl",
    "mesh_oit.wgsl",
    "mesh_instanced.wgsl",
    "mesh_instanced_oit.wgsl",
    "shadow.wgsl",
    "outline_mask.wgsl",
];

fn shader_source(name: &str) -> Option<&'static str> {
    SHADERS.iter().find(|s| s.name == name).map(|s| s.source)
}

/// Build the slot call sequence to splice into a stage marker region.
///
/// Each registered deformer at `stage` is emitted as:
///
/// ```text
/// if ((ctx.flags & (1u << SLOT)) != 0u) {
///     out = <name>__deform(out, ctx);
/// }
/// ```
fn build_stage_calls(stored: &[StoredDeformer], stage: DeformStage) -> String {
    let mut entries: Vec<&StoredDeformer> =
        stored.iter().filter(|d| d.desc.stage == stage).collect();
    entries.sort_by(|a, b| {
        a.desc
            .priority
            .cmp(&b.desc.priority)
            .then_with(|| a.desc.name.cmp(b.desc.name))
    });
    let mut out = String::new();
    for d in entries {
        out.push_str(&format!(
            "    if ((ctx.flags & (1u << {slot}u)) != 0u) {{\n        var ctx_{name} = ctx;\n        ctx_{name}.slot = {slot}u;\n        out = {name}__deform(out, ctx_{name});\n    }}\n",
            slot = d.slot,
            name = d.desc.name,
        ));
    }
    out
}

/// Rewrite the slot marker region for one stage with the provided call block.
fn rewrite_marker(source: &str, marker: &str, body: &str) -> Option<String> {
    let open = format!("// <viewport-deform-slots:{marker}>");
    let close = format!("// </viewport-deform-slots:{marker}>");
    let open_idx = source.find(&open)?;
    let close_idx = source.find(&close)?;
    if close_idx < open_idx {
        return None;
    }
    let mut out = String::with_capacity(source.len() + body.len());
    out.push_str(&source[..open_idx + open.len()]);
    out.push('\n');
    out.push_str(body);
    out.push_str(&source[close_idx..]);
    Some(out)
}

/// Prefix every top-level `fn`, `struct`, `var`, `let`, `const`, and `alias`
/// identifier appearing in `body` with `<name>__`, mapping references inside
/// the same body so they stay self-consistent. Simple regex-free walker:
/// finds `fn <ident>(`, `struct <ident>`, `let <ident>`, `const <ident>`
/// declarations and rewrites uses of the same identifier within the body.
fn identifier_prefix(name: &str, body: &str) -> String {
    let mut decls: Vec<String> = Vec::new();
    for tok in ["fn ", "struct ", "alias ", "const ", "var "] {
        let mut search_from = 0;
        while let Some(rel) = body[search_from..].find(tok) {
            let i = search_from + rel + tok.len();
            let rest = &body[i..];
            let end = rest
                .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            if end > 0 {
                let ident = rest[..end].to_string();
                if !ident.is_empty() && !decls.contains(&ident) {
                    decls.push(ident);
                }
            }
            search_from = i + end;
        }
    }
    // Replace whole-word occurrences of each declared identifier with the
    // prefixed form. Process longest-first so substring matches don't shadow.
    decls.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let mut out = body.to_string();
    for ident in &decls {
        let prefixed = format!("{name}__{ident}");
        out = replace_whole_word(&out, ident, &prefixed);
    }
    out
}

fn replace_whole_word(haystack: &str, needle: &str, replacement: &str) -> String {
    if needle.is_empty() {
        return haystack.to_string();
    }
    let bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let mut out = String::with_capacity(haystack.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + needle_bytes.len() <= bytes.len()
            && &bytes[i..i + needle_bytes.len()] == needle_bytes
        {
            let before_ok = i == 0 || {
                let c = bytes[i - 1] as char;
                !(c.is_alphanumeric() || c == '_')
            };
            let after_idx = i + needle_bytes.len();
            let after_ok = after_idx == bytes.len() || {
                let c = bytes[after_idx] as char;
                !(c.is_alphanumeric() || c == '_')
            };
            if before_ok && after_ok {
                out.push_str(replacement);
                i = after_idx;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Compose one shader's preprocessed source with the registered deformers.
/// Splices every registered body (identifier-prefixed) into the shader after
/// the `deform.wgsl` include, then rewrites the two stage markers to call
/// each body inside a flag-bit branch.
pub(crate) fn compose_shader(base: &str, stored: &[StoredDeformer]) -> String {
    let mut bodies = String::new();
    for d in stored {
        bodies.push_str(&format!(
            "\n// deformer: {name} (slot {slot}, stage {stage:?})\n",
            name = d.desc.name,
            slot = d.slot,
            stage = d.desc.stage,
        ));
        bodies.push_str(&identifier_prefix(d.desc.name, &d.desc.wgsl_body));
        bodies.push('\n');
    }

    // Splice bodies in just before the first viewport_deform_object_space fn
    // definition (which is inside the deform.wgsl include).
    let anchor = "fn viewport_deform_object_space(";
    let with_bodies = match base.find(anchor) {
        Some(idx) => {
            let mut s = String::with_capacity(base.len() + bodies.len());
            s.push_str(&base[..idx]);
            s.push_str(&bodies);
            s.push_str(&base[idx..]);
            s
        }
        None => return base.to_string(),
    };

    let object_calls = build_stage_calls(stored, DeformStage::ObjectSpace);
    let world_calls = build_stage_calls(stored, DeformStage::WorldSpace);

    let after_obj = rewrite_marker(&with_bodies, "object", &object_calls).unwrap_or(with_bodies);
    rewrite_marker(&after_obj, "world", &world_calls).unwrap_or(after_obj)
}

/// Assign a slot in the host range `[0, DEFORM_SLOT_COUNT_PUB)` to a new
/// descriptor. Returns the lowest unused host slot or `DeformSlotsExhausted`
/// when the host range is full. Internal in-crate deformers use
/// [`allocate_internal_slot`] instead.
pub(crate) fn allocate_slot(stored: &[StoredDeformer]) -> ViewportResult<usize> {
    let mut used = 0u32;
    for d in stored {
        used |= 1u32 << d.slot;
    }
    for slot in 0..REGISTRY_INTERNAL_SLOT_START {
        if used & (1u32 << slot) == 0 {
            return Ok(slot);
        }
    }
    Err(ViewportError::DeformSlotsExhausted {
        max: REGISTRY_INTERNAL_SLOT_START,
    })
}

/// Assign a slot in the internal range
/// `[DEFORM_SLOT_COUNT_PUB, REGISTRY_SLOT_COUNT)`. Used at renderer
/// construction to register the in-crate deformers (skinning, displacement)
/// without consuming a host slot.
#[allow(dead_code)]
pub(crate) fn allocate_internal_slot(stored: &[StoredDeformer]) -> ViewportResult<usize> {
    let mut used = 0u32;
    for d in stored {
        used |= 1u32 << d.slot;
    }
    for slot in REGISTRY_INTERNAL_SLOT_START..REGISTRY_SLOT_COUNT {
        if used & (1u32 << slot) == 0 {
            return Ok(slot);
        }
    }
    Err(ViewportError::DeformSlotsExhausted {
        max: REGISTRY_SLOT_COUNT - REGISTRY_INTERNAL_SLOT_START,
    })
}

/// Validate `name` is a fresh, non-empty WGSL-ish identifier.
pub(crate) fn validate_name(stored: &[StoredDeformer], name: &str) -> ViewportResult<()> {
    if name.is_empty()
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(true)
    {
        return Err(ViewportError::DeformShaderInvalid {
            reason: "deformer name must be a non-empty WGSL identifier (letters, digits, underscore; not starting with a digit)".to_string(),
        });
    }
    if stored.iter().any(|d| d.desc.name == name) {
        return Err(ViewportError::DeformNameTaken {
            name: name.to_string(),
        });
    }
    Ok(())
}

/// Run wgpu's shader-module path under an error scope. Returns `Ok(module)`
/// on success or a populated `DeformShaderInvalid` otherwise.
pub(crate) fn validate_with_wgpu(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> ViewportResult<wgpu::ShaderModule> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let captured = block_on_simple(device.pop_error_scope());
    if let Some(err) = captured {
        return Err(ViewportError::DeformShaderInvalid {
            reason: format!("{label}: {err}"),
        });
    }
    Ok(module)
}

/// Look up a preprocessed shader source by name. Returns `None` for unknown
/// shader names (callers treat this as an internal error).
pub(crate) fn lookup_source(name: &str) -> Option<&'static str> {
    shader_source(name)
}

/// Tiny sync executor that polls a future until it resolves. wgpu's
/// `pop_error_scope` resolves on the next driver poll, which the device
/// itself drives; spinning here is fine because validation completes
/// without going through the device's command queue.
fn block_on_simple<F: std::future::Future>(mut fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    // SAFETY: we own `fut` on the stack and never move it after this point.
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(name: &'static str, stage: DeformStage, priority: i32, body: &str) -> StoredDeformer {
        StoredDeformer {
            desc: DeformerDesc {
                name,
                stage,
                priority,
                wgsl_body: body.to_string(),
                per_vertex_stride: 4,
            },
            slot: 0,
        }
    }

    #[test]
    fn allocate_slot_walks_lowest_free_index() {
        let mut stored = Vec::<StoredDeformer>::new();
        assert_eq!(allocate_slot(&stored).unwrap(), 0);
        let mut d = desc("a", DeformStage::ObjectSpace, 0, "");
        d.slot = 0;
        stored.push(d);
        assert_eq!(allocate_slot(&stored).unwrap(), 1);
        let mut d = desc("b", DeformStage::ObjectSpace, 0, "");
        d.slot = 2;
        stored.push(d);
        assert_eq!(allocate_slot(&stored).unwrap(), 1);
    }

    #[test]
    fn allocate_slot_returns_exhausted_when_host_range_full() {
        let mut stored = Vec::<StoredDeformer>::new();
        for slot in 0..REGISTRY_INTERNAL_SLOT_START {
            let mut d = desc(
                Box::leak(format!("d{slot}").into_boxed_str()),
                DeformStage::ObjectSpace,
                0,
                "",
            );
            d.slot = slot;
            stored.push(d);
        }
        let err = allocate_slot(&stored).unwrap_err();
        assert!(matches!(
            err,
            ViewportError::DeformSlotsExhausted { max } if max == REGISTRY_INTERNAL_SLOT_START
        ));
    }

    #[test]
    fn allocate_internal_slot_uses_reserved_range() {
        let stored = Vec::<StoredDeformer>::new();
        let slot = allocate_internal_slot(&stored).unwrap();
        assert!(slot >= REGISTRY_INTERNAL_SLOT_START && slot < REGISTRY_SLOT_COUNT);
    }

    #[test]
    fn validate_name_rejects_duplicate() {
        let mut stored = Vec::<StoredDeformer>::new();
        stored.push(desc("wind", DeformStage::WorldSpace, 0, ""));
        let err = validate_name(&stored, "wind").unwrap_err();
        assert!(matches!(err, ViewportError::DeformNameTaken { .. }));
    }

    #[test]
    fn validate_name_rejects_invalid_identifier() {
        let stored = Vec::<StoredDeformer>::new();
        for bad in &["", "1wind", "wind!", " wind"] {
            assert!(matches!(
                validate_name(&stored, bad),
                Err(ViewportError::DeformShaderInvalid { .. })
            ));
        }
        assert!(validate_name(&stored, "wind_2").is_ok());
    }

    #[test]
    fn identifier_prefix_renames_locals() {
        let body = "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    return v;\n}\nfn helper() -> f32 { return 1.0; }\n";
        let out = identifier_prefix("wind", body);
        assert!(out.contains("fn wind__deform("));
        assert!(out.contains("fn wind__helper()"));
        // DeformVertex / DeformContext are external: must not get prefixed.
        assert!(out.contains("DeformVertex"));
        assert!(!out.contains("wind__DeformVertex"));
    }

    #[test]
    fn build_stage_calls_orders_by_priority_then_name() {
        let mut stored = vec![
            desc("z_late", DeformStage::ObjectSpace, 10, ""),
            desc("a_early", DeformStage::ObjectSpace, -5, ""),
            desc("m_mid", DeformStage::ObjectSpace, 0, ""),
            desc("world_only", DeformStage::WorldSpace, 0, ""),
        ];
        for (i, d) in stored.iter_mut().enumerate() {
            d.slot = i;
        }
        let obj = build_stage_calls(&stored, DeformStage::ObjectSpace);
        let pos_a = obj.find("a_early__deform").unwrap();
        let pos_m = obj.find("m_mid__deform").unwrap();
        let pos_z = obj.find("z_late__deform").unwrap();
        assert!(pos_a < pos_m);
        assert!(pos_m < pos_z);
        assert!(!obj.contains("world_only__deform"));
        let w = build_stage_calls(&stored, DeformStage::WorldSpace);
        assert!(w.contains("world_only__deform"));
    }

    #[test]
    fn compose_shader_zero_deformers_is_identity_passthrough() {
        let Some(src) = lookup_source("mesh.wgsl") else {
            // shader catalog not yet populated in test target; skip.
            return;
        };
        let composed = compose_shader(src, &[]);
        assert!(composed.contains("// <viewport-deform-slots:object>"));
        // No deformer calls injected.
        assert!(!composed.contains("__deform("));
    }

    #[test]
    fn compose_shader_splices_one_body_and_wires_call() {
        let Some(src) = lookup_source("mesh.wgsl") else {
            return;
        };
        let mut d = desc(
            "wind",
            DeformStage::WorldSpace,
            0,
            "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    return v;\n}\n",
        );
        d.slot = 2;
        let composed = compose_shader(src, &std::slice::from_ref(&d));
        assert!(composed.contains("fn wind__deform("));
        // World marker wired, object marker still empty.
        assert!(composed.contains("out = wind__deform(out, ctx_wind);"));
        assert!(composed.contains("ctx.flags & (1u << 2u)"));
        assert!(composed.contains("ctx_wind.slot = 2u;"));
        // Identifier of helper struct DeformVertex stays unprefixed.
        assert!(composed.contains("DeformVertex"));
    }
}
