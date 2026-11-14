//! Sprite and tile deformers: WGSL bodies a plugin registers once, composed
//! into every sprite and tilemap shader, that move each corner of the quads
//! of the batches and layers that opt in.
//!
//! A deformer is the sprite counterpart of a mesh deformer. Its body defines
//!
//! ```text
//! fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32>
//! ```
//!
//! and returns where the corner goes. Every registered deformer is composed
//! into every module and runs only where the batch's or layer's
//! [`SpriteDeformerSet`] holds it, so registering one rebuilds the sprite and
//! tilemap pipelines once and a batch that opts into none skips them all with
//! one branch. Deformers run in priority order, each seeing the corner the
//! last one moved, before a material's own `displace_sprite`.
//!
//! `slot` is the mesh deformer slot named by [`SpriteDeformerDesc::params`]:
//! the body reads that slot's parameters as
//! `deform_header.slot_params[slot * 4u + i]`, the block a mesh deformer's
//! `DeformSlotHandle` writes, so one write a frame drives a deformer's meshes,
//! sprites and tiles together.

use viewport_lib::DeformerId;
use viewport_lib::error::{ViewportError, ViewportResult};
use viewport_lib::plugin_api::wgsl_compose;

use super::effects::{SPRITE_CONTRACT, SPRITE_VERTEX_OF, TILE_VERTEX_OF, replace_once};
use super::types::{MAX_SPRITE_DEFORMERS, SpriteDeformerId};

/// A sprite and tile deformer to register with
/// [`SpriteDeformers::register_sprite_deformer`](super::SpriteDeformers::register_sprite_deformer).
#[derive(Clone, Debug)]
pub struct SpriteDeformerDesc {
    /// The deformer's name, a WGSL identifier: every top-level identifier in
    /// the body is prefixed with it.
    pub name: String,
    /// A body defining `fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32>`
    /// plus any helpers. `SpriteVertex` is the sprite vertex hook's contract.
    pub wgsl_body: String,
    /// Lower runs first. Deformers of equal priority run in registration
    /// order.
    pub priority: i32,
    /// The mesh deformer whose `slot_params` the body reads, or `None` for a
    /// body that reads none: `slot` is then `0xffffffff` and the body must not
    /// read `deform_header`.
    pub params: Option<DeformerId>,
}

impl SpriteDeformerDesc {
    /// A deformer named `name` with `wgsl_body`, at priority 0, reading no
    /// parameters.
    pub fn new(name: impl Into<String>, wgsl_body: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            wgsl_body: wgsl_body.into(),
            priority: 0,
            params: None,
        }
    }

    /// This deformer reading the parameters of mesh deformer `id`.
    pub fn with_params(mut self, id: DeformerId) -> Self {
        self.params = Some(id);
        self
    }

    /// This deformer at `priority`.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// A registered deformer as the sprite and tilemap types hold it.
#[derive(Clone, Debug)]
pub(crate) struct SpriteDeformerEntry {
    pub(crate) name: String,
    pub(crate) prefixed_body: String,
    pub(crate) priority: i32,
    /// The mesh slot whose parameters the body reads.
    pub(crate) slot: Option<u32>,
    /// The bit a [`SpriteDeformerSet`](super::SpriteDeformerSet) sets for it.
    pub(crate) bit: u32,
}

/// The marker in `sprite_corner` the deformer call replaces.
const SPRITE_MARKER: &str = "// <sprite-deform:vertex>";

/// The marker in `tile_vertex` the deformer call replaces.
const TILE_MARKER: &str = "// <tile-deform:vertex>";

/// The mesh deformers' shared header, which a body reads its parameters
/// from. Group 0 carries it wherever the mesh deformer registry exists.
const DEFORM_HEADER: &str = "
struct DeformHeader {
    time_seconds: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    slot_params: array<vec4<f32>, 32>,
};
@group(0) @binding(23) var<uniform> deform_header: DeformHeader;
";

/// One `if` per deformer, in priority order, each moving `v.world` when the
/// set in `mask` holds it.
fn calls(entries: &[SpriteDeformerEntry]) -> String {
    let mut order: Vec<&SpriteDeformerEntry> = entries.iter().collect();
    order.sort_by_key(|e| e.priority);
    let mut out = String::new();
    for e in order {
        out.push_str(&format!(
            "    if (mask & {bit}u) != 0u {{\n        v.world = {name}__deform_sprite(v, {slot}u);\n    }}\n",
            bit = 1u32 << e.bit,
            name = e.name,
            slot = e.slot.unwrap_or(u32::MAX),
        ));
    }
    out
}

/// The declarations and bodies both kinds of module take.
fn push_shared(out: &mut String, entries: &[SpriteDeformerEntry]) {
    if !out.contains("struct SpriteVertex {") {
        out.push('\n');
        out.push_str(SPRITE_CONTRACT);
    }
    if entries.iter().any(|e| e.slot.is_some()) && !out.contains("var<uniform> deform_header") {
        out.push_str(DEFORM_HEADER);
    }
    for e in entries {
        out.push('\n');
        out.push_str(&e.prefixed_body);
        out.push('\n');
    }
}

/// `source`, an assembled sprite module, with `entries` composed in. The
/// source as it is when there are none.
pub(crate) fn sprite_module(
    source: String,
    entries: &[SpriteDeformerEntry],
) -> Result<String, String> {
    if entries.is_empty() {
        return Ok(source);
    }
    let mut out = source;
    replace_once(
        &mut out,
        SPRITE_MARKER,
        "c = sprite_deform_vertex(c, corner, inst);",
    )?;
    push_shared(&mut out, entries);
    if !out.contains("fn sprite_vertex_of(") {
        out.push_str(SPRITE_VERTEX_OF);
    }
    out.push_str(&format!(
        "
// The registered deformers the batch opts into, in priority order.
fn sprite_deform_vertex(placed: SpriteCorner, corner: vec2<f32>, inst: SpriteInstance) -> SpriteCorner {{
    let mask = sprite_ub.mask.w;
    if mask == 0u {{
        return placed;
    }}
    var v = sprite_vertex_of(placed, corner, inst);
{calls}    var c = placed;
    c.clip = camera.view_proj * vec4<f32>(v.world, 1.0);
    c.world = v.world;
    return c;
}}
",
        calls = calls(entries)
    ));
    Ok(out)
}

/// `source`, an assembled tilemap module, with `entries` composed in. The
/// source as it is when there are none.
pub(crate) fn tile_module(
    source: String,
    entries: &[SpriteDeformerEntry],
) -> Result<String, String> {
    if entries.is_empty() {
        return Ok(source);
    }
    let mut out = source;
    replace_once(
        &mut out,
        TILE_MARKER,
        "world = tile_deform_vertex(world, corner, in, index, cell_origin, quad_min, quad_size, shift);",
    )?;
    push_shared(&mut out, entries);
    if !out.contains("fn tile_vertex_of(") {
        out.push_str(TILE_VERTEX_OF);
    }
    out.push_str(&format!(
        "
// The registered deformers the layer opts into, in priority order.
fn tile_deform_vertex(
    world: vec3<f32>,
    corner: vec2<f32>,
    in: TileIn,
    index: u32,
    cell_origin: vec2<f32>,
    quad_min: vec2<f32>,
    quad_size: vec2<f32>,
    shift: vec2<f32>,
) -> vec3<f32> {{
    let mask = layer.painter_layer.y;
    if mask == 0u {{
        return world;
    }}
    var v = tile_vertex_of(world, corner, in, index, cell_origin, quad_min, quad_size, shift);
{calls}    return v.world;
}}
",
        calls = calls(entries)
    ));
    Ok(out)
}

fn invalid(reason: String) -> ViewportError {
    ViewportError::ShadeShaderInvalid { reason }
}

/// Check `desc` against the deformer contract and make its entry, the next
/// after `existing`. Composition into the modules is checked by the caller.
pub(crate) fn entry(
    existing: &[SpriteDeformerEntry],
    desc: &SpriteDeformerDesc,
) -> ViewportResult<SpriteDeformerEntry> {
    let name = desc.name.as_str();
    if !wgsl_compose::is_identifier(name) {
        return Err(invalid(format!(
            "sprite deformer name '{name}' is not a WGSL identifier"
        )));
    }
    if existing.iter().any(|e| e.name == name) {
        return Err(ViewportError::ShadeNameTaken {
            name: name.to_string(),
        });
    }
    if existing.len() >= MAX_SPRITE_DEFORMERS {
        return Err(invalid(format!(
            "sprite deformer '{name}': at most {MAX_SPRITE_DEFORMERS} sprite deformers register"
        )));
    }
    let prefixed_body = wgsl_compose::prefix_identifiers(name, &desc.wgsl_body);
    if !prefixed_body.contains(&format!("fn {name}__deform_sprite(")) {
        return Err(invalid(format!(
            "sprite deformer '{name}' defines no `deform_sprite` function"
        )));
    }
    Ok(SpriteDeformerEntry {
        name: name.to_string(),
        prefixed_body,
        priority: desc.priority,
        slot: desc.params.map(|id| id.slot() as u32),
        bit: existing.len() as u32,
    })
}

/// Validate every module in `modules` with `entries` composed in.
pub(crate) fn validate(
    device: &viewport_lib::gpu::Device,
    label: &str,
    modules: impl IntoIterator<Item = (String, Result<String, String>)>,
) -> ViewportResult<()> {
    for (module, composed) in modules {
        let composed = composed.map_err(|e| invalid(format!("sprite deformer '{label}': {e}")))?;
        wgsl_compose::validate_module(
            device,
            &format!("sprite_deformer_{label}_{module}"),
            &composed,
        )?;
    }
    Ok(())
}

/// The id of the entry at `bit`.
pub(crate) fn id_of(entry: &SpriteDeformerEntry) -> SpriteDeformerId {
    SpriteDeformerId(entry.bit)
}
