//! Logic shared between the instanced and per-object mesh draw paths: the
//! common per-item material fields, and the predicate deciding which path an
//! item takes.

use super::*;

/// Whether an item can be drawn through the instanced path.
///
/// The instanced shader (`mesh_instanced.wgsl`) handles only the common case: a
/// plain mesh with a simple material. Anything that needs per-item state the
/// instanced shader does not read falls back to the per-object path. This is the
/// single source of truth for that decision, used both when building the batches
/// and when deciding the instanced-batch cache key. An item is excluded when it
/// is hidden, carries a scalar attribute, uses a two-sided material, a matcap, or
/// param-vis, has a pending compute-filter result (which needs a per-item index
/// buffer), has per-instance deform data, or its mesh has a position/normal
/// override buffer bound.
pub(super) fn is_instanceable(
    item: &SceneRenderItem,
    resources: &ViewportGpuResources,
    compute_filter_results: &[crate::resources::ComputeFilterResult],
) -> bool {
    !item.settings.hidden
        && item.active_attribute.is_none()
        && !item.material.is_two_sided()
        && item.material.matcap_id().is_none()
        && item.material.param_vis.is_none()
        && resources.mesh_store.get(item.mesh_id).is_some()
        && !compute_filter_results
            .iter()
            .any(|r| r.mesh_id == item.mesh_id)
        && !resources
            .deform
            .has_per_instance_deform_data(item.mesh_id, item.deform_instance)
        && resources.mesh_store.get(item.mesh_id).map_or(true, |m| {
            m.position_override_buffer.is_none() && m.normal_override_buffer.is_none()
        })
}

pub(super) struct CommonMaterial {
    pub(super) model: [[f32; 4]; 4],
    pub(super) colour: [f32; 4],
    pub(super) selected: u32,
    pub(super) ambient: f32,
    pub(super) diffuse: f32,
    pub(super) specular: f32,
    pub(super) shininess: f32,
    pub(super) has_texture: u32,
    pub(super) use_pbr: u32,
    pub(super) metallic: f32,
    pub(super) roughness: f32,
    pub(super) has_normal_map: u32,
    pub(super) has_ao_map: u32,
    pub(super) unlit: u32,
    pub(super) receive_shadows: u32,
    pub(super) use_flat: u32,
    pub(super) uv_transform: [f32; 4],
    pub(super) ao_range: [f32; 2],
}

pub(super) fn common_material(item: &SceneRenderItem) -> CommonMaterial {
    let m = &item.material;
    CommonMaterial {
        model: item.model,
        colour: [
            m.base_colour[0],
            m.base_colour[1],
            m.base_colour[2],
            item.settings.opacity,
        ],
        selected: if item.settings.selected { 1 } else { 0 },
        ambient: m.ambient,
        diffuse: m.diffuse,
        specular: m.specular,
        shininess: m.shininess,
        has_texture: if m.texture_id.is_some() { 1 } else { 0 },
        use_pbr: if m.is_pbr() { 1 } else { 0 },
        metallic: m.metallic,
        roughness: m.roughness,
        has_normal_map: if m.normal_map_id.is_some() { 1 } else { 0 },
        has_ao_map: if m.ao_map_id.is_some() { 1 } else { 0 },
        unlit: if item.settings.unlit { 1 } else { 0 },
        receive_shadows: if item.settings.receive_shadows { 1 } else { 0 },
        use_flat: if m.is_flat() { 1 } else { 0 },
        uv_transform: [m.uv_offset[0], m.uv_offset[1], m.uv_scale[0], m.uv_scale[1]],
        ao_range: m.ao_range,
    }
}
