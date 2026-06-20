//! Per-item material fields shared by the instanced and per-object draw paths.
//!
//! `ObjectUniform` (per-object) and `InstanceData` (instanced) are different GPU
//! structs, but the material and shading fields they have in common are derived
//! from the item the same way. This computes that common core once so the two
//! builders cannot drift apart. Each path fills its own extra fields separately:
//! the instanced path forces `wireframe` off, and the per-object path adds the
//! features the instanced shader does not handle (scalar attributes, matcaps,
//! per-side borders, overrides, and so on).

use super::*;

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
