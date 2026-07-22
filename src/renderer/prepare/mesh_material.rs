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
/// is hidden, carries a scalar attribute, carries a GPU vertex warp (the
/// instanced shader has no warp support), uses a styled back-face policy
/// (`DifferentColour`/`Tint`/`Pattern`, which need per-item back-face state), a
/// matcap, or param-vis, has a pending compute-filter result (which needs a
/// per-item index buffer), has per-instance deform data, or its mesh has a
/// position/normal override buffer bound. `Cull` and `Identical` back-face
/// policies both render through the instanced path: `Identical` batches use the
/// two-sided (`cull_mode: None`) instanced pipeline.
pub(crate) fn is_instanceable(
    item: &SceneRenderItem,
    resources: &DeviceResources,
    compute_filter_results: &[crate::resources::ComputeFilterResult],
) -> bool {
    !item.settings.hidden
        && item.active_attribute.is_none()
        // Material-plugin items select a per-item pipeline set and a group-3
        // params bind; the instanced path has neither, so they draw per-object.
        && item.material.shading_plugin.is_none()
        // A GPU vertex warp is a per-object-only feature: the instanced pipeline
        // has no warp support and would draw the mesh undeformed, ignoring
        // `warp_scale`. Keep warp items on the per-object path (matching the
        // per-object writer's own warp exception and the comment there).
        && item.warp_attribute.is_none()
        && !backface_needs_per_object(item)
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

/// Whether an item's back-face handling forces it onto the per-object path.
///
/// The instanced path admits the `Cull` and `Identical` policies (the latter via
/// the two-sided `cull_mode: None` solid pipeline), but only for opaque items: the
/// transparent instanced pipeline (OIT) is back-face culled, so a two-sided
/// transparent item would lose its back faces there and must stay per-object. The
/// styled policies (`DifferentColour`/`Tint`/`Pattern`) always go per-object. This
/// is the single predicate the instanced filter and both paint-path excluded
/// filters share, so they cannot drift.
pub(crate) fn backface_needs_per_object(item: &SceneRenderItem) -> bool {
    if item.material.backface_needs_per_object() {
        return true;
    }
    let transparent = item.settings.opacity < 1.0 || item.material.is_blend();
    item.material.is_two_sided() && transparent
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
    pub(super) normal_strength: f32,
    pub(super) has_ao_map: u32,
    pub(super) unlit: u32,
    pub(super) receive_shadows: u32,
    pub(super) use_flat: u32,
    pub(super) uv_transform: [f32; 4],
    pub(super) ao_range: [f32; 2],
    pub(super) alpha_cutoff: f32,
    pub(super) alpha_flag: u32,
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
        normal_strength: m.normal_strength,
        has_ao_map: if m.ao_map_id.is_some() { 1 } else { 0 },
        unlit: if item.settings.unlit { 1 } else { 0 },
        receive_shadows: if item.settings.receive_shadows { 1 } else { 0 },
        use_flat: if m.is_flat() { 1 } else { 0 },
        uv_transform: [m.uv_offset[0], m.uv_offset[1], m.uv_scale[0], m.uv_scale[1]],
        ao_range: m.ao_range,
        alpha_cutoff: match m.alpha_mode {
            crate::scene::material::AlphaMode::Mask(c) => c,
            _ => 0.5,
        },
        alpha_flag: match m.alpha_mode {
            crate::scene::material::AlphaMode::Mask(_) => 1,
            _ => 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::{AlphaMode, BackfacePolicy};

    /// A two-sided alpha-test (`Mask`) card is opaque and must stay on the
    /// instanced path: `backface_needs_per_object` only forces per-object for
    /// transparent (blend / opacity < 1) two-sided items, not for `Mask`.
    #[test]
    fn two_sided_mask_stays_instanceable() {
        let mut item = SceneRenderItem::default();
        item.material.backface_policy = BackfacePolicy::Identical;
        item.material.alpha_mode = AlphaMode::Mask(0.45);
        assert!(item.material.is_two_sided());
        assert!(!backface_needs_per_object(&item));
    }

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    /// A GPU vertex warp is per-object only: the instanced shader ignores
    /// `warp_scale`, so a warp item routed to the instanced path renders
    /// undeformed and its warp control does nothing. `is_instanceable` must
    /// exclude it.
    #[test]
    fn warp_item_is_not_instanceable() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mesh = crate::geometry::primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &mesh).unwrap();

        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        assert!(
            is_instanceable(&item, &resources, &[]),
            "a plain mesh item should instance",
        );

        item.warp_attribute = Some("warp".to_string());
        assert!(
            !is_instanceable(&item, &resources, &[]),
            "a warp item must fall back to the per-object path",
        );
    }

    /// The shared material mapping carries the cutoff and enable flag that the
    /// instanced shader reads. A `Mask` item enables the discard; anything else
    /// disables it.
    #[test]
    fn common_material_carries_alpha_cutout() {
        let mut item = SceneRenderItem::default();
        item.material.alpha_mode = AlphaMode::Mask(0.45);
        let cm = common_material(&item);
        assert_eq!(cm.alpha_flag, 1);
        assert!((cm.alpha_cutoff - 0.45).abs() < 1e-6);

        item.material.alpha_mode = AlphaMode::Opaque;
        let cm = common_material(&item);
        assert_eq!(cm.alpha_flag, 0);
    }
}
