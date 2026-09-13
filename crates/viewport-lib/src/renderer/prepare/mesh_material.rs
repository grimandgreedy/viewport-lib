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
/// instanced shader has no warp support), uses a matcap (a texture bind the
/// instanced path does not yet carry), has a pending
/// compute-filter result (which needs a per-item index buffer), carries
/// per-submesh materials, has per-instance deform data, or its mesh has a
/// position/normal override or baked lightmap. All four back-face policies now
/// instance: `Cull` and `Identical` use the one- and two-sided pipelines, and the
/// styled policies (`DifferentColour`/`Tint`/`Pattern`) run on the two-sided
/// pipeline, where the instanced shaders read the per-material policy/colour from
/// `material_gpu_buf`, flip the normal on back faces, and read the Pattern world
/// scale from `InstanceData`. Param-vis and premultiplied blend also instance
/// (their `param_vis` mode/scale and `alpha_mode` ride `material_gpu_buf`).
pub(crate) fn is_instanceable(
    item: &SceneRenderItem,
    resources: &DeviceResources,
    compute_filter_results: &[crate::resources::ComputeFilterResult],
) -> bool {
    !item.settings.hidden
        && item.active_attribute.is_none()
        // Material-plugin items instance once the plugin's instanced pipeline set
        // is built (the plugin's shading composed onto the instanced modules, on
        // the group-3 layout, for the active per-batch or bindless binding). Until
        // then, or on an unknown id where no instanced set exists, they draw through
        // the per-object path. A plugin that reads the per-vertex extension
        // attribute stays per-object regardless: the instanced path has no
        // per-vertex extension-attribute binding (its `surf.attr` is the
        // per-instance custom-data channel), so instancing such a plugin would feed
        // it the wrong data. This mirrors the per-object-only `active_attribute`
        // exclusion above.
        && match item.material.shading_plugin {
            None => true,
            Some(pid) => {
                resources.material_plugin_instanced_ready(pid)
                    && !resources.material_plugin_reads_vertex_attribute(pid)
            }
        }
        // A GPU vertex warp is a per-object-only feature: the instanced pipeline
        // has no warp support and would draw the mesh undeformed, ignoring
        // `warp_scale`. Keep warp items on the per-object path (matching the
        // per-object writer's own warp exception and the comment there).
        && item.warp_attribute.is_none()
        && item.material.matcap_id().is_none()
        // A per-material sampler (wrap/filter/aniso) is bound at group-1 binding 2
        // on the per-object path. The instanced/bindless path shares one sampler
        // across a batch (and, under bindless, across the whole texture array), so
        // it cannot honour a per-material sampler until a bindless sampler heap
        // carries one per slot. Until then, a material that sets a sampler draws
        // per-object so its wrap mode is not silently dropped.
        && item.material.selected_sampler().is_none()
        // Per-submesh materials mean one draw per index range, each with its
        // own object bind group; the instanced path draws the whole mesh in
        // one call with batch-level textures, so range items stay per-object.
        && item.submesh_materials.is_none()
        && resources.mesh_store.get(item.mesh_id).is_some()
        && !compute_filter_results
            .iter()
            .any(|r| r.mesh_id == item.mesh_id)
        && !resources
            .deform
            .has_per_instance_deform_data(item.mesh_id, item.deform_instance)
        && resources.mesh_store.get(item.mesh_id).map_or(true, |m| {
            m.position_override_buffer.is_none() && m.normal_override_buffer.is_none()
            // A baked lightmap is sampled only on the per-object path; the
            // instanced shader has no lightmap binding, so a lightmapped mesh must
            // stay per-object or its lightmap silently does not render.
                && m.lightmap.is_none()
        })
}

/// The per-range materials to draw `item` with, when it requests them and
/// they line up with the mesh's ranges. `None` means the item draws the
/// whole mesh with its single `material`: either it never set
/// `submesh_materials`, the mesh has no ranges, or the two disagree on
/// count (the mismatch falls back rather than guessing an assignment).
pub(crate) fn active_submesh_materials<'a>(
    item: &'a SceneRenderItem,
    mesh: &crate::resources::GpuMesh,
) -> Option<&'a [crate::scene::material::Material]> {
    let mats = item.submesh_materials.as_deref()?;
    if mesh.submeshes.is_empty() || mats.len() != mesh.submeshes.len() {
        if !mesh.submeshes.is_empty() {
            tracing::debug!(
                mesh_index = item.mesh_id.index(),
                materials = mats.len(),
                ranges = mesh.submeshes.len(),
                "submesh_materials count does not match the mesh's ranges; \
                 drawing with the item material"
            );
        }
        return None;
    }
    Some(mats)
}

/// Whether any of the item's draws belong in the opaque scene pass. For a
/// per-range item that is any opaque-material range; otherwise it is the
/// usual whole-item opacity/blend check.
pub(crate) fn has_opaque_draws(item: &SceneRenderItem, resources: &DeviceResources) -> bool {
    if item.settings.opacity < 1.0 {
        return false;
    }
    match resources
        .mesh_store
        .get(item.mesh_id)
        .and_then(|m| active_submesh_materials(item, m))
    {
        Some(mats) => mats.iter().any(|m| !m.is_blend()),
        None => !item.material.is_blend(),
    }
}

/// Whether any of the item's draws belong in the transparent pass (OIT on
/// the HDR path). The complement of [`has_opaque_draws`] per range: a
/// per-range item can be in both passes at once.
pub(crate) fn has_transparent_draws(item: &SceneRenderItem, resources: &DeviceResources) -> bool {
    if item.settings.opacity < 1.0 {
        return true;
    }
    match resources
        .mesh_store
        .get(item.mesh_id)
        .and_then(|m| active_submesh_materials(item, m))
    {
        Some(mats) => mats.iter().any(|m| m.is_blend()),
        None => item.material.is_blend(),
    }
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
    pub(super) ao_range: [f32; 2],
}

pub(super) fn common_material(item: &SceneRenderItem) -> CommonMaterial {
    let m = &item.material;
    CommonMaterial {
        model: item.model,
        colour: {
            let bc = m.base_colour.to_linear_rgb();
            [bc[0], bc[1], bc[2], item.settings.opacity]
        },
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
        ao_range: m.ao_range,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::{AlphaMode, BackfacePolicy};

    /// `Identical` is a two-sided policy but not a styled one, so it is not
    /// flagged for per-item back-face handling; a two-sided `Mask` card instances
    /// on the two-sided pipeline.
    #[test]
    fn identical_is_not_a_styled_backface() {
        let mut item = SceneRenderItem::default();
        item.material.backface_policy = BackfacePolicy::Identical;
        item.material.alpha_mode = AlphaMode::Mask(0.45);
        assert!(item.material.is_two_sided());
        assert!(!item.material.backface_needs_per_object());
    }

    /// Styled back-face policies now instance: the per-material policy/colour ride
    /// `material_gpu_buf` and the shader flips the normal and overrides the colour
    /// on back faces.
    #[test]
    fn styled_backface_is_instanceable() {
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
        item.material.backface_policy =
            BackfacePolicy::DifferentColour(crate::Colour::linear_rgb(1.0, 0.0, 0.0));
        assert!(
            is_instanceable(&item, &resources, &[]),
            "a styled-backface item should instance",
        );
    }

    /// A material-plugin item instances only once the plugin's instanced
    /// pipeline set is built: cold, it must fall back to the per-object path
    /// (where its plugin still shades); once built, `is_instanceable` admits it.
    #[test]
    fn shading_plugin_item_instances_once_its_set_is_built() {
        use crate::MaterialPlugin;
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        // A bare DeviceResources defaults to the per-batch texture path, which is
        // where the plugin instanced set is built.
        let mesh = crate::geometry::primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &mesh).unwrap();

        struct Toon;
        impl MaterialPlugin for Toon {
            fn name(&self) -> &'static str {
                "toon_instanceable_probe"
            }
            fn wgsl_body(&self) -> String {
                "fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {\n\
                 \x20   return surf.base_colour * light.radiance * light.shadow;\n\
                 }\n"
                .to_string()
            }
        }
        let id = resources
            .register_material_plugin(&device, &Toon)
            .expect("register");

        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.material.shading_plugin = Some(id);

        assert!(
            !is_instanceable(&item, &resources, &[]),
            "a plugin item stays per-object until its instanced set is built",
        );

        resources.ensure_instanced_pipelines(&device);
        resources.ensure_material_plugin_instanced_pipelines(&device, id);

        assert!(
            is_instanceable(&item, &resources, &[]),
            "a plugin item instances once its instanced set is ready",
        );
    }

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
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

    /// Straight and premultiplied blend both instance: the instanced OIT shader
    /// reads the per-material `alpha_mode` from `material_gpu_buf` and skips the
    /// premultiply for mode 3, mirroring the per-object OIT shader.
    #[test]
    fn blend_modes_are_instanceable() {
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
        item.material.alpha_mode = AlphaMode::Blend;
        assert!(
            is_instanceable(&item, &resources, &[]),
            "a straight-blend item should instance",
        );

        item.material.alpha_mode = AlphaMode::BlendPremultiplied;
        assert!(
            is_instanceable(&item, &resources, &[]),
            "a premultiplied-blend item should instance (alpha_mode rides the material buffer)",
        );
    }

    /// Param-vis instances: the instanced shaders read the per-material mode and
    /// scale from `material_gpu_buf` and run the procedural pattern.
    #[test]
    fn param_vis_is_instanceable() {
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
        item.material.param_vis = Some(crate::scene::material::ParamVis {
            mode: crate::scene::material::ParamVisMode::Checker,
            scale: 8.0,
        });
        assert!(
            is_instanceable(&item, &resources, &[]),
            "a param-vis item should instance (mode/scale ride the material buffer)",
        );
    }

    // Alpha cutout and emissive now ride the per-material block, covered by the
    // `material_gpu` tests (`scalars_pack_alpha_and_emissive`).

    /// Emissive- and metallic-roughness-textured materials now instance: the
    /// instanced shaders sample both maps (the textures are per-batch, bound on
    /// the instanced group-1 layout; the has-flags and MR ranges ride the material
    /// buffer).
    #[test]
    fn textured_pbr_maps_are_instanceable() {
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
        item.material.emissive = [2.0, 2.0, 2.0].into();
        item.material.emissive_texture_id = Some(crate::resources::TextureId::from_raw(1));
        assert!(
            is_instanceable(&item, &resources, &[]),
            "an emissive-textured item should instance",
        );

        item.material.metallic_roughness_texture_id =
            Some(crate::resources::TextureId::from_raw(2));
        assert!(
            is_instanceable(&item, &resources, &[]),
            "a metallic-roughness-textured item should instance",
        );
    }

    /// A baked lightmap is sampled only on the per-object path (the instanced
    /// shader has no lightmap binding), so a lightmapped mesh must not instance
    /// or its lightmap silently fails to render.
    #[test]
    fn lightmapped_mesh_is_not_instanceable() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mesh = crate::geometry::primitives::plane(1.0, 1.0);
        let mesh_id = resources.upload_mesh_data(&device, &mesh).unwrap();

        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        assert!(
            is_instanceable(&item, &resources, &[]),
            "a plain mesh item should instance",
        );

        let tex = resources
            .upload_texture(
                &device,
                &queue,
                crate::resources::TextureData::srgb(1, 1, [255u8, 255, 255, 255].to_vec()),
            )
            .unwrap();
        resources
            .set_lightmap(
                &device,
                mesh_id,
                &[glam::Vec2::ZERO; 4],
                crate::resources::lightmap::LightmapData::NonDirectional { radiance: tex },
                crate::resources::lightmap::LightmapMode::Replace,
            )
            .unwrap();
        assert!(
            !is_instanceable(&item, &resources, &[]),
            "a lightmapped mesh must fall back to the per-object path",
        );
    }
}
