//! Logic shared between the instanced and per-object mesh draw paths: the
//! common per-item material fields, and the predicate deciding which path an
//! item takes.

use super::*;

/// Why an item cannot be drawn through the instanced path.
///
/// [`fallback_reason`] returns the first test an item fails, so each
/// non-instanceable item has exactly one reason. `Hidden` is not a fallback:
/// a hidden item draws on neither path, so [`PerObjectReasons`] does not count
/// it.
///
/// [`PerObjectReasons`]: crate::renderer::stats::PerObjectReasons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FallbackReason {
    Hidden,
    Wireframe,
    ScalarAttribute,
    MaterialPluginNotReady,
    MaterialPluginVertexAttribute,
    Warp,
    Matcap,
    SubmeshMaterials,
    MissingMesh,
    PerInstanceDeform,
    MeshDeformSlots,
    VertexOverride,
    Lightmap,
    ForcePerObject,
}

/// Whether an item can be drawn through the instanced path.
///
/// The instanced shader (`mesh_instanced.wgsl`) handles only the common case: a
/// plain mesh with a simple material. Anything that needs per-item state the
/// instanced shader does not read falls back to the per-object path. This is the
/// single source of truth for that decision, used both when building the batches
/// and when deciding the instanced-batch cache key. [`fallback_reason`] holds
/// the tests and names the one an item fails.
pub(crate) fn is_instanceable(item: &SceneRenderItem, resources: &DeviceResources) -> bool {
    fallback_reason(item, resources).is_none()
}

/// The first reason `item` cannot be instanced, or `None` when it can.
///
/// All four back-face policies instance: `Cull` and `Identical` use the one-
/// and two-sided pipelines, and the styled policies (`DifferentColour`/`Tint`/
/// `Pattern`) run on the two-sided pipeline, where the instanced shaders read
/// the per-material policy/colour from `material_gpu_buf`, flip the normal on
/// back faces, and read the Pattern world scale from `InstanceData`. Param-vis
/// and premultiplied blend also instance (their `param_vis` mode/scale and
/// `alpha_mode` ride `material_gpu_buf`).
pub(crate) fn fallback_reason(
    item: &SceneRenderItem,
    resources: &DeviceResources,
) -> Option<FallbackReason> {
    use FallbackReason as R;
    if item.settings.hidden {
        return Some(R::Hidden);
    }
    // A wireframe item draws its edge lines, which only the per-object path
    // issues; an instanced batch would draw its filled triangles.
    if item.settings.wireframe {
        return Some(R::Wireframe);
    }
    if item.active_attribute.is_some() {
        return Some(R::ScalarAttribute);
    }
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
    if let Some(pid) = item.material.shading_plugin {
        if !resources.material_plugin_instanced_ready(pid) {
            return Some(R::MaterialPluginNotReady);
        }
        if resources.material_plugin_reads_vertex_attribute(pid) {
            return Some(R::MaterialPluginVertexAttribute);
        }
    }
    // A GPU vertex warp is a per-object-only feature: the instanced pipeline
    // has no warp support and would draw the mesh undeformed, ignoring
    // `warp_scale`. Keep warp items on the per-object path (matching the
    // per-object writer's own warp exception and the comment there).
    if item.warp_attribute.is_some() {
        return Some(R::Warp);
    }
    // A matcap is a texture bind the instanced path does not carry.
    if item.material.matcap_id().is_some() {
        return Some(R::Matcap);
    }
    // Per-submesh materials mean one draw per index range, each with its
    // own object bind group; the instanced path draws the whole mesh in
    // one call with batch-level textures, so range items stay per-object.
    if item.submesh_materials.is_some() {
        return Some(R::SubmeshMaterials);
    }
    let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
        return Some(R::MissingMesh);
    };
    if resources
        .deform
        .has_per_instance_deform_data(item.mesh_id, item.deform_instance)
    {
        return Some(R::PerInstanceDeform);
    }
    // Per-mesh deform slot data is per-object only, for the same reason as
    // warp above: the instanced draws bind the empty deform group and the
    // instanced shader passes no slot flags, so an instanced item reads no
    // slot data and renders undeformed. Keeping it per-object costs a
    // uniform write and a bind-group build per item and draws the shape the
    // consumer asked for. `FrameStats::deform_slots_ignored` counts any that
    // still reach an instanced batch, and should stay at zero because of
    // this line.
    if resources.deform.has_mesh_slot_data(item.mesh_id) {
        return Some(R::MeshDeformSlots);
    }
    if mesh.position_override_buffer.is_some() || mesh.normal_override_buffer.is_some() {
        return Some(R::VertexOverride);
    }
    // A baked lightmap is sampled only on the per-object path; the instanced
    // shader has no lightmap binding, so a lightmapped mesh must stay
    // per-object or its lightmap silently does not render.
    if mesh.lightmap.is_some() {
        return Some(R::Lightmap);
    }
    // Last, so an item with a real reason is counted under it.
    if item.settings.force_per_object {
        return Some(R::ForcePerObject);
    }
    None
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

/// Whether any of the item's draws belong in the transparent pass (OIT on
/// the HDR path). Per range for a per-range item: a
/// per-range item can be in both passes at once. A wireframe item has none.
pub(crate) fn has_transparent_draws(item: &SceneRenderItem, resources: &DeviceResources) -> bool {
    if item.settings.wireframe {
        return false;
    }
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

/// The per-instance facts of an item that both packers derive the same way.
/// Material scalars are not here: they live in the material block.
pub(super) struct CommonMaterial {
    pub(super) model: [[f32; 4]; 4],
    pub(super) colour: [f32; 4],
    pub(super) selected: u32,
    pub(super) has_texture: u32,
    pub(super) has_normal_map: u32,
    pub(super) has_ao_map: u32,
    pub(super) unlit: u32,
    pub(super) receive_shadows: u32,
}

/// `resolved` carries the material's texture slots looked up against the store.
/// The has-a-texture flags come from it rather than from `is_some()`: a slot
/// holding a handle whose texture has been freed binds the fallback, so the flag
/// has to say there is no texture, or the shader samples the fallback as though
/// it were authored.
pub(super) fn common_material(
    item: &SceneRenderItem,
    resolved: crate::resources::material_gpu::MaterialSlots,
) -> CommonMaterial {
    use crate::scene::material::TextureSlot::{Albedo, Ao, Normal};
    let m = &item.material;
    CommonMaterial {
        model: item.model,
        colour: {
            let bc = m.base_colour.to_linear_rgb();
            [bc[0], bc[1], bc[2], item.settings.opacity]
        },
        selected: if item.settings.selected { 1 } else { 0 },
        has_texture: if resolved.live(Albedo) { 1 } else { 0 },
        has_normal_map: if resolved.live(Normal) { 1 } else { 0 },
        has_ao_map: if resolved.live(Ao) { 1 } else { 0 },
        unlit: if item.settings.unlit { 1 } else { 0 },
        receive_shadows: if item.settings.receive_shadows { 1 } else { 0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Colour;
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
            is_instanceable(&item, &resources),
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
            !is_instanceable(&item, &resources),
            "a plugin item stays per-object until its instanced set is built",
        );

        resources.ensure_instanced_pipelines(&device);
        resources.ensure_material_plugin_instanced_pipelines(&device, id);

        assert!(
            is_instanceable(&item, &resources),
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
            is_instanceable(&item, &resources),
            "a plain mesh item should instance",
        );

        item.warp_attribute = Some("warp".to_string());
        assert!(
            !is_instanceable(&item, &resources),
            "a warp item must fall back to the per-object path",
        );
    }

    /// Per-mesh deform slot data is per-object only, the same as warp: the
    /// instanced draws bind the empty deform group and the instanced shader
    /// passes no slot flags, so an instanced item would render undeformed.
    /// `is_instanceable` must exclude it.
    #[test]
    fn per_mesh_deform_item_is_not_instanceable() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let mesh = crate::geometry::primitives::grid_plane(1.0, 1.0, 4, 4);
        let vertex_count = mesh.positions.len();
        let mesh_id = resources.upload_mesh_data(&device, &mesh).unwrap();

        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        assert!(
            is_instanceable(&item, &resources),
            "a plain mesh item should instance",
        );

        // One f32 per vertex on slot 0, the shape a displacement deformer takes.
        resources.attach_deform_slot(&device, mesh_id, 0, 4, &vec![0u8; vertex_count * 4]);
        assert!(
            !is_instanceable(&item, &resources),
            "an item whose mesh carries per-mesh deform data must fall back to \
             the per-object path, or it draws undeformed",
        );

        // Detaching restores instancing: the exclusion tracks the data, not a
        // one-way flag on the mesh.
        assert!(resources.detach_deform_slot(&device, mesh_id, 0));
        assert!(
            is_instanceable(&item, &resources),
            "detaching the slot puts the item back on the instanced path",
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
            is_instanceable(&item, &resources),
            "a straight-blend item should instance",
        );

        item.material.alpha_mode = AlphaMode::BlendPremultiplied;
        assert!(
            is_instanceable(&item, &resources),
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
            is_instanceable(&item, &resources),
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
        item.material.emissive = Colour::linear_rgb(2.0, 2.0, 2.0);
        item.material.emissive_texture_id = Some(crate::resources::TextureId::from_raw(1));
        assert!(
            is_instanceable(&item, &resources),
            "an emissive-textured item should instance",
        );

        item.material.metallic_roughness_texture_id =
            Some(crate::resources::TextureId::from_raw(2));
        assert!(
            is_instanceable(&item, &resources),
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
            is_instanceable(&item, &resources),
            "a plain mesh item should instance",
        );

        let tex = resources
            .upload_texture(
                &device,
                &queue,
                // Baked radiance, so linear: a lightmap slot rejects an sRGB upload.
                crate::resources::TextureData::linear(1, 1, [255u8, 255, 255, 255].to_vec()),
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
            !is_instanceable(&item, &resources),
            "a lightmapped mesh must fall back to the per-object path",
        );
    }
}
