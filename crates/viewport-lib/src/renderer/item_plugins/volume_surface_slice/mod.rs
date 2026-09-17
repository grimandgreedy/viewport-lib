//! The volume surface slice item type as an [`ItemTypePlugin`]: an arbitrary
//! uploaded mesh coloured by sampling a 3D scalar field at each fragment's
//! world position. Consumers submit [`VolumeSurfaceSliceItem`]s on
//! `SceneFrame::volume_surface_slices`; the renderer routes that field to this
//! plugin.
//!
//! The geometry belongs to the consumer's mesh store, so the draw hooks bind it
//! through [`MeshDraw`](crate::resources::MeshDraw) and the pick hooks read its
//! CPU arrays through [`MeshGeometry`](crate::resources::MeshGeometry) rather
//! than keeping copies.

mod pipeline;
pub(crate) mod types;

use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext,
};
use crate::renderer::{PickHit, PickId, PickMask, VolumeSurfaceSliceItem};
use crate::resources::HDR_COLOR_FORMAT;

pub(crate) const TYPE_NAME: &str = "viewport.volume_surface_slice";

impl PluginItemCollection for Vec<VolumeSurfaceSliceItem> {
    fn len(&self) -> usize {
        self.len()
    }
    fn item_settings(&self, index: usize) -> &crate::scene::material::ItemSettings {
        &self[index].settings
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Default)]
pub(crate) struct VolumeSurfaceSlicePlugin {
    gpu: Option<pipeline::SliceGpu>,
    /// Per drawn item, rebuilt each prepare.
    frame: Vec<pipeline::SliceFrame>,
    /// All items from the last prepared frame, hidden included, matching what
    /// the CPU pick cache used to retain.
    pick_items: Vec<VolumeSurfaceSliceItem>,
    /// Whether the frame's selection outline is active; the mask hook draws
    /// nothing when it is off.
    outline_active: bool,
}

impl ItemTypePlugin for VolumeSurfaceSlicePlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        self.outline_active = ctx.outline_selected;
        let items = items
            .as_any()
            .downcast_ref::<Vec<VolumeSurfaceSliceItem>>()
            .expect("volume surface slice collection is the SceneFrame field");
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::SliceGpu::new(device, ctx.resources));
        for item in items {
            if item.settings.hidden {
                continue;
            }
            if let Some(entry) = gpu.upload_item(device, queue, ctx.resources, item) {
                self.frame.push(entry);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if self.frame.is_empty() {
            return;
        }
        pass.set_pipeline(
            gpu.pipeline
                .for_format(ctx.target_format == HDR_COLOR_FORMAT),
        );
        for entry in &self.frame {
            pass.set_bind_group(1, &entry.bind_group, &[]);
            ctx.meshes.draw_indexed(pass, entry.mesh_id);
        }
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn outline_mask(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &OutlineMaskContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if !self.outline_active {
            return;
        }
        let mut bound = false;
        for entry in &self.frame {
            if !entry.selected {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.mask_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &entry.bind_group, &[]);
            ctx.meshes.draw_indexed(pass, entry.mesh_id);
        }
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE {
                continue;
            }
            let (Some(positions), Some(indices)) = (
                ctx.meshes.positions(item.mesh_id),
                ctx.meshes.indices(item.mesh_id),
            ) else {
                continue;
            };
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            let verts: Vec<parry3d::math::Vector> = positions
                .iter()
                .map(|p| {
                    let wp = model.transform_point3(glam::Vec3::from(*p));
                    parry3d::math::Vector::new(wp.x, wp.y, wp.z)
                })
                .collect();
            let tri_indices: Vec<[u32; 3]> = indices
                .chunks(3)
                .filter(|c| c.len() == 3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            if tri_indices.is_empty() {
                continue;
            }
            let Ok(trimesh) = parry3d::shape::TriMesh::new(verts, tri_indices) else {
                continue;
            };
            let parry_ray = parry3d::query::Ray::new(
                parry3d::math::Vector::new(ray.origin.x, ray.origin.y, ray.origin.z),
                parry3d::math::Vector::new(ray.direction.x, ray.direction.y, ray.direction.z),
            );
            use parry3d::query::RayCast as _;
            let Some(hit) = trimesh.cast_ray_and_get_normal(
                &parry3d::math::Pose::identity(),
                &parry_ray,
                f32::MAX,
                true,
            ) else {
                continue;
            };
            let toi = hit.time_of_impact;
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                let n = hit.normal;
                #[allow(deprecated)]
                let pick_hit = PickHit {
                    id: item.settings.pick_id.0,
                    sub_object: None,
                    world_pos: ray.origin + ray.direction * toi,
                    normal: glam::Vec3::new(n.x, n.y, n.z),
                    scalar_value: None,
                    sub_object_world_pos: None,
                };
                best = Some((toi, pick_hit));
            }
        }
        best
    }

    /// Hit when any of the mesh's world-space vertices projects into the rect.
    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> crate::renderer::PickRectResult {
        let mut result = crate::renderer::PickRectResult::default();
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return result;
        }
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE {
                continue;
            }
            let Some(positions) = ctx.meshes.positions(item.mesh_id) else {
                continue;
            };
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            let hit = positions.iter().any(|&p| {
                let wp = model.transform_point3(glam::Vec3::from(p));
                crate::plugin_api::pick_helpers::project_to_screen(
                    wp,
                    ctx.view_proj,
                    ctx.viewport_size,
                )
                .is_some_and(|s| {
                    s.x >= ctx.rect_min.x
                        && s.x <= ctx.rect_max.x
                        && s.y >= ctx.rect_min.y
                        && s.y <= ctx.rect_max.y
                })
            });
            if hit {
                result.objects.push(item.settings.pick_id.0);
            }
        }
        result
    }

    fn render_pick(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for entry in &self.frame {
            let Some((_, pick_bg)) = &entry.pick else {
                continue;
            };
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &entry.bind_group, &[]);
            pass.set_bind_group(2, pick_bg, &[]);
            ctx.meshes.draw_indexed(pass, entry.mesh_id);
        }
    }
}
