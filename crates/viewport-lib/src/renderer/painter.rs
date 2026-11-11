//! The renderer's own painter's-order content: mesh-instance batches and
//! surfaces carrying a [`PainterPlacement`](crate::PainterPlacement). They take
//! part in the painter sequence beside the item-type plugins, under
//! [`CORE_PAINTER`], sorting as if registered before every plugin.

use super::*;
use crate::plugin_api::PainterBatch;

/// The name the renderer's own painter content runs under in the painter
/// sequence.
pub(crate) const CORE_PAINTER: &str = "vpl.painter_meshes";

/// Set on a core painter batch handle that names a painter surface rather than
/// a mesh-instance batch.
const SURFACE_BATCH: u32 = 1 << 31;

/// One core painter draw, decoded from its batch handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CoreBatch {
    /// An index into `mesh_instance_gpu_data`.
    Instances(usize),
    /// An index into the viewport's painter surfaces, or on the compatibility
    /// tier into its painter mesh draws.
    Surface(usize),
}

impl CoreBatch {
    fn handle(self) -> u32 {
        match self {
            Self::Instances(i) => i as u32,
            Self::Surface(i) => SURFACE_BATCH | i as u32,
        }
    }

    pub(crate) fn decode(handle: u32) -> Self {
        if handle & SURFACE_BATCH != 0 {
            Self::Surface((handle & !SURFACE_BATCH) as usize)
        } else {
            Self::Instances(handle as usize)
        }
    }
}

/// Whether a surface draws in the painter sequence of a viewport culling with
/// `cull_mask`.
pub(crate) fn painter_surface(item: &SceneRenderItem, cull_mask: u32) -> bool {
    item.painter.is_some()
        && !item.settings.hidden
        && (item.settings.visibility_mask & cull_mask) != 0
}

impl ViewportRenderer {
    /// The renderer's own painter batches for `frame`'s viewport, in report
    /// order.
    pub(crate) fn core_painter_batches(&self, frame: &FrameData, out: &mut Vec<PainterBatch>) {
        for (i, gd) in self.mesh_instance_gpu_data.iter().enumerate() {
            if let Some(p) = gd.painter {
                out.push(PainterBatch::at(p, CoreBatch::Instances(i).handle()));
            }
        }
        if self.resources.scene_tier.is_built() {
            if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                for (j, (_, p)) in slot.painter_surfaces.iter().enumerate() {
                    out.push(PainterBatch::at(*p, CoreBatch::Surface(j).handle()));
                }
            }
        } else {
            for (j, p) in self.basic_painter_placements().enumerate() {
                out.push(PainterBatch::at(p, CoreBatch::Surface(j).handle()));
            }
        }
    }

    /// Whether the renderer itself has painter's-order draws in `frame`'s
    /// viewport.
    pub(crate) fn core_draws_painter(&self, frame: &FrameData) -> bool {
        self.mesh_instance_gpu_data
            .iter()
            .any(|gd| gd.painter.is_some())
            || if self.resources.scene_tier.is_built() {
                self.viewport_slots
                    .get(frame.camera.viewport_index)
                    .is_some_and(|s| !s.painter_surfaces.is_empty())
            } else {
                self.basic_painter_placements().next().is_some()
            }
    }

    /// Collect the viewport's painter surfaces from the submission and write
    /// their per-item uniforms. Full tier only.
    pub(super) fn prepare_painter_surfaces(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        let vp_idx = frame.camera.viewport_index;
        let SurfaceSubmission::Flat(items) = &frame.scene.surfaces;
        let cull_mask = frame.camera.cull_mask;
        let slot = &mut self.viewport_slots[vp_idx];
        slot.painter_surfaces.clear();
        slot.painter_surfaces.extend(
            items
                .iter()
                .enumerate()
                .filter(|(_, item)| painter_surface(item, cull_mask))
                .map(|(i, item)| (i, item.painter.expect("filtered on a placement"))),
        );
        if slot.painter_surfaces.is_empty() {
            slot.painter_objects.clear();
            return;
        }
        let mut entries = std::mem::take(&mut slot.painter_objects);
        let picked = slot.painter_surfaces.iter().map(|&(i, _)| &items[i]);
        Self::prepare_foreground_objects(&mut self.resources, &mut entries, picked, device, queue);
        self.viewport_slots[vp_idx].painter_objects = entries;
    }

    /// Draw the core painter batches `batches` names, in that order. `hdr` is
    /// `true` in the HDR painter pass, `false` in the LDR scene pass.
    pub(crate) fn paint_core_painter(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        frame: &FrameData,
        batches: &[u32],
        hdr: bool,
        camera_bg: &crate::gpu::BindGroup,
    ) {
        if !self.resources.scene_tier.is_built() {
            for &b in batches {
                if let CoreBatch::Surface(j) = CoreBatch::decode(b) {
                    self.paint_basic_painter(pass, camera_bg, j);
                }
            }
            pass.set_bind_group(0, camera_bg, &[]);
            return;
        }
        let resources = &self.resources;
        let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) else {
            return;
        };
        let SurfaceSubmission::Flat(items) = &frame.scene.surfaces;
        pass.set_bind_group(0, camera_bg, &[]);
        for &b in batches {
            match CoreBatch::decode(b) {
                CoreBatch::Instances(i) => {
                    let Some(batch) = self.mesh_instance_gpu_data.get(i) else {
                        continue;
                    };
                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                        continue;
                    };
                    if mesh.index_count == 0 {
                        continue;
                    }
                    let pipeline = if hdr {
                        resources.scene.hdr_painter(batch.blend)
                    } else {
                        resources.scene.ldr_painter()
                    };
                    // Still compiling: this batch draws next frame.
                    let Some(pipeline) = pipeline else {
                        continue;
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(1, &batch.bind_group, &[]);
                    pass.set_vertex_buffer(1, batch.instance_buf.slice(..));
                    pass.set_vertex_buffer(0, resources.geometry.vertex_slice(mesh.vertex_span));
                    pass.set_index_buffer(
                        resources.geometry.index_slice(mesh.index_span),
                        crate::gpu::IndexFormat::Uint32,
                    );
                    pass.draw_indexed(0..mesh.index_count, 0, 0..batch.instance_count);
                }
                CoreBatch::Surface(j) => {
                    let (Some(&(index, _)), Some(entry)) =
                        (slot.painter_surfaces.get(j), slot.painter_objects.get(j))
                    else {
                        continue;
                    };
                    let (Some(item), Some(bind_group)) = (items.get(index), &entry.bind_group)
                    else {
                        continue;
                    };
                    let pipeline = if hdr {
                        resources
                            .scene
                            .hdr_painter(crate::renderer::SpriteBlend::AlphaBlend)
                    } else {
                        resources.scene.ldr_painter()
                    };
                    let Some(pipeline) = pipeline else {
                        continue;
                    };
                    draw_painter_surface(
                        resources,
                        pass,
                        item,
                        bind_group,
                        &entry.uniform_buf,
                        pipeline,
                    );
                }
            }
        }
    }
}

/// Draw one painter surface whole with `pipeline`, through its own object
/// record.
fn draw_painter_surface(
    resources: &crate::resources::DeviceResources,
    pass: &mut crate::gpu::RenderPass<'_>,
    item: &SceneRenderItem,
    bind_group: &crate::gpu::BindGroup,
    records: &crate::gpu::Buffer,
    pipeline: &crate::gpu::RenderPipeline,
) {
    let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
        return;
    };
    pass.set_pipeline(pipeline);
    pass.set_bind_group(1, bind_group, &[]);
    pass.set_vertex_buffer(1, records.slice(..));
    let face_attribute = item.active_attribute.as_ref().is_some_and(|a| {
        matches!(
            a.kind,
            crate::resources::AttributeKind::Face
                | crate::resources::AttributeKind::FaceColour
                | crate::resources::AttributeKind::Halfedge
                | crate::resources::AttributeKind::Corner
        )
    });
    if face_attribute {
        let Some(fvb) = &mesh.face_vertex_buffer else {
            return;
        };
        pass.set_vertex_buffer(0, fvb.slice(..));
        pass.draw(0..mesh.index_count, 0..1);
        return;
    }
    // The record carries the mesh's base vertex, so the draw reads the whole
    // slab chunk, as the foreground pass does.
    pass.set_vertex_buffer(
        0,
        resources
            .geometry
            .vertex_chunk_slice(mesh.vertex_span.chunk),
    );
    pass.set_index_buffer(
        resources.geometry.index_chunk_slice(mesh.index_span.chunk),
        crate::gpu::IndexFormat::Uint32,
    );
    let first = resources.geometry.first_index(mesh.index_span);
    pass.draw_indexed(
        first..first + mesh.index_count,
        resources.geometry.base_vertex(mesh.vertex_span),
        0..1,
    );
}
