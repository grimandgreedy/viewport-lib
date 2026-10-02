//! The surface mask: a per-pixel record of which layers the item owning each
//! opaque pixel belongs to, held in the scene stencil.
//!
//! Screen-space effects that land on some surfaces and not others (decals)
//! read it. Every pixel starts as a member of every layer; the pass here
//! overwrites that for the items whose value some reader would refuse. Mesh
//! surfaces are stamped from this module, every other item type through
//! [`ItemTypePlugin::surface_mask`](crate::plugin_api::ItemTypePlugin::surface_mask).

use super::ViewportRenderer;
use crate::plugin_api::{
    SURFACE_MASK_DEFAULT,
    item_type::{surface_mask_needs_stamp, surface_mask_value},
};
use crate::renderer::types::FrameData;

/// One mesh surface to stamp. Its model matrix is the instance at `instance`
/// in the frame's instance buffer.
struct MeshStamp {
    mesh_id: crate::MeshId,
    instance: u32,
    value: u32,
    /// Checked against the viewport's cull mask at draw time: a surface the
    /// camera did not draw wrote no depth, so its stamp would land on whatever
    /// is behind it.
    visibility_mask: u32,
}

/// Per-frame surface mask state, rebuilt at the top of prepare.
pub(crate) struct SurfaceMaskState {
    /// The masks this frame's readers test against, folded and deduplicated.
    /// Empty when nothing reads the mask, which skips the pass.
    readers: Vec<u32>,
    mesh_stamps: Vec<MeshStamp>,
    models: Vec<[[f32; 4]; 4]>,
    model_buf: super::overlay_buffers::GrowBuffer,
    model_buf_handle: Option<crate::gpu::Buffer>,
    /// Whether some item-type plugin has an item that needs a stamp.
    plugin_stamps: bool,
    pipeline: Option<crate::gpu::RenderPipeline>,
}

impl SurfaceMaskState {
    pub(crate) fn new() -> Self {
        Self {
            readers: Vec::new(),
            mesh_stamps: Vec::new(),
            models: Vec::new(),
            model_buf: super::overlay_buffers::GrowBuffer::vertex("surface_mask_model_buf"),
            model_buf_handle: None,
            plugin_stamps: false,
            pipeline: None,
        }
    }

    fn active(&self) -> bool {
        !self.readers.is_empty() && (!self.mesh_stamps.is_empty() || self.plugin_stamps)
    }
}

impl ViewportRenderer {
    /// Work out which items need stamping this frame. Runs before plugin
    /// prepare, from the submitted items alone.
    pub(crate) fn collect_surface_mask(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        let state = &mut self.surface_mask;
        state.readers.clear();
        state.mesh_stamps.clear();
        state.models.clear();
        state.plugin_stamps = false;

        for (name, plugin) in self.item_type_plugins.iter() {
            let items = crate::plugin_api::ItemCollections::new(
                crate::renderer::item_plugins::plugin_collections_slice(frame, name),
            );
            if !items.is_empty() {
                plugin.surface_mask_readers(&items, &mut state.readers);
            }
        }
        // A reader with no bits lands nowhere whatever the mask holds, so it
        // gives no reason to stamp anything.
        for reader in &mut state.readers {
            *reader &= SURFACE_MASK_DEFAULT;
        }
        state.readers.retain(|reader| *reader != 0);
        state.readers.sort_unstable();
        state.readers.dedup();
        if state.readers.is_empty() {
            return;
        }

        let crate::SurfaceSubmission::Flat(ref surfaces) = frame.scene.surfaces;
        for item in surfaces.iter() {
            if item.settings.hidden {
                continue;
            }
            let value = surface_mask_value(&item.settings);
            if surface_mask_needs_stamp(value, &state.readers) {
                state.mesh_stamps.push(MeshStamp {
                    mesh_id: item.mesh_id,
                    instance: state.models.len() as u32,
                    value,
                    visibility_mask: item.settings.visibility_mask,
                });
                state.models.push(item.model);
            }
        }

        for (name, _) in self.item_type_plugins.iter() {
            let items = crate::plugin_api::ItemCollections::new(
                crate::renderer::item_plugins::plugin_collections_slice(frame, name),
            );
            let needs = (0..items.len()).any(|i| {
                let settings = items.item_settings(i);
                !settings.hidden
                    && surface_mask_needs_stamp(surface_mask_value(settings), &state.readers)
            });
            if needs {
                state.plugin_stamps = true;
                break;
            }
        }

        if !state.mesh_stamps.is_empty() {
            state.model_buf_handle = Some(state.model_buf.write(device, queue, &state.models));
            if state.pipeline.is_none() {
                state.pipeline = Some(build_mesh_pipeline(device, &self.resources));
            }
        }
    }

    /// Stamp the surface mask for one viewport. Runs after the opaque scene
    /// and its supersample resolve, so the stencil it writes is the one every
    /// later pass reads.
    pub(crate) fn encode_surface_mask(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        frame: &FrameData,
        vp_idx: usize,
    ) {
        let state = &self.surface_mask;
        if !state.active() {
            return;
        }
        let Some(slot) = self.viewport_slots.get(vp_idx) else {
            return;
        };
        let Some(slot_hdr) = slot.hdr.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("surface_mask_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: &slot_hdr.hdr_depth_view,
                depth_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                }),
                stencil_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_bind_group(0, &slot.camera_bind_group, &[]);

        let meshes = crate::resources::MeshDraw::new(&self.resources);
        if let (Some(pipeline), Some(models)) = (&state.pipeline, &state.model_buf_handle) {
            let cull_mask = frame.camera.cull_mask;
            let mut bound = false;
            for stamp in &state.mesh_stamps {
                if stamp.visibility_mask & cull_mask == 0 {
                    continue;
                }
                if !bound {
                    pass.set_pipeline(pipeline);
                    pass.set_vertex_buffer(1, models.slice(..));
                    bound = true;
                }
                pass.set_stencil_reference(stamp.value);
                meshes.draw_indexed_instance_range(
                    &mut pass,
                    stamp.mesh_id,
                    stamp.instance..stamp.instance + 1,
                );
            }
        }

        if state.plugin_stamps {
            let ctx = crate::plugin_api::SurfaceMaskContext {
                camera: &frame.camera.render_camera,
                viewport_index: vp_idx,
                frame_index: self.plugin_frame_index,
                meshes,
                readers: &state.readers,
            };
            for (name, plugin) in self.item_type_plugins.iter() {
                let items = crate::plugin_api::ItemCollections::new(
                    crate::renderer::item_plugins::plugin_collections_slice(frame, name),
                );
                if !items.is_empty() {
                    plugin.surface_mask(&mut pass, &ctx, &items);
                }
            }
        }
    }
}

/// The mesh stamp pipeline: position from the mesh arena's vertex, model
/// matrix from the per-instance buffer.
fn build_mesh_pipeline(
    device: &crate::gpu::Device,
    resources: &crate::resources::DeviceResources,
) -> crate::gpu::RenderPipeline {
    let shader = crate::resources::builders::wgsl_module(
        device,
        "surface_mask_mesh_shader",
        crate::resources::builders::wgsl_source!("surface_mask_mesh"),
    );
    let layout = crate::resources::builders::pipeline_layout(
        device,
        "surface_mask_mesh_layout",
        &[resources.shared_bindings().group0_layout],
    );
    // Position only; the stride is the full mesh vertex.
    let vertex_layout = crate::gpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::resources::Vertex>() as u64,
        step_mode: crate::gpu::VertexStepMode::Vertex,
        attributes: &[crate::gpu::VertexAttribute {
            format: crate::gpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    };
    const MODEL_ATTRS: [crate::gpu::VertexAttribute; 4] = crate::gpu::vertex_attr_array![
        1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4
    ];
    let model_layout = crate::gpu::VertexBufferLayout {
        array_stride: 64,
        step_mode: crate::gpu::VertexStepMode::Instance,
        attributes: &MODEL_ATTRS,
    };
    crate::resources::builders::build_surface_mask_pipeline(
        device,
        "surface_mask_mesh_pipeline",
        &layout,
        &shader,
        &[vertex_layout, model_layout],
        None,
    )
}
