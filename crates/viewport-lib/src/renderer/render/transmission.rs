//! The HDR transmission passes: copy the opaque scene, then draw the
//! transmissive meshes over it, each reading the copy for what lies behind.
//!
//! Transmissive meshes draw back to front without writing depth, and each
//! reads the copy taken before any of them drew: a transmissive surface seen
//! through another shows the environment, not the nearer glass.

use super::*;
use crate::renderer::render::pass_table::HdrPass;

impl ViewportRenderer {
    /// Whether this frame has a transmissive batch to draw.
    pub(super) fn has_transmissive_batches(&self) -> bool {
        self.instancing.batches.iter().any(|b| b.is_transmissive)
    }

    /// Size viewport `vp_idx`'s scene copy when the frame has transmissive
    /// meshes. Runs before the frame's encoder borrows the slot.
    pub(super) fn ensure_transmission_target(
        &mut self,
        device: &crate::gpu::Device,
        vp_idx: usize,
    ) {
        if !self.has_transmissive_batches() {
            return;
        }
        let slot = &mut self.viewport_slots[vp_idx];
        let Some(hdr) = slot.hdr.as_ref() else {
            return;
        };
        let [w, h] = hdr.scene_size;
        self.resources
            .transmission
            .ensure_target(device, &mut slot.transmission, w, h);
    }

    /// Copy the opaque scene colour and depth into the viewport's
    /// transmission target and build its mips.
    pub(super) fn hdr_opaque_copy(
        &mut self,
        device: &crate::gpu::Device,
        encoder: &mut crate::gpu::CommandEncoder,
        vp_idx: usize,
    ) {
        let slot = &mut self.viewport_slots[vp_idx];
        let (Some(hdr), Some(target)) = (slot.hdr.as_ref(), slot.transmission.as_mut()) else {
            return;
        };
        if self.resources.transmission.encode_mips(
            device,
            encoder,
            target,
            &hdr.hdr_view,
            &hdr.hdr_depth_only_view,
        ) {
            self.mark_pass_ran(vp_idx, HdrPass::OpaqueCopy);
        }
    }

    /// Draw the transmissive batches over the opaque scene, farthest first,
    /// depth-tested against the opaque depth without writing it.
    pub(super) fn hdr_transmission(
        &mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        vp_idx: usize,
        frame: &FrameData,
    ) {
        let resources = &self.resources;
        let slot = &self.viewport_slots[vp_idx];
        let (Some(hdr), Some(target), Some(empty)) = (
            slot.hdr.as_ref(),
            slot.transmission.as_ref(),
            resources.transmission.empty_group(),
        ) else {
            return;
        };
        let view = frame.camera.render_camera.view();
        let mut order: Vec<(f32, usize)> = self
            .instancing
            .batches
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_transmissive)
            .map(|(i, b)| {
                let depth = -view.transform_point3(glam::Vec3::from(b.sort_centre)).z;
                (depth, i)
            })
            .collect();
        // Farthest first; ties keep batch order.
        order.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("hdr_transmission_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: &hdr.hdr_view,
                resolve_target: None,
                ops: crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: &hdr.hdr_depth_view,
                depth_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_bind_group(0, &slot.camera_bind_group, &[]);
        pass.set_bind_group(2, empty, &[]);
        pass.set_bind_group(3, target.scene_group(), &[]);
        let mut bound = crate::renderer::types::BatchBound::default();
        for (_, i) in order {
            if crate::renderer::types::draw_batch(
                resources,
                &self.mesh_uniforms.material_bind_groups,
                &mut pass,
                &self.instancing.batches[i],
                &crate::renderer::types::BatchPass::Transmission,
                &mut bound,
            ) {
                self.frame_main_draw_commands
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        drop(pass);
        self.mark_pass_ran(vp_idx, HdrPass::Transmission);
    }
}
