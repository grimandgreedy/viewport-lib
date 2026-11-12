//! The painter mask pass: each frame, the item types that report masks draw
//! them into the viewport's mask target, and the slots' layer ranges go into
//! the uniform the painter content tests against. The light layers a plugin
//! reports are bound beside them.

use super::*;

impl ViewportRenderer {
    /// Collect this viewport's painter masks, draw them into its mask target
    /// and write the slots' ranges; with none, point the mask groups at the
    /// empty fallback. Takes the first light layers a plugin reports. Builds
    /// the groups the painter draws bind.
    pub(super) fn prepare_painter_masks(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        let vp_idx = frame.camera.viewport_index;
        let mut reports: Vec<crate::plugin_api::PainterMask> = Vec::new();
        let mut drawers: Vec<&'static str> = Vec::new();
        let mut lights: Option<crate::resources::painter_mask::FrameLights> = None;
        for (name, plugin) in self.item_type_plugins.iter() {
            if !self.plugins_with_items.contains(name) {
                continue;
            }
            if lights.is_none()
                && let Some(l) = plugin.painter_light_layers(vp_idx)
            {
                lights = Some(crate::resources::painter_mask::FrameLights {
                    view: l.view,
                    ranges: l.ranges.iter().map(|r| [*r.start(), *r.end()]).collect(),
                    scale: l.scale,
                    directions: l.directions,
                });
            }
            let before = reports.len();
            plugin.painter_masks(vp_idx, &mut reports);
            if reports.len() > before {
                drawers.push(name);
            }
        }
        let mut slots: [Option<(u32, u32)>; 4] = [None; 4];
        for mask in &reports {
            let slot = &mut slots[(mask.slot & 3) as usize];
            let (lo, hi) = (*mask.layers.start(), *mask.layers.end());
            *slot = Some(match *slot {
                Some((a, b)) => (a.min(lo), b.max(hi)),
                None => (lo, hi),
            });
        }
        let active = !drawers.is_empty();
        let state = &mut self.viewport_slots[vp_idx].painter_masks;
        state.active = active;
        state.set_lights(queue, lights);
        queue.write_buffer(
            &state.uniform,
            0,
            bytemuck::cast_slice(&crate::resources::painter_mask::painter_mask_uniform(
                &slots,
            )),
        );
        if active {
            let ppp = frame.scene_ppp();
            let size = frame.scene_size();
            let device_size = [
                (size[0] * ppp).round().max(1.0) as u32,
                (size[1] * ppp).round().max(1.0) as u32,
            ];
            let target = state.target(device, device_size).clone();
            let mut encoder =
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("painter_mask_encoder"),
                });
            {
                let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(any(wgpu29, wgpu30))]
                    multiview_mask: None,
                    label: Some("painter_mask_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: &target,
                        resolve_target: None,
                        ops: crate::gpu::Operations {
                            load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                            store: crate::gpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                let slot = &self.viewport_slots[vp_idx];
                let ctx = crate::plugin_api::PainterMaskContext {
                    camera: &frame.camera.render_camera,
                    viewport_size: glam::Vec2::from(frame.scene_size()),
                    viewport_index: vp_idx,
                    frame_index: self.plugin_frame_index,
                };
                for name in &drawers {
                    let Some(plugin) = self.item_type_plugins.get(name) else {
                        continue;
                    };
                    pass.set_bind_group(0, &slot.camera_bind_group, &[]);
                    let items = crate::plugin_api::ItemCollections::new(
                        crate::renderer::item_plugins::plugin_collections_slice(frame, name),
                    );
                    plugin.paint_painter_masks(&mut pass, &ctx, &items);
                }
            }
            sink.push(encoder.finish());
        }
        let resources = &self.resources;
        let state = &mut self.viewport_slots[vp_idx].painter_masks;
        state.bind_group(device, &resources.painter_mask);
        state.light_pick_group(device, &resources.painter_mask);
        if resources.scene_tier.is_built() {
            state.ldr_depth_read_group(
                device,
                &resources.painter_mask,
                &resources.scene_tier.material.depth_read_bgl,
                &resources.scene_tier.material.depth_read_sampler,
            );
        }
    }
}
