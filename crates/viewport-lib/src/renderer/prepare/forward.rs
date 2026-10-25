//! Per-frame prepare on the forward tier: the lights header and capped light
//! list of its group 0, and each viewport's camera and clip uniforms.

use super::*;
use crate::resources::FORWARD_MAX_LIGHTS;

impl ViewportRenderer {
    /// Rank the frame's lights into the forward light list and write it with
    /// its header. Shadows, image-based lighting and debug views are off on
    /// this tier, so the header carries only the count and the hemisphere.
    pub(super) fn prepare_forward_lights(
        &mut self,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        scene_fx: &SceneEffects<'_>,
    ) {
        let Some(forward) = self.resources.forward_tier.as_deref() else {
            return;
        };
        let lighting = scene_fx.lighting;
        let plugin_lights = lighting::collect_plugin_lights(
            &self.resources,
            &self.item_type_plugins,
            frame,
            self.plugin_frame_index,
        );
        let ranked = lighting::rank_lights(
            lighting
                .lights
                .iter()
                .chain(frame.scene.lights.iter())
                .chain(plugin_lights.iter()),
            frame,
            FORWARD_MAX_LIGHTS,
        );
        self.last_frustum_culled_lights = ranked.frustum_culled;
        self.last_lights_dropped = ranked.dropped;
        let packed: Vec<crate::resources::SingleLightUniform> = ranked
            .lights
            .iter()
            .map(|l| lighting::build_single_light_uniform(l, glam::Vec3::ZERO, 0.0, false))
            .collect();

        let mut header: crate::resources::LightsUniform = bytemuck::Zeroable::zeroed();
        header.count = packed.len() as u32;
        header.sky_colour = lighting.sky_colour.to_linear_rgb();
        header.hemisphere_intensity = lighting.hemisphere_intensity;
        header.ground_colour = lighting.ground_colour.to_linear_rgb();
        header.debug_vis_scale = 1.0;
        header.debug_vis_split_x = 0.5;
        queue.write_buffer(
            &forward.lights_header_buf,
            0,
            bytemuck::cast_slice(&[header]),
        );
        if !packed.is_empty() {
            queue.write_buffer(&forward.light_list_buf, 0, bytemuck::cast_slice(&packed));
        }
    }

    /// The viewport's camera and clip uniforms, which its forward group 0
    /// binds.
    pub(super) fn prepare_viewport_forward(
        &mut self,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
    ) {
        if self.resources.forward_tier.is_none() {
            return;
        }
        let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) else {
            return;
        };
        let (clip_uniform, clip_vols_uniform) =
            viewport_interaction::clip_uniforms(frame, viewport_fx);
        queue.write_buffer(
            &slot.clip_planes_buf,
            0,
            bytemuck::cast_slice(&[clip_uniform]),
        );
        queue.write_buffer(
            &slot.clip_volume_buf,
            0,
            bytemuck::cast_slice(&[clip_vols_uniform]),
        );
        // The forward tier draws through the LDR path, which keeps the lit
        // term in [0, 1].
        let camera_uniform = frame.camera.render_camera.camera_uniform();
        queue.write_buffer(&slot.camera_buf, 0, bytemuck::cast_slice(&[camera_uniform]));
    }
}
