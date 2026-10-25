//! Per-frame prepare on the forward tier: the lights header and capped light
//! list of its group 0, and each viewport's camera and clip uniforms.

use super::*;

impl ViewportRenderer {
    /// Rank the frame's lights into the forward light list and write it with
    /// its header. This tier's profile binds no shadow maps or environment,
    /// and the debug views are off.
    pub(super) fn prepare_forward_lights(
        &mut self,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        scene_fx: &SceneEffects<'_>,
    ) {
        let (Some(forward), Some(profile)) = (
            self.resources.forward_tier.as_deref(),
            self.resources.group0_profile.as_ref(),
        ) else {
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
            profile
                .uniform_light_list
                .unwrap_or(crate::resources::MAX_SCENE_LIGHTS),
        );
        self.last_frustum_culled_lights = ranked.frustum_culled;
        self.last_lights_dropped = ranked.dropped;
        let packed: Vec<crate::resources::SingleLightUniform> = ranked
            .lights
            .iter()
            .map(|l| lighting::build_single_light_uniform(l, glam::Vec3::ZERO, 0.0, false))
            .collect();

        let header = lighting::lights_header(
            lighting,
            packed.len() as u32,
            profile,
            &lighting::HeaderExtras::default(),
        );
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
