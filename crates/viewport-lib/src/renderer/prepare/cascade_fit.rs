//! Directional shadow cascades: the depth range they cover, the per-cascade
//! light matrices, and the shadow atlas uniform the lit shaders read.

use super::*;
use crate::renderer::shadow_state::LightingWrites;

impl ViewportRenderer {
    /// Fit the cascade caster's cascades to the frame camera, fill in the
    /// cascade fields of `light`, and upload the atlas and cascade uniforms.
    ///
    /// Cascades are fit to `frame.camera`, not to any per-viewport camera, so
    /// every split viewport shares one shadow atlas.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fit_cascades(
        resources: &DeviceResources,
        shadow: &mut ShadowState,
        viewport_slots: &[ViewportSlot],
        scene_fx: &SceneEffects<'_>,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        light: &mut LightingFrame,
    ) {
        let lighting = scene_fx.lighting;
        let camera = &frame.camera.render_camera;
        let (camera_near, camera_far) = camera.depth_range();
        let cascade_count = light.cascade_count;

        // When no consumer override is set, derive the auto extent from the
        // camera's far plane, clamped so very-long-far cameras don't blow out
        // the footprint and very-short-far ones don't lose precision.
        let (shadow_center, shadow_extent) = match lighting.shadows.extent_override {
            Some(extent) => (glam::Vec3::ZERO, extent),
            None => (
                glam::Vec3::ZERO,
                (camera_far.max(1.0) * 0.25).clamp(20.0, 200.0),
            ),
        };

        let dist = camera.distance();
        let shadow_near = (dist * 0.1).max(camera_near);
        let shadow_far = (dist * 1.5).max(shadow_extent).min(camera_far);
        let cascade_splits =
            compute_cascade_splits(shadow_near, shadow_far, cascade_count as u32, 0.75);

        let mut cascade_view_projs = [glam::Mat4::IDENTITY; 4];
        // Distance-based splits for fragment shader cascade selection.
        let mut cascade_split_distances = [0.0f32; 4];

        // A camera with no view gets one map over the scene extent instead.
        let use_csm = camera.view != glam::Mat4::IDENTITY;
        if let Some(light_dir) = light.csm_direction {
            if use_csm {
                for i in 0..cascade_count {
                    let split_near = if i == 0 {
                        camera_near.max(0.01)
                    } else {
                        cascade_splits[i - 1]
                    };
                    let split_far = cascade_splits[i];
                    cascade_view_projs[i] = compute_cascade_matrix(
                        light_dir,
                        camera.view,
                        camera.fov_y(),
                        camera.aspect(),
                        split_near,
                        split_far,
                        light.tile_size as f32,
                    );
                    cascade_split_distances[i] = split_far;
                }
            } else {
                cascade_view_projs[0] =
                    compute_shadow_matrix(light_dir, shadow_center, shadow_extent);
                cascade_split_distances[0] = camera_far;
            }
        }
        // No casting directional light: no cascades are rendered and the
        // shader's CSM sample short-circuits on `cascade_count == 0`.
        let effective_cascade_count = match light.csm_direction {
            None => 0,
            Some(_) if use_csm => cascade_count,
            Some(_) => 1,
        };

        // Cache shadow stats and log when cascade splits change.
        if cascade_split_distances != shadow.last_logged_cascade_splits {
            tracing::debug!(
                cascade_count = effective_cascade_count,
                splits = ?cascade_split_distances,
                shadow_extent = shadow_extent,
                "cascade splits changed"
            );
            shadow.last_logged_cascade_splits = cascade_split_distances;
        }
        shadow.last_cascade_count = effective_cascade_count as u32;
        shadow.last_cascade_splits = cascade_split_distances;
        shadow.last_shadow_extent = shadow_extent;
        shadow.last_shadow_atlas_resolution = light.atlas_size;
        shadow.last_contact_shadow_active = frame.effects.post_process.contact_shadows.enabled;

        // Atlas tile layout (2x2 grid of `tile_size` tiles filling the texture):
        // [0] = top-left, [1] = top-right, [2] = bottom-left, [3] = bottom-right
        let atlas_rects: [[f32; 4]; 8] = [
            [0.0, 0.0, 0.5, 0.5], // cascade 0
            [0.5, 0.0, 1.0, 0.5], // cascade 1
            [0.0, 0.5, 0.5, 1.0], // cascade 2
            [0.5, 0.5, 1.0, 1.0], // cascade 3
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4], // unused slots
        ];

        // Upload ShadowAtlasUniform (binding 5).
        {
            let mut vp_data = [[0.0f32; 4]; 16]; // 4 mat4s flattened
            for c in 0..4 {
                let cols = cascade_view_projs[c].to_cols_array_2d();
                for row in 0..4 {
                    vp_data[c * 4 + row] = cols[row];
                }
            }
            let shadow_atlas_uniform = ShadowAtlasUniform {
                cascade_view_proj: vp_data,
                cascade_splits: cascade_split_distances,
                cascade_count: effective_cascade_count as u32,
                // The backing texture size, after clamping: shader texel maths
                // (PCF radius, texel_world, blocker-search coords) is relative
                // to the real texture.
                atlas_size: light.atlas_size as f32,
                // Tier values match the dispatch in csm.wgsl.
                shadow_filter: match lighting.shadows.filter {
                    ShadowFilter::Pcf => 0,
                    ShadowFilter::Pcss => 1,
                    ShadowFilter::Hard => 2,
                    ShadowFilter::PcfHigh => 3,
                    ShadowFilter::PcssFast => 4,
                    _ => 0,
                },
                pcss_light_radius: lighting.shadows.pcss_light_radius,
                atlas_rects,
            };
            // The shared buffer and every per-viewport copy, only when the
            // contents changed. A slot made later is seeded from the cache
            // below, so it never misses a write.
            if LightingWrites::changed(&mut shadow.written.shadow_info, shadow_atlas_uniform) {
                queue.write_buffer(
                    &resources.scene_tier.shadow.info_buf,
                    0,
                    bytemuck::cast_slice(&[shadow_atlas_uniform]),
                );
                for slot in viewport_slots {
                    queue.write_buffer(
                        &slot.shadow_info_buf,
                        0,
                        bytemuck::cast_slice(&[shadow_atlas_uniform]),
                    );
                }
            }
            // Cache for viewport slots created later in the frame: a new slot's
            // buffer is seeded from this so its first frame does not sample
            // shadows through a zeroed uniform (see ensure_viewport_slot).
            shadow.last_shadow_atlas_uniform = shadow_atlas_uniform;
        }

        // Cache for ground plane ShadowOnly mode.
        shadow.last_cascade0_shadow_mat = cascade_view_projs[0];

        // Upload all cascade matrices to the shadow uniform buffer before the
        // shadow pass. wgpu batches write_buffer calls before the command
        // buffer, so every cascade slot is written up front; the cascade loop
        // then selects per-slot via dynamic offset.
        const SHADOW_SLOT_STRIDE: u64 = 256;
        for c in 0..4usize {
            let m = cascade_view_projs[c].to_cols_array_2d();
            if LightingWrites::changed(&mut shadow.written.cascades[c], m) {
                queue.write_buffer(
                    &resources.scene_tier.shadow.uniform_buf,
                    c as u64 * SHADOW_SLOT_STRIDE,
                    bytemuck::cast_slice(&m),
                );
            }
        }

        light.cascade_view_projs = cascade_view_projs;
        light.effective_cascade_count = effective_cascade_count;
    }
}

/// A single orthographic shadow map for a directional light, covering
/// `shadow_extent` around `shadow_center`. Used in place of the cascades when
/// the camera has no view matrix to fit them to.
fn compute_shadow_matrix(
    direction: glam::Vec3,
    shadow_center: glam::Vec3,
    shadow_extent: f32,
) -> glam::Mat4 {
    let dir = direction.normalize();
    let light_up = if dir.z.abs() > 0.99 {
        glam::Vec3::X
    } else {
        glam::Vec3::Z
    };
    let light_pos = shadow_center + dir * shadow_extent * 2.0;
    let light_view = glam::Mat4::look_at_rh(light_pos, shadow_center, light_up);
    let light_proj = glam::Mat4::orthographic_rh(
        -shadow_extent,
        shadow_extent,
        -shadow_extent,
        shadow_extent,
        0.01,
        shadow_extent * 5.0,
    );
    light_proj * light_view
}
