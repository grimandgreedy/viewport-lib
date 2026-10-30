//! Directional shadow cascades: the depth range they cover, the per-cascade
//! light matrices, and the shadow atlas uniform the lit shaders read.

use super::cascade_bounds::{fallback_range, fitted_range};
use super::*;
use crate::renderer::shadow_state::LightingWrites;

impl ViewportRenderer {
    /// Fit the cascade caster's cascades to the frame camera, fill in the
    /// cascade fields of `light`, and upload the atlas and cascade uniforms.
    ///
    /// Cascades are fit to `frame.camera`, not to any per-viewport camera, so
    /// every split viewport shares one shadow atlas. The depth range they
    /// cover comes from this frame's mesh instance bounds (see
    /// `cascade_bounds`), so it runs after the instanced batches are built.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fit_cascades(
        resources: &DeviceResources,
        shadow: &mut ShadowState,
        instancing: &InstancingState,
        item_type_plugins: &crate::renderer::item_plugins::registry::ItemPluginRegistry,
        viewport_slots: &[ViewportSlot],
        scene_fx: &SceneEffects<'_>,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        light: &mut LightingFrame,
        stats: &mut crate::renderer::stats::FrameStats,
    ) {
        let lighting = scene_fx.lighting;
        let camera = &frame.camera.render_camera;
        let (camera_near, camera_far) = camera.depth_range();
        let cascade_count = light.cascade_count;

        stats.shadow_bounds_rebuilds = 0;
        let bounds = &mut shadow.cascade_bounds;
        if light.csm_direction.is_some()
            && bounds.refresh(instancing.aabbs_gen, &instancing.cached_aabbs)
        {
            stats.shadow_bounds_rebuilds = 1;
        }

        // The old automatic extent, kept for frames with nothing batched to
        // fit to: a fraction of the far plane, clamped so very long or very
        // short far planes keep a usable footprint.
        let legacy_extent = (camera_far.max(1.0) * 0.25).clamp(20.0, 200.0);
        let fallback = fallback_range(camera_near, camera_far, legacy_extent);

        let ground = (frame.effects.ground_plane.mode != GroundPlaneMode::None)
            .then_some(frame.effects.ground_plane.height);
        let fitted = match light.csm_direction {
            Some(light_dir) if bounds.casters.is_some() => {
                fitted_range(camera, bounds, light_dir, ground)
            }
            _ => None,
        };
        // Item-type plugins cast into the cascades too, and their bounds are
        // not known here, so with any plugin items present the fitted range
        // only ever widens the fallback one.
        let plugin_items = item_type_plugins.iter().any(|(name, _)| {
            !crate::plugin_api::ItemCollections::new(
                crate::renderer::item_plugins::plugin_collections_slice(frame, name),
            )
            .is_empty()
        });
        let (shadow_near, shadow_far) = match fitted {
            Some((near, far)) if plugin_items => (near.min(fallback.0), far.max(fallback.1)),
            Some(range) => range,
            None => fallback,
        };
        // A set shadow distance replaces the far end, whatever the scene.
        let distance = lighting
            .shadows
            .shadow_distance
            .filter(|d| d.is_finite() && *d > 0.0);
        let (shadow_near, shadow_far) = match distance {
            Some(d) => {
                let far = d.min(camera_far);
                (shadow_near.min(far * 0.5), far)
            }
            None => (shadow_near, shadow_far),
        };
        // Shadows fade out where the cascades stop short of the scene: at a
        // set distance, or at the end of the fallback range. A fitted range
        // already reaches every shadow in view, and fading its far end
        // would light the farthest shadowed receivers.
        let ends_short = distance.is_some() || fitted.is_none_or(|(_, far)| shadow_far > far);
        let cascade_splits = compute_cascade_splits(
            shadow_near,
            shadow_far,
            cascade_count as u32,
            lighting.shadows.cascade_split,
        );
        // With depth clamp the cascade casters land in the map however far
        // in front of a cascade they are, so its depth range need not reach
        // them.
        let depth_clamp = resources.capabilities.depth_clip_control;

        // The single map a camera with no view matrix gets: the override
        // around the origin, or the casters' light-space half-extent around
        // their centre.
        let (shadow_center, shadow_extent) = match (
            lighting.shadows.extent_override,
            light.csm_direction,
            bounds.casters,
        ) {
            (Some(extent), _, _) => (glam::Vec3::ZERO, extent),
            (None, Some(light_dir), Some(casters)) => {
                (casters.centre(), casters.light_space_half_extent(light_dir))
            }
            (None, _, _) => (glam::Vec3::ZERO, legacy_extent),
        };

        let mut cascade_view_projs = [glam::Mat4::IDENTITY; 4];
        // Distance-based splits for fragment shader cascade selection.
        let mut cascade_split_distances = [0.0f32; 4];

        // A camera with no view gets one map over the scene extent instead.
        let use_csm = camera.view() != glam::Mat4::IDENTITY;
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
                        camera.view(),
                        camera.fov_y(),
                        camera.aspect(),
                        split_near,
                        split_far,
                        light.tile_size as f32,
                        |light_view, rect| {
                            (!depth_clamp)
                                .then(|| bounds.caster_max_z(light_view, rect))
                                .flatten()
                        },
                    )
                    .view_proj;
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
        let atlas_rects: [[f32; 4]; 4] = [
            [0.0, 0.0, 0.5, 0.5], // cascade 0
            [0.5, 0.0, 1.0, 0.5], // cascade 1
            [0.0, 0.5, 0.5, 1.0], // cascade 2
            [0.5, 0.5, 1.0, 1.0], // cascade 3
        ];
        // Where the cascades end along the view, and the share of that
        // distance they fade out over (0: no fade and no cut). The single
        // no-view map has no distance cut. The last two lanes are reserved
        // for physically sized soft shadows (sun angular radius, widest
        // penumbra).
        let fade_fraction = match lighting.shadows.fade_fraction {
            f if ends_short && f > 0.0 => f.min(1.0),
            _ => 0.0,
        };
        let range = if use_csm && effective_cascade_count > 0 {
            [
                cascade_split_distances[effective_cascade_count - 1],
                fade_fraction,
                0.0,
                0.0,
            ]
        } else {
            [0.0; 4]
        };

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
                range,
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

#[cfg(test)]
mod tests {
    use crate::renderer::shadows::{CascadeFit, compute_cascade_matrix, compute_cascade_splits};
    use glam::{Mat4, Vec3};

    fn sun() -> Vec3 {
        Vec3::new(0.3, -0.5, 0.8).normalize()
    }

    /// The second cascade of a camera at `eye` looking along +Y.
    fn fit(eye: Vec3, caster_max_z: Option<f32>) -> CascadeFit {
        let view = Mat4::look_at_rh(eye, eye + Vec3::Y, Vec3::Z);
        compute_cascade_matrix(sun(), view, 0.8, 1.5, 10.0, 40.0, 2048.0, |_, _| {
            caster_max_z
        })
    }

    #[test]
    fn cascade_depth_range_steps_rather_than_drifts() {
        let mut prev = fit(Vec3::new(0.0, -20.0, 1.7), None);
        let mut moves = 0;
        for i in 1..=400 {
            let t = i as f32;
            let eye = Vec3::new(t * 0.013, -20.0 + t * 0.007, 1.7 + t * 0.001);
            let f = fit(eye, None);
            assert_eq!(f.z_step, prev.z_step, "step {i}");
            for (a, b) in [(prev.min_z, f.min_z), (prev.max_z, f.max_z)] {
                let steps = (b - a) / f.z_step;
                assert!(
                    steps == 0.0 || (steps.abs() - 1.0).abs() < 1e-3,
                    "step {i}: {a} -> {b} is {steps} quanta of {}",
                    f.z_step
                );
                moves += (steps != 0.0) as u32;
            }
            assert_eq!((f.min_z / f.z_step).fract(), 0.0);
            assert_eq!((f.max_z / f.z_step).fract(), 0.0);
            prev = f;
        }
        assert!(moves > 0, "the walk crossed no quantum");
    }

    #[test]
    fn casters_in_front_of_a_cascade_widen_its_range_toward_the_light() {
        let eye = Vec3::new(0.0, -20.0, 1.7);
        let own = fit(eye, None);
        // A caster just above the cascade's own range: the margin covers it.
        assert_eq!(fit(eye, Some(own.max_z - 6.0)).max_z, own.max_z);
        // A tall caster well in front of it is taken in.
        let tall = own.max_z + 40.0;
        let wide = fit(eye, Some(tall));
        assert!(wide.max_z >= tall, "{} < {tall}", wide.max_z);
        assert!(wide.min_z <= own.min_z);
    }

    #[test]
    fn fraction_splits_fall_at_the_given_fractions() {
        let close = |a: [f32; 4], b: [f32; 4]| {
            assert!(
                a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-4),
                "{a:?} != {b:?}"
            );
        };
        let s = compute_cascade_splits(
            2.0,
            102.0,
            4,
            crate::CascadeSplit::Fractions([0.125, 0.25, 0.5]),
        );
        close(s, [14.5, 27.0, 52.0, 102.0]);
        // Out of order fractions never make a cascade run backwards.
        let s = compute_cascade_splits(
            0.0,
            100.0,
            4,
            crate::CascadeSplit::Fractions([0.5, 0.25, 2.0]),
        );
        close(s, [50.0, 50.0, 100.0, 100.0]);
        // Two cascades read the first fraction only.
        let s = compute_cascade_splits(
            0.0,
            100.0,
            2,
            crate::CascadeSplit::Fractions([0.3, 0.6, 0.9]),
        );
        close(s, [30.0, 100.0, 100.0, 100.0]);
    }
}
