//! Directional shadow cascades: the depth range they cover, the per-cascade
//! light matrices, and the shadow atlas uniform the lit shaders read.

use super::*;
use crate::renderer::shadow_state::LightingWrites;

impl ViewportRenderer {
    /// Fit the cascade caster's cascades to the frame camera, fill in the
    /// cascade fields of `light`, and upload the atlas and cascade uniforms.
    ///
    /// Cascades are fit to `frame.camera`, not to any per-viewport camera, so
    /// every split viewport shares one shadow atlas. The depth range they
    /// cover comes from this frame's mesh instance bounds (see
    /// [`fitted_range`]), so it runs after the instanced batches are built.
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
        let bounds = match light.csm_direction {
            Some(_) => {
                if shadow
                    .cascade_bounds
                    .refresh(instancing.aabbs_gen, &instancing.cached_aabbs)
                {
                    stats.shadow_bounds_rebuilds = 1;
                }
                shadow
                    .cascade_bounds
                    .casters
                    .zip(shadow.cascade_bounds.scene)
            }
            None => None,
        };

        // The old automatic extent, kept for frames with nothing batched to
        // fit to: a fraction of the far plane, clamped so very long or very
        // short far planes keep a usable footprint.
        let legacy_extent = (camera_far.max(1.0) * 0.25).clamp(20.0, 200.0);
        let fallback = fallback_range(camera_near, camera_far, legacy_extent);

        let fitted = light
            .csm_direction
            .zip(bounds)
            .and_then(|(light_dir, (casters, scene))| {
                let ground = (frame.effects.ground_plane.mode != GroundPlaneMode::None)
                    .then_some(frame.effects.ground_plane.height);
                fitted_range(camera, casters, scene, light_dir, ground)
            });
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
        let cascade_splits =
            compute_cascade_splits(shadow_near, shadow_far, cascade_count as u32, 0.75);

        // The single map a camera with no view matrix gets: the override
        // around the origin, or the casters' light-space half-extent around
        // their centre.
        let (shadow_center, shadow_extent) = match (
            lighting.shadows.extent_override,
            light.csm_direction,
            bounds,
        ) {
            (Some(extent), _, _) => (glam::Vec3::ZERO, extent),
            (None, Some(light_dir), Some((casters, _))) => {
                (casters.centre(), casters.light_space_half_extent(light_dir))
            }
            (None, _, _) => (glam::Vec3::ZERO, legacy_extent),
        };

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

/// World-space axis-aligned bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) min: glam::Vec3,
    pub(crate) max: glam::Vec3,
}

impl Bounds {
    fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn intersection(self, other: Self) -> Option<Self> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        min.cmple(max).all().then_some(Self { min, max })
    }

    fn centre(self) -> glam::Vec3 {
        (self.min + self.max) * 0.5
    }

    fn corners(self) -> [glam::Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            glam::Vec3::new(a.x, a.y, a.z),
            glam::Vec3::new(b.x, a.y, a.z),
            glam::Vec3::new(a.x, b.y, a.z),
            glam::Vec3::new(b.x, b.y, a.z),
            glam::Vec3::new(a.x, a.y, b.z),
            glam::Vec3::new(b.x, a.y, b.z),
            glam::Vec3::new(a.x, b.y, b.z),
            glam::Vec3::new(b.x, b.y, b.z),
        ]
    }

    /// Half the widest side of these bounds seen from a directional light
    /// shining along `-light_dir`, in the light basis `compute_shadow_matrix`
    /// uses.
    fn light_space_half_extent(self, light_dir: glam::Vec3) -> f32 {
        let dir = light_dir.normalize();
        let up = if dir.z.abs() > 0.99 {
            glam::Vec3::X
        } else {
            glam::Vec3::Z
        };
        let view = glam::Mat4::look_at_rh(dir, glam::Vec3::ZERO, up);
        let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
        for c in self.corners() {
            let p = view.transform_point3(c);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        ((hi - lo) * 0.5).max_element().max(1e-3)
    }
}

/// The caster and scene bounds of the mesh instances, cached against
/// `InstancingState::aabbs_gen` so a frame whose instance bounds did not
/// change reuses them.
#[derive(Default)]
pub(crate) struct CascadeBounds {
    key: Option<u64>,
    /// Union of the instances that cast shadows.
    pub(crate) casters: Option<Bounds>,
    /// Union of every instance.
    pub(crate) scene: Option<Bounds>,
}

impl CascadeBounds {
    /// Rebuild the unions from `aabbs` unless they were built at `generation`.
    /// Returns whether they were rebuilt.
    pub(crate) fn refresh(
        &mut self,
        generation: u64,
        aabbs: &[crate::resources::InstanceAabb],
    ) -> bool {
        if self.key == Some(generation) {
            return false;
        }
        self.key = Some(generation);
        self.casters = None;
        self.scene = None;
        for a in aabbs {
            let b = Bounds {
                min: a.min.into(),
                max: a.max.into(),
            };
            if !b.min.is_finite() || !b.max.is_finite() {
                continue;
            }
            self.scene = Some(self.scene.map_or(b, |s| s.union(b)));
            if a.cast_shadows != 0 {
                self.casters = Some(self.casters.map_or(b, |s| s.union(b)));
            }
        }
        true
    }
}

/// The view-depth range the cascades cover when there is nothing to fit
/// them to. `typical` stands in for the orbit radius the formula was first
/// written against: the log-space midpoint of the depth range.
fn fallback_range(camera_near: f32, camera_far: f32, extent: f32) -> (f32, f32) {
    let typical = (camera_near.max(1e-6) * camera_far).sqrt();
    let near = (typical * 0.1).max(camera_near);
    let far = (typical * 1.5).max(extent).min(camera_far);
    (near, far)
}

/// The view-depth range in which a fragment can be in a cast shadow.
///
/// Casters shadow what lies behind them along the light, so the casters'
/// bounds are swept away from the light (`light_dir` points towards it) far
/// enough to cross the scene, or to reach the ground plane at `ground` when
/// one is drawn, and clipped to the receivers: the scene bounds, widened to
/// the ground plane. The range is that region's depth span along the view,
/// clamped to `[2 near, far]` of the camera. `None` when no part of it is in
/// front of the camera within the far plane.
pub(crate) fn fitted_range(
    camera: &RenderCamera,
    casters: Bounds,
    scene: Bounds,
    light_dir: glam::Vec3,
    ground: Option<f32>,
) -> Option<(f32, f32)> {
    let (camera_near, camera_far) = camera.depth_range();
    let dir = light_dir.normalize_or_zero();
    let mut receivers = scene;
    let mut sweep = (scene.max - scene.min).length();
    if let Some(height) = ground {
        receivers.min = glam::Vec3::new(f32::MIN, f32::MIN, receivers.min.z.min(height));
        receivers.max = glam::Vec3::new(f32::MAX, f32::MAX, receivers.max.z.max(height));
        if dir.z > 1e-3 {
            sweep = sweep.max((casters.max.z - height) / dir.z);
        }
    }
    let sweep = sweep.min(camera_far * 2.0);
    let shifted = Bounds {
        min: casters.min - dir * sweep,
        max: casters.max - dir * sweep,
    };
    let region = casters.union(shifted).intersection(receivers)?;

    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for c in region.corners() {
        let depth = -camera.view.transform_point3(c).z;
        lo = lo.min(depth);
        hi = hi.max(depth);
    }
    let near_limit = camera_near * 2.0;
    if hi <= near_limit || lo >= camera_far {
        return None;
    }
    let near = lo.max(near_limit);
    let far = hi.min(camera_far);
    (far > near * 1.001).then_some((near, far))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn bounds(min: [f32; 3], max: [f32; 3]) -> Bounds {
        Bounds {
            min: min.into(),
            max: max.into(),
        }
    }

    /// Eye at the origin looking along +Y, Z up.
    fn camera() -> RenderCamera {
        let cam = crate::camera::Camera {
            center: Vec3::Y * 0.01,
            distance: 0.01,
            orientation: glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            ..Default::default()
        };
        RenderCamera::from_camera(&cam)
    }

    #[test]
    fn range_spans_the_casters_and_their_shadows_within_the_scene() {
        let scene = bounds([-5.0, 0.0, 0.0], [5.0, 40.0, 3.0]);
        let casters = bounds([-1.0, 9.0, 0.0], [1.0, 11.0, 2.0]);
        // Light from straight overhead: shadows fall under the casters.
        let (near, far) = fitted_range(&camera(), casters, scene, Vec3::Z, None).unwrap();
        assert!(
            (near - 9.0).abs() < 1e-3 && (far - 11.0).abs() < 1e-3,
            "{near} {far}"
        );
        // A low light from -Y throws them on along +Y, as far as the scene goes.
        let low = Vec3::new(0.0, -1.0, 0.2);
        let (near, far) = fitted_range(&camera(), casters, scene, low, None).unwrap();
        assert!(
            (near - 9.0).abs() < 1e-3 && (far - 40.0).abs() < 1e-3,
            "{near} {far}"
        );
    }

    #[test]
    fn a_ground_plane_receives_shadows_past_the_scene() {
        let scene = bounds([-1.0, 9.0, 0.0], [1.0, 11.0, 2.0]);
        let low = Vec3::new(0.0, -1.0, 0.2).normalize();
        let (_, far) = fitted_range(&camera(), scene, scene, low, None).unwrap();
        assert!((far - 11.0).abs() < 1e-3, "{far}");
        // The top of the caster throws its shadow 2 / 0.2 = 10 m along +Y.
        let (_, far) = fitted_range(&camera(), scene, scene, low, Some(0.0)).unwrap();
        assert!((far - 21.0).abs() < 1e-2, "{far}");
    }

    #[test]
    fn casters_behind_the_camera_fit_nothing() {
        let scene = bounds([-1.0, -11.0, 0.0], [1.0, -9.0, 2.0]);
        assert!(fitted_range(&camera(), scene, scene, Vec3::Z, None).is_none());
    }

    #[test]
    fn bounds_rebuild_once_per_generation() {
        let aabb = |x: f32, cast: u32| crate::resources::InstanceAabb {
            min: [x, 0.0, 0.0],
            batch_index: 0,
            max: [x + 1.0, 1.0, 1.0],
            cast_shadows: cast,
        };
        let mut b = CascadeBounds::default();
        assert!(b.refresh(0, &[aabb(0.0, 1), aabb(5.0, 0)]));
        assert_eq!(b.casters, Some(bounds([0.0; 3], [1.0; 3])));
        assert_eq!(b.scene, Some(bounds([0.0; 3], [6.0, 1.0, 1.0])));
        assert!(!b.refresh(0, &[]));
        assert!(b.refresh(1, &[aabb(5.0, 0)]));
        assert_eq!(b.casters, None);
    }
}
