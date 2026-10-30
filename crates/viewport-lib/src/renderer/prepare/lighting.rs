//! Per-frame lighting setup: lights, punctual shadow layers and the
//! clustered-light build.

use super::shadow_casters::{ShadowCasters, select_shadow_casters};
use super::*;
use crate::renderer::punctual_shadow_pool::{SLOT_LAYERS, ShadowKey, ShadowRequest};
use crate::renderer::shadow_state::LightingWrites;

/// Near plane of the punctual shadow projections, in world units.
const PUNCTUAL_SHADOW_NEAR: f32 = 0.1;
/// Widest half-angle a spot light's shadow frustum covers. A cone wider than
/// this is unshadowed past it.
const MAX_SPOT_SHADOW_HALF_ANGLE: f32 = 85.0 * std::f32::consts::PI / 180.0;

impl ViewportRenderer {
    /// Compute this frame's lighting: cull and pack lights, choose the shadow
    /// casters and place the punctual ones in the shadow array, and run the
    /// clustered-light build. Updates the cached shadow state and returns the
    /// punctual-layer data the shadow depth pass consumes, with the cascade
    /// caster for `fit_cascades` to fit.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_lighting(
        resources: &mut DeviceResources,
        shadow: &mut ShadowState,
        last_cluster_stats: &mut Option<crate::resources::gpu::clustered::ClusterStats>,
        last_frustum_culled_lights: &mut u32,
        last_lights_dropped: &mut u32,
        item_type_plugins: &crate::renderer::item_plugins::registry::ItemPluginRegistry,
        frame_index: u64,
        scene_fx: &SceneEffects<'_>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        origin: &crate::renderer::origin::OriginState,
        sink: &mut crate::renderer::SubmitSink,
        ts_query_set: Option<&crate::gpu::QuerySet>,
        ts_written_mask: &std::sync::atomic::AtomicU32,
    ) -> LightingFrame {
        let lighting = scene_fx.lighting;
        // Ranking culls in world space; everything after it works relative to
        // the origin, with the camera the GPU draws with.
        let camera = origin.camera(&frame.camera.render_camera);
        let plugin_lights = collect_plugin_lights(resources, item_type_plugins, frame, frame_index);
        let ranked = rank_lights(
            lighting
                .lights
                .iter()
                .chain(frame.scene.lights.iter())
                .chain(plugin_lights.iter()),
            frame,
            crate::resources::MAX_SCENE_LIGHTS,
        );
        *last_frustum_culled_lights = ranked.frustum_culled;
        *last_lights_dropped = ranked.dropped;
        let rel_sources: Vec<LightSource> = if origin.is_zero() {
            Vec::new()
        } else {
            ranked
                .lights
                .iter()
                .map(|(_, l)| rel_light_source(l, origin))
                .collect()
        };
        let rel_lights: Vec<(usize, &LightSource)>;
        let combined_lights = if origin.is_zero() {
            &ranked.lights
        } else {
            rel_lights = ranked
                .lights
                .iter()
                .zip(&rel_sources)
                .map(|((i, _), l)| (*i, l))
                .collect();
            &rel_lights
        };

        let light_count = combined_lights.len() as u32;

        // Build the per-light entries that get uploaded to the storage buffer.
        let mut lights_packed: Vec<SingleLightUniform> = combined_lights
            .iter()
            .map(|(_, src)| build_single_light_uniform(src))
            .collect();

        // Shadow casters: the directional light that takes the cascades, and
        // the point and spot lights given layers of the punctual array.
        let casters = {
            let settings_lights = lighting.lights.len();
            let scene_lights = frame.scene.lights.len();
            let key_of = |source: usize| {
                if source < settings_lights {
                    ShadowKey {
                        source: 0,
                        index: source as u32,
                    }
                } else if source < settings_lights + scene_lights {
                    ShadowKey {
                        source: 1,
                        index: (source - settings_lights) as u32,
                    }
                } else {
                    ShadowKey {
                        source: 2,
                        index: (source - settings_lights - scene_lights) as u32,
                    }
                }
            };
            let budget = if lighting.shadows.enabled {
                crate::resources::shadow::punctual_layers_for(
                    lighting.shadows.punctual_shadow_layers,
                    device.limits().max_texture_array_layers,
                )
            } else {
                0
            };
            shadow
                .punctual_pool
                .set_capacity((budget / SLOT_LAYERS) as usize);
            let pool = &shadow.punctual_pool;
            select_shadow_casters(
                combined_lights,
                key_of,
                camera.eye_position(),
                budget,
                |key| pool.holds(key),
            )
        };
        let (punctual_faces, punctual_layers_in_use) = Self::place_punctual_shadows(
            resources,
            shadow,
            &casters,
            combined_lights,
            &mut lights_packed,
            lighting,
            device,
            queue,
        );

        // The cascade caster's direction. The cascades themselves are fit by
        // `fit_cascades` once the instance bounds for this frame are known.
        let cascade_count = lighting.shadows.cascade_count.clamp(1, 4) as usize;
        // The atlas texture is sized from the setting; `prepare_shadow_pass`
        // (re)allocates it at `tile_size * 2` before rasterising cascades.
        let atlas_size = resources.shadow_atlas_size(lighting.shadows.atlas_resolution);
        let tile_size = atlas_size / 2;
        let csm_direction = casters
            .cascades
            .and_then(|i| match combined_lights[i].1.kind {
                LightKind::Directional { direction } => {
                    Some(glam::Vec3::from(direction).normalize())
                }
                _ => None,
            });
        if let Some(i) = casters.cascades {
            if lighting.shadows.enabled {
                lights_packed[i].shadow_kind = 1;
            }
        }

        // Upload lights uniform.
        // IBL fields from environment map settings.
        if resources.scene_tier.ibl.zones_dirty {
            crate::resources::material::environment::write_environment_zones(resources, queue);
        }
        let lit = scene_fx.environment.as_ref().filter(|env| {
            crate::resources::material::environment::select_lighting_environment(
                resources,
                device,
                queue,
                env.environment,
            )
        });
        let (ibl_enabled, ibl_intensity, ibl_rotation, ibl_diffuse_scale, ibl_specular_scale) =
            match lit {
                Some(env) => (
                    1u32,
                    crate::resources::material::environment::environment_multiplier(
                        &resources.scene_tier.ibl,
                        env.environment,
                        lighting.environment_intensity,
                    ),
                    env.rotation,
                    env.diffuse_scale,
                    env.specular_scale,
                ),
                None => (0, 0.0, 0.0, 0.0, 0.0),
            };

        let debug_vis_mode = frame.effects.debug.debug_vis.pack_mode();
        let debug_vis_scale = if frame.effects.debug.debug_vis.active {
            frame.effects.debug.debug_vis.scale.max(0.001)
        } else {
            1.0
        };
        // The lit pipelines normally compile without the debug-vis block, so
        // they do not carry its registers on every draw; swap in the full
        // shader variant while debug vis is active, and back when it stops.
        let keep_debug_block = debug_vis_mode != 0 || resources.force_debug_vis_shaders;
        resources.set_debug_vis_shaders(device, keep_debug_block);

        let lights_uniform = lights_header(
            lighting,
            light_count,
            resources
                .group0_profile
                .as_ref()
                .expect("the scene tier has a lit group 0"),
            &HeaderExtras {
                debug_vis_mode,
                debug_vis_scale,
                debug_vis_split_x: if frame.effects.debug.debug_vis.active {
                    frame.effects.debug.debug_vis.split_x.clamp(0.0, 1.0)
                } else {
                    0.5
                },
                ibl_enabled,
                ibl_intensity,
                ibl_rotation,
                ibl_diffuse_scale,
                ibl_specular_scale,
                env_zone_count: resources.scene_tier.ibl.env_zone_count,
                custom_data_base: (resources.scene_tier.lighting.custom_data_base_bytes / 16)
                    as u32,
            },
        );
        if LightingWrites::changed(&mut shadow.written.lights, lights_uniform) {
            queue.write_buffer(
                &resources.scene_tier.lighting.uniform_buf,
                0,
                bytemuck::cast_slice(&[lights_uniform]),
            );
        }
        // Upload the per-light array to the storage buffer at binding 13.
        // Slots past `count` are left as-is; the shader bounds its loop on
        // `lights_uniform.count` so stale tail entries are never sampled.
        if !lights_packed.is_empty()
            && LightingWrites::slice_changed(&mut shadow.written.light_storage, &lights_packed)
        {
            queue.write_buffer(
                &resources.scene_tier.lighting.storage_buf,
                0,
                bytemuck::cast_slice(&lights_packed),
            );
        }

        // Clustered shading : transform the survivors to view space, upload
        // the active-light array, refresh the cluster grid uniform, then
        // dispatch clear + build. Lit fragment shaders read the resulting
        // per-cluster light index ranges from bindings 14 / 15 / 16 of the
        // camera bind group.
        {
            use crate::resources::gpu::clustered::{
                ActiveLightView, CLUSTER_COUNT, CLUSTER_X_TILES, CLUSTER_Y_TILES, CLUSTER_Z_SLICES,
                ClusterGridUniform,
            };

            let view_mat = camera.view();
            // Perspective parameters : derive tan(half_fov) from the proj
            // matrix diagonal (proj[0][0] = 1/tan(half_fov_x), with the wgpu
            // RH perspective matrices used here).
            let proj = frame.camera.render_camera.projection();
            let p00 = proj.col(0).x.max(1e-6);
            let p11 = proj.col(1).y.max(1e-6);
            let tan_half_fov_x = 1.0 / p00;
            let tan_half_fov_y = 1.0 / p11;
            let near = frame.camera.render_camera.depth_range().0.max(0.01);
            let far = frame.camera.render_camera.depth_range().1.max(near + 0.01);

            // Fit the cluster depth range to the punctual lights' actual
            // reach instead of the camera far plane. With the default
            // far of 1000 the log-uniform slices are hundreds of metres
            // deep at distance, so every distant cluster intersects most
            // of the scene's lights and far fragments iterate lights that
            // cannot reach them. Clamping far to max(light depth + range)
            // concentrates all slices where lights exist. Fragments
            // beyond the fitted far clamp to the last slice (see
            // cluster_index_for), whose lights all attenuate to zero at
            // that distance, so shading is unchanged; only the list
            // tightness improves.
            let mut light_far = 0.0f32;
            for (_, l) in combined_lights.iter() {
                let (position, range) = match l.kind {
                    LightKind::Point {
                        position, range, ..
                    } => (position, range),
                    LightKind::Spot {
                        position, range, ..
                    } => (position, range),
                    LightKind::Directional { .. } => continue,
                    _ => continue,
                };
                let view_depth = -view_mat.transform_point3(glam::Vec3::from(position)).z;
                light_far = light_far.max(view_depth + range);
            }
            let far = if light_far > 0.0 {
                light_far.clamp(near * 2.0, far)
            } else {
                far
            };

            // Spend the z-slice budget where geometry is. Log-uniform slices
            // from the camera near plane put roughly half the slices inside
            // the first few metres of a metre-scale scene, leaving the
            // distant slabs several times thicker than they need to be, so
            // far clusters collect lights that cannot reach most of their
            // fragments. Instead the first slice covers everything up to
            // far/32 (fragments nearer than the cluster near plane clamp to
            // slice 0, whose AABB in cluster_build.wgsl extends to depth 0)
            // and the remaining slices run log-uniform from there, which
            // keeps the slice-to-slice depth ratio constant regardless of
            // scene scale.
            let cluster_near = (far / 32.0).max(near);
            let far = far.max(cluster_near * 1.01);

            let active_count = lights_packed.len() as u32;
            // Skip the cluster build entirely for scenes with only a handful
            // of lights : straight per-fragment iteration is cheaper than the
            // lookup-table indirection at that scale. A consumer-set debug
            // override also forces the fallback path so the two can be A/B'd
            // for correctness checks.
            // The cluster grid is built in compute; without it the frame shades
            // every light directly and the grid is never read, so it needs no
            // clear either.
            let compute = resources.capabilities.compute;
            let want_clusters = compute
                && !frame.effects.debug.force_cluster_fallback
                && active_count > crate::resources::gpu::clustered::SMALL_N_THRESHOLD;
            // The clear is owed only once a build has written the grid, so a
            // viewport whose light count never reaches the cluster threshold
            // composes neither compute pipeline. While they compile on a
            // worker the frame takes the per-light fallback.
            if compute && (want_clusters || resources.scene_tier.clustered.grid_dirty()) {
                if resources
                    .scene_tier
                    .clustered
                    .ensure_pipelines(device, &resources.pipeline_compiler)
                {
                    resources.camera_bind_groups_dirty = true;
                }
            }
            let use_clusters = want_clusters && resources.scene_tier.clustered.pipelines_ready();
            let fallback_flag = if use_clusters { 0.0 } else { 1.0 };
            let grid_uniform = ClusterGridUniform {
                dimensions: [
                    CLUSTER_X_TILES,
                    CLUSTER_Y_TILES,
                    CLUSTER_Z_SLICES,
                    CLUSTER_COUNT,
                ],
                depth: [
                    cluster_near,
                    far,
                    (far / cluster_near).ln(),
                    active_count as f32,
                ],
                screen: [
                    frame.camera.render_camera.aspect().max(0.01),
                    1.0,
                    fallback_flag,
                    0.0,
                ],
                proj_scale: [tan_half_fov_x, tan_half_fov_y, 0.0, 0.0],
                view: view_mat.to_cols_array_2d(),
            };
            if LightingWrites::changed(&mut shadow.written.cluster_grid, grid_uniform) {
                resources
                    .scene_tier
                    .clustered
                    .write_grid_uniform(queue, &grid_uniform);
            }

            // Build the view-space ActiveLight array. Order matches
            // `lights_packed` / `light_storage_buf` so light_indices[j] from
            // the build pass is a valid index into lights_storage too.
            let active_lights: Vec<ActiveLightView> = combined_lights
                .iter()
                .map(|(_, l)| {
                    let (view_pos, range, light_type, view_dir, cos_outer) = match l.kind {
                        LightKind::Directional { direction } => {
                            let world_dir = glam::Vec3::from(direction).normalize_or_zero();
                            let view_dir =
                                view_mat.transform_vector3(world_dir).normalize_or_zero();
                            (glam::Vec3::ZERO, f32::INFINITY, 0u32, view_dir, 1.0)
                        }
                        LightKind::Point {
                            position, range, ..
                        } => {
                            let view_pos = view_mat.transform_point3(glam::Vec3::from(position));
                            (view_pos, range, 1u32, glam::Vec3::ZERO, 1.0)
                        }
                        LightKind::Spot {
                            position,
                            direction,
                            range,
                            outer_angle,
                            ..
                        } => {
                            let view_pos = view_mat.transform_point3(glam::Vec3::from(position));
                            let view_dir = view_mat
                                .transform_vector3(glam::Vec3::from(direction))
                                .normalize_or_zero();
                            (view_pos, range, 2u32, view_dir, outer_angle.cos())
                        }
                        _ => unreachable!("unhandled LightKind variant"),
                    };
                    ActiveLightView {
                        view_pos_range: [view_pos.x, view_pos.y, view_pos.z, range],
                        type_pad: [light_type, 0, 0, 0],
                        spot_data: [view_dir.x, view_dir.y, view_dir.z, cos_outer],
                    }
                })
                .collect();
            if LightingWrites::slice_changed(&mut shadow.written.active_lights, &active_lights) {
                resources
                    .scene_tier
                    .clustered
                    .write_active_lights(queue, &active_lights);
            }

            // Pass 0 when below threshold so dispatch_frame runs the clear
            // (keeping the buffers in a defined state) but skips the build.
            let build_count = if use_clusters { active_count } else { 0 };
            let cluster_ts = if build_count > 0 {
                if ts_query_set.is_some() {
                    ts_written_mask.fetch_or(
                        1 << crate::renderer::GPU_TS_CLUSTER,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                ts_query_set
            } else {
                None
            };
            // Nothing to build and nothing to clear: no command buffer.
            if compute && (build_count > 0 || resources.scene_tier.clustered.grid_dirty()) {
                let mut encoder =
                    device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                        label: Some("cluster_frame_encoder"),
                    });
                resources.scene_tier.clustered.dispatch_frame(
                    &mut encoder,
                    build_count,
                    cluster_ts,
                );
                sink.push(encoder.finish());
            }

            // Optional host readback for the debug stats panel. Synchronous;
            // off by default.
            if frame.effects.debug.cluster_stats_request {
                let stats = resources.scene_tier.clustered.read_stats(
                    device,
                    queue,
                    active_count,
                    !use_clusters,
                );
                *last_cluster_stats = Some(stats);
            }
        }

        LightingFrame {
            punctual_faces,
            punctual_layers_in_use,
            tile_size,
            atlas_size,
            cascade_count,
            csm_direction,
            cascade_view_projs: [glam::Mat4::IDENTITY; 4],
            effective_cascade_count: 0,
        }
    }
}

impl ViewportRenderer {
    /// Place this frame's punctual casters in the shadow array: give each its
    /// layers, write its `shadow_kind` and `shadow_layer` (and a spot's
    /// matrix) into `lights_packed`, size the array, and upload the per-layer
    /// uniforms. Returns the layers to render and how many are in use.
    #[allow(clippy::too_many_arguments)]
    fn place_punctual_shadows(
        resources: &mut DeviceResources,
        shadow: &mut ShadowState,
        casters: &ShadowCasters,
        lights: &[(usize, &LightSource)],
        lights_packed: &mut [SingleLightUniform],
        lighting: &LightingSettings,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) -> (Vec<PunctualShadowFace>, u32) {
        let requests: Vec<ShadowRequest> = casters
            .punctual
            .iter()
            .map(|c| ShadowRequest {
                key: c.key,
                cube: c.cube,
            })
            .collect();
        let placed = shadow.punctual_pool.place(&requests);
        if casters.punctual.is_empty() {
            return (Vec::new(), 0);
        }
        let resolution = crate::resources::shadow::punctual_resolution_for(
            lighting.shadows.punctual_shadow_resolution,
            resources.capabilities.texture_dimension_2d,
        );
        // The array is a placeholder, or too small, until a frame places a
        // light in it; it has to exist before the uniform upload below and
        // any bind group that samples it. A new array holds nothing.
        let layers = shadow.punctual_pool.used_slots() as u32 * SLOT_LAYERS;
        if resources.ensure_punctual_shadow_array(device, layers, resolution) {
            shadow.punctual_pool.invalidate();
        }
        resources.ensure_point_shadow_pipeline(device);

        // Cube faces in layer order (forward, up): +X, -X, +Y, -Y, +Z, -Z.
        const CUBE_FACES: [(glam::Vec3, glam::Vec3); 6] = [
            (glam::Vec3::X, glam::Vec3::NEG_Y),
            (glam::Vec3::NEG_X, glam::Vec3::NEG_Y),
            (glam::Vec3::Y, glam::Vec3::Z),
            (glam::Vec3::NEG_Y, glam::Vec3::NEG_Z),
            (glam::Vec3::Z, glam::Vec3::NEG_Y),
            (glam::Vec3::NEG_Z, glam::Vec3::NEG_Y),
        ];
        let mut faces = Vec::new();
        let mut in_use = 0;
        for (c, base) in casters.punctual.iter().zip(&placed) {
            let Some(base) = *base else {
                continue;
            };
            let packed = &mut lights_packed[c.light];
            let mut light_hash = crate::renderer::shader_hashes::fnv1a_hash(&[]);
            match lights[c.light].1.kind {
                LightKind::Point {
                    position, range, ..
                } => {
                    let light_pos = glam::Vec3::from(position);
                    fnv_mix(&mut light_hash, &[2]);
                    fnv_mix(
                        &mut light_hash,
                        bytemuck::cast_slice(&[position[0], position[1], position[2], range]),
                    );
                    packed.shadow_kind = 2;
                    packed.shadow_layer = base as i32;
                    packed.point_shadow_near = PUNCTUAL_SHADOW_NEAR;
                    let proj = punctual_projection(std::f32::consts::FRAC_PI_2, range);
                    for (f, (forward, up)) in CUBE_FACES.iter().enumerate() {
                        let view = glam::Mat4::look_at_rh(light_pos, light_pos + *forward, *up);
                        faces.push(PunctualShadowFace {
                            layer: base + f as u32,
                            base,
                            light_hash,
                            view_proj: proj * view,
                            light_pos,
                            range,
                        });
                    }
                    in_use += SLOT_LAYERS;
                }
                LightKind::Spot {
                    position,
                    direction,
                    range,
                    outer_angle,
                    ..
                } => {
                    let light_pos = glam::Vec3::from(position);
                    let view_proj = spot_shadow_matrix(
                        light_pos,
                        glam::Vec3::from(direction),
                        outer_angle,
                        range,
                        resolution,
                    );
                    fnv_mix(&mut light_hash, &[3]);
                    fnv_mix(
                        &mut light_hash,
                        bytemuck::cast_slice(&view_proj.to_cols_array()),
                    );
                    fnv_mix(&mut light_hash, &range.to_le_bytes());
                    packed.light_view_proj = view_proj.to_cols_array_2d();
                    packed.shadow_kind = 3;
                    packed.shadow_layer = base as i32;
                    packed.point_shadow_near = PUNCTUAL_SHADOW_NEAR;
                    faces.push(PunctualShadowFace {
                        layer: base,
                        base,
                        light_hash,
                        view_proj,
                        light_pos,
                        range,
                    });
                    in_use += 1;
                }
                _ => {}
            }
        }

        // Per-layer uniforms: view_proj, light_pos and range, padded to the
        // 256-byte dynamic-offset stride.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PunctualFaceUniform {
            view_proj: [[f32; 4]; 4], // 0..64
            // Packed vec4: xyz = light_pos, w = range.
            light_pos: [f32; 4], // 64..80
            _pad: [f32; 44],     // pad to 256
        }
        for fc in &faces {
            let entry = PunctualFaceUniform {
                view_proj: fc.view_proj.to_cols_array_2d(),
                light_pos: [fc.light_pos.x, fc.light_pos.y, fc.light_pos.z, fc.range],
                _pad: [0.0; 44],
            };
            queue.write_buffer(
                &resources.scene_tier.shadow.point_face_buf,
                u64::from(fc.layer) * crate::resources::shadow::PUNCTUAL_FACE_STRIDE,
                bytemuck::cast_slice(&[entry]),
            );
        }
        (faces, in_use)
    }
}

/// The Y-flipped perspective a punctual shadow layer is rendered with, the
/// orientation a hardware cube-map face expects. The flip mirrors winding, so
/// the pipeline treats CW as front-facing (see `build_shadow_point_pipeline`).
fn punctual_projection(fov_y: f32, range: f32) -> glam::Mat4 {
    glam::Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0))
        * glam::Mat4::perspective_rh(
            fov_y,
            1.0,
            PUNCTUAL_SHADOW_NEAR,
            range.max(PUNCTUAL_SHADOW_NEAR + 0.01),
        )
}

/// A spot light's shadow view-projection: the cone's outer half-angle plus a
/// two-texel guard for the filter taps, up to [`MAX_SPOT_SHADOW_HALF_ANGLE`],
/// looking down the spot's axis.
pub(super) fn spot_shadow_matrix(
    position: glam::Vec3,
    direction: glam::Vec3,
    outer_angle: f32,
    range: f32,
    resolution: u32,
) -> glam::Mat4 {
    let guard = (2.0 / resolution.max(1) as f32).atan();
    let half = (outer_angle + guard).clamp(guard, MAX_SPOT_SHADOW_HALF_ANGLE);
    let dir = direction.normalize();
    let up = if dir.z.abs() > 0.99 {
        glam::Vec3::X
    } else {
        glam::Vec3::Z
    };
    let view = glam::Mat4::look_at_rh(position, position + dir, up);
    punctual_projection(2.0 * half, range) * view
}

/// Fold `bytes` into an FNV-1a hash.
pub(super) fn fnv_mix(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= b as u64;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

/// `src` with a point or spot light's position made relative to the origin.
fn rel_light_source(
    src: &LightSource,
    origin: &crate::renderer::origin::OriginState,
) -> LightSource {
    let mut l = src.clone();
    match &mut l.kind {
        LightKind::Point { position, .. } | LightKind::Spot { position, .. } => {
            *position = origin.rel(glam::Vec3::from(*position)).to_array();
        }
        _ => {}
    }
    l
}

/// `u` with a point or spot light's position made relative to the origin.
pub(super) fn rel_light(
    mut u: SingleLightUniform,
    origin: &crate::renderer::origin::OriginState,
) -> SingleLightUniform {
    if u.light_type != 0 {
        u.pos_or_dir = origin.rel(glam::Vec3::from(u.pos_or_dir)).to_array();
    }
    u
}

/// Convert a `LightSource` to `SingleLightUniform`, with no shadow assigned.
pub(super) fn build_single_light_uniform(src: &LightSource) -> SingleLightUniform {
    let identity = glam::Mat4::IDENTITY.to_cols_array_2d();
    match &src.kind {
        LightKind::Directional { direction } => SingleLightUniform {
            light_view_proj: identity,
            pos_or_dir: *direction,
            light_type: 0,
            colour: src.colour.to_linear_rgb(),
            intensity: src.intensity,
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            shadow_kind: 0,
            spot_direction: [0.0, -1.0, 0.0],
            shadow_layer: -1,
            point_shadow_near: 0.1,
            radius: 0.0,
            channel_mask: src.channel_mask,
            _reserved: 0,
        },
        LightKind::Point {
            position,
            range,
            radius,
        } => SingleLightUniform {
            light_view_proj: identity,
            pos_or_dir: *position,
            light_type: 1,
            colour: src.colour.to_linear_rgb(),
            intensity: src.intensity,
            range: *range,
            inner_angle: 0.0,
            outer_angle: 0.0,
            shadow_kind: 0,
            spot_direction: [0.0, -1.0, 0.0],
            shadow_layer: -1,
            point_shadow_near: 0.1,
            radius: *radius,
            channel_mask: src.channel_mask,
            _reserved: 0,
        },
        LightKind::Spot {
            position,
            direction,
            range,
            inner_angle,
            outer_angle,
            radius,
        } => SingleLightUniform {
            light_view_proj: identity,
            pos_or_dir: *position,
            light_type: 2,
            colour: src.colour.to_linear_rgb(),
            intensity: src.intensity,
            range: *range,
            inner_angle: *inner_angle,
            outer_angle: *outer_angle,
            shadow_kind: 0,
            spot_direction: *direction,
            shadow_layer: -1,
            point_shadow_near: 0.1,
            radius: *radius,
            channel_mask: src.channel_mask,
            _reserved: 0,
        },
        _ => unreachable!("unhandled LightKind variant"),
    }
}

/// Lights contributed by item types. An emissive scatter volume becomes a
/// virtual point light this way, so nearby opaque surfaces pick up warm
/// illumination without the consumer authoring a matching light by hand.
/// Appended after the consumer's own lights, and dropped like any other if
/// the per-frame cap is hit.
pub(super) fn collect_plugin_lights(
    resources: &DeviceResources,
    item_type_plugins: &crate::renderer::item_plugins::registry::ItemPluginRegistry,
    frame: &FrameData,
    frame_index: u64,
) -> Vec<LightSource> {
    let ctx = crate::plugin_api::LightContext {
        resources,
        camera: &frame.camera.render_camera,
        frame_index,
    };
    item_type_plugins
        .iter()
        .filter_map(|(name, plugin)| {
            let items = crate::plugin_api::ItemCollections::new(
                crate::renderer::item_plugins::plugin_collections_slice(frame, name),
            );
            Some(plugin.contribute_lights(&items, &ctx))
        })
        .flatten()
        .collect()
}

/// A frame's lights after the frustum cull and the cap, in upload order.
pub(super) struct RankedLights<'a> {
    /// Each light with its index in the iterator `rank_lights` was given,
    /// which stays put when the ranking reorders the list.
    pub(super) lights: Vec<(usize, &'a LightSource)>,
    /// Point and spot lights the frustum cull removed.
    pub(super) frustum_culled: u32,
    /// Lights that survived the cull but fell past the cap.
    pub(super) dropped: u32,
}

/// Cull `lights` against the camera frustum and rank what is left, keeping at
/// most `cap`. Each survivor comes back with its index in `lights`.
pub(super) fn rank_lights<'a>(
    lights: impl Iterator<Item = &'a LightSource>,
    frame: &FrameData,
    cap: usize,
) -> RankedLights<'a> {
    // CPU per-frame frustum cull. Sphere-vs-frustum for point lights,
    // cone-vs-frustum for spot lights, trivial-true for directional.
    // Dropping off-screen lights here keeps the cluster build pass and
    // the per-fragment iteration both bounded by what is actually visible.
    // Directional lights always survive : they affect every fragment.
    let cull_frustum = crate::camera::frustum::Frustum::from_view_proj(
        &frame.camera.render_camera.culling_view_proj(),
    );
    let mut frustum_culled = 0u32;
    let raw_lights: Vec<(usize, &LightSource)> = lights
        .enumerate()
        .filter(|(_, l)| match l.kind {
            LightKind::Directional { .. } => true,
            LightKind::Point {
                position, range, ..
            } => {
                let keep = !cull_frustum.cull_sphere(glam::Vec3::from(position), range);
                if !keep {
                    frustum_culled += 1;
                }
                keep
            }
            LightKind::Spot {
                position,
                direction,
                range,
                outer_angle,
                ..
            } => {
                let axis = glam::Vec3::from(direction).normalize_or_zero();
                if axis.length_squared() < 1e-8 {
                    return true;
                }
                let keep =
                    !cull_frustum.cull_cone(glam::Vec3::from(position), axis, outer_angle, range);
                if !keep {
                    frustum_culled += 1;
                }
                keep
            }
            _ => false,
        })
        .collect();

    // Rank the light array: the cascaded-shadow caster pinned at slot 0 (the
    // first directional light that casts, else the first directional), the
    // rest ordered by
    // `importance * proximity_weight` descending (stable, so equal ranks
    // keep submission order). The order does two jobs: over the cap the
    // tail is dropped, and the cluster build keeps each cell's lights in
    // array order, so a cluster that overflows its fixed slice drops the
    // lowest-priority lights deterministically. Directional lights are
    // treated as infinitely close (proximity_weight = 1). Shading itself is
    // order-independent.
    let camera_pos = frame.camera.render_camera.eye_position();
    let directional = |l: &LightSource| matches!(l.kind, LightKind::Directional { .. });
    let directional_first = raw_lights
        .iter()
        .position(|(_, l)| directional(l) && l.cast_shadows)
        .or_else(|| raw_lights.iter().position(|(_, l)| directional(l)));
    let mut out: Vec<(usize, &LightSource)> = Vec::with_capacity(cap.min(raw_lights.len()));
    if let Some(i) = directional_first {
        out.push(raw_lights[i]);
    }
    let mut rest: Vec<(f32, (usize, &LightSource))> = raw_lights
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != directional_first)
        .map(|(_, &(source, l))| {
            let proximity = match l.kind {
                LightKind::Directional { .. } => 1.0,
                LightKind::Point {
                    position, range, ..
                }
                | LightKind::Spot {
                    position, range, ..
                } => {
                    let d = (glam::Vec3::from(position) - camera_pos).length();
                    // Light is fully relevant within `range`; fades to ~0 beyond 4x range.
                    (range / (range + d.max(0.0))).clamp(0.0, 1.0)
                }
                _ => 1.0,
            };
            (l.importance.max(0.0) * proximity, (source, l))
        })
        .collect();
    rest.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = cap.saturating_sub(out.len());
    let dropped = rest.len().saturating_sub(take) as u32;
    out.extend(rest.into_iter().take(take).map(|(_, l)| l));
    RankedLights {
        lights: out,
        frustum_culled,
        dropped,
    }
}

/// The lights header fields that come from state only some group 0 profiles
/// have: the environment, the debug views and the indirect-light offsets.
pub(super) struct HeaderExtras {
    pub(super) debug_vis_mode: u32,
    pub(super) debug_vis_scale: f32,
    pub(super) debug_vis_split_x: f32,
    pub(super) ibl_enabled: u32,
    pub(super) ibl_intensity: f32,
    pub(super) ibl_rotation: f32,
    pub(super) ibl_diffuse_scale: f32,
    pub(super) ibl_specular_scale: f32,
    pub(super) env_zone_count: u32,
    pub(super) custom_data_base: u32,
}

impl Default for HeaderExtras {
    /// Debug views off and no environment.
    fn default() -> Self {
        Self {
            debug_vis_mode: 0,
            debug_vis_scale: 1.0,
            debug_vis_split_x: 0.5,
            ibl_enabled: 0,
            ibl_intensity: 0.0,
            ibl_rotation: 0.0,
            ibl_diffuse_scale: 0.0,
            ibl_specular_scale: 0.0,
            env_zone_count: 0,
            custom_data_base: 0,
        }
    }
}

/// The lights header (group 0 binding 3) for `count` lights, on every tier.
/// Shadows count as enabled only where the profile binds real shadow maps.
pub(super) fn lights_header(
    lighting: &LightingSettings,
    count: u32,
    profile: &crate::resources::group0::Group0Profile,
    extras: &HeaderExtras,
) -> LightsUniform {
    LightsUniform {
        count,
        shadow_bias: lighting.shadows.bias,
        shadows_enabled: u32::from(lighting.shadows.enabled && profile.shadows),
        debug_vis_mode: extras.debug_vis_mode,
        sky_colour: lighting.sky_colour.to_linear_rgb(),
        hemisphere_intensity: lighting.hemisphere_intensity,
        ground_colour: lighting.ground_colour.to_linear_rgb(),
        debug_vis_scale: extras.debug_vis_scale,
        ibl_enabled: if profile.ibl { extras.ibl_enabled } else { 0 },
        ibl_intensity: extras.ibl_intensity,
        ibl_rotation: extras.ibl_rotation,
        ibl_diffuse_scale: extras.ibl_diffuse_scale,
        debug_vis_split_x: extras.debug_vis_split_x,
        env_zone_count: extras.env_zone_count,
        ibl_specular_scale: extras.ibl_specular_scale,
        custom_data_base: extras.custom_data_base,
        ao_flags: ao_flags(lighting),
        _pad_ao: [0; 3],
    }
}

/// `ao_flags` bit: apply the multi-bounce AO fit. Matches `AO_FLAG_MULTI_BOUNCE`
/// in scene_lighting.wgsl.
const AO_FLAG_MULTI_BOUNCE: u32 = 1;
/// `ao_flags` bit: material AO shadows directional lights. Matches
/// `AO_FLAG_MICRO_SHADOW` in scene_lighting.wgsl.
const AO_FLAG_MICRO_SHADOW: u32 = 2;

/// The lights header's `ao_flags` word for `lighting`.
fn ao_flags(lighting: &LightingSettings) -> u32 {
    let mut flags = 0;
    if lighting.multi_bounce_ao {
        flags |= AO_FLAG_MULTI_BOUNCE;
    }
    if lighting.ao_micro_shadowing {
        flags |= AO_FLAG_MICRO_SHADOW;
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::{CameraFrame, RenderCamera, SceneFrame, SurfaceSubmission};

    fn frame() -> FrameData {
        let cf = CameraFrame::new(RenderCamera::default(), [256.0, 256.0]);
        let sf = SceneFrame::new(SurfaceSubmission::Flat(std::sync::Arc::from(Vec::new())));
        FrameData::new(cf, sf)
    }

    fn point(importance: f32) -> LightSource {
        let mut l = LightSource::point_candela([0.0, 0.0, 0.0], Candela(10.0), 1.0e6, 0.0);
        l.importance = importance;
        l
    }

    #[test]
    fn ranked_lights_keep_their_source_index() {
        let lights = [point(1.0), LightSource::default(), point(5.0), point(2.0)];
        let ranked = rank_lights(lights.iter(), &frame(), 16);
        let order: Vec<usize> = ranked.lights.iter().map(|(i, _)| *i).collect();
        // The directional light is pinned first, the points follow by importance.
        assert_eq!(order, vec![1, 2, 3, 0]);
        for (i, l) in &ranked.lights {
            assert!(std::ptr::eq(*l, &lights[*i]));
        }
    }

    #[test]
    fn the_casting_directional_light_ranks_first() {
        let mut quiet = LightSource::default();
        quiet.cast_shadows = false;
        let lights = [point(1.0), quiet, LightSource::default()];
        let ranked = rank_lights(lights.iter(), &frame(), 16);
        assert_eq!(ranked.lights[0].0, 2);
    }

    /// The spot frustum covers the cone plus its tap guard, so a point on the
    /// cone's rim lands just inside the map's edge.
    #[test]
    fn the_spot_frustum_frames_the_cone() {
        let outer = 0.6f32;
        let vp = spot_shadow_matrix(
            glam::Vec3::new(0.0, 0.0, 3.0),
            glam::Vec3::NEG_Z,
            outer,
            10.0,
            1024,
        );
        let rim = glam::Vec3::new(2.0 * outer.tan(), 0.0, 1.0);
        let clip = vp * rim.extend(1.0);
        let edge = (clip.x / clip.w).abs().max((clip.y / clip.w).abs());
        assert!(edge < 1.0 && edge > 0.99, "rim at ndc {edge}");
        let axis = vp * glam::Vec4::new(0.0, 0.0, 1.0, 1.0);
        assert!((axis.x / axis.w).abs() < 1e-5 && (axis.y / axis.w).abs() < 1e-5);
        // A cone past the cap is framed at the cap.
        let wide = spot_shadow_matrix(glam::Vec3::ZERO, glam::Vec3::X, 1.5, 10.0, 1024);
        let p00 = wide.row(0).truncate().length();
        assert!((1.0 / p00 - MAX_SPOT_SHADOW_HALF_ANGLE.tan()).abs() < 1e-3);
    }
}
