//! Per-frame lighting and shadow setup (lights, point-shadow slots, CSM
//! cascades, shadow atlas uniform, clustered-light build).

use super::*;

impl ViewportRenderer {
    /// Compute this frame's lighting: cull and pack lights, allocate point-light
    /// shadow slots, fit the directional CSM cascades, build the shadow atlas
    /// uniform, and run the clustered-light build. Updates the cached shadow
    /// state and returns the transient cascade / point-shadow data the shadow
    /// depth pass consumes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_lighting(
        resources: &mut DeviceResources,
        shadow: &mut ShadowState,
        last_cluster_stats: &mut Option<crate::resources::gpu::clustered::ClusterStats>,
        last_frustum_culled_lights: &mut u32,
        viewport_slots: &[ViewportSlot],
        scene_fx: &SceneEffects<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
        ts_query_set: Option<&wgpu::QuerySet>,
        ts_written_mask: &std::sync::atomic::AtomicU32,
    ) -> LightingFrame {
        let lighting = scene_fx.lighting;
        // Compute scene center / extent for shadow framing.
        //
        // When no consumer override is set, derive the auto extent from the
        // camera's far plane: a fraction of the view distance the camera
        // already reports as visible, clamped so very-long-far cameras don't
        // blow out the cascade footprint and very-short-far ones don't lose
        // contact-shadow precision. The earlier fixed 20.0 default broke FPS
        // cameras with scenes deeper than 20 units, since shadow coverage
        // capped at that distance and casters past it dropped out of every
        // cascade.
        let (shadow_center, shadow_extent) = if let Some(extent) = lighting.shadow_extent_override {
            (glam::Vec3::ZERO, extent)
        } else {
            let camera_far = frame.camera.render_camera.far.max(1.0);
            let auto_extent = (camera_far * 0.25).clamp(20.0, 200.0);
            (glam::Vec3::ZERO, auto_extent)
        };

        /// Build a light-space view-projection matrix for shadow mapping.
        fn compute_shadow_matrix(
            kind: &LightKind,
            shadow_center: glam::Vec3,
            shadow_extent: f32,
        ) -> glam::Mat4 {
            match kind {
                LightKind::Directional { direction } => {
                    let dir = glam::Vec3::from(*direction).normalize();
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
                LightKind::Point { position, range } => {
                    let pos = glam::Vec3::from(*position);
                    let to_center = (shadow_center - pos).normalize();
                    let light_up = if to_center.z.abs() > 0.99 {
                        glam::Vec3::X
                    } else {
                        glam::Vec3::Z
                    };
                    let light_view = glam::Mat4::look_at_rh(pos, shadow_center, light_up);
                    let light_proj =
                        glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, *range);
                    light_proj * light_view
                }
                LightKind::Spot {
                    position,
                    direction,
                    range,
                    ..
                } => {
                    let pos = glam::Vec3::from(*position);
                    let dir = glam::Vec3::from(*direction).normalize();
                    let look_target = pos + dir;
                    let up = if dir.z.abs() > 0.99 {
                        glam::Vec3::X
                    } else {
                        glam::Vec3::Z
                    };
                    let light_view = glam::Mat4::look_at_rh(pos, look_target, up);
                    let light_proj =
                        glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, *range);
                    light_proj * light_view
                }
            }
        }

        /// Derive virtual point lights from emissive scatter volumes so
        /// nearby opaque surfaces receive warm light from "fire-like" volumes.
        ///
        /// Cheap approximation: one virtual `Point` light per emissive
        /// volume, placed at the shape's centre. Intensity scales with
        /// `emission_strength * density`; range scales with the shape's
        /// longest axis. For `ColourSource::Ramp`, the colour is sampled
        /// from the CPU-side LUT at the "hot end" of the ramp (the point
        /// where emission contributes most), then multiplied by the tint.
        fn derive_scatter_volume_virtual_lights(
            items: &[crate::renderer::types::ScatterVolumeItem],
            colourmaps_cpu: &[[[u8; 4]; 256]],
        ) -> Vec<LightSource> {
            use crate::scene::scatter_volume::{
                ColourSource, Emission, EmissionCurve, ScatterShape,
            };
            // Sample the LUT at the value where emission peaks. For Linear
            // and Power curves emission grows with density, so the centre
            // of the volume (highest local density, typically remap = 1)
            // dominates the illumination. Threshold emission is a step
            // function; sampling just past the threshold is the most
            // representative point.
            fn lut_sample(lut: &[[u8; 4]; 256], t: f32) -> [f32; 3] {
                let idx = (t.clamp(0.0, 1.0) * 255.0).round() as usize;
                let p = lut[idx];
                [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            }
            let mut lights: Vec<LightSource> = Vec::new();
            for item in items {
                if item.settings.hidden {
                    continue;
                }
                let (strength, sample_t) = match item.volume.emission {
                    Emission::None => (0.0, 0.0),
                    Emission::Strength { strength, curve } => match curve {
                        EmissionCurve::Linear | EmissionCurve::Power(_) => (strength, 0.95),
                        EmissionCurve::Threshold(min_d) => {
                            (strength, (min_d + 0.05).clamp(0.0, 1.0))
                        }
                    },
                };
                if strength <= 0.0 {
                    continue;
                }
                let centre = item.volume.shape_centre();
                let (extent, size) = match item.volume.shape {
                    ScatterShape::Box(b) => {
                        let half = (b.max - b.min) * 0.5;
                        let r = half.length();
                        (r, r)
                    }
                    ScatterShape::Sphere { radius, .. } => (radius, radius),
                };
                let tint: [f32; 3] = match item.volume.colour {
                    ColourSource::Flat(rgb) => rgb,
                    ColourSource::Ramp(_) => [1.0, 1.0, 1.0],
                };
                let ramp_sample: [f32; 3] = match item.volume.colour {
                    ColourSource::Flat(_) => [1.0, 1.0, 1.0],
                    ColourSource::Ramp(id) => match colourmaps_cpu.get(id.0) {
                        Some(lut) => lut_sample(lut, sample_t),
                        None => [1.0, 1.0, 1.0],
                    },
                };
                let colour = [
                    tint[0] * ramp_sample[0],
                    tint[1] * ramp_sample[1],
                    tint[2] * ramp_sample[2],
                ];
                // Intensity model: emission * density folded into a unitless
                // scalar. Volume size enters through `range` rather than
                // intensity to keep illumination consistent across resizes.
                let intensity = strength * item.volume.density * item.settings.opacity;
                if !(intensity > 0.0) {
                    continue;
                }
                let range = (size * 4.0).max(extent * 2.0);
                let mut light = LightSource::default();
                light.kind = crate::renderer::types::LightKind::Point {
                    position: centre,
                    range,
                };
                light.colour = colour;
                light.intensity = intensity;
                lights.push(light);
            }
            lights
        }

        /// Convert a `LightSource` to `SingleLightUniform`, computing shadow matrix for lights[0].
        fn build_single_light_uniform(
            src: &LightSource,
            shadow_center: glam::Vec3,
            shadow_extent: f32,
            compute_shadow: bool,
        ) -> SingleLightUniform {
            let shadow_mat = if compute_shadow {
                compute_shadow_matrix(&src.kind, shadow_center, shadow_extent)
            } else {
                glam::Mat4::IDENTITY
            };

            match &src.kind {
                LightKind::Directional { direction } => SingleLightUniform {
                    light_view_proj: shadow_mat.to_cols_array_2d(),
                    pos_or_dir: *direction,
                    light_type: 0,
                    colour: src.colour,
                    intensity: src.intensity,
                    range: 0.0,
                    inner_angle: 0.0,
                    outer_angle: 0.0,
                    _pad_align: 0,
                    spot_direction: [0.0, -1.0, 0.0],
                    point_shadow_slot: -1,
                    point_shadow_near: 0.1,
                    _pad: [0.0; 3],
                },
                LightKind::Point { position, range } => SingleLightUniform {
                    light_view_proj: shadow_mat.to_cols_array_2d(),
                    pos_or_dir: *position,
                    light_type: 1,
                    colour: src.colour,
                    intensity: src.intensity,
                    range: *range,
                    inner_angle: 0.0,
                    outer_angle: 0.0,
                    _pad_align: 0,
                    spot_direction: [0.0, -1.0, 0.0],
                    point_shadow_slot: -1,
                    point_shadow_near: 0.1,
                    _pad: [0.0; 3],
                },
                LightKind::Spot {
                    position,
                    direction,
                    range,
                    inner_angle,
                    outer_angle,
                } => SingleLightUniform {
                    light_view_proj: shadow_mat.to_cols_array_2d(),
                    pos_or_dir: *position,
                    light_type: 2,
                    colour: src.colour,
                    intensity: src.intensity,
                    range: *range,
                    inner_angle: *inner_angle,
                    outer_angle: *outer_angle,
                    _pad_align: 0,
                    spot_direction: *direction,
                    point_shadow_slot: -1,
                    point_shadow_near: 0.1,
                    _pad: [0.0; 3],
                },
            }
        }

        // Derive virtual point lights from emissive scatter volumes. These
        // give nearby opaque surfaces warm illumination from "fire-like"
        // volumes without the consumer having to author a matching light by
        // hand. The lights are appended after the consumer's own lights and
        // are dropped if the per-frame cap is hit.
        let virtual_scatter_lights = derive_scatter_volume_virtual_lights(
            &frame.scene.scatter_volumes,
            &resources.content.colourmaps_cpu,
        );
        let raw_lights_unculled: Vec<&LightSource> = lighting
            .lights
            .iter()
            .chain(frame.scene.lights.iter())
            .chain(virtual_scatter_lights.iter())
            .collect();

        // CPU per-frame frustum cull. Sphere-vs-frustum for point lights,
        // cone-vs-frustum for spot lights, trivial-true for directional.
        // Dropping off-screen lights here keeps the cluster build pass and
        // the per-fragment iteration both bounded by what is actually visible.
        // Directional lights always survive : they affect every fragment.
        let cull_frustum = crate::camera::frustum::Frustum::from_view_proj(
            &frame.camera.render_camera.view_proj(),
        );
        let mut frustum_culled = 0u32;
        let raw_lights: Vec<&LightSource> = raw_lights_unculled
            .into_iter()
            .filter(|l| match l.kind {
                LightKind::Directional { .. } => true,
                LightKind::Point { position, range } => {
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
                    let keep = !cull_frustum.cull_cone(
                        glam::Vec3::from(position),
                        axis,
                        outer_angle,
                        range,
                    );
                    if !keep {
                        frustum_culled += 1;
                    }
                    keep
                }
            })
            .collect();
        *last_frustum_culled_lights = frustum_culled;

        // Rank the light array: first directional light pinned at slot 0 (the
        // cascaded-shadow caster), the rest ordered by
        // `importance * proximity_weight` descending (stable, so equal ranks
        // keep submission order). The order does two jobs: over
        // MAX_SCENE_LIGHTS the tail is dropped, and the cluster build keeps
        // each cell's lights in array order, so a cluster that overflows its
        // fixed slice drops the lowest-priority lights deterministically.
        // Directional lights are treated as infinitely close
        // (proximity_weight = 1). Shading itself is order-independent.
        let combined_lights: Vec<&LightSource> = {
            let camera_pos = glam::Vec3::from(frame.camera.render_camera.eye_position);
            let directional_first = raw_lights
                .iter()
                .position(|l| matches!(l.kind, LightKind::Directional { .. }));
            let mut out: Vec<&LightSource> = Vec::with_capacity(crate::resources::MAX_SCENE_LIGHTS);
            if let Some(i) = directional_first {
                out.push(raw_lights[i]);
            }
            let mut rest: Vec<(f32, &LightSource)> = raw_lights
                .iter()
                .enumerate()
                .filter(|(i, _)| Some(*i) != directional_first)
                .map(|(_, l)| {
                    let proximity = match l.kind {
                        LightKind::Directional { .. } => 1.0,
                        LightKind::Point { position, range }
                        | LightKind::Spot {
                            position, range, ..
                        } => {
                            let d = (glam::Vec3::from(position) - camera_pos).length();
                            // Light is fully relevant within `range`; fades to ~0 beyond 4x range.
                            (range / (range + d.max(0.0))).clamp(0.0, 1.0)
                        }
                    };
                    (l.importance.max(0.0) * proximity, *l)
                })
                .collect();
            rest.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let take = crate::resources::MAX_SCENE_LIGHTS - out.len();
            out.extend(rest.into_iter().take(take).map(|(_, l)| l));
            out
        };

        let light_count = combined_lights.len() as u32;

        // Build the per-light entries that get uploaded to the storage buffer.
        let mut lights_packed: Vec<SingleLightUniform> = combined_lights
            .iter()
            .enumerate()
            .map(|(i, src)| build_single_light_uniform(src, shadow_center, shadow_extent, i == 0))
            .collect();

        // ---------------------------------------------------------------
        // Point-light shadow pool: allocate a cubemap slot per shadow-casting
        // Point light, build the six per-face view-projection matrices, and
        // mutate each light's `point_shadow_slot`/`near` so the lit pass can
        // sample the cubemap. Faces are rendered later in this function once
        // the cascade pass has finished.
        // ---------------------------------------------------------------
        const POINT_SHADOW_NEAR: f32 = 0.1;
        let mut point_shadow_faces: Vec<PointShadowFace> = Vec::new();
        if matches!(
            lighting.point_shadow_mode,
            crate::renderer::types::PointShadowMode::Cube
        ) && lighting.shadows_enabled
        {
            shadow.point_shadow_frame = shadow.point_shadow_frame.wrapping_add(1);
            shadow
                .point_shadow_pool
                .begin_frame(shadow.point_shadow_frame);
            // Only as many point lights can have shadows as the cubemap pool
            // holds. Select that many deterministically (nearest to the
            // camera, importance-weighted) instead of letting every casting
            // light acquire a slot: the pool evicts on overflow, so a scene
            // with hundreds of default-constructed point lights (cast_shadows
            // defaults to true) would otherwise queue six shadow faces per
            // light per frame and thrash every slot. Lights beyond the pool
            // render unshadowed, exactly as if their acquire had failed.
            let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
            let mut point_casters: Vec<(usize, glam::Vec3, f32, f32)> = combined_lights
                .iter()
                .enumerate()
                .filter(|(_, src)| src.cast_shadows)
                .filter_map(|(i, src)| match src.kind {
                    LightKind::Point { position, range } => {
                        let pos = glam::Vec3::from(position);
                        let score = pos.distance(eye) / src.importance.max(0.001);
                        Some((i, pos, range, score))
                    }
                    _ => None,
                })
                .collect();
            point_casters
                .sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
            point_casters.truncate(crate::renderer::types::MAX_POINT_SHADOW_LIGHTS as usize);
            for &(i, light_pos, range, _) in &point_casters {
                let key = crate::renderer::point_shadow_pool::LightKey(i as u32);
                let Some(slot) = shadow.point_shadow_pool.acquire(key) else {
                    continue;
                };
                lights_packed[i].point_shadow_slot = slot as i32;
                lights_packed[i].point_shadow_near = POINT_SHADOW_NEAR;
                // Six standard cubemap face directions (forward, up). Order:
                //   0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
                let faces: [(glam::Vec3, glam::Vec3); 6] = [
                    (glam::Vec3::X, glam::Vec3::NEG_Y),
                    (glam::Vec3::NEG_X, glam::Vec3::NEG_Y),
                    (glam::Vec3::Y, glam::Vec3::Z),
                    (glam::Vec3::NEG_Y, glam::Vec3::NEG_Z),
                    (glam::Vec3::Z, glam::Vec3::NEG_Y),
                    (glam::Vec3::NEG_Z, glam::Vec3::NEG_Y),
                ];
                // Y-flipped projection: cubemap face (u, v) sampling
                // (WebGPU/OpenGL convention) expects content with V growing
                // in the opposite direction of `look_at_rh`'s positive view-Y.
                // Without this flip, every face stores its content mirrored
                // along V, so the lit pass looks up shadows in the wrong
                // image rows and shadows land far from the objects that cast
                // them. The pipeline compensates by treating CW as front-face
                // (see `build_shadow_point_pipeline`).
                let proj = glam::Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0))
                    * glam::Mat4::perspective_rh(
                        std::f32::consts::FRAC_PI_2,
                        1.0,
                        POINT_SHADOW_NEAR,
                        range.max(POINT_SHADOW_NEAR + 0.01),
                    );
                for (f, (forward, up)) in faces.iter().enumerate() {
                    let view = glam::Mat4::look_at_rh(light_pos, light_pos + *forward, *up);
                    point_shadow_faces.push(PointShadowFace {
                        slot,
                        face: f as u32,
                        view_proj: proj * view,
                        light_pos,
                        range,
                    });
                }
            }

            // Upload per-face uniforms (view_proj + light_pos + range,
            // padded to 256-byte dynamic-offset stride).
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct PointFaceUniform {
                view_proj: [[f32; 4]; 4], // 0..64
                // Packed vec4: xyz = light_pos, w = range.
                light_pos: [f32; 4], // 64..80
                _pad: [f32; 44],     // pad to 256
            }
            for fc in &point_shadow_faces {
                let entry = PointFaceUniform {
                    view_proj: fc.view_proj.to_cols_array_2d(),
                    light_pos: [fc.light_pos.x, fc.light_pos.y, fc.light_pos.z, fc.range],
                    _pad: [0.0; 44],
                };
                let layer = fc.slot * 6 + fc.face;
                let offset = layer as u64 * POINT_FACE_STRIDE;
                queue.write_buffer(
                    &resources.shadow_point_face_buf,
                    offset,
                    bytemuck::cast_slice(&[entry]),
                );
            }
        }

        // -------------------------------------------------------------------
        // Compute CSM cascade matrices for lights[0] (directional).
        // Cascades are fit to frame.camera, not to any per-viewport camera, so
        // every split viewport shares one shadow atlas.
        // -------------------------------------------------------------------
        let cascade_count = lighting.shadow_cascade_count.clamp(1, 4) as usize;
        let atlas_res = lighting.shadow_atlas_resolution.max(64);
        let tile_size = atlas_res / 2;

        let dist = frame.camera.render_camera.distance;
        let shadow_near = (dist * 0.1).max(frame.camera.render_camera.near);
        let shadow_far = (dist * 1.5)
            .max(shadow_extent)
            .min(frame.camera.render_camera.far);
        let cascade_splits =
            compute_cascade_splits(shadow_near, shadow_far, cascade_count as u32, 0.75);

        let light_dir_for_csm = if light_count > 0 {
            match &combined_lights[0].kind {
                LightKind::Directional { direction } => glam::Vec3::from(*direction).normalize(),
                LightKind::Point { position, .. } => {
                    (glam::Vec3::from(*position) - shadow_center).normalize()
                }
                LightKind::Spot {
                    position,
                    direction,
                    ..
                } => {
                    let _ = position;
                    glam::Vec3::from(*direction).normalize()
                }
            }
        } else {
            glam::Vec3::new(0.3, 1.0, 0.5).normalize()
        };

        let mut cascade_view_projs = [glam::Mat4::IDENTITY; 4];
        // Distance-based splits for fragment shader cascade selection.
        let mut cascade_split_distances = [0.0f32; 4];

        // Determine if we should use CSM (directional light + valid camera data).
        let use_csm = light_count > 0
            && matches!(combined_lights[0].kind, LightKind::Directional { .. })
            && frame.camera.render_camera.view != glam::Mat4::IDENTITY;

        if use_csm {
            for i in 0..cascade_count {
                let split_near = if i == 0 {
                    frame.camera.render_camera.near.max(0.01)
                } else {
                    cascade_splits[i - 1]
                };
                let split_far = cascade_splits[i];
                cascade_view_projs[i] = compute_cascade_matrix(
                    light_dir_for_csm,
                    frame.camera.render_camera.view,
                    frame.camera.render_camera.fov,
                    frame.camera.render_camera.aspect,
                    split_near,
                    split_far,
                    tile_size as f32,
                );
                cascade_split_distances[i] = split_far;
            }
        } else {
            // Fallback: single shadow map covering the whole scene (legacy behavior).
            let primary_shadow_mat = if light_count > 0 {
                compute_shadow_matrix(&combined_lights[0].kind, shadow_center, shadow_extent)
            } else {
                glam::Mat4::IDENTITY
            };
            cascade_view_projs[0] = primary_shadow_mat;
            cascade_split_distances[0] = frame.camera.render_camera.far;
        }
        // A primary light with `cast_shadows` off gets no shadow map at all:
        // no cascades are rendered and the shader's CSM sample short-circuits
        // on `cascade_count == 0`. Point lights manage their own cubemap slots
        // and are unaffected.
        let primary_casts_shadows = light_count > 0 && combined_lights[0].cast_shadows;
        let effective_cascade_count = if !primary_casts_shadows {
            0
        } else if use_csm {
            cascade_count
        } else {
            1
        };

        // D8: cache shadow stats and log when cascade splits change.
        {
            if cascade_split_distances != shadow.last_logged_cascade_splits {
                tracing::debug!(
                    cascade_count = effective_cascade_count,
                    splits = ?cascade_split_distances,
                    shadow_extent = shadow_extent,
                    camera_dist = frame.camera.render_camera.distance,
                    "cascade splits changed"
                );
                shadow.last_logged_cascade_splits = cascade_split_distances;
            }
            shadow.last_cascade_count = effective_cascade_count as u32;
            shadow.last_cascade_splits = cascade_split_distances;
            shadow.last_shadow_extent = shadow_extent;
            shadow.last_shadow_atlas_resolution = lighting.shadow_atlas_resolution.max(64);
            shadow.last_contact_shadow_active = frame.effects.post_process.contact_shadows;
        }

        // Atlas tile layout (2x2 grid):
        // [0] = top-left, [1] = top-right, [2] = bottom-left, [3] = bottom-right
        //
        // UV extents are computed from tile_size relative to the fixed SHADOW_ATLAS_SIZE
        // texture. When atlas_res < SHADOW_ATLAS_SIZE only the top-left portion of the
        // texture is rendered into (via scissor rect), so the UV rects must match that
        // footprint rather than always covering 0.0..0.5.
        let tile_uv = tile_size as f32 / crate::resources::SHADOW_ATLAS_SIZE as f32;
        let atlas_rects: [[f32; 4]; 8] = [
            [0.0, 0.0, tile_uv, tile_uv],                     // cascade 0
            [tile_uv, 0.0, tile_uv * 2.0, tile_uv],           // cascade 1
            [0.0, tile_uv, tile_uv, tile_uv * 2.0],           // cascade 2
            [tile_uv, tile_uv, tile_uv * 2.0, tile_uv * 2.0], // cascade 3
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
                // The backing texture size, not the requested resolution: shader
                // texel math (PCF radius, texel_world, blocker-search coords) is
                // relative to the real texture. The requested resolution enters
                // through the atlas rects, which shrink when it is lowered.
                atlas_size: crate::resources::SHADOW_ATLAS_SIZE as f32,
                shadow_filter: match lighting.shadow_filter {
                    ShadowFilter::Pcf => 0,
                    ShadowFilter::Pcss => 1,
                },
                pcss_light_radius: lighting.pcss_light_radius,
                atlas_rects,
            };
            queue.write_buffer(
                &resources.shadow_info_buf,
                0,
                bytemuck::cast_slice(&[shadow_atlas_uniform]),
            );
            // Write to all per-viewport slot buffers so each viewport's bind group
            // references correctly populated shadow info.
            for slot in viewport_slots {
                queue.write_buffer(
                    &slot.shadow_info_buf,
                    0,
                    bytemuck::cast_slice(&[shadow_atlas_uniform]),
                );
            }
            // Cache for viewport slots created later in the frame: a new slot's
            // buffer is seeded from this so its first frame does not sample
            // shadows through a zeroed uniform (see ensure_viewport_slot).
            shadow.last_shadow_atlas_uniform = shadow_atlas_uniform;
        }

        // The primary shadow matrix is still stored in lights[0].light_view_proj for
        // backward compat with the non-instanced shadow pass uniform.
        let _primary_shadow_mat = cascade_view_projs[0];
        // Cache for ground plane ShadowOnly mode.
        shadow.last_cascade0_shadow_mat = cascade_view_projs[0];

        // Upload lights uniform.
        // IBL fields from environment map settings.
        let (ibl_enabled, ibl_intensity, ibl_rotation, show_skybox) =
            if let Some(env) = scene_fx.environment {
                if resources.ibl_irradiance_view.is_some() {
                    (
                        1u32,
                        env.intensity,
                        env.rotation,
                        if env.show_skybox { 1u32 } else { 0 },
                    )
                } else {
                    (0, 0.0, 0.0, 0)
                }
            } else {
                (0, 0.0, 0.0, 0)
            };

        let debug_vis_mode = lighting.debug_vis.pack_mode();
        let debug_vis_scale = if lighting.debug_vis.active {
            lighting.debug_vis.scale.max(0.001)
        } else {
            1.0
        };

        let lights_uniform = LightsUniform {
            count: light_count,
            shadow_bias: lighting.shadow_bias,
            shadows_enabled: if lighting.shadows_enabled { 1 } else { 0 },
            debug_vis_mode,
            sky_colour: lighting.sky_colour,
            hemisphere_intensity: lighting.hemisphere_intensity,
            ground_colour: lighting.ground_colour,
            debug_vis_scale,
            ibl_enabled,
            ibl_intensity,
            ibl_rotation,
            show_skybox,
            debug_vis_split_x: if lighting.debug_vis.active {
                lighting.debug_vis.split_x.clamp(0.0, 1.0)
            } else {
                0.5
            },
            _pad_dbg: [0u32; 3],
        };
        queue.write_buffer(
            &resources.light_uniform_buf,
            0,
            bytemuck::cast_slice(&[lights_uniform]),
        );
        // Upload the per-light array to the storage buffer at binding 13.
        // Slots past `count` are left as-is; the shader bounds its loop on
        // `lights_uniform.count` so stale tail entries are never sampled.
        if !lights_packed.is_empty() {
            queue.write_buffer(
                &resources.light_storage_buf,
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

            let view_mat = frame.camera.render_camera.view;
            // Perspective parameters : derive tan(half_fov) from the proj
            // matrix diagonal (proj[0][0] = 1/tan(half_fov_x), with the wgpu
            // RH perspective matrices used here).
            let proj = frame.camera.render_camera.projection;
            let p00 = proj.col(0).x.max(1e-6);
            let p11 = proj.col(1).y.max(1e-6);
            let tan_half_fov_x = 1.0 / p00;
            let tan_half_fov_y = 1.0 / p11;
            let near = frame.camera.render_camera.near.max(0.01);
            let far = frame.camera.render_camera.far.max(near + 0.01);

            let active_count = lights_packed.len() as u32;
            // Skip the cluster build entirely for scenes with only a handful
            // of lights : straight per-fragment iteration is cheaper than the
            // lookup-table indirection at that scale. A consumer-set debug
            // override also forces the fallback path so the two can be A/B'd
            // for correctness checks.
            let use_clusters = !frame.viewport.force_cluster_fallback
                && active_count > crate::resources::gpu::clustered::SMALL_N_THRESHOLD;
            let fallback_flag = if use_clusters { 0.0 } else { 1.0 };
            let grid_uniform = ClusterGridUniform {
                dimensions: [
                    CLUSTER_X_TILES,
                    CLUSTER_Y_TILES,
                    CLUSTER_Z_SLICES,
                    CLUSTER_COUNT,
                ],
                depth: [near, far, (far / near).ln(), active_count as f32],
                screen: [
                    frame.camera.render_camera.aspect.max(0.01),
                    1.0,
                    fallback_flag,
                    0.0,
                ],
                proj_scale: [tan_half_fov_x, tan_half_fov_y, 0.0, 0.0],
                view: view_mat.to_cols_array_2d(),
            };
            resources.clustered.write_grid_uniform(queue, &grid_uniform);

            // Build the view-space ActiveLight array. Order matches
            // `lights_packed` / `light_storage_buf` so light_indices[j] from
            // the build pass is a valid index into lights_storage too.
            let active_lights: Vec<ActiveLightView> = combined_lights
                .iter()
                .map(|l| {
                    let (view_pos, range, light_type, view_dir, cos_outer) = match l.kind {
                        LightKind::Directional { direction } => {
                            let world_dir = glam::Vec3::from(direction).normalize_or_zero();
                            let view_dir =
                                view_mat.transform_vector3(world_dir).normalize_or_zero();
                            (glam::Vec3::ZERO, f32::INFINITY, 0u32, view_dir, 1.0)
                        }
                        LightKind::Point { position, range } => {
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
                    };
                    ActiveLightView {
                        view_pos_range: [view_pos.x, view_pos.y, view_pos.z, range],
                        type_pad: [light_type, 0, 0, 0],
                        spot_data: [view_dir.x, view_dir.y, view_dir.z, cos_outer],
                    }
                })
                .collect();
            resources
                .clustered
                .write_active_lights(queue, &active_lights);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cluster_frame_encoder"),
            });
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
            resources
                .clustered
                .dispatch_frame(&mut encoder, build_count, cluster_ts);
            sink.push(encoder.finish());

            // Optional host readback for the debug stats panel. Synchronous;
            // off by default.
            if frame.viewport.cluster_stats_request {
                let stats =
                    resources
                        .clustered
                        .read_stats(device, queue, active_count, !use_clusters);
                *last_cluster_stats = Some(stats);
            }
        }

        // Upload all cascade matrices to the shadow uniform buffer before the shadow pass.
        // wgpu batches write_buffer calls before the command buffer, so we must write ALL
        // cascade slots up-front; the cascade loop then selects per-slot via dynamic offset.
        const SHADOW_SLOT_STRIDE: u64 = 256;
        for c in 0..4usize {
            queue.write_buffer(
                &resources.shadow_uniform_buf,
                c as u64 * SHADOW_SLOT_STRIDE,
                bytemuck::cast_slice(&cascade_view_projs[c].to_cols_array_2d()),
            );
        }

        LightingFrame {
            point_shadow_faces,
            tile_size,
            cascade_view_projs,
            effective_cascade_count,
        }
    }
}
