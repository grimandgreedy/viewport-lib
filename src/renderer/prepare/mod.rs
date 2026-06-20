use super::types::{ClipShape, SceneEffects, ViewportEffects};
use super::*;
use crate::resources::CurveMeshOutlineItem;
use wgpu::util::DeviceExt;

mod math;
mod overlay_geometry;
mod projection;
mod scene_uploads;
mod viewport_interaction;
mod viewport_misc;
mod viewport_overlays;
mod wireframe;

use math::*;
use overlay_geometry::*;
use projection::*;
use wireframe::*;

impl ViewportRenderer {
    /// Scene-global prepare stage: compute filters, lighting, shadow pass, batching, scivis.
    ///
    /// Call once per frame before any `prepare_viewport_internal` calls.
    ///
    /// Reads `scene_fx` for lighting, IBL, and compute filters.  Also reads
    /// `frame.camera` for shadow cascade computation.
    pub(super) fn prepare_scene_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        scene_fx: &SceneEffects<'_>,
    ) {
        // Drain the upload-job runner. Worker results received since the last
        // frame are observed, GPU submissions are polled for completion, and
        // any registered completion callbacks fire on this thread.
        match self.upload_budget {
            Some(d) => self.resources.process_uploads_with_budget(
                device,
                queue,
                crate::resources::FrameBudget::from_now(d),
            ),
            None => self.resources.process_uploads(device, queue),
        }

        // GPU compute filtering.
        // Dispatch before the render pass. Completely skipped when list is empty (zero overhead).
        if !scene_fx.compute_filter_items.is_empty() {
            self.compute_filter_results =
                self.resources
                    .run_compute_filters(device, queue, scene_fx.compute_filter_items);
        } else {
            self.compute_filter_results.clear();
        }

        // Ensure built-in colourmaps and matcaps are uploaded on first frame.
        self.resources.ensure_colourmaps_initialized(device, queue);
        self.resources.ensure_matcaps_initialized(device, queue);

        // Capture an immutable view of the plugin map before splitting
        // off the mutable resources borrow. Used by per-cascade plugin
        // shadow dispatch deep inside the shadow_pass scope.
        let item_type_plugins_ptr = &self.item_type_plugins
            as *const std::collections::HashMap<
                &'static str,
                Box<dyn crate::plugin_api::ItemTypePlugin>,
            >;
        let plugin_frame_index = self.plugin_frame_index;

        let resources = &mut self.resources;
        let lighting = scene_fx.lighting;

        // Read scene items from the surface submission, then extend with the
        // boundary draws contributed by opaque volume meshes (items in
        // `volume_meshes` whose `transparency` is `None`). The owned vector
        // keeps these extra items alive for the whole prepare pass.
        let scene_items_owned: Vec<SceneRenderItem> = {
            let surfaces = match &frame.scene.surfaces {
                SurfaceSubmission::Flat(items) => items.as_ref(),
            };
            let extra = frame
                .scene
                .volume_meshes
                .iter()
                .filter(|item| item.transparency.is_none())
                .map(|item| item.to_render_item());
            surfaces.iter().cloned().chain(extra).collect()
        };
        let scene_items: &[SceneRenderItem] = &scene_items_owned;

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
            &resources.colourmaps_cpu,
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
        self.last_frustum_culled_lights = frustum_culled;

        // Apply the per-frame cap. When over the cap, keep the first directional
        // light at slot 0 (the cascaded-shadow caster) and rank the rest by
        // `importance * proximity_weight`, dropping the tail. Directional lights
        // are treated as infinitely close (proximity_weight = 1).
        let combined_lights: Vec<&LightSource> = if raw_lights.len()
            <= crate::resources::MAX_SCENE_LIGHTS
        {
            raw_lights
        } else {
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
        const POINT_FACE_STRIDE: u64 = 256;
        struct PointShadowFace {
            slot: u32,
            face: u32,
            view_proj: glam::Mat4,
            light_pos: glam::Vec3,
            range: f32,
        }
        let mut point_shadow_faces: Vec<PointShadowFace> = Vec::new();
        if matches!(
            lighting.point_shadow_mode,
            crate::renderer::types::PointShadowMode::Cube
        ) && lighting.shadows_enabled
        {
            self.point_shadow_frame = self.point_shadow_frame.wrapping_add(1);
            self.point_shadow_pool.begin_frame(self.point_shadow_frame);
            for (i, src) in combined_lights.iter().enumerate() {
                if !src.cast_shadows {
                    continue;
                }
                let (light_pos, range) = match src.kind {
                    LightKind::Point { position, range } => (glam::Vec3::from(position), range),
                    _ => continue,
                };
                let key = crate::renderer::point_shadow_pool::LightKey(i as u32);
                let Some(slot) = self.point_shadow_pool.acquire(key) else {
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
        let effective_cascade_count = if use_csm { cascade_count } else { 1 };

        // D8: cache shadow stats and log when cascade splits change.
        {
            if cascade_split_distances != self.last_logged_cascade_splits {
                tracing::debug!(
                    cascade_count = effective_cascade_count,
                    splits = ?cascade_split_distances,
                    shadow_extent = shadow_extent,
                    camera_dist = frame.camera.render_camera.distance,
                    "cascade splits changed"
                );
                self.last_logged_cascade_splits = cascade_split_distances;
            }
            self.last_cascade_count = effective_cascade_count as u32;
            self.last_cascade_splits = cascade_split_distances;
            self.last_shadow_extent = shadow_extent;
            self.last_shadow_atlas_resolution = lighting.shadow_atlas_resolution.max(64);
            self.last_contact_shadow_active = frame.effects.post_process.contact_shadows;
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
            for slot in &self.viewport_slots {
                queue.write_buffer(
                    &slot.shadow_info_buf,
                    0,
                    bytemuck::cast_slice(&[shadow_atlas_uniform]),
                );
            }
            // Cache for viewport slots created later in the frame: a new slot's
            // buffer is seeded from this so its first frame does not sample
            // shadows through a zeroed uniform (see ensure_viewport_slot).
            self.last_shadow_atlas_uniform = shadow_atlas_uniform;
        }

        // The primary shadow matrix is still stored in lights[0].light_view_proj for
        // backward compat with the non-instanced shadow pass uniform.
        let _primary_shadow_mat = cascade_view_projs[0];
        // Cache for ground plane ShadowOnly mode.
        self.last_cascade0_shadow_mat = cascade_view_projs[0];

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
            use crate::resources::clustered::{
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
                && active_count > crate::resources::clustered::SMALL_N_THRESHOLD;
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
            resources
                .clustered
                .dispatch_frame(&mut encoder, build_count);
            queue.submit(std::iter::once(encoder.finish()));

            // Optional host readback for the debug stats panel. Synchronous;
            // off by default.
            if frame.viewport.cluster_stats_request {
                let stats =
                    resources
                        .clustered
                        .read_stats(device, queue, active_count, !use_clusters);
                self.last_cluster_stats = Some(stats);
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

        // Per-frame batch upload counters.  Populated inside the instancing
        // block and folded into FrameStats at the end of prepare_scene_internal.
        let mut batches_reuploaded = 0u32;
        let mut batches_skipped = 0u32;

        // -- Instancing preparation --
        // Determine instancing mode BEFORE per-object uniforms so we can skip them.
        let visible_count = scene_items.iter().filter(|i| !i.settings.hidden).count();
        let prev_use_instancing = self.use_instancing;
        self.use_instancing = visible_count > INSTANCING_THRESHOLD;

        // If instancing mode changed (e.g. objects added/removed crossing the threshold),
        // clear batches so the generation check below forces a rebuild.
        if self.use_instancing != prev_use_instancing {
            self.instanced_batches.clear();
            self.last_scene_generation = u64::MAX;
            self.last_scene_items_count = usize::MAX;
        }
        if self.use_instancing != prev_use_instancing {
            tracing::debug!(
                visible_objects = visible_count,
                threshold = INSTANCING_THRESHOLD,
                instanced = self.use_instancing,
                "instancing mode changed"
            );
        }

        // Per-object uniform writes : needed for the non-instanced path, wireframe mode,
        // and for any items with active scalar attributes or two-sided materials
        // (both bypass the instanced path).
        let has_scalar_items = scene_items.iter().any(|i| i.active_attribute.is_some());
        let has_two_sided_items = scene_items.iter().any(|i| i.material.is_two_sided());
        let has_matcap_items = scene_items.iter().any(|i| i.material.matcap_id().is_some());
        let has_param_vis_items = scene_items.iter().any(|i| i.material.param_vis.is_some());
        let has_wireframe_items = scene_items.iter().any(|i| i.settings.wireframe);
        let has_normal_vis_items = scene_items.iter().any(|i| i.show_normals);
        // Items whose mesh has a position or normal override bound. The shader
        // flag that drives the override binding lives in the per-item
        // ObjectUniform, so we must enter the write loop for those items even
        // if no other "special" flag is set anywhere in the scene.
        let has_override_items = scene_items.iter().any(|i| {
            resources.mesh_store.get(i.mesh_id).map_or(false, |m| {
                m.position_override_buffer.is_some() || m.normal_override_buffer.is_some()
            })
        });
        // Skinned items are excluded from the batched-instanced path (the
        // instanced shader does not consume the skin palette), so they need
        // per-item object uniform writes too.
        let has_skinned_items = scene_items.iter().any(|i| {
            resources
                .deform
                .has_per_instance_deform_data(i.mesh_id, i.deform_instance)
        });
        // Collect per-item uniforms when wireframe mode is on so we can give each
        // visible item its own bind group (the mesh's shared object_uniform_buf gets
        // overwritten when multiple items reference the same MeshId).
        let mut wireframe_uniforms: Vec<ObjectUniform> = Vec::new();
        let collect_wf_uniforms = frame.viewport.wireframe_mode;
        if !self.use_instancing
            || frame.viewport.wireframe_mode
            || has_scalar_items
            || has_two_sided_items
            || has_matcap_items
            || has_param_vis_items
            || has_wireframe_items
            || has_normal_vis_items
            || has_override_items
            || has_skinned_items
        {
            // Make sure the per-item object pool can index every scene item.
            // Pool entries for items that go through the instanced path stay None.
            if self.per_item_object_bind_groups.len() < scene_items.len() {
                self.per_item_object_bind_groups
                    .resize_with(scene_items.len(), || None);
            }
            if self.per_item_object_cache_keys.len() < scene_items.len() {
                self.per_item_object_cache_keys.resize(scene_items.len(), 0);
            }
            for (item_idx, item) in scene_items.iter().enumerate() {
                // When instancing is active, skip items that will be rendered
                // via the instanced path. They don't need per-object uniform
                // writes; writing them anyway causes O(n) write_buffer calls
                // for the whole scene whenever any single item is two-sided.
                //
                // Items whose mesh has a GPU position or normal override bound
                // (`set_position_override_buffer` / `set_normal_override_buffer`)
                // MUST go through this per-item write path. The override
                // binding at group 1 binding 13/14 is selected by the shader's
                // `has_position_override` / `has_normal_override` flag, which
                // lives in the per-item `ObjectUniform`. Skipping the write
                // leaves the flag at the default 0, the shader ignores the
                // override binding, and the consumer's compute output silently
                // does nothing.
                let mesh_has_override = resources.mesh_store.get(item.mesh_id).map_or(false, |m| {
                    m.position_override_buffer.is_some() || m.normal_override_buffer.is_some()
                });
                let item_is_skinned = resources
                    .deform
                    .has_per_instance_deform_data(item.mesh_id, item.deform_instance);
                if self.use_instancing
                    && !frame.viewport.wireframe_mode
                    && item.active_attribute.is_none()
                    && !item.material.is_two_sided()
                    && item.material.matcap_id().is_none()
                    && item.material.param_vis.is_none()
                    && !item.settings.wireframe
                    && item.warp_attribute.is_none()
                    && !item.show_normals
                    && !mesh_has_override
                    && !item_is_skinned
                {
                    continue;
                }

                if resources.mesh_store.get(item.mesh_id).is_none() {
                    tracing::warn!(
                        mesh_index = item.mesh_id.index(),
                        "scene item mesh_index invalid, skipping"
                    );
                    continue;
                };
                let m = &item.material;
                // Compute scalar attribute range.
                let (has_attr, s_min, s_max) = if let Some(attr_ref) = &item.active_attribute {
                    let range =
                        item.scalar_range
                            .or_else(|| {
                                resources.mesh_store.get(item.mesh_id).and_then(|mesh| {
                                    mesh.attribute_ranges.get(&attr_ref.name).copied()
                                })
                            })
                            .unwrap_or((0.0, 1.0));
                    (1u32, range.0, range.1)
                } else {
                    (0u32, 0.0, 1.0)
                };
                let obj_uniform = ObjectUniform {
                    model: item.model,
                    colour: [
                        m.base_colour[0],
                        m.base_colour[1],
                        m.base_colour[2],
                        item.settings.opacity,
                    ],
                    selected: if item.settings.selected { 1 } else { 0 },
                    wireframe: if frame.viewport.wireframe_mode || item.settings.wireframe {
                        1
                    } else {
                        0
                    },
                    ambient: m.ambient,
                    diffuse: m.diffuse,
                    specular: m.specular,
                    shininess: m.shininess,
                    has_texture: if m.texture_id.is_some() { 1 } else { 0 },
                    use_pbr: if m.is_pbr() { 1 } else { 0 },
                    metallic: m.metallic,
                    roughness: m.roughness,
                    has_normal_map: if m.normal_map_id.is_some() { 1 } else { 0 },
                    has_ao_map: if m.ao_map_id.is_some() { 1 } else { 0 },
                    has_attribute: has_attr,
                    scalar_min: s_min,
                    scalar_max: s_max,
                    receive_shadows: if item.settings.receive_shadows { 1 } else { 0 },
                    nan_colour: item.nan_colour.unwrap_or([0.0; 4]),
                    use_nan_colour: if item.nan_colour.is_some() { 1 } else { 0 },
                    use_matcap: if m.matcap_id().is_some() { 1 } else { 0 },
                    matcap_blendable: m
                        .matcap_id()
                        .map_or(0, |id| if id.blendable { 1 } else { 0 }),
                    unlit: if item.settings.unlit { 1 } else { 0 },
                    use_face_colour: u32::from(item.active_attribute.as_ref().map_or(false, |a| {
                        a.kind == crate::resources::AttributeKind::FaceColour
                    })),
                    uv_vis_mode: m.param_vis.map_or(0, |pv| pv.mode as u32),
                    uv_vis_scale: m.param_vis.map_or(8.0, |pv| pv.scale),
                    backface_policy: match m.backface_policy {
                        crate::scene::material::BackfacePolicy::Cull => 0,
                        crate::scene::material::BackfacePolicy::Identical => 1,
                        crate::scene::material::BackfacePolicy::DifferentColour(_) => 2,
                        crate::scene::material::BackfacePolicy::Tint(_) => 3,
                        crate::scene::material::BackfacePolicy::Pattern(cfg) => {
                            4 + cfg.pattern as u32
                        }
                    },
                    backface_colour: match m.backface_policy {
                        crate::scene::material::BackfacePolicy::DifferentColour(c) => {
                            [c[0], c[1], c[2], 1.0]
                        }
                        crate::scene::material::BackfacePolicy::Tint(factor) => {
                            [factor, 0.0, 0.0, 1.0]
                        }
                        crate::scene::material::BackfacePolicy::Pattern(cfg) => {
                            let world_extent = resources
                                .mesh_store
                                .get(item.mesh_id)
                                .map(|mesh| {
                                    mesh.aabb
                                        .transformed(&glam::Mat4::from_cols_array_2d(&item.model))
                                        .longest_side()
                                })
                                .unwrap_or(1.0)
                                .max(1e-6);
                            let world_scale = cfg.scale / world_extent;
                            [cfg.colour[0], cfg.colour[1], cfg.colour[2], world_scale]
                        }
                        _ => [0.0; 4],
                    },
                    has_warp: if item.warp_attribute.is_some() { 1 } else { 0 },
                    warp_scale: item.warp_scale,
                    has_position_override: {
                        let mesh = resources.mesh_store.get(item.mesh_id);
                        if mesh.map_or(false, |m| m.position_override_buffer.is_some()) {
                            1
                        } else {
                            0
                        }
                    },
                    has_normal_override: {
                        let mesh = resources.mesh_store.get(item.mesh_id);
                        if mesh.map_or(false, |m| m.normal_override_buffer.is_some()) {
                            1
                        } else {
                            0
                        }
                    },
                    emissive: m.emissive,
                    use_flat: if m.is_flat() { 1 } else { 0 },
                    alpha_mode: match m.alpha_mode {
                        crate::scene::material::AlphaMode::Opaque => 0,
                        crate::scene::material::AlphaMode::Mask(_) => 1,
                        crate::scene::material::AlphaMode::Blend => 2,
                    },
                    alpha_cutoff: match m.alpha_mode {
                        crate::scene::material::AlphaMode::Mask(c) => c,
                        _ => 0.5,
                    },
                    has_metallic_roughness_tex: if m.metallic_roughness_texture_id.is_some() {
                        1
                    } else {
                        0
                    },
                    has_emissive_tex: if m.emissive_texture_id.is_some() {
                        1
                    } else {
                        0
                    },
                    uv_transform: [m.uv_offset[0], m.uv_offset[1], m.uv_scale[0], m.uv_scale[1]],
                    deform_flags: resources.deform.flag_bits(item.mesh_id),
                    _pad_after_deform: 0,
                    ao_range: m.ao_range,
                    metallic_range: m.metallic_range,
                    roughness_range: m.roughness_range,
                };

                let normal_obj_uniform = ObjectUniform {
                    model: item.model,
                    colour: [1.0, 1.0, 1.0, 1.0],
                    selected: 0,
                    wireframe: 0,
                    ambient: 0.15,
                    diffuse: 0.75,
                    specular: 0.4,
                    shininess: 32.0,
                    has_texture: 0,
                    use_pbr: 0,
                    metallic: 0.0,
                    roughness: 0.5,
                    has_normal_map: 0,
                    has_ao_map: 0,
                    has_attribute: 0,
                    scalar_min: 0.0,
                    scalar_max: 1.0,
                    receive_shadows: 1,
                    nan_colour: [0.0; 4],
                    use_nan_colour: 0,
                    use_matcap: 0,
                    matcap_blendable: 0,
                    unlit: 0,
                    use_face_colour: 0,
                    uv_vis_mode: 0,
                    uv_vis_scale: 8.0,
                    backface_policy: 0,
                    backface_colour: [0.0; 4],
                    has_warp: 0,
                    warp_scale: 1.0,
                    has_position_override: 0,
                    has_normal_override: 0,
                    emissive: [0.0; 3],
                    use_flat: 0,
                    alpha_mode: 0,
                    alpha_cutoff: 0.5,
                    has_metallic_roughness_tex: 0,
                    has_emissive_tex: 0,
                    uv_transform: [0.0, 0.0, 1.0, 1.0],
                    deform_flags: 0,
                    _pad_after_deform: 0,
                    ao_range: [0.0, 1.0],
                    metallic_range: [0.0, 1.0],
                    roughness_range: [0.0, 1.0],
                };

                // Collect per-item uniform for wireframe per-item bind groups.
                if collect_wf_uniforms && !item.settings.hidden {
                    wireframe_uniforms.push(obj_uniform);
                }

                // Write uniform data : use get() to read buffer references, then drop.
                {
                    let mesh = resources.mesh_store.get(item.mesh_id).unwrap();
                    queue.write_buffer(
                        &mesh.object_uniform_buf,
                        0,
                        bytemuck::cast_slice(&[obj_uniform]),
                    );
                    queue.write_buffer(
                        &mesh.normal_uniform_buf,
                        0,
                        bytemuck::cast_slice(&[normal_obj_uniform]),
                    );
                } // mesh borrow dropped here

                // Rebuild the object bind group if material/attribute/LUT/matcap/warp changed.
                resources.update_mesh_texture_bind_group(
                    device,
                    item.mesh_id,
                    item.material.texture_id,
                    item.material.normal_map_id,
                    item.material.ao_map_id,
                    item.colourmap_id,
                    item.active_attribute.as_ref().map(|a| a.name.as_str()),
                    item.material.matcap_id(),
                    item.warp_attribute.as_deref(),
                    item.material.metallic_roughness_texture_id,
                    item.material.emissive_texture_id,
                );

                // Per-item object pool: ensure this slot has its own uniform buffer +
                // bind group keyed on the item's position in scene_items. Multiple
                // scene items can share the same MeshId; the mesh's shared
                // object_uniform_buf above is stomped by whichever item wrote last,
                // so we maintain this parallel pool to guarantee each item draws
                // with its own transform/material.
                {
                    let uniform_size = std::mem::size_of::<ObjectUniform>() as u64;
                    while self.per_item_object_uniform_bufs.len() <= item_idx {
                        let buf = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("per_item_object_uniform"),
                            size: uniform_size,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        self.per_item_object_uniform_bufs.push(buf);
                    }
                    queue.write_buffer(
                        &self.per_item_object_uniform_bufs[item_idx],
                        0,
                        bytemuck::cast_slice(&[obj_uniform]),
                    );

                    let built = resources.build_per_item_object_bind_group(
                        device,
                        item.mesh_id,
                        &self.per_item_object_uniform_bufs[item_idx],
                        item.material.texture_id,
                        item.material.normal_map_id,
                        item.material.ao_map_id,
                        item.colourmap_id,
                        item.active_attribute.as_ref().map(|a| a.name.as_str()),
                        item.material.matcap_id(),
                        item.warp_attribute.as_deref(),
                        item.material.metallic_roughness_texture_id,
                        item.material.emissive_texture_id,
                    );
                    if let Some((bg, key)) = built {
                        let need_rebuild = self.per_item_object_bind_groups[item_idx].is_none()
                            || self.per_item_object_cache_keys[item_idx] != key;
                        if need_rebuild {
                            self.per_item_object_bind_groups[item_idx] = Some(bg);
                            self.per_item_object_cache_keys[item_idx] = key;
                        }
                    }
                }
            }
        }

        // Build per-item wireframe bind groups so each visible item gets its own
        // object uniform, avoiding the shared-MeshId overwrite problem.
        if !wireframe_uniforms.is_empty() {
            let n = wireframe_uniforms.len();
            let uniform_size = std::mem::size_of::<ObjectUniform>() as u64;

            // Grow the buffer/bind-group pools if needed. We never shrink them.
            while self.wireframe_uniform_bufs.len() < n {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wireframe_item_uniform"),
                    size: uniform_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("wireframe_item_bg"),
                    layout: &resources.object_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_texture.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&resources.material_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_normal_map_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_ao_map_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_lut_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: resources.fallback_scalar_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(
                                resources
                                    .fallback_matcap_view
                                    .as_ref()
                                    .unwrap_or(&resources.fallback_texture.view),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: resources.fallback_face_colour_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: resources.fallback_warp_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: wgpu::BindingResource::Sampler(&resources.lut_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_metallic_roughness_texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.fallback_emissive_texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: resources.fallback_position_override_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: resources.fallback_normal_override_buf.as_entire_binding(),
                        },
                    ],
                });
                self.wireframe_uniform_bufs.push(buf);
                self.wireframe_bind_groups.push(bg);
            }

            // Write each item's uniform into its dedicated buffer.
            for (i, uniform) in wireframe_uniforms.iter().enumerate() {
                queue.write_buffer(
                    &self.wireframe_uniform_bufs[i],
                    0,
                    bytemuck::cast_slice(std::slice::from_ref(uniform)),
                );
            }
        }

        if self.use_instancing {
            resources.ensure_instanced_pipelines(device);
            resources.ensure_hdr_instanced_pipelines(device);
            resources.ensure_oit_instanced_pipeline(device);

            // Generation-based cache: skip batch rebuild and GPU upload when nothing changed.
            // wireframe_mode removed from cache key : wireframe rendering
            // uses the per-object wireframe_pipeline, not the instanced path, so
            // instance data is now viewport-agnostic.
            //
            // Items with active_attribute, two-sided policy, matcap, or param_vis are
            // excluded from the instanced batch filter. Items whose mesh has an active
            // compute filter result are also excluded so the per-object path can apply
            // the filtered index buffer (instanced draws always use the full index buffer).
            // These flags are set on render items AFTER collect_render_items() (per-frame
            // mutations), so they do NOT bump the scene generation. Use last_instancable_count
            // as a cache key instead of a blanket has_per_frame_mutations flag; this allows
            // scenes that mix instanced and non-instanced items (e.g. one two-sided mesh +
            // many static boxes) to still hit the instanced batch cache on frames where the
            // filtered set is unchanged.
            let compute_filter_results = &self.compute_filter_results;
            let instancable_count = scene_items
                .iter()
                .filter(|item| {
                    !item.settings.hidden
                        && item.active_attribute.is_none()
                        && !item.material.is_two_sided()
                        && item.material.matcap_id().is_none()
                        && item.material.param_vis.is_none()
                        && resources.mesh_store.get(item.mesh_id).is_some()
                        && !compute_filter_results.iter().any(|r| r.mesh_id == item.mesh_id)
                        // Items whose mesh has a GPU position/normal override
                        // bound must use the per-object pipeline. `mesh_instanced.wgsl`
                        // has no awareness of the override binding, so an
                        // instanced draw silently ignores the consumer's
                        // compute output.
                        && resources
                            .mesh_store
                            .get(item.mesh_id)
                            .map_or(true, |m| {
                                m.position_override_buffer.is_none()
                                    && m.normal_override_buffer.is_none()
                            })
                })
                .count();
            let cache_valid = instancable_count == self.last_instancable_count
                && frame.scene.generation == self.last_scene_generation
                && frame.interaction.selection_generation == self.last_selection_generation
                && scene_items.len() == self.last_scene_items_count;

            if !cache_valid {
                // Cache miss : rebuild batches and upload instance data.
                let mut sorted_items: Vec<&SceneRenderItem> = scene_items
                    .iter()
                    .filter(|item| {
                        !item.settings.hidden
                            && item.active_attribute.is_none()
                            && !item.material.is_two_sided()
                            && item.material.matcap_id().is_none()
                            && item.material.param_vis.is_none()
                            && resources.mesh_store.get(item.mesh_id).is_some()
                            && !compute_filter_results.iter().any(|r| r.mesh_id == item.mesh_id)
                            // Items with per-instance deformer data need their
                            // own bind group at draw time and cannot share an
                            // instance batch.
                            && !resources
                                .deform
                                .has_per_instance_deform_data(item.mesh_id, item.deform_instance)
                            // Items with a position/normal override must use
                            // the per-object pipeline; see the matching check
                            // in `instancable_count` above.
                            && resources
                                .mesh_store
                                .get(item.mesh_id)
                                .map_or(true, |m| {
                                    m.position_override_buffer.is_none()
                                        && m.normal_override_buffer.is_none()
                                })
                    })
                    .collect();

                sorted_items.sort_unstable_by(|a, b| {
                    // Batch grouping key (must match the batch-split condition).
                    let batch_ord = (
                        a.mesh_id.index(),
                        a.material.texture_id,
                        a.material.normal_map_id,
                        a.material.ao_map_id,
                    )
                        .cmp(&(
                            b.mesh_id.index(),
                            b.material.texture_id,
                            b.material.normal_map_id,
                            b.material.ao_map_id,
                        ));
                    if batch_ord != std::cmp::Ordering::Equal {
                        return batch_ord;
                    }
                    // Within a batch, sort by model matrix for spatial coherence:
                    // column 3 (translation) first, then columns 0-2.  This keeps
                    // spatially close instances adjacent in the buffer, which
                    // reduces GPU cache pressure through the visibility-index
                    // indirection in the culled draw path.
                    for col in [3, 0, 1, 2] {
                        for row in 0..4 {
                            let ord = a.model[col][row]
                                .to_bits()
                                .cmp(&b.model[col][row].to_bits());
                            if ord != std::cmp::Ordering::Equal {
                                return ord;
                            }
                        }
                    }
                    // Final tiebreaker: pick_id is a stable, application-assigned
                    // per-object identity that is guaranteed unique for every
                    // pickable object. Placing it last (rather than in the batch
                    // key) ensures that any two objects with identical transforms
                    // still sort deterministically, regardless of the order they
                    // appear in the caller's scene_items slice.
                    a.settings.pick_id.0.cmp(&b.settings.pick_id.0)
                });

                let mut all_instances: Vec<InstanceData> = Vec::with_capacity(sorted_items.len());
                let mut all_aabbs: Vec<InstanceAabb> = Vec::with_capacity(sorted_items.len());
                let mut batch_metas: Vec<BatchMeta> = Vec::new();
                let mut instanced_batches: Vec<InstancedBatch> = Vec::new();

                if !sorted_items.is_empty() {
                    let mut batch_start = 0usize;
                    for i in 1..=sorted_items.len() {
                        let at_end = i == sorted_items.len();
                        let key_changed = !at_end && {
                            let a = sorted_items[batch_start];
                            let b = sorted_items[i];
                            a.mesh_id != b.mesh_id
                                || a.material.texture_id != b.material.texture_id
                                || a.material.normal_map_id != b.material.normal_map_id
                                || a.material.ao_map_id != b.material.ao_map_id
                        };

                        if at_end || key_changed {
                            let batch_items = &sorted_items[batch_start..i];
                            let rep = batch_items[0];
                            let instance_offset = all_instances.len() as u32;
                            let is_transparent = rep.settings.opacity < 1.0;

                            // All items in a batch share the same mesh_id (batch key).
                            // Look up the mesh once and reuse it for both index_count and
                            // per-instance AABB transforms, avoiding N redundant hash map
                            // lookups inside the inner loop.
                            let batch_idx = instanced_batches.len() as u32;
                            let batch_mesh = resources.mesh_store.get(rep.mesh_id);
                            let mesh_index_count = batch_mesh.map(|m| m.index_count).unwrap_or(0);

                            for item in batch_items {
                                let m = &item.material;
                                all_instances.push(InstanceData {
                                    model: item.model,
                                    colour: [
                                        m.base_colour[0],
                                        m.base_colour[1],
                                        m.base_colour[2],
                                        item.settings.opacity,
                                    ],
                                    selected: if item.settings.selected { 1 } else { 0 },
                                    wireframe: 0, // always 0 : wireframe uses per-object pipeline
                                    ambient: m.ambient,
                                    diffuse: m.diffuse,
                                    specular: m.specular,
                                    shininess: m.shininess,
                                    has_texture: if m.texture_id.is_some() { 1 } else { 0 },
                                    use_pbr: if m.is_pbr() { 1 } else { 0 },
                                    metallic: m.metallic,
                                    roughness: m.roughness,
                                    has_normal_map: if m.normal_map_id.is_some() { 1 } else { 0 },
                                    has_ao_map: if m.ao_map_id.is_some() { 1 } else { 0 },
                                    unlit: if item.settings.unlit { 1 } else { 0 },
                                    receive_shadows: if item.settings.receive_shadows {
                                        1
                                    } else {
                                        0
                                    },
                                    use_flat: if m.is_flat() { 1 } else { 0 },
                                    _pad_inst: 0,
                                    uv_transform: [
                                        m.uv_offset[0],
                                        m.uv_offset[1],
                                        m.uv_scale[0],
                                        m.uv_scale[1],
                                    ],
                                    ao_range: m.ao_range,
                                    _pad_ao_range: [0.0, 0.0],
                                });
                                if let Some(mesh) = batch_mesh {
                                    let model = glam::Mat4::from_cols_array_2d(&item.model);
                                    let world_aabb = mesh.aabb.transformed(&model);
                                    all_aabbs.push(InstanceAabb {
                                        min: world_aabb.min.into(),
                                        batch_index: batch_idx,
                                        max: world_aabb.max.into(),
                                        cast_shadows: if item.settings.cast_shadows {
                                            1
                                        } else {
                                            0
                                        },
                                    });
                                }
                            }

                            // vis_offset is the prefix sum of instance counts; since
                            // instances are laid out contiguously per batch, it equals
                            // instance_offset.
                            batch_metas.push(BatchMeta {
                                index_count: mesh_index_count,
                                first_index: 0,
                                instance_offset,
                                instance_count: batch_items.len() as u32,
                                vis_offset: instance_offset,
                                is_transparent: if is_transparent { 1 } else { 0 },
                                _pad: [0, 0],
                            });

                            instanced_batches.push(InstancedBatch {
                                mesh_id: rep.mesh_id,
                                texture_id: rep.material.texture_id,
                                normal_map_id: rep.material.normal_map_id,
                                ao_map_id: rep.material.ao_map_id,
                                instance_offset,
                                instance_count: batch_items.len() as u32,
                                is_transparent,
                            });

                            batch_start = i;
                        }
                    }
                }

                // Partial upload: when the batch structure is unchanged (same
                // count, same offsets and sizes per batch), compare each
                // batch's instance data against the cached CPU copy and only
                // write the sub-ranges that actually differ.  This avoids
                // re-uploading the full buffer when only a small fraction of
                // objects changed (e.g. one animated object in a large static
                // scene).
                //
                // A forced full upload (via `force_dirty()`) or any structural
                // change (different batch count, different instance counts)
                // falls back to the original full-upload path.
                let structure_preserved = self.cached_instance_count > 0
                    && all_instances.len() == self.cached_instance_count
                    && instanced_batches.len() == self.cached_instanced_batches.len()
                    && instanced_batches
                        .iter()
                        .zip(&self.cached_instanced_batches)
                        .all(|(a, b)| {
                            a.mesh_id == b.mesh_id
                                && a.instance_offset == b.instance_offset
                                && a.instance_count == b.instance_count
                        });
                let force = std::mem::replace(&mut self.force_full_upload, false);

                if structure_preserved && !force {
                    let inst_stride = std::mem::size_of::<InstanceData>() as u64;
                    let aabb_stride = std::mem::size_of::<InstanceAabb>() as u64;
                    // Ensure the hash vec is the right length (it should already be,
                    // but guard against a first-run edge case).
                    if self.cached_instance_hashes.len() != instanced_batches.len() {
                        self.cached_instance_hashes
                            .resize(instanced_batches.len(), 0);
                    }
                    for (bi, batch) in instanced_batches.iter().enumerate() {
                        let start = batch.instance_offset as usize;
                        let end = start + batch.instance_count as usize;
                        let new_bytes =
                            bytemuck::cast_slice::<InstanceData, u8>(&all_instances[start..end]);
                        let new_hash = hash_instance_bytes(new_bytes);
                        if new_hash != self.cached_instance_hashes[bi] {
                            if let Some(buf) = resources.instance_storage_buf.as_ref() {
                                queue.write_buffer(
                                    buf,
                                    batch.instance_offset as u64 * inst_stride,
                                    new_bytes,
                                );
                            }
                            if let Some(aabb_buf) = resources.instance_aabb_buf.as_ref() {
                                let aabb_bytes = bytemuck::cast_slice::<InstanceAabb, u8>(
                                    &all_aabbs[start..end],
                                );
                                queue.write_buffer(
                                    aabb_buf,
                                    batch.instance_offset as u64 * aabb_stride,
                                    aabb_bytes,
                                );
                            }
                            self.cached_instance_hashes[bi] = new_hash;
                            batches_reuploaded += 1;
                        } else {
                            batches_skipped += 1;
                        }
                    }
                } else {
                    resources.upload_instance_data(device, queue, &all_instances);
                    resources.upload_aabb_and_batch_meta(device, queue, &all_aabbs, &batch_metas);
                    batches_reuploaded = instanced_batches.len() as u32;
                    // Rebuild the hash cache so the next partial-upload check is seeded.
                    self.cached_instance_hashes.clear();
                    for batch in &instanced_batches {
                        let start = batch.instance_offset as usize;
                        let end = start + batch.instance_count as usize;
                        let bytes =
                            bytemuck::cast_slice::<InstanceData, u8>(&all_instances[start..end]);
                        self.cached_instance_hashes.push(hash_instance_bytes(bytes));
                    }
                }

                self.cached_instance_count = all_instances.len();
                self.cached_instanced_batches = instanced_batches;
                self.instanced_batches = self.cached_instanced_batches.clone();

                self.last_scene_generation = frame.scene.generation;
                self.last_selection_generation = frame.interaction.selection_generation;
                self.last_scene_items_count = scene_items.len();
                self.last_instancable_count = sorted_items.len();

                for batch in &self.instanced_batches {
                    resources.get_instance_bind_group(
                        device,
                        batch.texture_id,
                        batch.normal_map_id,
                        batch.ao_map_id,
                    );
                }
            } else {
                for batch in &self.instanced_batches {
                    resources.get_instance_bind_group(
                        device,
                        batch.texture_id,
                        batch.normal_map_id,
                        batch.ao_map_id,
                    );
                }
            }

            // ------------------------------------------------------------------
            // GPU cull dispatch
            //
            // Run `cull_instances` + `write_indirect_args` whenever GPU culling
            // is active and all required buffers are allocated.
            // ------------------------------------------------------------------
            if self.gpu_culling_enabled
                && !self.instanced_batches.is_empty()
                && self.cached_instance_count > 0
            {
                let instance_count = self.cached_instance_count as u32;
                let batch_count = self.instanced_batches.len() as u32;

                // Do all mutable borrows before taking immutable borrows from resources.
                if self.cull_resources.is_none() {
                    self.cull_resources =
                        Some(crate::renderer::indirect::CullResources::new(device));
                }
                resources.ensure_cull_instance_pipelines(device);
                for batch in &self.instanced_batches.clone() {
                    resources.get_instance_cull_bind_group(
                        device,
                        batch.texture_id,
                        batch.normal_map_id,
                        batch.ao_map_id,
                    );
                }

                // Now take immutable borrows to the GPU buffers for dispatch.
                if let (
                    Some(aabb_buf),
                    Some(meta_buf),
                    Some(counter_buf),
                    Some(vis_buf),
                    Some(indirect_buf),
                ) = (
                    resources.instance_aabb_buf.as_ref(),
                    resources.batch_meta_buf.as_ref(),
                    resources.batch_counter_buf.as_ref(),
                    resources.visibility_index_buf.as_ref(),
                    resources.indirect_args_buf.as_ref(),
                ) {
                    let vp_mat = frame.camera.render_camera.view_proj();
                    let cpu_frustum = crate::camera::frustum::Frustum::from_view_proj(&vp_mat);

                    let cull = self.cull_resources.as_ref().unwrap();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cull_encoder"),
                        });
                    let sub = crate::plugin_api::CullSubmission {
                        instance_aabbs: aabb_buf,
                        instance_count,
                        batch_meta: meta_buf,
                        batch_count,
                        counter: counter_buf,
                        visible_out: vis_buf,
                        indirect_out: indirect_buf,
                        shadow_pass: false,
                    };
                    cull.dispatch(&mut encoder, device, queue, &cpu_frustum, None, &sub);

                    // Copy indirect_args_buf to the CPU-readable staging buffer so the
                    // visible instance count can be read back next frame (one-frame lag).
                    let indirect_bytes = batch_count as u64 * 20;
                    if self.indirect_readback_buf.as_ref().map_or(0, |b| b.size()) < indirect_bytes
                    {
                        self.indirect_readback_buf =
                            Some(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("indirect_readback_buf"),
                                size: indirect_bytes,
                                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                                mapped_at_creation: false,
                            }));
                    }
                    if let Some(ref rb_buf) = self.indirect_readback_buf {
                        encoder.copy_buffer_to_buffer(indirect_buf, 0, rb_buf, 0, indirect_bytes);
                    }
                    queue.submit(std::iter::once(encoder.finish()));
                    self.indirect_readback_batch_count = batch_count;
                    self.indirect_readback_pending = true;
                }
            }
        }

        Self::upload_geometry_glyphs(resources, &mut self.point_cloud_gpu_data, &mut self.glyph_gpu_data, &mut self.sprite_gpu_data, &mut self.mesh_instance_gpu_data, &mut self.particle_gpu_data, &mut self.tensor_glyph_gpu_data, device, queue, frame);
        Self::upload_polylines(resources, &mut self.polyline_gpu_data, &mut self.polyline_selected_gpu_indices, &mut self.glyph_gpu_data, device, queue, frame);
        Self::upload_implicit_decals_mc(resources, &mut self.implicit_gpu_data, &mut self.pick_implicit_items, &mut self.decal_gpu_data, &mut self.decal_exclude_items, &mut self.mc_gpu_data, &mut self.pick_mc_items, device, queue, frame);
        Self::upload_images(resources, &mut self.screen_image_gpu_data, &mut self.overlay_image_gpu_data, device, queue, frame);
        Self::upload_tubes_ribbons(resources, &mut self.streamtube_gpu_data, &mut self.streamtube_selected_gpu_indices, &mut self.tube_gpu_data, &mut self.tube_selected_gpu_indices, &mut self.ribbon_gpu_data, &mut self.ribbon_selected_gpu_indices, device, queue, frame);
        Self::upload_slices(resources, &mut self.image_slice_gpu_data, &mut self.volume_surface_slice_gpu_data, device, queue, frame);
        let vp_size = frame.camera.viewport_size;
        // Surface LIC GPU data upload.
        // ------------------------------------------------------------------
        self.lic_gpu_data.clear();
        {
            let lic_scene_items: Vec<(&SceneRenderItem, &LicOverlay)> = scene_items
                .iter()
                .filter(|i| !i.settings.hidden)
                .filter_map(|i| i.lic.as_ref().map(|l| (i, l)))
                .collect();
            if !lic_scene_items.is_empty() {
                // The LIC surface pipeline is created inside ensure_hdr_shared (already called
                // before prepare_scene_internal runs), so no separate ensure call is needed here.
                for (item, lic) in &lic_scene_items {
                    if lic.vector_attribute.is_empty() {
                        continue;
                    }
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                        // Verify the vector attribute buffer exists before committing to this item.
                        if mesh
                            .vector_attribute_buffers
                            .contains_key(&lic.vector_attribute)
                        {
                            if let Some(bgl) = &resources.lic_surface_bgl {
                                use crate::resources::LicObjectUniform;
                                let model = item.model;
                                let obj_data = LicObjectUniform { model };
                                let obj_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some("lic_object_uniform"),
                                    size: std::mem::size_of::<LicObjectUniform>() as u64,
                                    usage: wgpu::BufferUsages::UNIFORM
                                        | wgpu::BufferUsages::COPY_DST,
                                    mapped_at_creation: false,
                                });
                                queue.write_buffer(&obj_buf, 0, bytemuck::cast_slice(&[obj_data]));
                                // Bind group (group 1): object uniform only.
                                // Flow vectors are bound as vertex buffer 1 in the render pass.
                                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("lic_surface_item_bg"),
                                    layout: bgl,
                                    entries: &[wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: obj_buf.as_entire_binding(),
                                    }],
                                });
                                self.lic_gpu_data.push(crate::resources::LicSurfaceGpuData {
                                    bind_group: bg,
                                    _object_uniform_buf: obj_buf,
                                    mesh_id: item.mesh_id,
                                    vector_attribute: lic.vector_attribute.clone(),
                                });
                            }
                        }
                    }
                }
                // Write LicAdvectUniform to the per-viewport buffer.
                if let Some(hdr) = self.viewport_slots[frame.camera.viewport_index]
                    .hdr
                    .as_ref()
                {
                    if let Some((_, first_lic)) = lic_scene_items.first() {
                        let [vw, vh] = hdr.scene_size;
                        let u = crate::resources::LicAdvectUniform {
                            steps: first_lic.config.steps,
                            step_size: first_lic.config.step_size,
                            vp_width: vw as f32,
                            vp_height: vh as f32,
                        };
                        queue.write_buffer(&hdr.lic_uniform_buf, 0, bytemuck::cast_slice(&[u]));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Volume GPU data upload.
        // Note: clip_planes are per-viewport but passed here for culling.
        // ------------------------------------------------------------------
        self.volume_gpu_data.clear();
        if !frame.scene.volumes.is_empty() {
            resources.ensure_volume_pipeline(device);
            let clip_objects_for_vol = &frame.effects.clip_objects;
            // Under budget pressure with allow_volume_quality_reduction, double the
            // step size (half the sample count) to reduce GPU raymarch cost.
            let vol_step_multiplier = if self.degradation_volume_quality_reduced {
                2.0_f32
            } else {
                1.0_f32
            };
            for item in &frame.scene.volumes {
                if item.settings.hidden {
                    continue;
                }
                let mut gpu = resources.upload_volume_frame(
                    device,
                    queue,
                    item,
                    clip_objects_for_vol,
                    vol_step_multiplier,
                );
                gpu.wireframe = frame.viewport.wireframe_mode || item.settings.wireframe;
                self.volume_gpu_data.push(gpu);
            }
        }

        // Volume wireframe overlay: OBB from bbox + model matrix.
        let need_vol_wf = frame.viewport.wireframe_mode
            || frame
                .scene
                .volumes
                .iter()
                .any(|v| !v.settings.hidden && v.settings.wireframe);
        if need_vol_wf {
            resources.ensure_polyline_pipeline(device);
            for item in &frame.scene.volumes {
                if item.settings.hidden {
                    continue;
                }
                if !(frame.viewport.wireframe_mode || item.settings.wireframe) {
                    continue;
                }
                let polyline = volume_obb_polyline(item);
                let gpu = resources.upload_polyline_per_frame(device, queue, &polyline, vp_size);
                self.polyline_gpu_data.push(gpu);
            }
        }

        // Transparent volume meshes wireframe: boundary mesh edge overlay.
        // Items rendering as opaque already participate in the standard
        // wireframe pass via the surface submission; here we only need to
        // gather boundary edges for items rendering through the projected-tet
        // path so they still get a wireframe overlay.
        self.tvm_wireframe_draws.clear();
        for item in &frame.scene.volume_meshes {
            if item.settings.hidden || item.transparency.is_none() {
                continue;
            }
            if !(item.settings.wireframe || frame.viewport.wireframe_mode) {
                continue;
            }
            if resources.mesh_store.get(item.boundary_mesh_id).is_none() {
                continue;
            }
            self.tvm_wireframe_draws.push(item.boundary_mesh_id);
        }
        if !self.tvm_wireframe_draws.is_empty() && self.tvm_wireframe_bg.is_none() {
            use wgpu::util::DeviceExt;
            let mut tvm_wf_uniform: crate::resources::ObjectUniform = bytemuck::Zeroable::zeroed();
            tvm_wf_uniform.model = glam::Mat4::IDENTITY.to_cols_array_2d();
            tvm_wf_uniform.colour = [0.75, 0.75, 0.75, 1.0];
            tvm_wf_uniform.wireframe = 1;
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tvm_wireframe_uniform"),
                contents: bytemuck::cast_slice(&[tvm_wf_uniform]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tvm_wireframe_bg"),
                layout: &resources.object_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_texture.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_normal_map_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_ao_map_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&resources.fallback_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: resources.fallback_scalar_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            resources
                                .fallback_matcap_view
                                .as_ref()
                                .unwrap_or(&resources.fallback_texture.view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: resources.fallback_face_colour_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: resources.fallback_warp_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&resources.lut_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_metallic_roughness_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(
                            &resources.fallback_emissive_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: resources.fallback_position_override_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: resources.fallback_normal_override_buf.as_entire_binding(),
                    },
                ],
            });
            self.tvm_wireframe_buf = Some(buf);
            self.tvm_wireframe_bg = Some(bg);
        }

        // -- Frame stats --
        {
            let total = scene_items.len() as u32;
            let visible = scene_items.iter().filter(|i| !i.settings.hidden).count() as u32;
            let mut draw_calls = 0u32;
            let mut triangles = 0u64;
            let instanced_batch_count = if self.use_instancing {
                self.instanced_batches.len() as u32
            } else {
                0
            };

            if self.use_instancing {
                for batch in &self.instanced_batches {
                    if let Some(mesh) = resources.mesh_store.get(batch.mesh_id) {
                        draw_calls += 1;
                        triangles += (mesh.index_count / 3) as u64 * batch.instance_count as u64;
                    }
                }
            } else {
                for item in scene_items {
                    if item.settings.hidden {
                        continue;
                    }
                    if let Some(mesh) = resources.mesh_store.get(item.mesh_id) {
                        draw_calls += 1;
                        triangles += (mesh.index_count / 3) as u64;
                    }
                }
            }

            self.last_stats = crate::renderer::stats::FrameStats {
                total_objects: total,
                visible_objects: visible,
                culled_objects: total.saturating_sub(visible),
                draw_calls,
                instanced_batches: instanced_batch_count,
                batches_reuploaded,
                batches_skipped,
                triangles_submitted: triangles,
                shadow_draw_calls: 0, // Updated below in shadow pass.
                gpu_culling_active: self.gpu_culling_enabled,
                // Clear stale readback if GPU culling is off this frame.
                gpu_visible_instances: if self.gpu_culling_enabled {
                    self.last_stats.gpu_visible_instances
                } else {
                    None
                },
                ..self.last_stats
            };
        }

        // ------------------------------------------------------------------
        // Shadow depth pass : CSM: render each cascade into its atlas tile.
        // Skip the pass entirely when over budget and shadow reduction is allowed.
        // ------------------------------------------------------------------
        let skip_shadows = self.degradation_shadows_skipped;

        // When skipping the shadow pass (budget pressure or empty scene), clear the
        // atlas to max depth so that stale values from a previous frame or a previous
        // showcase don't produce phantom shadows.
        if lighting.shadows_enabled && (skip_shadows || scene_items.is_empty()) {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_clear_encoder"),
            });
            let _ = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_clear_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &resources.shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            queue.submit(std::iter::once(enc.finish()));
        }

        if lighting.shadows_enabled && !scene_items.is_empty() && !skip_shadows {
            // ------------------------------------------------------------------
            // Shadow GPU cull dispatch
            //
            // For each active cascade, dispatch `cull_instances` + `write_indirect_args`
            // with the cascade frustum. Results land in `shadow_vis_bufs[c]` and
            // `shadow_indirect_bufs[c]`, consumed by the shadow render pass below.
            // All cascade dispatches share the same `batch_counter_buf`; each
            // `write_indirect_args` dispatch resets the counters for the next cascade.
            // ------------------------------------------------------------------
            if self.gpu_culling_enabled
                && self.use_instancing
                && !self.instanced_batches.is_empty()
                && self.cached_instance_count > 0
            {
                // Mutable operations first.
                if self.cull_resources.is_none() {
                    self.cull_resources =
                        Some(crate::renderer::indirect::CullResources::new(device));
                }
                resources.ensure_cull_instance_pipelines(device);
                for c in 0..effective_cascade_count {
                    resources.get_shadow_cull_instance_bind_group(device, c);
                }

                let instance_count = self.cached_instance_count as u32;
                let batch_count = self.instanced_batches.len() as u32;

                if let (Some(aabb_buf), Some(meta_buf), Some(counter_buf)) = (
                    resources.instance_aabb_buf.as_ref(),
                    resources.batch_meta_buf.as_ref(),
                    resources.batch_counter_buf.as_ref(),
                ) {
                    let cull = self.cull_resources.as_ref().unwrap();
                    let mut shadow_cull_encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("shadow_cull_encoder"),
                        });
                    for c in 0..effective_cascade_count {
                        if let (Some(shadow_vis_buf), Some(shadow_indirect_buf)) = (
                            resources.shadow_vis_bufs[c].as_ref(),
                            resources.shadow_indirect_bufs[c].as_ref(),
                        ) {
                            let cpu_frustum = crate::camera::frustum::Frustum::from_view_proj(
                                &cascade_view_projs[c],
                            );
                            let sub = crate::plugin_api::CullSubmission {
                                instance_aabbs: aabb_buf,
                                instance_count,
                                batch_meta: meta_buf,
                                batch_count,
                                counter: counter_buf,
                                visible_out: shadow_vis_buf,
                                indirect_out: shadow_indirect_buf,
                                shadow_pass: true,
                            };
                            cull.dispatch(
                                &mut shadow_cull_encoder,
                                device,
                                queue,
                                &cpu_frustum,
                                Some(c),
                                &sub,
                            );
                        }
                    }
                    queue.submit(std::iter::once(shadow_cull_encoder.finish()));
                }
            }

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_pass_encoder"),
            });
            {
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &resources.shadow_map_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                let mut shadow_draws = 0u32;
                let tile_px = tile_size as f32;

                if self.use_instancing {
                    let use_shadow_indirect = self.gpu_culling_enabled
                        && resources.shadow_instanced_cull_pipeline.is_some()
                        && resources.shadow_vis_bufs[0].is_some();

                    if use_shadow_indirect {
                        // GPU-culled indirect shadow path.
                        for cascade in 0..effective_cascade_count {
                            let tile_col = (cascade % 2) as f32;
                            let tile_row = (cascade / 2) as f32;
                            shadow_pass.set_viewport(
                                tile_col * tile_px,
                                tile_row * tile_px,
                                tile_px,
                                tile_px,
                                0.0,
                                1.0,
                            );
                            shadow_pass.set_scissor_rect(
                                (tile_col * tile_px) as u32,
                                (tile_row * tile_px) as u32,
                                tile_size,
                                tile_size,
                            );

                            // Write cascade view-projection matrix.
                            queue.write_buffer(
                                resources.shadow_instanced_cascade_bufs[cascade]
                                    .as_ref()
                                    .expect("shadow_instanced_cascade_bufs not allocated"),
                                0,
                                bytemuck::cast_slice(
                                    &cascade_view_projs[cascade].to_cols_array_2d(),
                                ),
                            );

                            let Some(pipeline) = resources.shadow_instanced_cull_pipeline.as_ref()
                            else {
                                continue;
                            };
                            let Some(cascade_bg) =
                                resources.shadow_instanced_cascade_bgs[cascade].as_ref()
                            else {
                                continue;
                            };
                            let Some(inst_cull_bg) =
                                resources.shadow_cull_instance_bgs[cascade].as_ref()
                            else {
                                continue;
                            };
                            let Some(shadow_indirect_buf) =
                                resources.shadow_indirect_bufs[cascade].as_ref()
                            else {
                                continue;
                            };

                            shadow_pass.set_pipeline(pipeline);
                            shadow_pass.set_bind_group(0, cascade_bg, &[]);
                            shadow_pass.set_bind_group(1, inst_cull_bg, &[]);

                            for (bi, batch) in self.instanced_batches.iter().enumerate() {
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                shadow_pass
                                    .draw_indexed_indirect(shadow_indirect_buf, bi as u64 * 20);
                                shadow_draws += 1;
                            }
                        }
                    } else if let (Some(pipeline), Some(instance_bg)) = (
                        &resources.shadow_instanced_pipeline,
                        self.instanced_batches.first().and_then(|b| {
                            resources.instance_bind_groups.get(&(
                                b.texture_id.unwrap_or(u64::MAX),
                                b.normal_map_id.unwrap_or(u64::MAX),
                                b.ao_map_id.unwrap_or(u64::MAX),
                            ))
                        }),
                    ) {
                        // Direct draw shadow path (fallback when GPU culling is off).
                        for cascade in 0..effective_cascade_count {
                            let tile_col = (cascade % 2) as f32;
                            let tile_row = (cascade / 2) as f32;
                            shadow_pass.set_viewport(
                                tile_col * tile_px,
                                tile_row * tile_px,
                                tile_px,
                                tile_px,
                                0.0,
                                1.0,
                            );
                            shadow_pass.set_scissor_rect(
                                (tile_col * tile_px) as u32,
                                (tile_row * tile_px) as u32,
                                tile_size,
                                tile_size,
                            );

                            shadow_pass.set_pipeline(pipeline);

                            queue.write_buffer(
                                resources.shadow_instanced_cascade_bufs[cascade]
                                    .as_ref()
                                    .expect("shadow_instanced_cascade_bufs not allocated"),
                                0,
                                bytemuck::cast_slice(
                                    &cascade_view_projs[cascade].to_cols_array_2d(),
                                ),
                            );

                            let cascade_bg = resources.shadow_instanced_cascade_bgs[cascade]
                                .as_ref()
                                .expect("shadow_instanced_cascade_bgs not allocated");
                            shadow_pass.set_bind_group(0, cascade_bg, &[]);
                            shadow_pass.set_bind_group(1, instance_bg, &[]);

                            for batch in &self.instanced_batches {
                                if batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                shadow_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                                shadow_draws += 1;
                            }
                        }
                    }

                    // Per-item shadow casters for items excluded from
                    // instanced batches.
                    //
                    // `sorted_items` in the instancing builder filters out
                    // items with: active scalar attributes, two-sided
                    // materials, matcap shading, UV/param visualisation,
                    // per-instance deformer data (skinned), and
                    // position/normal overrides. Those items render through
                    // the per-item lit path; without this loop they would
                    // also be silently dropped from the shadow atlas. Draw
                    // them here with the non-instanced `shadow_pipeline`
                    // (group 0 = `shadow_bind_group` with cascade dynamic
                    // offset, group 1 = mesh.object_bind_group, group 2 =
                    // per-mesh deform sidecar).
                    let filter_results = &self.compute_filter_results;
                    for cascade in 0..effective_cascade_count {
                        let tile_col = (cascade % 2) as f32;
                        let tile_row = (cascade / 2) as f32;
                        shadow_pass.set_viewport(
                            tile_col * tile_px,
                            tile_row * tile_px,
                            tile_px,
                            tile_px,
                            0.0,
                            1.0,
                        );
                        shadow_pass.set_scissor_rect(
                            (tile_col * tile_px) as u32,
                            (tile_row * tile_px) as u32,
                            tile_size,
                            tile_size,
                        );
                        shadow_pass.set_pipeline(&resources.shadow_pipeline);
                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow_bind_group,
                            &[cascade as u32 * 256],
                        );

                        let cascade_frustum = crate::camera::frustum::Frustum::from_view_proj(
                            &cascade_view_projs[cascade],
                        );

                        for item in scene_items.iter() {
                            if item.settings.hidden
                                || !item.settings.cast_shadows
                                || item.settings.opacity < 1.0
                            {
                                continue;
                            }
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };

                            // Mirror the inclusion filter from
                            // `sorted_items` (instancing builder). When
                            // every condition holds, the item was drawn by
                            // the instanced shadow path and must not be
                            // drawn again here.
                            let in_instanced_batch = item.active_attribute.is_none()
                                && !item.material.is_two_sided()
                                && item.material.matcap_id().is_none()
                                && item.material.param_vis.is_none()
                                && !filter_results.iter().any(|r| r.mesh_id == item.mesh_id)
                                && !resources.deform.has_per_instance_deform_data(
                                    item.mesh_id,
                                    item.deform_instance,
                                )
                                && mesh.position_override_buffer.is_none()
                                && mesh.normal_override_buffer.is_none();
                            if in_instanced_batch {
                                continue;
                            }

                            let world_aabb = mesh
                                .aabb
                                .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                            if cascade_frustum.cull_aabb(&world_aabb) {
                                continue;
                            }

                            shadow_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            shadow_pass.set_bind_group(
                                2,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance),
                                &[],
                            );
                            shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            shadow_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            shadow_draws += 1;
                        }
                    }
                } else {
                    for cascade in 0..effective_cascade_count {
                        let tile_col = (cascade % 2) as f32;
                        let tile_row = (cascade / 2) as f32;
                        shadow_pass.set_viewport(
                            tile_col * tile_px,
                            tile_row * tile_px,
                            tile_px,
                            tile_px,
                            0.0,
                            1.0,
                        );
                        shadow_pass.set_scissor_rect(
                            (tile_col * tile_px) as u32,
                            (tile_row * tile_px) as u32,
                            tile_size,
                            tile_size,
                        );

                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow_bind_group,
                            &[cascade as u32 * 256],
                        );
                        shadow_pass.set_pipeline(&resources.shadow_pipeline);

                        let cascade_frustum = crate::camera::frustum::Frustum::from_view_proj(
                            &cascade_view_projs[cascade],
                        );

                        for item in scene_items.iter() {
                            if item.settings.hidden {
                                continue;
                            }
                            if !item.settings.cast_shadows {
                                continue;
                            }
                            if item.settings.opacity < 1.0 {
                                continue;
                            }
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };

                            let world_aabb = mesh
                                .aabb
                                .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                            if cascade_frustum.cull_aabb(&world_aabb) {
                                continue;
                            }

                            shadow_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            shadow_pass.set_bind_group(
                                2,
                                resources
                                    .deform
                                    .instance_bind_group_for(item.mesh_id, item.deform_instance),
                                &[],
                            );
                            shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            shadow_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            shadow_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            shadow_draws += 1;
                        }
                    }
                }

                // Item-type plugin shadow casting: per-cascade dispatch
                // with viewport + scissor + cascade bind group set up by
                // the lib. The plugin map is read through a raw pointer
                // captured before the mutable resources borrow split off.
                let plugins = unsafe { &*item_type_plugins_ptr };
                if !plugins.is_empty() && !frame.scene.plugin_items.is_empty() {
                    for cascade in 0..effective_cascade_count {
                        let tile_col = (cascade % 2) as f32;
                        let tile_row = (cascade / 2) as f32;
                        shadow_pass.set_viewport(
                            tile_col * tile_px,
                            tile_row * tile_px,
                            tile_px,
                            tile_px,
                            0.0,
                            1.0,
                        );
                        shadow_pass.set_scissor_rect(
                            (tile_col * tile_px) as u32,
                            (tile_row * tile_px) as u32,
                            tile_size,
                            tile_size,
                        );
                        shadow_pass.set_bind_group(
                            0,
                            &resources.shadow_bind_group,
                            &[cascade as u32 * 256],
                        );
                        let ctx = crate::plugin_api::ShadowCastContext {
                            cascade_idx: cascade as u32,
                            light_view_proj: cascade_view_projs[cascade],
                            camera: &frame.camera.render_camera,
                            viewport_index: frame.camera.viewport_index,
                            frame_index: plugin_frame_index,
                        };
                        for (name, plugin) in plugins.iter() {
                            if let Some(items) = frame.scene.plugin_items.get(*name) {
                                plugin.cast_shadow_pass(&mut shadow_pass, &ctx, items.as_ref());
                            }
                        }
                    }
                }

                drop(shadow_pass);
                self.last_stats.shadow_draw_calls = shadow_draws;
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        // ----------------------------------------------------------------
        // Point-light cubemap shadow passes.
        //
        // One depth-only render pass per (slot, face). Reuses the cascade
        // shadow rasterisation conventions (front-face culling, opaque
        // casters only) but writes linear distance-to-light to frag_depth
        // via `shadow_point_pipeline`. Per-face culling uses the standard
        // CPU frustum from the face's view-projection.
        // ----------------------------------------------------------------
        if lighting.shadows_enabled && !scene_items.is_empty() && !point_shadow_faces.is_empty() {
            // Make sure each casting item's `mesh.object_uniform_buf` carries
            // the item's current world model matrix. When instancing is on,
            // the per-item write-buffer pass earlier in `prepare_scene_internal`
            // skips items that go through the instanced path, leaving the
            // shared per-mesh uniform stale. The cascade shadow pass works
            // around this by drawing instanced items through a different
            // pipeline; the point shadow path here renders every caster
            // through `shadow_point_pipeline` (non-instanced), so it needs
            // a fresh per-mesh write here. Multi-item-per-mesh scenes need a
            // dedicated per-item shadow uniform; the single-item case is the
            // priority bug fix.
            for item in scene_items.iter() {
                if item.settings.hidden
                    || !item.settings.cast_shadows
                    || item.settings.opacity < 1.0
                {
                    continue;
                }
                let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                    continue;
                };
                // Only the model matrix (offset 0, 64 bytes) is read by
                // `shadow_point.wgsl`. Write just that prefix so we don't
                // clobber the rest of the per-mesh `ObjectUniform`.
                queue.write_buffer(
                    &mesh.object_uniform_buf,
                    0,
                    bytemuck::cast_slice(&item.model),
                );
            }

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("point_shadow_encoder"),
            });
            for fc in &point_shadow_faces {
                let layer = fc.slot * 6 + fc.face;
                let view = &resources.point_shadow_face_views[layer as usize];
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("point_shadow_face_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&resources.shadow_point_pipeline);
                let dyn_offset = layer * POINT_FACE_STRIDE as u32;
                pass.set_bind_group(0, &resources.shadow_point_face_bind_group, &[dyn_offset]);

                let face_frustum = crate::camera::frustum::Frustum::from_view_proj(&fc.view_proj);

                for item in scene_items.iter() {
                    if item.settings.hidden
                        || !item.settings.cast_shadows
                        || item.settings.opacity < 1.0
                    {
                        continue;
                    }
                    let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                        continue;
                    };
                    let world_aabb = mesh
                        .aabb
                        .transformed(&glam::Mat4::from_cols_array_2d(&item.model));
                    if face_frustum.cull_aabb(&world_aabb) {
                        continue;
                    }
                    pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                    pass.set_bind_group(
                        2,
                        resources
                            .deform
                            .instance_bind_group_for(item.mesh_id, item.deform_instance),
                        &[],
                    );
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
            queue.submit(std::iter::once(enc.finish()));
        }
    }

    /// Per-viewport prepare stage: camera, clip planes, clip volume, grid, overlays, cap geometry, axes.
    ///
    /// Call once per viewport per frame, after `prepare_scene_internal`.
    /// Reads `viewport_fx` for clip planes, clip volume, cap fill, and post-process settings.
    pub(super) fn prepare_viewport_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
    ) {
        // Ensure a per-viewport camera slot exists for this viewport index.
        self.ensure_viewport_slot(device, frame.camera.viewport_index);
        self.prepare_clip_uniforms(queue, frame, viewport_fx);
        self.prepare_interaction_state(device, queue, frame, viewport_fx);
        self.prepare_outline_pass(device, queue, frame);
        self.prepare_sub_highlight(device, queue, frame);

        self.prepare_overlay_labels(device, queue, frame);
        self.prepare_scalar_bars(device, queue, frame);
        self.prepare_rulers(device, queue, frame);
        self.prepare_loading_bars(device, queue, frame);
        self.prepare_overlay_shapes(device, frame);
        self.prepare_splat_sort(device, queue, frame);
        self.prepare_splat_wireframe(device, queue, frame);
        self.prepare_sprite_wireframe(device, queue, frame);
        self.prepare_debug_buffer(device, frame);
        self.prepare_atlas_blit(queue, frame, viewport_fx);
    }

    /// Upload per-frame data to GPU buffers and render the shadow pass.
    /// Call before `paint()`.
    ///
    /// Returns [`crate::FrameStats`] with per-frame timing and upload metrics.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) -> crate::renderer::stats::FrameStats {
        let prepare_start = std::time::Instant::now();

        // Dispatch item-type plugin prepare work first so any GPU outputs
        // the plugin produces are visible to the rest of `prepare`.
        let plugin_bufs = self.dispatch_plugin_prepare(device, queue, frame);
        if !plugin_bufs.is_empty() {
            queue.submit(plugin_bufs);
        }

        // Run plugin culling for the current camera frustum so subsequent
        // plugin paint/shadow calls can skip culled items.
        if !self.item_type_plugins.is_empty() && !frame.scene.plugin_items.is_empty() {
            let vp = frame.camera.render_camera.view_proj();
            let frustum = crate::camera::frustum::Frustum::from_view_proj(&vp);
            self.dispatch_plugin_cull(&frustum, frame);
        }

        // Read back GPU timestamps from the previous frame, if available.
        // By the time prepare() is called, the previous frame's queue.submit() has
        // already happened, so it is safe to initiate the map here.
        if self.ts_needs_readback {
            if let Some(ref stg_buf) = self.ts_staging_buf {
                let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
                stg_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                // Non-blocking poll: flush any completed callbacks. GPU work from the
                // previous frame is almost certainly done by the time CPU reaches here.
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(std::time::Duration::from_millis(100)),
                    })
                    .ok();
                if rx.try_recv().unwrap_or(Err(wgpu::BufferAsyncError)).is_ok() {
                    let data = stg_buf.slice(..).get_mapped_range();
                    let t0 = u64::from_le_bytes(data[0..8].try_into().unwrap());
                    let t1 = u64::from_le_bytes(data[8..16].try_into().unwrap());
                    drop(data);
                    // ts_period is nanoseconds/tick; convert delta to milliseconds.
                    let gpu_ms = t1.saturating_sub(t0) as f32 * self.ts_period / 1_000_000.0;
                    self.last_stats.gpu_frame_ms = Some(gpu_ms);
                }
                stg_buf.unmap();
            }
            self.ts_needs_readback = false;
        }

        // Read back GPU-visible instance count from the previous frame's indirect args copy.
        // The cull pass from the previous frame has already been submitted and is almost
        // certainly done by the time prepare() is called; a short poll is enough.
        if self.indirect_readback_pending {
            if let Some(ref stg_buf) = self.indirect_readback_buf {
                let bytes = self.indirect_readback_batch_count as u64 * 20;
                if bytes > 0 {
                    let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
                    stg_buf
                        .slice(..bytes)
                        .map_async(wgpu::MapMode::Read, move |r| {
                            let _ = tx.send(r);
                        });
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(std::time::Duration::from_millis(100)),
                        })
                        .ok();
                    if rx.try_recv().unwrap_or(Err(wgpu::BufferAsyncError)).is_ok() {
                        let data = stg_buf.slice(..bytes).get_mapped_range();
                        let mut visible: u32 = 0;
                        for i in 0..self.indirect_readback_batch_count as usize {
                            // DrawIndexedIndirect layout: [index_count, instance_count, first_index, base_vertex, first_instance]
                            // instance_count is at byte offset 4 within each 20-byte entry.
                            let off = i * 20 + 4;
                            let n = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                            visible = visible.saturating_add(n);
                        }
                        drop(data);
                        self.last_stats.gpu_visible_instances = Some(visible);
                    }
                    stg_buf.unmap();
                }
            }
            self.indirect_readback_pending = false;
        }

        // Wall-clock duration since the previous prepare() call approximates the frame interval.
        let total_frame_ms = self
            .last_prepare_instant
            .map(|t| t.elapsed().as_secs_f32() * 1000.0)
            .unwrap_or(0.0);

        // Snapshot geometry upload bytes accumulated since the last frame, then reset.
        let upload_bytes = self.resources.frame_upload_bytes;
        self.resources.frame_upload_bytes = 0;

        // Resolve effective scale bounds and degradation flags.
        // When a preset is set it overrides the individual fields; the individual
        // fields are preserved so they restore when switching back to None.
        let policy = self.performance_policy;
        let (eff_min_scale, eff_max_scale, eff_allow_shadows, eff_allow_volumes, eff_allow_effects) =
            match policy.preset {
                Some(crate::renderer::stats::QualityPreset::High) => {
                    (1.0_f32, 1.0_f32, false, false, false)
                }
                Some(crate::renderer::stats::QualityPreset::Medium) => {
                    (0.75_f32, 1.0_f32, true, false, true)
                }
                Some(crate::renderer::stats::QualityPreset::Low) => {
                    (0.5_f32, 0.75_f32, true, true, true)
                }
                None => (
                    policy.min_render_scale,
                    policy.max_render_scale,
                    policy.allow_shadow_reduction,
                    policy.allow_volume_quality_reduction,
                    policy.allow_effect_throttling,
                ),
            };

        // Capture mode: force max render scale and suppress all degradation.
        // The adaptation controller is paused for the duration of the frame.
        let in_capture = self.runtime_mode == crate::renderer::stats::RuntimeMode::Capture;
        if in_capture {
            self.current_render_scale = eff_max_scale;
        }

        // When a preset is active, clamp current_render_scale to the preset's bounds
        // immediately, without requiring allow_dynamic_resolution. This ensures the
        // preset has a visible effect even when the adaptation controller is off.
        // The controller can still adjust within these bounds when enabled.
        if !in_capture && policy.preset.is_some() {
            self.current_render_scale = self
                .current_render_scale
                .clamp(eff_min_scale, eff_max_scale);
        }

        // Tiered degradation ladder.
        // Order: render scale -> shadows -> volumes -> effects.
        // The tier advances one step per over-budget frame once render scale has
        // reached its minimum (nothing more the controller can reduce).
        // The tier retreats one step per frame that is comfortably under budget,
        // reversing the ladder in the same order (effects first).
        // Capture mode resets the tier; otherwise advance/retreat based on budget.
        let missed_prev = self.last_stats.missed_budget;
        let under_prev = !self.last_stats.missed_budget
            && policy
                .target_fps
                .map(|fps| {
                    let budget = 1000.0 / fps;
                    let sig = self
                        .last_stats
                        .gpu_frame_ms
                        .unwrap_or(self.last_stats.total_frame_ms);
                    sig < budget * 0.8
                })
                .unwrap_or(true);
        if in_capture {
            self.degradation_tier = 0;
        } else {
            let at_min = !policy.allow_dynamic_resolution
                || self.current_render_scale <= eff_min_scale + 0.001;
            if missed_prev && at_min {
                self.degradation_tier = (self.degradation_tier + 1).min(3);
            } else if under_prev {
                self.degradation_tier = self.degradation_tier.saturating_sub(1);
            }
        }

        // Derive per-pass flags from the current tier and effective policy.
        // All flags are suppressed in Capture mode regardless of tier.
        self.degradation_shadows_skipped =
            !in_capture && self.degradation_tier >= 1 && eff_allow_shadows;
        self.degradation_volume_quality_reduced =
            !in_capture && self.degradation_tier >= 2 && eff_allow_volumes;
        self.degradation_effects_throttled =
            !in_capture && self.degradation_tier >= 3 && eff_allow_effects;

        // Cache scene items for renderer.pick() dispatch.
        {
            let surfaces = match &frame.scene.surfaces {
                SurfaceSubmission::Flat(items) => items.as_ref(),
            };
            // Mirror the rendering-side scene_items construction: opaque
            // volume meshes appear as boundary SceneRenderItems for face/vertex
            // picking via the BVH; cell-level remapping is then driven from
            // `pick_volume_mesh_items` (which keeps the full VolumeMeshItem so
            // `face_to_cell` is available).
            self.pick_scene_items = surfaces
                .iter()
                .cloned()
                .chain(
                    frame
                        .scene
                        .volume_meshes
                        .iter()
                        .filter(|item| item.transparency.is_none())
                        .map(|item| item.to_render_item()),
                )
                .collect();
            self.pick_point_cloud_items = frame.scene.point_clouds.clone();
            self.pick_splat_items = frame.scene.gaussian_splats.clone();
            self.pick_volume_items = frame.scene.volumes.clone();
            // Picking iterates the unified volume_meshes collection; no separate
            // transparent-only cache is needed any more.
            self.pick_scatter_volume_items = frame.scene.scatter_volumes.clone();
            self.prepared_scatter_volumes.clear();
            self.prepared_refraction_volumes.clear();
            let global_wireframe = frame.viewport.wireframe_mode;
            let eye = frame.camera.render_camera.eye_position;
            for item in &frame.scene.scatter_volumes {
                if item.settings.hidden || item.settings.wireframe || global_wireframe {
                    continue;
                }
                let mut flags: u32 = 0;
                if item.settings.unlit {
                    flags |= crate::scene::scatter_volume::SCATTER_FLAG_UNLIT;
                }
                if item.settings.receive_shadows {
                    flags |= crate::scene::scatter_volume::SCATTER_FLAG_RECEIVE_SHADOWS;
                }
                self.prepared_scatter_volumes.push((
                    item.volume.clone(),
                    item.settings.opacity,
                    flags,
                ));
                if item.volume.refraction.is_some() {
                    self.prepared_refraction_volumes
                        .push((item.volume.clone(), item.settings.opacity));
                }
            }
            // Sort back-to-front for the per-volume scatter draws. The
            // metric is the maximum corner distance of the volume's world
            // AABB from the eye, descending. Centroid distance flips order
            // when one volume contains another (huge fog containing a small
            // fire) -- the fire centroid can land on either side of the fog
            // centroid as the camera orbits, causing the alpha-over composite
            // to swap visibly. Sorting by far-corner distance keeps
            // containers (whose far corner is much further from the eye)
            // strictly behind contained volumes regardless of camera angle.
            self.prepared_scatter_volumes.sort_by(|a, b| {
                let aabb_a = a.0.world_aabb();
                let aabb_b = b.0.world_aabb();
                let far_corner = |aabb: &crate::Aabb| -> f32 {
                    let cx = if (aabb.min.x - eye[0]).abs() > (aabb.max.x - eye[0]).abs() {
                        aabb.min.x
                    } else {
                        aabb.max.x
                    };
                    let cy = if (aabb.min.y - eye[1]).abs() > (aabb.max.y - eye[1]).abs() {
                        aabb.min.y
                    } else {
                        aabb.max.y
                    };
                    let cz = if (aabb.min.z - eye[2]).abs() > (aabb.max.z - eye[2]).abs() {
                        aabb.min.z
                    } else {
                        aabb.max.z
                    };
                    (cx - eye[0]).powi(2) + (cy - eye[1]).powi(2) + (cz - eye[2]).powi(2)
                };
                let da = far_corner(&aabb_a);
                let db = far_corner(&aabb_b);
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });
            self.pick_volume_mesh_items = frame.scene.volume_meshes.clone();
            self.pick_polyline_items = frame.scene.polylines.clone();
            self.pick_glyph_items = frame.scene.glyphs.clone();
            self.pick_tensor_glyph_items = frame.scene.tensor_glyphs.clone();
            self.pick_sprite_items = frame.scene.sprite_items.clone();
            self.pick_streamtube_items = frame.scene.streamtube_items.clone();
            self.pick_tube_items = frame.scene.tube_items.clone();
            self.pick_ribbon_items = frame.scene.ribbon_items.clone();
            self.pick_image_slice_items = frame.scene.image_slices.clone();
            self.pick_volume_surface_slice_items = frame.scene.volume_surface_slices.clone();
            self.pick_screen_image_items = frame.scene.screen_images.clone();
        }

        let (scene_fx, viewport_fx) = frame.effects.split();
        self.prepare_scene_internal(device, queue, frame, &scene_fx);
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx);

        let cpu_prepare_ms = prepare_start.elapsed().as_secs_f32() * 1000.0;

        let budget_ms = policy.target_fps.map(|fps| 1000.0 / fps);

        // Controller signal: prefer gpu_frame_ms (excludes vsync wait, one-frame lag is
        // acceptable). Fall back to total_frame_ms when GPU timestamps are unavailable:
        // it reflects wall-clock frame duration and correctly fires over-budget at low
        // frame rates. cpu_prepare_ms is not used as a fallback because it only measures
        // CPU-side work and is low even when the GPU or driver is the bottleneck.
        let controller_ms = self.last_stats.gpu_frame_ms.unwrap_or(total_frame_ms);

        // Capture mode always reports missed_budget = false; degradation is suppressed.
        let missed_budget = !in_capture && budget_ms.map(|b| controller_ms > b).unwrap_or(false);

        // Adaptation controller: adjust render scale within effective bounds when enabled.
        // Uses controller_ms from the previous frame (gpu_frame_ms when available,
        // otherwise total_frame_ms). Paused in Capture mode.
        if policy.allow_dynamic_resolution && !in_capture {
            if let Some(budget) = budget_ms {
                if controller_ms > budget {
                    // Over budget: step down quickly.
                    self.current_render_scale =
                        (self.current_render_scale - 0.1).max(eff_min_scale);
                } else if controller_ms < budget * 0.8 {
                    // Comfortably under budget: recover slowly to avoid oscillation.
                    self.current_render_scale =
                        (self.current_render_scale + 0.05).min(eff_max_scale);
                }
            }
        }

        self.last_prepare_instant = Some(prepare_start);
        self.frame_counter = self.frame_counter.wrapping_add(1);

        let reported_render_scale = self.current_render_scale;

        let stats = crate::renderer::stats::FrameStats {
            cpu_prepare_ms,
            // gpu_frame_ms is updated by the timestamp readback above when available;
            // propagate the most recent value from last_stats.
            gpu_frame_ms: self.last_stats.gpu_frame_ms,
            total_frame_ms,
            render_scale: reported_render_scale,
            missed_budget,
            upload_bytes,
            shadows_skipped: self.degradation_shadows_skipped,
            volume_quality_reduced: self.degradation_volume_quality_reduced,
            // effects_throttled is set by the render path; carry forward here so
            // prepare()-only callers still see the previous frame's value until
            // paint_to()/render() updates it.
            effects_throttled: self.degradation_effects_throttled,
            ..self.last_stats
        };
        self.last_stats = stats;
        stats
    }
}
