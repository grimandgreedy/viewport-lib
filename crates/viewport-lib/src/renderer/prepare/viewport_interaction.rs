//! Per-viewport interaction prepare passes: clip-plane and clip-volume
//! uniforms, outline/x-ray masks, gizmo and axes geometry, the outline
//! offscreen edge-detection pass, and sub-object highlight geometry.

use super::*;
use crate::renderer::overlay_buffers::OverlayGeometrySlot;
use crate::resources::overlay::highlight::OutlineBinding;

impl ViewportRenderer {
    pub(super) fn prepare_clip_uniforms(
        &mut self,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
    ) {
        let gp_cascade0_mat = self.shadow.last_cascade0_shadow_mat.to_cols_array_2d();

        {
            let resources = &mut self.resources;

            // Upload clip planes + clip volume uniforms from clip_objects.
            {
                let (clip_uniform, clip_vols_uniform) =
                    clip_uniforms(frame, viewport_fx, &self.origin);
                if let Some((planes, volumes)) = clip_overflow(viewport_fx) {
                    resources.notices.report(
                        crate::notice::NoticeKey::single(
                            crate::NoticeCategory::Pipeline,
                            "clip_objects_over_cap",
                        ),
                        crate::Severity::Warning,
                        || crate::NoticeDetail::ClipObjectsOverCap { planes, volumes },
                    );
                }
                // Write to per-viewport slot buffer.
                if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
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
                }
                // Also write to shared buffers for legacy single-viewport callers.
                queue.write_buffer(
                    &resources.scene_tier.binds.clip_planes_buf,
                    0,
                    bytemuck::cast_slice(&[clip_uniform]),
                );
                queue.write_buffer(
                    &resources.scene_tier.binds.clip_volume_buf,
                    0,
                    bytemuck::cast_slice(&[clip_vols_uniform]),
                );
            }

            // Upload camera uniform to per-viewport slot buffer.
            let mut camera_uniform = frame
                .camera
                .camera_uniform(&self.frame_clock.current(), &self.origin);
            // The camera the GPU draws with, relative to the frame's origin.
            let gpu_camera = self.origin.camera(&frame.camera.render_camera);
            // On the HDR path the lit term may exceed 1.0; bound it at f16 max
            // so the Rgba16Float target stays finite. The LDR path keeps the
            // historical [0, 1] clamp.
            if frame.effects.display.is_hdr() {
                camera_uniform.lit_clamp = 65504.0;
            }
            // Write to shared buffer for legacy single-viewport callers.
            queue.write_buffer(
                &resources.scene_tier.binds.camera_uniform_buf,
                0,
                bytemuck::cast_slice(&[camera_uniform]),
            );
            // Write to the per-viewport slot buffer.
            if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                queue.write_buffer(&slot.camera_buf, 0, bytemuck::cast_slice(&[camera_uniform]));

                // Foreground pass camera: scene view with the (optional)
                // override projection. Only written when foreground work
                // exists this frame.
                if !frame.scene.foreground_items.is_empty() || !frame.scene.plugin_items.is_empty()
                {
                    let fg_camera = gpu_camera.foreground_camera(frame.effects.foreground.as_ref());
                    let mut fg_uniform = fg_camera.camera_uniform();
                    fg_uniform.lit_clamp = camera_uniform.lit_clamp;
                    fg_uniform.viewport = camera_uniform.viewport;
                    fg_uniform.time = camera_uniform.time;
                    fg_uniform.world_origin = camera_uniform.world_origin;
                    fg_uniform.origin_epoch = camera_uniform.origin_epoch;
                    queue.write_buffer(
                        &slot.foreground_camera_buf,
                        0,
                        bytemuck::cast_slice(&[fg_uniform]),
                    );
                }
            }

            // Upload grid uniform (full-screen analytical shader : no vertex buffers needed).
            if frame.viewport.show_grid {
                let eye = frame.camera.render_camera.eye_position();
                if !eye.is_finite() {
                    tracing::warn!(
                        eye_x = eye.x,
                        eye_y = eye.y,
                        eye_z = eye.z,
                        "grid skipped: eye_position is non-finite (camera distance overflow?)"
                    );
                } else {
                    let view_proj_mat = gpu_camera.view_proj().to_cols_array_2d();

                    let (spacing, minor_fade) = if frame.viewport.grid_cell_size > 0.0 {
                        (frame.viewport.grid_cell_size, 1.0_f32)
                    } else {
                        let vertical_depth = (eye.z - frame.viewport.grid_z).abs().max(1.0);
                        let world_per_pixel =
                            2.0 * (frame.camera.render_camera.fov_y() / 2.0).tan() * vertical_depth
                                / frame.camera.viewport_size[1].max(1.0);
                        let target = (world_per_pixel * 60.0).max(1e-9_f32);
                        let mut s = 1.0_f32;
                        let mut iters = 0u32;
                        while s < target {
                            s *= 10.0;
                            iters += 1;
                        }
                        let ratio = (target / s).clamp(0.0, 1.0);
                        let fade = if ratio < 0.5 {
                            1.0_f32
                        } else {
                            let t = (ratio - 0.5) * 2.0;
                            1.0 - t * t * (3.0 - 2.0 * t)
                        };
                        tracing::debug!(
                            eye_z = eye.z,
                            vertical_depth,
                            world_per_pixel,
                            target,
                            spacing = s,
                            lod_iters = iters,
                            ratio,
                            minor_fade = fade,
                            "grid LOD"
                        );
                        (s, fade)
                    };

                    let spacing_major = spacing * 10.0;
                    let snap_x = (eye.x / spacing_major).floor() * spacing_major;
                    let snap_y = (eye.y / spacing_major).floor() * spacing_major;
                    // The shader works relative to the origin; the snap point
                    // moves with it, subtracted in f64 so it stays small.
                    let origin = self.origin.origin();
                    let (rel_snap_x, rel_snap_y) = if self.origin.is_zero() {
                        (snap_x, snap_y)
                    } else {
                        (
                            (f64::from(snap_x) - origin.x) as f32,
                            (f64::from(snap_y) - origin.y) as f32,
                        )
                    };
                    tracing::debug!(
                        spacing_minor = spacing,
                        spacing_major,
                        snap_x,
                        snap_y,
                        eye_x = eye.x,
                        eye_y = eye.y,
                        eye_z = eye.z,
                        "grid snap"
                    );

                    let orient = frame.camera.render_camera.orientation();
                    let right = orient * glam::Vec3::X;
                    let up = orient * glam::Vec3::Y;
                    let back = orient * glam::Vec3::Z;
                    let cam_to_world = [
                        [right.x, right.y, right.z, 0.0_f32],
                        [up.x, up.y, up.z, 0.0_f32],
                        [back.x, back.y, back.z, 0.0_f32],
                    ];
                    let aspect =
                        frame.camera.viewport_size[0] / frame.camera.viewport_size[1].max(1.0);
                    let tan_half_fov = (frame.camera.render_camera.fov_y() / 2.0).tan();

                    let uniform = GridUniform {
                        view_proj: view_proj_mat,
                        cam_to_world,
                        tan_half_fov,
                        aspect,
                        _pad_ivp: [0.0; 2],
                        eye_pos: gpu_camera.eye_position().to_array(),
                        grid_z: self
                            .origin
                            .rel_plane_d(glam::Vec3::NEG_Z, frame.viewport.grid_z),
                        spacing_minor: spacing,
                        spacing_major,
                        snap_origin: [rel_snap_x, rel_snap_y],
                        colour_minor: {
                            let [r, g, b] = frame
                                .viewport
                                .grid_colour
                                .map(|c| c.to_linear_rgb())
                                .unwrap_or([0.55, 0.55, 0.55]);
                            [r, g, b, 0.4 * minor_fade]
                        },
                        colour_major: {
                            let [r, g, b] = frame
                                .viewport
                                .grid_colour
                                .map(|c| c.to_linear_rgb())
                                .unwrap_or([0.60, 0.60, 0.60]);
                            [r, g, b, 0.4 + 0.2 * minor_fade]
                        },
                    };
                    // Write to per-viewport slot buffer.
                    if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                        queue.write_buffer(&slot.grid_buf, 0, bytemuck::cast_slice(&[uniform]));
                    }
                    // Also write to shared buffer for legacy callers.
                    queue.write_buffer(
                        &resources.guides.grid_uniform_buf,
                        0,
                        bytemuck::cast_slice(&[uniform]),
                    );
                }
            }
            // ------------------------------------------------------------------
            // Ground plane uniform upload.
            // ------------------------------------------------------------------
            {
                let gp = &viewport_fx.ground_plane;
                let mode_u32: u32 = match gp.mode {
                    crate::renderer::types::GroundPlaneMode::None => 0,
                    crate::renderer::types::GroundPlaneMode::ShadowOnly => 1,
                    crate::renderer::types::GroundPlaneMode::Tile => 2,
                    crate::renderer::types::GroundPlaneMode::SolidColour => 3,
                };
                let orient = frame.camera.render_camera.orientation();
                let right = orient * glam::Vec3::X;
                let up = orient * glam::Vec3::Y;
                let back = orient * glam::Vec3::Z;
                let aspect = frame.camera.viewport_size[0] / frame.camera.viewport_size[1].max(1.0);
                let tan_half_fov = (frame.camera.render_camera.fov_y() / 2.0).tan();
                let vp = gpu_camera.view_proj().to_cols_array_2d();
                // The tiles' phase at the origin, so the checker stays put in
                // world space when drawn relative to it. Taken modulo a span of
                // 2^16 tiles, every tile size the shader's zoom levels reach
                // below it keeps its phase, and the offset stays small.
                let tile_origin = if self.origin.is_zero() {
                    [0.0; 2]
                } else {
                    let span = f64::from(gp.tile_size.max(1e-6)) * 65536.0;
                    let o = self.origin.origin();
                    [o.x.rem_euclid(span) as f32, o.y.rem_euclid(span) as f32]
                };
                let gp_uniform = crate::resources::GroundPlaneUniform {
                    view_proj: vp,
                    cam_right: [right.x, right.y, right.z, 0.0],
                    cam_up: [up.x, up.y, up.z, 0.0],
                    cam_back: [back.x, back.y, back.z, 0.0],
                    eye_pos: gpu_camera.eye_position().to_array(),
                    height: self.origin.rel_plane_d(glam::Vec3::NEG_Z, gp.height),
                    colour: gp.colour.to_linear_rgba(),
                    shadow_colour: gp.shadow_colour.to_linear_rgba(),
                    light_vp: gp_cascade0_mat,
                    tan_half_fov,
                    aspect,
                    tile_size: gp.tile_size,
                    shadow_bias: 0.002,
                    mode: mode_u32,
                    shadow_opacity: gp.shadow_opacity,
                    tile_origin,
                    colour2: gp.tile_colour2.to_linear_rgba(),
                };
                queue.write_buffer(
                    &resources.scene_tier.ground.uniform_buf,
                    0,
                    bytemuck::cast_slice(&[gp_uniform]),
                );
                if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                    queue.write_buffer(
                        &slot.ground_uniform_buf,
                        0,
                        bytemuck::cast_slice(&[gp_uniform]),
                    );
                }
            }
        } // `resources` mutable borrow dropped here.
    }

    pub(super) fn prepare_interaction_state(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
    ) {
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

        // ------------------------------------------------------------------
        // Build per-viewport interaction state into local variables.
        // Uses &self.resources (immutable) for BGL lookups; no conflict with
        // the slot borrow that follows.
        // ------------------------------------------------------------------

        let vp_idx = frame.camera.viewport_index;
        let origin = self.origin;

        // Outline mask buffers for selected objects (one per selected object).
        // Last frame's buffers are reused position by position.
        let mut old_outlines = std::mem::take(
            &mut self.viewport_slots[vp_idx]
                .selection_outlines
                .outline_object_buffers,
        )
        .into_iter()
        .map(|b| b.mask);
        let mut old_xray = std::mem::take(&mut self.viewport_slots[vp_idx].xray_object_buffers)
            .into_iter()
            .map(|(_, b)| b);
        let mut outline_object_buffers: Vec<OutlineObjectBuffers> = Vec::new();
        if frame.interaction.outline_selected {
            let resources = &self.resources;
            let colour_hdr = self.draws_hdr(frame);
            let clipping_active = DeviceResources::clipping_active(frame);
            for item in scene_items {
                // An item whose colour pipeline is still compiling is not on
                // screen, so it gets no outline until it is.
                if item.settings.hidden
                    || !item.settings.selected
                    || !resources.item_colour_ready(item, colour_hdr, clipping_active)
                {
                    continue;
                }
                // Mirror the mesh's position-override binding so the mask
                // rasterises the driven geometry, not the bind pose.
                let mesh = resources.mesh_store.get(item.mesh_id);
                let override_buf = mesh.and_then(|m| m.position_override_buffer.as_ref());
                let override_slice = mesh.and_then(|m| m.position_override_slice);
                let uniform = OutlineUniform {
                    model: origin.rel_model(item.model),
                    colour: [0.0; 4], // unused by mask shader
                    pixel_offset: 0.0,
                    has_position_override: if override_buf.is_some() { 1 } else { 0 },
                    position_override_base: override_slice.map_or(0, |s| s.base_element),
                    position_override_len: override_slice.map_or(u32::MAX, |s| s.element_count),
                    deform_flags: resources.scene_tier.deform.flag_bits(item.mesh_id),
                    deform_table: resources.scene_tier.deform.table_word(
                        item.mesh_id,
                        item.deform_instance,
                        mesh.and_then(|m| m.sidecar.as_ref()),
                    ),
                    _deform_pad: [0; 2],
                };
                let mask = OutlineBinding::reuse_or_new(
                    old_outlines.next(),
                    device,
                    queue,
                    &resources.scene_tier.outline.bind_group_layout,
                    uniform,
                    Some((
                        override_buf.unwrap_or(&resources.content.fallback_position_override_buf),
                        resources
                            .sidecar
                            .binding_buffer(mesh.and_then(|m| m.sidecar.as_ref())),
                    )),
                    "outline_mask_object_bg",
                );
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.mesh_id,
                    two_sided: resources.material_or_default(item.material).is_two_sided(),
                    mask,
                });
            }
            // Selected volume meshes: rasterise the boundary surface into the
            // outline mask. This covers both transparent (projected-tet) items,
            // whose boundary is not in the surface submission, and opaque items
            // submitted only through `volume_meshes` rather than as a separate
            // surface item. A host that also submits the opaque boundary as a
            // surface item draws the mask twice, but the R8 mask is a plain
            // white silhouette so the second draw is idempotent.
            for item in &frame.scene.volume_meshes {
                if item.settings.hidden || !item.settings.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: origin.rel_model(item.model),
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    has_position_override: 0,
                    position_override_base: 0,
                    position_override_len: u32::MAX,
                    deform_flags: 0,
                    deform_table: 0,
                    _deform_pad: [0; 2],
                };
                let mask = OutlineBinding::reuse_or_new(
                    old_outlines.next(),
                    device,
                    queue,
                    &resources.scene_tier.outline.bind_group_layout,
                    uniform,
                    Some((
                        &resources.content.fallback_position_override_buf,
                        resources.sidecar.fallback(),
                    )),
                    "outline_mask_object_bg",
                );
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.boundary_mesh_id,
                    two_sided: false,
                    mask,
                });
            }
        }

        // X-ray buffers for selected objects.
        let mut xray_object_buffers: Vec<(
            crate::resources::mesh::mesh_store::MeshId,
            OutlineBinding,
        )> = Vec::new();
        if frame.interaction.xray_selected {
            let resources = &self.resources;
            for item in scene_items {
                if item.settings.hidden || !item.settings.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: origin.rel_model(item.model),
                    colour: frame.interaction.xray_colour.to_linear_rgba(),
                    pixel_offset: 0.0,
                    has_position_override: 0,
                    position_override_base: 0,
                    position_override_len: u32::MAX,
                    deform_flags: 0,
                    deform_table: 0,
                    _deform_pad: [0; 2],
                };
                let binding = OutlineBinding::reuse_or_new(
                    old_xray.next(),
                    device,
                    queue,
                    &resources.scene_tier.outline.bind_group_layout,
                    uniform,
                    Some((
                        &resources.content.fallback_position_override_buf,
                        resources.sidecar.fallback(),
                    )),
                    "xray_object_bg",
                );
                xray_object_buffers.push((item.mesh_id, binding));
            }
        }

        // Constraint guide lines.
        let mut constraint_line_pool =
            std::mem::take(&mut self.viewport_slots[vp_idx].constraint_line_pool);
        let mut constraint_line_buffers = Vec::new();
        for (i, overlay) in frame.interaction.constraint_overlays.iter().enumerate() {
            if i == constraint_line_pool.len() {
                constraint_line_pool.push(OverlayGeometrySlot::new());
            }
            let (vertices, colour) =
                crate::resources::overlay::overlays::constraint_overlay_geometry(overlay);
            let indices: Vec<u32> = (0..vertices.len() as u32).collect();
            constraint_line_buffers.push(constraint_line_pool[i].write(
                device,
                queue,
                &self.resources.guides.overlay_bgl,
                &vertices,
                &indices,
                colour,
            ));
        }
        constraint_line_pool.truncate(constraint_line_buffers.len());

        // Clip-object visuals (outlines and the plane fill) are no longer drawn by
        // the renderer. The host builds them from `clip_plane::visual` and submits
        // them as ordinary scene primitives (polyline outlines and a translucent
        // fill mesh), tagged `ItemSettings::ignore_clip` so they stay visible where
        // the scene is clipped. The renderer only performs the clip operation and
        // the section cap fill below.

        // Section caps: one per clipping plane per item the clip applies to,
        // each at its plane's index in the clip-plane uniform so the shader
        // can skip that plane. Cached across frames on the viewport slot.
        let mut cap_cache = std::mem::take(&mut self.viewport_slots[vp_idx].cap_cache);
        cap_cache.begin_frame();
        let mut cap_buffers = Vec::new();
        let mut caps_built = 0u32;
        if viewport_fx.clip.cap_fill_enabled && clipping_planes(viewport_fx).next().is_some() {
            let hdr = self.draws_hdr(frame);
            self.resources.ensure_section_cap_pipeline(device, hdr);
            let resources = &self.resources;
            let cap_bgl = &resources.section_cap.as_ref().unwrap().bgl;
            for (plane_index, normal, distance, cap_colour) in clipping_planes(viewport_fx) {
                let plane_n = glam::Vec3::from(normal);
                // Caps are built and keyed relative to the origin, so a move
                // rebuilds them.
                let distance = origin.rel_plane_d(plane_n, distance);
                for item in scene_items
                    .iter()
                    .filter(|i| !i.settings.hidden && !i.settings.ignore_clip)
                {
                    let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                        continue;
                    };
                    let rel_model = origin.rel_model(item.model);
                    let model = glam::Mat4::from_cols_array_2d(&rel_model);
                    let world_aabb = mesh.aabb.transformed(&model);
                    if !world_aabb.intersects_plane(plane_n, distance) {
                        continue;
                    }
                    let (Some(pos), Some(idx)) = (&mesh.cpu_positions, &mesh.cpu_indices) else {
                        continue;
                    };
                    let colour = cap_colour.unwrap_or_else(|| {
                        let bc = item
                            .settings
                            .colour
                            .unwrap_or(resources.material_or_default(item.material).base_colour)
                            .to_linear_rgb();
                        [bc[0], bc[1], bc[2], 1.0]
                    });
                    let key = crate::renderer::section_caps::CapKey::new(
                        item.mesh_id,
                        mesh.content_rev,
                        &rel_model,
                        normal,
                        distance,
                        plane_index,
                    );
                    let (draw, built) = cap_cache.cap(device, queue, cap_bgl, key, colour, || {
                        crate::geometry::mesh::cap::generate_cap_mesh(
                            pos, idx, &model, plane_n, distance,
                        )
                    });
                    caps_built += built as u32;
                    cap_buffers.extend(draw);
                }
            }
        }
        cap_cache.evict_unseen();
        // Per viewport, like the overlay counters: the split API prepares a
        // viewport without assembling a `FrameStats`.
        self.last_stats.cap_meshes_built = caps_built;

        // The transform gizmo and the axes indicator are generated as 2D overlay
        // primitives during the overlay prepare (see `gizmo_overlay_items` and
        // `axes_overlay_items`); no per-viewport mesh upload happens here anymore.

        // ------------------------------------------------------------------
        // Assign all interaction state to the per-viewport slot.
        // ------------------------------------------------------------------
        {
            if !xray_object_buffers.is_empty() {
                self.resources.ensure_xray_pipeline(device);
            }
            if !constraint_line_buffers.is_empty() {
                self.resources.ensure_guide_overlay_pipelines(device);
            }
            let slot = &mut self.viewport_slots[vp_idx];
            slot.selection_outlines.outline_object_buffers = outline_object_buffers;
            slot.xray_object_buffers = xray_object_buffers;
            slot.constraint_line_buffers = constraint_line_buffers;
            slot.cap_buffers = cap_buffers;
            slot.constraint_line_pool = constraint_line_pool;
            slot.cap_cache = cap_cache;
        }
    }

    pub(super) fn prepare_outline_pass(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        if !self.outline_active(frame) {
            return;
        }
        let vp_idx = frame.camera.viewport_index;
        let ppp = frame.camera.pixels_per_point;
        let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let h = (frame.camera.viewport_size[1] * ppp).round() as u32;
        self.resources.ensure_outline_pipelines(device);
        // The mask tests against the LDR scene depth the HDR state holds.
        self.ensure_viewport_hdr(
            device,
            queue,
            vp_idx,
            w.max(1),
            h.max(1),
            frame.effects.post_process.ssaa_factor.max(1),
            self.current_render_scale,
            crate::resources::TargetGroups::LDR_DEPTH,
        );
        let size = self.viewport_slots[vp_idx]
            .hdr
            .as_ref()
            .expect("just ensured")
            .scene_size;
        self.encode_outline_passes(device, queue, frame, size, false, sink);
    }

    /// Record whether viewport `frame.camera.viewport_index` outlines anything
    /// this frame, which gates the mask and edge passes and the composite. The
    /// selected surfaces' mask buffers must already be built.
    ///
    /// Item-type plugins draw their own coverage into the mask, so they are
    /// not in the mask buffers; whether any has a selected item is recorded
    /// beside them.
    pub(super) fn outline_active(&mut self, frame: &FrameData) -> bool {
        let plugin_outline =
            frame.interaction.outline_selected && self.any_plugin_item_selected(frame);
        let outlines = &mut self.viewport_slots[frame.camera.viewport_index].selection_outlines;
        outlines.plugin_outline_present = plugin_outline;
        frame.interaction.outline_selected
            && (!outlines.outline_object_buffers.is_empty() || plugin_outline)
    }

    /// Draw the selection mask and run the edge pass into the viewport's
    /// outline targets at `size`, for the composite to blend over the scene.
    ///
    /// 1. Selected surfaces and the plugins' selected items draw white into
    ///    the R8 mask, tested against a cleared depth target: the LDR scene
    ///    depth on the full tier, the targets' own on the compatibility tier
    ///    (`own_depth`).
    /// 2. A fullscreen pass finds the mask's boundary and writes an
    ///    anti-aliased ring of the frame's outline colour and width.
    pub(super) fn encode_outline_passes(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        size: [u32; 2],
        own_depth: bool,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        let vp_idx = frame.camera.viewport_index;
        self.resources.ensure_outline_shared(device);
        let shared = self.resources.outline.as_ref().expect("just ensured");
        let slot = &mut self.viewport_slots[vp_idx];
        if !slot
            .outline_targets
            .as_ref()
            .is_some_and(|t| t.fits(size, own_depth))
        {
            slot.outline_targets = Some(crate::resources::outline::OutlineTargets::new(
                device,
                shared,
                self.resources.target_format,
                size,
                own_depth,
            ));
        }
        let slot = &self.viewport_slots[vp_idx];
        let targets = slot.outline_targets.as_ref().expect("just ensured");
        let [target_w, target_h] = targets.size();
        let edge_uniform = OutlineEdgeUniform {
            colour: frame.interaction.outline_colour.to_linear_rgba(),
            radius: frame.interaction.outline_width_px,
            viewport_w: target_w as f32,
            viewport_h: target_h as f32,
            _pad: 0.0,
        };
        queue.write_buffer(
            &targets.edge_uniform_buf,
            0,
            bytemuck::cast_slice(&[edge_uniform]),
        );
        let depth_view = match targets.depth_view() {
            Some(view) => view,
            None => {
                &slot
                    .hdr
                    .as_ref()
                    .expect("the full tier's LDR depth")
                    .outline_depth_view
            }
        };
        let Some((mask_pipeline, mask_two_sided)) = self.resources.outline_mask_pipelines() else {
            return;
        };

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("outline_offscreen_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("outline_mask_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: &targets.mask_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::plugin_api::depth::DEPTH_CLEAR),
                        store: crate::gpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &slot.camera_bind_group, &[]);
            for outlined in &slot.selection_outlines.outline_object_buffers {
                let Some(mesh) = self.resources.mesh_store.get(outlined.mesh_id) else {
                    continue;
                };
                pass.set_pipeline(if outlined.two_sided {
                    mask_two_sided
                } else {
                    mask_pipeline
                });
                pass.set_bind_group(1, &outlined.mask.bind_group, &[]);
                pass.set_vertex_buffer(0, self.resources.geometry.vertex_slice(mesh.vertex_span));
                pass.set_index_buffer(
                    self.resources.geometry.index_slice(mesh.index_span),
                    crate::gpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
            self.dispatch_plugin_outline_mask(&mut pass, frame);
        }
        {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("outline_edge_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: &targets.colour_view,
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
            pass.set_pipeline(&shared.edge_pipeline);
            pass.set_bind_group(0, &targets.edge_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        sink.push(encoder.finish());
    }

    pub(super) fn prepare_sub_highlight(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        let vp_idx = frame.camera.viewport_index;

        // ------------------------------------------------------------------
        // Sub-object highlight prepare: build GPU geometry from sub-selection
        // snapshot when the version has changed since the last frame.
        // ------------------------------------------------------------------
        {
            let w = frame.camera.viewport_size[0];
            let h = frame.camera.viewport_size[1];

            let has_sub_sel = frame.interaction.sub_selection.is_some();

            if has_sub_sel {
                let needs_rebuild = {
                    let slot = &self.viewport_slots[vp_idx];
                    let sel_version_changed = frame
                        .interaction
                        .sub_selection
                        .as_ref()
                        .map(|s| slot.sub_highlight_generation != s.version)
                        .unwrap_or(slot.sub_highlight_generation != u64::MAX);
                    sel_version_changed
                        || slot.sub_highlight.is_none()
                        || slot.sub_highlight_origin_epoch != self.origin.epoch()
                };
                if needs_rebuild {
                    self.resources.ensure_sub_highlight_pipelines(device);
                    let sel_ref = frame.interaction.sub_selection.as_ref();
                    let data = self.resources.build_sub_highlight(
                        device,
                        queue,
                        sel_ref,
                        &[],
                        frame
                            .interaction
                            .sub_highlight_face_fill_colour
                            .to_linear_rgba(),
                        frame.interaction.sub_highlight_edge_colour.to_linear_rgba(),
                        frame.interaction.sub_highlight_edge_width_px,
                        frame.interaction.sub_highlight_vertex_size_px,
                        w,
                        h,
                    );
                    let new_gen = frame
                        .interaction
                        .sub_selection
                        .as_ref()
                        .map(|s| s.version)
                        .unwrap_or(u64::MAX);
                    let slot = &mut self.viewport_slots[vp_idx];
                    slot.sub_highlight = Some(data);
                    slot.sub_highlight_generation = new_gen;
                    slot.sub_highlight_origin_epoch = self.origin.epoch();
                }
            } else {
                let slot = &mut self.viewport_slots[vp_idx];
                slot.sub_highlight = None;
                slot.sub_highlight_generation = u64::MAX;
            }
        }
    }
}

/// The clip-plane and clip-volume uniforms for a viewport, from its enabled
/// clip objects.
pub(super) fn clip_uniforms(
    frame: &FrameData,
    viewport_fx: &ViewportEffects<'_>,
    origin: &crate::renderer::origin::OriginState,
) -> (ClipPlanesUniform, ClipVolumesUniform) {
    let mut planes = [[0.0f32; 4]; 6];
    let mut count = 0u32;
    let mut clip_vols_uniform: ClipVolumesUniform = bytemuck::Zeroable::zeroed();
    let rel = |c: [f32; 3]| origin.rel(glam::Vec3::from(c)).to_array();

    for (i, normal, distance, _) in clipping_planes(viewport_fx) {
        let distance = origin.rel_plane_d(glam::Vec3::from(normal), distance);
        planes[i as usize] = [normal[0], normal[1], normal[2], distance];
        count = i + 1;
    }
    for obj in clipping_objects(viewport_fx) {
        if clip_vols_uniform.count as usize >= CLIP_VOLUME_MAX {
            break;
        }
        let idx = clip_vols_uniform.count as usize;
        let entry = match obj.shape {
            ClipShape::Box {
                center,
                half_extents,
                orientation,
            } => ClipVolumeEntry::from_box(rel(center), half_extents, orientation),
            ClipShape::Sphere { center, radius } => {
                ClipVolumeEntry::from_sphere(rel(center), radius)
            }
            ClipShape::Cylinder {
                center,
                axis,
                radius,
                half_length,
            } => ClipVolumeEntry::from_cylinder(rel(center), axis, radius, half_length),
            _ => continue,
        };
        clip_vols_uniform.volumes[idx] = entry;
        clip_vols_uniform.count += 1;
    }

    let ppp = frame.camera.pixels_per_point;
    let clip_uniform = ClipPlanesUniform {
        planes,
        count,
        _pad0: 0,
        // Physical pixels: clip_pos in the fragment shader is in physical pixels,
        // so split-screen and pixel-inspector buffer stride must match.
        viewport_width: (frame.camera.viewport_size[0] * ppp).max(1.0),
        viewport_height: (frame.camera.viewport_size[1] * ppp).max(1.0),
    };
    (clip_uniform, clip_vols_uniform)
}

/// The clip objects that clip geometry this frame.
fn clipping_objects<'a>(
    viewport_fx: &'a ViewportEffects<'_>,
) -> impl Iterator<Item = &'a crate::renderer::types::ClipObject> + 'a {
    viewport_fx
        .clip
        .objects
        .iter()
        .filter(|o| o.enabled && o.clip_geometry)
}

/// The planes the clip-plane uniform holds, with their index in it: normal,
/// distance and cap colour override. Planes past the uniform's six do not
/// clip.
fn clipping_planes<'a>(
    viewport_fx: &'a ViewportEffects<'_>,
) -> impl Iterator<Item = (u32, [f32; 3], f32, Option<[f32; 4]>)> + 'a {
    clipping_objects(viewport_fx)
        .filter_map(|o| match o.shape {
            ClipShape::Plane {
                normal,
                distance,
                cap_colour,
            } => Some((normal, distance, cap_colour)),
            _ => None,
        })
        .take(6)
        .zip(0u32..)
        .map(|((n, d, c), i)| (i, n, d, c))
}

/// Clipping planes and volumes in the frame, when either is past what the
/// clip uniforms hold.
fn clip_overflow(viewport_fx: &ViewportEffects<'_>) -> Option<(u32, u32)> {
    let (mut planes, mut volumes) = (0u32, 0u32);
    for o in clipping_objects(viewport_fx) {
        match o.shape {
            ClipShape::Plane { .. } => planes += 1,
            _ => volumes += 1,
        }
    }
    (planes > 6 || volumes as usize > CLIP_VOLUME_MAX).then_some((planes, volumes))
}

#[cfg(test)]
mod clip_plane_tests {
    use super::*;
    use crate::renderer::types::{ClipObject, EffectsFrame};

    fn effects(objects: Vec<ClipObject>) -> EffectsFrame {
        let mut fx = EffectsFrame::default();
        fx.clip.objects = objects;
        fx
    }

    /// A plane that does not clip, or is disabled, takes no slot, so the cap
    /// indices match the clip-plane uniform.
    #[test]
    fn plane_indices_skip_non_clipping_objects() {
        let mut inert = ClipObject::plane([0.0, 1.0, 0.0], 0.0);
        inert.clip_geometry = false;
        let mut off = ClipObject::plane([0.0, 0.0, 1.0], 0.0);
        off.enabled = false;
        let fx = effects(vec![
            inert,
            ClipObject::plane([1.0, 0.0, 0.0], 1.0),
            off,
            ClipObject::sphere([0.0; 3], 1.0),
            ClipObject::plane([-1.0, 0.0, 0.0], 2.0),
        ]);
        let (_, vfx) = fx.split();
        let planes: Vec<(u32, f32)> = clipping_planes(&vfx).map(|(i, _, d, _)| (i, d)).collect();
        assert_eq!(planes, vec![(0, 1.0), (1, 2.0)]);
        assert_eq!(clip_overflow(&vfx), None);
    }

    #[test]
    fn overflow_counts_clipping_objects() {
        let mut objects: Vec<ClipObject> = (0..7)
            .map(|i| ClipObject::plane([1.0, 0.0, 0.0], i as f32))
            .collect();
        let mut inert = ClipObject::sphere([0.0; 3], 1.0);
        inert.clip_geometry = false;
        objects.push(inert);
        let fx = effects(objects);
        let (_, vfx) = fx.split();
        assert_eq!(clipping_planes(&vfx).count(), 6);
        assert_eq!(clip_overflow(&vfx), Some((7, 0)));

        let fx = effects((0..5).map(|_| ClipObject::sphere([0.0; 3], 1.0)).collect());
        assert_eq!(clip_overflow(&fx.split().1), Some((0, 5)));
    }
}
