//! Per-viewport interaction prepare passes: clip-plane and clip-volume
//! uniforms, outline/x-ray masks, gizmo and axes geometry, the outline
//! offscreen edge-detection pass, and sub-object highlight geometry.

use super::*;

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
                let mut planes = [[0.0f32; 4]; 6];
                let mut count = 0u32;
                let mut clip_vols_uniform: ClipVolumesUniform = bytemuck::Zeroable::zeroed();

                for obj in viewport_fx
                    .clip
                    .objects
                    .iter()
                    .filter(|o| o.enabled && o.clip_geometry)
                {
                    match obj.shape {
                        ClipShape::Plane {
                            normal, distance, ..
                        } if count < 6 => {
                            planes[count as usize] = [normal[0], normal[1], normal[2], distance];
                            count += 1;
                        }
                        ClipShape::Box {
                            center,
                            half_extents,
                            orientation,
                        } if (clip_vols_uniform.count as usize) < CLIP_VOLUME_MAX => {
                            let idx = clip_vols_uniform.count as usize;
                            clip_vols_uniform.volumes[idx] =
                                ClipVolumeEntry::from_box(center, half_extents, orientation);
                            clip_vols_uniform.count += 1;
                        }
                        ClipShape::Sphere { center, radius }
                            if (clip_vols_uniform.count as usize) < CLIP_VOLUME_MAX =>
                        {
                            let idx = clip_vols_uniform.count as usize;
                            clip_vols_uniform.volumes[idx] =
                                ClipVolumeEntry::from_sphere(center, radius);
                            clip_vols_uniform.count += 1;
                        }
                        ClipShape::Cylinder {
                            center,
                            axis,
                            radius,
                            half_length,
                        } if (clip_vols_uniform.count as usize) < CLIP_VOLUME_MAX => {
                            let idx = clip_vols_uniform.count as usize;
                            clip_vols_uniform.volumes[idx] =
                                ClipVolumeEntry::from_cylinder(center, axis, radius, half_length);
                            clip_vols_uniform.count += 1;
                        }
                        _ => {}
                    }
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
                    &resources.binds.clip_planes_buf,
                    0,
                    bytemuck::cast_slice(&[clip_uniform]),
                );
                queue.write_buffer(
                    &resources.binds.clip_volume_buf,
                    0,
                    bytemuck::cast_slice(&[clip_vols_uniform]),
                );
            }

            // Upload camera uniform to per-viewport slot buffer.
            let mut camera_uniform = frame.camera.render_camera.camera_uniform();
            // On the HDR path the lit term may exceed 1.0; bound it at f16 max
            // so the Rgba16Float target stays finite. The LDR path keeps the
            // historical [0, 1] clamp.
            if frame.effects.display.is_hdr() {
                camera_uniform.lit_clamp = 65504.0;
            }
            // Write to shared buffer for legacy single-viewport callers.
            queue.write_buffer(
                &resources.binds.camera_uniform_buf,
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
                    let fg_camera = frame
                        .camera
                        .render_camera
                        .foreground_camera(frame.effects.foreground.as_ref());
                    let mut fg_uniform = fg_camera.camera_uniform();
                    fg_uniform.lit_clamp = camera_uniform.lit_clamp;
                    queue.write_buffer(
                        &slot.foreground_camera_buf,
                        0,
                        bytemuck::cast_slice(&[fg_uniform]),
                    );
                }
            }

            // Upload grid uniform (full-screen analytical shader : no vertex buffers needed).
            if frame.viewport.show_grid {
                let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
                if !eye.is_finite() {
                    tracing::warn!(
                        eye_x = eye.x,
                        eye_y = eye.y,
                        eye_z = eye.z,
                        "grid skipped: eye_position is non-finite (camera distance overflow?)"
                    );
                } else {
                    let view_proj_mat = frame.camera.render_camera.view_proj().to_cols_array_2d();

                    let (spacing, minor_fade) = if frame.viewport.grid_cell_size > 0.0 {
                        (frame.viewport.grid_cell_size, 1.0_f32)
                    } else {
                        let vertical_depth = (eye.z - frame.viewport.grid_z).abs().max(1.0);
                        let world_per_pixel =
                            2.0 * (frame.camera.render_camera.fov / 2.0).tan() * vertical_depth
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

                    let orient = frame.camera.render_camera.orientation;
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
                    let tan_half_fov = (frame.camera.render_camera.fov / 2.0).tan();

                    let uniform = GridUniform {
                        view_proj: view_proj_mat,
                        cam_to_world,
                        tan_half_fov,
                        aspect,
                        _pad_ivp: [0.0; 2],
                        eye_pos: frame.camera.render_camera.eye_position,
                        grid_z: frame.viewport.grid_z,
                        spacing_minor: spacing,
                        spacing_major,
                        snap_origin: [snap_x, snap_y],
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
                let orient = frame.camera.render_camera.orientation;
                let right = orient * glam::Vec3::X;
                let up = orient * glam::Vec3::Y;
                let back = orient * glam::Vec3::Z;
                let aspect = frame.camera.viewport_size[0] / frame.camera.viewport_size[1].max(1.0);
                let tan_half_fov = (frame.camera.render_camera.fov / 2.0).tan();
                let vp = frame.camera.render_camera.view_proj().to_cols_array_2d();
                let gp_uniform = crate::resources::GroundPlaneUniform {
                    view_proj: vp,
                    cam_right: [right.x, right.y, right.z, 0.0],
                    cam_up: [up.x, up.y, up.z, 0.0],
                    cam_back: [back.x, back.y, back.z, 0.0],
                    eye_pos: frame.camera.render_camera.eye_position,
                    height: gp.height,
                    colour: gp.colour.to_linear_rgba(),
                    shadow_colour: gp.shadow_colour.to_linear_rgba(),
                    light_vp: gp_cascade0_mat,
                    tan_half_fov,
                    aspect,
                    tile_size: gp.tile_size,
                    shadow_bias: 0.002,
                    mode: mode_u32,
                    shadow_opacity: gp.shadow_opacity,
                    _pad: [0.0; 2],
                    colour2: gp.tile_colour2.to_linear_rgba(),
                };
                queue.write_buffer(
                    &resources.ground.uniform_buf,
                    0,
                    bytemuck::cast_slice(&[gp_uniform]),
                );
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

        // Outline mask buffers for selected objects (one per selected object).
        let mut outline_object_buffers: Vec<OutlineObjectBuffers> = Vec::new();
        if frame.interaction.outline_selected {
            let resources = &self.resources;
            for item in scene_items {
                if item.settings.hidden || !item.settings.selected {
                    continue;
                }
                // Mirror the mesh's position-override binding so the mask
                // rasterises the driven geometry, not the bind pose.
                let mesh = resources.mesh_store.get(item.mesh_id);
                let override_buf = mesh.and_then(|m| m.position_override_buffer.as_ref());
                let override_slice = mesh.and_then(|m| m.position_override_slice);
                let uniform = OutlineUniform {
                    model: item.model,
                    colour: [0.0; 4], // unused by mask shader
                    pixel_offset: 0.0,
                    has_position_override: if override_buf.is_some() { 1 } else { 0 },
                    position_override_base: override_slice.map_or(0, |s| s.base_element),
                    position_override_len: override_slice.map_or(u32::MAX, |s| s.element_count),
                    deform_flags: resources.deform.flag_bits(item.mesh_id),
                    _deform_pad: [0; 3],
                };
                let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("outline_mask_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("outline_mask_object_bg"),
                    layout: &resources.outline.bind_group_layout,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: override_buf
                                .unwrap_or(&resources.content.fallback_position_override_buf)
                                .as_entire_binding(),
                        },
                    ],
                });
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.mesh_id,
                    two_sided: item.material.is_two_sided(),
                    deform_instance: item.deform_instance,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
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
                    model: item.model,
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    has_position_override: 0,
                    position_override_base: 0,
                    position_override_len: u32::MAX,
                    deform_flags: 0,
                    _deform_pad: [0; 3],
                };
                let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("outline_mask_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("outline_mask_object_bg"),
                    layout: &resources.outline.bind_group_layout,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: resources
                                .content
                                .fallback_position_override_buf
                                .as_entire_binding(),
                        },
                    ],
                });
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.boundary_mesh_id,
                    two_sided: false,
                    deform_instance: None,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                });
            }
        }

        // Screen image outlines: compute NDC bounds and create outline buffers.
        let mut screen_rect_outline_buffers: Vec<crate::resources::ScreenRectOutlineBuffers> =
            Vec::new();
        if frame.interaction.outline_selected
            && frame
                .scene
                .screen_images
                .iter()
                .any(|i| i.settings.selected)
        {
            self.resources
                .ensure_screen_rect_outline_mask_pipeline(device);
            let [vp_w, vp_h] = frame.camera.viewport_size;
            if let Some(bgl) = self.resources.screen_image.rect_outline_bgl.as_ref() {
                for item in &frame.scene.screen_images {
                    if item.settings.hidden
                        || !item.settings.selected
                        || item.width == 0
                        || item.height == 0
                    {
                        continue;
                    }
                    let [ndc_min_x, ndc_max_x, ndc_min_y, ndc_max_y] =
                        crate::renderer::types::viewport_anchored_ndc(
                            item.anchor_x,
                            item.anchor_y,
                            [
                                item.width as f32 * item.scale,
                                item.height as f32 * item.scale,
                            ],
                            [vp_w, vp_h],
                        );
                    #[repr(C)]
                    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
                    struct NdcRectUniform {
                        ndc_min: [f32; 2],
                        ndc_max: [f32; 2],
                    }
                    let uniform_data = NdcRectUniform {
                        ndc_min: [ndc_min_x, ndc_min_y],
                        ndc_max: [ndc_max_x, ndc_max_y],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("screen_rect_outline_uniform"),
                            contents: bytemuck::bytes_of(&uniform_data),
                            usage: crate::gpu::BufferUsages::UNIFORM,
                        });
                    let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("screen_rect_outline_bg"),
                        layout: bgl,
                        entries: &[crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });
                    screen_rect_outline_buffers.push(crate::resources::ScreenRectOutlineBuffers {
                        _uniform_buf: uniform_buf,
                        bind_group: bg,
                    });
                }
            }
        }

        // X-ray buffers for selected objects.
        let mut xray_object_buffers: Vec<(
            crate::resources::mesh::mesh_store::MeshId,
            crate::gpu::Buffer,
            crate::gpu::BindGroup,
        )> = Vec::new();
        if frame.interaction.xray_selected {
            let resources = &self.resources;
            for item in scene_items {
                if item.settings.hidden || !item.settings.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    colour: frame.interaction.xray_colour.to_linear_rgba(),
                    pixel_offset: 0.0,
                    has_position_override: 0,
                    position_override_base: 0,
                    position_override_len: u32::MAX,
                    deform_flags: 0,
                    _deform_pad: [0; 3],
                };
                let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("xray_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("xray_object_bg"),
                    layout: &resources.outline.bind_group_layout,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: resources
                                .content
                                .fallback_position_override_buf
                                .as_entire_binding(),
                        },
                    ],
                });
                xray_object_buffers.push((item.mesh_id, buf, bg));
            }
        }

        // Constraint guide lines.
        let mut constraint_line_buffers = Vec::new();
        for overlay in &frame.interaction.constraint_overlays {
            constraint_line_buffers.push(self.resources.create_constraint_overlay(device, overlay));
        }

        // Clip-object visuals (outlines and the plane fill) are no longer drawn by
        // the renderer. The host builds them from `clip_plane::visual` and submits
        // them as ordinary scene primitives (polyline outlines and a translucent
        // fill mesh), tagged `ItemSettings::ignore_clip` so they stay visible where
        // the scene is clipped. The renderer only performs the clip operation and
        // the section cap fill below.

        // Cap geometry for section-view cross-section fill.
        let mut cap_buffers = Vec::new();
        if viewport_fx.clip.cap_fill_enabled {
            for obj in viewport_fx.clip.objects.iter().filter(|o| o.enabled) {
                if let ClipShape::Plane {
                    normal,
                    distance,
                    cap_colour,
                    ..
                } = obj.shape
                {
                    let plane_n = glam::Vec3::from(normal);
                    for item in scene_items.iter().filter(|i| !i.settings.hidden) {
                        let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                            continue;
                        };
                        let model = glam::Mat4::from_cols_array_2d(&item.model);
                        let world_aabb = mesh.aabb.transformed(&model);
                        if !world_aabb.intersects_plane(plane_n, distance) {
                            continue;
                        }
                        let (Some(pos), Some(idx)) = (&mesh.cpu_positions, &mesh.cpu_indices)
                        else {
                            continue;
                        };
                        if let Some(cap) = crate::geometry::cap_geometry::generate_cap_mesh(
                            pos, idx, &model, plane_n, distance,
                        ) {
                            let bc = item.material.base_colour.to_linear_rgb();
                            let colour = cap_colour.unwrap_or([bc[0], bc[1], bc[2], 1.0]);
                            let buf = self.resources.upload_cap_geometry(device, &cap, colour);
                            cap_buffers.push(buf);
                        }
                    }
                }
            }
        }

        // The transform gizmo and the axes indicator are generated as 2D overlay
        // primitives during the overlay prepare (see `gizmo_overlay_items` and
        // `axes_overlay_items`); no per-viewport mesh upload happens here anymore.

        // ------------------------------------------------------------------
        // Assign all interaction state to the per-viewport slot.
        // ------------------------------------------------------------------
        {
            let slot = &mut self.viewport_slots[vp_idx];
            slot.selection_outlines.outline_object_buffers = outline_object_buffers;
            slot.selection_outlines.screen_rect_outline_buffers = screen_rect_outline_buffers;
            slot.xray_object_buffers = xray_object_buffers;
            slot.constraint_line_buffers = constraint_line_buffers;
            slot.cap_buffers = cap_buffers;
        }
    }

    pub(super) fn prepare_outline_pass(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        let vp_idx = frame.camera.viewport_index;

        // Item-type plugins draw their own selection coverage into the mask (they
        // are not tracked in the per-kind outline buffers). Record whether any
        // plugin has a selection this frame, so both the mask/edge pass below and
        // the composite (see `emit_outline_composite!`) run for plugin outlines.
        let plugin_outline = self.any_plugin_item_selected(frame);
        self.viewport_slots[vp_idx]
            .selection_outlines
            .plugin_outline_present = plugin_outline;

        // ------------------------------------------------------------------
        // Outline offscreen pass : screen-space edge detection.
        //
        // 1. Render selected objects to an R8 mask texture (white on black).
        // 2. Run a fullscreen edge-detection pass reading the mask and writing
        //    an anti-aliased outline ring to the outline colour texture.
        //
        // The outline colour texture is later composited onto the main target
        // by the composite pass in paint()/render().
        // ------------------------------------------------------------------
        if frame.interaction.outline_selected
            && (!self.viewport_slots[vp_idx]
                .selection_outlines
                .outline_object_buffers
                .is_empty()
                || !self.viewport_slots[vp_idx]
                    .selection_outlines
                    .screen_rect_outline_buffers
                    .is_empty()
                || plugin_outline)
        {
            let ppp = frame.camera.pixels_per_point;
            let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
            let h = (frame.camera.viewport_size[1] * ppp).round() as u32;

            // Ensure per-viewport HDR state exists (provides outline textures).
            self.ensure_viewport_hdr(
                device,
                queue,
                vp_idx,
                w.max(1),
                h.max(1),
                frame.effects.post_process.ssaa_factor.max(1),
                self.current_render_scale,
            );

            // Write edge-detection uniform (colour, radius, viewport size).
            {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let [scene_w, scene_h] = slot_hdr.scene_size;
                let edge_uniform = OutlineEdgeUniform {
                    colour: frame.interaction.outline_colour.to_linear_rgba(),
                    radius: frame.interaction.outline_width_px,
                    viewport_w: scene_w as f32,
                    viewport_h: scene_h as f32,
                    _pad: 0.0,
                };
                queue.write_buffer(
                    &slot_hdr.outline_edge_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[edge_uniform]),
                );
            }

            // Extract raw pointers for slot fields needed inside the render
            // passes alongside &self.resources borrows.
            let slot_ref = &self.viewport_slots[vp_idx];
            let outlines_ptr = &slot_ref.selection_outlines.outline_object_buffers
                as *const Vec<OutlineObjectBuffers>;
            let screen_rect_outlines_ptr = &slot_ref.selection_outlines.screen_rect_outline_buffers
                as *const Vec<crate::resources::ScreenRectOutlineBuffers>;
            let camera_bg_ptr = &slot_ref.camera_bind_group as *const crate::gpu::BindGroup;
            let slot_hdr = slot_ref.hdr.as_ref().unwrap();
            let mask_view_ptr = &slot_hdr.outline_mask_view as *const crate::gpu::TextureView;
            let colour_view_ptr = &slot_hdr.outline_colour_view as *const crate::gpu::TextureView;
            let depth_view_ptr = &slot_hdr.outline_depth_view as *const crate::gpu::TextureView;
            let edge_bg_ptr = &slot_hdr.outline_edge_bind_group as *const crate::gpu::BindGroup;
            // SAFETY: slot fields remain valid for the duration of this function;
            // no other code modifies these fields here.
            let (
                outlines,
                screen_rect_outlines,
                camera_bg,
                mask_view,
                colour_view,
                depth_view,
                edge_bg,
            ) = unsafe {
                (
                    &*outlines_ptr,
                    &*screen_rect_outlines_ptr,
                    &*camera_bg_ptr,
                    &*mask_view_ptr,
                    &*colour_view_ptr,
                    &*depth_view_ptr,
                    &*edge_bg_ptr,
                )
            };

            let mut encoder =
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("outline_offscreen_encoder"),
                });

            // Pass 1: render selected objects to R8 mask texture.
            {
                let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(any(wgpu29, wgpu30))]
                    multiview_mask: None,
                    label: Some("outline_mask_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: mask_view,
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
                            load: crate::gpu::LoadOp::Clear(1.0),
                            store: crate::gpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_bind_group(0, camera_bg, &[]);
                // Bind group 2 is required by outline_mask_pipeline and
                // outline_mask_two_sided_pipeline. Set the dummy here so it is
                // always valid; the mesh outline loop below overrides it per item.
                bind_deform_group!(
                    pass,
                    self.resources,
                    &self.resources.deform.dummy_bind_group
                );
                for outlined in outlines {
                    let Some(mesh) = self.resources.mesh_store.get(outlined.mesh_id) else {
                        continue;
                    };
                    let pipeline: &crate::gpu::RenderPipeline = if outlined.two_sided {
                        &self.resources.outline.mask_two_sided_pipeline
                    } else {
                        &self.resources.outline.mask_pipeline
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(1, &outlined.mask_bind_group, &[]);
                    bind_deform_group!(
                        pass,
                        self.resources,
                        self.resources
                            .deform
                            .instance_bind_group_for(outlined.mesh_id, outlined.deform_instance,)
                    );
                    pass.set_vertex_buffer(
                        0,
                        self.resources.geometry.vertex_slice(mesh.vertex_span),
                    );
                    // Use the compacted index buffer when a compute filter clipped this
                    // mesh, so the outline follows the filtered geometry (matching the
                    // scene pass) rather than the full mesh.
                    let filter = self
                        .compute_filter_results
                        .iter()
                        .find(|r| r.mesh_id == outlined.mesh_id);
                    let (index_slice, index_count) = match filter {
                        Some(f) => (f.index_buffer.slice(..), f.index_count),
                        None => (
                            self.resources.geometry.index_slice(mesh.index_span),
                            mesh.index_count,
                        ),
                    };
                    pass.set_index_buffer(index_slice, crate::gpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..index_count, 0, 0..1);
                }

                // Draw screen-space rect outlines for screen images.
                if !screen_rect_outlines.is_empty() {
                    if let Some(pipeline) = self
                        .resources
                        .screen_image
                        .rect_outline_mask_pipeline
                        .as_ref()
                    {
                        pass.set_pipeline(pipeline);
                        for sr in screen_rect_outlines {
                            pass.set_bind_group(0, &sr.bind_group, &[]);
                            pass.draw(0..6, 0..1);
                        }
                    }
                }

                // Draw screen-space rect outlines for screen images.
                if !screen_rect_outlines.is_empty() {
                    if let Some(pipeline) = self
                        .resources
                        .screen_image
                        .rect_outline_mask_pipeline
                        .as_ref()
                    {
                        pass.set_pipeline(pipeline);
                        for sr in screen_rect_outlines {
                            pass.set_bind_group(0, &sr.bind_group, &[]);
                            pass.draw(0..6, 0..1);
                        }
                    }
                }

                // Item-type plugin outline mask: each registered plugin
                // draws its selected items into the R8 mask.
                self.dispatch_plugin_outline_mask(&mut pass, frame);
            }

            // Pass 2: fullscreen edge detection (reads mask, writes colour).
            {
                let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                    #[cfg(any(wgpu29, wgpu30))]
                    multiview_mask: None,
                    label: Some("outline_edge_pass"),
                    color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                        view: colour_view,
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
                pass.set_pipeline(&self.resources.outline.edge_pipeline);
                pass.set_bind_group(0, edge_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            sink.push(encoder.finish());
        }
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
                    sel_version_changed || slot.sub_highlight.is_none()
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
                }
            } else {
                let slot = &mut self.viewport_slots[vp_idx];
                slot.sub_highlight = None;
                slot.sub_highlight_generation = u64::MAX;
            }
        }
    }
}
