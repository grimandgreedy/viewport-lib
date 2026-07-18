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
                    .clip_objects
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
                    &resources.clip_planes_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[clip_uniform]),
                );
                queue.write_buffer(
                    &resources.clip_volume_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[clip_vols_uniform]),
                );
            }

            // Upload camera uniform to per-viewport slot buffer.
            let camera_uniform = frame.camera.render_camera.camera_uniform();
            // Write to shared buffer for legacy single-viewport callers.
            queue.write_buffer(
                &resources.camera_uniform_buf,
                0,
                bytemuck::cast_slice(&[camera_uniform]),
            );
            // Write to the per-viewport slot buffer.
            if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                queue.write_buffer(&slot.camera_buf, 0, bytemuck::cast_slice(&[camera_uniform]));
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
                            let [r, g, b] =
                                frame.viewport.grid_colour.unwrap_or([0.55, 0.55, 0.55]);
                            [r, g, b, 0.4 * minor_fade]
                        },
                        colour_major: {
                            let [r, g, b] =
                                frame.viewport.grid_colour.unwrap_or([0.60, 0.60, 0.60]);
                            [r, g, b, 0.4 + 0.2 * minor_fade]
                        },
                    };
                    // Write to per-viewport slot buffer.
                    if let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) {
                        queue.write_buffer(&slot.grid_buf, 0, bytemuck::cast_slice(&[uniform]));
                    }
                    // Also write to shared buffer for legacy callers.
                    queue.write_buffer(
                        &resources.grid_uniform_buf,
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
                    colour: gp.colour,
                    shadow_colour: gp.shadow_colour,
                    light_vp: gp_cascade0_mat,
                    tan_half_fov,
                    aspect,
                    tile_size: gp.tile_size,
                    shadow_bias: 0.002,
                    mode: mode_u32,
                    shadow_opacity: gp.shadow_opacity,
                    _pad: [0.0; 2],
                    colour2: gp.tile_colour2,
                };
                queue.write_buffer(
                    &resources.ground_plane_uniform_buf,
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
                let uniform = OutlineUniform {
                    model: item.model,
                    colour: [0.0; 4], // unused by mask shader
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
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
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.mesh_id,
                    two_sided: item.material.is_two_sided(),
                    deform_instance: item.deform_instance,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                });
            }
            // Selected volume meshes rendered through projected-tet: use the
            // boundary surface for the outline mask. Opaque-mode items already
            // emit their outline via the standard surface submission, so skip
            // them here to avoid double draws.
            for item in &frame.scene.volume_meshes {
                if item.settings.hidden || !item.settings.selected || item.transparency.is_none() {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
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
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.boundary_mesh_id,
                    two_sided: false,
                    deform_instance: None,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                });
            }
            // Selected volume surface slices: use their mesh directly.
            for item in &frame.scene.volume_surface_slices {
                if item.settings.hidden || !item.settings.selected {
                    continue;
                }
                let uniform = OutlineUniform {
                    model: item.model,
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
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
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                outline_object_buffers.push(OutlineObjectBuffers {
                    mesh_id: item.mesh_id,
                    two_sided: true,
                    deform_instance: None,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                });
            }
        }

        // Splat outline buffers: point sprite discs for selected Gaussian splat sets.
        let mut splat_outline_buffers: Vec<crate::resources::SplatOutlineBuffers> = Vec::new();
        // Curve mesh outline items: streamtubes, tubes, ribbons rendered via outline_mask_pipeline.
        let mut streamtube_outline_items: Vec<CurveMeshOutlineItem> = Vec::new();
        let mut tube_outline_items: Vec<CurveMeshOutlineItem> = Vec::new();
        let mut ribbon_outline_items: Vec<CurveMeshOutlineItem> = Vec::new();
        // Polyline outline indices: indices into polyline_gpu_data for selected polylines.
        let mut polyline_outline_indices: Vec<usize> = Vec::new();
        // Each entry is (gpu_data_index, instance_ranges).
        // None = draw all instances (object-level selection).
        // Some(vec) = draw only these specific instance indices (sub-object Instance selection).
        let mut glyph_outline_indices: Vec<(usize, Option<Vec<u32>>)> = Vec::new();
        let mut tensor_glyph_outline_indices: Vec<(usize, Option<Vec<u32>>)> = Vec::new();
        let mut sprite_outline_indices: Vec<(usize, Option<Vec<u32>>)> = Vec::new();
        if frame.interaction.outline_selected {
            let resources = &self.resources;
            let view_proj = frame.camera.render_camera.view_proj();
            let [vp_w, vp_h] = frame.camera.viewport_size;
            for item in &frame.scene.gaussian_splats {
                let Some(gpu_set) = resources.content.gaussian_splat_store.get(item.source) else {
                    continue;
                };
                if item.settings.selected && !gpu_set.cpu_positions.is_empty() {
                    // Object-level: outline all splats.
                    // World-space radius covering the visible Gaussian tail (~3 sigma).
                    let mean_max_scale: f32 = if gpu_set.cpu_scales.is_empty() {
                        0.05
                    } else {
                        gpu_set
                            .cpu_scales
                            .iter()
                            .map(|s| s[0].max(s[1]).max(s[2]))
                            .sum::<f32>()
                            / gpu_set.cpu_scales.len() as f32
                    };
                    let world_radius = mean_max_scale * 3.0;

                    // Project the world radius to a pixel half-size at the cloud center.
                    // Use the camera right vector so the offset is always perpendicular
                    // to the view direction, avoiding the collapse when looking along X.
                    let model = glam::Mat4::from_cols_array_2d(&item.model);
                    let center_w = model.transform_point3(glam::Vec3::ZERO);
                    let cam_right = frame
                        .camera
                        .render_camera
                        .view
                        .row(0)
                        .truncate()
                        .normalize();
                    let p0_clip =
                        view_proj * glam::Vec4::new(center_w.x, center_w.y, center_w.z, 1.0);
                    let p1_world = center_w + cam_right * world_radius;
                    let p1_clip =
                        view_proj * glam::Vec4::new(p1_world.x, p1_world.y, p1_world.z, 1.0);
                    let pixel_radius = if p0_clip.w.abs() > 1e-6 && p1_clip.w.abs() > 1e-6 {
                        let p0_ndc = glam::Vec2::new(p0_clip.x, p0_clip.y) / p0_clip.w;
                        let p1_ndc = glam::Vec2::new(p1_clip.x, p1_clip.y) / p1_clip.w;
                        (p1_ndc - p0_ndc).length() * 0.5 * vp_w.max(vp_h)
                    } else {
                        world_radius * 100.0
                    };
                    let pixel_radius = pixel_radius.max(1.0);

                    let position_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("splat_outline_pos_buf"),
                            contents: bytemuck::cast_slice(gpu_set.cpu_positions.as_slice()),
                            usage: crate::gpu::BufferUsages::VERTEX,
                        });

                    let uniform = SplatOutlineMaskUniform {
                        model: item.model,
                        viewport_w: vp_w,
                        viewport_h: vp_h,
                        pixel_radius,
                        _pad: [0.0; 9],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("splat_outline_uniform_buf"),
                            contents: bytemuck::cast_slice(&[uniform]),
                            usage: crate::gpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("splat_outline_bg"),
                        layout: &resources.outline.bind_group_layout,
                        entries: &[crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });

                    let n = gpu_set.cpu_positions.len();
                    let size_data: Vec<f32> = vec![pixel_radius; n];
                    let size_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("splat_outline_size_buf"),
                            contents: bytemuck::cast_slice(&size_data),
                            usage: crate::gpu::BufferUsages::VERTEX,
                        });

                    splat_outline_buffers.push(crate::resources::SplatOutlineBuffers {
                        position_buf,
                        size_buf,
                        instance_count: n as u32,
                        _uniform_buf: uniform_buf,
                        bind_group,
                    });
                } else if !item.settings.selected && item.settings.pick_id != PickId::NONE {
                    // Per-splat sub-selection: outline only the selected splats.
                    let sub_sel = frame.interaction.sub_selection.as_ref();
                    let selected_indices: Vec<u32> = sub_sel
                        .iter()
                        .flat_map(|s| s.items.iter())
                        .filter_map(|(node_id, sub)| {
                            if *node_id == item.settings.pick_id.0 {
                                if let crate::renderer::SubObjectRef::Splat(i) = sub {
                                    return Some(*i);
                                }
                            }
                            None
                        })
                        .collect();
                    if selected_indices.is_empty() {
                        continue;
                    }

                    let model = glam::Mat4::from_cols_array_2d(&item.model);
                    let cam_right = frame
                        .camera
                        .render_camera
                        .view
                        .row(0)
                        .truncate()
                        .normalize();

                    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(selected_indices.len());
                    let mut sizes: Vec<f32> = Vec::with_capacity(selected_indices.len());
                    for &idx in &selected_indices {
                        let i = idx as usize;
                        if let Some(&pos) = gpu_set.cpu_positions.get(i) {
                            positions.push(pos);
                            let world_radius = if let Some(s) = gpu_set.cpu_scales.get(i) {
                                s[0].max(s[1]).max(s[2]) * 3.0
                            } else {
                                0.15
                            };
                            let center_w = model.transform_point3(glam::Vec3::from(pos));
                            let p0_clip = view_proj
                                * glam::Vec4::new(center_w.x, center_w.y, center_w.z, 1.0);
                            let p1_world = center_w + cam_right * world_radius;
                            let p1_clip = view_proj
                                * glam::Vec4::new(p1_world.x, p1_world.y, p1_world.z, 1.0);
                            let px = if p0_clip.w.abs() > 1e-6 && p1_clip.w.abs() > 1e-6 {
                                let p0_ndc = glam::Vec2::new(p0_clip.x, p0_clip.y) / p0_clip.w;
                                let p1_ndc = glam::Vec2::new(p1_clip.x, p1_clip.y) / p1_clip.w;
                                ((p1_ndc - p0_ndc).length() * 0.5 * vp_w.max(vp_h)).max(1.0)
                            } else {
                                world_radius * 100.0
                            };
                            sizes.push(px);
                        }
                    }
                    if positions.is_empty() {
                        continue;
                    }

                    let pixel_radius = sizes
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max)
                        .max(1.0);
                    let uniform = SplatOutlineMaskUniform {
                        model: item.model,
                        viewport_w: vp_w,
                        viewport_h: vp_h,
                        pixel_radius,
                        _pad: [0.0; 9],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("splat_sel_outline_uniform_buf"),
                            contents: bytemuck::cast_slice(&[uniform]),
                            usage: crate::gpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("splat_sel_outline_bg"),
                        layout: &resources.outline.bind_group_layout,
                        entries: &[crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });
                    let position_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("splat_sel_outline_pos_buf"),
                            contents: bytemuck::cast_slice(&positions),
                            usage: crate::gpu::BufferUsages::VERTEX,
                        });
                    let size_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("splat_sel_outline_size_buf"),
                            contents: bytemuck::cast_slice(&sizes),
                            usage: crate::gpu::BufferUsages::VERTEX,
                        });
                    splat_outline_buffers.push(crate::resources::SplatOutlineBuffers {
                        position_buf,
                        size_buf,
                        instance_count: positions.len() as u32,
                        _uniform_buf: uniform_buf,
                        bind_group,
                    });
                }
            }

            // Point cloud outline buffers: reuse the same point sprite mask pipeline.
            for item in &frame.scene.point_clouds {
                if item.settings.hidden || item.positions.is_empty() {
                    continue;
                }
                let pixel_radius = (item.point_size * 0.5).max(1.0);
                if item.settings.selected {
                    // Object-level: outline all points.
                    let position_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("pc_outline_pos_buf"),
                            contents: bytemuck::cast_slice(item.positions.as_slice()),
                            usage: crate::gpu::BufferUsages::VERTEX,
                        });
                    let uniform = SplatOutlineMaskUniform {
                        model: item.model,
                        viewport_w: vp_w,
                        viewport_h: vp_h,
                        pixel_radius,
                        _pad: [0.0; 9],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("pc_outline_uniform_buf"),
                            contents: bytemuck::cast_slice(&[uniform]),
                            usage: crate::gpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("pc_outline_bg"),
                        layout: &self.resources.outline.bind_group_layout,
                        entries: &[crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });
                    let n = item.positions.len();
                    let size_data: Vec<f32> = vec![pixel_radius; n];
                    let size_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("pc_outline_size_buf"),
                            contents: bytemuck::cast_slice(&size_data),
                            usage: crate::gpu::BufferUsages::VERTEX,
                        });
                    splat_outline_buffers.push(crate::resources::SplatOutlineBuffers {
                        position_buf,
                        size_buf,
                        instance_count: n as u32,
                        _uniform_buf: uniform_buf,
                        bind_group,
                    });
                } else if item.settings.pick_id != PickId::NONE {
                    // Per-point sub-selection: outline only the selected points.
                    let sub_sel = frame.interaction.sub_selection.as_ref();
                    let selected_positions: Vec<[f32; 3]> = sub_sel
                        .iter()
                        .flat_map(|s| s.items.iter())
                        .filter_map(|(node_id, sub)| {
                            if *node_id == item.settings.pick_id.0 {
                                if let crate::renderer::SubObjectRef::Point(i) = sub {
                                    return item.positions.get(*i as usize).copied();
                                }
                            }
                            None
                        })
                        .collect();
                    if selected_positions.is_empty() {
                        continue;
                    }
                    let n = selected_positions.len();
                    let uniform = SplatOutlineMaskUniform {
                        model: item.model,
                        viewport_w: vp_w,
                        viewport_h: vp_h,
                        pixel_radius,
                        _pad: [0.0; 9],
                    };
                    let uniform_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("pc_sel_outline_uniform_buf"),
                            contents: bytemuck::cast_slice(&[uniform]),
                            usage: crate::gpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("pc_sel_outline_bg"),
                        layout: &self.resources.outline.bind_group_layout,
                        entries: &[crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        }],
                    });
                    let position_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("pc_sel_outline_pos_buf"),
                            contents: bytemuck::cast_slice(&selected_positions),
                            usage: crate::gpu::BufferUsages::VERTEX,
                        });
                    let size_data = vec![pixel_radius; n];
                    let size_buf =
                        device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                            label: Some("pc_sel_outline_size_buf"),
                            contents: bytemuck::cast_slice(&size_data),
                            usage: crate::gpu::BufferUsages::VERTEX,
                        });
                    splat_outline_buffers.push(crate::resources::SplatOutlineBuffers {
                        position_buf,
                        size_buf,
                        instance_count: n as u32,
                        _uniform_buf: uniform_buf,
                        bind_group,
                    });
                }
            }

            // Glyph outline indices: record which glyph GPU data entries are selected
            // so the mask pass can render the actual instanced mesh.
            {
                let sub_sel = frame.interaction.sub_selection.as_ref();
                let mut gpu_idx = 0usize;
                for item in &frame.scene.glyphs {
                    if item.settings.hidden || item.positions.is_empty() || item.vectors.is_empty()
                    {
                        continue;
                    }
                    if item.settings.selected {
                        self.resources.ensure_glyph_outline_mask_pipeline(device);
                        glyph_outline_indices.push((gpu_idx, None));
                    } else if item.settings.pick_id != PickId::NONE {
                        // Check for per-instance sub-selection.
                        let instances: Vec<u32> = sub_sel
                            .iter()
                            .flat_map(|s| s.items.iter())
                            .filter_map(|(node_id, sub)| {
                                if *node_id == item.settings.pick_id.0 {
                                    if let crate::renderer::SubObjectRef::Instance(i) = sub {
                                        return Some(*i);
                                    }
                                }
                                None
                            })
                            .collect();
                        if !instances.is_empty() {
                            self.resources.ensure_glyph_outline_mask_pipeline(device);
                            glyph_outline_indices.push((gpu_idx, Some(instances)));
                        }
                    }
                    gpu_idx += 1;
                }
            }

            // Polyline outlines: collect indices of selected polylines so the mask
            // pass can draw their segment quads via the polyline_outline_mask_pipeline.
            if !self.polyline_selected_gpu_indices.is_empty() {
                self.resources.ensure_polyline_outline_mask_pipeline(device);
                polyline_outline_indices = self.polyline_selected_gpu_indices.clone();
            }

            // Sprite outline indices: record which sprite GPU data entries are selected
            // so the mask pass can render the actual billboard quads.
            {
                let sub_sel = frame.interaction.sub_selection.as_ref();
                for (i, item) in frame.scene.sprite_items.iter().enumerate() {
                    if item.settings.hidden || item.positions.is_empty() {
                        continue;
                    }
                    if item.settings.selected {
                        self.resources.ensure_sprite_outline_mask_pipeline(device);
                        sprite_outline_indices.push((i, None));
                    } else if item.settings.pick_id != PickId::NONE {
                        let instances: Vec<u32> = sub_sel
                            .iter()
                            .flat_map(|s| s.items.iter())
                            .filter_map(|(node_id, sub)| {
                                if *node_id == item.settings.pick_id.0 {
                                    if let crate::renderer::SubObjectRef::Instance(idx) = sub {
                                        return Some(*idx);
                                    }
                                }
                                None
                            })
                            .collect();
                        if !instances.is_empty() {
                            self.resources.ensure_sprite_outline_mask_pipeline(device);
                            sprite_outline_indices.push((i, Some(instances)));
                        }
                    }
                }
            }

            // Streamtube / Tube / Ribbon outline items: use the actual triangle mesh
            // geometry so the depth-buffer edge detection follows the tube silhouette.
            let make_curve_item = |index: usize, two_sided: bool| -> CurveMeshOutlineItem {
                let uniform = crate::resources::OutlineUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    colour: [1.0, 1.0, 1.0, 1.0],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                    deform_flags: 0,
                    _deform_pad: [0; 3],
                };
                let buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                    label: Some("curve_outline_uniform_buf"),
                    contents: bytemuck::cast_slice(&[uniform]),
                    usage: crate::gpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("curve_outline_mask_bg"),
                    layout: &self.resources.outline.bind_group_layout,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                CurveMeshOutlineItem {
                    index,
                    two_sided,
                    _mask_uniform_buf: buf,
                    mask_bind_group: bg,
                }
            };

            for &idx in &self.streamtube_selected_gpu_indices {
                streamtube_outline_items.push(make_curve_item(idx, false));
            }
            for &idx in &self.tube_selected_gpu_indices {
                tube_outline_items.push(make_curve_item(idx, false));
            }
            for &idx in &self.ribbon_selected_gpu_indices {
                ribbon_outline_items.push(make_curve_item(idx, true));
            }

            // Tensor glyph outline indices: same approach as arrow glyphs.
            {
                let sub_sel = frame.interaction.sub_selection.as_ref();
                let mut gpu_idx = 0usize;
                for item in &frame.scene.tensor_glyphs {
                    if item.settings.hidden || item.positions.is_empty() {
                        continue;
                    }
                    if item.settings.selected {
                        self.resources
                            .ensure_tensor_glyph_outline_mask_pipeline(device);
                        tensor_glyph_outline_indices.push((gpu_idx, None));
                    } else if item.settings.pick_id != PickId::NONE {
                        let instances: Vec<u32> = sub_sel
                            .iter()
                            .flat_map(|s| s.items.iter())
                            .filter_map(|(node_id, sub)| {
                                if *node_id == item.settings.pick_id.0 {
                                    if let crate::renderer::SubObjectRef::Instance(i) = sub {
                                        return Some(*i);
                                    }
                                }
                                None
                            })
                            .collect();
                        if !instances.is_empty() {
                            self.resources
                                .ensure_tensor_glyph_outline_mask_pipeline(device);
                            tensor_glyph_outline_indices.push((gpu_idx, Some(instances)));
                        }
                    }
                    gpu_idx += 1;
                }
            }
        }

        // Volume outline: record indices of selected volumes so the mask pass can
        // reuse their VolumeGpuData bind groups (which already contain model, 3D
        // texture, samplers, and LUTs needed by the ray-march mask shader).
        let mut volume_outline_indices: Vec<usize> = Vec::new();
        if frame.interaction.outline_selected {
            self.resources.ensure_volume_cube(device);
            self.resources.ensure_volume_pipeline(device);
            self.resources.ensure_volume_outline_mask_pipeline(device);
            for (i, item) in frame.scene.volumes.iter().enumerate() {
                if !item.settings.hidden && item.settings.selected {
                    volume_outline_indices.push(i);
                }
            }
        }

        // Image slice outlines: compute world-space quad corners and create inline vertex/index buffers.
        let mut raw_geom_outline_buffers: Vec<crate::resources::RawGeomOutlineBuffers> = Vec::new();
        if frame.interaction.outline_selected {
            let resources = &self.resources;
            for item in &frame.scene.image_slices {
                if item.settings.hidden || !item.settings.selected {
                    continue;
                }
                use crate::SliceAxis;
                let [bmin, bmax] = [item.bbox_min, item.bbox_max];
                let t = item.offset;
                let (v0, v1, v2, v3) = match item.axis {
                    SliceAxis::X => {
                        let x = bmin[0] + t * (bmax[0] - bmin[0]);
                        (
                            [x, bmin[1], bmin[2]],
                            [x, bmax[1], bmin[2]],
                            [x, bmax[1], bmax[2]],
                            [x, bmin[1], bmax[2]],
                        )
                    }
                    SliceAxis::Y => {
                        let y = bmin[1] + t * (bmax[1] - bmin[1]);
                        (
                            [bmin[0], y, bmin[2]],
                            [bmax[0], y, bmin[2]],
                            [bmax[0], y, bmax[2]],
                            [bmin[0], y, bmax[2]],
                        )
                    }
                    SliceAxis::Z => {
                        let z = bmin[2] + t * (bmax[2] - bmin[2]);
                        (
                            [bmin[0], bmin[1], z],
                            [bmax[0], bmin[1], z],
                            [bmax[0], bmax[1], z],
                            [bmin[0], bmax[1], z],
                        )
                    }
                };
                let verts: [[f32; 3]; 4] = [v0, v1, v2, v3];
                let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
                let vertex_buf =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("image_slice_outline_verts"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: crate::gpu::BufferUsages::VERTEX,
                    });
                let index_buf =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("image_slice_outline_indices"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: crate::gpu::BufferUsages::INDEX,
                    });
                let uniform = OutlineUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                    deform_flags: 0,
                    _deform_pad: [0; 3],
                };
                let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("outline_mask_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("outline_mask_object_bg"),
                    layout: &resources.outline.bind_group_layout,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    }],
                });
                raw_geom_outline_buffers.push(crate::resources::RawGeomOutlineBuffers {
                    vertex_buf,
                    index_buf,
                    index_count: 6,
                    two_sided: true,
                    _uniform_buf: uniform_buf,
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
                    use crate::ImageAnchor;
                    let img_w_ndc = 2.0 * item.width as f32 * item.scale / vp_w.max(1.0);
                    let img_h_ndc = 2.0 * item.height as f32 * item.scale / vp_h.max(1.0);
                    let (ndc_min_x, ndc_max_x, ndc_min_y, ndc_max_y) = match item.anchor {
                        ImageAnchor::TopLeft => (-1.0, -1.0 + img_w_ndc, 1.0 - img_h_ndc, 1.0),
                        ImageAnchor::TopRight => (1.0 - img_w_ndc, 1.0, 1.0 - img_h_ndc, 1.0),
                        ImageAnchor::BottomLeft => (-1.0, -1.0 + img_w_ndc, -1.0, -1.0 + img_h_ndc),
                        ImageAnchor::BottomRight => (1.0 - img_w_ndc, 1.0, -1.0, -1.0 + img_h_ndc),
                        _ => (
                            -img_w_ndc * 0.5,
                            img_w_ndc * 0.5,
                            -img_h_ndc * 0.5,
                            img_h_ndc * 0.5,
                        ),
                    };
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

        // Implicit surface outlines: record indices into implicit_gpu_data for selected items.
        let mut implicit_outline_indices: Vec<usize> = Vec::new();
        if frame.interaction.outline_selected {
            let mut gpu_idx = 0usize;
            for item in &frame.scene.gpu_implicit {
                if item.settings.hidden || item.primitives.is_empty() {
                    continue;
                }
                if item.settings.selected {
                    self.resources.ensure_implicit_pipeline(device);
                    self.resources.ensure_implicit_outline_mask_pipeline(device);
                    implicit_outline_indices.push(gpu_idx);
                }
                gpu_idx += 1;
            }
        }

        // MC surface outlines: build per-job outline uniform + bind group.
        let mut mc_outline_data: Vec<crate::resources::volume::gpu_marching_cubes::McOutlineItem> =
            Vec::new();
        if frame.interaction.outline_selected {
            for (i, job) in frame.scene.gpu_mc_jobs.iter().enumerate() {
                if job.settings.hidden || !job.settings.selected {
                    continue;
                }
                self.resources.ensure_mc_pipelines(device);
                self.resources.ensure_mc_outline_mask_pipeline(device);
                let uniform = OutlineUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
                    deform_flags: 0,
                    _deform_pad: [0; 3],
                };
                let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("mc_outline_uniform_buf"),
                    size: std::mem::size_of::<OutlineUniform>() as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[uniform]));
                let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("mc_outline_bg"),
                    layout: &self.resources.outline.bind_group_layout,
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                mc_outline_data.push(
                    crate::resources::volume::gpu_marching_cubes::McOutlineItem {
                        mc_gpu_idx: i,
                        _uniform_buf: buf,
                        mask_bind_group: bg,
                    },
                );
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
                    colour: frame.interaction.xray_colour,
                    pixel_offset: 0.0,
                    _pad: [0.0; 3],
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
                    entries: &[crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                xray_object_buffers.push((item.mesh_id, buf, bg));
            }
        }

        // Constraint guide lines.
        let mut constraint_line_buffers = Vec::new();
        for overlay in &frame.interaction.constraint_overlays {
            constraint_line_buffers.push(self.resources.create_constraint_overlay(device, overlay));
        }

        // Clip plane overlays : generated automatically from clip_objects with a colour set.
        let mut clip_plane_fill_buffers = Vec::new();
        let mut clip_plane_line_buffers = Vec::new();
        for obj in viewport_fx.clip_objects.iter().filter(|o| o.enabled) {
            // Skip if neither fill nor edge colour is set.
            if obj.colour.is_none() && obj.edge_colour.is_none() {
                continue;
            }
            if let ClipShape::Plane {
                normal,
                distance,
                display_center,
                ..
            } = obj.shape
            {
                let n = glam::Vec3::from(normal);
                // Use the caller-supplied display_center when available so that
                // lateral translations (tangent to the plane) are reflected in
                // the overlay quad position.  Fall back to the foot-of-normal
                // from the world origin when none is set.
                let center = display_center
                    .map(glam::Vec3::from)
                    .unwrap_or_else(|| n * (-distance));
                let active = obj.active;
                let hovered = obj.hovered || active;

                // Fill quad: derived from `colour`; transparent if not set.
                let fill_colour = if let Some(base_colour) = obj.colour {
                    if active {
                        [
                            base_colour[0] * 0.5,
                            base_colour[1] * 0.5,
                            base_colour[2] * 0.5,
                            base_colour[3] * 0.5,
                        ]
                    } else if hovered {
                        [
                            base_colour[0] * 0.8,
                            base_colour[1] * 0.8,
                            base_colour[2] * 0.8,
                            base_colour[3] * 0.6,
                        ]
                    } else {
                        [
                            base_colour[0] * 0.5,
                            base_colour[1] * 0.5,
                            base_colour[2] * 0.5,
                            base_colour[3] * 0.3,
                        ]
                    }
                } else {
                    [0.0, 0.0, 0.0, 0.0]
                };

                // Border edge: use `edge_colour` when set, otherwise derive from `colour`.
                let border_base = obj
                    .edge_colour
                    .or(obj.colour)
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                let border_colour = if active {
                    [border_base[0], border_base[1], border_base[2], 0.9]
                } else if hovered {
                    [border_base[0], border_base[1], border_base[2], 0.8]
                } else {
                    [
                        border_base[0] * 0.9,
                        border_base[1] * 0.9,
                        border_base[2] * 0.9,
                        0.6,
                    ]
                };

                let overlay = crate::interaction::manipulation::clip_plane::ClipPlaneOverlay {
                    center,
                    normal: n,
                    extent: obj.extent,
                    fill_colour,
                    border_colour,
                    _hovered: hovered,
                    _active: active,
                };
                if obj.colour.is_some() {
                    clip_plane_fill_buffers.push(
                        self.resources
                            .create_clip_plane_fill_overlay(device, &overlay),
                    );
                }
                clip_plane_line_buffers.push(
                    self.resources
                        .create_clip_plane_line_overlay(device, &overlay),
                );
            } else {
                // Box/Sphere/Cylinder: generate wireframe polyline overlay.
                // These use the clip-exempt pipeline so the outline is always fully visible,
                // even when multiple clip volumes are active (the user needs to see where each
                // clip is positioned to understand the combined result).
                let base_colour = obj.colour.unwrap_or([1.0, 1.0, 1.0, 1.0]);
                self.resources.ensure_polyline_no_clip_pipeline(device);
                match obj.shape {
                    ClipShape::Box {
                        center,
                        half_extents,
                        orientation,
                    } => {
                        let polyline =
                            clip_box_outline(center, half_extents, orientation, base_colour);
                        let vp_size = frame.camera.viewport_size;
                        let mut gpu = self
                            .resources
                            .upload_polyline_per_frame(device, queue, &polyline, vp_size);
                        gpu.skip_clip = true;
                        self.polyline_gpu_data.push(gpu);
                    }
                    ClipShape::Sphere { center, radius } => {
                        let polyline = clip_sphere_outline(center, radius, base_colour);
                        let vp_size = frame.camera.viewport_size;
                        let mut gpu = self
                            .resources
                            .upload_polyline_per_frame(device, queue, &polyline, vp_size);
                        gpu.skip_clip = true;
                        self.polyline_gpu_data.push(gpu);
                    }
                    ClipShape::Cylinder {
                        center,
                        axis,
                        radius,
                        half_length,
                    } => {
                        let polyline =
                            clip_cylinder_outline(center, axis, radius, half_length, base_colour);
                        let vp_size = frame.camera.viewport_size;
                        let mut gpu = self
                            .resources
                            .upload_polyline_per_frame(device, queue, &polyline, vp_size);
                        gpu.skip_clip = true;
                        self.polyline_gpu_data.push(gpu);
                    }
                    _ => {}
                }
            }
        }

        // Cap geometry for section-view cross-section fill.
        let mut cap_buffers = Vec::new();
        if viewport_fx.cap_fill_enabled {
            for obj in viewport_fx.clip_objects.iter().filter(|o| o.enabled) {
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
                            let bc = item.material.base_colour;
                            let colour = cap_colour.unwrap_or([bc[0], bc[1], bc[2], 1.0]);
                            let buf = self.resources.upload_cap_geometry(device, &cap, colour);
                            cap_buffers.push(buf);
                        }
                    }
                }
            }
        }

        // Axes indicator geometry (built here, written to slot buffer below).
        let axes_verts = if frame.viewport.show_axes_indicator
            && frame.camera.viewport_size[0] > 0.0
            && frame.camera.viewport_size[1] > 0.0
        {
            let verts = crate::interaction::widgets::axes_indicator::build_axes_geometry(
                frame.camera.viewport_size[0],
                frame.camera.viewport_size[1],
                frame.camera.render_camera.orientation,
            );
            if verts.is_empty() { None } else { Some(verts) }
        } else {
            None
        };

        // Gizmo mesh + uniform (built here, written to slot buffers below).
        let gizmo_update = frame.interaction.gizmo_model.map(|model| {
            let (verts, indices) = crate::interaction::manipulation::gizmo::build_gizmo_mesh(
                frame.interaction.gizmo_mode,
                frame.interaction.gizmo_hovered,
                frame.interaction.gizmo_space_orientation,
            );
            (verts, indices, model)
        });

        // ------------------------------------------------------------------
        // Assign all interaction state to the per-viewport slot.
        // ------------------------------------------------------------------
        {
            let slot = &mut self.viewport_slots[vp_idx];
            slot.outline_object_buffers = outline_object_buffers;
            slot.splat_outline_buffers = splat_outline_buffers;
            slot.streamtube_outline_items = streamtube_outline_items;
            slot.tube_outline_items = tube_outline_items;
            slot.ribbon_outline_items = ribbon_outline_items;
            slot.polyline_outline_indices = polyline_outline_indices;
            slot.volume_outline_indices = volume_outline_indices;
            slot.glyph_outline_indices = glyph_outline_indices;
            slot.tensor_glyph_outline_indices = tensor_glyph_outline_indices;
            slot.sprite_outline_indices = sprite_outline_indices;
            slot.raw_geom_outline_buffers = raw_geom_outline_buffers;
            slot.screen_rect_outline_buffers = screen_rect_outline_buffers;
            slot.implicit_outline_indices = implicit_outline_indices;
            slot.mc_outline_data = mc_outline_data;
            slot.xray_object_buffers = xray_object_buffers;
            slot.constraint_line_buffers = constraint_line_buffers;
            slot.clip_plane_fill_buffers = clip_plane_fill_buffers;
            slot.clip_plane_line_buffers = clip_plane_line_buffers;
            slot.cap_buffers = cap_buffers;

            // Axes: resize buffer if needed, then upload.
            if let Some(verts) = axes_verts {
                let byte_size = std::mem::size_of_val(verts.as_slice()) as u64;
                if byte_size > slot.axes_vertex_buffer.size() {
                    slot.axes_vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some("vp_axes_vertex_buf"),
                        size: byte_size,
                        usage: crate::gpu::BufferUsages::VERTEX
                            | crate::gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                queue.write_buffer(&slot.axes_vertex_buffer, 0, bytemuck::cast_slice(&verts));
                slot.axes_vertex_count = verts.len() as u32;
            } else {
                slot.axes_vertex_count = 0;
            }

            // Gizmo: resize buffers if needed, then upload mesh + uniform.
            if let Some((verts, indices, model)) = gizmo_update {
                let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
                let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);
                if vert_bytes.len() as u64 > slot.gizmo_vertex_buffer.size() {
                    slot.gizmo_vertex_buffer =
                        device.create_buffer(&crate::gpu::BufferDescriptor {
                            label: Some("vp_gizmo_vertex_buf"),
                            size: vert_bytes.len() as u64,
                            usage: crate::gpu::BufferUsages::VERTEX
                                | crate::gpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                }
                if idx_bytes.len() as u64 > slot.gizmo_index_buffer.size() {
                    slot.gizmo_index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some("vp_gizmo_index_buf"),
                        size: idx_bytes.len() as u64,
                        usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                queue.write_buffer(&slot.gizmo_vertex_buffer, 0, vert_bytes);
                queue.write_buffer(&slot.gizmo_index_buffer, 0, idx_bytes);
                slot.gizmo_index_count = indices.len() as u32;
                let uniform = crate::interaction::manipulation::gizmo::GizmoUniform {
                    model: model.to_cols_array_2d(),
                };
                queue.write_buffer(&slot.gizmo_uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
            }
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
                .outline_object_buffers
                .is_empty()
                || !self.viewport_slots[vp_idx].splat_outline_buffers.is_empty()
                || !self.viewport_slots[vp_idx]
                    .streamtube_outline_items
                    .is_empty()
                || !self.viewport_slots[vp_idx].tube_outline_items.is_empty()
                || !self.viewport_slots[vp_idx].ribbon_outline_items.is_empty()
                || !self.viewport_slots[vp_idx]
                    .polyline_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .volume_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx].glyph_outline_indices.is_empty()
                || !self.viewport_slots[vp_idx]
                    .tensor_glyph_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .sprite_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .raw_geom_outline_buffers
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .screen_rect_outline_buffers
                    .is_empty()
                || !self.viewport_slots[vp_idx]
                    .implicit_outline_indices
                    .is_empty()
                || !self.viewport_slots[vp_idx].mc_outline_data.is_empty())
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
                    colour: frame.interaction.outline_colour,
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
            let outlines_ptr = &slot_ref.outline_object_buffers as *const Vec<OutlineObjectBuffers>;
            let splat_outlines_ptr = &slot_ref.splat_outline_buffers
                as *const Vec<crate::resources::SplatOutlineBuffers>;
            let streamtube_outline_items_ptr =
                &slot_ref.streamtube_outline_items as *const Vec<CurveMeshOutlineItem>;
            let tube_outline_items_ptr =
                &slot_ref.tube_outline_items as *const Vec<CurveMeshOutlineItem>;
            let ribbon_outline_items_ptr =
                &slot_ref.ribbon_outline_items as *const Vec<CurveMeshOutlineItem>;
            let polyline_outline_idx_ptr = &slot_ref.polyline_outline_indices as *const Vec<usize>;
            let vol_outline_idx_ptr = &slot_ref.volume_outline_indices as *const Vec<usize>;
            let glyph_outline_idx_ptr =
                &slot_ref.glyph_outline_indices as *const Vec<(usize, Option<Vec<u32>>)>;
            let tensor_glyph_outline_idx_ptr =
                &slot_ref.tensor_glyph_outline_indices as *const Vec<(usize, Option<Vec<u32>>)>;
            let sprite_outline_idx_ptr =
                &slot_ref.sprite_outline_indices as *const Vec<(usize, Option<Vec<u32>>)>;
            let raw_geom_outlines_ptr = &slot_ref.raw_geom_outline_buffers
                as *const Vec<crate::resources::RawGeomOutlineBuffers>;
            let screen_rect_outlines_ptr = &slot_ref.screen_rect_outline_buffers
                as *const Vec<crate::resources::ScreenRectOutlineBuffers>;
            let implicit_outline_idx_ptr = &slot_ref.implicit_outline_indices as *const Vec<usize>;
            let mc_outlines_ptr = &slot_ref.mc_outline_data
                as *const Vec<crate::resources::volume::gpu_marching_cubes::McOutlineItem>;
            let glyph_gpu_ptr = &self.glyph_gpu_data as *const Vec<crate::resources::GlyphGpuData>;
            let tensor_glyph_gpu_ptr =
                &self.tensor_glyph_gpu_data as *const Vec<crate::resources::TensorGlyphGpuData>;
            let sprite_gpu_ptr =
                &self.sprite_gpu_data as *const Vec<crate::resources::SpriteGpuData>;
            let streamtube_gpu_ptr =
                &self.streamtube_gpu_data as *const Vec<crate::resources::StreamtubeGpuData>;
            let tube_gpu_ptr =
                &self.tube_gpu_data as *const Vec<crate::resources::StreamtubeGpuData>;
            let ribbon_gpu_ptr =
                &self.ribbon_gpu_data as *const Vec<crate::resources::StreamtubeGpuData>;
            let polyline_gpu_ptr =
                &self.polyline_gpu_data as *const Vec<crate::resources::PolylineGpuData>;
            let implicit_gpu_ptr = &self.implicit_gpu_data
                as *const Vec<crate::resources::volume::implicit::ImplicitGpuItem>;
            let mc_gpu_data_ptr = &self.mc_gpu_data
                as *const Vec<crate::resources::volume::gpu_marching_cubes::McFrameData>;
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
                splat_outlines,
                streamtube_outline_items,
                tube_outline_items,
                ribbon_outline_items,
                polyline_outline_idxs,
                vol_outline_indices,
                glyph_outline_indices,
                tensor_glyph_outline_indices,
                sprite_outline_indices,
                raw_geom_outlines,
                screen_rect_outlines,
                implicit_outline_idxs,
                mc_outlines,
                glyph_gpu_data,
                tensor_glyph_gpu_data,
                sprite_gpu_data,
                streamtube_gpu_data,
                tube_gpu_data,
                ribbon_gpu_data,
                polyline_gpu_data,
                implicit_gpu_data,
                mc_gpu_frame_data,
                camera_bg,
                mask_view,
                colour_view,
                depth_view,
                edge_bg,
            ) = unsafe {
                (
                    &*outlines_ptr,
                    &*splat_outlines_ptr,
                    &*streamtube_outline_items_ptr,
                    &*tube_outline_items_ptr,
                    &*ribbon_outline_items_ptr,
                    &*polyline_outline_idx_ptr,
                    &*vol_outline_idx_ptr,
                    &*glyph_outline_idx_ptr,
                    &*tensor_glyph_outline_idx_ptr,
                    &*sprite_outline_idx_ptr,
                    &*raw_geom_outlines_ptr,
                    &*screen_rect_outlines_ptr,
                    &*implicit_outline_idx_ptr,
                    &*mc_outlines_ptr,
                    &*glyph_gpu_ptr,
                    &*tensor_glyph_gpu_ptr,
                    &*sprite_gpu_ptr,
                    &*streamtube_gpu_ptr,
                    &*tube_gpu_ptr,
                    &*ribbon_gpu_ptr,
                    &*polyline_gpu_ptr,
                    &*implicit_gpu_ptr,
                    &*mc_gpu_data_ptr,
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
                    #[cfg(feature = "wgpu29")]
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
                pass.set_bind_group(2, &self.resources.deform.dummy_bind_group, &[]);
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
                    pass.set_bind_group(
                        2,
                        self.resources
                            .deform
                            .instance_bind_group_for(outlined.mesh_id, outlined.deform_instance),
                        &[],
                    );
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    // Use the compacted index buffer when a compute filter clipped this
                    // mesh, so the outline follows the filtered geometry (matching the
                    // scene pass) rather than the full mesh.
                    let filter = self
                        .compute_filter_results
                        .iter()
                        .find(|r| r.mesh_id == outlined.mesh_id);
                    let (index_buf, index_count) = match filter {
                        Some(f) => (&f.index_buffer, f.index_count),
                        None => (&mesh.index_buffer, mesh.index_count),
                    };
                    pass.set_index_buffer(index_buf.slice(..), crate::gpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..index_count, 0, 0..1);
                }

                // Draw Gaussian splat outline discs.  Each splat position expands to
                // a screen-space disc in the vertex shader (6 vertices per instance).
                // Depth is tested (splats behind selected meshes are culled) but not
                // written, so all visible splats in a cloud contribute to the mask.
                pass.set_pipeline(&self.resources.outline.splat_mask_pipeline);
                for splat in splat_outlines {
                    pass.set_bind_group(1, &splat.bind_group, &[]);
                    pass.set_vertex_buffer(0, splat.position_buf.slice(..));
                    pass.set_vertex_buffer(1, splat.size_buf.slice(..));
                    pass.draw(0..6, 0..splat.instance_count);
                }

                // Draw glyph instances into the mask using the actual instanced
                // mesh geometry so the outline follows arrow/sphere shapes.
                if !glyph_outline_indices.is_empty() {
                    if let Some(pipeline) = self.resources.glyph.outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        for (idx, instance_filter) in glyph_outline_indices {
                            if let Some(glyph) = glyph_gpu_data.get(*idx) {
                                pass.set_bind_group(0, camera_bg, &[]);
                                pass.set_bind_group(1, &glyph.uniform_bind_group, &[]);
                                pass.set_bind_group(2, &glyph.instance_bind_group, &[]);
                                pass.set_vertex_buffer(0, glyph.mesh_vertex_buffer.slice(..));
                                pass.set_index_buffer(
                                    glyph.mesh_index_buffer.slice(..),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                match instance_filter {
                                    None => {
                                        pass.draw_indexed(
                                            0..glyph.mesh_index_count,
                                            0,
                                            0..glyph.instance_count,
                                        );
                                    }
                                    Some(indices) => {
                                        for &i in indices {
                                            pass.draw_indexed(
                                                0..glyph.mesh_index_count,
                                                0,
                                                i..i + 1,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw tensor glyph instances into the mask (instanced ellipsoids).
                if !tensor_glyph_outline_indices.is_empty() {
                    if let Some(pipeline) =
                        self.resources.tensor_glyph.outline_mask_pipeline.as_ref()
                    {
                        pass.set_pipeline(pipeline);
                        for (idx, instance_filter) in tensor_glyph_outline_indices {
                            if let Some(tg) = tensor_glyph_gpu_data.get(*idx) {
                                pass.set_bind_group(0, camera_bg, &[]);
                                pass.set_bind_group(1, &tg.uniform_bind_group, &[]);
                                pass.set_bind_group(2, &tg.instance_bind_group, &[]);
                                pass.set_vertex_buffer(0, tg.mesh_vertex_buffer.slice(..));
                                pass.set_index_buffer(
                                    tg.mesh_index_buffer.slice(..),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                match instance_filter {
                                    None => {
                                        pass.draw_indexed(
                                            0..tg.mesh_index_count,
                                            0,
                                            0..tg.instance_count,
                                        );
                                    }
                                    Some(indices) => {
                                        for &i in indices {
                                            pass.draw_indexed(0..tg.mesh_index_count, 0, i..i + 1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw sprite billboards into the mask so the outline matches
                // each sprite's actual quad shape and per-instance size.
                if !sprite_outline_indices.is_empty() {
                    if let Some(pipeline) = self.resources.sprite.outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        for (idx, instance_filter) in sprite_outline_indices {
                            if let Some(sprite) = sprite_gpu_data.get(*idx) {
                                pass.set_bind_group(0, camera_bg, &[]);
                                pass.set_bind_group(1, &sprite.bind_group, &[]);
                                pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
                                match instance_filter {
                                    None => {
                                        pass.draw(0..6, 0..sprite.sprite_count);
                                    }
                                    Some(indices) => {
                                        for &i in indices {
                                            pass.draw(0..6, i..i + 1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw volumes into the mask using a simplified ray march so the
                // outline hugs the actual volume silhouette, not the AABB.
                if !vol_outline_indices.is_empty() {
                    if let Some(pipeline) = self.resources.volume.outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        for &idx in vol_outline_indices {
                            if let Some(vol) = self.volume_gpu_data.get(idx) {
                                pass.set_bind_group(1, &vol.bind_group, &[]);
                                pass.set_vertex_buffer(0, vol.vertex_buffer.slice(..));
                                pass.set_index_buffer(
                                    vol.index_buffer.slice(..),
                                    crate::gpu::IndexFormat::Uint32,
                                );
                                pass.draw_indexed(0..36, 0, 0..1);
                            }
                        }
                    }
                }

                // Draw inline-geometry quads for image slices.
                for raw in raw_geom_outlines {
                    let pipeline = if raw.two_sided {
                        &self.resources.outline.mask_two_sided_pipeline
                    } else {
                        &self.resources.outline.mask_pipeline
                    };
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(1, &raw.mask_bind_group, &[]);
                    pass.set_bind_group(2, &self.resources.deform.dummy_bind_group, &[]);
                    pass.set_vertex_buffer(0, raw.vertex_buf.slice(..));
                    pass.set_index_buffer(raw.index_buf.slice(..), crate::gpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..raw.index_count, 0, 0..1);
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

                // Draw GPU implicit surface outlines via ray-march to mask.
                if !implicit_outline_idxs.is_empty() {
                    if let Some(pipeline) = self.resources.implicit.outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, camera_bg, &[]);
                        for &idx in implicit_outline_idxs {
                            if let Some(gpu) = implicit_gpu_data.get(idx) {
                                pass.set_bind_group(1, &gpu.bind_group, &[]);
                                pass.draw(0..6, 0..1);
                            }
                        }
                    }
                }

                // Draw GPU marching cubes outlines (stride-24 vertex buffer, draw_indirect).
                if !mc_outlines.is_empty() {
                    if let Some(pipeline) = self.resources.mc.outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, camera_bg, &[]);
                        for mc_out in mc_outlines {
                            pass.set_bind_group(1, &mc_out.mask_bind_group, &[]);
                            if let Some(mc) = mc_gpu_frame_data.get(mc_out.mc_gpu_idx) {
                                if let Some(vol) = self.resources.mc.volumes.get(mc.volume_idx) {
                                    for slab in &vol.slabs {
                                        pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                                        pass.draw_indirect(&slab.indirect_buf, 0);
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw streamtube, tube, and ribbon mesh outlines. Streamtubes and
                // tubes use the back-face-culled pipeline; ribbons use the two-sided
                // pipeline because they are flat surfaces with no clear front face.
                pass.set_bind_group(0, camera_bg, &[]);
                pass.set_bind_group(2, &self.resources.deform.dummy_bind_group, &[]);
                let curve_draw_groups = [
                    (
                        streamtube_outline_items as &[CurveMeshOutlineItem],
                        streamtube_gpu_data as &[crate::resources::StreamtubeGpuData],
                    ),
                    (tube_outline_items, tube_gpu_data),
                    (ribbon_outline_items, ribbon_gpu_data),
                ];
                for (items, gpu_data_slice) in &curve_draw_groups {
                    for item in *items {
                        let pipeline = if item.two_sided {
                            &self.resources.outline.mask_two_sided_pipeline
                        } else {
                            &self.resources.outline.mask_pipeline
                        };
                        pass.set_pipeline(pipeline);
                        if let Some(gpu) = gpu_data_slice.get(item.index) {
                            pass.set_bind_group(1, &item.mask_bind_group, &[]);
                            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                            pass.set_index_buffer(
                                gpu.index_buffer.slice(..),
                                crate::gpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
                        }
                    }
                }

                // Draw polyline segment quads into the mask using the dedicated
                // polyline_outline_mask_pipeline (instance-expanded quads).
                if !polyline_outline_idxs.is_empty() {
                    if let Some(pipeline) = self.resources.polyline.outline_mask_pipeline.as_ref() {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, camera_bg, &[]);
                        for &idx in polyline_outline_idxs {
                            if let Some(pline) = polyline_gpu_data.get(idx) {
                                pass.set_bind_group(1, &pline.bind_group, &[]);
                                pass.set_vertex_buffer(0, pline.vertex_buffer.slice(..));
                                pass.draw(0..6, 0..pline.segment_count);
                            }
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
                    #[cfg(feature = "wgpu29")]
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
                        frame.interaction.sub_highlight_face_fill_colour,
                        frame.interaction.sub_highlight_edge_colour,
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
