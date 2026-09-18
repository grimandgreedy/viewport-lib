use super::*;

/// The streamtube and tube bind group layout. Uploads build their bind groups
/// against it, so it lives here with the stores rather than with the item
/// types' pipelines, and is created up front: it is a layout, not a compiled
/// pipeline. The streamtube and tube item types own their own render, pick and
/// mask pipelines.
pub(crate) struct StreamtubeResources {
    /// Bind group layout for streamtube and tube uniforms (group 1).
    pub(crate) bgl: crate::gpu::BindGroupLayout,
}

impl StreamtubeResources {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        Self {
            bgl: crate::resources::builders::uniform_bgl(
                device,
                "streamtube_bgl",
                crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            ),
        }
    }
}

/// The ribbon bind group layout. Uploads build their bind groups against it,
/// so it lives here with the store rather than with the item type's pipelines,
/// and is created up front: it is a layout, not a compiled pipeline. The
/// ribbon item type owns its own render, OIT, shadow, pick and mask pipelines.
///
/// The layout adds an optional streak texture and sampler alongside the shared
/// uniform binding. The fragment shader keys off `has_texture` and falls back
/// to the resolved colour when no texture is bound.
pub(crate) struct RibbonResources {
    /// Bind group layout for ribbons (group 1): uniform + optional streak
    /// texture + sampler.
    pub(crate) bgl: crate::gpu::BindGroupLayout,
}

impl RibbonResources {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        Self {
            bgl: crate::resources::builders::uniform_texture_sampler_bgl(
                device,
                "ribbon_bgl",
                crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
                crate::gpu::ShaderStages::FRAGMENT,
            ),
        }
    }
}

impl DeviceResources {
    /// Upload one [`StreamtubeItem`] to the GPU and return draw data.
    ///
    /// Generates a connected tube mesh CPU-side using a parallel-transport frame along
    /// each polyline strip, then uploads the result as a single owned vertex+index buffer.
    /// Adjacent rings are joined by quads (2 triangles each) giving a smooth, seamless tube
    /// without the z-fighting or inter-segment gaps that plagued the old instanced approach.
    pub(crate) fn upload_streamtube_per_frame(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::StreamtubeItem,
        wireframe: bool,
    ) -> StreamtubeGpuData {
        const SIDES: usize = 12; // tube cross-section resolution

        let radius = item.radius.max(f32::EPSILON);

        let mut verts: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        // Per-triangle segment / strip maps, filled in lockstep with `indices`
        // (one entry per triangle) so a GPU pick's `primitive_index` resolves to
        // a `SubObjectRef::Segment` / `Strip`. `seg_acc` accumulates the global
        // segment base per strip the same way the CPU picker's `strip_for_segment`
        // walks `strip_lengths` (each strip owns `len - 1` segments), so both
        // backends agree on segment numbering.
        let mut tri_segment: Vec<u32> = Vec::new();
        let mut tri_strip: Vec<u32> = Vec::new();
        let mut seg_acc: u32 = 0;

        let positions = &item.positions;
        let mut strip_start = 0usize;

        for (strip_idx, &strip_len) in item.strip_lengths.iter().enumerate() {
            let strip_idx = strip_idx as u32;
            let strip_len = strip_len as usize;
            let seg_base = seg_acc;
            seg_acc += strip_len.saturating_sub(1) as u32;
            let strip_end = (strip_start + strip_len).min(positions.len());
            let pts: Vec<glam::Vec3> = positions[strip_start..strip_end]
                .iter()
                .map(|&p| glam::Vec3::from(p))
                .collect();
            strip_start += strip_len;

            if pts.len() < 2 {
                continue;
            }

            // ---- Parallel transport frame ----------------------------------------
            // Seed: find an initial tangent and an arbitrary perpendicular.
            let t0 = (pts[1] - pts[0]).normalize_or_zero();
            if t0.length_squared() < 1e-10 {
                continue;
            }
            // Choose a reference vector not parallel to t0.
            let ref_v = if t0.x.abs() < 0.9 {
                glam::Vec3::X
            } else {
                glam::Vec3::Y
            };
            let mut u = t0.cross(ref_v).normalize(); // initial "up"

            // Emit rings for each point, transporting the frame forward.
            let ring_base = verts.len() as u32;
            let n_rings = pts.len();

            for (k, &pt) in pts.iter().enumerate() {
                // Tangent at this point (forward difference, except at the last point).
                let tangent = if k + 1 < pts.len() {
                    (pts[k + 1] - pt).normalize_or_zero()
                } else {
                    (pt - pts[k - 1]).normalize_or_zero()
                };

                // Transport u: project out the component along the new tangent.
                if k > 0 {
                    let t_prev = (pts[k] - pts[k - 1]).normalize_or_zero();
                    // Rodrigues rotation: rotate u by the same angle that t_prev -> tangent.
                    let axis = t_prev.cross(tangent);
                    let sin_a = axis.length().min(1.0);
                    if sin_a > 1e-6 {
                        let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                        let ax = axis / sin_a;
                        // Rodrigues: u' = u cos(a) + cross(ax, u) sin(a) + ax dot(ax, u) (1 - cos(a))
                        u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                        u = u.normalize_or_zero();
                    }
                }

                let v = tangent.cross(u).normalize_or_zero();

                // Emit SIDES vertices around the ring.
                for s in 0..SIDES {
                    let theta = 2.0 * std::f32::consts::PI * (s as f32) / (SIDES as f32);
                    let nx = theta.cos() * u.x + theta.sin() * v.x;
                    let ny = theta.cos() * u.y + theta.sin() * v.y;
                    let nz = theta.cos() * u.z + theta.sin() * v.z;
                    let normal = glam::Vec3::new(nx, ny, nz);
                    let world_pos = pt + normal * radius;
                    verts.push(Vertex {
                        position: world_pos.to_array(),
                        normal: normal.to_array(),
                        colour: [1.0, 1.0, 1.0, 1.0], // overridden by uniform in shader
                        uv: [0.0, 0.0],
                        tangent: [1.0, 0.0, 0.0, 1.0],
                    });
                }

                // Emit quad strip between ring k-1 and ring k.
                // Winding: outward-facing CCW (right-hand rule gives outward normal).
                // Verified: T1=(r0+s, r0+s1, r1+s) has dot(normal, Y) > 0 for s=0 on Z-axis tube.
                if k > 0 {
                    let r0 = ring_base + ((k - 1) * SIDES) as u32;
                    let r1 = ring_base + (k * SIDES) as u32;
                    let seg = seg_base + (k - 1) as u32;
                    for s in 0..SIDES {
                        let s1 = (s + 1) % SIDES;
                        indices.push(r0 + s as u32);
                        indices.push(r0 + s1 as u32);
                        indices.push(r1 + s as u32);

                        indices.push(r0 + s1 as u32);
                        indices.push(r1 + s1 as u32);
                        indices.push(r1 + s as u32);

                        tri_segment.push(seg);
                        tri_segment.push(seg);
                        tri_strip.push(strip_idx);
                        tri_strip.push(strip_idx);
                    }
                }
            }

            // Segment index for the caps: the end cap belongs to the last segment
            // of the strip, the start cap to the first.
            let last_seg = seg_base + (n_rings - 2) as u32;

            // End cap (flat fan at last ring, facing forward = outward at tube end).
            // CCW from the forward direction: (center, s, s1).
            {
                let last_ring = ring_base + ((n_rings - 1) * SIDES) as u32;
                let tangent = (pts[n_rings - 1] - pts[n_rings - 2]).normalize_or_zero();
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[n_rings - 1].to_array(),
                    normal: tangent.to_array(),
                    colour: [1.0, 1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                for s in 0..SIDES {
                    let s1 = (s + 1) % SIDES;
                    indices.push(cap_center_idx);
                    indices.push(last_ring + s as u32);
                    indices.push(last_ring + s1 as u32);
                    tri_segment.push(last_seg);
                    tri_strip.push(strip_idx);
                }
            }

            // Start cap (flat fan at first ring, facing backward = outward at tube start).
            // CCW from the backward direction = CW from forward = (center, s1, s).
            {
                let tangent = (pts[0] - pts[1]).normalize_or_zero();
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[0].to_array(),
                    normal: tangent.to_array(),
                    colour: [1.0, 1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                for s in 0..SIDES {
                    let s1 = (s + 1) % SIDES;
                    indices.push(cap_center_idx);
                    indices.push(ring_base + s1 as u32);
                    indices.push(ring_base + s as u32);
                    tri_segment.push(seg_base);
                    tri_strip.push(strip_idx);
                }
            }
        }

        // Upload vertex + index buffers.
        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("streamtube_vbuf"),
            size: vert_bytes.len().max(std::mem::size_of::<Vertex>()) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !vert_bytes.is_empty() {
            queue.write_buffer(&vertex_buffer, 0, vert_bytes);
        }

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("streamtube_ibuf"),
            size: idx_bytes.len().max(12) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !idx_bytes.is_empty() {
            queue.write_buffer(&index_buffer, 0, idx_bytes);
        }

        let index_count = indices.len() as u32;

        // Edge index buffer: deduplicated triangle edges as line-list pairs for wireframe.
        let edge_indices = crate::resources::mesh::geometry::generate_edge_indices(&indices);
        let edge_bytes: &[u8] = bytemuck::cast_slice(&edge_indices);
        let edge_index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("streamtube_edge_ibuf"),
            size: edge_bytes.len().max(8) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !edge_bytes.is_empty() {
            queue.write_buffer(&edge_index_buffer, 0, edge_bytes);
        }
        let edge_index_count = edge_indices.len() as u32;

        // Uniform buffer: model + colour + radius + use_vertex_colour + unlit + opacity + wireframe.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StreamtubeUniform {
            model: [[f32; 4]; 4],
            colour: [f32; 4],
            radius: f32,
            use_vertex_colour: u32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
            _pad: [f32; 3],
        }
        let uniform_data = StreamtubeUniform {
            model: item.model,
            colour: item.colour.to_linear_rgba(),
            radius,
            use_vertex_colour: 0,
            unlit: item.settings.unlit as u32,
            opacity: item.settings.opacity,
            wireframe: wireframe as u32,
            _pad: [0.0; 3],
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("streamtube_uniform_buf"),
            size: std::mem::size_of::<StreamtubeUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = &self.streamtube.bgl;
        let uniform_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("streamtube_uniform_bg"),
            layout: bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        StreamtubeGpuData {
            vertex_buffer,
            index_buffer,
            index_count,
            edge_index_buffer,
            edge_index_count,
            wireframe,
            uniform_bind_group,
            blend: crate::renderer::SpriteBlend::AlphaBlend,
            pick_id: crate::renderer::PickId::NONE,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            cast_shadows: true,
            oit_eligible: false,
            depth_write: true,
            node_pick_buffer: build_node_pick_buffer(
                device,
                queue,
                &tri_segment,
                &item.positions,
                &item.strip_lengths,
            ),
            tri_segment,
            tri_strip,
            _uniform_buf: uniform_buf,
        }
    }

    /// Pre-upload a streamtube and return a typed handle.
    ///
    /// Submit a [`StreamtubeRefItem`](crate::renderer::StreamtubeRefItem) on
    /// `SceneFrame::streamtube_refs` each frame to draw the tube at a
    /// per-frame model transform without rebuilding its mesh.
    ///
    /// Prefer [`ViewportRenderer::upload_streamtube`](crate::renderer::ViewportRenderer::upload_streamtube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_streamtube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::StreamtubeItem,
    ) -> crate::resources::StreamtubeId {
        let gpu = self.upload_streamtube_per_frame(device, queue, item, false);
        self.content.streamtube_store.insert_sized(gpu)
    }

    /// Remove a pre-uploaded streamtube.
    ///
    /// Prefer [`ViewportRenderer::drop_streamtube`](crate::renderer::ViewportRenderer::drop_streamtube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::drop_streamtube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn drop_streamtube(&mut self, id: crate::resources::StreamtubeId) -> bool {
        self.content.streamtube_store.remove(id).is_some()
    }

    /// Replace the geometry of a pre-uploaded streamtube, keeping the same id.
    ///
    /// Prefer [`ViewportRenderer::replace_streamtube`](crate::renderer::ViewportRenderer::replace_streamtube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::replace_streamtube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn replace_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::StreamtubeId,
        item: &crate::renderer::StreamtubeItem,
    ) -> bool {
        if !self.content.streamtube_store.contains(id) {
            return false;
        }
        let gpu = self.upload_streamtube_per_frame(device, queue, item, false);
        self.content
            .streamtube_store
            .replace_sized(id, gpu)
            .is_some()
    }

    // -------------------------------------------------------------------------
    // General Tube representation
    // -------------------------------------------------------------------------

    /// Upload one [`TubeItem`] to the GPU and return draw data.
    ///
    /// Generates a connected tube mesh CPU-side using a parallel-transport frame.
    /// Scalar values are baked into per-vertex colours using the CPU-side colourmap copy.
    /// Uses the same streamtube pipeline; sets `use_vertex_colour=1` when scalars are present.
    pub(crate) fn upload_tube_per_frame(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::TubeItem,
        wireframe: bool,
    ) -> StreamtubeGpuData {
        let sides = (item.sides.max(3)) as usize;

        // Resolve scalar-to-colour mapping upfront if scalars are provided.
        let (use_vertex_colour, lut_rgba): (u32, Option<[[u8; 4]; 256]>) =
            if !item.scalars.is_empty() {
                let lut = self
                    .content
                    .builtin_colourmap_ids
                    .and_then(|ids| {
                        let preset_id = item
                            .colourmap_id
                            .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                        self.content.colourmaps_cpu.get(preset_id.0).copied()
                    })
                    .unwrap_or([[128u8; 4]; 256]);
                (1, Some(lut))
            } else {
                (0, None)
            };

        let scalar_min = item
            .scalar_range
            .map(|r| r.0)
            .unwrap_or_else(|| item.scalars.iter().cloned().fold(f32::INFINITY, f32::min));
        let scalar_max = item.scalar_range.map(|r| r.1).unwrap_or_else(|| {
            item.scalars
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        });
        let scalar_range = (scalar_max - scalar_min).max(f32::EPSILON);

        // Helper: map a scalar value to an RGBA f32 colour from the LUT.
        let scalar_to_colour = |idx: usize| -> [f32; 4] {
            if let Some(ref lut) = lut_rgba {
                let s = *item.scalars.get(idx).unwrap_or(&0.0);
                let t = ((s - scalar_min) / scalar_range).clamp(0.0, 1.0);
                let lut_idx = ((t * 255.0).round() as usize).min(255);
                let c = lut[lut_idx];
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]
            } else {
                item.colour.to_linear_rgba()
            }
        };

        let mut verts: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        // Per-triangle segment / strip maps (see the streamtube builder for the
        // numbering convention). One entry per triangle in `indices`.
        let mut tri_segment: Vec<u32> = Vec::new();
        let mut tri_strip: Vec<u32> = Vec::new();
        let mut seg_acc: u32 = 0;

        let positions = &item.positions;
        let mut strip_start = 0usize;

        for (strip_idx, &strip_len) in item.strip_lengths.iter().enumerate() {
            let strip_idx = strip_idx as u32;
            let strip_len = strip_len as usize;
            let seg_base = seg_acc;
            seg_acc += strip_len.saturating_sub(1) as u32;
            let strip_end = (strip_start + strip_len).min(positions.len());
            let pts: Vec<glam::Vec3> = positions[strip_start..strip_end]
                .iter()
                .map(|&p| glam::Vec3::from(p))
                .collect();
            let pts_scalar_start = strip_start;
            strip_start += strip_len;

            if pts.len() < 2 {
                continue;
            }

            // Parallel transport frame (same as upload_streamtube).
            let t0 = (pts[1] - pts[0]).normalize_or_zero();
            if t0.length_squared() < 1e-10 {
                continue;
            }
            let ref_v = if t0.x.abs() < 0.9 {
                glam::Vec3::X
            } else {
                glam::Vec3::Y
            };
            let mut u = t0.cross(ref_v).normalize();

            let ring_base = verts.len() as u32;
            let n_rings = pts.len();

            for (k, &pt) in pts.iter().enumerate() {
                let tangent = if k + 1 < pts.len() {
                    (pts[k + 1] - pt).normalize_or_zero()
                } else {
                    (pt - pts[k - 1]).normalize_or_zero()
                };

                if k > 0 {
                    let t_prev = (pts[k] - pts[k - 1]).normalize_or_zero();
                    let axis = t_prev.cross(tangent);
                    let sin_a = axis.length().min(1.0);
                    if sin_a > 1e-6 {
                        let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                        let ax = axis / sin_a;
                        u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                        u = u.normalize_or_zero();
                    }
                }

                let v = tangent.cross(u).normalize_or_zero();

                // Per-point radius: from radius_attribute if provided, else uniform radius.
                let point_radius = item
                    .radius_attribute
                    .as_ref()
                    .and_then(|ra| ra.get(pts_scalar_start + k).copied())
                    .unwrap_or(item.radius)
                    .max(f32::EPSILON);

                let vertex_colour = scalar_to_colour(pts_scalar_start + k);

                for s in 0..sides {
                    let theta = 2.0 * std::f32::consts::PI * (s as f32) / (sides as f32);
                    let nx = theta.cos() * u.x + theta.sin() * v.x;
                    let ny = theta.cos() * u.y + theta.sin() * v.y;
                    let nz = theta.cos() * u.z + theta.sin() * v.z;
                    let normal = glam::Vec3::new(nx, ny, nz);
                    let world_pos = pt + normal * point_radius;
                    verts.push(Vertex {
                        position: world_pos.to_array(),
                        normal: normal.to_array(),
                        colour: vertex_colour,
                        uv: [0.0, 0.0],
                        tangent: [1.0, 0.0, 0.0, 1.0],
                    });
                }

                if k > 0 {
                    let r0 = ring_base + ((k - 1) * sides) as u32;
                    let r1 = ring_base + (k * sides) as u32;
                    let seg = seg_base + (k - 1) as u32;
                    for s in 0..sides {
                        let s1 = (s + 1) % sides;
                        indices.push(r0 + s as u32);
                        indices.push(r0 + s1 as u32);
                        indices.push(r1 + s as u32);

                        indices.push(r0 + s1 as u32);
                        indices.push(r1 + s1 as u32);
                        indices.push(r1 + s as u32);

                        tri_segment.push(seg);
                        tri_segment.push(seg);
                        tri_strip.push(strip_idx);
                        tri_strip.push(strip_idx);
                    }
                }
            }

            let last_seg = seg_base + (n_rings - 2) as u32;

            // End cap.
            {
                let last_ring = ring_base + ((n_rings - 1) * sides) as u32;
                let tangent = (pts[n_rings - 1] - pts[n_rings - 2]).normalize_or_zero();
                let cap_colour = scalar_to_colour(pts_scalar_start + n_rings - 1);
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[n_rings - 1].to_array(),
                    normal: tangent.to_array(),
                    colour: cap_colour,
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                for s in 0..sides {
                    let s1 = (s + 1) % sides;
                    indices.push(cap_center_idx);
                    indices.push(last_ring + s as u32);
                    indices.push(last_ring + s1 as u32);
                    tri_segment.push(last_seg);
                    tri_strip.push(strip_idx);
                }
            }

            // Start cap.
            {
                let tangent = (pts[0] - pts[1]).normalize_or_zero();
                let cap_colour = scalar_to_colour(pts_scalar_start);
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[0].to_array(),
                    normal: tangent.to_array(),
                    colour: cap_colour,
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                for s in 0..sides {
                    let s1 = (s + 1) % sides;
                    indices.push(cap_center_idx);
                    indices.push(ring_base + s1 as u32);
                    indices.push(ring_base + s as u32);
                    tri_segment.push(seg_base);
                    tri_strip.push(strip_idx);
                }
            }
        }

        // Upload vertex + index buffers.
        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tube_vbuf"),
            size: vert_bytes.len().max(std::mem::size_of::<Vertex>()) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !vert_bytes.is_empty() {
            queue.write_buffer(&vertex_buffer, 0, vert_bytes);
        }

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tube_ibuf"),
            size: idx_bytes.len().max(12) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !idx_bytes.is_empty() {
            queue.write_buffer(&index_buffer, 0, idx_bytes);
        }

        let index_count = indices.len() as u32;

        let edge_indices = crate::resources::mesh::geometry::generate_edge_indices(&indices);
        let edge_bytes: &[u8] = bytemuck::cast_slice(&edge_indices);
        let edge_index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tube_edge_ibuf"),
            size: edge_bytes.len().max(8) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !edge_bytes.is_empty() {
            queue.write_buffer(&edge_index_buffer, 0, edge_bytes);
        }
        let edge_index_count = edge_indices.len() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TubeUniform {
            model: [[f32; 4]; 4],
            colour: [f32; 4],
            radius: f32,
            use_vertex_colour: u32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
            _pad: [f32; 3],
        }
        let uniform_data = TubeUniform {
            model: item.model,
            colour: item.colour.to_linear_rgba(),
            radius: item.radius.max(f32::EPSILON),
            use_vertex_colour,
            unlit: item.settings.unlit as u32,
            opacity: item.settings.opacity,
            wireframe: wireframe as u32,
            _pad: [0.0; 3],
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("tube_uniform_buf"),
            size: std::mem::size_of::<TubeUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = &self.streamtube.bgl;
        let uniform_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("tube_uniform_bg"),
            layout: bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        StreamtubeGpuData {
            vertex_buffer,
            index_buffer,
            index_count,
            edge_index_buffer,
            edge_index_count,
            wireframe,
            uniform_bind_group,
            blend: crate::renderer::SpriteBlend::AlphaBlend,
            pick_id: crate::renderer::PickId::NONE,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            cast_shadows: true,
            oit_eligible: false,
            depth_write: true,
            node_pick_buffer: build_node_pick_buffer(
                device,
                queue,
                &tri_segment,
                &item.positions,
                &item.strip_lengths,
            ),
            tri_segment,
            tri_strip,
            _uniform_buf: uniform_buf,
        }
    }

    /// Pre-upload a general tube and return a typed handle.
    ///
    /// Prefer [`ViewportRenderer::upload_tube`](crate::renderer::ViewportRenderer::upload_tube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_tube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::TubeItem,
    ) -> crate::resources::TubeId {
        let gpu = self.upload_tube_per_frame(device, queue, item, false);
        self.content.tube_store.insert_sized(gpu)
    }

    /// Remove a pre-uploaded tube.
    ///
    /// Prefer [`ViewportRenderer::drop_tube`](crate::renderer::ViewportRenderer::drop_tube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::drop_tube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn drop_tube(&mut self, id: crate::resources::TubeId) -> bool {
        self.content.tube_store.remove(id).is_some()
    }

    /// Replace the geometry of a pre-uploaded tube, keeping the same id.
    ///
    /// Prefer [`ViewportRenderer::replace_tube`](crate::renderer::ViewportRenderer::replace_tube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::replace_tube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn replace_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::TubeId,
        item: &crate::renderer::TubeItem,
    ) -> bool {
        if !self.content.tube_store.contains(id) {
            return false;
        }
        let gpu = self.upload_tube_per_frame(device, queue, item, false);
        self.content.tube_store.replace_sized(id, gpu).is_some()
    }

    // -------------------------------------------------------------------------
    // Ribbon representation
    // -------------------------------------------------------------------------

    /// Build and upload GPU data for a `RibbonItem`.
    ///
    /// Each strip is swept as a flat quad surface. Two vertices are generated per
    /// point (left and right edges), connected as a triangle strip. The normal is
    /// the cross product of the tangent and the lateral direction `u`.
    pub(crate) fn upload_ribbon_per_frame(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::RibbonItem,
        wireframe: bool,
    ) -> StreamtubeGpuData {
        self.check_texture_slot(item.texture_id, crate::resources::TextureSlot::RibbonAlbedo);

        // Per-vertex RGBA (`colour_attribute`) takes precedence over the
        // scalar+LUT path and the flat `colour` fallback. Trails typically
        // drive only the alpha channel to fade along their length.
        let has_colour_attribute = !item.colour_attribute.is_empty();

        // Resolve LUT for scalar colouring.
        let (use_vertex_colour, lut_rgba): (u32, Option<[[u8; 4]; 256]>) = if has_colour_attribute {
            (1, None)
        } else if !item.scalars.is_empty() {
            let lut = self
                .content
                .builtin_colourmap_ids
                .and_then(|ids| {
                    let preset_id = item
                        .colourmap_id
                        .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                    self.content.colourmaps_cpu.get(preset_id.0).copied()
                })
                .unwrap_or([[128u8; 4]; 256]);
            (1, Some(lut))
        } else {
            (0, None)
        };

        let scalar_min = item
            .scalar_range
            .map(|r| r.0)
            .unwrap_or_else(|| item.scalars.iter().cloned().fold(f32::INFINITY, f32::min));
        let scalar_max = item.scalar_range.map(|r| r.1).unwrap_or_else(|| {
            item.scalars
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        });
        let scalar_range = (scalar_max - scalar_min).max(f32::EPSILON);

        let scalar_to_colour = |idx: usize| -> [f32; 4] {
            if has_colour_attribute {
                return item
                    .colour_attribute
                    .get(idx)
                    .copied()
                    .unwrap_or(item.colour)
                    .to_linear_rgba();
            }
            if let Some(ref lut) = lut_rgba {
                let s = *item.scalars.get(idx).unwrap_or(&0.0);
                let t = ((s - scalar_min) / scalar_range).clamp(0.0, 1.0);
                let lut_idx = ((t * 255.0).round() as usize).min(255);
                let c = lut[lut_idx];
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]
            } else {
                item.colour.to_linear_rgba()
            }
        };

        let mut verts: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        // Per-triangle segment / strip maps (see the streamtube builder for the
        // numbering convention). Ribbons emit two triangles per segment and no
        // caps.
        let mut tri_segment: Vec<u32> = Vec::new();
        let mut tri_strip: Vec<u32> = Vec::new();
        let mut seg_acc: u32 = 0;

        let positions = &item.positions;
        let mut strip_start = 0usize;

        for (strip_idx, &strip_len) in item.strip_lengths.iter().enumerate() {
            let strip_idx = strip_idx as u32;
            let strip_len = strip_len as usize;
            let seg_base = seg_acc;
            seg_acc += strip_len.saturating_sub(1) as u32;
            let strip_end = (strip_start + strip_len).min(positions.len());
            let pts: Vec<glam::Vec3> = positions[strip_start..strip_end]
                .iter()
                .map(|&p| glam::Vec3::from(p))
                .collect();
            let pts_start = strip_start;
            strip_start += strip_len;

            if pts.len() < 2 {
                continue;
            }

            // Build parallel transport frame.
            let t0 = (pts[1] - pts[0]).normalize_or_zero();
            if t0.length_squared() < 1e-10 {
                continue;
            }
            let ref_v = if t0.x.abs() < 0.9 {
                glam::Vec3::X
            } else {
                glam::Vec3::Y
            };
            let mut u = t0.cross(ref_v).normalize();

            // Per-vertex u along the strip. Defaults to cumulative-arc-length
            // normalised to [0, 1] when the host did not supply a `u_attribute`.
            let mut strip_u: Vec<f32> = Vec::with_capacity(pts.len());
            if item.u_attribute.is_empty() {
                let mut cum = 0.0_f32;
                strip_u.push(0.0);
                for k in 1..pts.len() {
                    cum += (pts[k] - pts[k - 1]).length();
                    strip_u.push(cum);
                }
                let total = strip_u.last().copied().unwrap_or(1.0).max(1e-6);
                for v in &mut strip_u {
                    *v /= total;
                }
            } else {
                for k in 0..pts.len() {
                    strip_u.push(*item.u_attribute.get(pts_start + k).unwrap_or(&0.0));
                }
            }

            let base = verts.len() as u32;

            for (k, &pt) in pts.iter().enumerate() {
                let tangent = if k + 1 < pts.len() {
                    (pts[k + 1] - pt).normalize_or_zero()
                } else {
                    (pt - pts[k - 1]).normalize_or_zero()
                };

                // Parallel transport: rotate u to stay perpendicular to new tangent.
                if k > 0 {
                    let t_prev = (pts[k] - pts[k - 1]).normalize_or_zero();
                    let axis = t_prev.cross(tangent);
                    let sin_a = axis.length().min(1.0);
                    if sin_a > 1e-6 {
                        let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                        let ax = axis / sin_a;
                        u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                        u = u.normalize_or_zero();
                    }
                }

                // If twist_attribute provided, align u with projection of that vector onto
                // the plane perpendicular to the tangent.
                let mut lateral = u;
                if let Some(ref twist) = item.twist_attribute {
                    if let Some(&tv) = twist.get(pts_start + k) {
                        let tv = glam::Vec3::from(tv);
                        let proj = tv - tangent * tangent.dot(tv);
                        if proj.length_squared() > 1e-10 {
                            lateral = proj.normalize();
                        }
                    }
                }

                let normal = tangent.cross(lateral).normalize_or_zero();
                let half_w = item
                    .width_attribute
                    .as_ref()
                    .and_then(|wa| wa.get(pts_start + k).copied())
                    .unwrap_or(item.width)
                    * 0.5;
                let colour = scalar_to_colour(pts_start + k);

                let uval = strip_u[k];
                // Left edge vertex. `uv.x` runs along the strip; `uv.y` is the
                // cross-strip coordinate that picks the left or right edge.
                verts.push(Vertex {
                    position: (pt + lateral * half_w).to_array(),
                    normal: normal.to_array(),
                    colour,
                    uv: [uval, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                // Right edge vertex.
                verts.push(Vertex {
                    position: (pt - lateral * half_w).to_array(),
                    normal: normal.to_array(),
                    colour,
                    uv: [uval, 1.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });

                // Connect to previous pair as two triangles.
                if k > 0 {
                    let r0 = base + ((k - 1) * 2) as u32;
                    let r1 = base + (k * 2) as u32;
                    // Triangle 1: r0+0, r0+1, r1+0
                    indices.push(r0);
                    indices.push(r0 + 1);
                    indices.push(r1);
                    // Triangle 2: r0+1, r1+1, r1+0
                    indices.push(r0 + 1);
                    indices.push(r1 + 1);
                    indices.push(r1);

                    let seg = seg_base + (k - 1) as u32;
                    tri_segment.push(seg);
                    tri_segment.push(seg);
                    tri_strip.push(strip_idx);
                    tri_strip.push(strip_idx);
                }
            }
        }

        // Upload vertex + index buffers.
        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("ribbon_vbuf"),
            size: vert_bytes.len().max(std::mem::size_of::<Vertex>()) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !vert_bytes.is_empty() {
            queue.write_buffer(&vertex_buffer, 0, vert_bytes);
        }

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("ribbon_ibuf"),
            size: idx_bytes.len().max(12) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !idx_bytes.is_empty() {
            queue.write_buffer(&index_buffer, 0, idx_bytes);
        }

        let index_count = indices.len() as u32;

        let edge_indices = crate::resources::mesh::geometry::generate_edge_indices(&indices);
        let edge_bytes: &[u8] = bytemuck::cast_slice(&edge_indices);
        let edge_index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("ribbon_edge_ibuf"),
            size: edge_bytes.len().max(8) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !edge_bytes.is_empty() {
            queue.write_buffer(&edge_index_buffer, 0, edge_bytes);
        }
        let edge_index_count = edge_indices.len() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RibbonUniform {
            model: [[f32; 4]; 4],
            colour: [f32; 4],
            radius: f32,
            use_vertex_colour: u32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
            has_texture: u32,
            receive_shadows: u32,
            _pad: f32,
        }
        let (texture_view, has_texture): (&crate::gpu::TextureView, u32) =
            if let Some(id) = item.texture_id {
                if let Some(tex) = self.content.textures.get(id) {
                    (&tex.view, 1)
                } else {
                    (&self.content.fallback_lut_view, 0)
                }
            } else {
                (&self.content.fallback_lut_view, 0)
            };
        let uniform_data = RibbonUniform {
            model: item.model,
            colour: item.colour.to_linear_rgba(),
            radius: item.width * 0.5,
            use_vertex_colour,
            unlit: item.settings.unlit as u32,
            opacity: item.settings.opacity,
            wireframe: wireframe as u32,
            has_texture,
            receive_shadows: item.settings.receive_shadows as u32,
            _pad: 0.0,
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("ribbon_uniform_buf"),
            size: std::mem::size_of::<RibbonUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = &self.ribbon.bgl;
        let uniform_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("ribbon_uniform_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(texture_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&self.material.sampler),
                },
            ],
        });

        // Only a ribbon that does not write depth goes through OIT. Deciding
        // this from the blend mode alone sent every default ribbon to the OIT
        // pass, which writes no depth, so an opaque ribbon was invisible to
        // decals, soft particles and everything else that reads scene depth.
        // Mirrors the sprite rule, which gates on `depth_write` the same way.
        let oit_eligible = !wireframe
            && !item.depth_write
            && matches!(
                item.blend,
                crate::renderer::SpriteBlend::AlphaBlend
                    | crate::renderer::SpriteBlend::Premultiplied
            );

        StreamtubeGpuData {
            vertex_buffer,
            index_buffer,
            index_count,
            edge_index_buffer,
            edge_index_count,
            wireframe,
            uniform_bind_group,
            blend: item.blend,
            pick_id: crate::renderer::PickId::NONE,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            cast_shadows: true,
            oit_eligible,
            depth_write: item.depth_write,
            node_pick_buffer: build_node_pick_buffer(
                device,
                queue,
                &tri_segment,
                &item.positions,
                &item.strip_lengths,
            ),
            tri_segment,
            tri_strip,
            _uniform_buf: uniform_buf,
        }
    }

    /// Pre-upload a ribbon and return a typed handle.
    ///
    /// Prefer [`ViewportRenderer::upload_ribbon`](crate::renderer::ViewportRenderer::upload_ribbon),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_ribbon instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::RibbonItem,
    ) -> crate::resources::RibbonId {
        let gpu = self.upload_ribbon_per_frame(device, queue, item, false);
        self.content.ribbon_store.insert_sized(gpu)
    }

    /// Remove a pre-uploaded ribbon.
    ///
    /// Prefer [`ViewportRenderer::drop_ribbon`](crate::renderer::ViewportRenderer::drop_ribbon),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::drop_ribbon instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn drop_ribbon(&mut self, id: crate::resources::RibbonId) -> bool {
        self.content.ribbon_store.remove(id).is_some()
    }

    /// Replace the geometry of a pre-uploaded ribbon, keeping the same id.
    ///
    /// Prefer [`ViewportRenderer::replace_ribbon`](crate::renderer::ViewportRenderer::replace_ribbon),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::replace_ribbon instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn replace_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::RibbonId,
        item: &crate::renderer::RibbonItem,
    ) -> bool {
        if !self.content.ribbon_store.contains(id) {
            return false;
        }
        let gpu = self.upload_ribbon_per_frame(device, queue, item, false);
        self.content.ribbon_store.replace_sized(id, gpu).is_some()
    }

    /// Start an asynchronous streamtube upload.
    ///
    /// Prefer [`ViewportRenderer::begin_upload_streamtube`](crate::renderer::ViewportRenderer::begin_upload_streamtube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::begin_upload_streamtube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    // The apply step inserts through the synchronous upload, which goes with it.
    #[allow(deprecated)]
    pub fn begin_upload_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::StreamtubeItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::StreamtubeId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let sid =
                            resources.upload_streamtube(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(sid);
                    }),
                ))
            })
        };
        self.job_results
            .streamtube
            .lock()
            .expect("streamtube result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`StreamtubeId`](crate::resources::StreamtubeId) produced by a
    /// completed [`begin_upload_streamtube`](Self::begin_upload_streamtube) job.
    ///
    /// Prefer [`ViewportRenderer::upload_result_streamtube`](crate::renderer::ViewportRenderer::upload_result_streamtube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_result_streamtube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_result_streamtube(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::StreamtubeId> {
        let mut map = self
            .job_results
            .streamtube
            .lock()
            .expect("streamtube result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(sid) => {
                map.remove(&id);
                Ok(sid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Start an asynchronous tube upload.
    ///
    /// Prefer [`ViewportRenderer::begin_upload_tube`](crate::renderer::ViewportRenderer::begin_upload_tube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::begin_upload_tube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    // The apply step inserts through the synchronous upload, which goes with it.
    #[allow(deprecated)]
    pub fn begin_upload_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::TubeItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::TubeId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let tid = resources.upload_tube(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(tid);
                    }),
                ))
            })
        };
        self.job_results
            .tube
            .lock()
            .expect("tube result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`TubeId`](crate::resources::TubeId) produced by a completed
    /// [`begin_upload_tube`](Self::begin_upload_tube) job.
    ///
    /// Prefer [`ViewportRenderer::upload_result_tube`](crate::renderer::ViewportRenderer::upload_result_tube),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_result_tube instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_result_tube(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TubeId> {
        let mut map = self
            .job_results
            .tube
            .lock()
            .expect("tube result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(tid) => {
                map.remove(&id);
                Ok(tid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Start an asynchronous ribbon upload.
    ///
    /// Prefer [`ViewportRenderer::begin_upload_ribbon`](crate::renderer::ViewportRenderer::begin_upload_ribbon),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::begin_upload_ribbon instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    // The apply step inserts through the synchronous upload, which goes with it.
    #[allow(deprecated)]
    pub fn begin_upload_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::RibbonItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::RibbonId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let rid =
                            resources.upload_ribbon(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(rid);
                    }),
                ))
            })
        };
        self.job_results
            .ribbon
            .lock()
            .expect("ribbon result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`RibbonId`](crate::resources::RibbonId) produced by a completed
    /// [`begin_upload_ribbon`](Self::begin_upload_ribbon) job.
    ///
    /// Prefer [`ViewportRenderer::upload_result_ribbon`](crate::renderer::ViewportRenderer::upload_result_ribbon),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_result_ribbon instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_result_ribbon(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::RibbonId> {
        let mut map = self
            .job_results
            .ribbon
            .lock()
            .expect("ribbon result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(rid) => {
                map.remove(&id);
                Ok(rid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }
}

#[cfg(test)]
mod tests {
    // These drive the DeviceResources upload calls directly, which is the
    // point: they test the methods the renderer-level ones forward to.
    #![allow(deprecated)]
    use crate::DeviceResources;
    use crate::renderer::{RibbonItem, StreamtubeItem, TubeItem};
    use crate::resources::UploadStatus;

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn sample_streamtube() -> StreamtubeItem {
        StreamtubeItem {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            strip_lengths: vec![3],
            radius: 0.1,
            ..Default::default()
        }
    }

    fn sample_tube() -> TubeItem {
        TubeItem {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            strip_lengths: vec![3],
            radius: 0.1,
            ..Default::default()
        }
    }

    fn sample_ribbon() -> RibbonItem {
        RibbonItem {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            strip_lengths: vec![3],
            width: 0.2,
            ..Default::default()
        }
    }

    fn drive_until_ready(
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::JobId,
        label: &'static str,
    ) {
        for _ in 0..200 {
            resources.process_uploads(device, queue);
            match resources.upload_status(id) {
                UploadStatus::Ready => return,
                UploadStatus::Failed(e) => panic!("{label} upload failed: {e:?}"),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                UploadStatus::Unknown => panic!("{label} job id disappeared"),
            }
        }
        panic!("{label} upload did not complete in time");
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn streamtube_default_model_is_identity() {
        assert_eq!(StreamtubeItem::default().model, IDENTITY);
    }

    #[test]
    fn tube_default_model_is_identity() {
        assert_eq!(TubeItem::default().model, IDENTITY);
    }

    #[test]
    fn ribbon_default_model_is_identity() {
        assert_eq!(RibbonItem::default().model, IDENTITY);
    }

    #[test]
    fn streamtube_carries_non_identity_model() {
        let mut m = IDENTITY;
        m[3] = [1.0, 2.0, 3.0, 1.0];
        let item = StreamtubeItem {
            model: m,
            ..StreamtubeItem::default()
        };
        assert_eq!(item.model[3], [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn tube_carries_non_identity_model() {
        let mut m = IDENTITY;
        m[3] = [1.0, 2.0, 3.0, 1.0];
        let item = TubeItem {
            model: m,
            ..TubeItem::default()
        };
        assert_eq!(item.model[3], [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn ribbon_carries_non_identity_model() {
        let mut m = IDENTITY;
        m[3] = [1.0, 2.0, 3.0, 1.0];
        let item = RibbonItem {
            model: m,
            ..RibbonItem::default()
        };
        assert_eq!(item.model[3], [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn upload_streamtube_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_streamtube(&device, &queue, &sample_streamtube());
        assert!(resources.content.streamtube_store.contains(id));
        assert!(resources.drop_streamtube(id));
        assert!(!resources.content.streamtube_store.contains(id));
    }

    #[test]
    fn upload_tube_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let start = resources.resident_bytes().scivis_bytes;
        let id = resources.upload_tube(&device, &queue, &sample_tube());
        assert!(resources.content.tube_store.contains(id));
        let after_upload = resources.resident_bytes().scivis_bytes;
        assert!(
            after_upload > start,
            "uploading a tube must increase resident scivis bytes"
        );
        assert!(resources.drop_tube(id));
        assert_eq!(
            resources.resident_bytes().scivis_bytes,
            start,
            "dropping the tube must return resident scivis bytes to the start"
        );
    }

    #[test]
    fn upload_ribbon_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_ribbon(&device, &queue, &sample_ribbon());
        assert!(resources.content.ribbon_store.contains(id));
        assert!(resources.drop_ribbon(id));
    }

    #[test]
    fn begin_upload_streamtube_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_streamtube(&device, &queue, sample_streamtube());
        drive_until_ready(&mut resources, &device, &queue, job, "streamtube");
        let id = resources.upload_result_streamtube(job).expect("ready");
        assert!(resources.content.streamtube_store.contains(id));
        let err = resources.upload_result_streamtube(job).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::JobResultMissing { .. }
        ));
    }

    #[test]
    fn begin_upload_tube_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_tube(&device, &queue, sample_tube());
        drive_until_ready(&mut resources, &device, &queue, job, "tube");
        let id = resources.upload_result_tube(job).expect("ready");
        assert!(resources.content.tube_store.contains(id));
    }

    #[test]
    fn begin_upload_ribbon_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_ribbon(&device, &queue, sample_ribbon());
        drive_until_ready(&mut resources, &device, &queue, job, "ribbon");
        let id = resources.upload_result_ribbon(job).expect("ready");
        assert!(resources.content.ribbon_store.contains(id));
    }
}

/// Per-frame GPU data for one streamtube item, created in `prepare()`.
///
/// The connected tube mesh (vertices + indices) is generated CPU-side for the
/// entire item (all strips) and uploaded as a single owned buffer pair.
#[derive(Clone)]
pub struct StreamtubeGpuData {
    /// Owned vertex buffer for the connected tube mesh (world-space positions + normals).
    pub(crate) vertex_buffer: crate::gpu::Buffer,
    /// Owned index buffer for the connected tube mesh (triangle indices).
    pub(crate) index_buffer: crate::gpu::Buffer,
    /// Number of triangle indices to draw (solid mode).
    pub(crate) index_count: u32,
    /// Owned index buffer for wireframe edges (deduplicated line-list pairs).
    pub(crate) edge_index_buffer: crate::gpu::Buffer,
    /// Number of edge indices to draw (wireframe mode).
    pub(crate) edge_index_count: u32,
    /// Whether this item should be drawn in wireframe mode.
    pub(crate) wireframe: bool,
    /// Bind group (group 1): tube uniform (colour, radius).
    pub(crate) uniform_bind_group: crate::gpu::BindGroup,
    /// Blend mode for the draw. Streamtubes always set this to
    /// `SpriteBlend::AlphaBlend`; ribbons honour the value from `RibbonItem`.
    pub(crate) blend: crate::renderer::SpriteBlend,
    /// Object pick id for GPU picking, set by the prepare loop from the source
    /// item's `settings.pick_id`. `PickId::NONE` (0) when the item is not pickable.
    pub(crate) pick_id: crate::renderer::PickId,
    /// Model matrix the streamtube shader applies to `vertex_buffer`. The pick
    /// pass uses the same matrix so its silhouette matches the rendered tube.
    pub(crate) model: [[f32; 4]; 4],
    /// Set from the source item's `settings.cast_shadows`, overwritten by the
    /// prepare loop alongside `pick_id`/`model`. Only Ribbon items currently
    /// read this in the shadow pass (see `shadow_pass.rs`'s ribbon caster
    /// loop); Streamtube/Tube data carries the same field for consistency but
    /// has no shadow-cast loop of its own yet.
    pub(crate) cast_shadows: bool,
    /// True when this batch qualifies for true (weighted-blended) OIT
    /// instead of ordinary alpha blending: `blend` is `AlphaBlend` or
    /// `Premultiplied` (not `Additive`) and the item is not drawn wireframe.
    /// Only ever `true` for Ribbon (`upload_ribbon_per_frame` computes it
    /// from `RibbonItem::blend`); Streamtube/Tube data always carries
    /// `false` since neither has an OIT pipeline of its own yet. Read by the
    /// HDR path to route the batch through `oit_pass` instead of the
    /// ordinary ribbon draw.
    pub(crate) oit_eligible: bool,
    /// Whether this item writes depth. Ribbons select a pipeline variant by it;
    /// tubes and streamtubes are always opaque and set it true.
    pub(crate) depth_write: bool,
    /// Per-triangle segment index, one entry per triangle in `index_buffer`
    /// (i.e. `index_count / 3` entries). Maps a GPU pick's `primitive_index`
    /// (the hit triangle) to the source curve segment, so a sub-object GPU pick
    /// resolves `SubObjectRef::Segment`. End/start cap triangles map to the last
    /// / first segment of their strip. Empty when the item is not pickable.
    pub(crate) tri_segment: Vec<u32>,
    /// Per-triangle strip index, parallel to [`tri_segment`](Self::tri_segment),
    /// used to resolve `SubObjectRef::Strip`. Empty when not pickable.
    pub(crate) tri_strip: Vec<u32>,
    /// Per-triangle node payload for the POLY_NODE pick pipeline: for each
    /// triangle (parallel to `tri_segment`), the local positions and global node
    /// indices of its segment's two endpoints. The pick fragment reads it by
    /// `primitive_index` and writes the endpoint nearer the hit. `None` when
    /// there is no geometry to pick.
    pub(crate) node_pick_buffer: Option<crate::gpu::Buffer>,
    // Keep uniform buffer alive.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
}

/// One triangle's segment-endpoint payload for the curve POLY_NODE pick shader.
/// Layout matches the WGSL `NodePair` struct (32 bytes): a `vec3` position keeps
/// 16-byte alignment, so each index sits in the padding slot after its position.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NodePairRaw {
    pub(crate) p0: [f32; 3],
    pub(crate) i0: u32,
    pub(crate) p1: [f32; 3],
    pub(crate) i1: u32,
}

/// Global node indices of a segment's two endpoints, accounting for multi-strip
/// layout (each strip of `slen` nodes owns `slen - 1` segments). Mirrors the
/// picking helper `segment_node_indices`, replicated here to keep the upload path
/// independent of the picking module.
fn node_endpoints(seg: u32, strip_lengths: &[u32], n_positions: usize) -> (usize, usize) {
    if strip_lengths.is_empty() {
        let a = seg as usize;
        if a + 1 < n_positions {
            return (a, a + 1);
        }
        return (0, 0);
    }
    let mut node_off = 0usize;
    let mut seg_off = 0u32;
    for &slen in strip_lengths {
        let slen = slen as usize;
        let segs = slen.saturating_sub(1) as u32;
        if seg < seg_off + segs {
            let k = (seg - seg_off) as usize;
            let a = node_off + k;
            if a + 1 < n_positions {
                return (a, a + 1);
            }
            return (0, 0);
        }
        seg_off += segs;
        node_off += slen;
    }
    (0, 0)
}

/// Build the per-triangle node payload buffer for the POLY_NODE pick pipeline.
/// One entry per triangle, ordered to match `tri_segment`. `None` when the item
/// has no triangles or control points.
fn build_node_pick_buffer(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    tri_segment: &[u32],
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
) -> Option<crate::gpu::Buffer> {
    if tri_segment.is_empty() || positions.is_empty() {
        return None;
    }
    let payload: Vec<NodePairRaw> = tri_segment
        .iter()
        .map(|&seg| {
            let (a, b) = node_endpoints(seg, strip_lengths, positions.len());
            NodePairRaw {
                p0: positions[a],
                i0: a as u32,
                p1: positions[b],
                i1: b as u32,
            }
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&payload);
    let buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("streamtube_node_pick_buffer"),
        size: bytes.len() as u64,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    Some(buffer)
}
