//! The curve payloads these item types hold on the consumer's behalf, and the
//! per-frame GPU data every streamtube, tube and ribbon draw is built from.
//!
//! All three produce the same [`StreamtubeGpuData`], so the three builders sit
//! together and the three stores are three handle spaces over one payload. An
//! inline item is rebuilt each frame; a `*RefItem` names a curve uploaded once
//! through the matching methods on
//! [`ViewportRenderer`](crate::renderer::ViewportRenderer).
//!
//! The two bind group layouts live here rather than with the pipelines, because
//! an upload builds its bind group against them and an upload can arrive long
//! before the first frame that draws one.

use crate::resources::{DeviceResources, Vertex};

pub(crate) use super::types::{RibbonId, StreamtubeId, TubeId};

/// The streamtube and tube bind group layout. Uploads build their bind groups
/// against it, so it lives here with the stores rather than with the item
/// types' pipelines, and is created up front: it is a layout, not a compiled
/// pipeline. The streamtube and tube item types own their own render, pick and
/// mask pipelines.
pub(super) struct StreamtubeResources {
    /// Bind group layout for streamtube and tube uniforms (group 1).
    pub(super) bgl: crate::gpu::BindGroupLayout,
}

impl StreamtubeResources {
    pub(super) fn new(device: &crate::gpu::Device) -> Self {
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
pub(super) struct RibbonResources {
    /// Bind group layout for ribbons (group 1): uniform + optional streak
    /// texture + sampler.
    pub(super) bgl: crate::gpu::BindGroupLayout,
}

impl RibbonResources {
    pub(super) fn new(device: &crate::gpu::Device) -> Self {
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

/// The renderer-owned handles one streamtube or tube upload binds, resolved
/// from `DeviceResources` before the geometry is built.
///
/// wgpu layouts are cheap clonable handles and a LUT is 1 KiB of plain data, so
/// resolving them up front is what lets the geometry work run on a worker
/// thread, where no `DeviceResources` borrow exists.
#[derive(Clone)]
pub(super) struct TubeBindings {
    bgl: crate::gpu::BindGroupLayout,
    /// The item's colourmap as 256 RGBA8 entries, defaulted to mid-grey when
    /// it names none and none is registered.
    lut: [[u8; 4]; 256],
}

/// The same for a ribbon, which also binds an optional streak texture.
#[derive(Clone)]
pub(super) struct RibbonBindings {
    bgl: crate::gpu::BindGroupLayout,
    lut: [[u8; 4]; 256],
    texture_view: crate::gpu::TextureView,
    has_texture: u32,
    sampler: crate::gpu::Sampler,
}

/// Resolve the CPU-side colourmap an item names. Viridis when it names none,
/// mid-grey when nothing is registered yet.
fn resolve_lut(
    resources: &DeviceResources,
    colourmap_id: Option<crate::resources::ColourmapId>,
) -> [[u8; 4]; 256] {
    let id = colourmap_id.unwrap_or_else(|| {
        resources.builtin_colourmap_id(crate::resources::BuiltinColourmap::Viridis)
    });
    resources
        .get_colourmap_rgba(id)
        .copied()
        .unwrap_or([[128u8; 4]; 256])
}

/// Resolve what a streamtube or tube upload needs from the renderer.
pub(super) fn resolve_tube_bindings(
    resources: &DeviceResources,
    layouts: &StreamtubeResources,
    colourmap_id: Option<crate::resources::ColourmapId>,
) -> TubeBindings {
    TubeBindings {
        bgl: layouts.bgl.clone(),
        lut: resolve_lut(resources, colourmap_id),
    }
}

/// Resolve what a ribbon upload needs from the renderer, reporting a streak
/// texture uploaded in the wrong colour space through the lib's slot check.
pub(super) fn resolve_ribbon_bindings(
    resources: &DeviceResources,
    layouts: &RibbonResources,
    item: &crate::renderer::RibbonItem,
) -> RibbonBindings {
    resources.check_texture_slot(item.texture_id, crate::resources::TextureSlot::RibbonAlbedo);
    RibbonBindings {
        lut: resolve_lut(resources, item.colourmap_id),
        ..resolve_ribbon_texture(resources, layouts, item.texture_id)
    }
}

/// The texture half of [`resolve_ribbon_bindings`], against an id rather than
/// an item, so a stored ribbon can be resolved again after the texture it names
/// is freed or swapped, where the item that built it is long gone.
///
/// The colour-space check is not repeated here: it was reported when the ribbon
/// was uploaded, and repeating it on every revalidation would spam the log. The
/// LUT is left at its default, because the caller either overrides it (the
/// upload above) or is rebinding, where the LUT is already baked into the
/// vertex colours and cannot change.
pub(super) fn resolve_ribbon_texture(
    resources: &DeviceResources,
    layouts: &RibbonResources,
    texture_id: Option<crate::resources::TextureId>,
) -> RibbonBindings {
    let (texture_view, has_texture) = match texture_id.and_then(|id| resources.texture_view(id)) {
        Some(view) => (view.clone(), 1),
        None => (resources.fallback_colourmap_view().clone(), 0),
    };
    RibbonBindings {
        bgl: layouts.bgl.clone(),
        lut: [[128u8; 4]; 256],
        texture_view,
        has_texture,
        sampler: resources.material_sampler().clone(),
    }
}

/// Point a stored ribbon's bind group at a freshly resolved streak texture,
/// keeping its geometry and its handle.
///
/// A ribbon is uploaded once and drawn for as long as the host holds the
/// handle, so the view it bound at upload can be freed or swapped underneath
/// it. Rebuilding is the only correct answer: dropping the ribbon would lose
/// content the host still owns, and leaving it alone would keep a freed texture
/// alive and keep sampling it.
pub(super) fn rebind_ribbon(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binds: &RibbonBindings,
    data: &mut StreamtubeGpuData,
) {
    let Some(rebind) = data.rebind.as_mut() else {
        return;
    };
    rebind.uniform.has_texture = binds.has_texture;
    queue.write_buffer(&data._uniform_buf, 0, bytemuck::bytes_of(&rebind.uniform));
    data.uniform_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some("ribbon_uniform_bg"),
        layout: &binds.bgl,
        entries: &[
            crate::gpu::BindGroupEntry {
                binding: 0,
                resource: data._uniform_buf.as_entire_binding(),
            },
            crate::gpu::BindGroupEntry {
                binding: 1,
                resource: crate::gpu::BindingResource::TextureView(&binds.texture_view),
            },
            crate::gpu::BindGroupEntry {
                binding: 2,
                resource: crate::gpu::BindingResource::Sampler(&binds.sampler),
            },
        ],
    });
}

/// Uniform block for one ribbon: transform, colour and the flags the fragment
/// shader keys off. Layout mirrors `RibbonUniform` in `ribbon.wgsl`.
///
/// Kept on the batch rather than written and forgotten, because `has_texture`
/// depends on a streak texture that can be freed after the upload: a
/// revalidation rewrites it without needing the item back.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RibbonUniform {
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

/// What a stored ribbon needs to rebind itself after the streak texture it
/// named is freed or swapped.
///
/// Only a ribbon that names one carries this: a tube or a streamtube binds no
/// host texture, and neither does a ribbon that draws untextured, so for those
/// there is nothing a free or a replace could invalidate.
#[derive(Clone)]
pub(crate) struct RibbonRebind {
    /// The streak texture the batch's bind group holds.
    pub(super) texture_id: crate::resources::TextureId,
    /// The uniform block as written, so the rebind can clear `has_texture`.
    pub(super) uniform: RibbonUniform,
}

/// Upload one [`StreamtubeItem`] to the GPU and return draw data.
///
/// Generates a connected tube mesh CPU-side using a parallel-transport frame along
/// each polyline strip, then uploads the result as a single owned vertex+index buffer.
/// Adjacent rings are joined by quads (2 triangles each) giving a smooth, seamless tube
/// without the z-fighting or inter-segment gaps that plagued the old instanced approach.
pub(super) fn build_streamtube(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binds: &TubeBindings,
    item: &crate::renderer::StreamtubeItem,
    wireframe: bool,
) -> StreamtubeGpuData {
    {
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

        let bgl = &binds.bgl;
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
            rebind: None,
            _uniform_buf: uniform_buf,
        }
    }
}

/// Upload one [`TubeItem`](crate::renderer::TubeItem) to the GPU and return
/// draw data.
///
/// Generates a connected tube mesh CPU-side using a parallel-transport frame.
/// Scalar values are baked into per-vertex colours using the CPU-side colourmap
/// copy. Uses the same streamtube pipeline; sets `use_vertex_colour = 1` when
/// scalars are present.
pub(super) fn build_tube(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binds: &TubeBindings,
    item: &crate::renderer::TubeItem,
    wireframe: bool,
) -> StreamtubeGpuData {
    {
        let sides = (item.sides.max(3)) as usize;

        // Resolve scalar-to-colour mapping upfront if scalars are provided.
        let (use_vertex_colour, lut_rgba): (u32, Option<[[u8; 4]; 256]>) =
            if !item.scalars.is_empty() {
                (1, Some(binds.lut))
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

        let bgl = &binds.bgl;
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
            rebind: None,
            _uniform_buf: uniform_buf,
        }
    }
}

/// Build the GPU data for one [`RibbonItem`](crate::renderer::RibbonItem).
///
/// Each strip is swept as a flat quad surface. Two vertices are generated per
/// point (left and right edges), connected as a triangle strip. The normal is
/// the cross product of the tangent and the lateral direction `u`.
pub(super) fn build_ribbon(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binds: &RibbonBindings,
    item: &crate::renderer::RibbonItem,
    wireframe: bool,
) -> StreamtubeGpuData {
    {
        // Per-vertex RGBA (`colour_attribute`) takes precedence over the
        // scalar+LUT path and the flat `colour` fallback. Trails typically
        // drive only the alpha channel to fade along their length.
        let has_colour_attribute = !item.colour_attribute.is_empty();

        // Resolve LUT for scalar colouring.
        let (use_vertex_colour, lut_rgba): (u32, Option<[[u8; 4]; 256]>) = if has_colour_attribute {
            (1, None)
        } else if !item.scalars.is_empty() {
            (1, Some(binds.lut))
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

        let (texture_view, has_texture): (&crate::gpu::TextureView, u32) =
            (&binds.texture_view, binds.has_texture);
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

        let bgl = &binds.bgl;
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
                    resource: crate::gpu::BindingResource::Sampler(&binds.sampler),
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
            rebind: item.texture_id.map(|texture_id| RibbonRebind {
                texture_id,
                uniform: uniform_data,
            }),
            _uniform_buf: uniform_buf,
        }
    }
}

/// GPU data for one curve draw, shared by all three types: the inline items
/// build it each frame, the three stores hold it across frames.
///
/// The connected mesh (vertices + indices) is generated CPU-side for the entire
/// item (all strips) and uploaded as a single owned buffer pair.
#[derive(Clone)]
pub(crate) struct StreamtubeGpuData {
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
    /// What this batch needs to rebind itself when the streak texture in its
    /// bind group is freed or swapped. `None` for a tube, a streamtube, or an
    /// untextured ribbon: none of them binds a texture the host can free.
    pub(crate) rebind: Option<RibbonRebind>,
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

// ---------------------------------------------------------------------------
// The three stores
// ---------------------------------------------------------------------------

/// Slotted store of pre-uploaded streamtubes.
pub(super) type StreamtubeStore =
    crate::resources::handle::SlotStore<StreamtubeGpuData, StreamtubeId>;
/// Slotted store of pre-uploaded tubes. Same payload, separate handle space.
pub(super) type TubeStore = crate::resources::handle::SlotStore<StreamtubeGpuData, TubeId>;
/// Slotted store of pre-uploaded ribbons. Same payload, separate handle space.
pub(super) type RibbonStore = crate::resources::handle::SlotStore<StreamtubeGpuData, RibbonId>;

impl crate::resources::handle::GpuByteSize for StreamtubeGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size()
            + self.index_buffer.size()
            + self.edge_index_buffer.size()
            + self._uniform_buf.size()
            + self.node_pick_buffer.as_ref().map_or(0, |b| b.size())
    }
}
