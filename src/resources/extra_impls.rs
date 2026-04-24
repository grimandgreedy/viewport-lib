use super::*;

pub(super) fn generate_edge_indices(triangle_indices: &[u32]) -> Vec<u32> {
    use std::collections::HashSet;
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    let mut result = Vec::new();

    for tri in triangle_indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let pairs = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
        for (a, b) in &pairs {
            // Canonical form: smaller index first, so (a,b) and (b,a) map to the same edge.
            let edge = if a < b { (*a, *b) } else { (*b, *a) };
            if edges.insert(edge) {
                result.push(*a);
                result.push(*b);
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Procedural unit cube mesh (24 vertices, 4 per face, 36 indices)
// ---------------------------------------------------------------------------

/// Generate a unit cube centered at the origin.
///
/// 24 vertices (4 per face with shared normals), 36 indices (2 triangles per face).
/// All vertices are white [1,1,1,1].
pub(super) fn build_unit_cube() -> (Vec<Vertex>, Vec<u32>) {
    let white = [1.0f32, 1.0, 1.0, 1.0];
    let mut verts: Vec<Vertex> = Vec::with_capacity(24);
    let mut idx: Vec<u32> = Vec::with_capacity(36);

    // Helper: add a face quad (4 vertices in CCW order) and its 2 triangles.
    let mut add_face = |positions: [[f32; 3]; 4], normal: [f32; 3]| {
        let base = verts.len() as u32;
        for pos in &positions {
            verts.push(Vertex {
                position: *pos,
                normal,
                color: white,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
        // Two triangles: (base, base+1, base+2) and (base, base+2, base+3)
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // +X face (right), normal [1, 0, 0]
    add_face(
        [
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
        ],
        [1.0, 0.0, 0.0],
    );

    // -X face (left), normal [-1, 0, 0]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5],
        ],
        [-1.0, 0.0, 0.0],
    );

    // +Y face (top), normal [0, 1, 0]
    add_face(
        [
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
        ],
        [0.0, 1.0, 0.0],
    );

    // -Y face (bottom), normal [0, -1, 0]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
        ],
        [0.0, -1.0, 0.0],
    );

    // +Z face (front), normal [0, 0, 1]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        [0.0, 0.0, 1.0],
    );

    // -Z face (back), normal [0, 0, -1]
    add_face(
        [
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ],
        [0.0, 0.0, -1.0],
    );

    (verts, idx)
}

// ---------------------------------------------------------------------------
// Procedural glyph arrow mesh (cone tip + cylinder shaft, local +Y axis)
// ---------------------------------------------------------------------------

/// Generate a unit arrow mesh aligned to local +Y.
///
/// The arrow consists of:
/// - A cylinder shaft from Y=0 to Y=0.7, radius 0.05.
/// - A cone tip from Y=0.7 to Y=1.0, base radius 0.12.
///
/// 16 segments around the circumference gives ~300 vertices.
pub(super) fn build_glyph_arrow() -> (Vec<Vertex>, Vec<u32>) {
    let white = [1.0f32, 1.0, 1.0, 1.0];
    let segments = 16usize;
    let mut verts: Vec<Vertex> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();

    let shaft_r = 0.05f32;
    let shaft_bot = 0.0f32;
    let shaft_top = 0.7f32;
    let cone_r = 0.12f32;
    let cone_bot = shaft_top;
    let cone_tip = 1.0f32;

    // Helper: append ring vertices at a given Y and radius with outward normals.
    let ring_verts = |verts: &mut Vec<Vertex>, y: f32, r: f32, normal_y: f32| {
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
            let (s, c) = angle.sin_cos();
            let nx = if r > 0.0 { c } else { 0.0 };
            let nz = if r > 0.0 { s } else { 0.0 };
            let len = (nx * nx + normal_y * normal_y + nz * nz).sqrt();
            verts.push(Vertex {
                position: [c * r, y, s * r],
                normal: [nx / len, normal_y / len, nz / len],
                color: white,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
    };

    // --- Shaft ---
    // Bottom ring (face down for the cap).
    let shaft_bot_base = verts.len() as u32;
    ring_verts(&mut verts, shaft_bot, shaft_r, 0.0);

    // Bottom cap center.
    let shaft_bot_center = verts.len() as u32;
    verts.push(Vertex {
        position: [0.0, shaft_bot, 0.0],
        normal: [0.0, -1.0, 0.0],
        color: white,
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 1.0],
    });

    // Bottom cap triangles.
    for i in 0..segments {
        let a = shaft_bot_base + i as u32;
        let b = shaft_bot_base + ((i + 1) % segments) as u32;
        idx.extend_from_slice(&[shaft_bot_center, b, a]);
    }

    // Side quads: two rings of shaft.
    let shaft_top_ring_base = verts.len() as u32;
    ring_verts(&mut verts, shaft_bot, shaft_r, 0.0); // duplicate bottom ring for side normals
    let shaft_top_ring_top = verts.len() as u32;
    ring_verts(&mut verts, shaft_top, shaft_r, 0.0);
    for i in 0..segments {
        let a = shaft_top_ring_base + i as u32;
        let b = shaft_top_ring_base + ((i + 1) % segments) as u32;
        let c = shaft_top_ring_top + i as u32;
        let d = shaft_top_ring_top + ((i + 1) % segments) as u32;
        idx.extend_from_slice(&[a, b, d, a, d, c]);
    }

    // --- Cone ---
    // Slanted normal angle for cone surface: rise=(cone_tip-cone_bot), run=cone_r.
    let cone_len = ((cone_tip - cone_bot).powi(2) + cone_r * cone_r).sqrt();
    let normal_y_cone = cone_r / cone_len; // outward Y component of slanted normal
    let normal_r_cone = (cone_tip - cone_bot) / cone_len;

    let cone_base_ring = verts.len() as u32;
    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let (s, c) = angle.sin_cos();
        verts.push(Vertex {
            position: [c * cone_r, cone_bot, s * cone_r],
            normal: [c * normal_r_cone, normal_y_cone, s * normal_r_cone],
            color: white,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        });
    }

    // Cone tip vertex (normals averaged around tip — just point up).
    let cone_tip_v = verts.len() as u32;
    verts.push(Vertex {
        position: [0.0, cone_tip, 0.0],
        normal: [0.0, 1.0, 0.0],
        color: white,
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 1.0],
    });

    for i in 0..segments {
        let a = cone_base_ring + i as u32;
        let b = cone_base_ring + ((i + 1) % segments) as u32;
        idx.extend_from_slice(&[a, b, cone_tip_v]);
    }

    // Cone base cap (flat, faces -Y).
    let cone_cap_base = verts.len() as u32;
    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let (s, c) = angle.sin_cos();
        verts.push(Vertex {
            position: [c * cone_r, cone_bot, s * cone_r],
            normal: [0.0, -1.0, 0.0],
            color: white,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        });
    }
    let cone_cap_center = verts.len() as u32;
    verts.push(Vertex {
        position: [0.0, cone_bot, 0.0],
        normal: [0.0, -1.0, 0.0],
        color: white,
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 1.0],
    });
    for i in 0..segments {
        let a = cone_cap_base + i as u32;
        let b = cone_cap_base + ((i + 1) % segments) as u32;
        idx.extend_from_slice(&[cone_cap_center, a, b]);
    }

    (verts, idx)
}

// ---------------------------------------------------------------------------
// Procedural icosphere (2 subdivisions, ~240 triangles)
// ---------------------------------------------------------------------------

/// Generate a unit sphere as an icosphere with 2 subdivisions.
///
/// Starts from a regular icosahedron and subdivides each triangle 2×.
pub(super) fn build_glyph_sphere() -> (Vec<Vertex>, Vec<u32>) {
    let white = [1.0f32, 1.0, 1.0, 1.0];

    // Icosahedron constants.
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;

    // 12 vertices of a regular icosahedron (not yet normalised).
    let raw_verts = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];

    let mut positions: Vec<[f32; 3]> = raw_verts
        .iter()
        .map(|v| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        })
        .collect();

    // 20 base triangles.
    let mut triangles: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    // Subdivide 2 times.
    for _ in 0..2 {
        let mut mid_cache: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let mut new_triangles: Vec<[usize; 3]> = Vec::with_capacity(triangles.len() * 4);

        let midpoint = |positions: &mut Vec<[f32; 3]>,
                        a: usize,
                        b: usize,
                        cache: &mut std::collections::HashMap<(usize, usize), usize>|
         -> usize {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = cache.get(&key) {
                return idx;
            }
            let pa = positions[a];
            let pb = positions[b];
            let mx = (pa[0] + pb[0]) * 0.5;
            let my = (pa[1] + pb[1]) * 0.5;
            let mz = (pa[2] + pb[2]) * 0.5;
            let l = (mx * mx + my * my + mz * mz).sqrt();
            let idx = positions.len();
            positions.push([mx / l, my / l, mz / l]);
            cache.insert(key, idx);
            idx
        };

        for tri in &triangles {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let ab = midpoint(&mut positions, a, b, &mut mid_cache);
            let bc = midpoint(&mut positions, b, c, &mut mid_cache);
            let ca = midpoint(&mut positions, c, a, &mut mid_cache);
            new_triangles.push([a, ab, ca]);
            new_triangles.push([b, bc, ab]);
            new_triangles.push([c, ca, bc]);
            new_triangles.push([ab, bc, ca]);
        }
        triangles = new_triangles;
    }

    let verts: Vec<Vertex> = positions
        .iter()
        .map(|&p| Vertex {
            position: p,
            normal: p, // unit sphere: position = normal
            color: white,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        })
        .collect();

    let idx: Vec<u32> = triangles
        .iter()
        .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
        .collect();

    (verts, idx)
}

// ---------------------------------------------------------------------------
// Attribute interpolation utilities
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase G — in-place attribute hot-swap
// ---------------------------------------------------------------------------

impl ViewportGpuResources {
    /// Write new scalar data into an existing attribute buffer in-place.
    ///
    /// No GPU buffer reallocation, no mesh re-upload, no bind group rebuild is
    /// required. The attribute bind group *will* be rebuilt on the next
    /// `prepare()` call if the scalar range changes (tracked via `last_tex_key`).
    ///
    /// # Errors
    ///
    /// - [`ViewportError::MeshSlotEmpty`](crate::error::ViewportError::MeshSlotEmpty) — `mesh_id` not found in the store.
    /// - [`ViewportError::AttributeNotFound`](crate::error::ViewportError::AttributeNotFound) — `name` not present on the mesh.
    /// - [`ViewportError::AttributeLengthMismatch`](crate::error::ViewportError::AttributeLengthMismatch) — `data.len()` differs from
    ///   the original upload (same-topology requirement).
    pub fn replace_attribute(
        &mut self,
        queue: &wgpu::Queue,
        mesh_id: crate::resources::mesh_store::MeshId,
        name: &str,
        data: &[f32],
    ) -> crate::error::ViewportResult<()> {
        // Resolve the mesh.
        let gpu_mesh =
            self.mesh_store
                .get_mut(mesh_id)
                .ok_or(crate::error::ViewportError::MeshSlotEmpty {
                    index: mesh_id.index(),
                })?;

        // Find the existing attribute buffer.
        let buffer = gpu_mesh.attribute_buffers.get(name).ok_or_else(|| {
            crate::error::ViewportError::AttributeNotFound {
                mesh_id: mesh_id.index(),
                name: name.to_string(),
            }
        })?;

        // Validate same topology (buffer size must match).
        let expected_elems = (buffer.size() / 4) as usize;
        if data.len() != expected_elems {
            return Err(crate::error::ViewportError::AttributeLengthMismatch {
                expected: expected_elems,
                got: data.len(),
            });
        }

        // Zero-copy in-place write via the wgpu staging belt.
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));

        // Recompute scalar range so LUT mapping stays accurate.
        let (min, max) = data
            .iter()
            .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        let range = if min > max { (0.0, 1.0) } else { (min, max) };
        gpu_mesh.attribute_ranges.insert(name.to_string(), range);

        // Force bind group rebuild on next prepare() by invalidating the key.
        gpu_mesh.last_tex_key = (
            gpu_mesh.last_tex_key.0,
            gpu_mesh.last_tex_key.1,
            gpu_mesh.last_tex_key.2,
            gpu_mesh.last_tex_key.3,
            u64::MAX, // attribute hash component
            gpu_mesh.last_tex_key.5,
        );

        Ok(())
    }

    /// Create a camera bind group (group 0) for the given per-viewport buffers.
    ///
    /// Per-viewport buffers (camera, clip planes, shadow info, clip volume) are
    /// passed explicitly. Scene-global resources (lights, shadow atlas, IBL) come
    /// from shared resources on `self`.
    ///
    /// NOTE: The initial bind group in `init.rs` is constructed inline (before
    /// `Self` exists). Keep the binding layout in sync when modifying either site.
    pub(crate) fn create_camera_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        clip_planes_buf: &wgpu::Buffer,
        shadow_info_buf: &wgpu::Buffer,
        clip_volume_buf: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        let irr = self
            .ibl_irradiance_view
            .as_ref()
            .unwrap_or(&self.ibl_fallback_view);
        let spec = self
            .ibl_prefiltered_view
            .as_ref()
            .unwrap_or(&self.ibl_fallback_view);
        let brdf = self
            .ibl_brdf_lut_view
            .as_ref()
            .unwrap_or(&self.ibl_fallback_brdf_view);
        let skybox = self
            .ibl_skybox_view
            .as_ref()
            .unwrap_or(&self.ibl_fallback_view);

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.light_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: clip_planes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: clip_volume_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(irr),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(spec),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(brdf),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.ibl_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(skybox),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// Phase G — GPU compute filter pipeline and dispatch
// ---------------------------------------------------------------------------

/// Output from a single GPU compute filter dispatch.
///
/// Contains a compacted index buffer (triangles that passed the filter)
/// and the count of valid indices. The renderer swaps this in during draw.
pub struct ComputeFilterResult {
    /// Output index buffer containing only passing triangles.
    pub index_buffer: wgpu::Buffer,
    /// Number of valid indices in `index_buffer` (may be 0 if all filtered).
    pub index_count: u32,
    /// Mesh index this result corresponds to.
    pub mesh_index: usize,
}

impl ViewportGpuResources {
    /// Lazily create the GPU compute filter pipeline on first use.
    fn ensure_compute_filter_pipeline(&mut self, device: &wgpu::Device) {
        if self.compute_filter_pipeline.is_some() {
            return;
        }

        // Build bind group layout.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_filter_bgl"),
            entries: &[
                // binding 0: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: vertices (f32 storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: source indices (u32 storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: scalars (f32 storage, read) — dummy for Clip
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: output compacted indices (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: atomic counter (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_filter_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_filter_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compute_filter.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_filter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.compute_filter_bgl = Some(bgl);
        self.compute_filter_pipeline = Some(pipeline);
    }

    // -----------------------------------------------------------------------
    // Phase J: OIT (order-independent transparency) resource management
    // -----------------------------------------------------------------------

    /// Ensure OIT accum/reveal textures, pipelines, and composite bind group exist
    /// for the given viewport size.  Call once per frame before the OIT pass.
    ///
    /// Early-returns immediately if the size is unchanged and all resources are present.
    #[allow(dead_code)]
    pub(crate) fn ensure_oit_targets(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);

        // Only recreate textures and the composite bind group when size changes.
        let need_textures = self.oit_size != [w, h] || self.oit_accum_texture.is_none();

        if need_textures {
            self.oit_size = [w, h];

            // Accum texture: Rgba16Float for accumulation of weighted color+alpha.
            let accum_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("oit_accum_texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

            // Reveal texture: R8Unorm for transmittance accumulation.
            let reveal_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("oit_reveal_texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let reveal_view = reveal_tex.create_view(&wgpu::TextureViewDescriptor::default());

            // Create or reuse the OIT sampler.
            let sampler = if self.oit_composite_sampler.is_none() {
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("oit_composite_sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                })
            } else {
                // We can't move out of self here, so create a new one.
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("oit_composite_sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                })
            };

            // Create BGL once.
            let bgl = if self.oit_composite_bgl.is_none() {
                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("oit_composite_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
                self.oit_composite_bgl = Some(bgl);
                self.oit_composite_bgl.as_ref().unwrap()
            } else {
                self.oit_composite_bgl.as_ref().unwrap()
            };

            // Composite bind group referencing the new texture views.
            let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oit_composite_bind_group"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&accum_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&reveal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            self.oit_accum_texture = Some(accum_tex);
            self.oit_accum_view = Some(accum_view);
            self.oit_reveal_texture = Some(reveal_tex);
            self.oit_reveal_view = Some(reveal_view);
            self.oit_composite_sampler = Some(sampler);
            self.oit_composite_bind_group = Some(composite_bg);
        }

        // Create pipelines once (they don't depend on viewport size).
        if self.oit_pipeline.is_none() {
            // Non-instanced OIT pipeline (mesh_oit.wgsl, group 0 = camera BGL, group 1 = object BGL).
            let oit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh_oit_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh_oit.wgsl").into()),
            });
            let oit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("oit_pipeline_layout"),
                bind_group_layouts: &[
                    &self.camera_bind_group_layout,
                    &self.object_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            // Accum blend: src=One, dst=One, Add (additive accumulation).
            let accum_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            // Reveal blend: src=Zero, dst=OneMinusSrcColor (multiplicative transmittance).
            let reveal_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            let oit_depth_stencil = wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            };

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("oit_pipeline"),
                layout: Some(&oit_layout),
                vertex: wgpu::VertexState {
                    module: &oit_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &oit_shader,
                    entry_point: Some("fs_oit_main"),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(accum_blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R8Unorm,
                            blend: Some(reveal_blend),
                            write_mask: wgpu::ColorWrites::RED,
                        }),
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(oit_depth_stencil.clone()),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });
            self.oit_pipeline = Some(pipeline);

            // Instanced OIT pipeline (mesh_instanced_oit.wgsl, two OIT targets).
            if let Some(ref instance_bgl) = self.instance_bind_group_layout {
                let instanced_oit_shader =
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("mesh_instanced_oit_shader"),
                        source: wgpu::ShaderSource::Wgsl(
                            include_str!("../shaders/mesh_instanced_oit.wgsl").into(),
                        ),
                    });
                let instanced_oit_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("oit_instanced_pipeline_layout"),
                        bind_group_layouts: &[&self.camera_bind_group_layout, instance_bgl],
                        push_constant_ranges: &[],
                    });
                let instanced_pipeline =
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("oit_instanced_pipeline"),
                        layout: Some(&instanced_oit_layout),
                        vertex: wgpu::VertexState {
                            module: &instanced_oit_shader,
                            entry_point: Some("vs_main"),
                            buffers: &[Vertex::buffer_layout()],
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &instanced_oit_shader,
                            entry_point: Some("fs_oit_main"),
                            targets: &[
                                Some(wgpu::ColorTargetState {
                                    format: wgpu::TextureFormat::Rgba16Float,
                                    blend: Some(accum_blend),
                                    write_mask: wgpu::ColorWrites::ALL,
                                }),
                                Some(wgpu::ColorTargetState {
                                    format: wgpu::TextureFormat::R8Unorm,
                                    blend: Some(reveal_blend),
                                    write_mask: wgpu::ColorWrites::RED,
                                }),
                            ],
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            cull_mode: Some(wgpu::Face::Back),
                            ..Default::default()
                        },
                        depth_stencil: Some(oit_depth_stencil),
                        multisample: wgpu::MultisampleState {
                            count: 1,
                            ..Default::default()
                        },
                        multiview: None,
                        cache: None,
                    });
                self.oit_instanced_pipeline = Some(instanced_pipeline);
            }
        }

        if self.oit_composite_pipeline.is_none() {
            let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oit_composite_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/oit_composite.wgsl").into(),
                ),
            });
            let bgl = self
                .oit_composite_bgl
                .as_ref()
                .expect("oit_composite_bgl must exist");
            let comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("oit_composite_pipeline_layout"),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            // Premultiplied alpha blend: One / OneMinusSrcAlpha — composites avg_color*(1-r) onto HDR.
            let premul_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };
            let comp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("oit_composite_pipeline"),
                layout: Some(&comp_layout),
                vertex: wgpu::VertexState {
                    module: &comp_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &comp_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });
            self.oit_composite_pipeline = Some(comp_pipeline);
        }
    }

    /// Dispatch GPU compute filters for all items in the list.
    ///
    /// Returns one [`ComputeFilterResult`] per item. The renderer uses these
    /// during `paint()` to override the mesh's default index buffer.
    ///
    /// This is a synchronous v1 implementation: it submits each dispatch
    /// individually and polls the device to read back the counter. This is
    /// acceptable for v1; async readback can be added later.
    pub fn run_compute_filters(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        items: &[crate::renderer::ComputeFilterItem],
    ) -> Vec<ComputeFilterResult> {
        if items.is_empty() {
            return Vec::new();
        }

        self.ensure_compute_filter_pipeline(device);

        // Dummy 4-byte buffer used as the scalar binding when doing a Clip filter.
        let dummy_scalar_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compute_filter_dummy_scalar"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut results = Vec::with_capacity(items.len());

        for item in items {
            // Resolve the mesh.
            let gpu_mesh = match self
                .mesh_store
                .get(crate::resources::mesh_store::MeshId(item.mesh_index))
            {
                Some(m) => m,
                None => continue,
            };

            let triangle_count = gpu_mesh.index_count / 3;
            if triangle_count == 0 {
                continue;
            }

            // Vertex stride: the Vertex struct is 64 bytes = 16 f32s.
            const VERTEX_STRIDE_F32: u32 = 16;

            // Build params uniform matching compute_filter.wgsl Params struct layout.
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct FilterParams {
                mode: u32,
                clip_type: u32,
                threshold_min: f32,
                threshold_max: f32,
                triangle_count: u32,
                vertex_stride_f32: u32,
                _pad: [u32; 2],
                // Plane params
                plane_nx: f32,
                plane_ny: f32,
                plane_nz: f32,
                plane_dist: f32,
                // Box params
                box_cx: f32,
                box_cy: f32,
                box_cz: f32,
                _padb0: f32,
                box_hex: f32,
                box_hey: f32,
                box_hez: f32,
                _padb1: f32,
                box_col0x: f32,
                box_col0y: f32,
                box_col0z: f32,
                _padb2: f32,
                box_col1x: f32,
                box_col1y: f32,
                box_col1z: f32,
                _padb3: f32,
                box_col2x: f32,
                box_col2y: f32,
                box_col2z: f32,
                _padb4: f32,
                // Sphere params
                sphere_cx: f32,
                sphere_cy: f32,
                sphere_cz: f32,
                sphere_radius: f32,
            }

            let mut params: FilterParams = bytemuck::Zeroable::zeroed();
            params.triangle_count = triangle_count;
            params.vertex_stride_f32 = VERTEX_STRIDE_F32;

            match item.kind {
                crate::renderer::ComputeFilterKind::Clip {
                    plane_normal,
                    plane_dist,
                } => {
                    params.mode = 0;
                    params.clip_type = 1;
                    params.plane_nx = plane_normal[0];
                    params.plane_ny = plane_normal[1];
                    params.plane_nz = plane_normal[2];
                    params.plane_dist = plane_dist;
                }
                crate::renderer::ComputeFilterKind::ClipBox {
                    center,
                    half_extents,
                    orientation,
                } => {
                    params.mode = 0;
                    params.clip_type = 2;
                    params.box_cx = center[0];
                    params.box_cy = center[1];
                    params.box_cz = center[2];
                    params.box_hex = half_extents[0];
                    params.box_hey = half_extents[1];
                    params.box_hez = half_extents[2];
                    params.box_col0x = orientation[0][0];
                    params.box_col0y = orientation[0][1];
                    params.box_col0z = orientation[0][2];
                    params.box_col1x = orientation[1][0];
                    params.box_col1y = orientation[1][1];
                    params.box_col1z = orientation[1][2];
                    params.box_col2x = orientation[2][0];
                    params.box_col2y = orientation[2][1];
                    params.box_col2z = orientation[2][2];
                }
                crate::renderer::ComputeFilterKind::ClipSphere { center, radius } => {
                    params.mode = 0;
                    params.clip_type = 3;
                    params.sphere_cx = center[0];
                    params.sphere_cy = center[1];
                    params.sphere_cz = center[2];
                    params.sphere_radius = radius;
                }
                crate::renderer::ComputeFilterKind::Threshold { min, max } => {
                    params.mode = 1;
                    params.threshold_min = min;
                    params.threshold_max = max;
                }
            }

            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute_filter_params"),
                size: std::mem::size_of::<FilterParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

            // Output index buffer (worst-case: all triangles pass).
            let out_index_size = (gpu_mesh.index_count as u64) * 4;
            let out_index_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute_filter_out_indices"),
                size: out_index_size.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDEX
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // 4-byte atomic counter buffer (cleared to 0).
            let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute_filter_counter"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            {
                let mut view = counter_buf.slice(..).get_mapped_range_mut();
                view[0..4].copy_from_slice(&0u32.to_le_bytes());
            }
            counter_buf.unmap();

            // Staging buffer to read back the counter.
            let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute_filter_counter_staging"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Pick the scalar buffer: named attribute or dummy.
            let scalar_buf_ref: &wgpu::Buffer = match &item.kind {
                crate::renderer::ComputeFilterKind::Threshold { .. } => {
                    if let Some(attr_name) = &item.attribute_name {
                        gpu_mesh
                            .attribute_buffers
                            .get(attr_name.as_str())
                            .unwrap_or(&dummy_scalar_buf)
                    } else {
                        &dummy_scalar_buf
                    }
                }
                // Clip variants don't use the scalar buffer.
                _ => &dummy_scalar_buf,
            };

            // Build bind group.
            let bgl = self.compute_filter_bgl.as_ref().unwrap();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute_filter_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu_mesh.vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gpu_mesh.index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scalar_buf_ref.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_index_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: counter_buf.as_entire_binding(),
                    },
                ],
            });

            // Encode and submit compute + counter copy.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute_filter_encoder"),
            });

            {
                let pipeline = self.compute_filter_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compute_filter_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let workgroups = triangle_count.div_ceil(64);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_buf, 0, 4);
            queue.submit(std::iter::once(encoder.finish()));

            // Synchronous readback (v1 — acceptable; async readback can follow later).
            let slice = staging_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            });

            let index_count = {
                let data = slice.get_mapped_range();
                u32::from_le_bytes([data[0], data[1], data[2], data[3]])
            };
            staging_buf.unmap();

            results.push(ComputeFilterResult {
                index_buffer: out_index_buf,
                index_count,
                mesh_index: item.mesh_index,
            });
        }

        results
    }

    // -----------------------------------------------------------------------
    // Phase K: GPU object-ID picking pipeline (lazily created)
    // -----------------------------------------------------------------------

    /// Lazily create the GPU pick pipeline and associated bind group layouts.
    ///
    /// No-op if already created. Called from `ViewportRenderer::pick_scene_gpu`
    /// on first invocation — zero overhead when GPU picking is never used.
    pub(crate) fn ensure_pick_pipeline(&mut self, device: &wgpu::Device) {
        if self.pick_pipeline.is_some() {
            return;
        }

        // --- group 0: minimal camera-only bind group layout ---
        // The pick shader only uses binding 0 (CameraUniform); the full
        // camera_bind_group_layout has 6 bindings and would require binding a
        // compatible full bind group. A separate minimal layout is cleaner.
        let pick_camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick_camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // --- group 1: PickInstance storage buffer ---
        let pick_instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick_instance_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pick_id_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pick_id.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pick_pipeline_layout"),
            bind_group_layouts: &[&pick_camera_bgl, &pick_instance_bgl],
            push_constant_ranges: &[],
        });

        // Vertex layout: reuse the 64-byte Vertex stride but only declare position (location 0).
        let pick_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress, // 64 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[pick_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    // location 0: R32Uint object ID
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None, // replace — no blending for integer targets
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location 1: R32Float depth
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1, // pick pass is always 1x (no MSAA)
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.pick_camera_bgl = Some(pick_camera_bgl);
        self.pick_bind_group_layout_1 = Some(pick_instance_bgl);
        self.pick_pipeline = Some(pipeline);
    }
}

// ---------------------------------------------------------------------------
// Attribute interpolation utilities
// ---------------------------------------------------------------------------


/// Linearly interpolate between two attribute buffers element-wise.
///
/// Both slices must have the same length. `t` is clamped to `[0.0, 1.0]`.
/// Returns a new `Vec<f32>` with `a[i] * (1 - t) + b[i] * t`.
///
/// Use this to blend per-vertex scalar attributes between two consecutive
/// timesteps when scrubbing the timeline at sub-frame resolution.
pub fn lerp_attributes(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let t = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| av * one_minus_t + bv * t)
        .collect()
}
