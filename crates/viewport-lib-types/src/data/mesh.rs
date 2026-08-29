//! Mesh submission payload: the CPU-side triangle mesh a consumer hands to the
//! renderer for upload.

use std::collections::HashMap;

use crate::data::attribute::AttributeData;

/// One contiguous run of the index buffer that draws with a single material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubmeshRange {
    /// First index in the mesh's index buffer.
    pub first_index: u32,
    /// Number of indices in the range (a multiple of 3 for triangle lists).
    pub index_count: u32,
}

/// Raw mesh data for upload to the GPU.
#[derive(Clone)]
#[non_exhaustive]
pub struct MeshData {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex normals (must be the same length as `positions`).
    pub normals: Vec<[f32; 3]>,
    /// Triangle index list (every 3 indices form one triangle).
    pub indices: Vec<u32>,
    /// Optional per-vertex UV coordinates. `None` means zero-fill [0.0, 0.0].
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Optional per-vertex tangents [tx, ty, tz, w] where w is handedness (+/-1.0).
    ///
    /// `None` = auto-compute from UVs if available, or zero-fill otherwise.
    /// Tangents are required for correct normal map rendering.
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Optional per-vertex RGBA colour in linear 0..1, multiplied into the base
    /// colour before lighting (same convention as glTF `COLOR_0`). `None` leaves
    /// vertices white [1, 1, 1, 1], a neutral multiply. Entries beyond the slice
    /// length also default to white, matching the forgiving `uvs`/`tangents` lookup.
    pub vertex_colours: Option<Vec<[f32; 4]>>,
    /// Named scalar attributes for per-vertex or per-cell scalar field visualisation.
    ///
    /// Keys are user-defined attribute names (e.g. `"pressure"`, `"velocity_mag"`).
    /// Cell attributes are averaged to vertices at upload time.
    pub attributes: HashMap<String, AttributeData>,
    /// Optional per-vertex `vec4<f32>` channel for material plugins.
    ///
    /// Uploaded to a per-mesh storage buffer and delivered to shading-hook
    /// bodies as `surf.attr` when the plugin sets `reads_vertex_attribute`
    /// (interpolated across the triangle, like any vertex attribute). The
    /// meaning of the four components is up to the plugin: blend masks,
    /// wind weights, bake data, etc.
    ///
    /// `None`, entries beyond the slice length, and meshes drawn without a
    /// reading plugin all resolve to `vec4(0.0)`.
    pub extension_attributes: Option<Vec<[f32; 4]>>,
    /// Optional material ranges partitioning the index buffer.
    ///
    /// Empty means the whole mesh draws with the item's single material,
    /// which is the behaviour for meshes that do not need ranges. When
    /// non-empty, an item can bind one material per range via
    /// `SceneRenderItem::submesh_materials`. See [`SubmeshRange`] for the
    /// index-sorting contract.
    pub submeshes: Vec<SubmeshRange>,
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            uvs: None,
            tangents: None,
            vertex_colours: None,
            attributes: HashMap::new(),
            extension_attributes: None,
            submeshes: Vec::new(),
        }
    }
}

impl MeshData {
    /// Build a mesh from its required geometry, leaving every optional channel
    /// (UVs, tangents, vertex colours, attributes, extension attributes,
    /// submeshes) empty.
    ///
    /// `MeshData` is `#[non_exhaustive]`, so it cannot be built with a struct
    /// literal from outside this crate; use this constructor and set the
    /// optional fields afterward.
    pub fn new(positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>, indices: Vec<u32>) -> Self {
        Self {
            positions,
            normals,
            indices,
            ..Self::default()
        }
    }

    /// Compute the local-space AABB from vertex positions.
    pub fn compute_aabb(&self) -> crate::scene::aabb::Aabb {
        crate::scene::aabb::Aabb::from_positions(&self.positions)
    }

    /// Sort triangles into contiguous per-material runs and fill `submeshes`.
    ///
    /// `triangle_materials[t]` is the material id of triangle `t` (the
    /// triangle at `indices[3t..3t+3]`). Triangles are stable-sorted by
    /// material id, the index buffer is rewritten in that order, and
    /// per-triangle attributes (`Cell`, `Face`, `FaceColour`, `Edge`,
    /// `Halfedge`, `Corner`) are permuted alongside so they keep addressing
    /// the same triangles. Per-vertex data is untouched.
    ///
    /// One range per distinct material id is written to `submeshes`, ordered
    /// by ascending id. Returns the distinct ids in that same order, so a
    /// caller with sparse or unordered ids can line up
    /// `SceneRenderItem::submesh_materials[i]` with the returned `ids[i]`.
    ///
    /// Importers that already emit per-material contiguous indices do not
    /// need this; it exists for meshes whose triangles arrive interleaved.
    ///
    /// # Errors
    ///
    /// [`ViewportError::SubmeshTriangleCountMismatch`](crate::error::ViewportError::SubmeshTriangleCountMismatch)
    /// if `triangle_materials.len() != indices.len() / 3` (or the index
    /// count is not a multiple of 3).
    pub fn sort_triangles_into_submeshes(
        &mut self,
        triangle_materials: &[u32],
    ) -> crate::error::ViewportResult<Vec<u32>> {
        let tri_count = self.indices.len() / 3;
        if self.indices.len() % 3 != 0 || triangle_materials.len() != tri_count {
            return Err(crate::error::ViewportError::SubmeshTriangleCountMismatch {
                triangles: tri_count,
                material_ids: triangle_materials.len(),
            });
        }

        let mut order: Vec<u32> = (0..tri_count as u32).collect();
        order.sort_by_key(|&t| triangle_materials[t as usize]);

        let old_indices = std::mem::take(&mut self.indices);
        self.indices = Vec::with_capacity(old_indices.len());
        for &t in &order {
            let base = t as usize * 3;
            self.indices.extend_from_slice(&old_indices[base..base + 3]);
        }

        // Per-triangle attribute data must follow its triangle to the new
        // position. Only permute channels whose length matches; a wrong
        // length is caught by upload validation, not silently reshuffled.
        fn permute<T: Copy>(values: &mut Vec<T>, order: &[u32], per_tri: usize) {
            if values.len() != order.len() * per_tri {
                return;
            }
            let old = std::mem::take(values);
            values.reserve(old.len());
            for &t in order {
                let base = t as usize * per_tri;
                values.extend_from_slice(&old[base..base + per_tri]);
            }
        }
        for data in self.attributes.values_mut() {
            match data {
                AttributeData::Cell(v) | AttributeData::Face(v) => permute(v, &order, 1),
                AttributeData::FaceColour(v) => permute(v, &order, 1),
                AttributeData::Edge(v) | AttributeData::Halfedge(v) | AttributeData::Corner(v) => {
                    permute(v, &order, 3)
                }
                AttributeData::Vertex(_) | AttributeData::VertexVector(_) => {}
            }
        }

        self.submeshes.clear();
        let mut ids = Vec::new();
        let mut run_start = 0usize;
        for i in 0..order.len() {
            let id = triangle_materials[order[i] as usize];
            let next_differs = order
                .get(i + 1)
                .map_or(true, |&t| triangle_materials[t as usize] != id);
            if next_differs {
                self.submeshes.push(SubmeshRange {
                    first_index: run_start as u32 * 3,
                    index_count: (i + 1 - run_start) as u32 * 3,
                });
                ids.push(id);
                run_start = i + 1;
            }
        }
        Ok(ids)
    }
}

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
