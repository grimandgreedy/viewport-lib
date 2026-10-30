//! Per-viewport cache of section cap geometry.
//!
//! A cap depends on the mesh's geometry, the item's model matrix and the clip
//! plane, so it is regenerated only when one of those changes. A key not seen
//! in a frame is evicted at the end of it.

use crate::resources::fast_hash::FastMap;
use crate::resources::mesh::mesh_store::MeshId;
use crate::resources::overlay::section_cap::CapUniform;

/// What a cached cap was generated from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CapKey {
    mesh: MeshId,
    /// `GpuMesh::content_rev`, which moves when geometry changes in place.
    content_rev: u64,
    /// Bit hash of the item's model matrix.
    model: u64,
    /// The plane's normal and distance, as bits.
    plane: [u32; 4],
    /// The plane's index in the clip-plane uniform.
    plane_index: u32,
    /// Counts identical keys within one frame (two items with the same mesh
    /// and transform), so each keeps its own colour.
    occurrence: u32,
}

impl CapKey {
    pub(crate) fn new(
        mesh: MeshId,
        content_rev: u64,
        model: &[[f32; 4]; 4],
        normal: [f32; 3],
        distance: f32,
        plane_index: u32,
    ) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = crate::resources::fast_hash::FastHasher::default();
        for col in model {
            for v in col {
                v.to_bits().hash(&mut h);
            }
        }
        Self {
            mesh,
            content_rev,
            model: h.finish(),
            plane: [
                normal[0].to_bits(),
                normal[1].to_bits(),
                normal[2].to_bits(),
                distance.to_bits(),
            ],
            plane_index,
            occurrence: 0,
        }
    }
}

/// One cap's GPU buffers, or `None` when the plane missed the mesh's
/// surface, cached so a miss is not retried every frame.
struct CachedCap {
    gpu: Option<CapGpu>,
    seen: bool,
}

struct CapGpu {
    vbuf: crate::gpu::Buffer,
    ibuf: crate::gpu::Buffer,
    index_count: u32,
    ubuf: crate::gpu::Buffer,
    bind_group: crate::gpu::BindGroup,
    uniform: CapUniform,
}

/// The caps one viewport drew last frame, keyed by what they were built from.
#[derive(Default)]
pub(crate) struct SectionCapCache {
    entries: FastMap<CapKey, CachedCap>,
    /// Keys handed out this frame, for `occurrence`.
    issued: FastMap<CapKey, u32>,
}

impl SectionCapCache {
    /// Start a frame: every entry is unseen until asked for again.
    pub(crate) fn begin_frame(&mut self) {
        for e in self.entries.values_mut() {
            e.seen = false;
        }
        self.issued.clear();
    }

    /// The cap for `key`, built with `build` when the key is new. Returns the
    /// draw, if the cap has geometry, and whether `build` ran.
    pub(crate) fn cap(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        layout: &crate::gpu::BindGroupLayout,
        mut key: CapKey,
        colour: [f32; 4],
        build: impl FnOnce() -> Option<crate::geometry::mesh::cap::CapMesh>,
    ) -> (Option<super::overlay_buffers::OverlayGeometryDraw>, bool) {
        let n = self.issued.entry(key).or_insert(0);
        key.occurrence = *n;
        *n += 1;

        let mut built = false;
        let entry = self.entries.entry(key).or_insert_with(|| {
            built = true;
            CachedCap {
                gpu: build()
                    .map(|mesh| upload(device, queue, layout, &mesh, key.plane_index, colour)),
                seen: false,
            }
        });
        entry.seen = true;
        let Some(gpu) = entry.gpu.as_mut() else {
            return (None, built);
        };
        if gpu.uniform.colour != colour {
            gpu.uniform.colour = colour;
            queue.write_buffer(&gpu.ubuf, 0, bytemuck::bytes_of(&gpu.uniform));
        }
        (
            Some((
                gpu.vbuf.clone(),
                gpu.ibuf.clone(),
                gpu.index_count,
                gpu.ubuf.clone(),
                gpu.bind_group.clone(),
            )),
            built,
        )
    }

    /// End a frame: drop every entry not asked for since `begin_frame`.
    pub(crate) fn evict_unseen(&mut self) {
        self.entries.retain(|_, e| e.seen);
    }
}

fn upload(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    layout: &crate::gpu::BindGroupLayout,
    mesh: &crate::geometry::mesh::cap::CapMesh,
    own_plane: u32,
    colour: [f32; 4],
) -> CapGpu {
    let buffer = |label, usage, bytes: &[u8]| {
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: usage | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytes);
        buf
    };
    // The cap is generated in world space, so its model matrix is identity.
    let uniform = CapUniform {
        model: glam::Mat4::IDENTITY.to_cols_array_2d(),
        colour,
        own_plane,
        _pad: [0; 3],
    };
    let vbuf = buffer(
        "section_cap_vbuf",
        crate::gpu::BufferUsages::VERTEX,
        bytemuck::cast_slice(&mesh.positions),
    );
    let ibuf = buffer(
        "section_cap_ibuf",
        crate::gpu::BufferUsages::INDEX,
        bytemuck::cast_slice(&mesh.indices),
    );
    let ubuf = buffer(
        "section_cap_ubuf",
        crate::gpu::BufferUsages::UNIFORM,
        bytemuck::bytes_of(&uniform),
    );
    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some("section_cap_bg"),
        layout,
        entries: &[crate::gpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        }],
    });
    CapGpu {
        vbuf,
        ibuf,
        index_count: mesh.indices.len() as u32,
        ubuf,
        bind_group,
        uniform,
    }
}
