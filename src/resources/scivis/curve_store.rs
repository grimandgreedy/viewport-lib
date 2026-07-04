//! Slot-based storage for pre-uploaded curve GPU data.
//!
//! Each store holds GPU data produced by the per-frame upload helpers, keyed
//! by a typed handle. Per-frame ref items (`PolylineRefItem` and friends) name
//! a store entry by handle and supply per-frame overrides (model matrix, etc.)
//! instead of resubmitting the geometry every frame.

use crate::resources::{
    GlyphGpuData, PointCloudGpuData, PolylineGpuData, SpriteGpuData, StreamtubeGpuData,
    TensorGlyphGpuData,
};

/// Resident GPU bytes for one uploaded curve entry, summed by
/// [`CurveStore::allocated_bytes`] and surfaced through
/// [`ViewportGpuResources::resident_bytes`](crate::resources::ViewportGpuResources::resident_bytes).
///
/// Counts the buffers this entry owns. Base meshes shared across glyph and
/// tensor-glyph batches are borrowed (`&'static`) from a single cached copy, so
/// they belong to that cache, not to each entry, and are not counted here.
pub(crate) trait GpuByteSize {
    fn gpu_bytes(&self) -> u64;
}

impl GpuByteSize for PolylineGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size() + self._uniform_buf.size()
    }
}

impl GpuByteSize for StreamtubeGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size()
            + self.index_buffer.size()
            + self.edge_index_buffer.size()
            + self._uniform_buf.size()
    }
}

impl GpuByteSize for PointCloudGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size()
            + self._uniform_buf.size()
            + self._scalar_buf.size()
            + self._colour_buf.size()
            + self._radius_buf.size()
            + self._transparency_buf.size()
    }
}

impl GpuByteSize for GlyphGpuData {
    fn gpu_bytes(&self) -> u64 {
        self._uniform_buf.size() + self._instance_buf.size()
    }
}

impl GpuByteSize for TensorGlyphGpuData {
    fn gpu_bytes(&self) -> u64 {
        self._uniform_buf.size() + self._instance_buf.size()
    }
}

impl GpuByteSize for SpriteGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size() + self._uniform_buf.size() + self._instance_buf.size()
    }
}

/// One store slot: the data (when occupied) and the slot's current generation.
struct CurveSlot<T> {
    data: Option<T>,
    generation: u32,
}

macro_rules! curve_store {
    ($(#[$id_doc:meta])* $id:ident, $store:ident, $data:ty) => {
        crate::resources::handle::slot_handle! {
            $(#[$id_doc])*
            pub struct $id;
        }

        pub(crate) struct $store {
            slots: Vec<CurveSlot<$data>>,
            free_list: Vec<usize>,
        }

        impl $store {
            pub fn new() -> Self {
                Self { slots: Vec::new(), free_list: Vec::new() }
            }

            pub fn insert(&mut self, data: $data) -> $id {
                if let Some(idx) = self.free_list.pop() {
                    let slot = &mut self.slots[idx];
                    slot.data = Some(data);
                    $id::new(idx as u32, slot.generation)
                } else {
                    let idx = self.slots.len();
                    self.slots.push(CurveSlot { data: Some(data), generation: 0 });
                    $id::new(idx as u32, 0)
                }
            }

            pub fn get(&self, id: $id) -> Option<&$data> {
                let slot = self.slots.get(id.index())?;
                if slot.generation != id.generation {
                    return None;
                }
                slot.data.as_ref()
            }

            pub fn replace(&mut self, id: $id, data: $data) -> bool {
                match self.slots.get_mut(id.index()) {
                    Some(slot) if slot.generation == id.generation && slot.data.is_some() => {
                        slot.data = Some(data);
                        true
                    }
                    _ => false,
                }
            }

            pub fn remove(&mut self, id: $id) -> bool {
                if let Some(slot) = self.slots.get_mut(id.index()) {
                    if slot.generation == id.generation && slot.data.is_some() {
                        slot.data = None;
                        slot.generation = slot.generation.wrapping_add(1);
                        self.free_list.push(id.index());
                        return true;
                    }
                }
                false
            }

            pub fn contains(&self, id: $id) -> bool {
                self.get(id).is_some()
            }

            #[allow(dead_code)]
            pub fn len(&self) -> usize {
                self.slots.iter().filter(|s| s.data.is_some()).count()
            }

            /// Total resident GPU bytes across every live entry in this store.
            pub fn allocated_bytes(&self) -> u64 {
                self.slots
                    .iter()
                    .filter_map(|s| s.data.as_ref())
                    .map(GpuByteSize::gpu_bytes)
                    .sum()
            }
        }
    };
}

curve_store!(
    /// Handle to a pre-uploaded polyline produced by
    /// [`ViewportGpuResources::upload_polyline`](crate::resources::ViewportGpuResources::upload_polyline).
    PolylineId,
    PolylineStore,
    PolylineGpuData
);

curve_store!(
    /// Handle to a pre-uploaded streamtube produced by
    /// [`ViewportGpuResources::upload_streamtube`](crate::resources::ViewportGpuResources::upload_streamtube).
    StreamtubeId,
    StreamtubeStore,
    StreamtubeGpuData
);

curve_store!(
    /// Handle to a pre-uploaded tube produced by
    /// [`ViewportGpuResources::upload_tube`](crate::resources::ViewportGpuResources::upload_tube).
    TubeId,
    TubeStore,
    StreamtubeGpuData
);

curve_store!(
    /// Handle to a pre-uploaded ribbon produced by
    /// [`ViewportGpuResources::upload_ribbon`](crate::resources::ViewportGpuResources::upload_ribbon).
    RibbonId,
    RibbonStore,
    StreamtubeGpuData
);

curve_store!(
    /// Handle to a pre-uploaded point cloud produced by
    /// [`ViewportGpuResources::upload_point_cloud`](crate::resources::ViewportGpuResources::upload_point_cloud).
    PointCloudId,
    PointCloudStore,
    PointCloudGpuData
);

curve_store!(
    /// Handle to a pre-uploaded glyph set produced by
    /// [`ViewportGpuResources::upload_glyph_set`](crate::resources::ViewportGpuResources::upload_glyph_set).
    GlyphSetId,
    GlyphSetStore,
    GlyphGpuData
);

curve_store!(
    /// Handle to a pre-uploaded tensor glyph set produced by
    /// [`ViewportGpuResources::upload_tensor_glyph_set`](crate::resources::ViewportGpuResources::upload_tensor_glyph_set).
    TensorGlyphSetId,
    TensorGlyphSetStore,
    TensorGlyphGpuData
);

curve_store!(
    /// Handle to a pre-uploaded sprite set produced by
    /// [`ViewportGpuResources::upload_sprite_set`](crate::resources::ViewportGpuResources::upload_sprite_set).
    /// Backs static billboards such as foliage, signage, and light flares.
    SpriteSetId,
    SpriteSetStore,
    SpriteGpuData
);

curve_store!(
    /// Handle to a pre-uploaded sprite instance set produced by
    /// [`ViewportGpuResources::upload_sprite_instance_set`](crate::resources::ViewportGpuResources::upload_sprite_instance_set).
    /// Backs entity sprites such as NPCs, item drops, and damage numbers.
    SpriteInstanceSetId,
    SpriteInstanceSetStore,
    SpriteGpuData
);
