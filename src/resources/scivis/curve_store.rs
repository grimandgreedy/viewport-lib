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

macro_rules! curve_store {
    ($(#[$id_doc:meta])* $id:ident, $store:ident, $data:ty) => {
        $(#[$id_doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $id(pub(crate) usize);

        impl $id {
            /// Build an id from a raw slot index.
            pub fn from_index(index: usize) -> Self { Self(index) }
            /// The raw slot index.
            pub fn index(&self) -> usize { self.0 }
        }

        pub(crate) struct $store {
            slots: Vec<Option<$data>>,
            free_list: Vec<usize>,
        }

        impl $store {
            pub fn new() -> Self {
                Self { slots: Vec::new(), free_list: Vec::new() }
            }

            pub fn insert(&mut self, data: $data) -> $id {
                if let Some(idx) = self.free_list.pop() {
                    self.slots[idx] = Some(data);
                    $id(idx)
                } else {
                    let idx = self.slots.len();
                    self.slots.push(Some(data));
                    $id(idx)
                }
            }

            pub fn get(&self, id: $id) -> Option<&$data> {
                self.slots.get(id.0)?.as_ref()
            }

            pub fn replace(&mut self, id: $id, data: $data) -> bool {
                match self.slots.get_mut(id.0) {
                    Some(slot) if slot.is_some() => { *slot = Some(data); true }
                    _ => false,
                }
            }

            pub fn remove(&mut self, id: $id) -> bool {
                if let Some(slot) = self.slots.get_mut(id.0) {
                    if slot.is_some() {
                        *slot = None;
                        self.free_list.push(id.0);
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
                self.slots.iter().filter(|s| s.is_some()).count()
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
