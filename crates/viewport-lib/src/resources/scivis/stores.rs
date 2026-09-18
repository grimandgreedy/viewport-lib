//! Slot stores for pre-uploaded scivis content.
//!
//! Each store holds GPU data produced by the upload helpers, keyed by a typed
//! handle. Per-frame ref items (`PolylineRefItem` and friends) name a store
//! entry by handle and supply per-frame overrides (model matrix, etc.) instead
//! of resubmitting the geometry every frame.
//!
//! The stores themselves are all [`SlotStore`] instances: this module declares
//! the handle types, states each payload's GPU byte charge, and pairs the two.

use crate::resources::handle::{GpuByteSize, SlotStore, slot_handle};
use crate::resources::{
    GlyphGpuData, PolylineGpuData, SpriteGpuData, StreamtubeGpuData, TensorGlyphGpuData,
};

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
            + self.node_pick_buffer.as_ref().map_or(0, |b| b.size())
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

slot_handle! {
    /// Handle to a pre-uploaded polyline produced by
    /// [`DeviceResources::upload_polyline`](crate::resources::DeviceResources::upload_polyline).
    pub struct PolylineId;
}

slot_handle! {
    /// Handle to a pre-uploaded streamtube produced by
    /// [`DeviceResources::upload_streamtube`](crate::resources::DeviceResources::upload_streamtube).
    pub struct StreamtubeId;
}

slot_handle! {
    /// Handle to a pre-uploaded tube produced by
    /// [`DeviceResources::upload_tube`](crate::resources::DeviceResources::upload_tube).
    pub struct TubeId;
}

slot_handle! {
    /// Handle to a pre-uploaded ribbon produced by
    /// [`DeviceResources::upload_ribbon`](crate::resources::DeviceResources::upload_ribbon).
    pub struct RibbonId;
}

slot_handle! {
    /// Handle to a pre-uploaded glyph set produced by
    /// [`DeviceResources::upload_glyph_set`](crate::resources::DeviceResources::upload_glyph_set).
    pub struct GlyphSetId;
}

slot_handle! {
    /// Handle to a pre-uploaded tensor glyph set produced by
    /// [`DeviceResources::upload_tensor_glyph_set`](crate::resources::DeviceResources::upload_tensor_glyph_set).
    pub struct TensorGlyphSetId;
}

slot_handle! {
    /// Handle to a pre-uploaded sprite set produced by
    /// [`DeviceResources::upload_sprite_set`](crate::resources::DeviceResources::upload_sprite_set).
    /// Backs static billboards such as foliage, signage, and light flares.
    pub struct SpriteSetId;
}

slot_handle! {
    /// Handle to a pre-uploaded sprite instance set produced by
    /// [`DeviceResources::upload_sprite_instance_set`](crate::resources::DeviceResources::upload_sprite_instance_set).
    /// Backs entity sprites such as NPCs, item drops, and damage numbers.
    pub struct SpriteInstanceSetId;
}

pub(crate) type PolylineStore = SlotStore<PolylineGpuData, PolylineId>;
pub(crate) type StreamtubeStore = SlotStore<StreamtubeGpuData, StreamtubeId>;
pub(crate) type TubeStore = SlotStore<StreamtubeGpuData, TubeId>;
pub(crate) type RibbonStore = SlotStore<StreamtubeGpuData, RibbonId>;
pub(crate) type GlyphSetStore = SlotStore<GlyphGpuData, GlyphSetId>;
pub(crate) type TensorGlyphSetStore = SlotStore<TensorGlyphGpuData, TensorGlyphSetId>;
pub(crate) type SpriteSetStore = SlotStore<SpriteGpuData, SpriteSetId>;
pub(crate) type SpriteInstanceSetStore = SlotStore<SpriteGpuData, SpriteInstanceSetId>;
