//! Slot stores for pre-uploaded scivis content.
//!
//! Each store holds GPU data produced by the upload helpers, keyed by a typed
//! handle. Per-frame ref items (`PolylineRefItem` and friends) name a store
//! entry by handle and supply per-frame overrides (model matrix, etc.) instead
//! of resubmitting the geometry every frame.
//!
//! The stores themselves are all [`SlotStore`] instances: this module declares
//! the handle types, states each payload's GPU byte charge, and pairs the two.

use crate::resources::PolylineGpuData;
use crate::resources::handle::{GpuByteSize, SlotStore, slot_handle};

impl GpuByteSize for PolylineGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size() + self._uniform_buf.size()
    }
}

slot_handle! {
    /// Handle to a pre-uploaded polyline produced by
    /// [`DeviceResources::upload_polyline`](crate::resources::DeviceResources::upload_polyline).
    pub struct PolylineId;
}

pub(crate) type PolylineStore = SlotStore<PolylineGpuData, PolylineId>;
