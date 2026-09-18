//! The polylines this item type holds on the consumer's behalf.
//!
//! A `PolylineRefItem` names a polyline uploaded once through the
//! `*_polyline` methods on [`ViewportRenderer`](crate::renderer::ViewportRenderer),
//! so a curve that does not change is built once and re-placed per frame.
//!
//! The payload and the builder that fills it are not here, and deliberately:
//! [`PolylineGpuData`](crate::resources::PolylineGpuData) is the shared line
//! substrate's output shape. Isolines, clip-object outlines, scatter bounds,
//! volume boxes and four item types' wireframe overlays all render through the
//! same pipelines and the same per-frame upload, so the substrate stays with
//! the renderer. What is this item type's alone is the store of pre-uploaded
//! curves, which is what lives here.

pub(crate) use super::types::PolylineId;
use crate::resources::PolylineGpuData;

/// Slotted store of pre-uploaded polylines.
///
/// A removed entry leaves an empty slot that a later upload reuses. Each slot
/// carries a generation bumped on removal, so a stale [`PolylineId`] resolves
/// to nothing rather than aliasing the curve now in its slot.
pub(super) type PolylineStore = crate::resources::handle::SlotStore<PolylineGpuData, PolylineId>;

impl crate::resources::handle::GpuByteSize for PolylineGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size() + self._uniform_buf.size()
    }
}
