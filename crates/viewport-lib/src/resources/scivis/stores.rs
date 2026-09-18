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
