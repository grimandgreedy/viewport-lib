//! Per-instance custom data buffer.
//!
//! Instances that share a material (and so batch together) can still vary a
//! small block of material inputs without breaking the batch: each instance
//! carries a raw `[f32; 8]` payload. This is the same channel Unity exposes as
//! `MaterialPropertyBlock` GPU-instanced properties and Unreal exposes as ISM /
//! HISM per-instance custom float data.
//!
//! The payloads live once in a scene-global storage buffer rather than being
//! duplicated into every per-instance record. Each instance carries a small
//! `custom_data_id` that indexes this buffer. The all-zero payload (the common
//! case, when no instance sets custom data) collapses to entry 0, so a scene
//! that never touches custom data uploads a single entry and pays nothing in the
//! per-instance record beyond the id it already had room for.
//!
//! The buffer is bound at group 0 (the scene-wide bind group) so every mesh
//! shader and draw variant reaches it without touching the per-object /
//! instanced group-1 layouts.
//!
//! Slot convention read by the built-in instanced shading:
//! - slots 0..3 (`data[0..3]`): added to the material's emissive (nits). Zero is
//!   a no-op, so leaving custom data unset changes nothing.
//! - slots 3..8: reserved as a raw channel for material plugins to consume. The
//!   built-in shading does not read them.

use std::collections::HashMap;

/// Number of raw floats carried per instance. Two `vec4` std430 slots.
pub const CUSTOM_DATA_FLOATS: usize = 8;

/// Maximum distinct custom-data blocks per frame. Scenes commonly vary custom
/// data across a palette (a bounded set of colour / scale variations) rather than
/// a genuinely-unique value per instance, so this bounds the number of *distinct*
/// blocks a frame holds. On overflow, further instances fall back to the zero
/// block (id 0) and [`CustomDataBuilder::overflowed`] is set so the caller can
/// log it. The buffer is preallocated at this size (32 bytes each, 8 MiB).
///
/// A workload with a genuinely-unique payload per instance beyond this count is
/// the case this deduplicated buffer serves least well (it degenerates to one
/// entry per instance); if such a profile matters, the storage moves to a
/// growable or directly instance-indexed buffer. The public `custom_data` API on
/// `ItemSettings` does not change with that.
pub(crate) const CUSTOM_DATA_CAPACITY: usize = 262144;

/// One instance's raw custom-data payload, vec4-packed for std430 alignment.
/// Matches the WGSL `CustomData` struct, 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceCustomData {
    pub(crate) data: [f32; CUSTOM_DATA_FLOATS],
}

const _: () = assert!(std::mem::size_of::<InstanceCustomData>() == 32);

impl InstanceCustomData {
    pub(crate) const ZERO: InstanceCustomData = InstanceCustomData {
        data: [0.0; CUSTOM_DATA_FLOATS],
    };
}

/// Per-frame interner that deduplicates custom-data blocks and assigns each a
/// stable `custom_data_id` (its index in the uploaded buffer). Entry 0 is always
/// the zero block, so any instance with no authored custom data maps to 0.
pub(crate) struct CustomDataBuilder {
    entries: Vec<InstanceCustomData>,
    lookup: HashMap<[u8; 32], u32>,
    /// Set when the capacity was hit and some instances were forced to entry 0.
    pub(crate) overflowed: bool,
}

impl Default for CustomDataBuilder {
    fn default() -> Self {
        let mut b = CustomDataBuilder {
            entries: Vec::new(),
            lookup: HashMap::new(),
            overflowed: false,
        };
        b.reset();
        b
    }
}

impl CustomDataBuilder {
    /// Clear to a single zero-block entry (id 0) for a new frame.
    pub(crate) fn reset(&mut self) {
        self.entries.clear();
        self.lookup.clear();
        self.overflowed = false;
        self.entries.push(InstanceCustomData::ZERO);
        self.lookup
            .insert(bytemuck::cast(InstanceCustomData::ZERO), 0);
    }

    /// Intern a raw payload, returning its `custom_data_id`. The all-zero payload
    /// short-circuits to 0 without hashing (the common case), so scenes that
    /// never set custom data pay no per-instance hashing cost. On overflow,
    /// returns 0.
    pub(crate) fn intern(&mut self, data: [f32; CUSTOM_DATA_FLOATS]) -> u32 {
        let block = InstanceCustomData { data };
        if block == InstanceCustomData::ZERO {
            return 0;
        }
        let key: [u8; 32] = bytemuck::cast(block);
        if let Some(&id) = self.lookup.get(&key) {
            return id;
        }
        if self.entries.len() >= CUSTOM_DATA_CAPACITY {
            self.overflowed = true;
            return 0;
        }
        let id = self.entries.len() as u32;
        self.entries.push(block);
        self.lookup.insert(key, id);
        id
    }

    /// The blocks to upload this frame (always at least the zero entry).
    pub(crate) fn entries(&self) -> &[InstanceCustomData] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_payload_interns_to_zero_without_growing() {
        let mut b = CustomDataBuilder::default();
        assert_eq!(b.intern([0.0; 8]), 0);
        // Only the reserved zero entry exists.
        assert_eq!(b.entries().len(), 1);
    }

    #[test]
    fn distinct_payload_makes_an_entry_and_dedups() {
        let mut b = CustomDataBuilder::default();
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let id = b.intern(a);
        assert_ne!(id, 0, "a non-zero payload is not the zero block");
        // Same payload interns to the same id (dedup), no new entry.
        assert_eq!(b.intern(a), id);
        assert_eq!(b.entries().len(), 2);
        // A different payload gets its own id.
        let b2 = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_ne!(b.intern(b2), id);
        assert_eq!(b.entries().len(), 3);
    }

    #[test]
    fn payload_bytes_survive_the_round_trip() {
        let mut b = CustomDataBuilder::default();
        let p = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let id = b.intern(p) as usize;
        assert_eq!(b.entries()[id].data, p);
    }
}
