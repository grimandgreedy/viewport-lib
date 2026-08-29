//! Slotted store with generational handles.
//!
//! The handle primitives themselves ([`ContentHandle`], the [`slot_handle!`] and
//! [`registry_handle!`] generators) live in `viewport-lib-types` so CPU-side
//! tools can name a resource without the renderer. They are re-exported here so
//! the renderer's `crate::resources::handle::*` paths keep resolving. This
//! module keeps [`SlotStore`], the renderer-side store the freeable content
//! stores wrap.

pub use viewport_lib_types::ids::ContentHandle;
pub(crate) use viewport_lib_types::slot_handle;

/// One slot in a [`SlotStore`]: the value when occupied, the slot's current
/// generation, and the GPU byte size charged for it.
struct Slot<T> {
    value: Option<T>,
    generation: u32,
    bytes: u64,
}

/// Slotted store with generational handles, a free list, and maintained
/// resident-byte and live-count totals.
///
/// A removed entry leaves an empty slot that a later insert reuses. Each slot
/// carries a generation bumped on removal, and the handle captures the
/// generation its slot had when it was issued. A handle whose generation no
/// longer matches its slot resolves to `None`, so a stale handle held across a
/// remove-then-reinsert cannot alias the value now in the slot.
///
/// This is the shared core behind the freeable content stores (meshes,
/// textures, LOD groups, Gaussian splat sets). Each of those wraps a
/// `SlotStore` and adds only what differs: how it measures an entry's GPU byte
/// size, and any lookups specific to that resource. Byte accounting is opt-in,
/// a store whose entries carry no measured size passes `0` for `bytes`.
pub(crate) struct SlotStore<T, H: ContentHandle> {
    slots: Vec<Slot<T>>,
    free_list: Vec<usize>,
    allocated_bytes: u64,
    live_count: usize,
    _handle: std::marker::PhantomData<H>,
}

impl<T, H: ContentHandle> SlotStore<T, H> {
    /// Insert a value charging `bytes` against it, reusing a free slot if one is
    /// available. Returns the handle carrying the slot's current generation.
    pub(crate) fn insert(&mut self, value: T, bytes: u64) -> H {
        self.allocated_bytes += bytes;
        self.live_count += 1;
        if let Some(idx) = self.free_list.pop() {
            let slot = &mut self.slots[idx];
            slot.value = Some(value);
            slot.bytes = bytes;
            H::from_parts(idx as u32, slot.generation)
        } else {
            let idx = self.slots.len();
            self.slots.push(Slot {
                value: Some(value),
                generation: 0,
                bytes,
            });
            H::from_parts(idx as u32, 0)
        }
    }

    /// The live slot for `id`, or `None` if the index is out of range, the slot
    /// is empty, or the handle's generation is stale.
    fn live_slot(&self, id: H) -> Option<&Slot<T>> {
        let slot = self.slots.get(id.index())?;
        if slot.generation != id.generation() {
            return None;
        }
        slot.value.is_some().then_some(slot)
    }

    /// Borrow the value for `id`, validating the generation. `None` for a stale
    /// handle, an empty slot, or an out-of-range index.
    pub(crate) fn get(&self, id: H) -> Option<&T> {
        self.live_slot(id)?.value.as_ref()
    }

    /// Mutably borrow the value for `id`, with the same generation check as
    /// [`get`](Self::get).
    pub(crate) fn get_mut(&mut self, id: H) -> Option<&mut T> {
        let slot = self.slots.get_mut(id.index())?;
        if slot.generation != id.generation() {
            return None;
        }
        slot.value.as_mut()
    }

    /// Borrow the value in a raw slot index without a generation check. For the
    /// per-frame draw path, where the index was already validated through
    /// [`get`](Self::get) earlier in the same frame.
    pub(crate) fn get_by_index(&self, index: usize) -> Option<&T> {
        self.slots.get(index)?.value.as_ref()
    }

    /// Mutable raw-index lookup, same contract as
    /// [`get_by_index`](Self::get_by_index).
    pub(crate) fn get_mut_by_index(&mut self, index: usize) -> Option<&mut T> {
        self.slots.get_mut(index)?.value.as_mut()
    }

    /// Swap the value in `id`'s slot, charging `bytes` in place of the old size
    /// and keeping the slot generation so `id` stays valid. Returns the old
    /// value, or `None` for a stale handle or an empty slot.
    ///
    /// The generation check is the in-flight guard: a stale `id` does not
    /// resolve, so it cannot overwrite whatever now occupies the slot.
    pub(crate) fn replace(&mut self, id: H, value: T, bytes: u64) -> Option<T> {
        let slot = self.slots.get_mut(id.index())?;
        if slot.generation != id.generation() || slot.value.is_none() {
            return None;
        }
        let old = slot.value.replace(value);
        self.allocated_bytes = self.allocated_bytes.saturating_sub(slot.bytes) + bytes;
        slot.bytes = bytes;
        old
    }

    /// Update the byte charge for `id`'s slot without swapping the value, for a
    /// store that mutates an entry in place through [`get_mut`](Self::get_mut)
    /// and needs to keep its resident-byte total accurate. Returns `false` for a
    /// stale handle or an empty slot.
    pub(crate) fn set_bytes(&mut self, id: H, bytes: u64) -> bool {
        let Some(slot) = self.slots.get_mut(id.index()) else {
            return false;
        };
        if slot.generation != id.generation() || slot.value.is_none() {
            return false;
        }
        self.allocated_bytes = self.allocated_bytes.saturating_sub(slot.bytes) + bytes;
        slot.bytes = bytes;
        true
    }

    /// Remove the value for `id`, bump the slot generation, free the slot, and
    /// drop its byte charge. Returns the removed value, or `None` for a stale
    /// handle or an empty slot.
    pub(crate) fn remove(&mut self, id: H) -> Option<T> {
        let slot = self.slots.get_mut(id.index())?;
        if slot.generation != id.generation() {
            return None;
        }
        let value = slot.value.take()?;
        self.allocated_bytes = self.allocated_bytes.saturating_sub(slot.bytes);
        slot.bytes = 0;
        // Bump so the just-removed handle no longer matches this slot.
        slot.generation = slot.generation.wrapping_add(1);
        self.free_list.push(id.index());
        self.live_count -= 1;
        Some(value)
    }

    /// Whether the slot for `id` holds a live value.
    pub(crate) fn contains(&self, id: H) -> bool {
        self.live_slot(id).is_some()
    }

    /// Number of occupied (non-empty) slots.
    pub(crate) fn len(&self) -> usize {
        self.live_count
    }

    /// Total number of slots (occupied plus free).
    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Total GPU bytes charged across every resident value.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Iterate every live value with its handle.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (H, &T)> {
        self.slots.iter().enumerate().filter_map(|(idx, slot)| {
            slot.value
                .as_ref()
                .map(|v| (H::from_parts(idx as u32, slot.generation), v))
        })
    }

    /// Mutably iterate every live value with its handle.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (H, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(idx, slot)| {
            let generation = slot.generation;
            slot.value
                .as_mut()
                .map(|v| (H::from_parts(idx as u32, generation), v))
        })
    }
}

impl<T, H: ContentHandle> Default for SlotStore<T, H> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            allocated_bytes: 0,
            live_count: 0,
            _handle: std::marker::PhantomData,
        }
    }
}
