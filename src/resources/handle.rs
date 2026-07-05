//! Shared generational handle primitive.
//!
//! Most GPU content the library stores (meshes, textures, splat sets, volumes,
//! curves) is keyed by a handle into a slotted store. A removed entry leaves an
//! empty slot that a later insert reuses; the handle carries the generation its
//! slot had when it was issued, and the store bumps that generation on removal.
//! A stale handle then resolves to nothing rather than aliasing whatever now
//! occupies the slot.
//!
//! [`slot_handle!`] generates one of these handle types with the standard
//! surface (an `INVALID` sentinel, `index`, and a private `new`), so every
//! content handle looks and behaves the same. [`ContentHandle`] is the common
//! interface over them.

/// Common interface implemented by every slotted content handle.
///
/// Lets code that does not care which resource a handle names (residency
/// accounting, generic validity checks) work over any of them.
pub trait ContentHandle: Copy + Eq {
    /// The handle value that refers to nothing. Store lookups always return
    /// `None` for it.
    const INVALID: Self;

    /// The raw slot index this handle points at.
    fn index(&self) -> usize;

    /// The generation this handle was issued against.
    fn generation(&self) -> u32;

    /// Build a handle from a raw slot index and generation. The store mints a
    /// handle this way on insert; outside code obtains handles from upload calls
    /// and treats them as opaque.
    fn from_parts(index: u32, generation: u32) -> Self;

    /// Whether this handle is anything other than [`INVALID`](Self::INVALID).
    /// A valid handle may still fail to resolve if its slot was freed.
    fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

/// Define a generational handle newtype with the standard surface.
///
/// Generates a `{ index: u32, generation: u32 }` struct plus its `INVALID`
/// sentinel, `index`, crate-internal `new`, and a [`ContentHandle`] impl. Pass
/// the doc comment and visibility; the derives and methods are fixed so every
/// handle matches.
macro_rules! slot_handle {
    ($(#[$meta:meta])* $vis:vis struct $name:ident;) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis struct $name {
            pub(crate) index: u32,
            pub(crate) generation: u32,
        }

        impl $name {
            /// A handle that refers to nothing. Store lookups always return
            /// `None` for it. Use it as the default / placeholder value where a
            /// real resource has not been assigned yet.
            pub const INVALID: $name = $name {
                index: u32::MAX,
                generation: u32::MAX,
            };

            /// The raw slot index this handle points at. Used to index parallel
            /// per-slot arrays; not meaningful for [`INVALID`](Self::INVALID).
            pub fn index(&self) -> usize {
                self.index as usize
            }

            /// Build a handle from raw parts. Crate-internal: outside code
            /// obtains a handle from an upload call and treats it as opaque.
            pub(crate) fn new(index: u32, generation: u32) -> Self {
                Self { index, generation }
            }
        }

        impl $crate::resources::handle::ContentHandle for $name {
            const INVALID: Self = <$name>::INVALID;

            fn index(&self) -> usize {
                self.index as usize
            }

            fn generation(&self) -> u32 {
                self.generation
            }

            fn from_parts(index: u32, generation: u32) -> Self {
                <$name>::new(index, generation)
            }
        }
    };
}

pub(crate) use slot_handle;

/// Define an append-only registry handle: a stable index into a grow-only store
/// that never frees or reuses slots, so it needs no generation.
///
/// These name resources created once and kept for the session (density volumes,
/// projected-tet meshes, GPU particle systems). The handle is opaque: it is
/// obtained from a create / upload call, and its inner index is crate-private so
/// it cannot be synthesised by hand. If a resource class later becomes
/// evictable, its handle graduates to [`slot_handle!`] and gains a generation;
/// because it is already opaque, that is an additive change for consumers.
macro_rules! registry_handle {
    ($(#[$meta:meta])* $vis:vis struct $name:ident;) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis struct $name(pub(crate) usize);

        impl $name {
            /// The raw registry index this handle points at. Stable for the
            /// session; useful for debug overlays. Do not synthesise handles by
            /// hand.
            pub fn index(&self) -> usize {
                self.0
            }
        }
    };
}

pub(crate) use registry_handle;

/// Append-only store keyed by a plain index.
///
/// Backs the grow-only content classes named by [`registry_handle!`] handles
/// (volume textures, projected-tet meshes, overlay textures): slots are never
/// freed or reused, so an index stays valid for the session. Compared to a raw
/// `Vec`, this centralises the push-returns-index bookkeeping and routes reads
/// through [`get`](Registry::get), which bounds-checks instead of panicking on
/// an out-of-range index.
pub(crate) struct Registry<T> {
    items: Vec<T>,
}

impl<T> Registry<T> {
    /// Append a value and return the index it was stored at.
    pub(crate) fn push(&mut self, value: T) -> usize {
        let index = self.items.len();
        self.items.push(value);
        index
    }

    /// Borrow the value at `index`, or `None` if it is out of range.
    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    /// Mutably borrow the value at `index`, or `None` if it is out of range.
    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)
    }

    /// Number of stored values. Also the index the next [`push`](Registry::push)
    /// will return.
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self { items: Vec::new() }
    }
}

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
