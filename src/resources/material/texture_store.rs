//! Slotted texture storage with generational ids.
//!
//! `TextureStore` gives user-uploaded textures the same slot-and-generation
//! model `MeshStore` gives meshes: a removed texture leaves an empty slot that a
//! later upload reuses, and each slot carries a generation counter bumped on
//! removal. Texture ids stay `u64` (materials hold them directly), with the
//! generation packed into the high 32 bits and the slot index in the low 32.
//! For a texture that is never freed the generation is 0, so its id equals its
//! slot index and behaves exactly like the old dense index. Once a slot is freed
//! and reused, a handle to the old texture carries a stale generation and
//! resolves to `None`, falling back to the fallback texture instead of aliasing
//! whatever now occupies the slot.

use crate::resources::GpuTexture;

/// Handle to a user-uploaded texture.
///
/// Wraps a packed slot index (low 32 bits) and generation (high 32 bits). A
/// handle whose texture was freed (its slot reused by a later upload) carries a
/// stale generation and resolves to `None` on lookup, falling back to the
/// fallback texture instead of aliasing whatever now occupies the slot. For a
/// texture that is never freed the generation is 0, so the handle's raw value
/// equals its dense slot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureId(pub(crate) u64);

impl TextureId {
    /// A handle that refers to no texture. Store lookups always return `None`
    /// for it. Use it as the default / placeholder value.
    pub const INVALID: TextureId = TextureId(u64::MAX);

    /// The raw slot index this handle points at.
    pub fn index(&self) -> usize {
        (self.0 as u32) as usize
    }

    /// The packed raw value, for keying caches and crossing the plugin boundary.
    pub(crate) fn raw(&self) -> u64 {
        self.0
    }
}

impl crate::resources::handle::ContentHandle for TextureId {
    const INVALID: Self = TextureId(u64::MAX);

    fn index(&self) -> usize {
        (self.0 as u32) as usize
    }

    fn generation(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    fn from_parts(index: u32, generation: u32) -> Self {
        pack_texture_id(index, generation)
    }
}

/// Pack a slot index and generation into a texture id.
///
/// Generation 0 leaves the high bits clear, so `pack(index, 0) == index` and a
/// never-freed texture keeps the dense-index id it always had.
fn pack_texture_id(index: u32, generation: u32) -> TextureId {
    TextureId(((generation as u64) << 32) | index as u64)
}

/// Slotted storage for user-uploaded GPU textures with generational ids, a free
/// list, and maintained byte / count totals. The id packs the slot index and
/// generation into a `u64`; the generic core keys on it through
/// [`ContentHandle`](crate::resources::handle::ContentHandle).
pub(crate) struct TextureStore {
    store: crate::resources::handle::SlotStore<GpuTexture, TextureId>,
}

impl TextureStore {
    /// Create an empty texture store.
    pub fn new() -> Self {
        Self {
            store: crate::resources::handle::SlotStore::default(),
        }
    }

    /// Insert a texture charging `bytes` against it, reusing a free slot if one
    /// is available. Returns the handle carrying the slot's generation.
    pub fn insert(&mut self, texture: GpuTexture, bytes: u64) -> TextureId {
        self.store.insert(texture, bytes)
    }

    /// Look up the live texture for `id`, or `None` if the index is out of
    /// range, the slot is empty, or the handle's generation is stale.
    pub fn get(&self, id: TextureId) -> Option<&GpuTexture> {
        self.store.get(id)
    }

    /// Remove the texture for `id`, dropping the `GpuTexture`, bumping the slot
    /// generation, freeing the slot, and decrementing the byte / count totals.
    ///
    /// Returns `true` if a texture was actually removed, `false` if the slot was
    /// empty, out of range, or the handle was stale.
    pub fn remove(&mut self, id: TextureId) -> bool {
        self.store.remove(id).is_some()
    }

    /// Swap the texture in `id`'s slot for `texture`, charging `bytes` in place
    /// of the old size and keeping the slot generation (so `id` stays valid).
    ///
    /// The generation check is the in-flight guard: a stale `id` (its slot freed
    /// and reused) does not resolve, so it cannot overwrite whatever now occupies
    /// the slot. Returns the dropped `GpuTexture` on success (the caller drops it
    /// after the frame that may still sample it), or `None` if `id` did not
    /// resolve to a live texture.
    pub fn replace(
        &mut self,
        id: TextureId,
        texture: GpuTexture,
        bytes: u64,
    ) -> Option<GpuTexture> {
        self.store.replace(id, texture, bytes)
    }

    /// Number of live (occupied) texture slots.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Total bytes charged for the textures currently resident in the store.
    pub fn allocated_bytes(&self) -> u64 {
        self.store.allocated_bytes()
    }
}
