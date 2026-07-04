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
}

/// Pack a slot index and generation into a texture id.
///
/// Generation 0 leaves the high bits clear, so `pack(index, 0) == index` and a
/// never-freed texture keeps the dense-index id it always had.
fn pack_texture_id(index: u32, generation: u32) -> TextureId {
    TextureId(((generation as u64) << 32) | index as u64)
}

/// Split a texture id back into `(index, generation)`.
fn unpack_texture_id(id: TextureId) -> (u32, u32) {
    (id.0 as u32, (id.0 >> 32) as u32)
}

/// One texture slot: the texture (when occupied), the slot's current
/// generation, and the byte size charged for it (for resident-bytes
/// accounting).
struct TexSlot {
    texture: Option<GpuTexture>,
    generation: u32,
    bytes: u64,
}

/// Slotted storage for user-uploaded GPU textures with generational ids, a free
/// list, and maintained byte / count totals.
pub(crate) struct TextureStore {
    slots: Vec<TexSlot>,
    free_list: Vec<usize>,
    allocated_bytes: u64,
    live_count: u32,
}

impl TextureStore {
    /// Create an empty texture store.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            allocated_bytes: 0,
            live_count: 0,
        }
    }

    /// Insert a texture charging `bytes` against it, reusing a free slot if one
    /// is available. Returns the handle carrying the slot's generation.
    pub fn insert(&mut self, texture: GpuTexture, bytes: u64) -> TextureId {
        self.allocated_bytes += bytes;
        self.live_count += 1;
        if let Some(idx) = self.free_list.pop() {
            let slot = &mut self.slots[idx];
            slot.texture = Some(texture);
            slot.bytes = bytes;
            pack_texture_id(idx as u32, slot.generation)
        } else {
            let idx = self.slots.len();
            self.slots.push(TexSlot {
                texture: Some(texture),
                generation: 0,
                bytes,
            });
            pack_texture_id(idx as u32, 0)
        }
    }

    /// Look up the live texture for `id`, or `None` if the index is out of
    /// range, the slot is empty, or the handle's generation is stale.
    pub fn get(&self, id: TextureId) -> Option<&GpuTexture> {
        let (index, generation) = unpack_texture_id(id);
        let slot = self.slots.get(index as usize)?;
        if slot.generation != generation {
            return None;
        }
        slot.texture.as_ref()
    }

    /// Remove the texture for `id`, dropping the `GpuTexture`, bumping the slot
    /// generation, freeing the slot, and decrementing the byte / count totals.
    ///
    /// Returns `true` if a texture was actually removed, `false` if the slot was
    /// empty, out of range, or the handle was stale.
    pub fn remove(&mut self, id: TextureId) -> bool {
        let (index, generation) = unpack_texture_id(id);
        if let Some(slot) = self.slots.get_mut(index as usize) {
            if slot.generation == generation && slot.texture.is_some() {
                slot.texture = None;
                slot.generation = slot.generation.wrapping_add(1);
                self.allocated_bytes = self.allocated_bytes.saturating_sub(slot.bytes);
                self.live_count = self.live_count.saturating_sub(1);
                slot.bytes = 0;
                self.free_list.push(index as usize);
                return true;
            }
        }
        false
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
        let (index, generation) = unpack_texture_id(id);
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation != generation || slot.texture.is_none() {
            return None;
        }
        let old = slot.texture.replace(texture);
        self.allocated_bytes = self.allocated_bytes.saturating_sub(slot.bytes) + bytes;
        slot.bytes = bytes;
        old
    }

    /// Number of live (occupied) texture slots.
    pub fn len(&self) -> usize {
        self.live_count as usize
    }

    /// Total bytes charged for the textures currently resident in the store.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }
}
