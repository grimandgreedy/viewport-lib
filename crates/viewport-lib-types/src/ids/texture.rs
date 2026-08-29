//! Handle to a user-uploaded texture.

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
pub struct TextureId(u64);

impl TextureId {
    /// A handle that refers to no texture. Store lookups always return `None`
    /// for it. Use it as the default / placeholder value.
    pub const INVALID: TextureId = TextureId(u64::MAX);

    /// The raw slot index this handle points at.
    pub fn index(&self) -> usize {
        (self.0 as u32) as usize
    }

    /// The packed raw value, for keying caches and crossing the plugin boundary.
    #[doc(hidden)]
    pub fn raw(&self) -> u64 {
        self.0
    }

    /// Build a handle from a packed raw value. Crate-internal: outside code
    /// obtains a handle from an upload call and treats it as opaque.
    #[doc(hidden)]
    pub fn from_raw(raw: u64) -> Self {
        TextureId(raw)
    }
}

impl crate::ids::ContentHandle for TextureId {
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
