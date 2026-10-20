//! Handle to a material in the renderer's material store.

crate::slot_handle! {
    /// Handle to a material in the store.
    ///
    /// Returned by `DeviceResources::upload_material` and edited in place with
    /// `replace_material`: every item holding the handle draws the new values
    /// on the next frame. A handle whose slot was freed carries a stale
    /// generation and draws [`MaterialId::DEFAULT`] instead of aliasing a
    /// material uploaded later into the slot.
    pub struct MaterialId;
}

impl MaterialId {
    /// The store's default material, `Material::default()`. Always resident and
    /// never freed, so an item that names no material still draws.
    pub const DEFAULT: MaterialId = MaterialId {
        index: 0,
        generation: 0,
    };
}

impl Default for MaterialId {
    fn default() -> Self {
        Self::DEFAULT
    }
}
