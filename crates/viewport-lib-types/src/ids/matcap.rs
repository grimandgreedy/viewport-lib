//! Handle to a matcap texture uploaded to the GPU.

/// Identifies a matcap texture uploaded to the GPU.
///
/// Obtained from `DeviceResources::upload_matcap` or `builtin_matcap_id`. An
/// append-only registry handle. The `blendable` flag controls whether the alpha
/// channel tints the base geometry colour (`true`) or the matcap fully replaces
/// the object colour (`false`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatcapId {
    /// Index into the GPU matcap texture store.
    index: usize,
    /// Whether the alpha channel blends with base geometry colour.
    pub blendable: bool,
}

impl MatcapId {
    /// The matcap store index this handle points at.
    #[doc(hidden)]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Build a handle from a store index and blend flag. Crate-internal: outside
    /// code obtains a handle from an upload call and treats it as opaque.
    #[doc(hidden)]
    pub fn from_parts(index: usize, blendable: bool) -> Self {
        Self { index, blendable }
    }
}
