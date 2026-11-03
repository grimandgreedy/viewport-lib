//! Handle for a registered runtime plugin.

/// Handle returned by `ViewportRuntime::add_plugin`; pass it to
/// `ViewportRuntime::remove_plugin` to unregister the plugin. Ids count up
/// from one per runtime and are never reused within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimePluginId(u64);

impl RuntimePluginId {
    /// Wrap a raw id. The runtime hands these out; a host only needs this to
    /// rebuild an id it stored as a number.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw id.
    pub const fn raw(self) -> u64 {
        self.0
    }
}
