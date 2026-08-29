//! Handle to one environment in the indexed IBL set.

/// Handle to one environment in the indexed set.
///
/// Layer 0 is the scene default; extra environments take layers 1.. up to the
/// fixed IBL capacity. The value is the array-texture layer the environment's
/// irradiance and prefiltered specular occupy.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvironmentMapId(u32);

impl EnvironmentMapId {
    /// The scene default environment (array layer 0).
    pub const DEFAULT: Self = Self(0);

    /// The array layer this environment occupies.
    pub fn index(self) -> u32 {
        self.0
    }

    /// Build a handle naming array layer `layer`. Crate-internal: outside code
    /// obtains a handle from an upload call and treats it as opaque.
    #[doc(hidden)]
    pub fn from_raw(layer: u32) -> Self {
        Self(layer)
    }
}
