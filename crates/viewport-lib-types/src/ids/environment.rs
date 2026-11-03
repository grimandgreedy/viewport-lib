//! Handle to one environment in the indexed IBL set.

/// Handle to one uploaded environment.
///
/// Returned by the environment upload. The handle names an array layer of the
/// fixed IBL set plus a generation, so a handle kept after
/// `free_environment` resolves to nothing rather than to whatever environment
/// reuses the layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvironmentMapId(u32);

impl EnvironmentMapId {
    const LAYER_BITS: u32 = 8;
    const LAYER_MASK: u32 = (1 << Self::LAYER_BITS) - 1;

    /// The array layer this environment occupies.
    pub fn index(self) -> u32 {
        self.0 & Self::LAYER_MASK
    }

    /// The generation of the layer this handle was issued for.
    #[doc(hidden)]
    pub fn generation(self) -> u32 {
        self.0 >> Self::LAYER_BITS
    }

    /// Build a handle naming array layer `layer` at `generation`. Crate-internal:
    /// outside code obtains a handle from an upload call and treats it as opaque.
    /// `layer` must be below 256; the generation wraps at 24 bits.
    #[doc(hidden)]
    pub fn from_parts(layer: u32, generation: u32) -> Self {
        debug_assert!(layer <= Self::LAYER_MASK);
        Self((layer & Self::LAYER_MASK) | (generation << Self::LAYER_BITS))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_round_trip() {
        let id = EnvironmentMapId::from_parts(31, 0x00AB_CDEF);
        assert_eq!(id.index(), 31);
        assert_eq!(id.generation(), 0x00AB_CDEF);
        assert_ne!(id, EnvironmentMapId::from_parts(31, 0x00AB_CDF0));
    }
}
