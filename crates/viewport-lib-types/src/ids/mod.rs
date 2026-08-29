//! Generational and registry handles for GPU content.
//!
//! The handle machinery ([`ContentHandle`], [`slot_handle!`],
//! [`registry_handle!`]) lives here; the individual handle types are defined in
//! the domain submodules and re-exported.
//!
//! Most GPU content the renderer stores (meshes, textures, splat sets, volumes,
//! curves) is keyed by a handle into a slotted store. A removed entry leaves an
//! empty slot that a later insert reuses; the handle carries the generation its
//! slot had when it was issued, and the store bumps that generation on removal.
//! A stale handle then resolves to nothing rather than aliasing whatever now
//! occupies the slot.
//!
//! [`slot_handle!`] generates one of these handle types with the standard
//! surface (an `INVALID` sentinel, `index`, and a private `new`), so every
//! content handle looks and behaves the same. [`ContentHandle`] is the common
//! interface over them. [`registry_handle!`] generates the append-only variant
//! for resources that are created once and never freed, so they need no
//! generation.

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
#[doc(hidden)]
#[macro_export]
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

            /// The generation this handle was issued against. Paired with
            /// [`index`](Self::index) it identifies a specific slot occupancy;
            /// a lookup fails once the slot's generation moves past it.
            pub fn generation(&self) -> u32 {
                self.generation
            }

            /// Build a handle from raw parts. Crate-internal: outside code
            /// obtains a handle from an upload call and treats it as opaque.
            pub(crate) fn new(index: u32, generation: u32) -> Self {
                Self { index, generation }
            }

            /// Fabricate a handle naming raw slot `index` at generation 0, the
            /// generation a freshly inserted slot carries.
            ///
            /// Hidden from the documented surface: production code obtains
            /// handles from upload calls and treats them as opaque. This exists
            /// for tests, benches, and fixtures that need to name a slot
            /// directly (or synthesize a deliberately out-of-range id).
            #[doc(hidden)]
            pub fn from_index(index: u32) -> Self {
                Self { index, generation: 0 }
            }
        }

        impl $crate::ids::ContentHandle for $name {
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

/// Define an append-only registry handle: a stable index into a grow-only store
/// that never frees or reuses slots, so it needs no generation.
///
/// These name resources created once and kept for the session (density volumes,
/// projected-tet meshes, GPU particle systems). The handle is opaque: it is
/// obtained from a create / upload call, and its inner index is crate-private so
/// it cannot be synthesised by hand. If a resource class later becomes
/// evictable, its handle graduates to [`slot_handle!`] and gains a generation;
/// because it is already opaque, that is an additive change for consumers.
#[doc(hidden)]
#[macro_export]
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

            /// Build a handle naming raw registry `index`.
            ///
            /// Hidden from the documented surface: production code obtains
            /// handles from a create / upload call and treats them as opaque.
            /// The store that owns the registry mints handles this way.
            #[doc(hidden)]
            pub fn from_index(index: usize) -> Self {
                Self(index)
            }
        }
    };
}

pub mod environment;
pub mod gpu;
pub mod matcap;
pub mod mesh;
pub mod pick;
pub mod splat;
pub mod texture;
pub mod volume;

pub use environment::EnvironmentMapId;
pub use gpu::{ExternalInstanceSetId, GpuParticleSystemId};
pub use matcap::MatcapId;
pub use mesh::{LodGroupId, MeshId};
pub use pick::PickId;
pub use splat::GaussianSplatId;
pub use texture::TextureId;
pub use volume::{ProjectedTetId, VolumeId};
