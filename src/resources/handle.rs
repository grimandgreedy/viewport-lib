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
