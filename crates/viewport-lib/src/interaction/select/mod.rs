//! Selection state: object multi-select and sub-object selection.
//!
//! [`selection::Selection`] tracks which objects are selected;
//! [`sub_object::SubSelection`] tracks which faces, vertices, edges, or points
//! within them are selected. Pick masks (`PickMask`) live under
//! `crate::renderer::picking`, alongside the picker that owns them.

/// Multi-select system for viewport objects.
pub mod selection;
/// Typed sub-object reference ([`sub_object::SubObjectRef`]) and the sub-object
/// selection set ([`sub_object::SubSelection`]).
pub mod sub_object;
