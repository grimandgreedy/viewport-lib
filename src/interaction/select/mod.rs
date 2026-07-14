//! Selection state: multi-select.
//!
//! Pick masks (`PickMask`) and sub-object references (`SubObjectRef`,
//! `SubSelection`) live under `crate::renderer::picking`, alongside the
//! unified picker that owns them.

/// Multi-select system for viewport objects.
pub mod selection;
