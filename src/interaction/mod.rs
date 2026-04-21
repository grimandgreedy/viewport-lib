/// World-space annotation labels and screen-projection utilities.
pub mod annotation;
/// Interactive clip plane controller: position and orient section planes.
pub mod clip_plane;
/// Transform gizmo (translate, rotate, scale) with hit testing.
pub mod gizmo;
/// Action-based input system with mode-sensitive key/mouse bindings.
pub mod input;
/// Object manipulation controller (move, rotate, scale with constraints and numeric input).
pub mod manipulation;
/// Ray-cast object picking.
pub mod picking;
/// Multi-select system for viewport objects.
pub mod selection;
/// Typed sub-object reference and sub-object selection set.
pub mod sub_object;
/// Transform snapping helpers and constraint overlay types.
pub mod snap;
