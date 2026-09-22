//! Interaction: what a resolved input frame does to the scene.
//!
//! # Hit-test order inside one viewport
//!
//! Several library-drawn things can claim the same press, and they are not layered
//! by a widget tree: this crate draws them, it does not arbitrate between them. An
//! application without a widget system chains them itself, in this order, stopping
//! at the first hit:
//!
//! 1. **The axes indicator**, [`axes_indicator::hit_test`](crate::interaction::widgets::axes_indicator::hit_test). Screen-space,
//!    drawn over everything, and cheapest to test: a 2D distance against a fixed
//!    corner of the rect.
//! 2. **The clip-plane handles**, [`clip_plane::hit_test_plane_quad`] and
//!    [`hit_test_normal_handle`](clip_plane::hit_test_normal_handle), when a clip
//!    object is being edited.
//! 3. **The transform gizmo**, [`manipulation::gizmo::Gizmo::hit_test`]. Drawn over
//!    the scene and depth-independent, so it wins against the geometry under it.
//! 4. **Scene picking**, [`query`](crate::interaction::query). Whatever is left.
//!
//! Application widgets drawn over the viewport are hit-tested before any of these,
//! because only the application can see them. Layering several interactive tools in
//! one pane is a widget system; viewport-lib-ui is one, and this crate will not grow
//! a second.

/// Action-based input system with mode-sensitive key/mouse bindings. Lives in
/// the `viewport-lib-input` crate; re-exported here so the renderer keeps its
/// `crate::interaction::input` path.
pub use viewport_lib_input::input;
/// Interactive clip-object manipulator: position and orient section planes (and the
/// other clip shapes' visuals). Kept separate from `manipulation` (the gizmo) so the
/// two can be lifted to separate companion crates independently.
pub mod clip_plane;
/// Object manipulation controller (move, rotate, scale with constraints and numeric input).
pub mod manipulation;
/// Scene queries: ray-cast picking and transform snapping.
pub mod query;
/// Selection state: multi-select, sub-object references, and pick masks.
pub mod select;
/// Interactive 3D probe and region widgets (line probe, sphere, box).
pub mod widgets;
