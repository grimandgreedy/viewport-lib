/// Action-based input system with mode-sensitive key/mouse bindings. Lives in
/// the `viewport-lib-input` crate; re-exported here so the renderer keeps its
/// `crate::interaction::input` path.
pub use viewport_lib_input::input;
/// Object manipulation controller (move, rotate, scale with constraints and numeric input).
pub mod manipulation;
/// Scene queries: ray-cast picking and transform snapping.
pub mod query;
/// Selection state: multi-select, sub-object references, and pick masks.
pub mod select;
/// Interactive 3D probe and region widgets (line probe, sphere, box).
pub mod widgets;
