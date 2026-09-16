//! Hit-test helpers for plugin pick implementations.
//!
//! Re-exports of the same routines the built-in CPU pickers use, so a
//! plugin item type's [`pick`](crate::plugin_api::ItemTypePlugin::pick) and
//! [`pick_rect`](crate::plugin_api::ItemTypePlugin::pick_rect) match the
//! built-in item types' pick tolerances and edge-case behaviour
//! (behind-camera rejection, screen-space distance metric, strip-aware
//! segment indexing).
//!
//! - [`project_to_screen`]: world point to pixel coordinates, the common
//!   first step of every screen-space proximity test.
//! - [`pick_closest_polyline_segment`]: closest segment of a set of strips
//!   to a click position, within a pixel threshold.
//! - [`segment_in_rect`]: 2D segment versus box-select rectangle.
//! - [`ray_triangle`]: Moller-Trumbore ray/triangle parameter.
//! - [`ray_unit_box_toi`]: ray versus the local unit box, for oriented-box
//!   proxies (transform the ray by the box's inverse model first).

pub use crate::renderer::picking::helpers::{
    pick_closest_polyline_segment, project_to_screen, ray_triangle, ray_unit_box_toi,
    segment_in_rect,
};
