//! Fixtures for [`ItemTypePlugin`](viewport_lib::plugin_api::ItemTypePlugin):
//! a consumer item type the renderer dispatches inside its own frame,
//! registered with `ViewportRenderer::with_item_type_plugin` and fed per frame
//! through `SceneFrame::submit_plugin_items`.
//!
//! - [`CountedItemCollection`]: the collection the fixtures are submitted
//!   with, shared by any variety here.
//! - [`LoggingItemTypePlugin`]: records the dispatch it receives and draws
//!   nothing.
//! - [`TriangleItemTypePlugin`]: draws a triangle in the opaque, shadow, and
//!   pick passes through the plugin pipeline builders, so the shared bind
//!   layout and the target descriptors are exercised from outside the crate.

mod collection;
mod logging;
mod triangle;

pub use collection::CountedItemCollection;
pub use logging::LoggingItemTypePlugin;
pub use triangle::TriangleItemTypePlugin;
