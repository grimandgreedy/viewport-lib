//! Fixtures for [`ItemTypePlugin`](viewport_lib::plugin_api::ItemTypePlugin):
//! a consumer item type the renderer dispatches inside its own frame,
//! registered with `ViewportRenderer::with_item_type_plugin` and fed per frame
//! through `SceneFrame::submit_plugin_items`.
//!
//! - [`CountedItemCollection`]: the collection the fixtures are submitted
//!   with, shared by any variety here.
//! - [`LoggingItemTypePlugin`]: records the dispatch it receives and draws
//!   nothing.

mod collection;
mod logging;

pub use collection::CountedItemCollection;
pub use logging::LoggingItemTypePlugin;
