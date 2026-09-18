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
//! - [`StoringItemTypePlugin`]: holds the content it draws and is uploaded
//!   into from outside the crate, synchronously and through the job runner,
//!   so the host-side route to a plugin-owned store is exercised too.
//! - [`ConformanceItemTypePlugin`]: all of the above in one type, plus the parts no
//!   other fixture covers: a bind group held over a host texture across frames
//!   and revalidated with a `ResourceGate`, a CPU pick, a wireframe and a
//!   selection affordance. It is the one to read when writing an item type of
//!   your own, and the one to extend when the seam grows.

mod collection;
mod conformance;
mod logging;
mod storing;
mod triangle;

pub use collection::CountedItemCollection;
pub use conformance::{ConformanceItemTypePlugin, ConformanceItems, QuadId};
pub use logging::LoggingItemTypePlugin;
pub use storing::{StoredId, StoringItemTypePlugin};
pub use triangle::TriangleItemTypePlugin;
