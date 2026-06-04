//! [`ItemTypePlugin`] trait and supporting types.
//!
//! A plugin that wants to ship a new scene item category implements
//! this trait, registers itself via
//! [`ViewportRenderer::with_item_type_plugin`](crate::renderer::ViewportRenderer::with_item_type_plugin),
//! and submits a per-frame collection through
//! [`SceneFrame::submit_plugin_items`](crate::renderer::SceneFrame::submit_plugin_items).
//!
//! Current surface: registration, per-frame `prepare`, and `paint` inside
//! the main HDR scene pass. Picking, outline-mask, shadow casting, frustum
//! cull, and transparent OIT integration arrive as additional
//! default-empty trait methods as those features are added.

use std::any::Any;

use crate::plugin_api::SharedBindings;
use crate::renderer::PickId;
use crate::scene::material::ItemSettings;

/// Per-frame item collection owned by the consumer and read by the lib.
///
/// A plugin defines its own collection type (typically a wrapper around a
/// `Vec<MyItem>`), implements this trait on it, and submits it via
/// [`SceneFrame::submit_plugin_items`](crate::renderer::SceneFrame::submit_plugin_items).
///
/// The lib reads the collection through this trait object during prepare
/// and paint; the plugin downcasts back to its concrete type inside
/// [`ItemTypePlugin::prepare`] and [`ItemTypePlugin::paint`] using
/// [`Any::downcast_ref`].
///
/// Implementations must be `Send + Sync` so the collection survives the
/// `SceneFrame` lifetime (Frame submission may be moved across threads in
/// some host setups).
pub trait PluginItemCollection: Any + Send + Sync {
    /// Number of items in the collection.
    fn len(&self) -> usize;

    /// `true` when [`len`](Self::len) is 0. Default reads `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `ItemSettings` for the item at `index`.
    ///
    /// The lib uses this to honour `hidden` (skip the item entirely),
    /// `selected` (route to the outline mask once outline integration
    /// lands), and other shared flags. The plugin is free to ignore
    /// `wireframe`, `unlit`, etc. for item types where they have no
    /// meaning.
    fn item_settings(&self, index: usize) -> &ItemSettings;

    /// Pick id for the item at `index`. Returns [`PickId::NONE`] when
    /// the item is not pickable.
    fn pick_id(&self, index: usize) -> PickId {
        self.item_settings(index).pick_id
    }

    /// Cast to `&dyn Any` so a plugin's `prepare` / `paint` can downcast
    /// back to its concrete collection type. Implementations should
    /// return `self`.
    fn as_any(&self) -> &dyn Any;
}

/// Information forwarded to a plugin's per-frame `prepare`.
///
/// Read-only camera + viewport state plus a frame counter. Plugins read
/// these to decide what to upload this frame; they do not modify them.
pub struct ItemFrameContext<'a> {
    /// Active render-camera snapshot for this viewport: view + projection
    /// matrices, eye position, near/far. Use
    /// `camera.view_proj()` for the combined matrix.
    pub camera: &'a crate::RenderCamera,
    /// Viewport extent in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Multi-viewport slot index. `0` for single-viewport apps.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter assigned by the lib.
    pub frame_index: u64,
}

/// Information forwarded to a plugin's `paint`.
///
/// The render pass is the lib's `hdr_scene_pass` with the shared group-0
/// bind group already bound. Plugin pipelines built via
/// [`build_opaque_pipeline`](crate::resources::ViewportGpuResources::build_opaque_pipeline)
/// drop in without further setup.
pub struct PaintContext<'a> {
    /// Active render-camera snapshot for this viewport.
    pub camera: &'a crate::RenderCamera,
    /// Viewport extent in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Multi-viewport slot index.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter.
    pub frame_index: u64,
}

/// A new scene item category supplied by a plugin.
///
/// Implementations are stored on the renderer via
/// [`ViewportRenderer::with_item_type_plugin`](crate::renderer::ViewportRenderer::with_item_type_plugin)
/// keyed by [`type_name`](Self::type_name). The lib invokes
/// [`init_gpu`](Self::init_gpu) once on registration, then
/// [`prepare`](Self::prepare) and [`paint`](Self::paint) every frame
/// against the collection submitted under the same name on
/// [`SceneFrame`](crate::renderer::SceneFrame).
///
/// The current surface is `init_gpu`, `prepare`, and `paint`. Picking,
/// outline-mask, shadow casting, and culling arrive as additional
/// default-empty methods when those integrations land.
pub trait ItemTypePlugin: Send + Sync + 'static {
    /// Stable name used as the [`SceneFrame::plugin_items`](crate::renderer::SceneFrame::plugin_items)
    /// key. Each registered plugin must have a unique name; registering a
    /// second plugin with the same name replaces the first.
    fn type_name(&self) -> &'static str;

    /// Build pipelines, persistent buffers, and bind group layouts. Called
    /// once when the plugin is registered with a renderer that has a
    /// device.
    ///
    /// The plugin reuses `shared.group0_layout` as group 0 of its pipeline
    /// layouts. See [`SharedBindings`](crate::plugin_api::SharedBindings)
    /// for the binding inventory.
    fn init_gpu(&mut self, _device: &wgpu::Device, _shared: &SharedBindings<'_>) {}

    /// Encode per-frame prepare work for this plugin's items.
    ///
    /// Returns a `Vec<wgpu::CommandBuffer>` that the renderer concatenates
    /// into the main `prepare` submission. Use this to upload per-frame
    /// instance data, run compute passes that produce inputs for `paint`,
    /// etc.
    ///
    /// `items` is the consumer's collection submitted under this plugin's
    /// [`type_name`](Self::type_name). Downcast via
    /// [`PluginItemCollection::as_any`] inside the implementation:
    ///
    /// ```ignore
    /// let coll = items.as_any().downcast_ref::<MyCollection>().unwrap();
    /// ```
    ///
    /// If no collection has been submitted, the renderer skips this
    /// method for the frame.
    fn prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _ctx: &ItemFrameContext<'_>,
        _items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        Vec::new()
    }

    /// Issue draw calls inside the lib's HDR scene pass.
    ///
    /// Called after built-in opaque geometry and before the skybox. The
    /// pass has the shared group-0 bind group bound on entry; plugins
    /// must restore it if they rebind group 0 themselves.
    ///
    /// Implementations should treat hidden items
    /// (`items.item_settings(i).hidden == true`) as drawn-nothing; the
    /// lib does not pre-filter the collection.
    fn paint<'a>(
        &'a self,
        _pass: &mut wgpu::RenderPass<'a>,
        _ctx: &PaintContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
    }
}
