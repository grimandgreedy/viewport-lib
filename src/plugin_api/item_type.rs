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

use crate::interaction::picking::PickHit;
use crate::plugin_api::SharedBindings;
use crate::renderer::PickId;
use crate::scene::material::ItemSettings;

/// World-space ray used for picking queries against plugin items.
///
/// The lib computes the ray once from the click position + viewport
/// + view-projection matrix and forwards the same ray to every plugin's
/// [`ItemTypePlugin::pick`] method.
#[derive(Clone, Copy, Debug)]
pub struct PickRay {
    /// World-space origin (camera eye in perspective; near-plane point in ortho).
    pub origin: glam::Vec3,
    /// World-space direction. Not required to be unit-length; plugins
    /// should normalize if their hit test depends on it.
    pub direction: glam::Vec3,
}

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
/// Read-only camera + viewport state, a frame counter, and a handle to
/// the upload-job runner. Plugins use the runner the same way built-in
/// uploads do: submit CPU work that returns a typed value, poll the
/// returned `JobId` next frame, and take the result when the status
/// reports `Ready`. See [`crate::resources::Jobs`] for the surface.
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
    /// Handle to the upload-job runner. Plugins call
    /// `ctx.jobs.submit_cpu(...)` to spawn background work and
    /// `ctx.jobs.take::<T>(id)` to retrieve the result once the matching
    /// `status` returns `Ready`.
    pub jobs: crate::resources::Jobs<'a>,
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

/// Information forwarded to a plugin's `cast_shadow_pass`.
///
/// The lib's shadow render pass is already begun on entry. The plugin
/// builds its pipeline against
/// [`shadow_target_desc`](crate::resources::ViewportGpuResources::shadow_target_desc)
/// and draws depth-only into the cascade tile selected by the lib's
/// `set_viewport` / `set_scissor_rect` calls.
pub struct ShadowCastContext<'a> {
    /// Cascade index, in `0..cascade_count`.
    pub cascade_idx: u32,
    /// Light-space view-projection matrix for this cascade.
    pub light_view_proj: glam::Mat4,
    /// Active render-camera snapshot.
    pub camera: &'a crate::RenderCamera,
    /// Multi-viewport slot index.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter.
    pub frame_index: u64,
}

/// Information forwarded to a plugin's `outline_mask`.
///
/// The lib's outline-mask render pass is already begun on entry and has
/// the shared group-0 camera bind group bound. The pass targets a single
/// `R8Unorm` colour attachment and the scene depth buffer; the plugin
/// builds its mask pipeline against
/// [`mask_target_desc`](crate::resources::ViewportGpuResources::mask_target_desc)
/// and draws into it with any value `> 0` for covered pixels.
pub struct OutlineMaskContext<'a> {
    /// Active render-camera snapshot.
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

    /// Issue draw calls for transparent items in the OIT pass.
    ///
    /// Called inside the lib's `oit_pass` render pass with the standard
    /// group-0 bindings already bound. Implementations build their
    /// pipeline via
    /// [`build_oit_pipeline`](crate::resources::ViewportGpuResources::build_oit_pipeline);
    /// the fragment shader must return
    /// [`OitOutput`](crate::plugin_api::shared_wgsl::SHARED_OIT_WGSL),
    /// writing both `@location(0)` (accum) and `@location(1)` (reveal).
    ///
    /// Plugins that ship only opaque items leave this empty.
    fn paint_transparent<'a>(
        &'a self,
        _pass: &mut wgpu::RenderPass<'a>,
        _ctx: &PaintContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
    }

    /// Issue draw calls into the lib's outline-mask render pass.
    ///
    /// Called once per frame inside the mask pass with the standard
    /// group-0 bind group bound. Implementations iterate `items`, draw
    /// only those whose `item_settings(i).selected` is `true`, and write
    /// any non-zero R8 value at covered fragments. Use
    /// [`build_mask_pipeline`](crate::resources::ViewportGpuResources::build_mask_pipeline)
    /// to construct a compatible pipeline; the fragment helper
    /// [`SHARED_MASK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_MASK_WGSL)
    /// provides the trivial `fs_mask` body.
    ///
    /// Plugins that do not participate in the outline highlight leave
    /// this empty.
    fn outline_mask<'a>(
        &'a self,
        _pass: &mut wgpu::RenderPass<'a>,
        _ctx: &OutlineMaskContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
    }

    /// Update internal culling state for the current camera frustum.
    ///
    /// Called once per frame before `paint`. The plugin records which
    /// items are inside the frustum so subsequent `paint` /
    /// `paint_transparent` / `cast_shadow_pass` calls can skip culled
    /// entries. The lib does not pre-filter the collection; the plugin
    /// owns the visibility decision.
    ///
    /// Default no-op: plugins that draw everything every frame leave
    /// this empty.
    fn cull(
        &mut self,
        _frustum: &crate::camera::frustum::Frustum,
        _ctx: &ItemFrameContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
    }

    /// Issue depth-only draw calls into one cascade tile of the lib's
    /// shadow atlas.
    ///
    /// Called once per active cascade inside the lib's `shadow_pass`
    /// render pass. The lib has already set the viewport + scissor rect
    /// for the cascade tile and bound the cascade-space camera at group
    /// 0. The plugin draws depth-only with a pipeline built via
    /// [`build_shadow_pipeline`](crate::resources::ViewportGpuResources::build_shadow_pipeline).
    ///
    /// Implementations skip items where `item_settings(i).cast_shadows`
    /// is false. Default no-op: plugins that do not cast shadows leave
    /// this empty.
    fn cast_shadow_pass<'a>(
        &'a self,
        _pass: &mut wgpu::RenderPass<'a>,
        _ctx: &ShadowCastContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
    }

    /// Hit-test this plugin's items against `ray` and return the closest
    /// hit, paired with its world-space time-of-impact.
    ///
    /// Called from the renderer's pick router after the built-in CPU and
    /// GPU pick paths. The router compares the returned `t` against every
    /// other candidate and chooses the closest.
    ///
    /// Implementations typically cache the world-space AABBs (or per-item
    /// geometry) on their own state during [`prepare`](Self::prepare); the
    /// trait method does not receive the per-frame collection because
    /// pick happens out-of-band with prepare and the plugin already has
    /// the data it needs.
    ///
    /// Plugins that prefer GPU picking can skip this method and instead
    /// render their pick-ids into the standard pick-id pass (a future
    /// hook), at which point the lib's GPU pick path returns plugin
    /// items transparently.
    fn pick(&self, _ray: &PickRay) -> Option<(f32, PickHit)> {
        None
    }
}
