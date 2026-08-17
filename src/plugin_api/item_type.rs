//! [`ItemTypePlugin`] trait and supporting types.
//!
//! A plugin that wants to ship a new scene item category implements
//! this trait, registers itself via
//! [`ViewportRenderer::with_item_type_plugin`](crate::renderer::ViewportRenderer::with_item_type_plugin),
//! and submits a per-frame collection through
//! [`SceneFrame::submit_plugin_items`](crate::renderer::SceneFrame::submit_plugin_items).
//!
//! Surface: registration, per-frame `prepare`, and drawing hooks the lib calls
//! inside its own passes: `paint` (opaque HDR scene), `paint_transparent`
//! (OIT), `outline_mask` (selection outline), `cast_shadow_pass` (shadow
//! cascades), `cull` (per-frame frustum cull), and `pick` / `render_pick` (CPU
//! and GPU picking). All hooks but `type_name` are default-empty; implement the
//! ones an item type needs.

use std::any::Any;

use crate::plugin_api::SharedBindings;
use crate::renderer::PickHit;
use crate::renderer::PickId;
use crate::scene::material::ItemSettings;

/// World-space ray used for picking queries against plugin items.
///
/// The lib computes the ray once from the click position + viewport
/// + view-projection matrix and forwards the same ray to every plugin's
/// [`ItemTypePlugin::pick`] method.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
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
#[non_exhaustive]
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
/// [`build_opaque_pipeline`](crate::resources::DeviceResources::build_opaque_pipeline)
/// drop in without further setup.
#[non_exhaustive]
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

/// Information forwarded to a plugin's `paint_depth_read`.
///
/// The read-only-depth pass runs after the opaque scene (and the built-in
/// sprite passes) and before OIT, with the scene depth attachment bound
/// read-only so the plugin's fragment shader can sample it. The pass has the
/// shared group-0 bind group bound on entry.
///
/// The context carries the scene depth as bindable resources. The fragment
/// shader samples the depth and reconstructs linear view-space depth through
/// [`SHARED_DEPTH_READ_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_READ_WGSL),
/// which matches the built-in soft-particle path so results agree with the
/// `Soft` sprite sub-mode.
///
/// Two ways to bind the depth, depending on whether the plugin has a spare
/// bind group (the default `max_bind_groups` is 4):
///
/// - **Fold into an existing group** (works even at four groups): declare the
///   depth texture + sampler at free binding indices inside a group the plugin
///   already owns, and build that group's bind group from
///   [`scene_depth`](Self::scene_depth) and
///   [`scene_depth_sampler`](Self::scene_depth_sampler).
/// - **Dedicated group** (when a group is free): bind the ready-made
///   [`scene_depth_bind_group`](Self::scene_depth_bind_group) at that group,
///   whose layout is
///   [`depth_read_bind_group_layout`](crate::resources::DeviceResources::depth_read_bind_group_layout);
///   list that layout at the matching slot of `extra_bind_group_layouts` when
///   building the pipeline via
///   [`build_depth_read_pipeline`](crate::resources::DeviceResources::build_depth_read_pipeline).
#[non_exhaustive]
pub struct DepthReadContext<'a> {
    /// Active render-camera snapshot for this viewport.
    pub camera: &'a crate::RenderCamera,
    /// Viewport extent in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Multi-viewport slot index.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter.
    pub frame_index: u64,
    /// Read-only depth-aspect view of the scene depth buffer (the opaque
    /// scene depth, plus the built-in sprite depth writes). Depth format,
    /// `TextureAspect::DepthOnly`. Bake this into a bind group of the plugin's
    /// own, or use the ready-made
    /// [`scene_depth_bind_group`](Self::scene_depth_bind_group).
    pub scene_depth: &'a crate::gpu::TextureView,
    /// Non-filtering clamp sampler matching the depth view. Pair it with
    /// [`scene_depth`](Self::scene_depth) when building a bind group; the same
    /// sampler backs [`scene_depth_bind_group`](Self::scene_depth_bind_group).
    pub scene_depth_sampler: &'a crate::gpu::Sampler,
    /// Ready-to-bind group carrying [`scene_depth`](Self::scene_depth) at
    /// binding 0 and [`scene_depth_sampler`](Self::scene_depth_sampler) at
    /// binding 1, matching
    /// [`depth_read_bind_group_layout`](crate::resources::DeviceResources::depth_read_bind_group_layout).
    /// A convenience for plugins with a spare bind group; bind it at that
    /// group. Plugins already using all four groups fold the two bindings into
    /// an existing group with [`scene_depth`](Self::scene_depth) +
    /// [`scene_depth_sampler`](Self::scene_depth_sampler) instead.
    pub scene_depth_bind_group: &'a crate::gpu::BindGroup,
}

/// Information forwarded to a plugin's `cast_shadow_pass`.
///
/// The lib's shadow render pass is already begun on entry. The plugin
/// builds its pipeline against
/// [`shadow_target_desc`](crate::resources::DeviceResources::shadow_target_desc)
/// and draws depth-only into the cascade tile selected by the lib's
/// `set_viewport` / `set_scissor_rect` calls.
#[non_exhaustive]
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
/// [`mask_target_desc`](crate::resources::DeviceResources::mask_target_desc)
/// and draws into it with any value `> 0` for covered pixels.
#[non_exhaustive]
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

/// Information forwarded to a plugin's `render_pick`.
///
/// The lib's pick-id render pass is already begun on entry, scissored to the
/// query region, with the shared group-0 camera bind group bound. The pass has
/// three colour targets (object id, primitive id, depth) plus the scene depth
/// format; the plugin builds its pipeline against
/// [`pick_target_desc`](crate::resources::DeviceResources::pick_target_desc)
/// (via [`build_pick_pipeline`](crate::resources::DeviceResources::build_pick_pipeline))
/// and writes each item's `PickId` at `@location(0)`. On read-back the lib maps
/// that id straight to a hit, so the plugin's items become GPU-pickable without
/// the CPU [`pick`](ItemTypePlugin::pick) fallback.
#[non_exhaustive]
pub struct PickPassContext<'a> {
    /// Active render-camera snapshot.
    pub camera: &'a crate::RenderCamera,
    /// Viewport extent in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Multi-viewport slot index.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter.
    pub frame_index: u64,
    /// The query's pick mask: which item types and sub-object levels the
    /// caller asked for. A plugin can pick a pipeline variant per level, or
    /// skip its draws when the mask holds nothing its items answer.
    pub mask: crate::renderer::PickMask,
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
/// Beyond `init_gpu`, `prepare`, and `paint`, the trait exposes
/// `paint_transparent`, `paint_depth_read`, `paint_foreground`,
/// `outline_mask`, `cull`, `cast_shadow_pass`, and `pick` as default-empty
/// hooks; implement the ones an item type needs.
///
/// Registration model: item-type plugins are singleton-by-type. The
/// `type_name` is the identity scene items reference to route themselves to a
/// renderer, so two plugins under the same name would be ambiguous and the
/// second registration replaces the first. This behavior is frozen. Runtime
/// and GPU plugins differ deliberately: they carry no external identity and
/// are multi-instance. See [`RuntimePlugin`](crate::runtime::RuntimePlugin)
/// and [`GpuPlugin`](crate::runtime::GpuPlugin).
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
    fn init_gpu(&mut self, _device: &crate::gpu::Device, _shared: &SharedBindings<'_>) {}

    /// Called when the wgpu device is recreated, e.g. after device loss or a
    /// host-driven reset. Every pipeline, buffer, texture, or bind group the
    /// plugin built against the old device is now invalid and must be rebuilt.
    ///
    /// The host invokes this via
    /// [`ViewportRenderer::notify_device_recreated`](crate::renderer::ViewportRenderer::notify_device_recreated);
    /// the renderer does not detect device loss on its own. After this call
    /// returns, [`init_gpu`](Self::init_gpu) is re-invoked on the plugin before
    /// it is next used, so a typical implementation drops its cached resources
    /// here and lets `init_gpu` rebuild them. Matches
    /// [`GpuPlugin::on_device_recreated`](crate::runtime::GpuPlugin::on_device_recreated).
    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {}

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
        _device: &crate::gpu::Device,
        _queue: &crate::gpu::Queue,
        _ctx: &ItemFrameContext<'_>,
        _items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
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
        _pass: &mut crate::gpu::RenderPass<'a>,
        _ctx: &PaintContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
    }

    /// Issue draw calls for transparent items in the OIT pass.
    ///
    /// Called inside the lib's `oit_pass` render pass with the standard
    /// group-0 bindings already bound. Implementations build their
    /// pipeline via
    /// [`build_oit_pipeline`](crate::resources::DeviceResources::build_oit_pipeline);
    /// the fragment shader must return
    /// [`OitOutput`](crate::plugin_api::shared_wgsl::SHARED_OIT_WGSL),
    /// writing both `@location(0)` (accum) and `@location(1)` (reveal).
    ///
    /// Plugins that ship only opaque items leave this empty.
    fn paint_transparent<'a>(
        &'a self,
        _pass: &mut crate::gpu::RenderPass<'a>,
        _ctx: &PaintContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
    }

    /// `true` when this plugin draws in the read-only-depth pass.
    ///
    /// The renderer opens that pass only when some plugin opts in;
    /// implementing [`paint_depth_read`](Self::paint_depth_read) without
    /// overriding this to `true` means the hook is never called, and a frame
    /// with no opted-in plugin issues no extra pass.
    fn draws_depth_read(&self) -> bool {
        false
    }

    /// Issue draw calls inside the lib's read-only-depth pass.
    ///
    /// The pass runs after the opaque scene (and the built-in sprite passes)
    /// and before OIT, with the scene depth attachment bound read-only so the
    /// plugin can sample it: the fragment shader compares its own fragment
    /// depth against the already-rendered opaque scene (soft particles,
    /// contact effects, depth-aware fog). The pass loads the HDR scene colour
    /// and blends over it; it tests against opaque depth but writes none.
    ///
    /// The pass has a group-0 bind group bound on entry. Build a compatible
    /// pipeline via
    /// [`build_depth_read_pipeline`](crate::resources::DeviceResources::build_depth_read_pipeline)
    /// and sample the depth through
    /// [`SHARED_DEPTH_READ_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_READ_WGSL).
    /// The plugin binds the scene depth in a group it owns: either the
    /// ready-made [`DepthReadContext::scene_depth_bind_group`] at a spare group,
    /// or, if it already uses all four bind groups, the two depth bindings
    /// folded into an existing group from
    /// [`DepthReadContext::scene_depth`] + [`DepthReadContext::scene_depth_sampler`].
    /// Plugins must restore group 0 if they rebind it.
    ///
    /// Only called when [`draws_depth_read`](Self::draws_depth_read) returns
    /// `true`.
    fn paint_depth_read<'a>(
        &'a self,
        _pass: &mut crate::gpu::RenderPass<'a>,
        _ctx: &DepthReadContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
    }

    /// `true` when this plugin draws in the foreground pass.
    ///
    /// The renderer opens the foreground pass only when foreground work
    /// exists; implementing [`paint_foreground`](Self::paint_foreground)
    /// without overriding this to `true` means the hook is never called.
    fn draws_foreground(&self) -> bool {
        false
    }

    /// Issue draw calls inside the lib's foreground pass.
    ///
    /// The foreground pass runs after the world is drawn. It loads the HDR
    /// colour and clears its own depth target, so foreground geometry draws
    /// over the scene without being occluded by or clipping into it. The
    /// pass has a group-0 bind group bound on entry whose camera at binding
    /// 0 carries the foreground projection (the scene projection, or the
    /// override from `EffectsFrame::foreground`) and whose clip planes are
    /// disabled; `ctx.camera` reflects the same projection. Plugins must
    /// restore group 0 if they rebind it. Build a compatible pipeline via
    /// [`build_foreground_pipeline`](crate::resources::DeviceResources::build_foreground_pipeline).
    ///
    /// Only called when [`draws_foreground`](Self::draws_foreground)
    /// returns `true`.
    fn paint_foreground<'a>(
        &'a self,
        _pass: &mut crate::gpu::RenderPass<'a>,
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
    /// [`build_mask_pipeline`](crate::resources::DeviceResources::build_mask_pipeline)
    /// to construct a compatible pipeline; the fragment helper
    /// [`SHARED_MASK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_MASK_WGSL)
    /// provides the trivial `fs_mask` body.
    ///
    /// Plugins that do not participate in the outline highlight leave
    /// this empty.
    fn outline_mask<'a>(
        &'a self,
        _pass: &mut crate::gpu::RenderPass<'a>,
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
    /// [`build_shadow_pipeline`](crate::resources::DeviceResources::build_shadow_pipeline).
    ///
    /// Implementations skip items where `item_settings(i).cast_shadows`
    /// is false. Default no-op: plugins that do not cast shadows leave
    /// this empty.
    fn cast_shadow_pass<'a>(
        &'a self,
        _pass: &mut crate::gpu::RenderPass<'a>,
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
    /// Plugins that prefer GPU picking can skip this method and implement
    /// [`render_pick`](Self::render_pick) instead, drawing their pick-ids
    /// into the shared pick pass so the lib's GPU pick path
    /// ([`pick_object`](crate::renderer::ViewportRenderer::pick_object) with
    /// [`PickBackend::Gpu`](crate::renderer::PickBackend::Gpu)) returns their
    /// items with no CPU ray-cast.
    fn pick(&self, _ray: &PickRay) -> Option<(f32, PickHit)> {
        None
    }

    /// Issue draw calls into the lib's pick-id pass.
    ///
    /// Called inside the pick render pass after the built-in geometry, with the
    /// shared group-0 camera bind group bound and the pass scissored to the
    /// query region. The plugin draws each pickable item with a pipeline whose
    /// three colour targets match the pass:
    /// [`pick_target_desc`](crate::resources::DeviceResources::pick_target_desc)
    /// spells them out, and
    /// [`SHARED_PICK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_WGSL)'s
    /// `viewport_pick_fs` writes the item's `pick_id` at `@location(0)` and the
    /// other two channels. A plugin with access to
    /// [`DeviceResources`](crate::resources::DeviceResources) can build the
    /// pipeline with `build_pick_pipeline`; one that only has `SharedBindings`
    /// (as at [`init_gpu`](Self::init_gpu)) hand-rolls it against those formats,
    /// the way the built-in pick pipelines do.
    ///
    /// Implementations skip hidden items and items whose
    /// [`pick_id`](PluginItemCollection::pick_id) is [`PickId::NONE`] (the clear
    /// value, which reads back as no hit). This is the GPU counterpart to
    /// [`pick`](Self::pick); a plugin implements whichever fits, or both.
    /// Default no-op: plugins relying on the CPU path leave this empty.
    ///
    /// For sub-object picking (face / vertex / edge on the drawn geometry),
    /// use the primitive-index fragment helper
    /// ([`SHARED_PICK_PRIM_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_PRIM_WGSL))
    /// instead of `viewport_pick_fs` when the device has
    /// [`PRIMITIVE_INDEX_FEATURE`](crate::gpu::PRIMITIVE_INDEX_FEATURE), and
    /// implement [`resolve_sub_object`](Self::resolve_sub_object) to map the
    /// read-back triangle index to the item's own sub-object ids.
    fn render_pick<'a>(
        &'a self,
        _pass: &mut crate::gpu::RenderPass<'a>,
        _ctx: &PickPassContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
    }

    /// Refine a GPU pick hit on one of this plugin's items to a sub-object.
    ///
    /// Called on read-back when a pick with a sub-object level in its mask
    /// (`FACE`, `VERTEX`, `EDGE`, ...) landed on an item this plugin drew via
    /// [`render_pick`](Self::render_pick). `primitive_index` is the value the
    /// plugin's pick fragment wrote into the primitive channel : the
    /// rasterised triangle index when the pipeline uses
    /// [`SHARED_PICK_PRIM_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_PRIM_WGSL)'s
    /// `viewport_pick_prim_fs`. `world_pos` is the hit's world position,
    /// reconstructed from the pick pass's depth channel, for snapping to the
    /// nearest vertex or edge of the hit triangle. `mask` is the query's full
    /// pick mask; answer the highest-priority level in it that the item type
    /// supports.
    ///
    /// The plugin owns its geometry, so only it can map a triangle index to a
    /// face / vertex / edge id. Return `None` for levels the item type does
    /// not answer; the hit then stays object-level. Default: `None` (all
    /// plugin picking stays object-level, as before this hook existed).
    ///
    /// Only called when the device has
    /// [`PRIMITIVE_INDEX_FEATURE`](crate::gpu::PRIMITIVE_INDEX_FEATURE);
    /// without it the primitive channel is all zeros and no refinement runs,
    /// matching the built-in surface fallback.
    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        _primitive_index: u32,
        _world_pos: glam::Vec3,
        _mask: crate::renderer::PickMask,
    ) -> Option<crate::renderer::SubObjectRef> {
        None
    }
}
