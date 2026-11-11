//! [`ItemTypePlugin`] trait and supporting types.
//!
//! A plugin that wants to ship a new scene item category implements
//! this trait, registers itself via
//! [`ViewportRenderer::with_item_type_plugin`](crate::renderer::ViewportRenderer::with_item_type_plugin),
//! and submits a per-frame collection through
//! [`SceneFrame::submit_plugin_items`](crate::renderer::SceneFrame::submit_plugin_items).
//!
//! Surface: registration, per-frame `prepare`, and drawing hooks the lib calls
//! inside its own passes: `paint` (opaque scene, HDR always and LDR for
//! plugins that opt in via `draws_ldr`), `paint_transparent`
//! (OIT), `outline_mask` (selection outline), `cast_shadow_pass` (shadow
//! cascades), `cull` (per-frame frustum cull), and `pick` / `render_pick` (CPU
//! and GPU picking). An item type whose work does not fit inside a pass the lib
//! begins uses `encode` instead, which hands over the frame's command encoder
//! at the points named by `encoder_scopes`. All hooks but `type_name` are
//! default-empty; implement the ones an item type needs.

use std::any::Any;

use crate::plugin_api::PluginResources;
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

/// Query information forwarded alongside the ray to a plugin's
/// [`ItemTypePlugin::pick`].
///
/// The ray is enough for solid geometry, but item types whose pick
/// tolerance lives in screen space (thin lines, point markers, screen-sized
/// handles) need the click position, viewport extent, and view-projection
/// to run a pixel-space proximity test; the helpers in
/// [`pick_helpers`](crate::plugin_api::pick_helpers) consume exactly these
/// fields. `mask` carries the query's requested levels so a plugin can
/// return `None` cheaply when the query asks for nothing its items answer.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct PickContext<'a> {
    /// Click position in the same pixel coordinates as `viewport_size`
    /// (origin top-left, y down).
    pub click_pos: glam::Vec2,
    /// Viewport extent in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Combined view-projection matrix the ray was derived from.
    pub view_proj: glam::Mat4,
    /// The query's pick mask.
    pub mask: crate::renderer::PickMask,
    /// Read handle for the CPU geometry of consumer-uploaded meshes. An item
    /// type whose geometry is a `MeshId` tests against these arrays; one that
    /// holds its own geometry ignores it.
    pub meshes: crate::resources::MeshGeometry<'a>,
}

/// Query information forwarded to a plugin's [`ItemTypePlugin::pick_rect`].
///
/// `rect_min` / `rect_max` are in pixel coordinates (origin top-left, y
/// down), the same space
/// [`project_to_screen`](crate::plugin_api::pick_helpers::project_to_screen)
/// maps into.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct RectPickContext<'a> {
    /// Rectangle minimum corner in pixels.
    pub rect_min: glam::Vec2,
    /// Rectangle maximum corner in pixels.
    pub rect_max: glam::Vec2,
    /// Viewport extent in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Combined view-projection matrix.
    pub view_proj: glam::Mat4,
    /// The query's pick mask.
    pub mask: crate::renderer::PickMask,
    /// Read handle for the CPU geometry of consumer-uploaded meshes. An item
    /// type whose geometry is a `MeshId` tests against these arrays; one that
    /// holds its own geometry ignores it.
    pub meshes: crate::resources::MeshGeometry<'a>,
}

/// One submittable item of an item type, tying the item struct to the type
/// name its plugin registered under.
///
/// Implementing this is what lets a `Vec<T>` be submitted with
/// [`SceneFrame::items_mut`](crate::renderer::SceneFrame::items_mut) and read
/// back by a plugin with [`ItemFrameContext::items_of`], with no per-type
/// wiring in the lib. An item type may have several item structs sharing one
/// `TYPE_NAME`: an inline form carrying its data and a reference form naming
/// pre-uploaded content, say, or a second reference form. They travel together
/// under the name and the plugin tells them apart by downcasting.
pub trait PluginItem: Any + Send + Sync + 'static {
    /// The registering plugin's [`ItemTypePlugin::type_name`].
    const TYPE_NAME: &'static str;

    /// Shared per-item flags (`hidden`, `selected`, `pick_id`, ...).
    fn settings(&self) -> &ItemSettings;
}

impl<T: PluginItem> PluginItemCollection for Vec<T> {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn item_settings(&self, index: usize) -> &ItemSettings {
        self[index].settings()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Every collection submitted under one item type's name this frame.
///
/// An item type may have several item forms sharing a type name: an inline
/// form carrying its data, a reference form naming pre-uploaded content, a
/// second reference form. They arrive together and a hook picks the one it
/// wants by type with [`of`](Self::of), so submission order does not matter.
#[derive(Clone, Copy)]
pub struct ItemCollections<'a>(&'a [Box<dyn PluginItemCollection>]);

impl<'a> ItemCollections<'a> {
    /// Wrap the collections submitted under one name.
    pub fn new(collections: &'a [Box<dyn PluginItemCollection>]) -> Self {
        Self(collections)
    }

    /// This frame's items of type `T`, or an empty slice if none were
    /// submitted.
    pub fn of<T: 'static>(&self) -> &'a [T] {
        self.0
            .iter()
            .find_map(|c| c.as_any().downcast_ref::<Vec<T>>())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// The submitted collection whose concrete type is `C`, for an item type
    /// that submits its own collection type rather than a plain `Vec`.
    pub fn downcast<C: PluginItemCollection>(&self) -> Option<&'a C> {
        self.0.iter().find_map(|c| c.as_any().downcast_ref::<C>())
    }

    /// Total items across every form.
    pub fn len(&self) -> usize {
        self.0.iter().map(|c| c.len()).sum()
    }

    /// `true` when no form carried anything.
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|c| c.is_empty())
    }

    /// Each submitted collection in turn.
    pub fn iter(&self) -> impl Iterator<Item = &'a dyn PluginItemCollection> {
        self.0.iter().map(|b| b.as_ref())
    }

    /// Settings for the item at `index`, counting across every form in
    /// submission order, so the range `0..len()` addresses each item once.
    pub fn item_settings(&self, index: usize) -> &'a ItemSettings {
        let mut rest = index;
        for c in self.0 {
            if rest < c.len() {
                return c.item_settings(rest);
            }
            rest -= c.len();
        }
        panic!("item index {index} out of range for {} items", self.len());
    }

    /// Pick id for the item at `index`, indexed as
    /// [`item_settings`](Self::item_settings).
    pub fn pick_id(&self, index: usize) -> PickId {
        self.item_settings(index).pick_id
    }
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
/// [`Any::downcast_ref`](std::any::Any#method.downcast_ref).
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

    /// Cast to `&mut dyn Any`, so a host can reach a submitted collection
    /// again to push into it. Implementations should return `self`.
    fn as_any_mut(&mut self) -> &mut dyn Any;
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
    /// The time this frame renders at: the value the shaders read as
    /// `camera.time`, with `seconds` in f64. It is `FrameData::time` when the
    /// host set one, and the renderer's own clock otherwise. Drive CPU-side
    /// animation (a tile frame, a flipbook index) from `time.seconds` so it
    /// stays in step with what the shaders draw.
    pub time: crate::FrameTime,
    /// The host's scene version counter for this frame (`SceneFrame::generation`).
    ///
    /// A plugin that rebuilds per-frame state from an inline collection can keep
    /// the value it last prepared at and skip the rebuild when this matches. That
    /// is what the instanced mesh path does with the same counter.
    ///
    /// **It is the host's counter, not the library's, and it says nothing on its
    /// own.** A host that never bumps it reports `0` every frame, and one that
    /// bumps it per frame reports a new value whether or not anything moved. It is
    /// also per frame rather than per collection: a bump means something in the
    /// scene changed, not that your items did.
    ///
    /// Because of that, never gate on the generation alone. Pair it with something
    /// that catches the case the host got wrong, as the instanced path pairs it
    /// with the item count: equal generation *and* equal count is the condition
    /// that is safe to skip on, and even then a host that mutates items in place
    /// without bumping draws stale content. If skipping is wrong for your type,
    /// ignore this field. Rebuilding every frame is the correct default and is
    /// what every built-in item type does.
    pub scene_generation: u64,
    /// Handle to the upload-job runner. Plugins call
    /// `ctx.jobs.submit_cpu(...)` to spawn background work and
    /// `ctx.jobs.take::<T>(id)` to retrieve the result once the matching
    /// `status` returns `Ready`.
    pub jobs: crate::resources::Jobs<'a>,
    /// Read-only view of the renderer's device resources for the duration of
    /// this call: texture views and samplers by id
    /// ([`texture_view`](crate::resources::DeviceResources::texture_view) /
    /// [`has_texture`](crate::resources::DeviceResources::has_texture)),
    /// colourmap LUT views
    /// ([`colourmap_view`](crate::resources::DeviceResources::colourmap_view)),
    /// and the target descriptors and samplers the pipeline builders expose.
    /// Borrows taken from it must not outlive `prepare` / `cull`; bake what
    /// the draw hooks need into the plugin's own bind groups here.
    ///
    /// A bind group baked here outlives the frame it was built in, so a plugin
    /// that bakes a texture view keeps a
    /// [`ResourceGate`](crate::resources::ResourceGate) beside its store and
    /// polls it against this before drawing: a texture the host has since freed
    /// is otherwise pinned by the bind group and still sampled, and a
    /// replacement swapped in behind a live id never arrives.
    pub resources: &'a crate::resources::DeviceResources,
    /// The frame's global wireframe toggle
    /// (`ViewportFrame::wireframe_mode`). Plugins whose item types have a
    /// wireframe representation switch pipelines on it the way the built-in
    /// types do; others ignore it.
    pub wireframe_mode: bool,
    /// Whether selected items should render the selection outline this frame
    /// (`InteractionFrame::outline_selected`). When `false`, a plugin can
    /// skip preparing outline-mask state entirely.
    pub outline_selected: bool,
    /// The frame's sub-object selection, when the host submitted one
    /// (`InteractionFrame::sub_selection`). Plugins that support sub-object
    /// highlighting read their own items' entries out of it.
    pub sub_selection: Option<&'a crate::renderer::SubSelectionRef>,
    /// The frame's clip objects (`EffectsFrame::clip`). Group 0 already carries
    /// the clip bindings, so a plugin whose shader reads them needs nothing
    /// here; this is for a plugin that bakes clipping into a uniform of its
    /// own, which a ray-marcher generally must, since it tests along the ray
    /// rather than per fragment.
    pub clip_objects: &'a [crate::renderer::ClipObject],
    /// Device pixels per logical pixel of the scene target, so
    /// `viewport_size * pixels_per_point` is the scene's size in device
    /// pixels: the viewport's scale factor, or one under a reference
    /// presentation.
    pub pixels_per_point: f32,
    /// The frame's light sources (`EffectsFrame::lighting.lights`), in world
    /// coordinates and in submission order, before the renderer ranks or caps
    /// them. For an item type that lights its own content from the scene's
    /// lights rather than through the group-0 light list.
    pub lights: &'a [crate::LightSource],
    /// `true` when the frame budget asked for reduced quality this frame
    /// (reported as `FrameStats::volume_quality_reduced`). A plugin with a
    /// cost knob, such as a march step count or a sample budget, should turn
    /// it down while this is set; one without ignores it. The value reflects
    /// the previous frame's measurement, since this frame's has not been
    /// taken yet.
    pub quality_reduced: bool,
    /// Every collection submitted under this item type's name this frame.
    ///
    /// One item type may submit several: an inline form carrying its data, a
    /// reference form naming pre-uploaded content, and so on. Sprite submits
    /// three. Read them by type with [`items_of`](Self::items_of) rather than
    /// by index; the order they were submitted in is not meaningful.
    pub collections: &'a [Box<dyn PluginItemCollection>],
    /// See [`world_origin`](Self::world_origin).
    pub(crate) world_origin: glam::DVec3,
}

impl<'a> ItemFrameContext<'a> {
    /// The origin this frame draws relative to, in world coordinates.
    ///
    /// The group-0 camera's matrices and eye, the light positions and the
    /// clip planes are all relative to it, while [`camera`](Self::camera)
    /// and the submitted items stay in world coordinates. An item type that
    /// writes world-space positions or model matrices into its own buffers
    /// subtracts this in f64 before the cast to f32
    /// (`(DVec3::from(p) - ctx.world_origin()).as_vec3()`), and rewrites them
    /// when it moves. It is zero under
    /// [`WorldOrigin::Zero`](crate::WorldOrigin::Zero); under the default
    /// [`snapped`](crate::WorldOrigin::snapped) origin it follows the camera.
    ///
    /// A model matrix written each frame takes the relative translation
    /// directly. Data uploaded once can stay as it is: draw it with a per-item
    /// offset, `(upload_origin - ctx.world_origin()).as_vec3()` where
    /// `upload_origin` is the origin at the upload, or rewrite it only when
    /// this value changes. CPU work that must agree with group 0 (a scissor
    /// from a projected box, a reprojection matrix) builds the camera's view
    /// relative to this origin rather than using [`camera`](Self::camera)
    /// as it is.
    pub fn world_origin(&self) -> glam::DVec3 {
        self.world_origin
    }

    /// This frame's items of type `T`, or an empty slice if the frame carried
    /// none. A type with several item forms calls this once per form.
    pub fn items_of<T: 'static>(&self) -> &'a [T] {
        self.collections
            .iter()
            .find_map(|c| c.as_any().downcast_ref::<Vec<T>>())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Alias of [`items_of`](Self::items_of), for reading a reference form.
    /// The two are the same lookup: a reference item is an item.
    pub fn refs_of<T: 'static>(&self) -> &'a [T] {
        self.items_of::<T>()
    }

    /// Report `detail` as a notice in the [`Item`](crate::NoticeCategory::Item)
    /// category, for a condition the notice vocabulary already names.
    ///
    /// Reports with the same `key` coalesce into one notice with a count, so
    /// reporting every frame is fine. `detail` is called only for the first
    /// report of `key` since the host last took the notices.
    pub fn report_detail(
        &self,
        severity: crate::Severity,
        key: &'static str,
        detail: impl FnOnce() -> crate::NoticeDetail,
    ) {
        self.resources.notices.report(
            crate::notice::NoticeKey::new(
                crate::NoticeCategory::Item,
                key,
                [crate::notice::name_id(key), 0],
            ),
            severity,
            detail,
        );
    }

    /// Report a condition the host should hear about, such as items this type
    /// skipped. It reaches the host through `ViewportRenderer::take_notices` as a
    /// [`NoticeDetail::Plugin`](crate::NoticeDetail::Plugin) in the
    /// [`Item`](crate::NoticeCategory::Item) category.
    ///
    /// Reports with the same `key` coalesce into one notice with a count, so
    /// reporting every frame is fine. `message` is called only for the first
    /// report of `key` since the host last took the notices, and not at all when
    /// `severity` is below the renderer's notice level.
    pub fn report_notice(
        &self,
        severity: crate::Severity,
        key: &'static str,
        message: impl FnOnce() -> String,
    ) {
        self.resources.notices.report(
            crate::notice::NoticeKey::new(
                crate::NoticeCategory::Item,
                "plugin",
                [crate::notice::name_id(key), 0],
            ),
            severity,
            || crate::NoticeDetail::Plugin {
                key,
                message: message(),
            },
        );
    }
}

/// Information forwarded to a plugin's `paint`.
///
/// The render pass is the lib's scene pass (HDR or LDR) with the shared
/// group-0 bind group already bound. Plugin pipelines built via
/// [`build_opaque_pipeline`](crate::resources::DeviceResources::build_opaque_pipeline)
/// drop into the HDR pass without further setup; a plugin that also opts
/// into the LDR pass via [`ItemTypePlugin::draws_ldr`] keeps a second
/// pipeline built against
/// [`ldr_opaque_target_desc`](crate::resources::DeviceResources::ldr_opaque_target_desc)
/// and selects between the two by [`is_hdr_pass`](Self::is_hdr_pass).
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
    /// Colour format of the pass's first colour target: the HDR scene format
    /// (`Rgba16Float`) in `paint` on the HDR path and in `paint_transparent`
    /// (the OIT accumulation target) and `paint_foreground`; the renderer's
    /// configured output format in `paint` on the LDR path. In `paint` and
    /// `paint_foreground` on the HDR path the pass has a second target, the
    /// indirect fraction (see
    /// [`OpaqueTargetDesc`](crate::plugin_api::OpaqueTargetDesc)). Choose the
    /// pipeline variant with [`is_hdr_pass`](Self::is_hdr_pass): the LDR
    /// output format can be `Rgba16Float` too.
    pub target_format: crate::gpu::TextureFormat,
    /// Draw handle for meshes the consumer uploaded through
    /// [`upload_mesh_data`](crate::resources::DeviceResources::upload_mesh_data).
    /// An item type whose geometry is a `MeshId` binds and draws it through
    /// this; one that owns its buffers ignores it.
    pub meshes: crate::resources::MeshDraw<'a>,
    /// Whether the pass is one of the HDR path's passes.
    pub(crate) hdr_pass: bool,
}

impl PaintContext<'_> {
    /// `true` when the pass belongs to the HDR path: `paint` on the HDR path,
    /// `paint_transparent` and `paint_foreground`. `false` for `paint` on the
    /// LDR path, whose single colour target is the renderer's output format.
    ///
    /// Select the HDR or LDR pipeline variant with this rather than by
    /// comparing [`target_format`](Self::target_format) with
    /// [`HDR_COLOR_FORMAT`](crate::resources::HDR_COLOR_FORMAT): a renderer
    /// whose output is `Rgba16Float` draws its LDR pass in that format too.
    pub fn is_hdr_pass(&self) -> bool {
        self.hdr_pass
    }
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
///
/// Either way, declare the depth as `texture_2d<f32>` (the layout entry is
/// unfilterable float, see
/// [`depth_texel_entry`](crate::plugin_api::builders::depth_texel_entry)) and
/// read it through the sampler: `textureSampleLevel(...).r`, or `depth_texel`
/// from [`SHARED_DEPTH_TEXEL_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_TEXEL_WGSL)
/// for one texel. A `texture_depth_2d` read with `textureLoad` or a
/// non-comparison sample does not build on wgpu's GL backend, and
/// `textureLoad` on the `texture_2d<f32>` reads 0 on GLES.
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
    /// Non-filtering, nearest, clamp sampler matching the depth view. Pair it with
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

/// Per-call context for [`ItemTypePlugin::paint_painter`].
#[non_exhaustive]
pub struct PainterContext<'a> {
    /// Active render-camera snapshot for this viewport.
    pub camera: &'a crate::RenderCamera,
    /// Viewport extent in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Multi-viewport slot index.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter.
    pub frame_index: u64,
    /// The painter layers this call draws, inclusive. For a call with
    /// [`batches`](Self::batches), the layers those batches span.
    pub layers: std::ops::RangeInclusive<u32>,
    /// For a plugin that reports
    /// [`painter_batches`](ItemTypePlugin::painter_batches): the batch handles
    /// this call draws, in the order to draw them. `None` for a plugin that
    /// reports none, which draws everything it has in
    /// [`layers`](Self::layers).
    pub batches: Option<&'a [u32]>,
    /// On the HDR path, the painter pass's ready-made scene depth bind group
    /// (laid out as [`DepthReadContext::scene_depth_bind_group`]); `None` in
    /// the LDR scene pass, whose depth attachment is writable.
    pub scene_depth_bind_group: Option<&'a crate::gpu::BindGroup>,
    /// On the full tier, a group laid out as
    /// [`depth_read_bind_group_layout`](crate::resources::DeviceResources::depth_read_bind_group_layout)
    /// in either pass: the scene depth on the HDR path (the same group as
    /// [`scene_depth_bind_group`](Self::scene_depth_bind_group)), a 1x1 depth
    /// in the LDR scene pass. Its bindings 2 and 3 carry this frame's painter
    /// masks, so a plugin with no spare group tests masks through it. `None`
    /// on the compatibility tier.
    pub depth_read_bind_group: Option<&'a crate::gpu::BindGroup>,
    /// This frame's painter masks, laid out as
    /// [`painter_mask_bind_group_layout`](crate::resources::DeviceResources::painter_mask_bind_group_layout),
    /// on every tier. Content that honours a [`MaskTest`](crate::MaskTest)
    /// binds it at a spare group and tests through
    /// [`SHARED_PAINTER_MASK_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PAINTER_MASK_WGSL).
    /// A frame with no masks binds an empty one.
    pub mask_bind_group: &'a crate::gpu::BindGroup,
    /// Draws an uploaded mesh by id, for a painter draw built on one.
    pub meshes: crate::resources::MeshDraw<'a>,
}

impl PainterContext<'_> {
    /// `true` in the HDR path's painter pass (`HDR_COLOR_FORMAT`, the
    /// renderer's sample count, one colour target, scene depth read-only),
    /// `false` in the LDR scene pass (the renderer's output format, one
    /// sample).
    pub fn is_hdr_pass(&self) -> bool {
        self.scene_depth_bind_group.is_some()
    }

    /// Whether `layer` is drawn by this call.
    pub fn includes(&self, layer: u32) -> bool {
        self.layers.contains(&layer)
    }
}

/// A painter mask a plugin draws this frame, as reported by
/// [`ItemTypePlugin::painter_masks`]: the slot its coverage goes into and the
/// painter layers it applies to.
///
/// Several masks may share a slot: their coverage is the union, and the
/// slot's layers run from the lowest first layer to the highest last layer
/// any of them names.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PainterMask {
    /// The slot, `0` to `PAINTER_MASK_SLOTS - 1`; higher values wrap.
    pub slot: u8,
    /// The painter layers content testing this slot is cut on.
    pub layers: std::ops::RangeInclusive<u32>,
}

impl PainterMask {
    /// A mask in `slot` over `layers`.
    pub fn new(slot: u8, layers: std::ops::RangeInclusive<u32>) -> Self {
        Self { slot, layers }
    }
}

/// Per-call context for [`ItemTypePlugin::paint_painter_masks`].
#[non_exhaustive]
pub struct PainterMaskContext<'a> {
    /// Active render-camera snapshot for this viewport.
    pub camera: &'a crate::RenderCamera,
    /// Viewport extent in logical pixels.
    pub viewport_size: glam::Vec2,
    /// Multi-viewport slot index.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter.
    pub frame_index: u64,
}

/// One of a plugin's painter's-order draws this frame, as reported by
/// [`ItemTypePlugin::painter_batches`].
///
/// The renderer sorts every plugin's batches into one sequence: ascending
/// `layer`, then ascending `sort_key`, then registration order of the plugins,
/// then the order each plugin reported them in. The renderer's own painter
/// content (mesh instances and surfaces with a
/// [`PainterPlacement`](crate::PainterPlacement)) sorts as if registered
/// before every plugin. A world overlay layer draws after every batch whose
/// layer is at or below its `order`.
///
/// A light layer (a batch built with [`light_layer`](Self::light_layer))
/// draws after every other batch and world overlay layer of its range, and
/// a batch marked [`unlit`](Self::unlit) on a layer in that range moves to
/// after it: the unlit content of a range draws above its lit content, in
/// its own layer and key order.
///
/// The sequence places each batch by its one key, so a batch must hold draws
/// at a single key. A plugin whose draw covers several keys reports it as one
/// batch per run of equal keys and draws consecutive batches of its own that
/// the sequence hands back together as one draw where it can, as the sprite
/// type does.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PainterBatch {
    /// Painter layer: lower layers draw first.
    pub layer: u32,
    /// Order within the layer, across plugins: lower keys draw first.
    /// Compared with `f32::total_cmp`. Compose it with
    /// [`painter_key`](crate::painter_key) to keep an entity's parts together.
    pub sort_key: f32,
    /// The plugin's own handle for the batch, handed back in
    /// [`PainterContext::batches`].
    pub batch: u32,
    /// Drawn after the light layer whose range holds `layer`, so 2D lighting
    /// leaves it as drawn. With no light layer over `layer` it draws in its
    /// place.
    pub unlit: bool,
    /// For a light layer: the first and last painter layer it lights. The
    /// batch draws at the top of that range, after its other content, and
    /// multiplies what is drawn by then. `None` for ordinary content.
    pub light_layers: Option<[u32; 2]>,
}

impl PainterBatch {
    /// A batch at `layer` with `sort_key`, named `batch` by the plugin.
    pub fn new(layer: u32, sort_key: f32, batch: u32) -> Self {
        Self {
            layer,
            sort_key,
            batch,
            unlit: false,
            light_layers: None,
        }
    }

    /// A batch at `placement`, named `batch` by the plugin, unlit when the
    /// placement is.
    pub fn at(placement: crate::PainterPlacement, batch: u32) -> Self {
        Self::new(placement.layer, placement.key, batch).with_unlit(placement.unlit)
    }

    /// This batch with [`unlit`](Self::unlit) set to `unlit`.
    pub fn with_unlit(mut self, unlit: bool) -> Self {
        self.unlit = unlit;
        self
    }

    /// A light layer over the painter layers `layers`, named `batch` by the
    /// plugin: placed at the range's last layer, after everything else drawn
    /// in the range and before the unlit content of the range.
    ///
    /// When two light layers name ranges that overlap, the sequence places
    /// unlit content by the first one reported.
    pub fn light_layer(layers: std::ops::RangeInclusive<u32>, batch: u32) -> Self {
        let (first, last) = (*layers.start(), *layers.end());
        Self {
            layer: last,
            sort_key: f32::INFINITY,
            batch,
            unlit: false,
            light_layers: Some([first, last]),
        }
    }
}

/// The painter's-order draws one
/// [`render_pick_painter`](ItemTypePlugin::render_pick_painter) call covers:
/// the same run of the sequence a
/// [`paint_painter`](ItemTypePlugin::paint_painter) call draws.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PainterRun<'a> {
    /// The painter layers this run covers, inclusive.
    pub layers: std::ops::RangeInclusive<u32>,
    /// For a plugin that reports
    /// [`painter_batches`](ItemTypePlugin::painter_batches): the batch handles
    /// of the run, in draw order. `None` for a plugin that reports none, which
    /// picks everything it has in [`layers`](Self::layers).
    pub batches: Option<&'a [u32]>,
}

impl PainterRun<'_> {
    /// Whether `layer` is covered by this run.
    pub fn includes(&self, layer: u32) -> bool {
        self.layers.contains(&layer)
    }
}

/// Where in the HDR frame an [`encode`](ItemTypePlugin::encode) call is
/// happening.
///
/// An item type that needs its own render or compute passes rather than draws
/// inside one the lib begins names the points it wants through
/// [`encoder_scopes`](ItemTypePlugin::encoder_scopes), and reads this off the
/// context to tell them apart when it asks for more than one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncoderScope {
    /// Right after the opaque scene and its supersample resolve, and before
    /// the selection sub-highlight and the depth-read pass.
    ///
    /// The scene colour holds the fully lit opaque image and the depth buffer
    /// is final for opaque geometry, but nothing non-opaque has drawn yet.
    /// A projection effect that stamps onto opaque surfaces (decals, projected
    /// textures) belongs here: its output is part of how the surface looks, so
    /// it has to sit underneath the selection affordances and the depth-read
    /// transparency that follow. Stamping at
    /// [`AfterOpaque`](Self::AfterOpaque) instead paints over soft particles
    /// and sub-object highlights that overlap the receiving surface.
    OnOpaqueSurfaces,
    /// After the opaque scene, the built-in sprite passes and the depth-read
    /// pass, and before the OIT resolve.
    ///
    /// The scene colour holds the lit opaque image plus the selection
    /// sub-highlight and the depth-read pass's output, so this is where an
    /// effect that samples the finished opaque image (refraction) belongs.
    /// Writes here are visible to every later pass. For an effect that
    /// composites onto surfaces rather than sampling them, use
    /// [`OnOpaqueSurfaces`](Self::OnOpaqueSurfaces).
    AfterOpaque,
    /// After the OIT resolve and the painter's-order content, and before the
    /// outline composite and foreground pass.
    ///
    /// The scene colour now holds opaque plus resolved transparency and the
    /// painter sequence (blended sprites and tiles, world overlay layers),
    /// which is what a participating-media or volumetric effect wants to
    /// composite over. Depth is still opaque-only: neither the OIT pass nor
    /// the painter pass writes any.
    ///
    /// The one scope the LDR path also has, for plugins that opt in through
    /// [`encodes_ldr`](ItemTypePlugin::encodes_ldr): after the scene content,
    /// the painter's-order content and the world overlay layers, and before
    /// the selection outline and the overlays.
    AfterTransparent,
}

/// Information forwarded to a plugin's [`encode`](ItemTypePlugin::encode).
///
/// Unlike the draw hooks, no render pass is begun: the plugin gets the command
/// encoder and the scene attachments, and opens whatever passes it needs. That
/// is the point of the hook: an item type whose work is several passes over
/// intermediate targets of its own cannot be expressed as draw calls inside a
/// pass the lib has already shaped.
///
/// The scene colour is handed over twice on purpose.
/// [`scene_colour`](Self::scene_colour) is the view to attach when writing to
/// it; [`scene_colour_texture`](Self::scene_colour_texture) is the texture to
/// `copy_texture_to_texture` from when the effect has to *read* the image it is
/// also writing, which a render pass cannot do in one step. Allocate the copy
/// target once per viewport size rather than per frame.
///
/// Build pipelines that write the scene colour against
/// [`opaque_target_desc`](crate::resources::DeviceResources::opaque_target_desc);
/// it names the same formats and sample count these attachments have.
#[non_exhaustive]
pub struct EncoderScopeContext<'a> {
    /// Which of the plugin's requested scopes this call is.
    pub scope: EncoderScope,
    /// The wgpu device, for passes, bind groups, and intermediate targets the
    /// effect allocates. Per-viewport allocations should be cached on the
    /// plugin and keyed by [`viewport_index`](Self::viewport_index) and
    /// [`scene_size`](Self::scene_size), not rebuilt every frame.
    pub device: &'a crate::gpu::Device,
    /// The wgpu queue, for uniform writes ahead of the passes encoded here.
    pub queue: &'a crate::gpu::Queue,
    /// Active render-camera snapshot for this viewport: view and projection
    /// matrices, eye position, near/far.
    pub camera: &'a crate::RenderCamera,
    /// Multi-viewport slot index. Effects with per-viewport state key on it.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter assigned by the lib.
    pub frame_index: u64,
    /// Size in pixels of the scene attachments below, render scale applied.
    /// Size intermediate targets from this, not from the viewport extent.
    pub scene_size: [u32; 2],
    /// Format of [`scene_colour`](Self::scene_colour):
    /// [`HDR_COLOR_FORMAT`](crate::resources::HDR_COLOR_FORMAT) on the HDR
    /// path, the renderer's output format on the LDR path.
    pub target_format: crate::gpu::TextureFormat,

    /// The shared group-0 bind group for this viewport, the same one the lib
    /// binds before the draw hooks. Bind it at group 0 in passes of the
    /// plugin's own so its shaders can use the shared camera, lighting and
    /// clip declarations.
    pub camera_bind_group: &'a crate::gpu::BindGroup,
    /// The scene colour target, as a view to attach and render into. On the
    /// HDR path this is the HDR target at scene resolution even under
    /// supersampling: the SSAA resolve runs before either scope, so from here
    /// on the frame is drawing into the same target it would without SSAA. On
    /// the LDR path it is the target the scene pass drew into.
    pub scene_colour: &'a crate::gpu::TextureView,
    /// The same target as a texture, as the source for a
    /// `copy_texture_to_texture` into a sampleable copy. An effect that reads
    /// the lit image while writing to it needs that copy; wgpu cannot bind one
    /// texture as both attachment and sampled resource in a pass.
    pub scene_colour_texture: &'a crate::gpu::Texture,
    /// The scene depth-stencil target, as a view to attach. Attach it with
    /// `LoadOp::Load` and, for an effect that must not disturb later passes,
    /// depth writes disabled.
    pub scene_depth: &'a crate::gpu::TextureView,
    /// Read-only depth-aspect view of the same target, for sampling scene
    /// depth inside the effect's own shaders. Without
    /// [`direct_depth_sampling`](crate::Capabilities::direct_depth_sampling)
    /// (wgpu's GL backend) it is a copy of the scene depth taken at this
    /// scope, a separate texture from the one
    /// [`scene_stencil_only`](Self::scene_stencil_only) views.
    pub scene_depth_only: &'a crate::gpu::TextureView,
    /// The frame's effects settings (`EffectsFrame`), read-only. An item type
    /// with quality or mode knobs that live on the frame rather than on its
    /// items reads them here; the built-in volumetric settings are the reason
    /// this is on the context at all.
    /// The stencil aspect of the scene depth buffer, sampleable as a
    /// `texture_2d<u32>`. It holds the surface mask: for each pixel, the
    /// layers of the item that owns it.
    ///
    /// A projection effect that lands only on some surfaces reads this at the
    /// pixel and skips it when the value shares no bit with the effect's own
    /// mask, cut down with [`surface_mask_bits`]. The type has to name that mask
    /// in [`surface_mask_readers`](ItemTypePlugin::surface_mask_readers), or
    /// the items it would refuse are never stamped. Paired with
    /// [`scene_depth_only`](Self::scene_depth_only), which reconstructs the
    /// receiving surface's position.
    ///
    /// Read it with `stencil_texel` from
    /// [`SHARED_DEPTH_TEXEL_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_TEXEL_WGSL)
    /// and a non-filtering sampler, not `textureLoad`: on GL the stencil is an
    /// integer texture that reads 0 unless a `NEAREST` sampler is bound.
    pub scene_stencil_only: &'a crate::gpu::TextureView,
    /// The frame's effects settings (`EffectsFrame`), read-only. An item type
    /// with quality or mode knobs that live on the frame rather than on its
    /// items reads them here; the built-in volumetric settings are the reason
    /// this is on the context at all.
    pub effects: &'a crate::EffectsFrame,
    /// Selection outline colour and width in pixels, as the consumer set them
    /// on `InteractionFrame`.
    ///
    /// For an item type that draws its own selection affordance here rather
    /// than through [`outline_mask`](ItemTypePlugin::outline_mask): the shared
    /// mask pass traces a ring around world-space geometry, which a
    /// screen-space projection has none of. Match these so a bespoke ring
    /// looks like the one the lib draws for everything else.
    pub outline_colour: crate::Colour,
    /// See [`outline_colour`](Self::outline_colour).
    pub outline_width_px: f32,
    /// Draw handle for consumer-uploaded meshes, for a pass that rasterises
    /// geometry the consumer uploaded rather than buffers of its own.
    pub meshes: crate::resources::MeshDraw<'a>,
    /// `true` when the frame budget asked for reduced quality this frame, the
    /// same signal [`ItemFrameContext::quality_reduced`] carries.
    pub quality_reduced: bool,
    /// Whether this is the HDR path's scope.
    pub(crate) hdr_pass: bool,
}

impl EncoderScopeContext<'_> {
    /// `true` on the HDR path, `false` at the LDR path's
    /// [`AfterTransparent`](EncoderScope::AfterTransparent) point. Choose the
    /// pipelines that write [`scene_colour`](Self::scene_colour) by this or by
    /// [`target_format`](Self::target_format).
    pub fn is_hdr_pass(&self) -> bool {
        self.hdr_pass
    }
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

/// Information forwarded to a plugin's
/// [`contribute_lights`](ItemTypePlugin::contribute_lights).
///
/// Lighting is prepared before any plugin's `prepare` runs, so this carries no
/// per-frame plugin state: derive the lights from the submitted items.
#[non_exhaustive]
pub struct LightContext<'a> {
    /// Uploaded resources, for anything the derivation needs to read. A volume
    /// tinted by a colourmap samples it here with
    /// [`get_colourmap_rgba`](crate::resources::DeviceResources::get_colourmap_rgba).
    pub resources: &'a crate::resources::DeviceResources,
    /// Active render-camera snapshot.
    pub camera: &'a crate::RenderCamera,
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
    /// Draw handle for meshes the consumer uploaded through
    /// [`upload_mesh_data`](crate::resources::DeviceResources::upload_mesh_data).
    /// An item type whose geometry is a `MeshId` binds and draws it through
    /// this; one that owns its buffers ignores it.
    pub meshes: crate::resources::MeshDraw<'a>,
}

/// The layers the surface mask holds, 0 to 7. Also the value every pixel
/// holds until an item stamps over it: a member of all eight.
pub const SURFACE_MASK_LAYERS: u32 = 0xFF;

/// The part of a 32-bit layer mask the surface mask holds: layers 0 to 7.
///
/// Layers 8 and up are dropped. An item with none of the low eight owns its
/// pixels with no layer at all, so nothing that reads the mask lands on it,
/// and a reader with none of them lands nowhere.
pub fn surface_mask_bits(mask: u32) -> u32 {
    mask & SURFACE_MASK_LAYERS
}

/// Information forwarded to a plugin's
/// [`surface_mask`](ItemTypePlugin::surface_mask).
///
/// The lib's surface-mask pass is already begun on entry and has the shared
/// group-0 camera bind group bound. The pass has no colour target; it attaches
/// the scene depth-stencil buffer, and a pipeline from
/// [`build_surface_mask_pipeline`](crate::plugin_api::builders::build_surface_mask_pipeline)
/// writes the stencil reference wherever the item's geometry passes the depth
/// test.
#[non_exhaustive]
pub struct SurfaceMaskContext<'a> {
    /// Active render-camera snapshot.
    pub camera: &'a crate::RenderCamera,
    /// Multi-viewport slot index.
    pub viewport_index: usize,
    /// Monotonically increasing frame counter.
    pub frame_index: u64,
    /// Draw handle for meshes the consumer uploaded through
    /// [`upload_mesh_data`](crate::resources::DeviceResources::upload_mesh_data).
    pub meshes: crate::resources::MeshDraw<'a>,
    pub(crate) readers: &'a [u32],
}

impl SurfaceMaskContext<'_> {
    /// The stencil reference to draw `settings`' item with, or `None` when the
    /// item needs no stamp this frame.
    ///
    /// An item is stamped only when its value would make some reader skip it.
    /// With every reader at the default mask that is no item at all, so a type
    /// can call this for each item and draw the few that answer `Some`:
    ///
    /// ```ignore
    /// if let Some(value) = ctx.stamp_for(settings) {
    ///     pass.set_stencil_reference(value);
    ///     // draw the item
    /// }
    /// ```
    pub fn stamp_for(&self, settings: &ItemSettings) -> Option<u32> {
        if settings.hidden {
            return None;
        }
        let value = surface_mask_bits(settings.visibility_mask);
        surface_mask_needs_stamp(value, self.readers).then_some(value)
    }
}

/// Whether a pixel holding `value` would be skipped by any of `readers`.
pub(crate) fn surface_mask_needs_stamp(value: u32, readers: &[u32]) -> bool {
    readers.iter().any(|reader| value & reader == 0)
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
    /// Draw handle for meshes the consumer uploaded through
    /// [`upload_mesh_data`](crate::resources::DeviceResources::upload_mesh_data).
    /// An item type whose geometry is a `MeshId` binds and draws it through
    /// this; one that owns its buffers ignores it.
    pub meshes: crate::resources::MeshDraw<'a>,
}

/// Upcast to [`Any`] so a host can downcast a registered
/// [`ItemTypePlugin`] back to its concrete type.
///
/// Blanket-implemented for every eligible type: no plugin implements this by
/// hand, and it exists only so
/// [`ViewportRenderer::item_type_plugin_mut`](crate::renderer::ViewportRenderer::item_type_plugin_mut)
/// can hand a registered plugin back as the type it was registered as.
pub trait AsAnyItemTypePlugin {
    /// Return `self` as `&dyn Any`.
    fn as_any_plugin(&self) -> &dyn Any;
    /// Return `self` as `&mut dyn Any`.
    fn as_any_plugin_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any> AsAnyItemTypePlugin for T {
    fn as_any_plugin(&self) -> &dyn Any {
        self
    }
    fn as_any_plugin_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Handle returned by
/// [`with_item_type_plugin`](crate::renderer::ViewportRenderer::with_item_type_plugin);
/// pass it to
/// [`remove_item_type_plugin`](crate::renderer::ViewportRenderer::remove_item_type_plugin)
/// to unregister. Ids are never reused within a renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemTypeId(pub(crate) u64);

/// A scene item category owned by a plugin: its GPU data, its pipelines, and
/// the draws that put it on screen.
///
/// Implement this to add an item type the library does not have. Register it
/// with
/// [`ViewportRenderer::with_item_type_plugin`](crate::renderer::ViewportRenderer::with_item_type_plugin),
/// and submit its items each frame through
/// [`SceneFrame::items_mut`](crate::renderer::SceneFrame::items_mut) or
/// [`submit_plugin_items`](crate::renderer::SceneFrame::submit_plugin_items).
///
/// [`type_name`](Self::type_name) is the only required method. Every other hook
/// is default-empty, so implement the ones the item type needs and leave the
/// rest: a type that only draws opaque geometry implements `prepare` and
/// `paint` and nothing else. The module documentation above lists what each
/// hook is called from and in what order.
///
/// The plugin owns its buffers, textures, layouts and pipelines, built from the
/// `&Device` its hooks are given. Content shared with other item types
/// (meshes, textures, volumes, colourmaps) stays with the library and is read
/// through [`ItemFrameContext::resources`].
///
/// Registration model: item-type plugins are singleton-by-type. The
/// `type_name` is the identity scene items reference to route themselves to a
/// renderer, so two plugins under the same name would be ambiguous: a second
/// registration under a taken name is refused with
/// [`ViewportError::ItemTypeNameTaken`](crate::ViewportError::ItemTypeNameTaken).
/// To replace a plugin, unregister it first with
/// [`remove_item_type_plugin`](crate::ViewportRenderer::remove_item_type_plugin). Runtime
/// and GPU plugins differ deliberately: they carry no external identity and
/// are multi-instance. See [`RuntimePlugin`](crate::runtime::RuntimePlugin)
/// and [`GpuPlugin`](crate::runtime::GpuPlugin).
pub trait ItemTypePlugin: AsAnyItemTypePlugin + Send + Sync + 'static {
    /// Stable name used as the [`SceneFrame::plugin_items`](crate::renderer::SceneFrame::plugin_items)
    /// key. Each registered plugin must have a unique name; registering a
    /// second plugin with the same name is refused with
    /// [`ViewportError::ItemTypeNameTaken`](crate::ViewportError::ItemTypeNameTaken).
    fn type_name(&self) -> &'static str;

    /// What the plugin needs from the device. Checked at registration: a
    /// plugin whose requirements the renderer's
    /// [`Capabilities`](crate::Capabilities) do not meet is declined with
    /// [`ViewportError::PluginUnsupported`](crate::ViewportError::PluginUnsupported).
    /// Defaults to the full tier, so a plugin that binds more storage buffers
    /// than [`Requirements::full_tier`](crate::Requirements::full_tier) names,
    /// or more bind groups, should say so.
    fn requires(&self) -> crate::capabilities::Requirements {
        crate::capabilities::Requirements::full_tier()
    }

    /// Build pipelines, persistent buffers, and bind group layouts. Called
    /// once when the plugin is registered with a renderer that has a
    /// device.
    ///
    /// Build pipelines through `resources` (`build_opaque_pipeline` and the
    /// rest), or list [`group0_layout`](PluginResources::group0_layout) as
    /// group 0 of a pipeline layout built by hand. See [`PluginResources`].
    fn init_gpu(&mut self, _device: &crate::gpu::Device, _resources: &dyn PluginResources) {}

    /// Build, or ask for, every pipeline this type can draw with. Called by
    /// `ViewportRenderer::warm_pipelines` when the type is named in the
    /// [`PipelineSet`](crate::PipelineSet) or the set is `PipelineSet::all()`.
    /// The default does nothing, so the type's pipelines are built by its
    /// first frame as before. A type whose pipelines live in
    /// [`LazyPipelines`](crate::plugin_api::LazyPipelines) calls
    /// `request_all` on them here.
    fn warm(&mut self, _device: &crate::gpu::Device, _resources: &crate::DeviceResources) {}

    /// Lights this item type contributes to the scene this frame.
    ///
    /// An item type that emits light changes how *every other* item type is
    /// lit, which is the one thing a plugin cannot express through its own
    /// draw hooks. The returned lights are appended after the consumer's own
    /// lights and go through the same per-frame frustum cull and count cap, so
    /// a type that returns many may see them dropped.
    ///
    /// Called once per frame before any `prepare`, so derive from `items`
    /// rather than from state a `prepare` would have built. Return an empty
    /// vector (the default) for a type that emits nothing.
    fn contribute_lights(
        &self,
        _items: &ItemCollections<'_>,
        _ctx: &LightContext<'_>,
    ) -> Vec<crate::renderer::LightSource> {
        Vec::new()
    }

    /// Resident GPU bytes held in stores this item type owns.
    ///
    /// Summed into [`ResidentBytes::plugin_bytes`](crate::resources::ResidentBytes::plugin_bytes)
    /// by [`ViewportRenderer::resident_bytes`](crate::renderer::ViewportRenderer::resident_bytes),
    /// so an item type that holds its own uploaded content shows up in the
    /// working-set figure an eviction policy budgets against. Leave it at the
    /// default when the type stores nothing of its own, or stores it through
    /// the shared upload calls on `DeviceResources` (which count it already).
    ///
    /// Count the persistent content buffers. Per-frame scratch and derived
    /// caches that grow and shrink with the viewport are not part of the
    /// evictable working set and are better left out.
    fn resident_bytes(&self) -> u64 {
        0
    }

    /// Draw calls this plugin issued for the most recently prepared frame.
    ///
    /// Summed into [`FrameStats::plugin_draw_calls`](crate::renderer::stats::FrameStats::plugin_draw_calls)
    /// and reported per plugin by
    /// [`ViewportRenderer::plugin_frame_counters`](crate::renderer::ViewportRenderer::plugin_frame_counters).
    /// `FrameStats::draw_calls` counts mesh-family draws only and cannot see
    /// yours, so a plugin that leaves this at the default is invisible in the
    /// frame's draw-call figure however much it draws.
    ///
    /// This is a per-frame count, not a running total: reset it in your
    /// `prepare` and accumulate as you record draws. The paint hooks take
    /// `&self`, so a count kept as draws are recorded lives in an atomic or a
    /// `Cell`. The renderer reads it when the host asks for the frame's
    /// stats, after the frame has painted, and does not reset it for you, so
    /// a plugin that only ever adds reports a number that climbs for the
    /// process lifetime. A frame with none of the plugin's items reports
    /// zero for it without calling this.
    fn draw_calls(&self) -> u32 {
        0
    }

    /// Bytes this plugin uploaded for the most recently prepared frame.
    ///
    /// Summed into [`FrameStats::plugin_upload_bytes`](crate::renderer::stats::FrameStats::plugin_upload_bytes).
    /// Same per-frame contract as [`Self::draw_calls`]: reset it in `prepare`.
    ///
    /// Count what crossed to the GPU this frame. That is a different question
    /// from [`Self::resident_bytes`], which is the steady-state working set: a
    /// plugin holding 100 MB and uploading nothing reports a large
    /// `resident_bytes` and a zero here, which is the healthy shape.
    fn upload_bytes(&self) -> u64 {
        0
    }

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
    ///
    /// With several viewports over one scene, this runs once per viewport
    /// (`prepare_viewport`), each with that viewport's camera, viewport and
    /// interaction state, before any of them paints. State built from those,
    /// and anything [`cull`](Self::cull) narrows, has to be kept per
    /// [`ItemFrameContext::viewport_index`] and read back by the draw hooks
    /// from their own context's `viewport_index`, or the first viewport
    /// paints what the last one prepared.
    /// [`PerViewport`](crate::plugin_api::PerViewport) holds such state.
    fn prepare(
        &mut self,
        _device: &crate::gpu::Device,
        _queue: &crate::gpu::Queue,
        _ctx: &ItemFrameContext<'_>,
        _items: &ItemCollections<'_>,
    ) -> Vec<crate::gpu::CommandBuffer> {
        Vec::new()
    }

    /// Issue draw calls inside the lib's scene pass.
    ///
    /// On the HDR path, called after built-in opaque geometry and the skybox.
    /// On the LDR path (`PipelineMode::Direct`, `paint`, `paint_to`,
    /// `paint_viewport`), called after the built-in scene content, but only
    /// for plugins that opt in via [`draws_ldr`](Self::draws_ldr); the two
    /// paths use different colour formats:
    /// [`PaintContext::is_hdr_pass`] says which path the pass is on and
    /// [`PaintContext::target_format`] names its format. The pass has
    /// the shared group-0 bind group bound on entry; plugins must restore it
    /// if they rebind group 0 themselves.
    ///
    /// Implementations should treat hidden items
    /// (`items.item_settings(i).hidden == true`) as drawn-nothing; the
    /// lib does not pre-filter the collection.
    fn paint(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// `true` when this plugin's `paint` also draws on the LDR path.
    ///
    /// The LDR scene pass targets the renderer's configured output format
    /// rather than the HDR scene format, so a plugin that returns `true`
    /// must keep a pipeline for that format (built against
    /// [`ldr_opaque_target_desc`](crate::resources::DeviceResources::ldr_opaque_target_desc))
    /// and pick it when [`PaintContext::is_hdr_pass`] is `false`. A plugin
    /// that returns `false` here is skipped on the LDR path
    /// (and the renderer warns once), never handed a pass their pipeline
    /// cannot draw into.
    ///
    /// `paint` and `paint_painter` run on the LDR path: the OIT,
    /// read-only-depth, and foreground passes exist on the HDR path alone, so
    /// `paint_transparent`, `paint_depth_read`, and `paint_foreground` are
    /// HDR-only regardless of this value. An encode after the scene opts in
    /// separately, through [`encodes_ldr`](Self::encodes_ldr).

    fn draws_ldr(&self) -> bool {
        false
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
    fn paint_transparent(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// `false` when this frame's items have nothing for
    /// [`paint_transparent`](Self::paint_transparent) in the viewport
    /// `viewport_index`.
    ///
    /// Read after [`prepare`](Self::prepare). The renderer runs the OIT pass
    /// when any plugin with items says `true`, so the default keeps a plugin
    /// that never overrides this drawing there; a plugin that knows its frame
    /// has no OIT draws returns `false` and, when nothing else is transparent,
    /// the frame skips the pass and its composite.
    fn draws_transparent(&self, _viewport_index: usize) -> bool {
        true
    }

    /// `true` when this plugin draws in the read-only-depth pass of the
    /// viewport `viewport_index`.
    ///
    /// The renderer opens that pass only when some plugin opts in;
    /// implementing [`paint_depth_read`](Self::paint_depth_read) without
    /// overriding this to `true` means the hook is never called, and a frame
    /// with no opted-in plugin issues no extra pass.
    fn draws_depth_read(&self, _viewport_index: usize) -> bool {
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
    ///
    /// Content that should sort with painter layers (blended sprites, tiles,
    /// 2D meshes, particles in a 2D scene) does not belong here: the renderer
    /// cannot tell such draws from depth-read effects, so they draw before the
    /// painter sequence whatever their layer. Report them through
    /// [`painter_batches`](Self::painter_batches) and draw them in
    /// [`paint_painter`](Self::paint_painter);
    /// [`PluginFrameCounters::painter_batches`](crate::renderer::stats::PluginFrameCounters::painter_batches)
    /// shows what joined the sequence.
    fn paint_depth_read(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &DepthReadContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// `true` when this frame's items have painter's-order draws for
    /// [`paint_painter`](Self::paint_painter) in the viewport `viewport_index`.
    ///
    /// Read after [`prepare`](Self::prepare). A frame where some plugin says
    /// `true` draws them on the HDR path in a read-only-depth pass of their
    /// own after the order-independent transparency resolves, and on the LDR
    /// path in the scene pass, after every plugin's [`paint`](Self::paint).
    fn draws_painter(&self, _viewport_index: usize) -> bool {
        false
    }

    /// This frame's painter's-order draws in the viewport `viewport_index`,
    /// pushed onto `out`, so they take
    /// their place in the sequence the renderer sorts across every plugin.
    ///
    /// Read after [`prepare`](Self::prepare) and [`cull`](Self::cull), only
    /// when [`draws_painter`](Self::draws_painter) returns `true`. Push one
    /// [`PainterBatch`] per draw that can be ordered on its own; the renderer
    /// then calls [`paint_painter`](Self::paint_painter) once per run of
    /// consecutive batches of this plugin, naming them in
    /// [`PainterContext::batches`]. Pushing nothing, the default, keeps the
    /// plugin out of the sequence: see `paint_painter` for where it draws then.
    ///
    /// Each batch is placed by its one key (see [`PainterBatch`]): report a
    /// draw that covers several keys as one batch per run of equal keys.
    /// Content meant to sort with painter layers must come through here; the
    /// renderer cannot see draws a plugin makes in
    /// [`paint`](Self::paint) or [`paint_depth_read`](Self::paint_depth_read)
    /// and does not warn about them. The count reported each frame is in
    /// [`PluginFrameCounters::painter_batches`](crate::renderer::stats::PluginFrameCounters::painter_batches).
    fn painter_batches(&self, _viewport_index: usize, _out: &mut Vec<PainterBatch>) {}

    /// Issue this frame's painter's-order draws.
    ///
    /// Painter's-order content (blended sprites, blended tile layers) is
    /// ordered by a `u32` layer, and world overlay layers interleave with it by
    /// the same number. A plugin that reports
    /// [`painter_batches`](Self::painter_batches) is called once per run of
    /// its own batches in the renderer's sequence and draws exactly the
    /// batches [`PainterContext::batches`] names, in that order, so its
    /// content interleaves with other plugins' by layer and sort key.
    ///
    /// A plugin that reports none is called once per run of layers between
    /// world overlay layers and light layers, ascending, after the sequenced
    /// batches of that run, and draws its content whose layer lies in
    /// [`PainterContext::layers`]; then the world layers or the light layer
    /// at the run's top layer draw. A frame with neither makes one such call
    /// covering every layer. Such plugins draw in registration order within a
    /// run, and their content is never unlit.
    ///
    /// The pass has the viewport's camera bind group at group 0 on entry.
    /// Only called when [`draws_painter`](Self::draws_painter) returns `true`,
    /// and on the LDR path only for a plugin that [`draws_ldr`](Self::draws_ldr).

    fn paint_painter(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PainterContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// The painter masks this plugin draws in the viewport `viewport_index`
    /// this frame, pushed onto `out`.
    ///
    /// Read after [`prepare`](Self::prepare). A frame where some plugin
    /// reports one clears the viewport's mask target and calls
    /// [`paint_painter_masks`](Self::paint_painter_masks) on every plugin that
    /// reported, before any painter content draws; content testing a slot
    /// nobody reported is not cut.
    fn painter_masks(&self, _viewport_index: usize, _out: &mut Vec<PainterMask>) {}

    /// Draw this frame's painter masks into the viewport's mask target.
    ///
    /// The target is [`PAINTER_MASK_FORMAT`](crate::resources::PAINTER_MASK_FORMAT)
    /// at the scene's device size, one sample, with no depth attachment and
    /// the viewport's camera bind group at group 0 on entry. Write a mask's
    /// coverage (`0` to `1`) into the channel of its slot (red for slot 0 to
    /// alpha for slot 3) and zero into the others, through
    /// [`PAINTER_MASK_BLEND`](crate::resources::PAINTER_MASK_BLEND). Runs on
    /// every tier, during prepare, so it needs no pass of the frame itself.
    fn paint_painter_masks(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PainterMaskContext<'_>,
        _items: &ItemCollections<'_>,
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
    fn paint_foreground(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// Build this frame's wireframe polylines for the item type.
    ///
    /// Called during `prepare` for every registered plugin. The returned items
    /// are uploaded and drawn through the shared line substrate, the same one
    /// clip-object outlines and consumer-submitted polylines render
    /// through, so a plugin needs no pipeline of its own for this. Return an
    /// empty vector, the default, for a type with nothing to draw.
    ///
    /// Implementations honour both triggers, since either means the same thing
    /// to the viewer: `ctx.wireframe_mode` for the whole frame, and
    /// `items.item_settings(i).wireframe` for one item. Skip hidden items.
    ///
    /// # Which lines to return
    ///
    /// [`ItemSettings::wireframe`](crate::scene::material::ItemSettings::wireframe)
    /// is documented as rendering an item's sub-objects as meshes, and that is
    /// the first thing to reach for: a Gaussian splat set draws a ring per
    /// splat, a sprite batch an outline per billboard. Lines that show what the
    /// item is made of.
    ///
    /// A bounding box is the other common answer, and it is a weaker one: it
    /// locates the item without describing it, and past a certain item count
    /// it degenerates into a box around everything. Prefer it where the item
    /// genuinely has no internal structure to show, as a participating-media
    /// volume does not, and be wary of it as a fallback for "too many
    /// sub-objects to draw": drawing nothing is often more honest.
    ///
    /// [`aabb_wireframe_polyline`](crate::aabb_wireframe_polyline) and
    /// [`sphere_wireframe_polyline`](crate::sphere_wireframe_polyline) build
    /// the two common bounds shapes.
    ///
    /// # Selection
    ///
    /// This is also where a type with no silhouette shows selection. The
    /// outline ring from [`outline_mask`](Self::outline_mask) traces rasterised
    /// geometry, which a volume, a cloud or a billboard does not present, so
    /// those types answer `item_settings(i).selected` here instead and draw
    /// their bounds. `[1.0, 0.9, 0.2, 1.0]` is the colour the built-in types
    /// use for that, distinct from the paler colour they use for plain
    /// wireframe mode.
    ///
    /// Set `settings.wireframe` on a returned item for the thin single-pixel
    /// line the built-in outlines use, rather than the screen-space thick line
    /// a data polyline gets.
    fn wireframe_polylines(
        &self,
        _items: &ItemCollections<'_>,
        _ctx: &ItemFrameContext<'_>,
    ) -> Vec<crate::renderer::PolylineItem> {
        Vec::new()
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
    ///
    /// # When the item type has no edges to trace
    ///
    /// This traces a ring around rasterised geometry, so it suits a type whose
    /// items are solid on screen. A type whose items are a cloud, a billboard
    /// or a participating-media volume has no silhouette worth ringing, and the
    /// built-in types in that position show selection as a bounds wireframe
    /// instead: a box for a volume, great circles for a sphere.
    ///
    /// That is not a built-in privilege. The polyline substrate those draw
    /// through takes consumer items too, and the builders are public, so a
    /// plugin's consumer draws the same affordance by pushing one item per
    /// frame:
    ///
    /// ```no_run
    /// # use viewport_lib::{Colour, FrameData, Aabb, PolylineItem, aabb_wireframe_polyline};
    /// # fn example(fd: &mut FrameData, bounds: &Aabb, selected: bool) {
    /// if selected {
    ///     // The same yellow the built-in bounds outlines use.
    ///     fd.scene
    ///         .items_mut::<PolylineItem>()
    ///         .push(aabb_wireframe_polyline(bounds, Colour::linear(1.0, 0.9, 0.2, 1.0)));
    /// }
    /// # }
    /// ```
    ///
    /// [`sphere_wireframe_polyline`](crate::sphere_wireframe_polyline) does the
    /// same for a sphere. Set `settings.wireframe` on the pushed item for the
    /// thin single-pixel line the built-in outlines use, rather than the
    /// screen-space thick line a data polyline gets.
    fn outline_mask(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// Draw items into the surface mask, so a screen-space effect can tell
    /// whose surface a pixel belongs to.
    ///
    /// The surface mask holds, for each pixel of the opaque image, the layers
    /// (`ItemSettings::visibility_mask`, cut down by [`surface_mask_bits`]) of
    /// the item that owns it. A decal reads it to land on some surfaces and
    /// not others. Every pixel starts as a member of every layer; this hook is
    /// where an item type overwrites that for the items that say otherwise.
    ///
    /// Called inside the lib's surface-mask pass, after the opaque scene and
    /// its supersample resolve and before
    /// [`EncoderScope::OnOpaqueSurfaces`], with group 0 bound. The pass runs
    /// only on frames where some type reads the mask and some item needs a
    /// stamp. For each item, ask
    /// [`SurfaceMaskContext::stamp_for`]; when it answers, set that as the
    /// stencil reference and draw the item's geometry with a pipeline from
    /// [`build_surface_mask_pipeline`](crate::plugin_api::builders::build_surface_mask_pipeline).
    ///
    /// Draw the same geometry the opaque pass drew. The stamp is depth-tested
    /// against the scene, so a vertex stage that lands somewhere else stamps
    /// the wrong pixels or none.
    ///
    /// A type that writes depth and leaves this empty is treated as a member
    /// of every layer: every decal lands on it.
    fn surface_mask(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &SurfaceMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// Name the masks this type will test the surface mask against this
    /// frame, one per distinct mask, cut down by [`surface_mask_bits`].
    ///
    /// Push nothing when the type does not read the mask, which is the
    /// default. The lib uses the list to decide which items need stamping, so
    /// a type that reads the mask without naming its masks here sees every
    /// pixel as a member of every layer.
    fn surface_mask_readers(&self, _items: &ItemCollections<'_>, _out: &mut Vec<u32>) {}

    /// The points in the HDR frame where this plugin wants an
    /// [`encode`](Self::encode) call.
    ///
    /// Empty by default, which is the answer for every item type that draws
    /// inside the passes the lib begins: those use `paint` and its siblings.
    /// Return one or more [`EncoderScope`]s only when the work genuinely needs
    /// passes of its own, and the renderer will call `encode` once per listed
    /// scope, in the order the frame reaches them rather than the order listed.
    ///
    /// Read after [`prepare`](Self::prepare), and only for a plugin with items
    /// this frame: a scope no plugin with items lists opens no pass.
    ///
    /// On the HDR path every scope runs. The LDR path runs
    /// [`AfterTransparent`](EncoderScope::AfterTransparent) alone, and only
    /// for a plugin that opts in through [`encodes_ldr`](Self::encodes_ldr).
    fn encoder_scopes(&self) -> &[EncoderScope] {
        &[]
    }

    /// `true` when this plugin's [`AfterTransparent`](EncoderScope::AfterTransparent)
    /// encode also runs on the LDR path.
    ///
    /// There the renderer ends its scene pass after the painter's-order
    /// content and world overlay layers, calls `encode`, and resumes with the
    /// selection outline, the sub-object highlight and the overlays. The
    /// context's scene colour is the LDR target in the renderer's output
    /// format ([`EncoderScopeContext::target_format`], with
    /// [`EncoderScopeContext::is_hdr_pass`] `false`), at one sample, so the
    /// plugin needs pipelines for that format as well. The scene colour is a
    /// texture that can be copied from: when the output cannot be, the frame
    /// draws into an intermediate target and copies it to the output at the
    /// end.
    ///
    /// The host-owned render pass (`pass().paint`, `paint_to`,
    /// `paint_viewport`) cannot be ended mid-frame, so there the encode is
    /// skipped and the renderer reports it once.
    fn encodes_ldr(&self) -> bool {
        false
    }

    /// `false` when this plugin's encode at `scope` never reads the scene
    /// depth.
    ///
    /// Without [`direct_depth_sampling`](crate::Capabilities::direct_depth_sampling)
    /// the renderer copies the scene depth before a scope so
    /// [`EncoderScopeContext::scene_depth_only`] can be sampled; it skips the
    /// copy when no plugin encoding at the scope says `true`. On such a frame
    /// `scene_depth_only` views the live depth buffer, which must not then be
    /// sampled while `scene_depth` is attached.
    fn encode_reads_depth(&self, _scope: EncoderScope) -> bool {
        true
    }

    /// Encode passes of this plugin's own into the frame's command encoder.
    ///
    /// Called once per scope named by [`encoder_scopes`](Self::encoder_scopes),
    /// with no render pass begun: the plugin opens the passes it needs against
    /// the scene attachments on `ctx`, or against intermediate targets it owns.
    /// This is the hook for work that is several passes over its own textures
    /// (march, resolve, composite) rather than draw calls inside a pass the lib
    /// has shaped.
    ///
    /// The encoder is the frame's, so everything encoded here lands in frame
    /// order between the neighbouring built-in passes. Two rules follow from
    /// that: leave the scene attachments in the state the next pass expects
    /// (attach with `LoadOp::Load`, and do not clear what you did not allocate),
    /// and do not submit : the lib owns submission.
    ///
    /// Implementations skip hidden items, as everywhere else. Default no-op.
    fn encode(
        &self,
        _encoder: &mut crate::gpu::CommandEncoder,
        _ctx: &EncoderScopeContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// Update internal culling state for the current camera frustum.
    ///
    /// Called once per frame and viewport, after `prepare` and before
    /// `paint`. Keep the result per [`ItemFrameContext::viewport_index`] (see
    /// [`prepare`](Self::prepare)). The plugin records which
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
        _items: &ItemCollections<'_>,
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
    fn cast_shadow_pass(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &ShadowCastContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// Hit-test this plugin's items against `ray` and return the closest
    /// hit, paired with its world-space time-of-impact.
    ///
    /// Called from the renderer's pick router after the built-in CPU and
    /// GPU pick paths. The router compares the returned `t` against every
    /// other candidate and chooses the closest. `ctx` carries the click
    /// position, viewport size, view-projection, and the query's mask, so
    /// item types with screen-space pick tolerances (thin lines, markers)
    /// can run pixel-space proximity tests via
    /// [`pick_helpers`](crate::plugin_api::pick_helpers) instead of pure
    /// ray-casts; a screen-space hit still returns a world-space `t` along
    /// the ray so it competes fairly with the other candidates.
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
    fn pick(&self, _ray: &PickRay, _ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        None
    }

    /// Return this plugin's items (and sub-elements) inside a screen-space
    /// rectangle (CPU box select).
    ///
    /// Called from the renderer's CPU
    /// [`pick_rect`](crate::renderer::ViewportRenderer::pick_rect) after the
    /// built-in item types; the returned result is merged into the query's.
    /// Implementations project their items with
    /// [`project_to_screen`](crate::plugin_api::pick_helpers::project_to_screen)
    /// (and
    /// [`segment_in_rect`](crate::plugin_api::pick_helpers::segment_in_rect)
    /// for edges), push the ids of items touching the rectangle into
    /// `objects`, and, when the mask asks for a sub-object level the item
    /// type answers, push `(id, sub_object)` pairs into `elements`. Skip
    /// hidden items and [`PickId::NONE`]. Like [`pick`](Self::pick), the
    /// plugin answers from state cached in [`prepare`](Self::prepare).
    ///
    /// The GPU rect path needs no counterpart: items drawn in
    /// [`render_pick`](Self::render_pick) are decoded there per pixel.
    /// Default: empty, matching plugins that only implement GPU picking.
    fn pick_rect(&self, _ctx: &RectPickContext<'_>) -> crate::renderer::PickRectResult {
        crate::renderer::PickRectResult::default()
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
    /// other two channels. Build the pipeline with
    /// [`build_pick_pipeline`](PluginResources::build_pick_pipeline) in
    /// [`init_gpu`](Self::init_gpu).
    ///
    /// Implementations skip hidden items and items whose
    /// [`pick_id`](PluginItemCollection::pick_id) is [`PickId::NONE`] (the clear
    /// value, which reads back as no hit). This is the GPU counterpart to
    /// [`pick`](Self::pick); a plugin implements whichever fits, or both.
    /// Default no-op: plugins relying on the CPU path leave this empty.
    ///
    /// Painter's-order content picks in
    /// [`render_pick_painter`](Self::render_pick_painter) instead, which runs
    /// after every plugin's `render_pick` in the painter sequence's order.
    ///
    /// For sub-object picking (face / vertex / edge on the drawn geometry),
    /// use the primitive-index fragment helper
    /// ([`SHARED_PICK_PRIM_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_PRIM_WGSL))
    /// instead of `viewport_pick_fs` when the device has
    /// [`PRIMITIVE_INDEX_FEATURE`](crate::gpu::PRIMITIVE_INDEX_FEATURE), and
    /// implement [`resolve_sub_object`](Self::resolve_sub_object) to map the
    /// read-back triangle index to the item's own sub-object ids.
    fn render_pick(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PickPassContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
    }

    /// Issue pick draws for one run of this frame's painter sequence.
    ///
    /// The pick pass draws every plugin's [`render_pick`](Self::render_pick)
    /// first, then walks the painter sequence in draw order and calls this
    /// once per run of this plugin, as [`paint_painter`](Self::paint_painter)
    /// is called: with the batch handles of the run, or for a plugin that
    /// reports no batches with the run's layers. The draw drawn last on
    /// screen therefore wins the pick across plugins, provided the pipeline
    /// tests the pick depth without writing it:
    /// [`build_pick_pipeline`](PluginResources::build_pick_pipeline) with
    /// `depth_write: false` and the
    /// [`DEPTH_COMPARE`](crate::plugin_api::depth::DEPTH_COMPARE) test.
    ///
    /// Only called when [`draws_painter`](Self::draws_painter) returns `true`.
    /// Default no-op.
    fn render_pick_painter(
        &self,
        _pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PickPassContext<'_>,
        _run: &PainterRun<'_>,
        _items: &ItemCollections<'_>,
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
    /// The plugin owns its geometry, so only it can map the channel value to
    /// a face / vertex / edge / instance id. Return `None` for levels the
    /// item type does not answer; the hit then stays object-level. Default:
    /// `None` (all plugin picking stays object-level, as before this hook
    /// existed).
    ///
    /// The mesh levels (`FACE`, `VERTEX`, `EDGE`, `CELL`) decode
    /// `@builtin(primitive_index)` and are only forwarded when the device
    /// has [`PRIMITIVE_INDEX_FEATURE`](crate::gpu::PRIMITIVE_INDEX_FEATURE);
    /// without it the renderer strips them from `mask` before this call, so
    /// a constant-0 channel is never misread as a triangle index. The
    /// `INSTANCE`, `SPLAT`, and `CLOUD_POINT` levels decode a shader-written
    /// index (for example
    /// [`SHARED_PICK_INSTANCE_WGSL`](crate::plugin_api::shared_wgsl::SHARED_PICK_INSTANCE_WGSL),
    /// or a pick fragment of the plugin's own that writes the element index)
    /// and are forwarded on every device, matching the built-in paths for
    /// those levels.
    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        _primitive_index: u32,
        _world_pos: glam::Vec3,
        _mask: crate::renderer::PickMask,
    ) -> Option<crate::renderer::SubObjectRef> {
        None
    }

    /// World-space position of a resolved sub-object feature, for snapping a
    /// gizmo to it.
    ///
    /// Called after [`resolve_sub_object`](Self::resolve_sub_object) (or the
    /// CPU [`pick`](Self::pick)) has named the feature, with the same
    /// `sub_object` it returned and the collection submitted this frame. Only
    /// the plugin knows where its features are, so a type that resolves
    /// [`SubObjectRef::Point`](crate::renderer::SubObjectRef::Point) answers
    /// with the point's world position: its index into the item's own
    /// positions, times the item's model.
    ///
    /// Return `None` for features with no single snap point (a cell, a
    /// segment, a strip) and for items whose positions are not reachable (a
    /// reference item whose data lives in an upload store rather than on the
    /// frame). The hit then keeps the pick's own world position, which lies on
    /// the feature but not at its centre.
    ///
    /// Default: `None`, which is the behaviour of every item type that has no
    /// point-like feature to snap to.
    fn sub_object_position(
        &self,
        _items: &ItemCollections<'_>,
        _pick_id: PickId,
        _sub_object: crate::renderer::SubObjectRef,
    ) -> Option<glam::Vec3> {
        None
    }

    /// World-space outline of a resolved sub-object, for the sub-object
    /// highlight: the corners of a convex polygon in order, which the
    /// highlight fills and traces.
    ///
    /// Asked for each selected sub-object whose geometry only the item type
    /// knows, such as a [`SubObjectRef::Tile`](crate::renderer::SubObjectRef::Tile),
    /// with the collection submitted this frame. Push the corners into `out`
    /// for a sub-object of one of this plugin's items and nothing otherwise;
    /// the first plugin to push wins. The highlight is rebuilt when an
    /// outline changes, so an item that moves keeps its highlight under it.
    ///
    /// Default: nothing, so the sub-object draws no highlight.
    fn sub_object_outline(
        &self,
        _items: &ItemCollections<'_>,
        _pick_id: PickId,
        _sub_object: crate::renderer::SubObjectRef,
        _out: &mut Vec<glam::Vec3>,
    ) {
    }
}

/// A registered item-type plugin plus the renderer-owned services an upload
/// into it needs, handed out together by
/// [`ViewportRenderer::item_type_plugin_host`](crate::renderer::ViewportRenderer::item_type_plugin_host).
///
/// A plugin that stores its own content takes an upload call of its own, and
/// that call usually wants three things at once: the plugin itself, the job
/// runner (to build buffers off the frame thread), and read access to the
/// content shared between item types (textures, colourmaps, volumes, meshes).
/// All three are separate fields of `ViewportRenderer`, so only the renderer
/// can lend them out together: taking the plugin through
/// [`item_type_plugin_mut`](crate::renderer::ViewportRenderer::item_type_plugin_mut)
/// borrows the whole renderer and the other two are then out of reach.
///
/// ```no_run
/// # use viewport_lib::plugin_api::ItemTypePlugin;
/// # use viewport_lib::renderer::ViewportRenderer;
/// # use viewport_lib::resources::{Jobs, JobId};
/// # struct MyPlugin;
/// # impl ItemTypePlugin for MyPlugin {
/// #     fn type_name(&self) -> &'static str { "my.type" }
/// # }
/// # impl MyPlugin {
/// #     fn begin_upload(&mut self, _: &Jobs<'_>, _: &viewport_lib::wgpu::Device) -> JobId { unimplemented!() }
/// # }
/// # fn demo(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device) -> Option<JobId> {
/// let host = renderer.item_type_plugin_host::<MyPlugin>("my.type")?;
/// Some(host.plugin.begin_upload(&host.jobs, device))
/// # }
/// ```
pub struct ItemTypeHost<'a, T> {
    /// The registered plugin, as the type it was registered as.
    pub plugin: &'a mut T,
    /// The upload-job runner, for CPU work that should not run on the frame
    /// thread. Submit here and collect the result from a later call; the
    /// runner is advanced by the renderer's `prepare`.
    pub jobs: crate::resources::Jobs<'a>,
    /// Read access to the content shared between item types: meshes,
    /// textures, 3D volumes and colourmaps, through the accessors on
    /// [`DeviceResources`](crate::resources::DeviceResources).
    pub resources: &'a crate::resources::DeviceResources,
}

#[cfg(test)]
mod surface_mask_tests {
    use super::*;

    #[test]
    fn the_mask_holds_the_low_eight_layers() {
        assert_eq!(surface_mask_bits(!0), SURFACE_MASK_LAYERS);
        assert_eq!(surface_mask_bits(0), 0);
        // Clearing one low layer clears it in the mask.
        assert_eq!(surface_mask_bits(!0b1), 0xFE);
        // An item on high layers only owns its pixels with no layer.
        assert_eq!(surface_mask_bits(!SURFACE_MASK_LAYERS), 0);
    }

    #[test]
    fn only_items_a_reader_would_refuse_are_stamped() {
        // Every reader at the default: nothing but a zero value needs a stamp.
        assert!(!surface_mask_needs_stamp(
            0b0000_0010,
            &[SURFACE_MASK_LAYERS]
        ));
        assert!(surface_mask_needs_stamp(0, &[SURFACE_MASK_LAYERS]));
        // A narrowed reader makes an item on another layer need one.
        assert!(surface_mask_needs_stamp(
            0b0000_0010,
            &[SURFACE_MASK_LAYERS, 0b0000_0001]
        ));
        assert!(!surface_mask_needs_stamp(0b0000_0011, &[0b0000_0001]));
        // Nothing reads the mask: nothing is stamped.
        assert!(!surface_mask_needs_stamp(0, &[]));
    }
}
