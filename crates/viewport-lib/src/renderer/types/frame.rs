use super::*;
use crate::interaction::manipulation::gizmo::{GizmoAxis, GizmoMode};
use crate::interaction::query::snap::ConstraintOverlay;
use crate::renderer::SubSelectionRef;
use crate::resources::CameraUniform;

// ---------------------------------------------------------------------------
// 0.2.0 grouped frame API types
// ---------------------------------------------------------------------------

/// Canonical renderer-facing camera state.
///
/// Replaces the flat camera fields that were previously scattered across
/// `FrameData`. Application-side orbit cameras resolve into this type
/// before frame submission.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RenderCamera {
    /// World-to-view transform matrix.
    pub view: glam::Mat4,
    /// View-to-clip (projection) matrix.
    pub projection: glam::Mat4,
    /// Camera eye position in world space.
    pub eye_position: [f32; 3],
    /// Camera forward direction in world space.
    pub forward: [f32; 3],
    /// Camera orientation quaternion.
    pub orientation: glam::Quat,
    /// Near clip plane distance. Default: 0.1.
    pub near: f32,
    /// Far clip plane distance. Default: 1000.0.
    pub far: f32,
    /// Orbit distance from camera to scene center (world units). Default: 10.0.
    /// Used internally to derive shadow cascade range so shadow quality tracks zoom level.
    pub distance: f32,
    /// Vertical field of view in radians. Default: PI/4.
    pub fov: f32,
    /// Aspect ratio (width / height). Default: 1.333.
    pub aspect: f32,
}

impl RenderCamera {
    /// Build the GPU-facing camera uniform from this camera's state.
    pub(crate) fn camera_uniform(&self) -> CameraUniform {
        let vp = self.view_proj();
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            eye_pos: self.eye_position,
            // LDR default; the per-viewport upload raises this on the HDR path.
            lit_clamp: 1.0,
            forward: self.forward,
            _pad1: 0.0,
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            view: self.view.to_cols_array_2d(),
        }
    }

    /// Combined view-projection matrix (projection * view).
    pub fn view_proj(&self) -> glam::Mat4 {
        self.projection * self.view
    }

    /// The camera the foreground pass renders with: this camera's view
    /// transform, with the projection replaced by the pass override when one
    /// is set. With no override (or no pass config) the scene camera is used
    /// unchanged.
    pub(crate) fn foreground_camera(&self, pass: Option<&ForegroundPass>) -> RenderCamera {
        let mut cam = self.clone();
        if let Some(p) = pass.and_then(|p| p.projection.as_ref()) {
            cam.fov = p.fov_y;
            cam.near = p.near.max(1e-4);
            cam.far = p.far.unwrap_or(self.far).max(cam.near * 2.0);
            cam.projection = glam::Mat4::perspective_rh(cam.fov, self.aspect, cam.near, cam.far);
        }
        cam
    }

    /// Build a `RenderCamera` from an app-side [`Camera`](crate::camera::Camera).
    ///
    /// This is the intended conversion path: resolve the orbit camera to a
    /// `RenderCamera` once per frame and pass it through `CameraFrame`.
    pub fn from_camera(cam: &crate::camera::Camera) -> Self {
        let eye = cam.eye_position();
        let forward = (cam.center - eye).normalize_or_zero();
        Self {
            view: cam.view_matrix(),
            projection: cam.proj_matrix(),
            eye_position: eye.to_array(),
            forward: forward.to_array(),
            orientation: cam.orientation,
            near: cam.effective_znear(),
            far: cam.effective_zfar(),
            fov: cam.fov_y,
            aspect: cam.aspect,
            distance: cam.distance,
        }
    }
}

impl Default for RenderCamera {
    fn default() -> Self {
        Self {
            view: glam::Mat4::IDENTITY,
            projection: glam::Mat4::IDENTITY,
            eye_position: [0.0, 0.0, 5.0],
            forward: [0.0, 0.0, -1.0],
            orientation: glam::Quat::IDENTITY,
            near: 0.1,
            far: 1000.0,
            fov: std::f32::consts::FRAC_PI_4,
            aspect: 1.333,
            distance: 10.0,
        }
    }
}

/// Camera submission state for one frame.
///
/// Groups the canonical render camera with viewport sizing and multi-viewport
/// slot index. This is the single owner of all camera-derived state submitted
/// to the renderer each frame.
#[non_exhaustive]
pub struct CameraFrame {
    /// Canonical renderer-facing camera state.
    pub render_camera: RenderCamera,
    /// Viewport size in logical pixels (egui points) (width, height). Default: [800.0, 600.0].
    pub viewport_size: [f32; 2],
    /// Physical pixels per logical pixel. Set from the window or egui context
    /// (e.g. `ui.ctx().pixels_per_point()`). Default: 1.0.
    ///
    /// Screen-space math (projection, picking, widget layout) and overlay
    /// positions/sizes are all in logical pixels and use `viewport_size`
    /// directly. `pixels_per_point` sizes the physical render target and
    /// rasterises overlay text at display resolution, so overlay labels stay
    /// crisp on HiDPI displays. At 1.0 the physical and logical spaces coincide.
    pub pixels_per_point: f32,
    /// Multi-viewport slot index. Default: 0 (single-viewport mode).
    pub viewport_index: usize,
}

impl Default for CameraFrame {
    fn default() -> Self {
        Self {
            render_camera: RenderCamera::default(),
            viewport_size: [800.0, 600.0],
            pixels_per_point: 1.0,
            viewport_index: 0,
        }
    }
}

impl CameraFrame {
    /// Build a camera frame from a render camera and viewport size.
    pub fn new(render_camera: RenderCamera, viewport_size: [f32; 2]) -> Self {
        Self {
            render_camera,
            viewport_size,
            pixels_per_point: 1.0,
            viewport_index: 0,
        }
    }

    /// Build a camera frame from an app-side camera and viewport size.
    pub fn from_camera(cam: &crate::camera::Camera, viewport_size: [f32; 2]) -> Self {
        Self::new(RenderCamera::from_camera(cam), viewport_size)
    }

    /// Set the physical pixels per logical pixel for this camera frame.
    ///
    /// Pass `ui.ctx().pixels_per_point()` from egui, or the equivalent scale
    /// factor from your window system. Sizes the physical render target and
    /// rasterises overlay text at display resolution; all screen-space math and
    /// overlay authoring stay in logical `viewport_size` units.
    pub fn with_pixels_per_point(mut self, pixels_per_point: f32) -> Self {
        self.pixels_per_point = pixels_per_point.max(0.001);
        self
    }

    /// Set the multi-viewport slot index for this camera frame.
    pub fn with_viewport_index(mut self, viewport_index: usize) -> Self {
        self.viewport_index = viewport_index;
        self
    }

    /// Set the multi-viewport slot from a ViewportId returned by ViewportRenderer::create_viewport.
    /// Prefer this over with_viewport_index when you have a ViewportId.
    pub fn with_viewport_id(mut self, id: crate::renderer::ViewportId) -> Self {
        self.viewport_index = id.0;
        self
    }
}

/// Surface submission type for world-space geometry.
///
/// For 0.2.0, only `Flat` submission is supported. This enum leaves room for
/// future large-scene or chunked submission without changing the `SceneFrame`
/// public type.
#[non_exhaustive]
pub enum SurfaceSubmission {
    /// A flat, reference-counted list of scene render items.
    ///
    /// Holding an `Arc<[SceneRenderItem]>` instead of a `Vec` means the per-frame
    /// submission cost is a single atomic refcount increment rather than a full deep
    /// copy of all items. Use [`SceneFrame::from_surface_items`] or
    /// [`SceneFrame::from_scene`] to construct this variant.
    Flat(std::sync::Arc<[SceneRenderItem]>),
}

impl Default for SurfaceSubmission {
    fn default() -> Self {
        SurfaceSubmission::Flat(std::sync::Arc::from([]))
    }
}

// ---------------------------------------------------------------------------
// Surface LIC
// ---------------------------------------------------------------------------

/// Configuration for Surface Line Integral Convolution.
///
/// Controls the advection quality and visual strength of the LIC effect.
/// All fields have sensible defaults via [`SurfaceLICConfig::default`].
///
/// The noise texture is viewport-sized (one independent random value per screen pixel).
/// Advection kernel length is `steps * step_size` pixels in each direction. Longer kernels
/// produce clearer, smoother streaks; shorter kernels give more contrast at lower GPU cost.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SurfaceLICConfig {
    /// Number of advection steps taken in each direction (forward and backward) from each pixel.
    /// More steps produce longer, clearer streaks at the cost of GPU time. Default: 20.
    pub steps: u32,
    /// Distance advanced per step, in screen pixels. Together with `steps`, controls total
    /// streak length: `steps * step_size` pixels each way. Default: 1.5.
    pub step_size: f32,
    /// How strongly the LIC intensity modulates the surface colour. At 0 there is no effect;
    /// at 1.0 the surface colour is scaled by up to 2x brighter or darkened to black depending
    /// on the local LIC value. Values above 1.0 increase contrast further. Default: 1.0.
    pub strength: f32,
}

impl Default for SurfaceLICConfig {
    fn default() -> Self {
        Self {
            steps: 20,
            step_size: 1.5,
            strength: 1.0,
        }
    }
}

/// World-space scene content for one frame.
///
/// Groups all renderable world-space content submitted to the renderer.
/// Surfaces are submitted through [`SurfaceSubmission`]; scientific
/// visualization primitives sit alongside them.
#[non_exhaustive]
pub struct SceneFrame {
    /// Scene version counter from `Scene::version()`. Default: 0 (triggers rebuild on first frame).
    ///
    /// The renderer uses this to skip batch rebuild and GPU upload when scene content
    /// has not changed since the previous frame.
    pub generation: u64,
    /// Surface geometry submission (opaque and transparent meshes).
    pub surfaces: SurfaceSubmission,
    /// Point cloud items to render this frame.
    pub point_clouds: Vec<PointCloudItem>,
    /// References to pre-uploaded point clouds.
    pub point_cloud_refs: Vec<PointCloudRefItem>,
    /// Instanced glyph items to render this frame.
    pub glyphs: Vec<GlyphItem>,
    /// References to pre-uploaded glyph sets.
    pub glyph_set_refs: Vec<GlyphSetRefItem>,
    /// Polyline (streamline) items to render this frame.
    pub polylines: Vec<PolylineItem>,
    /// References to pre-uploaded polylines (one entry per draw). Each
    /// `PolylineRefItem` carries a handle into the renderer's polyline store
    /// plus a per-frame model matrix and item settings.
    pub polyline_refs: Vec<PolylineRefItem>,
    /// Volume items to render this frame via GPU ray-marching.
    pub volumes: Vec<VolumeItem>,
    /// Isoline (contour line) items to render on mesh surfaces.
    pub isolines: Vec<crate::geometry::isoline::IsolineItem>,
    /// Streamtube items to render this frame.
    pub streamtube_items: Vec<StreamtubeItem>,
    /// References to pre-uploaded streamtubes.
    pub streamtube_refs: Vec<StreamtubeRefItem>,
    /// Screen-space image overlay items to render this frame.
    pub screen_images: Vec<ScreenImageItem>,
    /// GPU implicit surface items to render this frame.
    pub gpu_implicit: Vec<crate::resources::GpuImplicitItem>,
    /// GPU marching cubes jobs to dispatch this frame.
    pub gpu_mc_jobs: Vec<crate::resources::GpuMarchingCubesJob>,
    /// GPU compute filter items dispatched before the render pass.
    ///
    /// Each item references a pre-uploaded mesh and a compute kernel that
    /// rewrites its index buffer (culling, LOD selection). Empty by default;
    /// an empty list adds no dispatch and no allocations.
    pub compute_filter_items: Vec<ComputeFilterItem>,
    /// Unstructured volume meshes submitted this frame.
    ///
    /// Each [`VolumeMeshItem`] renders either as a boundary surface (default,
    /// through the standard mesh pipeline) or as a volumetric draw through
    /// projected tetrahedra (when [`VolumeMeshItem::transparency`] is set and
    /// the item was uploaded via
    /// [`upload_volume_mesh_with_transparency`](crate::resources::DeviceResources::upload_volume_mesh_with_transparency)).
    /// Cell-level picking, selection outlines, and wireframe overlays are
    /// driven from this collection regardless of mode.
    pub volume_meshes: Vec<VolumeMeshItem>,
    /// General tube items to render this frame.
    pub tube_items: Vec<TubeItem>,
    /// References to pre-uploaded tubes.
    pub tube_refs: Vec<TubeRefItem>,
    /// 2D image slice items to render this frame.
    pub image_slices: Vec<ImageSliceItem>,
    /// Tensor glyph items to render this frame.
    pub tensor_glyphs: Vec<TensorGlyphItem>,
    /// References to pre-uploaded tensor glyph sets.
    pub tensor_glyph_set_refs: Vec<TensorGlyphSetRefItem>,
    /// Ribbon items to render this frame.
    pub ribbon_items: Vec<RibbonItem>,
    /// References to pre-uploaded ribbons.
    pub ribbon_refs: Vec<RibbonRefItem>,
    /// Volume surface slice items to render this frame.
    pub volume_surface_slices: Vec<VolumeSurfaceSliceItem>,
    /// Billboard sprite items to render this frame.
    pub sprite_items: Vec<SpriteItem>,
    /// References to pre-uploaded sprite sets (static billboards).
    pub sprite_set_refs: Vec<SpriteSetRefItem>,
    /// References to pre-uploaded sprite instance sets (entity sprites).
    pub sprite_instance_set_refs: Vec<SpriteInstanceSetRefItem>,
    /// Mesh-instance batches to render this frame (mesh-based particles).
    pub mesh_instances: Vec<MeshInstanceItem>,
    /// GPU particle systems to advance and draw this frame.
    pub gpu_particle_systems: Vec<GpuParticleSystemItem>,
    /// External instance sets to draw this frame (mesh per element of a
    /// consumer-owned GPU positions buffer).
    pub external_instances: Vec<ExternalInstancesItem>,
    /// Gaussian splat items to render this frame.
    pub gaussian_splats: Vec<GaussianSplatItem>,
    /// Screen-space decal items to render this frame (D1).
    pub decals: Vec<DecalItem>,
    /// Participating-media volumes (fog, smoke, clouds) to render this frame.
    pub scatter_volumes: Vec<ScatterVolumeItem>,
    /// Scene-graph light sources to union with `EffectsFrame::lighting.lights`.
    ///
    /// Populate via [`crate::scene::scene::Scene::collect_lights`]. The renderer
    /// chains these with the frame-data lights before building the GPU uniform;
    /// consumers that use only `EffectsFrame::lighting` leave this empty.
    pub lights: Vec<LightSource>,
    /// Items drawn by the foreground pass, on top of the finished scene.
    ///
    /// Foreground items render after the world, against a freshly cleared
    /// depth buffer, so they are neither occluded by scene geometry nor clip
    /// into it. They skip instancing, shadow casting, frustum culling,
    /// picking, and selection outlines. Skinning, deformers, and materials
    /// work as for normal surface items. Configure the pass (optional
    /// override projection) via `EffectsFrame::foreground`.
    ///
    /// Empty by default; an empty list adds no pass and no allocations.
    pub foreground_items: Vec<SceneRenderItem>,
    /// Plugin item collections, keyed by
    /// [`ItemTypePlugin::type_name`](crate::plugin_api::ItemTypePlugin::type_name).
    ///
    /// Populated via [`Self::submit_plugin_items`]. Empty by default; the
    /// renderer iterates this map during `prepare` / `paint` and dispatches
    /// to the matching registered plugin. Entries whose `type_name` is not
    /// registered on the renderer are silently ignored.
    pub plugin_items:
        std::collections::HashMap<&'static str, Box<dyn crate::plugin_api::PluginItemCollection>>,
}

impl Default for SceneFrame {
    fn default() -> Self {
        Self {
            generation: 0,
            surfaces: SurfaceSubmission::default(),
            point_clouds: Vec::new(),
            point_cloud_refs: Vec::new(),
            glyphs: Vec::new(),
            glyph_set_refs: Vec::new(),
            polylines: Vec::new(),
            polyline_refs: Vec::new(),
            volumes: Vec::new(),
            isolines: Vec::new(),
            streamtube_items: Vec::new(),
            streamtube_refs: Vec::new(),
            screen_images: Vec::new(),
            gpu_implicit: Vec::new(),
            gpu_mc_jobs: Vec::new(),
            compute_filter_items: Vec::new(),
            volume_meshes: Vec::new(),
            tube_items: Vec::new(),
            tube_refs: Vec::new(),
            image_slices: Vec::new(),
            tensor_glyphs: Vec::new(),
            tensor_glyph_set_refs: Vec::new(),
            ribbon_items: Vec::new(),
            ribbon_refs: Vec::new(),
            volume_surface_slices: Vec::new(),
            sprite_items: Vec::new(),
            sprite_set_refs: Vec::new(),
            sprite_instance_set_refs: Vec::new(),
            mesh_instances: Vec::new(),
            gpu_particle_systems: Vec::new(),
            external_instances: Vec::new(),
            gaussian_splats: Vec::new(),
            decals: Vec::new(),
            scatter_volumes: Vec::new(),
            lights: Vec::new(),
            foreground_items: Vec::new(),
            plugin_items: std::collections::HashMap::new(),
        }
    }
}

impl SceneFrame {
    /// Build a scene frame from a surface submission.
    pub fn new(surfaces: SurfaceSubmission) -> Self {
        Self {
            surfaces,
            ..Self::default()
        }
    }

    /// Build a scene frame from a flat list of surface render items.
    ///
    /// The `Vec` is converted to an `Arc<[SceneRenderItem]>` so callers that
    /// submit the same list repeatedly can cheaply clone the `Arc` instead of
    /// cloning the underlying data.
    ///
    /// The frame's `generation` defaults to `0`. If items change every frame
    /// (e.g. from physics writeback or snapshot interpolation), chain
    /// [`with_generation`](Self::with_generation) to supply a value that
    /// changes with the content -- otherwise the renderer's instance-buffer
    /// cache will treat every frame as identical after the first upload:
    ///
    /// ```rust,ignore
    /// // Static scene -- generation 0 is fine, items never change.
    /// SceneFrame::from_surface_items(items)
    ///
    /// // Dynamic scene -- pass a value that increments when content changes.
    /// // scene.version() is the right source when writeback drives the items.
    /// SceneFrame::from_surface_items(items).with_generation(scene.version())
    /// ```
    pub fn from_surface_items(items: Vec<SceneRenderItem>) -> Self {
        Self::new(SurfaceSubmission::Flat(items.into()))
    }

    /// Override the generation stamp on a scene frame.
    ///
    /// The renderer compares `generation` against the previous frame to decide
    /// whether to rebuild and re-upload the instance buffer. A generation of
    /// `0` (the default for [`from_surface_items`](Self::from_surface_items))
    /// causes the cache to hit on every frame after the first, freezing
    /// rendered objects even if item transforms have changed.
    ///
    /// Pass any value that changes when content changes. `scene.version()` is
    /// the natural choice when items are derived from scene writeback.
    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    /// Submit a plugin item collection under `type_name`.
    ///
    /// The renderer dispatches per-frame work and draw calls to the
    /// matching [`ItemTypePlugin`](crate::plugin_api::ItemTypePlugin)
    /// registered with the same `type_name`. Submitting under a name
    /// that has no registered plugin is a no-op.
    ///
    /// Calling this twice with the same name replaces the previous
    /// submission.
    pub fn submit_plugin_items<C: crate::plugin_api::PluginItemCollection>(
        &mut self,
        type_name: &'static str,
        items: C,
    ) {
        self.plugin_items.insert(type_name, Box::new(items));
    }

    /// Build a scene frame from an already-allocated shared slice.
    ///
    /// Use this variant when you cache render items across frames:
    ///
    /// ```rust,ignore
    /// // First frame / on change: rebuild the Arc once.
    /// self.items_arc = Arc::from(scene.collect_render_items(&sel));
    ///
    /// // Every frame: zero-cost clone.
    /// SceneFrame::from_shared_items(Arc::clone(&self.items_arc), scene.version())
    /// ```
    pub fn from_shared_items(items: std::sync::Arc<[SceneRenderItem]>, generation: u64) -> Self {
        Self {
            generation,
            surfaces: SurfaceSubmission::Flat(items),
            ..Self::default()
        }
    }

    /// Build a scene frame by collecting render items from a [`Scene`](crate::scene::scene::Scene).
    ///
    /// Calls [`Scene::collect_render_items`](crate::scene::scene::Scene::collect_render_items)
    /// and stamps `generation` with the current scene version so the renderer can
    /// skip batch rebuilds on unchanged frames.
    ///
    /// This is the preferred constructor for the common single-viewport path.
    /// Use [`SceneFrame::from_surface_items`] when you need to assemble render items manually.
    pub fn from_scene(
        scene: &mut crate::scene::scene::Scene,
        selection: &crate::interaction::select::selection::Selection,
    ) -> Self {
        let items = scene.collect_render_items(selection);
        let lights = scene.collect_lights();
        let (light_glyphs, light_polylines) = crate::scene::build_light_glyphs(scene, selection);
        Self {
            generation: scene.version(),
            surfaces: SurfaceSubmission::Flat(items.into()),
            lights,
            glyphs: light_glyphs,
            polylines: light_polylines,
            ..Self::default()
        }
    }
}

/// Viewport presentation settings for one frame.
///
/// Groups background, grid, and axes indicator : the viewport chrome that is
/// independent of world-space content.
#[non_exhaustive]
pub struct ViewportFrame {
    /// Optional background/clear colour [r, g, b, a]. None = adapter default.
    pub background_colour: Option<[f32; 4]>,
    /// Whether to render the scene in wireframe mode. Default: false.
    pub wireframe_mode: bool,
    /// Whether to render the ground-plane grid. Default: false.
    pub show_grid: bool,
    /// Grid cell size in world units. Zero = camera-distance-based adaptive spacing.
    pub grid_cell_size: f32,
    /// Half-extent of the grid in world units. Zero = 1000 (effectively infinite).
    pub grid_half_extent: f32,
    /// World-space Z coordinate of the grid plane (3D mode only, Z-up). Default: 0.0.
    pub grid_z: f32,
    /// RGB colour for the grid lines. None = renderer default (mid-grey).
    pub grid_colour: Option<[f32; 3]>,
    /// Whether to draw the axes orientation indicator overlay. Default: true.
    pub show_axes_indicator: bool,
}

impl Default for ViewportFrame {
    fn default() -> Self {
        Self {
            background_colour: None,
            wireframe_mode: false,
            show_grid: false,
            grid_cell_size: 0.0,
            grid_half_extent: 0.0,
            grid_z: 0.0,
            grid_colour: None,
            show_axes_indicator: true,
        }
    }
}

/// Interaction and selection visualization state for one frame.
///
/// Groups the gizmo, selection overlays, constraint guides, outline, and
/// x-ray state : everything that communicates selection and interaction
/// feedback to the user.
#[non_exhaustive]
pub struct InteractionFrame {
    /// Selection version counter from `Selection::version()`. Default: 0.
    ///
    /// The renderer uses this to skip batch rebuild and GPU upload when selection
    /// state has not changed since the previous frame.
    pub selection_generation: u64,
    /// Gizmo model matrix. Some = selected object exists and gizmo should render.
    pub gizmo_model: Option<glam::Mat4>,
    /// Current gizmo interaction mode.
    pub gizmo_mode: GizmoMode,
    /// Current hovered gizmo axis.
    pub gizmo_hovered: GizmoAxis,
    /// Orientation for gizmo space (identity for world, object orientation for local).
    pub gizmo_space_orientation: glam::Quat,
    /// Constraint guide lines to render this frame.
    pub constraint_overlays: Vec<ConstraintOverlay>,
    /// Draw a stencil-outline ring around selected objects. Default: false.
    pub outline_selected: bool,
    /// RGBA colour of the selection outline ring. Default: white [1.0, 1.0, 1.0, 1.0].
    pub outline_colour: [f32; 4],
    /// Width of the outline ring in pixels. Default: 2.0.
    pub outline_width_px: f32,
    /// Render selected objects as a semi-transparent x-ray overlay. Default: false.
    pub xray_selected: bool,
    /// RGBA colour of the x-ray tint (should have alpha < 1). Default: [0.3, 0.7, 1.0, 0.25].
    pub xray_colour: [f32; 4],

    // --- Sub-object highlight ---
    /// Sub-object selection to highlight this frame.
    ///
    /// `None` = no sub-object highlights drawn. When `Some`, the renderer
    /// builds face fill, edge outline, and vertex/point sprite geometry from
    /// the snapshot and draws them after the opaque scene pass.
    pub sub_selection: Option<SubSelectionRef>,
    /// Fill colour (RGBA) for selected faces. The alpha component controls
    /// fill opacity. Default: translucent yellow `[1.0, 0.85, 0.0, 0.25]`.
    pub sub_highlight_face_fill_colour: [f32; 4],
    /// Edge colour (RGBA) for selected face outlines. Default: opaque yellow
    /// `[1.0, 0.85, 0.0, 1.0]`.
    pub sub_highlight_edge_colour: [f32; 4],
    /// Line width in pixels for face edge outlines. Default: `2.0`.
    pub sub_highlight_edge_width_px: f32,
    /// Point sprite size in pixels for selected vertices and point cloud
    /// points. Default: `10.0`.
    pub sub_highlight_vertex_size_px: f32,
}

impl Default for InteractionFrame {
    fn default() -> Self {
        Self {
            selection_generation: 0,
            gizmo_model: None,
            gizmo_mode: GizmoMode::Translate,
            gizmo_hovered: GizmoAxis::None,
            gizmo_space_orientation: glam::Quat::IDENTITY,
            constraint_overlays: Vec::new(),
            outline_selected: false,
            outline_colour: [1.0, 1.0, 1.0, 1.0],
            outline_width_px: 2.0,
            xray_selected: false,
            xray_colour: [0.3, 0.7, 1.0, 0.25],
            sub_selection: None,
            sub_highlight_face_fill_colour: [1.0, 0.85, 0.0, 0.25],
            sub_highlight_edge_colour: [1.0, 0.85, 0.0, 1.0],
            sub_highlight_edge_width_px: 2.0,
            sub_highlight_vertex_size_px: 10.0,
        }
    }
}

impl InteractionFrame {
    /// Build an interaction frame stamped with the current selection version.
    ///
    /// Sets `selection_generation` from [`Selection::version`](crate::interaction::select::selection::Selection::version)
    /// so the renderer can skip overlay rebuilds on unchanged frames.
    /// All other fields remain at their defaults.
    pub fn from_selection(selection: &crate::interaction::select::selection::Selection) -> Self {
        Self {
            selection_generation: selection.version(),
            ..Self::default()
        }
    }
}

// Ground-plane config (`GroundPlaneMode`, `GroundPlane`) lives in
// viewport-lib-types; re-exported so `crate::renderer::types::*` paths hold.
pub use viewport_lib_types::effects::ground::{GroundPlane, GroundPlaneMode};

/// When set on `EffectsFrame::environment`, the renderer uses the environment
/// map for PBR ambient lighting (irradiance + specular) and optionally renders
/// it as the scene background (skybox).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvironmentSettings {
    /// Absolute luminance scale in **nits** applied to the sampled environment -
    /// both the IBL contribution (diffuse irradiance + specular reflections) and
    /// the skybox background, so the lit surfaces and the visible sky stay
    /// physically consistent. Default: `1.0`.
    ///
    /// The stored environment map carries relative radiance; this scales it to
    /// the physical brightness the sky should read at, on the same nits scale as
    /// emissive surfaces (see [`Material::emissive_strength`](crate::Material)).
    /// A clear daytime sky is on the order of thousands of nits, so under
    /// photometric exposure a value near `1.0` leaves the environment nearly
    /// black; raise it to the sky's real luminance. The metering of an HDRI's
    /// own peaks is unchanged - this is a single physical multiplier, not a
    /// tonemap.
    pub intensity: f32,
    /// Y-axis rotation in radians. Default: 0.0.
    pub rotation: f32,
    /// Whether to render the environment as a visible skybox background.
    /// When false, IBL still contributes lighting but the background uses
    /// `ViewportFrame::background_colour`. Default: true.
    pub show_skybox: bool,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            rotation: 0.0,
            show_skybox: true,
        }
    }
}

// Scatter-volume pass config (`ScatterQuality`, `ScatterSettings`) lives in
// viewport-lib-types; re-exported so `crate::renderer::types::*` paths hold.
pub use viewport_lib_types::effects::scatter::{ScatterQuality, ScatterSettings};

/// Configuration for the foreground pass.
///
/// The foreground pass draws `SceneFrame::foreground_items` (and any
/// item-type plugin implementing `paint_foreground`) over the finished
/// scene, against a cleared depth buffer. Setting this to `Some` is only
/// needed to override the projection; the pass itself runs whenever
/// foreground items are submitted.
///
/// Notes:
/// - Foreground items are never sliced by scene clip planes.
/// - Transparent foreground items use sorted back-to-front alpha blending,
///   not the OIT pass.
/// - The pass runs in the HDR path and in the owned-encoder LDR path. It is
///   not available through `paint_to` style host-owned render passes, where
///   a cleared depth attachment cannot exist; submitting foreground items
///   there logs a warning once and draws nothing.
/// - In the LDR path, 2D overlays are drawn inside the scene pass, so
///   foreground items paint over them. The HDR path keeps overlays on top.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForegroundPass {
    /// Projection override for the pass. `None` reuses the scene projection
    /// (always-on-top gizmos, x-ray parts, HUD props). `Some` replaces the
    /// projection while keeping the scene view transform: a first-person
    /// item gets its own field of view and a near plane small enough for
    /// close-held geometry without changing world depth precision.
    pub projection: Option<ForegroundProjection>,
}

/// Override projection for the foreground pass.
///
/// Shares the main camera's view transform; only the projection differs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForegroundProjection {
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance. `None` uses the scene camera's far plane.
    pub far: Option<f32>,
}

/// Global rendering effects and modifiers for one frame.
///
/// Groups lighting, clipping, post-processing, and clip
/// volumes : effects that apply globally across the scene rather than to
/// individual objects.
#[non_exhaustive]
pub struct EffectsFrame {
    /// Per-frame lighting configuration.
    pub lighting: LightingSettings,
    /// Participating-media (scatter-volume) quality settings.
    pub scatter: ScatterSettings,
    /// Clip objects (planes/box/sphere) and cap-fill toggle. Default: no clipping.
    pub clip: ClipSettings,
    /// Display transform: pipeline mode (HDR / Direct), exposure, and tone-map
    /// operator. Default: HDR pipeline, neutral manual EV 0, KhronosNeutral.
    pub display: DisplaySettings,
    /// Post-processing effects (bloom, SSAO, DOF, ...). Default: all off. Every
    /// effect requires `display.mode == PipelineMode::Hdr`.
    pub post_process: PostProcessSettings,
    /// Foreground pass configuration (projection override). Default: None.
    /// The pass itself is driven by `SceneFrame::foreground_items`; this only
    /// carries pass-wide settings.
    pub foreground: Option<ForegroundPass>,
    /// Optional environment settings for IBL and skybox. Default: None.
    pub environment: Option<EnvironmentSettings>,
    /// Ground plane configuration. Default: mode = None (not drawn, zero overhead).
    pub ground_plane: GroundPlane,
    /// Debug overlays (shadow-atlas viewer). Default: off.
    pub debug: EffectsDebug,
}

impl Default for EffectsFrame {
    fn default() -> Self {
        Self {
            lighting: LightingSettings::default(),
            scatter: ScatterSettings::default(),
            clip: ClipSettings::default(),
            display: DisplaySettings::default(),
            post_process: PostProcessSettings::default(),
            foreground: None,
            environment: None,
            ground_plane: GroundPlane::default(),
            debug: EffectsDebug::default(),
        }
    }
}

/// Clip geometry for one frame: active clip objects plus the cross-section
/// cap-fill toggle. Grouped on [`EffectsFrame::clip`].
#[non_exhaustive]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClipSettings {
    /// Active clip objects (planes, boxes, spheres). Max 6 planes + 1 box/sphere.
    /// Default: empty (no clipping).
    pub objects: Vec<ClipObject>,
    /// Whether to render filled caps at clip plane cross-sections. Default: true.
    pub cap_fill_enabled: bool,
}

impl Default for ClipSettings {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            cap_fill_enabled: true,
        }
    }
}

// `EffectsDebug` lives in viewport-lib-types (effects::debug); re-exported
// so `crate::renderer::types::*` paths keep resolving.
pub use viewport_lib_types::effects::debug::EffectsDebug;

/// Scene-global effects for one frame, consumed by [`ViewportRenderer::prepare_scene`].
///
/// Groups the lighting, environment, and scatter configuration that applies
/// to the whole scene (not per-viewport). Construct directly or obtain via
/// [`EffectsFrame::split`]. Compute filter items travel with the scene content
/// on [`SceneFrame::compute_filter_items`].
///
/// # Multi-viewport usage
/// Call [`ViewportRenderer::prepare_scene`] once per frame with this struct.
/// Each viewport's per-viewport effects are passed separately via
/// [`ViewportEffects`] in [`ViewportRenderer::prepare_viewport`].
pub struct SceneEffects<'a> {
    /// Per-frame lighting configuration (drives the shadow pass and light uniform).
    pub lighting: &'a LightingSettings,
    /// Optional environment settings for IBL and skybox.
    pub environment: &'a Option<EnvironmentSettings>,
    /// Participating-media quality settings (scene-global).
    pub scatter: &'a ScatterSettings,
}

/// Per-viewport effects for one frame, consumed by [`ViewportRenderer::prepare_viewport`].
///
/// Groups the clip objects and post-processing settings that differ
/// per viewport. Construct directly or obtain via [`EffectsFrame::split`].
///
/// # Multi-viewport usage
/// Pass one `ViewportEffects` per viewport to [`ViewportRenderer::prepare_viewport`].
/// Scene-global effects are passed once via [`SceneEffects`] in
/// [`ViewportRenderer::prepare_scene`].
pub struct ViewportEffects<'a> {
    /// Clip objects and cap-fill toggle.
    pub clip: &'a ClipSettings,
    /// Display transform: pipeline mode, exposure, tone-map operator.
    pub display: &'a DisplaySettings,
    /// Post-processing effects (bloom, SSAO, DOF, ...).
    pub post_process: &'a PostProcessSettings,
    /// Foreground pass configuration (projection override).
    pub foreground: &'a Option<ForegroundPass>,
    /// Ground plane configuration for this viewport.
    pub ground_plane: &'a GroundPlane,
    /// Debug overlays (shadow-atlas viewer).
    pub debug: &'a EffectsDebug,
}

/// A coherent lighting + exposure starting point.
///
/// Whether a scene is "faithful" or "physical daylight" is not one field - it
/// emerges from the light magnitudes in [`EffectsFrame::lighting`] agreeing with
/// the exposure in [`EffectsFrame::display`]. Setting one without the other is the
/// common footgun: real-lux lights under the neutral default exposure clip to
/// white, and an adaptive daylight camera over nominal magnitudes silently loses
/// the "colour is data" look. `LightingPosture` sets both halves together via
/// [`EffectsFrame::with_posture`] so they cannot disagree.
///
/// This governs lighting and exposure only. The per-material
/// [`ShadingModel`](crate::ShadingModel) is chosen independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LightingPosture {
    /// "Colour is data": nominal light magnitudes shown at a fixed neutral
    /// exposure with no camera adaptation, so authored colours read on screen
    /// roughly as-is. The default posture, matching [`EffectsFrame::default`].
    #[default]
    Faithful,
    /// Physically-scaled daylight: real photometric magnitudes
    /// ([`LightingSettings::daylight`]) mapped down by an adaptive camera
    /// ([`ExposureSettings::automatic`]) so the sun does not clip to white.
    PhysicalDaylight,
}

impl EffectsFrame {
    /// Apply a [`LightingPosture`], setting [`EffectsFrame::lighting`] and the
    /// exposure on [`EffectsFrame::display`] as a matched pair.
    ///
    /// Pick the intent in one call instead of hand-pairing light magnitudes with
    /// an exposure mode - the two must agree or the scene clips (real lux at
    /// neutral exposure) or loses the colour-is-data look (nominal magnitudes
    /// under an adaptive camera). Call it early, then author your own lights
    /// (push onto `effects.lighting.lights`) and adjust other display fields:
    ///
    /// ```no_run
    /// # use viewport_lib::{EffectsFrame, LightingPosture};
    /// let effects = EffectsFrame::default().with_posture(LightingPosture::PhysicalDaylight);
    /// ```
    ///
    /// Leaves other display fields (pipeline mode, tone-map operator) and the
    /// per-material [`ShadingModel`](crate::ShadingModel) untouched.
    pub fn with_posture(mut self, posture: LightingPosture) -> Self {
        match posture {
            LightingPosture::Faithful => {
                self.lighting = LightingSettings::default();
                self.display.exposure = ExposureSettings::default();
            }
            LightingPosture::PhysicalDaylight => {
                self.lighting = LightingSettings::daylight();
                self.display.exposure = ExposureSettings::automatic();
            }
        }
        self
    }

    /// Decompose into scene-global and per-viewport effect references.
    ///
    /// Both halves borrow from `self` and cannot outlive the `EffectsFrame`.
    /// The scene half is passed to [`ViewportRenderer::prepare_scene`]; the
    /// viewport half is passed to [`ViewportRenderer::prepare_viewport`].
    ///
    /// Single-viewport callers can continue using [`ViewportRenderer::prepare`]
    /// directly without calling `split()`.
    pub fn split(&self) -> (SceneEffects<'_>, ViewportEffects<'_>) {
        (
            SceneEffects {
                lighting: &self.lighting,
                environment: &self.environment,
                scatter: &self.scatter,
            },
            ViewportEffects {
                clip: &self.clip,
                display: &self.display,
                post_process: &self.post_process,
                foreground: &self.foreground,
                ground_plane: &self.ground_plane,
                debug: &self.debug,
            },
        )
    }
}
