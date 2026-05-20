use super::*;
use crate::interaction::gizmo::{GizmoAxis, GizmoMode};
use crate::interaction::snap::ConstraintOverlay;
use crate::interaction::sub_object::SubSelectionRef;
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
    pub fn camera_uniform(&self) -> CameraUniform {
        let vp = self.view_proj();
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            eye_pos: self.eye_position,
            _pad: 0.0,
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
    /// The renderer uses this only for GPU texture allocation. All screen-space
    /// math (projection, picking, widget layout) uses `viewport_size` directly.
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
    /// factor from your window system. Only affects GPU texture allocation;
    /// all screen-space math continues to use `viewport_size`.
    pub fn with_pixels_per_point(mut self, pixels_per_point: f32) -> Self {
        self.pixels_per_point = pixels_per_point.max(0.001);
        self
    }

    /// Set the multi-viewport slot index for this camera frame.
    pub fn with_viewport_index(mut self, viewport_index: usize) -> Self {
        self.viewport_index = viewport_index;
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
// Phase 4: Surface LIC
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

/// A mesh surface to render with Surface LIC for one frame.
///
/// The mesh must have been uploaded via [`ViewportGpuResources::upload_mesh_data`]
/// with a `VertexVector` attribute matching `vector_attribute`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SurfaceLICItem {
    /// ID of the mesh to render (obtained from `upload_mesh_data`).
    pub mesh_id: crate::resources::mesh_store::MeshId,
    /// Name of the `AttributeData::VertexVector` attribute on that mesh.
    pub vector_attribute: String,
    /// Model matrix (row-major 4x4).
    pub model: [[f32; 4]; 4],
    /// LIC rendering configuration.
    pub config: SurfaceLICConfig,
    /// Per-item appearance overrides (hidden, unlit, opacity, wireframe).
    pub appearance: crate::scene::material::AppearanceSettings,
}

impl SurfaceLICItem {
    /// Create a new `SurfaceLICItem` with the given mesh, vector attribute, model matrix,
    /// and LIC configuration.
    pub fn new(
        mesh_id: crate::resources::mesh_store::MeshId,
        vector_attribute: impl Into<String>,
        model: [[f32; 4]; 4],
        config: SurfaceLICConfig,
    ) -> Self {
        Self {
            mesh_id,
            vector_attribute: vector_attribute.into(),
            model,
            config,
            appearance: crate::scene::material::AppearanceSettings::default(),
        }
    }
}

/// A transparent unstructured volume mesh rendered via projected tetrahedra.
///
/// Created by uploading a [`VolumeMeshData`](crate::resources::VolumeMeshData) with
/// [`ViewportGpuResources::upload_projected_tet_mesh`] and submitting the returned
/// [`ProjectedTetId`](crate::resources::ProjectedTetId) each frame.
///
/// # Picking
///
/// Set `pick_id` to a non-zero value and provide `volume_mesh_data` to enable cell-level
/// picking via `renderer.pick()` and `renderer.pick_rect()`. `volume_mesh_data` must be
/// the same data passed to `upload_projected_tet_mesh` for this item.
///
/// # Selection highlight
///
/// Set `selected = true` to draw an object-level outline ring around this mesh.
/// Also supply `boundary_mesh_id` with the `MeshId` returned by
/// [`upload_volume_mesh_data`](crate::resources::ViewportGpuResources::upload_volume_mesh_data)
/// for the same geometry; the outline pass uses that boundary surface to compute
/// the silhouette. Without `boundary_mesh_id` the `selected` flag has no visual effect.
#[derive(Clone)]
#[non_exhaustive]
pub struct TransparentVolumeMeshItem {
    /// Handle to the uploaded projected-tet mesh.
    pub id: crate::resources::ProjectedTetId,
    /// Pick ID returned when a cell in this mesh is hit.
    ///
    /// `0` means this mesh is not pickable. Set to a non-zero value alongside
    /// `volume_mesh_data` to enable cell picking.
    pub pick_id: u64,
    /// CPU mesh data for cell picking.
    ///
    /// Must match the data passed to `upload_projected_tet_mesh` for `id`.
    /// `None` disables cell-level picking regardless of `pick_id`.
    pub volume_mesh_data: Option<std::sync::Arc<crate::resources::volume_mesh::VolumeMeshData>>,
    /// Beer-Lambert extinction coefficient (world-space units).
    ///
    /// Higher values make the volume more opaque.  Typical range: 0.1 to 5.0.
    pub density: f32,
    /// Override the auto-detected scalar range `[min, max]` used to normalize colourmap lookup.
    ///
    /// `None` uses the range computed at upload time.
    pub scalar_range: Option<(f32, f32)>,
    /// Scalar threshold range. Tetrahedra whose scalar value falls outside
    /// `[threshold_min, threshold_max]` are discarded by the shader and not rendered.
    ///
    /// Operates on raw (un-normalized) scalar values, the same units as the data
    /// passed to `upload_projected_tet_mesh`. Default: no clipping (all tets rendered).
    pub threshold_min: f32,
    /// Upper scalar threshold. See `threshold_min`.
    pub threshold_max: f32,
    /// Per-item appearance overrides (hidden, unlit, opacity, wireframe).
    pub appearance: crate::scene::material::AppearanceSettings,
    /// Draw a selection outline ring around this mesh when `true`.
    ///
    /// Requires `boundary_mesh_id` to be set; see the type-level docs.
    pub selected: bool,
    /// Boundary surface mesh used for the selection outline mask pass.
    ///
    /// Pass the `MeshId` returned by `upload_volume_mesh_data` for the same
    /// geometry. The mesh positions are assumed to be in world space (identity
    /// model matrix).
    pub boundary_mesh_id: Option<crate::resources::mesh_store::MeshId>,
}

impl TransparentVolumeMeshItem {
    /// Create a visible item with default density of 1.0 and auto scalar range.
    pub fn new(id: crate::resources::ProjectedTetId) -> Self {
        Self {
            id,
            pick_id: 0,
            volume_mesh_data: None,
            density: 1.0,
            scalar_range: None,
            threshold_min: f32::NEG_INFINITY,
            threshold_max: f32::INFINITY,
            appearance: crate::scene::material::AppearanceSettings::default(),
            selected: false,
            boundary_mesh_id: None,
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
    /// Instanced glyph items to render this frame.
    pub glyphs: Vec<GlyphItem>,
    /// Polyline (streamline) items to render this frame.
    pub polylines: Vec<PolylineItem>,
    /// Volume items to render this frame via GPU ray-marching.
    pub volumes: Vec<VolumeItem>,
    /// Isoline (contour line) items to render on mesh surfaces.
    pub isolines: Vec<crate::geometry::isoline::IsolineItem>,
    /// Streamtube items to render this frame.
    pub streamtube_items: Vec<StreamtubeItem>,
    /// Camera frustum wireframe items to render this frame (Phase 10).
    pub camera_frustums: Vec<CameraFrustumItem>,
    /// Screen-space image overlay items to render this frame (Phase 10).
    pub screen_images: Vec<ScreenImageItem>,
    /// GPU implicit surface items to render this frame (Phase 16).
    pub gpu_implicit: Vec<crate::resources::GpuImplicitItem>,
    /// GPU marching cubes jobs to dispatch this frame (Phase 17).
    pub gpu_mc_jobs: Vec<crate::resources::GpuMarchingCubesJob>,
    /// Surface LIC items to render this frame (Phase 4).
    pub lic_items: Vec<SurfaceLICItem>,
    /// Transparent unstructured volume meshes rendered via projected tetrahedra (Phase 6).
    pub transparent_volume_meshes: Vec<TransparentVolumeMeshItem>,
    /// Opaque volume mesh items submitted for cell-level picking.
    ///
    /// Each entry enables `SubObjectRef::Cell` hits from `renderer.pick()` and
    /// `renderer.pick_rect()` for the mesh identified by `pick_id`. The item's
    /// `face_to_cell` map (from `upload_volume_mesh_data`) converts surface
    /// triangle hits to originating cell indices.
    ///
    /// Rendering still goes through `surfaces`; submit the `SceneRenderItem`
    /// from `VolumeMeshItem::to_render_item()` there as normal.
    pub volume_mesh_items: Vec<VolumeMeshItem>,
    /// General tube items to render this frame (Phase 3).
    pub tube_items: Vec<TubeItem>,
    /// 2D image slice items to render this frame (Phase 3).
    pub image_slices: Vec<ImageSliceItem>,
    /// Tensor glyph items to render this frame (Phase 5).
    pub tensor_glyphs: Vec<TensorGlyphItem>,
    /// Ribbon items to render this frame (Phase 8.1).
    pub ribbon_items: Vec<RibbonItem>,
    /// Volume surface slice items to render this frame (Phase 10).
    pub volume_surface_slices: Vec<VolumeSurfaceSliceItem>,
    /// Billboard sprite items to render this frame.
    pub sprite_items: Vec<SpriteItem>,
    /// Gaussian splat items to render this frame.
    pub gaussian_splats: Vec<GaussianSplatItem>,
}

impl Default for SceneFrame {
    fn default() -> Self {
        Self {
            generation: 0,
            surfaces: SurfaceSubmission::default(),
            point_clouds: Vec::new(),
            glyphs: Vec::new(),
            polylines: Vec::new(),
            volumes: Vec::new(),
            isolines: Vec::new(),
            streamtube_items: Vec::new(),
            camera_frustums: Vec::new(),
            screen_images: Vec::new(),
            gpu_implicit: Vec::new(),
            gpu_mc_jobs: Vec::new(),
            lic_items: Vec::new(),
            transparent_volume_meshes: Vec::new(),
            volume_mesh_items: Vec::new(),
            tube_items: Vec::new(),
            image_slices: Vec::new(),
            tensor_glyphs: Vec::new(),
            ribbon_items: Vec::new(),
            volume_surface_slices: Vec::new(),
            sprite_items: Vec::new(),
            gaussian_splats: Vec::new(),
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
        selection: &crate::interaction::selection::Selection,
    ) -> Self {
        let items = scene.collect_render_items(selection);
        Self {
            generation: scene.version(),
            surfaces: SurfaceSubmission::Flat(items.into()),
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
    /// Sets `selection_generation` from [`Selection::version`](crate::interaction::selection::Selection::version)
    /// so the renderer can skip overlay rebuilds on unchanged frames.
    /// All other fields remain at their defaults.
    pub fn from_selection(selection: &crate::interaction::selection::Selection) -> Self {
        Self {
            selection_generation: selection.version(),
            ..Self::default()
        }
    }
}

/// Environment map configuration for IBL (image-based lighting) and skybox.
///
/// Ground plane rendering mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GroundPlaneMode {
    /// No ground plane rendered (default, zero overhead).
    #[default]
    None,
    /// Invisible plane that receives and displays shadows only.
    ShadowOnly,
    /// Procedural checkerboard tile pattern.
    Tile,
    /// Flat solid colour.
    SolidColour,
}

/// Ground plane configuration for the viewport.
///
/// Renders a large horizontal plane at a configurable world-space Z height.
/// Provides spatial grounding without explicit scene geometry.
#[derive(Clone, Debug)]
pub struct GroundPlane {
    /// Rendering mode. Default: `None` (plane not drawn).
    pub mode: GroundPlaneMode,
    /// World-space Z coordinate of the ground plane. Default: `0.0`.
    pub height: f32,
    /// Primary colour for `Tile` and `SolidColour` modes. Default: `[0.3, 0.3, 0.3, 1.0]`.
    pub colour: [f32; 4],
    /// Secondary tile colour for `Tile` mode. Default: `[0.5, 0.5, 0.5, 1.0]`.
    pub tile_colour2: [f32; 4],
    /// Checker tile size in world units (`Tile` mode). Default: `1.0`.
    pub tile_size: f32,
    /// Shadow tint colour (`ShadowOnly` mode). Default: `[0.0, 0.0, 0.0, 1.0]`.
    pub shadow_colour: [f32; 4],
    /// Maximum shadow opacity (`ShadowOnly` mode). `0.0` = transparent, `1.0` = fully opaque. Default: `0.5`.
    pub shadow_opacity: f32,
}

impl Default for GroundPlane {
    fn default() -> Self {
        Self {
            mode: GroundPlaneMode::None,
            height: 0.0,
            colour: [1.0, 1.0, 1.0, 1.0],
            tile_colour2: [0.0, 0.0, 0.0, 1.0],
            tile_size: 1.0,
            shadow_colour: [0.0, 0.0, 0.0, 1.0],
            shadow_opacity: 0.5,
        }
    }
}

/// When set on `EffectsFrame::environment`, the renderer uses the environment
/// map for PBR ambient lighting (irradiance + specular) and optionally renders
/// it as the scene background (skybox).
#[derive(Clone, Debug)]
pub struct EnvironmentMap {
    /// Intensity multiplier for IBL contribution. Default: 1.0.
    pub intensity: f32,
    /// Y-axis rotation in radians. Default: 0.0.
    pub rotation: f32,
    /// Whether to render the environment as a visible skybox background.
    /// When false, IBL still contributes lighting but the background uses
    /// `ViewportFrame::background_colour`. Default: true.
    pub show_skybox: bool,
}

impl Default for EnvironmentMap {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            rotation: 0.0,
            show_skybox: true,
        }
    }
}

/// Global rendering effects and modifiers for one frame.
///
/// Groups lighting, clipping, post-processing, compute filtering, and clip
/// volumes : effects that apply globally across the scene rather than to
/// individual objects.
#[non_exhaustive]
pub struct EffectsFrame {
    /// Per-frame lighting configuration.
    pub lighting: LightingSettings,
    /// Active clip objects (planes, boxes, spheres). Max 6 planes + 1 box/sphere.
    /// Default: empty (no clipping).
    pub clip_objects: Vec<ClipObject>,
    /// Whether to render filled caps at clip plane cross-sections. Default: true.
    pub cap_fill_enabled: bool,
    /// Post-processing settings. Default: enabled (HDR pipeline active, all effects off).
    pub post_process: PostProcessSettings,
    /// GPU compute filter items dispatched before the render pass.
    pub compute_filter_items: Vec<ComputeFilterItem>,
    /// Optional environment map for IBL and skybox. Default: None.
    pub environment: Option<EnvironmentMap>,
    /// Ground plane configuration. Default: mode = None (not drawn, zero overhead).
    pub ground_plane: GroundPlane,
}

impl Default for EffectsFrame {
    fn default() -> Self {
        Self {
            lighting: LightingSettings::default(),
            clip_objects: Vec::new(),
            cap_fill_enabled: true,
            post_process: PostProcessSettings::default(),
            compute_filter_items: Vec::new(),
            environment: None,
            ground_plane: GroundPlane::default(),
        }
    }
}

/// Scene-global effects for one frame, consumed by [`ViewportRenderer::prepare_scene`].
///
/// Groups the lighting, environment, and compute-filter configuration that applies
/// to the whole scene (not per-viewport). Construct directly or obtain via
/// [`EffectsFrame::split`].
///
/// # Multi-viewport usage
/// Call [`ViewportRenderer::prepare_scene`] once per frame with this struct.
/// Each viewport's per-viewport effects are passed separately via
/// [`ViewportEffects`] in [`ViewportRenderer::prepare_viewport`].
pub struct SceneEffects<'a> {
    /// Per-frame lighting configuration (drives the shadow pass and light uniform).
    pub lighting: &'a LightingSettings,
    /// Optional environment map for IBL and skybox.
    pub environment: &'a Option<EnvironmentMap>,
    /// GPU compute filter items dispatched before the render pass.
    pub compute_filter_items: &'a [ComputeFilterItem],
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
    /// Active clip objects (planes, boxes, spheres).
    pub clip_objects: &'a [ClipObject],
    /// Whether to render filled caps at clip plane cross-sections.
    pub cap_fill_enabled: bool,
    /// Optional post-processing settings (tone mapping, bloom, SSAO).
    pub post_process: &'a PostProcessSettings,
    /// Ground plane configuration for this viewport.
    pub ground_plane: &'a GroundPlane,
}

impl EffectsFrame {
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
                compute_filter_items: &self.compute_filter_items,
            },
            ViewportEffects {
                clip_objects: &self.clip_objects,
                cap_fill_enabled: self.cap_fill_enabled,
                post_process: &self.post_process,
                ground_plane: &self.ground_plane,
            },
        )
    }
}
