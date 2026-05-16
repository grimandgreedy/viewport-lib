#![warn(missing_docs)]
//! `viewport-lib` : a 3D viewport library for `wgpu` applications.
//!
//! Built on `wgpu` and `glam`, with no required UI toolkit. The crate provides
//! the renderer, camera, picking, and interaction code; host applications keep
//! their own windowing and event loop.
//!
//! # Quick start (single viewport)
//!
//! 1. Create a [`ViewportRenderer`] from a `wgpu::Device` and target format.
//! 2. Upload meshes or volumes through [`ViewportGpuResources`].
//! 3. Build a [`FrameData`] each frame (camera via [`CameraFrame`] and
//!    [`RenderCamera`], scene content via [`SceneFrame`], viewport chrome via
//!    [`ViewportFrame`], etc.).
//! 4. Call [`ViewportRenderer::prepare`] and then [`ViewportRenderer::paint_to`].
//!
//! # Multi-viewport rendering
//!
//! To render the same scene from multiple independent cameras (e.g. a CAD
//! quad-view layout), use the split prepare/paint API:
//!
//! ```rust,ignore
//! // --- Setup (once at startup) ---
//! let vp_persp = renderer.create_viewport(&device);   // ViewportId(0)
//! let vp_top   = renderer.create_viewport(&device);   // ViewportId(1)
//! let vp_front = renderer.create_viewport(&device);   // ViewportId(2)
//!
//! // --- Each frame ---
//! // 1. Build per-viewport FrameData (shared scene, independent cameras).
//! let frame_persp = FrameData {
//!     camera: CameraFrame::from_camera(&cam_persp, vp_size).with_viewport_index(vp_persp.0),
//!     scene: shared_scene.clone(),
//!     ..FrameData::default()
//! };
//! // ... similarly for vp_top, vp_front ...
//!
//! // 2. Prepare scene data once (lighting, shadows, batching).
//! let (scene_fx, _) = frame_persp.effects.split();
//! renderer.prepare_scene(&device, &queue, &frame_persp, &scene_fx);
//!
//! // 3. Prepare per-viewport state (camera uniforms, clip planes, overlays).
//! renderer.prepare_viewport(&device, &queue, vp_persp, &frame_persp);
//! renderer.prepare_viewport(&device, &queue, vp_top,   &frame_top);
//! renderer.prepare_viewport(&device, &queue, vp_front, &frame_front);
//!
//! // 4a. LDR path : single render pass with viewport/scissor rects.
//! let mut rp = encoder.begin_render_pass(...);
//! rp.set_viewport(0.0, 0.0, half_w, half_h, 0.0, 1.0);
//! renderer.paint_viewport(&mut rp, vp_persp, &frame_persp);
//! rp.set_viewport(half_w, 0.0, half_w, half_h, 0.0, 1.0);
//! renderer.paint_viewport(&mut rp, vp_top, &frame_top);
//! // ...
//!
//! // 4b. HDR path : one command buffer per viewport, each into its own texture.
//! let cmd0 = renderer.render_viewport(&device, &queue, &view0, vp_persp, &frame_persp);
//! let cmd1 = renderer.render_viewport(&device, &queue, &view1, vp_top,   &frame_top);
//! queue.submit([cmd0, cmd1]);
//! ```
//!
//! Single-viewport applications require zero code changes:
//! [`prepare`](ViewportRenderer::prepare), [`paint_to`](ViewportRenderer::paint_to), and
//! [`render`](ViewportRenderer::render) continue to work as before.

/// Error types for the viewport library.
pub mod error;

/// Arcball camera, frustum, view presets, and animator.
pub mod camera;
/// BVH picking, marching cubes, isolines, and cap geometry.
pub mod geometry;
/// Gizmo, snap, selection, picking, and input.
pub mod interaction;
/// On-surface vector quantities (intrinsic vectors, Whitney one-forms).
pub mod quantities;
/// Main viewport renderer wrapping all GPU resources.
pub mod renderer;
/// Scene runtime: per-frame orchestration, plugin system, and physics hooks.
pub mod runtime;
/// GPU resource container (pipelines, buffers, bind groups).
pub mod resources;
/// Scene graph, material, traits, and AABB.
pub mod scene;
/// Axes orientation indicator.
pub mod widgets;

// ---------------------------------------------------------------------------
// Module re-exports : preserve old `viewport_lib::foo::Bar` paths.
// ---------------------------------------------------------------------------

pub use geometry::bvh;
pub use geometry::primitives;
pub use interaction::clip_plane;
pub use interaction::gizmo;
pub use interaction::input;
pub use interaction::manipulation;
pub use interaction::picking;
pub use interaction::selection;
pub use interaction::snap;
pub use scene::aabb;
pub use scene::material;
pub use scene::traits;
pub use widgets::axes_indicator;

// ---------------------------------------------------------------------------
// Flat re-exports : these form the public crate API.
// ---------------------------------------------------------------------------

pub use error::{ViewportError, ViewportResult};

pub use camera::animator::{CameraAnimator, CameraDamping, Easing};
pub use camera::camera::{Camera, CameraTarget, Projection};
pub use camera::frustum::{CullStats, Frustum};
pub use camera::track::{CameraTrack, interpolate_camera};
pub use camera::turntable::TurntableController;
pub use camera::view_preset::ViewPreset;

pub use scene::aabb::Aabb;
pub use scene::material::{
    AppearanceSettings, BackfacePattern, BackfacePolicy, Material, ParamVis, ParamVisMode,
    PatternConfig,
};
pub use scene::scene::{Group, GroupId, Layer, LayerId, Scene, SceneNode};
pub use scene::traits::{RenderMode, ViewportObject};

pub use geometry::bvh::PickAccelerator;
pub use geometry::implicit::{
    ImplicitRenderOptions, march_implicit_surface, march_implicit_surface_colour,
};
pub use geometry::isoline::{IsolineItem, extract_isolines};
pub use geometry::marching_cubes::{VolumeData, extract_isosurface};

pub use interaction::gizmo::{
    Gizmo, GizmoAxis, GizmoMode, GizmoSpace, PivotMode, gizmo_center_for_pivot,
};
pub use interaction::input::{
    Action, ActionState, Binding, FrameInput, InputMode, InputSystem, KeyCode, Modifiers,
    MouseButton, NavigationMode,
};
// New input pipeline : re-exported at crate root for convenience.
pub use interaction::input::{
    ActionFrame, BindingPreset, ButtonState, ModifiersMatch, NavigationActions,
    OrbitCameraController, ResolvedActionState, ScrollUnits, ViewportBinding, ViewportContext,
    ViewportEvent, ViewportGesture, ViewportInput, viewport_all_bindings,
};
pub use interaction::manipulation::solvers::{
    angular_rotation_from_cursor, constrained_scale, constrained_translation,
};
pub use interaction::manipulation::{
    GizmoInfo, ManipResult, ManipulationContext, ManipulationController, ManipulationKind,
    ManipulationState, TransformDelta,
};

pub use interaction::widgets::{
    BoxWidget, CylinderWidget, DiskWidget, LineProbeWidget, PlaneWidget, PolylineWidget,
    SphereWidget, SplineWidget, WidgetContext, WidgetResult,
};

pub use interaction::clip_plane::{
    ClipAxis, ClipPlaneContext, ClipPlaneController, ClipPlaneDelta, ClipPlaneHit, ClipPlaneResult,
    ClipPlaneSessionKind, hit_test_normal_handle, hit_test_plane_quad, plane_from_axis_preset,
    project_drag_onto_normal, ray_plane_intersection, snap_plane_distance,
};
pub use interaction::pick_mask::PickMask;
pub use interaction::picking::{
    GpuPickHit, PickHit, ProbeBinding, RectPickResult, nearest_vertex_on_hit,
    pick_gaussian_splat_cpu, pick_gaussian_splat_rect, pick_point_cloud_cpu, pick_rect,
    pick_scene_accelerated_with_probe_cpu, pick_scene_nodes_with_probe_cpu,
    pick_scene_with_probe_cpu, pick_transparent_volume_mesh_cpu, pick_transparent_volume_mesh_rect,
    pick_volume_cpu, pick_volume_rect, voxel_world_aabb,
};
pub use interaction::selection::{NodeId, Selection};
pub use interaction::snap::{ConstraintOverlay, SnapConfig};
pub use interaction::sub_object;
pub use interaction::sub_object::{
    CellSelectionInfo, PolylineSelectionInfo, SubObjectRef, SubSelection, SubSelectionRef,
    VolumeSelectionInfo,
};

pub use widgets::axes_indicator::AxisView;

pub use renderer::shader_hashes::ShaderValidation;
pub use renderer::stats::{FrameStats, PerformancePolicy, QualityPreset, RuntimeMode};
pub use renderer::{
    CameraFrame, CameraFrustumItem, ClipObject, ClipShape, ComputeFilterItem, ComputeFilterKind,
    EffectsFrame, EnvironmentMap, FilterMode, FrameData, GaussianSplatData, GaussianSplatId,
    GaussianSplatItem, GlyphItem, GlyphType, GroundPlane, GroundPlaneMode, ImageAnchor,
    ImageSliceItem, InteractionFrame, LabelAnchor, LabelItem, LightKind, LightSource,
    LightingSettings, LoadingBarAnchor, LoadingBarItem, OverlayFrame, OverlayImageItem, PickId,
    PickRectResult, PointCloudItem, PointRenderMode, PolylineItem, PostProcessSettings,
    RenderCamera, RibbonItem, RulerItem, ScalarBarAnchor, ScalarBarItem, ScalarBarOrientation,
    SceneEffects, SceneFrame, SceneRenderItem, ScreenImageItem, ShDegree, ShadowFilter, SliceAxis,
    SpriteItem, SpriteSizeMode, StreamtubeItem, SurfaceLICConfig, SurfaceLICItem,
    SurfaceSubmission, TensorGlyphItem, ToneMapping, TransparentVolumeMeshItem, TubeItem,
    ViewportEffects, ViewportFrame, ViewportId, ViewportRenderer, VolumeItem, VolumeMeshItem,
    VolumeSurfaceSliceItem, aabb_wireframe_polyline,
};

pub use quantities::{
    edge_one_form_to_glyphs, face_intrinsic_to_glyphs, polyline_edge_vectors_to_glyphs,
    polyline_node_vectors_to_glyphs, vertex_intrinsic_to_glyphs,
    volume_mesh_cell_vectors_to_glyphs, volume_mesh_vertex_vectors_to_glyphs,
};

#[allow(deprecated)]
pub use resources::ClipVolumeUniform;
pub use resources::colourmap_data::{
    export_paraview_xml_colourmap, lerp_colourmap_lut, parse_paraview_xml_colourmap,
};
pub use resources::mesh_store::MeshId;
pub use resources::sparse_volume::SparseVolumeGridData;
#[allow(deprecated)]
pub use resources::volume_mesh::{
    CELL_SENTINEL, TET_SENTINEL, VolumeMeshData, extract_clipped_volume_faces,
};
pub use resources::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColourmap, BuiltinMatcap, CLIP_VOLUME_MAX,
    CameraUniform, ClipVolumeEntry, ClipVolumesUniform, ColourmapId, ComputeFilterResult, FontError,
    FontHandle, GpuImplicitItem, GpuImplicitOptions, GpuMarchingCubesJob, ImplicitBlendMode,
    ImplicitPrimitive, LightUniform, LightsUniform, MatcapId, MeshData, ProjectedTetId,
    SingleLightUniform, ViewportGpuResources, VolumeGpuId, VolumeId, lerp_attributes,
};

pub use runtime::{
    ContactEvent, FixedStepIter, FixedTimestep, NodeTransformOp, RuntimeFrameContext,
    RuntimeOutput, RuntimePhase, RuntimePlugin, RuntimeStepContext, SceneRuntimeMode,
    SelectionOp, SimulationStepContext, TransformSnapshot, TransformSnapshotTable,
    TransformWriteback, ViewportRuntime,
};
