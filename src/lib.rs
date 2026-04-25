#![warn(missing_docs)]
//! `viewport-lib` — a 3D viewport library for `wgpu` applications.
//!
//! Built on `wgpu` and `glam`, with no required UI toolkit. The crate provides
//! the renderer, camera, picking, and interaction pieces; host applications keep
//! control of their own windowing and event loop.
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
//! // 4a. LDR path — single render pass with viewport/scissor rects.
//! let mut rp = encoder.begin_render_pass(...);
//! rp.set_viewport(0.0, 0.0, half_w, half_h, 0.0, 1.0);
//! renderer.paint_viewport(&mut rp, vp_persp, &frame_persp);
//! rp.set_viewport(half_w, 0.0, half_w, half_h, 0.0, 1.0);
//! renderer.paint_viewport(&mut rp, vp_top, &frame_top);
//! // ...
//!
//! // 4b. HDR path — one command buffer per viewport, each into its own texture.
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
/// Gizmo, snap, selection, annotation, picking, and input.
pub mod interaction;
/// Main viewport renderer wrapping all GPU resources.
pub mod renderer;
/// GPU resource container (pipelines, buffers, bind groups).
pub mod resources;
/// Scene graph, material, traits, and AABB.
pub mod scene;
/// On-surface vector quantities (intrinsic vectors, Whitney one-forms).
pub mod quantities;
/// Axes orientation indicator.
pub mod widgets;

// ---------------------------------------------------------------------------
// Module re-exports — preserve old `viewport_lib::foo::Bar` paths.
// ---------------------------------------------------------------------------

pub use geometry::bvh;
pub use geometry::primitives;
pub use interaction::annotation;
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
// Flat re-exports — these form the public crate API.
// ---------------------------------------------------------------------------

pub use error::{ViewportError, ViewportResult};

pub use camera::animator::{CameraAnimator, CameraDamping, Easing};
pub use camera::camera::{Camera, CameraTarget, Projection};
pub use camera::frustum::{CullStats, Frustum};
pub use camera::view_preset::ViewPreset;

pub use scene::aabb::Aabb;
pub use scene::material::{BackfacePolicy, Material, ParamVis, ParamVisMode};
pub use scene::scene::{Group, GroupId, Layer, LayerId, Scene, SceneNode};
pub use scene::traits::{RenderMode, ViewportObject};

pub use geometry::bvh::PickAccelerator;
pub use geometry::isoline::{IsolineItem, extract_isolines};
pub use geometry::marching_cubes::{VolumeData, extract_isosurface};

pub use interaction::annotation::AnnotationLabel;
#[cfg(feature = "egui")]
pub use interaction::annotation::draw_annotation_labels;
pub use interaction::annotation::{world_to_screen, world_to_screen_from_frame};
pub use interaction::gizmo::{
    Gizmo, GizmoAxis, GizmoMode, GizmoSpace, PivotMode, gizmo_center_for_pivot,
};
pub use interaction::input::{
    Action, ActionState, Binding, FrameInput, InputMode, InputSystem, KeyCode, Modifiers,
    MouseButton, NavigationMode,
};
// New input pipeline — re-exported at crate root for convenience.
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

pub use interaction::clip_plane::{
    ClipAxis, ClipPlaneContext, ClipPlaneController, ClipPlaneDelta, ClipPlaneHit,
    ClipPlaneResult, ClipPlaneSessionKind, hit_test_normal_handle,
    hit_test_plane_quad, plane_from_axis_preset, project_drag_onto_normal, ray_plane_intersection,
    snap_plane_distance,
};
pub use interaction::picking::{
    GpuPickHit, PickHit, ProbeBinding, RectPickResult, pick_rect,
    pick_scene_accelerated_with_probe, pick_scene_nodes_with_probe, pick_scene_with_probe,
};
pub use interaction::selection::{NodeId, Selection};
pub use interaction::snap::{ConstraintOverlay, SnapConfig};
pub use interaction::sub_object;
pub use interaction::sub_object::{SubObjectRef, SubSelection};

pub use widgets::axes_indicator::AxisView;

pub use renderer::shader_hashes::ShaderValidation;
pub use renderer::stats::FrameStats;
pub use renderer::{
    CameraFrame, CameraFrustumItem, ClipObject, ClipShape, ComputeFilterItem, ComputeFilterKind,
    EffectsFrame, EnvironmentMap, FilterMode, FrameData, GlyphItem, GlyphType, GroundPlane,
    GroundPlaneMode, ImageAnchor, InteractionFrame, LightKind, LightSource, LightingSettings,
    PointCloudItem, PointRenderMode, PolylineItem, PostProcessSettings, RenderCamera, SceneEffects,
    SceneFrame, SceneRenderItem, ScreenImageItem, ShadowFilter, StreamtubeItem, SurfaceSubmission,
    ToneMapping, ViewportEffects, ViewportFrame, ViewportId, ViewportRenderer, VolumeItem,
};
pub use renderer::{ScalarBar, ScalarBarAnchor, ScalarBarOrientation};

pub use quantities::{
    edge_one_form_to_glyphs, face_intrinsic_to_glyphs, vertex_intrinsic_to_glyphs,
};

pub use resources::colormap_data::{
    export_paraview_xml_colormap, lerp_colormap_lut, parse_paraview_xml_colormap,
};
pub use resources::mesh_store::MeshId;
pub use resources::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColormap, BuiltinMatcap, CameraUniform,
    ClipVolumeUniform, ColormapId, ComputeFilterResult, LightUniform, LightsUniform, MatcapId,
    MeshData, SingleLightUniform, ViewportGpuResources, VolumeId, lerp_attributes,
};
