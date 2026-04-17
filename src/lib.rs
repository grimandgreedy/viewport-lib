#![warn(missing_docs)]
//! `viewport-lib` — a 3D viewport library for `wgpu` applications.
//!
//! Built on `wgpu` and `glam`, with no required UI toolkit. The crate provides
//! the renderer, camera, picking, and interaction pieces; host applications keep
//! control of their own windowing and event loop.
//!
//! # Quick start
//!
//! 1. Create a [`ViewportRenderer`] from a `wgpu::Device` and target format.
//! 2. Upload meshes or volumes through [`ViewportGpuResources`].
//! 3. Build a [`FrameData`] each frame.
//! 4. Call [`ViewportRenderer::prepare`] and then [`ViewportRenderer::paint_to`].

/// Error types for the viewport library.
pub mod error;

/// Arcball camera, frustum, view presets, and animator.
pub mod camera;
/// Scene graph, material, traits, and AABB.
pub mod scene;
/// BVH picking, marching cubes, isolines, and cap geometry.
pub mod geometry;
/// Gizmo, snap, selection, annotation, picking, and input.
pub mod interaction;
/// Axes orientation indicator.
pub mod widgets;
/// Main viewport renderer wrapping all GPU resources.
pub mod renderer;
/// GPU resource container (pipelines, buffers, bind groups).
pub mod resources;

// ---------------------------------------------------------------------------
// Module re-exports — preserve old `viewport_lib::foo::Bar` paths.
// ---------------------------------------------------------------------------

pub use geometry::bvh;
pub use interaction::annotation;
pub use interaction::gizmo;
pub use interaction::input;
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

pub use camera::camera::{Camera, Projection};
pub use camera::animator::{CameraAnimator, CameraDamping, Easing};
pub use camera::frustum::{CullStats, Frustum};
pub use camera::view_preset::ViewPreset;

pub use scene::aabb::Aabb;
pub use scene::material::Material;
pub use scene::scene::{Group, GroupId, Layer, LayerId, Scene, SceneNode};
pub use scene::traits::{RenderMode, ViewportObject};

pub use geometry::bvh::PickAccelerator;
pub use geometry::isoline::{IsolineItem, extract_isolines};
pub use geometry::marching_cubes::{VolumeData, extract_isosurface};

pub use interaction::annotation::AnnotationLabel;
#[cfg(feature = "egui")]
pub use interaction::annotation::draw_annotation_labels;
pub use interaction::annotation::{world_to_screen, world_to_screen_from_frame};
pub use interaction::gizmo::{Gizmo, GizmoAxis, GizmoMode, GizmoSpace, PivotMode, gizmo_center_for_pivot};
pub use interaction::input::{
    Action, ActionState, Binding, FrameInput, InputMode, InputSystem, KeyCode, Modifiers,
    MouseButton,
};
pub use interaction::picking::{
    GpuPickHit, PickHit, ProbeBinding, RectPickResult, pick_rect,
    pick_scene_accelerated_with_probe, pick_scene_nodes_with_probe, pick_scene_with_probe,
};
pub use interaction::selection::{NodeId, Selection};
pub use interaction::snap::{ConstraintOverlay, SnapConfig};

pub use widgets::axes_indicator::AxisView;

pub use renderer::{
    ClipPlane, ClipVolume, ComputeFilterItem, ComputeFilterKind, FilterMode, FrameData, GlyphItem,
    GlyphType, LightKind, LightSource, LightingSettings, OverlayQuad, PointCloudItem,
    PointRenderMode, PolylineItem, PostProcessSettings, SceneRenderItem, ShadowFilter,
    StreamtubeItem, ToneMapping, ViewportRenderer, VolumeItem,
};
pub use renderer::{ScalarBar, ScalarBarAnchor, ScalarBarOrientation};
pub use renderer::stats::FrameStats;
pub use renderer::shader_hashes::ShaderValidation;

pub use resources::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColormap, CameraUniform, ClipVolumeUniform,
    ColormapId, ComputeFilterResult, LightUniform, LightsUniform, MeshData, SingleLightUniform,
    ViewportGpuResources, VolumeId, lerp_attributes,
};
pub use resources::colormap_data::{
    export_paraview_xml_colormap, lerp_colormap_lut, parse_paraview_xml_colormap,
};
pub use resources::mesh_store::MeshId;
