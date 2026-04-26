//! Feature showcase for `viewport-lib` using `eframe` / `egui`.
//!
//! Showcase modes are selected from the top panel (buttons 1–9).
//!
//! ## Showcase 1 : Rendering Basics
//!   - Directional / point light toggle
//!   - Orthographic / perspective projection toggle
//!   - Blinn-Phong lighting with specular highlights
//!
//! ## Showcase 2 : Scene Graph + Materials
//!   - Cycle materials on selected nodes
//!   - Toggle transparency / normal visualisation / hemisphere ambient
//!   - Cycle background colour
//!   - Add child node to selection (hierarchy demo)
//!   - Remove selected node; toggle layer B visibility
//!   - Click to select objects
//!
//! ## Showcase 3 : Performance at Scale
//!   - 1 000 000 boxes (100×100×100 grid) sharing a single mesh : GPU instancing
//!   - BVH-accelerated picking: click to select objects
//!   - Live FrameStats in the side panel
//!
//! ## Showcase 4 : Professional Interaction
//!   - Smooth camera with exponential damping (orbit/pan/zoom inertia)
//!   - Animated fly-to view presets (Front/Back/Left/Right/Top/Bottom/Isometric)
//!   - Zoom-to-fit selection (or entire scene)
//!   - Translate / Rotate / Scale gizmo with axis constraints
//!   - Gizmo space toggle (World / Local)
//!   - Snap configurations (off -> 0.5 u translation -> 15° rotation)
//!
//! ## Showcase 5 : Advanced Rendering
//!   - PBR vs Blinn-Phong comparison (left/right split)
//!   - Section-view clip plane (clips x < 0)
//!   - Selection outline (orange stencil ring)
//!   - X-ray mode (selected objects visible through occluders)
//!
//! ## Showcase 6 : Post-Processing
//!   - PBR scene with multiple material types
//!   - Tone mapping selection (Reinhard / ACES / Khronos Neutral)
//!   - Exposure, bloom, SSAO, FXAA toggles
//!   (Note: full HDR pipeline requires direct surface access : not available in
//!   the eframe callback model; settings are applied where supported by the LDR path.)
//!
//! ## Showcase 7 : Normal Maps + AO
//!   - Sphere with procedural brick normal map + AO vs plain reference sphere
//!   - Cube with tile normal map + AO
//!   - Toggle normal map / AO on/off; clip plane
//!
//! ## Showcase 8 : Shadows
//!   - Cascaded shadow maps with PCF / PCSS filtering
//!   - Cascade count control (1–4)
//!   - Contact shadows toggle
//!
//! ## Showcase 9 : Annotation Labels
//!   - `world_to_screen` projection with on-screen text via egui painter
//!   - Leader lines, anchor dots, semi-transparent text backgrounds
//!   - Behind-camera labels are automatically culled
//!
//! ## Showcase 10 : Camera Tools
//!   - Seven `ViewPreset` named views (Front/Back/Left/Right/Top/Bottom/Isometric)
//!   - Animated fly-to transitions with selectable easing curve
//!   - Animated / instant toggle to compare smooth vs snapped transitions
//!   - `ViewPreset::preferred_projection()` switches ortho/persp automatically
//!   - Explicit projection radio + per-frame FOV slider (perspective only)
//!
//! ## Showcase 19 : Matcap Shading
//!
//! ## Showcase 20 : Per-Face Attributes
//!   - Left:   Vertex attribute (Gouraud-interpolated scalar, colormapped)
//!   - Centre: Face attribute (flat per-triangle scalar, same data : no interpolation)
//!   - Right:  FaceColor attribute (direct per-face RGBA, no colormap)
//!   - Opacity slider on the FaceColor object exercises the OIT blending path
//!   - All eight built-in matcap presets (Clay/Wax/Candy/Flat blendable; Ceramic/Jade/Mud/Normal static)
//!   - Base-color picker shows how blendable presets tint with `base_color`
//!   - Custom procedural matcap uploaded via `upload_matcap()` with a user-controlled hue
//!
//! ## Common Camera Controls
//!   - Left-drag or Middle-drag: Orbit
//!   - Right-drag or Shift+Middle-drag: Pan
//!   - Scroll wheel: Zoom

use std::collections::HashMap;

use eframe::egui;
use viewport_lib::{
    Action, AnnotationLabel, AttributeKind, AttributeRef, BuiltinColormap, ButtonState, Camera,
    CameraAnimator, CameraFrame, ClipAxis, ClipObject, ColormapId, Easing, FrameData, FrameStats,
    Gizmo, GizmoAxis, GizmoInfo, GizmoMode, GizmoSpace, GlyphType, GroundPlane, GroundPlaneMode,
    KeyCode, LightKind, LightSource, LightingSettings, ManipResult, ManipulationContext,
    ManipulationController, MatcapId, Material, MeshData, MeshId, NodeId, OrbitCameraController,
    PickAccelerator, PostProcessSettings, Projection, RenderCamera, SceneFrame, SceneRenderItem,
    ScrollUnits, Selection, ShadowFilter, SnapConfig, ViewPreset, ViewportContext, ViewportEvent,
    ViewportRenderer, VolumeData, VolumeId,
    geometry::isoline::IsolineItem,
    gizmo::{self, compute_gizmo_scale},
    scene::{LayerId, Scene},
};

mod geometry;
mod hdr_viewport_callback;
mod multi_viewport_callback;
mod showcase_02_scene_graph;
mod showcase_03_performance;
mod showcase_04_interaction;
mod showcase_05_advanced_rendering;
mod showcase_06_post_process;
mod showcase_07_normal_maps;
mod showcase_08_shadows;
mod showcase_09_annotation;
mod showcase_10_camera_tools;
mod showcase_11_lights;
mod showcase_12_scalar_fields;
mod showcase_13_multi_viewport;
mod showcase_14_isolines;
mod showcase_15_point_clouds;
mod showcase_16_streamlines;
mod showcase_17_volume;
mod showcase_18_clip_volumes;
mod showcase_19_matcap;
mod showcase_20_face_attributes;
mod showcase_21_textures;
mod showcase_22_parameterization;
mod showcase_23_ground_plane;
mod showcase_24_surface_appearance;
mod showcase_25_surface_vectors;
mod showcase_26_volume_mesh;
mod showcase_27_auxiliary;
mod viewport_callback;

const BG_COLOR: [f32; 4] = [0.22, 0.22, 0.24, 1.0];

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : eframe Showcase",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            ..Default::default()
        },
        Box::new(|cc| {
            let wgpu_render_state = cc
                .wgpu_render_state
                .as_ref()
                .expect("eframe must be configured with the wgpu backend");

            let device = wgpu_render_state.device.clone();
            let queue = wgpu_render_state.queue.clone();
            let format = wgpu_render_state.target_format;

            let renderer = ViewportRenderer::new(&device, format);
            wgpu_render_state
                .renderer
                .write()
                .callback_resources
                .insert(renderer);

            // HDR blit resources (used by Showcase 24 and any other mode that
            // needs the full post-processing pipeline via HdrViewportCallback).
            hdr_viewport_callback::init_hdr_blit_resources(wgpu_render_state, format);

            // Pre-upload box meshes for Showcase 1 (4 independent slots).
            let box_mesh = unit_box_mesh();
            let mut mesh_indices = Vec::new();
            {
                let mut guard = wgpu_render_state.renderer.write();
                if let Some(vr) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                    for _ in 0..4 {
                        let idx = vr
                            .resources_mut()
                            .upload_mesh_data(&device, &box_mesh)
                            .expect("showcase 1 box mesh");
                        mesh_indices.push(idx);
                    }
                }
            }

            Ok(Box::new(App {
                device,
                queue,
                camera: Camera {
                    center: glam::Vec3::ZERO,
                    distance: 12.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                },
                controller: OrbitCameraController::viewport_all(),
                mode: ShowcaseMode::Basic,
                show_keybinds: false,
                mesh_indices,
                use_point_light: false,
                scene: Scene::new(),
                selection: Selection::new(),
                box_mesh_data: box_mesh,
                material_cycle: 0,
                bg_cycle: 0,
                sg_outline_width: 4.0,

                layer_b: None,
                layer_b_visible: true,
                scene_built: false,
                perf_scene: Scene::new(),
                perf_selection: Selection::new(),
                pick_accelerator: None,
                perf_mesh: None,
                last_stats: FrameStats::default(),
                perf_total_objects: 0,
                perf_scene_items_cache: Vec::new(),
                perf_scene_items_version: (u64::MAX, u64::MAX),
                perf_built: false,
                interact_scene: Scene::new(),
                interact_selection: Selection::new(),
                interact_animator: CameraAnimator::with_default_damping(),
                interact_gizmo: Gizmo::new(),
                manip: ManipulationController::new(),
                interact_transforms_snapshot: HashMap::new(),
                interact_left_held: false,
                interact_built: false,
                interact_gizmo_center: None,
                interact_gizmo_scale: 1.0,
                last_cursor_viewport: glam::Vec2::ZERO,
                adv_scene: Scene::new(),
                adv_selection: Selection::new(),
                adv_clip_enabled: false,
                adv_outline_on: true,
                adv_xray_on: false,
                adv_built: false,
                pp_scene: Scene::new(),
                pp_built: false,
                pp_shadow_pcss: true,
                pp_point_light_on: true,
                pp_dir_intensity: 4.0,
                nm_scene: Scene::new(),
                nm_built: false,
                nm_mapped_nodes: Vec::new(),
                nm_normal_on: true,
                nm_ao_on: true,
                nm_clip_enabled: false,
                nm_cap_fill: true,
                shd_scene: Scene::new(),
                shd_built: false,
                shd_cascade_count: 4,
                shd_pcss_on: false,
                shd_contact_on: false,
                ann_scene: Scene::new(),
                ann_built: false,
                ann_labels: Vec::new(),
                cam_tools_scene: Scene::new(),
                cam_tools_built: false,
                cam_animator: CameraAnimator::with_default_damping(),
                lights_scene: Scene::new(),
                lights_sources: vec![LightSource::default()],
                lights_built: false,
                lights_hemi_on: true,
                lights_hemi_intensity: 1.0,
                lights_sky_color: [1.0, 1.0, 1.0],
                lights_ground_color: [1.0, 1.0, 1.0],
                scalar_scene: Scene::new(),
                scalar_built: false,
                scalar_selection: Selection::new(),
                scalar_colormap: viewport_lib::BuiltinColormap::Viridis,
                scalar_range_auto: true,
                scalar_range: (0.0, 1.0),
                scalar_nan_on: false,
                scalar_bar_anchor: viewport_lib::ScalarBarAnchor::BottomRight,
                scalar_bar_orientation: viewport_lib::ScalarBarOrientation::Vertical,
                scalar_node_ids: [0; 3],
                scalar_mesh_indices: [0; 3],
                scalar_pick_positions: [Vec::new(), Vec::new(), Vec::new()],
                scalar_pick_indices: [Vec::new(), Vec::new(), Vec::new()],
                scalar_values: [Vec::new(), Vec::new(), Vec::new()],
                scalar_active_object: 0,
                mv_scene: Scene::new(),
                mv_selection: Selection::new(),
                mv_cameras: [
                    // TL: Perspective
                    Camera {
                        center: glam::Vec3::ZERO,
                        distance: 12.0,
                        orientation: glam::Quat::from_rotation_z(0.6)
                            * glam::Quat::from_rotation_x(1.1),
                        ..Camera::default()
                    },
                    // TR: Top (orthographic, looking -Z)
                    Camera {
                        center: glam::Vec3::ZERO,
                        distance: 10.0,
                        orientation: glam::Quat::IDENTITY,
                        projection: viewport_lib::Projection::Orthographic,
                        ..Camera::default()
                    },
                    // BL: Front (orthographic, looking -Y)
                    Camera {
                        center: glam::Vec3::ZERO,
                        distance: 10.0,
                        orientation: glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                        projection: viewport_lib::Projection::Orthographic,
                        ..Camera::default()
                    },
                    // BR: Right (orthographic, looking -X)
                    Camera {
                        center: glam::Vec3::ZERO,
                        distance: 10.0,
                        orientation: glam::Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)
                            * glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                        projection: viewport_lib::Projection::Orthographic,
                        ..Camera::default()
                    },
                ],
                mv_controllers: [
                    OrbitCameraController::viewport_primitives(),
                    OrbitCameraController::viewport_primitives(),
                    OrbitCameraController::viewport_primitives(),
                    OrbitCameraController::viewport_primitives(),
                ],
                mv_viewports: None,
                mv_built: false,
                mv_gizmo: Gizmo::new(),
                mv_gizmo_drag_active: false,
                mv_hovered_quad: 0,
                mv_cursor_local: glam::Vec2::ZERO,
                mv_gizmo_center: None,
                mv_gizmo_scales: [1.0; 4],
                mv_snap: SnapConfig::default(),
                mv_drag_accum_translation: glam::Vec3::ZERO,
                mv_drag_accum_rotation: 0.0,
                mv_drag_last_snapped_translation: glam::Vec3::ZERO,
                mv_drag_last_snapped_rotation: 0.0,
                iso_scene: Scene::new(),
                iso_mesh_index: 0,
                iso_positions: Vec::new(),
                iso_indices: Vec::new(),
                iso_scalars: Vec::new(),
                iso_grid_resolution: 128,
                iso_contour_count: 8,
                iso_line_color: [1.0, 1.0, 0.2, 1.0],
                iso_line_width: 1.5,
                iso_show_surface_color: true,
                iso_depth_bias: 0.005,
                iso_built: false,
                pc_built: false,
                pc_sub_mode: PcSubMode::PointCloud,
                pc_point_size: 4.0,
                pc_colormap: BuiltinColormap::Viridis,
                pc_scalar_range_manual: false,
                pc_scalar_range: (2.6, 3.4),
                pc_glyph_type: GlyphType::Arrow,
                pc_glyph_scale: 1.0,
                pc_glyph_magnitude_scale: true,
                pc_cloud_positions: Vec::new(),
                pc_cloud_scalars: Vec::new(),
                pc_field_positions: Vec::new(),
                pc_field_vectors: Vec::new(),
                stream_built: false,
                stream_use_tubes: false,
                stream_tube_radius: 0.06,
                stream_line_width: 4.0,
                stream_color_by_speed: true,
                stream_colormap: BuiltinColormap::Viridis,
                stream_flat_color: [0.4, 0.7, 1.0, 1.0],
                stream_seed_count: 32,
                stream_step_size: 0.08,
                stream_paths: Vec::new(),
                stream_scalars: Vec::new(),
                vol_built: false,
                vol_volume_id: None,
                vol_iso_mesh_index: None,
                vol_field: VolumeData {
                    data: Vec::new(),
                    dims: [1, 1, 1],
                    origin: [0.0; 3],
                    spacing: [1.0; 3],
                },
                vol_mode: showcase_17_volume::VolumeMode::VolumeOnly,
                vol_isovalue: 0.35,
                vol_color_lut: BuiltinColormap::Viridis,
                vol_opacity_scale: 1.0,
                vol_threshold: (0.05, 1.0),
                vol_step_scale: 1.0,
                vol_shading: true,
                vol_nan_on: false,
                vol_iso_material: {
                    let mut m = Material::from_color([0.6, 0.8, 1.0]);
                    m.roughness = 0.4;
                    m
                },
                clipvol_scene: Scene::new(),
                clipvol_built: false,
                clipvol_sub_mode: showcase_18_clip_volumes::ClipVolSubMode::BoxClip,
                clipvol_box_center: [0.0; 3],
                clipvol_box_half_extents: [2.5, 2.5, 2.5],
                clipvol_box_yaw: 0.0,
                clipvol_sphere_center: [0.0; 3],
                clipvol_sphere_radius: 2.8,
                clipvol_plane: {
                    let mut co = ClipObject::plane([0.0, 0.0, 1.0], 0.0);
                    co.color = Some([0.45, 0.82, 1.0, 0.5]);
                    co
                },
                clipvol_plane_axis: ClipAxis::Z,
                clipvol_gizmo: Gizmo::new(),
                clipvol_gizmo_center: None,
                clipvol_gizmo_scale: 1.0,
                clipvol_gizmo_drag_active: false,
                clipvol_show_overlay: true,
                matcap_scene: Scene::new(),
                matcap_built: false,

                matcap_builtin_node_ids: [0u64; 8],
                matcap_custom_node: None,
                matcap_custom_id: None,
                matcap_blendable_color: [0.7, 0.7, 0.7],
                matcap_custom_hue: 200.0,

                face_scene: Scene::new(),
                face_built: false,

                texture_scene: Scene::new(),
                texture_built: false,
                texture_plane_node: 0,
                face_mesh_indices: [0; 3],
                face_node_ids: [0; 3],
                face_colormap: BuiltinColormap::Viridis,
                face_opacity: 1.0,

                param_vis_scene: Scene::new(),
                param_vis_built: false,
                param_vis_node_ids: [0u64; 16],
                param_vis_scale: 8.0,
                param_vis_on: true,

                gp_scene: Scene::new(),
                gp_built: false,
                gp_mode: showcase_23_ground_plane::GpMode::Tile,
                gp_height: 0.0,
                gp_color: [0.3, 0.3, 0.3, 1.0],
                gp_tile_size: 1.0,
                gp_shadow_color: [0.0, 0.0, 0.0, 1.0],
                gp_shadow_opacity: 0.5,

                sa_built: false,
                sa_scene: Scene::new(),
                sa_clip_on: true,

                sv_built: false,
                sv_mode: showcase_25_surface_vectors::SvMode::VertexIntrinsic,
                sv_scale: 0.15,
                sv_density: 1.0,
                sv_glyph_density: -1.0,
                sv_mesh_index: [0; 3],
                sv_positions: [Vec::new(), Vec::new(), Vec::new()],
                sv_normals: [Vec::new(), Vec::new(), Vec::new()],
                sv_tangents: [None, None, None],
                sv_indices: [Vec::new(), Vec::new(), Vec::new()],
                sv_vertex_vecs: Vec::new(),
                sv_face_vecs: Vec::new(),
                sv_edge_vals: Vec::new(),

                vm_built: false,
                vm_mode: showcase_26_volume_mesh::VmMode::Hex,
                vm_tet_index: 0,
                vm_hex_index: 0,
                vm_colormap: BuiltinColormap::Viridis,
                vm_field: showcase_26_volume_mesh::VmField::Latitude,

                aux_built: false,
                aux_frustums: Vec::new(),
                aux_img_alpha: 1.0,
                aux_img_scale: 1.0,
            }))
        }),
    )
}

// ---------------------------------------------------------------------------
// Showcase mode
// ---------------------------------------------------------------------------

/// Sub-mode for Showcase 15 (Point Clouds & Glyphs).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PcSubMode {
    PointCloud,
    VectorField,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShowcaseMode {
    Basic,
    SceneGraph,
    Performance,
    Interaction,
    Advanced,
    PostProcess,
    NormalMaps,
    Shadows,
    Annotation,
    CameraTools,
    Lights,
    ScalarFields,
    MultiViewport,
    Isolines,
    PointClouds,
    Streamlines,
    Volume,
    ClipVolumes,
    Matcap,
    FaceAttributes,
    Textures,
    ParamVis,
    GroundPlane,
    SurfaceAppearance,
    SurfaceVectors,
    VolumeMesh,
    Auxiliary,
}

impl ShowcaseMode {
    fn label(self) -> &'static str {
        match self {
            Self::Basic => "1: Rendering Basics",
            Self::SceneGraph => "2: Scene Graph",
            Self::Performance => "3: Performance",
            Self::Interaction => "4: Interaction",
            Self::Advanced => "5: Advanced Rendering",
            Self::PostProcess => "6: Post-Processing",
            Self::NormalMaps => "7: Normal Maps",
            Self::Shadows => "8: Shadows",
            Self::Annotation => "9: Annotations",
            Self::CameraTools => "10: Camera Tools",
            Self::Lights => "11: Lights",
            Self::ScalarFields => "12: Scalar Fields",
            Self::MultiViewport => "13: Multi-Viewport",
            Self::Isolines => "14: Isolines & Contours",
            Self::PointClouds => "15: Point Clouds & Glyphs",
            Self::Streamlines => "16: Streamlines & Tubes",
            Self::Volume => "17: Volume & Isosurface",
            Self::ClipVolumes => "18: Clip Volumes",
            Self::Matcap => "19: Matcap Shading",
            Self::FaceAttributes => "20: Face Attributes",
            Self::Textures => "21: Textures",
            Self::ParamVis => "22: UV Parameterization",
            Self::GroundPlane => "23: Ground Plane",
            Self::SurfaceAppearance => "24: Surface Appearance",
            Self::SurfaceVectors => "25: Surface Vectors",
            Self::VolumeMesh => "26: Volume Meshes",
            Self::Auxiliary => "27: Auxiliary Structures",
        }
    }
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

pub(crate) struct App {
    // GPU handles (captured at startup for lazy mesh uploads).
    // wgpu::Device and Queue are internally ref-counted and implement Clone.
    device: eframe::wgpu::Device,
    queue: eframe::wgpu::Queue,

    camera: Camera,
    controller: OrbitCameraController,
    mode: ShowcaseMode,
    show_keybinds: bool,

    // --- Showcase 1 ---
    mesh_indices: Vec<usize>,
    use_point_light: bool,

    // --- Showcase 2 ---
    scene: Scene,
    selection: Selection,
    /// Shared box MeshData for on-demand uploads in later showcases.
    pub(crate) box_mesh_data: MeshData,
    material_cycle: usize,
    bg_cycle: usize,
    sg_outline_width: f32,
    pub(crate) layer_b: Option<LayerId>,
    layer_b_visible: bool,
    pub(crate) scene_built: bool,

    // --- Showcase 3 ---
    pub(crate) perf_scene: Scene,
    pub(crate) perf_selection: Selection,
    pub(crate) pick_accelerator: Option<PickAccelerator>,
    pub(crate) perf_mesh: Option<MeshId>,
    last_stats: FrameStats,
    pub(crate) perf_total_objects: u32,
    pub(crate) perf_scene_items_cache: Vec<SceneRenderItem>,
    pub(crate) perf_scene_items_version: (u64, u64),
    pub(crate) perf_built: bool,

    // --- Showcase 4 ---
    pub(crate) interact_scene: Scene,
    pub(crate) interact_selection: Selection,
    interact_animator: CameraAnimator,
    interact_gizmo: Gizmo,
    manip: ManipulationController,
    /// Per-node local-transform snapshots for Cancel/ConstraintChanged restore.
    interact_transforms_snapshot: HashMap<NodeId, glam::Mat4>,
    /// True while the left mouse button is held (raw, before egui drag threshold).
    interact_left_held: bool,
    pub(crate) interact_built: bool,
    interact_gizmo_center: Option<glam::Vec3>,
    interact_gizmo_scale: f32,
    last_cursor_viewport: glam::Vec2,

    // --- Showcase 5 ---
    pub(crate) adv_scene: Scene,
    pub(crate) adv_selection: Selection,
    adv_clip_enabled: bool,
    adv_outline_on: bool,
    adv_xray_on: bool,
    pub(crate) adv_built: bool,

    // --- Showcase 6 ---
    pub(crate) pp_scene: Scene,
    pub(crate) pp_built: bool,
    pp_shadow_pcss: bool,
    pp_point_light_on: bool,
    pp_dir_intensity: f32,

    // --- Showcase 7 ---
    pub(crate) nm_scene: Scene,
    pub(crate) nm_built: bool,
    /// (node_id, normal_map_id, ao_map_id) for every mapped object.
    pub(crate) nm_mapped_nodes: Vec<(NodeId, u64, u64)>,
    pub(crate) nm_normal_on: bool,
    pub(crate) nm_ao_on: bool,
    pub(crate) nm_clip_enabled: bool,
    pub(crate) nm_cap_fill: bool,

    // --- Showcase 8 ---
    pub(crate) shd_scene: Scene,
    pub(crate) shd_built: bool,
    pub(crate) shd_cascade_count: u32,
    shd_pcss_on: bool,
    shd_contact_on: bool,

    // --- Showcase 9 ---
    pub(crate) ann_scene: Scene,
    pub(crate) ann_built: bool,
    pub(crate) ann_labels: Vec<AnnotationLabel>,

    // --- Showcase 10 ---
    pub(crate) cam_tools_scene: Scene,
    pub(crate) cam_tools_built: bool,
    cam_animator: CameraAnimator,

    // --- Showcase 11 ---
    pub(crate) lights_scene: Scene,
    pub(crate) lights_sources: Vec<LightSource>,
    pub(crate) lights_built: bool,
    lights_hemi_on: bool,
    lights_hemi_intensity: f32,
    lights_sky_color: [f32; 3],
    lights_ground_color: [f32; 3],

    // --- Showcase 12 ---
    pub(crate) scalar_scene: Scene,
    pub(crate) scalar_built: bool,
    scalar_selection: Selection,
    /// ColormapId selected by the user; resolved at build time via builtin_colormap_id.
    scalar_colormap: viewport_lib::BuiltinColormap,
    scalar_range_auto: bool,
    scalar_range: (f32, f32),
    scalar_nan_on: bool,
    scalar_bar_anchor: viewport_lib::ScalarBarAnchor,
    scalar_bar_orientation: viewport_lib::ScalarBarOrientation,
    /// Stable node IDs for the scalar-field objects (sphere, wave grid, box).
    pub(crate) scalar_node_ids: [NodeId; 3],
    /// Mesh indices for the three scalar-field objects (sphere, wave grid, box).
    pub(crate) scalar_mesh_indices: [usize; 3],
    /// CPU-side triangle positions for picking each scalar-field object.
    pub(crate) scalar_pick_positions: [Vec<[f32; 3]>; 3],
    /// CPU-side triangle indices for picking each scalar-field object.
    pub(crate) scalar_pick_indices: [Vec<u32>; 3],
    /// CPU-side scalar values for each mesh (for range computation).
    pub(crate) scalar_values: [Vec<f32>; 3],
    scalar_active_object: usize,

    // --- Showcase 13 ---
    pub(crate) mv_scene: Scene,
    pub(crate) mv_selection: Selection,
    pub(crate) mv_cameras: [Camera; 4],
    mv_controllers: [OrbitCameraController; 4],
    pub(crate) mv_viewports: Option<[viewport_lib::ViewportId; 4]>,
    pub(crate) mv_built: bool,
    mv_gizmo: Gizmo,
    mv_gizmo_drag_active: bool,
    mv_hovered_quad: usize,
    mv_cursor_local: glam::Vec2,
    pub(crate) mv_gizmo_center: Option<glam::Vec3>,
    mv_gizmo_scales: [f32; 4],
    mv_snap: SnapConfig,
    mv_drag_accum_translation: glam::Vec3,
    mv_drag_accum_rotation: f32,
    mv_drag_last_snapped_translation: glam::Vec3,
    mv_drag_last_snapped_rotation: f32,

    // --- Showcase 14 ---
    pub(crate) iso_scene: Scene,
    pub(crate) iso_mesh_index: usize,
    /// CPU-side positions for IsolineItem re-submission.
    pub(crate) iso_positions: Vec<[f32; 3]>,
    /// CPU-side indices for IsolineItem re-submission.
    pub(crate) iso_indices: Vec<u32>,
    /// CPU-side scalar values for IsolineItem re-submission.
    pub(crate) iso_scalars: Vec<f32>,
    iso_grid_resolution: u32,
    iso_contour_count: usize,
    iso_line_color: [f32; 4],
    iso_line_width: f32,
    iso_show_surface_color: bool,
    iso_depth_bias: f32,
    pub(crate) iso_built: bool,

    // --- Showcase 15 ---
    pub(crate) pc_built: bool,
    pc_sub_mode: PcSubMode,
    pc_point_size: f32,
    pc_colormap: BuiltinColormap,
    pc_scalar_range_manual: bool,
    pc_scalar_range: (f32, f32),
    pc_glyph_type: GlyphType,
    pc_glyph_scale: f32,
    pc_glyph_magnitude_scale: bool,
    /// CPU-side point cloud positions (generated once, submitted every frame).
    pub(crate) pc_cloud_positions: Vec<[f32; 3]>,
    /// CPU-side point cloud scalar values (one per point).
    pub(crate) pc_cloud_scalars: Vec<f32>,
    /// CPU-side vector field base positions.
    pub(crate) pc_field_positions: Vec<[f32; 3]>,
    /// CPU-side vector field direction+magnitude vectors.
    pub(crate) pc_field_vectors: Vec<[f32; 3]>,

    // --- Showcase 16 ---
    pub(crate) stream_built: bool,
    stream_use_tubes: bool,
    stream_tube_radius: f32,
    stream_line_width: f32,
    stream_color_by_speed: bool,
    stream_colormap: BuiltinColormap,
    /// Flat RGBA color used when `stream_color_by_speed` is false, and as tube color.
    stream_flat_color: [f32; 4],
    stream_seed_count: usize,
    stream_step_size: f32,
    /// CPU-side streamline paths (one `Vec<[f32;3]>` per seed).
    pub(crate) stream_paths: Vec<Vec<[f32; 3]>>,
    /// Per-vertex speed values parallel to `stream_paths`.
    pub(crate) stream_scalars: Vec<Vec<f32>>,

    // --- Showcase 17 ---
    pub(crate) vol_built: bool,
    pub(crate) vol_volume_id: Option<VolumeId>,
    pub(crate) vol_iso_mesh_index: Option<usize>,
    /// CPU-side field kept for re-extraction on isovalue change.
    pub(crate) vol_field: VolumeData,
    vol_mode: showcase_17_volume::VolumeMode,
    vol_isovalue: f32,
    vol_color_lut: BuiltinColormap,
    vol_opacity_scale: f32,
    vol_threshold: (f32, f32),
    vol_step_scale: f32,
    vol_shading: bool,
    vol_nan_on: bool,
    vol_iso_material: Material,

    // --- Showcase 18 ---
    pub(crate) clipvol_scene: Scene,
    pub(crate) clipvol_built: bool,
    clipvol_sub_mode: showcase_18_clip_volumes::ClipVolSubMode,
    clipvol_box_center: [f32; 3],
    clipvol_box_half_extents: [f32; 3],
    clipvol_box_yaw: f32,
    clipvol_sphere_center: [f32; 3],
    clipvol_sphere_radius: f32,
    /// The clip object used in interactive-plane sub-mode.
    pub(crate) clipvol_plane: ClipObject,
    clipvol_plane_axis: ClipAxis,
    /// Gizmo state for clip volume manipulation.
    clipvol_gizmo: Gizmo,
    /// Cached gizmo center for hit-testing (updated end of each frame).
    clipvol_gizmo_center: Option<glam::Vec3>,
    /// Gizmo screen-space scale (updated end of each frame).
    clipvol_gizmo_scale: f32,
    /// True while a gizmo drag is in progress (suppresses orbit).
    clipvol_gizmo_drag_active: bool,
    /// Whether to show the clip object overlay (wireframe / fill quad).
    clipvol_show_overlay: bool,

    // --- Showcase 19 ---
    pub(crate) matcap_scene: Scene,
    pub(crate) matcap_built: bool,

    // --- Showcase 20 ---
    pub(crate) face_scene: Scene,
    pub(crate) face_built: bool,

    // --- Showcase 21 ---
    pub(crate) texture_scene: Scene,
    pub(crate) texture_built: bool,
    pub(crate) texture_plane_node: NodeId,

    // --- Showcase 22 ---
    pub(crate) param_vis_scene: Scene,
    pub(crate) param_vis_built: bool,
    /// Node IDs for the 16 param-vis objects (4 modes × 4 mesh types: sphere/cube/torus/plane).
    pub(crate) param_vis_node_ids: [NodeId; 16],
    param_vis_scale: f32,
    param_vis_on: bool,
    // --- Showcase 23 ---
    pub(crate) gp_scene: Scene,
    pub(crate) gp_built: bool,
    gp_mode: showcase_23_ground_plane::GpMode,
    gp_height: f32,
    gp_color: [f32; 4],
    gp_tile_size: f32,
    gp_shadow_color: [f32; 4],
    gp_shadow_opacity: f32,

    // --- Showcase 24 ---
    pub(crate) sa_built: bool,
    pub(crate) sa_scene: Scene,
    sa_clip_on: bool,

    // --- Showcase 25 ---
    pub(crate) sv_built: bool,
    sv_mode: showcase_25_surface_vectors::SvMode,
    sv_scale: f32,
    sv_density: f32,
    sv_glyph_density: f32,
    /// Mesh upload indices: [sphere, torus, plane].
    pub(crate) sv_mesh_index: [usize; 3],
    /// CPU-side positions for each mesh.
    pub(crate) sv_positions: [Vec<[f32; 3]>; 3],
    /// CPU-side normals for each mesh.
    pub(crate) sv_normals: [Vec<[f32; 3]>; 3],
    /// CPU-side explicit tangents (sphere only; others None).
    pub(crate) sv_tangents: [Option<Vec<[f32; 4]>>; 3],
    /// CPU-side indices for each mesh.
    pub(crate) sv_indices: [Vec<u32>; 3],
    /// Per-vertex intrinsic 2D vectors (sphere / vertex mode).
    pub(crate) sv_vertex_vecs: Vec<[f32; 2]>,
    /// Per-face intrinsic 2D vectors (torus / face mode).
    pub(crate) sv_face_vecs: Vec<[f32; 2]>,
    /// Per-directed-edge one-form values (plane / edge mode).
    pub(crate) sv_edge_vals: Vec<f32>,

    // --- Showcase 26 ---
    pub(crate) vm_built: bool,
    vm_mode: showcase_26_volume_mesh::VmMode,
    vm_tet_index: usize,
    vm_hex_index: usize,
    vm_colormap: BuiltinColormap,
    vm_field: showcase_26_volume_mesh::VmField,

    // --- Showcase 27 ---
    pub(crate) aux_built: bool,
    pub(crate) aux_frustums: Vec<viewport_lib::CameraFrustumItem>,
    aux_img_alpha: f32,
    aux_img_scale: f32,

    /// Mesh upload indices for the three face-attribute spheres.
    pub(crate) face_mesh_indices: [usize; 3],
    /// Node IDs for the three face-attribute spheres.
    pub(crate) face_node_ids: [NodeId; 3],
    face_colormap: BuiltinColormap,
    /// Opacity for the FaceColor object (tests OIT path when < 1.0).
    face_opacity: f32,
    /// NodeId for each of the 8 built-in preset spheres (matches BUILTIN_PRESETS order).
    pub(crate) matcap_builtin_node_ids: [NodeId; 8],
    pub(crate) matcap_custom_node: Option<NodeId>,
    pub(crate) matcap_custom_id: Option<MatcapId>,
    /// Base color applied to blendable matcap spheres.
    pub(crate) matcap_blendable_color: [f32; 3],
    /// Hue (0..360) for the custom matcap.
    pub(crate) matcap_custom_hue: f32,
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut cycle_dir = 0_i32;
        let mut toggle_keybinds = false;
        let mut tab_pressed = false;
        ctx.input(|i| {
            for event in &i.events {
                match event {
                    egui::Event::Key {
                        key,
                        pressed,
                        repeat,
                        modifiers,
                        ..
                    } if *pressed && !*repeat => {
                        let use_cycle =
                            (modifiers.ctrl || modifiers.command) && !modifiers.alt;
                        match key {
                            egui::Key::OpenBracket if use_cycle => cycle_dir = -1,
                            egui::Key::CloseBracket if use_cycle => cycle_dir = 1,
                            egui::Key::Tab => tab_pressed = true,
                            _ => {}
                        }
                    }
                    egui::Event::Text(t) if t == "?" => {
                        toggle_keybinds = true;
                    }
                    _ => {}
                }
            }
        });
        // Prevent Tab from cycling egui widget focus.
        // begin_pass already set focus_direction=Next, but widgets haven't been
        // laid out yet, so resetting the direction here stops focus from moving.
        if tab_pressed {
            ctx.memory_mut(|mem| mem.move_focus(egui::FocusDirection::None));
        }
        if toggle_keybinds {
            self.show_keybinds = !self.show_keybinds;
        }
        if cycle_dir != 0 {
            self.cycle_showcase(cycle_dir);
        }
        if tab_pressed {
            self.cycle_selection_tab();
        }

        // ---- Keybinds window ----
        egui::Window::new("Keybinds")
            .open(&mut self.show_keybinds)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                egui::Grid::new("keybinds_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        let binds: &[(&str, &str)] = &[
                            // --- Camera ---
                            ("Left drag", "Orbit"),
                            ("Middle drag", "Orbit"),
                            ("Right drag", "Pan"),
                            ("Middle + Shift drag", "Pan"),
                            ("Scroll", "Zoom"),
                            ("Ctrl + Scroll", "Orbit (two-axis)"),
                            ("Shift + Scroll", "Pan (two-axis)"),
                            // --- Selection ---
                            ("Click", "Select object"),
                            // --- Manipulation (Showcase 4) ---
                            ("G", "Move selected"),
                            ("R", "Rotate selected"),
                            ("S", "Scale selected"),
                            ("X / Y / Z", "Constrain to axis"),
                            ("Shift + X/Y/Z", "Exclude axis"),
                            ("0-9 / .", "Numeric input"),
                            ("Enter / Click", "Confirm"),
                            ("Esc", "Cancel"),
                            // --- App ---
                            ("Ctrl + [ / ]", "Cycle showcase"),
                            ("?", "Toggle this window"),
                        ];
                        for (key, action) in binds {
                            ui.strong(*key);
                            ui.label(*action);
                            ui.end_row();
                        }
                    });
            });

        // Lazy scene builds for the active mode.
        self.ensure_scene_built(frame);

        // ---- Top panel: mode switching ----
        egui::TopBottomPanel::top("mode_panel").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Showcase:");
                for mode in [
                    ShowcaseMode::Basic,
                    ShowcaseMode::SceneGraph,
                    ShowcaseMode::Performance,
                    ShowcaseMode::Interaction,
                    ShowcaseMode::Advanced,
                    ShowcaseMode::PostProcess,
                    ShowcaseMode::NormalMaps,
                    ShowcaseMode::Shadows,
                    ShowcaseMode::Annotation,
                    ShowcaseMode::CameraTools,
                    ShowcaseMode::Lights,
                    ShowcaseMode::ScalarFields,
                    ShowcaseMode::MultiViewport,
                    ShowcaseMode::Isolines,
                    ShowcaseMode::PointClouds,
                    ShowcaseMode::Streamlines,
                    ShowcaseMode::Volume,
                    ShowcaseMode::ClipVolumes,
                    ShowcaseMode::Matcap,
                    ShowcaseMode::FaceAttributes,
                    ShowcaseMode::Textures,
                    ShowcaseMode::ParamVis,
                    ShowcaseMode::GroundPlane,
                    ShowcaseMode::SurfaceAppearance,
                    ShowcaseMode::SurfaceVectors,
                    ShowcaseMode::VolumeMesh,
                    ShowcaseMode::Auxiliary,
                ] {
                    if ui
                        .selectable_label(self.mode == mode, mode.label())
                        .clicked()
                    {
                        self.switch_mode(mode);
                    }
                }
            });
        });

        // ---- Left panel: per-mode controls ----
        egui::SidePanel::left("controls_panel")
            .default_width(220.0)
            .show(ctx, |ui| {
                self.show_controls(ui, frame);
            });

        // ---- Central panel: 3-D viewport ----
        let panel_bg = if self.mode == ShowcaseMode::SceneGraph {
            background_color(self.bg_cycle)
        } else {
            BG_COLOR
        };
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgba_unmultiplied(
                (panel_bg[0] * 255.0) as u8,
                (panel_bg[1] * 255.0) as u8,
                (panel_bg[2] * 255.0) as u8,
                255,
            )))
            .show(ctx, |ui| {
            let available = ui.available_size();
            let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

            // Multi-viewport has its own full update path; bypass all single-viewport logic.
            if self.mode == ShowcaseMode::MultiViewport {
                self.update_multi_viewport(ctx, ui, rect, response, frame);
                return;
            }

            // ----- Camera controller -----
            let vp_hovered = response.hovered();
            self.controller.begin_frame(ViewportContext {
                hovered: vp_hovered,
                focused: vp_hovered,
                viewport_size: [rect.width(), rect.height()],
            });

            // Translate egui events -> ViewportEvents.
            let manip_active_for_text = self.manip.is_active();
            ui.input(|i| {
                let mods = viewport_lib::Modifiers {
                    alt: i.modifiers.alt,
                    shift: i.modifiers.shift,
                    ctrl: i.modifiers.command,
                };
                self.controller
                    .push_event(ViewportEvent::ModifiersChanged(mods));

                if let Some(pos) = i.pointer.interact_pos() {
                    let local = glam::Vec2::new(pos.x - rect.left(), pos.y - rect.top());
                    self.last_cursor_viewport = local;
                    self.controller
                        .push_event(ViewportEvent::PointerMoved { position: local });
                }

                for event in &i.events {
                    match event {
                        egui::Event::Key {
                            key,
                            pressed,
                            repeat,
                            ..
                        } if self.mode == ShowcaseMode::Interaction => {
                            if let Some(kc) = egui_key_to_keycode(*key) {
                                self.controller.push_event(ViewportEvent::Key {
                                    key: kc,
                                    state: if *pressed {
                                        ButtonState::Pressed
                                    } else {
                                        ButtonState::Released
                                    },
                                    repeat: *repeat,
                                });
                            }
                        }

                        egui::Event::Text(text) if manip_active_for_text => {
                            for c in text.chars() {
                                self.controller.push_event(ViewportEvent::Character(c));
                            }
                        }

                        egui::Event::PointerButton {
                            button,
                            pressed,
                            pos,
                            ..
                        } => {
                            let vp_button = match button {
                                egui::PointerButton::Primary => viewport_lib::MouseButton::Left,
                                egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
                                egui::PointerButton::Middle => viewport_lib::MouseButton::Middle,
                                _ => continue,
                            };

                            // Ignore presses that originate outside the viewport.
                            if *pressed && !rect.contains(*pos) {
                                continue;
                            }

                            // Track raw left-button held state for ManipulationContext.
                            if *button == egui::PointerButton::Primary {
                                if *pressed {
                                    self.interact_left_held = true;
                                } else {
                                    self.interact_left_held = false;
                                }
                            }

                            // Clip-vol gizmo : start drag.
                            if self.mode == ShowcaseMode::ClipVolumes
                                && *button == egui::PointerButton::Primary
                                && *pressed
                            {
                                let local =
                                    glam::Vec2::new(pos.x - rect.left(), pos.y - rect.top());
                                if let Some(center) = self.clipvol_gizmo_center {
                                    let w = rect.width();
                                    let h = rect.height();
                                    let vp_inv = self.camera.view_proj_matrix().inverse();
                                    let (ray_origin, ray_dir) =
                                        viewport_lib::picking::screen_to_ray(
                                            local,
                                            glam::Vec2::new(w, h),
                                            vp_inv,
                                        );
                                    let orient = self.clipvol_gizmo_orient();
                                    let hit = self.clipvol_gizmo.hit_test_oriented(
                                        ray_origin,
                                        ray_dir,
                                        center,
                                        self.clipvol_gizmo_scale,
                                        orient,
                                    );
                                    if hit != GizmoAxis::None {
                                        self.clipvol_gizmo.active_axis = hit;
                                        self.clipvol_gizmo_drag_active = true;
                                    }
                                }
                            }

                            // Clip-vol gizmo : end drag.
                            if self.mode == ShowcaseMode::ClipVolumes
                                && *button == egui::PointerButton::Primary
                                && !pressed
                                && self.clipvol_gizmo_drag_active
                            {
                                self.clipvol_gizmo_drag_active = false;
                                self.clipvol_gizmo.active_axis = GizmoAxis::None;
                            }

                            let state = if *pressed {
                                ButtonState::Pressed
                            } else {
                                ButtonState::Released
                            };
                            self.controller.push_event(ViewportEvent::MouseButton {
                                button: vp_button,
                                state,
                            });
                        }

                        egui::Event::MouseWheel { delta, .. } => {
                            let over_vp = i
                                .pointer
                                .hover_pos()
                                .map(|p| rect.contains(p))
                                .unwrap_or(false);
                            if over_vp {
                                self.controller.push_event(ViewportEvent::Wheel {
                                    delta: glam::Vec2::new(delta.x, delta.y),
                                    units: ScrollUnits::Pixels,
                                });
                            }
                        }

                        _ => {}
                    }
                }
            });

            // ----- Clip-vol gizmo drag (Showcase 18) -----
            if self.mode == ShowcaseMode::ClipVolumes
                && self.clipvol_gizmo_drag_active
                && response.dragged()
            {
                let drag_delta = response.drag_delta();
                let dx = drag_delta.x;
                let dy = drag_delta.y;
                if dx.abs() > 0.001 || dy.abs() > 0.001 {
                    self.apply_clipvol_gizmo_drag(dx, dy, rect.width(), rect.height());
                }
            }

            // ----- Advance camera animator (Showcases 4 and 10) -----
            if self.mode == ShowcaseMode::Interaction {
                let dt = ctx.input(|i| i.stable_dt.min(1.0 / 30.0));
                self.interact_animator.update(dt, &mut self.camera);
            }
            if self.mode == ShowcaseMode::CameraTools || self.mode == ShowcaseMode::Auxiliary {
                let dt = ctx.input(|i| i.stable_dt.min(1.0 / 30.0));
                self.cam_animator.update(dt, &mut self.camera);
            }

            // ----- ManipulationController update (Showcase 4 only) -----
            // For Interaction mode, orbit resolution is integrated here so that
            // the same ActionFrame drives both camera and gizmo.
            if self.mode == ShowcaseMode::Interaction {
                if self.interact_built {
                    let w = rect.width();
                    let h = rect.height();
                    let viewport_size = glam::Vec2::new(w, h);
                    let view_proj = self.camera.proj_matrix() * self.camera.view_matrix();

                    // Per-frame gizmo hover when no session is active.
                    if !self.manip.is_active() {
                        if let Some(center) = self.interact_gizmo_center {
                            let ray_origin = self.camera.eye_position();
                            let cursor = self.last_cursor_viewport;
                            let ndc_x = (cursor.x / w.max(1.0)) * 2.0 - 1.0;
                            let ndc_y = 1.0 - (cursor.y / h.max(1.0)) * 2.0;
                            let inv_vp = view_proj.inverse();
                            let far =
                                inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
                            let ray_dir =
                                (far - ray_origin).normalize_or_zero();
                            let orient = gizmo_orientation(
                                &self.interact_gizmo,
                                &self.interact_selection,
                                &self.interact_scene,
                            );
                            self.interact_gizmo.hovered_axis =
                                self.interact_gizmo.hit_test_oriented(
                                    ray_origin,
                                    ray_dir,
                                    center,
                                    self.interact_gizmo_scale,
                                    orient,
                                );
                        } else {
                            self.interact_gizmo.hovered_axis = GizmoAxis::None;
                        }
                    }

                    // Build GizmoInfo.
                    let orient = gizmo_orientation(
                        &self.interact_gizmo,
                        &self.interact_selection,
                        &self.interact_scene,
                    );
                    let gizmo_info = self.interact_gizmo_center.map(|center| GizmoInfo {
                        center,
                        scale: self.interact_gizmo_scale,
                        orientation: orient,
                        mode: self.interact_gizmo.mode,
                    });

                    // Build ManipulationContext.
                    let pointer_delta =
                        ctx.input(|i| glam::Vec2::new(i.pointer.delta().x, i.pointer.delta().y));
                    let manip_ctx = ManipulationContext {
                        camera: self.camera.clone(),
                        viewport_size,
                        cursor_viewport: Some(self.last_cursor_viewport),
                        pointer_delta,
                        selection_center: self.interact_gizmo_center,
                        gizmo: gizmo_info,
                        drag_started: response.drag_started(),
                        dragging: self.interact_left_held,
                        clicked: response.clicked(),
                    };

                    // Orbit: resolve (no camera movement) while manipulation is active.
                    let action_frame = if self.manip.is_active() {
                        self.controller.resolve()
                    } else {
                        self.controller.apply_to_camera(&mut self.camera)
                    };

                    // Tab cycles gizmo mode when no session is active.
                    if !self.manip.is_active()
                        && action_frame.is_active(Action::CycleGizmoMode)
                    {
                        self.interact_gizmo.mode = match self.interact_gizmo.mode {
                            GizmoMode::Translate => GizmoMode::Rotate,
                            GizmoMode::Rotate => GizmoMode::Scale,
                            GizmoMode::Scale => GizmoMode::Translate,
                            _ => GizmoMode::Translate,
                        };
                    }

                    match self.manip.update(&action_frame, manip_ctx) {
                        ManipResult::Update(delta) => {
                            self.apply_interact_delta(delta);
                        }
                        ManipResult::Cancel | ManipResult::ConstraintChanged => {
                            self.restore_interact_snapshots();
                        }
                        ManipResult::Commit => {
                            self.save_interact_snapshots();
                        }
                        ManipResult::None => {
                            if !self.manip.is_active() {
                                // Keep snapshots current so G/R/S always starts clean.
                                self.save_interact_snapshots();
                            }
                        }
                    }

                    // Click-to-select: only when no session is active.
                    if response.clicked() && !self.manip.is_active() {
                        let pick_pos = self.last_cursor_viewport;
                        self.handle_click_select(pick_pos, w, h);
                    }
                } else {
                    self.controller.apply_to_camera(&mut self.camera);
                }
            } else {
                // ----- Apply / resolve orbit controller (non-Interaction modes) -----
                let suppress_orbit =
                    self.mode == ShowcaseMode::ClipVolumes && self.clipvol_gizmo_drag_active;
                if suppress_orbit {
                    self.controller.resolve();
                } else {
                    self.controller.apply_to_camera(&mut self.camera);
                }
            }

            self.camera.set_aspect_ratio(rect.width(), rect.height());

            // ----- Click-to-select (non-Interaction modes) -----
            if response.clicked() && self.mode != ShowcaseMode::Interaction {
                let pick_pos = self.last_cursor_viewport;
                self.handle_click_select(pick_pos, rect.width(), rect.height());
            }

            // ----- Build frame data -----
            let frame_data = self.build_frame_data(rect.width(), rect.height(), frame);

            // ----- Update gizmo_center cache for next frame's hit-testing -----
            if self.mode == ShowcaseMode::Interaction {
                self.interact_gizmo_center =
                    gizmo::gizmo_center_from_selection(&self.interact_selection, |id| {
                        self.interact_scene.node(id).map(|n| {
                            let t = n.world_transform();
                            glam::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z)
                        })
                    });
                if let Some(center) = self.interact_gizmo_center {
                    self.interact_gizmo_scale = compute_gizmo_scale(
                        center,
                        self.camera.eye_position(),
                        self.camera.fov_y,
                        rect.height(),
                    );
                }
            }
            if self.mode == ShowcaseMode::ClipVolumes && self.clipvol_built {
                self.clipvol_gizmo_center = self.clip_gizmo_center();
                if let Some(center) = self.clipvol_gizmo_center {
                    self.clipvol_gizmo_scale = compute_gizmo_scale(
                        center,
                        self.camera.eye_position(),
                        self.camera.fov_y,
                        rect.height(),
                    );
                }
            }

            // ----- Schedule paint callback -----
            // Showcase 24 uses the HDR path so SSAA and post-processing work.
            if self.mode == ShowcaseMode::SurfaceAppearance {
                ui.painter()
                    .add(eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        hdr_viewport_callback::HdrViewportCallback {
                            frame: frame_data,
                            viewport_size: [rect.width() as u32, rect.height() as u32],
                        },
                    ));
            } else {
                ui.painter()
                    .add(eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        viewport_callback::ViewportCallback { frame: frame_data },
                    ));
            }

            // ----- Manipulation mode overlay (Showcase 4) -----
            if self.mode == ShowcaseMode::Interaction {
                if let Some(ms) = self.manip.state() {
                    let kind_label = match ms.kind {
                        viewport_lib::ManipulationKind::Move => "Move",
                        viewport_lib::ManipulationKind::Rotate => "Rotate",
                        viewport_lib::ManipulationKind::Scale => "Scale",
                    };
                    let axis_label = match ms.axis {
                        Some(GizmoAxis::X) => if ms.exclude_axis { " (YZ)" } else { " (X)" },
                        Some(GizmoAxis::Y) => if ms.exclude_axis { " (XZ)" } else { " (Y)" },
                        Some(GizmoAxis::Z) => if ms.exclude_axis { " (XY)" } else { " (Z)" },
                        _ => "",
                    };
                    let text = if let Some(ref numeric) = ms.numeric_display {
                        format!("{kind_label}{axis_label}: {numeric}")
                    } else {
                        format!("{kind_label}{axis_label}")
                    };
                    let font = egui::FontId::proportional(14.0);
                    let galley = ui.painter().layout_no_wrap(
                        text,
                        font,
                        egui::Color32::WHITE,
                    );
                    let pos = egui::pos2(
                        rect.center().x - galley.size().x / 2.0,
                        rect.max.y - 30.0,
                    );
                    let bg = egui::Rect::from_min_size(
                        pos - egui::vec2(6.0, 3.0),
                        galley.size() + egui::vec2(12.0, 6.0),
                    );
                    ui.painter().rect_filled(
                        bg,
                        3.0,
                        egui::Color32::from_black_alpha(180),
                    );
                    ui.painter().galley(pos, galley, egui::Color32::WHITE);
                    ctx.request_repaint();
                }
            }

            // ----- Annotation labels drawn on top of the 3-D viewport -----
            if self.mode == ShowcaseMode::Annotation {
                self.draw_annotation_labels(ui, rect);
            }

            // ----- Scalar bar overlay (Showcase 12) -----
            if self.mode == ShowcaseMode::ScalarFields {
                self.draw_scalar_bar(ui, rect, frame);
            }

            // ----- Cursor feedback -----
            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }

            // ----- Continuous repaint for animated camera -----
            if self.mode == ShowcaseMode::Interaction && self.interact_animator.is_animating() {
                ctx.request_repaint();
            }
            if (self.mode == ShowcaseMode::CameraTools || self.mode == ShowcaseMode::Auxiliary)
                && self.cam_animator.is_animating()
            {
                ctx.request_repaint();
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------

impl App {
    fn cycle_showcase(&mut self, dir: i32) {
        const SHOWCASE_MODES: [ShowcaseMode; 27] = [
            ShowcaseMode::Basic,
            ShowcaseMode::SceneGraph,
            ShowcaseMode::Performance,
            ShowcaseMode::Interaction,
            ShowcaseMode::Advanced,
            ShowcaseMode::PostProcess,
            ShowcaseMode::NormalMaps,
            ShowcaseMode::Shadows,
            ShowcaseMode::Annotation,
            ShowcaseMode::CameraTools,
            ShowcaseMode::Lights,
            ShowcaseMode::ScalarFields,
            ShowcaseMode::MultiViewport,
            ShowcaseMode::Isolines,
            ShowcaseMode::PointClouds,
            ShowcaseMode::Streamlines,
            ShowcaseMode::Volume,
            ShowcaseMode::ClipVolumes,
            ShowcaseMode::Matcap,
            ShowcaseMode::FaceAttributes,
            ShowcaseMode::Textures,
            ShowcaseMode::ParamVis,
            ShowcaseMode::GroundPlane,
            ShowcaseMode::SurfaceAppearance,
            ShowcaseMode::SurfaceVectors,
            ShowcaseMode::VolumeMesh,
            ShowcaseMode::Auxiliary,
        ];

        let Some(current) = SHOWCASE_MODES.iter().position(|&mode| mode == self.mode) else {
            return;
        };
        let len = SHOWCASE_MODES.len() as i32;
        let next = (current as i32 + dir).rem_euclid(len) as usize;
        self.switch_mode(SHOWCASE_MODES[next]);
    }

    fn cycle_selection_tab(&mut self) {
        let (scene, selection) = match self.mode {
            ShowcaseMode::SceneGraph => (&self.scene, &mut self.selection),
            ShowcaseMode::Advanced => (&self.adv_scene, &mut self.adv_selection),
            _ => return,
        };
        let walk = scene.walk_depth_first();
        if !walk.is_empty() {
            let current = selection.primary();
            let next_idx = match current {
                Some(id) => {
                    let pos = walk.iter().position(|(nid, _)| *nid == id);
                    pos.map(|i| (i + 1) % walk.len()).unwrap_or(0)
                }
                None => 0,
            };
            selection.select_one(walk[next_idx].0);
        }
    }

    fn switch_mode(&mut self, mode: ShowcaseMode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        // Camera resets are applied on first build in ensure_scene_built.
        // Switching back to an already-built showcase doesn't reset.
    }

    fn ensure_scene_built(&mut self, frame: &eframe::Frame) {
        let needs = match self.mode {
            ShowcaseMode::SceneGraph => !self.scene_built,
            ShowcaseMode::Performance => !self.perf_built,
            ShowcaseMode::Interaction => !self.interact_built,
            ShowcaseMode::Advanced => !self.adv_built,
            ShowcaseMode::PostProcess => !self.pp_built,
            ShowcaseMode::NormalMaps => !self.nm_built,
            ShowcaseMode::Shadows => !self.shd_built,
            ShowcaseMode::Annotation => !self.ann_built,
            ShowcaseMode::CameraTools => !self.cam_tools_built,
            ShowcaseMode::Lights => !self.lights_built,
            ShowcaseMode::ScalarFields => !self.scalar_built,
            ShowcaseMode::MultiViewport => !self.mv_built || self.mv_viewports.is_none(),
            ShowcaseMode::Isolines => !self.iso_built,
            ShowcaseMode::PointClouds => !self.pc_built,
            ShowcaseMode::Streamlines => !self.stream_built,
            ShowcaseMode::Volume => !self.vol_built,
            ShowcaseMode::ClipVolumes => !self.clipvol_built,
            ShowcaseMode::Matcap => !self.matcap_built,
            ShowcaseMode::FaceAttributes => !self.face_built,
            ShowcaseMode::Textures => !self.texture_built,
            ShowcaseMode::ParamVis => !self.param_vis_built,
            ShowcaseMode::GroundPlane => !self.gp_built,
            ShowcaseMode::SurfaceAppearance => !self.sa_built,
            ShowcaseMode::SurfaceVectors => !self.sv_built,
            ShowcaseMode::VolumeMesh => !self.vm_built,
            ShowcaseMode::Auxiliary => !self.aux_built,
            _ => false,
        };
        if !needs {
            return;
        }
        let rs = frame.wgpu_render_state().expect("wgpu must be enabled");
        let mut guard = rs.renderer.write();
        let renderer = guard
            .callback_resources
            .get_mut::<ViewportRenderer>()
            .expect("ViewportRenderer must be registered");

        match self.mode {
            ShowcaseMode::SceneGraph => self.build_scene_graph(renderer),
            ShowcaseMode::Performance => {
                self.build_perf_scene(renderer);
                self.camera.distance = 80.0;
            }
            ShowcaseMode::Interaction => {
                self.build_interact_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 12.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Advanced => {
                self.build_adv_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 2.0, 0.5),
                    distance: 14.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::PostProcess => {
                self.build_pp_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 0.5),
                    distance: 8.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::NormalMaps => {
                self.build_nm_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 0.8),
                    distance: 10.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Shadows => {
                self.build_shadow_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 1.0),
                    distance: 14.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Annotation => {
                self.build_annotation_scene(renderer);
                self.reset_annotation_camera();
            }
            ShowcaseMode::CameraTools => {
                self.build_camera_tools_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 12.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Lights => {
                self.build_lights_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 14.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::ScalarFields => {
                self.build_scalar_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 16.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::MultiViewport => {
                if self.mv_viewports.is_none() {
                    let vp0 = renderer.create_viewport(&rs.device);
                    let vp1 = renderer.create_viewport(&rs.device);
                    let vp2 = renderer.create_viewport(&rs.device);
                    let vp3 = renderer.create_viewport(&rs.device);
                    self.mv_viewports = Some([vp0, vp1, vp2, vp3]);
                }
                if !self.mv_built {
                    self.build_mv_scene(renderer);
                }
            }
            ShowcaseMode::Isolines => {
                self.build_iso_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 14.0,
                    orientation: glam::Quat::from_rotation_z(0.3)
                        * glam::Quat::from_rotation_x(0.8),
                    ..Camera::default()
                };
            }
            ShowcaseMode::PointClouds => {
                self.build_pc_scene();
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 12.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Streamlines => {
                self.build_stream_scene();
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 10.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Volume => {
                self.build_volume_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 12.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::ClipVolumes => {
                self.build_clipvol_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 14.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Textures => {
                self.build_texture_scene(renderer);
            }
            ShowcaseMode::ParamVis => {
                self.build_param_vis_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 22.0,
                    orientation: glam::Quat::from_rotation_z(0.3)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Matcap => {
                self.build_matcap_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, -0.5),
                    distance: 14.0,
                    orientation: glam::Quat::from_rotation_z(0.3)
                        * glam::Quat::from_rotation_x(0.9),
                    ..Camera::default()
                };
            }
            ShowcaseMode::FaceAttributes => {
                self.build_face_attr_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 16.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::GroundPlane => {
                self.build_ground_plane_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 1.0),
                    distance: 14.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::SurfaceAppearance => {
                self.build_sa_scene(renderer);
                self.camera = Camera {
                    // Pull back so both the front spheres and the background
                    // SSAA stress-grid are fully visible.
                    center: glam::Vec3::new(0.0, 0.0, -1.5),
                    distance: 16.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::SurfaceVectors => {
                self.build_sv_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 6.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::VolumeMesh => {
                self.build_vm_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 9.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Auxiliary => {
                self.build_aux_scene();
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 12.0,
                    orientation: glam::Quat::from_rotation_z(0.3)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

impl App {
    fn show_controls(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.horizontal(|ui| {
            ui.heading(self.mode.label());
            if ui.small_button("(?)").on_hover_text("Keybinds").clicked() {
                self.show_keybinds = !self.show_keybinds;
            }
        });
        ui.separator();

        match self.mode {
            ShowcaseMode::Basic => self.controls_basic(ui),
            ShowcaseMode::SceneGraph => self.controls_scene_graph(ui, frame),
            ShowcaseMode::Performance => self.controls_performance(ui),
            ShowcaseMode::Interaction => self.controls_interaction(ui),
            ShowcaseMode::Advanced => self.controls_advanced(ui),
            ShowcaseMode::PostProcess => self.controls_post_process(ui),
            ShowcaseMode::NormalMaps => self.controls_normal_maps(ui, frame),
            ShowcaseMode::Shadows => self.controls_shadows(ui),
            ShowcaseMode::Annotation => self.controls_annotation(ui),
            ShowcaseMode::CameraTools => self.controls_camera_tools(ui),
            ShowcaseMode::Lights => self.controls_lights(ui),
            ShowcaseMode::ScalarFields => self.controls_scalar_fields(ui),
            ShowcaseMode::MultiViewport => self.controls_mv(ui),
            ShowcaseMode::Isolines => self.controls_isolines(ui),
            ShowcaseMode::PointClouds => self.controls_point_clouds(ui),
            ShowcaseMode::Streamlines => self.controls_streamlines(ui),
            ShowcaseMode::Volume => self.controls_volume(ui, frame),
            ShowcaseMode::ClipVolumes => self.controls_clipvol(ui),
            ShowcaseMode::Matcap => self.controls_matcap(ui, frame),
            ShowcaseMode::FaceAttributes => self.controls_face_attr(ui),
            ShowcaseMode::Textures => self.controls_textures(ui),
            ShowcaseMode::ParamVis => self.controls_param_vis(ui),
            ShowcaseMode::GroundPlane => self.controls_ground_plane(ui),
            ShowcaseMode::SurfaceAppearance => self.controls_surface_appearance(ui),
            ShowcaseMode::SurfaceVectors => self.controls_surface_vectors(ui),
            ShowcaseMode::VolumeMesh => self.controls_volume_mesh(ui),
            ShowcaseMode::Auxiliary => self.controls_aux(ui),
        }
    }

    fn controls_basic(&mut self, ui: &mut egui::Ui) {
        ui.label("Projection:");
        ui.horizontal(|ui| {
            if ui
                .radio(
                    self.camera.projection == Projection::Perspective,
                    "Perspective",
                )
                .clicked()
            {
                self.camera.projection = Projection::Perspective;
            }
            if ui
                .radio(
                    self.camera.projection == Projection::Orthographic,
                    "Orthographic",
                )
                .clicked()
            {
                self.camera.projection = Projection::Orthographic;
            }
        });
        ui.separator();
        ui.label("Light:");
        ui.horizontal(|ui| {
            if ui.radio(!self.use_point_light, "Directional").clicked() {
                self.use_point_light = false;
            }
            if ui.radio(self.use_point_light, "Point").clicked() {
                self.use_point_light = true;
            }
        });
    }

    fn controls_scene_graph(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let sel = self.selection.len();
        let nodes = self.scene.node_count();
        ui.label(format!("Nodes: {nodes}  Selected: {sel}"));
        ui.separator();

        if ui.button("Cycle Material").clicked() {
            self.material_cycle += 1;
            let mat = material_preset(self.material_cycle);
            for &id in self.selection.iter() {
                self.scene.set_material(id, mat);
            }
        }

        if ui.button("Toggle Transparency").clicked() {
            for &id in self.selection.iter() {
                if let Some(node) = self.scene.node(id) {
                    let mut mat = *node.material();
                    mat.opacity = if mat.opacity < 1.0 { 1.0 } else { 0.4 };
                    self.scene.set_material(id, mat);
                }
            }
        }

        if ui.button("Toggle Normal Vis").clicked() {
            for &id in self.selection.iter() {
                if let Some(node) = self.scene.node(id) {
                    let show = !node.show_normals();
                    self.scene.set_show_normals(id, show);
                }
            }
        }

        ui.separator();

        ui.label("Outline width (px):");
        ui.add(egui::Slider::new(&mut self.sg_outline_width, 1.0..=8.0).step_by(0.5));

        ui.separator();

        if ui.button("Cycle Background").clicked() {
            self.bg_cycle += 1;
        }

        ui.separator();

        if ui.button("Add Child to Selected").clicked() {
            if let Some(parent_id) = self.selection.primary() {
                let rs = frame.wgpu_render_state().unwrap();
                let mut guard = rs.renderer.write();
                let renderer = guard
                    .callback_resources
                    .get_mut::<ViewportRenderer>()
                    .unwrap();
                let mesh = self.upload_box(renderer);
                let local = glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::splat(0.5),
                    glam::Quat::IDENTITY,
                    glam::Vec3::new(1.5, 0.0, 1.5),
                );
                let child_id = self.scene.add_named(
                    "Child",
                    Some(mesh),
                    local,
                    Material::from_color([1.0, 0.6, 0.2]),
                );
                self.scene.set_parent(child_id, Some(parent_id));
                self.selection.select_one(child_id);
            }
        }

        if ui.button("Remove Selected").clicked() {
            if let Some(id) = self.selection.primary() {
                let removed = self.scene.remove(id);
                for rid in &removed {
                    self.selection.remove(*rid);
                }
            }
        }

        ui.separator();

        if ui
            .checkbox(&mut self.layer_b_visible, "Layer B Visible")
            .changed()
        {
            self.scene
                .set_layer_visible(self.layer_b.unwrap(), self.layer_b_visible);
        }

        ui.separator();

        if ui.button("Cycle Selection (Tab)").clicked() {
            let walk = self.scene.walk_depth_first();
            if !walk.is_empty() {
                let current = self.selection.primary();
                let next_idx = match current {
                    Some(id) => {
                        let pos = walk.iter().position(|(nid, _)| *nid == id);
                        pos.map(|i| (i + 1) % walk.len()).unwrap_or(0)
                    }
                    None => 0,
                };
                self.selection.select_one(walk[next_idx].0);
            }
        }

        if ui.button("Clear Selection").clicked() {
            self.selection.clear();
        }
    }

    fn controls_performance(&mut self, ui: &mut egui::Ui) {
        let s = &self.last_stats;
        let total = self.perf_total_objects;
        let visible = s.total_objects;
        let culled = total.saturating_sub(visible);
        ui.label(format!("Total objects: {total}"));
        ui.label(format!("Visible: {visible}"));
        ui.label(format!("Culled: {culled}"));
        ui.label(format!("Draw calls: {}", s.draw_calls));
        ui.label(format!("Batches: {}", s.instanced_batches));
        ui.label(format!("Triangles: {}", s.triangles_submitted));
        ui.label(format!("Shadow draws: {}", s.shadow_draw_calls));
        ui.separator();
        ui.label("Click objects to select them.");
        if ui.button("Clear Selection").clicked() {
            self.perf_selection.clear();
        }
    }

    fn controls_interaction(&mut self, ui: &mut egui::Ui) {
        ui.label("Gizmo Mode:");
        ui.horizontal(|ui| {
            if ui
                .radio(
                    self.interact_gizmo.mode == GizmoMode::Translate,
                    "Translate",
                )
                .clicked()
            {
                self.interact_gizmo.mode = GizmoMode::Translate;
            }
            if ui
                .radio(self.interact_gizmo.mode == GizmoMode::Rotate, "Rotate")
                .clicked()
            {
                self.interact_gizmo.mode = GizmoMode::Rotate;
            }
            if ui
                .radio(self.interact_gizmo.mode == GizmoMode::Scale, "Scale")
                .clicked()
            {
                self.interact_gizmo.mode = GizmoMode::Scale;
            }
        });

        ui.separator();

        ui.label("Gizmo Space:");
        ui.horizontal(|ui| {
            if ui
                .radio(self.interact_gizmo.space == GizmoSpace::World, "World")
                .clicked()
            {
                self.interact_gizmo.space = GizmoSpace::World;
            }
            if ui
                .radio(self.interact_gizmo.space == GizmoSpace::Local, "Local")
                .clicked()
            {
                self.interact_gizmo.space = GizmoSpace::Local;
            }
        });

        ui.separator();
        ui.label("Shortcuts: G move · R rotate · S scale");
        ui.label("X / Y / Z : constrain axis  ·  Enter / click : confirm  ·  Esc : cancel");
        ui.separator();
        ui.label("View presets:");
        egui::Grid::new("view_presets_grid")
            .num_columns(4)
            .show(ui, |ui| {
                for (label, preset) in [
                    ("Front", ViewPreset::Front),
                    ("Back", ViewPreset::Back),
                    ("Left", ViewPreset::Left),
                    ("Right", ViewPreset::Right),
                    ("Top", ViewPreset::Top),
                    ("Bottom", ViewPreset::Bottom),
                    ("Iso", ViewPreset::Isometric),
                ] {
                    if ui.button(label).clicked() {
                        self.interact_animator.fly_to_full(
                            &self.camera,
                            self.camera.center,
                            self.camera.distance,
                            preset.orientation(),
                            preset.preferred_projection(),
                            0.6,
                            Easing::EaseInOutCubic,
                        );
                    }
                }
            });

        ui.separator();

        if ui.button("Zoom to Fit").clicked() {
            self.zoom_to_fit_interact();
        }

        ui.separator();

        if ui.button("Clear Selection").clicked() {
            self.interact_selection.clear();
        }
    }

    fn controls_advanced(&mut self, ui: &mut egui::Ui) {
        let sel = self.adv_selection.len();
        ui.label(format!("Selected: {sel}"));
        ui.separator();

        if ui
            .checkbox(&mut self.adv_clip_enabled, "Clip plane (x < 0)")
            .changed()
        {}
        if ui
            .checkbox(&mut self.adv_outline_on, "Selection outline")
            .changed()
        {}
        if ui
            .checkbox(&mut self.adv_xray_on, "X-ray selected")
            .changed()
        {}

        ui.separator();

        if ui.button("Cycle Selection (Tab)").clicked() {
            let walk = self.adv_scene.walk_depth_first();
            if !walk.is_empty() {
                let current = self.adv_selection.primary();
                let next_idx = match current {
                    Some(id) => {
                        let pos = walk.iter().position(|(nid, _)| *nid == id);
                        pos.map(|i| (i + 1) % walk.len()).unwrap_or(0)
                    }
                    None => 0,
                };
                self.adv_selection.select_one(walk[next_idx].0);
            }
        }

        if ui.button("Clear Selection").clicked() {
            self.adv_selection.clear();
        }
    }

    fn controls_post_process(&mut self, ui: &mut egui::Ui) {
        ui.label("Lighting:");
        ui.add(egui::Slider::new(&mut self.pp_dir_intensity, 0.0..=5.0).text("Dir. intensity"));
        ui.checkbox(&mut self.pp_point_light_on, "Point light");

        ui.separator();
        ui.label("Shadows:");
        ui.checkbox(&mut self.pp_shadow_pcss, "PCSS (soft shadows)");

        ui.separator();
        ui.weak("Bloom / SSAO / FXAA / tone-mapping require the HDR\npipeline (renderer.render()), not available in the\neframe paint-callback path.");
    }

    fn controls_normal_maps(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        if ui.checkbox(&mut self.nm_normal_on, "Normal map").changed() {
            let on = self.nm_normal_on;
            for &(node_id, nm_id, _) in &self.nm_mapped_nodes.clone() {
                if let Some(node) = self.nm_scene.node(node_id) {
                    let mut mat = *node.material();
                    mat.normal_map_id = if on { Some(nm_id) } else { None };
                    self.nm_scene.set_material(node_id, mat);
                }
            }
        }

        if ui.checkbox(&mut self.nm_ao_on, "AO map").changed() {
            let on = self.nm_ao_on;
            for &(node_id, _, ao_id) in &self.nm_mapped_nodes.clone() {
                if let Some(node) = self.nm_scene.node(node_id) {
                    let mut mat = *node.material();
                    mat.ao_map_id = if on { Some(ao_id) } else { None };
                    self.nm_scene.set_material(node_id, mat);
                }
            }
        }

        ui.checkbox(&mut self.nm_clip_enabled, "Clip plane");
        if self.nm_clip_enabled {
            ui.checkbox(&mut self.nm_cap_fill, "Cap fill");
        }

        ui.separator();
        let _ = frame;
    }

    fn controls_shadows(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("Cascades: {}", self.shd_cascade_count));
        ui.horizontal(|ui| {
            if ui
                .button("-")
                .on_hover_text("Decrease cascade count")
                .clicked()
            {
                self.shd_cascade_count = (self.shd_cascade_count - 1).max(1);
            }
            if ui
                .button("+")
                .on_hover_text("Increase cascade count")
                .clicked()
            {
                self.shd_cascade_count = (self.shd_cascade_count + 1).min(4);
            }
        });

        ui.separator();
        ui.label("Filter:");
        ui.horizontal(|ui| {
            if ui.radio(!self.shd_pcss_on, "PCF").clicked() {
                self.shd_pcss_on = false;
            }
            if ui.radio(self.shd_pcss_on, "PCSS").clicked() {
                self.shd_pcss_on = true;
            }
        });

        ui.separator();
        ui.checkbox(&mut self.shd_contact_on, "Contact Shadows");
    }

    fn controls_annotation(&mut self, ui: &mut egui::Ui) {
        ui.label("Labels are drawn directly on the viewport.");
        ui.separator();
        for (i, label) in self.ann_labels.iter().enumerate() {
            let view = self.camera.view_matrix();
            let proj = self.camera.proj_matrix();
            let screen = viewport_lib::world_to_screen(
                label.world_pos,
                &view,
                &proj,
                [800.0, 600.0], // approximate; just for display
            );
            let status = if screen.is_some() {
                "visible"
            } else {
                "clipped"
            };
            ui.label(format!("L{i}: \"{}\" : {status}", label.text));
        }
    }

    fn controls_camera_tools(&mut self, ui: &mut egui::Ui) {
        ui.label("Named Views:");
        egui::Grid::new("cam_view_presets")
            .num_columns(4)
            .show(ui, |ui| {
                for (label, preset) in [
                    ("Front", ViewPreset::Front),
                    ("Back", ViewPreset::Back),
                    ("Left", ViewPreset::Left),
                    ("Right", ViewPreset::Right),
                    ("Top", ViewPreset::Top),
                    ("Bottom", ViewPreset::Bottom),
                    ("Iso", ViewPreset::Isometric),
                ] {
                    if ui.button(label).clicked() {
                        self.cam_animator.fly_to_full(
                            &self.camera,
                            self.camera.center,
                            self.camera.distance,
                            preset.orientation(),
                            preset.preferred_projection(),
                            0.6,
                            Easing::EaseInOutCubic,
                        );
                    }
                }
            });
        ui.separator();
        ui.label("Projection:");
        ui.horizontal(|ui| {
            if ui
                .radio(
                    self.camera.projection == Projection::Perspective,
                    "Perspective",
                )
                .clicked()
            {
                self.camera.projection = Projection::Perspective;
            }
            if ui
                .radio(
                    self.camera.projection == Projection::Orthographic,
                    "Orthographic",
                )
                .clicked()
            {
                self.camera.projection = Projection::Orthographic;
            }
        });
        if self.camera.projection == Projection::Perspective {
            ui.separator();
            let mut fov_deg = self.camera.fov_y.to_degrees();
            ui.label(format!("FOV: {fov_deg:.0}°"));
            if ui
                .add(egui::Slider::new(&mut fov_deg, 20.0_f32..=120.0_f32).suffix("°"))
                .changed()
            {
                self.camera.fov_y = fov_deg.to_radians();
            }
        }
        ui.separator();
        ui.label("The colored boxes identify each axis:");
        ui.label("Red = +X,  Green = +Y,  Blue = +Z");
    }

    fn controls_lights(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("Lights: {}", self.lights_sources.len()));
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("+ Directional").clicked() && self.lights_sources.len() < 8 {
                self.lights_sources.push(LightSource {
                    kind: LightKind::Directional {
                        direction: [0.4, 0.3, 1.5],
                    },
                    color: [1.0, 1.0, 1.0],
                    intensity: 1.0,
                });
            }
            if ui.button("+ Point").clicked() && self.lights_sources.len() < 8 {
                self.lights_sources.push(LightSource {
                    kind: LightKind::Point {
                        position: [0.0, 3.0, 3.0],
                        range: 15.0,
                    },
                    color: [1.0, 0.9, 0.7],
                    intensity: 2.0,
                });
            }
            if ui.button("+ Spot").clicked() && self.lights_sources.len() < 8 {
                self.lights_sources.push(LightSource {
                    kind: LightKind::Spot {
                        position: [0.0, 3.0, 6.0],
                        direction: [0.0, 0.0, -1.0],
                        range: 20.0,
                        inner_angle: 0.25,
                        outer_angle: 0.45,
                    },
                    color: [0.8, 0.95, 1.0],
                    intensity: 3.0,
                });
            }
        });

        if ui.button("Reset to Default").clicked() {
            self.lights_sources = vec![LightSource::default()];
        }

        ui.separator();

        egui::ScrollArea::vertical()
            .max_height(300.0)
            .show(ui, |ui| {
                let mut to_remove: Option<usize> = None;
                let count = self.lights_sources.len();
                for i in 0..count {
                    let kind_label = match self.lights_sources[i].kind {
                        LightKind::Directional { .. } => "Directional",
                        LightKind::Point { .. } => "Point",
                        LightKind::Spot { .. } => "Spot",
                        _ => "Unknown",
                    };
                    egui::CollapsingHeader::new(format!("Light {i} ({kind_label})"))
                        .id_salt(i)
                        .show(ui, |ui| {
                            let src = &mut self.lights_sources[i];

                            // Color
                            ui.horizontal(|ui| {
                                ui.label("Color:");
                                let mut c = src.color;
                                if ui.color_edit_button_rgb(&mut c).changed() {
                                    src.color = c;
                                }
                            });

                            // Intensity
                            ui.horizontal(|ui| {
                                ui.label("Intensity:");
                                ui.add(egui::Slider::new(&mut src.intensity, 0.0..=10.0));
                            });

                            // Kind-specific params
                            #[allow(clippy::match_wildcard_for_catch_all)]
                            match &mut src.kind {
                                LightKind::Directional { direction } => {
                                    ui.label("Direction (toward light):");
                                    ui.horizontal(|ui| {
                                        ui.label("X:");
                                        ui.add(egui::DragValue::new(&mut direction[0]).speed(0.02));
                                        ui.label("Y:");
                                        ui.add(egui::DragValue::new(&mut direction[1]).speed(0.02));
                                        ui.label("Z:");
                                        ui.add(egui::DragValue::new(&mut direction[2]).speed(0.02));
                                    });
                                }
                                LightKind::Point { position, range } => {
                                    ui.label("Position:");
                                    ui.horizontal(|ui| {
                                        ui.label("X:");
                                        ui.add(egui::DragValue::new(&mut position[0]).speed(0.1));
                                        ui.label("Y:");
                                        ui.add(egui::DragValue::new(&mut position[1]).speed(0.1));
                                        ui.label("Z:");
                                        ui.add(egui::DragValue::new(&mut position[2]).speed(0.1));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Range:");
                                        ui.add(egui::Slider::new(range, 1.0..=50.0));
                                    });
                                }
                                LightKind::Spot {
                                    position,
                                    direction,
                                    range,
                                    inner_angle,
                                    outer_angle,
                                } => {
                                    ui.label("Position:");
                                    ui.horizontal(|ui| {
                                        ui.label("X:");
                                        ui.add(egui::DragValue::new(&mut position[0]).speed(0.1));
                                        ui.label("Y:");
                                        ui.add(egui::DragValue::new(&mut position[1]).speed(0.1));
                                        ui.label("Z:");
                                        ui.add(egui::DragValue::new(&mut position[2]).speed(0.1));
                                    });
                                    ui.label("Direction:");
                                    ui.horizontal(|ui| {
                                        ui.label("X:");
                                        ui.add(egui::DragValue::new(&mut direction[0]).speed(0.02));
                                        ui.label("Y:");
                                        ui.add(egui::DragValue::new(&mut direction[1]).speed(0.02));
                                        ui.label("Z:");
                                        ui.add(egui::DragValue::new(&mut direction[2]).speed(0.02));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Range:");
                                        ui.add(egui::Slider::new(range, 1.0..=50.0));
                                    });
                                    let mut inner_deg = inner_angle.to_degrees();
                                    let mut outer_deg = outer_angle.to_degrees();
                                    ui.horizontal(|ui| {
                                        ui.label("Inner cone:");
                                        if ui
                                            .add(
                                                egui::Slider::new(&mut inner_deg, 1.0..=45.0)
                                                    .suffix("°"),
                                            )
                                            .changed()
                                        {
                                            *inner_angle = inner_deg.to_radians();
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Outer cone:");
                                        if ui
                                            .add(
                                                egui::Slider::new(&mut outer_deg, 2.0..=89.0)
                                                    .suffix("°"),
                                            )
                                            .changed()
                                        {
                                            *outer_angle = outer_deg.to_radians();
                                        }
                                    });
                                }
                                _ => {}
                            }

                            if ui.button("Remove").clicked() {
                                to_remove = Some(i);
                            }
                        });
                }
                if let Some(idx) = to_remove {
                    self.lights_sources.remove(idx);
                }
            });

        ui.separator();
        ui.checkbox(&mut self.lights_hemi_on, "Hemisphere Ambient");
        if self.lights_hemi_on {
            ui.add(egui::Slider::new(&mut self.lights_hemi_intensity, 0.0..=1.0).text("Intensity"));
            ui.horizontal(|ui| {
                ui.label("Sky:");
                ui.color_edit_button_rgb(&mut self.lights_sky_color);
            });
            ui.horizontal(|ui| {
                ui.label("Ground:");
                ui.color_edit_button_rgb(&mut self.lights_ground_color);
            });
        }
    }

    fn set_scalar_active_object(&mut self, index: usize) {
        self.scalar_active_object = index;
        self.scalar_range_auto = true;

        let node_id = self.scalar_node_ids[index];
        if node_id != 0 {
            self.scalar_selection.select_one(node_id);
        } else {
            self.scalar_selection.clear();
        }

        if !self.scalar_values[index].is_empty() {
            let min = self.scalar_values[index]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let max = self.scalar_values[index]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            self.scalar_range = (min, max);
        }
    }

    fn controls_scalar_fields(&mut self, ui: &mut egui::Ui) {
        ui.label("Object:");
        for (i, label) in ["0: Sphere (height)", "1: Wave Grid", "2: Box (distance)"]
            .iter()
            .enumerate()
        {
            if ui.radio(self.scalar_active_object == i, *label).clicked() {
                self.set_scalar_active_object(i);
            }
        }

        ui.separator();
        ui.label("Colormap:");
        for (preset, label) in [
            (viewport_lib::BuiltinColormap::Viridis, "Viridis"),
            (viewport_lib::BuiltinColormap::Plasma, "Plasma"),
            (viewport_lib::BuiltinColormap::Greyscale, "Greyscale"),
            (viewport_lib::BuiltinColormap::Coolwarm, "Coolwarm"),
            (viewport_lib::BuiltinColormap::Rainbow, "Rainbow"),
        ] {
            if ui.radio(self.scalar_colormap == preset, label).clicked() {
                self.scalar_colormap = preset;
            }
        }

        ui.separator();
        ui.checkbox(&mut self.scalar_range_auto, "Auto Range");
        if !self.scalar_range_auto {
            ui.horizontal(|ui| {
                ui.label("Min:");
                ui.add(egui::DragValue::new(&mut self.scalar_range.0).speed(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("Max:");
                ui.add(egui::DragValue::new(&mut self.scalar_range.1).speed(0.01));
            });
        } else {
            let i = self.scalar_active_object;
            if !self.scalar_values[i].is_empty() {
                let min = self.scalar_values[i]
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min);
                let max = self.scalar_values[i]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                ui.label(format!("Range: [{min:.2}, {max:.2}]"));
            }
        }

        ui.separator();
        ui.checkbox(&mut self.scalar_nan_on, "Show NaN color (purple)");
        ui.label("(box object: values < threshold set to NaN)");

        ui.separator();
        ui.label("Scalar Bar:");
        for (anchor, label) in [
            (viewport_lib::ScalarBarAnchor::TopLeft, "Top-Left"),
            (viewport_lib::ScalarBarAnchor::TopRight, "Top-Right"),
            (viewport_lib::ScalarBarAnchor::BottomLeft, "Bottom-Left"),
            (viewport_lib::ScalarBarAnchor::BottomRight, "Bottom-Right"),
        ] {
            if ui.radio(self.scalar_bar_anchor == anchor, label).clicked() {
                self.scalar_bar_anchor = anchor;
            }
        }
        ui.horizontal(|ui| {
            if ui
                .radio(
                    self.scalar_bar_orientation == viewport_lib::ScalarBarOrientation::Vertical,
                    "Vertical",
                )
                .clicked()
            {
                self.scalar_bar_orientation = viewport_lib::ScalarBarOrientation::Vertical;
            }
            if ui
                .radio(
                    self.scalar_bar_orientation == viewport_lib::ScalarBarOrientation::Horizontal,
                    "Horizontal",
                )
                .clicked()
            {
                self.scalar_bar_orientation = viewport_lib::ScalarBarOrientation::Horizontal;
            }
        });
    }

    fn controls_isolines(&mut self, ui: &mut egui::Ui) {
        ui.label("Mesh resolution:");
        let res_resp = ui.add(
            egui::Slider::new(&mut self.iso_grid_resolution, 16..=256)
                .text("quads/side")
                .logarithmic(true),
        );
        if res_resp.drag_stopped() || res_resp.lost_focus() {
            self.iso_built = false;
        }

        ui.separator();
        ui.label("Contour levels:");
        ui.add(egui::Slider::new(&mut self.iso_contour_count, 2..=20).text("levels"));

        ui.separator();

        ui.label("Line color:");
        let mut color = egui::Color32::from_rgba_unmultiplied(
            (self.iso_line_color[0] * 255.0) as u8,
            (self.iso_line_color[1] * 255.0) as u8,
            (self.iso_line_color[2] * 255.0) as u8,
            (self.iso_line_color[3] * 255.0) as u8,
        );
        if ui.color_edit_button_srgba(&mut color).changed() {
            let [r, g, b, a] = color.to_array();
            self.iso_line_color = [
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ];
        }

        ui.separator();

        ui.label("Line width (px):");
        ui.add(egui::Slider::new(&mut self.iso_line_width, 0.5..=5.0).step_by(0.25));

        ui.separator();

        ui.checkbox(&mut self.iso_show_surface_color, "Surface scalar coloring");
        ui.label("(grey when off, wave-colored when on)");

        ui.separator();

        ui.label("Depth bias:");
        ui.add(
            egui::Slider::new(&mut self.iso_depth_bias, 0.0..=0.05)
                .step_by(0.001)
                .text("(z-fighting offset)"),
        );
    }
}

// ---------------------------------------------------------------------------
// Frame data assembly
// ---------------------------------------------------------------------------

impl App {
    fn build_frame_data(&mut self, w: f32, h: f32, frame: &eframe::Frame) -> FrameData {
        let mut adv_clip_objects: Vec<ClipObject> = vec![];
        let mut adv_outline = false;
        let mut adv_xray = false;
        let mut perf_outline = false;
        let mut interact_outline = false;
        let mut scene_graph_outline = false;
        let mut scene_graph_outline_width = 4.0_f32;

        let (scene_items, bg_color, lighting, scene_gen, sel_gen) = match self.mode {
            ShowcaseMode::Basic => {
                let positions = [
                    [-1.5, -1.5, 0.0f32],
                    [1.5, -1.5, 0.0],
                    [-1.5, 1.5, 0.0],
                    [1.5, 1.5, 0.0],
                ];
                let items: Vec<SceneRenderItem> = self
                    .mesh_indices
                    .iter()
                    .zip(&positions)
                    .map(|(&mesh_index, pos)| {
                        let model = glam::Mat4::from_translation(glam::Vec3::from(*pos));
                        let mut item = SceneRenderItem::default();
                        item.mesh_index = mesh_index;
                        item.model = model.to_cols_array_2d();
                        item
                    })
                    .collect();

                let lights = if self.use_point_light {
                    vec![LightSource {
                        kind: LightKind::Point {
                            position: [5.0, 5.0, 5.0],
                            range: 30.0,
                        },
                        ..LightSource::default()
                    }]
                } else {
                    vec![LightSource::default()]
                };
                let lighting = LightingSettings {
                    lights,
                    hemisphere_intensity: 0.25,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, 0u64, 0u64)
            }

            ShowcaseMode::SceneGraph => {
                let items = self.scene.collect_render_items(&self.selection);
                let bg = background_color(self.bg_cycle);
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                scene_graph_outline = !self.selection.is_empty();
                scene_graph_outline_width = self.sg_outline_width;
                let sg = self.scene.version();
                let ss = self.selection.version();
                (items, Some(bg), lighting, sg, ss)
            }

            ShowcaseMode::Performance => {
                let current_ver = (self.perf_scene.version(), self.perf_selection.version());
                if current_ver != self.perf_scene_items_version {
                    self.perf_scene_items_cache =
                        self.perf_scene.collect_render_items(&self.perf_selection);
                    self.perf_scene_items_version = current_ver;
                }
                let items = self.perf_scene_items_cache.clone();
                let sg = self.perf_scene.version();
                let ss = self.perf_selection.version();
                perf_outline = !self.perf_selection.is_empty();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, ss)
            }

            ShowcaseMode::Interaction => {
                let items = self
                    .interact_scene
                    .collect_render_items(&self.interact_selection);
                interact_outline = !self.interact_selection.is_empty();
                let sg = self.interact_scene.version();
                let ss = self.interact_selection.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, ss)
            }

            ShowcaseMode::Advanced => {
                let items = self.adv_scene.collect_render_items(&self.adv_selection);
                if self.adv_clip_enabled {
                    adv_clip_objects.push(ClipObject::plane([1.0, 0.0, 0.0], 0.0));
                }
                adv_outline = self.adv_outline_on && !self.adv_selection.is_empty();
                adv_xray = self.adv_xray_on && !self.adv_selection.is_empty();
                let sg = self.adv_scene.version();
                let ss = self.adv_selection.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, ss)
            }

            ShowcaseMode::PostProcess => {
                let items = self.pp_scene.collect_render_items(&Selection::new());
                let mut lights = vec![LightSource {
                    kind: LightKind::Directional {
                        direction: [0.6, 0.4, 1.0],
                    },
                    intensity: self.pp_dir_intensity,
                    ..LightSource::default()
                }];
                if self.pp_point_light_on {
                    lights.push(LightSource {
                        kind: LightKind::Point {
                            position: [3.0, 3.0, 3.0],
                            range: 15.0,
                        },
                        color: [1.0, 0.9, 0.7],
                        intensity: 2.0,
                        ..LightSource::default()
                    });
                }
                let lighting = LightingSettings {
                    lights,
                    shadows_enabled: true,
                    shadow_filter: if self.pp_shadow_pcss {
                        ShadowFilter::Pcss
                    } else {
                        ShadowFilter::Pcf
                    },
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                let sg = self.pp_scene.version();
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::NormalMaps => {
                let items = self.nm_scene.collect_render_items(&Selection::new());
                if self.nm_clip_enabled {
                    adv_clip_objects.push(ClipObject::plane([1.0, 0.0, 0.0], 0.0));
                }
                let lighting = LightingSettings {
                    lights: vec![
                        LightSource {
                            kind: LightKind::Directional {
                                direction: [0.5, 0.3, 1.0],
                            },
                            intensity: 2.5,
                            ..LightSource::default()
                        },
                        LightSource {
                            kind: LightKind::Point {
                                position: [3.0, 3.0, 3.0],
                                range: 15.0,
                            },
                            color: [1.0, 0.97, 0.93],
                            intensity: 2.0,
                            ..LightSource::default()
                        },
                    ],
                    shadows_enabled: true,
                    hemisphere_intensity: 0.4,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                let sg = self.nm_scene.version();
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::Shadows => {
                let items = self.shd_scene.collect_render_items(&Selection::new());
                let lighting = LightingSettings {
                    lights: vec![LightSource {
                        kind: LightKind::Directional {
                            direction: [0.5, 0.2, 1.2],
                        },
                        intensity: 2.0,
                        ..LightSource::default()
                    }],
                    shadows_enabled: true,
                    shadow_cascade_count: self.shd_cascade_count,
                    shadow_filter: if self.shd_pcss_on {
                        ShadowFilter::Pcss
                    } else {
                        ShadowFilter::Pcf
                    },
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                let sg = self.shd_scene.version();
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::Annotation => {
                let items = self.ann_scene.collect_render_items(&Selection::new());
                let sg = self.ann_scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::CameraTools => {
                let items = self.cam_tools_scene.collect_render_items(&Selection::new());
                let sg = self.cam_tools_scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::Lights => {
                let items = self.lights_scene.collect_render_items(&Selection::new());
                let lighting = LightingSettings {
                    lights: self.lights_sources.clone(),
                    hemisphere_intensity: if self.lights_hemi_on {
                        self.lights_hemi_intensity
                    } else {
                        0.0
                    },
                    sky_color: self.lights_sky_color,
                    ground_color: self.lights_ground_color,
                    ..LightingSettings::default()
                };
                let sg = self.lights_scene.version();
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::MultiViewport => {
                unreachable!("MultiViewport is handled before build_frame_data")
            }
            ShowcaseMode::Isolines => {
                let mut items = self.iso_scene.collect_render_items(&Selection::new());
                // Apply scalar coloring or flat grey depending on toggle.
                if self.iso_show_surface_color {
                    for item in items.iter_mut() {
                        item.active_attribute = Some(AttributeRef {
                            name: "wave".to_string(),
                            kind: AttributeKind::Vertex,
                        });
                        item.colormap_id = Some(ColormapId(BuiltinColormap::Coolwarm as usize));
                        item.two_sided = true;
                    }
                } else {
                    for item in items.iter_mut() {
                        item.two_sided = true;
                    }
                }
                let sg = self.iso_scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::PointClouds => {
                let lighting = App::pc_lighting();
                (App::pc_surface_items(), Some(BG_COLOR), lighting, 0, 0)
            }

            ShowcaseMode::Streamlines => {
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (vec![], Some(BG_COLOR), lighting, 0, 0)
            }

            ShowcaseMode::Volume => {
                // Isosurface mesh items go into surface_items when visible.
                let surface_items = if self.vol_mode != showcase_17_volume::VolumeMode::VolumeOnly {
                    self.make_iso_surface_item().into_iter().collect()
                } else {
                    vec![]
                };
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.6,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                (surface_items, Some(BG_COLOR), lighting, 0, 0)
            }

            ShowcaseMode::ClipVolumes => {
                let mut items = self.clipvol_scene.collect_render_items(&Selection::new());
                // Two-sided on all sub-modes: makes the sphere and ground plane visible
                // from the inside when clipped, showing the cut cross-section surface.
                for item in items.iter_mut() {
                    item.two_sided = true;
                }
                let sg = self.clipvol_scene.version();
                let lighting = LightingSettings {
                    lights: vec![LightSource {
                        kind: LightKind::Directional {
                            direction: [0.5, 0.3, 1.2],
                        },
                        intensity: 1.8,
                        ..LightSource::default()
                    }],
                    hemisphere_intensity: 0.4,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::ScalarFields => {
                const ATTR_NAMES: [&str; 3] = ["height", "wave", "distance"];
                let mut items = self
                    .scalar_scene
                    .collect_render_items(&self.scalar_selection);
                let colormap_id = viewport_lib::ColormapId(self.scalar_colormap as usize);
                let active_node_id = self.scalar_node_ids[self.scalar_active_object];
                let wave_node_id = self.scalar_node_ids[1];
                if let Some(item) = items.iter_mut().find(|item| item.pick_id == active_node_id) {
                    item.active_attribute = Some(viewport_lib::AttributeRef {
                        name: ATTR_NAMES[self.scalar_active_object].to_string(),
                        kind: viewport_lib::AttributeKind::Vertex,
                    });
                    item.colormap_id = Some(colormap_id);
                    item.scalar_range = if self.scalar_range_auto {
                        None
                    } else {
                        Some(self.scalar_range)
                    };
                    item.nan_color = if self.scalar_nan_on {
                        Some([0.85, 0.1, 0.85, 1.0])
                    } else {
                        None
                    };
                }
                if let Some(item) = items.iter_mut().find(|item| item.pick_id == wave_node_id) {
                    item.two_sided = true;
                }
                let sg = self.scalar_scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (
                    items,
                    Some(BG_COLOR),
                    lighting,
                    sg,
                    self.scalar_selection.version(),
                )
            }

            ShowcaseMode::Matcap => {
                let items = self.matcap_scene.collect_render_items(&Selection::new());
                let sg = self.matcap_scene.version();
                // Lighting is not used by matcap-shaded objects, but we still
                // need minimal settings for the framework.
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::Textures => {
                let mut items = self.texture_scene.collect_render_items(&Selection::new());
                let plane_node = self.texture_plane_node;
                if let Some(item) = items.iter_mut().find(|i| i.pick_id == plane_node) {
                    item.two_sided = true;
                }
                let sg = self.texture_scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 1.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::ParamVis => {
                let items = self.param_vis_scene.collect_render_items(&Selection::new());
                let sg = self.param_vis_scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::GroundPlane => {
                let items = self.gp_scene.collect_render_items(&Selection::new());
                let sg = self.gp_scene.version();
                let lighting = LightingSettings {
                    lights: vec![LightSource {
                        kind: LightKind::Directional {
                            direction: [0.4, 0.6, 1.0],
                        },
                        intensity: 1.5,
                        ..LightSource::default()
                    }],
                    shadows_enabled: true,
                    hemisphere_intensity: 0.3,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [0.3, 0.3, 0.3],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::FaceAttributes => {
                let mut items = self.face_scene.collect_render_items(&Selection::new());
                let colormap_id = ColormapId(self.face_colormap as usize);

                // Node 0: Vertex attribute (interpolated)
                // scalar_range left as None : renderer auto-detects from attribute_ranges.
                if let Some(item) = items
                    .iter_mut()
                    .find(|i| i.pick_id == self.face_node_ids[0])
                {
                    item.active_attribute = Some(AttributeRef {
                        name: "scalar".to_string(),
                        kind: AttributeKind::Vertex,
                    });
                    item.colormap_id = Some(colormap_id);
                }

                // Node 1: Face attribute (flat per-triangle)
                // scalar_range left as None : renderer auto-detects from attribute_ranges.
                if let Some(item) = items
                    .iter_mut()
                    .find(|i| i.pick_id == self.face_node_ids[1])
                {
                    item.active_attribute = Some(AttributeRef {
                        name: "scalar".to_string(),
                        kind: AttributeKind::Face,
                    });
                    item.colormap_id = Some(colormap_id);
                }

                // Node 2: FaceColor attribute (direct RGBA, no colormap)
                if let Some(item) = items
                    .iter_mut()
                    .find(|i| i.pick_id == self.face_node_ids[2])
                {
                    item.active_attribute = Some(AttributeRef {
                        name: "color".to_string(),
                        kind: AttributeKind::FaceColor,
                    });
                    item.material.opacity = self.face_opacity;
                }

                let sg = self.face_scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.4,
                    sky_color: [1.0, 1.0, 1.0],
                    ground_color: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOR), lighting, sg, 0)
            }

            ShowcaseMode::SurfaceAppearance => {
                let items = self.sa_scene_items();
                let sg = self.sa_scene.version();
                (items, Some(BG_COLOR), App::sa_lighting(), sg, 0)
            }

            ShowcaseMode::SurfaceVectors => {
                let surface_item = if self.sv_built {
                    vec![self.sv_surface_item()]
                } else {
                    vec![]
                };
                (surface_item, Some(BG_COLOR), App::sv_lighting(), 0, 0)
            }

            ShowcaseMode::VolumeMesh => {
                let items = self.vm_scene_items();
                (items, Some(BG_COLOR), App::vm_lighting(), 0, 0)
            }

            ShowcaseMode::Auxiliary => (vec![], Some(BG_COLOR), LightingSettings::default(), 0, 0),
        };

        // Gizmo matrices for Interaction and ClipVolumes modes.
        let (gizmo_model, gizmo_mode, gizmo_space_orient, gizmo_hovered) =
            if self.mode == ShowcaseMode::Interaction {
                let center = self.interact_gizmo_center;
                let model = center.map(|c| {
                    glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(self.interact_gizmo_scale),
                        glam::Quat::IDENTITY,
                        c,
                    )
                });
                let orient = gizmo_orientation(
                    &self.interact_gizmo,
                    &self.interact_selection,
                    &self.interact_scene,
                );
                let hovered = if let Some(state) = self.manip.state() {
                    state.axis.unwrap_or(GizmoAxis::None)
                } else {
                    self.interact_gizmo.hovered_axis
                };
                (model, self.interact_gizmo.mode, orient, hovered)
            } else if self.mode == ShowcaseMode::ClipVolumes && self.clipvol_built {
                let center = self.clipvol_gizmo_center;
                let orient = self.clipvol_gizmo_orient();
                let model = center.map(|c| {
                    glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(self.clipvol_gizmo_scale),
                        glam::Quat::IDENTITY,
                        c,
                    )
                });
                let hovered = if self.clipvol_gizmo.active_axis != GizmoAxis::None {
                    self.clipvol_gizmo.active_axis
                } else {
                    self.clipvol_gizmo.hovered_axis
                };
                (model, self.clipvol_gizmo.mode, orient, hovered)
            } else {
                (
                    None,
                    GizmoMode::Translate,
                    glam::Quat::IDENTITY,
                    GizmoAxis::None,
                )
            };

        let mut fd = FrameData::new(
            CameraFrame::from_camera(&self.camera, [w, h]),
            SceneFrame::from_surface_items(scene_items),
        );
        fd.effects.lighting = lighting;
        fd.viewport.show_grid = false;
        fd.viewport.show_axes_indicator = true;
        fd.viewport.background_color = bg_color;

        // Ground plane (Showcase 23).
        if self.mode == ShowcaseMode::GroundPlane {
            use showcase_23_ground_plane::GpMode;
            fd.effects.ground_plane = GroundPlane {
                mode: match self.gp_mode {
                    GpMode::None => GroundPlaneMode::None,
                    GpMode::ShadowOnly => GroundPlaneMode::ShadowOnly,
                    GpMode::Tile => GroundPlaneMode::Tile,
                    GpMode::SolidColor => GroundPlaneMode::SolidColor,
                },
                height: self.gp_height,
                color: self.gp_color,
                tile_size: self.gp_tile_size,
                shadow_color: self.gp_shadow_color,
                shadow_opacity: self.gp_shadow_opacity,
            };
        }
        // Clip objects for Showcase 24 (Surface Appearance).
        if self.mode == ShowcaseMode::SurfaceAppearance {
            adv_clip_objects.extend(self.sa_clip_objects());
        }
        fd.effects.clip_objects = adv_clip_objects;
        if self.mode == ShowcaseMode::NormalMaps {
            fd.effects.cap_fill_enabled = self.nm_cap_fill;
        }
        // Showcase 24 exists to show back face policies : cap fill would hide them.
        if self.mode == ShowcaseMode::SurfaceAppearance {
            fd.effects.cap_fill_enabled = false;
        }
        fd.interaction.gizmo_model = gizmo_model;
        fd.interaction.gizmo_mode = gizmo_mode;
        fd.interaction.gizmo_hovered = gizmo_hovered;
        fd.interaction.gizmo_space_orientation = gizmo_space_orient;
        fd.interaction.outline_selected = adv_outline
            || perf_outline
            || scene_graph_outline
            || interact_outline
            || (self.mode == ShowcaseMode::ScalarFields && !self.scalar_selection.is_empty());
        if scene_graph_outline {
            fd.interaction.outline_width_px = scene_graph_outline_width;
        }
        fd.interaction.xray_selected = adv_xray;
        fd.scene.generation = scene_gen;
        fd.interaction.selection_generation = sel_gen;

        // Volume item (Showcase 17) : submitted every frame when in volume mode.
        if self.mode == ShowcaseMode::Volume
            && self.vol_built
            && self.vol_mode != showcase_17_volume::VolumeMode::IsosurfaceOnly
        {
            if let Some(vol_item) = self.make_volume_item() {
                fd.scene.volumes.push(vol_item);
            }
        }

        // Clip volume (Showcase 18) : set every frame from current state.
        if self.mode == ShowcaseMode::ClipVolumes && self.clipvol_built {
            if let Some(clip_obj) = self.make_clip_object() {
                fd.effects.clip_objects.push(clip_obj);
            }
        }

        // Streamline / tube items (Showcase 16) : submitted every frame.
        if self.mode == ShowcaseMode::Streamlines && self.stream_built {
            if self.stream_use_tubes {
                fd.scene.streamtube_items.push(self.make_stream_tube_item());
            } else {
                fd.scene.polylines.push(self.make_stream_polyline_item());
            }
        }

        // Surface vector glyphs (Showcase 25) : submitted every frame.
        if self.mode == ShowcaseMode::SurfaceVectors && self.sv_built {
            fd.scene.glyphs.push(self.sv_glyph_item());
        }

        // Auxiliary frustums and screen images (Showcase 27) : submitted every frame.
        if self.mode == ShowcaseMode::Auxiliary && self.aux_built {
            fd.scene.camera_frustums = self.aux_frustums.clone();
            self.aux_push_screen_images(&mut fd);
        }

        // Point cloud / glyph items (Showcase 15) : submitted every frame.
        if self.mode == ShowcaseMode::PointClouds && self.pc_built {
            use crate::PcSubMode;
            match self.pc_sub_mode {
                PcSubMode::PointCloud => {
                    fd.scene.point_clouds.push(self.make_pc_point_cloud_item());
                }
                PcSubMode::VectorField => {
                    fd.scene.glyphs.push(self.make_pc_glyph_item());
                }
            }
        }

        // Isoline items (Showcase 14) : submitted every frame with current settings.
        if self.mode == ShowcaseMode::Isolines && self.iso_built {
            let scalar_min = self
                .iso_scalars
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let scalar_max = self
                .iso_scalars
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let range = scalar_max - scalar_min;
            let isovalues: Vec<f32> = (0..self.iso_contour_count)
                .map(|i| {
                    scalar_min + range * (i as f32 + 1.0) / (self.iso_contour_count as f32 + 1.0)
                })
                .collect();
            let mut iso_item = IsolineItem::default();
            iso_item.positions = self.iso_positions.clone();
            iso_item.indices = self.iso_indices.clone();
            iso_item.scalars = self.iso_scalars.clone();
            iso_item.isovalues = isovalues;
            iso_item.color = self.iso_line_color;
            iso_item.line_width = self.iso_line_width;
            iso_item.depth_bias = self.iso_depth_bias;
            fd.scene.isolines.push(iso_item);
        }

        // Post-process settings (Showcases 6–8).
        // Note: the full HDR pipeline uses renderer.render() which requires direct
        // surface access. In the eframe callback model we use prepare()+paint(),
        // so the post-process pass is not applied. Settings are stored for reference.
        match self.mode {
            ShowcaseMode::PostProcess => {
                // Clamp far plane for better cascade distribution.
                let mut rc = RenderCamera::from_camera(&self.camera);
                rc.far = self.camera.zfar.min(60.0);
                rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                fd.camera.render_camera = rc;
            }
            ShowcaseMode::Shadows => {
                fd.effects.post_process = PostProcessSettings {
                    enabled: false,
                    contact_shadows: self.shd_contact_on,
                    contact_shadow_max_distance: 0.18,
                    contact_shadow_steps: 32,
                    contact_shadow_thickness: 0.04,
                    ..PostProcessSettings::default()
                };
                // Clamp far plane for better cascade distribution.
                let mut rc = RenderCamera::from_camera(&self.camera);
                rc.far = self.camera.zfar.min(60.0);
                rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                fd.camera.render_camera = rc;
            }
            ShowcaseMode::NormalMaps => {
                fd.effects.post_process = PostProcessSettings {
                    enabled: false,
                    ..PostProcessSettings::default()
                };
                // Clamp far plane for better cascade distribution.
                let mut rc = RenderCamera::from_camera(&self.camera);
                rc.far = self.camera.zfar.min(60.0);
                rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                fd.camera.render_camera = rc;
            }
            ShowcaseMode::SurfaceAppearance => {}
            _ => {}
        }

        // Update stats from the last rendered frame (Performance mode).
        if self.mode == ShowcaseMode::Performance {
            let rs = frame.wgpu_render_state().unwrap();
            let guard = rs.renderer.read();
            if let Some(renderer) = guard.callback_resources.get::<ViewportRenderer>() {
                self.last_stats = renderer.last_frame_stats();
            }
        }

        fd
    }
}

// ---------------------------------------------------------------------------
// Selection / picking
// ---------------------------------------------------------------------------

impl App {
    fn handle_click_select(&mut self, pos: glam::Vec2, w: f32, h: f32) {
        let vp_inv = self.camera.view_proj_matrix().inverse();
        let (ray_origin, ray_dir) =
            viewport_lib::picking::screen_to_ray(pos, glam::Vec2::new(w, h), vp_inv);

        match self.mode {
            ShowcaseMode::SceneGraph => {
                let mut mesh_lookup = std::collections::HashMap::new();
                for node in self.scene.nodes() {
                    if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                        mesh_lookup.entry(mid).or_insert_with(|| {
                            (
                                self.box_mesh_data.positions.clone(),
                                self.box_mesh_data.indices.clone(),
                            )
                        });
                    }
                }
                let hit = viewport_lib::picking::pick_scene_nodes(
                    ray_origin,
                    ray_dir,
                    &self.scene,
                    &mesh_lookup,
                );
                if let Some(hit) = hit {
                    self.selection.select_one(hit.id);
                } else {
                    self.selection.clear();
                }
            }

            ShowcaseMode::Performance => {
                let mut mesh_lookup = std::collections::HashMap::new();
                if let Some(mesh) = self.perf_mesh {
                    mesh_lookup.insert(
                        mesh.index() as u64,
                        (
                            self.box_mesh_data.positions.clone(),
                            self.box_mesh_data.indices.clone(),
                        ),
                    );
                }
                let hit = if let Some(ref mut accel) = self.pick_accelerator {
                    viewport_lib::bvh::pick_scene_accelerated(
                        ray_origin,
                        ray_dir,
                        accel,
                        &mesh_lookup,
                    )
                } else {
                    None
                };
                if let Some(hit) = hit {
                    self.perf_selection.select_one(hit.id);
                } else {
                    self.perf_selection.clear();
                }
            }

            ShowcaseMode::Interaction => {
                let mut mesh_lookup = std::collections::HashMap::new();
                for node in self.interact_scene.nodes() {
                    if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                        mesh_lookup.entry(mid).or_insert_with(|| {
                            (
                                self.box_mesh_data.positions.clone(),
                                self.box_mesh_data.indices.clone(),
                            )
                        });
                    }
                }
                let hit = viewport_lib::picking::pick_scene_nodes(
                    ray_origin,
                    ray_dir,
                    &self.interact_scene,
                    &mesh_lookup,
                );
                if let Some(hit) = hit {
                    self.interact_selection.select_one(hit.id);
                } else {
                    self.interact_selection.clear();
                }
            }

            ShowcaseMode::Advanced => {
                let mut mesh_lookup = std::collections::HashMap::new();
                for node in self.adv_scene.nodes() {
                    if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                        mesh_lookup.entry(mid).or_insert_with(|| {
                            (
                                self.box_mesh_data.positions.clone(),
                                self.box_mesh_data.indices.clone(),
                            )
                        });
                    }
                }
                let hit = viewport_lib::picking::pick_scene_nodes(
                    ray_origin,
                    ray_dir,
                    &self.adv_scene,
                    &mesh_lookup,
                );
                if let Some(hit) = hit {
                    self.adv_selection.select_one(hit.id);
                } else {
                    self.adv_selection.clear();
                }
            }

            ShowcaseMode::ScalarFields => {
                let mut mesh_lookup = std::collections::HashMap::new();
                for i in 0..self.scalar_mesh_indices.len() {
                    mesh_lookup.insert(
                        self.scalar_mesh_indices[i] as u64,
                        (
                            self.scalar_pick_positions[i].clone(),
                            self.scalar_pick_indices[i].clone(),
                        ),
                    );
                }
                let hit = viewport_lib::picking::pick_scene_nodes(
                    ray_origin,
                    ray_dir,
                    &self.scalar_scene,
                    &mesh_lookup,
                );
                if let Some(hit) = hit {
                    if let Some(index) = self
                        .scalar_node_ids
                        .iter()
                        .position(|&node_id| node_id == hit.id)
                    {
                        self.set_scalar_active_object(index);
                    } else {
                        self.scalar_selection.select_one(hit.id);
                    }
                } else {
                    self.scalar_selection.clear();
                }
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Manipulation helpers (Showcase 4)
// ---------------------------------------------------------------------------

impl App {
    /// Apply a [`viewport_lib::TransformDelta`] to all selected scene nodes.
    ///
    /// Rotation and scale pivot around `interact_gizmo_center`.
    /// When `position_override` or `scale_override` is set (numeric input), the
    /// snapshot is restored first and the override applied as an absolute value.
    fn apply_interact_delta(&mut self, delta: viewport_lib::TransformDelta) {
        let Some(center) = self.interact_gizmo_center else {
            return;
        };

        let has_pos_override = delta.position_override.iter().any(|v| v.is_some());
        let has_scale_override = delta.scale_override.iter().any(|v| v.is_some());

        if has_pos_override || has_scale_override {
            // Numeric input: restore snapshot so the override is applied from
            // the pre-session position, not accumulated on top of previous deltas.
            self.restore_interact_snapshots();
        }

        let translation = if has_pos_override {
            glam::Vec3::new(
                delta.position_override[0].unwrap_or(0.0),
                delta.position_override[1].unwrap_or(0.0),
                delta.position_override[2].unwrap_or(0.0),
            )
        } else {
            delta.translation
        };

        let scale = if has_scale_override {
            glam::Vec3::new(
                delta.scale_override[0].unwrap_or(1.0),
                delta.scale_override[1].unwrap_or(1.0),
                delta.scale_override[2].unwrap_or(1.0),
            )
        } else {
            delta.scale
        };

        let rot_mat = glam::Mat4::from_quat(delta.rotation);
        let scale_mat = glam::Mat4::from_scale(scale);
        let translate_mat = glam::Mat4::from_translation(translation);
        let to_pivot = glam::Mat4::from_translation(-center);
        let from_pivot = glam::Mat4::from_translation(center);

        for id in self.interact_selection.iter().copied().collect::<Vec<_>>() {
            if let Some(node) = self.interact_scene.node(id) {
                let cur = node.local_transform();
                let new_t = translate_mat * from_pivot * rot_mat * scale_mat * to_pivot * cur;
                self.interact_scene.set_local_transform(id, new_t);
            }
        }
        self.interact_scene.update_transforms();
    }

    /// Snapshot the current local transforms of all selected nodes.
    fn save_interact_snapshots(&mut self) {
        self.interact_transforms_snapshot.clear();
        for id in self.interact_selection.iter().copied().collect::<Vec<_>>() {
            if let Some(node) = self.interact_scene.node(id) {
                self.interact_transforms_snapshot.insert(id, node.local_transform());
            }
        }
    }

    /// Restore local transforms from the last snapshot (used by Cancel / ConstraintChanged).
    fn restore_interact_snapshots(&mut self) {
        let ids: Vec<_> = self.interact_transforms_snapshot.keys().copied().collect();
        for id in ids {
            if let Some(&t) = self.interact_transforms_snapshot.get(&id) {
                self.interact_scene.set_local_transform(id, t);
            }
        }
        self.interact_scene.update_transforms();
    }

    fn zoom_to_fit_interact(&mut self) {
        // Needs renderer resources for mesh AABBs : but we don't have frame here.
        // Fall back to fitting all node world positions.
        let mut min = glam::Vec3::splat(f32::INFINITY);
        let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;

        let iter_nodes: Vec<_> = if !self.interact_selection.is_empty() {
            self.interact_selection.iter().copied().collect()
        } else {
            self.interact_scene
                .walk_depth_first()
                .iter()
                .map(|(id, _)| *id)
                .collect()
        };

        for nid in iter_nodes {
            if let Some(node) = self.interact_scene.node(nid) {
                let t = node.world_transform();
                let pos = glam::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z);
                min = min.min(pos - glam::Vec3::splat(0.6));
                max = max.max(pos + glam::Vec3::splat(0.6));
                any = true;
            }
        }

        if any {
            let aabb = viewport_lib::Aabb { min, max };
            let target = self.camera.fit_aabb_target(&aabb);
            self.interact_animator.fly_to(
                &self.camera,
                target.center,
                target.distance,
                target.orientation,
                0.6,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Clip-vol gizmo helpers (Showcase 18)
// ---------------------------------------------------------------------------

impl App {
    /// Compute the world-space gizmo center for the active clip sub-mode.
    pub(crate) fn clip_gizmo_center(&self) -> Option<glam::Vec3> {
        use showcase_18_clip_volumes::ClipVolSubMode;
        use viewport_lib::ClipShape;
        match self.clipvol_sub_mode {
            ClipVolSubMode::BoxClip => Some(glam::Vec3::from(self.clipvol_box_center)),
            ClipVolSubMode::SphereClip => Some(glam::Vec3::from(self.clipvol_sphere_center)),
            ClipVolSubMode::InteractivePlane => {
                if let ClipShape::Plane {
                    normal, distance, ..
                } = self.clipvol_plane.shape
                {
                    // Shader clips where dot(p, n) + distance < 0, so the plane
                    // sits at -normal * distance from the origin.
                    Some(glam::Vec3::from(normal) * (-distance))
                } else {
                    None
                }
            }
        }
    }

    /// Compute the gizmo orientation quaternion for the active clip sub-mode.
    pub(crate) fn clipvol_gizmo_orient(&self) -> glam::Quat {
        use showcase_18_clip_volumes::ClipVolSubMode;
        use viewport_lib::ClipShape;
        match self.clipvol_sub_mode {
            ClipVolSubMode::BoxClip => {
                glam::Quat::from_rotation_z(self.clipvol_box_yaw.to_radians())
            }
            ClipVolSubMode::SphereClip => glam::Quat::IDENTITY,
            ClipVolSubMode::InteractivePlane => {
                if let ClipShape::Plane { normal, .. } = self.clipvol_plane.shape {
                    let n = glam::Vec3::from(normal).normalize_or_zero();
                    if n.length_squared() > 0.001 {
                        glam::Quat::from_rotation_arc(glam::Vec3::Z, n)
                    } else {
                        glam::Quat::IDENTITY
                    }
                } else {
                    glam::Quat::IDENTITY
                }
            }
        }
    }

    fn apply_clipvol_gizmo_drag(&mut self, dx: f32, dy: f32, w: f32, h: f32) {
        use showcase_18_clip_volumes::ClipVolSubMode;
        use viewport_lib::ClipShape;

        let Some(center) = self.clipvol_gizmo_center else {
            return;
        };
        let drag_delta = glam::Vec2::new(dx, dy);
        let viewport_size = glam::Vec2::new(w, h);
        let vp = self.camera.view_proj_matrix();
        let view = self.camera.view_matrix();
        let axis = self.clipvol_gizmo.active_axis;
        let orient = self.clipvol_gizmo_orient();

        let axis_dir = |a: GizmoAxis| -> glam::Vec3 {
            let base = match a {
                GizmoAxis::X => glam::Vec3::X,
                GizmoAxis::Y => glam::Vec3::Y,
                GizmoAxis::Z => glam::Vec3::Z,
                _ => glam::Vec3::X,
            };
            orient * base
        };

        match self.clipvol_gizmo.mode {
            GizmoMode::Translate => {
                let delta = match axis {
                    GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                        let dir = axis_dir(axis);
                        let amount = gizmo::project_drag_onto_axis(
                            drag_delta,
                            dir,
                            vp,
                            center,
                            viewport_size,
                        );
                        dir * amount
                    }
                    GizmoAxis::XY => gizmo::project_drag_onto_plane(
                        drag_delta,
                        orient * glam::Vec3::X,
                        orient * glam::Vec3::Y,
                        vp,
                        center,
                        viewport_size,
                    ),
                    GizmoAxis::XZ => gizmo::project_drag_onto_plane(
                        drag_delta,
                        orient * glam::Vec3::X,
                        orient * glam::Vec3::Z,
                        vp,
                        center,
                        viewport_size,
                    ),
                    GizmoAxis::YZ => gizmo::project_drag_onto_plane(
                        drag_delta,
                        orient * glam::Vec3::Y,
                        orient * glam::Vec3::Z,
                        vp,
                        center,
                        viewport_size,
                    ),
                    GizmoAxis::Screen => gizmo::project_drag_onto_screen_plane(
                        drag_delta,
                        self.camera.right(),
                        self.camera.up(),
                        vp,
                        center,
                        viewport_size,
                    ),
                    _ => glam::Vec3::ZERO,
                };
                match self.clipvol_sub_mode {
                    ClipVolSubMode::BoxClip => {
                        self.clipvol_box_center[0] += delta.x;
                        self.clipvol_box_center[1] += delta.y;
                        self.clipvol_box_center[2] += delta.z;
                    }
                    ClipVolSubMode::SphereClip => {
                        self.clipvol_sphere_center[0] += delta.x;
                        self.clipvol_sphere_center[1] += delta.y;
                        self.clipvol_sphere_center[2] += delta.z;
                    }
                    ClipVolSubMode::InteractivePlane => {
                        if let ClipShape::Plane {
                            ref mut distance,
                            ref normal,
                            ..
                        } = self.clipvol_plane.shape
                        {
                            let n = glam::Vec3::from(*normal).normalize_or_zero();
                            // Plane sits at -normal*distance; moving by delta means
                            // new_distance = distance - delta.dot(n).
                            *distance -= delta.dot(n);
                        }
                    }
                }
            }

            GizmoMode::Rotate => {
                let angle = match axis {
                    GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                        let dir = axis_dir(axis);
                        gizmo::project_drag_onto_rotation(drag_delta, dir, view)
                    }
                    _ => 0.0,
                };
                if angle.abs() > 1e-6 {
                    match self.clipvol_sub_mode {
                        ClipVolSubMode::BoxClip => {
                            self.clipvol_box_yaw += angle.to_degrees();
                        }
                        ClipVolSubMode::InteractivePlane => {
                            if let ClipShape::Plane {
                                ref mut normal,
                                ref mut distance,
                                ..
                            } = self.clipvol_plane.shape
                            {
                                let n = glam::Vec3::from(*normal);
                                let rot_axis = axis_dir(axis);
                                let rot = glam::Quat::from_axis_angle(rot_axis, angle);
                                let new_n = (rot * n).normalize_or_zero();
                                // Plane sits at anchor = -n * distance. Keep the anchor
                                // fixed: new anchor = -new_n * new_distance = -n * distance
                                // -> new_distance = -(-n * distance).dot(new_n) / |new_n|²
                                //                = (n * distance).dot(new_n)
                                let anchor = n * (-*distance);
                                *distance = -(anchor.dot(new_n));
                                *normal = new_n.to_array();
                            }
                        }
                        ClipVolSubMode::SphereClip => {}
                    }
                }
            }

            GizmoMode::Scale => {
                let amount = match axis {
                    GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                        let dir = axis_dir(axis);
                        gizmo::project_drag_onto_axis(drag_delta, dir, vp, center, viewport_size)
                    }
                    _ => 0.0,
                };
                if amount.abs() > 1e-6 {
                    match self.clipvol_sub_mode {
                        ClipVolSubMode::BoxClip => {
                            let scale = 1.0 + amount;
                            match axis {
                                GizmoAxis::X => {
                                    self.clipvol_box_half_extents[0] =
                                        (self.clipvol_box_half_extents[0] * scale).max(0.1);
                                }
                                GizmoAxis::Y => {
                                    self.clipvol_box_half_extents[1] =
                                        (self.clipvol_box_half_extents[1] * scale).max(0.1);
                                }
                                GizmoAxis::Z => {
                                    self.clipvol_box_half_extents[2] =
                                        (self.clipvol_box_half_extents[2] * scale).max(0.1);
                                }
                                _ => {}
                            }
                        }
                        ClipVolSubMode::SphereClip => {
                            self.clipvol_sphere_radius =
                                (self.clipvol_sphere_radius + amount).max(0.1);
                        }
                        ClipVolSubMode::InteractivePlane => {}
                    }
                }
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Shared mesh upload helper
// ---------------------------------------------------------------------------

impl App {
    /// Upload a new box mesh slot into the renderer. Used by showcase build methods.
    pub(crate) fn upload_box(&self, renderer: &mut ViewportRenderer) -> MeshId {
        let idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &self.box_mesh_data)
            .expect("box mesh upload");
        MeshId::from_index(idx)
    }
}

// ---------------------------------------------------------------------------
// Material and background presets
// ---------------------------------------------------------------------------

fn material_preset(index: usize) -> Material {
    match index % 4 {
        0 => Material::default(),
        1 => {
            let mut m = Material::from_color([0.8, 0.2, 0.2]);
            m.specular = 0.8;
            m.shininess = 64.0;
            m.ambient = 0.1;
            m
        }
        2 => {
            let mut m = Material::from_color([0.2, 0.4, 0.9]);
            m.opacity = 0.5;
            m.specular = 0.9;
            m.shininess = 128.0;
            m
        }
        3 => {
            let mut m = Material::from_color([0.3, 0.7, 0.3]);
            m.specular = 0.1;
            m.shininess = 8.0;
            m.diffuse = 0.9;
            m
        }
        _ => unreachable!(),
    }
}

fn background_color(index: usize) -> [f32; 4] {
    match index % 3 {
        0 => BG_COLOR,
        1 => [0.05, 0.08, 0.15, 1.0],
        2 => [0.18, 0.16, 0.14, 1.0],
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Gizmo orientation helper
// ---------------------------------------------------------------------------

fn egui_key_to_keycode(key: egui::Key) -> Option<KeyCode> {
    match key {
        egui::Key::A => Some(KeyCode::A),
        egui::Key::D => Some(KeyCode::D),
        egui::Key::E => Some(KeyCode::E),
        egui::Key::F => Some(KeyCode::F),
        egui::Key::G => Some(KeyCode::G),
        egui::Key::Q => Some(KeyCode::Q),
        egui::Key::R => Some(KeyCode::R),
        egui::Key::S => Some(KeyCode::S),
        egui::Key::W => Some(KeyCode::W),
        egui::Key::X => Some(KeyCode::X),
        egui::Key::Y => Some(KeyCode::Y),
        egui::Key::Z => Some(KeyCode::Z),
        egui::Key::Tab => Some(KeyCode::Tab),
        egui::Key::Enter => Some(KeyCode::Enter),
        egui::Key::Escape => Some(KeyCode::Escape),
        egui::Key::Backspace => Some(KeyCode::Backspace),
        egui::Key::Backtick => Some(KeyCode::Backtick),
        egui::Key::Comma => Some(KeyCode::Comma),
        egui::Key::Period => Some(KeyCode::Period),
        egui::Key::OpenBracket => Some(KeyCode::LeftBracket),
        egui::Key::CloseBracket => Some(KeyCode::RightBracket),
        egui::Key::Slash => Some(KeyCode::Slash),
        _ => None,
    }
}

fn gizmo_orientation(gizmo: &Gizmo, selection: &Selection, scene: &Scene) -> glam::Quat {
    match gizmo.space {
        GizmoSpace::World => glam::Quat::IDENTITY,
        GizmoSpace::Local => selection
            .primary()
            .and_then(|id| scene.node(id))
            .map(|n| glam::Quat::from_mat4(&n.world_transform()))
            .unwrap_or(glam::Quat::IDENTITY),
    }
}

// ---------------------------------------------------------------------------
// Unit box mesh
// ---------------------------------------------------------------------------

fn unit_box_mesh() -> MeshData {
    #[rustfmt::skip]
    let positions: Vec<[f32; 3]> = vec![
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
    ];

    #[rustfmt::skip]
    let normals: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    ];

    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0,  1,  2,  0,  2,  3,
        4,  5,  6,  4,  6,  7,
        8,  9,  10, 8,  10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    ];

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}
