//! Feature showcase for `viewport-lib` on top of `winit` and `wgpu`.
//!
//! Showcase modes are toggled with **1**, **2**, **3**, **4**:
//!
//! ## Showcase 1 (default) - Rendering Basics
//!   - Directional / point / spot light toggle (press **L**)
//!   - Orthographic / perspective projection toggle (press **O**)
//!   - Blinn-Phong lighting with specular highlights
//!   - Directional shadow mapping with 3x3 PCF
//!   - 4x MSAA anti-aliasing
//!
//! ## Showcase 2 - Phase 2 & 3 Features
//!   - **M**: Cycle materials (default -> red metal -> blue glass -> green matte)
//!   - **T**: Toggle transparency on selected objects
//!   - **N**: Toggle normal visualization on selected objects
//!   - **H**: Toggle hemisphere ambient lighting
//!   - **B**: Cycle background color (dark -> blue -> warm gray)
//!   - **G**: Add a child node to the selected node (hierarchy demo)
//!   - **R**: Remove selected node (and children)
//!   - **V**: Toggle layer visibility (hides "Layer B")
//!   - **Click**: Select object under cursor (single-select)
//!   - **Tab**: Cycle selection through scene nodes
//!   - **Delete**: Remove selected node from scene
//!
//! ## Showcase 3 - Phase 4 Performance
//!   - 300 boxes (10x10x3 grid) sharing a single mesh - triggers GPU instancing
//!   - BVH-accelerated picking: click to select objects
//!   - Live `FrameStats` in title bar: objects, visible, culled, draws, batches, triangles
//!
//! ## Showcase 4 - Phase 5 Professional Interaction
//!   - Smooth camera with exponential damping (orbit/pan/zoom inertia)
//!   - **F/B/L/R/T/P/I**: Animated fly-to view presets (Front/Back/Left/Right/Top/Bottom/Isometric)
//!   - **Z**: Zoom-to-fit selection (or entire scene)
//!   - **Tab**: Cycle gizmo mode (Translate -> Rotate -> Scale)
//!   - **X**: Toggle gizmo space (World / Local)
//!   - **S**: Toggle snap (off -> 0.5 unit translation snap -> 15 deg rotation snap)
//!   - **Click**: Select objects, gizmo renders on selection
//!   - Title bar shows: gizmo mode, space, snap config, animator state
//!
//! ## Showcase 5 - Phase 6 Advanced Rendering
//!   - Side-by-side PBR vs Blinn-Phong comparison
//!   - **C**: Toggle section-view clip plane (clips x<0, reveals PBR side only)
//!   - **O**: Toggle selection outline (orange stencil ring)
//!   - **Q**: Toggle X-ray mode (selected objects visible through occluders)
//!   - **Click**: Select object under cursor
//!   - **Tab**: Cycle selection through scene nodes
//!
//! ## Common Controls
//!   - Left-drag or Middle-drag:  Orbit
//!   - Right-drag or Shift+Middle-drag: Pan
//!   - Scroll wheel: Zoom
//!   - Escape: Quit

use std::sync::Arc;

use viewport_lib::{
    Camera, CameraAnimator, CameraFrame, ClipPlane, Easing, FrameData, FrameStats, Gizmo,
    GizmoAxis, GizmoMode, GizmoSpace, LightKind, LightSource, LightingSettings, Material, MeshData,
    MeshId, NodeId, PickAccelerator, PostProcessSettings, Projection, RenderCamera, SceneFrame,
    SceneRenderItem, Selection, ShadowFilter, SnapConfig, ToneMapping, ViewPreset,
    ViewportRenderer,
    gizmo::{self, compute_gizmo_scale},
    scene::{LayerId, Scene},
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

mod annotation_demo;
mod geometry;
mod showcase_advanced;
mod showcase_interaction;
mod showcase_normal_maps;
mod showcase_performance;
mod showcase_post_process;
mod showcase_scene_graph;
mod showcase_shadows;

const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.001;
/// Reduced sensitivity for CameraAnimator paths - damping accumulates velocity,
/// so raw input must be scaled down to match the feel of direct manipulation.
const ANIM_ORBIT_SENSITIVITY: f32 = 0.0008;
const ANIM_ZOOM_SENSITIVITY: f32 = 0.0003;
const MSAA_SAMPLES: u32 = 4;
const BG_COLOR: [f32; 4] = [0.08, 0.08, 0.10, 1.0];

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");
}

// ---------------------------------------------------------------------------
// Showcase mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShowcaseMode {
    /// Original rendering basics (lights, projection).
    Basic,
    /// Phase 2 & 3: materials, transparency, scene graph, selection, layers.
    SceneGraph,
    /// Phase 4: frustum culling, instancing, BVH picking, frame stats.
    Performance,
    /// Phase 5: camera animator, view presets, gizmo modes, snap, zoom-to-fit.
    Interaction,
    /// Phase 6: PBR, clip planes, stencil outlines, x-ray.
    Advanced,
    /// Phase 6.2: HDR, tone mapping, bloom, SSAO.
    PostProcess,
    /// Phase 6.3: Normal maps, AO maps, tangent computation.
    NormalMaps,
    /// Phase 6.3: CSM shadows, PCSS, contact shadows.
    Shadows,
    /// Phase O: World-space annotation labels (world_to_screen demo).
    Annotation,
}

impl ShowcaseMode {
    fn label(self) -> &'static str {
        match self {
            Self::Basic => "1: Rendering Basics",
            Self::SceneGraph => "2: Scene Graph + Materials",
            Self::Performance => "3: Performance at Scale",
            Self::Interaction => "4: Professional Interaction",
            Self::Advanced => "5: Advanced Rendering",
            Self::PostProcess => "6: Post-Processing",
            Self::NormalMaps => "7: Normal Maps + AO",
            Self::Shadows => "8: Shadow Demo",
            Self::Annotation => "9: Annotation Labels",
        }
    }
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

pub(crate) struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    msaa_view: wgpu::TextureView,

    renderer: ViewportRenderer,
    camera: Camera,

    // -- Showcase 1 state --
    mesh_indices: Vec<usize>,
    use_point_light: bool,

    // -- Showcase 2 state --
    mode: ShowcaseMode,
    scene: Scene,
    selection: Selection,
    /// Cached box mesh data for on-demand uploads (each node needs its own GPU mesh slot).
    box_mesh_data: MeshData,
    material_cycle: usize,
    bg_cycle: usize,
    hemisphere_on: bool,
    layer_b: LayerId,
    layer_b_visible: bool,

    // -- Showcase 3 state --
    perf_scene: Scene,
    perf_selection: Selection,
    pick_accelerator: Option<PickAccelerator>,
    /// The single shared mesh used by all perf-scene boxes.
    perf_mesh: Option<MeshId>,
    last_stats: FrameStats,
    /// Total objects in the perf scene (before culling).
    perf_total_objects: u32,
    /// Cached render items for the performance scene. Rebuilt only when perf_scene.version() or
    /// perf_selection.version() changes. Avoids O(N) Vec allocation every frame for a static
    /// 1M-item scene.
    perf_scene_items_cache: Vec<SceneRenderItem>,
    /// The (perf_scene.version(), perf_selection.version()) at which the cache was last built.
    perf_scene_items_version: (u64, u64),

    // -- Showcase 4 state --
    interact_scene: Scene,
    interact_selection: Selection,
    interact_animator: CameraAnimator,
    interact_gizmo: Gizmo,
    interact_snap_cycle: usize,
    interact_snap: SnapConfig,
    /// Cumulative unsnapped translation during a gizmo drag (for snap accumulation).
    interact_drag_accum_translation: glam::Vec3,
    /// Cumulative unsnapped rotation angle during a gizmo drag (for snap accumulation).
    interact_drag_accum_rotation: f32,
    /// Last snapped translation applied (to compute per-frame snapped delta).
    interact_drag_last_snapped_translation: glam::Vec3,
    /// Last snapped rotation angle applied (to compute per-frame snapped delta).
    interact_drag_last_snapped_rotation: f32,
    interact_built: bool,
    /// Cached gizmo center + scale for the current frame (set during render, used for drag).
    interact_gizmo_center: Option<glam::Vec3>,
    interact_gizmo_scale: f32,
    /// Time of last frame for dt computation.
    last_instant: Option<std::time::Instant>,

    // -- Showcase 5 state --
    adv_scene: Scene,
    adv_selection: Selection,
    adv_clip_enabled: bool,
    adv_outline_on: bool,
    adv_xray_on: bool,
    adv_built: bool,

    // -- Showcase 6 state --
    pp_scene: Scene,
    pp_built: bool,
    pp_tone_mapping: viewport_lib::ToneMapping,
    pp_exposure: f32,
    pp_bloom: bool,
    pp_ssao: bool,
    pp_fxaa: bool,

    // -- Showcase 7 state --
    nm_scene: Scene,
    nm_built: bool,
    /// Texture IDs for normal map showcase: [nm_id, nm_id, ao_id].
    nm_tex_ids: [u64; 3],
    /// Tile texture IDs for the cube: [tile_nm_id, tile_ao_id].
    nm_tile_tex_ids: [u64; 2],
    /// NodeId of the sphere that has normal_map + AO applied (left sphere).
    nm_node: Option<NodeId>,
    /// NodeId of the cube with tile normal+AO maps.
    nm_cube_node: Option<NodeId>,
    nm_normal_on: bool,
    nm_ao_on: bool,
    nm_clip_enabled: bool,

    // -- Showcase 8 state --
    shd_scene: Scene,
    shd_built: bool,
    shd_cascade_count: u32,
    shd_pcss_on: bool,
    shd_contact_on: bool,

    // -- Showcase 9 state: annotation labels --
    /// The 3D scene for the annotation demo (marker boxes at each label's world_pos).
    ann_scene: Scene,
    /// Whether the annotation scene has been built.
    ann_built: bool,
    /// Annotation labels - each holds a world_pos, text, optional leader, colour, font_size.
    ann_labels: Vec<viewport_lib::AnnotationLabel>,

    // Mouse tracking.
    left_pressed: bool,
    middle_pressed: bool,
    right_pressed: bool,
    shift_held: bool,
    ctrl_held: bool,
    last_cursor: PhysicalPosition<f64>,
    /// True if the left button has moved more than the click threshold since it was pressed.
    /// Used to distinguish an orbit drag from a click-to-select.
    left_drag_active: bool,
}

impl AppState {
    fn create_depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("showcase_depth"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_msaa_view(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        w: u32,
        h: u32,
    ) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("showcase_msaa"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Upload a fresh box mesh and return its MeshId.
    /// Each scene node needs its own GPU mesh slot (own uniform buffer).
    pub(crate) fn upload_box(&mut self) -> MeshId {
        let idx = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &self.box_mesh_data)
            .expect("box mesh upload");
        MeshId::from_index(idx)
    }

    fn update_title(&self) {
        let mode_label = self.mode.label();
        let extra = match self.mode {
            ShowcaseMode::Basic => {
                let proj = match self.camera.projection {
                    Projection::Perspective => "Perspective",
                    Projection::Orthographic => "Orthographic",
                    _ => "Unknown",
                };
                let light = if self.use_point_light {
                    "Point"
                } else {
                    "Directional"
                };
                format!("{proj} / {light}  [O proj] [L light]")
            }
            ShowcaseMode::SceneGraph => {
                let sel_count = self.selection.len();
                let node_count = self.scene.node_count();
                format!(
                    "Nodes: {node_count} | Selected: {sel_count}  [M mat] [T transp] [N normals] [H hemi] [B bg] [G child] [R remove] [V layer]"
                )
            }
            ShowcaseMode::Performance => {
                let s = &self.last_stats;
                let total = self.perf_total_objects;
                let visible = s.total_objects; // renderer sees post-cull items
                let culled = total.saturating_sub(visible);
                format!(
                    "Objects: {total} | Visible: {visible} | Culled: {culled} | Draws: {} | Batches: {} | Tris: {} | ShadowDraws: {}",
                    s.draw_calls, s.instanced_batches, s.triangles_submitted, s.shadow_draw_calls,
                )
            }
            ShowcaseMode::Interaction => {
                let mode = match self.interact_gizmo.mode {
                    GizmoMode::Translate => "Translate",
                    GizmoMode::Rotate => "Rotate",
                    GizmoMode::Scale => "Scale",
                    _ => "Unknown",
                };
                let space = match self.interact_gizmo.space {
                    GizmoSpace::World => "World",
                    GizmoSpace::Local => "Local",
                };
                let snap_str = match (
                    &self.interact_snap.translation,
                    &self.interact_snap.rotation,
                ) {
                    (Some(t), _) => format!("Snap: {t}u"),
                    (_, Some(r)) => format!("Snap: {:.0}°", r.to_degrees()),
                    _ => "Snap: Off".to_string(),
                };
                let anim = if self.interact_animator.is_animating() {
                    " [animating]"
                } else {
                    ""
                };
                format!(
                    "{mode} | {space} | {snap_str}{anim}  [Tab gizmo] [X space] [S snap] [FBLRTPI view] [Z fit]"
                )
            }
            ShowcaseMode::Advanced => {
                let clip = if self.adv_clip_enabled { "ON" } else { "off" };
                let outline = if self.adv_outline_on { "ON" } else { "off" };
                let xray = if self.adv_xray_on { "ON" } else { "off" };
                let sel = self.adv_selection.len();
                format!(
                    "Clip:{clip} Outline:{outline} XRay:{xray} | Selected:{sel}  [C clip] [O outline] [Q xray] [click select]"
                )
            }
            ShowcaseMode::PostProcess => {
                let tm = match self.pp_tone_mapping {
                    ToneMapping::Reinhard => "Reinhard",
                    ToneMapping::Aces => "ACES",
                    ToneMapping::KhronosNeutral => "KhronosNeutral",
                };
                let bloom = if self.pp_bloom { "ON" } else { "off" };
                let ssao = if self.pp_ssao { "ON" } else { "off" };
                let fxaa = if self.pp_fxaa { "ON" } else { "off" };
                format!(
                    "ToneMap:{tm} Exp:{:.1} Bloom:{bloom} SSAO:{ssao} FXAA:{fxaa}  [T tone] [[/]] exposure] [B bloom] [A ssao] [X fxaa]",
                    self.pp_exposure
                )
            }
            ShowcaseMode::NormalMaps => {
                let nm = if self.nm_normal_on { "ON" } else { "off" };
                let ao = if self.nm_ao_on { "ON" } else { "off" };
                let clip = if self.nm_clip_enabled { "ON" } else { "off" };
                format!("NormalMap:{nm} AO:{ao} Clip:{clip}  [N normal] [O ao] [C clip]")
            }
            ShowcaseMode::Shadows => {
                let filter = if self.shd_pcss_on { "PCSS" } else { "PCF" };
                let contact = if self.shd_contact_on { "ON" } else { "off" };
                let n = self.shd_cascade_count;
                format!(
                    "Cascades:{n} Filter:{filter} Contact:{contact}  [[ / ] cascades] [P pcss] [K contact]"
                )
            }
            ShowcaseMode::Annotation => {
                // Project all labels and show screen positions in the title bar.
                // L3 should always read "clipped" because its world_pos is behind the camera.
                self.annotation_title()
            }
        };
        self.window
            .set_title(&format!("viewport-lib Showcase [{mode_label}] - {extra}"));
    }
}

// ---------------------------------------------------------------------------
// Material presets for cycling
// ---------------------------------------------------------------------------

fn material_preset(index: usize) -> Material {
    match index % 4 {
        0 => Material::default(),
        1 => { let mut m = Material::from_color([0.8, 0.2, 0.2]); m.specular = 0.8; m.shininess = 64.0; m.ambient = 0.1; m },
        2 => { let mut m = Material::from_color([0.2, 0.4, 0.9]); m.opacity = 0.5; m.specular = 0.9; m.shininess = 128.0; m },
        3 => { let mut m = Material::from_color([0.3, 0.7, 0.3]); m.specular = 0.1; m.shininess = 8.0; m.diffuse = 0.9; m },
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
// ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("viewport-lib Showcase [1: Rendering Basics]")
                        .with_inner_size(winit::dpi::LogicalSize::new(1000u32, 700u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("showcase_device"),
            ..Default::default()
        }))
        .expect("Failed to create wgpu device");

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_view = AppState::create_depth_view(&device, config.width, config.height);
        let msaa_view = AppState::create_msaa_view(&device, format, config.width, config.height);

        let mut renderer = ViewportRenderer::with_sample_count(&device, format, MSAA_SAMPLES);

        // Upload meshes for showcase 1 (4 separate uploads).
        let box_mesh = unit_box_mesh();
        let mut mesh_indices = Vec::new();
        for _ in 0..4 {
            let idx = renderer
                .resources_mut()
                .upload_mesh_data(&device, &box_mesh)
                .expect("built-in mesh");
            mesh_indices.push(idx);
        }

        let camera = Camera {
            center: glam::Vec3::ZERO,
            distance: 12.0,
            orientation: glam::Quat::from_rotation_y(0.6) * glam::Quat::from_rotation_x(-0.4),
            ..Camera::default()
        };

        // Create Scene with a second layer.
        let mut scene = Scene::new();
        let layer_b = scene.add_layer("Layer B");

        let mut app_state = AppState {
            window,
            surface,
            device,
            queue,
            surface_config: config,
            depth_view,
            msaa_view,
            renderer,
            camera,
            mesh_indices,
            use_point_light: false,
            mode: ShowcaseMode::Basic,
            scene,
            selection: Selection::new(),
            box_mesh_data: box_mesh,
            material_cycle: 0,
            bg_cycle: 0,
            hemisphere_on: false,
            layer_b,
            layer_b_visible: true,
            perf_scene: Scene::new(),
            perf_selection: Selection::new(),
            pick_accelerator: None,
            perf_mesh: None,
            last_stats: FrameStats::default(),
            perf_total_objects: 0,
            perf_scene_items_cache: Vec::new(),
            perf_scene_items_version: (u64::MAX, u64::MAX),
            interact_scene: Scene::new(),
            interact_selection: Selection::new(),
            interact_animator: CameraAnimator::with_default_damping(),
            interact_gizmo: Gizmo::new(),
            interact_snap_cycle: 0,
            interact_snap: SnapConfig::default(),
            interact_drag_accum_translation: glam::Vec3::ZERO,
            interact_drag_accum_rotation: 0.0,
            interact_drag_last_snapped_translation: glam::Vec3::ZERO,
            interact_drag_last_snapped_rotation: 0.0,
            interact_built: false,
            interact_gizmo_center: None,
            interact_gizmo_scale: 1.0,
            last_instant: None,
            adv_scene: Scene::new(),
            adv_selection: Selection::new(),
            adv_clip_enabled: false,
            adv_outline_on: false,
            adv_xray_on: false,
            adv_built: false,
            pp_scene: Scene::new(),
            pp_built: false,
            pp_tone_mapping: ToneMapping::Aces,
            pp_exposure: 1.0,
            pp_bloom: true,
            pp_ssao: false,
            pp_fxaa: false,
            nm_scene: Scene::new(),
            nm_built: false,
            nm_tex_ids: [0, 0, 0],
            nm_tile_tex_ids: [0, 0],
            nm_node: None,
            nm_cube_node: None,
            nm_normal_on: true,
            nm_ao_on: true,
            nm_clip_enabled: false,
            shd_scene: Scene::new(),
            shd_built: false,
            shd_cascade_count: 4,
            shd_pcss_on: false,
            shd_contact_on: false,
            ann_scene: Scene::new(),
            ann_built: false,
            ann_labels: Vec::new(),
            left_pressed: false,
            middle_pressed: false,
            right_pressed: false,
            shift_held: false,
            ctrl_held: false,
            last_cursor: PhysicalPosition::new(0.0, 0.0),
            left_drag_active: false,
        };

        app_state.build_scene();
        app_state.update_title();
        self.state = Some(app_state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.surface_config.width = new_size.width;
                    state.surface_config.height = new_size.height;
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    state.depth_view =
                        AppState::create_depth_view(&state.device, new_size.width, new_size.height);
                    state.msaa_view = AppState::create_msaa_view(
                        &state.device,
                        state.surface_config.format,
                        new_size.width,
                        new_size.height,
                    );
                    state.window.request_redraw();
                }
            }

            // --- Keyboard ---
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match &event.logical_key {
                    // Mode switching.
                    Key::Character(c) if c.as_str() == "1" => {
                        state.mode = ShowcaseMode::Basic;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c) if c.as_str() == "2" => {
                        state.mode = ShowcaseMode::SceneGraph;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c) if c.as_str() == "3" => {
                        if state.perf_mesh.is_none() {
                            state.build_perf_scene();
                            state.camera.distance = 80.0;
                        }
                        state.mode = ShowcaseMode::Performance;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c) if c.as_str() == "4" => {
                        if !state.interact_built {
                            state.build_interact_scene();
                            state.camera = Camera {
                                center: glam::Vec3::ZERO,
                                distance: 12.0,
                                orientation: glam::Quat::from_rotation_y(0.6)
                                    * glam::Quat::from_rotation_x(-0.4),
                                ..Camera::default()
                            };
                        }
                        state.mode = ShowcaseMode::Interaction;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c) if c.as_str() == "5" => {
                        if !state.adv_built {
                            state.build_adv_scene();
                            state.camera = Camera {
                                center: glam::Vec3::new(0.0, 0.5, 2.0),
                                distance: 14.0,
                                orientation: glam::Quat::from_rotation_y(0.4)
                                    * glam::Quat::from_rotation_x(-0.3),
                                ..Camera::default()
                            };
                        }
                        state.mode = ShowcaseMode::Advanced;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c) if c.as_str() == "6" => {
                        if !state.pp_built {
                            state.build_pp_scene();
                            state.camera = Camera {
                                center: glam::Vec3::new(0.0, 0.5, 0.0),
                                distance: 8.0,
                                orientation: glam::Quat::from_rotation_y(0.6)
                                    * glam::Quat::from_rotation_x(-0.45),
                                ..Camera::default()
                            };
                        }
                        state.mode = ShowcaseMode::PostProcess;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c) if c.as_str() == "7" => {
                        if !state.nm_built {
                            state.build_nm_scene();
                            state.camera = Camera {
                                center: glam::Vec3::new(0.0, 0.0, 0.0),
                                distance: 6.0,
                                orientation: glam::Quat::from_rotation_y(0.5)
                                    * glam::Quat::from_rotation_x(-0.4),
                                ..Camera::default()
                            };
                        }
                        state.mode = ShowcaseMode::NormalMaps;
                        state.update_title();
                        state.window.request_redraw();
                    }

                    Key::Character(c) if c.as_str() == "8" => {
                        if !state.shd_built {
                            state.build_shadow_scene();
                            state.camera = Camera {
                                center: glam::Vec3::new(0.0, 1.0, 0.0),
                                distance: 14.0,
                                orientation: glam::Quat::from_rotation_y(0.4)
                                    * glam::Quat::from_rotation_x(-0.4),
                                ..Camera::default()
                            };
                        }
                        state.mode = ShowcaseMode::Shadows;
                        state.update_title();
                        state.window.request_redraw();
                    }

                    Key::Character(c) if c.as_str() == "9" => {
                        if !state.ann_built {
                            state.build_annotation_scene();
                            state.reset_annotation_camera();
                            state.ann_built = true;
                        }
                        state.mode = ShowcaseMode::Annotation;
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // -- Showcase 1 keys --
                    Key::Character(c)
                        if (c.as_str() == "o" || c.as_str() == "O")
                            && state.mode == ShowcaseMode::Basic =>
                    {
                        state.camera.projection = match state.camera.projection {
                            Projection::Perspective => Projection::Orthographic,
                            _ => Projection::Perspective,
                        };
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c)
                        if (c.as_str() == "l" || c.as_str() == "L")
                            && state.mode == ShowcaseMode::Basic =>
                    {
                        state.use_point_light = !state.use_point_light;
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // -- Showcase 2 keys --
                    // M: Cycle material on selected nodes.
                    Key::Character(c)
                        if (c.as_str() == "m" || c.as_str() == "M")
                            && state.mode == ShowcaseMode::SceneGraph =>
                    {
                        state.material_cycle += 1;
                        let mat = material_preset(state.material_cycle);
                        for &id in state.selection.iter() {
                            state.scene.set_material(id, mat);
                        }
                        state.window.request_redraw();
                    }

                    // T: Toggle transparency on selected.
                    Key::Character(c)
                        if (c.as_str() == "t" || c.as_str() == "T")
                            && state.mode == ShowcaseMode::SceneGraph =>
                    {
                        for &id in state.selection.iter() {
                            if let Some(node) = state.scene.node(id) {
                                let mut mat = *node.material();
                                mat.opacity = if mat.opacity < 1.0 { 1.0 } else { 0.4 };
                                state.scene.set_material(id, mat);
                            }
                        }
                        state.window.request_redraw();
                    }

                    // N: Toggle normal visualization on selected (SceneGraph showcase).
                    Key::Character(c)
                        if (c.as_str() == "n" || c.as_str() == "N")
                            && state.mode == ShowcaseMode::SceneGraph =>
                    {
                        for &id in state.selection.iter() {
                            if let Some(node) = state.scene.node(id) {
                                let show = !node.show_normals();
                                state.scene.set_show_normals(id, show);
                            }
                        }
                        state.window.request_redraw();
                    }

                    // N: Toggle normal map on/off (NormalMaps showcase).
                    Key::Character(c)
                        if (c.as_str() == "n" || c.as_str() == "N")
                            && state.mode == ShowcaseMode::NormalMaps =>
                    {
                        state.nm_normal_on = !state.nm_normal_on;
                        // Toggle brick sphere normal map.
                        if let Some(id) = state.nm_node {
                            if let Some(node) = state.nm_scene.node(id) {
                                let nm_id = state.nm_tex_ids[0];
                                let mut mat = *node.material();
                                mat.normal_map_id = if state.nm_normal_on {
                                    Some(nm_id)
                                } else {
                                    None
                                };
                                state.nm_scene.set_material(id, mat);
                            }
                        }
                        // Toggle tile cube normal map.
                        if let Some(id) = state.nm_cube_node {
                            if let Some(node) = state.nm_scene.node(id) {
                                let tile_nm_id = state.nm_tile_tex_ids[0];
                                let mut mat = *node.material();
                                mat.normal_map_id = if state.nm_normal_on {
                                    Some(tile_nm_id)
                                } else {
                                    None
                                };
                                state.nm_scene.set_material(id, mat);
                            }
                        }
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // O: Toggle AO map on/off (NormalMaps showcase).
                    Key::Character(c)
                        if (c.as_str() == "o" || c.as_str() == "O")
                            && state.mode == ShowcaseMode::NormalMaps =>
                    {
                        state.nm_ao_on = !state.nm_ao_on;
                        // Toggle brick sphere AO map.
                        if let Some(id) = state.nm_node {
                            if let Some(node) = state.nm_scene.node(id) {
                                let ao_id = state.nm_tex_ids[2];
                                let mut mat = *node.material();
                                mat.ao_map_id = if state.nm_ao_on { Some(ao_id) } else { None };
                                state.nm_scene.set_material(id, mat);
                            }
                        }
                        // Toggle tile cube AO map.
                        if let Some(id) = state.nm_cube_node {
                            if let Some(node) = state.nm_scene.node(id) {
                                let tile_ao_id = state.nm_tile_tex_ids[1];
                                let mut mat = *node.material();
                                mat.ao_map_id = if state.nm_ao_on {
                                    Some(tile_ao_id)
                                } else {
                                    None
                                };
                                state.nm_scene.set_material(id, mat);
                            }
                        }
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // C: Toggle clip plane (NormalMaps showcase).
                    Key::Character(c)
                        if (c.as_str() == "c" || c.as_str() == "C")
                            && state.mode == ShowcaseMode::NormalMaps =>
                    {
                        state.nm_clip_enabled = !state.nm_clip_enabled;
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // -- Showcase 8 keys --
                    // [: Decrease cascade count, ]: Increase cascade count.
                    Key::Character(c)
                        if c.as_str() == "[" && state.mode == ShowcaseMode::Shadows =>
                    {
                        state.shd_cascade_count = (state.shd_cascade_count - 1).max(1);
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c)
                        if c.as_str() == "]" && state.mode == ShowcaseMode::Shadows =>
                    {
                        state.shd_cascade_count = (state.shd_cascade_count + 1).min(4);
                        state.update_title();
                        state.window.request_redraw();
                    }
                    // P: Toggle PCF/PCSS.
                    Key::Character(c)
                        if (c.as_str() == "p" || c.as_str() == "P")
                            && state.mode == ShowcaseMode::Shadows =>
                    {
                        state.shd_pcss_on = !state.shd_pcss_on;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    // K: Toggle contact shadows.
                    Key::Character(c)
                        if (c.as_str() == "k" || c.as_str() == "K")
                            && state.mode == ShowcaseMode::Shadows =>
                    {
                        state.shd_contact_on = !state.shd_contact_on;
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // H: Toggle hemisphere ambient.
                    Key::Character(c)
                        if (c.as_str() == "h" || c.as_str() == "H")
                            && state.mode == ShowcaseMode::SceneGraph =>
                    {
                        state.hemisphere_on = !state.hemisphere_on;
                        state.window.request_redraw();
                    }

                    // B: Cycle background color.
                    Key::Character(c)
                        if (c.as_str() == "b" || c.as_str() == "B")
                            && state.mode == ShowcaseMode::SceneGraph =>
                    {
                        state.bg_cycle += 1;
                        state.window.request_redraw();
                    }

                    // G: Add child node to selected (hierarchy demo).
                    Key::Character(c)
                        if (c.as_str() == "g" || c.as_str() == "G")
                            && state.mode == ShowcaseMode::SceneGraph =>
                    {
                        if let Some(parent_id) = state.selection.primary() {
                            let mesh = state.upload_box();
                            // Offset child by (1, 1, 0) in local space, scaled down.
                            let local = glam::Mat4::from_scale_rotation_translation(
                                glam::Vec3::splat(0.5),
                                glam::Quat::IDENTITY,
                                glam::Vec3::new(1.5, 1.5, 0.0),
                            );
                            let child_id = state.scene.add_named(
                                "Child",
                                Some(mesh),
                                local,
                                Material::from_color([1.0, 0.6, 0.2]),
                            );
                            state.scene.set_parent(child_id, Some(parent_id));
                            state.selection.select_one(child_id);
                            state.update_title();
                        }
                        state.window.request_redraw();
                    }

                    // R / Delete: Remove selected node (and children).
                    Key::Character(c)
                        if (c.as_str() == "r" || c.as_str() == "R")
                            && state.mode == ShowcaseMode::SceneGraph =>
                    {
                        if let Some(id) = state.selection.primary() {
                            let removed = state.scene.remove(id);
                            for rid in &removed {
                                state.selection.remove(*rid);
                            }
                            state.update_title();
                        }
                        state.window.request_redraw();
                    }
                    Key::Named(NamedKey::Delete) if state.mode == ShowcaseMode::SceneGraph => {
                        if let Some(id) = state.selection.primary() {
                            let removed = state.scene.remove(id);
                            for rid in &removed {
                                state.selection.remove(*rid);
                            }
                            state.update_title();
                        }
                        state.window.request_redraw();
                    }

                    // V: Toggle layer B visibility.
                    Key::Character(c)
                        if (c.as_str() == "v" || c.as_str() == "V")
                            && state.mode == ShowcaseMode::SceneGraph =>
                    {
                        state.layer_b_visible = !state.layer_b_visible;
                        state
                            .scene
                            .set_layer_visible(state.layer_b, state.layer_b_visible);
                        state.window.request_redraw();
                    }

                    // -- Showcase 5 keys --
                    // C: Toggle clip plane.
                    Key::Character(c)
                        if (c.as_str() == "c" || c.as_str() == "C")
                            && state.mode == ShowcaseMode::Advanced =>
                    {
                        state.adv_clip_enabled = !state.adv_clip_enabled;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    // O: Toggle selection outline.
                    Key::Character(c)
                        if (c.as_str() == "o" || c.as_str() == "O")
                            && state.mode == ShowcaseMode::Advanced =>
                    {
                        state.adv_outline_on = !state.adv_outline_on;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    // Q: Toggle x-ray.
                    Key::Character(c)
                        if (c.as_str() == "q" || c.as_str() == "Q")
                            && state.mode == ShowcaseMode::Advanced =>
                    {
                        state.adv_xray_on = !state.adv_xray_on;
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // F: Focus camera on selected objects (showcases 3, 5, 7).
                    Key::Character(c)
                        if (c.as_str() == "f" || c.as_str() == "F")
                            && matches!(
                                state.mode,
                                ShowcaseMode::Performance
                                    | ShowcaseMode::Advanced
                                    | ShowcaseMode::NormalMaps
                            ) =>
                    {
                        // Compute world AABB of selected items (or all if nothing selected).
                        let resources = state.renderer.resources();
                        let mut combined_min = glam::Vec3::splat(f32::INFINITY);
                        let mut combined_max = glam::Vec3::splat(f32::NEG_INFINITY);
                        let mut any = false;

                        let (scene, selection) = match state.mode {
                            ShowcaseMode::Performance => (&state.perf_scene, &state.perf_selection),
                            ShowcaseMode::Advanced => (&state.adv_scene, &state.adv_selection),
                            ShowcaseMode::NormalMaps => (&state.nm_scene, &state.perf_selection), // no selection for nm
                            _ => unreachable!(),
                        };

                        let iter_ids: Vec<_> = if !selection.is_empty() {
                            selection.iter().copied().collect()
                        } else {
                            scene.walk_depth_first().iter().map(|(id, _)| *id).collect()
                        };

                        for nid in iter_ids {
                            if let Some(node) = scene.node(nid) {
                                if let Some(mid) =
                                    viewport_lib::traits::ViewportObject::mesh_id(node)
                                {
                                    if let Some(gpu_mesh) = resources.mesh(mid as usize) {
                                        let world_aabb =
                                            gpu_mesh.aabb.transformed(&node.world_transform());
                                        combined_min = combined_min.min(world_aabb.min);
                                        combined_max = combined_max.max(world_aabb.max);
                                        any = true;
                                    }
                                }
                            }
                        }

                        if any {
                            let aabb = viewport_lib::Aabb {
                                min: combined_min,
                                max: combined_max,
                            };
                            state.camera.frame_aabb(&aabb);
                        }
                        state.window.request_redraw();
                    }

                    // -- Showcase 4 keys --
                    // Tab: Cycle gizmo mode.
                    Key::Named(NamedKey::Tab) if state.mode == ShowcaseMode::Interaction => {
                        state.interact_gizmo.mode = match state.interact_gizmo.mode {
                            GizmoMode::Translate => GizmoMode::Rotate,
                            GizmoMode::Rotate => GizmoMode::Scale,
                            _ => GizmoMode::Translate,
                        };
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // X: Toggle gizmo space.
                    Key::Character(c)
                        if (c.as_str() == "x" || c.as_str() == "X")
                            && state.mode == ShowcaseMode::Interaction =>
                    {
                        state.interact_gizmo.space = match state.interact_gizmo.space {
                            GizmoSpace::World => GizmoSpace::Local,
                            GizmoSpace::Local => GizmoSpace::World,
                        };
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // S: Cycle snap config.
                    Key::Character(c)
                        if (c.as_str() == "s" || c.as_str() == "S")
                            && state.mode == ShowcaseMode::Interaction =>
                    {
                        state.interact_snap_cycle = (state.interact_snap_cycle + 1) % 3;
                        state.interact_snap = match state.interact_snap_cycle {
                            0 => SnapConfig::default(), // off
                            1 => SnapConfig {
                                translation: Some(0.5),
                                ..SnapConfig::default()
                            },
                            2 => SnapConfig {
                                rotation: Some(std::f32::consts::PI / 12.0),
                                ..SnapConfig::default()
                            },
                            _ => unreachable!(),
                        };
                        state.update_title();
                        state.window.request_redraw();
                    }

                    // View presets: F/B/L/R/T/P/I (Front/Back/Left/Right/Top/Bottom/Isometric).
                    Key::Character(c)
                        if state.mode == ShowcaseMode::Interaction
                            && matches!(
                                c.as_str(),
                                "f" | "F"
                                    | "b"
                                    | "B"
                                    | "l"
                                    | "L"
                                    | "r"
                                    | "R"
                                    | "t"
                                    | "T"
                                    | "p"
                                    | "P"
                                    | "i"
                                    | "I"
                            ) =>
                    {
                        let preset = match c.as_str().to_ascii_lowercase().as_str() {
                            "f" => Some(ViewPreset::Front),
                            "b" => Some(ViewPreset::Back),
                            "l" => Some(ViewPreset::Left),
                            "r" => Some(ViewPreset::Right),
                            "t" => Some(ViewPreset::Top),
                            "p" => Some(ViewPreset::Bottom), // P for "bottom plane"
                            "i" => Some(ViewPreset::Isometric),
                            _ => None,
                        };
                        if let Some(preset) = preset {
                            state.interact_animator.fly_to_full(
                                &state.camera,
                                state.camera.center,
                                state.camera.distance,
                                preset.orientation(),
                                preset.preferred_projection(),
                                0.6,
                                Easing::EaseInOutCubic,
                            );
                            state.window.request_redraw();
                        }
                    }

                    // Z: Zoom-to-fit selection (or entire scene).
                    Key::Character(c)
                        if (c.as_str() == "z" || c.as_str() == "Z")
                            && state.mode == ShowcaseMode::Interaction =>
                    {
                        // Compute AABB of selected objects, or entire scene.
                        let resources = state.renderer.resources();
                        let mut combined_min = glam::Vec3::splat(f32::INFINITY);
                        let mut combined_max = glam::Vec3::splat(f32::NEG_INFINITY);
                        let mut any = false;

                        let iter_nodes: Vec<_> = if !state.interact_selection.is_empty() {
                            state.interact_selection.iter().copied().collect()
                        } else {
                            state
                                .interact_scene
                                .walk_depth_first()
                                .iter()
                                .map(|(id, _)| *id)
                                .collect()
                        };

                        for nid in iter_nodes {
                            if let Some(node) = state.interact_scene.node(nid) {
                                if let Some(mid) =
                                    viewport_lib::traits::ViewportObject::mesh_id(node)
                                {
                                    if let Some(gpu_mesh) = resources.mesh(mid as usize) {
                                        let world_aabb =
                                            gpu_mesh.aabb.transformed(&node.world_transform());
                                        combined_min = combined_min.min(world_aabb.min);
                                        combined_max = combined_max.max(world_aabb.max);
                                        any = true;
                                    }
                                }
                            }
                        }

                        if any {
                            let aabb = viewport_lib::Aabb {
                                min: combined_min,
                                max: combined_max,
                            };
                            let target = state.camera.fit_aabb_target(&aabb);
                            state.interact_animator.fly_to(
                                &state.camera,
                                target.center,
                                target.distance,
                                target.orientation,
                                0.6,
                            );
                            state.window.request_redraw();
                        }
                    }

                    // Tab: Cycle selection through scene nodes.
                    Key::Named(NamedKey::Tab) if state.mode == ShowcaseMode::SceneGraph => {
                        let walk = state.scene.walk_depth_first();
                        if !walk.is_empty() {
                            let current = state.selection.primary();
                            let next_idx = match current {
                                Some(id) => {
                                    let pos = walk.iter().position(|(nid, _)| *nid == id);
                                    match pos {
                                        Some(i) => (i + 1) % walk.len(),
                                        None => 0,
                                    }
                                }
                                None => 0,
                            };
                            state.selection.select_one(walk[next_idx].0);
                            state.update_title();
                        }
                        state.window.request_redraw();
                    }

                    Key::Named(NamedKey::Tab) if state.mode == ShowcaseMode::Advanced => {
                        let walk = state.adv_scene.walk_depth_first();
                        if !walk.is_empty() {
                            let current = state.adv_selection.primary();
                            let next_idx = match current {
                                Some(id) => {
                                    let pos = walk.iter().position(|(nid, _)| *nid == id);
                                    match pos {
                                        Some(i) => (i + 1) % walk.len(),
                                        None => 0,
                                    }
                                }
                                None => 0,
                            };
                            state.adv_selection.select_one(walk[next_idx].0);
                            state.update_title();
                        }
                        state.window.request_redraw();
                    }

                    // -- Showcase 6 keys --
                    Key::Character(c)
                        if (c.as_str() == "t" || c.as_str() == "T")
                            && state.mode == ShowcaseMode::PostProcess =>
                    {
                        state.pp_tone_mapping = match state.pp_tone_mapping {
                            ToneMapping::Reinhard => ToneMapping::Aces,
                            ToneMapping::Aces => ToneMapping::KhronosNeutral,
                            ToneMapping::KhronosNeutral => ToneMapping::Reinhard,
                        };
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c)
                        if c.as_str() == "[" && state.mode == ShowcaseMode::PostProcess =>
                    {
                        state.pp_exposure = (state.pp_exposure - 0.1).max(0.1);
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c)
                        if c.as_str() == "]" && state.mode == ShowcaseMode::PostProcess =>
                    {
                        state.pp_exposure = (state.pp_exposure + 0.1).min(5.0);
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c)
                        if (c.as_str() == "b" || c.as_str() == "B")
                            && state.mode == ShowcaseMode::PostProcess =>
                    {
                        state.pp_bloom = !state.pp_bloom;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c)
                        if (c.as_str() == "a" || c.as_str() == "A")
                            && state.mode == ShowcaseMode::PostProcess =>
                    {
                        state.pp_ssao = !state.pp_ssao;
                        state.update_title();
                        state.window.request_redraw();
                    }
                    Key::Character(c)
                        if (c.as_str() == "x" || c.as_str() == "X")
                            && state.mode == ShowcaseMode::PostProcess =>
                    {
                        state.pp_fxaa = !state.pp_fxaa;
                        state.update_title();
                        state.window.request_redraw();
                    }

                    Key::Named(NamedKey::Escape) => {
                        if state.mode == ShowcaseMode::SceneGraph && !state.selection.is_empty() {
                            state.selection.clear();
                            state.update_title();
                            state.window.request_redraw();
                        } else if state.mode == ShowcaseMode::Performance
                            && !state.perf_selection.is_empty()
                        {
                            state.perf_selection.clear();
                            state.update_title();
                            state.window.request_redraw();
                        } else if state.mode == ShowcaseMode::Advanced
                            && !state.adv_selection.is_empty()
                        {
                            state.adv_selection.clear();
                            state.update_title();
                            state.window.request_redraw();
                        } else {
                            event_loop.exit();
                        }
                    }
                    _ => {}
                }
            }

            // --- Mouse ---
            WindowEvent::ModifiersChanged(mods) => {
                state.shift_held = mods.state().shift_key();
                state.ctrl_held = mods.state().control_key();
            }

            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                let pressed = btn_state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        state.left_pressed = pressed;
                        if pressed {
                            state.left_drag_active = false;
                        }
                    }
                    MouseButton::Middle => state.middle_pressed = pressed,
                    MouseButton::Right => state.right_pressed = pressed,
                    _ => {}
                }

                // Click-to-select in Scene Graph mode (on release, only if not a drag).
                if button == MouseButton::Left
                    && !pressed
                    && !state.left_drag_active
                    && state.mode == ShowcaseMode::SceneGraph
                {
                    // Ray-cast picking against actual mesh triangles.
                    let w = state.surface_config.width as f32;
                    let h = state.surface_config.height as f32;
                    let vp = state.camera.view_proj_matrix();
                    let vp_inv = vp.inverse();
                    let cursor =
                        glam::Vec2::new(state.last_cursor.x as f32, state.last_cursor.y as f32);
                    let viewport_size = glam::Vec2::new(w, h);

                    let (ray_origin, ray_dir) =
                        viewport_lib::picking::screen_to_ray(cursor, viewport_size, vp_inv);

                    // Build mesh lookup: map each node's mesh_id (as u64) to CPU geometry.
                    // All nodes use the same box geometry but different mesh slot indices.
                    let mut mesh_lookup = std::collections::HashMap::new();
                    for node in state.scene.nodes() {
                        if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                            mesh_lookup.entry(mid).or_insert_with(|| {
                                (
                                    state.box_mesh_data.positions.clone(),
                                    state.box_mesh_data.indices.clone(),
                                )
                            });
                        }
                    }

                    let hit = viewport_lib::picking::pick_scene_nodes(
                        ray_origin,
                        ray_dir,
                        &state.scene,
                        &mesh_lookup,
                    );

                    if let Some(hit) = hit {
                        let id = hit.id;
                        if state.shift_held {
                            state.selection.toggle(id);
                        } else {
                            state.selection.select_one(id);
                        }
                    } else if !state.shift_held {
                        state.selection.clear();
                    }
                    state.update_title();
                }

                // Interaction mode: gizmo drag start on press, select on release.
                if button == MouseButton::Left && state.mode == ShowcaseMode::Interaction {
                    if pressed {
                        // Hit-test the gizmo on mouse press.
                        if let Some(center) = state.interact_gizmo_center {
                            let w = state.surface_config.width as f32;
                            let h = state.surface_config.height as f32;
                            let vp_inv = state.camera.view_proj_matrix().inverse();
                            let cursor = glam::Vec2::new(
                                state.last_cursor.x as f32,
                                state.last_cursor.y as f32,
                            );
                            let (ray_origin, ray_dir) = viewport_lib::picking::screen_to_ray(
                                cursor,
                                glam::Vec2::new(w, h),
                                vp_inv,
                            );

                            let orient = match state.interact_gizmo.space {
                                GizmoSpace::World => glam::Quat::IDENTITY,
                                GizmoSpace::Local => state
                                    .interact_selection
                                    .primary()
                                    .and_then(|id| state.interact_scene.node(id))
                                    .map(|n| glam::Quat::from_mat4(&n.world_transform()))
                                    .unwrap_or(glam::Quat::IDENTITY),
                            };
                            let hit = state.interact_gizmo.hit_test_oriented(
                                ray_origin,
                                ray_dir,
                                center,
                                state.interact_gizmo_scale,
                                orient,
                            );
                            if hit != GizmoAxis::None {
                                state.interact_gizmo.active_axis = hit;
                                state.interact_gizmo.drag_start_mouse = Some(cursor);
                                // Reset snap accumulators for a fresh drag.
                                state.interact_drag_accum_translation = glam::Vec3::ZERO;
                                state.interact_drag_accum_rotation = 0.0;
                                state.interact_drag_last_snapped_translation = glam::Vec3::ZERO;
                                state.interact_drag_last_snapped_rotation = 0.0;
                            }
                        }
                    } else {
                        // Mouse release: end gizmo drag or do click-to-select.
                        if state.interact_gizmo.active_axis != GizmoAxis::None {
                            // Was dragging gizmo - just end the drag, don't select.
                            state.interact_gizmo.active_axis = GizmoAxis::None;
                            state.interact_gizmo.drag_start_mouse = None;
                        } else if !state.left_drag_active {
                            // Click-to-select (only if not a drag).
                            let w = state.surface_config.width as f32;
                            let h = state.surface_config.height as f32;
                            let vp_inv = state.camera.view_proj_matrix().inverse();
                            let cursor = glam::Vec2::new(
                                state.last_cursor.x as f32,
                                state.last_cursor.y as f32,
                            );
                            let (ray_origin, ray_dir) = viewport_lib::picking::screen_to_ray(
                                cursor,
                                glam::Vec2::new(w, h),
                                vp_inv,
                            );

                            let mut mesh_lookup = std::collections::HashMap::new();
                            for node in state.interact_scene.nodes() {
                                if let Some(mid) =
                                    viewport_lib::traits::ViewportObject::mesh_id(node)
                                {
                                    mesh_lookup.entry(mid).or_insert_with(|| {
                                        (
                                            state.box_mesh_data.positions.clone(),
                                            state.box_mesh_data.indices.clone(),
                                        )
                                    });
                                }
                            }

                            let hit = viewport_lib::picking::pick_scene_nodes(
                                ray_origin,
                                ray_dir,
                                &state.interact_scene,
                                &mesh_lookup,
                            );

                            if let Some(hit) = hit {
                                let id = hit.id;
                                if state.shift_held {
                                    state.interact_selection.toggle(id);
                                } else {
                                    state.interact_selection.select_one(id);
                                }
                            } else if !state.shift_held {
                                state.interact_selection.clear();
                            }
                            state.update_title();
                        }
                    }
                }

                // Click-to-select in Performance mode (BVH-accelerated).
                if button == MouseButton::Left
                    && !pressed
                    && !state.left_drag_active
                    && state.mode == ShowcaseMode::Performance
                {
                    let w = state.surface_config.width as f32;
                    let h = state.surface_config.height as f32;
                    let vp_inv = state.camera.view_proj_matrix().inverse();
                    let cursor =
                        glam::Vec2::new(state.last_cursor.x as f32, state.last_cursor.y as f32);
                    let (ray_origin, ray_dir) =
                        viewport_lib::picking::screen_to_ray(cursor, glam::Vec2::new(w, h), vp_inv);

                    // Build mesh lookup - all perf-scene nodes share the same box geometry.
                    let mut mesh_lookup = std::collections::HashMap::new();
                    if let Some(mesh) = state.perf_mesh {
                        mesh_lookup.insert(
                            mesh.index() as u64,
                            (
                                state.box_mesh_data.positions.clone(),
                                state.box_mesh_data.indices.clone(),
                            ),
                        );
                    }

                    let hit = if let Some(ref mut accel) = state.pick_accelerator {
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
                        let id = hit.id;
                        if state.shift_held {
                            state.perf_selection.toggle(id);
                        } else {
                            state.perf_selection.select_one(id);
                        }
                    } else if !state.shift_held {
                        state.perf_selection.clear();
                    }
                    state.update_title();
                }

                // Click-to-select in Advanced mode.
                if button == MouseButton::Left
                    && !pressed
                    && !state.left_drag_active
                    && state.mode == ShowcaseMode::Advanced
                {
                    let w = state.surface_config.width as f32;
                    let h = state.surface_config.height as f32;
                    let vp_inv = state.camera.view_proj_matrix().inverse();
                    let cursor =
                        glam::Vec2::new(state.last_cursor.x as f32, state.last_cursor.y as f32);
                    let (ray_origin, ray_dir) =
                        viewport_lib::picking::screen_to_ray(cursor, glam::Vec2::new(w, h), vp_inv);

                    let mut mesh_lookup = std::collections::HashMap::new();
                    for node in state.adv_scene.nodes() {
                        if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                            mesh_lookup.entry(mid).or_insert_with(|| {
                                (
                                    state.box_mesh_data.positions.clone(),
                                    state.box_mesh_data.indices.clone(),
                                )
                            });
                        }
                    }

                    let hit = viewport_lib::picking::pick_scene_nodes(
                        ray_origin,
                        ray_dir,
                        &state.adv_scene,
                        &mesh_lookup,
                    );

                    if let Some(hit) = hit {
                        let id = hit.id;
                        if state.shift_held {
                            state.adv_selection.toggle(id);
                        } else {
                            state.adv_selection.select_one(id);
                        }
                    } else if !state.shift_held {
                        state.adv_selection.clear();
                    }
                    state.update_title();
                }

                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let dx = (position.x - state.last_cursor.x) as f32;
                let dy = (position.y - state.last_cursor.y) as f32;
                state.last_cursor = position;

                // Mark as drag if the left button is held and has moved more than 3px total.
                if state.left_pressed && (dx * dx + dy * dy) > 9.0 {
                    state.left_drag_active = true;
                }

                let any_drag = state.left_pressed || state.middle_pressed || state.right_pressed;
                if !any_drag || (dx.abs() < 0.001 && dy.abs() < 0.001) {
                    return;
                }

                let is_pan = state.right_pressed || (state.middle_pressed && state.shift_held);

                if state.mode == ShowcaseMode::Interaction {
                    // If dragging a gizmo handle, apply transform instead of orbiting.
                    if state.interact_gizmo.active_axis != GizmoAxis::None {
                        if let Some(center) = state.interact_gizmo_center {
                            let drag_delta = glam::Vec2::new(dx, dy);
                            let w = state.surface_config.width as f32;
                            let h = state.surface_config.height as f32;
                            let viewport_size = glam::Vec2::new(w, h);
                            let vp = state.camera.view_proj_matrix();
                            let view = state.camera.view_matrix();
                            let axis = state.interact_gizmo.active_axis;

                            let orient = match state.interact_gizmo.space {
                                GizmoSpace::World => glam::Quat::IDENTITY,
                                GizmoSpace::Local => state
                                    .interact_selection
                                    .primary()
                                    .and_then(|id| state.interact_scene.node(id))
                                    .map(|n| glam::Quat::from_mat4(&n.world_transform()))
                                    .unwrap_or(glam::Quat::IDENTITY),
                            };

                            let axis_dir = |a: GizmoAxis| -> glam::Vec3 {
                                let base = match a {
                                    GizmoAxis::X => glam::Vec3::X,
                                    GizmoAxis::Y => glam::Vec3::Y,
                                    GizmoAxis::Z => glam::Vec3::Z,
                                    _ => glam::Vec3::X,
                                };
                                orient * base
                            };

                            match state.interact_gizmo.mode {
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
                                            state.camera.right(),
                                            state.camera.up(),
                                            vp,
                                            center,
                                            viewport_size,
                                        ),
                                        _ => glam::Vec3::ZERO,
                                    };
                                    let snap_delta =
                                        if let Some(inc) = state.interact_snap.translation {
                                            // Accumulate raw delta, snap the cumulative total,
                                            // and apply only the change since the last snapped value.
                                            state.interact_drag_accum_translation += delta;
                                            let snapped_total = viewport_lib::snap::snap_vec3(
                                                state.interact_drag_accum_translation,
                                                inc,
                                            );
                                            let step = snapped_total
                                                - state.interact_drag_last_snapped_translation;
                                            state.interact_drag_last_snapped_translation =
                                                snapped_total;
                                            step
                                        } else {
                                            delta
                                        };
                                    // Apply translation to all selected nodes.
                                    for id in
                                        state.interact_selection.iter().copied().collect::<Vec<_>>()
                                    {
                                        if let Some(node) = state.interact_scene.node(id) {
                                            let cur = node.local_transform();
                                            let new_t =
                                                glam::Mat4::from_translation(snap_delta) * cur;
                                            state.interact_scene.set_local_transform(id, new_t);
                                        }
                                    }
                                    state.interact_scene.update_transforms();
                                }
                                GizmoMode::Rotate => {
                                    let angle = match axis {
                                        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                                            let dir = axis_dir(axis);
                                            gizmo::project_drag_onto_rotation(drag_delta, dir, view)
                                        }
                                        _ => 0.0,
                                    };
                                    let snap_angle = if let Some(inc) = state.interact_snap.rotation
                                    {
                                        // Accumulate raw angle, snap the cumulative total,
                                        // and apply only the change since the last snapped value.
                                        state.interact_drag_accum_rotation += angle;
                                        let snapped_total = viewport_lib::snap::snap_angle(
                                            state.interact_drag_accum_rotation,
                                            inc,
                                        );
                                        let step = snapped_total
                                            - state.interact_drag_last_snapped_rotation;
                                        state.interact_drag_last_snapped_rotation = snapped_total;
                                        step
                                    } else {
                                        angle
                                    };
                                    if snap_angle.abs() > 1e-6 {
                                        let rot_axis = axis_dir(axis);
                                        let rot = glam::Quat::from_axis_angle(rot_axis, snap_angle);
                                        for id in state
                                            .interact_selection
                                            .iter()
                                            .copied()
                                            .collect::<Vec<_>>()
                                        {
                                            if let Some(node) = state.interact_scene.node(id) {
                                                let cur = node.local_transform();
                                                let pos = cur.w_axis.truncate();
                                                // Rotate around the gizmo center.
                                                let to_origin =
                                                    glam::Mat4::from_translation(-center);
                                                let from_origin =
                                                    glam::Mat4::from_translation(center);
                                                let rot_mat = glam::Mat4::from_quat(rot);
                                                let new_t = from_origin * rot_mat * to_origin * cur;
                                                let _ = pos; // suppress unused
                                                state.interact_scene.set_local_transform(id, new_t);
                                            }
                                        }
                                        state.interact_scene.update_transforms();
                                    }
                                }
                                GizmoMode::Scale => {
                                    let amount = match axis {
                                        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                                            let dir = axis_dir(axis);
                                            gizmo::project_drag_onto_axis(
                                                drag_delta,
                                                dir,
                                                vp,
                                                center,
                                                viewport_size,
                                            )
                                        }
                                        _ => 0.0,
                                    };
                                    if amount.abs() > 1e-6 {
                                        let scale_vec = match axis {
                                            GizmoAxis::X => glam::Vec3::new(1.0 + amount, 1.0, 1.0),
                                            GizmoAxis::Y => glam::Vec3::new(1.0, 1.0 + amount, 1.0),
                                            GizmoAxis::Z => glam::Vec3::new(1.0, 1.0, 1.0 + amount),
                                            _ => glam::Vec3::ONE,
                                        };
                                        for id in state
                                            .interact_selection
                                            .iter()
                                            .copied()
                                            .collect::<Vec<_>>()
                                        {
                                            if let Some(node) = state.interact_scene.node(id) {
                                                let cur = node.local_transform();
                                                let scale_mat = glam::Mat4::from_scale(scale_vec);
                                                let new_t = cur * scale_mat;
                                                state.interact_scene.set_local_transform(id, new_t);
                                            }
                                        }
                                        state.interact_scene.update_transforms();
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else if is_pan {
                        // Route through CameraAnimator for damped motion.
                        let cam = &state.camera;
                        let viewport_h = state.surface_config.height as f32;
                        let pan_scale = 2.0 * cam.distance * (cam.fov_y / 2.0).tan() / viewport_h;
                        state
                            .interact_animator
                            .apply_pan(dx * pan_scale * 0.4, -dy * pan_scale * 0.4);
                    } else {
                        state
                            .interact_animator
                            .apply_orbit(dx * ANIM_ORBIT_SENSITIVITY, dy * ANIM_ORBIT_SENSITIVITY);
                    }
                } else if is_pan {
                    let viewport_h = state.surface_config.height as f32;
                    state.camera.pan_pixels(glam::vec2(dx, dy), viewport_h);
                } else {
                    state
                        .camera
                        .orbit(dx * ORBIT_SENSITIVITY, dy * ORBIT_SENSITIVITY);
                }

                state.window.request_redraw();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let (scroll_x, scroll_y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x * 28.0, y * 28.0),
                    MouseScrollDelta::PixelDelta(px) => (px.x as f32, px.y as f32),
                };
                if state.ctrl_held {
                    // Ctrl + Scroll → orbit (two-axis, matching the controller preset).
                    if state.mode == ShowcaseMode::Interaction {
                        state.interact_animator.apply_orbit(
                            scroll_x * ANIM_ORBIT_SENSITIVITY,
                            scroll_y * ANIM_ORBIT_SENSITIVITY,
                        );
                    } else {
                        state
                            .camera
                            .orbit(scroll_x * ORBIT_SENSITIVITY, scroll_y * ORBIT_SENSITIVITY);
                    }
                } else {
                    if state.mode == ShowcaseMode::Interaction {
                        let zoom_delta = -scroll_y * ANIM_ZOOM_SENSITIVITY * state.camera.distance;
                        state.interact_animator.apply_zoom(zoom_delta);
                    } else {
                        state
                            .camera
                            .zoom_by_factor(1.0 - scroll_y * ZOOM_SENSITIVITY);
                    }
                }
                state.window.request_redraw();
            }

            WindowEvent::RotationGesture { delta, .. } => {
                let angle = delta.to_radians();
                if state.mode == ShowcaseMode::Interaction {
                    state.interact_animator.apply_orbit(angle, 0.0);
                } else {
                    state.camera.orbit(angle, 0.0);
                }
                state.window.request_redraw();
            }

            // --- Render ---
            WindowEvent::RedrawRequested => {
                let frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                    Err(e) => {
                        eprintln!("Surface error: {e:?}");
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let w = state.surface_config.width as f32;
                let h = state.surface_config.height as f32;

                state.camera.set_aspect_ratio(w, h);

                let mut adv_clip_planes: Vec<ClipPlane> = vec![];
                let mut adv_outline = false;
                let mut adv_xray = false;

                let (scene_items, bg_color, lighting) = match state.mode {
                    ShowcaseMode::Basic => {
                        // Original 4-box grid.
                        let positions = [
                            [-1.5, 0.0, -1.5],
                            [1.5, 0.0, -1.5],
                            [-1.5, 0.0, 1.5],
                            [1.5, 0.0, 1.5],
                        ];
                        let items: Vec<SceneRenderItem> = state
                            .mesh_indices
                            .iter()
                            .zip(&positions)
                            .map(|(&mesh_index, pos)| {
                                let model = glam::Mat4::from_translation(glam::Vec3::from(*pos));
                                {
                                    let mut item = SceneRenderItem::default();
                                    item.mesh_index = mesh_index;
                                    item.model = model.to_cols_array_2d();
                                    item
                                }
                            })
                            .collect();

                        let lighting = if state.use_point_light {
                            LightingSettings {
                                lights: vec![LightSource {
                                    kind: LightKind::Point {
                                        position: [5.0, 5.0, 5.0],
                                        range: 30.0,
                                    },
                                    ..LightSource::default()
                                }],
                                ..LightingSettings::default()
                            }
                        } else {
                            LightingSettings::default()
                        };

                        (items, Some(BG_COLOR), lighting)
                    }
                    ShowcaseMode::SceneGraph => {
                        // Collect from Scene.
                        let items = state.scene.collect_render_items(&state.selection);

                        let bg = background_color(state.bg_cycle);

                        let lighting = LightingSettings {
                            hemisphere_intensity: if state.hemisphere_on { 0.3 } else { 0.0 },
                            ..LightingSettings::default()
                        };

                        (items, Some(bg), lighting)
                    }
                    ShowcaseMode::Performance => {
                        // Cache scene items keyed on (scene version, selection version) to avoid
                        // O(N) Vec allocation every frame for the static 1M-item scene. The
                        // renderer's generation cache then skips the sort+instance-buffer rebuild
                        // on static frames, making orbit fast after the initial build.
                        let current_ver =
                            (state.perf_scene.version(), state.perf_selection.version());
                        if current_ver != state.perf_scene_items_version {
                            state.perf_scene_items_cache =
                                state.perf_scene.collect_render_items(&state.perf_selection);
                            state.perf_scene_items_version = current_ver;
                        }
                        let items = state.perf_scene_items_cache.clone();

                        let lighting = LightingSettings::default();
                        (items, Some(BG_COLOR), lighting)
                    }
                    ShowcaseMode::Advanced => {
                        let items = state.adv_scene.collect_render_items(&state.adv_selection);
                        if state.adv_clip_enabled {
                            adv_clip_planes.push(ClipPlane {
                                normal: [1.0, 0.0, 0.0],
                                distance: 0.0,
                                enabled: true,
                                cap_color: None,
                            });
                        }
                        adv_outline = state.adv_outline_on && !state.adv_selection.is_empty();
                        adv_xray = state.adv_xray_on && !state.adv_selection.is_empty();
                        (items, Some(BG_COLOR), LightingSettings::default())
                    }
                    ShowcaseMode::PostProcess => {
                        let items = state.pp_scene.collect_render_items(&Selection::new());
                        (
                            items,
                            Some(BG_COLOR),
                            LightingSettings {
                                lights: vec![
                                    LightSource {
                                        kind: LightKind::Directional {
                                            direction: [0.5, 1.0, 0.3],
                                        },
                                        intensity: 2.0,
                                        ..LightSource::default()
                                    },
                                    LightSource {
                                        kind: LightKind::Point {
                                            position: [3.0, 3.0, 3.0],
                                            range: 15.0,
                                        },
                                        color: [1.0, 0.9, 0.7],
                                        intensity: 3.0,
                                        ..LightSource::default()
                                    },
                                ],
                                shadows_enabled: true,
                                shadow_extent_override: Some(7.0),
                                ..LightingSettings::default()
                            },
                        )
                    }
                    ShowcaseMode::NormalMaps => {
                        let items = state.nm_scene.collect_render_items(&Selection::new());
                        if state.nm_clip_enabled {
                            adv_clip_planes.push(ClipPlane {
                                normal: [1.0, 0.0, 0.0],
                                distance: 0.0,
                                enabled: true,
                                cap_color: None,
                            });
                        }
                        (
                            items,
                            Some(BG_COLOR),
                            LightingSettings {
                                lights: vec![
                                    LightSource {
                                        kind: LightKind::Directional {
                                            direction: [0.4, 1.0, 0.3],
                                        },
                                        intensity: 1.5,
                                        ..LightSource::default()
                                    },
                                    LightSource {
                                        kind: LightKind::Point {
                                            position: [2.0, 2.0, 2.0],
                                            range: 10.0,
                                        },
                                        color: [1.0, 0.95, 0.9],
                                        intensity: 2.0,
                                        ..LightSource::default()
                                    },
                                ],
                                shadows_enabled: false,
                                ..LightingSettings::default()
                            },
                        )
                    }
                    ShowcaseMode::Shadows => {
                        let items = state.shd_scene.collect_render_items(&Selection::new());
                        (
                            items,
                            Some(BG_COLOR),
                            LightingSettings {
                                lights: vec![LightSource {
                                    kind: LightKind::Directional {
                                        direction: [0.5, 1.0, 0.3],
                                    },
                                    intensity: 2.0,
                                    ..LightSource::default()
                                }],
                                shadows_enabled: true,
                                shadow_cascade_count: state.shd_cascade_count,
                                shadow_filter: if state.shd_pcss_on {
                                    ShadowFilter::Pcss
                                } else {
                                    ShadowFilter::Pcf
                                },
                                ..LightingSettings::default()
                            },
                        )
                    }
                    ShowcaseMode::Annotation => {
                        // Collect marker boxes from the annotation scene.
                        let items = state.ann_scene.collect_render_items(&Selection::new());
                        // Labels are projected via world_to_screen in update_title each frame.
                        (items, Some(BG_COLOR), LightingSettings::default())
                    }
                    ShowcaseMode::Interaction => {
                        // Advance the camera animator.
                        let now = std::time::Instant::now();
                        let dt = state
                            .last_instant
                            .map(|prev| now.duration_since(prev).as_secs_f32())
                            .unwrap_or(1.0 / 60.0)
                            .min(1.0 / 30.0); // Clamp so idle gaps don't skip entire animations.
                        state.last_instant = Some(now);
                        state.interact_animator.update(dt, &mut state.camera);

                        let items = state
                            .interact_scene
                            .collect_render_items(&state.interact_selection);
                        let lighting = LightingSettings::default();
                        (items, Some(BG_COLOR), lighting)
                    }
                };

                let clear_color = bg_color.unwrap_or(BG_COLOR);

                // Compute gizmo model matrix for Interaction mode.
                let (gizmo_model, gizmo_mode, gizmo_space_orient) =
                    if state.mode == ShowcaseMode::Interaction {
                        let center =
                            gizmo::gizmo_center_from_selection(&state.interact_selection, |id| {
                                state.interact_scene.node(id).map(|n| {
                                    let t = n.world_transform();
                                    glam::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z)
                                })
                            });
                        // Cache gizmo center + scale for hit-testing in mouse events.
                        state.interact_gizmo_center = center;
                        let model = center.map(|c| {
                            let scale = compute_gizmo_scale(
                                c,
                                state.camera.eye_position(),
                                state.camera.fov_y,
                                h,
                            );
                            state.interact_gizmo_scale = scale;
                            glam::Mat4::from_scale_rotation_translation(
                                glam::Vec3::splat(scale),
                                glam::Quat::IDENTITY,
                                c,
                            )
                        });
                        let orient = match state.interact_gizmo.space {
                            GizmoSpace::World => glam::Quat::IDENTITY,
                            GizmoSpace::Local => {
                                // Use primary selection's orientation.
                                state
                                    .interact_selection
                                    .primary()
                                    .and_then(|id| state.interact_scene.node(id))
                                    .map(|n| glam::Quat::from_mat4(&n.world_transform()))
                                    .unwrap_or(glam::Quat::IDENTITY)
                            }
                        };
                        (model, state.interact_gizmo.mode, orient)
                    } else {
                        (None, GizmoMode::Translate, glam::Quat::IDENTITY)
                    };

                // Compute generation counters for the active scene and selection.
                // These allow the renderer to skip batch rebuild and GPU upload on static frames.
                let (scene_gen, sel_gen) = match state.mode {
                    ShowcaseMode::Basic => (0u64, 0u64),
                    ShowcaseMode::SceneGraph => (state.scene.version(), state.selection.version()),
                    ShowcaseMode::Performance => {
                        (state.perf_scene.version(), state.perf_selection.version())
                    }
                    ShowcaseMode::Advanced => {
                        (state.adv_scene.version(), state.adv_selection.version())
                    }
                    ShowcaseMode::PostProcess => (state.pp_scene.version(), 0u64),
                    ShowcaseMode::NormalMaps => (state.nm_scene.version(), 0u64),
                    ShowcaseMode::Shadows => (state.shd_scene.version(), 0u64),
                    ShowcaseMode::Annotation => (state.ann_scene.version(), 0u64),
                    ShowcaseMode::Interaction => (
                        state.interact_scene.version(),
                        state.interact_selection.version(),
                    ),
                };

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(scene_items),
                );
                frame_data.effects.lighting = lighting;
                frame_data.viewport.wireframe_mode = false;
                frame_data.interaction.gizmo_model = gizmo_model;
                frame_data.interaction.gizmo_mode = gizmo_mode;
                frame_data.interaction.gizmo_hovered =
                    if state.interact_gizmo.active_axis != GizmoAxis::None {
                        state.interact_gizmo.active_axis
                    } else {
                        state.interact_gizmo.hovered_axis
                    };
                frame_data.interaction.gizmo_space_orientation = gizmo_space_orient;
                frame_data.viewport.show_grid = true;
                frame_data.viewport.show_axes_indicator = true;
                frame_data.viewport.background_color = bg_color;
                frame_data.effects.clip_planes = adv_clip_planes;
                frame_data.interaction.outline_selected = adv_outline;
                frame_data.interaction.xray_selected = adv_xray;
                frame_data.scene.generation = scene_gen;
                frame_data.interaction.selection_generation = sel_gen;

                if matches!(
                    state.mode,
                    ShowcaseMode::PostProcess | ShowcaseMode::NormalMaps | ShowcaseMode::Shadows
                ) {
                    // HDR path: use render() which handles the full post-processing pipeline.
                    let mut fd = frame_data;
                    // Only Shadows mode uses CSM (needs clamped far plane for better cascade
                    // distribution). PostProcess and NormalMaps keep the default RenderCamera.
                    if state.mode == ShowcaseMode::Shadows {
                        // Cap far at 60m so cascade splits land within the ~20m scene
                        // (default zfar=1000 would push all 4 cascades to 62/127/230/1000m,
                        // wasting resolution and leaving the whole scene in cascade 0).
                        let mut rc = RenderCamera::from_camera(&state.camera);
                        rc.far = state.camera.zfar.min(60.0);
                        rc.projection =
                            glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                        fd.camera.render_camera = rc;
                    }
                    fd.effects.post_process = if state.mode == ShowcaseMode::PostProcess {
                        PostProcessSettings {
                            enabled: true,
                            tone_mapping: state.pp_tone_mapping,
                            exposure: state.pp_exposure,
                            bloom: state.pp_bloom,
                            bloom_threshold: 0.6,
                            bloom_intensity: 1.0,
                            ssao: state.pp_ssao,
                            fxaa: state.pp_fxaa,
                            ..PostProcessSettings::default()
                        }
                    } else if state.mode == ShowcaseMode::Shadows {
                        PostProcessSettings {
                            enabled: true,
                            tone_mapping: ToneMapping::Aces,
                            exposure: 1.0,
                            contact_shadows: state.shd_contact_on,
                            contact_shadow_max_distance: 0.18,
                            contact_shadow_steps: 32,
                            contact_shadow_thickness: 0.04,
                            ..PostProcessSettings::default()
                        }
                    } else {
                        // Showcase 7: HDR enabled for correct PBR lighting; no bloom/SSAO/FXAA.
                        PostProcessSettings {
                            enabled: true,
                            tone_mapping: ToneMapping::Aces,
                            exposure: 1.0,
                            ..PostProcessSettings::default()
                        }
                    };
                    let cmd = state
                        .renderer
                        .render(&state.device, &state.queue, &view, &fd);
                    state.queue.submit(std::iter::once(cmd));
                    frame.present();
                } else {
                    state
                        .renderer
                        .prepare(&state.device, &state.queue, &frame_data);

                    let mut encoder =
                        state
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("showcase_encoder"),
                            });

                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("showcase_render_pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &state.msaa_view,
                                    resolve_target: Some(&view),
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: clear_color[0] as f64,
                                            g: clear_color[1] as f64,
                                            b: clear_color[2] as f64,
                                            a: clear_color[3] as f64,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: &state.depth_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: wgpu::StoreOp::Discard,
                                        }),
                                        stencil_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(0),
                                            store: wgpu::StoreOp::Store,
                                        }),
                                    },
                                ),
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });

                        render_pass.set_viewport(0.0, 0.0, w, h, 0.0, 1.0);
                        state.renderer.paint_to(&mut render_pass, &frame_data);
                    }

                    state.queue.submit(std::iter::once(encoder.finish()));
                    frame.present();
                }

                // Update stats in title bar for Performance mode.
                if state.mode == ShowcaseMode::Performance {
                    state.last_stats = state.renderer.last_frame_stats();
                    state.update_title();
                }

                // Update title each frame for Annotation mode so projected screen
                // coordinates in the title bar track camera orbit in real time.
                if state.mode == ShowcaseMode::Annotation {
                    state.update_title();
                }

                // Request continuous redraws while the camera animator is active.
                if state.mode == ShowcaseMode::Interaction && state.interact_animator.is_animating()
                {
                    state.update_title();
                    state.window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Box mesh helper
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
