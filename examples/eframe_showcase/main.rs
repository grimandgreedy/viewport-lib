//! Feature showcase for `viewport-lib` using `eframe` / `egui`.

use std::collections::HashMap;

use eframe::egui;
use viewport_lib::{
    Action, AttributeKind, AttributeRef, BackfacePolicy, BuiltinColourmap, ButtonState, Camera,
    CameraAnimator, CameraFrame, CellSelectionInfo, ClipObject, ColourmapId, FrameData,
    GaussianSplatItem, GizmoAxis, GizmoInfo, GizmoMode, GlyphItem, GlyphType, GpuImplicitItem,
    GpuImplicitOptions, GpuMarchingCubesJob, GroundPlane, GroundPlaneMode, ImageAnchor,
    ImplicitBlendMode, ImplicitPrimitive, LightKind, LightSource, LightingSettings, ManipResult,
    ManipulationContext, MeshData, MeshId, OrbitCameraController, PickId, PointCloudItem,
    PolylineItem, PolylineSelectionInfo, PostProcessSettings, RenderCamera, RibbonItem,
    RuntimeMode, SceneFrame, SceneRenderItem, ScreenImageItem, ScrollUnits, Selection,
    ShadowFilter, SpriteItem, StreamtubeItem, SubSelectionRef, TensorGlyphItem,
    TransparentVolumeMeshItem, TubeItem, ViewportContext, ViewportEvent, ViewportRenderer,
    VolumeMeshItem, VolumeSurfaceSliceItem,
    geometry::isoline::IsolineItem,
    gizmo::{self, compute_gizmo_scale},
    scene::Scene,
};

use viewport_lib::{LoadingBarAnchor, LoadingBarItem};

mod geometry;
mod gizmo_helpers;
mod multi_viewport_callback;
mod shared;
mod showcase_01_basic;
mod showcase_02_scene_graph;
mod showcase_03_performance;
mod showcase_04_interaction;
mod showcase_05_materials_and_visibility;
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
mod showcase_24_backface_policy;
mod showcase_25_surface_vectors;
mod showcase_26_volume_mesh;
mod showcase_27_camera_framing;
mod showcase_28_curve_network_quantities;
mod showcase_29_depth_composite_images;
mod showcase_30_implicit_surface;
mod showcase_31_sparse_volume_grid;
mod showcase_32_extended_quantities;
mod showcase_33_picking_levels;
mod showcase_34_labels;
mod showcase_35_overlay;
mod showcase_36_playback_runtime;
mod showcase_37_probe_widgets;
mod showcase_38_surface_lic;
mod showcase_39_tensor_glyphs;
mod showcase_40_vertex_warp;
mod showcase_41_sprites;
mod showcase_42_gaussian_splats;
mod viewport_callback;

const BG_COLOUR: [f32; 4] = [0.22, 0.22, 0.24, 1.0];

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
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                    eframe::egui_wgpu::WgpuSetupCreateNew {
                        // Request INDIRECT_FIRST_INSTANCE when the adapter supports it so
                        // GPU-driven culling is available on Metal (Apple Silicon) and Vulkan.
                        device_descriptor: std::sync::Arc::new(|adapter| {
                            use eframe::wgpu;
                            let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                                wgpu::Limits::downlevel_webgl2_defaults()
                            } else {
                                wgpu::Limits::default()
                            };
                            wgpu::DeviceDescriptor {
                                label: Some("viewport-lib showcase device"),
                                required_features: adapter.features()
                                    & wgpu::Features::INDIRECT_FIRST_INSTANCE,
                                required_limits: wgpu::Limits {
                                    max_texture_dimension_2d: 8192,
                                    ..base_limits
                                },
                                ..Default::default()
                            }
                        }),
                        ..Default::default()
                    },
                ),
                ..Default::default()
            },
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

            let box_mesh = viewport_lib::primitives::cube(1.0);

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
                mode_gen: 0,
                show_keybinds: false,
                basic_state: showcase_01_basic::BasicState::default(),
                sg_state: showcase_02_scene_graph::SgState::default(),
                box_mesh_data: box_mesh,
                perf_state: showcase_03_performance::PerfState::default(),
                interact_state: showcase_04_interaction::InteractState::default(),
                materials_visibility_state:
                    showcase_05_materials_and_visibility::MaterialsVisibilityState::default(),
                pp_state: showcase_06_post_process::PostProcessState::default(),
                nm_state: showcase_07_normal_maps::NormalMapsState::default(),
                shd_state: showcase_08_shadows::ShadowsState::default(),
                ann_state: showcase_09_annotation::AnnotationState::default(),
                ct_state: showcase_10_camera_tools::CameraToolsState::default(),
                cam_animator: CameraAnimator::with_default_damping(),
                lights_state: showcase_11_lights::LightsState::default(),
                scalar_state: showcase_12_scalar_fields::ScalarFieldsState::default(),
                mv_state: showcase_13_multi_viewport::MvState::default(),
                iso_state: showcase_14_isolines::IsolinesState::default(),
                pc_state: showcase_15_point_clouds::PointCloudsState::default(),
                stream_state: showcase_16_streamlines::StreamlinesState::default(),
                vol_state: showcase_17_volume::VolumeState::default(),
                clipvol_state: showcase_18_clip_volumes::ClipVolState::default(),
                matcap_state: showcase_19_matcap::MatcapState::default(),

                face_state: showcase_20_face_attributes::FaceAttrState::default(),

                texture_state: showcase_21_textures::TextureState::default(),

                param_vis_state: showcase_22_parameterization::ParamVisState::default(),

                gp_state: showcase_23_ground_plane::GroundPlaneState::default(),

                sa_state: showcase_24_backface_policy::SaState::default(),

                sv_state: showcase_25_surface_vectors::SvState::default(),

                vm_state: showcase_26_volume_mesh::VmState::default(),

                cnq_state: showcase_28_curve_network_quantities::CnqState::default(),

                dc_state: showcase_29_depth_composite_images::DcState::default(),

                is_state: showcase_30_implicit_surface::IsState::default(),

                eq_state: showcase_32_extended_quantities::EqState::default(),

                svg_state: showcase_31_sparse_volume_grid::SvgState::default(),

                aux_state: showcase_27_camera_framing::AuxState::default(),

                pl_state: showcase_33_picking_levels::PlState::default(),
                lbl_state: showcase_34_labels::LblState::default(),

                ovl_state: showcase_35_overlay::OvlState::default(),

                pb_state: showcase_36_playback_runtime::PbState::default(),

                pw_state: showcase_37_probe_widgets::ProbeWidgetState::new(),

                lic_state: showcase_38_surface_lic::LicState::default(),

                tg_state: showcase_39_tensor_glyphs::TensorGlyphState::default(),

                warp_state: showcase_40_vertex_warp::VertexWarpState::default(),
                sprite_state: showcase_41_sprites::SpriteState::default(),
                splat_state: showcase_42_gaussian_splats::GaussianSplatsState::default(),
            }))
        }),
    )
}

use showcase_27_camera_framing::AuxSubMode;

// ---------------------------------------------------------------------------
// Showcase mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShowcaseMode {
    Basic,
    SceneGraph,
    Performance,
    Interaction,
    MaterialsVisibility,
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
    BackfacePolicy,
    SurfaceVectors,
    VolumeMesh,
    Auxiliary,
    CurveNetworkQuantities,
    DepthCompositeImages,
    ImplicitSurface,
    SparseVolumeGrid,
    ExtendedQuantities,
    PickLevels,
    Labels,
    Overlay,
    PlaybackRuntime,
    ProbeWidgets,
    SurfaceLIC,
    TensorGlyphs,
    VertexWarp,
    Sprites,
    GaussianSplats,
}

impl ShowcaseMode {
    fn label(self) -> &'static str {
        match self {
            Self::Basic => "1: Rendering Basics",
            Self::SceneGraph => "2: Scene Graph",
            Self::Performance => "3: Performance",
            Self::Interaction => "4: Interaction",
            Self::MaterialsVisibility => "5: Materials and Visibility",
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
            Self::BackfacePolicy => "24: Backface Policy",
            Self::SurfaceVectors => "25: Surface Vectors",
            Self::VolumeMesh => "26: Volume Meshes",
            Self::Auxiliary => "27: Camera Framing & HUD",
            Self::CurveNetworkQuantities => "28: Curve Network Quantities",
            Self::DepthCompositeImages => "29: Depth-Composited Images",
            Self::ImplicitSurface => "30: Implicit Surfaces",
            Self::SparseVolumeGrid => "31: Sparse Volume Grid",
            Self::ExtendedQuantities => "32: Extended Quantities",
            Self::PickLevels => "33: Picking Levels",
            Self::Labels => "34: Labels",
            Self::Overlay => "35: Overlay Composition",
            Self::PlaybackRuntime => "36: Playback Runtime Control",
            Self::ProbeWidgets => "37: Probe Widgets",
            Self::SurfaceLIC => "38: Surface LIC",
            Self::TensorGlyphs => "39: Tensor Glyphs",
            Self::VertexWarp => "40: GPU Vertex Warp",
            Self::Sprites => "41: Sprites & Particles",
            Self::GaussianSplats => "42: Gaussian Splats",
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
    /// Smooth camera animator used by CameraTools and Auxiliary showcases.
    cam_animator: CameraAnimator,
    mode: ShowcaseMode,
    /// Increments on every mode switch; used as scene generation for showcases
    /// that build items manually (no Scene struct) so the renderer's instance
    /// cache is always invalidated when changing showcases.
    mode_gen: u64,
    show_keybinds: bool,

    // --- Showcase 1 ---
    pub(crate) basic_state: showcase_01_basic::BasicState,

    // --- Showcase 2 ---
    pub(crate) sg_state: showcase_02_scene_graph::SgState,
    /// Shared box MeshData for on-demand uploads in later showcases.
    pub(crate) box_mesh_data: MeshData,

    // --- Showcase 3 ---
    pub(crate) perf_state: showcase_03_performance::PerfState,

    // --- Showcase 4 ---
    pub(crate) interact_state: showcase_04_interaction::InteractState,

    // --- Showcase 5 ---
    pub(crate) materials_visibility_state:
        showcase_05_materials_and_visibility::MaterialsVisibilityState,

    // --- Showcase 6 ---
    pub(crate) pp_state: showcase_06_post_process::PostProcessState,

    // --- Showcase 7 ---
    pub(crate) nm_state: showcase_07_normal_maps::NormalMapsState,

    // --- Showcase 8 ---
    pub(crate) shd_state: showcase_08_shadows::ShadowsState,

    // --- Showcase 9 ---
    pub(crate) ann_state: showcase_09_annotation::AnnotationState,

    // --- Showcase 10 ---
    pub(crate) ct_state: showcase_10_camera_tools::CameraToolsState,

    // --- Showcase 11 ---
    pub(crate) lights_state: showcase_11_lights::LightsState,

    // --- Showcase 12 ---
    pub(crate) scalar_state: showcase_12_scalar_fields::ScalarFieldsState,

    // --- Showcase 13 ---
    pub(crate) mv_state: showcase_13_multi_viewport::MvState,

    // --- Showcase 14 ---
    pub(crate) iso_state: showcase_14_isolines::IsolinesState,

    // --- Showcase 15 ---
    pub(crate) pc_state: showcase_15_point_clouds::PointCloudsState,

    // --- Showcase 16 ---
    pub(crate) stream_state: showcase_16_streamlines::StreamlinesState,

    // --- Showcase 17 ---
    pub(crate) vol_state: showcase_17_volume::VolumeState,

    // --- Showcase 18 ---
    pub(crate) clipvol_state: showcase_18_clip_volumes::ClipVolState,

    // --- Showcase 19 ---
    pub(crate) matcap_state: showcase_19_matcap::MatcapState,

    // --- Showcase 20 ---
    pub(crate) face_state: showcase_20_face_attributes::FaceAttrState,

    // --- Showcase 21 ---
    pub(crate) texture_state: showcase_21_textures::TextureState,

    // --- Showcase 22 ---
    pub(crate) param_vis_state: showcase_22_parameterization::ParamVisState,
    // --- Showcase 23 ---
    pub(crate) gp_state: showcase_23_ground_plane::GroundPlaneState,

    // --- Showcase 24 ---
    pub(crate) sa_state: showcase_24_backface_policy::SaState,

    // --- Showcase 25 ---
    pub(crate) sv_state: showcase_25_surface_vectors::SvState,

    // --- Showcase 26 ---
    pub(crate) vm_state: showcase_26_volume_mesh::VmState,

    // --- Showcase 28 ---
    pub(crate) cnq_state: showcase_28_curve_network_quantities::CnqState,

    // --- Showcase 29 ---
    dc_state: showcase_29_depth_composite_images::DcState,

    // --- Showcase 30 ---
    is_state: showcase_30_implicit_surface::IsState,

    // --- Showcase 31 ---
    pub(crate) svg_state: showcase_31_sparse_volume_grid::SvgState,

    // --- Showcase 27 ---
    pub(crate) aux_state: showcase_27_camera_framing::AuxState,

    // --- Showcase 32 ---
    pub(crate) eq_state: showcase_32_extended_quantities::EqState,

    // --- Showcase 33 ---
    pub(crate) pl_state: showcase_33_picking_levels::PlState,

    // --- Showcase 34 ---
    pub(crate) lbl_state: showcase_34_labels::LblState,

    // --- Showcase 35 ---
    pub(crate) ovl_state: showcase_35_overlay::OvlState,

    // --- Showcase 36 ---
    pub(crate) pb_state: showcase_36_playback_runtime::PbState,

    // --- Showcase 37 ---
    pub(crate) pw_state: showcase_37_probe_widgets::ProbeWidgetState,

    // --- Showcase 38 ---
    pub(crate) lic_state: showcase_38_surface_lic::LicState,

    // --- Showcase 39 ---
    pub(crate) tg_state: showcase_39_tensor_glyphs::TensorGlyphState,

    // --- Showcase 40 ---
    pub(crate) warp_state: showcase_40_vertex_warp::VertexWarpState,

    // --- Showcase 41 ---
    pub(crate) sprite_state: showcase_41_sprites::SpriteState,

    // --- Showcase 42 ---
    pub(crate) splat_state: showcase_42_gaussian_splats::GaussianSplatsState,
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut cycle_dir = 0_i32;
        let mut toggle_keybinds = false;
        let mut tab_pressed = false;
        let mut escape_pressed = false;
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
                        let use_cycle = (modifiers.ctrl || modifiers.command) && !modifiers.alt;
                        match key {
                            egui::Key::OpenBracket if use_cycle => cycle_dir = -1,
                            egui::Key::CloseBracket if use_cycle => cycle_dir = 1,
                            egui::Key::Tab => tab_pressed = true,
                            egui::Key::Escape => escape_pressed = true,
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
        if escape_pressed && self.mode == ShowcaseMode::Auxiliary {
            self.aux_state.active_frustum = None;
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

        // Poll for a completed async perf scene build.
        let completed = self
            .perf_state
            .build_rx
            .as_ref()
            .and_then(|rx: &std::sync::mpsc::Receiver<_>| rx.try_recv().ok());
        if let Some((scene, pick_acc)) = completed {
            self.perf_state.scene = scene;
            self.perf_state.pick_accelerator = Some(pick_acc);
            // Pre-warm items cache so the first rendered frame has no stall.
            self.perf_state.scene_items_cache = std::sync::Arc::from(
                self.perf_state
                    .scene
                    .collect_render_items(&self.perf_state.selection),
            );
            self.perf_state.scene_items_version = (
                self.perf_state.scene.version(),
                self.perf_state.selection.version(),
            );
            self.perf_state.total_objects = 1_000_000;
            self.perf_state.build_rx = None;
            self.perf_state.build_progress = None;
            self.perf_state.built = true;
        }

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
                    ShowcaseMode::MaterialsVisibility,
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
                    ShowcaseMode::BackfacePolicy,
                    ShowcaseMode::SurfaceVectors,
                    ShowcaseMode::VolumeMesh,
                    ShowcaseMode::Auxiliary,
                    ShowcaseMode::CurveNetworkQuantities,
                    ShowcaseMode::DepthCompositeImages,
                    ShowcaseMode::ImplicitSurface,
                    ShowcaseMode::SparseVolumeGrid,
                    ShowcaseMode::ExtendedQuantities,
                    ShowcaseMode::PickLevels,
                    ShowcaseMode::Labels,
                    ShowcaseMode::Overlay,
                    ShowcaseMode::PlaybackRuntime,
                    ShowcaseMode::ProbeWidgets,
                    ShowcaseMode::SurfaceLIC,
                    ShowcaseMode::TensorGlyphs,
                    ShowcaseMode::VertexWarp,
                    ShowcaseMode::Sprites,
                    ShowcaseMode::GaussianSplats,
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
            showcase_02_scene_graph::background_colour(self.sg_state.bg_cycle)
        } else {
            BG_COLOUR
        };
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE.fill(egui::Color32::from_rgba_unmultiplied(
                    (panel_bg[0] * 255.0) as u8,
                    (panel_bg[1] * 255.0) as u8,
                    (panel_bg[2] * 255.0) as u8,
                    255,
                )),
            )
            .show(ctx, |ui| {
                let available = ui.available_size();
                let (rect, response) =
                    ui.allocate_exact_size(available, egui::Sense::click_and_drag());

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
                let manip_active_for_text = self.interact_state.manip.is_active();
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
                        self.interact_state.last_cursor_viewport = local;
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
                                if let Some(kc) = shared::egui_key_to_keycode(*key) {
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
                                    egui::PointerButton::Secondary => {
                                        viewport_lib::MouseButton::Right
                                    }
                                    egui::PointerButton::Middle => {
                                        viewport_lib::MouseButton::Middle
                                    }
                                    _ => continue,
                                };

                                // Ignore presses that originate outside the viewport.
                                if *pressed && !rect.contains(*pos) {
                                    continue;
                                }

                                // Track raw left-button held state for ManipulationContext.
                                if *button == egui::PointerButton::Primary {
                                    if *pressed {
                                        self.interact_state.left_held = true;
                                    } else {
                                        self.interact_state.left_held = false;
                                    }
                                }

                                // Clip-vol gizmo : start drag.
                                if self.mode == ShowcaseMode::ClipVolumes
                                    && *button == egui::PointerButton::Primary
                                    && *pressed
                                {
                                    let local =
                                        glam::Vec2::new(pos.x - rect.left(), pos.y - rect.top());
                                    if let Some(center) = self.clipvol_state.gizmo_center {
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
                                        let hit = self.clipvol_state.gizmo.hit_test_oriented(
                                            ray_origin,
                                            ray_dir,
                                            center,
                                            self.clipvol_state.gizmo_scale,
                                            orient,
                                        );
                                        if hit != GizmoAxis::None {
                                            self.clipvol_state.gizmo.active_axis = hit;
                                            self.clipvol_state.gizmo_drag_active = true;
                                        }
                                    }
                                }

                                // Clip-vol gizmo : end drag.
                                if self.mode == ShowcaseMode::ClipVolumes
                                    && *button == egui::PointerButton::Primary
                                    && !pressed
                                    && self.clipvol_state.gizmo_drag_active
                                {
                                    self.clipvol_state.gizmo_drag_active = false;
                                    self.clipvol_state.gizmo.active_axis = GizmoAxis::None;
                                }

                                // PickLevels: track drag start for rubber-band box select.
                                if self.mode == ShowcaseMode::PickLevels
                                    && *button == egui::PointerButton::Primary
                                    && *pressed
                                {
                                    let local =
                                        glam::Vec2::new(pos.x - rect.left(), pos.y - rect.top());
                                    self.pl_state.drag_start = Some(local);
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

                            egui::Event::MouseWheel { unit, delta, .. } => {
                                let over_vp = i
                                    .pointer
                                    .hover_pos()
                                    .map(|p| rect.contains(p))
                                    .unwrap_or(false);
                                if over_vp {
                                    let units = match unit {
                                        egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                                        egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                                        egui::MouseWheelUnit::Page => ScrollUnits::Pages,
                                    };
                                    self.controller.push_event(ViewportEvent::Wheel {
                                        delta: glam::Vec2::new(delta.x, delta.y),
                                        units,
                                    });
                                }
                            }

                            _ => {}
                        }
                    }
                });

                // ----- PickLevels: update shift state and fire box-select on drag end -----
                if self.mode == ShowcaseMode::PickLevels {
                    self.pl_state.shift_held = ctx.input(|i| i.modifiers.shift);
                    if response.drag_stopped() {
                        if let Some(drag_start) = self.pl_state.drag_start.take() {
                            let drag_end = self.interact_state.last_cursor_viewport;
                            if (drag_end - drag_start).length() > 4.0 {
                                let shift = self.pl_state.shift_held;
                                if self.pl_state.unified_mode {
                                    let vp_size = glam::Vec2::new(rect.width(), rect.height());
                                    let view_proj = self.camera.view_proj_matrix();
                                    let rs = frame.wgpu_render_state().expect("wgpu required");
                                    let guard = rs.renderer.read();
                                    if let Some(renderer) =
                                        guard.callback_resources.get::<ViewportRenderer>()
                                    {
                                        self.handle_pl_unified_box_select(
                                            drag_start, drag_end, vp_size, view_proj, shift,
                                            renderer,
                                        );
                                    }
                                } else {
                                    self.handle_pl_box_select(
                                        drag_start,
                                        drag_end,
                                        rect.width(),
                                        rect.height(),
                                        shift,
                                    );
                                }
                            }
                        }
                    }
                    // Clear drag start if the button was released below egui's drag threshold.
                    if !ctx.input(|i| i.pointer.primary_down()) {
                        self.pl_state.drag_start = None;
                    }
                }

                // ----- Clip-vol gizmo drag (Showcase 18) -----
                if self.mode == ShowcaseMode::ClipVolumes
                    && self.clipvol_state.gizmo_drag_active
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
                    self.interact_state.animator.update(dt, &mut self.camera);
                }
                if self.mode == ShowcaseMode::CameraTools || self.mode == ShowcaseMode::Auxiliary {
                    let dt = ctx.input(|i| i.stable_dt.min(1.0 / 30.0));
                    self.cam_animator.update(dt, &mut self.camera);
                    if self.mode == ShowcaseMode::Auxiliary {
                        match self.aux_state.sub_mode {
                            AuxSubMode::Turntable if self.aux_state.turntable_running => {
                                self.aux_state.turntable.update(dt, &mut self.camera);
                            }
                            AuxSubMode::Track if self.aux_state.track_playing => {
                                self.aux_state.track_t += dt as f64;
                                if self.aux_state.track_t > self.aux_state.track.duration() {
                                    self.aux_state.track_t = 0.0;
                                }
                                let target = viewport_lib::interpolate_camera(
                                    &self.aux_state.track,
                                    self.aux_state.track_t,
                                );
                                self.camera.center = target.center;
                                self.camera.set_distance(target.distance);
                                self.camera.set_orientation(target.orientation);
                            }
                            _ => {}
                        }
                        ctx.request_repaint();
                    }
                }

                // ----- ManipulationController update (Showcase 4 only) -----
                // For Interaction mode, orbit resolution is integrated here so that
                // the same ActionFrame drives both camera and gizmo.
                if self.mode == ShowcaseMode::Interaction {
                    if self.interact_state.built {
                        let w = rect.width();
                        let h = rect.height();
                        let viewport_size = glam::Vec2::new(w, h);
                        let view_proj = self.camera.proj_matrix() * self.camera.view_matrix();

                        // Per-frame gizmo hover when no session is active.
                        if !self.interact_state.manip.is_active() {
                            if let Some(center) = self.interact_state.gizmo_center {
                                let ray_origin = self.camera.eye_position();
                                let cursor = self.interact_state.last_cursor_viewport;
                                let ndc_x = (cursor.x / w.max(1.0)) * 2.0 - 1.0;
                                let ndc_y = 1.0 - (cursor.y / h.max(1.0)) * 2.0;
                                let inv_vp = view_proj.inverse();
                                let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
                                let ray_dir = (far - ray_origin).normalize_or_zero();
                                let orient = gizmo_helpers::gizmo_orientation(
                                    &self.interact_state.gizmo,
                                    &self.interact_state.selection,
                                    &self.interact_state.scene,
                                );
                                self.interact_state.gizmo.hovered_axis =
                                    self.interact_state.gizmo.hit_test_oriented(
                                        ray_origin,
                                        ray_dir,
                                        center,
                                        self.interact_state.gizmo_scale,
                                        orient,
                                    );
                            } else {
                                self.interact_state.gizmo.hovered_axis = GizmoAxis::None;
                            }
                        }

                        // Build GizmoInfo.
                        let orient = gizmo_helpers::gizmo_orientation(
                            &self.interact_state.gizmo,
                            &self.interact_state.selection,
                            &self.interact_state.scene,
                        );
                        let gizmo_info = self.interact_state.gizmo_center.map(|center| GizmoInfo {
                            center,
                            scale: self.interact_state.gizmo_scale,
                            orientation: orient,
                            mode: self.interact_state.gizmo.mode,
                        });

                        // Build ManipulationContext.
                        let pointer_delta = ctx
                            .input(|i| glam::Vec2::new(i.pointer.delta().x, i.pointer.delta().y));
                        let manip_ctx = ManipulationContext {
                            camera: self.camera.clone(),
                            viewport_size,
                            cursor_viewport: Some(self.interact_state.last_cursor_viewport),
                            pointer_delta,
                            selection_center: self.interact_state.gizmo_center,
                            gizmo: gizmo_info,
                            drag_started: response.drag_started(),
                            dragging: self.interact_state.left_held,
                            clicked: response.clicked(),
                        };

                        // Orbit: resolve (no camera movement) while manipulation is active.
                        let action_frame = if self.interact_state.manip.is_active() {
                            self.controller.resolve()
                        } else {
                            self.controller.apply_to_camera(&mut self.camera)
                        };

                        // Tab cycles gizmo mode when no session is active.
                        if !self.interact_state.manip.is_active()
                            && action_frame.is_active(Action::CycleGizmoMode)
                        {
                            self.interact_state.gizmo.mode = match self.interact_state.gizmo.mode {
                                GizmoMode::Translate => GizmoMode::Rotate,
                                GizmoMode::Rotate => GizmoMode::Scale,
                                GizmoMode::Scale => GizmoMode::Translate,
                                _ => GizmoMode::Translate,
                            };
                        }

                        match self.interact_state.manip.update(&action_frame, manip_ctx) {
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
                                if !self.interact_state.manip.is_active() {
                                    // Keep snapshots current so G/R/S always starts clean.
                                    self.save_interact_snapshots();
                                }
                            }
                        }

                        // Click-to-select: only when no session is active.
                        if response.clicked() && !self.interact_state.manip.is_active() {
                            let pick_pos = self.interact_state.last_cursor_viewport;
                            self.handle_click_select(pick_pos, w, h);
                        }
                    } else {
                        self.controller.apply_to_camera(&mut self.camera);
                    }
                } else {
                    // ----- Apply / resolve orbit controller (non-Interaction modes) -----
                    let suppress_orbit = (self.mode == ShowcaseMode::ClipVolumes
                        && self.clipvol_state.gizmo_drag_active)
                        || (self.mode == ShowcaseMode::PickLevels
                            && self.pl_state.drag_start.is_some())
                        || (self.mode == ShowcaseMode::ProbeWidgets
                            && self.pw_state.suppress_orbit);
                    if suppress_orbit {
                        self.controller.resolve();
                    } else {
                        self.controller.apply_to_camera(&mut self.camera);
                    }
                }

                self.camera.set_aspect_ratio(rect.width(), rect.height());

                // ----- Spline widget update (Showcase 4) -----
                if self.mode == ShowcaseMode::Interaction && self.interact_state.built {
                    let render_cam =
                        CameraFrame::from_camera(&self.camera, [rect.width(), rect.height()])
                            .render_camera;
                    let widget_ctx = viewport_lib::WidgetContext {
                        camera: render_cam,
                        viewport_size: glam::Vec2::new(rect.width(), rect.height()),
                        cursor_viewport: self.interact_state.last_cursor_viewport,
                        drag_started: response.drag_started(),
                        dragging: response.dragged(),
                        released: response.drag_stopped(),
                        double_clicked: false,
                    };
                    self.interact_state.spline.update(&widget_ctx);
                }

                // ----- Probe widgets update (Showcase 37) -----
                if self.mode == ShowcaseMode::ProbeWidgets && self.pw_state.built {
                    let render_cam =
                        CameraFrame::from_camera(&self.camera, [rect.width(), rect.height()])
                            .render_camera;
                    let widget_ctx = viewport_lib::WidgetContext {
                        camera: render_cam,
                        viewport_size: glam::Vec2::new(rect.width(), rect.height()),
                        cursor_viewport: self.interact_state.last_cursor_viewport,
                        drag_started: response.drag_started(),
                        dragging: response.dragged(),
                        released: response.drag_stopped(),
                        double_clicked: response.double_clicked(),
                    };
                    self.update_probe_widgets(widget_ctx);
                }

                // ----- Click-to-select (non-Interaction modes) -----
                if response.clicked() && self.mode != ShowcaseMode::Interaction {
                    let pick_pos = self.interact_state.last_cursor_viewport;
                    // Unified pick for PickLevels showcase uses renderer.pick().
                    if self.mode == ShowcaseMode::PickLevels && self.pl_state.unified_mode {
                        let vp_size = glam::Vec2::new(rect.width(), rect.height());
                        let view_proj = self.camera.view_proj_matrix();
                        let shift = self.pl_state.shift_held;
                        let rs = frame.wgpu_render_state().expect("wgpu required");
                        let guard = rs.renderer.read();
                        if let Some(renderer) = guard.callback_resources.get::<ViewportRenderer>() {
                            self.handle_pl_unified_click(
                                pick_pos, vp_size, view_proj, shift, renderer,
                            );
                        }
                    } else {
                        self.handle_click_select(pick_pos, rect.width(), rect.height());
                    }
                }

                // ----- Voxel paint: flush painted cell to GPU -----
                if self.svg_state.paint_dirty && self.mode == ShowcaseMode::SparseVolumeGrid {
                    self.svg_state.paint_dirty = false;
                    let rs = frame.wgpu_render_state().expect("wgpu required");
                    let mut guard = rs.renderer.write();
                    if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                        let _ = renderer.resources_mut().replace_sparse_volume_grid_data(
                            &self.device,
                            &self.queue,
                            self.svg_state.paint_mesh_id,
                            &self.svg_state.paint_data,
                        );
                    }
                }

                // ----- Build frame data -----
                let frame_data = self.build_frame_data(
                    rect.width(),
                    rect.height(),
                    ui.ctx().pixels_per_point(),
                    frame,
                );

                // ----- Update gizmo_center cache for next frame's hit-testing -----
                if self.mode == ShowcaseMode::Interaction {
                    self.interact_state.gizmo_center =
                        gizmo::gizmo_center_from_selection(&self.interact_state.selection, |id| {
                            self.interact_state.scene.node(id).map(|n| {
                                let t = n.world_transform();
                                glam::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z)
                            })
                        });
                    if let Some(center) = self.interact_state.gizmo_center {
                        self.interact_state.gizmo_scale = compute_gizmo_scale(
                            center,
                            self.camera.eye_position(),
                            self.camera.fov_y,
                            rect.height(),
                        );
                    }
                }
                if self.mode == ShowcaseMode::ClipVolumes && self.clipvol_state.built {
                    self.clipvol_state.gizmo_center = self.clip_gizmo_center();
                    if let Some(center) = self.clipvol_state.gizmo_center {
                        self.clipvol_state.gizmo_scale = compute_gizmo_scale(
                            center,
                            self.camera.eye_position(),
                            self.camera.fov_y,
                            rect.height(),
                        );
                    }
                }

                // ----- Schedule paint callback -----
                ui.painter()
                    .add(eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        viewport_callback::ViewportCallback::new(frame_data),
                    ));

                // ----- PickLevels: rubber-band drag rect overlay -----
                if self.mode == ShowcaseMode::PickLevels {
                    if let Some(drag_start) = self.pl_state.drag_start {
                        let drag_end = self.interact_state.last_cursor_viewport;
                        if response.dragged() && (drag_end - drag_start).length() > 4.0 {
                            let a =
                                egui::pos2(rect.left() + drag_start.x, rect.top() + drag_start.y);
                            let b = egui::pos2(rect.left() + drag_end.x, rect.top() + drag_end.y);
                            let sel_rect = egui::Rect::from_two_pos(a, b);
                            ui.painter().rect(
                                sel_rect,
                                0.0,
                                egui::Color32::from_rgba_unmultiplied(255, 200, 50, 20),
                                egui::Stroke::new(
                                    1.5,
                                    egui::Color32::from_rgba_unmultiplied(255, 200, 50, 200),
                                ),
                                egui::StrokeKind::Outside,
                            );
                        }
                    }
                }

                // ----- Manipulation mode overlay (Showcase 4) -----
                if self.mode == ShowcaseMode::Interaction {
                    if let Some(ms) = self.interact_state.manip.state() {
                        let kind_label = match ms.kind {
                            viewport_lib::ManipulationKind::Move => "Move",
                            viewport_lib::ManipulationKind::Rotate => "Rotate",
                            viewport_lib::ManipulationKind::Scale => "Scale",
                        };
                        let axis_label = match ms.axis {
                            Some(GizmoAxis::X) => {
                                if ms.exclude_axis {
                                    " (YZ)"
                                } else {
                                    " (X)"
                                }
                            }
                            Some(GizmoAxis::Y) => {
                                if ms.exclude_axis {
                                    " (XZ)"
                                } else {
                                    " (Y)"
                                }
                            }
                            Some(GizmoAxis::Z) => {
                                if ms.exclude_axis {
                                    " (XY)"
                                } else {
                                    " (Z)"
                                }
                            }
                            _ => "",
                        };
                        let text = if let Some(ref numeric) = ms.numeric_display {
                            format!("{kind_label}{axis_label}: {numeric}")
                        } else {
                            format!("{kind_label}{axis_label}")
                        };
                        let font = egui::FontId::proportional(14.0);
                        let galley = ui
                            .painter()
                            .layout_no_wrap(text, font, egui::Color32::WHITE);
                        let pos =
                            egui::pos2(rect.center().x - galley.size().x / 2.0, rect.max.y - 30.0);
                        let bg = egui::Rect::from_min_size(
                            pos - egui::vec2(6.0, 3.0),
                            galley.size() + egui::vec2(12.0, 6.0),
                        );
                        ui.painter()
                            .rect_filled(bg, 3.0, egui::Color32::from_black_alpha(180));
                        ui.painter().galley(pos, galley, egui::Color32::WHITE);
                        ctx.request_repaint();
                    }
                }

                // (Annotation labels now render natively via OverlayFrame.)
                if self.mode == ShowcaseMode::BackfacePolicy {
                    self.draw_sa_labels(ui, rect);
                    self.draw_sa_row_labels(ui, rect);
                }

                // ----- Cursor feedback -----
                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }

                // ----- Continuous repaint for background build progress -----
                if self.perf_state.build_rx.is_some() {
                    ctx.request_repaint();
                }

                // ----- Continuous repaint for animated camera -----
                if self.mode == ShowcaseMode::Interaction
                    && self.interact_state.animator.is_animating()
                {
                    ctx.request_repaint();
                }
                if (self.mode == ShowcaseMode::CameraTools || self.mode == ShowcaseMode::Auxiliary)
                    && self.cam_animator.is_animating()
                {
                    ctx.request_repaint();
                }
                // ----- Playback runtime: advance time and request repaint -----
                if self.mode == ShowcaseMode::PlaybackRuntime
                    && self.pb_state.mode == RuntimeMode::Playback
                {
                    let dt = ctx.input(|i| i.stable_dt.min(1.0 / 15.0));
                    self.pb_state.time += dt;
                    ctx.request_repaint();
                }
                // ----- Sprites: simulate particles and advance atlas frame -----
                if self.mode == ShowcaseMode::Sprites && self.sprite_state.built {
                    let dt = ctx.input(|i| i.stable_dt.min(1.0 / 30.0));
                    showcase_41_sprites::update_sprites(self, dt);
                    ctx.request_repaint();
                }
                // ----- Gaussian splats: advance slow rotation -----
                if self.mode == ShowcaseMode::GaussianSplats && self.splat_state.built {
                    let dt = ctx.input(|i| i.stable_dt.min(1.0 / 30.0));
                    showcase_42_gaussian_splats::update_gaussian_splats(self, dt);
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
        const SHOWCASE_MODES: [ShowcaseMode; 42] = [
            ShowcaseMode::Basic,
            ShowcaseMode::SceneGraph,
            ShowcaseMode::Performance,
            ShowcaseMode::Interaction,
            ShowcaseMode::MaterialsVisibility,
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
            ShowcaseMode::BackfacePolicy,
            ShowcaseMode::SurfaceVectors,
            ShowcaseMode::VolumeMesh,
            ShowcaseMode::Auxiliary,
            ShowcaseMode::CurveNetworkQuantities,
            ShowcaseMode::DepthCompositeImages,
            ShowcaseMode::ImplicitSurface,
            ShowcaseMode::SparseVolumeGrid,
            ShowcaseMode::ExtendedQuantities,
            ShowcaseMode::PickLevels,
            ShowcaseMode::Labels,
            ShowcaseMode::Overlay,
            ShowcaseMode::PlaybackRuntime,
            ShowcaseMode::ProbeWidgets,
            ShowcaseMode::SurfaceLIC,
            ShowcaseMode::TensorGlyphs,
            ShowcaseMode::VertexWarp,
            ShowcaseMode::Sprites,
            ShowcaseMode::GaussianSplats,
        ];

        let Some(current) = SHOWCASE_MODES.iter().position(|&mode| mode == self.mode) else {
            return;
        };
        let len = SHOWCASE_MODES.len() as i32;
        let next = (current as i32 + dir).rem_euclid(len) as usize;
        self.switch_mode(SHOWCASE_MODES[next]);
    }

    fn cycle_selection_tab(&mut self) {
        if self.mode == ShowcaseMode::Auxiliary {
            // Cycle: overview -> A -> B -> C -> overview -> ...
            let next = match self.aux_state.active_frustum {
                None => Some(0),
                Some(i) if i + 1 < self.aux_state.frustums.len() => Some(i + 1),
                _ => None,
            };
            match next {
                Some(i) => {
                    let t = self.aux_state.frustums[i].camera_view_target();
                    self.cam_animator.fly_to(
                        &self.camera,
                        t.center,
                        t.distance,
                        t.orientation,
                        0.8,
                    );
                    self.aux_state.active_frustum = Some(i);
                }
                None => {
                    self.cam_animator.fly_to(
                        &self.camera,
                        glam::Vec3::new(0.0, 0.0, 0.5),
                        30.0,
                        glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.0),
                        0.8,
                    );
                    self.aux_state.active_frustum = None;
                }
            }
            return;
        }
        let (scene, selection) = match self.mode {
            ShowcaseMode::SceneGraph => (&self.sg_state.scene, &mut self.sg_state.selection),
            ShowcaseMode::MaterialsVisibility => (
                &self.materials_visibility_state.scene,
                &mut self.materials_visibility_state.selection,
            ),
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
        self.mode_gen = self.mode_gen.wrapping_add(1);
        if let Some(cam) = shared::opening_camera(mode) {
            self.camera = cam;
        }
        // Camera resets are applied on first build in ensure_scene_built.
        // Switching back to an already-built showcase doesn't reset.
    }

    fn ensure_scene_built(&mut self, frame: &eframe::Frame) {
        let needs = match self.mode {
            ShowcaseMode::SceneGraph => !self.sg_state.built,
            ShowcaseMode::Performance => {
                !self.perf_state.built && self.perf_state.build_rx.is_none()
            }
            ShowcaseMode::Interaction => !self.interact_state.built,
            ShowcaseMode::MaterialsVisibility => !self.materials_visibility_state.built,
            ShowcaseMode::PostProcess => !self.pp_state.built,
            ShowcaseMode::NormalMaps => !self.nm_state.built,
            ShowcaseMode::Shadows => !self.shd_state.built,
            ShowcaseMode::Annotation => !self.ann_state.built,
            ShowcaseMode::CameraTools => !self.ct_state.built,
            ShowcaseMode::Lights => !self.lights_state.built,
            ShowcaseMode::ScalarFields => !self.scalar_state.built,
            ShowcaseMode::MultiViewport => {
                !self.mv_state.built || self.mv_state.viewports.is_none()
            }
            ShowcaseMode::Isolines => !self.iso_state.built,
            ShowcaseMode::PointClouds => !self.pc_state.built,
            ShowcaseMode::Streamlines => !self.stream_state.built,
            ShowcaseMode::Volume => !self.vol_state.built,
            ShowcaseMode::ClipVolumes => !self.clipvol_state.built,
            ShowcaseMode::Matcap => !self.matcap_state.built,
            ShowcaseMode::FaceAttributes => !self.face_state.built,
            ShowcaseMode::Textures => !self.texture_state.built,
            ShowcaseMode::ParamVis => !self.param_vis_state.built,
            ShowcaseMode::GroundPlane => !self.gp_state.built,
            ShowcaseMode::BackfacePolicy => !self.sa_state.built,
            ShowcaseMode::SurfaceVectors => !self.sv_state.built,
            ShowcaseMode::VolumeMesh => !self.vm_state.built,
            ShowcaseMode::Auxiliary => !self.aux_state.built,
            ShowcaseMode::DepthCompositeImages => !self.dc_state.built,
            ShowcaseMode::ImplicitSurface => !self.is_state.built,
            ShowcaseMode::SparseVolumeGrid => !self.svg_state.built,
            ShowcaseMode::ExtendedQuantities => !self.eq_state.built,
            ShowcaseMode::PickLevels => !self.pl_state.built,
            ShowcaseMode::Labels => !self.lbl_state.built,
            ShowcaseMode::Overlay => !self.ovl_state.cloud_built,
            ShowcaseMode::PlaybackRuntime => !self.pb_state.built,
            ShowcaseMode::ProbeWidgets => !self.pw_state.built,
            ShowcaseMode::SurfaceLIC => !self.lic_state.built,
            ShowcaseMode::TensorGlyphs => !self.tg_state.built,
            ShowcaseMode::VertexWarp => !self.warp_state.built,
            ShowcaseMode::Sprites => !self.sprite_state.built,
            ShowcaseMode::GaussianSplats => !self.splat_state.built,
            ShowcaseMode::Basic => self.basic_state.mesh_id.is_none(),
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
            ShowcaseMode::Basic => self.build_basic_scene(renderer),
            ShowcaseMode::SceneGraph => self.build_scene_graph(renderer),
            ShowcaseMode::Performance => {
                // Upload the box mesh on the main thread (requires GPU access).
                let mesh = self.upload_box(renderer);
                self.perf_state.mesh = Some(mesh);
                self.perf_state.scene = Scene::new();
                self.perf_state.selection.clear();
                self.camera.distance = 80.0;

                // Capture the mesh AABB before releasing the renderer borrow.
                let mesh_aabb = renderer.resources().mesh(mesh).map(|m| m.aabb);

                let progress = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let progress_clone = std::sync::Arc::clone(&progress);
                let (tx, rx) = std::sync::mpsc::channel();

                std::thread::spawn(move || {
                    let result = showcase_03_performance::build_perf_scene_threaded(
                        mesh,
                        mesh_aabb,
                        &progress_clone,
                    );
                    let _ = tx.send(result);
                });

                self.perf_state.build_progress = Some(progress);
                self.perf_state.build_rx = Some(rx);
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
            ShowcaseMode::MaterialsVisibility => {
                self.build_materials_visibility_scene(renderer);
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
                if self.mv_state.viewports.is_none() {
                    let vp0 = renderer.create_viewport(&rs.device);
                    let vp1 = renderer.create_viewport(&rs.device);
                    let vp2 = renderer.create_viewport(&rs.device);
                    let vp3 = renderer.create_viewport(&rs.device);
                    self.mv_state.viewports = Some([vp0, vp1, vp2, vp3]);
                }
                if !self.mv_state.built {
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
                self.ensure_surface_slice_mesh(renderer);
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
            ShowcaseMode::BackfacePolicy => {
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
                self.build_aux_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 0.5),
                    distance: 30.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::DepthCompositeImages => {
                self.build_dc_scene(renderer);
            }
            ShowcaseMode::ImplicitSurface => {
                self.build_implicit_scene(renderer);
            }
            ShowcaseMode::SparseVolumeGrid => {
                self.build_svg_scene(renderer);
                self.camera = viewport_lib::Camera {
                    center: glam::Vec3::ZERO,
                    distance: 28.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(0.8),
                    ..viewport_lib::Camera::default()
                };
            }
            ShowcaseMode::ExtendedQuantities => {
                self.build_eq_scene(renderer);
                self.camera = viewport_lib::Camera {
                    center: glam::Vec3::ZERO,
                    distance: 18.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(0.8),
                    ..viewport_lib::Camera::default()
                };
            }
            ShowcaseMode::PickLevels => {
                self.build_pl_scene(renderer);
            }
            ShowcaseMode::Labels => {
                self.build_labels_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 0.0),
                    distance: 20.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Overlay => {
                let (positions, scalars) = showcase_35_overlay::build_ovl_cloud();
                self.ovl_state.cloud_positions = positions;
                self.ovl_state.cloud_scalars = scalars;
                self.ovl_state.cloud_built = true;
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 1.57, 0.0),
                    distance: 9.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::PlaybackRuntime => {
                showcase_36_playback_runtime::build_pb_scene(self, renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 0.0),
                    distance: 18.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                };
            }
            ShowcaseMode::ProbeWidgets => {
                self.build_probe_widgets_scene(renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 8.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::SurfaceLIC => {
                showcase_38_surface_lic::build_lic_scene(self, renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 7.5),
                    distance: 18.0,
                    orientation: glam::Quat::from_rotation_x(0.45),
                    ..Camera::default()
                };
            }
            ShowcaseMode::TensorGlyphs => {
                showcase_39_tensor_glyphs::build_tensor_glyph_scene(self, renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 0.0),
                    distance: 16.0,
                    orientation: glam::Quat::from_rotation_x(0.15),
                    ..Camera::default()
                };
            }
            ShowcaseMode::VertexWarp => {
                showcase_40_vertex_warp::build_warp_scene(self, renderer);
                self.camera = Camera {
                    center: glam::Vec3::new(0.0, 0.0, 0.0),
                    distance: 10.0,
                    orientation: glam::Quat::from_rotation_x(0.6),
                    ..Camera::default()
                };
            }
            ShowcaseMode::Sprites => {
                showcase_41_sprites::build_sprite_scene(self, renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 10.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.0),
                    ..Camera::default()
                };
            }
            ShowcaseMode::GaussianSplats => {
                showcase_42_gaussian_splats::build_gaussian_splat_scene(self, renderer);
                self.camera = Camera {
                    center: glam::Vec3::ZERO,
                    distance: 12.0,
                    orientation: glam::Quat::from_rotation_x(0.5),
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
            ShowcaseMode::Basic => showcase_01_basic::controls_basic(self, ui),
            ShowcaseMode::SceneGraph => {
                showcase_02_scene_graph::controls_scene_graph(self, ui, frame)
            }
            ShowcaseMode::Performance => showcase_03_performance::controls_performance(self, ui),
            ShowcaseMode::Interaction => showcase_04_interaction::controls_interaction(self, ui),
            ShowcaseMode::MaterialsVisibility => {
                showcase_05_materials_and_visibility::controls_materials_visibility(self, ui)
            }
            ShowcaseMode::PostProcess => showcase_06_post_process::controls_post_process(self, ui),
            ShowcaseMode::NormalMaps => showcase_07_normal_maps::controls_normal_maps(self, ui),
            ShowcaseMode::Shadows => showcase_08_shadows::controls_shadows(self, ui),
            ShowcaseMode::Annotation => showcase_09_annotation::controls_annotation(self, ui),
            ShowcaseMode::CameraTools => showcase_10_camera_tools::controls_camera_tools(self, ui),
            ShowcaseMode::Lights => showcase_11_lights::controls_lights(self, ui),
            ShowcaseMode::ScalarFields => {
                showcase_12_scalar_fields::controls_scalar_fields(self, ui)
            }
            ShowcaseMode::MultiViewport => showcase_13_multi_viewport::controls_mv(self, ui),
            ShowcaseMode::Isolines => showcase_14_isolines::controls_isolines(self, ui),
            ShowcaseMode::PointClouds => showcase_15_point_clouds::controls_point_clouds(self, ui),
            ShowcaseMode::Streamlines => showcase_16_streamlines::controls_streamlines(self, ui),
            ShowcaseMode::Volume => showcase_17_volume::controls_volume(self, ui, frame),
            ShowcaseMode::ClipVolumes => showcase_18_clip_volumes::controls_clipvol(self, ui),
            ShowcaseMode::Matcap => showcase_19_matcap::controls_matcap(self, ui, frame),
            ShowcaseMode::FaceAttributes => {
                showcase_20_face_attributes::controls_face_attr(self, ui)
            }
            ShowcaseMode::Textures => showcase_21_textures::controls_textures(self, ui),
            ShowcaseMode::ParamVis => showcase_22_parameterization::controls_param_vis(self, ui),
            ShowcaseMode::GroundPlane => showcase_23_ground_plane::controls_ground_plane(self, ui),
            ShowcaseMode::BackfacePolicy => {
                showcase_24_backface_policy::controls_surface_appearance(self, ui)
            }
            ShowcaseMode::SurfaceVectors => {
                showcase_25_surface_vectors::controls_surface_vectors(self, ui)
            }
            ShowcaseMode::VolumeMesh => showcase_26_volume_mesh::controls_volume_mesh(self, ui),
            ShowcaseMode::Auxiliary => showcase_27_camera_framing::controls_aux(self, ui),
            ShowcaseMode::CurveNetworkQuantities => {
                showcase_28_curve_network_quantities::controls_cnq(self, ui)
            }
            ShowcaseMode::DepthCompositeImages => {
                showcase_29_depth_composite_images::controls_dc(self, ui)
            }
            ShowcaseMode::ImplicitSurface => {
                showcase_30_implicit_surface::controls_implicit(self, ui)
            }
            ShowcaseMode::SparseVolumeGrid => {
                showcase_31_sparse_volume_grid::controls_sparse_volume_grid(self, ui)
            }
            ShowcaseMode::ExtendedQuantities => {
                showcase_32_extended_quantities::controls_eq(self, ui)
            }
            ShowcaseMode::PickLevels => showcase_33_picking_levels::controls_pick_levels(self, ui),
            ShowcaseMode::Labels => showcase_34_labels::controls_labels(self, ui),
            ShowcaseMode::Overlay => showcase_35_overlay::controls_overlay(self, ui),
            ShowcaseMode::PlaybackRuntime => {
                showcase_36_playback_runtime::controls_pb(self, ui, frame)
            }
            ShowcaseMode::ProbeWidgets => self.controls_probe_widgets(ui),
            ShowcaseMode::SurfaceLIC => showcase_38_surface_lic::controls_lic(self, ui),
            ShowcaseMode::TensorGlyphs => {
                showcase_39_tensor_glyphs::controls_tensor_glyphs(self, ui)
            }
            ShowcaseMode::VertexWarp => showcase_40_vertex_warp::controls_warp(self, ui),
            ShowcaseMode::Sprites => showcase_41_sprites::controls_sprites(self, ui),
            ShowcaseMode::GaussianSplats => {
                showcase_42_gaussian_splats::controls_gaussian_splats(self, ui)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Frame data assembly
// ---------------------------------------------------------------------------

impl App {
    fn build_frame_data(
        &mut self,
        w: f32,
        h: f32,
        pixels_per_point: f32,
        frame: &eframe::Frame,
    ) -> FrameData {
        let mut adv_clip_objects: Vec<ClipObject> = vec![];
        let mut adv_outline = false;
        let mut adv_xray = false;
        let mut perf_outline = false;
        let mut interact_outline = false;
        let mut scene_graph_outline = false;
        let mut scene_graph_outline_width = 4.0_f32;

        let mut eq_glyphs: Vec<GlyphItem> = Vec::new();
        let mut eq_pcs: Vec<PointCloudItem> = Vec::new();

        // Performance showcase uses a cached Arc<[SceneRenderItem]> to avoid a per-frame
        // deep clone of the 1M-item Vec. Set by the Performance arm below; None for all others.
        let mut perf_arc: Option<std::sync::Arc<[SceneRenderItem]>> = None;

        let (scene_items, bg_colour, lighting, scene_gen, sel_gen) = match self.mode {
            ShowcaseMode::Basic => {
                let items = self.basic_scene_items();

                let lights = if self.basic_state.use_point_light {
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
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, 0u64, 0u64)
            }

            ShowcaseMode::SceneGraph => {
                let items = self
                    .sg_state
                    .scene
                    .collect_render_items(&self.sg_state.selection);
                let bg = showcase_02_scene_graph::background_colour(self.sg_state.bg_cycle);
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                scene_graph_outline = !self.sg_state.selection.is_empty();
                scene_graph_outline_width = self.sg_state.outline_width;
                let sg = self.sg_state.scene.version();
                let ss = self.sg_state.selection.version();
                (items, Some(bg), lighting, sg, ss)
            }

            ShowcaseMode::Performance => {
                let current_ver = (
                    self.perf_state.scene.version(),
                    self.perf_state.selection.version(),
                );
                if current_ver != self.perf_state.scene_items_version {
                    self.perf_state.scene_items_cache = std::sync::Arc::from(
                        self.perf_state
                            .scene
                            .collect_render_items(&self.perf_state.selection),
                    );
                    self.perf_state.scene_items_version = current_ver;
                }
                // Arc::clone is a single atomic refcount increment — no data copy.
                perf_arc = Some(std::sync::Arc::clone(&self.perf_state.scene_items_cache));
                let sg = self.perf_state.scene.version();
                let ss = self.perf_state.selection.version();
                perf_outline = !self.perf_state.selection.is_empty();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (vec![], Some(BG_COLOUR), lighting, sg, ss)
            }

            ShowcaseMode::Interaction => {
                let items = self
                    .interact_state
                    .scene
                    .collect_render_items(&self.interact_state.selection);
                interact_outline = !self.interact_state.selection.is_empty();
                let sg = self.interact_state.scene.version();
                let ss = self.interact_state.selection.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, ss)
            }

            ShowcaseMode::MaterialsVisibility => {
                let items = self
                    .materials_visibility_state
                    .scene
                    .collect_render_items(&self.materials_visibility_state.selection);
                if self.materials_visibility_state.clip_enabled {
                    adv_clip_objects.push(ClipObject::plane([1.0, 0.0, 0.0], 0.0));
                }
                adv_outline = self.materials_visibility_state.outline_on
                    && !self.materials_visibility_state.selection.is_empty();
                adv_xray = self.materials_visibility_state.xray_on
                    && !self.materials_visibility_state.selection.is_empty();
                let sg = self.materials_visibility_state.scene.version();
                let ss = self.materials_visibility_state.selection.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, ss)
            }

            ShowcaseMode::PostProcess => {
                let items = self.pp_state.scene.collect_render_items(&Selection::new());
                let mut lights = vec![LightSource {
                    kind: LightKind::Directional {
                        direction: [0.6, 0.4, 1.0],
                    },
                    intensity: self.pp_state.dir_intensity,
                    ..LightSource::default()
                }];
                if self.pp_state.point_light_on {
                    lights.push(LightSource {
                        kind: LightKind::Point {
                            position: [3.0, 3.0, 3.0],
                            range: 15.0,
                        },
                        colour: [1.0, 0.9, 0.7],
                        intensity: 2.0,
                        ..LightSource::default()
                    });
                }
                let lighting = LightingSettings {
                    lights,
                    shadows_enabled: true,
                    shadow_filter: if self.pp_state.shadow_pcss {
                        ShadowFilter::Pcss
                    } else {
                        ShadowFilter::Pcf
                    },
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                let sg = self.pp_state.scene.version();
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::NormalMaps => {
                let items = self.nm_state.scene.collect_render_items(&Selection::new());
                if self.nm_state.clip_enabled {
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
                            colour: [1.0, 0.97, 0.93],
                            intensity: 2.0,
                            ..LightSource::default()
                        },
                    ],
                    shadows_enabled: true,
                    hemisphere_intensity: 0.4,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                let sg = self.nm_state.scene.version();
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::Shadows => {
                let items = self.shd_state.scene.collect_render_items(&Selection::new());
                let lighting = LightingSettings {
                    lights: vec![LightSource {
                        kind: LightKind::Directional {
                            direction: [0.5, 0.2, 1.2],
                        },
                        intensity: 2.0,
                        ..LightSource::default()
                    }],
                    shadows_enabled: true,
                    shadow_cascade_count: self.shd_state.cascade_count,
                    shadow_filter: if self.shd_state.pcss_on {
                        ShadowFilter::Pcss
                    } else {
                        ShadowFilter::Pcf
                    },
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                let sg = self.shd_state.scene.version();
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::Annotation => {
                let items = self.ann_state.scene.collect_render_items(&Selection::new());
                let sg = self.ann_state.scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::CameraTools => {
                let items = self.ct_state.scene.collect_render_items(&Selection::new());
                let sg = self.ct_state.scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::Lights => {
                let mut items = self
                    .lights_state
                    .scene
                    .collect_render_items(&Selection::new());
                if !self.lights_state.unlit_sphere {
                    items.retain(|item| !item.appearance.unlit);
                }
                let lighting = LightingSettings {
                    lights: self.lights_state.sources.clone(),
                    hemisphere_intensity: if self.lights_state.hemi_on {
                        self.lights_state.hemi_intensity
                    } else {
                        0.0
                    },
                    sky_colour: self.lights_state.sky_colour,
                    ground_colour: self.lights_state.ground_colour,
                    ..LightingSettings::default()
                };
                let sg = self.lights_state.scene.version();
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::MultiViewport => {
                unreachable!("MultiViewport is handled before build_frame_data")
            }
            ShowcaseMode::Isolines => {
                let mut items = self.iso_state.scene.collect_render_items(&Selection::new());
                // Apply scalar colouring or flat grey depending on toggle.
                if self.iso_state.show_surface_colour {
                    for item in items.iter_mut() {
                        item.active_attribute = Some(AttributeRef {
                            name: "wave".to_string(),
                            kind: AttributeKind::Vertex,
                        });
                        item.colourmap_id = Some(ColourmapId(BuiltinColourmap::Coolwarm as usize));
                        item.material.backface_policy = BackfacePolicy::Identical;
                    }
                } else {
                    for item in items.iter_mut() {
                        item.material.backface_policy = BackfacePolicy::Identical;
                    }
                }
                let sg = self.iso_state.scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::PointClouds => {
                let lighting = App::pc_lighting();
                (App::pc_surface_items(), Some(BG_COLOUR), lighting, 0, 0)
            }

            ShowcaseMode::Streamlines => {
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (vec![], Some(BG_COLOUR), lighting, 0, 0)
            }

            ShowcaseMode::Volume => {
                // Isosurface mesh items go into surface_items when visible.
                let surface_items =
                    if self.vol_state.mode != showcase_17_volume::VolumeMode::VolumeOnly {
                        self.make_iso_surface_item().into_iter().collect()
                    } else {
                        vec![]
                    };
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.6,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                (surface_items, Some(BG_COLOUR), lighting, 0, 0)
            }

            ShowcaseMode::ClipVolumes => {
                use showcase_18_clip_volumes::SceneMode;
                let items = if self.clipvol_state.scene_mode == SceneMode::Mesh {
                    let mut items = self
                        .clipvol_state
                        .scene
                        .collect_render_items(&Selection::new());
                    // Show inside faces when clipped so the cross-section is visible.
                    for item in items.iter_mut() {
                        item.material.backface_policy = BackfacePolicy::Identical;
                    }
                    items
                } else {
                    Vec::new()
                };
                let sg = self.clipvol_state.scene.version();
                let lighting = LightingSettings {
                    lights: vec![LightSource {
                        kind: LightKind::Directional {
                            direction: [0.5, 0.3, 1.2],
                        },
                        intensity: 1.8,
                        ..LightSource::default()
                    }],
                    hemisphere_intensity: 0.4,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::ScalarFields => {
                const ATTR_NAMES: [&str; 3] = ["height", "wave", "distance"];
                let mut items = self
                    .scalar_state
                    .scene
                    .collect_render_items(&self.scalar_state.selection);
                let colourmap_id = viewport_lib::ColourmapId(self.scalar_state.colourmap as usize);
                let active_node_id = self.scalar_state.node_ids[self.scalar_state.active_object];
                let wave_node_id = self.scalar_state.node_ids[1];
                if let Some(item) = items
                    .iter_mut()
                    .find(|item| item.pick_id == PickId(active_node_id))
                {
                    item.active_attribute = Some(viewport_lib::AttributeRef {
                        name: ATTR_NAMES[self.scalar_state.active_object].to_string(),
                        kind: viewport_lib::AttributeKind::Vertex,
                    });
                    item.colourmap_id = Some(colourmap_id);
                    item.scalar_range = if self.scalar_state.range_auto {
                        None
                    } else {
                        Some(self.scalar_state.range)
                    };
                    item.nan_colour = if self.scalar_state.nan_on {
                        Some([0.85, 0.1, 0.85, 1.0])
                    } else {
                        None
                    };
                }
                if let Some(item) = items
                    .iter_mut()
                    .find(|item| item.pick_id == PickId(wave_node_id))
                {
                    item.material.backface_policy = BackfacePolicy::Identical;
                }
                let sg = self.scalar_state.scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (
                    items,
                    Some(BG_COLOUR),
                    lighting,
                    sg,
                    self.scalar_state.selection.version(),
                )
            }

            ShowcaseMode::Matcap => {
                let items = self
                    .matcap_state
                    .scene
                    .collect_render_items(&Selection::new());
                let sg = self.matcap_state.scene.version();
                // Lighting is not used by matcap-shaded objects, but we still
                // need minimal settings for the framework.
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::Textures => {
                let mut items = self
                    .texture_state
                    .scene
                    .collect_render_items(&Selection::new());
                let plane_node = self.texture_state.plane_node;
                if let Some(item) = items.iter_mut().find(|i| i.pick_id == PickId(plane_node)) {
                    item.material.backface_policy = BackfacePolicy::Identical;
                }
                let sg = self.texture_state.scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 1.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::ParamVis => {
                let items = self
                    .param_vis_state
                    .scene
                    .collect_render_items(&Selection::new());
                let sg = self.param_vis_state.scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::GroundPlane => {
                let items = self.gp_state.scene.collect_render_items(&Selection::new());
                let sg = self.gp_state.scene.version();
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
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [0.3, 0.3, 0.3],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::FaceAttributes => {
                let mut items = self
                    .face_state
                    .scene
                    .collect_render_items(&Selection::new());
                let colourmap_id = ColourmapId(self.face_state.colourmap as usize);

                // Node 0: Vertex attribute (interpolated)
                // scalar_range left as None : renderer auto-detects from attribute_ranges.
                if let Some(item) = items
                    .iter_mut()
                    .find(|i| i.pick_id == PickId(self.face_state.node_ids[0]))
                {
                    item.active_attribute = Some(AttributeRef {
                        name: "scalar".to_string(),
                        kind: AttributeKind::Vertex,
                    });
                    item.colourmap_id = Some(colourmap_id);
                }

                // Node 1: Face attribute (flat per-triangle)
                // scalar_range left as None : renderer auto-detects from attribute_ranges.
                if let Some(item) = items
                    .iter_mut()
                    .find(|i| i.pick_id == PickId(self.face_state.node_ids[1]))
                {
                    item.active_attribute = Some(AttributeRef {
                        name: "scalar".to_string(),
                        kind: AttributeKind::Face,
                    });
                    item.colourmap_id = Some(colourmap_id);
                }

                // Node 2: FaceColour attribute (direct RGBA, no colourmap)
                if let Some(item) = items
                    .iter_mut()
                    .find(|i| i.pick_id == PickId(self.face_state.node_ids[2]))
                {
                    item.active_attribute = Some(AttributeRef {
                        name: "colour".to_string(),
                        kind: AttributeKind::FaceColour,
                    });
                    item.appearance.opacity = self.face_state.opacity;
                }

                let sg = self.face_state.scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.4,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::BackfacePolicy => {
                let items = self.sa_scene_items();
                let sg = self.sa_state.scene.version();
                (items, Some(BG_COLOUR), App::sa_lighting(), sg, 0)
            }

            ShowcaseMode::SurfaceVectors => {
                let surface_item = if self.sv_state.built {
                    vec![self.sv_surface_item()]
                } else {
                    vec![]
                };
                (surface_item, Some(BG_COLOUR), App::sv_lighting(), 0, 0)
            }

            ShowcaseMode::VolumeMesh => {
                // Per-frame CPU clip section: regenerate the clipped mesh slot
                // so section faces reflect the current plane position.
                if self.vm_state.clip_on && self.vm_state.built {
                    if let Some(rs) = frame.wgpu_render_state() {
                        let mut guard = rs.renderer.write();
                        if let Some(renderer) =
                            guard.callback_resources.get_mut::<ViewportRenderer>()
                        {
                            let data = self.vm_active_data();
                            let clip_planes = [self.vm_clip_plane()];
                            match self.vm_state.clipped_item.as_ref() {
                                None => {
                                    if let Ok((id, f2c)) =
                                        renderer.resources_mut().upload_clipped_volume_mesh_data(
                                            &rs.device,
                                            &data,
                                            &clip_planes,
                                        )
                                    {
                                        let mut item = viewport_lib::VolumeMeshItem::new(id, f2c);
                                        item.material.backface_policy =
                                            viewport_lib::BackfacePolicy::Identical;
                                        self.vm_state.clipped_item = Some(item);
                                    }
                                }
                                Some(existing) => {
                                    if let Ok(f2c) =
                                        renderer.resources_mut().replace_clipped_volume_mesh_data(
                                            &rs.device,
                                            &rs.queue,
                                            existing.mesh_id,
                                            &data,
                                            &clip_planes,
                                        )
                                    {
                                        if let Some(item) = self.vm_state.clipped_item.as_mut() {
                                            item.face_to_cell = f2c;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Rebuild PT meshes when the scalar field or colourmap changes.
                let pt_needs_rebuild = self.vm_state.built
                    && (self.vm_state.field != self.vm_state.pt_field
                        || self.vm_state.colourmap != self.vm_state.pt_colourmap);
                if pt_needs_rebuild {
                    if let Some(rs) = frame.wgpu_render_state() {
                        let mut guard = rs.renderer.write();
                        if let Some(renderer) =
                            guard.callback_resources.get_mut::<ViewportRenderer>()
                        {
                            self.rebuild_pt_meshes(
                                renderer,
                                &rs.device,
                                self.vm_state.field,
                                self.vm_state.colourmap,
                            );
                            self.vm_state.pt_field = self.vm_state.field;
                            self.vm_state.pt_colourmap = self.vm_state.colourmap;
                        }
                    }
                }

                let items = self.vm_scene_items();
                (items, Some(BG_COLOUR), App::vm_lighting(), 0, 0)
            }

            ShowcaseMode::Auxiliary => {
                let items = self.aux_state.scene.collect_render_items(&Selection::new());
                (items, Some(BG_COLOUR), App::aux_lighting(), 0, 0)
            }

            ShowcaseMode::CurveNetworkQuantities => {
                (vec![], Some(BG_COLOUR), LightingSettings::default(), 0, 0)
            }

            ShowcaseMode::DepthCompositeImages => (
                self.dc_scene_items(),
                Some(BG_COLOUR),
                App::dc_lighting(),
                0,
                0,
            ),

            ShowcaseMode::ImplicitSurface => (
                self.implicit_scene_items(),
                Some(BG_COLOUR),
                App::implicit_lighting(),
                self.mode_gen,
                0,
            ),

            ShowcaseMode::SparseVolumeGrid => (
                self.svg_scene_items(),
                Some(BG_COLOUR),
                App::svg_lighting(),
                self.mode_gen,
                0,
            ),

            ShowcaseMode::ExtendedQuantities => {
                let (items, glyphs, pcs) = self.eq_scene_items();
                eq_glyphs = glyphs;
                eq_pcs = pcs;
                (items, Some(BG_COLOUR), LightingSettings::default(), 0, 0)
            }

            ShowcaseMode::PickLevels => {
                let mut items = self
                    .pl_state
                    .scene
                    .collect_render_items(&self.pl_state.selection);
                // TVM boundary surface rendered as an opaque mesh alongside the scene.
                if let Some(tvm_mesh_id) = self.pl_state.tvm_mesh_id {
                    let mut tvm_item = SceneRenderItem::default();
                    tvm_item.mesh_id = tvm_mesh_id;
                    tvm_item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
                    tvm_item.pick_id = PickId(11);
                    tvm_item.selected = self.pl_state.selection.contains(11);
                    tvm_item.material = viewport_lib::Material::from_colour([0.8, 0.45, 0.2]);
                    items.push(tvm_item);
                }
                let sg = self.pl_state.scene.version();
                let lighting = LightingSettings {
                    lights: vec![
                        viewport_lib::LightSource {
                            kind: viewport_lib::LightKind::Directional {
                                direction: [0.4, 0.3, 1.5],
                            },
                            intensity: 0.7,
                            ..Default::default()
                        },
                        viewport_lib::LightSource {
                            kind: viewport_lib::LightKind::Directional {
                                direction: [-0.3, -0.2, -1.0],
                            },
                            intensity: 0.35,
                            ..Default::default()
                        },
                    ],
                    hemisphere_intensity: 0.7,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                (
                    items,
                    Some(BG_COLOUR),
                    lighting,
                    sg,
                    self.pl_state.selection.version(),
                )
            }

            ShowcaseMode::Labels => {
                let items = self.lbl_state.scene.collect_render_items(&Selection::new());
                let sg = self.lbl_state.scene.version();
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::Overlay => (
                Vec::new(),
                Some(BG_COLOUR),
                LightingSettings::default(),
                0,
                0,
            ),

            ShowcaseMode::PlaybackRuntime => {
                // Apply renderer settings and update deforming mesh.
                let topology_changed = self.pb_state.grid_resolution
                    != self.pb_state.last_grid_resolution
                    || self.pb_state.grid_layers != self.pb_state.last_grid_layers;
                let need_mesh_update =
                    self.pb_state.mode == RuntimeMode::Playback || topology_changed;
                if let Some(rs) = frame.wgpu_render_state() {
                    let mut guard = rs.renderer.write();
                    if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                        if need_mesh_update {
                            if let Some(mesh_id) = self.pb_state.mesh_id {
                                let t0 = std::time::Instant::now();
                                let mesh = showcase_36_playback_runtime::build_sine_grid(
                                    self.pb_state.grid_resolution,
                                    self.pb_state.grid_layers,
                                    self.pb_state.time,
                                );
                                if topology_changed {
                                    let _ = renderer
                                        .resources_mut()
                                        .replace_mesh_data(&rs.device, &rs.queue, mesh_id, &mesh);
                                    self.pb_state.last_grid_resolution =
                                        self.pb_state.grid_resolution;
                                    self.pb_state.last_grid_layers = self.pb_state.grid_layers;
                                } else {
                                    let _ = renderer.resources_mut().write_mesh_positions_normals(
                                        &rs.queue,
                                        mesh_id,
                                        &mesh.positions,
                                        &mesh.normals,
                                    );
                                }
                                self.pb_state.upload_ms = t0.elapsed().as_secs_f32() * 1000.0;
                            }
                        } else {
                            self.pb_state.upload_ms = 0.0;
                        }
                        renderer.set_runtime_mode(self.pb_state.mode);
                        renderer.set_performance_policy(self.pb_state.policy);
                        if !self.pb_state.policy.allow_dynamic_resolution {
                            renderer.set_render_scale(self.pb_state.manual_render_scale);
                        }
                        self.pb_state.last_stats = renderer.last_frame_stats();
                    }
                }

                // Update rolling stats history.
                self.pb_state
                    .stats_history
                    .push_back(self.pb_state.last_stats.total_frame_ms);
                if self.pb_state.stats_history.len() > 60 {
                    self.pb_state.stats_history.pop_front();
                }

                let items = showcase_36_playback_runtime::pb_scene_items(self);
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [1.0, 1.0, 1.0],
                    ..LightingSettings::default()
                };
                let sg = self.pb_state.scene.version();
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::ProbeWidgets => {
                let items = self.pw_state.scene.collect_render_items(&Selection::new());
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.6,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                let sg = self.pw_state.scene.version();
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::SurfaceLIC => {
                let items = self.lic_state.scene.collect_render_items(&Selection::new());
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.5,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [0.7, 0.7, 0.7],
                    ..LightingSettings::default()
                };
                let sg = self.lic_state.scene.version();
                (items, Some(BG_COLOUR), lighting, sg, 0)
            }

            ShowcaseMode::TensorGlyphs => {
                let lighting = LightingSettings {
                    hemisphere_intensity: 0.6,
                    sky_colour: [1.0, 1.0, 1.0],
                    ground_colour: [0.8, 0.8, 0.8],
                    ..LightingSettings::default()
                };
                let items = showcase_39_tensor_glyphs::beam_scene_items(self);
                (items, Some(BG_COLOUR), lighting, 0, 0)
            }

            ShowcaseMode::VertexWarp => {
                let items = showcase_40_vertex_warp::warp_scene_items(self);
                (
                    items,
                    Some(BG_COLOUR),
                    showcase_40_vertex_warp::warp_lighting(),
                    0,
                    0,
                )
            }

            ShowcaseMode::Sprites => {
                let items = showcase_41_sprites::sprite_scene_items(self);
                (
                    items,
                    Some(BG_COLOUR),
                    showcase_41_sprites::sprite_lighting(),
                    0,
                    0,
                )
            }

            ShowcaseMode::GaussianSplats => {
                (vec![], Some(BG_COLOUR), LightingSettings::default(), 0, 0)
            }
        };

        // Gizmo matrices for Interaction and ClipVolumes modes.
        let (gizmo_model, gizmo_mode, gizmo_space_orient, gizmo_hovered) =
            if self.mode == ShowcaseMode::Interaction {
                let center = self.interact_state.gizmo_center;
                let model = center.map(|c| {
                    glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(self.interact_state.gizmo_scale),
                        glam::Quat::IDENTITY,
                        c,
                    )
                });
                let orient = gizmo_helpers::gizmo_orientation(
                    &self.interact_state.gizmo,
                    &self.interact_state.selection,
                    &self.interact_state.scene,
                );
                let hovered = if let Some(state) = self.interact_state.manip.state() {
                    state.axis.unwrap_or(GizmoAxis::None)
                } else {
                    self.interact_state.gizmo.hovered_axis
                };
                (model, self.interact_state.gizmo.mode, orient, hovered)
            } else if self.mode == ShowcaseMode::ClipVolumes && self.clipvol_state.built {
                let center = self.clipvol_state.gizmo_center;
                let orient = self.clipvol_gizmo_orient();
                let model = center.map(|c| {
                    glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(self.clipvol_state.gizmo_scale),
                        glam::Quat::IDENTITY,
                        c,
                    )
                });
                let hovered = if self.clipvol_state.gizmo.active_axis != GizmoAxis::None {
                    self.clipvol_state.gizmo.active_axis
                } else {
                    self.clipvol_state.gizmo.hovered_axis
                };
                (model, self.clipvol_state.gizmo.mode, orient, hovered)
            } else {
                (
                    None,
                    GizmoMode::Translate,
                    glam::Quat::IDENTITY,
                    GizmoAxis::None,
                )
            };

        let mut fd = FrameData::new(
            CameraFrame::from_camera(&self.camera, [w, h]).with_pixels_per_point(pixels_per_point),
            if let Some(arc) = perf_arc {
                SceneFrame::from_shared_items(arc, scene_gen)
            } else {
                SceneFrame::from_surface_items(scene_items)
            },
        );
        fd.effects.lighting = lighting;
        if self.mode == ShowcaseMode::PickLevels {
            fd.viewport.wireframe_mode = self.pl_state.wireframe;
        }
        if self.mode == ShowcaseMode::VolumeMesh {
            fd.viewport.wireframe_mode = self.vm_state.wireframe;
        }
        fd.viewport.show_grid = false;
        fd.viewport.show_axes_indicator = true;
        fd.viewport.background_colour = bg_colour;

        // Ground plane (Showcase 23).
        if self.mode == ShowcaseMode::GroundPlane {
            use showcase_23_ground_plane::GpMode;
            fd.effects.ground_plane = GroundPlane {
                mode: match self.gp_state.mode {
                    GpMode::None => GroundPlaneMode::None,
                    GpMode::ShadowOnly => GroundPlaneMode::ShadowOnly,
                    GpMode::Tile => GroundPlaneMode::Tile,
                    GpMode::SolidColour => GroundPlaneMode::SolidColour,
                },
                height: self.gp_state.height,
                colour: self.gp_state.colour,
                tile_size: self.gp_state.tile_size,
                shadow_colour: self.gp_state.shadow_colour,
                shadow_opacity: self.gp_state.shadow_opacity,
            };
        }
        // Clip objects for Showcase 24 (Surface Appearance).
        if self.mode == ShowcaseMode::BackfacePolicy {
            adv_clip_objects.extend(self.sa_clip_objects());
        }
        if self.mode == ShowcaseMode::VolumeMesh {
            adv_clip_objects.extend(self.vm_clip_objects());
        }
        fd.effects.clip_objects = adv_clip_objects;
        if self.mode == ShowcaseMode::NormalMaps {
            fd.effects.cap_fill_enabled = self.nm_state.cap_fill;
        }
        // Showcase 24 exists to show back face policies : cap fill would hide them.
        if self.mode == ShowcaseMode::BackfacePolicy {
            fd.effects.cap_fill_enabled = false;
        }
        if self.mode == ShowcaseMode::VolumeMesh {
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
            || (self.mode == ShowcaseMode::ScalarFields && !self.scalar_state.selection.is_empty())
            || (self.mode == ShowcaseMode::PickLevels && !self.pl_state.selection.is_empty())
            || (self.mode == ShowcaseMode::PickLevels
                && self
                    .pl_state
                    .sub_selection
                    .iter()
                    .any(|(id, sub)| match sub {
                        viewport_lib::SubObjectRef::Instance(_) => matches!(*id, 31 | 32 | 33 | 34),
                        viewport_lib::SubObjectRef::Point(_) => *id == 100,
                        viewport_lib::SubObjectRef::Splat(_) => *id == 10,
                        _ => false,
                    }));
        if scene_graph_outline {
            fd.interaction.outline_width_px = scene_graph_outline_width;
        }
        if self.mode == ShowcaseMode::PickLevels {
            fd.interaction.outline_colour = [1.0, 0.85, 0.0, 1.0];
        }
        fd.interaction.xray_selected = adv_xray;
        fd.scene.generation = scene_gen;
        fd.interaction.selection_generation = sel_gen;

        // Transparent volume mesh (Showcase 26) : submitted every frame when transparent mode is on.
        if self.mode == ShowcaseMode::VolumeMesh {
            if let Some(item) = self.vm_transparent_item() {
                fd.scene.transparent_volume_meshes.push(item);
                // OIT runs inside the HDR path; enable post-processing to activate it.
                fd.effects.post_process.enabled = true;
            }
        }

        // Volume item (Showcase 17) : submitted every frame when in volume mode.
        if self.mode == ShowcaseMode::Volume
            && self.vol_state.built
            && self.vol_state.mode != showcase_17_volume::VolumeMode::IsosurfaceOnly
        {
            if let Some(vol_item) = self.make_volume_item() {
                fd.scene.volumes.push(vol_item);
            }
        }

        // Image slice (Showcase 17) : submitted every frame when enabled.
        if self.mode == ShowcaseMode::Volume && self.vol_state.built && self.vol_state.show_slice {
            if let Some(slice_item) = self.make_image_slice_item() {
                fd.scene.image_slices.push(slice_item);
            }
        }

        // Volume surface slice (Showcase 17) : submitted every frame when enabled.
        if self.mode == ShowcaseMode::Volume
            && self.vol_state.built
            && self.vol_state.show_surface_slice
        {
            if let Some(item) = self.make_volume_surface_slice_item() {
                fd.scene.volume_surface_slices.push(item);
            }
        }

        // Clip volume (Showcase 18) : set every frame from current state.
        if self.mode == ShowcaseMode::ClipVolumes && self.clipvol_state.built {
            fd.effects.clip_objects.extend(self.make_clip_objects());
            // Volume mode: ray-march the density field with the same clip objects applied.
            if self.clipvol_state.scene_mode == showcase_18_clip_volumes::SceneMode::Volume {
                if let Some(vol) = self.make_clipvol_volume_item() {
                    fd.scene.volumes.push(vol);
                }
            }
        }

        // Streamline / tube items (Showcase 16) : submitted every frame.
        if self.mode == ShowcaseMode::Streamlines && self.stream_state.built {
            use showcase_16_streamlines::StreamRenderMode;
            match self.stream_state.render_mode {
                StreamRenderMode::Polylines => {
                    fd.scene.polylines.push(self.make_stream_polyline_item());
                }
                StreamRenderMode::Streamtube => {
                    fd.scene.streamtube_items.push(self.make_stream_tube_item());
                }
                StreamRenderMode::GeneralTube => {
                    fd.scene
                        .tube_items
                        .push(self.make_stream_general_tube_item());
                }
                StreamRenderMode::Ribbon => {
                    fd.scene.ribbon_items.push(self.make_stream_ribbon_item());
                }
            }
        }

        // Spline widget polyline + handles (Showcase 4) : submitted every frame.
        if self.mode == ShowcaseMode::Interaction && self.interact_state.built {
            fd.scene
                .polylines
                .push(self.interact_state.spline.polyline_item(9900));
            let render_cam = CameraFrame::from_camera(&self.camera, [w, h]).render_camera;
            let spline_ctx = viewport_lib::WidgetContext {
                camera: render_cam,
                viewport_size: glam::Vec2::new(w, h),
                cursor_viewport: self.interact_state.last_cursor_viewport,
                drag_started: false,
                dragging: false,
                released: false,
                double_clicked: false,
            };
            fd.scene
                .glyphs
                .push(self.interact_state.spline.handle_glyphs(9901, &spline_ctx));
        }

        // Surface vector glyphs (Showcase 25) : submitted every frame.
        if self.mode == ShowcaseMode::SurfaceVectors && self.sv_state.built {
            fd.scene.glyphs.push(self.sv_glyph_item());
        }

        // Extended quantity glyphs and point clouds (Showcase 32) : submitted every frame.
        if self.mode == ShowcaseMode::ExtendedQuantities && self.eq_state.built {
            fd.scene.glyphs.extend(eq_glyphs);
            fd.scene.point_clouds.extend(eq_pcs);
        }

        // Picking Levels (Showcase 33) : point cloud, sub-selection highlights, and hit marker.
        if self.mode == ShowcaseMode::PickLevels && self.pl_state.built {
            // Background point cloud (blue). Id=100 avoids collision with scene
            // NodeIds (which start at 1 and cover the 4 mesh objects here).
            if !self.pl_state.pc_positions.is_empty() {
                let mut pc = PointCloudItem::default();
                pc.positions = self.pl_state.pc_positions.clone();
                pc.point_size = 18.0;
                pc.default_colour = [0.5, 0.8, 1.0, 1.0];
                pc.id = 100;
                pc.selected = self.pl_state.selection.contains(100);
                fd.scene.point_clouds.push(pc);
            }
            // Gaussian splat grid (pickable via pick_id=10).
            if let Some(splat_id) = self.pl_state.splat_id {
                let mut item = GaussianSplatItem::default();
                item.id = splat_id;
                item.model = showcase_33_picking_levels::pl_splat_model().to_cols_array_2d();
                item.pick_id = 10;
                item.selected = self.pl_state.selection.contains(10);
                fd.scene.gaussian_splats.push(item);
            }
            // Hex cylinder: transparent volume mesh (pick_id=12).
            // Rendered via OIT; boundary mesh used for the selection outline.
            if let (Some(tet_id), Some(tet_data)) = (
                self.pl_state.tvm_tet_id,
                self.pl_state.tvm_tet_data.as_ref(),
            ) {
                let mut tvm = TransparentVolumeMeshItem::new(tet_id);
                tvm.pick_id = 12;
                tvm.volume_mesh_data = Some(tet_data.clone());
                tvm.selected = self.pl_state.selection.contains(12);
                tvm.boundary_mesh_id = self.pl_state.tvm_tet_mesh_id;
                tvm.density = 2.0;
                fd.scene.transparent_volume_meshes.push(tvm);
            }
            // TVM capsule (pick_id=11) submitted as VolumeMeshItem so renderer.pick()
            // with PointLike mask can return Cell sub_objects for it via face_to_cell.
            if let Some(mesh_id) = self.pl_state.tvm_mesh_id {
                if !self.pl_state.tvm_face_to_cell.is_empty() {
                    let mut item =
                        VolumeMeshItem::new(mesh_id, self.pl_state.tvm_face_to_cell.clone());
                    item.pick_id = PickId(11);
                    fd.scene.volume_mesh_items.push(item);
                }
            }
            // Sub-object highlight pass (face fill, edge outline, vertex/point sprites).
            if !self.pl_state.sub_selection.is_empty() {
                let mut mesh_lookup: HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)> = HashMap::new();
                let mut model_matrices: HashMap<u64, glam::Mat4> = HashMap::new();
                for node in self.pl_state.scene.nodes() {
                    let node_id = viewport_lib::traits::ViewportObject::id(node);
                    let model = viewport_lib::traits::ViewportObject::model_matrix(node);
                    model_matrices.insert(node_id, model);
                    if let Some(mesh_key) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                        if let Some(data) = self.pl_state.mesh_lookup.get(&mesh_key) {
                            mesh_lookup.insert(node_id, data.clone());
                        }
                    }
                }
                let mut point_positions: HashMap<u64, Vec<[f32; 3]>> = HashMap::new();
                point_positions.insert(100, self.pl_state.pc_positions.clone());
                let mut voxel_lookup: HashMap<u64, viewport_lib::VolumeSelectionInfo> =
                    HashMap::new();
                if self.pl_state.volume_id.is_some() {
                    voxel_lookup.insert(
                        20,
                        viewport_lib::VolumeSelectionInfo {
                            dims: [16, 16, 16],
                            bbox_min: [0.0, 0.0, 0.0],
                            bbox_max: [4.0, 4.0, 4.0],
                            model: glam::Mat4::from_translation(glam::vec3(-2.0, -1.0, -6.0))
                                .to_cols_array_2d(),
                        },
                    );
                }
                let mut cell_lookup: HashMap<u64, CellSelectionInfo> = HashMap::new();
                if let Some(tvm_data) = &self.pl_state.tvm_data {
                    cell_lookup.insert(
                        11,
                        CellSelectionInfo {
                            positions: tvm_data.positions.clone(),
                            cells: tvm_data.cells.clone(),
                        },
                    );
                }
                if let Some(tet_data) = &self.pl_state.tvm_tet_data {
                    cell_lookup.insert(
                        12,
                        CellSelectionInfo {
                            positions: tet_data.positions.clone(),
                            cells: tet_data.cells.clone(),
                        },
                    );
                }
                let mut polyline_lookup: HashMap<u64, PolylineSelectionInfo> = HashMap::new();
                if !self.pl_state.polyline_positions.is_empty() {
                    polyline_lookup.insert(
                        30,
                        PolylineSelectionInfo {
                            positions: self.pl_state.polyline_positions.clone(),
                            strip_lengths: self.pl_state.polyline_strip_lengths.clone(),
                        },
                    );
                }
                let mut curve_family_lookup: HashMap<u64, PolylineSelectionInfo> = HashMap::new();
                if !self.pl_state.streamtube_positions.is_empty() {
                    curve_family_lookup.insert(
                        40,
                        PolylineSelectionInfo {
                            positions: self.pl_state.streamtube_positions.clone(),
                            strip_lengths: self.pl_state.streamtube_strip_lengths.clone(),
                        },
                    );
                }
                if !self.pl_state.tube_positions.is_empty() {
                    curve_family_lookup.insert(
                        41,
                        PolylineSelectionInfo {
                            positions: self.pl_state.tube_positions.clone(),
                            strip_lengths: self.pl_state.tube_strip_lengths.clone(),
                        },
                    );
                }
                if !self.pl_state.ribbon_positions.is_empty() {
                    curve_family_lookup.insert(
                        42,
                        PolylineSelectionInfo {
                            positions: self.pl_state.ribbon_positions.clone(),
                            strip_lengths: self.pl_state.ribbon_strip_lengths.clone(),
                        },
                    );
                }
                fd.interaction.sub_selection = Some(
                    SubSelectionRef::new(
                        &self.pl_state.sub_selection,
                        mesh_lookup,
                        model_matrices,
                        point_positions,
                    )
                    .with_voxels(voxel_lookup)
                    .with_cells(cell_lookup)
                    .with_polylines(polyline_lookup)
                    .with_curve_families(curve_family_lookup),
                );
                fd.interaction.sub_highlight_face_fill_colour = [1.0, 0.85, 0.0, 0.25];
                fd.interaction.sub_highlight_edge_colour = [1.0, 0.85, 0.0, 1.0];
                fd.interaction.sub_highlight_edge_width_px = 5.0;
                fd.interaction.sub_highlight_vertex_size_px = 14.0;
            }
            // Orange crosshair marker at the click point.
            if self.pl_state.show_hit_marker {
                if let Some(marker_pos) = self.pl_state.hit_marker {
                    let mut marker = PointCloudItem::default();
                    marker.positions = vec![marker_pos.to_array()];
                    marker.point_size = 16.0;
                    marker.default_colour = [1.0, 0.35, 0.0, 1.0];
                    fd.scene.point_clouds.push(marker);
                }
            }
            // Volume render.
            if let Some(vol_id) = self.pl_state.volume_id {
                let mut vol = viewport_lib::VolumeItem::default();
                vol.volume_id = vol_id;
                vol.model =
                    glam::Mat4::from_translation(glam::vec3(-2.0, -1.0, -6.0)).to_cols_array_2d();
                vol.bbox_min = [0.0, 0.0, 0.0];
                vol.bbox_max = [4.0, 4.0, 4.0];
                vol.scalar_range = (0.0, 1.0);
                vol.threshold_min = 0.15;
                vol.threshold_max = 1.0;
                vol.opacity_scale = 0.6;
                vol.enable_shading = true;
                vol.selected = self.pl_state.selection.contains(20);
                vol.pick_id = 20;
                vol.volume_data = self
                    .pl_state
                    .volume_data
                    .as_ref()
                    .map(|d| std::sync::Arc::new(d.clone()));
                fd.scene.volumes.push(vol);
            }
            // Polyline: 3 strips (pick_id=30).
            if !self.pl_state.polyline_positions.is_empty() {
                let mut pl = PolylineItem::default();
                pl.positions = self.pl_state.polyline_positions.clone();
                pl.strip_lengths = self.pl_state.polyline_strip_lengths.clone();
                pl.default_colour = [0.2, 0.85, 0.35, 1.0];
                pl.line_width = 3.0;
                pl.id = 30;
                pl.selected = self.pl_state.selection.contains(30);
                fd.scene.polylines.push(pl);
            }
            // Arrow glyphs (pick_id=31).
            if !self.pl_state.arrow_glyph_positions.is_empty() {
                let n = self.pl_state.arrow_glyph_positions.len();
                let mut g = GlyphItem::default();
                g.positions = self.pl_state.arrow_glyph_positions.clone();
                g.vectors = vec![[0.0, 0.0, 1.0]; n];
                g.scale = 0.8;
                g.scale_by_magnitude = false;
                g.use_default_colour = true;
                g.default_colour = [0.75, 0.1, 1.0, 1.0];
                g.glyph_type = GlyphType::Arrow;
                g.id = 31;
                g.selected = self.pl_state.selection.contains(31);
                fd.scene.glyphs.push(g);
            }
            // Tensor glyphs (pick_id=32).
            if !self.pl_state.tensor_glyph_positions.is_empty() {
                let mut tg = TensorGlyphItem::default();
                tg.positions = self.pl_state.tensor_glyph_positions.clone();
                tg.eigenvalues = self.pl_state.tensor_glyph_eigenvalues.clone();
                tg.eigenvectors = self.pl_state.tensor_glyph_eigenvectors.clone();
                tg.scale = 0.5;
                tg.colourmap_id = Some(ColourmapId(BuiltinColourmap::Coolwarm as usize));
                tg.id = 32;
                tg.selected = self.pl_state.selection.contains(32);
                fd.scene.tensor_glyphs.push(tg);
            }
            // Sprites (pick_id=33).
            if !self.pl_state.sprite_positions.is_empty() {
                let mut s = SpriteItem::default();
                s.positions = self.pl_state.sprite_positions.clone();
                s.sizes = self.pl_state.sprite_sizes.clone();
                s.colours = self.pl_state.sprite_colours.clone();
                s.default_colour = [1.0, 0.90, 0.20, 1.0];
                s.default_size = 28.0;
                s.depth_write = true;
                s.id = 33;
                s.selected = self.pl_state.selection.contains(33);
                fd.scene.sprite_items.push(s);
            }
            if !self.pl_state.xo_sprite_positions.is_empty() {
                let mut s = SpriteItem::default();
                s.positions = self.pl_state.xo_sprite_positions.clone();
                s.sizes = self.pl_state.xo_sprite_sizes.clone();
                s.colours = self.pl_state.xo_sprite_colours.clone();
                s.default_colour = [0.5, 0.5, 1.0, 1.0];
                s.default_size = 30.0;
                s.depth_write = true;
                s.id = 34;
                s.selected = self.pl_state.selection.contains(34);
                fd.scene.sprite_items.push(s);
            }
            // Streamtube (pick_id=40).
            if !self.pl_state.streamtube_positions.is_empty() {
                let mut st = StreamtubeItem::default();
                st.positions = self.pl_state.streamtube_positions.clone();
                st.strip_lengths = self.pl_state.streamtube_strip_lengths.clone();
                st.radius = 0.12;
                st.colour = [0.3, 0.8, 0.55, 1.0];
                st.id = 40;
                st.selected = self.pl_state.selection.contains(40);
                fd.scene.streamtube_items.push(st);
            }
            // Tube (pick_id=41).
            if !self.pl_state.tube_positions.is_empty() {
                let mut tb = TubeItem::default();
                tb.positions = self.pl_state.tube_positions.clone();
                tb.strip_lengths = self.pl_state.tube_strip_lengths.clone();
                tb.radius = 0.15;
                tb.colour = [0.85, 0.4, 0.2, 1.0];
                tb.id = 41;
                tb.selected = self.pl_state.selection.contains(41);
                fd.scene.tube_items.push(tb);
            }
            // Ribbon (pick_id=42).
            if !self.pl_state.ribbon_positions.is_empty() {
                let mut rb = RibbonItem::default();
                rb.positions = self.pl_state.ribbon_positions.clone();
                rb.strip_lengths = self.pl_state.ribbon_strip_lengths.clone();
                rb.width = 0.4;
                rb.colour = [0.6, 0.3, 0.9, 1.0];
                rb.id = 42;
                rb.selected = self.pl_state.selection.contains(42);
                fd.scene.ribbon_items.push(rb);
            }
            // Volume surface slice (pick_id=51): plane mesh coloured by the same volume.
            // Plane mesh is a 3.6x3.6 XZ quad; model places it at (0, 1, -4) and tilts it
            // 60 degrees around X so the plane makes a 30 degree angle with the XY plane.
            if let (Some(vol_id), Some(mesh_id)) =
                (self.pl_state.volume_id, self.pl_state.surface_slice_mesh_id)
            {
                let mut item = VolumeSurfaceSliceItem::default();
                item.volume_id = vol_id;
                item.mesh_id = mesh_id;
                item.bbox_min = [-2.0, -1.0, -6.0];
                item.bbox_max = [2.0, 3.0, -2.0];
                item.scalar_range = (0.0, 1.0);
                item.model = (glam::Mat4::from_translation(glam::vec3(0.0, 1.0, -4.0))
                    * glam::Mat4::from_rotation_x(60_f32.to_radians()))
                .to_cols_array_2d();
                item.id = 51;
                item.selected = self.pl_state.selection.contains(51);
                fd.scene.volume_surface_slices.push(item);
            }
            // Screen image (pick_id=52): small checkerboard pinned to the top-right corner.
            {
                let iw = 48u32;
                let ih = 48u32;
                let pixels: Vec<[u8; 4]> = (0..iw * ih)
                    .map(|i| {
                        let x = i % iw;
                        let y = i / iw;
                        if (x / 6 + y / 6) % 2 == 0 {
                            [255, 200, 50, 220]
                        } else {
                            [80, 50, 200, 220]
                        }
                    })
                    .collect();
                let mut img = ScreenImageItem::default();
                img.pixels = pixels;
                img.width = iw;
                img.height = ih;
                img.anchor = ImageAnchor::TopRight;
                img.scale = 2.0;
                img.id = 52;
                img.selected = self.pl_state.selection.contains(52);
                fd.scene.screen_images.push(img);
            }
            // GPU implicit (pick_id=53): two smooth-blended spheres at (13, 0, 0) / (15, 0, 0).
            {
                let centers: [[f32; 3]; 2] = [[13.0, 0.0, 0.0], [15.0, 0.0, 0.0]];
                let colours: [[f32; 4]; 2] = [[0.9, 0.4, 0.2, 1.0], [0.2, 0.5, 1.0, 1.0]];
                let mut item = GpuImplicitItem::default();
                for i in 0..2 {
                    let mut prim = ImplicitPrimitive::zeroed();
                    prim.kind = 1; // sphere
                    prim.blend = 0.8;
                    prim.params[0] = centers[i][0];
                    prim.params[1] = centers[i][1];
                    prim.params[2] = centers[i][2];
                    prim.params[3] = 1.2; // radius
                    prim.colour = colours[i];
                    item.primitives.push(prim);
                }
                item.blend_mode = ImplicitBlendMode::SmoothUnion;
                item.march_options = GpuImplicitOptions {
                    max_steps: 64,
                    step_scale: 0.9,
                    hit_threshold: 1e-3,
                    max_distance: self.camera.zfar,
                };
                item.id = 53;
                item.selected = self.pl_state.selection.contains(53);
                fd.scene.gpu_implicit.push(item);
            }
            // GPU marching cubes (pick_id=54): gyroid surface centered near (13, 0, -5).
            if let Some(mc_vol_id) = self.pl_state.mc_volume_id {
                let mut mat = viewport_lib::Material::from_colour([0.5, 0.8, 0.55]);
                mat.roughness = 0.4;
                fd.scene.gpu_mc_jobs.push(GpuMarchingCubesJob {
                    volume_id: mc_vol_id,
                    isovalue: 0.0,
                    material: mat,
                    id: 54,
                    appearance: Default::default(),
                    selected: self.pl_state.selection.contains(54),
                    cpu_data: self.pl_state.mc_volume_data.clone(),
                });
            }
        }

        // Curve network quantities (Showcase 28) : submitted every frame.
        if self.mode == ShowcaseMode::CurveNetworkQuantities {
            fd.scene
                .polylines
                .push(showcase_28_curve_network_quantities::make_cnq_polyline_item(self));
        }

        // Depth-composite screen image (Showcase 29) : submitted every frame.
        if self.mode == ShowcaseMode::DepthCompositeImages {
            self.dc_push_screen_image(&mut fd);
        }

        // Implicit surface (Showcase 30) : CPU sphere-march, GPU implicit, or GPU MC — re-submitted every frame.
        if self.mode == ShowcaseMode::ImplicitSurface {
            self.push_implicit_screen_image(&mut fd, w as u32, h as u32);
            self.push_gpu_implicit(&mut fd);
            self.push_gpu_mc_job(&mut fd);
        }

        // Auxiliary frustums and screen images (Showcase 27) : submitted every frame.
        if self.mode == ShowcaseMode::Auxiliary && self.aux_state.built {
            fd.scene.camera_frustums = self.aux_state.frustums.clone();
            self.aux_push_screen_images(&mut fd);
        }

        // Point cloud / glyph items (Showcase 15) : submitted every frame.
        if self.mode == ShowcaseMode::PointClouds && self.pc_state.built {
            use showcase_15_point_clouds::PcSubMode;
            match self.pc_state.sub_mode {
                PcSubMode::PointCloud => {
                    fd.scene.point_clouds.push(self.make_pc_point_cloud_item());
                }
                PcSubMode::VectorField => {
                    fd.scene.glyphs.push(self.make_pc_glyph_item());
                }
                PcSubMode::PointGaussian => {
                    fd.scene.point_clouds.push(self.make_pc_gaussian_item());
                }
            }
            if self.pc_state.ssao_enabled {
                fd.effects.post_process = PostProcessSettings {
                    enabled: true,
                    ssao: true,
                    ..PostProcessSettings::default()
                };
            }
        }

        // Isoline items (Showcase 14) : submitted every frame with current settings.
        if self.mode == ShowcaseMode::Isolines && self.iso_state.built {
            let scalar_min = self
                .iso_state
                .scalars
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let scalar_max = self
                .iso_state
                .scalars
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let range = scalar_max - scalar_min;
            let isovalues: Vec<f32> = (0..self.iso_state.contour_count)
                .map(|i| {
                    scalar_min
                        + range * (i as f32 + 1.0) / (self.iso_state.contour_count as f32 + 1.0)
                })
                .collect();
            let mut iso_item = IsolineItem::default();
            iso_item.positions = self.iso_state.positions.clone();
            iso_item.indices = self.iso_state.indices.clone();
            iso_item.scalars = self.iso_state.scalars.clone();
            iso_item.isovalues = isovalues;
            iso_item.colour = self.iso_state.line_colour;
            iso_item.line_width = self.iso_state.line_width;
            iso_item.depth_bias = self.iso_state.depth_bias;
            fd.scene.isolines.push(iso_item);
        }

        // Post-process settings (Showcases 6–8).
        // Note: the full HDR pipeline uses renderer.render() which requires direct
        // surface access. In the eframe callback model we use prepare()+paint(),
        // so the post-process pass is not applied. Settings are stored for reference.
        match self.mode {
            ShowcaseMode::PostProcess => {
                // Cap far plane for better cascade distribution, but track orbit
                // distance so the scene doesn't disappear when zooming out.
                let mut rc = RenderCamera::from_camera(&self.camera);
                rc.far = (self.camera.distance * 3.0).max(60.0);
                rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                fd.camera.render_camera = rc;
                if self.pp_state.dof_enabled {
                    fd.effects.post_process = PostProcessSettings {
                        enabled: true,
                        dof_enabled: true,
                        dof_focal_distance: self.pp_state.dof_focal_dist,
                        dof_focal_range: self.pp_state.dof_focal_range,
                        dof_max_blur_radius: self.pp_state.dof_max_blur,
                        ..PostProcessSettings::default()
                    };
                }
            }
            ShowcaseMode::Shadows => {
                fd.effects.post_process = PostProcessSettings {
                    enabled: false,
                    contact_shadows: self.shd_state.contact_on,
                    contact_shadow_max_distance: 0.18,
                    contact_shadow_steps: 32,
                    contact_shadow_thickness: 0.04,
                    ..PostProcessSettings::default()
                };
                // Cap far plane for better cascade distribution, but track orbit
                // distance so the scene doesn't disappear when zooming out.
                let mut rc = RenderCamera::from_camera(&self.camera);
                rc.far = (self.camera.distance * 3.0).max(60.0);
                rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                fd.camera.render_camera = rc;
            }
            ShowcaseMode::NormalMaps => {
                fd.effects.post_process = PostProcessSettings {
                    enabled: false,
                    ..PostProcessSettings::default()
                };
                // Cap far plane for better cascade distribution, but track orbit
                // distance so the scene doesn't disappear when zooming out.
                let mut rc = RenderCamera::from_camera(&self.camera);
                rc.far = (self.camera.distance * 3.0).max(60.0);
                rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                fd.camera.render_camera = rc;
            }
            ShowcaseMode::Auxiliary => {
                // Cap far plane for better cascade distribution, but track orbit
                // distance so the scene doesn't disappear when zooming out.
                let mut rc = RenderCamera::from_camera(&self.camera);
                rc.far = (self.camera.distance * 3.0).max(60.0);
                rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                fd.camera.render_camera = rc;
            }
            ShowcaseMode::Lights => {
                // Cap the far plane so depth values span a useful range for EDL.
                // Without this, all scene geometry clusters near depth 0.99, making
                // the log-space neighbor differences too small to see.
                let mut rc = RenderCamera::from_camera(&self.camera);
                rc.far = (self.camera.distance * 3.0).max(30.0);
                rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
                fd.camera.render_camera = rc;
                if self.lights_state.edl_enabled {
                    fd.effects.post_process = PostProcessSettings {
                        enabled: true,
                        edl_enabled: true,
                        edl_radius: self.lights_state.edl_radius,
                        edl_strength: self.lights_state.edl_strength,
                        ..PostProcessSettings::default()
                    };
                }
            }
            ShowcaseMode::BackfacePolicy => {}
            _ => {}
        }

        // Overlay labels (Showcase 9 and 34): populate OverlayFrame.
        if self.mode == ShowcaseMode::Annotation && self.ann_state.built {
            fd.overlays.labels = self.ann_state.labels.clone();
        }
        if self.mode == ShowcaseMode::ScalarFields && self.scalar_state.built {
            fd.overlays.scalar_bars = vec![self.scalar_bar_item()];
        }
        if self.mode == ShowcaseMode::Overlay {
            let (labels, bar, ruler) = showcase_35_overlay::build_overlay_frame(self);
            fd.overlays.labels = labels;
            fd.overlays.scalar_bars = vec![bar];
            if let Some(r) = ruler {
                fd.overlays.rulers = vec![r];
            }
            if self.ovl_state.cloud_built {
                let mut pc = PointCloudItem::default();
                pc.positions = self.ovl_state.cloud_positions.clone();
                pc.scalars = self.ovl_state.cloud_scalars.clone();
                pc.scalar_range = Some((-1.5, 1.5));
                pc.colourmap_id = Some(ColourmapId(self.ovl_state.colourmap as usize));
                pc.point_size = 4.0;
                fd.scene.point_clouds.push(pc);
            }
        }
        if self.mode == ShowcaseMode::Labels && self.lbl_state.built {
            // World-anchored part labels (built once, filtered by toggle).
            if self.lbl_state.show_part_labels {
                fd.overlays
                    .labels
                    .extend(self.lbl_state.labels.iter().cloned());
            }
            // Screen-anchored labels (title, legend, feature demos) sized to viewport.
            fd.overlays
                .labels
                .extend(self.build_label_screen_overlays(w, h));
        }

        // Loading bar while the async perf scene build is in flight.
        if self.mode == ShowcaseMode::Performance {
            if let Some(ref progress) = self.perf_state.build_progress {
                let n = progress.load(std::sync::atomic::Ordering::Relaxed);
                let label = format!(
                    "Building scene\u{2026} {} / 1 000 000",
                    showcase_03_performance::format_count(n),
                );
                fd.overlays.loading_bars.push(LoadingBarItem {
                    progress: n as f32 / 1_000_000.0,
                    label: Some(label),
                    anchor: LoadingBarAnchor::BottomCenter,
                    ..LoadingBarItem::default()
                });
            }
        }

        // Update stats and apply GPU culling toggle (Performance mode).
        if self.mode == ShowcaseMode::Performance {
            let rs = frame.wgpu_render_state().unwrap();
            let mut guard = rs.renderer.write();
            if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                if self.perf_state.gpu_culling {
                    renderer.enable_gpu_driven_culling();
                } else {
                    renderer.disable_gpu_driven_culling();
                }
                self.perf_state.last_stats = renderer.last_frame_stats();
            }
        }
        // Probe widget render items (Showcase 37) : submitted every frame.
        if self.mode == ShowcaseMode::ProbeWidgets && self.pw_state.built {
            let render_cam = CameraFrame::from_camera(&self.camera, [w, h]).render_camera;
            let widget_ctx = viewport_lib::WidgetContext {
                camera: render_cam,
                viewport_size: glam::Vec2::new(w, h),
                cursor_viewport: self.interact_state.last_cursor_viewport,
                drag_started: false,
                dragging: false,
                released: false,
                double_clicked: false,
            };
            use showcase_37_probe_widgets::PwSubMode;
            let state = &self.pw_state;
            match state.sub_mode {
                PwSubMode::LineProbe => {
                    fd.scene.polylines.push(state.probe.polyline_item(0));
                    fd.scene
                        .glyphs
                        .push(state.probe.handle_glyphs(100, &widget_ctx));
                }
                PwSubMode::Sphere => {
                    fd.effects.clip_objects.push(state.sphere.clip_object());
                    fd.scene.polylines.push(state.sphere.wireframe_item(0));
                    fd.scene
                        .glyphs
                        .push(state.sphere.handle_glyphs(100, &widget_ctx));
                }
                PwSubMode::Box => {
                    fd.scene.polylines.push(state.bw.wireframe_item(0));
                    fd.scene.polylines.push(state.bw.rotation_arcs_item(1));
                    fd.scene
                        .glyphs
                        .push(state.bw.handle_glyphs(100, &widget_ctx));
                }
                PwSubMode::Plane => {
                    fd.scene.polylines.push(state.plane.plane_item(0));
                    fd.scene
                        .glyphs
                        .push(state.plane.handle_glyphs(100, &widget_ctx));
                }
                PwSubMode::Disk => {
                    fd.scene.polylines.push(state.disk.wireframe_item(0));
                    fd.scene
                        .glyphs
                        .push(state.disk.handle_glyphs(100, &widget_ctx));
                }
                PwSubMode::Cylinder => {
                    fd.scene.polylines.push(state.cylinder.wireframe_item(0));
                    fd.scene
                        .glyphs
                        .push(state.cylinder.handle_glyphs(100, &widget_ctx));
                }
                PwSubMode::Polyline => {
                    fd.scene.polylines.push(state.polyline.polyline_item(0));
                    fd.scene
                        .glyphs
                        .push(state.polyline.handle_glyphs(100, &widget_ctx));
                }
            }

            // Point cloud: unselected in blue, selected in orange -- rendered as Gaussian splats.
            let unsel: Vec<[f32; 3]> = state
                .cloud_positions
                .iter()
                .zip(state.selected.iter())
                .filter(|(_, s)| !**s)
                .map(|(p, _)| *p)
                .collect();
            let sel: Vec<[f32; 3]> = state
                .cloud_positions
                .iter()
                .zip(state.selected.iter())
                .filter(|(_, s)| **s)
                .map(|(p, _)| *p)
                .collect();
            if !unsel.is_empty() {
                let mut pc = PointCloudItem::default();
                pc.positions = unsel;
                pc.default_colour = [0.5, 0.7, 1.0, 1.0];
                pc.gaussian = true;
                pc.point_size = 8.0;
                fd.scene.point_clouds.push(pc);
            }
            if !sel.is_empty() {
                let mut pc = PointCloudItem::default();
                pc.positions = sel;
                pc.default_colour = [1.0, 0.55, 0.1, 1.0];
                pc.gaussian = true;
                pc.point_size = 14.0;
                fd.scene.point_clouds.push(pc);
            }
        }

        // Surface LIC render items (Showcase 38) : submitted every frame when built.
        // LIC compositing happens inside the tone-map pass, so the HDR pipeline
        // must be active (post_process.enabled = true).
        if self.mode == ShowcaseMode::SurfaceLIC && self.lic_state.built {
            showcase_38_surface_lic::submit_lic_items(self, &mut fd);
            if !fd.scene.lic_items.is_empty() {
                fd.effects.post_process.enabled = true;
            }
        }

        // Tensor glyph items (Showcase 39) : submitted every frame when built.
        if self.mode == ShowcaseMode::TensorGlyphs && self.tg_state.built {
            showcase_39_tensor_glyphs::submit_tensor_glyphs(self, &mut fd);
        }

        // Sprite items and ring polylines (Showcase 41) : submitted every frame when built.
        if self.mode == ShowcaseMode::Sprites && self.sprite_state.built {
            fd.scene
                .sprite_items
                .extend(showcase_41_sprites::sprite_items(self));
            fd.scene
                .polylines
                .extend(showcase_41_sprites::ring_polylines(self));
        }

        // Gaussian splat items (Showcase 42) : submitted every frame when built.
        if self.mode == ShowcaseMode::GaussianSplats && self.splat_state.built {
            fd.scene
                .gaussian_splats
                .extend(showcase_42_gaussian_splats::gaussian_splat_items(self));
        }

        // PlaybackRuntime stats are updated inside the build_frame_data PlaybackRuntime arm.

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
                for node in self.sg_state.scene.nodes() {
                    if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                        mesh_lookup.entry(mid).or_insert_with(|| {
                            (
                                self.box_mesh_data.positions.clone(),
                                self.box_mesh_data.indices.clone(),
                            )
                        });
                    }
                }
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin,
                    ray_dir,
                    &self.sg_state.scene,
                    &mesh_lookup,
                );
                if let Some(hit) = hit {
                    self.sg_state.selection.select_one(hit.id);
                } else {
                    self.sg_state.selection.clear();
                }
            }

            ShowcaseMode::Performance => {
                let mut mesh_lookup = std::collections::HashMap::new();
                if let Some(mesh) = self.perf_state.mesh {
                    mesh_lookup.insert(
                        mesh.index() as u64,
                        (
                            self.box_mesh_data.positions.clone(),
                            self.box_mesh_data.indices.clone(),
                        ),
                    );
                }
                let hit = if let Some(ref mut accel) = self.perf_state.pick_accelerator {
                    viewport_lib::bvh::pick_scene_accelerated_cpu(
                        ray_origin,
                        ray_dir,
                        accel,
                        &mesh_lookup,
                    )
                } else {
                    None
                };
                if let Some(hit) = hit {
                    self.perf_state.selection.select_one(hit.id);
                } else {
                    self.perf_state.selection.clear();
                }
            }

            ShowcaseMode::Interaction => {
                let mut mesh_lookup = std::collections::HashMap::new();
                for node in self.interact_state.scene.nodes() {
                    if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                        mesh_lookup.entry(mid).or_insert_with(|| {
                            (
                                self.box_mesh_data.positions.clone(),
                                self.box_mesh_data.indices.clone(),
                            )
                        });
                    }
                }
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin,
                    ray_dir,
                    &self.interact_state.scene,
                    &mesh_lookup,
                );
                if let Some(hit) = hit {
                    self.interact_state.selection.select_one(hit.id);
                } else {
                    self.interact_state.selection.clear();
                }
            }

            ShowcaseMode::MaterialsVisibility => {
                let mut mesh_lookup = std::collections::HashMap::new();
                for node in self.materials_visibility_state.scene.nodes() {
                    if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                        mesh_lookup.entry(mid).or_insert_with(|| {
                            (
                                self.box_mesh_data.positions.clone(),
                                self.box_mesh_data.indices.clone(),
                            )
                        });
                    }
                }
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin,
                    ray_dir,
                    &self.materials_visibility_state.scene,
                    &mesh_lookup,
                );
                if let Some(hit) = hit {
                    self.materials_visibility_state.selection.select_one(hit.id);
                } else {
                    self.materials_visibility_state.selection.clear();
                }
            }

            ShowcaseMode::ScalarFields => {
                let mut mesh_lookup = std::collections::HashMap::new();
                for i in 0..self.scalar_state.mesh_indices.len() {
                    mesh_lookup.insert(
                        self.scalar_state.mesh_indices[i].index() as u64,
                        (
                            self.scalar_state.pick_positions[i].clone(),
                            self.scalar_state.pick_indices[i].clone(),
                        ),
                    );
                }
                let hit = viewport_lib::picking::pick_scene_nodes_cpu(
                    ray_origin,
                    ray_dir,
                    &self.scalar_state.scene,
                    &mesh_lookup,
                );
                if let Some(hit) = hit {
                    if let Some(index) = self
                        .scalar_state
                        .node_ids
                        .iter()
                        .position(|&node_id| node_id == hit.id)
                    {
                        self.scalar_state.set_active_object(index);
                    } else {
                        self.scalar_state.selection.select_one(hit.id);
                    }
                } else {
                    self.scalar_state.selection.clear();
                }
            }

            ShowcaseMode::PickLevels => {
                let shift = self.pl_state.shift_held;
                if self.pl_state.unified_mode {
                    // Unified path: renderer.pick() requires renderer access.
                    // Handled separately in the viewport event section.
                } else {
                    self.handle_pl_click(pos, w, h, shift);
                }
            }

            ShowcaseMode::SparseVolumeGrid => {
                self.handle_svg_paint_click(pos, w, h);
            }

            _ => {}
        }
    }
}
