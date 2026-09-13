//! Feature showcase for `viewport-lib` using `eframe` / `egui`.

use crate::eframe::egui;
use viewport_lib as vpl;
use viewport_lib::wgpu;
pub use viewport_lib_examples_eframe::eframe;
use vpl::{
    Action, ButtonState, Camera, CameraAnimator, CameraFrame, ClipObject, FrameData, GizmoAxis,
    GizmoInfo, GizmoMode, GroundPlane, GroundPlaneMode, LightingSettings, ManipResult,
    ManipulationContext, MeshData, MeshId, OffscreenViewportTarget, OrbitCameraController,
    PickBackend, PickMask, RuntimeMode, SceneFrame, SceneRenderItem, ScrollUnits, ViewportContext,
    ViewportEvent, ViewportRenderer,
    gizmo::{self, compute_gizmo_scale},
};

mod geometry;
mod gizmo_helpers;
mod registry;
mod shared;
mod showcase_01_basic;
mod showcase_02_scene_graph;
mod showcase_03_ground_plane;
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
mod showcase_23_performance;
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
mod showcase_43_scene_runtime;
mod showcase_44_debug_draw;
mod showcase_45_skinned_animation;
mod showcase_46_decals;
mod showcase_47_lighting_consistency;
mod showcase_48_scatter_volumes;
mod showcase_49_scene_lights;
mod showcase_50_gpu_wave;
mod showcase_51_async_uploads;
mod showcase_52_lod;
mod showcase_53_vertex_colours;
mod showcase_54_custom_shading;
mod showcase_55_foreground_pass;
mod showcase_56_submesh_materials;
mod showcase_57_photometric_lighting;
mod showcase_58_physically_based_surfaces;
mod showcase_59_vector_art;

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
                        // Request the features the renderer can use when the adapter has
                        // them (GPU-driven culling, GPU frame timings); eframe does not
                        // request any by default.
                        device_descriptor: std::sync::Arc::new(|adapter| {
                            use eframe::wgpu;
                            let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                                wgpu::Limits::downlevel_webgl2_defaults()
                            } else {
                                vpl::ViewportRenderer::recommended_device_limits(adapter)
                            };
                            wgpu::DeviceDescriptor {
                                label: Some("viewport-lib showcase device"),
                                required_features:
                                    vpl::ViewportRenderer::recommended_device_features(adapter),
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

            // sRGB render format so the tonemap encode happens; the offscreen
            // targets hand egui non-sRGB views so the encode survives the sample.
            let mut renderer =
                ViewportRenderer::new(&device, OffscreenViewportTarget::render_format(format));
            // Compile the custom-shading plugin pipelines now, at startup,
            // rather than on the frame that showcase opens: the ~45 pipeline
            // builds would otherwise stall that frame. See
            // prewarm_custom_shading_plugins.
            showcase_54_custom_shading::prewarm_custom_shading_plugins(&device, &mut renderer);
            wgpu_render_state
                .renderer
                .write()
                .callback_resources
                .insert(renderer);

            let box_mesh = vpl::primitives::cube(1.0);

            Ok(Box::new(App {
                device,
                queue,
                viewport_target: None,
                mv_targets: Vec::new(),
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
                pending_pick: None,
                basic_state: showcase_01_basic::BasicState::default(),
                sg_state: showcase_02_scene_graph::SgState::default(),
                box_mesh_data: box_mesh,
                perf_state: showcase_23_performance::PerfState::default(),
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

                gp_state: showcase_03_ground_plane::GroundPlaneState::default(),

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
                rt_state: showcase_43_scene_runtime::RtDemoState::default(),
                dbg_draw_state: showcase_44_debug_draw::DbgDrawState::default(),
                skin_state: showcase_45_skinned_animation::Skin47State::default(),
                decal46_state: showcase_46_decals::Decal46State::default(),
                lc_state: showcase_47_lighting_consistency::LcState::default(),
                svol_state: showcase_48_scatter_volumes::SvolState::default(),
                sl_state: showcase_49_scene_lights::SlState::default(),
                wave_state: showcase_50_gpu_wave::WaveState::default(),
                async_uploads_state: showcase_51_async_uploads::AsyncUploadsState::default(),
                lod_state: showcase_52_lod::LodState::default(),
                vcol_state: showcase_53_vertex_colours::VertexColourState::default(),
                cs_state: showcase_54_custom_shading::CustomShadingState::default(),
                fg_state: showcase_55_foreground_pass::ForegroundState::default(),
                submesh_state: showcase_56_submesh_materials::SubmeshState::default(),
                lighting_state: showcase_57_photometric_lighting::PhotometricLightingState::default(
                ),
                surfaces_state:
                    showcase_58_physically_based_surfaces::PhysicallyBasedSurfacesState::default(),
                va_state: showcase_59_vector_art::VectorArtState::default(),
                last_cluster_stats: None,
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
    SceneRuntime,
    DebugDraw,
    SkinnedAnimation,
    Decals,
    LightingConsistency,
    ScatterVolumes,
    SceneLights,
    GpuWave,
    AsyncUploads,
    Lod,
    VertexColours,
    CustomShading,
    Foreground,
    SubmeshMaterials,
    PhotometricLighting,
    PhysicallyBasedSurfaces,
    VectorArt,
}

// `ShowcaseMode::label` and the menu order live in `registry.rs`.

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// An offscreen colour target and its egui texture id, recreated on resize.
pub(crate) struct Target {
    pub(crate) inner: OffscreenViewportTarget,
    pub(crate) id: egui::TextureId,
}

pub(crate) struct App {
    // GPU handles (captured at startup for lazy mesh uploads).
    // wgpu::Device and Queue are internally ref-counted and implement Clone.
    device: eframe::wgpu::Device,
    queue: eframe::wgpu::Queue,

    /// Offscreen target for the single-viewport showcases (sRGB dual-view so the
    /// tonemap encode survives egui's sample).
    viewport_target: Option<Target>,
    /// Offscreen targets for the four multi-viewport quadrants.
    mv_targets: Vec<Target>,

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

    /// Deferred object/instance pick request from the input handler. Set on a
    /// click; consumed at the render site, where the renderer, device, queue,
    /// and the on-screen `FrameData` are all in scope, by `pick_object`. Cursor
    /// in viewport pixels.
    pending_pick: Option<glam::Vec2>,

    // --- Showcase 1 ---
    pub(crate) basic_state: showcase_01_basic::BasicState,

    // --- Showcase 2 ---
    pub(crate) sg_state: showcase_02_scene_graph::SgState,
    /// Shared box MeshData for on-demand uploads in later showcases.
    pub(crate) box_mesh_data: MeshData,

    // --- Showcase 23 ---
    pub(crate) perf_state: showcase_23_performance::PerfState,

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
    // --- Showcase 3 ---
    pub(crate) gp_state: showcase_03_ground_plane::GroundPlaneState,

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

    // --- Showcase 43 ---
    pub(crate) rt_state: showcase_43_scene_runtime::RtDemoState,

    // --- Showcase 44 ---
    pub(crate) dbg_draw_state: showcase_44_debug_draw::DbgDrawState,

    // --- Showcase 45 ---
    pub(crate) skin_state: showcase_45_skinned_animation::Skin47State,

    // --- Showcase 48 ---
    pub(crate) decal46_state: showcase_46_decals::Decal46State,

    // --- Showcase 49 ---
    pub(crate) lc_state: showcase_47_lighting_consistency::LcState,
    pub(crate) svol_state: showcase_48_scatter_volumes::SvolState,

    // --- Showcase 51 ---
    pub(crate) sl_state: showcase_49_scene_lights::SlState,

    // --- Showcase 50 ---
    pub(crate) wave_state: showcase_50_gpu_wave::WaveState,

    // --- Showcase 51 ---
    pub(crate) async_uploads_state: showcase_51_async_uploads::AsyncUploadsState,

    // --- Showcase 52 ---
    pub(crate) lod_state: showcase_52_lod::LodState,

    // --- Showcase 53 ---
    pub(crate) vcol_state: showcase_53_vertex_colours::VertexColourState,

    // --- Showcase 54 ---
    pub(crate) cs_state: showcase_54_custom_shading::CustomShadingState,
    pub(crate) fg_state: showcase_55_foreground_pass::ForegroundState,
    pub(crate) submesh_state: showcase_56_submesh_materials::SubmeshState,

    // --- Showcase 57: Photometric Lighting (units/presets + falloff + exposure) ---
    pub(crate) lighting_state: showcase_57_photometric_lighting::PhotometricLightingState,

    // --- Showcase 58: Physically-Based Surfaces (shading parity + emissive/IBL) ---
    pub(crate) surfaces_state: showcase_58_physically_based_surfaces::PhysicallyBasedSurfacesState,
    pub(crate) va_state: showcase_59_vector_art::VectorArtState,

    /// Latest cluster build stats pulled from the renderer, surfaced by the
    /// scene-lights controls panel.
    pub(crate) last_cluster_stats: Option<vpl::resources::gpu::clustered::ClusterStats>,
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Pull the latest cluster stats off the renderer for the scene-lights
        // controls panel. Cheap : the renderer only does the readback when the
        // panel previously requested it.
        if let Some(rs) = frame.wgpu_render_state() {
            let guard = rs.renderer.read();
            if let Some(renderer) = guard.callback_resources.get::<ViewportRenderer>() {
                if let Some(stats) = renderer.cluster_stats() {
                    self.last_cluster_stats = Some(stats);
                }
            }
        }

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
        if let Some(scene) = completed {
            self.perf_state.scene = scene;
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
            self.perf_state.total_objects = 125_000;
            self.perf_state.build_rx = None;
            self.perf_state.build_progress = None;
            self.perf_state.built = true;
        }

        // Lazy scene builds for the active mode.
        self.ensure_scene_built(frame);

        // ---- Top panel: showcase selector ----
        // One dropdown over the whole set, with a heading per registry group so
        // related demos are found together rather than scanned for.
        egui::TopBottomPanel::top("mode_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Showcase:");
                let mut chosen = None;
                egui::ComboBox::from_id_salt("showcase_selector")
                    .width(260.0)
                    .selected_text(self.mode.label())
                    .show_ui(ui, |ui| {
                        let mut group = None;
                        for entry in &registry::SHOWCASES {
                            if group != Some(entry.group) {
                                if group.is_some() {
                                    ui.separator();
                                }
                                ui.label(egui::RichText::new(entry.group.title()).small().strong());
                                group = Some(entry.group);
                            }
                            if ui
                                .selectable_label(self.mode == entry.mode, entry.label())
                                .clicked()
                            {
                                chosen = Some(entry.mode);
                            }
                        }
                    });
                if let Some(mode) = chosen {
                    self.switch_mode(mode);
                }
                ui.separator();
                ui.weak("Ctrl + [ / ] to cycle");
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
            showcase_02_scene_graph::background_colour(self.sg_state.bg_cycle).unwrap_or(BG_COLOUR)
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
                    let mods = vpl::Modifiers {
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
                                    egui::PointerButton::Primary => vpl::MouseButton::Left,
                                    egui::PointerButton::Secondary => vpl::MouseButton::Right,
                                    egui::PointerButton::Middle => vpl::MouseButton::Middle,
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
                                        let (ray_origin, ray_dir) = vpl::picking::screen_to_ray(
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
                                    let device = self.device.clone();
                                    let queue = self.queue.clone();
                                    let pick_frame =
                                        showcase_33_picking_levels::pl_build_pick_frame(
                                            self,
                                            rect.width(),
                                            rect.height(),
                                            ctx.pixels_per_point(),
                                        );
                                    let rs = frame.wgpu_render_state().expect("wgpu required");
                                    let mut guard = rs.renderer.write();
                                    if let Some(renderer) =
                                        guard.callback_resources.get_mut::<ViewportRenderer>()
                                    {
                                        self.handle_pl_unified_box_select(
                                            drag_start,
                                            drag_end,
                                            shift,
                                            renderer,
                                            &device,
                                            &queue,
                                            &pick_frame,
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
                                let target = vpl::interpolate_camera(
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

                // ----- Foreground fly-around + focus rack (Showcase 55) -----
                if self.mode == ShowcaseMode::Foreground {
                    let dt = ctx.input(|i| i.stable_dt.min(1.0 / 30.0));
                    showcase_55_foreground_pass::update_foreground(self, dt);
                    ctx.request_repaint();
                }

                // ----- Submesh rocket spin (Showcase 56) -----
                if self.mode == ShowcaseMode::SubmeshMaterials && self.submesh_state.spin {
                    let dt = ctx.input(|i| i.stable_dt.min(1.0 / 30.0));
                    self.submesh_state.angle += dt * 0.5;
                    ctx.request_repaint();
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
                            _ => {}
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
                            && self.pw_state.suppress_orbit)
                        || (self.mode == ShowcaseMode::VertexColours
                            && self.vcol_state.paint_mode
                            && (response.dragged() || response.drag_started()));
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
                    let widget_ctx = vpl::WidgetContext {
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
                    let widget_ctx = vpl::WidgetContext {
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
                    // Unified pick for PickLevels showcase uses renderer.pick_object(),
                    // which dispatches to the GPU or CPU backend per the UI toggle.
                    if self.mode == ShowcaseMode::PickLevels && self.pl_state.unified_mode {
                        let shift = self.pl_state.shift_held;
                        let device = self.device.clone();
                        let queue = self.queue.clone();
                        let pick_frame = showcase_33_picking_levels::pl_build_pick_frame(
                            self,
                            rect.width(),
                            rect.height(),
                            ctx.pixels_per_point(),
                        );
                        let rs = frame.wgpu_render_state().expect("wgpu required");
                        let mut guard = rs.renderer.write();
                        if let Some(renderer) =
                            guard.callback_resources.get_mut::<ViewportRenderer>()
                        {
                            self.handle_pl_unified_click(
                                pick_pos,
                                shift,
                                renderer,
                                &device,
                                &queue,
                                &pick_frame,
                            );
                        }
                    } else if (self.mode == ShowcaseMode::TensorGlyphs && self.tg_state.built)
                        || self.mode == ShowcaseMode::Decals
                    {
                        // Resolved by the unified GPU picker at the render site.
                        self.pending_pick = Some(pick_pos);
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
                let dt_frame = ui.ctx().input(|i| i.stable_dt.min(1.0 / 15.0));
                let frame_data = self.build_frame_data(
                    rect.width(),
                    rect.height(),
                    ui.ctx().pixels_per_point(),
                    frame,
                    dt_frame,
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

                // ----- Render into the offscreen target and display it -----
                // The target's sRGB dual-view keeps the tonemap encode through
                // egui's sample (see `OffscreenViewportTarget`).
                {
                    let rs = frame.wgpu_render_state().expect("wgpu required");
                    let ppp = ui.ctx().pixels_per_point();
                    let size_px = [
                        (rect.width() * ppp).round().max(1.0) as u32,
                        (rect.height() * ppp).round().max(1.0) as u32,
                    ];
                    let mut guard = rs.renderer.write();
                    if self
                        .viewport_target
                        .as_ref()
                        .map_or(true, |t| t.inner.size() != size_px)
                    {
                        let inner =
                            OffscreenViewportTarget::new(&self.device, rs.target_format, size_px);
                        let id = guard.register_native_texture(
                            &self.device,
                            inner.sample_view(),
                            eframe::wgpu::FilterMode::Linear,
                        );
                        self.viewport_target = Some(Target { inner, id });
                    }
                    let tex_id = self.viewport_target.as_ref().unwrap().id;
                    if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                        let cmd = renderer.owned().render(
                            &self.device,
                            &self.queue,
                            self.viewport_target.as_ref().unwrap().inner.render_view(),
                            &frame_data,
                        );
                        self.queue.submit(std::iter::once(cmd));
                        // Resolve any deferred click pick against the frame just
                        // drawn, using the unified GPU picker.
                        self.apply_pending_pick(renderer, &frame_data);
                    }
                    drop(guard);
                    ui.painter().image(
                        tex_id,
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }

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
                            vpl::ManipulationKind::Move => "Move",
                            vpl::ManipulationKind::Rotate => "Rotate",
                            vpl::ManipulationKind::Scale => "Scale",
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
                // ----- Skinned animation: step runtime -----
                if self.mode == ShowcaseMode::SkinnedAnimation && self.skin_state.built {
                    let dt = ctx.input(|i| i.stable_dt.min(0.25));
                    let cursor = self.interact_state.last_cursor_viewport;
                    let viewport_size = glam::Vec2::new(rect.width(), rect.height());
                    let clicked = response.clicked();
                    showcase_45_skinned_animation::update_skin47(
                        self,
                        dt,
                        cursor,
                        viewport_size,
                        clicked,
                    );
                    ctx.request_repaint();
                }
                // ----- Debug draw: step simulation -----
                if self.mode == ShowcaseMode::DebugDraw && self.dbg_draw_state.built {
                    let dt = ctx.input(|i| i.stable_dt.min(0.25));
                    showcase_44_debug_draw::update_dbg_draw(self, dt);
                    ctx.request_repaint();
                }
                // ----- Scene runtime: step simulation -----
                if self.mode == ShowcaseMode::SceneRuntime && self.rt_state.built {
                    let dt = ctx.input(|i| i.stable_dt.min(0.25));
                    showcase_43_scene_runtime::update_rt_demo(self, dt);
                    if !self.rt_state.paused
                        || self.rt_state.demo == showcase_43_scene_runtime::RuntimeDemo::Orbit
                    {
                        ctx.request_repaint();
                    }
                }
                // ----- Decals: advance live decal ages (D4) -----
                if self.mode == ShowcaseMode::Decals && self.decal46_state.built {
                    let dt = ctx.input(|i| i.stable_dt.min(1.0 / 30.0));
                    showcase_46_decals::update_decal46(self, dt);
                    if !self.decal46_state.scene.collect_decal_items().is_empty() {
                        ctx.request_repaint();
                    }
                }
                // ----- Overlay (35): keep repainting so OverlayFrame::time
                // advances and the built-in opacity animations (FadeIn,
                // FadeOut, Pulse) update without requiring user input.
                if self.mode == ShowcaseMode::Overlay {
                    ctx.request_repaint();
                }
                // ----- Lighting consistency (49): request repaint while the
                // second-light rotation animation is on so the cross-type
                // response is continuously observable.
                if self.mode == ShowcaseMode::LightingConsistency
                    && self.lc_state.second_light_enabled
                    && self.lc_state.second_light_animate
                {
                    ctx.request_repaint();
                }
                // ----- Scene lights (51): request repaint while orbit animation is on.
                if self.mode == ShowcaseMode::SceneLights && self.sl_state.animate {
                    ctx.request_repaint();
                }
                // ----- LOD (52): keep repainting while the camera dolly animates.
                if self.mode == ShowcaseMode::Lod && self.lod_state.auto_dolly {
                    ctx.request_repaint();
                }
                // ----- GPU wave (50): always animate while built and not paused.
                if self.mode == ShowcaseMode::GpuWave
                    && self.wave_state.built
                    && !self.wave_state.paused
                {
                    ctx.request_repaint();
                }
                // ----- Exposure (59): keep repainting so exposure-control changes
                // take effect and auto-exposure adaptation runs. Exposure lives in
                // `EffectsFrame`, not the scene, so it never dirties the scene; the
                // renderer is on-demand, so without a repaint request a slider change
                // is dropped until something else re-renders. Auto-exposure smoothing
                // also needs continuous frames.
                if self.mode == ShowcaseMode::PhotometricLighting
                    || self.mode == ShowcaseMode::PhysicallyBasedSurfaces
                {
                    ctx.request_repaint();
                }

                // ----- Vertex colours (53): drive the animated grid and apply
                // paint strokes. Both go through `update_vertex_colours`, an
                // in-place GPU write, so nothing here re-uploads a mesh.
                if self.mode == ShowcaseMode::VertexColours && self.vcol_state.built {
                    let dt = ctx.input(|i| i.stable_dt).min(0.1);
                    let cursor = self.interact_state.last_cursor_viewport;
                    let view_proj = self.camera.view_proj_matrix();
                    let vp_w = rect.width();
                    let vp_h = rect.height();
                    let queue = self.queue.clone();
                    let animate = self.vcol_state.animate;
                    let do_clear = std::mem::take(&mut self.vcol_state.clear_requested);
                    let do_paint = self.vcol_state.paint_mode
                        && response.hovered()
                        && (response.dragged() || response.drag_started() || response.clicked());

                    let rs = frame.wgpu_render_state().expect("wgpu required");
                    let mut guard = rs.renderer.write();
                    if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                        if do_clear {
                            showcase_53_vertex_colours::vcol_clear_paint(
                                &mut self.vcol_state,
                                renderer,
                                &queue,
                            );
                        }
                        if animate {
                            showcase_53_vertex_colours::vcol_animate(
                                &mut self.vcol_state,
                                renderer,
                                &queue,
                                dt,
                            );
                        }
                        if do_paint {
                            showcase_53_vertex_colours::vcol_paint(
                                &mut self.vcol_state,
                                renderer,
                                &queue,
                                cursor,
                                vp_w,
                                vp_h,
                                view_proj,
                            );
                        }
                    }
                    drop(guard);
                    if animate {
                        ctx.request_repaint();
                    }
                }

                // ----- Async uploads (51): advance per-asset state machines
                // from upload_status, and keep repainting so the orbit camera
                // animates even while no input is happening.
                if self.mode == ShowcaseMode::AsyncUploads && self.async_uploads_state.built {
                    let rs = frame.wgpu_render_state().expect("wgpu");
                    let mut guard = rs.renderer.write();
                    let renderer = guard
                        .callback_resources
                        .get_mut::<ViewportRenderer>()
                        .expect("renderer");
                    self.async_uploads_update(renderer);
                    drop(guard);
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
        let Some(current) = registry::SHOWCASES.iter().position(|e| e.mode == self.mode) else {
            return;
        };
        let len = registry::SHOWCASES.len() as i32;
        let next = (current as i32 + dir).rem_euclid(len) as usize;
        self.switch_mode(registry::SHOWCASES[next].mode);
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
                    let t = showcase_27_camera_framing::frustum_view_target(
                        &self.aux_state.frustums[i],
                    );
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

    /// Run the active showcase's lazy scene build if it has not run yet. Both
    /// halves dispatch to the showcase's own module: the predicate first, so the
    /// renderer lock is only taken on the frame a build actually happens.
    fn ensure_scene_built(&mut self, frame: &eframe::Frame) {
        let needs = match self.mode {
            ShowcaseMode::Basic => showcase_01_basic::needs_build(self),
            ShowcaseMode::SceneGraph => showcase_02_scene_graph::needs_build(self),
            ShowcaseMode::MaterialsVisibility => {
                showcase_05_materials_and_visibility::needs_build(self)
            }
            ShowcaseMode::ParamVis => showcase_22_parameterization::needs_build(self),
            ShowcaseMode::BackfacePolicy => showcase_24_backface_policy::needs_build(self),
            ShowcaseMode::Interaction => showcase_04_interaction::needs_build(self),
            ShowcaseMode::CameraTools => showcase_10_camera_tools::needs_build(self),
            ShowcaseMode::MultiViewport => showcase_13_multi_viewport::needs_build(self),
            ShowcaseMode::Auxiliary => showcase_27_camera_framing::needs_build(self),
            ShowcaseMode::ProbeWidgets => showcase_37_probe_widgets::needs_build(self),
            ShowcaseMode::GroundPlane => showcase_03_ground_plane::needs_build(self),
            ShowcaseMode::PostProcess => showcase_06_post_process::needs_build(self),
            ShowcaseMode::Foreground => showcase_55_foreground_pass::needs_build(self),
            ShowcaseMode::NormalMaps => showcase_07_normal_maps::needs_build(self),
            ShowcaseMode::Shadows => showcase_08_shadows::needs_build(self),
            ShowcaseMode::Lights => showcase_11_lights::needs_build(self),
            ShowcaseMode::Matcap => showcase_19_matcap::needs_build(self),
            ShowcaseMode::LightingConsistency => {
                showcase_47_lighting_consistency::needs_build(self)
            }
            ShowcaseMode::SceneLights => showcase_49_scene_lights::needs_build(self),
            ShowcaseMode::PhotometricLighting => {
                showcase_57_photometric_lighting::needs_build(self)
            }
            ShowcaseMode::Textures => showcase_21_textures::needs_build(self),
            ShowcaseMode::Decals => showcase_46_decals::needs_build(self),
            ShowcaseMode::VertexColours => showcase_53_vertex_colours::needs_build(self),
            ShowcaseMode::SubmeshMaterials => showcase_56_submesh_materials::needs_build(self),
            ShowcaseMode::PhysicallyBasedSurfaces => {
                showcase_58_physically_based_surfaces::needs_build(self)
            }
            ShowcaseMode::ScalarFields => showcase_12_scalar_fields::needs_build(self),
            ShowcaseMode::Isolines => showcase_14_isolines::needs_build(self),
            ShowcaseMode::PointClouds => showcase_15_point_clouds::needs_build(self),
            ShowcaseMode::Streamlines => showcase_16_streamlines::needs_build(self),
            ShowcaseMode::FaceAttributes => showcase_20_face_attributes::needs_build(self),
            ShowcaseMode::SurfaceVectors => showcase_25_surface_vectors::needs_build(self),
            ShowcaseMode::CurveNetworkQuantities => {
                showcase_28_curve_network_quantities::needs_build(self)
            }
            ShowcaseMode::ExtendedQuantities => showcase_32_extended_quantities::needs_build(self),
            ShowcaseMode::SurfaceLIC => showcase_38_surface_lic::needs_build(self),
            ShowcaseMode::TensorGlyphs => showcase_39_tensor_glyphs::needs_build(self),
            ShowcaseMode::Volume => showcase_17_volume::needs_build(self),
            ShowcaseMode::ClipVolumes => showcase_18_clip_volumes::needs_build(self),
            ShowcaseMode::VolumeMesh => showcase_26_volume_mesh::needs_build(self),
            ShowcaseMode::ImplicitSurface => showcase_30_implicit_surface::needs_build(self),
            ShowcaseMode::SparseVolumeGrid => showcase_31_sparse_volume_grid::needs_build(self),
            ShowcaseMode::ScatterVolumes => showcase_48_scatter_volumes::needs_build(self),
            ShowcaseMode::Annotation => showcase_09_annotation::needs_build(self),
            ShowcaseMode::DepthCompositeImages => {
                showcase_29_depth_composite_images::needs_build(self)
            }
            ShowcaseMode::Labels => showcase_34_labels::needs_build(self),
            ShowcaseMode::Overlay => showcase_35_overlay::needs_build(self),
            ShowcaseMode::VectorArt => showcase_59_vector_art::needs_build(self),
            ShowcaseMode::Sprites => showcase_41_sprites::needs_build(self),
            ShowcaseMode::GaussianSplats => showcase_42_gaussian_splats::needs_build(self),
            ShowcaseMode::PlaybackRuntime => showcase_36_playback_runtime::needs_build(self),
            ShowcaseMode::VertexWarp => showcase_40_vertex_warp::needs_build(self),
            ShowcaseMode::SceneRuntime => showcase_43_scene_runtime::needs_build(self),
            ShowcaseMode::DebugDraw => showcase_44_debug_draw::needs_build(self),
            ShowcaseMode::SkinnedAnimation => showcase_45_skinned_animation::needs_build(self),
            ShowcaseMode::GpuWave => showcase_50_gpu_wave::needs_build(self),
            ShowcaseMode::CustomShading => showcase_54_custom_shading::needs_build(self),
            ShowcaseMode::Performance => showcase_23_performance::needs_build(self),
            ShowcaseMode::PickLevels => showcase_33_picking_levels::needs_build(self),
            ShowcaseMode::AsyncUploads => showcase_51_async_uploads::needs_build(self),
            ShowcaseMode::Lod => showcase_52_lod::needs_build(self),
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
            ShowcaseMode::Basic => showcase_01_basic::build(self, renderer),
            ShowcaseMode::SceneGraph => showcase_02_scene_graph::build(self, renderer),
            ShowcaseMode::MaterialsVisibility => {
                showcase_05_materials_and_visibility::build(self, renderer)
            }
            ShowcaseMode::ParamVis => showcase_22_parameterization::build(self, renderer),
            ShowcaseMode::BackfacePolicy => showcase_24_backface_policy::build(self, renderer),
            ShowcaseMode::Interaction => showcase_04_interaction::build(self, renderer),
            ShowcaseMode::CameraTools => showcase_10_camera_tools::build(self, renderer),
            ShowcaseMode::MultiViewport => showcase_13_multi_viewport::build(self, renderer),
            ShowcaseMode::Auxiliary => showcase_27_camera_framing::build(self, renderer),
            ShowcaseMode::ProbeWidgets => showcase_37_probe_widgets::build(self, renderer),
            ShowcaseMode::GroundPlane => showcase_03_ground_plane::build(self, renderer),
            ShowcaseMode::PostProcess => showcase_06_post_process::build(self, renderer),
            ShowcaseMode::Foreground => showcase_55_foreground_pass::build(self, renderer),
            ShowcaseMode::NormalMaps => showcase_07_normal_maps::build(self, renderer),
            ShowcaseMode::Shadows => showcase_08_shadows::build(self, renderer),
            ShowcaseMode::Lights => showcase_11_lights::build(self, renderer),
            ShowcaseMode::Matcap => showcase_19_matcap::build(self, renderer),
            ShowcaseMode::LightingConsistency => {
                showcase_47_lighting_consistency::build(self, renderer)
            }
            ShowcaseMode::SceneLights => showcase_49_scene_lights::build(self, renderer),
            ShowcaseMode::PhotometricLighting => {
                showcase_57_photometric_lighting::build(self, renderer)
            }
            ShowcaseMode::Textures => showcase_21_textures::build(self, renderer),
            ShowcaseMode::Decals => showcase_46_decals::build(self, renderer),
            ShowcaseMode::VertexColours => showcase_53_vertex_colours::build(self, renderer),
            ShowcaseMode::SubmeshMaterials => showcase_56_submesh_materials::build(self, renderer),
            ShowcaseMode::PhysicallyBasedSurfaces => {
                showcase_58_physically_based_surfaces::build(self, renderer)
            }
            ShowcaseMode::ScalarFields => showcase_12_scalar_fields::build(self, renderer),
            ShowcaseMode::Isolines => showcase_14_isolines::build(self, renderer),
            ShowcaseMode::PointClouds => showcase_15_point_clouds::build(self, renderer),
            ShowcaseMode::Streamlines => showcase_16_streamlines::build(self, renderer),
            ShowcaseMode::FaceAttributes => showcase_20_face_attributes::build(self, renderer),
            ShowcaseMode::SurfaceVectors => showcase_25_surface_vectors::build(self, renderer),
            ShowcaseMode::CurveNetworkQuantities => {
                showcase_28_curve_network_quantities::build(self, renderer)
            }
            ShowcaseMode::ExtendedQuantities => {
                showcase_32_extended_quantities::build(self, renderer)
            }
            ShowcaseMode::SurfaceLIC => showcase_38_surface_lic::build(self, renderer),
            ShowcaseMode::TensorGlyphs => showcase_39_tensor_glyphs::build(self, renderer),
            ShowcaseMode::Volume => showcase_17_volume::build(self, renderer),
            ShowcaseMode::ClipVolumes => showcase_18_clip_volumes::build(self, renderer),
            ShowcaseMode::VolumeMesh => showcase_26_volume_mesh::build(self, renderer),
            ShowcaseMode::ImplicitSurface => showcase_30_implicit_surface::build(self, renderer),
            ShowcaseMode::SparseVolumeGrid => showcase_31_sparse_volume_grid::build(self, renderer),
            ShowcaseMode::ScatterVolumes => showcase_48_scatter_volumes::build(self, renderer),
            ShowcaseMode::Annotation => showcase_09_annotation::build(self, renderer),
            ShowcaseMode::DepthCompositeImages => {
                showcase_29_depth_composite_images::build(self, renderer)
            }
            ShowcaseMode::Labels => showcase_34_labels::build(self, renderer),
            ShowcaseMode::Overlay => showcase_35_overlay::build(self, renderer),
            ShowcaseMode::VectorArt => showcase_59_vector_art::build(self, renderer),
            ShowcaseMode::Sprites => showcase_41_sprites::build(self, renderer),
            ShowcaseMode::GaussianSplats => showcase_42_gaussian_splats::build(self, renderer),
            ShowcaseMode::PlaybackRuntime => showcase_36_playback_runtime::build(self, renderer),
            ShowcaseMode::VertexWarp => showcase_40_vertex_warp::build(self, renderer),
            ShowcaseMode::SceneRuntime => showcase_43_scene_runtime::build(self, renderer),
            ShowcaseMode::DebugDraw => showcase_44_debug_draw::build(self, renderer),
            ShowcaseMode::SkinnedAnimation => showcase_45_skinned_animation::build(self, renderer),
            ShowcaseMode::GpuWave => showcase_50_gpu_wave::build(self, renderer),
            ShowcaseMode::CustomShading => showcase_54_custom_shading::build(self, renderer),
            ShowcaseMode::Performance => showcase_23_performance::build(self, renderer),
            ShowcaseMode::PickLevels => showcase_33_picking_levels::build(self, renderer),
            ShowcaseMode::AsyncUploads => showcase_51_async_uploads::build(self, renderer),
            ShowcaseMode::Lod => showcase_52_lod::build(self, renderer),
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
            ShowcaseMode::Performance => showcase_23_performance::controls_performance(self, ui),
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
            ShowcaseMode::GroundPlane => showcase_03_ground_plane::controls_ground_plane(self, ui),
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
            ShowcaseMode::SceneRuntime => showcase_43_scene_runtime::controls_rt_demo(self, ui),
            ShowcaseMode::DebugDraw => showcase_44_debug_draw::controls_dbg_draw(self, ui),
            ShowcaseMode::SkinnedAnimation => {
                showcase_45_skinned_animation::controls_skin47(self, ui)
            }
            ShowcaseMode::Decals => showcase_46_decals::controls_decal46(self, ui),
            ShowcaseMode::LightingConsistency => {
                showcase_47_lighting_consistency::controls_lc(self, ui)
            }
            ShowcaseMode::ScatterVolumes => showcase_48_scatter_volumes::controls_svol(self, ui),
            ShowcaseMode::SceneLights => showcase_49_scene_lights::controls_sl(self, ui),
            ShowcaseMode::GpuWave => showcase_50_gpu_wave::controls_wave(self, ui),
            ShowcaseMode::AsyncUploads => {
                showcase_51_async_uploads::controls_async_uploads(self, ui, frame)
            }
            ShowcaseMode::Lod => showcase_52_lod::controls_lod(self, ui),
            ShowcaseMode::VertexColours => {
                showcase_53_vertex_colours::controls_vertex_colour(self, ui)
            }
            ShowcaseMode::CustomShading => {
                showcase_54_custom_shading::controls_custom_shading(self, ui)
            }
            ShowcaseMode::Foreground => showcase_55_foreground_pass::controls_foreground(self, ui),
            ShowcaseMode::SubmeshMaterials => {
                showcase_56_submesh_materials::controls_submesh(self, ui)
            }
            ShowcaseMode::PhotometricLighting => {
                showcase_57_photometric_lighting::controls_photometric_lighting(self, ui)
            }
            ShowcaseMode::PhysicallyBasedSurfaces => {
                showcase_58_physically_based_surfaces::controls_physically_based_surfaces(self, ui)
            }
            ShowcaseMode::VectorArt => showcase_59_vector_art::controls_vector_art(self, ui),
        }
    }
}

// ---------------------------------------------------------------------------
// Frame data assembly
// ---------------------------------------------------------------------------

/// What a showcase contributes to the frame: its render items, plus the few
/// per-frame values the host reads back out of it.
pub(crate) struct SceneContents {
    pub(crate) items: Vec<SceneRenderItem>,
    /// Replaces the viewport background for this frame when set.
    pub(crate) bg_colour: Option<[f32; 4]>,
    pub(crate) lighting: LightingSettings,
    /// Scene and selection versions, so the renderer's instance cache can see
    /// when the items really changed. Showcases that assemble items by hand
    /// leave both at 0 and rely on the mode generation instead.
    pub(crate) scene_gen: u64,
    pub(crate) sel_gen: u64,
}

/// Frame settings a showcase may set from its `scene` hook on top of the items
/// it returns. Most showcases set none of them and leave this at its default.
pub(crate) struct SceneOverrides {
    pub(crate) clip_objects: Vec<ClipObject>,
    pub(crate) outline: bool,
    pub(crate) xray: bool,
    pub(crate) perf_outline: bool,
    pub(crate) scene_graph_outline: bool,
    pub(crate) scene_graph_outline_width: f32,
    /// Hands the host a cached item list to draw instead of `items`, so a large
    /// static scene is not deep-cloned every frame.
    pub(crate) cached_items: Option<std::sync::Arc<[SceneRenderItem]>>,
}

impl Default for SceneOverrides {
    fn default() -> Self {
        Self {
            clip_objects: Vec::new(),
            outline: false,
            xray: false,
            perf_outline: false,
            scene_graph_outline: false,
            scene_graph_outline_width: 4.0,
            cached_items: None,
        }
    }
}

/// Per-frame inputs a showcase's `frame` hook may need, beyond the app state
/// and the frame data it is filling in.
pub(crate) struct FrameCtx<'a> {
    pub(crate) frame: &'a eframe::Frame,
    /// Viewport size in logical points.
    pub(crate) w: f32,
    pub(crate) h: f32,
    /// Seconds since the previous frame.
    pub(crate) dt: f32,
}

impl App {
    fn build_frame_data(
        &mut self,
        w: f32,
        h: f32,
        pixels_per_point: f32,
        frame: &eframe::Frame,
        dt: f32,
    ) -> FrameData {
        if self.mode == ShowcaseMode::Lod {
            showcase_52_lod::update_lod(self, dt);
        }

        let mut overrides = SceneOverrides::default();
        let contents = match self.mode {
            ShowcaseMode::Basic => showcase_01_basic::scene(self, frame, &mut overrides),
            ShowcaseMode::SceneGraph => showcase_02_scene_graph::scene(self, frame, &mut overrides),
            ShowcaseMode::MaterialsVisibility => {
                showcase_05_materials_and_visibility::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::ParamVis => {
                showcase_22_parameterization::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::BackfacePolicy => {
                showcase_24_backface_policy::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Interaction => {
                showcase_04_interaction::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::CameraTools => {
                showcase_10_camera_tools::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::MultiViewport => {
                showcase_13_multi_viewport::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Auxiliary => {
                showcase_27_camera_framing::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::ProbeWidgets => {
                showcase_37_probe_widgets::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::GroundPlane => {
                showcase_03_ground_plane::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::PostProcess => {
                showcase_06_post_process::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Foreground => {
                showcase_55_foreground_pass::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::NormalMaps => showcase_07_normal_maps::scene(self, frame, &mut overrides),
            ShowcaseMode::Shadows => showcase_08_shadows::scene(self, frame, &mut overrides),
            ShowcaseMode::Lights => showcase_11_lights::scene(self, frame, &mut overrides),
            ShowcaseMode::Matcap => showcase_19_matcap::scene(self, frame, &mut overrides),
            ShowcaseMode::LightingConsistency => {
                showcase_47_lighting_consistency::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::SceneLights => {
                showcase_49_scene_lights::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::PhotometricLighting => {
                showcase_57_photometric_lighting::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Textures => showcase_21_textures::scene(self, frame, &mut overrides),
            ShowcaseMode::Decals => showcase_46_decals::scene(self, frame, &mut overrides),
            ShowcaseMode::VertexColours => {
                showcase_53_vertex_colours::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::SubmeshMaterials => {
                showcase_56_submesh_materials::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::PhysicallyBasedSurfaces => {
                showcase_58_physically_based_surfaces::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::ScalarFields => {
                showcase_12_scalar_fields::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Isolines => showcase_14_isolines::scene(self, frame, &mut overrides),
            ShowcaseMode::PointClouds => {
                showcase_15_point_clouds::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Streamlines => {
                showcase_16_streamlines::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::FaceAttributes => {
                showcase_20_face_attributes::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::SurfaceVectors => {
                showcase_25_surface_vectors::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::CurveNetworkQuantities => {
                showcase_28_curve_network_quantities::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::ExtendedQuantities => {
                showcase_32_extended_quantities::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::SurfaceLIC => showcase_38_surface_lic::scene(self, frame, &mut overrides),
            ShowcaseMode::TensorGlyphs => {
                showcase_39_tensor_glyphs::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Volume => showcase_17_volume::scene(self, frame, &mut overrides),
            ShowcaseMode::ClipVolumes => {
                showcase_18_clip_volumes::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::VolumeMesh => showcase_26_volume_mesh::scene(self, frame, &mut overrides),
            ShowcaseMode::ImplicitSurface => {
                showcase_30_implicit_surface::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::SparseVolumeGrid => {
                showcase_31_sparse_volume_grid::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::ScatterVolumes => {
                showcase_48_scatter_volumes::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Annotation => showcase_09_annotation::scene(self, frame, &mut overrides),
            ShowcaseMode::DepthCompositeImages => {
                showcase_29_depth_composite_images::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Labels => showcase_34_labels::scene(self, frame, &mut overrides),
            ShowcaseMode::Overlay => showcase_35_overlay::scene(self, frame, &mut overrides),
            ShowcaseMode::VectorArt => showcase_59_vector_art::scene(self, frame, &mut overrides),
            ShowcaseMode::Sprites => showcase_41_sprites::scene(self, frame, &mut overrides),
            ShowcaseMode::GaussianSplats => {
                showcase_42_gaussian_splats::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::PlaybackRuntime => {
                showcase_36_playback_runtime::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::VertexWarp => showcase_40_vertex_warp::scene(self, frame, &mut overrides),
            ShowcaseMode::SceneRuntime => {
                showcase_43_scene_runtime::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::DebugDraw => showcase_44_debug_draw::scene(self, frame, &mut overrides),
            ShowcaseMode::SkinnedAnimation => {
                showcase_45_skinned_animation::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::GpuWave => showcase_50_gpu_wave::scene(self, frame, &mut overrides),
            ShowcaseMode::CustomShading => {
                showcase_54_custom_shading::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Performance => {
                showcase_23_performance::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::PickLevels => {
                showcase_33_picking_levels::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::AsyncUploads => {
                showcase_51_async_uploads::scene(self, frame, &mut overrides)
            }
            ShowcaseMode::Lod => showcase_52_lod::scene(self, frame, &mut overrides),
        };
        let SceneContents {
            items: scene_items,
            bg_colour,
            lighting,
            scene_gen,
            sel_gen,
        } = contents;
        let SceneOverrides {
            clip_objects: mut adv_clip_objects,
            outline: adv_outline,
            xray: adv_xray,
            perf_outline,
            scene_graph_outline,
            scene_graph_outline_width,
            cached_items: perf_arc,
        } = overrides;

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
            showcase_33_picking_levels::pl_configure_frame(self, &mut fd);
        }
        fd.viewport.show_grid = self.mode == ShowcaseMode::GroundPlane
            && self.gp_state.mode == showcase_03_ground_plane::GpMode::Grid;
        if self.mode == ShowcaseMode::GroundPlane
            && self.gp_state.mode == showcase_03_ground_plane::GpMode::Grid
        {
            fd.viewport.grid_colour = Some(self.gp_state.grid_colour.into());
            fd.viewport.grid_z = self.gp_state.height;
        }
        fd.viewport.show_axes_indicator = true;
        fd.effects.debug.force_cluster_fallback =
            self.mode == ShowcaseMode::SceneLights && self.sl_state.force_cluster_fallback;
        fd.effects.debug.cluster_stats_request =
            self.mode == ShowcaseMode::SceneLights && self.sl_state.show_cluster_stats;
        fd.viewport.background_colour = bg_colour.map(Into::into);

        // Ground plane (Showcase 3).
        if self.mode == ShowcaseMode::GroundPlane {
            use showcase_03_ground_plane::GpMode;
            fd.effects.ground_plane = GroundPlane {
                mode: match self.gp_state.mode {
                    GpMode::None | GpMode::Grid => GroundPlaneMode::None,
                    GpMode::ShadowOnly => GroundPlaneMode::ShadowOnly,
                    GpMode::Tile => GroundPlaneMode::Tile,
                    GpMode::SolidColour => GroundPlaneMode::SolidColour,
                },
                height: self.gp_state.height,
                colour: self.gp_state.colour.into(),
                tile_colour2: self.gp_state.tile_colour2.into(),
                tile_size: self.gp_state.tile_size,
                shadow_colour: self.gp_state.shadow_colour.into(),
                shadow_opacity: self.gp_state.shadow_opacity,
            };
        }
        // Clip objects for Showcase 24 (Surface Appearance).
        if self.mode == ShowcaseMode::BackfacePolicy {
            adv_clip_objects.extend(self.sa_clip_objects());
        }
        fd.effects.clip.objects = adv_clip_objects;
        if self.mode == ShowcaseMode::NormalMaps {
            fd.effects.clip.cap_fill_enabled = self.nm_state.cap_fill;
        }
        // Showcase 24 exists to show back face policies : cap fill would hide them.
        if self.mode == ShowcaseMode::BackfacePolicy {
            fd.effects.clip.cap_fill_enabled = false;
        }
        if self.mode == ShowcaseMode::VolumeMesh {
            showcase_26_volume_mesh::vm_configure_frame(self, &mut fd);
        }
        fd.interaction.gizmo_model = gizmo_model;
        fd.interaction.gizmo_mode = gizmo_mode;
        fd.interaction.gizmo_hovered = gizmo_hovered;
        fd.interaction.gizmo_space_orientation = gizmo_space_orient;
        fd.interaction.outline_selected = adv_outline
            || perf_outline
            || scene_graph_outline
            || (self.mode == ShowcaseMode::Interaction
                && showcase_04_interaction::interact_outline_selected(self))
            || (self.mode == ShowcaseMode::ScalarFields && !self.scalar_state.selection.is_empty())
            || (self.mode == ShowcaseMode::PickLevels
                && showcase_33_picking_levels::pl_outline_selected(self))
            || (self.mode == ShowcaseMode::SkinnedAnimation
                && !self.skin_state.selection.is_empty())
            || (self.mode == ShowcaseMode::LightingConsistency && self.lc_state.bcast_selected);
        if scene_graph_outline {
            fd.interaction.outline_width_px = scene_graph_outline_width;
        }
        fd.interaction.xray_selected = adv_xray;
        fd.scene.generation = scene_gen;
        fd.interaction.selection_generation = sel_gen;

        // Everything a showcase re-submits per frame: extra render items,
        // overlays, and effect settings that are not part of its scene.
        let frame_ctx = FrameCtx { frame, w, h, dt };
        match self.mode {
            ShowcaseMode::Basic => showcase_01_basic::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::SceneGraph => showcase_02_scene_graph::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::MaterialsVisibility => {
                showcase_05_materials_and_visibility::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::ParamVis => {
                showcase_22_parameterization::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::BackfacePolicy => {
                showcase_24_backface_policy::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Interaction => showcase_04_interaction::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::CameraTools => showcase_10_camera_tools::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::MultiViewport => {
                showcase_13_multi_viewport::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Auxiliary => showcase_27_camera_framing::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::ProbeWidgets => {
                showcase_37_probe_widgets::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::GroundPlane => showcase_03_ground_plane::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::PostProcess => showcase_06_post_process::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::Foreground => {
                showcase_55_foreground_pass::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::NormalMaps => showcase_07_normal_maps::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::Shadows => showcase_08_shadows::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::Lights => showcase_11_lights::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::Matcap => showcase_19_matcap::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::LightingConsistency => {
                showcase_47_lighting_consistency::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::SceneLights => showcase_49_scene_lights::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::PhotometricLighting => {
                showcase_57_photometric_lighting::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Textures => showcase_21_textures::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::Decals => showcase_46_decals::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::VertexColours => {
                showcase_53_vertex_colours::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::SubmeshMaterials => {
                showcase_56_submesh_materials::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::PhysicallyBasedSurfaces => {
                showcase_58_physically_based_surfaces::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::ScalarFields => {
                showcase_12_scalar_fields::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Isolines => showcase_14_isolines::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::PointClouds => showcase_15_point_clouds::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::Streamlines => showcase_16_streamlines::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::FaceAttributes => {
                showcase_20_face_attributes::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::SurfaceVectors => {
                showcase_25_surface_vectors::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::CurveNetworkQuantities => {
                showcase_28_curve_network_quantities::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::ExtendedQuantities => {
                showcase_32_extended_quantities::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::SurfaceLIC => showcase_38_surface_lic::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::TensorGlyphs => {
                showcase_39_tensor_glyphs::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Volume => showcase_17_volume::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::ClipVolumes => showcase_18_clip_volumes::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::VolumeMesh => showcase_26_volume_mesh::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::ImplicitSurface => {
                showcase_30_implicit_surface::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::SparseVolumeGrid => {
                showcase_31_sparse_volume_grid::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::ScatterVolumes => {
                showcase_48_scatter_volumes::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Annotation => showcase_09_annotation::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::DepthCompositeImages => {
                showcase_29_depth_composite_images::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Labels => showcase_34_labels::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::Overlay => showcase_35_overlay::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::VectorArt => showcase_59_vector_art::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::Sprites => showcase_41_sprites::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::GaussianSplats => {
                showcase_42_gaussian_splats::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::PlaybackRuntime => {
                showcase_36_playback_runtime::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::VertexWarp => showcase_40_vertex_warp::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::SceneRuntime => {
                showcase_43_scene_runtime::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::DebugDraw => showcase_44_debug_draw::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::SkinnedAnimation => {
                showcase_45_skinned_animation::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::GpuWave => showcase_50_gpu_wave::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::CustomShading => {
                showcase_54_custom_shading::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Performance => showcase_23_performance::frame(self, &mut fd, &frame_ctx),
            ShowcaseMode::PickLevels => {
                showcase_33_picking_levels::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::AsyncUploads => {
                showcase_51_async_uploads::frame(self, &mut fd, &frame_ctx)
            }
            ShowcaseMode::Lod => showcase_52_lod::frame(self, &mut fd, &frame_ctx),
        }

        fd
    }
}

// ---------------------------------------------------------------------------
// Selection / picking
// ---------------------------------------------------------------------------

impl App {
    fn handle_click_select(&mut self, pos: glam::Vec2, w: f32, h: f32) {
        match self.mode {
            // Object-level selection modes: defer the pick to the render site,
            // where the renderer and the on-screen `FrameData` are in scope, and
            // resolve it with the unified GPU picker (`pick_object`). See
            // `apply_pending_pick`.
            ShowcaseMode::SceneGraph
            | ShowcaseMode::Performance
            | ShowcaseMode::Interaction
            | ShowcaseMode::MaterialsVisibility
            | ShowcaseMode::ScalarFields => {
                self.pending_pick = Some(pos);
            }

            // Sparse volume grid uses the click to paint a voxel, not to select.
            ShowcaseMode::SparseVolumeGrid => {
                self.handle_svg_paint_click(pos, w, h);
            }

            // PickLevels per-type reference path; the unified path is handled in
            // the viewport event section against a dedicated pick frame.
            ShowcaseMode::PickLevels => {
                if !self.pl_state.unified_mode {
                    let shift = self.pl_state.shift_held;
                    self.handle_pl_click(pos, w, h, shift);
                }
            }

            _ => {}
        }
    }

    /// Resolve a deferred click pick against the on-screen frame using the
    /// unified GPU picker, and route the hit to the active mode's selection.
    ///
    /// Called from the render site so the renderer, device, queue, and the
    /// `FrameData` that was just drawn are all available. `pick_object` reads the
    /// scene straight from `frame_data`, so the pick matches exactly what is on
    /// screen.
    fn apply_pending_pick(&mut self, renderer: &mut ViewportRenderer, frame_data: &FrameData) {
        let Some(pos) = self.pending_pick.take() else {
            return;
        };
        let mask = match self.mode {
            // Tensor glyph instances and beam-mesh cells are both point-like.
            ShowcaseMode::TensorGlyphs => PickMask::POINT_LIKE,
            _ => PickMask::OBJECT,
        };
        let hit = renderer.pick_object(
            PickBackend::Gpu,
            pos,
            frame_data,
            &self.device,
            &self.queue,
            mask,
        );

        match self.mode {
            ShowcaseMode::SceneGraph => match hit {
                Some(h) => self.sg_state.selection.select_one(h.id),
                None => self.sg_state.selection.clear(),
            },
            ShowcaseMode::Performance => match hit {
                Some(h) => self.perf_state.selection.select_one(h.id),
                None => self.perf_state.selection.clear(),
            },
            ShowcaseMode::Interaction => match hit {
                Some(h) => self.interact_state.selection.select_one(h.id),
                None => self.interact_state.selection.clear(),
            },
            ShowcaseMode::MaterialsVisibility => match hit {
                Some(h) => self.materials_visibility_state.selection.select_one(h.id),
                None => self.materials_visibility_state.selection.clear(),
            },
            ShowcaseMode::ScalarFields => match hit {
                Some(h) => {
                    // The scalar-field objects are cycled by index; a hit on one
                    // makes it the active object, otherwise it is a plain select.
                    if let Some(index) =
                        self.scalar_state.node_ids.iter().position(|&id| id == h.id)
                    {
                        self.scalar_state.set_active_object(index);
                    } else {
                        self.scalar_state.selection.select_one(h.id);
                    }
                }
                None => self.scalar_state.selection.clear(),
            },
            ShowcaseMode::TensorGlyphs => {
                showcase_39_tensor_glyphs::tg_apply_pick(self, hit);
            }
            ShowcaseMode::Decals => {
                // Decal placement uses the hit's surface position and normal.
                if let Some(h) = hit {
                    showcase_46_decals::decal46_place(self, &h);
                }
            }
            _ => {}
        }
    }
}
