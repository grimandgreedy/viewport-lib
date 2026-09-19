//! Feature showcase for `viewport-lib` using `eframe` / `egui`.

use crate::eframe::egui;
use viewport_lib as vpl;
pub use viewport_lib_examples_eframe::eframe;
use vpl::{
    ButtonState, Camera, CameraAnimator, CameraFrame, ClipObject, FrameData, GizmoAxis, GizmoMode,
    GroundPlane, GroundPlaneMode, LightingSettings, MeshData, MeshId, OffscreenViewportTarget,
    OrbitCameraController, PickBackend, PickMask, SceneFrame, SceneRenderItem, ScrollUnits,
    ViewportContext, ViewportEvent, ViewportRenderer,
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
                cursor_viewport: glam::Vec2::ZERO,
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

    /// Cursor position in viewport pixels, updated once per frame from the
    /// pointer. Shared: several showcases read it for picking and painting,
    /// so it belongs to the host rather than to any one of them.
    pub(crate) cursor_viewport: glam::Vec2,

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

        // ---- Top panel: mode switching ----
        egui::TopBottomPanel::top("mode_panel").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Showcase:");
                let mut chosen = None;
                for entry in &registry::SHOWCASES {
                    if ui
                        .selectable_label(self.mode == entry.mode, entry.label())
                        .clicked()
                    {
                        chosen = Some(entry.mode);
                    }
                }
                if let Some(mode) = chosen {
                    self.switch_mode(mode);
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

                // Built once here and handed to every per-showcase hook below;
                // `response` and `rect` do not change for the rest of the frame.
                let viewport_cx = ViewportCtx {
                    egui: ctx,
                    frame,
                    response: &response,
                    rect,
                };

                // A showcase may drive the whole viewport itself (the
                // multi-viewport one does), in which case the host's
                // single-viewport path below is skipped entirely.
                if self.showcase_viewport_override(ui, &viewport_cx) {
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
                        self.cursor_viewport = local;
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

                // ----- Per-showcase drag_input -----
                self.showcase_drag_input(&viewport_cx);
                // ----- Per-showcase advance -----
                self.showcase_advance(&viewport_cx);
                // ----- Camera control -----
                // A showcase that needs the resulting `ActionFrame` (gizmo
                // manipulation) drives the controller itself; otherwise the
                // host applies it, or resolves it without moving the camera
                // when the showcase is using the drag for something else.
                if !self.showcase_drive_camera(&viewport_cx) {
                    if self.showcase_suppress_orbit(&viewport_cx) {
                        self.controller.resolve();
                    } else {
                        self.controller.apply_to_camera(&mut self.camera);
                    }
                }

                self.camera.set_aspect_ratio(rect.width(), rect.height());

                // ----- Per-showcase widgets -----
                self.showcase_widgets(&viewport_cx);
                // ----- Click-to-select -----
                // Showcase 4 routes its own clicks from the manipulation block
                // above, so that a click ending a gizmo drag is not a selection.
                if response.clicked() && self.mode != ShowcaseMode::Interaction {
                    let click_cx = ClickCtx {
                        frame,
                        pos: self.cursor_viewport,
                        w: rect.width(),
                        h: rect.height(),
                        pixels_per_point: ctx.pixels_per_point(),
                    };
                    self.handle_click_select(&click_cx);
                }

                // ----- Per-showcase flush_gpu -----
                self.showcase_flush_gpu(&viewport_cx);
                // ----- Build frame data -----
                let dt_frame = ui.ctx().input(|i| i.stable_dt.min(1.0 / 15.0));
                let frame_data = self.build_frame_data(
                    rect.width(),
                    rect.height(),
                    ui.ctx().pixels_per_point(),
                    frame,
                    dt_frame,
                );

                // ----- Per-showcase cache_gizmo -----
                self.showcase_cache_gizmo(&viewport_cx);
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

                // ----- Per-showcase viewport overlay -----
                self.showcase_overlay(ui, &viewport_cx);

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

                // ----- Per-showcase animation and repaint requests -----
                self.showcase_tick(&viewport_cx);
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
    /// Let the active showcase draw its own egui overlay over the viewport.
    fn showcase_overlay(&mut self, ui: &mut egui::Ui, cx: &ViewportCtx) {
        self.mode.showcase().overlay(self, ui, cx)
    }

    /// Let the active showcase advance its animation and request a repaint.
    fn showcase_tick(&mut self, cx: &ViewportCtx) {
        self.mode.showcase().tick(self, cx)
    }

    /// Handle drag gestures this showcase owns, before the camera controller runs.
    fn showcase_drag_input(&mut self, cx: &ViewportCtx) {
        self.mode.showcase().drag_input(self, cx)
    }

    /// Advance this showcase's own camera animation or object motion for the frame.
    fn showcase_advance(&mut self, cx: &ViewportCtx) {
        self.mode.showcase().advance(self, cx)
    }

    /// Update this showcase's interactive widgets for the frame.
    fn showcase_widgets(&mut self, cx: &ViewportCtx) {
        self.mode.showcase().widgets(self, cx)
    }

    /// Flush any per-frame GPU writes this showcase has queued.
    fn showcase_flush_gpu(&mut self, cx: &ViewportCtx) {
        self.mode.showcase().flush_gpu(self, cx)
    }

    /// Cache gizmo placement for next frame's hit-testing.
    fn showcase_cache_gizmo(&mut self, cx: &ViewportCtx) {
        self.mode.showcase().cache_gizmo(self, cx)
    }

    /// Let the active showcase take over the whole viewport for this frame.
    fn showcase_viewport_override(&mut self, ui: &mut egui::Ui, cx: &ViewportCtx) -> bool {
        self.mode.showcase().viewport_override(self, ui, cx)
    }

    /// Let the active showcase drive the orbit controller itself. Returns true
    /// if it did, in which case the host leaves the camera alone.
    fn showcase_drive_camera(&mut self, cx: &ViewportCtx) -> bool {
        self.mode.showcase().drive_camera(self, cx)
    }

    /// Whether the active showcase wants the orbit resolved without moving the
    /// camera this frame.
    fn showcase_suppress_orbit(&self, cx: &ViewportCtx) -> bool {
        self.mode.showcase().suppress_orbit(self, cx)
    }

    fn ensure_scene_built(&mut self, frame: &eframe::Frame) {
        let needs = self.mode.showcase().needs_build(self);
        if !needs {
            return;
        }
        let rs = frame.wgpu_render_state().expect("wgpu must be enabled");
        let mut guard = rs.renderer.write();
        let renderer = guard
            .callback_resources
            .get_mut::<ViewportRenderer>()
            .expect("ViewportRenderer must be registered");

        self.mode.showcase().build(self, renderer)
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

        self.mode.showcase().controls(self, ui, frame)
    }
}

/// One showcase. Implemented by a zero-sized marker per showcase module; the
/// scene state itself lives in a field on [`App`]. Every method has a default,
/// so a showcase writes only the hooks it actually uses.
pub(crate) trait Showcase {
    /// Whether the host should call [`build`](Self::build) before the next frame.
    fn needs_build(&self, _app: &crate::App) -> bool {
        false
    }

    /// Build the scene and frame the opening camera, once, on the first frame
    /// after this showcase becomes active.
    fn build(&self, _app: &mut crate::App, _renderer: &mut vpl::ViewportRenderer) {}

    /// Collect the render items and lighting for this frame.
    fn scene(
        &self,
        _app: &mut crate::App,
        _frame: &crate::eframe::Frame,
        _out: &mut crate::SceneOverrides,
    ) -> crate::SceneContents {
        crate::SceneContents::empty()
    }

    /// Fold per-frame extras into the assembled frame: items, overlays, and
    /// effect settings that are not part of the scene.
    fn frame(&self, _app: &mut crate::App, _fd: &mut vpl::FrameData, _ctx: &crate::FrameCtx) {}

    /// Draw an egui overlay on top of the rendered viewport.
    fn overlay(
        &self,
        _app: &mut crate::App,
        _ui: &mut crate::eframe::egui::Ui,
        _cx: &crate::ViewportCtx,
    ) {
    }

    /// Advance animation and request another frame, after the viewport is drawn.
    fn tick(&self, _app: &mut crate::App, _cx: &crate::ViewportCtx) {}

    /// Route a plain viewport click that no gizmo or widget consumed.
    fn on_click(&self, _app: &mut crate::App, _cx: &crate::ClickCtx) {}

    /// Handle drag gestures this showcase owns, before the camera controller runs.
    fn drag_input(&self, _app: &mut crate::App, _cx: &crate::ViewportCtx) {}

    /// Advance this showcase's own camera animation or object motion.
    fn advance(&self, _app: &mut crate::App, _cx: &crate::ViewportCtx) {}

    /// Update interactive widgets for the frame.
    fn widgets(&self, _app: &mut crate::App, _cx: &crate::ViewportCtx) {}

    /// Flush any per-frame GPU writes this showcase has queued.
    fn flush_gpu(&self, _app: &mut crate::App, _cx: &crate::ViewportCtx) {}

    /// Cache gizmo placement for next frame's hit-testing.
    fn cache_gizmo(&self, _app: &mut crate::App, _cx: &crate::ViewportCtx) {}

    /// Take over the whole viewport for this frame. Returning true skips the
    /// host's single-viewport path entirely.
    fn viewport_override(
        &self,
        _app: &mut crate::App,
        _ui: &mut crate::eframe::egui::Ui,
        _cx: &crate::ViewportCtx,
    ) -> bool {
        false
    }

    /// Drive the orbit controller directly, for showcases that need the
    /// resulting `ActionFrame`. Returning true leaves the camera to this
    /// showcase.
    fn drive_camera(&self, _app: &mut crate::App, _cx: &crate::ViewportCtx) -> bool {
        false
    }

    /// Whether the orbit should resolve without moving the camera this frame,
    /// because this showcase is using the drag for something of its own.
    fn suppress_orbit(&self, _app: &crate::App, _cx: &crate::ViewportCtx) -> bool {
        false
    }

    /// Controls panel for this showcase, drawn in the left side panel.
    fn controls(
        &self,
        _app: &mut crate::App,
        _ui: &mut crate::eframe::egui::Ui,
        _frame: &crate::eframe::Frame,
    ) {
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

impl SceneContents {
    /// An empty scene: no items, no background override, default lighting.
    pub(crate) fn empty() -> Self {
        Self {
            items: Vec::new(),
            bg_colour: None,
            lighting: LightingSettings::default(),
            scene_gen: 0,
            sel_gen: 0,
        }
    }
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

/// What a viewport click carries, passed to a showcase's `on_click` hook.
pub(crate) struct ClickCtx<'a> {
    pub(crate) frame: &'a eframe::Frame,
    /// Cursor position in viewport pixels.
    pub(crate) pos: glam::Vec2,
    /// Viewport size in logical points.
    pub(crate) w: f32,
    pub(crate) h: f32,
    pub(crate) pixels_per_point: f32,
}

/// The viewport widget's per-frame egui handles, passed to a showcase's
/// `overlay` and `tick` hooks.
pub(crate) struct ViewportCtx<'a> {
    pub(crate) egui: &'a egui::Context,
    pub(crate) frame: &'a eframe::Frame,
    pub(crate) response: &'a egui::Response,
    pub(crate) rect: egui::Rect,
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
        let contents = self.mode.showcase().scene(self, frame, &mut overrides);
        let SceneContents {
            items: scene_items,
            bg_colour,
            lighting,
            scene_gen,
            sel_gen,
        } = contents;
        let SceneOverrides {
            clip_objects: adv_clip_objects,
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
        self.mode.showcase().frame(self, &mut fd, &frame_ctx);

        fd
    }
}

// ---------------------------------------------------------------------------
// Selection / picking
// ---------------------------------------------------------------------------

impl App {
    /// Route a plain viewport click to the active showcase.
    fn handle_click_select(&mut self, cx: &ClickCtx) {
        self.mode.showcase().on_click(self, cx)
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
