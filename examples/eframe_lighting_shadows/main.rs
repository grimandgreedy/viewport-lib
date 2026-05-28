//! Lighting and shadows test example using eframe / egui.
//!
//! Use the left panel to adjust all lighting and shadow settings.
//! Switch between scene tabs at the top to test different material configurations.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    AtlasViewerCorner, BackfacePolicy, BuiltinMatcap, ButtonState, Camera, CameraFrame,
    DebugOutputMode, DebugQuantity, DebugVis, FrameData, LightKind, LightSource, LightingSettings,
    MatcapId, Material, MeshId, OrbitCameraController, SceneFrame, SceneRenderItem, ScrollUnits,
    ShadowDebugStats, ShadowFilter, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};

// Percy photo: pre-converted raw RGBA (2203 x 2009).
const PERCY_WIDTH: u32 = 2203;
const PERCY_HEIGHT: u32 = 2009;
const PERCY_RGBA: &[u8] = include_bytes!("../eframe_showcase/percy.rgba");

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Lighting & Shadows",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 900.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let device = &rs.device;
            let queue = &rs.queue;
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);

            let (
                m_ground, m_sphere, m_cube, m_torus,
                m_ground2, m_clay, m_ceramic, m_metal, m_rough, m_cube2, m_percy,
                tex_percy,
            );
            {
                let res = renderer.resources_mut();

                res.ensure_matcaps_initialized(device, queue);

                // Tab 1: Basic geometry
                m_ground = res
                    .upload_mesh_data(device, &primitives::cuboid(24.0, 24.0, 0.5))
                    .expect("ground");
                m_sphere = res
                    .upload_mesh_data(device, &primitives::sphere(0.6, 32, 16))
                    .expect("sphere");
                m_cube = res
                    .upload_mesh_data(device, &primitives::cube(1.0))
                    .expect("cube");
                m_torus = res
                    .upload_mesh_data(device, &primitives::torus(0.5, 0.18, 40, 20))
                    .expect("torus");

                // Tab 2: Material variety.
                // Each sphere needs its own mesh upload so the non-instanced path gives each
                // object independent GPU state (object_uniform_buf / bind_group).
                m_ground2 = res
                    .upload_mesh_data(device, &primitives::cuboid(24.0, 24.0, 0.5))
                    .expect("ground2");
                m_clay = res
                    .upload_mesh_data(device, &primitives::sphere(0.7, 40, 20))
                    .expect("clay sphere");
                m_ceramic = res
                    .upload_mesh_data(device, &primitives::sphere(0.7, 40, 20))
                    .expect("ceramic sphere");
                m_metal = res
                    .upload_mesh_data(device, &primitives::sphere(0.7, 40, 20))
                    .expect("metal sphere");
                m_rough = res
                    .upload_mesh_data(device, &primitives::sphere(0.7, 40, 20))
                    .expect("rough sphere");
                m_cube2 = res
                    .upload_mesh_data(device, &primitives::cube(1.2))
                    .expect("cube2");

                let percy_aspect = PERCY_HEIGHT as f32 / PERCY_WIDTH as f32;
                m_percy = res
                    .upload_mesh_data(
                        device,
                        &primitives::plane(4.5, 4.5 * percy_aspect),
                    )
                    .expect("percy plane");
                tex_percy = res
                    .upload_texture(device, queue, PERCY_WIDTH, PERCY_HEIGHT, PERCY_RGBA)
                    .expect("percy texture");
            }

            let matcap_clay = renderer.resources().builtin_matcap_id(BuiltinMatcap::Clay);
            let matcap_ceramic = renderer.resources().builtin_matcap_id(BuiltinMatcap::Ceramic);

            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App::new(
                m_ground, m_sphere, m_cube, m_torus,
                m_ground2, m_clay, m_ceramic, m_metal, m_rough, m_cube2,
                m_percy, tex_percy, matcap_clay, matcap_ceramic,
            )))
        }),
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Basic,
    Materials,
}

struct App {
    // Camera / input
    camera: Camera,
    controller: OrbitCameraController,
    cursor_viewport: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,

    // Scene selection
    tab: Tab,

    // Mesh / texture / matcap IDs
    m_ground: MeshId,
    m_sphere: MeshId,
    m_cube: MeshId,
    m_torus: MeshId,
    m_ground2: MeshId,
    m_clay: MeshId,
    m_ceramic: MeshId,
    m_metal: MeshId,
    m_rough: MeshId,
    m_cube2: MeshId,
    m_percy: MeshId,
    tex_percy: u64,
    matcap_clay: MatcapId,
    matcap_ceramic: MatcapId,

    // Light source parameters.
    // Stored per-kind so switching kinds does not lose previously entered values.
    light_kind: u8, // 0 = Directional, 1 = Point, 2 = Spot
    light_colour: [f32; 3],
    light_intensity: f32,
    dir_direction: [f32; 3],
    point_position: [f32; 3],
    point_range: f32,
    spot_position: [f32; 3],
    spot_direction: [f32; 3],
    spot_range: f32,
    spot_inner_deg: f32,
    spot_outer_deg: f32,

    // Shadow settings
    shadows_enabled: bool,
    shadow_bias: f32,
    shadow_cascade_count: u32,
    shadow_filter: ShadowFilter,
    pcss_light_radius: f32,
    shadow_atlas_resolution: u32,
    shadow_extent_enabled: bool,
    shadow_extent_value: f32,

    // Hemisphere ambient
    hemisphere_intensity: f32,
    sky_colour: [f32; 3],
    ground_colour: [f32; 3],

    // Debug visualization
    debug_vis_active: bool,
    debug_vis_mode_replace: bool,
    debug_vis_splitscreen: bool,
    debug_vis_split_x: f32,
    debug_vis_r: DebugQuantity,
    debug_vis_g: DebugQuantity,
    debug_vis_b: DebugQuantity,
    debug_vis_scale: f32,

    // Shadow atlas viewer
    show_shadow_atlas: bool,
    atlas_viewer_corner: AtlasViewerCorner,
    atlas_viewer_scale: f32,

    // Pixel inspector (D7)
    pixel_inspector_active: bool,
    pixel_read_req: std::sync::Arc<std::sync::Mutex<Option<(u32, u32)>>>,
    pixel_read_res: std::sync::Arc<std::sync::Mutex<Option<[f32; 4]>>>,
    last_picked_pos: Option<(u32, u32)>,
    last_picked_values: Option<[f32; 4]>,

    // Instancing status (updated each frame by the paint callback)
    instancing_status: std::sync::Arc<std::sync::Mutex<(bool, usize)>>,
    // Shadow debug stats (updated each frame by the paint callback)
    shadow_stats: std::sync::Arc<std::sync::Mutex<Option<ShadowDebugStats>>>,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        m_ground: MeshId, m_sphere: MeshId, m_cube: MeshId, m_torus: MeshId,
        m_ground2: MeshId, m_clay: MeshId, m_ceramic: MeshId,
        m_metal: MeshId, m_rough: MeshId, m_cube2: MeshId,
        m_percy: MeshId, tex_percy: u64,
        matcap_clay: MatcapId, matcap_ceramic: MatcapId,
    ) -> Self {
        Self {
            camera: Camera {
                distance: 18.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            cursor_viewport: None,
            cursor_prev: None,
            tab: Tab::Basic,
            m_ground, m_sphere, m_cube, m_torus,
            m_ground2, m_clay, m_ceramic, m_metal, m_rough, m_cube2,
            m_percy, tex_percy, matcap_clay, matcap_ceramic,
            // Match showcase_27 defaults: a known-problematic setup useful for reproducing bugs.
            light_kind: 0,
            light_colour: [1.0, 0.97, 0.90],
            light_intensity: 0.8,
            dir_direction: [0.3, -0.5, 0.8],
            point_position: [0.0, 0.0, 5.0],
            point_range: 20.0,
            spot_position: [0.0, -8.0, 6.0],
            spot_direction: [0.0, 0.5, -0.6],
            spot_range: 20.0,
            spot_inner_deg: 15.0,
            spot_outer_deg: 25.0,
            shadows_enabled: true,
            shadow_bias: 0.0,
            shadow_cascade_count: 4,
            shadow_filter: ShadowFilter::Pcf,
            pcss_light_radius: 0.02,
            shadow_atlas_resolution: 4096,
            shadow_extent_enabled: false,
            shadow_extent_value: 20.0,
            hemisphere_intensity: 0.2,
            sky_colour: [0.8, 0.9, 1.0],
            ground_colour: [0.5, 0.55, 0.6],
            debug_vis_active: false,
            debug_vis_mode_replace: true,
            debug_vis_splitscreen: false,
            debug_vis_split_x: 0.5,
            debug_vis_r: DebugQuantity::ShadowFactor,
            debug_vis_g: DebugQuantity::Zero,
            debug_vis_b: DebugQuantity::Zero,
            debug_vis_scale: 1.0,
            show_shadow_atlas: false,
            atlas_viewer_corner: AtlasViewerCorner::BottomRight,
            atlas_viewer_scale: 0.3,
            pixel_inspector_active: false,
            pixel_read_req: std::sync::Arc::new(std::sync::Mutex::new(None)),
            pixel_read_res: std::sync::Arc::new(std::sync::Mutex::new(None)),
            last_picked_pos: None,
            last_picked_values: None,
            instancing_status: std::sync::Arc::new(std::sync::Mutex::new((false, 0))),
            shadow_stats: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    fn build_lighting(&self) -> LightingSettings {
        let kind = match self.light_kind {
            0 => LightKind::Directional {
                direction: self.dir_direction,
            },
            1 => LightKind::Point {
                position: self.point_position,
                range: self.point_range,
            },
            _ => LightKind::Spot {
                position: self.spot_position,
                direction: self.spot_direction,
                range: self.spot_range,
                inner_angle: self.spot_inner_deg.to_radians(),
                outer_angle: self.spot_outer_deg.to_radians(),
            },
        };
        {
            let mut _t = LightingSettings::default();
            _t.lights = vec![{
                let mut _t = LightSource::default();
                _t.kind = kind;
                _t.colour = self.light_colour;
                _t.intensity = self.light_intensity;
                _t
            }];
            _t.shadows_enabled = self.shadows_enabled;
            _t.shadow_bias = self.shadow_bias;
            _t.shadow_cascade_count = self.shadow_cascade_count;
            _t.shadow_filter = self.shadow_filter;
            _t.pcss_light_radius = self.pcss_light_radius;
            _t.shadow_atlas_resolution = self.shadow_atlas_resolution;
            _t.shadow_extent_override = if self.shadow_extent_enabled {
                Some(self.shadow_extent_value)
            } else {
                None
            };
            _t.hemisphere_intensity = self.hemisphere_intensity;
            _t.sky_colour = self.sky_colour;
            _t.ground_colour = self.ground_colour;
            _t.debug_vis = {
                let mut dv = DebugVis::default();
                dv.active = self.debug_vis_active;
                dv.mode = if self.debug_vis_splitscreen {
                    DebugOutputMode::SplitScreen
                } else if self.debug_vis_mode_replace {
                    DebugOutputMode::Replace
                } else {
                    DebugOutputMode::TintOverlay
                };
                dv.split_x = self.debug_vis_split_x;
                dv.channel_r = self.debug_vis_r;
                dv.channel_g = self.debug_vis_g;
                dv.channel_b = self.debug_vis_b;
                dv.scale = self.debug_vis_scale;
                dv
            };
            _t
        }
    }

    fn build_basic_items(&self) -> Vec<SceneRenderItem> {
        let mut items = Vec::new();

        // Ground platform: light warm sand. Cuboid with top surface at z=0.
        let mut ground = SceneRenderItem::default();
        ground.mesh_id = self.m_ground;
        ground.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25))
            .to_cols_array_2d();
        ground.material = Material::from_colour([0.88, 0.84, 0.76]);
        ground.material.roughness = 0.85;
        ground.material.backface_policy = BackfacePolicy::Cull;
        items.push(ground);
        
        // Sphere: light sage green. Acne is easy to spot on light curved surfaces.
        let mut sphere = SceneRenderItem::default();
        sphere.mesh_id = self.m_sphere;
        sphere.model =
            glam::Mat4::from_translation(glam::Vec3::new(-4.0, 0.0, 0.6)).to_cols_array_2d();
        sphere.material = Material::from_colour([0.78, 0.90, 0.80]);
        items.push(sphere);

        // Cube: light periwinkle. Flat faces isolate bias and gap artifacts.
        let mut cube = SceneRenderItem::default();
        cube.mesh_id = self.m_cube;
        cube.model =
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.5)).to_cols_array_2d();
        cube.material = Material::from_colour([0.78, 0.83, 0.95]);
        items.push(cube);

        // Torus: light peach. Mixed geometry useful for bright-spot test.
        // Z offset 0.18 = torus minor radius, places the bottom of the tube on the ground.
        let mut torus = SceneRenderItem::default();
        torus.mesh_id = self.m_torus;
        torus.model =
            glam::Mat4::from_translation(glam::Vec3::new(4.0, 0.0, 0.18)).to_cols_array_2d();
        torus.material = Material::from_colour([0.95, 0.82, 0.74]);
        items.push(torus);

        items
    }

    fn build_materials_items(&self) -> Vec<SceneRenderItem> {
        let mut items = Vec::new();

        // Ground platform: neutral light grey, top surface at z=0.
        let mut ground = SceneRenderItem::default();
        ground.mesh_id = self.m_ground2;
        ground.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25))
            .to_cols_array_2d();
        ground.material = Material::from_colour([0.85, 0.85, 0.85]);
        ground.material.roughness = 0.85;
        items.push(ground);

        // Clay matcap sphere (blendable): off-white base exposes matcap tint while keeping
        // surfaces light enough to show shadow detail.
        let mut clay = SceneRenderItem::default();
        clay.mesh_id = self.m_clay;
        clay.model =
            glam::Mat4::from_translation(glam::Vec3::new(-4.5, 2.5, 0.7)).to_cols_array_2d();
        clay.material = Material::from_colour([0.92, 0.90, 0.88]);
        clay.material.shading_model = viewport_lib::ShadingModel::Matcap(self.matcap_clay);
        items.push(clay);

        // Ceramic matcap sphere (static): white base, high-contrast sheen.
        // Good for checking shadow edge quality.
        let mut ceramic = SceneRenderItem::default();
        ceramic.mesh_id = self.m_ceramic;
        ceramic.model =
            glam::Mat4::from_translation(glam::Vec3::new(-1.5, 2.5, 0.7)).to_cols_array_2d();
        ceramic.material = Material::from_colour([1.0, 1.0, 1.0]);
        ceramic.material.shading_model = viewport_lib::ShadingModel::Matcap(self.matcap_ceramic);
        items.push(ceramic);

        // PBR metallic sphere: near-white, very smooth. Specular response reveals shadow
        // terminator and banding at glancing angles.
        let mut metal = SceneRenderItem::default();
        metal.mesh_id = self.m_metal;
        metal.model =
            glam::Mat4::from_translation(glam::Vec3::new(1.5, 2.5, 0.7)).to_cols_array_2d();
        metal.material = Material::pbr([0.95, 0.95, 0.93], 1.0, 0.08);
        items.push(metal);

        // PBR rough sphere: near-white diffuse. High roughness maximises shadow acne visibility.
        let mut rough = SceneRenderItem::default();
        rough.mesh_id = self.m_rough;
        rough.model =
            glam::Mat4::from_translation(glam::Vec3::new(4.5, 2.5, 0.7)).to_cols_array_2d();
        rough.material = Material::pbr([0.95, 0.95, 0.95], 0.0, 0.95);
        items.push(rough);

        // Plain diffuse cube: light cream. Flat surfaces at cardinal angles show cascade bands
        // as clean rectangular stripes rather than complex shadow shapes.
        let mut cube = SceneRenderItem::default();
        cube.mesh_id = self.m_cube2;
        cube.model =
            glam::Mat4::from_translation(glam::Vec3::new(-3.5, -2.5, 0.6)).to_cols_array_2d();
        cube.material = Material::from_colour([0.96, 0.94, 0.90]);
        items.push(cube);

        // Percy photo plane: lying flat on the ground, slightly raised to avoid Z-fighting.
        // Real-world image content makes acne and bias gaps easy to judge perceptually.
        let mut percy = SceneRenderItem::default();
        percy.mesh_id = self.m_percy;
        percy.model =
            glam::Mat4::from_translation(glam::Vec3::new(2.5, -2.5, 0.005)).to_cols_array_2d();
        percy.material = Material::default();
        percy.material.texture_id = Some(self.tex_percy);
        percy.material.backface_policy = BackfacePolicy::Cull;
        items.push(percy);

        items
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("lighting_panel")
            .min_width(250.0)
            .max_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.ui_lighting_panel(ui);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Tab bar
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Basic, "Basic");
                ui.selectable_value(&mut self.tab, Tab::Materials, "Materials");
            });
            ui.separator();

            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.controller.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            });

            ui.input(|i| {
                self.controller.push_event(ViewportEvent::ModifiersChanged(
                    viewport_lib::Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    },
                ));

                let local_pos = i
                    .pointer
                    .interact_pos()
                    .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
                self.cursor_prev = self.cursor_viewport;
                self.cursor_viewport = local_pos;
                if let Some(pos) = local_pos {
                    self.controller
                        .push_event(ViewportEvent::PointerMoved { position: pos });
                }

                for event in &i.events {
                    match event {
                        egui::Event::PointerButton {
                            button, pressed, ..
                        } => {
                            let vp_button = match button {
                                egui::PointerButton::Primary => viewport_lib::MouseButton::Left,
                                egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
                                egui::PointerButton::Middle => viewport_lib::MouseButton::Middle,
                                _ => continue,
                            };
                            self.controller.push_event(ViewportEvent::MouseButton {
                                button: vp_button,
                                state: if *pressed {
                                    ButtonState::Pressed
                                } else {
                                    ButtonState::Released
                                },
                            });
                        }
                        egui::Event::MouseWheel { unit, delta, .. } => {
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
                        _ => {}
                    }
                }
            });

            let w = rect.width();
            let h = rect.height();
            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(w, h);

            let items = match self.tab {
                Tab::Basic => self.build_basic_items(),
                Tab::Materials => self.build_materials_items(),
            };

            let ppp = ui.ctx().pixels_per_point();
            let mut fd = FrameData::new(
                CameraFrame::from_camera(&self.camera, [w, h])
                    .with_pixels_per_point(ppp),
                SceneFrame::from_surface_items(items),
            );
            fd.effects.lighting = self.build_lighting();
            fd.effects.show_shadow_atlas = self.show_shadow_atlas;
            fd.effects.atlas_viewer_corner = self.atlas_viewer_corner;
            fd.effects.atlas_viewer_scale = self.atlas_viewer_scale;

            // Pixel inspector: on left-click, submit the clicked pixel for readback.
            // Convert from CSS pixels (egui) to physical pixels (what the shader writes).
            if self.pixel_inspector_active && self.debug_vis_active {
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let ppp = ui.ctx().pixels_per_point();
                        let px = ((pos.x - rect.left()) * ppp) as u32;
                        let py = ((pos.y - rect.top()) * ppp) as u32;
                        self.last_picked_pos = Some((px, py));
                        if let Ok(mut req) = self.pixel_read_req.lock() {
                            *req = Some((px, py));
                        }
                    }
                }
            }

            // Collect pixel readback result from previous frame.
            if let Ok(mut res) = self.pixel_read_res.lock() {
                if let Some(vals) = res.take() {
                    self.last_picked_values = Some(vals);
                }
            }

            ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
                rect,
                viewport_callback::ViewportCallback {
                    frame: fd,
                    instancing_status: self.instancing_status.clone(),
                    pixel_read_req: self.pixel_read_req.clone(),
                    pixel_read_res: self.pixel_read_res.clone(),
                    shadow_stats: self.shadow_stats.clone(),
                },
            ));

            // Status bar: show shader path and cascade count (data is one frame behind).
            let (is_instanced, batch_count) = self.instancing_status.lock()
                .map(|g| *g)
                .unwrap_or((false, 0));
            let path_label = if is_instanced {
                format!("Shader path: Instanced ({} batches)", batch_count)
            } else {
                "Shader path: Per-object".to_string()
            };
            let status_text = format!("{}   Cascades: {}", path_label, self.shadow_cascade_count);
            ui.painter().text(
                egui::pos2(rect.left() + 8.0, rect.bottom() - 20.0),
                egui::Align2::LEFT_BOTTOM,
                &status_text,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgba_premultiplied(20,20,20, 200),
            );

            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Lighting panel UI
// ---------------------------------------------------------------------------

impl App {
    fn ui_lighting_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Lighting");
        ui.separator();

        // Light source
        egui::CollapsingHeader::new("Light Source")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.light_kind, 0, "Directional");
                    ui.radio_value(&mut self.light_kind, 1, "Point");
                    ui.radio_value(&mut self.light_kind, 2, "Spot");
                });
                ui.add_space(4.0);

                match self.light_kind {
                    0 => {
                        ui.label("Direction (toward light):");
                        ui_vec3(ui, &mut self.dir_direction, 0.01);
                    }
                    1 => {
                        ui.label("Position:");
                        ui_vec3(ui, &mut self.point_position, 0.1);
                        ui.add(
                            egui::Slider::new(&mut self.point_range, 1.0..=100.0).text("Range"),
                        );
                    }
                    _ => {
                        ui.label("Position:");
                        ui_vec3(ui, &mut self.spot_position, 0.1);
                        ui.label("Direction:");
                        ui_vec3(ui, &mut self.spot_direction, 0.01);
                        ui.add(
                            egui::Slider::new(&mut self.spot_range, 1.0..=100.0).text("Range"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.spot_inner_deg, 1.0..=89.0)
                                .text("Inner angle"),
                        );
                        // Keep outer >= inner
                        self.spot_outer_deg = self.spot_outer_deg.max(self.spot_inner_deg);
                        ui.add(
                            egui::Slider::new(
                                &mut self.spot_outer_deg,
                                self.spot_inner_deg..=89.0,
                            )
                            .text("Outer angle"),
                        );
                    }
                }

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Colour:");
                    ui.color_edit_button_rgb(&mut self.light_colour);
                });
                ui.add(
                    egui::Slider::new(&mut self.light_intensity, 0.0..=1.0).text("Intensity"),
                );
            });

        ui.add_space(4.0);

        // Hemisphere ambient
        egui::CollapsingHeader::new("Hemisphere Ambient")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.hemisphere_intensity, 0.0..=2.0)
                        .text("Intensity"),
                );
                ui.horizontal(|ui| {
                    ui.label("Sky:   ");
                    ui.color_edit_button_rgb(&mut self.sky_colour);
                });
                ui.horizontal(|ui| {
                    ui.label("Ground:");
                    ui.color_edit_button_rgb(&mut self.ground_colour);
                });
            });

        ui.add_space(4.0);

        // Shadows
        egui::CollapsingHeader::new("Shadows")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut self.shadows_enabled, "Enabled");
                ui.add_space(2.0);

                ui.add(
                    egui::Slider::new(&mut self.shadow_bias, 0.0..=0.005)
                        .text("Bias")
                        .custom_formatter(|v, _| format!("{:.5}", v)),
                );

                ui.add_space(4.0);
                ui.label("Cascade count:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.shadow_cascade_count, 1, "1");
                    ui.radio_value(&mut self.shadow_cascade_count, 2, "2");
                    ui.radio_value(&mut self.shadow_cascade_count, 4, "4");
                });


                ui.add_space(4.0);
                ui.label("Filter:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::Pcf, "PCF");
                    ui.radio_value(&mut self.shadow_filter, ShadowFilter::Pcss, "PCSS");
                });
                if self.shadow_filter == ShadowFilter::Pcss {
                    ui.add(
                        egui::Slider::new(&mut self.pcss_light_radius, 0.001..=0.2)
                            .text("PCSS radius"),
                    );
                }

                ui.add_space(4.0);
                ui.label("Atlas resolution:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.shadow_atlas_resolution, 1024, "1K");
                    ui.radio_value(&mut self.shadow_atlas_resolution, 2048, "2K");
                    ui.radio_value(&mut self.shadow_atlas_resolution, 4096, "4K");
                });

                ui.add_space(4.0);
                ui.checkbox(&mut self.shadow_extent_enabled, "Override extent");
                if self.shadow_extent_enabled {
                    ui.add(
                        egui::Slider::new(&mut self.shadow_extent_value, 2.0..=100.0)
                            .text("Extent (m)"),
                    );
                }
            });

        ui.add_space(4.0);

        egui::CollapsingHeader::new("Shadow Atlas Viewer")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.show_shadow_atlas, "Show atlas");
                    if self.show_shadow_atlas {
                        egui::ComboBox::from_id_salt("atlas_corner")
                            .selected_text(atlas_corner_label(self.atlas_viewer_corner))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.atlas_viewer_corner, AtlasViewerCorner::TopLeft, "Top-left");
                                ui.selectable_value(&mut self.atlas_viewer_corner, AtlasViewerCorner::TopRight, "Top-right");
                                ui.selectable_value(&mut self.atlas_viewer_corner, AtlasViewerCorner::BottomLeft, "Bottom-left");
                                ui.selectable_value(&mut self.atlas_viewer_corner, AtlasViewerCorner::BottomRight, "Bottom-right");
                            });
                    }
                });
                if self.show_shadow_atlas {
                    ui.add(egui::Slider::new(&mut self.atlas_viewer_scale, 0.1..=0.6).text("Size"));
                }
            });

        ui.add_space(4.0);

        egui::CollapsingHeader::new("Debug Visualization")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.debug_vis_active, "Active");
                if self.debug_vis_active {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        if ui.small_button("Atlas UV").clicked() {
                            self.debug_vis_r = viewport_lib::DebugQuantity::AtlasUvX;
                            self.debug_vis_g = viewport_lib::DebugQuantity::AtlasUvY;
                            self.debug_vis_b = viewport_lib::DebugQuantity::Zero;
                        }
                        if ui.small_button("Depth compare").clicked() {
                            self.debug_vis_r = viewport_lib::DebugQuantity::BiasedDepth;
                            self.debug_vis_g = viewport_lib::DebugQuantity::SurfaceDepth;
                            self.debug_vis_b = viewport_lib::DebugQuantity::NdotL;
                        }
                        if ui.small_button("Direct light").clicked() {
                            self.debug_vis_r = viewport_lib::DebugQuantity::DirectLightLuminance;
                            self.debug_vis_g = viewport_lib::DebugQuantity::AmbientLuminance;
                            self.debug_vis_b = viewport_lib::DebugQuantity::Zero;
                            self.debug_vis_scale = 1.0;
                        }
                        if ui.small_button("IBL split").clicked() {
                            self.debug_vis_r = viewport_lib::DebugQuantity::IblDiffuseLuminance;
                            self.debug_vis_g = viewport_lib::DebugQuantity::IblSpecularLuminance;
                            self.debug_vis_b = viewport_lib::DebugQuantity::Zero;
                            self.debug_vis_scale = 1.0;
                        }
                    });
                    ui.add_space(4.0);
                    ui.label("Mode:");
                    ui.horizontal(|ui| {
                        if ui.radio(!self.debug_vis_splitscreen && self.debug_vis_mode_replace, "Replace").clicked() {
                            self.debug_vis_splitscreen = false;
                            self.debug_vis_mode_replace = true;
                        }
                        if ui.radio(!self.debug_vis_splitscreen && !self.debug_vis_mode_replace, "Tint overlay").clicked() {
                            self.debug_vis_splitscreen = false;
                            self.debug_vis_mode_replace = false;
                        }
                        if ui.radio(self.debug_vis_splitscreen, "Split screen").clicked() {
                            self.debug_vis_splitscreen = true;
                        }
                    });
                    if self.debug_vis_splitscreen {
                        ui.add(egui::Slider::new(&mut self.debug_vis_split_x, 0.0..=1.0).text("Split"));
                    }
                    ui.add_space(4.0);
                    ui.label("R channel:");
                    egui::ComboBox::from_id_salt("dbg_r")
                        .selected_text(debug_quantity_label(self.debug_vis_r))
                        .show_ui(ui, |ui| {
                            for &q in DebugQuantity::all_variants() {
                                ui.selectable_value(&mut self.debug_vis_r, q, debug_quantity_label(q));
                            }
                        });
                    ui.label("G channel:");
                    egui::ComboBox::from_id_salt("dbg_g")
                        .selected_text(debug_quantity_label(self.debug_vis_g))
                        .show_ui(ui, |ui| {
                            for &q in DebugQuantity::all_variants() {
                                ui.selectable_value(&mut self.debug_vis_g, q, debug_quantity_label(q));
                            }
                        });
                    ui.label("B channel:");
                    egui::ComboBox::from_id_salt("dbg_b")
                        .selected_text(debug_quantity_label(self.debug_vis_b))
                        .show_ui(ui, |ui| {
                            for &q in DebugQuantity::all_variants() {
                                ui.selectable_value(&mut self.debug_vis_b, q, debug_quantity_label(q));
                            }
                        });
                    ui.add_space(4.0);
                    ui.add(
                        egui::Slider::new(&mut self.debug_vis_scale, 0.1..=100.0)
                            .text("Scale")
                            .logarithmic(true),
                    );
                }
            });

        ui.add_space(4.0);

        egui::CollapsingHeader::new("Pixel Inspector")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.pixel_inspector_active, "Active");
                if self.pixel_inspector_active {
                    if !self.debug_vis_active {
                        ui.label("Enable Debug Visualization first.");
                    } else {
                        ui.label("Click a pixel in the viewport to read its debug values.");
                    }
                }

                if let Some((px, py)) = self.last_picked_pos {
                    ui.add_space(4.0);
                    ui.label(format!("Last picked: ({}, {})", px, py));
                    if let Some(vals) = self.last_picked_values {
                        ui.horizontal(|ui| {
                            ui.label("R");
                            ui.label(format!("{} ({:.4})", debug_quantity_label(self.debug_vis_r), vals[0]));
                        });
                        ui.horizontal(|ui| {
                            ui.label("G");
                            ui.label(format!("{} ({:.4})", debug_quantity_label(self.debug_vis_g), vals[1]));
                        });
                        ui.horizontal(|ui| {
                            ui.label("B");
                            ui.label(format!("{} ({:.4})", debug_quantity_label(self.debug_vis_b), vals[2]));
                        });
                    } else {
                        ui.label("(reading...)");
                    }
                }
            });

        ui.add_space(4.0);

        // D8: Frame stats footer.
        egui::CollapsingHeader::new("Frame Stats")
            .default_open(false)
            .show(ui, |ui| {
                let stats = self.shadow_stats.lock().ok().and_then(|g| *g);
                if let Some(s) = stats {
                    let path = if s.using_instanced_path {
                        format!("Instanced ({} batches)", s.instanced_batch_count)
                    } else {
                        "Per-object".to_string()
                    };
                    ui.label(format!("Path:     {}", path));
                    let splits: Vec<String> = s.cascade_splits[..s.cascade_count as usize]
                        .iter()
                        .map(|v| format!("{:.1}", v))
                        .collect();
                    let splits_str = splits.join(" / ");
                    ui.label(format!("Cascades: {}  splits: {}", s.cascade_count, splits_str));
                    let extent_label = if self.shadow_extent_enabled {
                        format!("{:.1} m (override)", s.shadow_extent_world)
                    } else {
                        format!("{:.1} m (auto)", s.shadow_extent_world)
                    };
                    ui.label(format!("Atlas:    {} px   Extent: {}", s.shadow_atlas_resolution, extent_label));
                    let contact = if s.contact_shadow_active { "enabled" } else { "disabled" };
                    ui.label(format!("Contact:  {}", contact));
                } else {
                    ui.label("(no data yet)");
                }
            });

        ui.add_space(4.0);

        // D9: Diagnostic presets.
        egui::CollapsingHeader::new("Diagnostic Presets")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("One-click setups for each known bug class.");
                ui.add_space(4.0);

                if ui.button("Peter-panning / shadow gap").clicked() {
                    self.apply_preset_peter_panning();
                }
                if ui.button("Shadow acne").clicked() {
                    self.apply_preset_acne();
                }
                if ui.button("Cascade band / seam").clicked() {
                    self.apply_preset_cascade_band();
                }
                if ui.button("Ghost shapes (orbit)").clicked() {
                    self.apply_preset_ghost_shapes();
                }
                if ui.button("Contact shadow bright spot").clicked() {
                    self.apply_preset_contact_bright_spot();
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Diagnostic preset helpers
// ---------------------------------------------------------------------------

impl App {
    fn apply_preset_peter_panning(&mut self) {
        // Side-on view of the sphere at (-4, 0, 0.6), low camera angle.
        // SplitScreen: left = normal, right = ShadowFactor.
        self.tab = Tab::Basic;
        self.camera.center = glam::Vec3::new(-4.0, 0.0, 0.6);
        self.camera.distance = 7.0;
        // Looking from the Y+ side, slightly above horizontal.
        self.camera.orientation =
            glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2) * glam::Quat::from_rotation_x(1.42);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::ShadowFactor;
        self.debug_vis_g = DebugQuantity::Zero;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 1.0;
        self.debug_vis_split_x = 0.5;
    }

    fn apply_preset_acne(&mut self) {
        // Steep top-down view of the rough sphere in the Materials tab.
        // SplitScreen: left = normal, right = ShadowFactor (acne appears as bright speckles).
        self.tab = Tab::Materials;
        self.camera.center = glam::Vec3::new(4.5, 2.5, 0.7);
        self.camera.distance = 4.0;
        // Looking down at steep angle.
        self.camera.orientation =
            glam::Quat::from_rotation_z(0.3) * glam::Quat::from_rotation_x(0.5);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::ShadowFactor;
        self.debug_vis_g = DebugQuantity::Zero;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 1.0;
        self.debug_vis_split_x = 0.5;
    }

    fn apply_preset_cascade_band(&mut self) {
        // Pulled back to see the full scene from a moderate angle.
        // SplitScreen: left = normal, right = CascadeIndex (bands as distinct colours).
        self.tab = Tab::Basic;
        self.camera.center = glam::Vec3::new(0.0, 0.0, 0.5);
        self.camera.distance = 22.0;
        self.camera.orientation =
            glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.0);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::CascadeIndex;
        self.debug_vis_g = DebugQuantity::Zero;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 0.33;
        self.debug_vis_split_x = 0.5;
    }

    fn apply_preset_ghost_shapes(&mut self) {
        // Mid-distance orbit view. Orbit while watching AtlasUV on the right half.
        // Unstable UVs during orbit point to cascade matrix instability.
        self.tab = Tab::Basic;
        self.camera.center = glam::Vec3::new(0.0, 0.0, 0.5);
        self.camera.distance = 12.0;
        self.camera.orientation =
            glam::Quat::from_rotation_z(1.0) * glam::Quat::from_rotation_x(1.15);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::AtlasUvX;
        self.debug_vis_g = DebugQuantity::AtlasUvY;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 1.0;
        self.debug_vis_split_x = 0.5;
    }

    fn apply_preset_contact_bright_spot(&mut self) {
        // Looking at the torus from above at a low angle so the inner ring is visible.
        // SplitScreen: left = normal, right = ContactShadowFactor.
        self.tab = Tab::Basic;
        self.camera.center = glam::Vec3::new(4.0, 0.0, 0.18);
        self.camera.distance = 5.0;
        self.camera.orientation =
            glam::Quat::from_rotation_z(-0.4) * glam::Quat::from_rotation_x(1.3);
        self.debug_vis_active = true;
        self.debug_vis_splitscreen = true;
        self.debug_vis_r = DebugQuantity::ContactShadowFactor;
        self.debug_vis_g = DebugQuantity::Zero;
        self.debug_vis_b = DebugQuantity::Zero;
        self.debug_vis_scale = 1.0;
        self.debug_vis_split_x = 0.5;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Three drag-value fields for a vec3, laid out horizontally.
fn ui_vec3(ui: &mut egui::Ui, v: &mut [f32; 3], speed: f64) {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut v[0]).speed(speed).prefix("x: "));
        ui.add(egui::DragValue::new(&mut v[1]).speed(speed).prefix("y: "));
        ui.add(egui::DragValue::new(&mut v[2]).speed(speed).prefix("z: "));
    });
}

/// Display label for an AtlasViewerCorner variant.
#[allow(unreachable_patterns)]
fn atlas_corner_label(c: AtlasViewerCorner) -> &'static str {
    match c {
        AtlasViewerCorner::TopLeft => "Top-left",
        AtlasViewerCorner::TopRight => "Top-right",
        AtlasViewerCorner::BottomLeft => "Bottom-left",
        AtlasViewerCorner::BottomRight => "Bottom-right",
        _ => "?",
    }
}

/// Display label for a DebugQuantity variant.
fn debug_quantity_label(q: DebugQuantity) -> &'static str {
    match q {
        DebugQuantity::Zero => "Zero (black)",
        DebugQuantity::One => "One (white)",
        DebugQuantity::CascadeIndex => "Cascade index",
        DebugQuantity::ShadowFactor => "Shadow factor",
        DebugQuantity::ContactShadowFactor => "Contact shadow",
        DebugQuantity::NdotL => "N dot L",
        DebugQuantity::NormalBiasMagnitude => "Normal bias magnitude",
        DebugQuantity::AtlasUvX => "Atlas UV X",
        DebugQuantity::AtlasUvY => "Atlas UV Y",
        DebugQuantity::TileUvX => "Tile UV X",
        DebugQuantity::TileUvY => "Tile UV Y",
        DebugQuantity::BiasedDepth => "Biased depth",
        DebugQuantity::SurfaceDepth => "Surface depth",
        DebugQuantity::WorldNormalX => "World normal X",
        DebugQuantity::WorldNormalY => "World normal Y",
        DebugQuantity::WorldNormalZ => "World normal Z",
        DebugQuantity::Roughness => "Roughness",
        DebugQuantity::Metallic => "Metallic",
        DebugQuantity::AoFactor => "AO factor",
        DebugQuantity::DirectLightLuminance => "Direct light lum.",
        DebugQuantity::AmbientLuminance => "Ambient lum.",
        DebugQuantity::IblDiffuseLuminance => "IBL diffuse lum.",
        DebugQuantity::IblSpecularLuminance => "IBL specular lum.",
        DebugQuantity::EmissiveLuminance => "Emissive lum.",
        _ => "Unknown",
    }
}
