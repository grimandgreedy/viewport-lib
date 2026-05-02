//! Scalar colormap + BackfacePolicy::Pattern repro test
//!
//! Tests the bug report: surface with an active scalar attribute and
//! BackfacePolicy::Pattern renders white instead of colorized.
//!
//! A sphere is set up with a vertex scalar attribute ("pressure", 0..1
//! linearly by vertex index) and a Viridis colormap. Its backface policy
//! is set to Pattern::Checker so is_two_sided() returns true, which routes
//! it to the solid_two_sided pipeline. It should render colorized; if it
//! renders white or a solid color the bug is confirmed.
//!
//! Navigation: Left drag / Middle drag = orbit, Right drag = pan, Scroll = zoom.

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    AttributeData, AttributeKind, AttributeRef, BackfacePattern, BackfacePolicy, BuiltinColormap,
    ButtonState, Camera, CameraFrame, FrameData, LightingSettings, MeshData, SceneFrame,
    SceneRenderItem, ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer,
    OrbitCameraController, primitives,
};
use std::sync::{Arc, Mutex};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : scalar + backface pattern repro",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
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
            let res = renderer.resources_mut();

            // Initialize colormaps so builtin_colormap_id is available.
            res.ensure_colormaps_initialized(device, queue);
            let colormap_id = res.builtin_colormap_id(BuiltinColormap::Viridis);

            // Build a sphere with a vertex scalar attribute (pressure: 0..1 by vertex index).
            let sphere_data = primitives::sphere(0.8, 24, 12);
            let n_verts = sphere_data.positions.len();
            let scalars: Vec<f32> = (0..n_verts)
                .map(|i| i as f32 / (n_verts - 1).max(1) as f32)
                .collect();

            let mut mesh_data = MeshData::default();
            mesh_data.positions = sphere_data.positions;
            mesh_data.normals = sphere_data.normals;
            mesh_data.indices = sphere_data.indices;
            mesh_data
                .attributes
                .insert("pressure".to_string(), AttributeData::Vertex(scalars));

            let mesh_id = res.upload_mesh_data(device, &mesh_data).unwrap();

            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App::new(mesh_id, colormap_id)))
        }),
    )
}

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    scene_items: Vec<SceneRenderItem>,
    cursor_viewport: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,
    colormap_id: viewport_lib::ColormapId,
    use_pattern: bool,
}

impl App {
    fn new(mesh_id: viewport_lib::MeshId, colormap_id: viewport_lib::ColormapId) -> Self {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.visible = true;
        item.active_attribute = Some(AttributeRef {
            name: "pressure".into(),
            kind: AttributeKind::Vertex,
        });
        item.colormap_id = Some(colormap_id);
        item.scalar_range = Some((0.0, 1.0));
        // Pattern backface policy: this makes is_two_sided() return true.
        item.material.backface_policy = BackfacePolicy::Pattern {
            pattern: BackfacePattern::Checker,
            color: [0.08, 0.35, 0.90],
        };

        Self {
            camera: Camera {
                distance: 4.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            scene_items: vec![item],
            cursor_viewport: None,
            cursor_prev: None,
            colormap_id,
            use_pattern: true,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            // Toggle instructions.
            let policy_label = if self.use_pattern {
                "BackfacePolicy::Pattern (is_two_sided=true) — press T to toggle"
            } else {
                "BackfacePolicy::Cull (is_two_sided=false) — press T to toggle"
            };
            ui.painter().text(
                rect.min + egui::vec2(12.0, 12.0),
                egui::Align2::LEFT_TOP,
                policy_label,
                egui::FontId::proportional(16.0),
                egui::Color32::WHITE,
            );
            ui.painter().text(
                rect.min + egui::vec2(12.0, 36.0),
                egui::Align2::LEFT_TOP,
                "Sphere should render Viridis colormap in both modes",
                egui::FontId::proportional(14.0),
                egui::Color32::from_rgb(200, 200, 200),
            );

            // Toggle backface policy on T press.
            if ui.input(|i| i.key_pressed(egui::Key::T)) {
                self.use_pattern = !self.use_pattern;
                let item = &mut self.scene_items[0];
                item.material.backface_policy = if self.use_pattern {
                    BackfacePolicy::Pattern {
                        pattern: BackfacePattern::Checker,
                        color: [0.08, 0.35, 0.90],
                    }
                } else {
                    BackfacePolicy::Cull
                };
            }

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
                        egui::Event::PointerButton { button, pressed, .. } => {
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

            let mut frame_data = FrameData::new(
                CameraFrame::from_camera(&self.camera, [w, h]),
                SceneFrame::from_surface_items(self.scene_items.clone()),
            );
            frame_data.effects.lighting = LightingSettings::default();

            ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
                rect,
                viewport_callback::ViewportCallback {
                    frame: frame_data,
                    pick_cursor: None,
                    pick_result: Arc::new(Mutex::new(None)),
                },
            ));
        });
    }
}
