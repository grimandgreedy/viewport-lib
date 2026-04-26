//! Minimal viewport-lib example using eframe / egui.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, LightingSettings, ManipResult,
    ManipulationContext, ManipulationController, Material, OrbitCameraController, SceneFrame,
    SceneRenderItem, ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Minimal",
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
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);
            let res = renderer.resources_mut();

            let m_sphere = res
                .upload_mesh_data(device, &primitives::sphere(0.6, 24, 12))
                .unwrap();
            let m_cube = res
                .upload_mesh_data(device, &primitives::cube(1.0))
                .unwrap();
            let m_torus = res
                .upload_mesh_data(device, &primitives::torus(0.5, 0.18, 32, 16))
                .unwrap();

            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App::new(m_sphere, m_cube, m_torus)))
        }),
    )
}

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    manip: ManipulationController,
    scene_items: Vec<SceneRenderItem>,
    selected: Option<usize>,
    transforms_snapshot: Vec<[[f32; 4]; 4]>,
    // Per-frame drag/click state for ManipulationContext
    cursor_viewport: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,
    left_held: bool,
    drag_started: bool,
    clicked: bool,
    press_origin: Option<glam::Vec2>,
}

impl App {
    fn new(m_sphere: usize, m_cube: usize, m_torus: usize) -> Self {
        let mut make = |mesh_index, [x, y, z]: [f32; 3], color: [f32; 3]| {
            let mut item = SceneRenderItem::default();
            item.mesh_index = mesh_index;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
            item.material = Material::from_color(color);
            item.two_sided = true;
            item
        };

        Self {
            camera: Camera {
                distance: 10.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            manip: ManipulationController::new(),
            scene_items: vec![
                make(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2]),
                make(m_cube, [0.0, 0.0, 0.0], [0.4, 0.6, 0.9]),
                make(m_torus, [2.5, 0.0, 0.0], [0.3, 0.8, 0.4]),
            ],
            selected: None,
            transforms_snapshot: Vec::new(),
            cursor_viewport: None,
            cursor_prev: None,
            left_held: false,
            drag_started: false,
            clicked: false,
            press_origin: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            // Reset per-frame flags.
            self.drag_started = false;
            self.clicked = false;

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

                // Cursor position in viewport-local space.
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

                            if *button == egui::PointerButton::Primary {
                                if *pressed {
                                    self.left_held = true;
                                    self.press_origin = self.cursor_viewport;
                                    self.drag_started = true;
                                } else {
                                    let is_click = self
                                        .press_origin
                                        .zip(self.cursor_viewport)
                                        .map(|(o, c)| (c - o).length() < 5.0)
                                        .unwrap_or(false);
                                    if is_click {
                                        self.clicked = true;
                                    }
                                    self.left_held = false;
                                    self.press_origin = None;
                                }
                            }
                        }
                        egui::Event::MouseWheel { delta, .. } => {
                            self.controller.push_event(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units: ScrollUnits::Pixels,
                            });
                        }
                        _ => {}
                    }
                }
            });

            let w = rect.width();
            let h = rect.height();

            let pointer_delta = self
                .cursor_viewport
                .zip(self.cursor_prev)
                .map(|(c, p)| c - p)
                .unwrap_or(glam::Vec2::ZERO);

            let selection_center = self.selected.map(|i| {
                let col = self.scene_items[i].model[3];
                glam::Vec3::new(col[0], col[1], col[2])
            });

            let manip_ctx = ManipulationContext {
                camera: self.camera.clone(),
                viewport_size: glam::Vec2::new(w, h),
                cursor_viewport: self.cursor_viewport,
                pointer_delta,
                selection_center,
                gizmo: None,
                drag_started: self.drag_started,
                dragging: self.left_held,
                clicked: self.clicked,
            };

            let action_frame = if self.manip.is_active() {
                self.controller.resolve()
            } else {
                self.controller.apply_to_camera(&mut self.camera)
            };
            self.camera.set_aspect_ratio(w, h);

            match self.manip.update(&action_frame, manip_ctx) {
                ManipResult::Update(delta) => {
                    if let Some(idx) = self.selected {
                        let current = glam::Mat4::from_cols_array_2d(&self.scene_items[idx].model);
                        let delta_mat = glam::Mat4::from_scale_rotation_translation(
                            delta.scale,
                            delta.rotation,
                            delta.translation,
                        );
                        self.scene_items[idx].model = (delta_mat * current).to_cols_array_2d();
                    }
                }
                ManipResult::Commit => {}
                ManipResult::Cancel | ManipResult::ConstraintChanged => {
                    for (item, snap) in self
                        .scene_items
                        .iter_mut()
                        .zip(self.transforms_snapshot.iter())
                    {
                        item.model = *snap;
                    }
                }
                ManipResult::None => {
                    self.transforms_snapshot = self.scene_items.iter().map(|i| i.model).collect();
                }
            }

            let mut frame_data = FrameData::new(
                CameraFrame::from_camera(&self.camera, [w, h]),
                SceneFrame::from_surface_items(self.scene_items.clone()),
            );
            frame_data.effects.lighting = LightingSettings::default();

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback { frame: frame_data },
                ));

            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}
