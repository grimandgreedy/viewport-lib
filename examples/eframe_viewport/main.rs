//! `eframe` / `egui` integration for `viewport-lib`.
//!
//! Uses `egui_wgpu` callback resources to share the host renderer and drive the
//! viewport from an ordinary `egui` app.

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    ButtonState, CameraFrame, FrameData, LightKind, LightSource, LightingSettings,
    OrbitCameraController, SceneFrame, ScrollUnits, ViewportContext, ViewportEvent,
    ViewportRenderer, primitives,
};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib - eframe Example",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 600.0]),
            // The viewport's GPU pipelines require a depth-stencil attachment
            // (Depth24PlusStencil8 - stencil is used for selection outlines).
            depth_buffer: 24,
            stencil_buffer: 8,
            ..Default::default()
        },
        Box::new(|cc| {
            // Register the ViewportRenderer as a callback resource so the
            // paint callback can access it during rendering.
            let wgpu_render_state = cc
                .wgpu_render_state
                .as_ref()
                .expect("eframe must be configured with wgpu backend");
            let device = &wgpu_render_state.device;
            let format = wgpu_render_state.target_format;
            let renderer = ViewportRenderer::new(device, format);
            wgpu_render_state
                .renderer
                .write()
                .callback_resources
                .insert(renderer);

            // Upload the box mesh once at startup.
            let mut guard = wgpu_render_state.renderer.write();
            if let Some(vr) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                // Pre-upload a few meshes so objects added later can reference them.
                let cube = primitives::cube(1.0);
                for _ in 0..16 {
                    vr.resources_mut()
                        .upload_mesh_data(device, &cube)
                        .expect("built-in mesh");
                }
            }

            Ok(Box::new(App::default()))
        }),
    )
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct App {
    objects: Vec<SceneObj>,
    next_id: u64,
    camera: viewport_lib::Camera,
    controller: OrbitCameraController,
}

#[derive(Clone)]
struct SceneObj {
    id: u64,
    name: String,
    position: [f32; 3],
    mesh_index: usize,
}

impl Default for App {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            next_id: 1,
            camera: viewport_lib::Camera {
                center: glam::Vec3::ZERO,
                distance: 12.0,
                orientation: glam::Quat::from_rotation_y(0.6) * glam::Quat::from_rotation_x(-0.4),
                ..viewport_lib::Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Left panel: object list ---
        egui::SidePanel::left("object_panel")
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Objects");
                if ui.button("+ Add Box").clicked() {
                    let id = self.next_id;
                    self.next_id += 1;
                    let n = self.objects.len() as f32;
                    let x = (n % 4.0) * 2.0 - 3.0;
                    let z = (n / 4.0).floor() * 2.0 - 3.0;
                    // Each object uses mesh slot = its index (pre-uploaded at startup).
                    let mesh_index = self.objects.len();
                    self.objects.push(SceneObj {
                        id,
                        name: format!("Box {id}"),
                        position: [x, 0.0, z],
                        mesh_index,
                    });
                }

                ui.separator();

                let mut to_remove = None;
                for obj in &self.objects {
                    ui.horizontal(|ui| {
                        ui.label(&obj.name);
                        if ui.small_button("x").clicked() {
                            to_remove = Some(obj.id);
                        }
                    });
                }
                if let Some(id) = to_remove {
                    self.objects.retain(|o| o.id != id);
                }
            });

        // --- Central panel: viewport ---
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

            // Begin frame for input controller.
            self.controller.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            });

            // Translate egui events to ViewportEvents.
            ui.input(|i| {
                // Modifier state
                let mods = viewport_lib::Modifiers {
                    alt: i.modifiers.alt,
                    shift: i.modifiers.shift,
                    ctrl: i.modifiers.command,
                };
                self.controller
                    .push_event(ViewportEvent::ModifiersChanged(mods));

                // Pointer position (viewport-local coordinates)
                if let Some(pos) = i.pointer.interact_pos() {
                    let local = glam::Vec2::new(pos.x - rect.left(), pos.y - rect.top());
                    self.controller
                        .push_event(ViewportEvent::PointerMoved { position: local });
                }

                // Mouse buttons and wheel events
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
                            self.controller.push_event(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units: ScrollUnits::Pixels,
                            });
                        }
                        _ => {}
                    }
                }
            });

            // Apply to camera.
            self.controller.apply_to_camera(&mut self.camera);

            // Update camera aspect ratio.
            self.camera.set_aspect_ratio(rect.width(), rect.height());

            // Build scene items.
            let scene_items: Vec<viewport_lib::SceneRenderItem> = self
                .objects
                .iter()
                .map(|obj| {
                    let model = glam::Mat4::from_translation(glam::Vec3::from(obj.position));
                    {
                        let mut item = viewport_lib::SceneRenderItem::default();
                        item.mesh_index = obj.mesh_index;
                        item.model = model.to_cols_array_2d();
                        item
                    }
                })
                .collect();

            let mut frame_data = FrameData::new(
                CameraFrame::from_camera(&self.camera, [rect.width(), rect.height()]),
                SceneFrame::from_surface_items(scene_items),
            );
            frame_data.effects.lighting = {
                let mut ls = LightingSettings::default();
                // 8 point lights, one per octant, aimed toward the origin.
                let d = 10.0_f32;
                let s = d / 3.0_f32.sqrt();
                ls.lights = [
                    [s, s, s],
                    [-s, s, s],
                    [s, -s, s],
                    [-s, -s, s],
                    [s, s, -s],
                    [-s, s, -s],
                    [s, -s, -s],
                    [-s, -s, -s],
                ]
                .into_iter()
                .map(|position| LightSource {
                    kind: LightKind::Point {
                        position,
                        range: d * 2.0,
                    },
                    color: [1.0, 1.0, 1.0],
                    intensity: 0.5,
                })
                .collect();
                ls
            };
            frame_data.viewport.show_grid = true;
            frame_data.viewport.show_axes_indicator = true;

            // Schedule the viewport paint callback.
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback { frame: frame_data },
                ));

            // Show grab cursor when hovering the viewport.
            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}
