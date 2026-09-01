//! Step through every scene in the shared test/bench catalogue.
//!
//! This is the visual front end for `viewport-lib-testkit`: the same scene
//! definitions the headless counter and snapshot tests render are shown here for
//! manual inspection. Pick a scene on the left, jump to a named camera, or orbit
//! freely.
//!
//! Navigation:
//!   Left / Middle drag : orbit
//!   Right drag         : pan
//!   Scroll             : zoom

mod viewport_callback;

use eframe::egui;
use viewport_lib::wgpu;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, Modifiers, MouseButton, OrbitCameraController,
    SceneFrame, ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer,
};
use viewport_lib_testkit::{BuildCtx, BuiltScene, NamedCamera, catalogue};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Catalogue Viewer",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 860.0]),
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

            // Build every catalogue scene once, uploading all assets up front.
            let mut builts: Vec<BuiltScene> = Vec::new();
            let mut names: Vec<&'static str> = Vec::new();
            let mut cameras: Vec<Vec<NamedCamera>> = Vec::new();
            for scene in catalogue() {
                let built = {
                    let mut ctx = BuildCtx {
                        res: renderer.resources_mut(),
                        device,
                        queue,
                    };
                    (scene.build)(&mut ctx)
                };
                builts.push(built);
                names.push(scene.name);
                cameras.push(scene.cameras);
            }

            let camera = cameras[0][0].camera.clone();
            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App {
                builts,
                names,
                cameras,
                selected: 0,
                camera,
                controller: OrbitCameraController::viewport_primitives(),
                cursor: None,
                cursor_prev: None,
            }))
        }),
    )
}

struct App {
    builts: Vec<BuiltScene>,
    names: Vec<&'static str>,
    cameras: Vec<Vec<NamedCamera>>,
    selected: usize,
    camera: Camera,
    controller: OrbitCameraController,
    cursor: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("scenes").show(ctx, |ui| {
            ui.heading("Catalogue");
            ui.label(format!("{} scenes", self.names.len()));
            ui.separator();
            for (i, name) in self.names.iter().enumerate() {
                ui.selectable_value(&mut self.selected, i, *name);
            }
            ui.separator();
            ui.label("Cameras");
            for cam in &self.cameras[self.selected] {
                if ui.button(cam.name).clicked() {
                    self.camera = cam.camera.clone();
                }
            }
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

                self.controller.begin_frame(ViewportContext {
                    hovered: response.hovered(),
                    focused: response.has_focus(),
                    viewport_size: [rect.width(), rect.height()],
                });

                ui.input(|i| {
                    self.controller
                        .push_event(ViewportEvent::ModifiersChanged(Modifiers {
                            alt: i.modifiers.alt,
                            shift: i.modifiers.shift,
                            ctrl: i.modifiers.command,
                        }));
                    let local = i
                        .pointer
                        .interact_pos()
                        .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
                    self.cursor_prev = self.cursor;
                    self.cursor = local;
                    if let Some(pos) = local {
                        self.controller
                            .push_event(ViewportEvent::PointerMoved { position: pos });
                    }
                    for event in &i.events {
                        match event {
                            egui::Event::PointerButton {
                                button, pressed, ..
                            } => {
                                let vp = match button {
                                    egui::PointerButton::Primary => MouseButton::Left,
                                    egui::PointerButton::Secondary => MouseButton::Right,
                                    egui::PointerButton::Middle => MouseButton::Middle,
                                    _ => continue,
                                };
                                self.controller.push_event(ViewportEvent::MouseButton {
                                    button: vp,
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

                let (w, h) = (rect.width(), rect.height());
                self.controller.apply_to_camera(&mut self.camera);
                self.camera.set_aspect_ratio(w, h);

                let built = &self.builts[self.selected];
                let mut fd = FrameData::new(
                    CameraFrame::from_camera(&self.camera, [w, h])
                        .with_pixels_per_point(ui.ctx().pixels_per_point()),
                    SceneFrame::from_surface_items(built.items.clone()),
                );
                fd.effects.lighting = built.lighting.clone();
                if let Some(bg) = built.background {
                    fd.viewport.background_colour = Some(bg);
                }

                ui.painter()
                    .add(eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        viewport_callback::ViewportCallback { frame: fd },
                    ));

                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            });
    }
}
