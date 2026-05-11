//! Remote rendering example using eframe / egui.
//!
//! Demonstrates a client-server split where the viewport renderer runs on a
//! separate "server" thread and streams rendered frames back to a thin client
//! that only handles input and display.
//!
//! Architecture:
//!
//!   Client (this eframe app)          Server (server::run_server)
//!   ----------------------------      ----------------------------
//!   capture input                     headless wgpu device
//!   update Camera                     ViewportRenderer
//!   send RenderRequest  ---------->   render_offscreen()
//!   display frame       <----------   send RenderFrame
//!
//! The channels are a stand-in for a real network transport. In a production
//! setup the two sides would run as separate processes (or on separate
//! machines). The message types use Camera directly here because it is
//! in-process; over a real network you would serialize them -- Camera
//! implements Serialize/Deserialize via the serde feature.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom

mod server;

use std::sync::mpsc::{self, Receiver, Sender};

use eframe::egui;
use server::{RenderFrame, RenderRequest};
use viewport_lib::{
    ButtonState, Camera, OrbitCameraController, ScrollUnits, ViewportContext, ViewportEvent,
};

fn main() -> eframe::Result {
    let (req_tx, req_rx) = mpsc::channel::<RenderRequest>();
    let (frame_tx, frame_rx) = mpsc::channel::<RenderFrame>();

    std::thread::spawn(move || server::run_server(req_rx, frame_tx));

    eframe::run_native(
        "viewport-lib : Remote Rendering",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(App::new(req_tx, frame_rx)))),
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    tx: Sender<RenderRequest>,
    rx: Receiver<RenderFrame>,
    current_frame: Option<RenderFrame>,
    current_texture: Option<egui::TextureHandle>,
}

impl App {
    fn new(tx: Sender<RenderRequest>, rx: Receiver<RenderFrame>) -> Self {
        Self {
            camera: Camera {
                distance: 10.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            tx,
            rx,
            current_frame: None,
            current_texture: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            let w = rect.width();
            let h = rect.height();

            // Feed input into the orbit controller.
            self.controller.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [w, h],
            });

            ui.input(|i| {
                self.controller
                    .push_event(ViewportEvent::ModifiersChanged(viewport_lib::Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    }));

                let local_pos = i
                    .pointer
                    .interact_pos()
                    .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
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
                                egui::PointerButton::Secondary => {
                                    viewport_lib::MouseButton::Right
                                }
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

            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(w, h);

            // Send the latest camera state to the server.
            let _ = self.tx.send(RenderRequest {
                camera: self.camera.clone(),
                width: w as u32,
                height: h as u32,
            });

            // Receive the latest rendered frame (discard all but the newest).
            while let Ok(frame) = self.rx.try_recv() {
                self.current_frame = Some(frame);
            }

            // Upload new frame to an egui texture when one arrives.
            if let Some(frame) = self.current_frame.take() {
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [frame.width as usize, frame.height as usize],
                    &frame.pixels,
                );
                self.current_texture = Some(ctx.load_texture(
                    "remote-frame",
                    color_image,
                    egui::TextureOptions::LINEAR,
                ));
            }

            // Draw the texture to fill the panel.
            if let Some(ref tex) = self.current_texture {
                ui.painter().image(
                    tex.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            } else {
                // While the first frame is in flight, show a placeholder.
                ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(40));
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Waiting for server...",
                    egui::FontId::proportional(18.0),
                    egui::Color32::GRAY,
                );
            }

            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });

        // Drive continuous repaints so new frames are displayed promptly.
        ctx.request_repaint();
    }
}
