//! Modular viewport-lib showcase, built on `ViewportSession` + eframe.
//!
//! The host here is deliberately small: it owns the window, one shared
//! `ViewportSession`, and an offscreen texture egui displays as an image. Each
//! showcase (see `showcases/`) is self-contained in its own file and owns its
//! scene, camera controllers, and interaction. Pick a showcase from the side
//! panel; the host resets the scene and calls the new one's `setup`.
//!
//! Run with: cargo run --example showcase --features "example-egui,egui-adapter"

mod showcase;
mod showcases;

use eframe::{egui, wgpu};
use viewport_lib::input::adapters::from_egui;
use viewport_lib::{
    ManipulationController, Modifiers, ViewportContext, ViewportEvent, ViewportSession,
};

use showcase::{SetupCtx, Showcase, ShowcaseCtx};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : showcase",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            // One session, shared across showcases; manipulation is always
            // attached (idle when nothing is selected).
            let mut session = ViewportSession::new(&rs.device, rs.target_format)
                .with_manipulation(ManipulationController::new());

            let mut list = showcases::all();
            let mut setup = SetupCtx {
                session: &mut session,
                device: &rs.device,
            };
            list[0].setup(&mut setup);

            Ok(Box::new(App {
                session,
                list,
                active: 0,
                target: None,
            }))
        }),
    )
}

/// The offscreen render target and its egui texture registration.
struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    id: egui::TextureId,
    size: [u32; 2],
}

struct App {
    session: ViewportSession,
    list: Vec<Box<dyn Showcase>>,
    active: usize,
    target: Option<Target>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let rs = frame.wgpu_render_state().expect("wgpu backend required");
        let dt = ctx.input(|i| i.stable_dt).min(0.1);

        let count = self.list.len();
        let mut switch_to = None;

        // Ctrl/Cmd + [ / ] cycles through the showcases (wrapping).
        ctx.input(|i| {
            let cycle = (i.modifiers.command || i.modifiers.ctrl) && !i.modifiers.alt;
            if cycle && i.key_pressed(egui::Key::CloseBracket) {
                switch_to = Some((self.active + 1) % count);
            }
            if cycle && i.key_pressed(egui::Key::OpenBracket) {
                switch_to = Some((self.active + count - 1) % count);
            }
        });

        // Top bar: numbered selector laid out horizontally across the top.
        egui::TopBottomPanel::top("showcases").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("Showcases:");
                for (i, sc) in self.list.iter().enumerate() {
                    let label = format!("{}. {}", i + 1, sc.name());
                    if ui.selectable_label(i == self.active, label).clicked() {
                        switch_to = Some(i);
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak("Ctrl/Cmd + [ / ] to cycle");
                });
            });
        });

        if let Some(i) = switch_to.filter(|&i| i != self.active) {
            showcase::reset_session(&mut self.session);
            self.active = i;
            let mut setup = SetupCtx {
                session: &mut self.session,
                device: &rs.device,
            };
            self.list[self.active].setup(&mut setup);
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
                let ppp = ui.ctx().pixels_per_point();
                let size = [
                    (rect.width() * ppp).round().max(1.0) as u32,
                    (rect.height() * ppp).round().max(1.0) as u32,
                ];

                if self.target.as_ref().map_or(true, |t| t.size != size) {
                    let texture = rs.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("showcase_offscreen"),
                        size: wgpu::Extent3d {
                            width: size[0],
                            height: size[1],
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: rs.target_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let id = rs.renderer.write().register_native_texture(
                        &rs.device,
                        &view,
                        wgpu::FilterMode::Linear,
                    );
                    self.target = Some(Target {
                        _texture: texture,
                        view,
                        id,
                        size,
                    });
                }
                let target = self.target.as_ref().unwrap();

                // Logical viewport size + pixels_per_point; cursor stays logical.
                // A viewport-local rect never takes egui keyboard focus, so treat
                // it as focused while hovered, or the input resolver drops key
                // events (G/R/S, WASD).
                self.session.begin_frame(ViewportContext {
                    hovered: response.hovered(),
                    focused: response.hovered(),
                    viewport_size: [rect.width(), rect.height()],
                });
                self.session.set_pixels_per_point(ppp);

                let origin = glam::Vec2::new(rect.left(), rect.top());
                let mut keys_pressed = Vec::new();
                let keys_down: Vec<egui::Key> =
                    ui.input(|i| i.keys_down.iter().copied().collect());
                ui.input(|i| {
                    self.session
                        .handle_event(ViewportEvent::ModifiersChanged(Modifiers {
                            alt: i.modifiers.alt,
                            shift: i.modifiers.shift,
                            ctrl: i.modifiers.command,
                        }));
                    for event in &i.events {
                        if let egui::Event::Key {
                            key,
                            pressed: true,
                            repeat: false,
                            ..
                        } = event
                        {
                            keys_pressed.push(*key);
                        }
                        if let Some(ev) = from_egui(event, origin) {
                            self.session.handle_event(ev);
                        }
                    }
                });

                {
                    let mut sctx = ShowcaseCtx::new(
                        &mut self.session,
                        dt,
                        response.hovered(),
                        response.hovered(),
                        [rect.width(), rect.height()],
                        &keys_pressed,
                        &keys_down,
                    );
                    self.list[self.active].update(&mut sctx);
                }

                let cmd = self.session.render(&rs.device, &rs.queue, &target.view);
                rs.queue.submit(std::iter::once(cmd));
                ui.painter().image(
                    target.id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                // Help text over the top-left corner of the viewport.
                egui::Area::new(egui::Id::new("showcase_help"))
                    .fixed_pos(rect.left_top() + egui::vec2(12.0, 12.0))
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            self.list[self.active].ui(ui);
                        });
                    });

                // Showcase-owned controls (e.g. camera toggle) over the top-right.
                egui::Area::new(egui::Id::new("showcase_overlay"))
                    .fixed_pos(rect.right_top() + egui::vec2(-12.0, 12.0))
                    .pivot(egui::Align2::RIGHT_TOP)
                    .show(ui.ctx(), |ui| {
                        self.list[self.active].viewport_overlay(ui);
                    });

                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            });

        ctx.request_repaint();
    }
}
