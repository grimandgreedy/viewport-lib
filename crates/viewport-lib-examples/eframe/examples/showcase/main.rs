//! Modular viewport-lib showcase, built on `ViewportInstance` + eframe.
//!
//! The host here is deliberately small: it owns the window, one shared
//! `ViewportInstance`, and an offscreen texture egui displays as an image. Each
//! showcase (see `showcases/`) is self-contained in its own file and owns its
//! scene, camera controllers, and interaction. Pick a showcase from the side
//! panel; the host resets the scene and calls the new one's `setup`.
//!
//! Run with: cargo run --release -p viewport-lib-examples-eframe --example showcase
//!
//! It builds on every wgpu leg. On wgpu 30 the path tracer and the lightmap
//! baker trace in hardware where the GPU has ray queries, Apple silicon
//! included:
//!
//! cargo run --release -p viewport-lib-examples-eframe --no-default-features \
//!     --features wgpu30,showcase --example showcase
//!
//! `VPL_SHOWCASE=<n>` opens on showcase n (1-based) instead of the first.

mod camera;
mod showcase;
mod showcases;
mod ui;

use crate::eframe::{egui, wgpu};
use viewport_lib as vpl;
pub use viewport_lib_examples_eframe::eframe;
use vpl::input::adapters::from_egui;
use vpl::{
    ManipulationController, Modifiers, OffscreenViewportTarget, ViewportContext, ViewportEvent,
    ViewportInstance,
};

use showcase::{SetupCtx, Showcase, ShowcaseCtx};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : showcase",
        native_options(),
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            // One session, shared across showcases; manipulation is always
            // attached (idle when nothing is selected).
            // sRGB render format so the tonemap encode happens; the offscreen
            // target hands egui a non-sRGB view so the encode survives its sample.
            let mut session = ViewportInstance::new(
                &rs.device,
                OffscreenViewportTarget::render_format(rs.target_format),
            )
            .with_manipulation(ManipulationController::new());
            viewport_lib_plugins::item_types::install(session.renderer_mut(), &rs.device);

            let mut list = showcases::all();
            // `VPL_SHOWCASE=<n>` opens on showcase n (1-based) instead of the first.
            let active = std::env::var("VPL_SHOWCASE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| (1..=list.len()).contains(&n))
                .map_or(0, |n| n - 1);
            let mut setup = SetupCtx {
                session: &mut session,
                device: &rs.device,
                queue: &rs.queue,
            };
            list[active].setup(&mut setup);

            Ok(Box::new(App {
                session,
                camera: camera::CameraRig::new(),
                list,
                active,
                target: None,
                show_controls: false,
            }))
        }),
    )
}

/// The examples' device setup, plus ray queries where the adapter offers them,
/// so the path tracer and the lightmap baker trace in hardware.
fn native_options() -> eframe::NativeOptions {
    let mut options = viewport_lib_examples_eframe::native_options([1280.0, 800.0]);
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        let base = setup.device_descriptor.clone();
        setup.device_descriptor = std::sync::Arc::new(move |adapter| {
            let mut desc = base(adapter);
            desc.label = Some("viewport-lib showcase device");
            vpl::raytrace::request_ray_queries(adapter, &mut desc);
            desc
        });
    }
    options
}

/// The offscreen render target and its egui texture registration.
struct Target {
    inner: OffscreenViewportTarget,
    id: egui::TextureId,
}

struct App {
    session: ViewportInstance,
    camera: camera::CameraRig,
    list: Vec<Box<dyn Showcase>>,
    active: usize,
    target: Option<Target>,
    show_controls: bool,
}

// eframe 0.35 replaced `App::update(&Context, ..)` with `App::ui(&mut Ui, ..)`,
// handing the app a margin-free root Ui instead of the Context. On the 0.33 leg a
// frameless central panel makes the same Ui, so one body serves every leg.
#[cfg(feature = "wgpu27")]
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| self.frame_ui(ui, frame));
    }
}

#[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.frame_ui(ui, frame);
    }
}

// egui 0.35 folded `TopBottomPanel` and `SidePanel` into one `Panel`.
#[cfg(feature = "wgpu27")]
fn top_panel(ui: &mut egui::Ui, id: &'static str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::TopBottomPanel::top(id).show_inside(ui, add_contents);
}

#[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
fn top_panel(ui: &mut egui::Ui, id: &'static str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Panel::top(id).show(ui, add_contents);
}

#[cfg(feature = "wgpu27")]
fn right_panel(
    ui: &mut egui::Ui,
    id: &'static str,
    width: f32,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    egui::SidePanel::right(id)
        .resizable(false)
        .default_width(width)
        .show_inside(ui, add_contents);
}

#[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
fn right_panel(
    ui: &mut egui::Ui,
    id: &'static str,
    width: f32,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    egui::Panel::right(id)
        .resizable(false)
        .default_size(width)
        .show(ui, add_contents);
}

#[cfg(feature = "wgpu27")]
fn central_panel(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show_inside(ui, add_contents);
}

#[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
fn central_panel(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ui, add_contents);
}

impl App {
    fn frame_ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = &ui.ctx().clone();
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
        top_panel(ui, "showcases", |ui| {
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
                queue: &rs.queue,
            };
            self.list[self.active].setup(&mut setup);
        }

        // Right-side controls panel for showcases that have live controls.
        if self.list[self.active].has_controls() {
            right_panel(ui, "showcase_controls", 240.0, |ui| {
                ui.add_space(4.0);
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.list[self.active].panel(ui);
                });
            });
        }

        central_panel(ui, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let ppp = ui.ctx().pixels_per_point();
            let size = [
                (rect.width() * ppp).round().max(1.0) as u32,
                (rect.height() * ppp).round().max(1.0) as u32,
            ];

            if self
                .target
                .as_ref()
                .map_or(true, |t| t.inner.size() != size)
            {
                let inner = OffscreenViewportTarget::new(&rs.device, rs.target_format, size);
                let id = rs.renderer.write().register_native_texture(
                    &rs.device,
                    inner.sample_view(),
                    wgpu::FilterMode::Linear,
                );
                self.target = Some(Target { inner, id });
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
            let keys_down: Vec<egui::Key> = ui.input(|i| i.keys_down.iter().copied().collect());
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
                    &mut self.camera,
                    &rs.device,
                    &rs.queue,
                    ppp,
                    dt,
                    response.hovered(),
                    response.hovered(),
                    [rect.width(), rect.height()],
                    &keys_pressed,
                    &keys_down,
                );
                self.list[self.active].update(&mut sctx);
            }

            let cmd = self
                .session
                .render(&rs.device, &rs.queue, target.inner.render_view());
            rs.queue.submit(std::iter::once(cmd));
            ui.painter().image(
                target.id,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Info box over the top-left: what this showcase is.
            let title = self.list[self.active].name().to_string();
            let description = self.list[self.active].description().to_string();
            ui::info_box(
                ui.ctx(),
                rect.left_top() + egui::vec2(12.0, 12.0),
                &title,
                &description,
            );

            // Showcase-owned controls over the top-centre (e.g. a mode chip).
            egui::Area::new(egui::Id::new("showcase_top_overlay"))
                .fixed_pos(rect.center_top() + egui::vec2(0.0, 12.0))
                .pivot(egui::Align2::CENTER_TOP)
                .show(ui.ctx(), |ui| {
                    self.list[self.active].top_overlay(ui);
                });

            // Shared orbit/fly toggle over the top-right.
            egui::Area::new(egui::Id::new("camera_toggle"))
                .fixed_pos(rect.right_top() + egui::vec2(-12.0, 12.0))
                .pivot(egui::Align2::RIGHT_TOP)
                .show(ui.ctx(), |ui| {
                    self.camera.overlay(ui);
                });

            // `?` button over the bottom-right opens the controls modal.
            egui::Area::new(egui::Id::new("showcase_help_btn"))
                .fixed_pos(rect.right_bottom() + egui::vec2(-12.0, -12.0))
                .pivot(egui::Align2::RIGHT_BOTTOM)
                .show(ui.ctx(), |ui| {
                    if ui::help_button(ui) {
                        self.show_controls = true;
                    }
                });
            // General camera controls first, then this showcase's own.
            ui::controls_modal(ui.ctx(), &mut self.show_controls, &title, |ui| {
                self.camera.controls(ui);
                ui.separator();
                self.list[self.active].controls(ui);
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
