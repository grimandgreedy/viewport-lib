//! The picking scene on each device tier, side by side.
//!
//! One tab per tier: the full tier, the compatibility tier a WebGL2 or GLES
//! device gets, and the overlay-only tier below it. Every tab holds its own renderer
//! on this machine's device, capped with `RendererConfig::with_max_tier`, and
//! builds the same scene as the showcase's picking demo: one of every pickable
//! item type. The panel says what each tier did with each of them: which item
//! types it declined at registration, which uploads it refused, which items it
//! drew and which it skipped, and how picking resolves there.
//!
//! The cap reduces what the renderer reads from the device, not the device
//! itself, so this shows what a lesser device draws without checking that the
//! pipelines would fit within its limits. The testkit's device profiles do
//! that.
//!
//! Run with: cargo run --release -p viewport-lib-examples-eframe --example tiers
//!
//! `VPL_TIER=<n>` opens on tab n (1-based) instead of the first.
//! Cmd+[ and Cmd+] (Ctrl on Windows and Linux) step through the tabs.

#[path = "../showcase/camera.rs"]
mod camera;
#[path = "../showcase/showcases/04_picking.rs"]
mod picking;
#[path = "../showcase/showcase.rs"]
#[allow(dead_code)]
mod showcase;
#[path = "../showcase/ui.rs"]
#[allow(dead_code)]
mod ui;

use crate::eframe::{egui, wgpu};
use viewport_lib as vpl;
pub use viewport_lib_examples_eframe::eframe;
use vpl::input::adapters::from_egui;
use vpl::{
    ManipulationController, Modifiers, OffscreenViewportTarget, RendererConfig, Tier,
    ViewportContext, ViewportEvent, ViewportInstance,
};

use showcase::{SetupCtx, Showcase, ShowcaseCtx};

const TIERS: [Tier; 3] = [Tier::Full, Tier::Compatibility, Tier::OverlayOnly];

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : device tiers",
        viewport_lib_examples_eframe::native_options([1360.0, 820.0]),
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let tabs = TIERS
                .iter()
                .map(|&tier| Tab::new(&rs.device, &rs.queue, rs.target_format, tier))
                .collect();
            let active = std::env::var("VPL_TIER")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| (1..=TIERS.len()).contains(&n))
                .map_or(0, |n| n - 1);
            Ok(Box::new(App {
                tabs,
                active,
                camera: camera::CameraRig::new(),
                show_controls: false,
            }))
        }),
    )
}

/// The offscreen render target and its egui texture registration.
struct Target {
    inner: OffscreenViewportTarget,
    id: egui::TextureId,
}

/// One tier's renderer, its copy of the picking scene, and what it declined.
struct Tab {
    tier: Tier,
    session: ViewportInstance,
    scene: picking::PickingShowcase,
    declined: Vec<(&'static str, vpl::ViewportError)>,
    target: Option<Target>,
}

impl Tab {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        tier: Tier,
    ) -> Self {
        let config = RendererConfig::new(OffscreenViewportTarget::render_format(target_format))
            .with_max_tier(tier);
        let mut session = ViewportInstance::with_config(device, &config)
            .with_manipulation(ManipulationController::new());
        // Below the full tier there is no GPU pick pass; picks run on the CPU
        // against the cached scene items.
        session.renderer_mut().set_cpu_pick_cache(true);
        let declined = viewport_lib_plugins::item_types::install(session.renderer_mut(), device);
        let mut scene = picking::PickingShowcase::new();
        scene.setup(&mut SetupCtx {
            session: &mut session,
            device,
            queue,
        });
        Self {
            tier,
            session,
            scene,
            declined,
            target: None,
        }
    }

    fn label(&self) -> &'static str {
        match self.tier {
            Tier::Full => "Full",
            Tier::Compatibility => "Compatibility (WebGL2, GLES)",
            _ => "Overlay-only",
        }
    }

    fn summary(&self) -> &'static str {
        match self.tier {
            Tier::Full => {
                "Everything: storage buffers, compute, shadows, IBL, GPU picking \
                 with sub-object highlights."
            }
            Tier::Compatibility => {
                "No storage buffers or compute. Meshes draw through a basic-lit \
                 path, and the item types that draw without storage buffers \
                 register; the rest are declined. Clicks and box selection \
                 pick on the CPU."
            }
            _ => {
                "Fewer than three bind groups: only overlays draw. Every scene \
                 upload is refused and nothing picks."
            }
        }
    }

    /// The report panel for this tier, read after the frame was prepared.
    fn report(&mut self, ui: &mut egui::Ui) {
        ui.heading(format!("Tier: {}", self.label()));
        ui.label(self.summary());
        ui.separator();

        ui.strong("Item types declined at registration");
        if self.declined.is_empty() {
            ui.weak("none");
        }
        for (name, e) in &self.declined {
            ui.label(format!("{name}: {e}"))
                .on_hover_text(format!("{e:?}"));
        }
        ui.separator();

        ui.strong("Uploads refused");
        if self.scene.refused.is_empty() {
            ui.weak("none");
        }
        for (what, why) in &self.scene.refused {
            ui.label(format!("{what}: {why}"));
        }
        ui.separator();

        ui.strong("This frame");
        let renderer_tier = self.session.renderer_mut().tier();
        let registered: Vec<(&'static str, usize, bool)> = {
            let names: Vec<(&'static str, usize)> = self
                .session
                .frame_data_mut()
                .scene
                .plugin_items
                .iter()
                .map(|(name, c)| (*name, c.iter().map(|c| c.len()).sum()))
                .collect();
            let renderer = self.session.renderer_mut();
            let mut v: Vec<_> = names
                .into_iter()
                .map(|(name, n)| (name, n, renderer.has_item_type_plugin(name)))
                .collect();
            v.sort_by_key(|(name, _, _)| *name);
            v
        };
        let surfaces = match &self.session.frame_data_mut().scene.surfaces {
            vpl::SurfaceSubmission::Flat(items) => items.len(),
            _ => 0,
        };
        let volume_meshes = self.session.frame_data_mut().scene.volume_meshes.len();
        let drawn = |yes: bool| if yes { "drawn" } else { "skipped" };
        ui.label(format!(
            "meshes ({surfaces}): {}",
            drawn(renderer_tier >= Tier::Compatibility)
        ));
        ui.label(format!(
            "volume meshes ({volume_meshes}): {}",
            drawn(renderer_tier >= Tier::Full)
        ));
        for (name, n, has) in registered {
            let status = if has {
                "drawn"
            } else {
                "skipped, no item type registered"
            };
            ui.label(format!("{name} ({n}): {status}"));
        }
        let stats = self.session.renderer_mut().last_frame_stats();
        ui.label(format!(
            "items_skipped_by_tier: {}",
            stats.items_skipped_by_tier
        ));
        if stats.lights_dropped > 0 {
            ui.label(format!("lights_dropped: {}", stats.lights_dropped));
        }
        ui.separator();
    }
}

struct App {
    tabs: Vec<Tab>,
    active: usize,
    camera: camera::CameraRig,
    show_controls: bool,
}

// eframe 0.35 replaced `App::update(&Context, ..)` with `App::ui(&mut Ui, ..)`.
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
fn right_panel(ui: &mut egui::Ui, id: &'static str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::SidePanel::right(id)
        .resizable(true)
        .default_width(320.0)
        .show_inside(ui, add_contents);
}

#[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
fn right_panel(ui: &mut egui::Ui, id: &'static str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Panel::right(id)
        .resizable(true)
        .default_size(320.0)
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

        // Cmd+[ and Cmd+] (Ctrl on Windows and Linux) step through the tabs.
        let n = self.tabs.len();
        ctx.input_mut(|i| {
            if i.consume_key(egui::Modifiers::COMMAND, egui::Key::OpenBracket) {
                self.active = (self.active + n - 1) % n;
            }
            if i.consume_key(egui::Modifiers::COMMAND, egui::Key::CloseBracket) {
                self.active = (self.active + 1) % n;
            }
        });

        top_panel(ui, "tiers", |ui| {
            ui.horizontal(|ui| {
                ui.strong("Tier:");
                for (i, tab) in self.tabs.iter().enumerate() {
                    if ui.selectable_label(i == self.active, tab.label()).clicked() {
                        self.active = i;
                    }
                }
                ui.weak("Cmd+[ / Cmd+] to switch");
            });
        });

        let tab = &mut self.tabs[self.active];
        right_panel(ui, "tier_report", |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                tab.report(ui);
                tab.scene.panel(ui);
            });
        });

        central_panel(ui, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let ppp = ui.ctx().pixels_per_point();
            let size = [
                (rect.width() * ppp).round().max(1.0) as u32,
                (rect.height() * ppp).round().max(1.0) as u32,
            ];
            if tab.target.as_ref().map_or(true, |t| t.inner.size() != size) {
                if let Some(old) = tab.target.take() {
                    rs.renderer.write().free_texture(&old.id);
                }
                let inner = OffscreenViewportTarget::new(&rs.device, rs.target_format, size);
                let id = rs.renderer.write().register_native_texture(
                    &rs.device,
                    inner.sample_view(),
                    wgpu::FilterMode::Linear,
                );
                tab.target = Some(Target { inner, id });
            }

            tab.session.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.hovered(),
                viewport_size: [rect.width(), rect.height()],
            });
            tab.session.set_pixels_per_point(ppp);

            let origin = glam::Vec2::new(rect.left(), rect.top());
            let mut keys_pressed = Vec::new();
            let keys_down: Vec<egui::Key> = ui.input(|i| i.keys_down.iter().copied().collect());
            ui.input(|i| {
                tab.session
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
                        tab.session.handle_event(ev);
                    }
                }
            });

            {
                let mut sctx = ShowcaseCtx::new(
                    &mut tab.session,
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
                tab.scene.update(&mut sctx);
            }

            let target = tab.target.as_ref().unwrap();
            let cmd = tab
                .session
                .render(&rs.device, &rs.queue, target.inner.render_view());
            rs.queue.submit(std::iter::once(cmd));
            ui.painter().image(
                target.id,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            ui::info_box(
                ui.ctx(),
                rect.left_top() + egui::vec2(12.0, 12.0),
                tab.label(),
                tab.summary(),
            );
            egui::Area::new(egui::Id::new("camera_toggle"))
                .fixed_pos(rect.right_top() + egui::vec2(-12.0, 12.0))
                .pivot(egui::Align2::RIGHT_TOP)
                .show(ui.ctx(), |ui| {
                    self.camera.overlay(ui);
                });
            egui::Area::new(egui::Id::new("tiers_help_btn"))
                .fixed_pos(rect.right_bottom() + egui::vec2(-12.0, -12.0))
                .pivot(egui::Align2::RIGHT_BOTTOM)
                .show(ui.ctx(), |ui| {
                    if ui::help_button(ui) {
                        self.show_controls = true;
                    }
                });
            ui::controls_modal(ui.ctx(), &mut self.show_controls, "Device tiers", |ui| {
                self.camera.controls(ui);
            });
        });

        ctx.request_repaint();
    }
}
