//! Minimal embedded viewport-lib example using eframe / egui and `ViewportInstance`.
//!
//! The session renders into an app-owned offscreen texture, which egui displays
//! as an image. This keeps the whole session model (renderer, scene, camera,
//! input) behind one object; the app only owns the texture and translates events
//! with `viewport_lib::input::adapters::from_egui`. The egui paint-callback route
//! is not used here because it requires the session to live in egui's
//! `Send + Sync` callback resources, which a session carrying a runtime is not.
//!
//! Viewport size and pointer coordinates stay in logical points; the session's
//! `pixels_per_point` sizes the physical render target and keeps overlays and the
//! axes indicator crisp on HiDPI. The offscreen texture is therefore allocated at
//! `points * pixels_per_point` (physical) to match the target the renderer sizes.
//!
//! Navigation: left/middle drag orbit, right drag pan, scroll zoom.

use crate::eframe::{egui, wgpu};
use viewport_lib as vpl;
use viewport_lib::Colour;
pub use viewport_lib_examples_eframe::eframe;
use vpl::input::adapters::from_egui;
use vpl::{
    Material, Modifiers, NodeId, OffscreenViewportTarget, OrbitCameraController, RendererConfig,
    ViewportContext, ViewportEvent, ViewportInstance, primitives,
};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : minimal (egui)",
        viewport_lib_examples_eframe::native_options([1280.0, 720.0]),
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            // Render into the sRGB variant of egui's surface format so the
            // renderer's linear tonemap output gets the linear->sRGB encode; the
            // offscreen target (below) carries the matching dual-view wiring.
            //
            // Seeded with the pipeline cache the last run saved, so its shader
            // compilation is not paid again. The data is only used on a device
            // with a pipeline cache, and anything stale is discarded.
            let cache = std::fs::read(pipeline_cache_path()).ok();
            let mut session = ViewportInstance::with_config(
                &rs.device,
                &RendererConfig::new(OffscreenViewportTarget::render_format(rs.target_format))
                    .with_pipeline_cache_data(cache),
            );

            let sphere = session
                .resources_mut()
                .upload_mesh_data(&rs.device, &primitives::sphere(0.6, 24, 12))
                .unwrap();
            let cube = session
                .resources_mut()
                .upload_mesh_data(&rs.device, &primitives::cube(1.0))
                .unwrap();
            let torus = session
                .resources_mut()
                .upload_mesh_data(&rs.device, &primitives::torus(0.5, 0.18, 32, 16))
                .unwrap();

            let scene = session.scene_mut();
            scene.add(
                Some(sphere),
                glam::Mat4::from_translation(glam::Vec3::new(-2.5, 0.0, 0.0)),
                Material::from_colour(Colour::linear_rgb(0.75, 0.28, 0.05)),
            );
            let cube_id = scene.add(
                Some(cube),
                glam::Mat4::IDENTITY,
                Material::from_colour(Colour::linear_rgb(0.12, 0.3, 0.7)),
            );
            scene.add(
                Some(torus),
                glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, 0.0)),
                Material::from_colour(Colour::linear_rgb(0.1, 0.55, 0.2)),
            );
            session.camera_mut().distance = 10.0;

            Ok(Box::new(App {
                session,
                orbit: OrbitCameraController::new_stateless(),
                cube_id,
                target: None,
                cache_saved: false,
            }))
        }),
    )
}

/// The offscreen render target and its egui texture registration.
struct Target {
    inner: OffscreenViewportTarget,
    id: egui::TextureId,
}

struct App {
    session: ViewportInstance,
    orbit: OrbitCameraController,
    cube_id: NodeId,
    target: Option<Target>,
    /// Whether this run has written its pipeline cache back yet.
    cache_saved: bool,
}

/// Where the example keeps its pipeline cache. An application would use its
/// per-user cache directory.
fn pipeline_cache_path() -> std::path::PathBuf {
    std::env::temp_dir().join("viewport-lib-eframe-minimal.pipeline_cache")
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

impl App {
    fn frame_ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let rs = frame.wgpu_render_state().expect("wgpu backend required");
        let time = ctx.input(|i| i.time) as f32;

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
                // Render target in physical pixels so it stays sharp on HiDPI.
                let ppp = ui.ctx().pixels_per_point();
                let size = [
                    (rect.width() * ppp).round().max(1.0) as u32,
                    (rect.height() * ppp).round().max(1.0) as u32,
                ];

                // (Re)create the offscreen target and its egui texture id when the
                // viewport size changes. `OffscreenViewportTarget` owns the sRGB
                // dual-view so the tonemap encode survives egui's sample; we render
                // into its sRGB view and register its non-sRGB view with egui.
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

                // Screen-space state (viewport size, cursor) stays in logical
                // points; pixels_per_point sizes the physical render target and
                // keeps overlays and the axes indicator crisp on HiDPI.
                self.session.begin_frame(ViewportContext {
                    hovered: response.hovered(),
                    focused: response.has_focus(),
                    viewport_size: [rect.width(), rect.height()],
                });
                self.session.set_pixels_per_point(ppp);
                let origin = glam::Vec2::new(rect.left(), rect.top());
                ui.input(|i| {
                    self.session
                        .handle_event(ViewportEvent::ModifiersChanged(Modifiers {
                            alt: i.modifiers.alt,
                            shift: i.modifiers.shift,
                            ctrl: i.modifiers.command,
                        }));
                    for event in &i.events {
                        if let Some(ev) = from_egui(event, origin) {
                            self.session.handle_event(ev);
                        }
                    }
                });

                // Z-up: spin the cube about the world up axis, before assembly.
                let spin = glam::Mat4::from_rotation_z(time);
                self.session
                    .scene_mut()
                    .set_local_transform(self.cube_id, spin);
                self.session.update_orbit(&mut self.orbit);

                // Render into the offscreen texture and display it in the panel.
                let cmd = self
                    .session
                    .render(&rs.device, &rs.queue, target.inner.render_view());
                rs.queue.submit(std::iter::once(cmd));
                // The first frame has compiled what this scene draws with, so
                // this is the point to write the pipeline cache back.
                if !self.cache_saved {
                    self.cache_saved = true;
                    if let Some(data) = self.session.pipeline_cache_data() {
                        let _ = std::fs::write(pipeline_cache_path(), data);
                    }
                }
                ui.painter().image(
                    target.id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            });

        ctx.request_repaint();
    }
}
