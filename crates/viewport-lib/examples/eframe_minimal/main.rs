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

use eframe::{egui, wgpu};
use viewport_lib as vpl;
use vpl::input::adapters::from_egui;
use vpl::{
    Material, Modifiers, NodeId, OrbitCameraController, ViewportContext, ViewportEvent,
    ViewportInstance, primitives,
};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : minimal (egui)",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let mut session = ViewportInstance::new(&rs.device, rs.target_format);

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
                Material::from_colour([0.9, 0.5, 0.2]),
            );
            let cube_id = scene.add(
                Some(cube),
                glam::Mat4::IDENTITY,
                Material::from_colour([0.4, 0.6, 0.9]),
            );
            scene.add(
                Some(torus),
                glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, 0.0)),
                Material::from_colour([0.3, 0.8, 0.4]),
            );
            session.camera_mut().distance = 10.0;

            Ok(Box::new(App {
                session,
                orbit: OrbitCameraController::viewport_all(),
                cube_id,
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
    session: ViewportInstance,
    orbit: OrbitCameraController,
    cube_id: NodeId,
    target: Option<Target>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let rs = frame.wgpu_render_state().expect("wgpu backend required");
        let time = ctx.input(|i| i.time) as f32;

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
                // Render target in physical pixels so it stays sharp on HiDPI.
                let ppp = ui.ctx().pixels_per_point();
                let size = [
                    (rect.width() * ppp).round().max(1.0) as u32,
                    (rect.height() * ppp).round().max(1.0) as u32,
                ];

                // (Re)create the offscreen target and its egui texture id when the
                // viewport size changes.
                if self.target.as_ref().map_or(true, |t| t.size != size) {
                    let texture = rs.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("viewport_offscreen"),
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
                let cmd = self.session.render(&rs.device, &rs.queue, &target.view);
                rs.queue.submit(std::iter::once(cmd));
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
