//! Built-in vs external post effects, same scene, live toggles.
//!
//! Contact shadows can run Off, through the built-in implementation, or
//! through this crate's external producer copy; because the copy is
//! pixel-identical, flipping Built-in <-> External should produce no
//! visible change (that is the point). Bloom is the built-in (its copy was
//! retired after validation). The vfx stage stack (colour grade, depth
//! fog, edge detect) layers on top through the stage chain.
//!
//! Run with `cargo run --release -p viewport-lib-post-effects --example toggle`.
//! The frame-time readout makes this the A/B vehicle for pricing the
//! external dispatch against the built-ins on a target GPU.

use eframe::{egui, wgpu};
use viewport_lib as vpl;
use viewport_lib_post_effects::{
    ContactShadowEffect, ContactShadowEffectSettings, SettingsHandle, VfxSettings, vfx_stack,
};
use vpl::input::adapters::from_egui;
use vpl::{
    LightKind, LightSource, Material, Modifiers, OffscreenViewportTarget, OrbitCameraController,
    ViewportContext, ViewportEvent, ViewportInstance, primitives,
};

/// Slanted key light: also fed to the external contact-shadow effect, which
/// cannot read the frame's lights through the post-effect context.
const LIGHT_DIRECTION: [f32; 3] = [0.5, 0.3, 0.8];

#[derive(Clone, Copy, PartialEq)]
enum Impl {
    Off,
    Builtin,
    External,
}

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : post-effect toggle",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let mut session = ViewportInstance::new(
                &rs.device,
                OffscreenViewportTarget::render_format(rs.target_format),
            );

            let sphere = session
                .resources_mut()
                .upload_mesh_data(&rs.device, &primitives::sphere(0.6, 32, 16))
                .unwrap();
            let cube = session
                .resources_mut()
                .upload_mesh_data(&rs.device, &primitives::cube(1.0))
                .unwrap();

            let scene = session.scene_mut();
            // Ground slab for the contact shadows to land on.
            scene.add(
                Some(cube),
                glam::Mat4::from_scale(glam::Vec3::new(10.0, 10.0, 0.1))
                    * glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.5)),
                Material::from_colour([0.65, 0.65, 0.65]),
            );
            // Emissive sphere for the bloom to pick up.
            let mut glow = Material::from_colour([0.05, 0.05, 0.05]);
            glow.emissive = [5.0, 3.5, 1.5].into();
            scene.add(
                Some(sphere),
                glam::Mat4::from_translation(glam::Vec3::new(-1.2, 0.0, 0.7)),
                glow,
            );
            // Plain cube sitting near the ground for the contact shadow.
            scene.add(
                Some(cube),
                glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.45))
                    * glam::Mat4::from_scale(glam::Vec3::splat(0.8)),
                Material::from_colour([0.25, 0.4, 0.65]),
            );
            session.camera_mut().distance = 8.0;

            let mut light = LightSource::default();
            light.kind = LightKind::Directional {
                direction: LIGHT_DIRECTION,
            };
            session.effects_mut().lighting.lights = vec![light];

            // The default bloom parameters (threshold 1.0, intensity 0.1)
            // are too subtle to read on a small emissive sphere; use the
            // demo values. The per-frame sync below mirrors these into the
            // external copy, so both implementations run identically.
            let pp = &mut session.effects_mut().post_process;
            pp.bloom.threshold = 0.7;
            pp.bloom.intensity = 2.0;

            // Register the external contact-shadow copy (self-gated off
            // until toggled) and the vfx stage stack.
            let (cs, cs_handle) = ContactShadowEffect::new(ContactShadowEffectSettings {
                enabled: false,
                light_direction: LIGHT_DIRECTION,
                ..Default::default()
            });
            session
                .renderer_mut()
                .add_post_effect_producer(Box::new(cs));
            let (stages, vfx_handle) = vfx_stack();
            for (stage, order) in stages {
                session.renderer_mut().add_post_effect_stage(stage, order);
            }
            {
                let mut s = vfx_handle.lock().unwrap();
                s.colour_grade.enabled = false;
                s.depth_fog.enabled = false;
                s.edge_detect.enabled = false;
            }

            Ok(Box::new(App {
                session,
                orbit: OrbitCameraController::viewport_all(),
                target: None,
                bloom: true,
                cs_impl: Impl::Builtin,
                cs_handle,
                vfx_handle,
                frame_ms: 0.0,
            }))
        }),
    )
}

struct Target {
    inner: OffscreenViewportTarget,
    id: egui::TextureId,
}

struct App {
    session: ViewportInstance,
    orbit: OrbitCameraController,
    target: Option<Target>,
    bloom: bool,
    cs_impl: Impl,
    cs_handle: SettingsHandle<ContactShadowEffectSettings>,
    vfx_handle: SettingsHandle<VfxSettings>,
    frame_ms: f32,
}

fn impl_picker(ui: &mut egui::Ui, label: &str, value: &mut Impl) {
    ui.label(label);
    ui.horizontal(|ui| {
        ui.selectable_value(value, Impl::Off, "Off");
        ui.selectable_value(value, Impl::Builtin, "Built-in");
        ui.selectable_value(value, Impl::External, "External");
    });
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let rs = frame.wgpu_render_state().expect("wgpu backend required");
        let dt = ctx.input(|i| i.stable_dt);
        self.frame_ms = self.frame_ms * 0.95 + dt * 1000.0 * 0.05;

        egui::SidePanel::right("controls").show(ctx, |ui| {
            ui.heading("Post effects");
            ui.checkbox(&mut self.bloom, "Bloom (built-in)");
            impl_picker(ui, "Contact shadows", &mut self.cs_impl);
            ui.separator();
            ui.heading("Stage stack");
            {
                let mut vfx = self.vfx_handle.lock().unwrap();
                ui.checkbox(&mut vfx.colour_grade.enabled, "Colour grade");
                ui.checkbox(&mut vfx.depth_fog.enabled, "Depth fog");
                ui.checkbox(&mut vfx.edge_detect.enabled, "Edge detect");
            }
            ui.separator();
            ui.label(format!("frame: {:.2} ms", self.frame_ms));
        });

        // Route contact shadows to their implementation. The built-in
        // switches live on PostProcessSettings; the external copy is
        // self-gated through its handle, mirroring the built-in parameters.
        {
            let pp = &mut self.session.effects_mut().post_process;
            pp.bloom.enabled = self.bloom;
            pp.contact_shadows.enabled = self.cs_impl == Impl::Builtin;
            let mut cs = self.cs_handle.lock().unwrap();
            cs.enabled = self.cs_impl == Impl::External;
            cs.max_distance = pp.contact_shadows.max_distance;
            cs.steps = pp.contact_shadows.steps;
            cs.thickness = pp.contact_shadows.thickness;
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
                self.session.update_orbit(&mut self.orbit);

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
            });

        ctx.request_repaint();
    }
}
