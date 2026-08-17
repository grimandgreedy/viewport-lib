//! Exposure & auto-exposure test example (eframe / egui).
//!
//! A focused, standalone counterpart to showcase 59 in the big monolith: one
//! lit scene (ground + a row of spheres running dark -> bright albedo, with cast
//! shadows) viewed under the three `ExposureMode`s.
//!
//! - Manual EV        : a fixed EV100.
//! - Physical camera  : aperture / shutter / ISO drive EV100.
//! - Automatic        : the HDR target is metered each frame and the exposure
//!                      adapts to hold a steady mid-grey.
//!
//! The scene-brightness slider is the thing to play with: under Manual /
//! Physical it changes how bright the image is; under Automatic the image stays
//! put while the metered exposure moves.
//!
//! Structure mirrors `eframe_minimal`: a `ViewportInstance` renders into an
//! app-owned offscreen texture that egui displays as an image, and events come
//! through the standard `from_egui` adapter driving an `OrbitCameraController`
//! (left/middle drag orbit, right drag pan, scroll zoom). The offscreen texture
//! is the sRGB variant of egui's format so the tone map's linear output gets the
//! hardware linear->sRGB encode on write; egui runs on a non-sRGB surface, and
//! painting straight into its pass would skip that encode and read ~2.4x too
//! dark. See docs/issues/eframe-offscreen-target-skips-srgb-encode.
//!
//! The viewport is repainted every frame (`ctx.request_repaint()`) so exposure
//! changes take effect live and auto-exposure smoothing runs - viewport-lib only
//! re-renders on demand, and exposure lives outside the scene, so without a
//! repaint request a control change would otherwise be dropped.

use eframe::{egui, wgpu};
use viewport_lib::input::adapters::from_egui;
use viewport_lib::{
    AutoExposure, ExposureMode, ExposureSettings, LightKind, LightSource, LightingSettings,
    Material, Modifiers, OrbitCameraController, ViewportContext, ViewportEvent, ViewportInstance,
    primitives,
};

const COLUMNS: usize = 6;
const COL_SPACING: f32 = 2.4;

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Exposure & Auto-Exposure",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            // eframe creates its device with default limits (8 storage buffers
            // per stage), but the auto-exposure compute path needs the higher
            // limits the renderer recommends. Request them, or bring-up panics.
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                    eframe::egui_wgpu::WgpuSetupCreateNew {
                        device_descriptor: std::sync::Arc::new(|adapter| {
                            let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                                wgpu::Limits::downlevel_webgl2_defaults()
                            } else {
                                viewport_lib::ViewportRenderer::recommended_device_limits(adapter)
                            };
                            wgpu::DeviceDescriptor {
                                label: Some("viewport-lib exposure device"),
                                required_features:
                                    viewport_lib::ViewportRenderer::recommended_device_features(
                                        adapter,
                                    ),
                                required_limits: wgpu::Limits {
                                    max_texture_dimension_2d: 8192,
                                    ..base_limits
                                },
                                ..Default::default()
                            }
                        }),
                        ..Default::default()
                    },
                ),
                ..Default::default()
            },
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            // The session draws into an app-owned offscreen texture built as the
            // sRGB variant of egui's format (see the module docs), so build the
            // session for that same format.
            let format = rs.target_format.add_srgb_suffix();
            let mut session = ViewportInstance::new(&rs.device, format);

            let span = (COLUMNS - 1) as f32 * COL_SPACING;
            let ground_mesh = session
                .resources_mut()
                .upload_mesh_data(&rs.device, &primitives::cuboid(span + 8.0, 12.0, 0.4))
                .expect("ground");
            let sphere_mesh = session
                .resources_mut()
                .upload_mesh_data(&rs.device, &primitives::sphere(0.8, 48, 24))
                .expect("sphere");

            // Static scene: a ground slab and a row of spheres from dark to bright
            // albedo. Geometry never changes, so add it once here.
            let scene = session.scene_mut();
            scene.add(
                Some(ground_mesh),
                glam::Mat4::from_translation(glam::Vec3::new(span * 0.5, 0.0, -0.2)),
                {
                    let mut m = Material::from_colour([0.45, 0.45, 0.48]);
                    m.roughness = 0.95;
                    m
                },
            );
            for c in 0..COLUMNS {
                let x = c as f32 * COL_SPACING;
                let a = 0.05 + (c as f32 / (COLUMNS - 1) as f32) * 0.85;
                let mut m = Material::from_colour([a, a, a]);
                m.roughness = 0.6;
                m.metallic = 0.0;
                scene.add(
                    Some(sphere_mesh),
                    glam::Mat4::from_translation(glam::Vec3::new(x, 0.8, 0.8)),
                    m,
                );
            }

            // Background is a persistent viewport setting.
            session.viewport_frame_mut().background_colour = Some([0.12, 0.12, 0.14, 1.0]);

            let cam = session.camera_mut();
            cam.center = glam::Vec3::new((COLUMNS as f32 - 1.0) * COL_SPACING * 0.5, 0.4, 0.7);
            cam.distance = 17.0;
            cam.orientation =
                glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.2);

            Ok(Box::new(App {
                session,
                orbit: OrbitCameraController::viewport_all(),
                target: None,
                light_intensity: 4.0,
                mode: ModeSel::Automatic,
                compensation: 0.0,
                manual_ev: 0.0,
                aperture: 1.4,
                shutter_denom: 30.0,
                iso: 3200.0,
                auto: AutoExposure::default(),
                smooth: true,
            }))
        }),
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModeSel {
    Manual,
    Physical,
    Automatic,
}

/// The app-owned offscreen render target and its egui texture registration.
struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    id: egui::TextureId,
    size: [u32; 2],
}

struct App {
    session: ViewportInstance,
    orbit: OrbitCameraController,
    target: Option<Target>,

    // Scene
    light_intensity: f32,

    // Exposure
    mode: ModeSel,
    compensation: f32,
    manual_ev: f32,
    aperture: f32,
    shutter_denom: f32,
    iso: f32,
    auto: AutoExposure,
    smooth: bool,
}

impl App {
    fn build_lighting(&self) -> LightingSettings {
        let mut sun = LightSource::default();
        sun.cast_shadows = true;
        sun.kind = LightKind::Directional {
            direction: [0.4, 0.5, 0.9],
        };
        sun.intensity = self.light_intensity;

        let mut l = LightingSettings::default();
        l.lights = vec![sun];
        l.shadows_enabled = true;
        // A softer sky fill so shadows aren't near-black: less extreme metering
        // swings when the camera points into shadow (closer to real ambient/GI).
        l.hemisphere_intensity = 0.15;
        l.sky_colour = [0.6, 0.7, 0.9];
        l.ground_colour = [0.25, 0.22, 0.2];
        l
    }

    fn build_exposure(&self, dt: f32) -> ExposureSettings {
        let mode = match self.mode {
            ModeSel::Manual => ExposureMode::Manual { ev: self.manual_ev },
            ModeSel::Physical => ExposureMode::PhysicalCamera {
                aperture: self.aperture,
                shutter: 1.0 / self.shutter_denom.max(1.0),
                iso: self.iso.max(1.0),
            },
            ModeSel::Automatic => {
                let mut a = self.auto;
                a.dt = if self.smooth { dt } else { 0.0 };
                ExposureMode::Automatic(a)
            }
        };
        ExposureSettings::from_mode(mode).with_compensation(self.compensation)
    }

    fn readout_ev(&self) -> Option<f32> {
        match self.mode {
            ModeSel::Manual => Some(self.manual_ev),
            ModeSel::Physical => {
                let n = self.aperture;
                let t = 1.0 / self.shutter_denom.max(1.0);
                let iso = self.iso.max(1.0);
                Some((n * n / t).log2() + (100.0 / iso).log2())
            }
            ModeSel::Automatic => None,
        }
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let rs = frame
            .wgpu_render_state()
            .expect("wgpu backend required")
            .clone();

        egui::SidePanel::left("exposure_panel")
            .min_width(280.0)
            .max_width(340.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.ui_panel(ui));
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

                // Offscreen target in physical pixels so it stays sharp on HiDPI;
                // (re)created when the panel size changes.
                let ppp = ui.ctx().pixels_per_point();
                let size = [
                    (rect.width() * ppp).round().max(1.0) as u32,
                    (rect.height() * ppp).round().max(1.0) as u32,
                ];
                if self.target.as_ref().map_or(true, |t| t.size != size) {
                    let texture = rs.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("exposure_offscreen"),
                        size: wgpu::Extent3d {
                            width: size[0],
                            height: size[1],
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: rs.target_format.add_srgb_suffix(),
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

                // Screen-space state stays in logical points; pixels_per_point
                // sizes the physical render target and keeps overlays crisp.
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

                // Per-frame exposure + lighting (both driven by live controls).
                let dt = ctx.input(|i| i.stable_dt).min(0.1);
                let lighting = self.build_lighting();
                let exposure = self.build_exposure(dt);
                let eff = self.session.effects_mut();
                eff.lighting = lighting;
                eff.exposure = exposure;

                self.session.update_orbit(&mut self.orbit);

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

        // Repaint every frame: exposure lives outside the scene, so a control
        // change would otherwise not dirty anything and be dropped; auto-exposure
        // smoothing also needs continuous frames.
        ctx.request_repaint();
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

impl App {
    fn ui_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Exposure");
        ui.separator();
        ui.label("One lit scene under three exposure modes. Drive the scene-brightness slider and watch how each mode responds.");
        ui.add_space(6.0);

        ui.add(
            egui::Slider::new(&mut self.light_intensity, 0.1..=40.0)
                .logarithmic(true)
                .text("Scene brightness (light intensity)"),
        );
        ui.add(egui::Slider::new(&mut self.compensation, -3.0..=3.0).text("Exposure compensation (stops)"));
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Mode:");
            ui.selectable_value(&mut self.mode, ModeSel::Manual, "Manual EV");
            ui.selectable_value(&mut self.mode, ModeSel::Physical, "Physical");
            ui.selectable_value(&mut self.mode, ModeSel::Automatic, "Automatic");
        });
        ui.separator();

        match self.mode {
            ModeSel::Manual => {
                ui.add(egui::Slider::new(&mut self.manual_ev, -6.0..=16.0).text("EV100"));
                ui.label("Fixed exposure. Higher EV = darker. Change scene brightness and the image clips or crushes.");
            }
            ModeSel::Physical => {
                ui.add(egui::Slider::new(&mut self.aperture, 1.0..=22.0).text("Aperture f/"));
                ui.add(
                    egui::Slider::new(&mut self.shutter_denom, 4.0..=4000.0)
                        .logarithmic(true)
                        .text("Shutter 1/x s"),
                );
                ui.add(
                    egui::Slider::new(&mut self.iso, 50.0..=6400.0)
                        .logarithmic(true)
                        .text("ISO"),
                );
                ui.label("EV100 = log2(N^2 / t) + log2(100 / ISO). f-stops are calibrated for photometric magnitudes (Phase 4); until units are pinned, use fast/wide/high-ISO settings - daylight settings read black.");
            }
            ModeSel::Automatic => {
                ui.checkbox(&mut self.smooth, "Smooth adaptation (dt > 0)");
                ui.add(egui::Slider::new(&mut self.auto.adaptation, 0.0..=1.0).text("Adaptation strength"));
                ui.add(egui::Slider::new(&mut self.auto.min_ev, -10.0..=6.0).text("EV min"));
                ui.add(egui::Slider::new(&mut self.auto.max_ev, 4.0..=20.0).text("EV max"));
                ui.add(egui::Slider::new(&mut self.auto.speed_up, 0.1..=10.0).text("Adapt speed (brighten)"));
                ui.add(egui::Slider::new(&mut self.auto.speed_down, 0.1..=10.0).text("Adapt speed (darken)"));
                ui.add(egui::Slider::new(&mut self.auto.low_percent, 0.0..=0.9).text("Meter low clip"));
                ui.add(egui::Slider::new(&mut self.auto.high_percent, 0.1..=1.0).text("Meter high clip"));
                ui.add(egui::Slider::new(&mut self.auto.center_weight, 0.0..=1.0).text("Center weighting"));
                ui.label("Metering holds the image steady while scene brightness moves. dt<=0 snaps; smoothing eases. Center weighting keeps exposure stable on zoom/pan (set 0 to meter the whole frame).");
            }
        }

        ui.separator();
        match self.readout_ev() {
            Some(ev) => ui.label(format!("EV100: {ev:.2}  (before {:+.1} stops compensation)", self.compensation)),
            None => ui.label("EV100: metered on the GPU each frame (adapts live)."),
        };
    }
}
