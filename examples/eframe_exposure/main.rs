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
//! put while the metered exposure moves. Orbit/zoom under Automatic to confirm
//! exposure stays stable (background is excluded from the meter).
//!
//! The viewport is repainted every frame (`ctx.request_repaint()`) so exposure
//! changes take effect live and auto-exposure smoothing runs - viewport-lib only
//! re-renders on demand, and exposure lives outside the scene, so without a
//! repaint request a control change would otherwise be dropped.
//!
//! Navigation: left/middle drag = orbit, right drag = pan, scroll = zoom.

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    AutoExposure, ButtonState, Camera, CameraFrame, ExposureMode, ExposureSettings, FrameData,
    LightKind, LightSource, LightingSettings, Material, MeshId, OrbitCameraController, SceneFrame,
    SceneRenderItem, ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
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
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            // eframe/egui runs on a non-sRGB surface, but the tone map outputs
            // linear light and relies on an sRGB render target for the hardware
            // linear->sRGB encode. Using the sRGB variant here makes the encode
            // happen; without it every colour reads ~2.4x too dark (crushed
            // shadows). See docs/issues/eframe-offscreen-target-skips-srgb-encode.
            let format = rs.target_format.add_srgb_suffix();
            let mut renderer = ViewportRenderer::new(&rs.device, format);

            let span = (COLUMNS - 1) as f32 * COL_SPACING;
            let (ground, sphere) = {
                let res = renderer.resources_mut();
                let ground = res
                    .upload_mesh_data(&rs.device, &primitives::cuboid(span + 8.0, 12.0, 0.4))
                    .expect("ground");
                let sphere = res
                    .upload_mesh_data(&rs.device, &primitives::sphere(0.8, 48, 24))
                    .expect("sphere");
                (ground, sphere)
            };

            rs.renderer.write().callback_resources.insert(renderer);
            Ok(Box::new(App::new(ground, sphere)))
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

struct App {
    camera: Camera,
    controller: OrbitCameraController,

    ground: MeshId,
    sphere: MeshId,

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
    fn new(ground: MeshId, sphere: MeshId) -> Self {
        Self {
            camera: Camera {
                center: glam::Vec3::new((COLUMNS as f32 - 1.0) * COL_SPACING * 0.5, 0.4, 0.7),
                distance: 17.0,
                orientation: glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.2),
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            ground,
            sphere,
            light_intensity: 4.0,
            mode: ModeSel::Automatic,
            compensation: 0.0,
            manual_ev: 0.0,
            // Low-light defaults: the scene is pre-photometric (intensities ~1,
            // not lux/candela), so daylight f-stops read black until units are
            // pinned. Wide aperture / slow shutter / high ISO expose it.
            aperture: 1.4,
            shutter_denom: 30.0,
            iso: 3200.0,
            auto: AutoExposure::default(),
            // Interactive default: ease exposure over time so orbit/zoom don't
            // snap the brightness. Snap (dt<=0) is the library default for
            // one-shot / dirty-only renders.
            smooth: true,
        }
    }

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

    fn build_items(&self) -> Vec<SceneRenderItem> {
        let mut items = Vec::new();
        let span = (COLUMNS - 1) as f32 * COL_SPACING;

        let mut ground = SceneRenderItem::default();
        ground.mesh_id = self.ground;
        ground.model =
            glam::Mat4::from_translation(glam::Vec3::new(span * 0.5, 0.0, -0.2)).to_cols_array_2d();
        ground.material = Material::from_colour([0.45, 0.45, 0.48]);
        ground.material.roughness = 0.95;
        items.push(ground);

        for c in 0..COLUMNS {
            let x = c as f32 * COL_SPACING;
            let a = 0.05 + (c as f32 / (COLUMNS - 1) as f32) * 0.85;
            let mut s = SceneRenderItem::default();
            s.mesh_id = self.sphere;
            s.model =
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.8, 0.8)).to_cols_array_2d();
            s.material = Material::from_colour([a, a, a]);
            s.material.roughness = 0.6;
            s.material.metallic = 0.0;
            items.push(s);
        }
        items
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
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("exposure_panel")
            .min_width(280.0)
            .max_width(340.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.ui_panel(ui));
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.controller.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            });
            ui.input(|i| {
                let local = i
                    .pointer
                    .interact_pos()
                    .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
                if let Some(pos) = local {
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
                                egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
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

            let (w, h) = (rect.width(), rect.height());
            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(w, h);

            let dt = ctx.input(|i| i.stable_dt).min(0.1);
            let ppp = ui.ctx().pixels_per_point();
            let mut fd = FrameData::new(
                CameraFrame::from_camera(&self.camera, [w, h]).with_pixels_per_point(ppp),
                SceneFrame::from_surface_items(self.build_items()),
            );
            fd.viewport.background_colour = Some([0.12, 0.12, 0.14, 1.0]);
            fd.effects.lighting = self.build_lighting();
            fd.effects.exposure = self.build_exposure(dt);

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
