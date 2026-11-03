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

use viewport_lib::Colour;
mod viewport_callback;

use eframe::egui;
use viewport_lib::input::adapters::{AdapterCtx, egui as vpl_egui};
use viewport_lib::{
    Camera, CameraFrame, FrameData, OrbitCameraController, SceneFrame, ViewportContext,
    ViewportInput, ViewportRenderer,
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

            // The paint callback draws into eframe's own pass, whose depth
            // attachment is 24-bit.
            let mut renderer = ViewportRenderer::with_config(
                device,
                &viewport_lib::RendererConfig::new(format)
                    .with_scene_depth_format(eframe::wgpu::TextureFormat::Depth24PlusStencil8),
            );

            // Build every catalogue scene once, uploading all assets up front.
            let mut builts: Vec<BuiltScene> = Vec::new();
            let mut names: Vec<&'static str> = Vec::new();
            let mut cameras: Vec<Vec<NamedCamera>> = Vec::new();
            for scene in catalogue() {
                let built = {
                    let mut ctx = BuildCtx {
                        renderer: &mut renderer,
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
                controller: OrbitCameraController::new_stateless(),
                input: ViewportInput::from_preset(viewport_lib::BindingPreset::Viewer),
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
    /// Owns the bindings and resolves events into an `ActionFrame`. The
    /// controller keeps only its sensitivities and applies the frame.
    input: ViewportInput,
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

                let ownership = vpl_egui::ownership(&response, ui.ctx(), rect, false);
                self.input.begin_frame(ViewportContext::new(
                    ownership,
                    [rect.width(), rect.height()],
                ));
                // Pan reads the viewport height off the controller.
                self.controller
                    .set_viewport_size([rect.width(), rect.height()]);

                let actx = AdapterCtx::new(1.0, glam::Vec2::new(rect.left(), rect.top()));
                let mut events = Vec::new();
                ui.input(|i| {
                    let local = i
                        .pointer
                        .interact_pos()
                        .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
                    self.cursor_prev = self.cursor;
                    self.cursor = local;
                    vpl_egui::begin_frame(i, &actx, &mut events);
                    for event in &i.events {
                        vpl_egui::translate(event, &actx, &mut events);
                    }
                });
                for ev in events {
                    self.input.push_event(ev);
                }

                let (w, h) = (rect.width(), rect.height());
                let action_frame = self.input.resolve();
                let dt = ctx.input(|i| i.stable_dt);
                self.controller.apply(&mut self.camera, &action_frame, dt);
                self.camera.set_aspect_ratio(w, h);

                let built = &self.builts[self.selected];
                let mut fd = FrameData::new(
                    CameraFrame::from_camera(&self.camera, [w, h])
                        .with_pixels_per_point(ui.ctx().pixels_per_point()),
                    SceneFrame::from_surface_items(built.items.clone()),
                );
                fd.effects.lighting = built.lighting.clone();
                if let Some(bg) = built.background {
                    fd.viewport.background_colour = Some(Colour::from_linear_array(bg));
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
