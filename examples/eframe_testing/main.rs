//! Shadow cascade diagnostic test
//!
//! Scene: large flat ground plane + several shadow-casting boxes.
//! Side panel shows live cascade splits and camera state.
//! Use the controls to change cascade count and shadow settings,
//! then zoom out until the artifact appears to correlate its position
//! with the cascade boundary readout.
//!
//! Key bindings:
//!   Left drag / Middle drag = orbit
//!   Right drag = pan
//!   Scroll = zoom

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    BackfacePolicy, ButtonState, Camera, CameraFrame, FrameData, LightKind, LightSource,
    LightingSettings, Material, MeshId, OrbitCameraController, SceneFrame, SceneRenderItem,
    ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};

// Solid unlit colours for the subsurface objects. Picked to be visually distinct
// so any bleed-through is immediately identifiable by colour.
const COLOUR_MAGENTA: [f32; 3] = [1.0, 0.0, 1.0];
const COLOUR_CYAN: [f32; 3] = [0.0, 1.0, 1.0];

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : shadow cascade debug",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 900.0]),
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
            let _queue = &rs.queue;

            let mut renderer = ViewportRenderer::new(device, rs.target_format);
            let res = renderer.resources_mut();

            // Ground plane: large flat slab so cascade boundaries are visible.
            let ground = primitives::cuboid(40.0, 40.0, 0.3);
            let ground_id = res.upload_mesh_data(device, &ground).unwrap();

            // Shadow casters: boxes at different distances from centre.
            let box_mesh = primitives::cuboid(1.0, 1.0, 2.0);
            let box_id = res.upload_mesh_data(device, &box_mesh).unwrap();

            let sphere_mesh = primitives::sphere(0.8, 24, 12);
            let sphere_id = res.upload_mesh_data(device, &sphere_mesh).unwrap();

            // Unlit objects placed below the ground plane. Any visible bleed-through
            // of their colours indicates the depth issue we are diagnosing.
            let unlit_sphere_mesh = primitives::sphere(2.0, 24, 12);
            let unlit_sphere_id = res.upload_mesh_data(device, &unlit_sphere_mesh).unwrap();
            let unlit_box_mesh = primitives::cuboid(3.0, 3.0, 3.0);
            let unlit_box_id = res.upload_mesh_data(device, &unlit_box_mesh).unwrap();

            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App::new(
                ground_id,
                box_id,
                sphere_id,
                unlit_sphere_id,
                unlit_box_id,
            )))
        }),
    )
}

// ---------------------------------------------------------------------------
// Cascade split formula (mirrors shadows.rs so we can show live splits)
// ---------------------------------------------------------------------------

fn cascade_splits(near: f32, far: f32, count: u32) -> Vec<f32> {
    let n = count.min(4) as usize;
    (1..=n)
        .map(|i| {
            let p = i as f32 / n as f32;
            near * (far / near).powf(p)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    ground_id: MeshId,
    box_id: MeshId,
    sphere_id: MeshId,
    unlit_sphere_id: MeshId,
    unlit_box_id: MeshId,
    lighting: LightingSettings,
    show_hemisphere: bool,
    log_next_frame: bool,
    ground_two_sided: bool,
    show_subsurface: bool,
    gpu_culling: bool,
}

impl App {
    fn new(
        ground_id: MeshId,
        box_id: MeshId,
        sphere_id: MeshId,
        unlit_sphere_id: MeshId,
        unlit_box_id: MeshId,
    ) -> Self {
        let mut lighting = LightingSettings::default();
        lighting.lights = vec![{
            let mut _t = LightSource::default();
            _t.kind = LightKind::Directional {
                direction: [0.4, 0.3, 1.5],
            };
            _t.colour = [1.0, 1.0, 1.0];
            _t.intensity = 1.0;
            _t
        }];
        lighting.hemisphere_intensity = 0.4;
        lighting.shadow_cascade_count = 4;
        lighting.shadows_enabled = true;

        Self {
            camera: Camera {
                distance: 20.0,
                zfar: 1000.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            ground_id,
            box_id,
            sphere_id,
            unlit_sphere_id,
            unlit_box_id,
            lighting,
            show_hemisphere: true,
            log_next_frame: false,
            ground_two_sided: true,
            show_subsurface: true,
            gpu_culling: true,
        }
    }

    fn build_scene(&self) -> Vec<SceneRenderItem> {
        let ground_mat = Material::pbr([0.7, 0.7, 0.7], 0.0, 0.8);
        let box_mat = Material::pbr([0.8, 0.5, 0.3], 0.0, 0.6);
        let sphere_mat = Material::pbr([0.4, 0.6, 0.9], 0.0, 0.3);

        let ground = {
            let mut item = SceneRenderItem::default();
            item.mesh_id = self.ground_id;

            item.material = {
                let mut m = ground_mat;
                m.backface_policy = if self.ground_two_sided {
                    BackfacePolicy::Identical
                } else {
                    BackfacePolicy::Cull
                };
                m
            };
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.15)).to_cols_array_2d();
            item
        };

        // Boxes at distances 2m, 5m, 10m, 20m from centre.
        let box_positions = [
            glam::Vec3::new(2.0, 0.0, 1.0),
            glam::Vec3::new(-5.0, 3.0, 1.0),
            glam::Vec3::new(10.0, -4.0, 1.0),
            glam::Vec3::new(-15.0, 8.0, 1.0),
        ];

        let mut items = vec![ground];
        for pos in &box_positions {
            let mut item = SceneRenderItem::default();
            item.mesh_id = self.box_id;

            item.material = box_mat;
            item.model = glam::Mat4::from_translation(*pos).to_cols_array_2d();
            items.push(item);
        }

        // Sphere at the origin, slightly above ground, as a depth-fight probe.
        {
            let mut item = SceneRenderItem::default();
            item.mesh_id = self.sphere_id;

            item.material = sphere_mat;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.8)).to_cols_array_2d();
            items.push(item);
        }

        // Unlit objects below the ground plane. Their colours bleed through if
        // the depth issue is active. Magenta sphere at the centre, cyan box
        // offset so both are independently visible.
        if self.show_subsurface {
            {
                let mut item = SceneRenderItem::default();
                item.mesh_id = self.unlit_sphere_id;
                item.settings.unlit = true;
                item.material.base_colour = COLOUR_MAGENTA;
                item.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -2.0))
                    .to_cols_array_2d();
                items.push(item);
            }

            {
                let mut item = SceneRenderItem::default();
                item.mesh_id = self.unlit_box_id;
    
                item.settings.unlit = true;
                item.material.base_colour = COLOUR_CYAN;
                item.model = glam::Mat4::from_translation(glam::Vec3::new(8.0, 5.0, -2.5))
                    .to_cols_array_2d();
                items.push(item);
            }
        }

        items
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Side panel: controls + live readout.
        egui::SidePanel::left("controls")
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Shadow debug");
                ui.separator();

                ui.label("Cascade count");
                let mut cc = self.lighting.shadow_cascade_count as i32;
                if ui.add(egui::Slider::new(&mut cc, 1..=4)).changed() {
                    self.lighting.shadow_cascade_count = cc as u32;
                }


                ui.label("Shadow bias");
                ui.add(
                    egui::Slider::new(&mut self.lighting.shadow_bias, 0.0..=0.01).logarithmic(true),
                );

                ui.checkbox(&mut self.lighting.shadows_enabled, "Shadows enabled");

                if ui
                    .checkbox(&mut self.show_hemisphere, "Hemisphere ambient")
                    .changed()
                {
                    self.lighting.hemisphere_intensity =
                        if self.show_hemisphere { 0.4 } else { 0.0 };
                }

                ui.separator();
                ui.label("Geometry / GPU");
                ui.checkbox(
                    &mut self.ground_two_sided,
                    "Ground two-sided (Identical vs Cull)",
                );
                ui.checkbox(
                    &mut self.show_subsurface,
                    "Show subsurface objects (magenta/cyan)",
                );
                ui.checkbox(&mut self.gpu_culling, "GPU driven culling");

                ui.separator();
                ui.label("Camera");

                let dist = self.camera.distance;
                let eff_near = self.camera.effective_znear();
                let eff_far = self.camera.effective_zfar();
                let precision_limit = (2.0_f32 * eff_near * eff_far / (eff_far - eff_near)).sqrt();
                ui.label(format!("distance:         {:.2}", dist));
                ui.label(format!("znear (eff):      {:.4}", eff_near));
                ui.label(format!("effective far:    {:.2}", eff_far));
                ui.label(format!("near/far ratio:   {:.0}:1", eff_far / eff_near));
                ui.label(format!("depth limit @2m:  {:.1}m", precision_limit));

                ui.separator();
                ui.label("Cascade splits (world depth)");
                let shadow_far = (self.camera.distance * 3.0).max(10.0).min(eff_far);
                let splits = cascade_splits(
                    eff_near,
                    shadow_far,
                    self.lighting.shadow_cascade_count,
                );
                let mut prev = eff_near;
                for (i, &s) in splits.iter().enumerate() {
                    ui.label(format!("  [{}] {:.2} .. {:.2}", i, prev, s));
                    prev = s;
                }

                ui.separator();
                if ui.button("Log cascade matrices (P)").clicked() || self.log_next_frame {
                    self.log_next_frame = false;
                    eprintln!("=== CASCADE STATE ===");
                    eprintln!("  camera distance: {:.2}", dist);
                    eprintln!("  near: {:.4}  shadow_far: {:.2}", eff_near, shadow_far);
                    let splits = cascade_splits(
                        eff_near,
                        shadow_far,
                        self.lighting.shadow_cascade_count,
                    );
                    for (i, &s) in splits.iter().enumerate() {
                        let lo = if i == 0 { eff_near } else { splits[i - 1] };
                        eprintln!("  cascade[{}]: {:.3} .. {:.3}", i, lo, s);
                    }
                    eprintln!("=== END ===");
                }

                ui.small("P = log to terminal");
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
                if i.key_pressed(egui::Key::P) {
                    self.log_next_frame = true;
                }

                self.controller.push_event(ViewportEvent::ModifiersChanged(
                    viewport_lib::Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    },
                ));

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
                            let vp_btn = match button {
                                egui::PointerButton::Primary => viewport_lib::MouseButton::Left,
                                egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
                                egui::PointerButton::Middle => viewport_lib::MouseButton::Middle,
                                _ => continue,
                            };
                            self.controller.push_event(ViewportEvent::MouseButton {
                                button: vp_btn,
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

            let w = rect.width();
            let h = rect.height();
            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(w, h);

            let mut lighting = self.lighting.clone();
            if !self.show_hemisphere {
                lighting.hemisphere_intensity = 0.0;
            }

            let frame_data = FrameData::new(
                CameraFrame::from_camera(&self.camera, [w, h]),
                SceneFrame::from_surface_items(self.build_scene()),
            )
            .with_lighting(lighting);

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback {
                        frame: frame_data,
                        gpu_culling: self.gpu_culling,
                    },
                ));
        });
    }
}
