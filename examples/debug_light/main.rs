//! Debug example reproducing the lighting setup from
//! hamilton_engine_v2/examples/cloth/cloth_fast_drape.rs.
//!
//! Scene: a flat gray floor plane with a small flat platform (slab)
//! floating above it. Lighting is `LightingSettings::default()`.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    BackfacePolicy, ButtonState, Camera, CameraFrame, FrameData, LightingSettings, Material,
    MeshId, Modifiers, MouseButton, OrbitCameraController, SceneFrame, SceneRenderItem,
    ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};

const SLAB_HALF_EXTENTS: [f32; 3] = [0.5, 0.5, 0.005];
const SLAB_HEIGHT: f32 = 0.6;

const FLOOR_COLOUR: [f32; 3] = [0.35, 0.35, 0.38];
const SLAB_COLOUR: [f32; 3] = [0.85, 0.55, 0.35];

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Debug Light",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 720.0]),
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
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);
            let res = renderer.resources_mut();

            let floor_mesh = res
                .upload_mesh_data(device, &primitives::plane(20.0, 20.0))
                .expect("floor mesh upload");
            let slab_mesh = res
                .upload_mesh_data(
                    device,
                    &primitives::cuboid(
                        2.0 * SLAB_HALF_EXTENTS[0],
                        2.0 * SLAB_HALF_EXTENTS[1],
                        2.0 * SLAB_HALF_EXTENTS[2],
                    ),
                )
                .expect("slab mesh upload");

            rs.renderer.write().callback_resources.insert(renderer);
            Ok(Box::new(App::new(floor_mesh, slab_mesh)))
        }),
    )
}

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    floor_mesh: MeshId,
    slab_mesh: MeshId,
}

impl App {
    fn new(floor_mesh: MeshId, slab_mesh: MeshId) -> Self {
        Self {
            camera: Camera {
                center: glam::Vec3::new(0.0, 0.0, 1.0),
                distance: 5.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            floor_mesh,
            slab_mesh,
        }
    }

    fn build_items(&self) -> Vec<SceneRenderItem> {
        let mut floor = SceneRenderItem::default();
        floor.mesh_id = self.floor_mesh;
        floor.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        floor.material = {
            let mut m = Material::from_colour(FLOOR_COLOUR);
            m.backface_policy = BackfacePolicy::Identical;
            m
        };

        let mut slab = SceneRenderItem::default();
        slab.mesh_id = self.slab_mesh;
        slab.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, SLAB_HEIGHT))
            .to_cols_array_2d();
        slab.material = Material::from_colour(SLAB_COLOUR);

        vec![slab, floor]
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.controller.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            });

            ui.input(|i| {
                self.controller
                    .push_event(ViewportEvent::ModifiersChanged(Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    }));
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
                            let vp_btn = match button {
                                egui::PointerButton::Primary => MouseButton::Left,
                                egui::PointerButton::Secondary => MouseButton::Right,
                                egui::PointerButton::Middle => MouseButton::Middle,
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

            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(rect.width(), rect.height());

            let mut frame_data = FrameData::new(
                CameraFrame::from_camera(&self.camera, [rect.width(), rect.height()])
                    .with_pixels_per_point(ui.ctx().pixels_per_point()),
                SceneFrame::from_surface_items(self.build_items()),
            );
            frame_data.effects.lighting = LightingSettings::default();

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback { frame: frame_data },
                ));
        });

        ctx.request_repaint();
    }
}
