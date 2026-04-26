//! Primitives showcase : every geometry primitive displayed in a single viewport.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, LightKind, LightSource, LightingSettings,
    Material, MeshId, OrbitCameraController, SceneFrame, SceneRenderItem, ScrollUnits,
    ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Primitives Showcase",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
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

            let mut renderer = ViewportRenderer::new(device, rs.target_format);
            let res = renderer.resources_mut();

            macro_rules! mesh {
                ($data:expr) => {
                    res.upload_mesh_data(device, &$data).expect("upload mesh")
                };
            }

            // Row 0 : basic solids
            let m_cube = mesh!(primitives::cube(1.0));
            let m_cuboid = mesh!(primitives::cuboid(2.0, 0.75, 1.0));
            let m_sphere = mesh!(primitives::sphere(0.6, 32, 16));
            let m_icosphere = mesh!(primitives::icosphere(0.6, 3));

            // Row 1 : round / capped
            let m_ellipsoid = mesh!(primitives::ellipsoid(0.9, 0.5, 0.6, 28, 14));
            let m_hemisphere = mesh!(primitives::hemisphere(0.65, 32, 16));
            let m_cone = mesh!(primitives::cone(0.55, 1.1, 28));
            let m_cylinder = mesh!(primitives::cylinder(0.4, 1.1, 28));

            // Row 2 : curved surfaces
            let m_capsule = mesh!(primitives::capsule(0.38, 1.4, 24, 16));
            let m_torus = mesh!(primitives::torus(0.55, 0.2, 40, 24));
            let m_disk = mesh!(primitives::disk(0.65, 40));
            let m_ring = mesh!(primitives::ring(0.3, 0.65, 48));

            // Row 3 : flat / composite
            let m_plane = mesh!(primitives::plane(1.8, 1.8));
            let m_grid_plane = mesh!(primitives::grid_plane(1.8, 1.8, 8, 8));
            let m_frustum = mesh!(primitives::frustum(
                std::f32::consts::FRAC_PI_4, // 45° fov — tighter than 60°
                4.0 / 3.0,
                0.12,
                1.1
            ));
            let m_arrow = mesh!(primitives::arrow(0.07, 0.18, 0.28, 24));

            // Row 4 : spring variants
            let m_spring_a = mesh!(primitives::spring(0.35, 0.08, 5.0, 14));
            let m_spring_b = mesh!(primitives::spring(0.28, 0.12, 3.0, 18));

            rs.renderer.write().callback_resources.insert(renderer);

            let cx = [-5.25f32, -1.75, 1.75, 5.25];
            let rz = [-7.0f32, -3.5, 0.0, 3.5, 7.0];

            let mut item = |mesh_id: MeshId, x, z, color: [f32; 3]| {
                let mut s = SceneRenderItem::default();
                s.mesh_id = mesh_id;
                s.model =
                    glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, z)).to_cols_array_2d();
                s.material = Material::from_color(color);
                s.two_sided = true;
                s
            };

            let scene_items = vec![
                item(m_cube, cx[0], rz[0], [0.75, 0.75, 0.75]),
                item(m_cuboid, cx[1], rz[0], [0.35, 0.55, 0.90]),
                item(m_sphere, cx[2], rz[0], [0.90, 0.50, 0.20]),
                item(m_icosphere, cx[3], rz[0], [0.25, 0.75, 0.40]),
                item(m_ellipsoid, cx[0], rz[1], [0.70, 0.30, 0.85]),
                item(m_hemisphere, cx[1], rz[1], [0.50, 0.80, 0.35]),
                item(m_cone, cx[2], rz[1], [0.85, 0.20, 0.25]),
                item(m_cylinder, cx[3], rz[1], [0.20, 0.70, 0.80]),
                item(m_capsule, cx[0], rz[2], [0.90, 0.45, 0.65]),
                item(m_torus, cx[1], rz[2], [0.85, 0.70, 0.15]),
                item(m_disk, cx[2], rz[2], [0.40, 0.65, 0.95]),
                item(m_ring, cx[3], rz[2], [0.55, 0.90, 0.30]),
                item(m_plane, cx[0], rz[3], [0.60, 0.55, 0.75]),
                item(m_grid_plane, cx[1], rz[3], [0.80, 0.50, 0.25]),
                {
                    // Rotate 180° around X so the frustum opens upward (+Z) in the z-up scene.
                    let mut s = item(m_frustum, cx[2], rz[3], [0.30, 0.65, 0.90]);
                    s.model = (glam::Mat4::from_translation(glam::Vec3::new(cx[2], 0.0, rz[3]))
                        * glam::Mat4::from_rotation_x(std::f32::consts::PI))
                        .to_cols_array_2d();
                    s
                },
                item(m_arrow, cx[3], rz[3], [0.90, 0.25, 0.30]),
                item(m_spring_a, cx[0], rz[4], [0.85, 0.30, 0.75]),
                item(m_spring_b, cx[1], rz[4], [0.55, 0.35, 0.90]),
            ];

            Ok(Box::new(App::new(scene_items)))
        }),
    )
}

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    scene_items: Vec<SceneRenderItem>,
}

impl App {
    fn new(scene_items: Vec<SceneRenderItem>) -> Self {
        Self {
            camera: Camera {
                center: glam::Vec3::ZERO,
                distance: 28.0,
                // Z-up: compose a 30° azimuth around Z with a ~57° tilt from vertical.
                // orientation * Y ≈ world Z (screen up), orientation * Z points from center to eye.
                orientation: glam::Quat::from_rotation_z(0.5)
                    * glam::Quat::from_rotation_x(1.0),
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            scene_items,
        }
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
                self.controller.push_event(ViewportEvent::ModifiersChanged(
                    viewport_lib::Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    },
                ));

                if let Some(pos) = i.pointer.interact_pos() {
                    self.controller.push_event(ViewportEvent::PointerMoved {
                        position: glam::Vec2::new(pos.x - rect.left(), pos.y - rect.top()),
                    });
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
                        egui::Event::MouseWheel { delta, .. } => {
                            self.controller.push_event(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units: ScrollUnits::Pixels,
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

            let mut frame_data = FrameData::new(
                CameraFrame::from_camera(&self.camera, [w, h]),
                SceneFrame::from_surface_items(self.scene_items.clone()),
            );
            frame_data.effects.lighting = LightingSettings {
                lights: vec![
                    // Key light: warm, from upper front-right.
                    LightSource {
                        kind: LightKind::Directional {
                            direction: [0.4, -0.5, 1.2],
                        },
                        color: [1.0, 0.97, 0.92],
                        intensity: 1.0,
                    },
                    // Cool fill from the opposite side.
                    LightSource {
                        kind: LightKind::Directional {
                            direction: [-0.8, 0.6, 0.3],
                        },
                        color: [0.70, 0.82, 1.0],
                        intensity: 0.35,
                    },
                ],
                hemisphere_intensity: 0.35,
                sky_color: [0.80, 0.90, 1.0],
                ground_color: [0.35, 0.28, 0.22],
                ..LightingSettings::default()
            };
            frame_data.viewport.show_grid = false;
            frame_data.viewport.show_axes_indicator = true;

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback { frame: frame_data },
                ));

            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}
