//! `eframe` / `egui` integration for `viewport-lib`.
//!
//! Uses `egui_wgpu` callback resources to share the host renderer and drive the
//! viewport from an ordinary `egui` app.

mod viewport_callback;

use viewport_lib::{
    Action, ActionState, FrameInput, InputSystem, KeyCode, MeshData, Modifiers,
    MouseButton, ViewportRenderer,
};
use eframe::egui;

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib - eframe Example",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 600.0]),
            // The viewport's GPU pipelines require a depth-stencil attachment
            // (Depth24PlusStencil8 - stencil is used for selection outlines).
            depth_buffer: 24,
            stencil_buffer: 8,
            ..Default::default()
        },
        Box::new(|cc| {
            // Register the ViewportRenderer as a callback resource so the
            // paint callback can access it during rendering.
            let wgpu_render_state = cc
                .wgpu_render_state
                .as_ref()
                .expect("eframe must be configured with wgpu backend");
            let device = &wgpu_render_state.device;
            let format = wgpu_render_state.target_format;
            let renderer = ViewportRenderer::new(device, format);
            wgpu_render_state
                .renderer
                .write()
                .callback_resources
                .insert(renderer);

            // Upload the box mesh once at startup.
            let box_mesh = unit_box_mesh();
            let mut guard = wgpu_render_state.renderer.write();
            if let Some(vr) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                // Pre-upload a few meshes so objects added later can reference them.
                for _ in 0..16 {
                    vr.resources_mut()
                        .upload_mesh_data(device, &box_mesh)
                        .expect("built-in mesh");
                }
            }

            Ok(Box::new(App::default()))
        }),
    )
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct App {
    objects: Vec<SceneObj>,
    next_id: u64,
    camera: viewport_lib::Camera,
    input: InputSystem,
}

#[derive(Clone)]
struct SceneObj {
    id: u64,
    name: String,
    position: [f32; 3],
    mesh_index: usize,
}

impl Default for App {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            next_id: 1,
            camera: viewport_lib::Camera {
                center: glam::Vec3::ZERO,
                distance: 12.0,
                orientation: glam::Quat::from_rotation_y(0.6) * glam::Quat::from_rotation_x(0.4),
                ..viewport_lib::Camera::default()
            },
            input: InputSystem::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Camera control constants shared with the other viewport examples.
// ---------------------------------------------------------------------------

const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.001;
const MIN_DISTANCE: f32 = 0.1;

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Left panel: object list ---
        egui::SidePanel::left("object_panel")
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Objects");
                if ui.button("+ Add Box").clicked() {
                    let id = self.next_id;
                    self.next_id += 1;
                    let n = self.objects.len() as f32;
                    let x = (n % 4.0) * 2.0 - 3.0;
                    let z = (n / 4.0).floor() * 2.0 - 3.0;
                    // Each object uses mesh slot = its index (pre-uploaded at startup).
                    let mesh_index = self.objects.len();
                    self.objects.push(SceneObj {
                        id,
                        name: format!("Box {id}"),
                        position: [x, 0.0, z],
                        mesh_index,
                    });
                }

                ui.separator();

                let mut to_remove = None;
                for obj in &self.objects {
                    ui.horizontal(|ui| {
                        ui.label(&obj.name);
                        if ui.small_button("x").clicked() {
                            to_remove = Some(obj.id);
                        }
                    });
                }
                if let Some(id) = to_remove {
                    self.objects.retain(|o| o.id != id);
                }
            });

        // --- Central panel: viewport ---
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

            // Translate egui input into a framework-agnostic FrameInput.
            let frame_input = build_frame_input(ui, &response);

            // --- Camera: Orbit ---
            if let ActionState::Active { delta } = self.input.query(Action::Orbit, &frame_input) {
                let q_yaw = glam::Quat::from_rotation_y(-delta.x * ORBIT_SENSITIVITY);
                let q_pitch = glam::Quat::from_rotation_x(-delta.y * ORBIT_SENSITIVITY);
                self.camera.orientation = (q_yaw * self.camera.orientation * q_pitch).normalize();
            }
            // Ctrl+scroll orbit (2-axis)
            let cd = frame_input.ctrl_scroll_orbit_delta;
            if cd != glam::Vec2::ZERO {
                let q_yaw = glam::Quat::from_rotation_y(-cd.x * ORBIT_SENSITIVITY);
                let q_pitch = glam::Quat::from_rotation_x(-cd.y * ORBIT_SENSITIVITY);
                self.camera.orientation = (q_yaw * self.camera.orientation * q_pitch).normalize();
            }

            // --- Camera: Pan ---
            if let ActionState::Active { delta } = self.input.query(Action::Pan, &frame_input) {
                let pan_scale =
                    2.0 * self.camera.distance * (self.camera.fov_y / 2.0).tan() / rect.height();
                self.camera.center -= self.camera.right() * delta.x * pan_scale;
                self.camera.center += self.camera.up() * delta.y * pan_scale;
            }
            // Shift+scroll pan (2-axis)
            let sp = frame_input.shift_scroll_pan_delta;
            if sp != glam::Vec2::ZERO {
                let pan_scale =
                    2.0 * self.camera.distance * (self.camera.fov_y / 2.0).tan() / rect.height();
                self.camera.center -= self.camera.right() * sp.x * pan_scale;
                self.camera.center += self.camera.up() * sp.y * pan_scale;
            }

            // --- Camera: Zoom ---
            if let ActionState::Active { delta } = self.input.query(Action::Zoom, &frame_input) {
                self.camera.distance =
                    (self.camera.distance * (1.0 - delta.y * ZOOM_SENSITIVITY)).max(MIN_DISTANCE);
            }

            // Update camera aspect ratio.
            if rect.height() > 0.0 {
                self.camera.aspect = rect.width() / rect.height();
            }

            // Build scene items.
            let scene_items: Vec<viewport_lib::SceneRenderItem> = self
                .objects
                .iter()
                .map(|obj| {
                    let model = glam::Mat4::from_translation(glam::Vec3::from(obj.position));
                    {
                        let mut item = viewport_lib::SceneRenderItem::default();
                        item.mesh_index = obj.mesh_index;
                        item.model = model.to_cols_array_2d();
                        item
                    }
                })
                .collect();

            let frame_data = {
                let mut fd = viewport_lib::FrameData::default();
                fd.camera_uniform = viewport_lib::CameraUniform {
                    view_proj: self.camera.view_proj_matrix().to_cols_array_2d(),
                    eye_pos: self.camera.eye_position().into(),
                    _pad: 0.0,
                };
                fd.lighting = viewport_lib::LightingSettings::default();
                fd.eye_pos = self.camera.eye_position().into();
                fd.scene_items = scene_items;
                fd.show_grid = true;
                fd.show_axes_indicator = true;
                fd.viewport_size = [rect.width(), rect.height()];
                fd.camera_orientation = self.camera.orientation;
                fd
            };

            // Schedule the viewport paint callback.
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback { frame: frame_data },
                ));

            // Show grab cursor when hovering the viewport.
            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// egui -> FrameInput adapter
// ---------------------------------------------------------------------------

fn egui_key_to_keycode(key: egui::Key) -> Option<KeyCode> {
    match key {
        egui::Key::A => Some(KeyCode::A),
        egui::Key::B => Some(KeyCode::B),
        egui::Key::C => Some(KeyCode::C),
        egui::Key::D => Some(KeyCode::D),
        egui::Key::E => Some(KeyCode::E),
        egui::Key::F => Some(KeyCode::F),
        egui::Key::G => Some(KeyCode::G),
        egui::Key::H => Some(KeyCode::H),
        egui::Key::I => Some(KeyCode::I),
        egui::Key::J => Some(KeyCode::J),
        egui::Key::K => Some(KeyCode::K),
        egui::Key::L => Some(KeyCode::L),
        egui::Key::M => Some(KeyCode::M),
        egui::Key::N => Some(KeyCode::N),
        egui::Key::O => Some(KeyCode::O),
        egui::Key::P => Some(KeyCode::P),
        egui::Key::Q => Some(KeyCode::Q),
        egui::Key::R => Some(KeyCode::R),
        egui::Key::S => Some(KeyCode::S),
        egui::Key::T => Some(KeyCode::T),
        egui::Key::U => Some(KeyCode::U),
        egui::Key::V => Some(KeyCode::V),
        egui::Key::W => Some(KeyCode::W),
        egui::Key::X => Some(KeyCode::X),
        egui::Key::Y => Some(KeyCode::Y),
        egui::Key::Z => Some(KeyCode::Z),
        egui::Key::Tab => Some(KeyCode::Tab),
        egui::Key::Enter => Some(KeyCode::Enter),
        egui::Key::Escape => Some(KeyCode::Escape),
        egui::Key::Backtick => Some(KeyCode::Backtick),
        _ => None,
    }
}

fn build_frame_input(ui: &egui::Ui, response: &egui::Response) -> FrameInput {
    use std::collections::HashSet;

    let mut keys_pressed = HashSet::new();
    let mut keys_held = HashSet::new();

    ui.input(|i| {
        for event in &i.events {
            if let egui::Event::Key {
                key,
                pressed,
                repeat,
                ..
            } = event
            {
                if let Some(kc) = egui_key_to_keycode(*key) {
                    if *pressed && !*repeat {
                        keys_pressed.insert(kc);
                    }
                    if *pressed {
                        keys_held.insert(kc);
                    }
                }
            }
        }
        for key in [
            egui::Key::A,
            egui::Key::B,
            egui::Key::C,
            egui::Key::D,
            egui::Key::E,
            egui::Key::F,
            egui::Key::G,
            egui::Key::H,
            egui::Key::I,
            egui::Key::J,
            egui::Key::K,
            egui::Key::L,
            egui::Key::M,
            egui::Key::N,
            egui::Key::O,
            egui::Key::P,
            egui::Key::Q,
            egui::Key::R,
            egui::Key::S,
            egui::Key::T,
            egui::Key::U,
            egui::Key::V,
            egui::Key::W,
            egui::Key::X,
            egui::Key::Y,
            egui::Key::Z,
            egui::Key::Tab,
            egui::Key::Enter,
            egui::Key::Escape,
            egui::Key::Backtick,
        ] {
            if i.key_down(key) {
                if let Some(kc) = egui_key_to_keycode(key) {
                    keys_held.insert(kc);
                }
            }
        }
    });

    let modifiers = ui.input(|i| Modifiers {
        alt: i.modifiers.alt,
        shift: i.modifiers.shift,
        ctrl: i.modifiers.command,
    });

    let mut drag_started = HashSet::new();
    let mut dragging = HashSet::new();
    let mut clicked = HashSet::new();

    for (egui_btn, our_btn) in [
        (egui::PointerButton::Primary, MouseButton::Left),
        (egui::PointerButton::Secondary, MouseButton::Right),
        (egui::PointerButton::Middle, MouseButton::Middle),
    ] {
        if response.drag_started_by(egui_btn) {
            drag_started.insert(our_btn);
        }
        if response.dragged_by(egui_btn) {
            dragging.insert(our_btn);
        }
        if response.clicked_by(egui_btn) {
            clicked.insert(our_btn);
        }
    }

    let drag_delta_egui = response.drag_delta();
    let scroll_delta = if response.hovered() {
        ui.input(|i| i.smooth_scroll_delta.y).clamp(-150.0, 150.0)
    } else {
        0.0
    };
    let pointer_delta_egui = ui.input(|i| i.pointer.delta());

    let ctrl_scroll_orbit_delta = if response.hovered() {
        let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
        if ctrl {
            ui.input(|i| {
                let mut d = egui::Vec2::ZERO;
                for e in &i.events {
                    if let egui::Event::MouseWheel { delta, .. } = e {
                        d += *delta;
                    }
                }
                glam::Vec2::new(d.x, d.y)
            })
        } else {
            glam::Vec2::ZERO
        }
    } else {
        glam::Vec2::ZERO
    };

    let shift_scroll_pan_delta = if response.hovered() {
        let shift = ui.input(|i| i.modifiers.shift);
        if shift {
            ui.input(|i| {
                let mut d = egui::Vec2::ZERO;
                for e in &i.events {
                    if let egui::Event::MouseWheel { delta, .. } = e {
                        d += *delta;
                    }
                }
                glam::Vec2::new(d.x, d.y)
            })
        } else {
            glam::Vec2::ZERO
        }
    } else {
        glam::Vec2::ZERO
    };

    FrameInput {
        keys_pressed,
        keys_held,
        modifiers,
        drag_started,
        dragging,
        drag_delta: glam::Vec2::new(drag_delta_egui.x, drag_delta_egui.y),
        scroll_delta,
        pointer_delta: glam::Vec2::new(pointer_delta_egui.x, pointer_delta_egui.y),
        clicked,
        hovered: response.hovered(),
        ctrl_scroll_orbit_delta,
        shift_scroll_pan_delta,
    }
}

// ---------------------------------------------------------------------------
// Box mesh helper
// ---------------------------------------------------------------------------

fn unit_box_mesh() -> MeshData {
    #[rustfmt::skip]
    let positions: Vec<[f32; 3]> = vec![
        // Front face
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        // Back face
        [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        // Top face
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        // Bottom face
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
        // Right face
        [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
        // Left face
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
    ];

    #[rustfmt::skip]
    let normals: Vec<[f32; 3]> = vec![
        // Front
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        // Back
        [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
        // Top
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        // Bottom
        [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
        // Right
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        // Left
        [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    ];

    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0,  1,  2,  0,  2,  3,   // Front
        4,  5,  6,  4,  6,  7,   // Back
        8,  9,  10, 8,  10, 11,  // Top
        12, 13, 14, 12, 14, 15,  // Bottom
        16, 17, 18, 16, 18, 19,  // Right
        20, 21, 22, 20, 22, 23,  // Left
    ];

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}
