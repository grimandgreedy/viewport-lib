//! Demonstrates the two built-in input presets.
//!
//! Press **1** — Camera only (`ViewportPrimitives`):
//!   Left/Middle drag → Orbit  ·  Right drag → Pan  ·  Scroll → Zoom
//!   Ctrl+Scroll → Orbit (two-axis)  ·  Shift+Scroll → Pan (two-axis)
//!
//! Press **2** — Full controls (`ViewportAll`) + `ManipulationController`:
//!   Ctrl+Scroll → Orbit  ·  Right drag → Pan  ·  Scroll → Zoom
//!   G move  R rotate  S scale  ·  X/Y/Z constrain  ·  Shift+X/Y/Z exclude
//!   Enter / left-click → confirm  ·  Esc → cancel

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, LightingSettings, ManipResult,
    ManipulationContext, ManipulationController, Material, OrbitCameraController, SceneFrame,
    SceneRenderItem, ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
    KeyCode,
};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib — Input Controllers",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 600.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc.wgpu_render_state.as_ref().expect("wgpu backend required");
            let device = &rs.device;
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);
            let res = renderer.resources_mut();
            let m_box    = res.upload_mesh_data(device, &primitives::cube(1.0)).unwrap();
            let m_sphere = res.upload_mesh_data(device, &primitives::sphere(0.75, 32, 16)).unwrap();

            rs.renderer.write().callback_resources.insert(renderer);
            Ok(Box::new(App::new(m_box, m_sphere)))
        }),
    )
}

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Camera navigation only — ViewportPrimitives preset.
    Primitives,
    /// Full controls + object manipulation — ViewportAll preset.
    All,
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    mode: Mode,
    camera: Camera,

    // Mode 1
    ctrl_primitives: OrbitCameraController,

    // Mode 2
    ctrl_all: OrbitCameraController,
    manip: ManipulationController,
    sphere_model: [[f32; 4]; 4],
    snapshot: [[f32; 4]; 4],

    m_box: usize,
    m_sphere: usize,

    // Per-frame drag / click state (for ManipulationContext)
    cursor_viewport: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,
    left_held: bool,
    drag_started: bool,
    clicked: bool,
    press_origin: Option<glam::Vec2>,
}

impl App {
    fn new(m_box: usize, m_sphere: usize) -> Self {
        let identity = glam::Mat4::IDENTITY.to_cols_array_2d();
        Self {
            mode: Mode::Primitives,
            camera: Camera {
                distance: 10.0,
                orientation: glam::Quat::from_rotation_y(0.5)
                    * glam::Quat::from_rotation_x(-0.3),
                ..Camera::default()
            },
            ctrl_primitives: OrbitCameraController::viewport_primitives(),
            ctrl_all: OrbitCameraController::viewport_all(),
            manip: ManipulationController::new(),
            sphere_model: identity,
            snapshot: identity,
            m_box,
            m_sphere,
            cursor_viewport: None,
            cursor_prev: None,
            left_held: false,
            drag_started: false,
            clicked: false,
            press_origin: None,
        }
    }
}

fn egui_key_to_keycode(key: egui::Key) -> Option<KeyCode> {
    match key {
        egui::Key::A => Some(KeyCode::A),
        egui::Key::D => Some(KeyCode::D),
        egui::Key::E => Some(KeyCode::E),
        egui::Key::F => Some(KeyCode::F),
        egui::Key::G => Some(KeyCode::G),
        egui::Key::Q => Some(KeyCode::Q),
        egui::Key::R => Some(KeyCode::R),
        egui::Key::S => Some(KeyCode::S),
        egui::Key::W => Some(KeyCode::W),
        egui::Key::X => Some(KeyCode::X),
        egui::Key::Y => Some(KeyCode::Y),
        egui::Key::Z => Some(KeyCode::Z),
        egui::Key::Tab       => Some(KeyCode::Tab),
        egui::Key::Enter     => Some(KeyCode::Enter),
        egui::Key::Escape    => Some(KeyCode::Escape),
        egui::Key::Backspace => Some(KeyCode::Backspace),
        egui::Key::Backtick  => Some(KeyCode::Backtick),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Mode switch: consume 1/2 before any controller sees them.
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Key { key, pressed: true, repeat: false, .. } = event {
                    let next = match key {
                        egui::Key::Num1 => Some(Mode::Primitives),
                        egui::Key::Num2 => Some(Mode::All),
                        _ => None,
                    };
                    if let Some(m) = next {
                        if m != self.mode {
                            // Flush held state in the outgoing controller.
                            match self.mode {
                                Mode::Primitives => self.ctrl_primitives.push_event(ViewportEvent::FocusLost),
                                Mode::All        => self.ctrl_all.push_event(ViewportEvent::FocusLost),
                            }
                            self.mode = m;
                            self.manip.reset();
                        }
                    }
                }
            }
        });

        // ---- Mode bar ----
        egui::TopBottomPanel::top("mode_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, Mode::Primitives, "1 · Camera only");
                ui.selectable_value(&mut self.mode, Mode::All,        "2 · Manipulation");
                ui.separator();
                ui.small(match self.mode {
                    Mode::Primitives =>
                        "Left/Middle drag: orbit  ·  Right drag: pan  ·  Scroll: zoom  ·  Ctrl+Scroll: orbit  ·  Shift+Scroll: pan",
                    Mode::All =>
                        "Ctrl+Scroll: orbit  ·  Right drag: pan  ·  Scroll: zoom  ·  G move  R rotate  S scale  ·  X/Y/Z: constrain  ·  Shift+X/Y/Z: exclude  ·  Enter/Esc: confirm/cancel",
                });
            });
        });

        // ---- Viewport ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.drag_started = false;
            self.clicked = false;

            let vp_ctx = ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            };
            match self.mode {
                Mode::Primitives => self.ctrl_primitives.begin_frame(vp_ctx),
                Mode::All        => self.ctrl_all.begin_frame(vp_ctx),
            }

            // ---- Collect events into a Vec, then push outside the closure ----
            let mut vp_events: Vec<ViewportEvent> = Vec::new();
            let manip_active = self.manip.is_active();

            ui.input(|i| {
                vp_events.push(ViewportEvent::ModifiersChanged(viewport_lib::Modifiers {
                    alt:  i.modifiers.alt,
                    shift: i.modifiers.shift,
                    ctrl: i.modifiers.command,
                }));

                let local_pos = i.pointer.interact_pos().map(|p| {
                    glam::Vec2::new(p.x - rect.left(), p.y - rect.top())
                });
                self.cursor_prev = self.cursor_viewport;
                self.cursor_viewport = local_pos;
                if let Some(pos) = local_pos {
                    vp_events.push(ViewportEvent::PointerMoved { position: pos });
                }

                for event in &i.events {
                    match event {
                        egui::Event::Key { key, pressed, repeat, .. } => {
                            if matches!(key, egui::Key::Num1 | egui::Key::Num2) {
                                continue; // consumed above
                            }
                            if let Some(kc) = egui_key_to_keycode(*key) {
                                vp_events.push(ViewportEvent::Key {
                                    key: kc,
                                    state: if *pressed {
                                        ButtonState::Pressed
                                    } else {
                                        ButtonState::Released
                                    },
                                    repeat: *repeat,
                                });
                            }
                        }
                        egui::Event::PointerButton { button, pressed, .. } => {
                            let vp_button = match button {
                                egui::PointerButton::Primary   => viewport_lib::MouseButton::Left,
                                egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
                                egui::PointerButton::Middle    => viewport_lib::MouseButton::Middle,
                                _ => continue,
                            };
                            vp_events.push(ViewportEvent::MouseButton {
                                button: vp_button,
                                state: if *pressed {
                                    ButtonState::Pressed
                                } else {
                                    ButtonState::Released
                                },
                            });
                            if *button == egui::PointerButton::Primary {
                                if *pressed {
                                    self.left_held = true;
                                    self.press_origin = self.cursor_viewport;
                                    self.drag_started = true;
                                } else {
                                    let is_click = self
                                        .press_origin
                                        .zip(self.cursor_viewport)
                                        .map(|(o, c)| (c - o).length() < 5.0)
                                        .unwrap_or(false);
                                    if is_click {
                                        self.clicked = true;
                                    }
                                    self.left_held = false;
                                    self.press_origin = None;
                                }
                            }
                        }
                        egui::Event::MouseWheel { delta, .. } => {
                            vp_events.push(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units: ScrollUnits::Pixels,
                            });
                        }
                        egui::Event::Text(text) if manip_active => {
                            for c in text.chars() {
                                vp_events.push(ViewportEvent::Character(c));
                            }
                        }
                        _ => {}
                    }
                }
            });

            // Push collected events to the active controller.
            let ctrl = match self.mode {
                Mode::Primitives => &mut self.ctrl_primitives,
                Mode::All        => &mut self.ctrl_all,
            };
            for e in vp_events {
                ctrl.push_event(e);
            }

            // ---- Camera + scene ----
            let w = rect.width();
            let h = rect.height();
            let pointer_delta = self
                .cursor_viewport
                .zip(self.cursor_prev)
                .map(|(c, p)| c - p)
                .unwrap_or(glam::Vec2::ZERO);

            let scene_items: Vec<SceneRenderItem> = match self.mode {
                // ------------------------------------------------------------------
                // Mode 1: camera only — three boxes
                // ------------------------------------------------------------------
                Mode::Primitives => {
                    self.ctrl_primitives.apply_to_camera(&mut self.camera);

                    [
                        ([-2.0f32, 0.0, 0.0], [0.4f32, 0.6, 0.9]),
                        ([ 0.0,    0.0, 0.0], [0.9,    0.5, 0.2]),
                        ([ 2.0,    0.0, 0.0], [0.3,    0.8, 0.4]),
                    ]
                    .iter()
                    .map(|&(pos, color)| {
                        let mut item = SceneRenderItem::default();
                        item.mesh_index = self.m_box;
                        item.model = glam::Mat4::from_translation(glam::Vec3::from(pos))
                            .to_cols_array_2d();
                        item.material = Material { base_color: color, ..Material::default() };
                        item
                    })
                    .collect()
                }

                // ------------------------------------------------------------------
                // Mode 2: full controls — one sphere, always selected
                // ------------------------------------------------------------------
                Mode::All => {
                    let sphere_pos = glam::Vec3::new(
                        self.sphere_model[3][0],
                        self.sphere_model[3][1],
                        self.sphere_model[3][2],
                    );

                    let manip_ctx = ManipulationContext {
                        camera: self.camera.clone(),
                        viewport_size: glam::Vec2::new(w, h),
                        cursor_viewport: self.cursor_viewport,
                        pointer_delta,
                        selection_center: Some(sphere_pos),
                        gizmo: None,
                        drag_started: self.drag_started,
                        dragging: self.left_held,
                        clicked: self.clicked,
                    };

                    let action_frame = if self.manip.is_active() {
                        self.ctrl_all.resolve()
                    } else {
                        self.ctrl_all.apply_to_camera(&mut self.camera)
                    };

                    match self.manip.update(&action_frame, manip_ctx) {
                        ManipResult::Update(delta) => {
                            let current = glam::Mat4::from_cols_array_2d(&self.sphere_model);
                            let delta_mat = glam::Mat4::from_scale_rotation_translation(
                                delta.scale,
                                delta.rotation,
                                delta.translation,
                            );
                            self.sphere_model = (delta_mat * current).to_cols_array_2d();
                        }
                        ManipResult::Cancel | ManipResult::ConstraintChanged => {
                            self.sphere_model = self.snapshot;
                        }
                        ManipResult::Commit => {}
                        ManipResult::None => {
                            self.snapshot = self.sphere_model;
                        }
                    }

                    let mut item = SceneRenderItem::default();
                    item.mesh_index = self.m_sphere;
                    item.model = self.sphere_model;
                    item.material = Material { base_color: [0.4, 0.7, 1.0], ..Material::default() };
                    item.selected = true;
                    vec![item]
                }
            };

            self.camera.set_aspect_ratio(w, h);

            let mut frame_data = FrameData::new(
                CameraFrame::from_camera(&self.camera, [w, h]),
                SceneFrame::from_surface_items(scene_items),
            );
            frame_data.effects.lighting = LightingSettings::default();
            frame_data.viewport.show_grid = true;
            frame_data.viewport.show_axes_indicator = true;

            ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
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
