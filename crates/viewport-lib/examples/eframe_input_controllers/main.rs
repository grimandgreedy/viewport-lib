//! Demonstrates the two built-in input presets.
//!
//! Cmd+[ : Camera only (`ViewportPrimitives`):
//!   Left/Middle drag -> Orbit  |  Right drag -> Pan  |  Scroll -> Zoom
//!   Ctrl+Scroll -> Orbit (two-axis)  |  Shift+Scroll -> Pan (two-axis)
//!
//! Cmd+] : Full controls (`ViewportAll`) + `ManipulationController`:
//!   Ctrl+Scroll -> Orbit  |  Right drag -> Pan  |  Scroll -> Zoom
//!   Click -> select  |  Shift+Click -> multi-select
//!   G move  R rotate  S scale  |  X/Y/Z constrain  |  Shift+X/Y/Z exclude
//!   Tab -> cycle gizmo mode (translate / rotate / scale)
//!   Type digits / minus / period -> numeric input while G/R/S active
//!   Enter / left-click -> confirm  |  Esc -> cancel
//!   [ -> cycle pivot forward  |  ] -> cycle pivot backward
//!
//! First person / Third person: body-attached cameras following a WASD-driven
//! body on one shared `Camera`. Move the mouse to look, WASD to move; Cmd+[ and
//! Cmd+] cycle through all four modes. Switching modes calls `sync_from_camera`
//! so the view continues instead of snapping.

use std::sync::{Arc, Mutex};
use viewport_lib as vpl;

use eframe::{egui, wgpu};
use vpl::{
    Action, ButtonState, Camera, CameraFrame, FirstPersonCameraController, FrameData, Gizmo,
    GizmoAxis, GizmoInfo, GizmoMode, InteractionFrame, KeyCode, LightingSettings, ManipResult,
    ManipulationContext, ManipulationController, ManipulationKind, Material, MeshId,
    OffscreenViewportTarget, OrbitCameraController, PickId, PivotMode, SceneFrame, SceneRenderItem,
    ScrollUnits, Selection, ThirdPersonCameraController, ViewportContext, ViewportEvent,
    ViewportInput, ViewportRenderer, gizmo_center_for_pivot, primitives, viewport_all_bindings,
    wish_xy_from_actions,
};

/// Body movement speed in world units per second for the character cameras.
const MOVE_SPEED: f32 = 6.0;
/// Eye height above the body origin for the first-person camera.
const EYE_HEIGHT: f32 = 1.7;
/// Height of the player capsule drawn in third-person.
const CAPSULE_HEIGHT: f32 = 1.8;

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Input Controllers",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 600.0]),
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

            // sRGB render format; the offscreen target keeps the encode through egui.
            let mut renderer = ViewportRenderer::new(
                device,
                OffscreenViewportTarget::render_format(rs.target_format),
            );
            let res = renderer.resources_mut();
            let m_box = res
                .upload_mesh_data(device, &primitives::cube(1.0))
                .unwrap();
            let m_sphere = res
                .upload_mesh_data(device, &primitives::sphere(0.75, 32, 16))
                .unwrap();
            let m_capsule = res
                .upload_mesh_data(device, &primitives::capsule(0.4, CAPSULE_HEIGHT, 16, 8))
                .unwrap();

            Ok(Box::new(App::new(renderer, m_box, m_sphere, m_capsule)))
        }),
    )
}

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Camera navigation only : ViewportPrimitives preset.
    Primitives,
    /// Full controls + object manipulation : ViewportAll preset.
    All,
    /// Body-attached first-person camera following a WASD-driven body.
    FirstPerson,
    /// Body-attached third-person orbit-and-boom camera following the body.
    ThirdPerson,
}

impl Mode {
    /// Cycle order for the Cmd+[ / Cmd+] shortcuts.
    const ORDER: [Mode; 4] = [
        Mode::Primitives,
        Mode::All,
        Mode::FirstPerson,
        Mode::ThirdPerson,
    ];

    fn index(self) -> usize {
        Self::ORDER.iter().position(|&m| m == self).unwrap_or(0)
    }

    fn next(self) -> Self {
        Self::ORDER[(self.index() + 1) % Self::ORDER.len()]
    }

    fn prev(self) -> Self {
        Self::ORDER[(self.index() + Self::ORDER.len() - 1) % Self::ORDER.len()]
    }
}

// ---------------------------------------------------------------------------
// Scene object
// ---------------------------------------------------------------------------

struct Object {
    model: glam::Mat4,
    snapshot: glam::Mat4,
    mesh: MeshId,
    colour: [f32; 3],
}

impl Object {
    fn new(pos: glam::Vec3, mesh: MeshId, colour: [f32; 3]) -> Self {
        let model = glam::Mat4::from_translation(pos);
        Self {
            model,
            snapshot: model,
            mesh,
            colour,
        }
    }

    fn position(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.model.w_axis.x,
            self.model.w_axis.y,
            self.model.w_axis.z,
        )
    }

    fn save_snapshot(&mut self) {
        self.snapshot = self.model;
    }

    fn restore_snapshot(&mut self) {
        self.model = self.snapshot;
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

/// The offscreen colour target and its egui texture id, recreated on resize.
struct Target {
    inner: OffscreenViewportTarget,
    id: egui::TextureId,
}

struct App {
    renderer: ViewportRenderer,
    target: Option<Target>,
    mode: Mode,
    camera: Camera,

    // Mode 1
    ctrl_primitives: OrbitCameraController,

    // Mode 2
    ctrl_all: OrbitCameraController,

    // Play modes: one shared resolver, two body-attached appliers.
    input_play: ViewportInput,
    ctrl_fp: FirstPersonCameraController,
    ctrl_tp: ThirdPersonCameraController,
    /// World-space body the character cameras follow (host-owned).
    body_pos: glam::Vec3,
    /// Mesh used to draw the body marker in third-person.
    body_mesh: MeshId,
    /// Previous frame's mode, to detect entry into a play mode for sync.
    prev_mode: Mode,

    objects: Vec<Object>,
    selection: Selection,
    gizmo: Gizmo,
    gizmo_center: Option<glam::Vec3>,
    gizmo_scale: f32,
    manip: ManipulationController,
    active_manip_kind: ManipulationKind,

    /// Increments whenever any `obj.model` changes. Passed as `SceneFrame::generation`
    /// so the renderer knows to re-upload instance transforms.
    scene_generation: u64,
    /// Increments whenever `selection` changes. Passed as `InteractionFrame::selection_generation`
    /// so the renderer knows to rebuild the selection outline.
    selection_generation: u64,

    // Per-frame pointer state
    cursor_viewport: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,
    left_held: bool,
    drag_started: bool,
    clicked: bool,
    shift_held: bool,
    press_origin: Option<glam::Vec2>,

    // GPU picking
    /// Cursor passed to `pick_scene_gpu` in the current frame's prepare callback.
    pick_cursor: Option<glam::Vec2>,
    /// Result written by prepare, drained at the start of the next frame.
    pick_result: Arc<Mutex<Option<u64>>>,
}

impl App {
    fn new(
        renderer: ViewportRenderer,
        m_box: MeshId,
        m_sphere: MeshId,
        m_capsule: MeshId,
    ) -> Self {
        Self {
            renderer,
            target: None,
            mode: Mode::Primitives,
            camera: Camera {
                distance: 12.0,
                ..Camera::default()
            },
            ctrl_primitives: OrbitCameraController::viewport_primitives(),
            ctrl_all: OrbitCameraController::viewport_all(),
            // Resolver for the play modes: viewport_all provides the WASD/Fly
            // movement actions. Look is driven from raw pointer motion in the
            // update loop, so it follows the mouse without holding a button.
            input_play: ViewportInput::new(viewport_all_bindings()),
            ctrl_fp: FirstPersonCameraController::new(
                FirstPersonCameraController::DEFAULT_SENSITIVITY,
                FirstPersonCameraController::DEFAULT_PITCH_CLAMP,
            ),
            ctrl_tp: ThirdPersonCameraController::new(0.005, 0.005),
            body_pos: glam::Vec3::new(0.0, -4.0, 0.0),
            body_mesh: m_capsule,
            prev_mode: Mode::Primitives,
            objects: vec![
                Object::new(glam::Vec3::new(-3.0, 0.0, 0.0), m_box, [0.4, 0.6, 0.9]),
                Object::new(glam::Vec3::new(0.0, 0.0, 0.0), m_sphere, [0.9, 0.5, 0.2]),
                Object::new(glam::Vec3::new(3.0, 0.0, 0.0), m_box, [0.3, 0.8, 0.4]),
            ],
            selection: Selection::new(),
            gizmo: Gizmo::new(),
            gizmo_center: None,
            gizmo_scale: 1.0,
            manip: ManipulationController::new(),
            active_manip_kind: ManipulationKind::Move,
            scene_generation: 1,
            selection_generation: 1,
            cursor_viewport: None,
            cursor_prev: None,
            left_held: false,
            drag_started: false,
            clicked: false,
            shift_held: false,
            press_origin: None,
            pick_cursor: None,
            pick_result: Arc::new(Mutex::new(None)),
        }
    }

    /// Apply a `TransformDelta` to all selected objects.
    ///
    /// Rotation and scale pivot depends on the current `PivotMode`:
    /// - `IndividualOrigins` : each object transforms around its own centre.
    /// - Everything else : all objects transform around the shared `gizmo_center`.
    /// Apply a `TransformDelta` to all selected objects.
    ///
    /// When `position_override` or `scale_override` is set (numeric input mode),
    /// the snapshot is restored first so the override is an absolute offset from
    /// the pre-session state rather than accumulated on top of previous deltas.
    fn apply_delta(&mut self, delta: vpl::TransformDelta) {
        let has_pos_override = delta.position_override.iter().any(|v| v.is_some());
        let has_scale_override = delta.scale_override.iter().any(|v| v.is_some());

        if has_pos_override || has_scale_override {
            self.restore_snapshots();
        }

        let translation = if has_pos_override {
            glam::Vec3::new(
                delta.position_override[0].unwrap_or(0.0),
                delta.position_override[1].unwrap_or(0.0),
                delta.position_override[2].unwrap_or(0.0),
            )
        } else {
            delta.translation
        };

        let scale = if has_scale_override {
            glam::Vec3::new(
                delta.scale_override[0].unwrap_or(1.0),
                delta.scale_override[1].unwrap_or(1.0),
                delta.scale_override[2].unwrap_or(1.0),
            )
        } else {
            delta.scale
        };

        let pivot_mode = self.gizmo.pivot_mode;
        let gizmo_center = self.gizmo_center.unwrap_or(glam::Vec3::ZERO);

        let rot_mat = glam::Mat4::from_quat(delta.rotation);
        let scale_mat = glam::Mat4::from_scale(scale);
        let translate_mat = glam::Mat4::from_translation(translation);

        for id in self.selection.iter().copied().collect::<Vec<_>>() {
            let obj = &mut self.objects[id as usize];

            let pivot = match pivot_mode {
                PivotMode::IndividualOrigins => obj.position(),
                _ => gizmo_center,
            };

            let to_pivot = glam::Mat4::from_translation(-pivot);
            let from_pivot = glam::Mat4::from_translation(pivot);

            // translate * (from_pivot * rot * scale * to_pivot) * old_model
            obj.model = translate_mat * from_pivot * rot_mat * scale_mat * to_pivot * obj.model;
        }
        self.scene_generation += 1;
    }

    fn restore_snapshots(&mut self) {
        for id in self.selection.iter().copied().collect::<Vec<_>>() {
            self.objects[id as usize].restore_snapshot();
        }
        self.scene_generation += 1;
    }

    fn save_snapshots(&mut self) {
        for id in self.selection.iter().copied().collect::<Vec<_>>() {
            self.objects[id as usize].save_snapshot();
        }
    }

    fn recompute_gizmo_center(&mut self) {
        self.gizmo_center = gizmo_center_for_pivot(&self.gizmo.pivot_mode, &self.selection, |id| {
            Some(self.objects[id as usize].position())
        });
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
        egui::Key::Tab => Some(KeyCode::Tab),
        egui::Key::Enter => Some(KeyCode::Enter),
        egui::Key::Escape => Some(KeyCode::Escape),
        egui::Key::Backspace => Some(KeyCode::Backspace),
        egui::Key::Backtick => Some(KeyCode::Backtick),
        egui::Key::Comma => Some(KeyCode::Comma),
        egui::Key::Period => Some(KeyCode::Period),
        egui::Key::OpenBracket => Some(KeyCode::LeftBracket),
        egui::Key::CloseBracket => Some(KeyCode::RightBracket),
        egui::Key::Slash => Some(KeyCode::Slash),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, eframe_frame: &mut eframe::Frame) {
        // Drain GPU pick result from the previous frame's prepare callback.
        // Extract while holding the lock, then drop the guard before calling
        // &mut self methods (recompute_gizmo_center requires exclusive access).
        let pick_raw: Option<u64> = self
            .pick_result
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        if let Some(raw_id) = pick_raw {
            // pick_id is 1-indexed (0 = miss); convert to object index.
            if raw_id == 0 {
                if !self.shift_held {
                    self.selection.clear();
                    self.recompute_gizmo_center();
                    self.selection_generation += 1;
                }
            } else {
                let idx = (raw_id - 1) as usize;
                if idx < self.objects.len() {
                    if self.shift_held {
                        if self.selection.contains(idx as u64) {
                            self.selection.remove(idx as u64);
                        } else {
                            self.selection.add(idx as u64);
                        }
                    } else {
                        self.selection.clear();
                        self.selection.select_one(idx as u64);
                    }
                    self.recompute_gizmo_center();
                    self.selection_generation += 1;
                }
            }
        }

        self.pick_cursor = None;

        // Mode switch: Cmd+[ -> Primitives, Cmd+] -> All.
        // Also intercept Tab so egui does not steal it for focus cycling.
        let mut tab_pressed = false;
        ctx.input(|i| {
            let cmd = i.modifiers.command;
            for event in &i.events {
                if let egui::Event::Key {
                    key,
                    pressed: true,
                    repeat: false,
                    ..
                } = event
                {
                    match key {
                        egui::Key::Tab => tab_pressed = true,
                        egui::Key::OpenBracket if cmd => {
                            self.mode = self.mode.prev();
                            self.manip.reset();
                        }
                        egui::Key::CloseBracket if cmd => {
                            self.mode = self.mode.next();
                            self.manip.reset();
                        }
                        _ => {}
                    }
                }
            }
        });
        if tab_pressed {
            ctx.memory_mut(|mem| mem.move_focus(egui::FocusDirection::None));
        }

        // ---- Mode bar ----
        egui::TopBottomPanel::top("mode_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, Mode::Primitives, "Camera only");
                ui.selectable_value(&mut self.mode, Mode::All, "Manipulation");
                ui.selectable_value(&mut self.mode, Mode::FirstPerson, "First person");
                ui.selectable_value(&mut self.mode, Mode::ThirdPerson, "Third person");
                ui.separator();
                match self.mode {
                    Mode::Primitives => {
                        ui.small("Left/Middle drag: orbit  |  Right drag: pan  |  Scroll: zoom");
                    }
                    Mode::All => {
                        ui.small(
                            "Click: select  |  Shift+Click: multi-select  |  G/R/S: move/rotate/scale  |  Tab: cycle gizmo  |  X/Y/Z: constrain  |  digits: numeric  |  [/]: pivot",
                        );
                        if !self.selection.is_empty() {
                            ui.separator();
                            ui.small(format!("Pivot: {}", self.gizmo.pivot_mode.label()));
                        }
                    }
                    Mode::FirstPerson => {
                        ui.small("Move mouse: look  |  WASD: move the body (eye follows it)");
                    }
                    Mode::ThirdPerson => {
                        ui.small("Move mouse: orbit the body  |  WASD: move the body (red marker)");
                    }
                }
            });
        });

        // ---- Viewport ----
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

                self.drag_started = false;
                self.clicked = false;

                let vp_ctx = ViewportContext {
                    hovered: response.hovered(),
                    // Treat any hover as focus so keyboard shortcuts work without
                    // requiring a separate egui-focus click on the canvas widget.
                    focused: response.hovered() || response.has_focus(),
                    viewport_size: [rect.width(), rect.height()],
                };
                match self.mode {
                    Mode::Primitives => self.ctrl_primitives.begin_frame(vp_ctx),
                    Mode::All => self.ctrl_all.begin_frame(vp_ctx),
                    Mode::FirstPerson | Mode::ThirdPerson => self.input_play.begin_frame(vp_ctx),
                }

                let mut vp_events: Vec<ViewportEvent> = Vec::new();
                let manip_active = self.manip.is_active();

                ui.input(|i| {
                    self.shift_held = i.modifiers.shift;

                    vp_events.push(ViewportEvent::ModifiersChanged(vpl::Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    }));

                    let local_pos = i
                        .pointer
                        .interact_pos()
                        .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
                    self.cursor_prev = self.cursor_viewport;
                    self.cursor_viewport = local_pos;
                    if let Some(pos) = local_pos {
                        vp_events.push(ViewportEvent::PointerMoved { position: pos });
                    }

                    let cmd = i.modifiers.command;

                    for event in &i.events {
                        match event {
                            egui::Event::Key {
                                key,
                                pressed,
                                repeat,
                                ..
                            } => {
                                // Cmd+[/] are consumed for mode switching above; don't
                                // also forward them as pivot-cycle keys to the controller.
                                if cmd
                                    && matches!(
                                        key,
                                        egui::Key::OpenBracket | egui::Key::CloseBracket
                                    )
                                {
                                    continue;
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
                            egui::Event::PointerButton {
                                button, pressed, ..
                            } => {
                                let vp_button = match button {
                                    egui::PointerButton::Primary => vpl::MouseButton::Left,
                                    egui::PointerButton::Secondary => vpl::MouseButton::Right,
                                    egui::PointerButton::Middle => vpl::MouseButton::Middle,
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
                            egui::Event::MouseWheel { unit, delta, .. } => {
                                let units = match unit {
                                    egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                                    egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                                    egui::MouseWheelUnit::Page => ScrollUnits::Pages,
                                };
                                vp_events.push(ViewportEvent::Wheel {
                                    delta: glam::Vec2::new(delta.x, delta.y),
                                    units,
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

                match self.mode {
                    Mode::Primitives => {
                        for e in vp_events {
                            self.ctrl_primitives.push_event(e);
                        }
                    }
                    Mode::All => {
                        for e in vp_events {
                            self.ctrl_all.push_event(e);
                        }
                    }
                    Mode::FirstPerson | Mode::ThirdPerson => {
                        for e in vp_events {
                            self.input_play.push_event(e);
                        }
                    }
                }

                // On entry to a play mode, adopt the current view so the camera
                // continues from where it was instead of snapping.
                if self.mode != self.prev_mode {
                    match self.mode {
                        Mode::FirstPerson => self.ctrl_fp.sync_from_camera(&self.camera),
                        Mode::ThirdPerson => self.ctrl_tp.sync_from_camera(&self.camera),
                        _ => {}
                    }
                }
                self.prev_mode = self.mode;

                let w = rect.width();
                let h = rect.height();
                let viewport_size = glam::Vec2::new(w, h);
                let pointer_delta = self
                    .cursor_viewport
                    .zip(self.cursor_prev)
                    .map(|(c, p)| c - p)
                    .unwrap_or(glam::Vec2::ZERO);

                // ---------------------------------------------------------------
                // Mode 1: camera only : three static boxes
                // ---------------------------------------------------------------
                let scene_items: Vec<SceneRenderItem> = match self.mode {
                    Mode::Primitives => {
                        self.ctrl_primitives.apply_to_camera(&mut self.camera);

                        // Render the same objects as Mode 2 so the scene looks
                        // identical across modes (no confusing shape swaps).
                        self.objects
                            .iter()
                            .map(|obj| {
                                let mut item = SceneRenderItem::default();
                                item.mesh_id = obj.mesh;
                                item.model = obj.model.to_cols_array_2d();
                                item.material = Material::from_colour(obj.colour);
                                item
                            })
                            .collect()
                    }

                    // ---------------------------------------------------------------
                    // Play modes: one resolver, body-attached cameras
                    // ---------------------------------------------------------------
                    Mode::FirstPerson | Mode::ThirdPerson => {
                        // Keep animating so look and movement stay smooth instead
                        // of only advancing on sparse input events.
                        ui.ctx().request_repaint();
                        let dt = ui.input(|i| i.stable_dt);

                        // input_play resolves the WASD/Fly movement actions; look
                        // follows raw pointer motion so the camera turns with the
                        // mouse without holding a button.
                        let mut frame = self.input_play.resolve();
                        frame.navigation.orbit = pointer_delta;

                        let (fwd, right) = if self.mode == Mode::FirstPerson {
                            (self.ctrl_fp.forward_dir(), self.ctrl_fp.right_dir())
                        } else {
                            (self.ctrl_tp.forward_dir(), self.ctrl_tp.right_dir())
                        };
                        let wish = wish_xy_from_actions(&frame, fwd, right);
                        if wish != glam::Vec2::ZERO {
                            self.body_pos += glam::Vec3::new(wish.x, wish.y, 0.0) * MOVE_SPEED * dt;
                            // The capsule marker moved; bump the generation so the
                            // renderer re-uploads its instance transform.
                            self.scene_generation += 1;
                        }

                        if self.mode == Mode::FirstPerson {
                            let eye = self.body_pos + glam::Vec3::Z * EYE_HEIGHT;
                            self.ctrl_fp.apply(&mut self.camera, &frame, eye);
                        } else {
                            self.ctrl_tp.apply(&mut self.camera, &frame, self.body_pos);
                        }

                        let mut items: Vec<SceneRenderItem> = self
                            .objects
                            .iter()
                            .map(|obj| {
                                let mut item = SceneRenderItem::default();
                                item.mesh_id = obj.mesh;
                                item.model = obj.model.to_cols_array_2d();
                                item.material = Material::from_colour(obj.colour);
                                item
                            })
                            .collect();

                        // Third-person: draw a player capsule standing on the
                        // body position (the controllers treat body_pos as feet).
                        if self.mode == Mode::ThirdPerson {
                            let mut marker = SceneRenderItem::default();
                            marker.mesh_id = self.body_mesh;
                            let stand = self.body_pos + glam::Vec3::Z * (CAPSULE_HEIGHT * 0.5);
                            marker.model = glam::Mat4::from_translation(stand).to_cols_array_2d();
                            marker.material = Material::from_colour([0.85, 0.2, 0.3]);
                            items.push(marker);
                        }
                        items
                    }

                    // ---------------------------------------------------------------
                    // Mode 2: full controls : three selectable objects
                    // ---------------------------------------------------------------
                    Mode::All => {
                        let camera_view = self.camera.view_matrix();
                        let camera_proj = self.camera.proj_matrix();
                        let view_proj = camera_proj * camera_view;

                        // Single action frame for the whole frame : resolve() or
                        // apply_to_camera() must only be called once per frame.
                        let action_frame = if self.manip.is_active() {
                            self.ctrl_all.resolve()
                        } else {
                            self.ctrl_all.apply_to_camera(&mut self.camera)
                        };

                        // Tab cycles the gizmo mode when no manipulation is active.
                        if !self.manip.is_active() && action_frame.is_active(Action::CycleGizmoMode)
                        {
                            self.gizmo.mode = match self.gizmo.mode {
                                GizmoMode::Translate => GizmoMode::Rotate,
                                GizmoMode::Rotate => GizmoMode::Scale,
                                GizmoMode::Scale => GizmoMode::Translate,
                                _ => GizmoMode::Translate,
                            };
                        }

                        // Pivot mode cycling : check the action frame we already have.
                        let cycle_fwd = action_frame.is_active(Action::CyclePivotModeForward);
                        let cycle_bwd = action_frame.is_active(Action::CyclePivotModeBackward);
                        if cycle_fwd || cycle_bwd {
                            if cycle_fwd {
                                self.gizmo.cycle_pivot_forward();
                            } else {
                                self.gizmo.cycle_pivot_backward();
                            }
                            self.recompute_gizmo_center();
                            if self.manip.is_active() {
                                self.manip.reset();
                                self.restore_snapshots();
                                if let Some(center) = self.gizmo_center {
                                    self.manip.begin(self.active_manip_kind, center);
                                }
                            }
                        }

                        // Update gizmo hover from the current cursor position.
                        if let (Some(cursor), Some(center)) =
                            (self.cursor_viewport, self.gizmo_center)
                        {
                            if !self.manip.is_active() {
                                let ray_origin = self.camera.eye_position();
                                let ndc_x = (cursor.x / w.max(1.0)) * 2.0 - 1.0;
                                let ndc_y = 1.0 - (cursor.y / h.max(1.0)) * 2.0;
                                let inv_vp = view_proj.inverse();
                                let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
                                let ray_dir = (far - ray_origin).normalize_or(glam::Vec3::NEG_Z);
                                self.gizmo.hovered_axis = self.gizmo.hit_test(
                                    ray_origin,
                                    ray_dir,
                                    center,
                                    self.gizmo_scale,
                                );
                            }
                        }

                        // Build ManipulationContext.
                        let gizmo_info = self.gizmo_center.map(|center| GizmoInfo {
                            center,
                            scale: self.gizmo_scale,
                            orientation: glam::Quat::IDENTITY,
                            mode: self.gizmo.mode,
                        });

                        let manip_ctx = ManipulationContext {
                            camera: self.camera.clone(),
                            viewport_size,
                            cursor_viewport: self.cursor_viewport,
                            pointer_delta,
                            selection_center: self.gizmo_center,
                            gizmo: gizmo_info,
                            drag_started: self.drag_started,
                            dragging: self.left_held,
                            clicked: self.clicked,
                        };

                        match self.manip.update(&action_frame, manip_ctx) {
                            ManipResult::Update(delta) => {
                                // Delta is incremental : accumulate, do NOT restore snapshot.
                                self.apply_delta(delta);
                                self.recompute_gizmo_center();
                            }
                            ManipResult::Cancel | ManipResult::ConstraintChanged => {
                                self.restore_snapshots();
                                self.recompute_gizmo_center();
                            }
                            ManipResult::Commit => {
                                self.save_snapshots();
                                self.recompute_gizmo_center();
                                // If a click confirmed the session, consume it so
                                // click-to-select doesn't fire on the same frame.
                                self.clicked = false;
                            }
                            ManipResult::None => {
                                if self.manip.is_active() {
                                    // Session just started this frame : record its kind.
                                    if let Some(state) = self.manip.state() {
                                        self.active_manip_kind = state.kind;
                                    }
                                } else {
                                    // Idle : keep snapshot current so G/R/S starts clean.
                                    self.save_snapshots();
                                }
                            }
                            _ => {}
                        }

                        // Click-to-select: schedule a GPU pick; result arrives next frame.
                        if self.clicked && !self.manip.is_active() {
                            self.pick_cursor = self.cursor_viewport;
                        }

                        // Recompute gizmo scale each frame so it stays screen-size-stable.
                        if let Some(center) = self.gizmo_center {
                            self.gizmo_scale = vpl::gizmo::compute_gizmo_scale(
                                center,
                                self.camera.eye_position(),
                                self.camera.fov_y,
                                h,
                            );
                        }

                        // Build scene render items.
                        self.objects
                            .iter()
                            .enumerate()
                            .map(|(i, obj)| {
                                let mut item = SceneRenderItem::default();
                                item.mesh_id = obj.mesh;
                                item.model = obj.model.to_cols_array_2d();
                                item.material = Material::from_colour(obj.colour);
                                item.settings.selected = self.selection.contains(i as u64);
                                // pick_id is 1-indexed so 0 can mean "no hit".
                                item.settings.pick_id = PickId((i as u64) + 1);
                                item
                            })
                            .collect()
                    }
                };

                self.camera.set_aspect_ratio(w, h);

                let mut scene_frame = SceneFrame::from_surface_items(scene_items);
                scene_frame.generation = self.scene_generation;

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&self.camera, [w, h])
                        .with_pixels_per_point(ui.ctx().pixels_per_point()),
                    scene_frame,
                );
                frame_data.effects.lighting = LightingSettings::default();
                frame_data.viewport.show_grid = true;
                frame_data.viewport.show_axes_indicator = true;

                // Gizmo rendering (Mode 2 only).
                if self.mode == Mode::All {
                    if let Some(center) = self.gizmo_center {
                        let gizmo_model = glam::Mat4::from_scale_rotation_translation(
                            glam::Vec3::splat(self.gizmo_scale),
                            glam::Quat::IDENTITY,
                            center,
                        );
                        let mut interaction = InteractionFrame::default();
                        interaction.gizmo_model = Some(gizmo_model);
                        interaction.gizmo_mode = self.gizmo.mode;
                        interaction.gizmo_hovered = self.gizmo.hovered_axis;
                        interaction.gizmo_space_orientation = glam::Quat::IDENTITY;
                        interaction.outline_selected = true;
                        interaction.selection_generation = self.selection_generation;
                        frame_data.interaction = interaction;
                    } else {
                        frame_data.interaction.outline_selected = true;
                        frame_data.interaction.selection_generation = self.selection_generation;
                    }
                }

                // Render into the offscreen target (sRGB dual-view keeps the encode
                // through egui's sample), then GPU-pick, then display the image.
                let rs = eframe_frame
                    .wgpu_render_state()
                    .expect("wgpu backend required");
                let ppp = ui.ctx().pixels_per_point();
                let size_px = [
                    (w * ppp).round().max(1.0) as u32,
                    (h * ppp).round().max(1.0) as u32,
                ];
                if self.target.as_ref().map_or(true, |t| t.inner.size() != size_px) {
                    let inner = OffscreenViewportTarget::new(&rs.device, rs.target_format, size_px);
                    let id = rs.renderer.write().register_native_texture(
                        &rs.device,
                        inner.sample_view(),
                        wgpu::FilterMode::Linear,
                    );
                    self.target = Some(Target { inner, id });
                }
                let target = self.target.as_ref().unwrap();
                let render_view = target.inner.render_view();
                let tex_id = target.id;
                let cmd =
                    self.renderer
                        .owned()
                        .render(&rs.device, &rs.queue, render_view, &frame_data);
                rs.queue.submit(std::iter::once(cmd));

                // GPU pick after prepare/render for this frame; drained next frame.
                if let Some(cursor) = self.pick_cursor {
                    let hit = self
                        .renderer
                        .pick_scene_gpu(&rs.device, &rs.queue, cursor, &frame_data);
                    let id = hit.map(|h| h.object_id.0).unwrap_or(0);
                    if let Ok(mut slot) = self.pick_result.lock() {
                        *slot = Some(id);
                    }
                }

                ui.painter().image(
                    tex_id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                // Manipulation status label: shown at the bottom-centre of the
                // viewport while a G/R/S session is active.
                if self.mode == Mode::All {
                    if let Some(ms) = self.manip.state() {
                        let kind_label = match ms.kind {
                            ManipulationKind::Move => "Move",
                            ManipulationKind::Rotate => "Rotate",
                            ManipulationKind::Scale => "Scale",
                        };
                        let axis_label = match ms.axis {
                            Some(GizmoAxis::X) => {
                                if ms.exclude_axis {
                                    " (YZ)"
                                } else {
                                    " (X)"
                                }
                            }
                            Some(GizmoAxis::Y) => {
                                if ms.exclude_axis {
                                    " (XZ)"
                                } else {
                                    " (Y)"
                                }
                            }
                            Some(GizmoAxis::Z) => {
                                if ms.exclude_axis {
                                    " (XY)"
                                } else {
                                    " (Z)"
                                }
                            }
                            _ => "",
                        };
                        let text = if let Some(ref numeric) = ms.numeric_display {
                            format!("{kind_label}{axis_label}: {numeric}")
                        } else {
                            format!("{kind_label}{axis_label}")
                        };
                        let font = egui::FontId::proportional(14.0);
                        let galley = ui
                            .painter()
                            .layout_no_wrap(text, font, egui::Color32::WHITE);
                        let pos =
                            egui::pos2(rect.center().x - galley.size().x / 2.0, rect.max.y - 30.0);
                        let bg = egui::Rect::from_min_size(
                            pos - egui::vec2(6.0, 3.0),
                            galley.size() + egui::vec2(12.0, 6.0),
                        );
                        ui.painter()
                            .rect_filled(bg, 3.0, egui::Color32::from_black_alpha(180));
                        ui.painter().galley(pos, galley, egui::Color32::WHITE);
                        ctx.request_repaint();
                    }
                }

                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            });
    }
}
