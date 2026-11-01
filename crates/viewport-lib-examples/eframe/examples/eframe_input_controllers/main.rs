//! Demonstrates the two built-in input presets.
//!
//! Cmd+[ : Camera only (`ViewportPrimitives`):
//!   Left/Middle drag -> Orbit  |  Right drag -> Pan  |  Scroll -> Zoom
//!   Ctrl+Scroll -> Orbit (two-axis)  |  Shift+Scroll -> Pan (two-axis)
//!
//! Cmd+] : Full controls (`ViewportAll`) + the `viewport-lib-gizmo` transform gizmo:
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
use viewport_lib::Colour;
use viewport_lib::input::adapters::{AdapterCtx, egui as vpl_egui};
pub use viewport_lib_examples_eframe::eframe;

use crate::eframe::{egui, wgpu};
use viewport_lib_gizmo::{
    FrameInput as GizmoInput, GizmoDriver, GizmoStyle, ManipResult, PivotMode, Selection,
    TransformDelta, build_numeric_label, gizmo_centre_for_pivot,
};
use vpl::{
    Action, BindingPreset, Camera, CameraFrame, FirstPersonCameraController, FrameData, KeyCode,
    LightingSettings, MeshId, OffscreenViewportTarget, OrbitCameraController, PickId, SceneFrame,
    SceneRenderItem, ThirdPersonCameraController, ViewportContext, ViewportEvent, ViewportInput,
    ViewportRenderer, primitives, viewport_default_bindings, wish_xy_from_actions,
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
        viewport_lib_examples_eframe::native_options([900.0, 600.0]),
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

    // Mode 1: the Viewer binding scheme.
    ctrl_primitives: OrbitCameraController,
    input_primitives: ViewportInput,

    // Mode 2: the Default binding scheme.
    ctrl_all: OrbitCameraController,
    input_all: ViewportInput,

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
    /// The transform gizmo: hover, mode, pivot, and the G/R/S session.
    gizmo: GizmoDriver,
    gizmo_style: GizmoStyle,
    gizmo_centre: Option<glam::Vec3>,

    /// Increments whenever any `obj.model` changes. Passed as `SceneFrame::generation`
    /// so the renderer knows to re-upload instance transforms.
    scene_generation: u64,
    /// Increments whenever `selection` changes. Passed as `InteractionFrame::selection_generation`
    /// so the renderer knows to rebuild the selection outline.
    selection_generation: u64,

    // Per-frame pointer state
    cursor_viewport: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,
    shift_held: bool,

    // GPU picking
    /// Cursor passed to `pick_scene_gpu` in the current frame's prepare callback.
    pick_cursor: Option<glam::Vec2>,
    /// Result written by prepare, drained at the start of the next frame.
    pick_result: Arc<Mutex<Option<u64>>>,
}

impl App {
    fn new(renderer: ViewportRenderer, m_box: MeshId, m_sphere: MeshId, m_capsule: MeshId) -> Self {
        Self {
            renderer,
            target: None,
            mode: Mode::Primitives,
            camera: Camera::orbit(glam::DVec3::ZERO, 12.0, Camera::default().orientation()),
            ctrl_primitives: OrbitCameraController::new_stateless(),
            input_primitives: ViewportInput::from_preset(BindingPreset::Viewer),
            ctrl_all: OrbitCameraController::new_stateless(),
            input_all: ViewportInput::from_preset(BindingPreset::Default),
            // Resolver for the play modes: viewport_all provides the WASD/Fly
            // movement actions. Look is driven from raw pointer motion in the
            // update loop, so it follows the mouse without holding a button.
            input_play: ViewportInput::new(viewport_default_bindings()),
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
            gizmo: GizmoDriver::new(),
            gizmo_style: GizmoStyle::default(),
            gizmo_centre: None,
            scene_generation: 1,
            selection_generation: 1,
            cursor_viewport: None,
            cursor_prev: None,
            shift_held: false,
            pick_cursor: None,
            pick_result: Arc::new(Mutex::new(None)),
        }
    }

    /// Apply a `TransformDelta` to all selected objects.
    ///
    /// The delta is the whole change since the session began, so each object
    /// starts again from its snapshot. Rotation and scale pivot about the
    /// session's centre, or about each object's own origin under
    /// `IndividualOrigins`. Typed values replace the dragged ones per axis, read
    /// relative to the start pose; rotation is typed in degrees.
    fn apply_delta(&mut self, delta: TransformDelta, centre: glam::Vec3) {
        let pick = |typed: [Option<f32>; 3], dragged: glam::Vec3| {
            glam::Vec3::new(
                typed[0].unwrap_or(dragged.x),
                typed[1].unwrap_or(dragged.y),
                typed[2].unwrap_or(dragged.z),
            )
        };
        let translation = pick(delta.position_override, delta.translation);
        let scale = pick(delta.scale_override, delta.scale);
        let rotation = if delta.rotation_override.iter().any(Option::is_some) {
            let [x, y, z] = delta
                .rotation_override
                .map(|v| v.unwrap_or(0.0).to_radians());
            glam::Quat::from_euler(glam::EulerRot::XYZ, x, y, z)
        } else {
            delta.rotation
        };

        let individual = self.gizmo.pivot_mode() == PivotMode::IndividualOrigins;
        let rot_mat = glam::Mat4::from_quat(rotation);
        let scale_mat = glam::Mat4::from_scale(scale);
        let translate_mat = glam::Mat4::from_translation(translation);

        for id in self.selection.iter().copied().collect::<Vec<_>>() {
            let obj = &mut self.objects[id as usize];
            let pivot = if individual {
                obj.snapshot.w_axis.truncate()
            } else {
                centre
            };
            let to_pivot = glam::Mat4::from_translation(-pivot);
            let from_pivot = glam::Mat4::from_translation(pivot);
            obj.model = translate_mat * from_pivot * rot_mat * scale_mat * to_pivot * obj.snapshot;
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

    fn recompute_gizmo_centre(&mut self) {
        self.gizmo_centre =
            gizmo_centre_for_pivot(&self.gizmo.pivot_mode(), &self.selection, |id| {
                Some(self.objects[id as usize].position())
            });
    }

    /// Drop a session in progress when the mode changes, putting the selection
    /// back where it started. The gizmo keeps its mode and pivot.
    fn end_gizmo_session(&mut self) {
        if self.gizmo.is_active() {
            self.restore_snapshots();
        }
        let (mode, pivot) = (self.gizmo.mode(), self.gizmo.pivot_mode());
        self.gizmo = GizmoDriver::new();
        self.gizmo.set_mode(mode);
        self.gizmo.set_pivot_mode(pivot);
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, eframe_frame: &mut eframe::Frame) {
        // Drain GPU pick result from the previous frame's prepare callback.
        // Extract while holding the lock, then drop the guard before calling
        // &mut self methods (recompute_gizmo_centre requires exclusive access).
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
                    self.recompute_gizmo_centre();
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
                    self.recompute_gizmo_centre();
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
                            self.end_gizmo_session();
                        }
                        egui::Key::CloseBracket if cmd => {
                            self.mode = self.mode.next();
                            self.end_gizmo_session();
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
                            ui.small(format!("Pivot: {}", self.gizmo.pivot_mode().label()));
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

                let ownership =
                    vpl_egui::ownership(&response, ui.ctx(), rect, self.gizmo.is_active());
                let vp_ctx = ViewportContext::new(ownership, [rect.width(), rect.height()]);
                match self.mode {
                    Mode::Primitives => self.input_primitives.begin_frame(vp_ctx),
                    Mode::All => self.input_all.begin_frame(vp_ctx),
                    Mode::FirstPerson | Mode::ThirdPerson => self.input_play.begin_frame(vp_ctx),
                }
                // Pan reads the viewport height off whichever orbit controller applies.
                self.ctrl_primitives.set_viewport_size(vp_ctx.viewport_size);
                self.ctrl_all.set_viewport_size(vp_ctx.viewport_size);

                let actx = AdapterCtx::new(1.0, glam::Vec2::new(rect.left(), rect.top()));
                let mut translated = Vec::new();
                let mut cmd = false;
                ui.input(|i| {
                    self.shift_held = i.modifiers.shift;
                    cmd = i.modifiers.command;

                    let local_pos = i
                        .pointer
                        .interact_pos()
                        .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
                    self.cursor_prev = self.cursor_viewport;
                    self.cursor_viewport = local_pos;

                    vpl_egui::begin_frame(i, &actx, &mut translated);
                    for event in &i.events {
                        vpl_egui::translate(event, &actx, &mut translated);
                    }
                });

                let mut vp_events: Vec<ViewportEvent> = Vec::new();
                for ev in translated {
                    match ev {
                        // Cmd+[/] are consumed for mode switching above; don't also
                        // forward them as pivot-cycle keys to the controller.
                        ViewportEvent::Key {
                            key: KeyCode::LeftBracket | KeyCode::RightBracket,
                            ..
                        } if cmd => continue,
                        _ => {}
                    }
                    vp_events.push(ev);
                }

                match self.mode {
                    Mode::Primitives => {
                        for e in vp_events {
                            self.input_primitives.push_event(e);
                        }
                    }
                    Mode::All => {
                        for e in vp_events {
                            self.input_all.push_event(e);
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
                        let frame = self.input_primitives.resolve();
                        let dt = ctx.input(|i| i.stable_dt);
                        self.ctrl_primitives.apply(&mut self.camera, &frame, dt);

                        // Render the same objects as Mode 2 so the scene looks
                        // identical across modes (no confusing shape swaps).
                        self.objects
                            .iter()
                            .map(|obj| {
                                let mut item = SceneRenderItem::default();
                                item.mesh_id = obj.mesh;
                                item.model = obj.model.into();
                                // The default material, coloured per item.
                                item.settings.colour =
                                    Some(Colour::from_linear_rgb_array(obj.colour));
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
                                item.model = obj.model.into();
                                item.settings.colour =
                                    Some(Colour::from_linear_rgb_array(obj.colour));
                                item
                            })
                            .collect();

                        // Third-person: draw a player capsule standing on the
                        // body position (the controllers treat body_pos as feet).
                        if self.mode == Mode::ThirdPerson {
                            let mut marker = SceneRenderItem::default();
                            marker.mesh_id = self.body_mesh;
                            let stand = self.body_pos + glam::Vec3::Z * (CAPSULE_HEIGHT * 0.5);
                            marker.model = glam::Mat4::from_translation(stand).into();
                            marker.settings.colour = Some(Colour::linear_rgb(0.85, 0.2, 0.3));
                            items.push(marker);
                        }
                        items
                    }

                    // ---------------------------------------------------------------
                    // Mode 2: full controls : three selectable objects
                    // ---------------------------------------------------------------
                    Mode::All => {
                        // One resolve for the whole frame; the camera holds still
                        // while a gizmo session owns the pointer.
                        let action_frame = self.input_all.resolve();
                        if !self.gizmo.is_active() {
                            let dt = ctx.input(|i| i.stable_dt);
                            self.ctrl_all.apply(&mut self.camera, &action_frame, dt);
                        }

                        // Tab cycles the gizmo mode, [ and ] the pivot. The driver
                        // ignores both while a session runs, so Tab then steps
                        // the numeric field instead.
                        if action_frame.is_active(Action::CycleGizmoMode) {
                            self.gizmo.cycle_mode();
                        }
                        if action_frame.is_active(Action::CyclePivotModeForward) {
                            self.gizmo.cycle_pivot_forward();
                        } else if action_frame.is_active(Action::CyclePivotModeBackward) {
                            self.gizmo.cycle_pivot_backward();
                        }
                        self.recompute_gizmo_centre();

                        let input = GizmoInput::from_action(
                            &action_frame,
                            &self.camera,
                            viewport_size,
                            self.gizmo_centre,
                        );
                        let mut clicked = action_frame.pointer.clicked;
                        match self.gizmo.update(input) {
                            ManipResult::Update(delta) => {
                                if let Some(state) = self.gizmo.state() {
                                    self.apply_delta(delta, state.centre);
                                }
                                self.recompute_gizmo_centre();
                            }
                            ManipResult::Cancel | ManipResult::ConstraintChanged => {
                                self.restore_snapshots();
                                self.recompute_gizmo_centre();
                            }
                            ManipResult::Commit => {
                                self.save_snapshots();
                                self.recompute_gizmo_centre();
                                // A click that confirmed the session is not a
                                // selection click.
                                clicked = false;
                            }
                            // Idle: keep the snapshots current so G/R/S starts clean.
                            ManipResult::None if !self.gizmo.is_active() => self.save_snapshots(),
                            _ => {}
                        }

                        // Click-to-select: schedule a GPU pick; result arrives next frame.
                        if clicked && !self.gizmo.is_active() {
                            self.pick_cursor = self.cursor_viewport;
                        }

                        // Build scene render items.
                        self.objects
                            .iter()
                            .enumerate()
                            .map(|(i, obj)| {
                                let mut item = SceneRenderItem::default();
                                item.mesh_id = obj.mesh;
                                item.model = obj.model.into();
                                item.settings.colour =
                                    Some(Colour::from_linear_rgb_array(obj.colour));
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

                // Gizmo and readout (Mode 2 only), drawn as overlays.
                if self.mode == Mode::All {
                    frame_data.interaction.outline_selected = true;
                    frame_data.interaction.selection_generation = self.selection_generation;
                    let overlays = &mut frame_data.overlays;
                    self.gizmo.build_view_overlays(
                        &self.gizmo_style,
                        &self.camera,
                        viewport_size,
                        true,
                        &mut overlays.shapes,
                        &mut overlays.polylines,
                    );
                    if let (Some(text), Some(centre)) =
                        (self.gizmo.readout(), self.gizmo.gizmo_centre())
                    {
                        overlays
                            .labels
                            .push(build_numeric_label(&self.gizmo_style, &text, centre));
                        ctx.request_repaint();
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
                if self
                    .target
                    .as_ref()
                    .map_or(true, |t| t.inner.size() != size_px)
                {
                    if let Some(old) = self.target.take() {
                        rs.renderer.write().free_texture(&old.id);
                    }
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
                    let hit =
                        self.renderer
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

                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            });
    }
}
