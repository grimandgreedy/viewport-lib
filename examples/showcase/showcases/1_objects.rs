//! Objects and manipulation: a handful of primitives you can click to select
//! and move/rotate/scale, with a camera that toggles between orbit and
//! first-person on the backtick (`) key.
//!
//! This is the whole showcase in one file: it owns both camera controllers, the
//! selection, and the manipulation-apply logic. The host only feeds it events
//! and renders the frame it assembles.

use eframe::egui;
use glam::{Mat4, Vec3};
use viewport_lib::{
    Action, FirstPersonCameraController, ManipResult, Material, NodeId, OrbitCameraController,
    PickMask, ViewportContext, primitives,
};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

#[derive(Clone, Copy, PartialEq)]
enum CameraMode {
    Orbit,
    FirstPerson,
}

pub struct ObjectsShowcase {
    orbit: OrbitCameraController,
    fp: FirstPersonCameraController,
    mode: CameraMode,
    /// First-person eye position, moved by WASD; the FP controller only looks.
    eye: Vec3,
    move_speed: f32,
    selected: Option<NodeId>,
    /// Selected node transform captured while idle, for cancel/constraint restore.
    idle_transform: Option<Mat4>,
    /// Set by the viewport toggle button; consumed next `update` like a ` press.
    toggle_requested: bool,
}

impl ObjectsShowcase {
    pub fn new() -> Self {
        Self {
            orbit: OrbitCameraController::viewport_all(),
            fp: FirstPersonCameraController::new(
                FirstPersonCameraController::DEFAULT_SENSITIVITY,
                FirstPersonCameraController::DEFAULT_PITCH_CLAMP,
            ),
            mode: CameraMode::Orbit,
            eye: Vec3::ZERO,
            move_speed: 6.0,
            selected: None,
            idle_transform: None,
            toggle_requested: false,
        }
    }
}

impl Showcase for ObjectsShowcase {
    fn name(&self) -> &str {
        "Objects & manipulation"
    }

    fn setup(&mut self, ctx: &mut SetupCtx) {
        let cube = ctx
            .session
            .resources_mut()
            .upload_mesh_data(ctx.device, &primitives::cube(1.0))
            .unwrap();
        let sphere = ctx
            .session
            .resources_mut()
            .upload_mesh_data(ctx.device, &primitives::sphere(0.7, 24, 12))
            .unwrap();
        let torus = ctx
            .session
            .resources_mut()
            .upload_mesh_data(ctx.device, &primitives::torus(0.6, 0.22, 32, 16))
            .unwrap();

        // Z-up: lay a few objects out on the ground plane.
        let objects: [(_, Vec3, [f32; 3]); 5] = [
            (cube, Vec3::new(-3.0, 0.0, 0.5), [0.85, 0.35, 0.30]),
            (sphere, Vec3::new(-1.2, 0.5, 0.7), [0.90, 0.70, 0.25]),
            (torus, Vec3::new(0.8, -0.6, 0.6), [0.35, 0.75, 0.45]),
            (cube, Vec3::new(2.4, 0.8, 0.5), [0.35, 0.55, 0.85]),
            (sphere, Vec3::new(3.6, -0.8, 0.7), [0.65, 0.45, 0.80]),
        ];
        for (mesh, pos, colour) in objects {
            ctx.session.scene_mut().add(
                Some(mesh),
                Mat4::from_translation(pos),
                Material::from_colour(colour),
            );
        }

        // Outline the selected object so picking is visible (default white).
        ctx.session
            .set_selection_outline(true, [1.0, 1.0, 1.0, 1.0], 2.0);

        // Show the X-Y plane grid at Z=0 for spatial reference.
        let vp = ctx.session.viewport_frame_mut();
        vp.show_grid = true;
        vp.grid_cell_size = 1.0;

        ctx.session.camera_mut().distance = 12.0;
        self.mode = CameraMode::Orbit;
        self.selected = None;
        self.idle_transform = None;
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        // Read everything off the context before borrowing the session mutably.
        let toggle = ctx.key_pressed(egui::Key::Backtick) || self.toggle_requested;
        self.toggle_requested = false;
        // Arrow keys drive first-person movement alongside WASD.
        let arrow_fwd = ctx.key_down(egui::Key::ArrowUp);
        let arrow_back = ctx.key_down(egui::Key::ArrowDown);
        let arrow_left = ctx.key_down(egui::Key::ArrowLeft);
        let arrow_right = ctx.key_down(egui::Key::ArrowRight);
        let dt = ctx.dt;
        let view_ctx = ViewportContext {
            hovered: ctx.hovered,
            focused: ctx.focused,
            viewport_size: ctx.viewport_size,
        };
        let session = &mut *ctx.session;

        // Resolve input once so the pointer/action state is available before we
        // drive the camera (which resolves again, harmlessly).
        let action = session.resolve().clone();

        // Backtick toggles the camera controller, syncing so the view does not jump.
        if toggle {
            self.mode = match self.mode {
                CameraMode::Orbit => {
                    self.fp.sync_from_camera(session.camera());
                    self.eye = session.camera().eye_position();
                    CameraMode::FirstPerson
                }
                CameraMode::FirstPerson => {
                    // Hand the orbit controller a sane centre ahead of the eye.
                    let eye = session.camera().eye_position();
                    let aim = self.fp.aim_dir();
                    let camera = session.camera_mut();
                    camera.center = eye + aim * 8.0;
                    camera.distance = 8.0;
                    CameraMode::Orbit
                }
            };
        }

        // Click to select (only when not mid-manipulation).
        if !session.is_manipulating() && action.pointer.clicked {
            if let Some(cursor) = action.pointer.cursor {
                match session.pick(cursor, PickMask::OBJECT) {
                    Some(hit) => {
                        session.selection_mut().select_one(hit.id);
                        self.selected = Some(hit.id);
                    }
                    None => {
                        session.selection_mut().clear();
                        self.selected = None;
                    }
                }
                self.idle_transform = None;
            }
        }

        // Drive the camera and assemble the frame.
        match self.mode {
            CameraMode::Orbit => {
                session.update_orbit(&mut self.orbit);
            }
            CameraMode::FirstPerson => {
                // A manipulation drag owns the pointer; do not also look/move then.
                if !session.is_manipulating() {
                    let mut dir = Vec3::ZERO;
                    if action.is_active(Action::FlyForward) || arrow_fwd {
                        dir += self.fp.forward_dir();
                    }
                    if action.is_active(Action::FlyBackward) || arrow_back {
                        dir -= self.fp.forward_dir();
                    }
                    if action.is_active(Action::FlyRight) || arrow_right {
                        dir += self.fp.right_dir();
                    }
                    if action.is_active(Action::FlyLeft) || arrow_left {
                        dir -= self.fp.right_dir();
                    }
                    if action.is_active(Action::FlyUp) {
                        dir += Vec3::Z;
                    }
                    if action.is_active(Action::FlyDown) {
                        dir -= Vec3::Z;
                    }
                    let speed = if action.is_active(Action::FlySpeedBoost) {
                        self.move_speed * 3.0
                    } else {
                        self.move_speed
                    };
                    if dir != Vec3::ZERO {
                        self.eye += dir.normalize() * speed * dt;
                    }
                    self.fp.apply(session.camera_mut(), &action, self.eye);
                }
                session.frame(view_ctx);
            }
        }

        // Apply the manipulation result to the selected node.
        match session.last_manip() {
            ManipResult::Update(delta) => {
                // Rotate and scale pivot around the session center (the object's
                // own position at session start), not the world origin, so the
                // object turns and grows in place. Translation just adds.
                let center = session
                    .manip_state()
                    .map(|s| s.center)
                    .unwrap_or(Vec3::ZERO);
                if let Some(id) = self.selected {
                    let current = session
                        .scene()
                        .node(id)
                        .map(|n| n.local_transform())
                        .unwrap_or(Mat4::IDENTITY);
                    let to_pivot = Mat4::from_translation(-center);
                    let from_pivot = Mat4::from_translation(center);
                    let rot = Mat4::from_quat(delta.rotation);
                    let scale = Mat4::from_scale(delta.scale);
                    let translate = Mat4::from_translation(delta.translation);
                    let new_transform =
                        translate * from_pivot * rot * scale * to_pivot * current;
                    session.scene_mut().set_local_transform(id, new_transform);
                }
            }
            ManipResult::Cancel | ManipResult::ConstraintChanged => {
                if let (Some(id), Some(base)) = (self.selected, self.idle_transform) {
                    session.scene_mut().set_local_transform(id, base);
                }
            }
            ManipResult::None => {
                // Idle: snapshot the selected node so cancel/constraint can restore it.
                self.idle_transform = self
                    .selected
                    .and_then(|id| session.scene().node(id).map(|n| n.local_transform()));
            }
            _ => {}
        }
    }

    fn description(&self) -> &str {
        "A handful of primitives you can click to select, then grab, rotate, or \
         scale. The camera toggles between orbit and first-person."
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Click an object to select it.");
        ui.separator();
        ui.strong("Manipulate");
        ui.label("G / R / S: grab / rotate / scale");
        ui.label("X / Y / Z: constrain to axis");
        ui.label("Enter or click: confirm    Esc: cancel");
        ui.separator();
        ui.strong("Camera");
        ui.label("` or the top-right toggle: orbit <-> fly");
        ui.label("WASD or arrow keys: move (first person)");
    }

    fn viewport_overlay(&mut self, ui: &mut egui::Ui) {
        let active = match self.mode {
            CameraMode::Orbit => 0,
            CameraMode::FirstPerson => 1,
        };
        // Two modes, so any pick is a toggle; route it through the same request
        // flag the ` key uses.
        if crate::ui::segmented(ui, active, &["orbit", "fly"]).is_some() {
            self.toggle_requested = true;
        }
    }
}
