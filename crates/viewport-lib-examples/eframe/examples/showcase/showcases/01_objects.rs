//! Objects and manipulation: a handful of primitives you can click to select
//! and move/rotate/scale with the `viewport-lib-gizmo` transform gizmo. The
//! camera (orbit/fly) is the host's shared rig; this file owns the selection,
//! the gizmo, and applying its result.

use crate::eframe::egui;
use glam::{Mat4, Quat, Vec2, Vec3};
use viewport_lib as vpl;
use viewport_lib::Colour;
use viewport_lib_gizmo::actions;
use viewport_lib_gizmo::{
    FrameInput as GizmoInput, GizmoAxis, GizmoDriver, GizmoStyle, ManipResult, TransformDelta,
    build_numeric_label,
};
use vpl::{Material, NodeId, PickMask, primitives};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

pub struct ObjectsShowcase {
    selected: Option<NodeId>,
    /// Selected node transform when the session began, for applying the delta
    /// and for cancel/constraint restore. Kept current while idle.
    start_transform: Option<Mat4>,
    gizmo: GizmoDriver,
    style: GizmoStyle,
}

impl ObjectsShowcase {
    pub fn new() -> Self {
        Self {
            selected: None,
            start_transform: None,
            gizmo: GizmoDriver::new(),
            style: GizmoStyle::default(),
        }
    }
}

/// Apply a delta (the whole change since the session began) to `start` about
/// `centre`. Typed values replace the dragged ones per axis, read relative to
/// the start pose; rotation is typed in degrees.
fn compose(delta: &TransformDelta, centre: Vec3, start: Mat4) -> Mat4 {
    let pick = |typed: [Option<f32>; 3], dragged: Vec3| {
        Vec3::new(
            typed[0].unwrap_or(dragged.x),
            typed[1].unwrap_or(dragged.y),
            typed[2].unwrap_or(dragged.z),
        )
    };
    let rotation = if delta.rotation_override.iter().any(Option::is_some) {
        let [x, y, z] = delta
            .rotation_override
            .map(|v| v.unwrap_or(0.0).to_radians());
        Quat::from_euler(glam::EulerRot::XYZ, x, y, z)
    } else {
        delta.rotation
    };
    Mat4::from_translation(pick(delta.position_override, delta.translation))
        * Mat4::from_translation(centre)
        * Mat4::from_quat(rotation)
        * Mat4::from_scale(pick(delta.scale_override, delta.scale))
        * Mat4::from_translation(-centre)
        * start
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
            .upload_mesh_data(ctx.device, &primitives::torus(0.6, 0.22, 96, 48))
            .unwrap();

        // Z-up: lay a few objects out on the ground plane.
        let objects: [(_, Vec3, [f32; 3]); 5] = [
            (cube, Vec3::new(-3.0, 0.0, 0.5), [0.65, 0.09, 0.07]),
            (sphere, Vec3::new(-1.2, 0.5, 0.7), [0.72, 0.42, 0.04]),
            (torus, Vec3::new(0.8, -0.6, 0.6), [0.10, 0.50, 0.18]),
            (cube, Vec3::new(2.4, 0.8, 0.5), [0.10, 0.26, 0.68]),
            (sphere, Vec3::new(3.6, -0.8, 0.7), [0.38, 0.12, 0.62]),
        ];
        for (mesh, pos, colour) in objects {
            let material = ctx
                .session
                .resources_mut()
                .upload_material(Material::from_colour(Colour::from_linear_rgb_array(colour)))
                .unwrap();
            ctx.session
                .scene_mut()
                .add(Some(mesh), Mat4::from_translation(pos), material);
        }

        // Outline the selected object so picking is visible (default white).
        ctx.session
            .set_selection_outline(true, [1.0, 1.0, 1.0, 1.0], 2.0);

        // Show the X-Y plane grid at Z=0 for spatial reference. Leave
        // grid_cell_size at 0 so the grid uses adaptive spacing: as the camera
        // zooms out, minor lines fade and the spacing rescales in powers of ten
        // instead of crowding into a solid plane.
        let vp = ctx.session.viewport_frame_mut();
        vp.show_grid = true;

        ctx.session.camera_mut().set_orbit_distance(12.0);
        self.selected = None;
        self.start_transform = None;
        self.gizmo = GizmoDriver::new();
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        let size = Vec2::from(ctx.viewport_size);
        let session = &mut *ctx.session;
        let action = session.resolve().clone();
        // Whether the gizmo had the pointer coming into this frame.
        let was_busy = self.gizmo.is_active() || self.gizmo.hovered_axis() != GizmoAxis::None;

        // Tab cycles the gizmo mode (ignored while a session runs).
        if action.is_active(actions::CYCLE_GIZMO_MODE) {
            self.gizmo.cycle_mode();
        }

        // Drive the gizmo: hover, handle drags, and G/R/S sessions.
        let node = |id| session.scene().node(id);
        let centre = self
            .selected
            .and_then(node)
            .map(|n| n.world_transform().translation.as_vec3());
        let camera = session.camera().clone();
        let result = self
            .gizmo
            .update(GizmoInput::from_action(&action, &camera, size, centre));
        let mut committed = false;
        match result {
            ManipResult::Update(delta) => {
                if let (Some(id), Some(start), Some(state)) =
                    (self.selected, self.start_transform, self.gizmo.state())
                {
                    let transform = compose(&delta, state.centre, start);
                    session.scene_mut().set_local_transform(id, transform);
                }
            }
            ManipResult::Cancel | ManipResult::ConstraintChanged => {
                if let (Some(id), Some(start)) = (self.selected, self.start_transform) {
                    session.scene_mut().set_local_transform(id, start);
                }
            }
            ManipResult::Commit => committed = true,
            _ => {}
        }
        if !self.gizmo.is_active() {
            // Idle: keep the start pose current so the next session begins clean.
            self.start_transform = self
                .selected
                .and_then(|id| session.scene().node(id))
                .map(|n| n.local_transform().to_mat4());
        }

        // Click to select, unless the click went to the gizmo.
        if action.pointer.clicked && !committed && !was_busy && !self.gizmo.is_active() {
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
                self.start_transform = None;
            }
        }

        // Keep the pointer from the camera while a handle is hovered or a
        // session runs, and drop the camera's drag for a press the gizmo took.
        let grabbing = self.gizmo.is_active() || self.gizmo.hovered_axis() != GizmoAxis::None;
        let took_press = action.pointer.pressed && (was_busy || grabbing);
        ctx.report_tool(grabbing, self.gizmo.is_active(), took_press);

        // Shared orbit/fly camera drives the view and assembles the frame.
        ctx.drive_camera();

        // Draw the gizmo and its readout into the assembled frame.
        let session = &mut *ctx.session;
        let camera = session.camera().clone();
        let overlays = &mut session.frame_data_mut().overlays;
        self.gizmo.build_view_overlays(
            &self.style,
            &camera,
            size,
            true,
            &mut overlays.shapes,
            &mut overlays.polylines,
        );
        if let (Some(text), Some(centre)) = (self.gizmo.readout(), self.gizmo.gizmo_centre()) {
            overlays
                .labels
                .push(build_numeric_label(&self.style, &text, centre));
        }
    }

    fn description(&self) -> &str {
        "A handful of primitives you can click to select, then grab, rotate, or \
         scale in place."
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Click an object to select it.");
        ui.separator();
        ui.strong("Manipulate");
        ui.label("Drag a handle, or G / R / S: grab / rotate / scale");
        ui.label("Tab: cycle translate / rotate / scale handles");
        ui.label("X / Y / Z: constrain to axis    Shift+X / Y / Z: exclude");
        ui.label("Digits: type a value");
        ui.label("Enter or click: confirm    Esc: cancel");
    }
}
