//! Objects and manipulation: a handful of primitives you can click to select
//! and move/rotate/scale. The camera (orbit/fly) is the host's shared rig; this
//! file only owns the selection and the manipulation-apply logic.

use eframe::egui;
use glam::{Mat4, Vec3};
use viewport_lib::{ManipResult, Material, NodeId, PickMask, primitives};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

pub struct ObjectsShowcase {
    selected: Option<NodeId>,
    /// Selected node transform captured while idle, for cancel/constraint restore.
    idle_transform: Option<Mat4>,
}

impl ObjectsShowcase {
    pub fn new() -> Self {
        Self {
            selected: None,
            idle_transform: None,
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
            .upload_mesh_data(ctx.device, &primitives::torus(0.6, 0.22, 96, 48))
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

        // Show the X-Y plane grid at Z=0 for spatial reference. Leave
        // grid_cell_size at 0 so the grid uses adaptive spacing: as the camera
        // zooms out, minor lines fade and the spacing rescales in powers of ten
        // instead of crowding into a solid plane.
        let vp = ctx.session.viewport_frame_mut();
        vp.show_grid = true;

        ctx.session.camera_mut().distance = 12.0;
        self.selected = None;
        self.idle_transform = None;
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        // Click to select (only when not mid-manipulation). Scoped so the session
        // borrow ends before we drive the shared camera.
        {
            let session = &mut *ctx.session;
            let action = session.resolve().clone();
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
        }

        // Shared orbit/fly camera drives the view and assembles the frame.
        ctx.drive_camera();

        // Apply the manipulation result to the selected node.
        let session = &mut *ctx.session;
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
                    let new_transform = translate * from_pivot * rot * scale * to_pivot * current;
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
         scale in place."
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Click an object to select it.");
        ui.separator();
        ui.strong("Manipulate");
        ui.label("G / R / S: grab / rotate / scale");
        ui.label("X / Y / Z: constrain to axis");
        ui.label("Enter or click: confirm    Esc: cancel");
    }
}
