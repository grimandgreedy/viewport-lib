//! Showcase 47: Skeletal Animation (CPU Skinning)
//!
//! Demonstrates the skeletal animation substrate: Skeleton, Pose, SkeletonPlugin,
//! and per-frame write_mesh_positions_normals to upload deformed vertex data.
//!
//! Scene: a two-joint arm built from a ring-stack mesh. The lower half is fully
//! weighted to the root joint; the upper half is fully weighted to the child joint.
//! A short transition region blends between them. A timer plugin writes a Pose each
//! frame, rotating the child joint so the arm bends at the joint seam.
//!
//! Per-frame loop:
//!   1. runtime.step() - PoseDriver writes Pose; SkeletonPlugin deforms and emits SkinnedMeshUpdate
//!   2. In build_frame_data: apply write_mesh_positions_normals for each SkinnedMeshUpdate
//!   3. Render normally
//!
//! Controls:
//! - Speed slider: how fast the arm bends

use eframe::egui;
use viewport_lib::{
    ActionFrame, Joint, Material, MeshId, MeshData, Pose, RuntimeFrameContext, RuntimePlugin,
    RuntimeStepContext, Skeleton, SkeletonPlugin, SkinnedMeshUpdate, SkinWeights,
    ViewportRuntime,
    runtime::plugin::phase,
    scene::{Scene, material::BackfacePolicy},
    selection::Selection,
};

// ---------------------------------------------------------------------------
// Demo selector
// ---------------------------------------------------------------------------
//
// Showcase 47 grows additively across the skeletal-animation plan phases.
// Each phase adds a new entry to this enum and the supporting state on
// `Skin47State`; earlier entries stay selectable so the showcase remains a
// running tour of the substrate's capabilities.

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Skin47Demo {
    /// Phase 1: hand-written sine-wave pose driving a two-joint arm.
    SineWaveArm,
}

impl Skin47Demo {
    fn label(self) -> &'static str {
        match self {
            Skin47Demo::SineWaveArm => "Sine-wave arm",
        }
    }
}

use crate::App;

// ---------------------------------------------------------------------------
// Speed resource
// ---------------------------------------------------------------------------

/// Per-frame speed control written from the app and read by PoseDriver.
struct BendSpeed(pub f32);

// ---------------------------------------------------------------------------
// Pose driver plugin
// ---------------------------------------------------------------------------

/// Writes a Pose to resources each frame, rotating the child joint.
/// Reads BendSpeed from resources if present; otherwise uses its default.
struct PoseDriver {
    time: f32,
    default_speed: f32,
}

impl RuntimePlugin for PoseDriver {
    fn priority(&self) -> i32 {
        phase::ANIMATE
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext) {
        let speed = ctx.resources.get::<BendSpeed>().map(|s| s.0).unwrap_or(self.default_speed);
        self.time += ctx.dt * speed;
        let angle = (self.time).sin() * std::f32::consts::FRAC_PI_4;
        // Joint 1's local transform must reproduce its bind-pose position
        // (translate(0, 0, JOINT_Z)) before applying the per-frame rotation.
        // Skipping the bind translation collapses the upper arm onto the lower
        // one, which is what made earlier versions of this showcase look wrong.
        let mut pose = Pose::identity(2);
        pose.local_transforms[1] = glam::Affine3A::from_translation(glam::Vec3::new(0.0, 0.0, JOINT_Z))
            * glam::Affine3A::from_rotation_x(angle);
        ctx.resources.insert(pose);
    }
}

// ---------------------------------------------------------------------------
// Mesh generation
// ---------------------------------------------------------------------------

const RINGS: usize = 64;     // rings along the arm length (dense enough for a smooth bend)
const SIDES: usize = 20;     // vertices per ring
const ARM_LENGTH: f32 = 4.0; // total arm length along Z
const ARM_RADIUS: f32 = 0.35;
const JOINT_Z: f32 = 2.0;    // world Z where the joint sits
const BLEND_HALF_WIDTH: f32 = 0.75; // half-width of the joint's blend band along Z

/// Build a ring-stack arm mesh. Returns (positions, normals, indices, skin_weights).
fn build_arm_mesh() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>, SkinWeights) {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut joint_indices: Vec<[u8; 4]> = Vec::new();
    let mut joint_weights: Vec<[f32; 4]> = Vec::new();

    // Generate ring vertices.
    for r in 0..=RINGS {
        let t = r as f32 / RINGS as f32;
        let z = t * ARM_LENGTH;
        // Smoothstep blend across [JOINT_Z - BLEND_HALF_WIDTH, JOINT_Z + BLEND_HALF_WIDTH]
        // so the seam reads as a soft bend rather than a kink.
        let u = ((z - (JOINT_Z - BLEND_HALF_WIDTH)) / (2.0 * BLEND_HALF_WIDTH)).clamp(0.0, 1.0);
        let w1 = u * u * (3.0 - 2.0 * u);
        let w0 = 1.0 - w1;

        for s in 0..SIDES {
            let angle = (s as f32 / SIDES as f32) * std::f32::consts::TAU;
            let nx = angle.cos();
            let ny = angle.sin();
            positions.push([nx * ARM_RADIUS, ny * ARM_RADIUS, z]);
            normals.push([nx, ny, 0.0]);
            joint_indices.push([0, 1, 0, 0]);
            joint_weights.push([w0, w1, 0.0, 0.0]);
        }
    }

    // Generate quad indices for each ring pair.
    for r in 0..RINGS {
        let base = r * SIDES;
        let next = base + SIDES;
        for s in 0..SIDES {
            let a = (base + s) as u32;
            let b = (base + (s + 1) % SIDES) as u32;
            let c = (next + (s + 1) % SIDES) as u32;
            let d = (next + s) as u32;
            indices.push(a); indices.push(b); indices.push(d);
            indices.push(b); indices.push(c); indices.push(d);
        }
    }

    let skin_weights = SkinWeights { joint_indices, joint_weights };
    (positions, normals, indices, skin_weights)
}

/// Build the two-joint arm skeleton.
/// Joint 0: root at origin, inverse_bind = identity.
/// Joint 1: child at (0,0,JOINT_Z), parent=0, inverse_bind = translate(0,0,-JOINT_Z).
fn build_arm_skeleton() -> Skeleton {
    Skeleton::new(vec![
        Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: glam::Affine3A::IDENTITY,
        },
        Joint {
            name: "forearm".into(),
            parent: Some(0),
            inverse_bind: glam::Affine3A::from_translation(-glam::Vec3::new(0.0, 0.0, JOINT_Z)),
        },
    ])
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct Skin47State {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub runtime: ViewportRuntime,
    pub mesh_id: Option<MeshId>,
    /// CPU bind-pose vertex data kept for the SkeletonPlugin.
    pub bind_positions: Vec<[f32; 3]>,
    pub bind_normals: Vec<[f32; 3]>,
    pub speed: f32,
    /// Active demo from the sidebar selector.
    pub demo: Skin47Demo,
    /// Pending deformation updates to apply to the GPU on the next frame.
    pub pending_updates: Vec<SkinnedMeshUpdate>,
}

impl Default for Skin47State {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            runtime: ViewportRuntime::new(),
            mesh_id: None,
            bind_positions: Vec::new(),
            bind_normals: Vec::new(),
            speed: 1.0,
            demo: Skin47Demo::SineWaveArm,
            pending_updates: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

pub(crate) fn build_skin47_scene(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    let (positions, normals, indices, skin_weights) = build_arm_mesh();

    let mut mesh_data = MeshData::default();
    mesh_data.positions = positions.clone();
    mesh_data.normals = normals.clone();
    mesh_data.indices = indices;
    mesh_data.skin_weights = Some(skin_weights.clone());

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &mesh_data)
        .expect("arm mesh upload");

    app.skin47_state.mesh_id = Some(mesh_id);
    app.skin47_state.bind_positions = positions.clone();
    app.skin47_state.bind_normals = normals.clone();

    let skeleton = build_arm_skeleton();

    let plugin = SkeletonPlugin::new(
        skeleton,
        mesh_id,
        positions,
        normals,
        skin_weights,
    );

    let pose_driver = PoseDriver { time: 0.0, default_speed: app.skin47_state.speed };

    app.skin47_state.runtime = ViewportRuntime::new()
        .with_plugin(pose_driver)
        .with_plugin(plugin);

    // Add the arm to the scene. Render both sides so the inside of the bend
    // stays visible if LBS flips a normal near the seam.
    let mut mat = Material::from_colour([0.6, 0.75, 0.9]);
    mat.backface_policy = BackfacePolicy::Tint(0.4);
    app.skin47_state.scene.add(Some(mesh_id), glam::Mat4::IDENTITY, mat);

    app.skin47_state.built = true;
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

pub(crate) fn update_skin47(app: &mut App, dt: f32) {
    let camera = app.camera.clone();
    let frame_ctx = RuntimeFrameContext {
        dt,
        camera: &camera,
        viewport_size: glam::Vec2::new(800.0, 600.0),
        input: &ActionFrame::default(),
        pick_hit: None,
        clicked: false,
        drag_started: false,
        dragging: false,
        pointer_delta: glam::Vec2::ZERO,
        cursor_viewport: None,
        shift_held: false,
    };

    let output = app.skin47_state.runtime.step(
        &mut app.skin47_state.scene,
        &mut app.skin47_state.selection,
        &frame_ctx,
    );

    // Store updates so build_frame_data can apply them to the GPU.
    app.skin47_state.pending_updates = output.skinned_mesh_updates;
}

/// Apply pending skinned mesh updates to the GPU. Called from build_frame_data.
pub(crate) fn apply_skin47_updates(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    for u in app.skin47_state.pending_updates.drain(..) {
        let _ = renderer
            .resources_mut()
            .write_mesh_positions_normals(&app.queue, u.mesh_id, &u.positions, &u.normals);
    }
}

// ---------------------------------------------------------------------------
// Scene items
// ---------------------------------------------------------------------------

pub(crate) fn skin47_scene_items(app: &mut App) -> Vec<viewport_lib::SceneRenderItem> {
    app.skin47_state
        .scene
        .collect_render_items(&app.skin47_state.selection)
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_skin47(app: &mut App, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label("Skeletal animation tour. Demos are added as the substrate grows.");
        ui.add_space(6.0);
        ui.separator();

        ui.label("Demo:");
        let current = app.skin47_state.demo;
        egui::ComboBox::from_id_salt("skin47_demo")
            .selected_text(current.label())
            .show_ui(ui, |ui| {
                for d in [Skin47Demo::SineWaveArm] {
                    ui.selectable_value(&mut app.skin47_state.demo, d, d.label());
                }
            });
        ui.add_space(6.0);
        ui.separator();

        match app.skin47_state.demo {
            Skin47Demo::SineWaveArm => {
                ui.label("Bending speed:");
                let mut speed = app.skin47_state.speed;
                if ui
                    .add(egui::Slider::new(&mut speed, 0.0..=4.0).text("rad/s"))
                    .changed()
                {
                    app.skin47_state.speed = speed;
                    // PoseDriver reads BendSpeed from resources each frame.
                    app.skin47_state.runtime.resources_mut().insert(BendSpeed(speed));
                }
                ui.add_space(6.0);

                ui.separator();
                ui.label("How it works:");
                ui.small("- Skeleton: 2 joints (root at origin, forearm at z=2.0).");
                ui.small("- Pose: child joint rotates around X each frame.");
                ui.small("- SkeletonPlugin: runs at POST_SIM, reads Pose, runs LBS.");
                ui.small("- Output: SkinnedMeshUpdate pushed to output.skinned_mesh_updates.");
                ui.small("- App: calls write_mesh_positions_normals before rendering.");
            }
        }
    });
}
