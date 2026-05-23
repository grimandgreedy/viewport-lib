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
    ActionFrame, AnimationClip, BuiltinMatcap, Channel, ClipPlayerPlugin, Interpolation, Joint,
    Material, MatcapId, MeshId, MeshData, Pose, RuntimeFrameContext, RuntimePlugin,
    RuntimeStepContext, Sampler, Skeleton, SkeletonPlugin, SkinnedActor, SkinnedActorPart,
    SkinnedActorPlugin, SkinnedMeshUpdate, SkinnedPoseUpdate, SkinningPath, SkinWeights, Track,
    TrackValues, ViewportRuntime,
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
    /// Phase 2: pose sampled from an AnimationClip via ClipPlayerPlugin.
    ClipDrivenArm,
    /// Phase 3: skinned mesh imported from a glTF file, animated by one of its
    /// own clips. Falls back to a placeholder when the asset is missing.
    GltfCharacter,
    /// Phase 4: N independently-animated copies of the glTF character on a
    /// grid. One SkinnedActorPlugin owns the shared skeleton and N actors.
    Crowd,
}

impl Skin47Demo {
    fn label(self) -> &'static str {
        match self {
            Skin47Demo::SineWaveArm => "Sine-wave arm",
            Skin47Demo::ClipDrivenArm => "Clip-driven arm",
            Skin47Demo::GltfCharacter => "glTF character",
            Skin47Demo::Crowd => "Crowd",
        }
    }

    fn all() -> [Skin47Demo; 4] {
        [
            Skin47Demo::SineWaveArm,
            Skin47Demo::ClipDrivenArm,
            Skin47Demo::GltfCharacter,
            Skin47Demo::Crowd,
        ]
    }
}

/// Path the glTF character demo looks for. Drop a `.glb` (or `.gltf` with its
/// buffers/textures next to it) at this path to enable the demo. See the
/// in-app help text for details.
pub(crate) const GLTF_DEMO_PATH: &str = "examples/eframe_showcase/assets/character.glb";

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
    pub bind_skin_weights: Option<SkinWeights>,
    pub speed: f32,
    /// Demo selected in the sidebar.
    pub demo: Skin47Demo,
    /// Demo whose runtime is currently active. When `demo != active_demo`,
    /// the runtime is rebuilt on the next frame.
    pub active_demo: Skin47Demo,
    /// Pending CPU-path deformation updates to apply to the GPU on the next frame.
    pub pending_updates: Vec<SkinnedMeshUpdate>,
    /// Pending GPU-path joint palette updates from the runtime.
    pub pending_pose_updates: Vec<SkinnedPoseUpdate>,
    /// Active deformation path (CPU or GPU). Toggling rebuilds the runtime so
    /// the plugins emit the right output channel.
    pub path: SkinningPath,
    /// Path whose runtime is currently active. Rebuild when `path != active_path`.
    pub active_path: SkinningPath,
    /// Mesh IDs for which `set_skin_weights` has already been called. Avoids
    /// re-uploading the weights buffer on every GPU-path switch.
    pub skin_weights_uploaded: std::collections::HashSet<MeshId>,
    /// Rolling per-frame timing (microseconds) of `runtime.step()`, for the
    /// CPU/GPU comparison readout. Reset on path switch.
    pub last_step_us: f32,
    /// Mesh IDs uploaded for each loaded glTF part. Empty when no asset is
    /// available or the demo file is absent.
    pub gltf_mesh_ids: Vec<MeshId>,
    pub gltf_asset: Option<GltfCharacterAsset>,
    /// True when the glTF demo path was checked and the file was absent.
    pub gltf_missing: bool,
    /// Scene node IDs currently present for each demo. Populated lazily when
    /// a demo becomes active and dropped when it deactivates so the user only
    /// sees the meshes belonging to the selected demo.
    pub arm_node: Option<viewport_lib::NodeId>,
    pub gltf_nodes: Vec<viewport_lib::NodeId>,
    pub crowd_nodes: Vec<viewport_lib::NodeId>,
    /// Per-actor uploaded mesh IDs for the crowd demo. Outer Vec indexes the
    /// actor; inner Vec is one MeshId per part. Empty until the crowd demo
    /// has been activated at least once.
    pub crowd_actor_meshes: Vec<Vec<MeshId>>,
    /// Number of actors the user has requested in the crowd demo. The actual
    /// number lazily grows as the user increases the slider; meshes already
    /// uploaded are reused on subsequent activations.
    pub crowd_count: usize,
    /// Crowd count whose runtime/scene is currently active. Differs from
    /// `crowd_count` between the slider moving and `apply_skin47_updates`
    /// running (the latter has renderer access for uploads).
    pub crowd_count_active: usize,
    /// Highest count ever requested. Used to size the existing actor pool so
    /// the slider can scale up to that without re-uploading.
    pub crowd_max_uploaded: usize,

    // -----------------------------------------------------------------------
    // Appearance toggles (Phase 5.3 demo)
    //
    // Apply to every skinned node currently in the scene. Bumping
    // `appearance_version` re-stamps the scene on the next frame.
    // -----------------------------------------------------------------------
    /// Per-item opacity. < 1.0 routes draws through the skinned transparent
    /// pipeline (and HDR OIT when HDR is the active path).
    pub opacity: f32,
    /// Force-wireframe per node (independent of viewport.wireframe_mode).
    pub per_item_wireframe: bool,
    /// `use_pbr` on the material. Exercises the Cook-Torrance fragment branch
    /// reused by the skinned variant.
    pub use_pbr: bool,
    /// When true, set the material's matcap to a built-in matcap. Exercises
    /// the matcap fragment branch through the skinned variant.
    pub use_matcap: bool,
    /// When true, set `backface_policy = Tint(0.4)` so the skinned two-sided
    /// pipeline takes over.
    pub two_sided: bool,
    /// Bumped whenever any of the above changes. The per-frame update applies
    /// the new values to all skinned nodes when this differs from
    /// `applied_appearance_version`.
    pub appearance_version: u32,
    pub applied_appearance_version: u32,
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
            bind_skin_weights: None,
            speed: 1.0,
            demo: Skin47Demo::SineWaveArm,
            active_demo: Skin47Demo::SineWaveArm,
            pending_updates: Vec::new(),
            pending_pose_updates: Vec::new(),
            path: SkinningPath::Cpu,
            active_path: SkinningPath::Cpu,
            skin_weights_uploaded: std::collections::HashSet::new(),
            last_step_us: 0.0,
            gltf_mesh_ids: Vec::new(),
            gltf_asset: None,
            gltf_missing: false,
            arm_node: None,
            gltf_nodes: Vec::new(),
            crowd_nodes: Vec::new(),
            crowd_actor_meshes: Vec::new(),
            crowd_count: 6,
            crowd_count_active: 0,
            crowd_max_uploaded: 0,
            opacity: 1.0,
            per_item_wireframe: false,
            use_pbr: false,
            use_matcap: false,
            two_sided: true,
            appearance_version: 0,
            applied_appearance_version: 0,
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

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &mesh_data)
        .expect("arm mesh upload");

    app.skin47_state.mesh_id = Some(mesh_id);
    app.skin47_state.bind_positions = positions.clone();
    app.skin47_state.bind_normals = normals.clone();
    app.skin47_state.bind_skin_weights = Some(skin_weights.clone());

    // Try to upload the glTF character if its asset file is present.
    let gltf_path = std::path::Path::new(GLTF_DEMO_PATH);
    if let Some(asset) = try_load_gltf_character(gltf_path) {
        for part in &asset.parts {
            let id = renderer
                .resources_mut()
                .upload_mesh_data(&app.device, &part.mesh_data)
                .expect("glTF part mesh upload");
            app.skin47_state.gltf_mesh_ids.push(id);
        }
        app.skin47_state.gltf_asset = Some(asset);
    } else {
        app.skin47_state.gltf_missing = true;
    }

    app.skin47_state.runtime = build_runtime_for_demo(
        app.skin47_state.demo,
        mesh_id,
        &positions,
        &normals,
        &skin_weights,
        &app.skin47_state.gltf_mesh_ids,
        app.skin47_state.gltf_asset.as_ref(),
        &app.skin47_state.crowd_actor_meshes,
        app.skin47_state.crowd_count,
        app.skin47_state.speed,
        app.skin47_state.path,
    );
    app.skin47_state.active_demo = app.skin47_state.demo;
    populate_scene_for_demo(&mut app.skin47_state);

    // Force the appearance toggles to apply on the first frame, otherwise
    // the scene shows whatever material build_skin47_scene set and ignores
    // the sidebar defaults until the user toggles something.
    app.skin47_state.appearance_version =
        app.skin47_state.appearance_version.wrapping_add(1);

    app.skin47_state.built = true;
}

/// Make the scene contain only the meshes belonging to the currently active
/// demo. Removes whatever the previous demo added and inserts the new demo's
/// meshes. Idempotent: calling with the same demo is a no-op.
fn populate_scene_for_demo(state: &mut Skin47State) {
    let arm_demos = matches!(state.demo, Skin47Demo::SineWaveArm | Skin47Demo::ClipDrivenArm);
    let gltf_demo = matches!(state.demo, Skin47Demo::GltfCharacter);
    let crowd_demo = matches!(state.demo, Skin47Demo::Crowd);

    // Drop the arm node if it shouldn't be visible.
    if !arm_demos {
        if let Some(id) = state.arm_node.take() {
            state.scene.remove(id);
        }
    }
    // Drop glTF nodes if they shouldn't be visible.
    if !gltf_demo {
        for id in state.gltf_nodes.drain(..) {
            state.scene.remove(id);
        }
    }
    // Drop crowd nodes whenever we're not showing the crowd OR whenever the
    // crowd size has changed (the simplest correct behavior).
    if !crowd_demo {
        for id in state.crowd_nodes.drain(..) {
            state.scene.remove(id);
        }
    }

    // Add the arm if needed and not already present.
    if arm_demos && state.arm_node.is_none() {
        if let Some(mesh_id) = state.mesh_id {
            let mut mat = Material::from_colour([0.6, 0.75, 0.9]);
            mat.backface_policy = BackfacePolicy::Tint(0.4);
            state.arm_node = Some(state.scene.add(Some(mesh_id), glam::Mat4::IDENTITY, mat));
        }
    }

    // Add glTF parts if needed and not already present.
    if gltf_demo && state.gltf_nodes.is_empty() {
        if let Some(asset) = state.gltf_asset.as_ref() {
            for (id, part) in state.gltf_mesh_ids.iter().zip(asset.parts.iter()) {
                let mut gmat = Material::from_colour(part.colour);
                gmat.backface_policy = BackfacePolicy::Tint(0.4);
                let node = state.scene.add(Some(*id), part.scene_transform, gmat);
                state.gltf_nodes.push(node);
            }
        }
    }

    // Add crowd parts. One node per (actor, part); the actor's world
    // translation is composed on top of the part's import-time transform so
    // the Z-up rotation still applies.
    if crowd_demo && state.crowd_nodes.is_empty() {
        if let Some(asset) = state.gltf_asset.as_ref() {
            let total = state.crowd_count.min(state.crowd_actor_meshes.len());
            for actor_idx in 0..total {
                let translation = crowd_translation(actor_idx, total);
                let actor_offset = glam::Mat4::from_translation(translation);
                for (mid, part) in state.crowd_actor_meshes[actor_idx]
                    .iter()
                    .zip(asset.parts.iter())
                {
                    let mut gmat = Material::from_colour(part.colour);
                    gmat.backface_policy = BackfacePolicy::Tint(0.4);
                    let node = state.scene.add(
                        Some(*mid),
                        actor_offset * part.scene_transform,
                        gmat,
                    );
                    state.crowd_nodes.push(node);
                }
            }
        }
    }
}

/// Upload extra mesh copies for the crowd so each actor has its own GPU
/// vertex buffer to deform into. Idempotent: already-uploaded actors are
/// reused; calling with a smaller count is a no-op (extra uploads stay).
fn ensure_crowd_uploads(
    state: &mut Skin47State,
    device: &wgpu::Device,
    renderer: &mut viewport_lib::ViewportRenderer,
    desired: usize,
) {
    let Some(asset) = state.gltf_asset.as_ref() else {
        return;
    };
    while state.crowd_actor_meshes.len() < desired {
        let mut actor_meshes = Vec::with_capacity(asset.parts.len());
        for part in &asset.parts {
            let id = renderer
                .resources_mut()
                .upload_mesh_data(device, &part.mesh_data)
                .expect("crowd part upload");
            actor_meshes.push(id);
        }
        state.crowd_actor_meshes.push(actor_meshes);
    }
    state.crowd_max_uploaded = state.crowd_max_uploaded.max(desired);
}

// ---------------------------------------------------------------------------
// Demo wiring
// ---------------------------------------------------------------------------

/// Build a fresh runtime for a given demo. `SkeletonPlugin` takes ownership of
/// the bind-pose vertex arrays and skin weights, so the demo state stores its
/// own copies and clones them in here per rebuild.
fn build_runtime_for_demo(
    demo: Skin47Demo,
    arm_mesh_id: MeshId,
    arm_positions: &[[f32; 3]],
    arm_normals: &[[f32; 3]],
    arm_skin_weights: &SkinWeights,
    gltf_mesh_ids: &[MeshId],
    gltf_asset: Option<&GltfCharacterAsset>,
    crowd_actor_meshes: &[Vec<MeshId>],
    crowd_count: usize,
    speed: f32,
    path: SkinningPath,
) -> ViewportRuntime {
    match demo {
        Skin47Demo::SineWaveArm => {
            let skeleton_plugin = SkeletonPlugin::new(
                build_arm_skeleton(),
                arm_mesh_id,
                arm_positions.to_vec(),
                arm_normals.to_vec(),
                arm_skin_weights.clone(),
            )
            .with_path(path);
            let pose_driver = PoseDriver { time: 0.0, default_speed: speed };
            ViewportRuntime::new()
                .with_plugin(pose_driver)
                .with_plugin(skeleton_plugin)
        }
        Skin47Demo::ClipDrivenArm => {
            let skeleton_plugin = SkeletonPlugin::new(
                build_arm_skeleton(),
                arm_mesh_id,
                arm_positions.to_vec(),
                arm_normals.to_vec(),
                arm_skin_weights.clone(),
            )
            .with_path(path);
            let bind_pose = bind_pose_for_arm();
            let clip = build_bend_clip();
            let player = ClipPlayerPlugin::new(clip, bind_pose).with_speed(speed);
            ViewportRuntime::new()
                .with_plugin(player)
                .with_plugin(skeleton_plugin)
        }
        Skin47Demo::GltfCharacter => {
            // Fall back to an empty runtime when the asset is missing; the
            // sidebar shows an explanatory message in this state.
            let Some(asset) = gltf_asset else {
                return ViewportRuntime::new();
            };
            if gltf_mesh_ids.len() != asset.parts.len() {
                return ViewportRuntime::new();
            }
            let clip_idx = asset.active_clip.min(asset.clips.len().saturating_sub(1));
            let player = ClipPlayerPlugin::new(
                asset.clips[clip_idx].clone(),
                asset.bind_pose.clone(),
            )
            .with_speed(speed);

            // Build one SkeletonPlugin per mesh part, all sharing the same
            // skeleton and reading the same Pose written by the player.
            let mut runtime = ViewportRuntime::new().with_plugin(player);
            for (gid, part) in gltf_mesh_ids.iter().zip(asset.parts.iter()) {
                let plugin = SkeletonPlugin::new(
                    asset.skeleton.clone(),
                    *gid,
                    part.bind_positions.clone(),
                    part.bind_normals.clone(),
                    part.skin_weights.clone(),
                )
                .with_path(path);
                runtime = runtime.with_plugin(plugin);
            }
            runtime
        }
        Skin47Demo::Crowd => {
            let Some(asset) = gltf_asset else {
                return ViewportRuntime::new();
            };
            if crowd_actor_meshes.len() < crowd_count {
                return ViewportRuntime::new();
            }

            // Build one SkinnedActorPlugin holding all crowd actors. Each
            // actor stages its playhead by an even fraction of its clip
            // duration so the crowd reads as out-of-phase loops.
            let mut actors: Vec<SkinnedActor> = Vec::with_capacity(crowd_count);
            for actor_idx in 0..crowd_count {
                let mesh_ids = &crowd_actor_meshes[actor_idx];
                let mut parts: Vec<SkinnedActorPart> = Vec::with_capacity(asset.parts.len());
                for (gid, part) in mesh_ids.iter().zip(asset.parts.iter()) {
                    parts.push(SkinnedActorPart {
                        mesh_id: *gid,
                        bind_positions: part.bind_positions.clone(),
                        bind_normals: part.bind_normals.clone(),
                        skin_weights: part.skin_weights.clone(),
                    });
                }
                let clip_index = actor_idx % asset.clips.len();
                let phase = if asset.clips[clip_index].duration > 0.0 {
                    (actor_idx as f32 / crowd_count.max(1) as f32) * asset.clips[clip_index].duration
                } else {
                    0.0
                };
                actors.push(
                    SkinnedActor::new(parts)
                        .with_clip(clip_index)
                        .with_playhead(phase)
                        .with_speed(speed),
                );
            }

            let plugin =
                SkinnedActorPlugin::new(asset.skeleton.clone(), asset.bind_pose.clone(), asset.clips.clone())
                    .with_actors(actors)
                    .with_path(path);
            ViewportRuntime::new().with_plugin(plugin)
        }
    }
}

/// Grid spacing (in world units) between crowd members. Set roughly to the
/// character footprint so they read as distinct without overlapping.
const CROWD_SPACING: f32 = 3.0;

/// Compute a world-space translation for one crowd actor placed on a roughly
/// square grid centered on the origin.
fn crowd_translation(actor_idx: usize, total: usize) -> glam::Vec3 {
    let cols = (total as f32).sqrt().ceil().max(1.0) as usize;
    let row = (actor_idx / cols) as f32;
    let col = (actor_idx % cols) as f32;
    let rows = ((total + cols - 1) / cols).max(1) as f32;
    let cx = (cols as f32 - 1.0) * 0.5;
    let cy = (rows - 1.0) * 0.5;
    glam::Vec3::new((col - cx) * CROWD_SPACING, (row - cy) * CROWD_SPACING, 0.0)
}

/// Bind pose for the two-joint arm: joint 0 identity, joint 1 translated to
/// `(0, 0, JOINT_Z)`. The clip's rotation track only modifies joint 1's
/// rotation; translation is preserved from this bind pose.
fn bind_pose_for_arm() -> Pose {
    let mut p = Pose::identity(2);
    p.local_transforms[1] = glam::Affine3A::from_translation(glam::Vec3::new(0.0, 0.0, JOINT_Z));
    p
}

// ---------------------------------------------------------------------------
// glTF -> runtime adapter
// ---------------------------------------------------------------------------

/// One skinned mesh part loaded from a glTF character. Owns the bind data
/// because SkeletonPlugin takes ownership when constructed.
pub(crate) struct GltfMeshPart {
    pub name: String,
    pub mesh_data: MeshData,
    pub bind_positions: Vec<[f32; 3]>,
    pub bind_normals: Vec<[f32; 3]>,
    pub skin_weights: SkinWeights,
    pub colour: [f32; 3],
    /// Scene transform from the io loader. Carries the Y-up -> Z-up rotation
    /// applied by viewport-lib-io's glTF loader.
    pub scene_transform: glam::Mat4,
}

/// What `try_load_gltf_character` produces on success. A character may consist
/// of many mesh parts that all share one skeleton (common for game-style rigs
/// with separate meshes per body region).
pub(crate) struct GltfCharacterAsset {
    pub parts: Vec<GltfMeshPart>,
    pub skeleton: Skeleton,
    pub bind_pose: Pose,
    pub clips: Vec<AnimationClip>,
    pub clip_names: Vec<String>,
    pub active_clip: usize,
}

/// Load all skinned mesh parts + skeleton + animations from a glTF file via
/// viewport-lib-io and convert them to runtime types. Returns `None` if the
/// file is missing or has no skinned mesh.
fn try_load_gltf_character(path: &std::path::Path) -> Option<GltfCharacterAsset> {
    use viewport_lib_io::{
        AnimationChannel as IoChannel, AnimationInterpolation as IoInterp,
        AnimationTrackValues as IoTrackValues, Joint as IoJoint, Skeleton as IoSkeleton,
    };

    if !path.exists() {
        return None;
    }
    let scene = viewport_lib_io::loaders::gltf::scene_from_path(path).ok()?;

    // Pick the skeleton that the first skinned mesh references; convert all
    // mesh parts that target it.
    let first_skinned = scene
        .meshes
        .iter()
        .find(|m| m.skeleton_index.is_some() && m.mesh.skin_weights.is_some())?;
    let skeleton_idx = first_skinned.skeleton_index.unwrap();
    let io_skeleton: &IoSkeleton = scene.skeletons.get(skeleton_idx)?;

    // Convert skeleton.
    let joints: Vec<Joint> = io_skeleton
        .joints
        .iter()
        .map(|j: &IoJoint| Joint {
            name: j.name.clone(),
            parent: j.parent,
            inverse_bind: glam::Affine3A::from_mat4(j.inverse_bind),
        })
        .collect();
    let skeleton = Skeleton::new(joints);
    let bind_pose = bind_pose_from_skeleton(&skeleton);

    // Convert every mesh part that targets this skeleton. Assign each a
    // distinct colour so the rig is readable on screen.
    const PART_COLOURS: [[f32; 3]; 6] = [
        [0.85, 0.7, 0.55],
        [0.70, 0.55, 0.45],
        [0.55, 0.65, 0.80],
        [0.80, 0.55, 0.55],
        [0.55, 0.75, 0.55],
        [0.75, 0.75, 0.55],
    ];
    let mut parts = Vec::new();
    for io_mesh in scene.meshes.iter() {
        if io_mesh.skeleton_index != Some(skeleton_idx) {
            continue;
        }
        let io_sw = match io_mesh.mesh.skin_weights.as_ref() {
            Some(sw) => sw,
            None => continue,
        };
        let skin_weights = SkinWeights {
            joint_indices: io_sw.joint_indices.clone(),
            joint_weights: io_sw.joint_weights.clone(),
        };
        let mut mesh_data = MeshData::default();
        mesh_data.positions = io_mesh.mesh.positions.clone();
        mesh_data.normals = io_mesh.mesh.normals.clone();
        mesh_data.indices = io_mesh.mesh.indices.clone();
        let colour = PART_COLOURS[parts.len() % PART_COLOURS.len()];
        parts.push(GltfMeshPart {
            name: io_mesh.name.clone(),
            mesh_data,
            bind_positions: io_mesh.mesh.positions.clone(),
            bind_normals: io_mesh.mesh.normals.clone(),
            skin_weights,
            colour,
            scene_transform: io_mesh.transform,
        });
    }
    if parts.is_empty() {
        return None;
    }

    // Convert every clip targeting this skeleton.
    let mut clips = Vec::new();
    let mut clip_names = Vec::new();
    for io_clip in scene.animations.iter().filter(|c| c.skeleton_index == skeleton_idx) {
        let mut tracks = Vec::with_capacity(io_clip.tracks.len());
        for t in &io_clip.tracks {
            let interp = match t.sampler.interpolation {
                IoInterp::Step => Interpolation::Step,
                IoInterp::Linear => Interpolation::Linear,
                // CubicSpline is preserved on import but not yet sampled.
                IoInterp::CubicSpline => continue,
            };
            let channel = match t.channel {
                IoChannel::Translation => Channel::Translation,
                IoChannel::Rotation => Channel::Rotation,
                IoChannel::Scale => Channel::Scale,
            };
            let values = match &t.sampler.values {
                IoTrackValues::Vec3(v) => TrackValues::Vec3(v.clone()),
                IoTrackValues::Quat(v) => TrackValues::Quat(v.clone()),
            };
            tracks.push(Track {
                joint: t.joint,
                channel,
                sampler: Sampler {
                    interpolation: interp,
                    times: t.sampler.times.clone(),
                    values,
                },
            });
        }
        if tracks.is_empty() {
            continue;
        }
        clips.push(AnimationClip {
            duration: io_clip.duration,
            tracks,
        });
        clip_names.push(io_clip.name.clone());
    }
    if clips.is_empty() {
        return None;
    }

    Some(GltfCharacterAsset {
        parts,
        skeleton,
        bind_pose,
        clips,
        clip_names,
        active_clip: 0,
    })
}

/// Reconstruct the bind pose as local transforms by inverting the bind-world
/// chain implied by `inverse_bind`. For root joints, local = inverse_bind^-1.
/// For child joints, local = parent_bind_world^-1 * bind_world.
fn bind_pose_from_skeleton(skeleton: &Skeleton) -> Pose {
    let n = skeleton.joint_count();
    let mut bind_world = vec![glam::Affine3A::IDENTITY; n];
    let mut pose = Pose::identity(n);
    for (i, joint) in skeleton.joints().iter().enumerate() {
        let inverse_bind = joint.inverse_bind;
        // bind_world = inverse_bind^-1 if the matrix is invertible.
        let bind_world_i = inverse_bind.inverse();
        bind_world[i] = bind_world_i;
        let local = match joint.parent {
            Some(p) => bind_world[p as usize].inverse() * bind_world_i,
            None => bind_world_i,
        };
        pose.local_transforms[i] = local;
    }
    pose
}

/// A short hand-authored clip that bends the forearm joint back and forth
/// around the X axis. Two-second loop, +/- 45 deg, linear interpolation.
fn build_bend_clip() -> AnimationClip {
    let bend = std::f32::consts::FRAC_PI_4;
    let rot_keys = vec![
        glam::Quat::IDENTITY,
        glam::Quat::from_rotation_x(bend),
        glam::Quat::IDENTITY,
        glam::Quat::from_rotation_x(-bend),
        glam::Quat::IDENTITY,
    ];
    let times = vec![0.0, 0.5, 1.0, 1.5, 2.0];

    AnimationClip {
        duration: 2.0,
        tracks: vec![Track {
            joint: 1,
            channel: Channel::Rotation,
            sampler: Sampler {
                interpolation: Interpolation::Linear,
                times,
                values: TrackValues::Quat(rot_keys),
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

pub(crate) fn update_skin47(app: &mut App, dt: f32) {
    // Rebuild the runtime if the user selected a different demo or switched
    // CPU/GPU path. Path change also re-stamps `skin_instance` on the scene
    // nodes so the renderer routes through the skinned pipeline variant on
    // GPU and through the static pipeline on CPU.
    let demo_changed = app.skin47_state.demo != app.skin47_state.active_demo;
    let path_changed = app.skin47_state.path != app.skin47_state.active_path;
    if demo_changed || path_changed {
        if let (Some(mesh_id), Some(skin_weights)) = (
            app.skin47_state.mesh_id,
            app.skin47_state.bind_skin_weights.clone(),
        ) {
            app.skin47_state.runtime = build_runtime_for_demo(
                app.skin47_state.demo,
                mesh_id,
                &app.skin47_state.bind_positions,
                &app.skin47_state.bind_normals,
                &skin_weights,
                &app.skin47_state.gltf_mesh_ids,
                app.skin47_state.gltf_asset.as_ref(),
                &app.skin47_state.crowd_actor_meshes,
                app.skin47_state.crowd_count,
                app.skin47_state.speed,
                app.skin47_state.path,
            );
            app.skin47_state.active_demo = app.skin47_state.demo;
            app.skin47_state.active_path = app.skin47_state.path;
            populate_scene_for_demo(&mut app.skin47_state);
            restamp_skin_instances(&mut app.skin47_state);
            // New nodes need the current appearance toggles applied on next
            // pass through `apply_skin47_updates`.
            app.skin47_state.applied_appearance_version =
                app.skin47_state.appearance_version.wrapping_sub(1);
        }
    }

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

    let t0 = std::time::Instant::now();
    let output = app.skin47_state.runtime.step(
        &mut app.skin47_state.scene,
        &mut app.skin47_state.selection,
        &frame_ctx,
    );
    let step_us = t0.elapsed().as_micros() as f32;
    // Exponential smoothing so the readout is steady enough to read.
    let alpha = 0.1_f32;
    app.skin47_state.last_step_us =
        alpha * step_us + (1.0 - alpha) * app.skin47_state.last_step_us;

    // Hand the right update channel through to build_frame_data.
    app.skin47_state.pending_updates = output.skinned_mesh_updates;
    app.skin47_state.pending_pose_updates = output.skinned_pose_updates;
}

/// Apply the current appearance toggles to every scene node belonging to the
/// active demo. Re-stamps each frame only when the appearance version bumps,
/// so unchanged frames pay no cost.
fn apply_skin47_appearance(state: &mut Skin47State, renderer: &viewport_lib::ViewportRenderer) {
    if state.appearance_version == state.applied_appearance_version {
        return;
    }
    state.applied_appearance_version = state.appearance_version;

    let matcap_id: Option<MatcapId> = if state.use_matcap {
        Some(renderer.resources().builtin_matcap_id(BuiltinMatcap::Jade))
    } else {
        None
    };

    // Each (node, base_colour) pair gets its own material so the per-node
    // colour set by build_skin47_scene / populate_scene_for_demo survives the
    // re-stamp. Read the colour back off the existing material so a future
    // edit to the base palette doesn't have to come back to this function.
    let mut nodes: Vec<viewport_lib::NodeId> = Vec::new();
    if let Some(arm) = state.arm_node {
        nodes.push(arm);
    }
    nodes.extend(state.gltf_nodes.iter().copied());
    nodes.extend(state.crowd_nodes.iter().copied());

    for node_id in nodes {
        let Some(node) = state.scene.node(node_id) else {
            continue;
        };
        let base_colour = node.material().base_colour;
        let mut mat = Material::from_colour(base_colour);
        mat.use_pbr = state.use_pbr;
        mat.matcap_id = matcap_id;
        mat.backface_policy = if state.two_sided {
            BackfacePolicy::Tint(0.4)
        } else {
            BackfacePolicy::Cull
        };
        state.scene.set_material(node_id, mat);

        let mut app = viewport_lib::scene::material::AppearanceSettings::default();
        app.opacity = state.opacity;
        app.wireframe = state.per_item_wireframe;
        state.scene.set_appearance(node_id, app);
    }
}

/// Stamp `skin_instance` on every scene node that participates in the active
/// demo so the renderer routes them through the skinned pipeline variant on
/// the GPU path. Clears it on the CPU path so the same nodes fall back to the
/// static pipeline (CPU writes deformed vertex buffers directly into the
/// bind-pose mesh).
fn restamp_skin_instances(state: &mut Skin47State) {
    let gpu = state.path == SkinningPath::Gpu;
    if let Some(arm_node) = state.arm_node {
        state.scene.set_skin_instance(arm_node, gpu.then_some(0));
    }
    for node in &state.gltf_nodes {
        state.scene.set_skin_instance(*node, gpu.then_some(0));
    }
    // Crowd: instance id is actor_idx. Nodes are laid out actor-major
    // (all parts of actor 0, then all parts of actor 1, ...).
    let parts_per_actor = state
        .gltf_asset
        .as_ref()
        .map(|a| a.parts.len())
        .unwrap_or(1)
        .max(1);
    for (i, node) in state.crowd_nodes.iter().enumerate() {
        let actor_idx = (i / parts_per_actor) as u32;
        state.scene.set_skin_instance(*node, gpu.then_some(actor_idx));
    }
}

/// Apply pending skinned mesh updates to the GPU. Called from build_frame_data.
///
/// Also performs deferred uploads + runtime rebuilds for the crowd demo, since
/// this is the only per-frame hook that has both `device` and `renderer`.
pub(crate) fn apply_skin47_updates(app: &mut App, renderer: &mut viewport_lib::ViewportRenderer) {
    // Crowd: respond to slider changes by uploading extra mesh copies (if
    // needed) and rebuilding the runtime + scene at the new actor count.
    if app.skin47_state.demo == Skin47Demo::Crowd
        && app.skin47_state.crowd_count != app.skin47_state.crowd_count_active
    {
        let desired = app.skin47_state.crowd_count;
        ensure_crowd_uploads(&mut app.skin47_state, &app.device, renderer, desired);

        // Drop existing crowd nodes so populate_scene_for_demo rebuilds with
        // the new count and grid layout.
        for id in app.skin47_state.crowd_nodes.drain(..) {
            app.skin47_state.scene.remove(id);
        }

        if let (Some(mesh_id), Some(skin_weights)) = (
            app.skin47_state.mesh_id,
            app.skin47_state.bind_skin_weights.clone(),
        ) {
            app.skin47_state.runtime = build_runtime_for_demo(
                app.skin47_state.demo,
                mesh_id,
                &app.skin47_state.bind_positions,
                &app.skin47_state.bind_normals,
                &skin_weights,
                &app.skin47_state.gltf_mesh_ids,
                app.skin47_state.gltf_asset.as_ref(),
                &app.skin47_state.crowd_actor_meshes,
                desired,
                app.skin47_state.speed,
                app.skin47_state.path,
            );
        }
        populate_scene_for_demo(&mut app.skin47_state);
        restamp_skin_instances(&mut app.skin47_state);
        app.skin47_state.applied_appearance_version =
            app.skin47_state.appearance_version.wrapping_sub(1);
        app.skin47_state.crowd_count_active = desired;
    }

    // Apply appearance toggles (opacity, wireframe, PBR, matcap, two-sided)
    // to every skinned node in the active demo.
    apply_skin47_appearance(&mut app.skin47_state, renderer);

    // CPU path: blit deformed positions/normals back into the bind-pose
    // vertex buffer for every emitted mesh.
    for u in app.skin47_state.pending_updates.drain(..) {
        let _ = renderer
            .resources_mut()
            .write_mesh_positions_normals(&app.queue, u.mesh_id, &u.positions, &u.normals);
    }

    // GPU path: lazily upload skin weights the first time we see a mesh on
    // the GPU path, then push the per-(mesh, instance) joint palette for the
    // current frame. The renderer's skinned pipeline variant takes over once
    // both the weights sidecar and the palette are present.
    if !app.skin47_state.pending_pose_updates.is_empty() {
        let pose_updates = std::mem::take(&mut app.skin47_state.pending_pose_updates);
        for u in &pose_updates {
            if !app.skin47_state.skin_weights_uploaded.contains(&u.mesh_id) {
                if let Some(w) = weights_for_mesh(&app.skin47_state, u.mesh_id) {
                    renderer
                        .resources_mut()
                        .set_skin_weights(&app.device, u.mesh_id, &w);
                    app.skin47_state.skin_weights_uploaded.insert(u.mesh_id);
                }
            }
            renderer.resources_mut().set_skin_palette(
                &app.device,
                &app.queue,
                u.mesh_id,
                u.instance_id,
                &u.joint_matrices,
            );
        }
    }
}

/// Lookup the bind-pose skin weights for `mesh_id` across all the assets the
/// showcase has loaded. Returns `None` if the mesh wasn't recognised; the
/// caller treats that as "not skinnable this frame" and falls back to the
/// static pipeline.
fn weights_for_mesh(state: &Skin47State, mesh_id: MeshId) -> Option<SkinWeights> {
    if Some(mesh_id) == state.mesh_id {
        return state.bind_skin_weights.clone();
    }
    if let Some(asset) = state.gltf_asset.as_ref() {
        for (i, id) in state.gltf_mesh_ids.iter().enumerate() {
            if *id == mesh_id {
                return Some(asset.parts[i].skin_weights.clone());
            }
        }
        for (actor_idx, mesh_ids) in state.crowd_actor_meshes.iter().enumerate() {
            let _ = actor_idx;
            for (i, id) in mesh_ids.iter().enumerate() {
                if *id == mesh_id {
                    return Some(asset.parts[i].skin_weights.clone());
                }
            }
        }
    }
    None
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
                for d in Skin47Demo::all() {
                    ui.selectable_value(&mut app.skin47_state.demo, d, d.label());
                }
            });
        ui.add_space(6.0);

        ui.label("Skinning path:");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut app.skin47_state.path, SkinningPath::Cpu, "CPU");
            ui.selectable_value(&mut app.skin47_state.path, SkinningPath::Gpu, "GPU");
        });
        ui.small(format!(
            "runtime.step() avg: {:.0} us",
            app.skin47_state.last_step_us
        ));
        ui.small("CPU: re-uploads vertex buffers each frame.");
        ui.small("GPU: uploads joint palette only; shader does LBS.");
        ui.add_space(6.0);
        ui.separator();

        ui.label("Appearance (skinned pipeline coverage):");
        ui.small("Toggles apply to every node in the active demo.");
        ui.small("Use these with GPU path to exercise the 5.3 variants.");
        ui.add_space(4.0);
        let mut changed = false;
        let mut opacity = app.skin47_state.opacity;
        if ui
            .add(egui::Slider::new(&mut opacity, 0.05..=1.0).text("Opacity"))
            .changed()
        {
            app.skin47_state.opacity = opacity;
            changed = true;
        }
        if ui
            .checkbox(&mut app.skin47_state.per_item_wireframe, "Wireframe")
            .changed()
        {
            changed = true;
        }
        if ui
            .checkbox(&mut app.skin47_state.use_pbr, "PBR shading")
            .changed()
        {
            changed = true;
        }
        if ui
            .checkbox(&mut app.skin47_state.use_matcap, "Matcap (Jade)")
            .changed()
        {
            changed = true;
        }
        if ui
            .checkbox(&mut app.skin47_state.two_sided, "Two-sided (Tint)")
            .changed()
        {
            changed = true;
        }
        if changed {
            app.skin47_state.appearance_version =
                app.skin47_state.appearance_version.wrapping_add(1);
        }
        ui.small("Opacity < 1 -> skinned transparent pipeline.");
        ui.small("Wireframe -> skinned wireframe pipeline.");
        ui.small("PBR / Matcap -> fragment-stage branches (shared with static).");
        ui.small("Two-sided -> skinned two-sided pipeline.");
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
            Skin47Demo::ClipDrivenArm => {
                ui.label("Playback speed:");
                let mut speed = app.skin47_state.speed;
                if ui
                    .add(egui::Slider::new(&mut speed, 0.0..=4.0).text("x"))
                    .changed()
                {
                    app.skin47_state.speed = speed;
                    // Rebuild on the next frame so the new speed takes effect.
                    app.skin47_state.active_demo = Skin47Demo::SineWaveArm;
                }
                ui.add_space(6.0);

                ui.separator();
                ui.label("How it works:");
                ui.small("- AnimationClip: one rotation track on the forearm joint.");
                ui.small("- 5 keyframes over 2 seconds: 0, +45, 0, -45, 0 deg.");
                ui.small("- ClipPlayerPlugin: samples the clip at the playhead, writes Pose.");
                ui.small("- SkeletonPlugin: same as before, reads Pose and runs LBS.");
                ui.small("- Bind pose carries joint 1 translation; clip only animates rotation.");
            }
            Skin47Demo::GltfCharacter => {
                if app.skin47_state.gltf_asset.is_none() {
                    ui.label("No glTF asset loaded.");
                    ui.add_space(6.0);
                    ui.small("Drop a skinned .glb (or .gltf with its buffers next to it) at:");
                    ui.code(GLTF_DEMO_PATH);
                    ui.add_space(4.0);
                    ui.small("Restart the showcase to load it.");
                    if app.skin47_state.gltf_missing {
                        ui.add_space(4.0);
                        ui.small("(File not found on the last startup.)");
                    }
                } else {
                    // Clip selector. Switching clips triggers a runtime
                    // rebuild on the next frame.
                    let mut switch_clip: Option<usize> = None;
                    if let Some(asset) = app.skin47_state.gltf_asset.as_ref() {
                        ui.label(format!(
                            "{} mesh parts, {} animation clips",
                            asset.parts.len(),
                            asset.clips.len(),
                        ));
                        ui.add_space(4.0);
                        ui.label("Clip:");
                        let current = asset.active_clip;
                        let current_name = asset
                            .clip_names
                            .get(current)
                            .cloned()
                            .unwrap_or_else(|| format!("clip {current}"));
                        egui::ComboBox::from_id_salt("skin47_gltf_clip")
                            .selected_text(current_name)
                            .show_ui(ui, |ui| {
                                for (i, name) in asset.clip_names.iter().enumerate() {
                                    if ui.selectable_label(i == current, name).clicked() {
                                        switch_clip = Some(i);
                                    }
                                }
                            });
                    }
                    if let Some(i) = switch_clip {
                        if let Some(asset) = app.skin47_state.gltf_asset.as_mut() {
                            asset.active_clip = i;
                        }
                        // Force a runtime rebuild next frame.
                        app.skin47_state.active_demo = Skin47Demo::SineWaveArm;
                    }
                    ui.add_space(6.0);

                    ui.label("Playback speed:");
                    let mut speed = app.skin47_state.speed;
                    if ui
                        .add(egui::Slider::new(&mut speed, 0.0..=4.0).text("x"))
                        .changed()
                    {
                        app.skin47_state.speed = speed;
                        app.skin47_state.active_demo = Skin47Demo::SineWaveArm;
                    }
                    ui.add_space(6.0);
                    ui.separator();
                    ui.label("How it works:");
                    ui.small("- viewport-lib-io decodes the .glb to neutral io types.");
                    ui.small("- An adapter converts io Skeleton/Clip to runtime types.");
                    ui.small("- One ClipPlayerPlugin + N SkeletonPlugins (one per mesh part).");
                    ui.small("- All parts share the skeleton and the Pose written by the player.");
                    ui.small("- Bone-parented (non-skinned) rigs are auto-converted at import time.");
                }
            }
            Skin47Demo::Crowd => {
                if app.skin47_state.gltf_asset.is_none() {
                    ui.label("No glTF asset loaded.");
                    ui.add_space(6.0);
                    ui.small("The crowd demo reuses the glTF character; drop a .glb at:");
                    ui.code(GLTF_DEMO_PATH);
                    ui.add_space(4.0);
                    ui.small("Restart the showcase to load it.");
                } else {
                    ui.label("Crowd size:");
                    let mut n = app.skin47_state.crowd_count;
                    let max_n = 32usize;
                    if ui
                        .add(egui::Slider::new(&mut n, 1..=max_n).text("actors"))
                        .changed()
                    {
                        app.skin47_state.crowd_count = n;
                        // apply_skin47_updates will notice the mismatch and
                        // upload + rebuild on the next frame.
                    }
                    ui.add_space(6.0);

                    ui.label("Playback speed:");
                    let mut speed = app.skin47_state.speed;
                    if ui
                        .add(egui::Slider::new(&mut speed, 0.0..=4.0).text("x"))
                        .changed()
                    {
                        app.skin47_state.speed = speed;
                        // Force a rebuild on the next apply pass.
                        app.skin47_state.crowd_count_active =
                            app.skin47_state.crowd_count_active.wrapping_add(1);
                    }
                    ui.add_space(6.0);

                    if let Some(asset) = app.skin47_state.gltf_asset.as_ref() {
                        ui.label(format!(
                            "{} actors x {} parts = {} skinned meshes",
                            app.skin47_state.crowd_count,
                            asset.parts.len(),
                            app.skin47_state.crowd_count * asset.parts.len(),
                        ));
                    }

                    ui.separator();
                    ui.label("How it works:");
                    ui.small("- One SkinnedActorPlugin owns the shared skeleton and N actors.");
                    ui.small("- Each actor has its own clip choice, playhead, and play state.");
                    ui.small("- Playheads are staggered around each clip's duration to de-phase.");
                    ui.small("- N x parts unique GPU meshes; one vertex-buffer write per part per frame.");
                    ui.small("- Phase 5 (GPU skinning) replaces those writes with one joint-palette upload per actor.");
                }
            }
        }
    });
}
