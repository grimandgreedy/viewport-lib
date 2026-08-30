//! Installing GPU skinning in one call.
//!
//! Shows the `ViewportPlugin` installer path: build a skinned mesh and a pose
//! plugin, hand them to `SkinnedMeshFeature`, and install the deformer plus the
//! runtime plugin with a single `install_plugin` call. Each frame, one
//! `handle.apply(...)` drains the pose updates and uploads the joint palette,
//! replacing the hand-written drain-and-attach loop a host would otherwise run.
//!
//! Headless: it builds its own wgpu device and renders nothing to screen. Run
//! with:
//!
//!     cargo run --example skinned-feature
//!
//! Compare with `examples/eframe_showcase/showcase_45_skinned_animation.rs`,
//! which wires the same three pieces by hand across two frame hooks.

use glam::{Affine3A, Vec3};
use viewport_lib as vpl;
use vpl::plugins::skeleton::{
    Joint, Pose, Skeleton, SkeletonPlugin, SkinnedMeshFeature, SkinningPath,
};
use vpl::plugins::skinning::SkinWeights;
use vpl::wgpu;
use vpl::{MeshData, MeshId, ViewportRenderer, ViewportRuntime, install_plugin};

const ARM_LENGTH: f32 = 4.0;
const ARM_RADIUS: f32 = 0.5;
const JOINT_Z: f32 = 2.0;
const RINGS: usize = 8;
const SIDES: usize = 8;

// A capsule-ish tube split across two joints: the lower half is weighted to the
// root, the upper half to the forearm, with a soft blend at the seam.
fn build_arm_mesh() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>, SkinWeights) {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut joint_indices = Vec::new();
    let mut joint_weights = Vec::new();

    for r in 0..=RINGS {
        let z = (r as f32 / RINGS as f32) * ARM_LENGTH;
        let w1 = ((z - JOINT_Z) / ARM_LENGTH + 0.5).clamp(0.0, 1.0);
        let w0 = 1.0 - w1;
        for s in 0..SIDES {
            let angle = (s as f32 / SIDES as f32) * std::f32::consts::TAU;
            let (nx, ny) = (angle.cos(), angle.sin());
            positions.push([nx * ARM_RADIUS, ny * ARM_RADIUS, z]);
            normals.push([nx, ny, 0.0]);
            joint_indices.push([0, 1, 0, 0]);
            joint_weights.push([w0, w1, 0.0, 0.0]);
        }
    }

    for r in 0..RINGS {
        let base = r * SIDES;
        let next = base + SIDES;
        for s in 0..SIDES {
            let a = (base + s) as u32;
            let b = (base + (s + 1) % SIDES) as u32;
            let c = (next + (s + 1) % SIDES) as u32;
            let d = (next + s) as u32;
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }

    let skin_weights = SkinWeights {
        joint_indices,
        joint_weights,
    };
    (positions, normals, indices, skin_weights)
}

fn build_arm_skeleton() -> Skeleton {
    Skeleton::new(vec![
        Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: Affine3A::IDENTITY,
        },
        Joint {
            name: "forearm".into(),
            parent: Some(0),
            inverse_bind: Affine3A::from_translation(-Vec3::new(0.0, 0.0, JOINT_Z)),
        },
    ])
}

// Bend the forearm joint by `angle` about X. At angle 0 this reproduces the
// bind pose, so the mesh is undeformed.
fn bent_pose(angle: f32) -> Pose {
    let mut pose = Pose::identity(2);
    pose.local_transforms[1] =
        Affine3A::from_translation(Vec3::new(0.0, 0.0, JOINT_Z)) * Affine3A::from_rotation_x(angle);
    pose
}

fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()
    else {
        eprintln!("no wgpu adapter available; nothing to demonstrate");
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("skinned-feature"),
        required_limits: vpl::ViewportRenderer::recommended_device_limits(&adapter),
        ..Default::default()
    }))
    .expect("device");

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut runtime = ViewportRuntime::new();
    let mut scene = vpl::scene::scene::Scene::new();
    let mut selection = vpl::interaction::select::selection::Selection::new();

    // Upload the mesh the host would draw.
    let (positions, normals, indices, skin_weights) = build_arm_mesh();
    let mut mesh_data = MeshData::default();
    mesh_data.positions = positions;
    mesh_data.normals = normals;
    mesh_data.indices = indices;
    let mesh_id: MeshId = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh_data)
        .expect("mesh upload");

    // Build the pose plugin and install the whole feature in one call: this
    // registers the skinning deformer on the renderer and the pose plugin on
    // the runtime, and returns a handle that uploads palettes each frame.
    let pose_plugin = SkeletonPlugin::new(
        build_arm_skeleton(),
        mesh_id,
        mesh_data.positions.clone(),
        mesh_data.normals.clone(),
        skin_weights.clone(),
    )
    .with_path(SkinningPath::Gpu);

    let feature = SkinnedMeshFeature::new(pose_plugin).with_weights(mesh_id, skin_weights);
    let mut handle = install_plugin(feature, &device, &queue, Some(&mut runtime), &mut renderer)
        .expect("install skinning feature");

    println!(
        "installed: deformer registered = {}",
        renderer
            .resources()
            .deformer_id_by_name("viewport_skin")
            .is_some()
    );

    // Per-frame loop: drive the pose, step the runtime, and apply the pose
    // updates with a single call. In a windowed host this is the whole
    // skinning glue.
    for frame in 0..6 {
        let angle = (frame as f32) * 0.15;
        runtime.resources_mut().insert(bent_pose(angle));

        let mut frame_ctx = vpl::runtime::RuntimeFrameContext::default();
        frame_ctx.dt = 1.0 / 60.0;
        let mut output = runtime.step(&mut scene, &mut selection, &frame_ctx);

        handle.apply(renderer.resources_mut(), &device, &queue, &mut output);

        println!(
            "frame {frame}: bend {angle:.2} rad, mesh skinned = {}",
            handle.skinning().is_skinned_mesh(mesh_id)
        );
    }

    println!("done: one install call, one apply call per frame");
}
