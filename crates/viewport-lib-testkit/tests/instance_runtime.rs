//! A runtime attached to a `ViewportInstance`: attaching, taking and
//! replacing it, installing features through the instance, what
//! `step_runtime` hands back, and GPU plugins across `rebuild`.

use viewport_lib::plugins::skeleton::{
    Joint, Pose, Skeleton, SkeletonPlugin, SkinnedMeshFeature, SkinningPath,
};
use viewport_lib::plugins::skinning::SkinWeights;
use viewport_lib::runtime::gpu_plugin::gpu_phase;
use viewport_lib::runtime::{CameraFollow, CameraFollowTarget};
use viewport_lib::wgpu;
use viewport_lib::{
    InputOwnership, Material, OffscreenViewportTarget, RendererConfig, ViewportContext,
    ViewportError, ViewportInstance, ViewportRuntime, primitives,
};
use viewport_lib_testkit::fixtures::{
    CallLog, ConstantOffsetDeformer, DeformAndStepInstaller, LoggingGpuPlugin, LoggingRuntimePlugin,
};
use viewport_lib_testkit::{DeviceProfile, headless_device_with};

const SIZE: u32 = 64;
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// A device with the renderer's recommended limits, which deformers and
/// skinning need.
fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(&DeviceProfile::low_power("instance-runtime"))
}

fn instance(device: &wgpu::Device) -> ViewportInstance {
    let config = RendererConfig::new(OffscreenViewportTarget::render_format(SURFACE_FORMAT))
        .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    ViewportInstance::with_config(device, &config)
}

/// Assemble and render one frame into `target`.
fn render(
    instance: &mut ViewportInstance,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &OffscreenViewportTarget,
) {
    instance.frame(ViewportContext::new(
        InputOwnership::default(),
        [SIZE as f32, SIZE as f32],
    ));
    let cmd = instance.render(device, queue, target.render_view());
    queue.submit(std::iter::once(cmd));
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(5)),
    });
}

/// A runtime holding one logging runtime plugin labelled `label`.
fn labelled_runtime(log: &CallLog, label: &'static str) -> ViewportRuntime {
    ViewportRuntime::new().with_plugin(LoggingRuntimePlugin::new(
        log.clone(),
        label,
        viewport_lib::runtime::plugin::phase::ANIMATE,
    ))
}

/// `set_runtime` attaches and hands back the runtime it replaced,
/// `take_runtime` detaches, and `step_runtime` drives whichever is attached.
#[test]
fn set_take_and_replace_a_runtime() {
    let Some((device, _queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let log = CallLog::new();
    let mut instance = instance(&device);
    assert!(instance.take_runtime().is_none());
    assert!(instance.step_runtime(0.1).events.is_empty());

    assert!(
        instance
            .set_runtime(labelled_runtime(&log, "first"))
            .is_none()
    );
    let _ = instance.step_runtime(0.1);
    assert_eq!(log.count("first:step"), 1);

    let replaced = instance
        .set_runtime(labelled_runtime(&log, "second"))
        .expect("the first runtime comes back");
    assert_eq!(replaced.plugin_count(), 1);
    let _ = instance.step_runtime(0.1);
    assert_eq!(log.count("first:step"), 1, "the replaced runtime stopped");
    assert_eq!(log.count("second:step"), 1);

    let taken = instance.take_runtime().expect("the second runtime");
    assert_eq!(taken.plugin_count(), 1);
    assert!(instance.runtime_mut().is_none());
    let _ = instance.step_runtime(0.1);
    assert_eq!(
        log.count("second:step"),
        1,
        "a taken runtime is not stepped"
    );
}

/// A runtime moved off an instance and attached again initialises its GPU
/// plugins again before their next `pre_prepare`.
#[test]
fn a_reattached_runtime_initialises_its_gpu_plugins_again() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    let log = CallLog::new();
    let mut instance = instance(&device);
    instance.set_runtime(
        ViewportRuntime::new()
            .with_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE)),
    );
    render(&mut instance, &device, &queue, &target);
    let runtime = instance.take_runtime().expect("runtime");
    render(&mut instance, &device, &queue, &target);
    instance.set_runtime(runtime);
    render(&mut instance, &device, &queue, &target);
    let pre: Vec<String> = log
        .entries()
        .into_iter()
        .filter(|e| !e.starts_with("post_paint"))
        .collect();
    assert_eq!(
        pre,
        ["init_gpu", "pre_prepare:0", "init_gpu", "pre_prepare:1"]
    );
}

/// `install_plugin` passes the attached runtime and the instance's renderer,
/// so a feature spanning both lands in one call, and `uninstall_plugin`
/// takes it out again. Without a runtime, a feature that needs one fails
/// before registering anything.
#[test]
fn install_through_the_instance_with_and_without_a_runtime() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let log = CallLog::new();
    let mut instance = instance(&device);

    let missing =
        instance.install_plugin(DeformAndStepInstaller::new(log.clone()), &device, &queue);
    assert!(matches!(
        missing,
        Err(ViewportError::PluginInstallMissing { .. })
    ));
    assert!(
        instance
            .resources()
            .deformer_id_by_name(ConstantOffsetDeformer::NAME)
            .is_none(),
        "a missing runtime leaves the renderer untouched"
    );

    instance.set_runtime(ViewportRuntime::new());
    let handle =
        match instance.install_plugin(DeformAndStepInstaller::new(log.clone()), &device, &queue) {
            Ok(handle) => handle,
            Err(err) => {
                eprintln!("skipping: this device cannot register the feature: {err}");
                return;
            }
        };
    assert_eq!(
        instance
            .resources()
            .deformer_id_by_name(ConstantOffsetDeformer::NAME),
        Some(handle.deformer)
    );
    let _ = instance.step_runtime(0.1);
    assert_eq!(log.count("installed_step:step"), 1);

    instance
        .uninstall_plugin::<DeformAndStepInstaller>(handle, &device, &queue)
        .expect("uninstall");
    let _ = instance.step_runtime(0.1);
    assert_eq!(
        log.count("installed_step:step"),
        1,
        "the uninstalled plugin is not stepped"
    );
    assert!(
        instance
            .resources()
            .deformer_id_by_name(ConstantOffsetDeformer::NAME)
            .is_none()
    );
}

/// `rebuild` keeps the attached runtime, and its GPU plugins run `init_gpu`
/// again on the new device before the first render's `pre_prepare` there.
#[test]
fn rebuild_initialises_gpu_plugins_on_the_new_device() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let Some((new_device, new_queue)) = self::device() else {
        panic!("no adapter for the rebuilt device");
    };
    let log = CallLog::new();
    let mut instance = instance(&device).with_runtime(
        ViewportRuntime::new()
            .with_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE)),
    );
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    render(&mut instance, &device, &queue, &target);

    instance.rebuild(&new_device).expect("rebuild");
    assert!(
        instance.runtime_mut().is_some(),
        "rebuild keeps the runtime"
    );
    let new_target = OffscreenViewportTarget::new(&new_device, SURFACE_FORMAT, [SIZE, SIZE]);
    render(&mut instance, &new_device, &new_queue, &new_target);
    let pre: Vec<String> = log
        .entries()
        .into_iter()
        .filter(|e| !e.starts_with("post_paint"))
        .collect();
    assert_eq!(
        pre,
        ["init_gpu", "pre_prepare:0", "init_gpu", "pre_prepare:1"]
    );
}

/// A skeleton feature's `SkinnedPoseUpdate`, read from `step_runtime`'s
/// output, reaches the feature's handle: it uploads the mesh's weights and
/// attaches its joint palette.
#[test]
fn skinned_pose_updates_from_step_runtime_reach_the_skinning_handle() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let mut instance = instance(&device).with_runtime(ViewportRuntime::new());
    let cube = primitives::cube(1.0);
    let mesh = instance.upload_mesh(&device, &cube).expect("mesh");
    let vertices = cube.positions.len();
    let weights = SkinWeights {
        joint_indices: vec![[0, 0, 0, 0]; vertices],
        joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; vertices],
    };
    let skeleton = Skeleton::new(vec![Joint {
        name: "root".into(),
        parent: None,
        inverse_bind: glam::Affine3A::IDENTITY,
    }]);
    let pose_plugin = SkeletonPlugin::new(
        skeleton,
        mesh,
        cube.positions.clone(),
        cube.normals.clone(),
        weights.clone(),
    )
    .with_path(SkinningPath::Gpu);
    let mut handle = match instance.install_plugin(
        SkinnedMeshFeature::new(pose_plugin).with_weights(mesh, weights),
        &device,
        &queue,
    ) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("skipping: this device cannot install skinning: {err}");
            return;
        }
    };
    instance
        .runtime_mut()
        .expect("runtime")
        .resources_mut()
        .insert(Pose::identity(1));
    let slot = handle.skinning().deformer_id().slot();
    assert!(!handle.skinning().is_skinned_mesh(mesh));
    assert!(!instance.resources().has_deform_slot_instance(mesh, 0, slot));

    let mut output = instance.step_runtime(1.0 / 60.0);
    handle.apply(instance.resources_mut(), &device, &queue, &mut output);

    assert!(
        handle.skinning().is_skinned_mesh(mesh),
        "the update uploaded the mesh's weights"
    );
    assert!(
        instance.resources().has_deform_slot_instance(mesh, 0, slot),
        "the update attached the joint palette"
    );
}

/// With camera follow set, `step_runtime`'s output carries the followed
/// node's position as a `CameraFollowTarget`.
#[test]
fn step_runtime_output_carries_the_camera_follow_target() {
    let Some((device, _queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let mut instance = instance(&device);
    let material = instance
        .resources_mut()
        .upload_material(Material::default())
        .expect("material");
    let at = glam::Vec3::new(1.0, 2.0, 3.0);
    let node = instance
        .scene_mut()
        .add(None, glam::Mat4::from_translation(at), material);
    instance.set_runtime(
        ViewportRuntime::new().with_camera_follow(CameraFollow::Node {
            id: node,
            offset: glam::Vec3::new(0.0, 0.0, 1.0),
            look_at: false,
        }),
    );
    let output = instance.step_runtime(1.0 / 60.0);
    let target = output
        .events
        .read::<CameraFollowTarget>()
        .next()
        .expect("a camera follow target");
    assert!(
        (target.0 - (at + glam::Vec3::Z)).length() < 1e-5,
        "target {:?}",
        target.0
    );
}
