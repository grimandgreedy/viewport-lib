//! Integration tests for runtime camera command writeback.

use viewport_lib::camera::camera::{Camera, CameraTarget};
use viewport_lib::runtime::{
    CameraCommand, RuntimeFrameContext, RuntimeOutput, RuntimePlugin, RuntimeStepContext,
    ViewportRuntime, plugin::phase,
};
use viewport_lib::interaction::input::ActionFrame;
use viewport_lib::interaction::selection::Selection;
use viewport_lib::scene::scene::Scene;

fn make_frame(camera: &Camera, input: &ActionFrame) -> RuntimeFrameContext {
    let mut frame = RuntimeFrameContext::default();
    frame.dt = 1.0 / 60.0;
    frame.camera = camera.clone();
    frame.viewport_size = glam::Vec2::new(800.0, 600.0);
    frame.input = input.clone();
    frame
}

fn step(runtime: &mut ViewportRuntime, camera: &Camera) -> RuntimeOutput {
    let input = ActionFrame::default();
    let mut scene = Scene::new();
    let mut sel = Selection::new();
    runtime.step(&mut scene, &mut sel, &make_frame(camera, &input))
}

// --- Plugins -----------------------------------------------------------------

struct SetCenterPlugin(glam::Vec3);

impl RuntimePlugin for SetCenterPlugin {
    fn priority(&self) -> i32 { phase::ANIMATE }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        ctx.output.camera_commands.push(CameraCommand::SetCenter(self.0));
    }
}

struct OffsetCenterPlugin(glam::Vec3);

impl RuntimePlugin for OffsetCenterPlugin {
    fn priority(&self) -> i32 { phase::POST_SIM }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        ctx.output.camera_commands.push(CameraCommand::OffsetCenter(self.0));
    }
}

struct SetDistancePlugin(f32);

impl RuntimePlugin for SetDistancePlugin {
    fn priority(&self) -> i32 { phase::ANIMATE }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        ctx.output.camera_commands.push(CameraCommand::SetDistance(self.0));
    }
}

struct BlendPlugin { target: CameraTarget, weight: f32 }

impl RuntimePlugin for BlendPlugin {
    fn priority(&self) -> i32 { phase::ANIMATE }
    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        ctx.output.camera_commands.push(CameraCommand::BlendToward {
            target: self.target,
            weight: self.weight,
        });
    }
}

// --- Tests -------------------------------------------------------------------

#[test]
fn set_center_applies_to_camera() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(SetCenterPlugin(glam::Vec3::new(1.0, 2.0, 3.0)));
    let camera = Camera::default();
    let output = step(&mut runtime, &camera);

    let mut cam = camera.clone();
    output.apply_camera_commands(&mut cam);
    assert!((cam.center - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
}

#[test]
fn offset_center_adds_to_current_position() {
    let mut camera = Camera::default();
    camera.center = glam::Vec3::new(5.0, 0.0, 0.0);

    let mut runtime = ViewportRuntime::new()
        .with_plugin(OffsetCenterPlugin(glam::Vec3::new(1.0, 0.0, 0.0)));
    let output = step(&mut runtime, &camera);

    output.apply_camera_commands(&mut camera);
    assert!((camera.center.x - 6.0).abs() < 1e-5);
}

#[test]
fn set_distance_applies() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(SetDistancePlugin(10.0));
    let camera = Camera::default();
    let output = step(&mut runtime, &camera);

    let mut cam = camera.clone();
    output.apply_camera_commands(&mut cam);
    assert!((cam.distance - 10.0).abs() < 1e-5);
}

#[test]
fn blend_toward_interpolates() {
    let target = CameraTarget {
        center: glam::Vec3::new(10.0, 0.0, 0.0),
        distance: 20.0,
        orientation: glam::Quat::IDENTITY,
    };
    let mut runtime = ViewportRuntime::new()
        .with_plugin(BlendPlugin { target, weight: 0.5 });

    let mut camera = Camera::default();
    camera.center = glam::Vec3::ZERO;
    camera.distance = 4.0;

    let output = step(&mut runtime, &camera);
    output.apply_camera_commands(&mut camera);

    // 50% blend: center should be halfway to (10,0,0)
    assert!((camera.center.x - 5.0).abs() < 1e-4);
    // distance: 4 + (20 - 4) * 0.5 = 12
    assert!((camera.distance - 12.0).abs() < 1e-4);
}

#[test]
fn blend_at_weight_one_snaps_to_target() {
    let target = CameraTarget {
        center: glam::Vec3::new(7.0, 8.0, 9.0),
        distance: 5.0,
        orientation: glam::Quat::IDENTITY,
    };
    let mut runtime = ViewportRuntime::new()
        .with_plugin(BlendPlugin { target, weight: 1.0 });

    let camera = Camera::default();
    let output = step(&mut runtime, &camera);

    let mut cam = camera.clone();
    output.apply_camera_commands(&mut cam);
    assert!((cam.center - glam::Vec3::new(7.0, 8.0, 9.0)).length() < 1e-5);
    assert!((cam.distance - 5.0).abs() < 1e-5);
}

#[test]
fn commands_applied_in_emission_order() {
    // SetCenter at ANIMATE sets center to (10,0,0).
    // OffsetCenter at POST_SIM adds (1,0,0).
    // Result must be (11,0,0), not (1,0,0) or (10,0,0).
    let mut runtime = ViewportRuntime::new()
        .with_plugin(SetCenterPlugin(glam::Vec3::new(10.0, 0.0, 0.0)))
        .with_plugin(OffsetCenterPlugin(glam::Vec3::new(1.0, 0.0, 0.0)));

    let camera = Camera::default();
    let output = step(&mut runtime, &camera);
    assert_eq!(output.camera_commands.len(), 2);

    let mut cam = camera.clone();
    output.apply_camera_commands(&mut cam);
    assert!((cam.center.x - 11.0).abs() < 1e-5);
}

#[test]
fn no_commands_leaves_camera_unchanged() {
    let mut runtime = ViewportRuntime::new();
    let camera = Camera::default();
    let original_center = camera.center;
    let original_distance = camera.distance;
    let output = step(&mut runtime, &camera);

    let mut cam = camera.clone();
    output.apply_camera_commands(&mut cam);
    assert_eq!(cam.center, original_center);
    assert_eq!(cam.distance, original_distance);
}

#[test]
fn camera_commands_cleared_each_frame() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(SetCenterPlugin(glam::Vec3::ONE));
    let camera = Camera::default();
    step(&mut runtime, &camera);
    let output2 = step(&mut runtime, &camera);
    // Each frame emits exactly one command.
    assert_eq!(output2.camera_commands.len(), 1);
}

#[test]
fn camera_follow_target_unaffected_by_commands() {
    // camera_follow_target and camera_commands are independent.
    let mut runtime = ViewportRuntime::new()
        .with_plugin(SetCenterPlugin(glam::Vec3::ONE));
    let camera = Camera::default();
    let output = step(&mut runtime, &camera);
    assert!(output.camera_follow_target.is_none());
    assert_eq!(output.camera_commands.len(), 1);
}

#[test]
fn existing_output_fields_unaffected() {
    let mut runtime = ViewportRuntime::new()
        .with_plugin(SetCenterPlugin(glam::Vec3::ZERO));
    let camera = Camera::default();
    let output = step(&mut runtime, &camera);
    assert!(output.contact_events.is_empty());
    assert!(output.selection_ops.is_empty());
    assert!(output.node_transform_ops.is_empty());
    assert!(output.events.is_empty());
}

#[test]
fn set_orientation_applies() {
    let q = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    struct SetOrientationPlugin(glam::Quat);
    impl RuntimePlugin for SetOrientationPlugin {
        fn priority(&self) -> i32 { phase::ANIMATE }
        fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
            ctx.output.camera_commands.push(CameraCommand::SetOrientation(self.0));
        }
    }

    let mut runtime = ViewportRuntime::new().with_plugin(SetOrientationPlugin(q));
    let camera = Camera::default();
    let output = step(&mut runtime, &camera);

    let mut cam = camera.clone();
    output.apply_camera_commands(&mut cam);
    assert!(cam.orientation.dot(q).abs() > 0.9999);
}
