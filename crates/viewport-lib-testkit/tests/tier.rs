//! Device tiers: which tier a device class lands on, what the renderer does
//! there, and that the capability reads replacing platform checks build valid
//! pipelines.
//!
//! The profiles model a device class on whatever adapter runs the test: wgpu
//! enforces requested limits, and the renderer reads the downlevel flags the
//! profile hands it.

use viewport_lib::plugin_api::ItemTypePlugin;
use viewport_lib::{Capabilities, Requirements, Tier, ViewportError, ViewportRenderer, wgpu};
use viewport_lib_testkit::{DeviceProfile, Harness, catalogue, frame_for, headless_device_full};

const W: u32 = 160;
const H: u32 = 120;

#[test]
fn webgl2_profile_is_overlay_only_and_construction_says_so() {
    let Some(d) = headless_device_full(&DeviceProfile::downlevel_webgl2("tier-webgl2")) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let caps = Capabilities::from_device(&d.device, Some(&d.downlevel));
    assert_eq!(caps.tier(), Tier::OverlayOnly);
    assert!(!caps.compute);
    assert!(!caps.vertex_storage && !caps.fragment_storage);
    assert!(!caps.cube_array_textures);

    // Without the adapter's flags the zero compute limits still rule compute
    // out, so the tier does not depend on the flags being passed.
    assert_eq!(
        Capabilities::from_device(&d.device, None).tier(),
        Tier::OverlayOnly
    );

    let config = viewport_lib::RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT)
        .with_downlevel_capabilities(d.downlevel);
    match ViewportRenderer::try_with_config(&d.device, &config) {
        Err(ViewportError::DeviceUnsupported { reason }) => {
            assert!(reason.contains("overlay-only"), "{reason}");
            assert!(reason.contains("compute shaders"), "{reason}");
        }
        Err(other) => panic!("expected DeviceUnsupported, got {other}"),
        Ok(_) => panic!("the renderer built on a WebGL2-limits device"),
    }
}

#[test]
fn iced_two_bind_groups_build_on_the_full_tier() {
    let profile = DeviceProfile::harness().max_bind_groups(2);
    let Some(harness) = Harness::with_profile(&profile) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), Tier::Full);
    assert!(harness.renderer.capabilities().downlevel_known);
}

/// Render every catalogue scene on `profile`. A pipeline whose shader does not
/// match its layout is a validation error, which panics the test.
fn render_catalogue(profile: &DeviceProfile) -> Option<Harness> {
    let mut harness = Harness::with_profile(profile)?;
    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        let _ = harness.render(&frame, W, H);
    }
    Some(harness)
}

#[test]
fn every_scene_builds_without_cube_array_textures() {
    let profile = DeviceProfile::high_performance("tier-no-cube-arrays")
        .with_recommended_features()
        .without_downlevel(wgpu::DownlevelFlags::CUBE_ARRAY_TEXTURES);
    let Some(harness) = render_catalogue(&profile) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(!harness.renderer.capabilities().cube_array_textures);
    assert_eq!(harness.renderer.tier(), Tier::Full);
}

#[test]
fn every_scene_builds_without_indirect_execution() {
    let profile = DeviceProfile::high_performance("tier-no-indirect")
        .with_recommended_features()
        .without_downlevel(wgpu::DownlevelFlags::INDIRECT_EXECUTION);
    let Some(harness) = render_catalogue(&profile) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(!harness.renderer.capabilities().indirect_execution);
}

#[test]
fn the_tier_reaches_frame_stats() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = &catalogue()[0];
    let built = harness.build_scene(scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let stats = harness.render_two_frames(&frame, W, H);
    assert_eq!(stats.tier, Tier::Full);
}

/// An item type that asks for more storage buffers than any device grants.
struct Greedy;

impl ItemTypePlugin for Greedy {
    fn type_name(&self) -> &'static str {
        "tier_test.greedy"
    }
    fn requires(&self) -> Requirements {
        Requirements::full_tier().with_storage_buffers_per_stage(u32::MAX)
    }
    fn init_gpu(&mut self, _: &wgpu::Device, _: &viewport_lib::plugin_api::SharedBindings<'_>) {
        panic!("init_gpu ran for a declined plugin");
    }
}

#[test]
fn a_plugin_the_device_cannot_run_is_declined() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    match harness
        .renderer
        .with_item_type_plugin(&harness.device, Box::new(Greedy))
    {
        Err(ViewportError::PluginUnsupported { type_name, missing }) => {
            assert_eq!(type_name, "tier_test.greedy");
            assert_eq!(missing.len(), 1, "{missing:?}");
        }
        other => panic!("expected PluginUnsupported, got {other:?}"),
    }
    assert!(!harness.renderer.has_item_type_plugin("tier_test.greedy"));
}

struct GreedyGpu;

impl viewport_lib::runtime::GpuPlugin for GreedyGpu {
    fn type_name(&self) -> &'static str {
        "tier_test.greedy_gpu"
    }
    fn requires(&self) -> Requirements {
        Requirements::full_tier().with_bind_groups(u32::MAX)
    }
    fn init_gpu(&mut self, _: &wgpu::Device) {
        panic!("init_gpu ran for a declined GPU plugin");
    }
}

#[test]
fn a_gpu_plugin_the_device_cannot_run_is_declined() {
    let Some(harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut runtime = viewport_lib::ViewportRuntime::new().with_gpu_plugin(GreedyGpu);
    let camera = viewport_lib::Camera::default();
    let ctx = viewport_lib::runtime::GpuFrameContext::new(&camera, glam::Vec2::ONE, 0.0, 0);
    let _ = runtime.pre_prepare(&harness.device, &harness.queue, &ctx);
    assert!(matches!(
        runtime.declined_gpu_plugins(),
        [ViewportError::PluginUnsupported { type_name, .. }] if type_name == "tier_test.greedy_gpu"
    ));
}
