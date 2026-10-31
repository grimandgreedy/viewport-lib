//! Device tiers: which tier a device class lands on, what the renderer does
//! there, and that the capability reads replacing platform checks build valid
//! pipelines.
//!
//! The profiles model a device class on whatever adapter runs the test: wgpu
//! enforces requested limits, and the renderer reads the downlevel flags the
//! profile hands it.

use viewport_lib::plugin_api::ItemTypePlugin;
use viewport_lib::{Capabilities, Requirements, Tier, ViewportError, ViewportRenderer, wgpu};
use viewport_lib_testkit::scenes::overlays;
use viewport_lib_testkit::{DeviceProfile, Harness, catalogue, frame_for, headless_device_full};

const W: u32 = 160;
const H: u32 = 120;

#[test]
fn webgl2_profile_builds_a_forward_renderer() {
    let Some(d) = headless_device_full(&DeviceProfile::downlevel_webgl2("tier-webgl2")) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let caps = Capabilities::from_device(&d.device, Some(&d.downlevel));
    assert_eq!(caps.tier(), Tier::Compatibility);
    assert!(!caps.compute);
    assert!(!caps.vertex_storage && !caps.fragment_storage);
    assert!(!caps.cube_array_textures);

    // Without the adapter's flags the zero compute limits still rule compute
    // out, so the tier does not depend on the flags being passed.
    assert_eq!(
        Capabilities::from_device(&d.device, None).tier(),
        Tier::Compatibility
    );

    // Construction builds nothing the device cannot create: a layout over its
    // limits would fail validation here, and validation failures panic.
    let config = viewport_lib::RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT)
        .with_downlevel_capabilities(d.downlevel);
    let mut renderer = ViewportRenderer::try_with_config(&d.device, &config)
        .expect("a compatibility renderer builds on WebGL2 limits");
    assert_eq!(renderer.tier(), Tier::Compatibility);
    assert_scene_calls_refused(&mut renderer, &d.device, &d.queue);
}

/// WebGL2's limits with two bind groups: below the compatibility tier, so only the
/// overlay family builds.
fn overlay_only_profile() -> DeviceProfile {
    DeviceProfile::downlevel_webgl2("tier-overlay-only").max_bind_groups(2)
}

#[test]
fn two_bind_groups_build_an_overlay_only_renderer() {
    let Some(d) = headless_device_full(&overlay_only_profile()) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let config = viewport_lib::RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT)
        .with_downlevel_capabilities(d.downlevel);
    let mut renderer = ViewportRenderer::try_with_config(&d.device, &config)
        .expect("an overlay-only renderer builds on WebGL2 limits with two bind groups");
    assert_eq!(renderer.tier(), Tier::OverlayOnly);
    assert_scene_calls_refused(&mut renderer, &d.device, &d.queue);
    // No item type registers without a lit group 0, whatever it asks for.
    match renderer.with_item_type_plugin(&d.device, Box::new(Modest)) {
        Err(ViewportError::TierUnsupported {
            needed: Tier::Compatibility,
            have: Tier::OverlayOnly,
        }) => {}
        other => panic!("expected TierUnsupported, got {other:?}"),
    }
    assert!(!renderer.has_item_type_plugin("tier_test.modest"));
}

/// One bind group is below the overlay floor: the fallible constructors return
/// `DeviceUnsupported` instead of panicking.
#[test]
fn one_bind_group_is_refused_without_a_panic() {
    let profile = DeviceProfile::downlevel_webgl2("tier-unsupported").max_bind_groups(1);
    let Some(d) = headless_device_full(&profile) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let config = viewport_lib::RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT)
        .with_downlevel_capabilities(d.downlevel);
    match ViewportRenderer::try_with_config(&d.device, &config) {
        Err(ViewportError::DeviceUnsupported { .. }) => {}
        other => panic!("expected DeviceUnsupported, got {:?}", other.err()),
    }
    match viewport_lib::ViewportInstance::try_with_config(&d.device, &config) {
        Err(ViewportError::DeviceUnsupported { .. }) => {}
        other => panic!("expected DeviceUnsupported, got {:?}", other.err()),
    }
}

/// The calls that add scene content say why they cannot.
fn assert_scene_calls_refused(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let have = renderer.tier();
    assert!(have < Tier::Full);
    // Meshes and textures upload wherever there is a lit group 0 to draw
    // them with: the compatibility tier's basic mesh path.
    let mesh = viewport_lib::geometry::primitives::cube(1.0);
    let mesh = renderer.resources_mut().upload_mesh_data(device, &mesh);
    let image = viewport_lib::TextureData::srgb(1, 1, vec![255; 4]);
    let texture = renderer
        .resources_mut()
        .upload_texture(device, queue, image);
    if have == Tier::Compatibility {
        mesh.expect("meshes upload on the compatibility tier");
        texture.expect("textures upload on the compatibility tier");
    } else {
        for e in [
            mesh.map(|_| ()).unwrap_err(),
            texture.map(|_| ()).unwrap_err(),
        ] {
            assert!(
                matches!(
                    e,
                    ViewportError::TierUnsupported { needed: Tier::Compatibility, have: h } if h == have
                ),
                "expected TierUnsupported, got {e}"
            );
        }
    }

    // Matcaps draw on the full tier only: the built-ins are a no-op to
    // initialise and an upload says why it is refused.
    renderer
        .resources_mut()
        .ensure_matcaps_initialized(device, queue);
    let matcap = viewport_lib::TextureData::linear(256, 256, vec![255; 256 * 256 * 4]);
    match renderer
        .resources_mut()
        .upload_matcap(device, queue, matcap, false)
    {
        Err(ViewportError::TierUnsupported {
            needed: Tier::Full,
            have: h,
        }) if h == have => {}
        other => panic!("expected TierUnsupported, got {other:?}"),
    }

    // The renderer's own getters answer rather than reaching for resources
    // that were never built.
    assert_eq!(renderer.shadow_allocation_bytes(), 0);
    let _ = renderer.resident_bytes();
    assert!(renderer.cluster_stats().is_none());
    let _ = renderer.shadow_debug_stats();
    let _ = renderer.feature_set_violations();
    assert_eq!(renderer.instanced_batch_count(), 0);
    renderer.set_occlusion_culling(true);
    renderer.enable_gpu_driven_culling();
    assert_eq!(renderer.resources().registered_deformer_count(), 0);
}

/// A device with storage buffers and no compute. It lands on the compatibility tier,
/// and its overlay layouts build.
fn no_compute_profile() -> DeviceProfile {
    DeviceProfile::harness().without_downlevel(wgpu::DownlevelFlags::COMPUTE_SHADERS)
}

#[test]
fn no_compute_draws_overlays_as_the_full_tier_does() {
    assert_overlays_match_the_full_tier(&no_compute_profile());
}

#[test]
fn overlay_only_draws_overlays_as_the_full_tier_does() {
    assert_overlays_match_the_full_tier(&overlay_only_profile());
}

/// The overlay family binds no storage buffer and no compute, so a device with
/// WebGL2's limits draws every overlay scene exactly as the full tier does.
#[test]
fn webgl2_draws_overlays_as_the_full_tier_does() {
    assert_overlays_match_the_full_tier(&DeviceProfile::downlevel_webgl2("tier-webgl2-overlays"));
}

/// Render every overlay scene on `profile` and on the full tier's LDR path, and
/// require the same pixels.
fn assert_overlays_match_the_full_tier(profile: &DeviceProfile) {
    let Some(mut full) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some(mut overlay) = Harness::with_profile(profile) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_scene_calls_refused(&mut overlay.renderer, &overlay.device, &overlay.queue);
    // None of the renderer's own item types accepts a tier below the full one
    // yet.
    assert!(
        !overlay
            .renderer
            .has_item_type_plugin("viewport_lib.point_cloud")
    );

    // Every overlay scene, then the shapes again with a filtered backdrop
    // blur, which draws through its own pass.
    let blurred = overlays::scenes()
        .into_iter()
        .find(|s| s.name == "overlay_shapes")
        .unwrap();
    let cases = overlays::scenes()
        .into_iter()
        .map(|s| (s, false))
        .chain([(blurred, true)]);
    for (scene, blur) in cases {
        let camera = &scene.cameras[0].camera;
        let with_blur = |frame: &mut viewport_lib::FrameData| {
            if blur {
                for shape in &mut frame.overlays.shapes {
                    shape.style.backdrop = viewport_lib::BackdropEffects::default()
                        .with_blur(4.0)
                        .with_filters(1.0, 0.25, 0.0);
                }
            }
        };
        // The overlay scenes draw over a lit backdrop, which the overlay-only
        // renderer leaves out and the compatibility tier lights differently; leave
        // it out of both.
        let mut built = full.build_scene(&scene);
        built.items.clear();
        let mut frame = frame_for(&built, camera, [W as f32, H as f32]);
        with_blur(&mut frame);
        // The full tier on its LDR path is what the overlay-only renderer draws.
        frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        let want = full.render(&frame, W, H);
        assert!(
            want.chunks_exact(4).any(|p| p != &want[..4]),
            "{}: the reference draws nothing",
            scene.name
        );

        let mut built = overlay.build_scene(&scene);
        built.items.clear();
        let mut frame = frame_for(&built, camera, [W as f32, H as f32]);
        with_blur(&mut frame);
        let got = overlay.render(&frame, W, H);
        let stats = overlay.stats();
        assert_eq!(stats.items_skipped_by_tier, 0, "{}", scene.name);
        assert!(stats.hdr_drawn_as_ldr, "{}", scene.name);
        let differing = want
            .chunks_exact(4)
            .zip(got.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differing, 0, "{}: {differing} pixels differ", scene.name);
    }
}

#[test]
fn overlay_only_skips_scene_content_and_counts_it() {
    let Some(mut full) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some(mut overlay) = Harness::with_profile(&overlay_only_profile()) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Items naming meshes the full renderer holds: the overlay-only renderer
    // has none, and draws the frame without them.
    let scene = &catalogue()[0];
    let built = full.build_scene(scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    assert!(!built.items.is_empty());
    let pixels = overlay.render(&frame, W, H);
    let stats = overlay.stats();
    assert_eq!(stats.items_skipped_by_tier as usize, built.items.len());
    assert_eq!(stats.tier, Tier::OverlayOnly);
    // Nothing but the clear colour.
    assert!(pixels.chunks_exact(4).all(|p| p == &pixels[..4]));
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

/// Without direct depth sampling (wgpu's GL backend) plugins sample a copy of
/// the scene depth: the depth-read pass and the encoder scopes. Every scene
/// builds that way, and the soft sprites and decals, which read the copy, draw
/// the same pixels as from the scene depth itself.
#[test]
fn every_scene_builds_without_direct_depth_sampling() {
    let profile = DeviceProfile::high_performance("tier-no-direct-depth")
        .with_recommended_features()
        .without_downlevel(wgpu::DownlevelFlags::READ_ONLY_DEPTH_STENCIL);
    let Some(mut copied) = render_catalogue(&profile) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(!copied.renderer.capabilities().direct_depth_sampling);
    let Some(mut direct) = Harness::with_profile(
        &DeviceProfile::high_performance("tier-direct-depth").with_recommended_features(),
    ) else {
        return;
    };
    // A device without it (GL) has no direct read to compare against.
    if !direct.renderer.capabilities().direct_depth_sampling {
        return;
    }
    for name in ["sprites_soft", "decals"] {
        let scene = catalogue()
            .into_iter()
            .find(|s| s.name == name)
            .expect("the scene is in the catalogue");
        let mut draw = |harness: &mut Harness| {
            let built = harness.build_scene(&scene);
            let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
            let _ = harness.render(&frame, W, H);
            harness.render(&frame, W, H)
        };
        assert!(
            draw(&mut copied) == draw(&mut direct),
            "{name} differs when it reads a copy of the scene depth"
        );
    }
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
    fn init_gpu(&mut self, _: &wgpu::Device, _: &dyn viewport_lib::plugin_api::PluginResources) {
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

/// An item type that asks for nothing.
struct Modest;

impl ItemTypePlugin for Modest {
    fn type_name(&self) -> &'static str {
        "tier_test.modest"
    }
    fn requires(&self) -> Requirements {
        Requirements::none()
    }
    fn init_gpu(&mut self, _: &wgpu::Device, _: &dyn viewport_lib::plugin_api::PluginResources) {
        panic!("init_gpu ran on an overlay-only renderer");
    }
}

/// An item type that accepts the compatibility tier and, in `init_gpu`, builds an
/// LDR pipeline lit through the shared scene-lighting helper with the plugin
/// pipeline builder, recording any validation error.
#[derive(Default)]
struct ForwardLit {
    built: Option<Result<(), String>>,
    /// Whether the resources offered the read-only-depth layout and the
    /// material fallbacks, which only the full tier has.
    full_tier_extras: Option<bool>,
}

const FORWARD_LIT_BODY: &str = "
@group(0) @binding(0) var<uniform> test_view_proj: mat4x4<f32>;
@group(0) @binding(3) var<uniform> test_lights: Lights;

struct TestOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> TestOut {
    var out: TestOut;
    out.clip = test_view_proj * vec4<f32>(position, 1.0);
    out.world = position;
    return out;
}

@fragment
fn fs_main(in: TestOut) -> @location(0) vec4<f32> {
    let lit = apply_scene_lighting(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0), false, in.world, test_lights);
    return vec4<f32>(lit, 1.0);
}
";

impl ItemTypePlugin for ForwardLit {
    fn type_name(&self) -> &'static str {
        "tier_test.forward_lit"
    }
    fn requires(&self) -> Requirements {
        Requirements::compatibility_tier()
    }
    fn init_gpu(
        &mut self,
        device: &wgpu::Device,
        resources: &dyn viewport_lib::plugin_api::PluginResources,
    ) {
        let source = format!(
            "{}\n{}",
            viewport_lib::plugin_api::shared_wgsl::SHARED_SCENE_LIGHTING_WGSL,
            FORWARD_LIT_BODY
        );
        self.full_tier_extras = Some(
            resources.depth_read_bind_group_layout().is_some()
                && resources.depth_read_sampler().is_some()
                && resources
                    .fallback_texture_view(viewport_lib::TextureSlot::Albedo)
                    .is_some(),
        );
        let scope = viewport_lib::gpu::push_error_scope(device, wgpu::ErrorFilter::Validation);
        let module = resources.lazy_module(device, "forward_lit", &source);
        let vertex_layouts = [wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }];
        let _ = resources.build_ldr_opaque_pipeline(
            device,
            &viewport_lib::plugin_api::PluginPipelineOpts::new(
                Some("forward_lit"),
                module.get(),
                "vs_main",
                "fs_main",
                &vertex_layouts,
            ),
        );
        self.built = Some(match pollster::block_on(scope.pop()) {
            None => Ok(()),
            Some(e) => Err(e.to_string()),
        });
    }
}

/// On the compatibility tier an item type that accepts it registers and builds a lit
/// pipeline against the compatibility group 0; one that needs the full tier is
/// declined. The same plugin builds on the full tier too.
#[test]
fn compatibility_tier_registers_item_types_that_accept_it() {
    for profile in [
        DeviceProfile::downlevel_webgl2("tier-compatibility-plugin"),
        DeviceProfile::harness(),
    ] {
        let Some(mut harness) = Harness::with_profile(&profile) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let tier = harness.renderer.tier();
        harness
            .renderer
            .with_item_type_plugin(&harness.device, Box::new(ForwardLit::default()))
            .unwrap_or_else(|e| panic!("{tier}: {e}"));
        let plugin = harness
            .renderer
            .item_type_plugin::<ForwardLit>("tier_test.forward_lit")
            .expect("registered");
        assert_eq!(plugin.built, Some(Ok(())), "{tier}");
        assert_eq!(plugin.full_tier_extras, Some(tier == Tier::Full), "{tier}");

        if tier == Tier::Compatibility {
            match harness
                .renderer
                .with_item_type_plugin(&harness.device, Box::new(Greedy))
            {
                Err(ViewportError::PluginUnsupported { .. }) => {}
                other => panic!("expected PluginUnsupported, got {other:?}"),
            }
            // Of the built-in types, those that draw without storage buffers
            // register here and the rest are declined.
            for (name, registered) in [
                (
                    viewport_lib_plugins::item_types::gpu_implicit::TYPE_NAME,
                    true,
                ),
                (viewport_lib_plugins::item_types::volume::TYPE_NAME, true),
                (
                    viewport_lib_plugins::item_types::curves::TUBE_TYPE_NAME,
                    true,
                ),
                (
                    viewport_lib_plugins::item_types::point_cloud::TYPE_NAME,
                    true,
                ),
                (
                    viewport_lib_plugins::item_types::vector_field::TYPE_NAME,
                    true,
                ),
                (
                    viewport_lib_plugins::item_types::gaussian_splat::TYPE_NAME,
                    true,
                ),
                (viewport_lib_plugins::item_types::sprite::TYPE_NAME, false),
                ("vpl.polyline", true),
            ] {
                assert_eq!(
                    harness.renderer.has_item_type_plugin(name),
                    registered,
                    "{name}"
                );
            }
        }
    }
}

/// An upload to a built-in item type the renderer declined returns the error it
/// was declined with, and the calls with nothing to return do nothing. A
/// renderer that never ran `install` names the missing type.
#[test]
fn uploads_to_a_declined_item_type_return_why_it_was_declined() {
    use viewport_lib::plugin_api::{Handles, Uploads};
    use viewport_lib_plugins::item_types::gpu_particles::{
        GpuParticleSystemConfig, GpuParticleSystems,
    };
    use viewport_lib_plugins::item_types::sprite::{self, SpriteItem, SpriteSetId};

    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("tier-declined-upload"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), Tier::Compatibility);
    let declined = harness
        .renderer
        .item_type_declined(sprite::TYPE_NAME)
        .cloned()
        .expect("sprites are declined on the compatibility tier");

    let mut sprites = SpriteItem::default();
    sprites.positions = vec![[0.0, 0.0, 0.0]; 4];
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let result: Result<SpriteSetId, _> = harness.renderer.upload(&device, &queue, &sprites);
    match result {
        Err(e) => assert_eq!(e.to_string(), declined.to_string()),
        Ok(_) => panic!("a declined type took an upload"),
    }
    assert!(!harness.renderer.release(SpriteSetId::from_index(0)));
    assert!(
        harness
            .renderer
            .create_gpu_particle_system(&device, &queue, &GpuParticleSystemConfig::default())
            .is_err()
    );
    harness.renderer.drop_gpu_particle_system(
        viewport_lib_plugins::item_types::gpu_particles::GpuParticleSystemId::from_index(0),
    );

    let config = viewport_lib::RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT);
    let mut bare = ViewportRenderer::try_with_config(&device, &config).expect("renderer");
    match Uploads::<SpriteItem>::upload(&mut bare, &device, &queue, &sprites) {
        Err(ViewportError::ItemTypePluginMissing { type_name }) => {
            assert_eq!(type_name, sprite::TYPE_NAME)
        }
        other => panic!("expected ItemTypePluginMissing, got {:?}", other.err()),
    }
}

/// The compatibility tier's light list holds `COMPATIBILITY_MAX_LIGHTS`, and the frame says how
/// many it left out. The full tier's holds all of them.
#[test]
fn forward_light_list_caps_and_counts_what_it_drops() {
    let lights = viewport_lib::resources::COMPATIBILITY_MAX_LIGHTS + 4;
    for (profile, dropped) in [
        (
            DeviceProfile::downlevel_webgl2("tier-compatibility-lights"),
            4,
        ),
        (DeviceProfile::harness(), 0),
    ] {
        let Some(mut harness) = Harness::with_profile(&profile) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // Lights and no scene content: the compatibility tier has no meshes to draw
        // yet, and the cap does not depend on what they light.
        let mut frame = viewport_lib::FrameData::new(
            viewport_lib::CameraFrame::from_camera(
                &viewport_lib::Camera::default(),
                [W as f32, H as f32],
            ),
            viewport_lib::SceneFrame::default(),
        );
        // The default lighting carries a light of its own.
        frame.effects.lighting.lights.clear();
        frame.scene.lights = vec![viewport_lib::LightSource::default(); lights];
        let _ = harness.render(&frame, W, H);
        assert_eq!(
            harness.stats().lights_dropped,
            dropped,
            "{}",
            harness.renderer.tier()
        );
    }
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

/// Copy a `W`x`H` four-byte texture back to the CPU, rows padded to the copy
/// alignment.
fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let row = (W * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tier_read_back"),
        size: (row * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    viewport_lib::gpu::mapped_range(slice).to_vec()
}

fn target(device: &wgpu::Device, format: wgpu::TextureFormat, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Prepare through the split API, then draw once into a host-owned pass and
/// once through `render_viewport`. Returns both images.
fn draw_split(harness: &mut Harness, frame: &mut viewport_lib::FrameData) -> (Vec<u8>, Vec<u8>) {
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let r = &mut harness.renderer;
    let id = r.create_viewport(&device);
    frame.camera.viewport_index = id.index();
    let colour = target(&device, Harness::DEFAULT_TARGET_FORMAT, "tier_colour");
    let depth = target(
        &device,
        wgpu::TextureFormat::Depth24PlusStencil8,
        "tier_depth",
    );
    let (colour_view, depth_view) = (
        colour.create_view(&Default::default()),
        depth.create_view(&Default::default()),
    );

    let (scene_fx, _) = frame.effects.split();
    let token = r.pass().prepare_scene(&device, &queue, frame, &scene_fx);
    r.pass()
        .prepare_viewport(&device, &queue, &token, id, frame);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tier_host_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &colour_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(viewport_lib::plugin_api::depth::DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            // The tail fields differ between the wgpu legs.
            ..Default::default()
        });
        r.pass().paint_viewport(&mut rp, id, frame);
    }
    queue.submit([encoder.finish()]);
    let painted = read_back(&device, &queue, &colour);

    let owned = target(&device, Harness::DEFAULT_TARGET_FORMAT, "tier_owned");
    let token = r.owned().prepare_scene(&device, &queue, frame, &scene_fx);
    r.owned()
        .prepare_viewport(&device, &queue, &token, id, frame);
    let cmd = r.owned().render_viewport(
        &device,
        &queue,
        &owned.create_view(&Default::default()),
        id,
        frame,
    );
    queue.submit([cmd]);
    (painted, read_back(&device, &queue, &owned))
}

#[test]
fn overlay_only_draws_through_the_split_and_host_pass_paths() {
    assert_split_paths_match_the_full_tier(&no_compute_profile());
    assert_split_paths_match_the_full_tier(&DeviceProfile::downlevel_webgl2("tier-webgl2-split"));
}

fn assert_split_paths_match_the_full_tier(profile: &DeviceProfile) {
    let Some(mut full) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some(mut overlay) = Harness::with_profile(profile) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = overlays::scenes()
        .into_iter()
        .find(|s| s.name == "overlay_shapes")
        .unwrap();
    let camera = &scene.cameras[0].camera;
    let mut built = full.build_scene(&scene);
    built.items.clear();
    let mut want_frame = frame_for(&built, camera, [W as f32, H as f32]);
    want_frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
    let (want_painted, want_owned) = draw_split(&mut full, &mut want_frame);
    assert!(
        want_painted
            .chunks_exact(4)
            .any(|p| p != &want_painted[..4])
    );

    let mut built = overlay.build_scene(&scene);
    built.items.clear();
    let mut frame = frame_for(&built, camera, [W as f32, H as f32]);
    let (painted, owned) = draw_split(&mut overlay, &mut frame);
    assert!(painted == want_painted, "the host-pass paint differs");
    assert!(owned == want_owned, "render_viewport differs");
}

/// A mesh on the compatibility tier draws through both halves of the split API: into
/// a host-owned pass and through `render_viewport`.
/// Picked faces of a mesh and a picked voxel of a volume are highlighted on the
/// compatibility tier, through the host's own pass and the owned-encoder
/// path alike.
#[test]
fn compatibility_sub_object_highlights_draw_through_the_split_and_host_pass_paths() {
    let Some(mut harness) = Harness::with_profile(&DeviceProfile::downlevel_webgl2(
        "tier-compatibility-highlight",
    )) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let cube = viewport_lib::geometry::primitives::cube(2.0);
    let mesh = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &cube)
        .expect("meshes upload on the compatibility tier");
    let mut item = viewport_lib::SceneRenderItem::default();
    item.mesh_id = mesh;
    item.settings.colour = Some(viewport_lib::Colour::linear_rgb(0.9, 0.3, 0.1));
    let mut camera = viewport_lib::Camera::default();
    camera.set_orbit_distance(6.0);
    camera.set_aspect_ratio(W as f32, H as f32);
    let base = || {
        let mut frame = viewport_lib::FrameData::default();
        frame.camera.render_camera = viewport_lib::RenderCamera::from_camera(&camera);
        frame.camera.viewport_size = [W as f32, H as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.interaction.sub_highlight_face_fill_colour =
            viewport_lib::Colour::linear(0.0, 1.0, 0.0, 1.0);
        frame.interaction.sub_highlight_edge_colour =
            viewport_lib::Colour::linear_rgb(0.0, 1.0, 0.0);
        frame.interaction.sub_highlight_edge_width_px = 3.0;
        frame
    };
    let green = |pixels: &[u8]| {
        // `read_back` keeps the copy's padded rows and the target's BGRA order.
        let row = (W * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize;
        (0..H as usize)
            .flat_map(|y| (0..W as usize).map(move |x| y * row + x * 4))
            .filter(|&i| {
                let [b, g, r] = [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32];
                g > r + 100 && g > b + 100
            })
            .count()
    };

    // Every face of the cube.
    let mut faces = viewport_lib::SubSelection::new();
    for face in 0..(cube.indices.len() / 3) as u32 {
        faces.add(1, viewport_lib::SubObjectRef::Face(face));
    }
    let mut frame = base();
    frame.scene.surfaces = viewport_lib::SurfaceSubmission::Flat(vec![item].into());
    frame.interaction.sub_selection = Some(viewport_lib::SubSelectionRef::new(
        &faces,
        [(1, (cube.positions.clone(), cube.indices.clone()))].into(),
        [(1, glam::Mat4::IDENTITY)].into(),
        Default::default(),
    ));
    let (painted, owned) = draw_split(&mut harness, &mut frame);
    for (name, pixels) in [("paint_viewport", &painted), ("render_viewport", &owned)] {
        let n = green(pixels);
        assert!(
            n > 500,
            "{name}: the picked faces must be filled, {n} green pixels"
        );
    }

    // One voxel of a 4 x 4 x 4 grid, drawn as its box's edges.
    let mut voxels = viewport_lib::SubSelection::new();
    voxels.add(2, viewport_lib::SubObjectRef::Voxel(0));
    let mut frame = base();
    frame.interaction.sub_selection = Some(
        viewport_lib::SubSelectionRef::new(
            &voxels,
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .with_voxels(
            [(
                2,
                viewport_lib::VolumeSelectionInfo {
                    dims: [4, 4, 4],
                    bbox_min: [-1.0, -1.0, -1.0],
                    bbox_max: [1.0, 1.0, 1.0],
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                },
            )]
            .into(),
        ),
    );
    let (painted, owned) = draw_split(&mut harness, &mut frame);
    for (name, pixels) in [("paint_viewport", &painted), ("render_viewport", &owned)] {
        let n = green(pixels);
        assert!(
            n > 30,
            "{name}: the picked voxel must be outlined, {n} green pixels"
        );
    }
}

/// A selected mesh on the compatibility tier is outlined through the host's own
/// pass (`paint_viewport`) and the owned-encoder path (`render_viewport`): the
/// composite draws in both.
#[test]
fn compatibility_outlines_draw_through_the_split_and_host_pass_paths() {
    let Some(mut harness) = Harness::with_profile(&DeviceProfile::downlevel_webgl2(
        "tier-compatibility-outline",
    )) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let cube = viewport_lib::geometry::primitives::cube(2.0);
    let mesh = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &cube)
        .expect("meshes upload on the compatibility tier");
    let mut item = viewport_lib::SceneRenderItem::default();
    item.mesh_id = mesh;
    item.settings.colour = Some(viewport_lib::Colour::linear_rgb(0.9, 0.3, 0.1));
    item.settings.selected = true;
    let mut camera = viewport_lib::Camera::default();
    camera.set_orbit_distance(6.0);
    camera.set_aspect_ratio(W as f32, H as f32);
    let mut frame = viewport_lib::FrameData::default();
    frame.camera.render_camera = viewport_lib::RenderCamera::from_camera(&camera);
    frame.camera.viewport_size = [W as f32, H as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.interaction.outline_selected = true;
    frame.interaction.outline_colour = viewport_lib::Colour::linear_rgb(0.0, 1.0, 0.0);
    frame.interaction.outline_width_px = 3.0;
    frame.scene.surfaces = viewport_lib::SurfaceSubmission::Flat(vec![item].into());

    let (painted, owned) = draw_split(&mut harness, &mut frame);
    // `read_back` keeps the copy's padded rows and the target's BGRA order.
    let row = (W * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize;
    for (name, pixels) in [("paint_viewport", &painted), ("render_viewport", &owned)] {
        let green = (0..H as usize)
            .flat_map(|y| (0..W as usize).map(move |x| y * row + x * 4))
            .filter(|&i| {
                let [b, g, r] = [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32];
                g > r + 100 && g > b + 100
            })
            .count();
        assert!(
            green > 100,
            "{name}: the cube must be outlined, {green} green pixels"
        );
    }
}

#[test]
fn forward_meshes_draw_through_the_split_and_host_pass_paths() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("tier-compatibility-split"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let cube = viewport_lib::geometry::primitives::cube(2.0);
    let mesh = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &cube)
        .expect("meshes upload on the compatibility tier");
    let mut item = viewport_lib::SceneRenderItem::default();
    item.mesh_id = mesh;
    item.settings.colour = Some(viewport_lib::Colour::linear_rgb(0.9, 0.3, 0.1));
    let mut camera = viewport_lib::Camera::default();
    camera.set_orbit_distance(6.0);
    camera.set_aspect_ratio(W as f32, H as f32);
    let mut frame = viewport_lib::FrameData::default();
    frame.camera.render_camera = viewport_lib::RenderCamera::from_camera(&camera);
    frame.camera.viewport_size = [W as f32, H as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = viewport_lib::SurfaceSubmission::Flat(vec![item].into());

    let (painted, owned) = draw_split(&mut harness, &mut frame);
    // `read_back` keeps the copy's padded rows and the target's BGRA order.
    assert_eq!(
        Harness::DEFAULT_TARGET_FORMAT,
        wgpu::TextureFormat::Bgra8UnormSrgb
    );
    let row = (W * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    for (name, pixels) in [("paint_viewport", &painted), ("render_viewport", &owned)] {
        let i = ((H / 2) * row + (W / 2) * 4) as usize;
        let centre = &pixels[i..i + 4];
        assert!(
            centre[2] as i32 > centre[0] as i32 + 40,
            "{name}: the cube must cover the centre, got {centre:?}"
        );
    }
}

/// Every scene item a frame submits that `renderer` cannot draw at its tier,
/// counted the way `FrameStats` counts what it skips: the compatibility tier draws
/// the surfaces and the items of the types registered on it.
fn scene_items(frame: &viewport_lib::FrameData, renderer: &ViewportRenderer) -> usize {
    let tier = renderer.tier();
    let surfaces = match &frame.scene.surfaces {
        viewport_lib::SurfaceSubmission::Flat(items) => items.len(),
        _ => panic!("a surface submission this test does not count"),
    };
    let plugin_items: usize = frame
        .scene
        .plugin_items
        .iter()
        .filter(|(name, _)| tier != Tier::Compatibility || !renderer.has_item_type_plugin(name))
        .flat_map(|(_, collections)| collections)
        .map(|c| c.len())
        .sum();
    let surfaces = if tier == Tier::Compatibility {
        0
    } else {
        surfaces
    };
    surfaces
        + frame.scene.volume_meshes.len()
        + frame.scene.mesh_instances.len()
        + frame.scene.foreground_items.len()
        + plugin_items
}

#[test]
fn overlay_only_renders_every_catalogue_scene_without_reaching_the_scene_tier() {
    sweep_catalogue(&no_compute_profile());
    sweep_catalogue(&DeviceProfile::downlevel_webgl2("tier-webgl2-sweep"));
}

fn sweep_catalogue(profile: &DeviceProfile) {
    // The overlay-only renderer reads the scene tier through a slot that
    // panics when a path is missing its tier check, so this is the sweep for
    // such a path: every scene the catalogue has, scivis types included. The
    // scenes are built on a full renderer, since most of their uploads are refused on
    // a renderer below the full tier, and the frames handed across. Their mesh
    // ids name nothing on the second renderer, so its surfaces draw nothing.
    let Some(mut full) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some(mut overlay) = Harness::with_profile(profile) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(overlay.renderer.tier() < Tier::Full);
    for scene in catalogue() {
        let built = full.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        let _ = overlay.render(&frame, W, H);
        let stats = overlay.stats();
        assert_eq!(
            stats.items_skipped_by_tier as usize,
            scene_items(&frame, &overlay.renderer),
            "{}",
            scene.name
        );
    }
}
