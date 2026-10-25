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
    assert_eq!(caps.tier(), Tier::Forward);
    assert!(!caps.compute);
    assert!(!caps.vertex_storage && !caps.fragment_storage);
    assert!(!caps.cube_array_textures);

    // Without the adapter's flags the zero compute limits still rule compute
    // out, so the tier does not depend on the flags being passed.
    assert_eq!(
        Capabilities::from_device(&d.device, None).tier(),
        Tier::Forward
    );

    // Construction builds nothing the device cannot create: a layout over its
    // limits would fail validation here, and validation failures panic.
    let config = viewport_lib::RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT)
        .with_downlevel_capabilities(d.downlevel);
    let mut renderer = ViewportRenderer::try_with_config(&d.device, &config)
        .expect("a forward renderer builds on WebGL2 limits");
    assert_eq!(renderer.tier(), Tier::Forward);
    assert_scene_calls_refused(&mut renderer, &d.device, &d.queue);
}

/// WebGL2's limits with two bind groups: below the forward tier, so only the
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
            needed: Tier::Forward,
            have: Tier::OverlayOnly,
        }) => {}
        other => panic!("expected TierUnsupported, got {other:?}"),
    }
    assert!(!renderer.has_item_type_plugin("tier_test.modest"));
}

/// The calls that add scene content say why they cannot.
fn assert_scene_calls_refused(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let have = renderer.tier();
    assert!(have < Tier::Full);
    let refused = |e: ViewportError| {
        assert!(
            matches!(
                e,
                ViewportError::TierUnsupported { needed: Tier::Full, have: h } if h == have
            ),
            "expected TierUnsupported, got {e}"
        )
    };
    let mesh = viewport_lib::geometry::primitives::cube(1.0);
    refused(
        renderer
            .resources_mut()
            .upload_mesh_data(device, &mesh)
            .unwrap_err(),
    );
    let image = viewport_lib::TextureData::srgb(1, 1, vec![255; 4]);
    refused(
        renderer
            .resources_mut()
            .upload_texture(device, queue, image)
            .unwrap_err(),
    );

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

/// A device with storage buffers and no compute. It lands on the forward tier,
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
        // renderer leaves out; leave it out of the reference too.
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

        let built = overlay.build_scene(&scene);
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
    let Some(mut overlay) = Harness::with_profile(&no_compute_profile()) else {
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
    assert_eq!(stats.tier, Tier::Forward);
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

/// An item type that asks for nothing.
struct Modest;

impl ItemTypePlugin for Modest {
    fn type_name(&self) -> &'static str {
        "tier_test.modest"
    }
    fn requires(&self) -> Requirements {
        Requirements::none()
    }
    fn init_gpu(&mut self, _: &wgpu::Device, _: &viewport_lib::plugin_api::SharedBindings<'_>) {
        panic!("init_gpu ran on an overlay-only renderer");
    }
}

/// An item type that accepts the forward tier and, in `init_gpu`, builds a
/// pipeline lit through the shared scene-lighting helper against the shared
/// group 0, recording any validation error.
#[derive(Default)]
struct ForwardLit {
    built: Option<Result<(), String>>,
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
        Requirements::forward_tier()
    }
    fn init_gpu(
        &mut self,
        device: &wgpu::Device,
        shared: &viewport_lib::plugin_api::SharedBindings<'_>,
    ) {
        let source = format!(
            "{}\n{}",
            viewport_lib::plugin_api::shared_wgsl::SHARED_SCENE_LIGHTING_WGSL,
            FORWARD_LIT_BODY
        );
        let source = shared.adapt_wgsl(&source).into_owned();
        let scope = viewport_lib::gpu::push_error_scope(device, wgpu::ErrorFilter::Validation);
        use viewport_lib::plugin_api::builders;
        let module = builders::wgsl_module(device, "forward_lit", source);
        let layout = builders::pipeline_layout(device, "forward_lit", &[shared.group0_layout]);
        let _ = builders::render_pipeline(
            device,
            builders::RenderPipelineDesc {
                label: "forward_lit",
                layout: &layout,
                vertex_module: &module,
                vertex_entry: "vs_main",
                vertex_buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(Harness::DEFAULT_TARGET_FORMAT.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                cache: None,
            },
        );
        self.built = Some(match pollster::block_on(scope.pop()) {
            None => Ok(()),
            Some(e) => Err(e.to_string()),
        });
    }
}

/// On the forward tier an item type that accepts it registers and builds a lit
/// pipeline against the forward group 0; one that needs the full tier is
/// declined. The same plugin builds on the full tier too.
#[test]
fn forward_tier_registers_item_types_that_accept_it() {
    for profile in [
        DeviceProfile::downlevel_webgl2("tier-forward-plugin"),
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

        if tier == Tier::Forward {
            match harness
                .renderer
                .with_item_type_plugin(&harness.device, Box::new(Greedy))
            {
                Err(ViewportError::PluginUnsupported { .. }) => {}
                other => panic!("expected PluginUnsupported, got {other:?}"),
            }
            // Nothing the renderer ships with accepts the forward tier yet.
            assert!(
                !harness
                    .renderer
                    .has_item_type_plugin("viewport_lib.point_cloud")
            );
        }
    }
}

/// The forward light list holds `FORWARD_MAX_LIGHTS`, and the frame says how
/// many it left out. The full tier's holds all of them.
#[test]
fn forward_light_list_caps_and_counts_what_it_drops() {
    let lights = viewport_lib::resources::FORWARD_MAX_LIGHTS + 4;
    for (profile, dropped) in [
        (DeviceProfile::downlevel_webgl2("tier-forward-lights"), 4),
        (DeviceProfile::harness(), 0),
    ] {
        let Some(mut harness) = Harness::with_profile(&profile) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // Lights and no scene content: the forward tier has no meshes to draw
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
                    load: wgpu::LoadOp::Clear(1.0),
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

    let built = overlay.build_scene(&scene);
    let mut frame = frame_for(&built, camera, [W as f32, H as f32]);
    let (painted, owned) = draw_split(&mut overlay, &mut frame);
    assert!(painted == want_painted, "the host-pass paint differs");
    assert!(owned == want_owned, "render_viewport differs");
}

/// Every scene item a frame submits, counted the way `FrameStats` counts what
/// an overlay-only renderer skips.
fn scene_items(frame: &viewport_lib::FrameData) -> usize {
    let surfaces = match &frame.scene.surfaces {
        viewport_lib::SurfaceSubmission::Flat(items) => items.len(),
        _ => panic!("a surface submission this test does not count"),
    };
    let plugin_items: usize = frame
        .scene
        .plugin_items
        .values()
        .flatten()
        .map(|c| c.len())
        .sum();
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
    // scenes are built on a full renderer, since their uploads are refused on
    // the overlay-only one, and the frames handed across.
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
            scene_items(&frame),
            "{}",
            scene.name
        );
    }
}
