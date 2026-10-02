//! A frame drawn on a fresh renderer, where each pipeline group is built by the
//! first frame that asks for it, has to match the same frame drawn on a
//! renderer with every group built up front.
//!
//! Most passes skip quietly when their pipeline is missing. A build trigger
//! narrower than the draw condition therefore shows up as a wrong image, not a
//! panic, and this comparison is what catches it. Each case also has to differ
//! from the plain frame, so a feature that drew nothing on either renderer
//! cannot pass by agreeing with itself.

use crate::renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission};
use crate::{Camera, Material, ViewportRenderer};

const SIZE: u32 = 96;
const FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rgba8UnormSrgb;

/// A device for the cases to run on. With `recommended` it carries the
/// features and limits a runner asks for, which switches on GPU culling,
/// bindless textures and the deform sidecar where the adapter has them.
/// Without, it is a default device, where the same frames go through the
/// plain instanced and per-object paths.
fn device(recommended: bool) -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
    let instance = crate::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &crate::gpu::RequestAdapterOptions {
            power_preference: crate::gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            #[cfg(wgpu30)]
            apply_limit_buckets: false,
        },
    ))
    .ok()?;
    let descriptor = if recommended {
        crate::gpu::DeviceDescriptor {
            required_features: ViewportRenderer::recommended_device_features(&adapter),
            required_limits: ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }
    } else {
        crate::gpu::DeviceDescriptor::default()
    };
    pollster::block_on(adapter.request_device(&descriptor)).ok()
}

/// Build every lazily built pipeline group, as a renderer that had drawn
/// everything already would hold them.
fn build_everything(
    renderer: &mut ViewportRenderer,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
) {
    let res = &mut renderer.resources;
    res.ensure_hdr_pipelines(device, queue, FORMAT);
    res.ensure_ldr_mesh_pipelines(device);
    res.ensure_instanced_pipelines(device);
    res.ensure_ldr_instanced_pipelines(device);
    res.ensure_hdr_instanced_pipelines(device);
    res.ensure_oit_instanced_pipeline(device);
    res.ensure_cull_instance_pipelines(device);
    res.ensure_hdr_cull_pipelines(device);
    res.ensure_oit_cull_pipelines(device);
    res.ensure_outline_pipelines(device);
    res.ensure_xray_pipeline(device);
    res.ensure_ground_plane_pipeline(device);
    res.ensure_skybox_pipeline(device);
    res.ensure_cascade_shadow_pipelines(device);
    res.ensure_point_shadow_pipeline(device);
}

struct Meshes {
    cube: crate::MeshId,
    flow_quad: crate::MeshId,
}

fn upload(renderer: &mut ViewportRenderer, device: &crate::gpu::Device) -> Meshes {
    let cube = renderer
        .resources_mut()
        .upload_mesh_data(device, &crate::geometry::primitives::cube(1.0))
        .unwrap();
    let mut quad = crate::resources::MeshData::default();
    quad.positions = vec![
        [-0.5, -0.5, 0.6],
        [0.5, -0.5, 0.6],
        [0.5, 0.5, 0.6],
        [-0.5, 0.5, 0.6],
    ];
    quad.normals = vec![[0.0, 0.0, 1.0]; 4];
    quad.indices = vec![0, 1, 2, 0, 2, 3];
    quad.attributes.insert(
        "flow".to_string(),
        crate::AttributeData::VertexVector(vec![[1.0, 0.0, 0.0]; 4]),
    );
    let flow_quad = renderer
        .resources_mut()
        .upload_mesh_data(device, &quad)
        .unwrap();
    Meshes { cube, flow_quad }
}

fn cube_item(meshes: &Meshes, x: f32, colour: [f32; 3]) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = meshes.cube;
    item.model = glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
    item.material = Material::from_colour(colour);
    item
}

/// One cube: the per-object path.
fn base_frame(meshes: &Meshes) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.1, 0.1, 0.12, 1.0].into());
    frame.scene.surfaces =
        SurfaceSubmission::Flat(vec![cube_item(meshes, 0.0, [0.8, 0.5, 0.3])].into());
    frame
}

/// Three cubes sharing a mesh and material: the instanced path.
fn instanced_frame(meshes: &Meshes) -> FrameData {
    let mut frame = base_frame(meshes);
    frame.scene.surfaces = SurfaceSubmission::Flat(
        (0..3)
            .map(|i| cube_item(meshes, i as f32 * 1.3 - 1.3, [0.8, 0.5, 0.3]))
            .collect::<Vec<_>>()
            .into(),
    );
    frame
}

type Setup = fn(&Meshes, &mut ViewportRenderer) -> FrameData;

fn render_case(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    setup: Setup,
    eager: bool,
) -> Vec<u8> {
    let mut renderer = ViewportRenderer::new(device, FORMAT);
    if eager {
        build_everything(&mut renderer, device, queue);
    }
    let meshes = upload(&mut renderer, device);
    let frame = setup(&meshes, &mut renderer);
    renderer.render_offscreen(device, queue, &frame, SIZE, SIZE)
}

fn check(name: &str, setup: Setup, plain: Setup) {
    for recommended in [true, false] {
        let Some((device, queue)) = device(recommended) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lazy = render_case(&device, &queue, setup, false);
        let eager = render_case(&device, &queue, setup, true);
        assert!(
            lazy == eager,
            "{name} (recommended device: {recommended}): the first frame on a fresh renderer \
             differs from the same frame with every pipeline prebuilt, so some pass did not \
             find its pipeline"
        );
        let baseline = render_case(&device, &queue, plain, true);
        assert!(
            eager != baseline,
            "{name} (recommended device: {recommended}): the feature changed nothing, so this \
             case cannot tell a skipped pass from a drawn one"
        );
    }
}

/// The same comparison for a feature switched on part-way through a session.
///
/// A viewport holds full-size targets only for what its frames have used, and
/// promotes a group when a frame first asks for it. Drawing `first` and then
/// `then` on one renderer has to give the image `then` gives on a fresh one:
/// the promotion must not lose a target another pass already drew into, and
/// nothing may be left bound to a stand-in.
fn check_mid_session(name: &str, first: Setup, then: Setup) {
    for recommended in [true, false] {
        let Some((device, queue)) = device(recommended) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let fresh = render_case(&device, &queue, then, false);

        let mut renderer = ViewportRenderer::new(&device, FORMAT);
        let meshes = upload(&mut renderer, &device);
        // Distinct generations, so the second frame's items are not taken for
        // the first's.
        let mut frame = first(&meshes, &mut renderer);
        frame.scene.generation = 1;
        let _ = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
        let mut frame = then(&meshes, &mut renderer);
        frame.scene.generation = 2;
        let later = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
        assert!(
            later == fresh,
            "{name} (recommended device: {recommended}): switched on after an earlier frame, the \
             image differs from the same frame on a fresh renderer"
        );
    }
}

/// An empty frame, for cases where the mesh itself is the feature.
fn empty(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
    let mut frame = base_frame(meshes);
    frame.scene.surfaces = SurfaceSubmission::Flat(Vec::new().into());
    frame
}

fn plain(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
    base_frame(meshes)
}

fn plain_instanced(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
    instanced_frame(meshes)
}

#[test]
fn hdr_mesh() {
    check("hdr mesh", plain, empty);
}

#[test]
fn hdr_instanced_mesh() {
    check("hdr instanced mesh", plain_instanced, empty);
}

#[test]
fn direct_mesh() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        frame.effects.display.mode = crate::PipelineMode::Direct;
        frame
    }
    fn none(meshes: &Meshes, r: &mut ViewportRenderer) -> FrameData {
        let mut frame = empty(meshes, r);
        frame.effects.display.mode = crate::PipelineMode::Direct;
        frame
    }
    check("direct mesh", setup, none);
    // And across a change of path, in both directions.
    check_mid_session("hdr then direct", plain, setup);
    check_mid_session("direct then hdr", setup, plain);
}

#[test]
fn direct_instanced_mesh() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = instanced_frame(meshes);
        frame.effects.display.mode = crate::PipelineMode::Direct;
        frame
    }
    fn none(meshes: &Meshes, r: &mut ViewportRenderer) -> FrameData {
        let mut frame = empty(meshes, r);
        frame.effects.display.mode = crate::PipelineMode::Direct;
        frame
    }
    check("direct instanced mesh", setup, none);
}

#[test]
fn bloom() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        frame.effects.post_process.bloom.enabled = true;
        frame.effects.post_process.bloom.threshold = 0.0;
        frame.effects.post_process.bloom.intensity = 1.0;
        frame
    }
    check("bloom", setup, plain);
    check_mid_session("bloom", plain, setup);
}

#[test]
fn ssao() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = instanced_frame(meshes);
        frame.effects.post_process.ssao = true;
        frame
    }
    check("ssao", setup, plain_instanced);
    check_mid_session("ssao", plain_instanced, setup);
}

#[test]
fn fxaa() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        frame.effects.post_process.fxaa = true;
        frame
    }
    check("fxaa", setup, plain);
    check_mid_session("fxaa", plain, setup);
}

#[test]
fn depth_of_field() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        frame.effects.post_process.dof.enabled = true;
        frame.effects.post_process.dof.focal_distance = 0.5;
        frame.effects.post_process.dof.focal_range = 0.1;
        frame.effects.post_process.dof.max_blur_radius = 8.0;
        frame
    }
    check("depth of field", setup, plain);
    check_mid_session("depth of field", plain, setup);
}

#[test]
fn supersampling() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        frame.effects.post_process.ssaa_factor = 2;
        frame
    }
    check("ssaa", setup, plain);
}

#[test]
fn automatic_exposure() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        frame.effects.display.exposure = crate::ExposureSettings::automatic();
        frame
    }
    check("automatic exposure", setup, plain);
}

#[test]
fn render_scale() {
    fn setup(meshes: &Meshes, renderer: &mut ViewportRenderer) -> FrameData {
        renderer.set_render_scale(0.5);
        base_frame(meshes)
    }
    check("render scale", setup, plain);
}

#[test]
fn transparent_mesh() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        let mut glass = cube_item(meshes, 0.6, [0.2, 0.6, 0.9]);
        glass.settings.opacity = 0.5;
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![glass].into());
        frame
    }
    check("transparent mesh", setup, empty);
    check_mid_session("transparent mesh", plain, setup);
}

#[test]
fn transparent_instanced_mesh() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        frame.scene.surfaces = SurfaceSubmission::Flat(
            (0..3)
                .map(|i| {
                    let mut glass = cube_item(meshes, i as f32 * 1.3 - 1.3, [0.2, 0.6, 0.9]);
                    glass.settings.opacity = 0.5;
                    glass
                })
                .collect::<Vec<_>>()
                .into(),
        );
        frame
    }
    check("transparent instanced mesh", setup, empty);
}

#[test]
fn foreground_item() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        frame.scene.foreground_items = vec![cube_item(meshes, 0.9, [0.9, 0.2, 0.2])];
        frame
    }
    check("foreground item", setup, plain);
    check_mid_session("foreground item", plain, setup);
}

#[test]
fn selection_outline() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        let mut item = cube_item(meshes, 0.0, [0.8, 0.5, 0.3]);
        item.settings.selected = true;
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        frame.interaction.outline_selected = true;
        frame
    }
    check("selection outline", setup, plain);
    check_mid_session("selection outline", plain, setup);
}

#[test]
fn surface_lic() {
    fn setup(meshes: &Meshes, _: &mut ViewportRenderer) -> FrameData {
        let mut frame = base_frame(meshes);
        let mut item = SceneRenderItem::default();
        item.mesh_id = meshes.flow_quad;
        item.material = Material::from_colour([0.7, 0.7, 0.7]);
        item.settings.unlit = true;
        let mut lic = crate::LicOverlay::new("flow", crate::SurfaceLICConfig::default());
        lic.config.strength = 2.0;
        item.lic = Some(lic);
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        frame
    }
    fn without(meshes: &Meshes, r: &mut ViewportRenderer) -> FrameData {
        let mut frame = setup(meshes, r);
        let crate::SurfaceSubmission::Flat(items) = &frame.scene.surfaces;
        let mut items = items.to_vec();
        items[0].lic = None;
        frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
        frame
    }
    check("surface lic", setup, without);
    check_mid_session("surface lic", plain, setup);
}
