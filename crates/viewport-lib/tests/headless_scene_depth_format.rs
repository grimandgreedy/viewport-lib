//! The scene depth format: `RendererConfig::scene_depth_format` picks the
//! format every scene pipeline and depth target uses, a 32-bit float depth
//! draws the same picture as the 24-bit default, and a request the device
//! cannot honour falls back with a notice.

use viewport_lib::wgpu;
use viewport_lib::{Colour, LightKind, NoticeDetail, RendererConfig};

mod common;
use common::*;
use viewport_lib_testkit::{DeviceProfile, headless_device_with};

const SIZE: u32 = 192;
const F24: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;
const F32S8: wgpu::TextureFormat = wgpu::TextureFormat::Depth32FloatStencil8;

/// Per-channel difference, in 8-bit levels, a pixel may show between the two
/// depth formats. Depth precision moves which triangle wins at silhouettes and
/// where a shadow edge falls, so pixels there change by a few levels. Apple
/// GPUs store `Depth24PlusStencil8` as 32-bit float, so there the two frames
/// match exactly.
const TOLERANCE: u8 = 24;
/// Share of pixels allowed past [`TOLERANCE`]: silhouette and shadow-edge
/// pixels only.
const OUTLIER_SHARE: f64 = 0.002;

fn renderer(device: &wgpu::Device, depth: Option<wgpu::TextureFormat>) -> ViewportRenderer {
    let mut config = RendererConfig::new(wgpu::TextureFormat::Rgba8UnormSrgb)
        .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    if let Some(format) = depth {
        config = config.with_scene_depth_format(format);
    }
    ViewportRenderer::with_config(device, &config)
}

/// A sphere, a cube and a torus on a ground slab under a low directional sun,
/// so the frame carries cast shadows, silhouettes over the ground and contact.
fn trio_frame(renderer: &mut ViewportRenderer, device: &wgpu::Device, hdr: bool) -> FrameData {
    use viewport_lib::geometry::primitives;
    let mut items = Vec::new();
    let mut add = |renderer: &mut ViewportRenderer,
                   mesh: MeshData,
                   pos: [f32; 3],
                   colour: Colour,
                   pick: u64| {
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(device, &mesh)
            .unwrap();
        let mut item = SceneRenderItem::new(
            mesh_id,
            upload_material(renderer, Material::pbr(colour, 0.2, 0.5)),
        );
        item.model = glam::Mat4::from_translation(glam::Vec3::from(pos)).to_cols_array_2d();
        item.settings.pick_id = PickId(pick);
        items.push(item);
    };
    add(
        renderer,
        primitives::sphere(0.8, 32, 16),
        [-2.5, 0.0, 0.8],
        Colour::linear_rgb(0.9, 0.5, 0.2),
        1,
    );
    add(
        renderer,
        primitives::cube(1.2),
        [0.0, 0.0, 0.6],
        Colour::linear_rgb(0.4, 0.6, 0.9),
        2,
    );
    add(
        renderer,
        primitives::torus(0.6, 0.22, 32, 16),
        [2.5, 0.0, 0.22],
        Colour::linear_rgb(0.3, 0.8, 0.4),
        3,
    );
    add(
        renderer,
        primitives::cuboid(40.0, 40.0, 0.4),
        [0.0, 0.0, -0.2],
        Colour::linear_rgb(0.55, 0.55, 0.58),
        4,
    );

    let camera = Camera {
        center: glam::Vec3::ZERO,
        distance: 9.0,
        orientation: glam::Quat::from_rotation_z(0.7) * glam::Quat::from_rotation_x(1.0),
        ..Camera::default()
    };
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&camera);
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.1, 0.1, 0.12, 1.0));
    frame.effects.display.mode = if hdr {
        viewport_lib::PipelineMode::Hdr
    } else {
        viewport_lib::PipelineMode::Direct
    };
    let mut sun = viewport_lib::LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [1.0, 0.35, 0.6],
    };
    sun.intensity = 1.2;
    frame.effects.lighting.lights = vec![sun];
    frame.effects.lighting.hemisphere_intensity = 0.3;
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    frame
}

/// Pixels past [`TOLERANCE`] in any channel, and the largest difference.
fn compare(a: &[u8], b: &[u8]) -> (usize, u8) {
    let mut outliers = 0;
    let mut max = 0;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = pa
            .iter()
            .zip(pb)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap();
        max = max.max(d);
        if d > TOLERANCE {
            outliers += 1;
        }
    }
    (outliers, max)
}

/// The same shadowed scene drawn with a `Depth32FloatStencil8` scene depth
/// matches the 24-bit render within [`TOLERANCE`] on both paths, and a GPU
/// pick against the float depth finds the object under the cursor.
#[test]
fn float_depth_matches_the_24_bit_render() {
    let Some((device, queue)) = headless_device_with(
        &DeviceProfile::low_power("scene-depth-32f").require(wgpu::Features::DEPTH32FLOAT_STENCIL8),
    ) else {
        eprintln!("skipping: no adapter with DEPTH32FLOAT_STENCIL8");
        return;
    };
    let mut base = renderer(&device, None);
    let mut float = renderer(&device, Some(F32S8));
    assert_eq!(base.scene_depth_format(), F24);
    assert_eq!(float.scene_depth_format(), F32S8);
    assert_eq!(float.capabilities().scene_depth_format, F32S8);
    let desc = float.resources().opaque_target_desc();
    assert_eq!(desc.depth_format, F32S8);
    assert_eq!(
        float.resources().pick_target_desc().depth_stencil_format,
        F32S8
    );
    assert!(float.take_notices().is_empty());

    for hdr in [true, false] {
        let frame_a = trio_frame(&mut base, &device, hdr);
        let frame_b = trio_frame(&mut float, &device, hdr);
        let a = base.render_offscreen(&device, &queue, &frame_a, SIZE, SIZE);
        let b = float.render_offscreen(&device, &queue, &frame_b, SIZE, SIZE);
        let background = &a[..4];
        let covered = a.chunks_exact(4).filter(|p| *p != background).count();
        assert!(
            covered > (SIZE * SIZE / 4) as usize,
            "hdr={hdr}: the scene covers only {covered} pixels"
        );
        let (outliers, max) = compare(&a, &b);
        let allowed = ((SIZE * SIZE) as f64 * OUTLIER_SHARE).ceil() as usize;
        println!("hdr={hdr}: {outliers} pixels past {TOLERANCE} levels, max diff {max}");
        assert!(
            outliers <= allowed,
            "hdr={hdr}: {outliers} pixels differ by more than {TOLERANCE} levels \
             (allowed {allowed}, max {max})"
        );

        // The cube sits at the centre of the view.
        let centre = glam::Vec2::splat(SIZE as f32 / 2.0);
        let hit_a = base.pick_scene_gpu(&device, &queue, centre, &frame_a);
        let hit_b = float.pick_scene_gpu(&device, &queue, centre, &frame_b);
        assert_eq!(
            hit_a.map(|h| h.object_id),
            hit_b.map(|h| h.object_id),
            "the two depth formats picked different objects"
        );
        assert!(hit_b.is_some(), "the pick under the cube found nothing");
    }
}

/// On a device created without `DEPTH32FLOAT_STENCIL8` a request for the float
/// depth falls back to 24-bit and says so in a notice.
#[test]
fn float_depth_without_the_feature_falls_back_with_a_notice() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    if device
        .features()
        .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8)
    {
        eprintln!("skipping: the default test device has DEPTH32FLOAT_STENCIL8");
        return;
    }
    let mut renderer = renderer(&device, Some(F32S8));
    assert_eq!(renderer.scene_depth_format(), F24);
    let notices = renderer.take_notices();
    assert!(
        notices.iter().any(|n| matches!(
            &n.detail,
            NoticeDetail::SceneDepthFormatFallback { requested, used, .. }
                if requested == "Depth32FloatStencil8" && used == "Depth24PlusStencil8"
        )),
        "no fallback notice: {notices:?}"
    );
}

/// A depth format without a stencil aspect is refused at construction.
#[test]
fn a_format_without_stencil_is_refused() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let config = RendererConfig::new(wgpu::TextureFormat::Rgba8UnormSrgb)
        .with_scene_depth_format(wgpu::TextureFormat::Depth32Float);
    assert!(matches!(
        ViewportRenderer::try_with_config(&device, &config),
        Err(ViewportError::InvalidSceneDepthFormat { .. })
    ));
}
