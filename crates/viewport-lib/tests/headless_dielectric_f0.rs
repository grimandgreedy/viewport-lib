//! Dielectric F0 control: `Material::reflectance` drives the PBR specular, and a
//! surface hook's `f0_override` replaces it only when its `w` is set.
//!
//! Part of the headless integration suite. Shared device and mesh helpers live
//! in tests/common/mod.rs.

use viewport_lib::Colour;
use viewport_lib::wgpu;

mod common;
use common::*;

const SIZE: u32 = 96;

/// Render one black PBR sphere under a single directional light of
/// `intensity` from behind the camera, with no ambient, so every lit pixel is
/// the specular lobe.
fn render_sphere_with(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material: Material,
    mode: viewport_lib::PipelineMode,
    intensity: f32,
) -> Vec<u8> {
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::sphere(1.0, 64, 32))
        .unwrap();
    let mut cam = Camera::default();
    cam.set_distance(4.0);
    cam.set_aspect_ratio(1.0, 1.0);
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame.viewport.environment_background = viewport_lib::EnvironmentBackground::colour();
    frame.effects.lighting.hemisphere_intensity = 0.0;
    frame.effects.display.mode = mode;
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: (cam.eye_position() - cam.center).normalize().to_array(),
    };
    light.intensity = intensity;
    light.cast_shadows = false;
    frame.effects.lighting.lights = vec![light];
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.material = upload_material(renderer, material);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    frame.scene.generation = GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    renderer.render_offscreen(device, queue, &frame, SIZE, SIZE)
}

/// `render_sphere_with` without a tone map (the Direct pipeline), at a light
/// intensity that keeps the brightest highlight below 1, so pixel luminance is
/// linear in radiance up to 8-bit rounding.
fn render_sphere(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material: Material,
) -> Vec<u8> {
    render_sphere_with(
        renderer,
        device,
        queue,
        material,
        viewport_lib::PipelineMode::Direct,
        0.5,
    )
}

/// Peak linear luminance of an sRGB-encoded RGBA8 image.
fn peak_luminance(pixels: &[u8]) -> f32 {
    let decode = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    pixels
        .chunks_exact(4)
        .map(|p| 0.2126 * decode(p[0]) + 0.7152 * decode(p[1]) + 0.0722 * decode(p[2]))
        .fold(0.0, f32::max)
}

fn dark_dielectric() -> Material {
    Material::pbr(Colour::linear_rgb(0.0, 0.0, 0.0), 0.0, 0.3)
}

fn renderer(device: &wgpu::Device) -> ViewportRenderer {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    renderer
}

fn with_reflectance(reflectance: f32) -> Material {
    let mut m = dark_dielectric();
    m.reflectance = reflectance;
    m
}

/// F0 = 0.16 r^2: reflectance 1.0 gives four times the default's F0, so the
/// highlight peak of a black dielectric is about four times brighter.
#[test]
fn full_reflectance_quadruples_the_highlight() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer(&device);
    let peak = |renderer: &mut ViewportRenderer, reflectance| {
        peak_luminance(&render_sphere(
            renderer,
            &device,
            &queue,
            with_reflectance(reflectance),
        ))
    };
    let d = peak(&mut renderer, 0.5);
    let f = peak(&mut renderer, 1.0);
    assert!(d > 0.05, "the default dielectric shows no highlight ({d})");
    let ratio = f / d;
    assert!(
        (3.5..=4.5).contains(&ratio),
        "reflectance 1.0 should give 4x the default highlight, got {ratio}"
    );
    assert_eq!(
        peak(&mut renderer, 0.0),
        0.0,
        "reflectance 0 on a black dielectric leaves nothing lit"
    );

    // Through the default tone map the curve compresses the brighter peak, so
    // the ratio only has to stay well clear of 1.
    let hdr = |renderer: &mut ViewportRenderer, reflectance| {
        peak_luminance(&render_sphere_with(
            renderer,
            &device,
            &queue,
            with_reflectance(reflectance),
            viewport_lib::PipelineMode::Hdr,
            1.0,
        ))
    };
    let hdr_ratio = hdr(&mut renderer, 1.0) / hdr(&mut renderer, 0.5);
    assert!(
        (2.5..=5.0).contains(&hdr_ratio),
        "tone-mapped highlight ratio {hdr_ratio}"
    );
}

/// A surface hook that keeps the material and only sets `f0_override` from
/// two parameters: `f0` for the rgb and `use_f0` for `w`.
struct F0OverridePlugin;

impl viewport_lib::MaterialPlugin for F0OverridePlugin {
    fn name(&self) -> &'static str {
        "f0_override_test"
    }
    fn wgsl_body(&self) -> String {
        "\
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {
    var ov: SurfaceOverride;
    ov.base_colour = surf.base_colour;
    ov.normal = surf.normal;
    ov.metallic = surf.metallic;
    ov.roughness = surf.roughness;
    ov.alpha = surf.alpha;
    ov.f0_override = vec4<f32>(vec3<f32>(params.f0), params.use_f0);
    return ov;
}
"
        .to_string()
    }
    fn param_layout(&self) -> Vec<viewport_lib::ParamDesc> {
        vec![
            viewport_lib::ParamDesc::f32("f0", 0.0),
            viewport_lib::ParamDesc::f32("use_f0", 0.0),
        ]
    }
}

/// `f0_override.w == 0` keeps the material's dielectric F0 whatever the rgb;
/// `w == 1` replaces it, matching a material authored with that F0.
#[test]
fn surface_hook_f0_override_honours_w() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer(&device);
    let plugin = renderer
        .resources_mut()
        .register_material_plugin(&device, &F0OverridePlugin)
        .expect("register material plugin");
    let hooked = |reflectance: f32, f0: f32, use_f0: f32| {
        let mut m = dark_dielectric()
            .with_plugin(plugin)
            .with_param("f0", f0)
            .with_param("use_f0", use_f0);
        m.reflectance = reflectance;
        m
    };

    // w = 0: the rgb is ignored, and the result is the material's own F0.
    let unset = render_sphere(&mut renderer, &device, &queue, hooked(0.5, 0.0, 0.0));
    let ignored = render_sphere(&mut renderer, &device, &queue, hooked(0.5, 0.16, 0.0));
    assert_eq!(
        unset, ignored,
        "f0_override.rgb must be ignored when w is 0"
    );

    // w = 1 with F0 = 0.16 on a default-reflectance material matches
    // reflectance 1.0 with the override off.
    let replaced = render_sphere(&mut renderer, &device, &queue, hooked(0.5, 0.16, 1.0));
    let authored = render_sphere(&mut renderer, &device, &queue, hooked(1.0, 0.0, 0.0));
    let max_diff = replaced
        .iter()
        .zip(&authored)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(
        max_diff <= 1,
        "f0_override (w = 1) must match the authored reflectance (max diff {max_diff})"
    );
    let ratio = peak_luminance(&replaced) / peak_luminance(&unset);
    assert!(
        (3.0..=5.0).contains(&ratio),
        "f0_override 0.16 should brighten the highlight about 4x, got {ratio}"
    );
}
