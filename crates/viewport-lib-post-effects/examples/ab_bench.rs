//! Headless frame-time A/B: the built-in contact shadows vs this crate's
//! external copy, plus an effects-off baseline. (A bloom copy was part of
//! the original A/B and was retired after validation.)
//!
//! Renders the same scene through three configurations on one device and
//! prints end-to-end frame percentiles (each frame is submitted and waited
//! on, so the numbers are synchronous full-frame costs, comparable within
//! this run). Writes a JSON line per configuration to `AB_OUT` when set.
//!
//! Env knobs: `AB_FRAMES` (default 300), `AB_WARMUP` (30), `AB_WIDTH`
//! (1600), `AB_HEIGHT` (900), `AB_GRID` (20: the scene is GRID x GRID
//! cubes), `AB_OUT` (JSON output path).
//!
//! Run: `cargo run --release -p viewport-lib-post-effects --example ab_bench`

use std::time::Instant;

use viewport_lib::wgpu;
use viewport_lib::{
    Camera, LightKind, LightSource, Material,
    renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, ViewportRenderer},
    resources::MeshData,
};
use viewport_lib_post_effects::{ContactShadowEffect, ContactShadowEffectSettings};
use viewport_lib_testkit::{DeviceProfile, headless_device_with};

const LIGHT_DIRECTION: [f32; 3] = [0.5, 0.3, 0.8];
const CS: ContactShadowEffectSettings = ContactShadowEffectSettings {
    enabled: true,
    max_distance: 0.6,
    steps: 16,
    thickness: 0.4,
    light_direction: LIGHT_DIRECTION,
};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn box_mesh() -> MeshData {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    mesh.normals = vec![
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    mesh.indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    mesh
}

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Off,
    Builtin,
    External,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::Off => "off",
            Variant::Builtin => "builtin",
            Variant::External => "external",
        }
    }
}

fn main() {
    let frames = env_u32("AB_FRAMES", 300) as usize;
    let warmup = env_u32("AB_WARMUP", 30) as usize;
    let width = env_u32("AB_WIDTH", 1600);
    let height = env_u32("AB_HEIGHT", 900);
    let grid = env_u32("AB_GRID", 20) as i32;

    let Some((device, queue)) =
        headless_device_with(&DeviceProfile::high_performance("post-effect-ab"))
    else {
        eprintln!("no GPU adapter available");
        std::process::exit(1);
    };

    println!(
        "post-effect A/B: {width}x{height}, {grid}x{grid} cubes, {frames} frames (+{warmup} warmup)"
    );

    let mut out_lines = Vec::new();
    for variant in [Variant::Off, Variant::Builtin, Variant::External] {
        let ms = run_variant(
            &device, &queue, variant, width, height, grid, frames, warmup,
        );
        let pct = |p: f64| ms[((ms.len() - 1) as f64 * p) as usize];
        println!(
            "{:<10} p50 {:7.3} ms   p90 {:7.3} ms   p99 {:7.3} ms",
            variant.label(),
            pct(0.50),
            pct(0.90),
            pct(0.99)
        );
        out_lines.push(format!(
            "{{\"variant\":\"{}\",\"width\":{width},\"height\":{height},\"grid\":{grid},\"frames\":{frames},\"frame_ms_p50\":{:.4},\"frame_ms_p90\":{:.4},\"frame_ms_p99\":{:.4}}}",
            variant.label(),
            pct(0.50),
            pct(0.90),
            pct(0.99)
        ));
    }

    if let Ok(path) = std::env::var("AB_OUT") {
        std::fs::write(&path, out_lines.join("\n") + "\n").expect("write AB_OUT");
        println!("wrote {path}");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_variant(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    variant: Variant,
    width: u32,
    height: u32,
    grid: i32,
    frames: usize,
    warmup: usize,
) -> Vec<f64> {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &box_mesh())
        .unwrap();

    if variant == Variant::External {
        let (cs, _h) = ContactShadowEffect::new(CS);
        renderer.add_post_effect_producer(Box::new(cs));
    }

    // Ground slab plus a GRID x GRID field of cubes, every eighth emissive
    // so bright sources are spread across the frame.
    let mut items = Vec::new();
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = mesh;
    ground.model =
        (glam::Mat4::from_scale(glam::Vec3::new(grid as f32 * 1.6, grid as f32 * 1.6, 0.1))
            * glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.5)))
        .to_cols_array_2d();
    ground.material = Material::from_colour([0.6, 0.6, 0.6]);
    items.push(ground);
    for x in 0..grid {
        for y in 0..grid {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh;
            let pos = glam::Vec3::new(
                (x - grid / 2) as f32 * 1.5,
                (y - grid / 2) as f32 * 1.5,
                0.45,
            );
            item.model = (glam::Mat4::from_translation(pos)
                * glam::Mat4::from_scale(glam::Vec3::splat(0.8)))
            .to_cols_array_2d();
            let i = (x * grid + y) as usize;
            item.material = Material::from_colour([0.3, 0.4, 0.6]);
            if i % 8 == 0 {
                item.material.emissive = [4.0, 2.8, 1.2].into();
            }
            items.push(item);
        }
    }

    let cam = Camera {
        distance: grid as f32 * 1.8,
        ..Camera::default()
    };
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = width as f32 / height as f32;
        rc
    };
    frame.camera.viewport_size = [width as f32, height as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.12, 0.13, 0.15, 1.0].into());
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: LIGHT_DIRECTION,
    };
    frame.effects.lighting.lights = vec![light];
    let pp = &mut frame.effects.post_process;
    pp.contact_shadows.enabled = variant == Variant::Builtin;
    pp.contact_shadows.max_distance = CS.max_distance;
    pp.contact_shadows.steps = CS.steps;
    pp.contact_shadows.thickness = CS.thickness;
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ab_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut samples = Vec::with_capacity(frames);
    for i in 0..warmup + frames {
        let t0 = Instant::now();
        renderer.render_to_texture(device, queue, &view, &frame);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        if i >= warmup {
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples
}
