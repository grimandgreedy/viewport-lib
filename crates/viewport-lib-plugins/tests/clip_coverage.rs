//! What a clip box and a clip plane remove from each item type, measured by
//! drawing rather than by reading shaders.
//!
//! Each item sits across the plane x = 0 and is drawn three ways: unclipped,
//! with a box that keeps the half x < 0, and with a plane that keeps the same
//! half. A type that honours the clip loses about half its pixels; a type that
//! ignores it loses none. The comparison is between frames from the same
//! camera, so it does not depend on where x = 0 lands on screen.
//!
//! Then a pick at a pixel the box cut away, through the GPU pick pass and the
//! CPU ray cast, with the box in the frame. A pick there should find nothing:
//! the viewer is looking at empty space.
//!
//! Run with `--nocapture` for the table. The test asserts only that every item
//! drew; the coverage it reports is the subject of
//! `clipping-coverage-uneven-across-item-types` and is not yet a contract.

#![cfg(feature = "item-types")]

use viewport_lib::plugin_api::{PluginItem, Uploads};
use viewport_lib::wgpu;
use viewport_lib::vplg::volume::grid::VolumeData;
use viewport_lib::{
    CameraFrame, ClipObject, Colour, ColourSource, ColourmapId, Material, MeshId, PolylineItem,
    SceneFrame, SceneRenderItem, SizeSource, SurfaceSubmission, primitives,
};
use viewport_lib_plugins::item_types::curves::TubeItem;
use viewport_lib_plugins::item_types::gaussian_splat::{
    GaussianSplatData, GaussianSplatItem, ShDegree,
};
use viewport_lib_plugins::item_types::gpu_implicit::{
    GpuImplicitItem, ImplicitBlendMode, ImplicitPrimitive,
};
use viewport_lib_plugins::item_types::gpu_marching_cubes::{GpuMarchingCubesItem, McVolumes};
use viewport_lib_plugins::item_types::point_cloud::PointCloudItem;
use viewport_lib_plugins::item_types::vector_field::VectorFieldItem;
use viewport_lib_plugins::item_types::volume::VolumeItem;

mod common;
use common::*;

const W: u32 = 160;
const H: u32 = 120;
const ID: u64 = 4242;

/// Keep x < 0: a box from x = -10 to 0, and the plane -x >= 0.
fn box_clip() -> ClipObject {
    ClipObject::box_shape(
        [-5.0, 0.0, 0.0],
        [5.0, 10.0, 10.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
}

fn plane_clip() -> ClipObject {
    ClipObject::plane([-1.0, 0.0, 0.0], 0.0)
}

fn frame(scene: SceneFrame, clip: Option<ClipObject>) -> FrameData {
    let mut camera = Camera {
        distance: 7.0,
        ..Camera::default()
    };
    camera.set_aspect_ratio(W as f32, H as f32);
    let mut frame = FrameData::new(CameraFrame::from_camera(&camera, [W as f32, H as f32]), scene);
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    if let Some(c) = clip {
        frame.effects.clip.objects.push(c);
    }
    frame
}

/// A scene each time it is asked for: a frame takes its scene by value and a
/// `SceneFrame` is not `Clone`.
type Scene = Box<dyn Fn() -> SceneFrame>;

fn one<T: PluginItem + Clone + 'static>(item: T) -> Scene {
    Box::new(move || {
        let mut scene = SceneFrame::default();
        scene.items_mut::<T>().push(item.clone());
        scene
    })
}

/// Which pixels differ from the empty frame by more than a rounding step.
fn drawn(base: &[u8], pixels: &[u8]) -> Vec<bool> {
    base.chunks_exact(4)
        .zip(pixels.chunks_exact(4))
        .map(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > 8))
        .collect()
}

struct Row {
    name: &'static str,
    full: usize,
    boxed: usize,
    planed: usize,
    gpu_pick_in_cut: Option<bool>,
    cpu_pick_in_cut: Option<bool>,
}

/// A drawn-pixel mask as a greyscale image, for looking at what a row counted.
fn write_mask(path: &str, mask: &[bool]) {
    let mut out = format!("P5\n{W} {H}\n255\n").into_bytes();
    out.extend(mask.iter().map(|&d| if d { 255u8 } else { 0 }));
    std::fs::write(path, out).expect("write mask");
}

fn removed(full: usize, kept: usize) -> f32 {
    if full == 0 { 0.0 } else { 100.0 * (1.0 - kept as f32 / full as f32) }
}

#[test]
fn clip_coverage_by_item_type() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);

    let mut scenes: Vec<(&'static str, Scene)> = Vec::new();

    let sphere = renderer
        .resources_mut()
        .upload_mesh_data(&device, &primitives::sphere(1.4, 32, 16))
        .expect("sphere");
    scenes.push(("surface mesh", surface(sphere)));
    scenes.push(("polyline", one(polyline())));
    scenes.push(("point cloud", one(point_cloud())));
    scenes.push(("tube", one(tube())));
    scenes.push(("GPU implicit", one(implicit())));
    scenes.push(("volume", one(volume(&mut renderer, &device, &queue))));
    scenes.push(("vector field", one(vector_field(&mut renderer, &device))));
    scenes.push(("marching cubes", one(marching_cubes(&mut renderer, &device, &queue))));
    scenes.push(("Gaussian splats", one(splats(&mut renderer, &device, &queue))));

    let empty = renderer.render_offscreen(&device, &queue, &frame(SceneFrame::default(), None), W, H);

    let mut rows = Vec::new();
    for (name, scene) in scenes {
        let full_frame = frame(scene(), None);
        let full = drawn(&empty, &renderer.render_offscreen(&device, &queue, &full_frame, W, H));
        let boxed_frame = frame(scene(), Some(box_clip()));
        let boxed = drawn(&empty, &renderer.render_offscreen(&device, &queue, &boxed_frame, W, H));
        let planed = drawn(
            &empty,
            &renderer.render_offscreen(&device, &queue, &frame(scene(), Some(plane_clip())), W, H),
        );

        // A pixel the box cut away, as far from the kept part as there is.
        let cut = (0..(W * H) as usize)
            .filter(|&i| full[i] && !boxed[i])
            .max_by_key(|&i| i % W as usize);
        let (gpu_pick_in_cut, cpu_pick_in_cut) = match cut {
            Some(i) => {
                let at = glam::Vec2::new((i % W as usize) as f32 + 0.5, (i / W as usize) as f32 + 0.5);
                let _ = renderer.pass().prepare(&device, &queue, &boxed_frame);
                let gpu = renderer
                    .pick_object(PickBackend::Gpu, at, &boxed_frame, &device, &queue, PickMask::OBJECT)
                    .is_some();
                let cpu = renderer
                    .pick_object(PickBackend::Cpu, at, &boxed_frame, &device, &queue, PickMask::OBJECT)
                    .is_some();
                (Some(gpu), Some(cpu))
            }
            None => (None, None),
        };

        if let Ok(dir) = std::env::var("CLIP_COVERAGE_DUMP") {
            for (tag, mask) in [("full", &full), ("box", &boxed), ("plane", &planed)] {
                write_mask(&format!("{dir}/{}-{tag}.pgm", name.replace(' ', "_")), mask);
            }
        }
        let count = |v: &[bool]| v.iter().filter(|&&d| d).count();
        rows.push(Row {
            name,
            full: count(&full),
            boxed: count(&boxed),
            planed: count(&planed),
            gpu_pick_in_cut,
            cpu_pick_in_cut,
        });
    }

    let yes_no = |v: Option<bool>| match v {
        Some(true) => "hit",
        Some(false) => "nothing",
        None => "-",
    };
    eprintln!(
        "\n{:<18} {:>7} {:>13} {:>15} {:>16} {:>16}",
        "item", "pixels", "box removes", "plane removes", "GPU pick in cut", "CPU pick in cut"
    );
    for r in &rows {
        eprintln!(
            "{:<18} {:>7} {:>12.0}% {:>14.0}% {:>16} {:>16}",
            r.name,
            r.full,
            removed(r.full, r.boxed),
            removed(r.full, r.planed),
            yes_no(r.gpu_pick_in_cut),
            yes_no(r.cpu_pick_in_cut),
        );
    }
    for r in &rows {
        assert!(r.full > 200, "{} drew {} pixels", r.name, r.full);
    }
}

fn surface(mesh: MeshId) -> Scene {
    Box::new(move || {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.settings.colour = Some(Colour::linear_rgb(0.6, 0.6, 0.7));
        item.settings.pick_id = PickId(ID);
        let mut scene = SceneFrame::default();
        scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        scene
    })
}

fn polyline() -> PolylineItem {
    let n = 64;
    let mut line = PolylineItem::default();
    line.positions = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            [-1.6 + 3.2 * t, 0.8 * (t * 12.0).sin(), 0.0]
        })
        .collect();
    line.strip_lengths = vec![n as u32];
    line.line_width = 6.0;
    line.default_colour = Colour::linear_rgb(0.9, 0.8, 0.2);
    line.settings.pick_id = PickId(ID);
    line
}

fn point_cloud() -> PointCloudItem {
    let n = 3000;
    let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let z = 1.0 - (i as f32 / (n - 1) as f32) * 2.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let theta = golden * i as f32;
            [theta.cos() * r * 1.5, theta.sin() * r * 1.5, z * 1.5]
        })
        .collect();
    let mut pc = PointCloudItem::default();
    pc.colour = ColourSource::Scalar {
        values: positions.iter().map(|p| p[2]).collect(),
        range: None,
        colourmap: None,
    };
    pc.positions = positions;
    pc.size = SizeSource::Uniform(6.0);
    pc.settings.pick_id = PickId(ID);
    pc
}

fn tube() -> TubeItem {
    let n = 96;
    let mut tube = TubeItem::default();
    tube.positions = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let a = t * 3.0 * std::f32::consts::TAU;
            [1.2 * a.cos(), 1.2 * a.sin(), 2.4 * (t - 0.5)]
        })
        .collect();
    tube.strip_lengths = vec![n as u32];
    tube.radius = 0.15;
    tube.colour = Colour::linear(0.85, 0.35, 0.2, 1.0);
    tube.settings.pick_id = PickId(ID);
    tube
}

fn implicit() -> GpuImplicitItem {
    let sphere = |c: [f32; 3], r: f32, colour: Colour| {
        let mut p = ImplicitPrimitive::zeroed();
        p.kind = 1;
        p.blend = 0.35;
        p.params[..3].copy_from_slice(&c);
        p.params[3] = r;
        p.colour = colour;
        p
    };
    let mut item = GpuImplicitItem::default();
    item.primitives = vec![
        sphere([-0.7, 0.0, 0.3], 0.8, Colour::linear(0.38, 0.12, 0.62, 1.0)),
        sphere([0.7, 0.2, 0.4], 0.65, Colour::linear(0.12, 0.5, 0.55, 1.0)),
    ];
    item.blend_mode = ImplicitBlendMode::SmoothUnion;
    item.settings.pick_id = PickId(ID);
    item
}

fn volume(renderer: &mut ViewportRenderer, device: &wgpu::Device, queue: &wgpu::Queue) -> VolumeItem {
    let n = 24u32;
    let c = (n - 1) as f32 * 0.5;
    let mut data = Vec::with_capacity((n * n * n) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let d = glam::Vec3::new(x as f32 - c, y as f32 - c, z as f32 - c) / c;
                data.push(0.65 - d.length());
            }
        }
    }
    let volume_id = renderer.resources_mut().upload_volume(device, queue, &data, [n; 3]);
    let mut v = VolumeItem::default();
    v.volume_id = volume_id;
    v.colour_lut = Some(ColourmapId(0));
    v.scalar_range = (-0.4, 0.65);
    v.threshold_min = -0.4;
    v.threshold_max = 0.65;
    v.bbox_min = [-1.6; 3];
    v.bbox_max = [1.6; 3];
    v.enable_shading = true;
    v.settings.pick_id = PickId(ID);
    v
}

fn vector_field(renderer: &mut ViewportRenderer, device: &wgpu::Device) -> VectorFieldItem {
    let shape = renderer
        .resources_mut()
        .upload_mesh_data(device, &primitives::arrow(0.08, 0.2, 0.35, 12))
        .expect("arrow");
    let mut field = VectorFieldItem::new(shape);
    for i in 0..8 {
        for j in 0..8 {
            let x = (i as f32 - 3.5) * 0.4;
            let y = (j as f32 - 3.5) * 0.4;
            field.positions.push([x, y, 0.0]);
            field.vectors.push([-y * 0.3, x * 0.3, 0.25]);
        }
    }
    field.scale = 0.8;
    field.settings.pick_id = PickId(ID);
    field
}

fn marching_cubes(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> GpuMarchingCubesItem {
    let n: u32 = 32;
    let origin = [-1.6f32; 3];
    let step = 3.2 / (n - 1) as f32;
    let mut data = Vec::with_capacity((n * n * n) as usize);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let p = glam::Vec3::new(ix as f32, iy as f32, iz as f32) * step + glam::Vec3::from(origin);
                data.push(1.3 - p.length());
            }
        }
    }
    let vol = VolumeData {
        data,
        dims: [n; 3],
        origin,
        spacing: [step; 3],
    };
    let volume_id = renderer.upload_volume_for_mc(device, queue, &vol).expect("mc volume");
    let mut item = GpuMarchingCubesItem {
        volume_id,
        isovalue: 0.0,
        material: Material::from_colour(Colour::linear_rgb(0.45, 0.48, 0.52)),
        settings: Default::default(),
        cpu_data: None,
    };
    item.settings.pick_id = PickId(ID);
    item
}

fn splats(renderer: &mut ViewportRenderer, device: &wgpu::Device, queue: &wgpu::Queue) -> GaussianSplatItem {
    const SH0: f32 = 0.282_094_79;
    let mut sd = GaussianSplatData::default();
    sd.sh_degree = ShDegree::Zero;
    let count = 24;
    for i in 0..count {
        let a = i as f32 / count as f32 * std::f32::consts::TAU;
        sd.positions.push([1.4 * a.cos(), 1.4 * a.sin(), 0.0]);
        sd.scales.push([0.3; 3]);
        sd.rotations.push([0.0, 0.0, 0.0, 1.0]);
        sd.opacities.push(0.85);
        sd.sh_coefficients
            .extend_from_slice(&[(0.8 - 0.5) / SH0, (0.4 - 0.5) / SH0, (0.3 - 0.5) / SH0]);
    }
    let source = renderer.upload(device, queue, &sd).expect("splats");
    let mut item = GaussianSplatItem::default();
    item.source = source;
    item.settings.pick_id = PickId(ID);
    item
}
