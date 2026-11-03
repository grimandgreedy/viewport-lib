//! The built-in item types on the page: one item of each type that registers
//! below the full tier, drawn alone into an offscreen target under error scopes
//! and read back, so a type whose shaders build but draw nothing on this driver
//! fails here instead of passing silently.
//!
//! Below the full tier the Gaussian splat depth sort runs on the CPU, and this
//! target has one thread, so opening the page with `?sort` also times it at
//! 100k and 1M splats (`?sort=<count>` for one size). Off by default: headless
//! Chrome with its GPU process off rasterises in software, where a million
//! splats take seconds a frame and the timings say nothing.

use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use viewport_lib::plugin_api::{PluginItem, Uploads};
use viewport_lib::vplg::volume::grid::VolumeData;
use viewport_lib::{
    Camera, CameraFrame, Colour, ColourSource, ColourmapId, FrameData, Material, SceneFrame,
    SizeSource, Tier, ViewportRenderer, ViewportResult, primitives, wgpu,
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
use web_time::Instant;

use crate::Results;

const W: u32 = 128;
const H: u32 = 96;

/// Fewer pixels than this differing from an empty frame counts as drawing
/// nothing. Every item below covers well over a thousand at this size.
const MIN_PIXELS: usize = 200;

/// The splat set sizes `?sort` times the sort at.
const SORT_COUNTS: [usize; 2] = [100_000, 1_000_000];

/// The page's device and renderer, borrowed for the checks.
pub struct Gpu<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub renderer: &'a mut ViewportRenderer,
    pub format: wgpu::TextureFormat,
}

/// Register the built-in item types, then draw one item of each type the tier
/// runs and time the splat sort.
pub async fn check(gpu: &mut Gpu<'_>, results: &mut Results) {
    let declined = viewport_lib_plugins::item_types::install(gpu.renderer, gpu.device);
    results.pass(format!(
        "built-in item types registered, {} declined on this tier",
        declined.len()
    ));
    let target = Target::new(gpu.device, gpu.format);
    let empty = match target.draw(gpu, SceneFrame::default(), 0.0).await {
        Ok(pixels) => pixels,
        Err(e) => {
            results.fail(format!("empty item-type frame: {e}"));
            return;
        }
    };

    let mut cases: Vec<(&str, Case)> = vec![
        ("GPU implicit surface", case(implicit())),
        ("tube", case(tube())),
        ("point cloud", case(point_cloud())),
    ];
    let uploads: [(&str, ViewportResult<Case>); 4] = [
        ("volume", volume(gpu).map(case)),
        ("vector field", vector_field(gpu).map(case)),
        ("marching cubes isosurface", marching_cubes(gpu).map(case)),
        ("Gaussian splats", splats(gpu, 12, 0.3).map(case)),
    ];
    for (name, scene) in uploads {
        match scene {
            Ok(scene) => cases.push((name, scene)),
            Err(e) => results.fail(format!("{name} upload: {e}")),
        }
    }

    for (name, scene) in cases {
        // A pipeline compiling in the background draws nothing on its first
        // frame, so the second frame is the one read.
        if let Err(e) = target.draw(gpu, scene(), 0.0).await {
            results.fail(format!("{name}: {e}"));
            continue;
        }
        match target.draw(gpu, scene(), 0.0).await {
            Ok(pixels) => {
                let drawn = differing(&empty, &pixels);
                if drawn >= MIN_PIXELS {
                    results.pass(format!("{name} drawn, {drawn} pixels"));
                } else {
                    results.fail(format!(
                        "{name} drew {drawn} pixels with no GPU error, expected {MIN_PIXELS} or more"
                    ));
                }
            }
            Err(e) => results.fail(format!("{name}: {e}")),
        }
    }

    if gpu.renderer.tier() == Tier::Full {
        results.pass("splat depth sort runs on the GPU at this tier, so it is not timed");
        return;
    }
    let Some(counts) = sort_counts() else {
        results.pass("splat depth sort not timed; open the page with ?sort to time it");
        return;
    };
    for count in counts {
        match time_sort(gpu, &target, count).await {
            Ok((moved, still)) => results.pass(format!(
                "splat frame at {count} splats: {moved:.1} ms with the eye moved (re-sorted), \
                 {still:.1} ms with it still, so about {:.1} ms a million splats to sort",
                (moved - still).max(0.0) * 1_000_000.0 / count as f32
            )),
            Err(e) => results.fail(format!("splat sort at {count} splats: {e}")),
        }
    }
}

/// The splat counts the page's `?sort` query asks for, or `None` without it.
#[cfg(target_arch = "wasm32")]
fn sort_counts() -> Option<Vec<usize>> {
    let search = web_sys::window()?.location().search().ok()?;
    let value = search
        .trim_start_matches('?')
        .split('&')
        .find_map(|kv| kv.strip_prefix("sort").map(|v| v.trim_start_matches('=')))?;
    Some(match value.parse() {
        Ok(count) => vec![count],
        Err(_) => SORT_COUNTS.to_vec(),
    })
}

/// Natively the sort runs on every launch, at both sizes.
#[cfg(not(target_arch = "wasm32"))]
fn sort_counts() -> Option<Vec<usize>> {
    Some(SORT_COUNTS.to_vec())
}

/// Builds the frame for one check.
type Case = Box<dyn Fn() -> SceneFrame>;

/// A frame holding one item.
fn one<T: PluginItem>(item: T) -> SceneFrame {
    let mut scene = SceneFrame::default();
    scene.items_mut::<T>().push(item);
    scene
}

fn case<T: PluginItem + Clone>(item: T) -> Case {
    Box::new(move || one(item.clone()))
}

/// How many pixels differ from `base` by more than a rounding step in a colour
/// channel.
fn differing(base: &[u8], pixels: &[u8]) -> usize {
    base.chunks_exact(4)
        .zip(pixels.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > 8))
        .count()
}

/// The CPU time of a splat frame's `render` call with the eye moved since the
/// last frame, which re-sorts, and with it still, which does not. Medians of
/// three each.
async fn time_sort(gpu: &mut Gpu<'_>, target: &Target, count: usize) -> Result<(f32, f32), String> {
    let item = splats(gpu, count, 0.01).map_err(|e| e.to_string())?;
    // The first frame packs the set into its data textures; not timed.
    target.draw(gpu, one(item.clone()), 0.0).await?;
    let (mut moved, mut still) = (Vec::new(), Vec::new());
    for i in 1..=3 {
        let yaw = 0.4 * i as f32;
        moved.push(target.time(gpu, one(item.clone()), yaw).await?);
        still.push(target.time(gpu, one(item.clone()), yaw).await?);
    }
    Ok((median(moved), median(still)))
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// An offscreen colour target the checks draw into, and a buffer to read it
/// back through.
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
}

impl Target {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("item type probe"),
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
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // W * 4 bytes is a multiple of the 256-byte row alignment.
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("item type probe readback"),
            size: (W * H * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            texture,
            view,
            readback,
        }
    }

    fn frame(scene: SceneFrame, yaw: f32) -> FrameData {
        let mut camera = Camera::orbit(glam::DVec3::ZERO, 7.0, Camera::default().orientation());
        camera.set_orbit_orientation(glam::Quat::from_rotation_z(yaw) * camera.orientation());
        camera.set_aspect_ratio(W as f32, H as f32);
        FrameData::new(
            CameraFrame::from_camera(&camera, [W as f32, H as f32]),
            scene,
        )
    }

    /// Draw `scene` under validation and internal error scopes and return the
    /// target's pixels. Internal, because a GLSL compile or link failure on
    /// wgpu's GL backend is reported as one.
    async fn draw(
        &self,
        gpu: &mut Gpu<'_>,
        scene: SceneFrame,
        yaw: f32,
    ) -> Result<Vec<u8>, String> {
        let internal = viewport_lib::gpu::push_error_scope(gpu.device, wgpu::ErrorFilter::Internal);
        let validation =
            viewport_lib::gpu::push_error_scope(gpu.device, wgpu::ErrorFilter::Validation);
        let cmd = gpu.renderer.owned().render(
            gpu.device,
            gpu.queue,
            &self.view,
            &Self::frame(scene, yaw),
        );
        gpu.queue.submit(std::iter::once(cmd));
        let pixels = self.read(gpu).await;
        let validation = validation.pop().await;
        let internal = internal.pop().await;
        if let Some(e) = validation.or(internal) {
            return Err(e.to_string());
        }
        pixels
    }

    /// The CPU time of `render` for `scene`, in milliseconds. Reads the frame
    /// back afterwards so the GPU work does not pile up across timed frames.
    async fn time(&self, gpu: &mut Gpu<'_>, scene: SceneFrame, yaw: f32) -> Result<f32, String> {
        let frame = Self::frame(scene, yaw);
        let t0 = Instant::now();
        let cmd = gpu
            .renderer
            .owned()
            .render(gpu.device, gpu.queue, &self.view, &frame);
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        gpu.queue.submit(std::iter::once(cmd));
        self.read(gpu).await?;
        Ok(ms)
    }

    async fn read(&self, gpu: &Gpu<'_>) -> Result<Vec<u8>, String> {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("item type probe copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(W * 4),
                    rows_per_image: Some(H),
                },
            },
            wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.readback.slice(..);
        let state = Arc::new(Mutex::new(MapState::default()));
        let callback_state = state.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let mut s = callback_state.lock().unwrap();
            s.result = Some(result);
            if let Some(waker) = s.waker.take() {
                waker.wake();
            }
        });
        wait_for_map(gpu.device, &state).await?;
        let pixels = viewport_lib::gpu::mapped_range(slice).to_vec();
        self.readback.unmap();
        Ok(pixels)
    }
}

#[derive(Default)]
struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

/// Wait for a `map_async` callback. Natively the device is polled until it
/// fires. In a browser the WebGPU backend resolves it on its own, while wgpu's
/// GL backend resolves it only from `poll`, so the page polls and yields to the
/// browser between tries.
async fn wait_for_map(device: &wgpu::Device, state: &Arc<Mutex<MapState>>) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(10)),
    });
    #[cfg(target_arch = "wasm32")]
    for _ in 0..10_000 {
        if state.lock().unwrap().result.is_some() {
            break;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        yield_to_browser().await;
    }
    let result = std::future::poll_fn(|cx| {
        let mut s = state.lock().unwrap();
        match s.result.take() {
            Some(r) => Poll::Ready(r),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    })
    .await;
    result.map_err(|e| format!("reading the frame back: {e}"))
}

/// Let the browser run a task, so a GL fence can signal between polls.
#[cfg(target_arch = "wasm32")]
async fn yield_to_browser() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        if let Some(window) = web_sys::window() {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 1);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
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
    item
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
    tube
}

fn point_cloud() -> PointCloudItem {
    // A Fibonacci sphere coloured by height.
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
    pc
}

fn volume(gpu: &mut Gpu<'_>) -> ViewportResult<VolumeItem> {
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
    let volume_id = gpu
        .renderer
        .resources_mut()
        .upload_volume(gpu.device, gpu.queue, &data, [n; 3]);
    let mut v = VolumeItem::default();
    v.volume_id = volume_id;
    v.colour_lut = Some(ColourmapId(0));
    v.scalar_range = (-0.4, 0.65);
    v.threshold_min = -0.4;
    v.threshold_max = 0.65;
    v.bbox_min = [-1.6; 3];
    v.bbox_max = [1.6; 3];
    v.enable_shading = true;
    Ok(v)
}

fn vector_field(gpu: &mut Gpu<'_>) -> ViewportResult<VectorFieldItem> {
    let shape = gpu
        .renderer
        .resources_mut()
        .upload_mesh_data(gpu.device, &primitives::arrow(0.08, 0.2, 0.35, 12))?;
    let mut field = VectorFieldItem::new(shape);
    for i in 0..8 {
        for j in 0..8 {
            let x = (i as f32 - 3.5) * 0.6;
            let y = (j as f32 - 3.5) * 0.6;
            field.positions.push([x, y, 0.0]);
            field.vectors.push([-y * 0.3, x * 0.3, 0.25]);
        }
    }
    field.scale = 0.8;
    Ok(field)
}

fn marching_cubes(gpu: &mut Gpu<'_>) -> ViewportResult<GpuMarchingCubesItem> {
    // A gyroid over a 32^3 grid, at isovalue 0.
    let n: u32 = 32;
    let origin = [-2.0f32; 3];
    let step = 4.0 / (n - 1) as f32;
    let mut data = Vec::with_capacity((n * n * n) as usize);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let p = glam::Vec3::new(ix as f32, iy as f32, iz as f32) * step
                    + glam::Vec3::from(origin);
                let q = p * 1.6;
                data.push(q.x.sin() * q.y.cos() + q.y.sin() * q.z.cos() + q.z.sin() * q.x.cos());
            }
        }
    }
    let vol = VolumeData {
        data,
        dims: [n; 3],
        origin,
        spacing: [step; 3],
    };
    let volume_id = gpu
        .renderer
        .upload_volume_for_mc(gpu.device, gpu.queue, &vol)?;
    Ok(GpuMarchingCubesItem {
        volume_id,
        isovalue: 0.0,
        material: Material::from_colour(Colour::linear_rgb(0.45, 0.48, 0.52)),
        settings: Default::default(),
        cpu_data: None,
    })
}

/// `count` degree-0 splats of radius `scale`: a ring for a handful, a filled
/// ball for the timed sets.
fn splats(gpu: &mut Gpu<'_>, count: usize, scale: f32) -> ViewportResult<GaussianSplatItem> {
    const SH0: f32 = 0.282_094_79;
    let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let mut sd = GaussianSplatData::default();
    sd.sh_degree = ShDegree::Zero;
    for i in 0..count {
        let t = i as f32 / count as f32;
        let p = if count <= 64 {
            let a = t * std::f32::consts::TAU;
            [1.4 * a.cos(), 1.4 * a.sin(), 0.0]
        } else {
            // Spread through a ball, so the sort order changes with the eye.
            let z = 1.0 - 2.0 * t;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let a = golden * i as f32;
            let s = 1.5 * ((i * 7919) % count) as f32 / count as f32;
            [a.cos() * r * s, a.sin() * r * s, z * s]
        };
        sd.positions.push(p);
        sd.scales.push([scale; 3]);
        sd.rotations.push([0.0, 0.0, 0.0, 1.0]);
        sd.opacities.push(0.85);
        let (r, g, b) = (0.2 + 0.7 * t, 0.3, 0.85 - 0.6 * t);
        sd.sh_coefficients
            .extend_from_slice(&[(r - 0.5) / SH0, (g - 0.5) / SH0, (b - 0.5) / SH0]);
    }
    let source = gpu.renderer.upload(gpu.device, gpu.queue, &sd)?;
    let mut item = GaussianSplatItem::default();
    item.source = source;
    Ok(item)
}
