//! What each item type costs: its registration, and the pipelines it compiles on
//! the first frame that draws one.
//!
//! Registration is timed per plugin on one renderer. First use is measured per
//! type on a fresh harness: a mesh-only frame runs first so the renderer's own
//! pipelines are already built, then the type's catalogue scene is drawn and
//! the build log read back, so what is listed belongs to that frame alone.
//!
//! ```bash
//! cargo run --release --example item_type_cost
//! ```

use std::time::Instant;

use viewport_lib::resources::build_log;
use viewport_lib::{ViewportRenderer, wgpu};
use viewport_lib_item_types as types;
use viewport_lib_testkit::{
    DeviceProfile, Harness, frame_for, headless_device_with, scene_by_name,
};

const SIZE: u32 = 512;

/// The catalogue scene that draws each item type.
const SCENES: &[(&str, &str)] = &[
    ("image slice", "image_slice"),
    ("volume surface slice", "volume_surface_slice"),
    ("point cloud", "point_cloud"),
    ("gaussian splat", "gaussian_splats"),
    ("gpu implicit", "gpu_implicit"),
    ("gpu marching cubes", "gpu_marching_cubes"),
    ("volume", "volume"),
    ("streamtube", "streamtubes"),
    ("tube", "tubes"),
    ("tensor field", "tensor_fields"),
    ("vector field", "vector_fields"),
    ("ribbon", "ribbons"),
    ("sprite", "sprites"),
    ("gpu particles", "gpu_particles"),
    ("scatter volume", "scatter_volume"),
    ("polyline (built in)", "polyline"),
    ("decal", "decals"),
];

fn ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}

fn main() {
    build_log::enable();

    // ---- Construction and registration, on one renderer.
    let (device, _queue) = headless_device_with(&DeviceProfile::harness()).expect("no GPU adapter");
    let t = Instant::now();
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let new_ms = ms(t);
    let new_builds = build_log::drain();
    println!(
        "ViewportRenderer::new        {new_ms:7.2} ms, {} pipelines/modules",
        new_builds.len()
    );

    println!();
    println!("registration, per item type");
    let mut total = 0.0;
    macro_rules! register {
        ($($name:literal => $plugin:ty),* $(,)?) => {$(
            let t = Instant::now();
            renderer.with_item_type_plugin(&device, Box::new(<$plugin>::default()));
            let took = ms(t);
            total += took;
            let built = build_log::drain();
            let built_ms: f32 = built.iter().map(|(_, ms)| ms).sum();
            println!(
                "  {:<22} {took:6.3} ms, {} pipelines/modules ({built_ms:.2} ms)",
                $name,
                built.len()
            );
        )*};
    }
    register!(
        "image slice" => types::ImageSlicePlugin,
        "volume surface slice" => types::VolumeSurfaceSlicePlugin,
        "point cloud" => types::PointCloudPlugin,
        "gaussian splat" => types::GaussianSplatPlugin,
        "gpu implicit" => types::GpuImplicitPlugin,
        "gpu marching cubes" => types::GpuMarchingCubesPlugin,
        "volume" => types::VolumePlugin,
        "streamtube" => types::StreamtubePlugin,
        "tube" => types::TubePlugin,
        "tensor field" => types::TensorFieldPlugin,
        "vector field" => types::VectorFieldPlugin,
        "ribbon" => types::RibbonPlugin,
        "external instances" => types::ExternalInstancesPlugin,
        "sprite" => types::SpritePlugin,
        "gpu particles" => types::GpuParticlesPlugin,
        "scatter volume" => types::ScatterVolumePlugin,
        "decal" => types::DecalPlugin,
    );
    println!("  {:<22} {total:6.3} ms", "all seventeen");
    drop(renderer);

    // ---- First use, per type, each on a fresh harness.
    println!();
    println!("first frame that draws each type (renderer's own pipelines already built)");
    println!(
        "  {:<22} {:>8} {:>10} {:>10}",
        "type", "objects", "build ms", "frame ms"
    );
    let warm = scene_by_name("primitives_trio").expect("primitives_trio");
    for (label, name) in SCENES {
        let Some(scene) = scene_by_name(name) else {
            println!("  {label:<22} (no catalogue scene named {name})");
            continue;
        };
        let mut h = Harness::new().expect("no GPU adapter");
        let built = h.build_scene(&warm);
        let frame = frame_for(&built, &warm.cameras[0].camera, [SIZE as f32, SIZE as f32]);
        let _ = h.render(&frame, SIZE, SIZE);
        let _ = h.render(&frame, SIZE, SIZE);
        let _ = build_log::drain();

        let built = h.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [SIZE as f32, SIZE as f32]);
        // Uploading the scene's content may itself compile something.
        let upload_builds = build_log::drain();
        let t = Instant::now();
        let _ = h.render(&frame, SIZE, SIZE);
        let frame_ms = ms(t);
        let mut builds = build_log::drain();
        builds.extend(upload_builds);
        let build_ms: f32 = builds.iter().map(|(_, ms)| ms).sum();
        println!(
            "  {label:<22} {:>8} {build_ms:>10.2} {frame_ms:>10.2}",
            builds.len()
        );
        if std::env::var("VPL_ITEM_TYPE_DETAIL").is_ok() {
            builds.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (l, ms) in builds.iter().take(6) {
                println!("      {l:<44} {ms:7.3} ms");
            }
        }
    }
}
