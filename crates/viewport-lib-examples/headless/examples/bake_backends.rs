//! Times a directional lightmap solve on each ray traversal backend.
//!
//! The same scene and the same texel surfaces are solved with the software
//! (compute BVH) backend and, where the device offers ray queries, the
//! hardware one, so the two can be compared on one machine:
//!
//!   cargo run --release -p viewport-lib-examples-headless --example bake-backends \
//!       --features raytrace-hardware
//!
//! Without `raytrace-hardware`, or on a device without ray queries, only the
//! software backend runs.

use glam::Vec3;
use std::time::Instant;
use viewport_lib as vpl;
use vpl::primitives;
use vpl::raytrace::{RtBackend, RtLight, RtMaterial, RtScene, RtSettings, TexelSurfaces, Tracer};

/// Atlas side in texels; the surfaces are a floor grid of this size.
const SIDE: u32 = 1024;
const SAMPLES: u32 = 64;

fn add_primitive(scene: &mut RtScene, mesh: &vpl::MeshData, origin: Vec3, colour: [f32; 3]) {
    let positions: Vec<Vec3> = mesh
        .positions
        .iter()
        .map(|p| Vec3::from(*p) + origin)
        .collect();
    let normals: Vec<Vec3> = mesh.normals.iter().map(|n| Vec3::from(*n)).collect();
    scene.add_mesh(
        &positions,
        &mesh.indices,
        Some(&normals),
        RtMaterial {
            base_colour: colour.into(),
            roughness: 0.8,
            ..RtMaterial::default()
        },
    );
}

fn main() {
    let instance = vpl::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&vpl::gpu::RequestAdapterOptions {
        power_preference: vpl::gpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .expect("no GPU adapter");
    let offers_ray_query = adapter.features().contains(vpl::gpu::RAY_QUERY_FEATURE);
    println!(
        "{} ({:?}), ray queries offered: {offers_ray_query}",
        adapter.get_info().name,
        adapter.get_info().backend
    );
    let features = if offers_ray_query {
        vpl::gpu::RAY_QUERY_FEATURE
    } else {
        vpl::gpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&vpl::gpu::DeviceDescriptor {
        required_features: features,
        // Ray queries also need the acceleration-structure limits, which
        // default to zero; ask for everything the adapter has.
        required_limits: if offers_ray_query {
            adapter.limits()
        } else {
            vpl::gpu::Limits::default()
        },
        // Ray queries are an experimental wgpu feature and have to be opted into.
        experimental_features: if offers_ray_query {
            unsafe { vpl::gpu::ExperimentalFeatures::enabled() }
        } else {
            vpl::gpu::ExperimentalFeatures::default()
        },
        ..Default::default()
    }))
    .expect("no device");

    // A Z-up room: floor, two walls, and a few objects to bounce between.
    let mut scene = RtScene::new();
    add_primitive(
        &mut scene,
        &primitives::cuboid(20.0, 14.0, 0.2),
        Vec3::new(0.0, 0.0, -0.1),
        [0.7, 0.7, 0.7],
    );
    add_primitive(
        &mut scene,
        &primitives::cuboid(0.2, 14.0, 7.0),
        Vec3::new(-10.0, 0.0, 3.5),
        [0.7, 0.2, 0.2],
    );
    add_primitive(
        &mut scene,
        &primitives::cuboid(20.0, 0.2, 7.0),
        Vec3::new(0.0, 7.0, 3.5),
        [0.6, 0.6, 0.6],
    );
    add_primitive(
        &mut scene,
        &primitives::torus(1.9, 0.7, 64, 32),
        Vec3::new(0.0, 1.0, 1.1),
        [0.8, 0.6, 0.3],
    );
    add_primitive(
        &mut scene,
        &primitives::icosphere(1.5, 4),
        Vec3::new(-4.6, -1.5, 1.5),
        [0.3, 0.5, 0.8],
    );
    add_primitive(
        &mut scene,
        &primitives::torus(1.3, 0.5, 96, 48),
        Vec3::new(0.0, -4.2, 1.35),
        [0.5, 0.8, 0.4],
    );
    scene.add_light(RtLight::Directional {
        direction: [0.4, -0.3, 0.85],
        colour: [3.0, 2.9, 2.7].into(),
    });
    println!("triangles: {}", scene.triangle_count());

    // Texel surfaces: the floor, one texel per atlas cell, facing up.
    let texels = (SIDE * SIDE) as usize;
    let mut world_pos = Vec::with_capacity(texels);
    for y in 0..SIDE {
        for x in 0..SIDE {
            let u = (x as f32 + 0.5) / SIDE as f32;
            let v = (y as f32 + 0.5) / SIDE as f32;
            world_pos.push([(u - 0.5) * 19.0, (v - 0.5) * 13.0, 0.001, 1.0]);
        }
    }
    let world_normal = vec![[0.0, 0.0, 1.0, 0.0]; texels];
    let surfaces = TexelSurfaces {
        width: SIDE,
        height: SIDE,
        world_pos: &world_pos,
        world_normal: &world_normal,
    };
    let settings = RtSettings {
        samples: SAMPLES,
        max_bounces: 4,
        denoise: false,
        seed: 0,
    };

    let mut backends = vec![RtBackend::Software];
    if cfg!(feature = "raytrace-hardware") && offers_ray_query {
        backends.push(RtBackend::Hardware);
    }
    for backend in backends {
        let t = Instant::now();
        let mut tracer = Tracer::new_with_backend(&device, &queue, &scene, backend);
        let setup_ms = t.elapsed().as_secs_f32() * 1000.0;
        // One untimed solve first, so shader compilation and first-use costs
        // stay out of the comparison.
        let first = tracer.bake_directional(&device, &queue, &surfaces, &settings);
        // Mean irradiance, so the backends can be checked to agree.
        let mean: f64 = first
            .irradiance
            .chunks_exact(4)
            .map(|px| f64::from(px[0] + px[1] + px[2]) / 3.0)
            .sum::<f64>()
            / texels as f64;
        let mut best = f32::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let _ = tracer.bake_directional(&device, &queue, &surfaces, &settings);
            best = best.min(t.elapsed().as_secs_f32() * 1000.0);
        }
        println!(
            "{:?} (built {:?}): setup {setup_ms:.0} ms, solve {best:.0} ms for {:.2} Mtexel at {SAMPLES} samples, mean irradiance {mean:.4}",
            backend,
            tracer.backend(),
            texels as f32 / 1.0e6
        );
    }
}
