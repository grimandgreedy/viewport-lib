//! CPU micro-benchmarks: the statistical (criterion) layer of the suite.
//!
//! These measure the per-frame and upload work that runs on the CPU. Criterion
//! stores a baseline and reports `% change` per cell, flagging regressions far
//! below the eyeball floor. Each group is one axis of the matrix:
//!
//! - `primitive_gen` / `frustum_cull`: device-free CPU math, swept by size.
//! - `upload`: `upload_mesh_data` swept by triangle count (needs a device).
//! - `prepare`: `prepare()` cost vs object count, instanced vs per-object
//!   (needs a device). The instanced/per-object pair is benched side by side so
//!   a change that helps one path and hurts the other is visible.
//!
//! The triangle/object sweeps feed `scripts/fit_costs.py`, which fits
//! `t ~= a + b * n` (fixed cost vs per-item cost) and reports the crossover.
//!
//! Device benches skip silently when no GPU adapter is present.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use viewport_lib::{
    Aabb, BackfacePolicy, Camera, CameraFrame, FrameData, Frustum, Material, MeshId,
    PickAccelerator, Scene, SceneFrame, SceneRenderItem, primitives,
};
use viewport_lib_testkit::{Harness, meshes};

/// Triangle counts to sweep upload/generation by. `icosphere(r, sub)` has
/// `20 * 4^sub` triangles, so these are the subdivision levels 1..6.
const ICO_SUBDIVS: &[(u32, u64)] = &[
    (1, 80),
    (2, 320),
    (3, 1280),
    (4, 5120),
    (5, 20480),
    (6, 81920),
];

fn bench_primitive_gen(c: &mut Criterion) {
    let mut g = c.benchmark_group("primitive_gen");
    for &(sub, tris) in ICO_SUBDIVS {
        g.throughput(Throughput::Elements(tris));
        g.bench_with_input(BenchmarkId::new("icosphere", tris), &sub, |b, &sub| {
            b.iter(|| black_box(primitives::icosphere(1.0, sub)));
        });
    }
    // A concave generator from the corpus, for contrast with the convex sphere.
    g.bench_function("torus_knot_240x16", |b| {
        b.iter(|| black_box(meshes::torus_knot(2, 3, 240, 16, 0.32)));
    });
    g.bench_function("heightfield_96x96", |b| {
        b.iter(|| black_box(meshes::heightfield(96, 96, 4.0, 0.8)));
    });
    g.finish();
}

fn bench_frustum_cull(c: &mut Criterion) {
    let mut g = c.benchmark_group("frustum_cull");
    let view = Mat4::look_at_rh(Vec3::new(0.0, -60.0, 40.0), Vec3::ZERO, Vec3::Z);
    let proj = Mat4::perspective_rh(0.9, 1.6, 0.1, 1000.0);
    let frustum = Frustum::from_view_proj(&(proj * view));

    for &count in &[1_000usize, 10_000, 100_000] {
        // A cube grid of unit AABBs.
        let side = (count as f64).cbrt().ceil() as i32;
        let mut aabbs = Vec::with_capacity(count);
        'outer: for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let c = Vec3::new(x as f32, y as f32, z as f32) * 2.0;
                    aabbs.push(Aabb::from_positions(&[
                        (c - Vec3::splat(0.5)).to_array(),
                        (c + Vec3::splat(0.5)).to_array(),
                    ]));
                    if aabbs.len() == count {
                        break 'outer;
                    }
                }
            }
        }
        g.throughput(Throughput::Elements(count as u64));
        g.bench_with_input(BenchmarkId::new("cull_aabb", count), &aabbs, |b, aabbs| {
            b.iter(|| {
                let mut visible = 0u32;
                for a in aabbs {
                    if !frustum.cull_aabb(a) {
                        visible += 1;
                    }
                }
                black_box(visible)
            });
        });
    }
    g.finish();
}

fn bench_upload(c: &mut Criterion) {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping upload bench: no GPU adapter");
        return;
    };
    let mut g = c.benchmark_group("upload");
    for &(sub, tris) in ICO_SUBDIVS {
        let mesh = primitives::icosphere(1.0, sub);
        g.throughput(Throughput::Elements(tris));
        g.bench_with_input(BenchmarkId::new("mesh_data", tris), &mesh, |b, mesh| {
            b.iter(|| {
                let id: MeshId = harness
                    .renderer
                    .resources_mut()
                    .upload_mesh_data(&harness.device, mesh)
                    .expect("upload");
                // Remove immediately so GPU memory does not grow across iterations.
                harness.renderer.resources_mut().free_mesh(id);
            });
        });
    }
    g.finish();
}

/// Build a scene of `count` cubes. When `instanced` they share one mesh and
/// material (one batch); otherwise each carries a styled back-face policy that
/// forces the per-object path.
fn cube_scene(mesh: MeshId, count: usize, instanced: bool) -> Vec<SceneRenderItem> {
    let side = (count as f64).cbrt().ceil() as i32;
    let mut items = Vec::with_capacity(count);
    'outer: for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let mut it = SceneRenderItem::default();
                it.mesh_id = mesh;
                it.model = Mat4::from_translation(Vec3::new(x as f32, y as f32, z as f32) * 1.5)
                    .to_cols_array_2d();
                it.material = Material::pbr([0.7, 0.6, 0.5], 0.2, 0.5);
                if !instanced {
                    // A styled back-face policy defeats batching, forcing the
                    // per-object path (uniform write + bind-group build each).
                    it.material.backface_policy = BackfacePolicy::Tint(0.4);
                }
                items.push(it);
                if items.len() == count {
                    break 'outer;
                }
            }
        }
    }
    items
}

fn bench_prepare(c: &mut Criterion) {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping prepare bench: no GPU adapter");
        return;
    };
    let cube = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &primitives::cube(1.0))
        .expect("cube upload");

    let camera = Camera {
        distance: 200.0,
        ..Camera::default()
    };

    let mut g = c.benchmark_group("prepare");
    for &count in &[1usize, 100, 1_000] {
        for &instanced in &[true, false] {
            let label = if instanced { "instanced" } else { "per_object" };
            let items = cube_scene(cube, count, instanced);
            g.throughput(Throughput::Elements(count as u64));
            g.bench_with_input(BenchmarkId::new(label, count), &items, |b, items| {
                b.iter(|| {
                    let frame = FrameData::new(
                        CameraFrame::from_camera(&camera, [1280.0, 720.0]),
                        SceneFrame::from_surface_items(items.clone()),
                    );
                    let bufs =
                        harness
                            .renderer
                            .pass()
                            .prepare(&harness.device, &harness.queue, &frame);
                    black_box(bufs.len())
                });
            });
        }
    }
    g.finish();
}

fn bench_pick(c: &mut Criterion) {
    // CPU picking: build a BVH over a scene of N cube nodes and ray-cast through
    // it. Device-free (PickAccelerator works on retained CPU data).
    let cube = primitives::cube(1.0);
    let aabb = cube.compute_aabb();
    let mesh_id = MeshId::from_index(0);
    let mut mesh_lookup: HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)> = HashMap::new();
    mesh_lookup.insert(0, (cube.positions.clone(), cube.indices.clone()));

    let build_scene = |count: usize| -> Scene {
        let side = (count as f64).cbrt().ceil() as i32;
        let mut scene = Scene::new();
        let mut n = 0;
        'outer: for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let m = Mat4::from_translation(Vec3::new(x as f32, y as f32, z as f32) * 2.0);
                    scene.add(Some(mesh_id), m, Material::pbr([0.6, 0.6, 0.6], 0.0, 0.5));
                    n += 1;
                    if n == count {
                        break 'outer;
                    }
                }
            }
        }
        scene
    };

    let mut build = c.benchmark_group("pick_build");
    for &count in &[100usize, 1_000, 10_000] {
        let scene = build_scene(count);
        build.throughput(Throughput::Elements(count as u64));
        build.bench_with_input(BenchmarkId::new("bvh", count), &scene, |b, scene| {
            b.iter(|| black_box(PickAccelerator::build_from_scene(scene, |_| Some(aabb))));
        });
    }
    build.finish();

    let mut cast = c.benchmark_group("pick_cast");
    for &count in &[100usize, 1_000, 10_000] {
        let scene = build_scene(count);
        let mut acc = PickAccelerator::build_from_scene(&scene, |_| Some(aabb));
        let origin = Vec3::new(0.0, 0.0, 1000.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        cast.throughput(Throughput::Elements(count as u64));
        cast.bench_with_input(BenchmarkId::new("ray", count), &count, |b, _| {
            b.iter(|| black_box(acc.pick(origin, dir, &mesh_lookup)));
        });
    }
    cast.finish();
}

criterion_group!(
    benches,
    bench_primitive_gen,
    bench_frustum_cull,
    bench_upload,
    bench_prepare,
    bench_pick
);
criterion_main!(benches);
