//! Build + query timing for the CPU `PickAccelerator`, next to a brute-force
//! linear scan that uses the same parry narrow phase but no broad-phase pruning.
//!
//! Public-API only, so the same bench compiles against either the hand-rolled BVH
//! or the spatial-query backend and their numbers line up. The linear-scan column
//! is the baseline the renderer's own CPU pick (a linear scan today) would be
//! replaced by, so it is recorded here for an easy comparison when that migration
//! lands.
//!
//! Run: cargo bench --bench pick_bench

use std::collections::HashMap;
use std::time::Instant;

use glam::{Mat4, Vec3};
use parry3d::math::Vector;
use parry3d::query::{Ray as PRay, RayCast};
use parry3d::shape::TriMesh;
use viewport_lib::{Aabb, Material, MeshId, PickAccelerator, Scene};

/// Unit cube geometry (positions + triangle indices).
fn unit_cube() -> (Vec<[f32; 3]>, Vec<u32>) {
    let positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    (positions, indices)
}

/// Tiny deterministic LCG returning f32 in [0, 1).
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// `n` unit cubes scattered through a volume whose side grows with `n` so the
/// spacing stays roughly constant. Returns the scene (for the accelerator) and the
/// translations (for the linear scan, which needs no Scene introspection).
fn scatter(n: usize) -> (Scene, Vec<Vec3>) {
    let mut rng = Rng(0xC0FFEE_1234_5678);
    let span = 2.0 * (n as f32).cbrt();
    let mut scene = Scene::new();
    let mut positions = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Vec3::new(
            rng.range(-span, span),
            rng.range(-span, span),
            rng.range(-span, span),
        );
        positions.push(t);
        scene.add(
            Some(MeshId::from_index(0)),
            Mat4::from_translation(t),
            Material::default(),
        );
    }
    scene.update_transforms();
    (scene, positions)
}

/// Rays sweeping through the volume from a shell around it.
fn query_rays(count: usize, span: f32) -> Vec<(Vec3, Vec3)> {
    let mut rng = Rng(0xBEEF_9999_0001);
    (0..count)
        .map(|_| {
            let origin = Vec3::new(
                rng.range(-1.5 * span, 1.5 * span),
                rng.range(-1.5 * span, 1.5 * span),
                2.0 * span + rng.range(0.0, span),
            );
            let target = Vec3::new(
                rng.range(-span, span),
                rng.range(-span, span),
                rng.range(-span, span),
            );
            (origin, (target - origin).normalize())
        })
        .collect()
}

fn unit_aabb() -> Aabb {
    Aabb {
        min: Vec3::splat(-0.5),
        max: Vec3::splat(0.5),
    }
}

/// Brute-force nearest hit: cast every cube's shared unit `TriMesh` (translation
/// only, so the ray shifts by -position). Same narrow phase as the accelerator,
/// without the BVH prune.
fn linear_pick(tm: &TriMesh, positions: &[Vec3], o: Vec3, d: Vec3) -> Option<f32> {
    let mut best = f32::INFINITY;
    for &p in positions {
        let lo = o - p;
        let ray = PRay::new(Vector::new(lo.x, lo.y, lo.z), Vector::new(d.x, d.y, d.z));
        if let Some(toi) = tm.cast_local_ray(&ray, f32::MAX, true) {
            if toi < best {
                best = toi;
            }
        }
    }
    best.is_finite().then_some(best)
}

fn min_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    const QUERIES: usize = 10_000;
    const BUILD_REPS: usize = 3;
    const QUERY_REPS: usize = 5;

    let (positions, indices) = unit_cube();
    let mut mesh_lookup = HashMap::new();
    mesh_lookup.insert(0u64, (positions.clone(), indices.clone()));

    // Shared TriMesh for the linear scan.
    let verts: Vec<Vector> = positions.iter().map(|p| Vector::new(p[0], p[1], p[2])).collect();
    let tris: Vec<[u32; 3]> = indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let cube_tm = TriMesh::new(verts, tris).unwrap();

    println!(
        "\n{:>9} | {:>10} | {:>10} | {:>12} | {:>13}",
        "objects", "build ms", "refit ms", "bvh q ms/10k", "linear q ms/10k",
    );
    println!("{}", "-".repeat(67));

    for &n in &[1_000usize, 10_000, 100_000] {
        let (scene, obj_positions) = scatter(n);
        let span = 2.0 * (n as f32).cbrt();
        let rays = query_rays(QUERIES, span);

        let build_ms = min_ms(BUILD_REPS, || {
            let _ = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));
        });

        // Warm the accelerator's TriMesh cache so query timing is traversal +
        // narrow phase, not the one-time parry build.
        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));
        for (o, d) in rays.iter().take(16) {
            let _ = accel.pick(*o, *d, &mesh_lookup);
        }

        // Refit the same scene (re-reads every leaf AABB, keeps topology). The
        // per-frame cost for a moving scene that was built once.
        let refit_ms = min_ms(QUERY_REPS, || {
            assert!(accel.refit_from_scene(&scene, |_| Some(unit_aabb())));
        });

        let bvh_ms = min_ms(QUERY_REPS, || {
            for (o, d) in &rays {
                let _ = accel.pick(*o, *d, &mesh_lookup);
            }
        });

        let linear_ms = min_ms(QUERY_REPS, || {
            for (o, d) in &rays {
                let _ = linear_pick(&cube_tm, &obj_positions, *o, *d);
            }
        });

        println!(
            "{:>9} | {:>10.2} | {:>10.2} | {:>12.2} | {:>13.2}",
            n, build_ms, refit_ms, bvh_ms, linear_ms,
        );
    }
    println!();
}
