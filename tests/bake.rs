//! Texel G-buffer functional tests.
//!
//! These run the UV-space rasterisation on a real headless GPU and check the
//! per-texel world position/normal readback end to end. Run with:
//!   cargo test --features bake --test bake

#![cfg(feature = "bake")]

use glam::Mat4;
use viewport_lib::bake::{TexelGeometry, rasterize_texel_gbuffer};

/// A headless device/queue, or `None` when no adapter is available (skip).
fn device_queue() -> Option<(viewport_lib::gpu::Device, viewport_lib::gpu::Queue)> {
    let instance = viewport_lib::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &viewport_lib::gpu::RequestAdapterOptions {
            power_preference: viewport_lib::gpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        },
    ))
    .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&viewport_lib::gpu::DeviceDescriptor::default()))
            .ok()?;
    Some((device, queue))
}

/// A unit quad on the z=0 plane whose UV1 fills the whole `[0,1]` atlas, so every
/// texel is covered. Corners are placed so UV `(0,0)` sits at world `(-1,-1)`.
fn full_quad() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
    let positions = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let normals = vec![[0.0, 0.0, 1.0]; 4];
    let uv1 = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices = vec![0u32, 1, 2, 0, 2, 3];
    (positions, normals, uv1, indices)
}

/// A quad filling UV space covers every texel; the readback world positions land
/// on the quad and the normals point along it.
#[test]
fn full_quad_covers_every_texel() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let (positions, normals, uv1, indices) = full_quad();
    let geom = TexelGeometry {
        positions: &positions,
        normals: &normals,
        uv1: &uv1,
        indices: &indices,
        model: Mat4::IDENTITY,
    };
    let (w, h) = (8u32, 8u32);
    let gb = rasterize_texel_gbuffer(&device, &queue, &geom, w, h);

    assert_eq!(gb.world_pos.len(), (w * h) as usize);
    assert_eq!(
        gb.covered_count(),
        (w * h) as usize,
        "quad should fill the atlas"
    );

    for (i, p) in gb.world_pos.iter().enumerate() {
        assert_eq!(p[3], 1.0, "texel {i} should be covered");
        assert!(p[0] >= -1.01 && p[0] <= 1.01, "x {} off the quad", p[0]);
        assert!(p[1] >= -1.01 && p[1] <= 1.01, "y {} off the quad", p[1]);
        assert!(p[2].abs() < 1e-3, "z {} should be ~0", p[2]);
        let n = gb.world_normal[i];
        assert!(
            (n[2].abs() - 1.0).abs() < 1e-3,
            "normal {n:?} should be +/-Z"
        );
    }
}

/// The UV -> atlas mapping puts UV `(0,0)` at the top-left texel and `v` growing
/// downward, so the top-left texel's world position is near the `(-1,-1)` corner
/// and the bottom-left texel's is near `(-1,+1)`.
#[test]
fn uv_origin_maps_to_top_left() {
    let Some((device, queue)) = device_queue() else {
        return;
    };
    let (positions, normals, uv1, indices) = full_quad();
    let geom = TexelGeometry {
        positions: &positions,
        normals: &normals,
        uv1: &uv1,
        indices: &indices,
        model: Mat4::IDENTITY,
    };
    let (w, h) = (8u32, 8u32);
    let gb = rasterize_texel_gbuffer(&device, &queue, &geom, w, h);

    let at = |x: u32, y: u32| gb.world_pos[(y * w + x) as usize];
    let top_left = at(0, 0);
    let bottom_left = at(0, h - 1);
    // Top row is UV v~0 -> world y ~ -1; bottom row is UV v~1 -> world y ~ +1.
    assert!(
        top_left[1] < -0.5,
        "top-left y {} should be near -1",
        top_left[1]
    );
    assert!(
        bottom_left[1] > 0.5,
        "bottom-left y {} should be near +1",
        bottom_left[1]
    );
    assert!(
        top_left[0] < -0.5,
        "left column x {} should be near -1",
        top_left[0]
    );
}

/// A model transform carries through to the baked world positions and normals.
#[test]
fn model_transform_is_applied() {
    let Some((device, queue)) = device_queue() else {
        return;
    };
    let (positions, normals, uv1, indices) = full_quad();
    let model = Mat4::from_translation(glam::Vec3::new(10.0, 0.0, 0.0));
    let geom = TexelGeometry {
        positions: &positions,
        normals: &normals,
        uv1: &uv1,
        indices: &indices,
        model,
    };
    let gb = rasterize_texel_gbuffer(&device, &queue, &geom, 8, 8);
    for p in gb.world_pos.iter().filter(|p| p[3] != 0.0) {
        // Translated +10 in x: every covered texel sits near x in [9, 11].
        assert!(p[0] >= 8.9 && p[0] <= 11.1, "x {} not translated", p[0]);
    }
}

/// An empty mesh returns an all-empty buffer and touches nothing.
#[test]
fn empty_mesh_is_all_invalid() {
    let Some((device, queue)) = device_queue() else {
        return;
    };
    let geom = TexelGeometry {
        positions: &[],
        normals: &[],
        uv1: &[],
        indices: &[],
        model: Mat4::IDENTITY,
    };
    let gb = rasterize_texel_gbuffer(&device, &queue, &geom, 8, 8);
    assert_eq!(gb.covered_count(), 0);
    assert!(gb.world_pos.iter().all(|p| p[3] == 0.0));
}
