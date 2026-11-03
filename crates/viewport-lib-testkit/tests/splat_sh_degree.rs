//! Degree-3 Gaussian splat sets: each splat reads its own 48 coefficients,
//! and the degree 2 and 3 terms follow the view on the full tier.

use viewport_lib::plugin_api::Uploads;
use viewport_lib_plugins::item_types::gaussian_splat::{
    GaussianSplatData, GaussianSplatItem, ShDegree,
};
use viewport_lib_testkit::{BuiltScene, DeviceProfile, Harness, frame_for};

const W: u32 = 256;
const H: u32 = 192;
const SH0: f32 = 0.282_094_79;

/// The camera on the -y side of the origin at `yaw` 0, on the +y side at pi.
fn view(yaw: f32) -> viewport_lib::Camera {
    viewport_lib::Camera::orbit(
        glam::DVec3::ZERO,
        4.0,
        glam::Quat::from_rotation_z(yaw) * glam::Quat::from_rotation_x(1.5708),
    )
}

fn render(harness: &mut Harness, data: &GaussianSplatData, yaw: f32) -> Vec<u8> {
    let (device, queue) = (harness.device.clone(), harness.queue.clone());
    let id = harness
        .renderer
        .upload(&device, &queue, data)
        .expect("upload");
    let mut item = GaussianSplatItem::default();
    item.source = id;
    let built = BuiltScene {
        gaussian_splats: vec![item],
        ..Default::default()
    };
    let frame = frame_for(&built, &view(yaw), [W as f32, H as f32]);
    let _ = harness.render(&frame, W, H);
    harness.render(&frame, W, H)
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [i32; 3] {
    let at = ((y * W + x) * 4) as usize;
    [
        pixels[at] as i32,
        pixels[at + 1] as i32,
        pixels[at + 2] as i32,
    ]
}

/// A degree-3 set with every coefficient zero but those `set` writes.
fn degree3(positions: Vec<[f32; 3]>, set: impl Fn(&mut [f32], usize)) -> GaussianSplatData {
    let n = positions.len();
    let mut data = GaussianSplatData::default();
    data.positions = positions;
    data.scales = vec![[0.25; 3]; n];
    data.rotations = vec![[0.0, 0.0, 0.0, 1.0]; n];
    data.opacities = vec![0.97; n];
    data.sh_degree = ShDegree::Three;
    data.sh_coefficients = vec![0.0; n * 48];
    for (i, block) in data.sh_coefficients.chunks_exact_mut(48).enumerate() {
        set(block, i);
    }
    data
}

fn second_splat_reads_its_own_coefficients(harness: &mut Harness) {
    let dc = |c: [f32; 3]| c.map(|v| (v - 0.5) / SH0);
    let colours = [dc([1.0, 0.1, 0.1]), dc([0.1, 0.1, 1.0])];
    let data = degree3(vec![[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], |block, i| {
        block[..3].copy_from_slice(&colours[i]);
    });
    let pixels = render(harness, &data, 0.0);
    let row: Vec<_> = (0..W).map(|x| pixel(&pixels, x, H / 2)).collect();
    assert!(
        row.iter().any(|c| c[0] - c[2] > 60),
        "the first splat is red"
    );
    assert!(
        row.iter().any(|c| c[2] - c[0] > 60),
        "the second splat is blue, from its own coefficients"
    );
}

#[test]
fn a_degree_three_splat_reads_its_own_coefficients() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    second_splat_reads_its_own_coefficients(&mut harness);
}

#[test]
fn a_degree_three_splat_reads_its_own_coefficients_on_the_compatibility_tier() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("splat-sh-degree-three"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    second_splat_reads_its_own_coefficients(&mut harness);
}

/// Only the Y(3,-3) coefficient is set, a term in y^3, so the colour swings
/// from red to blue as the eye moves from -y to +y. Degree 1 alone would draw
/// grey from both sides.
#[test]
fn degree_three_terms_follow_the_view() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let data = degree3(vec![[0.0; 3]], |block, _| {
        block[27..30].copy_from_slice(&[2.0, 0.0, -2.0]);
    });
    let one_way = pixel(&render(&mut harness, &data, 0.0), W / 2, H / 2);
    let other_way = pixel(
        &render(&mut harness, &data, std::f32::consts::PI),
        W / 2,
        H / 2,
    );
    assert!(
        one_way[0] - one_way[2] > 60,
        "red with the eye on -y: {one_way:?}"
    );
    assert!(
        other_way[2] - other_way[0] > 60,
        "blue with the eye on +y: {other_way:?}"
    );
}
