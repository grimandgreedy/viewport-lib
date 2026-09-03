//! Regression tests: a `replace_texture` update must reach the screen on the
//! instanced draw path, including its GPU-culling indirect variant.
//!
//! Two planes that share one mesh force the instanced path
//! (`is_using_instanced_path()` is true for 2+ visible items). Each test renders
//! once, calls `replace_texture` on one plane's texture with a very different
//! colour, renders again, and checks the framebuffer changed. A byte-identical
//! pair of renders means the update was dropped somewhere in the path.

use super::types::FrameData;
use super::{CameraFrame, RenderCamera, SceneFrame, ViewportRenderer};
use crate::camera::Camera;
use crate::resources::TextureId;
use crate::scene::material::{BackfacePolicy, Material};

fn headless_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
    let instance = crate::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &crate::gpu::RequestAdapterOptions {
            power_preference: crate::gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            #[cfg(wgpu30)]
            apply_limit_buckets: false,
        },
    ))
    .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
            label: Some("instanced_texture_tests"),
            required_limits: crate::renderer::ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }))
        .ok()?;
    Some((device, queue))
}

const W: u32 = 128;
const H: u32 = 128;

// A dark start colour and a light swap colour. Both differ on every channel so
// the weighted checksum below registers the change (a green <-> blue swap, which
// only trades the G and B channels, would slip past a plain byte sum).
const C_START: [u8; 4] = [10, 20, 30, 255];
const C_SWAP: [u8; 4] = [200, 180, 160, 255];

// A camera looking straight down the -Z axis at the XY plane, so the ground
// planes (normal +Z) face the camera and cover a large part of the frame.
fn top_down_camera() -> Camera {
    let mut cam = Camera::default();
    cam.orientation = glam::Quat::IDENTITY; // identity = top view in a Z-up world
    cam.center = glam::Vec3::ZERO;
    cam.distance = 5.0;
    cam.aspect = W as f32 / H as f32;
    cam
}

fn solid_rgba(w: u32, h: u32, colour: [u8; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        v.extend_from_slice(&colour);
    }
    v
}

fn unlit_settings() -> crate::scene::material::ItemSettings {
    let mut s = crate::scene::material::ItemSettings::default();
    s.unlit = true; // output raw albedo, no lighting mixed in
    s
}

// Channel-weighted checksum. A plain byte sum is blind to some colour swaps (it
// can just move a constant between two channels), so weight R/G/B/A differently
// to make any colour change register.
fn checksum(bytes: &[u8]) -> u64 {
    const WEIGHT: [u64; 4] = [2, 3, 5, 7];
    bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| WEIGHT[i % 4] * b as u64)
        .sum()
}

fn frame_for(items: Vec<crate::SceneRenderItem>) -> FrameData {
    let cf = CameraFrame::new(
        RenderCamera::from_camera(&top_down_camera()),
        [W as f32, H as f32],
    );
    FrameData::new(cf, SceneFrame::from_surface_items(items))
}

/// A textured plane at world x, unlit and two-sided (the material a windowed
/// compositor uses for its client planes), sharing `mesh`.
fn textured_plane(
    mesh: crate::resources::mesh::mesh_store::MeshId,
    tex: TextureId,
    x: f32,
) -> crate::SceneRenderItem {
    let mut material = Material::textured(tex);
    material.backface_policy = BackfacePolicy::Identical;
    crate::SceneRenderItem {
        mesh_id: mesh,
        model: glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d(),
        material,
        settings: unlit_settings(),
        ..Default::default()
    }
}

/// Two textured planes side by side sharing one mesh, plus the second plane's
/// texture id (the one a test replaces). Both start at `C_START`.
fn two_textured_planes(
    renderer: &mut ViewportRenderer,
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
) -> (crate::SceneRenderItem, crate::SceneRenderItem, TextureId) {
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &crate::primitives::plane(2.0, 2.0))
        .unwrap();
    let tex_a = renderer
        .resources_mut()
        .upload_texture(device, queue, 2, 2, &solid_rgba(2, 2, C_START))
        .unwrap();
    let tex_b = renderer
        .resources_mut()
        .upload_texture(device, queue, 2, 2, &solid_rgba(2, 2, C_START))
        .unwrap();
    (
        textured_plane(mesh, tex_a, -1.05),
        textured_plane(mesh, tex_b, 1.05),
        tex_b,
    )
}

/// The direct instanced draw path (no GPU culling) reflects a `replace_texture`.
/// Runs on every backend; on Metal this is the only instanced path available.
#[test]
fn instanced_path_reflects_replace_texture() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping instanced_path_reflects_replace_texture: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);
    let (item_a, item_b, tex_b) = two_textured_planes(&mut renderer, &device, &queue);

    let sum1 = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_for(vec![item_a.clone(), item_b.clone()]),
        W,
        H,
    ));
    assert!(
        renderer.is_using_instanced_path(),
        "two planes sharing one mesh must select the instanced path"
    );

    renderer
        .resources_mut()
        .replace_texture(&device, &queue, tex_b, 2, 2, &solid_rgba(2, 2, C_SWAP))
        .unwrap();

    let sum2 = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_for(vec![item_a, item_b]),
        W,
        H,
    ));

    assert_ne!(
        sum1, sum2,
        "replace_texture must update the plane on the instanced path (sum1={sum1} sum2={sum2})"
    );
}

/// A textured plane batched next to an untextured sibling on the same mesh (the
/// shape of a client window drawn alongside an untextured cursor marker) still
/// reflects a `replace_texture` of the textured plane.
#[test]
fn instanced_path_reflects_replace_texture_with_untextured_sibling() {
    let Some((device, queue)) = headless_device() else {
        eprintln!(
            "skipping instanced_path_reflects_replace_texture_with_untextured_sibling: no GPU adapter"
        );
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);

    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &crate::primitives::plane(2.0, 2.0))
        .unwrap();
    let tex = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 2, 2, &solid_rgba(2, 2, C_START))
        .unwrap();

    let textured = textured_plane(mesh, tex, -1.05);
    // The sibling shares the mesh but carries no texture, only a base colour.
    let mut untextured_mat = Material::default();
    untextured_mat.backface_policy = BackfacePolicy::Identical;
    untextured_mat.base_colour = crate::Colour::srgb(0.85, 0.15, 0.5, 1.0);
    let untextured = crate::SceneRenderItem {
        mesh_id: mesh,
        model: glam::Mat4::from_translation(glam::Vec3::new(1.05, 0.0, 0.0)).to_cols_array_2d(),
        material: untextured_mat,
        settings: unlit_settings(),
        ..Default::default()
    };

    let sum1 = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_for(vec![textured.clone(), untextured.clone()]),
        W,
        H,
    ));
    assert!(renderer.is_using_instanced_path());

    renderer
        .resources_mut()
        .replace_texture(&device, &queue, tex, 2, 2, &solid_rgba(2, 2, C_SWAP))
        .unwrap();

    let sum2 = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_for(vec![textured, untextured]),
        W,
        H,
    ));

    assert_ne!(
        sum1, sum2,
        "replace_texture must update the textured plane even beside an untextured \
         sibling (sum1={sum1} sum2={sum2})"
    );
}

/// Regression test for a `replace_texture` update being dropped on the GPU-driven
/// culling indirect draw path.
///
/// That path binds each batch's cull bind group (which samples the albedo view at
/// binding 1) as the draw's group 1. Those bind groups are keyed by texture id and
/// cached per viewport; `replace_texture` swaps the view under a stable id, so the
/// key does not change and the cache used to keep drawing the old view. The tests
/// above cannot catch this: GPU culling needs `INDIRECT_FIRST_INSTANCE`, which
/// Metal lacks, so on macOS they exercise the direct instanced draw (which was
/// already correct). This test enables culling and skips where it is unsupported,
/// so it is the Vulkan/DX12 leg that covers the indirect path.
#[test]
fn gpu_culling_indirect_path_reflects_replace_texture() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping gpu_culling_indirect_path_reflects_replace_texture: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);
    if !renderer.is_gpu_culling_supported() {
        eprintln!(
            "skipping gpu_culling_indirect_path_reflects_replace_texture: \
             device has no INDIRECT_FIRST_INSTANCE (e.g. Metal)"
        );
        return;
    }
    renderer.enable_gpu_driven_culling();

    let (item_a, item_b, tex_b) = two_textured_planes(&mut renderer, &device, &queue);

    // The default frame is the HDR pipeline, so with culling on this exercises the
    // indirect draw path.
    let sum1 = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_for(vec![item_a.clone(), item_b.clone()]),
        W,
        H,
    ));
    assert!(
        renderer.is_using_instanced_path(),
        "two planes sharing one mesh must select the instanced path"
    );

    renderer
        .resources_mut()
        .replace_texture(&device, &queue, tex_b, 2, 2, &solid_rgba(2, 2, C_SWAP))
        .unwrap();

    let sum2 = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_for(vec![item_a, item_b]),
        W,
        H,
    ));

    assert_ne!(
        sum1, sum2,
        "replace_texture must update the texture on the GPU-culling indirect path \
         (sum1={sum1} sum2={sum2})"
    );
}
