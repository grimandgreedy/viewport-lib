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

// ---------------------------------------------------------------------------
// Alpha-cutout shadow regression.
// ---------------------------------------------------------------------------

// Framebuffer size for the cutout-shadow scene. Wider than tall so the ground
// receiver fills the frame at the camera angle below.
const EW: u32 = 256;
const EH: u32 = 144;

// The receiver region that catches the caster's shadow but never the caster
// itself. Found by rendering the scene three ways (caster + shadow, caster
// without shadow, ground only) and classifying each cell: the caster paints the
// upper band (x >= 128, y < 48) while its shadow lands here, well clear of it.
// Sampling only this rectangle isolates the shadow from the caster's own
// surface, so a change here can only come from the shadow silhouette.
const SHADOW_X0: usize = 64;
const SHADOW_X1: usize = 120;
const SHADOW_Y0: usize = 48;
const SHADOW_Y1: usize = 80;

// Channel-weighted checksum over a sub-rectangle of an RGBA framebuffer.
fn region_checksum(bytes: &[u8], width: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> u64 {
    const WEIGHT: [u64; 4] = [2, 3, 5, 7];
    let mut sum = 0u64;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * width + x) * 4;
            for c in 0..4 {
                sum += WEIGHT[c] * bytes[i + c] as u64;
            }
        }
    }
    sum
}

/// Two alpha-mask shadow casters share one mesh (so the instanced draw path is
/// selected) and drop a shadow onto a large ground receiver. The cutout shadow
/// silhouette is produced by sampling the caster's albedo alpha in the shadow
/// depth pass. Replacing the caster's albedo (an opaque mask -> a fully
/// transparent one) under a stable `TextureId` must change the shadow: the solid
/// shadow the opaque mask casts disappears once the mask is transparent.
///
/// The shadow cull bind groups carry the albedo view sampled during the shadow
/// pass. They were only rebuilt on an instance-buffer rebuild, so a
/// `replace_texture` (which swaps the view under an unchanged id) used to leave
/// the shadow frozen at the first frame's cutout while the lit surface updated.
///
/// Only the receiver region that the shadow falls on is checksummed, and the
/// caster never paints there, so the measured change comes from the shadow
/// silhouette rather than the caster's own surface. The CPU-cull shadow path
/// (devices without `INDIRECT_FIRST_INSTANCE`, e.g. Metal) is exercised here
/// too, so this is not gated on GPU culling.
#[test]
fn instanced_cutout_shadow_reflects_replace_texture() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping instanced_cutout_shadow_reflects_replace_texture: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb);

    let ground = renderer
        .resources_mut()
        .upload_mesh_data(&device, &crate::primitives::cuboid(24.0, 24.0, 0.5))
        .unwrap();
    let caster_mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &crate::primitives::plane(4.0, 4.0))
        .unwrap();
    // Start opaque: alpha 255 everywhere, so the cutout keeps every texel and the
    // caster throws a solid shadow.
    let tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            2,
            2,
            &solid_rgba(2, 2, [255, 255, 255, 255]),
        )
        .unwrap();

    use crate::scene::material::AlphaMode;
    // `with_casters` toggles the casters for the shadow-present sanity check.
    let build = |with_casters: bool| -> FrameData {
        let mut items = Vec::new();
        let mut g = crate::SceneRenderItem::default();
        g.mesh_id = ground;
        g.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
        g.material = Material::from_colour([0.85, 0.85, 0.85]);
        items.push(g);
        if with_casters {
            // Two casters sharing one mesh and one texture -> a single instanced
            // batch of 2, which forces the instanced path.
            for x in [-1.0f32, 1.0] {
                let mut c = crate::SceneRenderItem::default();
                c.mesh_id = caster_mesh;
                c.model =
                    glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 5.0)).to_cols_array_2d();
                let mut m = Material::textured(tex);
                m.backface_policy = BackfacePolicy::Identical;
                m.alpha_mode = AlphaMode::Mask(0.5);
                c.material = m;
                items.push(c);
            }
        }
        let mut cam = Camera {
            distance: 16.0,
            ..Camera::default()
        };
        cam.center = glam::Vec3::new(-4.0, 0.0, 0.0);
        cam.orientation = glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.0);
        cam.set_aspect_ratio(EW as f32, EH as f32);
        let cf = CameraFrame::from_camera(&cam, [EW as f32, EH as f32]);
        let mut fd = FrameData::new(cf, SceneFrame::from_surface_items(items));
        // One low, offset sun so the shadow lands beside the casters (not under
        // them), plus a faint hemisphere fill so the shadow reads as clearly
        // darker than the lit ground.
        let mut l = crate::LightingSettings::default();
        l.lights = vec![{
            let mut s = crate::LightSource::default();
            s.kind = crate::LightKind::Directional {
                direction: [1.6, 0.0, 1.0],
            };
            s.intensity = 1.0;
            s
        }];
        l.shadows.enabled = true;
        l.hemisphere_intensity = 0.05;
        fd.effects.lighting = l;
        fd
    };

    let region = |bytes: &[u8]| {
        region_checksum(
            bytes,
            EW as usize,
            SHADOW_X0,
            SHADOW_X1,
            SHADOW_Y0,
            SHADOW_Y1,
        )
    };

    // Frame 1: opaque caster -> a solid shadow in the sampled region.
    let frame1 = renderer.render_offscreen(&device, &queue, &build(true), EW, EH);
    assert!(
        renderer.is_using_instanced_path(),
        "two casters sharing one mesh must select the instanced path"
    );
    let sum1 = region(&frame1);

    // Sanity: the region actually holds a shadow. Compare against the same scene
    // with the casters removed (ground only, shadows still on). If the region did
    // not darken, there is no shadow to guard and the test would be vacuous.
    let ground_only = renderer.render_offscreen(&device, &queue, &build(false), EW, EH);
    let sum_lit = region(&ground_only);
    assert!(
        sum1 + 5_000 < sum_lit,
        "frame 1 must show a shadow in the sampled region (shadowed={sum1} lit={sum_lit}); \
         without a visible shadow the test cannot guard the cutout update"
    );

    // Swap the albedo under the same id for a fully transparent mask (alpha 0
    // everywhere). The cutout now discards every texel, so the caster throws no
    // shadow. The item set is unchanged (same meshes, same TextureId).
    renderer
        .resources_mut()
        .replace_texture(
            &device,
            &queue,
            tex,
            2,
            2,
            &solid_rgba(2, 2, [255, 255, 255, 0]),
        )
        .unwrap();

    // Frame 2: transparent caster -> the shadow is gone.
    let frame2 = renderer.render_offscreen(&device, &queue, &build(true), EW, EH);
    let sum2 = region(&frame2);

    assert_ne!(
        sum1, sum2,
        "replace_texture on an alpha-cutout shadow caster must update the shadow \
         (shadowed={sum1} after-swap={sum2}); a frozen shadow means the cull bind \
         groups were not invalidated on the texture change"
    );
    // The transparent mask casts no shadow, so the region must brighten back
    // toward the fully-lit ground.
    assert!(
        sum2 > sum1 + 5_000,
        "the shadow must disappear when the mask becomes transparent \
         (shadowed={sum1} after-swap={sum2})"
    );
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
