//! Regression tests: a `replace_texture` update must reach the screen on the
//! instanced draw path, including its GPU-culling indirect variant.
//!
//! Two planes that share one mesh force the instanced path
//! (`is_using_instanced_path()` is true for 2+ visible items). Each test renders
//! once, calls `replace_texture` on one plane's texture with a very different
//! colour, renders again, and checks the framebuffer changed. A byte-identical
//! pair of renders means the update was dropped somewhere in the path.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::renderer::{CameraFrame, SceneFrame};
use viewport_lib::resources::{DeformStage, TextureData, TextureId};

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

fn unlit_settings() -> viewport_lib::ItemSettings {
    let mut s = viewport_lib::ItemSettings::default();
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

fn frame_for(items: Vec<viewport_lib::SceneRenderItem>) -> FrameData {
    let cf = CameraFrame::new(
        RenderCamera::from_camera(&top_down_camera()),
        [W as f32, H as f32],
    );
    FrameData::new(cf, SceneFrame::from_surface_items(items))
}

/// `frame_for` with a distinct scene generation.
///
/// Material identity is not part of the instanced batch cache key, so two frames
/// differing only by material reuse the cached batch and render identically. A
/// test comparing a textured render against an untextured one has to move the
/// generation or it measures nothing. Frames that differ by an upload, a free or
/// a replace do not need this: those move an epoch, which rebuilds on its own.
fn frame_gen(items: Vec<viewport_lib::SceneRenderItem>, generation: u64) -> FrameData {
    let mut fd = frame_for(items);
    fd.scene.generation = generation;
    fd
}

/// A textured plane at world x, unlit and two-sided (the material a windowed
/// compositor uses for its client planes), sharing `mesh`.
fn textured_plane(mesh: MeshId, tex: TextureId, x: f32) -> viewport_lib::SceneRenderItem {
    let mut material = Material::textured(tex);
    material.backface_policy = BackfacePolicy::Identical;
    let mut item = viewport_lib::SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
    item.material = material;
    item.settings = unlit_settings();
    item
}

/// Two textured planes side by side sharing one mesh, plus the second plane's
/// texture id (the one a test replaces). Both start at `C_START`.
fn two_textured_planes(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (
    viewport_lib::SceneRenderItem,
    viewport_lib::SceneRenderItem,
    TextureId,
) {
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::plane(2.0, 2.0))
        .unwrap();
    let tex_a = renderer
        .resources_mut()
        .upload_texture(
            device,
            queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_START)),
        )
        .unwrap();
    let tex_b = renderer
        .resources_mut()
        .upload_texture(
            device,
            queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_START)),
        )
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
/// (devices without `INDIRECT_FIRST_INSTANCE`, e.g. WebGPU or older hardware) is
/// exercised here too, so this is not gated on GPU culling.
#[test]
fn instanced_cutout_shadow_reflects_replace_texture() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping instanced_cutout_shadow_reflects_replace_texture: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let ground = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::cuboid(24.0, 24.0, 0.5))
        .unwrap();
    let caster_mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(4.0, 4.0))
        .unwrap();
    // Start opaque: alpha 255 everywhere, so the cutout keeps every texel and the
    // caster throws a solid shadow.
    let tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, [255, 255, 255, 255])),
        )
        .unwrap();

    use viewport_lib::AlphaMode;
    // `with_casters` toggles the casters for the shadow-present sanity check.
    let build = |with_casters: bool| -> FrameData {
        let mut items = Vec::new();
        let mut g = viewport_lib::SceneRenderItem::default();
        g.mesh_id = ground;
        g.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
        g.material = Material::from_colour([0.85, 0.85, 0.85]);
        items.push(g);
        if with_casters {
            // Two casters sharing one mesh and one texture -> a single instanced
            // batch of 2, which forces the instanced path.
            for x in [-1.0f32, 1.0] {
                let mut c = viewport_lib::SceneRenderItem::default();
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
        let mut l = viewport_lib::LightingSettings::default();
        l.lights = vec![{
            let mut s = viewport_lib::LightSource::default();
            s.kind = viewport_lib::LightKind::Directional {
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
            TextureData::srgb(2, 2, solid_rgba(2, 2, [255, 255, 255, 0])),
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
/// Runs on every backend; where GPU culling is unavailable (WebGPU, older
/// hardware) it is the only instanced path.
#[test]
fn instanced_path_reflects_replace_texture() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping instanced_path_reflects_replace_texture: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
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
        .replace_texture(
            &device,
            &queue,
            tex_b,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
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
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(2.0, 2.0))
        .unwrap();
    let tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_START)),
        )
        .unwrap();

    let textured = textured_plane(mesh, tex, -1.05);
    // The sibling shares the mesh but carries no texture, only a base colour.
    let mut untextured_mat = Material::default();
    untextured_mat.backface_policy = BackfacePolicy::Identical;
    untextured_mat.base_colour = viewport_lib::Colour::srgb(0.85, 0.15, 0.5, 1.0);
    let mut untextured = viewport_lib::SceneRenderItem::default();
    untextured.mesh_id = mesh;
    untextured.model =
        glam::Mat4::from_translation(glam::Vec3::new(1.05, 0.0, 0.0)).to_cols_array_2d();
    untextured.material = untextured_mat;
    untextured.settings = unlit_settings();

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
        .replace_texture(
            &device,
            &queue,
            tex,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
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
/// above exercise the direct instanced draw (which was already correct); this one
/// enables GPU culling to cover the indirect path, and skips where
/// `INDIRECT_FIRST_INSTANCE` is unsupported (WebGPU, older hardware). It runs on
/// Vulkan, DX12, and modern Apple Silicon Metal, which all report the feature.
#[test]
fn gpu_culling_indirect_path_reflects_replace_texture() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping gpu_culling_indirect_path_reflects_replace_texture: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    if !renderer.is_gpu_culling_supported() {
        eprintln!(
            "skipping gpu_culling_indirect_path_reflects_replace_texture: \
             device has no INDIRECT_FIRST_INSTANCE (e.g. WebGPU or older hardware)"
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
        .replace_texture(
            &device,
            &queue,
            tex_b,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
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

// ---------------------------------------------------------------------------
// Bindless material-texture path (Apple Silicon Metal / Vulkan / DX12).
// ---------------------------------------------------------------------------

/// A headless device that requests the bindless texture-array feature set, plus
/// the recommended limits. Returns `None` when no adapter is available or the
/// adapter does not offer the feature set (Metal pre-Tier-2, WebGPU, older HW),
/// so the test skips instead of failing there.
fn headless_bindless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = viewport_lib::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        #[cfg(wgpu30)]
        apply_limit_buckets: false,
    }))
    .ok()?;
    if !adapter
        .features()
        .contains(viewport_lib::gpu::BINDLESS_TEXTURE_FEATURES)
    {
        return None;
    }
    // Request the full recommended feature set (as a real consumer does), so the
    // GPU-culled bindless path is exercised where the adapter also supports
    // indirect draws, not just the direct path.
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("bindless_tests"),
        required_features: ViewportRenderer::recommended_device_features(&adapter),
        required_limits: ViewportRenderer::recommended_device_limits(&adapter),
        ..Default::default()
    }))
    .ok()
}

// Four distinct solid colours, each far apart on every channel so the checksum
// registers every one.
const BINDLESS_COLOURS: [[u8; 4]; 4] = [
    [220, 40, 40, 255],
    [40, 200, 60, 255],
    [50, 70, 230, 255],
    [230, 210, 40, 255],
];

/// Build a scene of four unlit textured planes sharing one mesh, each with its
/// own solid-colour texture, laid out in a row and visible top-down. Returns the
/// frame plus the item list. `renderer` uploads the mesh and textures.
fn bindless_scene(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> FrameData {
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::plane(0.8, 0.8))
        .unwrap();
    let xs = [-1.5f32, -0.5, 0.5, 1.5];
    let items = BINDLESS_COLOURS
        .iter()
        .zip(xs)
        .map(|(colour, x)| {
            let tex = renderer
                .resources_mut()
                .upload_texture(
                    device,
                    queue,
                    TextureData::srgb(2, 2, solid_rgba(2, 2, *colour)),
                )
                .unwrap();
            textured_plane(mesh, tex, x)
        })
        .collect::<Vec<_>>();
    frame_for(items)
}

/// The bindless colour path renders pixel-identically to the per-batch path, and
/// collapses the four distinct-texture instances of one mesh into a single batch.
///
/// Both renderers draw the same scene; the only difference is that the bindless
/// device enabled the texture-array feature set, so it binds one array and drops
/// the texture ids from the batch key. Unlit albedo output makes the comparison
/// exact. The bindless device requests the full recommended feature set, so where
/// the adapter also reports `INDIRECT_FIRST_INSTANCE` (Vulkan, DX12, modern Apple
/// Silicon Metal) this exercises the GPU-culled bindless path, not just the direct one.
#[test]
fn bindless_matches_per_batch_and_collapses_batches() {
    let Some((bd, bq)) = headless_bindless_device() else {
        eprintln!("skipping: no adapter with the bindless texture feature set");
        return;
    };
    let Some((pd, pq)) = headless_device() else {
        eprintln!("skipping: no adapter available");
        return;
    };

    // Per-batch reference (default features, no bindless).
    let mut per_batch = ViewportRenderer::new(&pd, wgpu::TextureFormat::Rgba8UnormSrgb);
    let per_frame = bindless_scene(&mut per_batch, &pd, &pq);
    let per_img = per_batch.render_offscreen(&pd, &pq, &per_frame, W, H);
    let per_batches = per_batch.last_frame_stats().instanced_batches;

    // Bindless device.
    let mut bindless = ViewportRenderer::new(&bd, wgpu::TextureFormat::Rgba8UnormSrgb);
    let bindless_frame = bindless_scene(&mut bindless, &bd, &bq);
    let bindless_img = bindless.render_offscreen(&bd, &bq, &bindless_frame, W, H);
    let bindless_batches = bindless.last_frame_stats().instanced_batches;

    // Both paths drew through the instanced path.
    assert!(
        per_batches >= 4,
        "per-batch path should keep one batch per distinct texture (got {per_batches})",
    );
    // Bindless drops the texture ids from the key: four distinct-texture planes on
    // one mesh collapse to a single batch.
    assert_eq!(
        bindless_batches, 1,
        "bindless should collapse the four instances into one batch (got {bindless_batches})",
    );
    assert!(
        bindless_batches < per_batches,
        "bindless batch count ({bindless_batches}) must be below per-batch ({per_batches})",
    );

    // Pixel parity: the two paths render the same image.
    assert_eq!(
        checksum(&bindless_img),
        checksum(&per_img),
        "bindless and per-batch must render the same image",
    );
}

/// Registering a deformer rebuilds the instanced pipelines through a second build
/// path (`rebuild_mesh_pipelines`), which must pick the same bindless group-1
/// layout that `ensure_*` did. If it falls back to the per-batch layout, the
/// bindless bind group set at draw is incompatible with the rebuilt pipeline and
/// the GPU-culled draw fails validation. This renders once (building the
/// pipelines), registers a deformer (forcing the rebuild), and renders again on a
/// bindless device with GPU culling, which used to panic.
#[test]
fn bindless_survives_deformer_pipeline_rebuild() {
    let Some((device, queue)) = headless_bindless_device() else {
        eprintln!("skipping: no adapter with the bindless texture feature set");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let frame = bindless_scene(&mut renderer, &device, &queue);

    // First render builds the instanced (and, with GPU culling, cull) pipelines.
    let _ = renderer.render_offscreen(&device, &queue, &frame, W, H);

    // Registering a deformer marks the mesh pipelines dirty; the next prepare
    // rebuilds them through the second build path.
    let body =
        "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    return v;\n}\n";
    renderer
        .resources_mut()
        .register_deformer(
            &device,
            viewport_lib::resources::DeformerDesc {
                name: "bindless_noop",
                stage: DeformStage::ObjectSpace,
                priority: 0,
                wgsl_body: body.to_string(),
                per_vertex_stride: 4,
            },
        )
        .expect("register deformer");
    // Rendering again drives the rebuilt bindless pipelines; a layout mismatch
    // here is a validation panic.
    let img = renderer.render_offscreen(&device, &queue, &frame, W, H);
    let batches = renderer.last_frame_stats().instanced_batches;
    assert_eq!(
        batches, 1,
        "bindless still collapses to one batch after the deformer rebuild (got {batches})",
    );
    assert!(
        checksum(&img) > 0,
        "the rebuilt bindless pipelines must still render the scene",
    );
}

/// The explicit `MeshInstanceItem` draw path binds its own per-batch group 1 and
/// pins material_id 0, so it must keep using the per-batch instanced pipelines
/// even under bindless (its `hdr_transparent` / `additive` / `premultiplied`
/// pipelines). If those went bindless, its per-batch bind group would meet a
/// bindless pipeline (a validation error), and its texture would index the array
/// out of bounds. This submits a textured mesh-instance batch and renders it in
/// HDR on a bindless device, which used to panic.
#[test]
fn bindless_keeps_mesh_instance_path_per_batch() {
    use viewport_lib::renderer::{MeshInstanceItem, SpriteBlend};
    let Some((device, queue)) = headless_bindless_device() else {
        eprintln!("skipping: no adapter with the bindless texture feature set");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(0.8, 0.8))
        .unwrap();
    let tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, [220, 40, 40, 255])),
        )
        .unwrap();

    let mut item = MeshInstanceItem::default();
    item.mesh_id = mesh;
    item.texture_id = Some(tex); // has_texture = 1: the crash needs a bound texture
    item.blend = SpriteBlend::AlphaBlend; // -> hdr_transparent pipeline
    item.transforms = [-0.5f32, 0.5]
        .iter()
        .map(|x| glam::Mat4::from_translation(glam::Vec3::new(*x, 0.0, 0.0)).to_cols_array_2d())
        .collect();
    item.colours = vec![viewport_lib::Colour::linear_rgb(1.0, 1.0, 1.0); 2];

    let mut frame = frame_for(vec![]);
    frame.scene.mesh_instances = vec![item];
    frame.effects.display.mode = viewport_lib::PipelineMode::Hdr;

    // A layout mismatch here is a validation panic.
    let img = renderer.render_offscreen(&device, &queue, &frame, W, H);
    assert!(
        checksum(&img) > 0,
        "the mesh-instance batch must render under bindless",
    );
}

/// A material plugin whose final colour is exactly `surf.base_colour` (albedo x
/// tint), so its output is deterministic and driven entirely by the sampled
/// albedo. Under bindless the albedo is fetched from the texture array by the
/// material's per-slot index, so this doubles as a check that the bindlessified
/// plugin shader indexes the array correctly.
#[cfg(test)]
struct AlbedoPlugin;
#[cfg(test)]
impl viewport_lib::MaterialPlugin for AlbedoPlugin {
    fn name(&self) -> &'static str {
        "bindless_albedo_test"
    }
    fn wgsl_body(&self) -> String {
        "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return surf.base_colour;
}
"
        .to_string()
    }
}

/// The four distinct-albedo planes of `bindless_scene`, each drawing through the
/// `AlbedoPlugin` material plugin. Returns the frame plus the plugin id.
#[cfg(test)]
fn bindless_plugin_scene(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> FrameData {
    let plugin = renderer
        .resources_mut()
        .register_material_plugin(device, &AlbedoPlugin)
        .expect("register plugin");
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::plane(0.8, 0.8))
        .unwrap();
    let xs = [-1.5f32, -0.5, 0.5, 1.5];
    let items = BINDLESS_COLOURS
        .iter()
        .zip(xs)
        .map(|(colour, x)| {
            let tex = renderer
                .resources_mut()
                .upload_texture(
                    device,
                    queue,
                    TextureData::srgb(2, 2, solid_rgba(2, 2, *colour)),
                )
                .unwrap();
            let mut it = textured_plane(mesh, tex, x);
            it.material.shading_plugin = Some(plugin);
            it
        })
        .collect::<Vec<_>>();
    frame_for(items)
}

/// A material plugin instances under bindless (its group-1 shape rewritten to the
/// texture array) and renders pixel-identically to the per-batch plugin path.
///
/// Both renderers draw the same four distinct-albedo plugin planes. The per-batch
/// device keeps one batch per texture; the bindless device drops the texture ids
/// and collapses them to a single batch, indexing the albedo array by material.
/// The plugin's output is the sampled albedo, so a wrong index would change the
/// image. Neither path may drop a plugin item to the per-object path.
#[test]
fn bindless_plugin_instances_and_matches_per_batch() {
    let Some((bd, bq)) = headless_bindless_device() else {
        eprintln!("skipping: no adapter with the bindless texture feature set");
        return;
    };
    let Some((pd, pq)) = headless_device() else {
        eprintln!("skipping: no adapter available");
        return;
    };

    let mut per_batch = ViewportRenderer::new(&pd, wgpu::TextureFormat::Rgba8UnormSrgb);
    let per_frame = bindless_plugin_scene(&mut per_batch, &pd, &pq);
    let per_img = per_batch.render_offscreen(&pd, &pq, &per_frame, W, H);
    let per_stats = per_batch.last_frame_stats();

    let mut bindless = ViewportRenderer::new(&bd, wgpu::TextureFormat::Rgba8UnormSrgb);
    let bindless_frame = bindless_plugin_scene(&mut bindless, &bd, &bq);
    let bindless_img = bindless.render_offscreen(&bd, &bq, &bindless_frame, W, H);
    let bindless_stats = bindless.last_frame_stats();

    // Both paths instanced every plugin item.
    assert_eq!(
        per_stats.per_object_items, 0,
        "per-batch plugin items should instance, not fall per-object",
    );
    assert_eq!(
        bindless_stats.per_object_items, 0,
        "bindless plugin items should instance, not fall per-object",
    );

    // Bindless drops the texture ids: four distinct-albedo plugin planes on one
    // mesh collapse to a single batch, where per-batch keeps four.
    assert!(
        per_stats.instanced_batches >= 4,
        "per-batch keeps one plugin batch per texture (got {})",
        per_stats.instanced_batches,
    );
    assert_eq!(
        bindless_stats.instanced_batches, 1,
        "bindless collapses the plugin instances into one batch (got {})",
        bindless_stats.instanced_batches,
    );

    // Pixel parity: the bindlessified plugin shader indexed the albedo array
    // correctly and matches the per-batch plugin shading exactly.
    assert_eq!(
        checksum(&bindless_img),
        checksum(&per_img),
        "bindless and per-batch plugin shading must render the same image",
    );
}

// ---------------------------------------------------------------------------
// Free-epoch validation: a free that touches nothing the batches name must not
// rebuild them, and a free that does touch them must.
// ---------------------------------------------------------------------------

/// Whether the last frame rebuilt the instanced batch list.
///
/// A rebuild reports every batch as either re-uploaded or skipped; a cache hit
/// reports neither, because no upload is attempted at all. Reading only
/// `batches_reuploaded` would miss a rebuild on a static scene, where every
/// batch compares equal and the whole rebuild lands in `batches_skipped`.
fn batches_rebuilt(renderer: &ViewportRenderer) -> bool {
    let s = renderer.last_frame_stats();
    s.batches_reuploaded + s.batches_skipped > 0
}

#[test]
fn freeing_an_unreferenced_texture_keeps_the_batch_cache() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping freeing_an_unreferenced_texture_keeps_the_batch_cache: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let (item_a, item_b, _tex_b) = two_textured_planes(&mut renderer, &device, &queue);
    let items = vec![item_a, item_b];

    // A spare no render item names: the shape of a streaming eviction freeing a
    // resource belonging to some other part of the world.
    let spare = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
        .unwrap();

    renderer.render_offscreen(&device, &queue, &frame_for(items.clone()), W, H);
    assert!(renderer.is_using_instanced_path());
    // Second frame settles the cache: the first is always a build.
    renderer.render_offscreen(&device, &queue, &frame_for(items.clone()), W, H);
    assert!(
        !batches_rebuilt(&renderer),
        "an unchanged scene must hit the batch cache before the free is tested"
    );

    assert!(renderer.resources_mut().free_texture(spare));
    renderer.render_offscreen(&device, &queue, &frame_for(items), W, H);

    assert!(
        !batches_rebuilt(&renderer),
        "freeing a texture no batch references must not rebuild the batch list"
    );
}

#[test]
fn freeing_a_referenced_texture_rebuilds_the_batch_cache() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping freeing_a_referenced_texture_rebuilds_the_batch_cache: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let (item_a, item_b, tex_b) = two_textured_planes(&mut renderer, &device, &queue);
    let items = vec![item_a, item_b];

    renderer.render_offscreen(&device, &queue, &frame_for(items.clone()), W, H);
    renderer.render_offscreen(&device, &queue, &frame_for(items.clone()), W, H);
    assert!(!batches_rebuilt(&renderer));

    // This one a batch does name, so the validation must fail and force a rebuild.
    assert!(renderer.resources_mut().free_texture(tex_b));
    renderer.render_offscreen(&device, &queue, &frame_for(items), W, H);

    assert!(
        batches_rebuilt(&renderer),
        "freeing a texture a batch references must rebuild the batch list"
    );
}

#[test]
fn replacing_a_texture_rebuilds_the_batch_cache() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping replacing_a_texture_rebuilds_the_batch_cache: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let (item_a, item_b, tex_b) = two_textured_planes(&mut renderer, &device, &queue);
    let items = vec![item_a, item_b];

    renderer.render_offscreen(&device, &queue, &frame_for(items.clone()), W, H);
    renderer.render_offscreen(&device, &queue, &frame_for(items.clone()), W, H);
    assert!(!batches_rebuilt(&renderer));

    // A replace keeps the id live, so no liveness check can see it. The view
    // epoch is what must force the rebuild here; the sibling tests in this file
    // check the pixels actually change.
    renderer
        .resources_mut()
        .replace_texture(
            &device,
            &queue,
            tex_b,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
        .unwrap();
    renderer.render_offscreen(&device, &queue, &frame_for(items), W, H);

    assert!(
        batches_rebuilt(&renderer),
        "a replaced texture keeps its id, so only the view epoch can force the rebuild"
    );
}

// ---------------------------------------------------------------------------
// Freeing, rather than replacing.
//
// The caches above validate themselves against a free by checking whether the
// resources they name still resolve, and fall back to a wholesale rebuild only
// when a view moved under a live id. The tests below cover the corners of that
// split which the replace-path tests above do not reach. Every one of them fails
// as a cache wrongly *kept*: the render does not change when it should, which
// also means the freed resource is still pinned.
// ---------------------------------------------------------------------------

/// Freeing a *mesh* reaches the caches, not just freeing a texture.
///
/// `cached_batches_resolve` and `MaterialBindGroup::resources_resolve` both check
/// the mesh id as well as the texture ids, and nothing exercised that.
#[test]
fn freeing_a_mesh_drops_its_items_from_the_cached_batches() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping freeing_a_mesh_drops_its_items_from_the_cached_batches: no adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let keep = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(1.6, 1.6))
        .unwrap();
    let doomed = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(1.6, 1.6))
        .unwrap();
    let tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
        .unwrap();

    // Two items per mesh so each mesh forms its own instanced batch.
    let items = |with_doomed: bool| {
        let mut v = vec![
            textured_plane(keep, tex, -1.8),
            textured_plane(keep, tex, -0.6),
        ];
        if with_doomed {
            v.push(textured_plane(doomed, tex, 0.6));
            v.push(textured_plane(doomed, tex, 1.8));
        }
        v
    };

    let both = checksum(&renderer.render_offscreen(&device, &queue, &frame_for(items(true)), W, H));
    assert!(renderer.is_using_instanced_path());
    // The oracle: what the frame looks like when those items were never submitted.
    let without =
        checksum(&renderer.render_offscreen(&device, &queue, &frame_for(items(false)), W, H));
    assert_ne!(
        both, without,
        "the second mesh must be visible, or this test cannot see its own subject"
    );

    assert!(renderer.resources_mut().free_mesh(doomed));
    // Same item list as the first frame, including the items whose mesh is gone.
    let after =
        checksum(&renderer.render_offscreen(&device, &queue, &frame_for(items(true)), W, H));
    assert_eq!(
        after, without,
        "items whose mesh was freed must stop drawing (after-free={after} \
         never-submitted={without} before-free={both}); an unchanged frame means \
         the batch list was kept and is still holding the freed mesh's buffers"
    );
}

/// A texture reached only through `submesh_materials` is part of what the caches
/// depend on. The item's own material names no texture at all, so a cache that
/// only looked at `item.material` would not know the submesh texture was gone.
#[test]
fn freeing_a_texture_used_only_by_a_submesh_material_reaches_the_screen() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping freeing_a_texture_used_only_by_a_submesh_material: no adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // The mesh needs a real submesh range: `active_submesh_materials` falls back
    // to the item material when the mesh has none, which would make this test
    // observe nothing. One range covering every index is the simplest that counts.
    let mut data = viewport_lib::primitives::plane(2.0, 2.0);
    data.submeshes = vec![viewport_lib::resources::SubmeshRange {
        first_index: 0,
        index_count: data.indices.len() as u32,
    }];
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &data)
        .unwrap();
    let tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
        .unwrap();

    // The item's own material is untextured; only the submesh material names the
    // texture, so a cache that looked at `item.material` alone would not know the
    // submesh texture had gone.
    let item = |with_submesh_texture: bool| {
        let mut it = viewport_lib::SceneRenderItem::default();
        it.mesh_id = mesh;
        it.material = Material::from_colour([0.1, 0.1, 0.1]);
        it.settings = unlit_settings();
        let mut sub = Material::from_colour([0.1, 0.1, 0.1]);
        if with_submesh_texture {
            sub.texture_id = Some(tex);
        }
        it.submesh_materials = Some(vec![sub]);
        it
    };

    let textured = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_gen(vec![item(true), item(true)], 1),
        W,
        H,
    ));
    let untextured = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_gen(vec![item(false), item(false)], 2),
        W,
        H,
    ));
    assert_ne!(
        textured, untextured,
        "the submesh texture must reach the draw, or this test observes nothing"
    );

    assert!(renderer.resources_mut().free_texture(tex));
    let after = checksum(&renderer.render_offscreen(
        &device,
        &queue,
        &frame_gen(vec![item(true), item(true)], 3),
        W,
        H,
    ));
    assert_eq!(
        after, untextured,
        "freeing a texture named only by a submesh material must reach the screen \
         (after-free={after} untextured={untextured} textured={textured}); an \
         unchanged frame means the cache never knew it depended on that texture"
    );
}

/// `update_texture_view` re-points an externally-owned view under a live id.
/// No liveness check can see that, so it bumps the view epoch and forces the
/// wholesale rebuild. (The branch notes called this `replace_external_texture`,
/// which is not the name of the method.)
#[test]
fn update_texture_view_reaches_the_screen() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping update_texture_view_reaches_the_screen: no adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Two caller-owned textures, the kind a compositor hands over per frame.
    let external = |colour: [u8; 4]| {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("external_test_texture"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &t,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &solid_rgba(2, 2, colour),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(2),
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        let view = t.create_view(&wgpu::TextureViewDescriptor::default());
        (t, view)
    };
    let (_keep_a, view_a) = external(C_START);
    let (_keep_b, view_b) = external(C_SWAP);

    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(2.0, 2.0))
        .unwrap();
    let id = renderer
        .resources_mut()
        .register_texture_view(&device, &view_a, 2, 2);
    let items = vec![
        textured_plane(mesh, id, -1.05),
        textured_plane(mesh, id, 1.05),
    ];

    let before =
        checksum(&renderer.render_offscreen(&device, &queue, &frame_for(items.clone()), W, H));
    assert!(renderer.is_using_instanced_path());

    assert!(
        renderer
            .resources_mut()
            .update_texture_view(&device, id, &view_b, 2, 2),
        "re-pointing a registered external view should succeed"
    );
    let after = checksum(&renderer.render_offscreen(&device, &queue, &frame_for(items), W, H));
    assert_ne!(
        before, after,
        "re-pointing an external view must reach the screen (before={before} \
         after={after}); the id stays live, so only the view epoch can carry this"
    );
}

/// A free landing in the same frame as an in-flight async upload's apply.
///
/// The upload's apply closure checks the slot generation, so it must not write
/// into a slot the free has already recycled. The render is the end-to-end check:
/// whatever the ordering, the freed texture must not come back.
#[test]
fn a_free_racing_an_in_flight_upload_does_not_resurrect_the_texture() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping a_free_racing_an_in_flight_upload: no adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(2.0, 2.0))
        .unwrap();
    let doomed = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_SWAP)),
        )
        .unwrap();
    let items = vec![
        textured_plane(mesh, doomed, -1.05),
        textured_plane(mesh, doomed, 1.05),
    ];
    let untextured: Vec<viewport_lib::SceneRenderItem> = items
        .iter()
        .cloned()
        .map(|mut it| {
            it.material = Material::default();
            it.settings = unlit_settings();
            it
        })
        .collect();

    let before =
        checksum(&renderer.render_offscreen(&device, &queue, &frame_gen(items.clone(), 1), W, H));
    let oracle =
        checksum(&renderer.render_offscreen(&device, &queue, &frame_gen(untextured, 2), W, H));
    assert_ne!(before, oracle, "the texture must be visible to begin with");

    // Start an upload, free the in-use texture, then let the upload land. The
    // upload takes the recycled slot, so the item's handle is stale either way.
    let job = renderer
        .resources_mut()
        .begin_upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, C_START)),
        )
        .expect("begin upload");
    assert!(renderer.resources_mut().free_texture(doomed));
    // The job queue is pumped by `process_uploads`, which `prepare` calls; nothing
    // advances it on its own, so a bare spin here never finishes.
    let mut arrived = None;
    for _ in 0..1000 {
        renderer.resources_mut().process_uploads(&device, &queue);
        if let Ok(id) = renderer.resources_mut().upload_result_texture(job) {
            arrived = Some(id);
            break;
        }
        std::thread::yield_now();
    }
    let arrived = arrived.expect("the queued texture upload should land within 1000 pumps");

    let after = checksum(&renderer.render_offscreen(&device, &queue, &frame_gen(items, 3), W, H));
    assert_eq!(
        after,
        oracle,
        "an item holding a freed id must render untextured even when a later \
         upload took its slot (after={after} untextured={oracle} before={before}); \
         the arrived texture landed in slot {} and the freed one was slot {}",
        arrived.index(),
        doomed.index()
    );
}

/// The alpha-cutout shadow path, on a free rather than a replace.
///
/// `shadow_cull_bind_groups_stale` forces `bundle_key = None`, re-recording the
/// shadow bundle. The replace case is covered above; a caster whose albedo is
/// *freed* is the corner that was not.
///
/// The direction is deliberate. Start with a fully transparent mask, so the
/// cutout discards every texel and the caster throws no shadow; freeing the
/// texture leaves the material with no albedo at all, so the cutout falls back to
/// the base colour's alpha and the caster becomes solid. The shadow therefore
/// *appears*. Starting opaque would change nothing visible and the test would
/// pass without observing anything.
#[test]
fn instanced_cutout_shadow_reflects_free_texture() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping instanced_cutout_shadow_reflects_free_texture: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let ground = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::cuboid(24.0, 24.0, 0.5))
        .unwrap();
    let caster_mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(4.0, 4.0))
        .unwrap();
    // Fully transparent: the cutout discards every texel, so no shadow.
    let tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            TextureData::srgb(2, 2, solid_rgba(2, 2, [255, 255, 255, 0])),
        )
        .unwrap();

    use viewport_lib::AlphaMode;
    let build = || -> FrameData {
        let mut items = Vec::new();
        let mut g = viewport_lib::SceneRenderItem::default();
        g.mesh_id = ground;
        g.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
        g.material = Material::from_colour([0.85, 0.85, 0.85]);
        items.push(g);
        for x in [-1.0f32, 1.0] {
            let mut c = viewport_lib::SceneRenderItem::default();
            c.mesh_id = caster_mesh;
            c.model = glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 5.0)).to_cols_array_2d();
            let mut m = Material::textured(tex);
            m.backface_policy = BackfacePolicy::Identical;
            m.alpha_mode = AlphaMode::Mask(0.5);
            c.material = m;
            items.push(c);
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
        let mut l = viewport_lib::LightingSettings::default();
        l.lights = vec![{
            let mut s = viewport_lib::LightSource::default();
            s.kind = viewport_lib::LightKind::Directional {
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

    // Frame 1: transparent mask, so the region is lit.
    let lit = region(&renderer.render_offscreen(&device, &queue, &build(), EW, EH));
    assert!(
        renderer.is_using_instanced_path(),
        "two casters sharing one mesh must select the instanced path"
    );

    assert!(renderer.resources_mut().free_texture(tex));

    // Frame 2: the mask is gone, so the caster is solid and shadows the region.
    let shadowed = region(&renderer.render_offscreen(&device, &queue, &build(), EW, EH));
    assert!(
        shadowed + 5_000 < lit,
        "freeing a cutout caster's albedo must re-record the shadow bundle \
         (lit={lit} after-free={shadowed}); an unchanged region means the cull \
         bind groups and the bundle survived the free"
    );
}
