//! Pipeline rebuilds and the two-bind-group device draw path.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

/// Toggling DebugVis swaps the lit pipelines between the stripped shader
/// variant (no debug storage write, early-Z friendly) and the full debug
/// variant, and back. Exercises the rebuild path end to end: a validation
/// error in either module variant fails the prepare calls.
#[test]
fn debug_vis_toggle_rebuilds_lit_pipelines() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // Default: stripped variant.
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // On: rebuilds with the debug block present.
    frame.effects.lighting.debug_vis.active = true;
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // Off again: rebuilds stripped.
    frame.effects.lighting.debug_vis.active = false;
    let _ = renderer.pass().prepare(&device, &queue, &frame);
}

/// Regression test for the recurring class of bug where a draw site
/// unconditionally binds wgpu group index 2. iced_wgpu 0.14 requests a device
/// with `max_bind_groups: 2` and gives consumers no way to raise it, so any
/// unconditional `set_bind_group(2, ...)` fails wgpu validation the moment a
/// mesh, shadow, or HDR pass draws on that device. This has been fixed and
/// silently reintroduced by new draw sites multiple times (see
/// `docs/issues/iced-max-bind-groups-2-draw-path-incomplete.md`) because the
/// only thing that previously caught it was manually running the iced
/// example. This test exercises the same crash surface headlessly on every
/// `cargo test`, with no reliance on remembering to gate a new call site.
///
/// wgpu's default (no error-scope) behaviour is to raise the validation
/// error through the uncaptured-error handler, which panics the calling
/// thread; a regression here fails the test the same way it crashed iced.
#[test]
fn two_bind_group_device_renders_without_validation_errors() {
    let Some((device, queue)) = headless_device_limited_bind_groups() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(
        device.limits().max_bind_groups,
        2,
        "adapter did not honour the requested 2-group limit; test would not \
         be exercising the iced crash surface"
    );

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    // Default lighting has shadows_enabled = true, and mesh items default to
    // cast_shadows = true, so the shadow pass (the site that crashed the
    // iced example even after the mesh draw path was gated) runs below.

    // One opaque box, one transparent box so the OIT pipeline family draws
    // too, not just the opaque path.
    let mut opaque = SceneRenderItem::default();
    opaque.mesh_id = mesh_id;
    let mut transparent = SceneRenderItem::default();
    transparent.mesh_id = mesh_id;
    transparent.settings.opacity = 0.5;
    transparent.model =
        glam::Mat4::from_translation(glam::Vec3::new(1.5, 0.0, 0.0)).to_cols_array_2d();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![opaque, transparent].into());

    // LDR path.
    frame.effects.post_process.enabled = false;
    let ldr_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(ldr_pixels.len(), 64 * 64 * 4);

    // HDR path: exercises the deform/OIT binds inside hdr_path.rs and the
    // full tonemap/bloom/SSAO/FXAA post chain.
    frame.effects.post_process.enabled = true;
    let hdr_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(hdr_pixels.len(), 64 * 64 * 4);

    // Wireframe mode is a separate draw path (wireframe pipeline, its own
    // group-2 deform dummy bind) from both the solid and HDR paths above.
    frame.viewport.wireframe_mode = true;
    let wireframe_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(wireframe_pixels.len(), 64 * 64 * 4);
}

/// Two non-instanced items that share one mesh but differ in transform and
/// colour must each draw with their own data. The per-object path writes each
/// item's `ObjectUniform` to a distinct slot in the shared object-data buffer
/// and draws it with that slot as `@builtin(instance_index)`; if the indexing
/// collapsed (every draw at instance 0) both boxes would render with the first
/// item's red colour and the second item's green would never appear. A styled
/// back-face policy forces the non-instanced path so this exercises the indexed
/// object-data buffer rather than the instanced path.
#[test]
fn per_object_items_select_distinct_object_data() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [128.0, 128.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.post_process.enabled = false;

    let make = |x: f32, colour: [f32; 3]| {
        let mut it = SceneRenderItem::default();
        it.mesh_id = mesh_id;
        it.material.base_colour = colour;
        // Styled back faces => is_instanceable is false => per-object path.
        it.material.backface_policy = BackfacePolicy::DifferentColour([0.0, 0.0, 0.0]);
        it.settings.unlit = true;
        it.model = glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
        it
    };
    frame.scene.surfaces = SurfaceSubmission::Flat(
        vec![make(-1.0, [1.0, 0.0, 0.0]), make(1.0, [0.0, 1.0, 0.0])].into(),
    );

    let px = renderer.render_offscreen(&device, &queue, &frame, 128, 128);
    let (mut has_red, mut has_green) = (false, false);
    for p in px.chunks_exact(4) {
        if p[0] > 128 && p[1] < 80 && p[2] < 80 {
            has_red = true;
        }
        if p[1] > 128 && p[0] < 80 && p[2] < 80 {
            has_green = true;
        }
    }
    assert!(has_red, "first per-object item (red) did not render");
    assert!(
        has_green,
        "second per-object item (green) did not render: items collapsed to one object-data slot"
    );
}

/// Building a renderer and rendering the full HDR + shadow + OIT + per-object
/// mesh path on a device requested with exactly `recommended_device_limits`
/// (not the adapter's full caps) must succeed. This is the path a
/// limits-following consumer takes, and it guards the requirement two ways:
/// `ViewportRenderer::new`'s up-front assert fails on any backend if the device
/// is under the storage-buffer requirement, and on backends that enforce the
/// limit at pipeline-layout creation (Vulkan, DX12) an over-budget layout also
/// fails here. Metal does not enforce it at layout creation, so on the local
/// box this is a smoke test that the whole path builds under the recommended
/// limits; the enforcing-backend catch needs a Vulkan/DX12 run.
#[test]
fn renderer_fits_recommended_device_limits() {
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // Construction builds the LDR mesh pipeline layout (group 0 + group 1
    // storage buffers): the original wgpu29 failure was here.
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // A styled-backface (per-object) opaque item plus a transparent item, so the
    // per-object object-data storage binding, the OIT layout, and the shadow
    // pass (default lighting casts shadows) all build under the capped limits.
    let mut opaque = SceneRenderItem::default();
    opaque.mesh_id = mesh_id;
    opaque.material.backface_policy = BackfacePolicy::DifferentColour([0.0, 0.0, 0.0]);
    let mut transparent = SceneRenderItem::default();
    transparent.mesh_id = mesh_id;
    transparent.settings.opacity = 0.5;
    transparent.model =
        glam::Mat4::from_translation(glam::Vec3::new(1.5, 0.0, 0.0)).to_cols_array_2d();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![opaque, transparent].into());

    // HDR path builds the clustered lit / OIT pipeline layouts.
    frame.effects.post_process.enabled = true;
    let hdr = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(hdr.len(), 64 * 64 * 4);

    // LDR path (its own mesh pipeline layout family).
    frame.effects.post_process.enabled = false;
    let ldr = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(ldr.len(), 64 * 64 * 4);
}

/// `apply_tuning` then `tuning()` round-trips the persistent knobs. Uses
/// `gpu_driven_culling: false` (always honored) so the snapshot equals the input
/// exactly regardless of device support, and a render scale inside the default
/// policy bounds so it is not clamped.
#[test]
fn render_tuning_round_trips() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Non-exhaustive: consumers start from Default and set fields.
    let mut want = viewport_lib::RenderTuning::default();
    want.gpu_driven_culling = false;
    want.occlusion_culling = true;
    want.render_scale = 0.75;
    want.runtime_mode = viewport_lib::RuntimeMode::Playback;
    want.upload_budget = Some(std::time::Duration::from_millis(2));
    want.cpu_pick_cache = true;
    want.diagnostics.force_multi_draw = true;
    want.diagnostics.force_po_discard = true;
    renderer.apply_tuning(&want);
    assert_eq!(renderer.tuning(), want);
}

/// Overlay SDF shapes that exercise the new stacked-shadow storage buffer,
/// rotation pivot, and textured backdrop-filter paths must all render without
/// tripping wgpu validation. The solid pipeline now carries a group-0 storage
/// buffer bind group for shadow layers; a missing or mismatched binding would
/// fail validation the moment the shapes draw. A shape with a coloured fill
/// over a mid-grey background must leave visible non-background pixels.
#[test]
fn overlay_shape_shadow_layers_and_pivot_render() {
    use viewport_lib::{OverlayFill, OverlayShape, OverlayShapeItem, ShadowLayer};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 96u32;
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.3, 0.3, 0.3, 1.0]);

    // A shape with two outer shadow layers plus two inner shadow layers,
    // rotated about an off-centre pivot. Fill is bright so it stands out.
    frame.overlays.shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 8.0 },
            [24.0, 24.0],
            [48.0, 48.0],
        )
        .with_fill(OverlayFill::Solid([0.9, 0.2, 0.1, 1.0]))
        .with_border([1.0, 1.0, 1.0, 1.0], 2.0)
        .with_rotation(0.5)
        .with_rotation_pivot([10.0, 6.0])
        .with_shadows(vec![
            ShadowLayer::new([0.0, 0.0, 0.0, 0.5], 12.0, [0.0, 6.0]),
            ShadowLayer::new([0.0, 0.0, 0.0, 0.6], 4.0, [0.0, 2.0]),
        ])
        .with_inner_shadows(vec![
            ShadowLayer::new([1.0, 1.0, 1.0, 0.3], 5.0, [0.0, -2.0]),
            ShadowLayer::new([0.0, 0.0, 0.0, 0.4], 8.0, [0.0, 3.0]),
        ]),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    // Look for a clearly red-dominant pixel produced by the shape fill.
    let mut found = false;
    for i in (0..px.len()).step_by(4) {
        let (r, g, b) = (px[i] as i32, px[i + 1] as i32, px[i + 2] as i32);
        if r > 150 && r - g > 60 && r - b > 60 {
            found = true;
            break;
        }
    }
    assert!(found, "expected the red shadowed shape to render");
}

/// A backdrop-blur shape with colour filters (saturation / brightness /
/// hue-shift) must render through the blur + textured-overlay pipelines
/// without wgpu validation errors. The filters are encoded into the blur
/// vertex `extras` slot and applied in the textured shader's blur branch.
#[test]
fn overlay_shape_backdrop_filters_render() {
    use viewport_lib::{OverlayFill, OverlayShape, OverlayShapeItem};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 96u32;
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.2, 0.5, 0.8, 1.0]);

    frame.overlays.shapes = vec![
        OverlayShapeItem::new(OverlayShape::Circle, [24.0, 24.0], [48.0, 48.0])
            .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 0.1]))
            .with_backdrop_blur(8.0)
            .with_backdrop_filters(0.4, 0.9, 1.0),
    ];

    // Just needs to complete without a validation panic; also sanity-check we
    // got a full frame back.
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert_eq!(px.len(), (size * size * 4) as usize);
}

/// A capsule rotated about its bottom end (an off-centre pivot) must not clip
/// against its own axis-aligned bounding quad: the prepare pass grows the quad
/// to the AABB of the rotated shape. Rotated ~90 degrees, a tall narrow
/// capsule lies horizontally and must paint pixels well outside its original
/// narrow column.
#[test]
fn overlay_shape_pivot_rotation_not_clipped() {
    use viewport_lib::{OverlayFill, OverlayShape, OverlayShapeItem};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 160u32;
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.1, 0.1, 0.1, 1.0]);

    // Tall narrow capsule: box x in [60, 76], centre (68, 80). Pivot at the
    // bottom end; rotate 90 degrees so the hand swings out horizontally.
    frame.overlays.shapes = vec![
        OverlayShapeItem::new(OverlayShape::Capsule, [60.0, 20.0], [16.0, 120.0])
            .with_fill(OverlayFill::Solid([0.95, 0.75, 0.2, 1.0]))
            .with_rotation(std::f32::consts::FRAC_PI_2)
            .with_rotation_pivot([0.0, 60.0]),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let is_yellow = |i: usize| {
        let (r, g, b) = (px[i] as i32, px[i + 1] as i32, px[i + 2] as i32);
        r > 150 && g > 110 && b < 130 && r - b > 50
    };
    // Look for the capsule far outside its original column (x well past 76).
    let mut found_far = false;
    for y in 0..size {
        for x in 110..size {
            if is_yellow(((y * size + x) * 4) as usize) {
                found_far = true;
                break;
            }
        }
        if found_far {
            break;
        }
    }
    assert!(
        found_far,
        "rotated capsule should paint outside its original box (no clip)"
    );
}
