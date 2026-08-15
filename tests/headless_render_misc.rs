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

