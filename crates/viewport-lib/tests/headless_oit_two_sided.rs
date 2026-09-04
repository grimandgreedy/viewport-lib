//! Two-sided transparent surfaces must draw their back faces on the HDR/OIT path.
//!
//! Regression for the OIT pipelines being hardcoded to `cull_mode: Back`: a
//! two-sided material at opacity < 1 lost every back-facing triangle on the HDR
//! path (the LDR path and the opaque pipelines were fine). The scene here is a
//! single quad wound so the camera sees its back face: `Cull` renders nothing,
//! `Identical` (two-sided) must render it, and at opacity 0.75 that has to hold
//! through the OIT pass, not just when opaque.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

/// A flat quad in the z = 0 plane, wound so its front face points at -Z. Viewed
/// from +Z the camera sees the back face, which `Cull` discards.
fn back_facing_quad() -> MeshData {
    let positions = vec![
        [-1.5, -1.5, 0.0],
        [1.5, -1.5, 0.0],
        [1.5, 1.5, 0.0],
        [-1.5, 1.5, 0.0],
    ];
    let normals = vec![[0.0, 0.0, -1.0]; 4];
    // Wound clockwise as seen from +Z, so +Z is the back face.
    let indices = vec![0, 2, 1, 0, 3, 2];
    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}

fn coverage(px: &[u8]) -> usize {
    // Any pixel with a visible red channel is the quad (unlit red on black).
    px.chunks_exact(4).filter(|p| p[0] > 40).count()
}

#[test]
fn two_sided_transparent_draws_back_faces_on_hdr_oit() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let size = 128u32;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &back_facing_quad())
        .unwrap();

    // Camera straight above the quad (+Z), looking down the -Z axis at the
    // back face. Small pitch is the top-down view (see the testkit's cameras).
    let cam = {
        let mut c = Camera::default();
        c.center = glam::Vec3::ZERO;
        c.distance = 6.0;
        c.orientation = glam::Quat::from_rotation_x(0.15);
        c
    };
    let render = |renderer: &mut ViewportRenderer, two_sided: bool, opacity: f32| {
        let mut frame = FrameData::default();
        frame.camera.render_camera = {
            let mut rc = RenderCamera::from_camera(&cam);
            rc.aspect = 1.0;
            rc
        };
        frame.camera.viewport_size = [size as f32, size as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0].into());
        frame.effects.display.mode = viewport_lib::PipelineMode::Hdr;
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.material.base_colour = [1.0, 0.0, 0.0].into();
        item.settings.unlit = true;
        item.settings.opacity = opacity;
        if two_sided {
            item.material.backface_policy = BackfacePolicy::Identical;
        }
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        coverage(&renderer.render_offscreen(&device, &queue, &frame, size, size))
    };

    // One-sided: the back face is culled, so opaque or transparent it is empty.
    // This anchors the geometry: the camera really is looking at the back face.
    let one_sided_opaque = render(&mut renderer, false, 1.0);
    assert_eq!(
        one_sided_opaque, 0,
        "one-sided quad viewed from behind should cull to nothing (coverage {one_sided_opaque}); \
         the test is not looking at the back face"
    );

    // Two-sided opaque: back face draws (control for the two-sided pipeline).
    let two_sided_opaque = render(&mut renderer, true, 1.0);
    assert!(
        two_sided_opaque > 1000,
        "two-sided opaque quad did not draw its back face (coverage {two_sided_opaque})"
    );

    // Two-sided at opacity 0.75 through the OIT pass: this is the regression.
    // Before the fix the OIT pipeline was back-face culled, so this was 0.
    let two_sided_transparent = render(&mut renderer, true, 0.75);
    assert!(
        two_sided_transparent > 1000,
        "two-sided transparent quad lost its back faces on the HDR/OIT path \
         (coverage {two_sided_transparent}); the OIT pipeline is back-face culled"
    );
}
