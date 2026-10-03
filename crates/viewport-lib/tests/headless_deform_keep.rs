//! A deformer that defines `keep` removes surface in the colour passes, per item.
//!
//! The test deformer cuts with a plane carried in per-instance slot data, so it
//! reaches only items that select that instance. Two quads share one mesh over
//! a red backdrop; one selects the cut and loses half, the other stays whole.
//! Skips when no adapter is available or the device cannot run deformers.

mod common;
use common::*;

use viewport_lib::renderer::{CameraFrame, SceneFrame};
use viewport_lib::resources::{DeformStage, DeformerDesc};
use viewport_lib::wgpu;

const W: u32 = 128;
const H: u32 = 128;

/// Keeps the side of a world-space plane `dot(p, n) + d >= 0`, with `(n, d)`
/// read from the item's per-instance data. Items without that data are kept.
const PLANE_CUT: &str = "\
fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    return v;
}

fn keep(v: DeformVertex, ctx: DeformContext) -> f32 {
    if deform_instance_slot_stride(ctx.slot) == 0u {
        return 1.0;
    }
    let n = vec3<f32>(
        deform_read_instance_f32(ctx.slot, 0u, 0u),
        deform_read_instance_f32(ctx.slot, 0u, 1u),
        deform_read_instance_f32(ctx.slot, 0u, 2u),
    );
    return dot(v.position, n) + deform_read_instance_f32(ctx.slot, 0u, 3u);
}
";

fn plane_cut() -> DeformerDesc {
    DeformerDesc {
        name: "test_plane_cut",
        stage: DeformStage::WorldSpace,
        priority: 0,
        wgsl_body: PLANE_CUT.to_string(),
        per_vertex_stride: 4,
    }
}

fn top_down() -> RenderCamera {
    let mut cam = Camera::default();
    cam.orientation = glam::Quat::IDENTITY;
    cam.center = glam::Vec3::ZERO;
    cam.distance = 5.0;
    cam.aspect = W as f32 / H as f32;
    RenderCamera::from_camera(&cam)
}

fn flat(mesh: MeshId, model: glam::Mat4, colour: [f32; 3]) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = model.to_cols_array_2d();
    item.material = Material::from_colour(viewport_lib::Colour::linear_rgb(
        colour[0], colour[1], colour[2],
    ));
    item.settings.unlit = true;
    item
}

/// The pixel a world point lands on, read back as RGBA.
fn pixel_at(img: &[u8], rc: &RenderCamera, world: glam::Vec3) -> [u8; 4] {
    let clip = rc.view_proj() * world.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = (((ndc.x * 0.5 + 0.5) * W as f32) as u32).min(W - 1);
    let y = (((0.5 - ndc.y * 0.5) * H as f32) as u32).min(H - 1);
    let i = ((y * W + x) * 4) as usize;
    [img[i], img[i + 1], img[i + 2], img[i + 3]]
}

fn is_red(p: [u8; 4]) -> bool {
    p[0] > 150 && p[1] < 80 && p[2] < 80
}

fn is_white(p: [u8; 4]) -> bool {
    p[0] > 150 && p[1] > 150 && p[2] > 150
}

#[test]
fn a_keep_hook_cuts_only_the_item_that_selects_it() {
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let quad = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(1.0, 1.0))
        .unwrap();
    let id = match renderer
        .resources_mut()
        .register_deformer(&device, plane_cut())
    {
        Ok(id) => id,
        Err(e) => {
            eprintln!("skipping: device cannot run deformers ({e})");
            return;
        }
    };

    // Keep x < -0.6 on the left quad: n = (-1, 0, 0), d = -0.6.
    let plane: [f32; 4] = [-1.0, 0.0, 0.0, -0.6];
    renderer.resources_mut().attach_deform_slot_instance(
        &device,
        &queue,
        quad,
        1,
        id.slot(),
        16,
        bytemuck::cast_slice(&plane),
    );

    let left = glam::Mat4::from_translation(glam::Vec3::new(-0.6, 0.0, 0.0));
    let right = glam::Mat4::from_translation(glam::Vec3::new(0.6, 0.0, 0.0));
    let mut cut = flat(quad, left, [1.0, 1.0, 1.0]);
    cut.deform_instance = Some(1);
    let whole = flat(quad, right, [1.0, 1.0, 1.0]);
    let backdrop = flat(
        quad,
        glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(4.0),
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 0.0, -0.5),
        ),
        [1.0, 0.0, 0.0],
    );

    let rc = top_down();
    let mut frame = FrameData::new(
        CameraFrame::new(rc.clone(), [W as f32, H as f32]),
        SceneFrame::from_surface_items(vec![backdrop, cut, whole]),
    );
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // HDR as well as the default: the HDR opaque pass has a discard-free
    // early-Z twin, which a cut item must not be drawn with.
    for mode in [
        viewport_lib::PipelineMode::default(),
        viewport_lib::PipelineMode::Hdr,
    ] {
        frame.effects.display.mode = mode;
        let img = renderer.render_offscreen(&device, &queue, &frame, W, H);
        let probe = |x: f32| pixel_at(&img, &rc, glam::Vec3::new(x, 0.25, 0.0));
        assert!(
            is_white(probe(-0.85)),
            "{mode:?}: the kept half of the cut quad is drawn"
        );
        assert!(
            is_red(probe(-0.35)),
            "{mode:?}: the removed half of the cut quad shows the backdrop, got {:?}",
            probe(-0.35)
        );
        assert!(
            is_white(probe(0.35)),
            "{mode:?}: the quad without the cut is whole"
        );
        assert!(
            is_white(probe(0.85)),
            "{mode:?}: the quad without the cut is whole"
        );
    }
}

/// The same scene through the OIT path: a transparent quad is cut too.
#[test]
fn a_keep_hook_cuts_transparent_items() {
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let quad = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(1.0, 1.0))
        .unwrap();
    let id = match renderer
        .resources_mut()
        .register_deformer(&device, plane_cut())
    {
        Ok(id) => id,
        Err(e) => {
            eprintln!("skipping: device cannot run deformers ({e})");
            return;
        }
    };
    let plane: [f32; 4] = [-1.0, 0.0, 0.0, 0.0];
    renderer.resources_mut().attach_deform_slot_instance(
        &device,
        &queue,
        quad,
        1,
        id.slot(),
        16,
        bytemuck::cast_slice(&plane),
    );

    let mut glass = flat(quad, glam::Mat4::IDENTITY, [1.0, 1.0, 1.0]);
    glass.settings.opacity = 0.9;
    glass.deform_instance = Some(1);
    let backdrop = flat(
        quad,
        glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(4.0),
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 0.0, -0.5),
        ),
        [1.0, 0.0, 0.0],
    );

    let rc = top_down();
    let mut frame = FrameData::new(
        CameraFrame::new(rc.clone(), [W as f32, H as f32]),
        SceneFrame::from_surface_items(vec![backdrop, glass]),
    );
    frame.effects.display.mode = viewport_lib::PipelineMode::Hdr;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    let img = renderer.render_offscreen(&device, &queue, &frame, W, H);

    let kept = pixel_at(&img, &rc, glam::Vec3::new(-0.3, 0.25, 0.0));
    let removed = pixel_at(&img, &rc, glam::Vec3::new(0.3, 0.25, 0.0));
    assert!(
        kept[1] > 120,
        "the kept half is tinted by the glass, got {kept:?}"
    );
    assert!(
        is_red(removed),
        "the removed half shows the backdrop untinted, got {removed:?}"
    );
}
