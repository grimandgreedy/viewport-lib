//! A snapped world origin far from zero: the instanced batches repack only
//! when the origin moves, and a clipped mesh draws the same cut it draws at
//! the world origin.

use viewport_lib::{
    CameraFrame, ClipObject, Colour, FrameData, Material, MeshId, RenderCamera, SceneFrame,
    SceneRenderItem, WorldOrigin, primitives,
};
use viewport_lib_testkit::Harness;

const W: u32 = 160;
const H: u32 = 120;

/// 2^20 m from the world origin. Every coordinate below is a multiple of
/// 1/8, which f32 holds exactly out here, so the inputs carry no rounding.
const FAR: f32 = 1_048_576.0;

fn cube(harness: &mut Harness) -> MeshId {
    harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &primitives::cube(1.0))
        .expect("cube upload")
}

fn items(harness: &mut Harness, mesh: MeshId, offset: glam::Vec3) -> Vec<SceneRenderItem> {
    let material = harness
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.65)))
        .expect("material upload");
    // A 4 x 4 grid, enough to take the cached-batch path.
    (0..16)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh;
            item.material = material;
            let at = glam::Vec3::new((i % 4) as f32 * 1.5, (i / 4) as f32 * 1.5, 0.0);
            item.model = glam::Mat4::from_translation(offset + at).into();
            item
        })
        .collect()
}

fn frame(items: &[SceneRenderItem], eye: glam::Vec3, target: glam::Vec3) -> FrameData {
    let camera = RenderCamera::from_matrices(
        glam::Mat4::look_at_rh(eye, target, glam::Vec3::Z),
        glam::Mat4::perspective_infinite_reverse_rh(0.9, W as f32 / H as f32, 0.1),
    )
    .with_depth_range(0.1, 200.0);
    FrameData::new(
        CameraFrame::new(camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(items.to_vec()).with_generation(1),
    )
}

/// The camera moving inside one snap cell repacks nothing; crossing out of
/// it moves the origin and repacks on exactly that frame.
#[test]
fn origin_snaps_repack_once() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness
        .renderer
        .set_world_origin(WorldOrigin::snapped().with_cell(16.0));
    let mesh = cube(&mut harness);
    let offset = glam::Vec3::new(FAR, FAR, 0.0);
    let items = items(&mut harness, mesh, offset);
    let target = offset + glam::Vec3::new(2.25, 2.25, 0.0);
    let eye_at = |dx: f32| offset + glam::Vec3::new(dx, -8.0, 6.0);

    let _ = harness.render_two_frames(&frame(&items, eye_at(0.0), target), W, H);
    let settled = harness.renderer.world_origin();

    // Inside the cell: the eye stays within 12 m (0.75 cells) of the origin.
    for step in 1..=8 {
        let _ = harness.render(&frame(&items, eye_at(step as f32 * 0.5), target), W, H);
        let s = harness.stats();
        assert_eq!(s.batches_reuploaded, 0, "step {step} repacked a batch");
        assert_eq!(
            harness.renderer.world_origin(),
            settled,
            "step {step} snapped"
        );
    }

    // Past it: one frame repacks, the next does not.
    let mut repacked = Vec::new();
    for step in 0..4 {
        let _ = harness.render(
            &frame(&items, eye_at(20.0 + step as f32 * 0.5), target),
            W,
            H,
        );
        repacked.push(harness.stats().batches_reuploaded > 0);
    }
    assert_ne!(harness.renderer.world_origin(), settled, "never snapped");
    assert_eq!(
        repacked,
        [true, false, false, false],
        "frames that repacked after the camera left the cell"
    );
}

/// A mesh 2^20 m out, cut by a clip plane, draws under a snapped origin as the
/// same scene does at the world origin.
#[test]
fn clip_plane_parity_far_from_origin() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mesh = cube(&mut harness);
    let render = |harness: &mut Harness, offset: glam::Vec3, origin: WorldOrigin| {
        harness.renderer.set_world_origin(origin);
        let items = items(harness, mesh, offset);
        let target = offset + glam::Vec3::new(2.25, 2.25, 0.0);
        let eye = offset + glam::Vec3::new(-3.0, -7.0, 5.0);
        let mut f = frame(&items, eye, target);
        // The plane x = offset.x + 2.125, keeping the side below it.
        let d = offset.x + 2.125;
        f.effects
            .clip
            .objects
            .push(ClipObject::plane([-1.0, 0.0, 0.0], d));
        let _ = harness.render(&f, W, H);
        harness.render(&f, W, H)
    };
    let reference = render(&mut harness, glam::Vec3::ZERO, WorldOrigin::Zero);
    let snapped = render(
        &mut harness,
        glam::Vec3::new(FAR, FAR, 0.0),
        WorldOrigin::snapped().with_cell(16.0),
    );
    let zero_far = render(
        &mut harness,
        glam::Vec3::new(FAR, FAR, 0.0),
        WorldOrigin::Zero,
    );
    let differing = |img: &[u8]| {
        img.chunks_exact(4)
            .zip(reference.chunks_exact(4))
            .filter(|(a, b)| a.iter().zip(*b).any(|(x, y)| x.abs_diff(*y) > 2))
            .count()
    };
    let (snapped_px, zero_px) = (differing(&snapped), differing(&zero_far));
    eprintln!(
        "pixels differing from the world-origin render by more than 2/255: \
         snapped {snapped_px}, zero {zero_px} of {}",
        W * H
    );
    // The cut and every edge land where they do at the world origin, to
    // within rounding on a handful of edge pixels.
    assert!(
        snapped_px <= (W * H / 200) as usize,
        "snapped render differs in {snapped_px} pixels"
    );
}
