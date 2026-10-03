//! The cut deformer: each shape keeps the side it says, flipping keeps the
//! other, several cuts keep what all of them keep, a range reads its field from
//! values or from a mesh attribute, and the cut lands on the position other
//! deformers moved the mesh to. Built against the public API only, as any
//! dependent would.
//!
//! The scenes are a white grid quad over a red backdrop, seen from above.

#![cfg(feature = "deformers")]

use viewport_lib::renderer::{CameraFrame, FrameData, RenderCamera, SceneFrame};
use viewport_lib::resources::{AttributeData, DeformStage, DeformerDesc};
use viewport_lib::{Camera, Material, MeshId, SceneRenderItem, ViewportRenderer, gpu};
use viewport_lib_plugins::deformers::cut::{Cut, CutDeformer};
use viewport_lib_testkit::{DeviceProfile, headless_device_with};

const W: u32 = 128;
const H: u32 = 128;

struct Scene {
    device: gpu::Device,
    queue: gpu::Queue,
    renderer: ViewportRenderer,
    cut: CutDeformer,
    grid: MeshId,
    backdrop: MeshId,
}

/// A renderer with the cut installed, a 1 x 1 grid quad carrying an `x`
/// attribute (its own x coordinate), and a backdrop mesh. `None` without an
/// adapter.
fn scene() -> Option<Scene> {
    let (device, queue) = headless_device_with(
        &DeviceProfile::high_performance("cut-deformer").with_recommended_features(),
    )?;
    let mut renderer = ViewportRenderer::new(&device, gpu::TextureFormat::Rgba8UnormSrgb);
    let cut = CutDeformer::install(renderer.resources_mut(), &device).ok()?;
    let mut grid_data = viewport_lib::primitives::grid_plane(1.0, 1.0, 32, 32);
    let xs = grid_data.positions.iter().map(|p| p[0]).collect();
    grid_data
        .attributes
        .insert("x".to_string(), AttributeData::Vertex(xs));
    let grid = renderer
        .resources_mut()
        .upload_mesh_data(&device, &grid_data)
        .unwrap();
    let backdrop = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(4.0, 4.0))
        .unwrap();
    Some(Scene {
        device,
        queue,
        renderer,
        cut,
        grid,
        backdrop,
    })
}

fn camera() -> RenderCamera {
    let mut cam = Camera::default();
    cam.orientation = glam::Quat::IDENTITY;
    cam.center = glam::Vec3::ZERO;
    cam.distance = 5.0;
    cam.aspect = 1.0;
    RenderCamera::from_camera(&cam)
}

fn flat(mesh: MeshId, translation: glam::Vec3, colour: [f32; 3]) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::from_translation(translation).to_cols_array_2d();
    item.material = Material::from_colour(viewport_lib::Colour::linear_rgb(
        colour[0], colour[1], colour[2],
    ));
    item.settings.unlit = true;
    item
}

impl Scene {
    /// Render the grid quad, selecting instance 1, over the backdrop, and
    /// report whether each probe point shows the quad (white) rather than the
    /// backdrop (red).
    fn shows_quad(&mut self, extra: Vec<SceneRenderItem>, probes: &[[f32; 2]]) -> Vec<bool> {
        let mut quad = flat(self.grid, glam::Vec3::ZERO, [1.0, 1.0, 1.0]);
        quad.deform_instance = Some(1);
        let backdrop = flat(
            self.backdrop,
            glam::Vec3::new(0.0, 0.0, -0.5),
            [1.0, 0.0, 0.0],
        );
        let mut items = vec![backdrop, quad];
        items.extend(extra);
        let rc = camera();
        let mut frame = FrameData::new(
            CameraFrame::new(rc.clone(), [W as f32, H as f32]),
            SceneFrame::from_surface_items(items),
        );
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        let img = self
            .renderer
            .render_offscreen(&self.device, &self.queue, &frame, W, H);
        probes
            .iter()
            .map(|p| {
                let clip = rc.view_proj() * glam::Vec4::new(p[0], p[1], 0.0, 1.0);
                let ndc = clip.truncate() / clip.w;
                let x = (((ndc.x * 0.5 + 0.5) * W as f32) as u32).min(W - 1);
                let y = (((0.5 - ndc.y * 0.5) * H as f32) as u32).min(H - 1);
                let i = ((y * W + x) * 4) as usize;
                let (r, g) = (img[i], img[i + 1]);
                assert!(
                    r > 150 && (g > 150 || g < 80),
                    "probe {p:?} is neither quad nor backdrop: {:?}",
                    &img[i..i + 4]
                );
                g > 150
            })
            .collect()
    }

    fn set(&mut self, cuts: &[Cut]) {
        self.cut.set(
            self.renderer.resources_mut(),
            &self.device,
            &self.queue,
            self.grid,
            1,
            cuts,
        );
    }
}

#[test]
fn each_shape_keeps_its_side_and_flipping_keeps_the_other() {
    let Some(mut s) = scene() else {
        eprintln!("skipping: no GPU adapter or no deformer support");
        return;
    };
    let left = [-0.3, 0.1];
    let right = [0.3, 0.1];
    let centre = [0.0, 0.05];
    let corner = [0.4, 0.4];

    s.set(&[Cut::plane([1.0, 0.0, 0.0], 0.0)]);
    assert_eq!(s.shows_quad(vec![], &[left, right]), [false, true], "plane");
    s.set(&[Cut::plane([1.0, 0.0, 0.0], 0.0).flipped()]);
    assert_eq!(
        s.shows_quad(vec![], &[left, right]),
        [true, false],
        "flipped plane"
    );

    s.set(&[Cut::sphere([0.0, 0.0, 0.0], 0.25)]);
    assert_eq!(
        s.shows_quad(vec![], &[centre, corner]),
        [true, false],
        "sphere"
    );
    s.set(&[Cut::sphere([0.0, 0.0, 0.0], 0.25).flipped()]);
    assert_eq!(
        s.shows_quad(vec![], &[centre, corner]),
        [false, true],
        "flipped sphere"
    );

    s.set(&[Cut::aabb([-0.2, -0.2, -1.0], [0.2, 0.2, 1.0])]);
    assert_eq!(
        s.shows_quad(vec![], &[centre, corner]),
        [true, false],
        "box"
    );
    let quarter_turn = std::f32::consts::FRAC_1_SQRT_2;
    s.set(&[Cut::oriented_box(
        [0.0, 0.0, 0.0],
        [
            [quarter_turn, quarter_turn, 0.0],
            [-quarter_turn, quarter_turn, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [0.6, 0.1, 1.0],
    )]);
    // A thin bar along the diagonal: the corner on it is kept, the off-axis
    // point is not.
    assert_eq!(
        s.shows_quad(vec![], &[[0.3, 0.3], [0.3, -0.3]]),
        [true, false],
        "oriented box"
    );
}

#[test]
fn several_cuts_keep_what_every_cut_keeps() {
    let Some(mut s) = scene() else {
        eprintln!("skipping: no GPU adapter or no deformer support");
        return;
    };
    s.set(&[
        Cut::plane([1.0, 0.0, 0.0], 0.0),
        Cut::plane([0.0, 1.0, 0.0], 0.0),
    ]);
    assert_eq!(
        s.shows_quad(
            vec![],
            &[[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]]
        ),
        [true, false, false, false]
    );
    // An empty set keeps everything.
    s.set(&[]);
    assert_eq!(
        s.shows_quad(vec![], &[[0.3, 0.3], [-0.3, -0.3]]),
        [true, true]
    );
}

#[test]
fn a_range_cuts_the_same_from_values_and_from_an_attribute() {
    let Some(mut s) = scene() else {
        eprintln!("skipping: no GPU adapter or no deformer support");
        return;
    };
    let probes = [[-0.3, 0.1], [0.3, 0.1]];
    s.set(&[Cut::range(0.0, 1.0)]);
    // No field yet: a range has nothing to test, and keeps everything.
    assert_eq!(s.shows_quad(vec![], &probes), [true, true], "no field");

    let xs: Vec<f32> = viewport_lib::primitives::grid_plane(1.0, 1.0, 32, 32)
        .positions
        .iter()
        .map(|p| p[0])
        .collect();
    s.cut
        .set_field(s.renderer.resources_mut(), &s.device, s.grid, &xs);
    assert_eq!(
        s.shows_quad(vec![], &probes),
        [false, true],
        "field from values"
    );

    s.cut
        .clear_field(s.renderer.resources_mut(), &s.device, s.grid);
    s.cut
        .set_field_from_attribute(s.renderer.resources_mut(), &s.device, s.grid, "x")
        .unwrap();
    assert_eq!(
        s.shows_quad(vec![], &probes),
        [false, true],
        "field from attribute"
    );

    assert!(
        s.cut
            .set_field_from_attribute(s.renderer.resources_mut(), &s.device, s.grid, "missing")
            .is_err()
    );
}

#[test]
fn only_the_item_that_selects_the_cut_is_cut() {
    let Some(mut s) = scene() else {
        eprintln!("skipping: no GPU adapter or no deformer support");
        return;
    };
    s.set(&[Cut::plane([1.0, 0.0, 0.0], 0.0)]);
    // A second item on the same mesh, moved off to y = 1.2, selects nothing.
    let other = flat(s.grid, glam::Vec3::new(0.0, 1.2, 0.0), [1.0, 1.0, 1.0]);
    assert_eq!(
        s.shows_quad(vec![other], &[[-0.3, 0.1], [-0.3, 1.2]]),
        [false, true]
    );
    s.cut
        .clear(s.renderer.resources_mut(), &s.device, &s.queue, s.grid, 1);
    assert_eq!(s.shows_quad(vec![], &[[-0.3, 0.1]]), [true], "cleared");
}

#[test]
fn the_cut_lands_where_other_deformers_moved_the_mesh() {
    let Some(mut s) = scene() else {
        eprintln!("skipping: no GPU adapter or no deformer support");
        return;
    };
    // Slides the mesh +0.5 along x in object space, for any mesh with data in
    // its slot.
    let slide = s
        .renderer
        .resources_mut()
        .register_deformer(
            &s.device,
            DeformerDesc {
                name: "test_slide",
                stage: DeformStage::ObjectSpace,
                priority: 0,
                wgsl_body: "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    \
                            var out = v;\n    out.position.x = out.position.x + 0.5;\n    return out;\n}\n"
                    .to_string(),
                per_vertex_stride: 4,
            },
        )
        .unwrap();
    let vertex_count = viewport_lib::primitives::grid_plane(1.0, 1.0, 32, 32)
        .positions
        .len();
    s.renderer.resources_mut().attach_deform_slot(
        &s.device,
        s.grid,
        slide.slot(),
        4,
        bytemuck::cast_slice(&vec![0.0f32; vertex_count]),
    );
    // The slid quad spans x in [0, 1]. Keep x >= 0.6 in world space.
    s.set(&[Cut::plane([1.0, 0.0, 0.0], 0.6)]);
    assert_eq!(
        s.shows_quad(vec![], &[[0.3, 0.1], [0.8, 0.1]]),
        [false, true]
    );
}

#[test]
fn install_finds_the_registered_deformer() {
    let Some(mut s) = scene() else {
        eprintln!("skipping: no GPU adapter or no deformer support");
        return;
    };
    let again = CutDeformer::install(s.renderer.resources_mut(), &s.device).unwrap();
    assert_eq!(again.deformer_id(), s.cut.deformer_id());
}

#[test]
fn install_reports_a_device_without_deformer_support() {
    let Some((device, _queue)) =
        headless_device_with(&DeviceProfile::low_power("cut-two-groups").max_bind_groups(2))
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, gpu::TextureFormat::Rgba8UnormSrgb);
    assert!(CutDeformer::install(renderer.resources_mut(), &device).is_err());
}
