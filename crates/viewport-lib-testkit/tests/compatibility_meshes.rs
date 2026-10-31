//! Meshes on the compatibility tier: a lit, a textured and a colourmapped mesh drawn
//! through the basic mesh path on WebGL2 limits, with goldens of their own.
//!
//! The compatibility tier's frame does not match the full tier's (no shadows, IBL or
//! PBR), so these references live in `tests/snapshots/<backend>-compatibility/` and
//! are blessed the same way as the catalogue's:
//!
//!     BLESS=1 cargo test --release --test compatibility_meshes

use std::path::Path;
use viewport_lib::{
    AttributeData, AttributeKind, AttributeRef, BuiltinColourmap, Camera, Colour, FrameData,
    Material, MaterialId, RenderCamera, SceneRenderItem, SurfaceSubmission, Tier,
};
use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::{DeviceProfile, Harness, catalogue, frame_for};

const W: u32 = 256;
const H: u32 = 192;
const BACKGROUND: [f32; 4] = [0.05, 0.05, 0.08, 1.0];

fn forward_harness(label: &'static str) -> Option<Harness> {
    let harness = Harness::with_profile(&DeviceProfile::downlevel_webgl2(label))?;
    assert_eq!(harness.renderer.tier(), Tier::Compatibility);
    Some(harness)
}

fn scene_camera() -> Camera {
    let mut camera = Camera::default();
    camera.set_center(glam::Vec3::ZERO);
    camera.set_distance(6.0);
    camera.set_orientation(glam::Quat::from_rotation_z(0.5) * glam::Quat::from_rotation_x(1.1));
    camera.set_aspect_ratio(W as f32, H as f32);
    camera
}

fn frame(items: Vec<SceneRenderItem>) -> FrameData {
    let camera = scene_camera();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&camera);
    frame.camera.viewport_size = [W as f32, H as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::from_linear_array(BACKGROUND));
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    frame
}

fn item(mesh_id: viewport_lib::MeshId, material: MaterialId, at: [f32; 3]) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = material;
    item.model = glam::Mat4::from_translation(glam::Vec3::from(at)).into();
    item
}

fn material(harness: &mut Harness, m: Material) -> MaterialId {
    harness
        .renderer
        .resources_mut()
        .upload_material(m)
        .expect("upload material")
}

fn render(harness: &mut Harness, frame: &FrameData) -> Vec<u8> {
    let _ = harness.render(frame, W, H);
    harness.render(frame, W, H)
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// Check against `<adapter>-compatibility/<name>.png` in the mode the
/// environment names (see `golden::GoldenMode::from_env`).
fn check_golden(harness: &Harness, name: &str, pixels: Vec<u8>) {
    let snapshots = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
    let dir = snapshots.join(format!("{}-compatibility", harness.backend_dir_name()));
    let mode = GoldenMode::from_env();
    if mode != GoldenMode::Bless {
        // The driver stamp lives with the adapter's main reference set.
        let stamp_dir = snapshots.join(harness.backend_dir_name());
        if let Err(e) = golden::check_driver_stamp(&stamp_dir, harness.driver_info().unwrap_or(""))
        {
            panic!("{e}");
        }
    }
    let tolerances = golden::load_tolerances(&dir.join("tolerances.txt"));
    let tolerance = tolerances.get(name).copied().unwrap_or(Tolerance::EXACT);
    let actual = RgbaImage::from_raw(W, H, pixels).expect("frame -> image");
    match golden::check(&dir, name, &actual, tolerance, mode) {
        Outcome::Blessed => eprintln!("blessed {name} in {}", dir.display()),
        Outcome::Rendered { path } => eprintln!("render only: {} (not compared)", path.display()),
        Outcome::Match => {}
        Outcome::Missing { .. } => panic!("{}", golden::missing_message(name, &dir)),
        Outcome::Mismatch { comparison, .. } => panic!(
            "{name}: max_channel_delta={} pixels_over={}/{}",
            comparison.max_channel_delta, comparison.pixels_over, comparison.total_pixels
        ),
    }
}

/// A lit sphere, an opaque cube and a smaller translucent cube between it
/// and the camera, through which the opaque one shows.
#[test]
fn compatibility_tier_draws_lit_meshes() {
    let Some(mut harness) = forward_harness("compatibility-lit") else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let sphere = viewport_lib::geometry::primitives::sphere(1.0, 48, 24);
    let cube = viewport_lib::geometry::primitives::cube(1.2);
    let resources = harness.renderer.resources_mut();
    let sphere = resources
        .upload_mesh_data(&harness.device, &sphere)
        .unwrap();
    let cube = resources.upload_mesh_data(&harness.device, &cube).unwrap();
    let red = material(
        &mut harness,
        Material::from_colour(Colour::linear_rgb(0.8, 0.2, 0.15)),
    );
    let grey = material(
        &mut harness,
        Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.6)),
    );
    let opaque_at = glam::Vec3::new(1.4, 0.0, 0.0);
    let towards_eye = (scene_camera().eye_position() - opaque_at).normalize();
    let mut glass = item(cube, grey, (opaque_at + towards_eye * 1.3).to_array());
    glass.model = (glass.model.to_mat4()
        * glam::Mat4::from_scale(glam::Vec3::splat(0.6)))
    .into();
    glass.settings.opacity = 0.4;
    glass.settings.colour = Some(Colour::linear_rgb(0.2, 0.4, 0.9));
    let frame = frame(vec![
        item(sphere, red, [-1.2, 0.0, 0.0]),
        item(cube, grey, opaque_at.to_array()),
        glass,
    ]);
    let pixels = render(&mut harness, &frame);

    let background = pixel(&pixels, 2, 2);
    let drawn = pixels
        .chunks_exact(4)
        .filter(|p| p[..3] != background[..3])
        .count();
    assert!(
        drawn > (W * H / 10) as usize,
        "the meshes must cover part of the frame: {drawn} pixels"
    );
    let stats = harness.renderer.last_frame_stats();
    assert_eq!(stats.items_skipped_by_tier, 0, "surfaces draw on this tier");
    check_golden(&harness, "lit_meshes", pixels);
}

/// A cube with a two-colour checker albedo, 8 texels to a cell, shows both
/// colours.
#[test]
fn compatibility_tier_draws_a_textured_mesh() {
    let Some(mut harness) = forward_harness("compatibility-textured") else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut rgba = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            let on = (x / 8 + y / 8) % 2 == 0;
            rgba.extend_from_slice(if on {
                &[230, 200, 40, 255]
            } else {
                &[30, 90, 200, 255]
            });
        }
    }
    let texture = harness
        .renderer
        .resources_mut()
        .upload_texture(
            &harness.device,
            &harness.queue,
            viewport_lib::TextureData::srgb(64, 64, rgba),
        )
        .expect("textures upload on the compatibility tier");
    let cube = viewport_lib::geometry::primitives::cube(2.0);
    let cube = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &cube)
        .unwrap();
    let mut m = Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0));
    m.texture_id = Some(texture);
    let m = material(&mut harness, m);
    let frame = frame(vec![item(cube, m, [0.0, 0.0, 0.0])]);
    let pixels = render(&mut harness, &frame);

    let (mut yellow, mut blue) = (0, 0);
    for p in pixels.chunks_exact(4) {
        let [r, g, b] = [p[0] as i32, p[1] as i32, p[2] as i32];
        if r > b + 40 && g > b + 20 {
            yellow += 1;
        } else if b > r + 40 {
            blue += 1;
        }
    }
    assert!(
        yellow > 200 && blue > 200,
        "both checker colours must show: yellow {yellow}, blue {blue}"
    );
    check_golden(&harness, "textured_mesh", pixels);
}

/// A sphere coloured by its height through viridis: the top reads yellow and
/// the bottom purple.
#[test]
fn compatibility_tier_draws_a_colourmapped_mesh() {
    let Some(mut harness) = forward_harness("compatibility-colourmap") else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut sphere = viewport_lib::geometry::primitives::sphere(1.5, 48, 24);
    let height: Vec<f32> = sphere.positions.iter().map(|p| p[2]).collect();
    sphere
        .attributes
        .insert("height".to_string(), AttributeData::Vertex(height));
    let resources = harness.renderer.resources_mut();
    let sphere = resources
        .upload_mesh_data(&harness.device, &sphere)
        .unwrap();
    let viridis = resources.builtin_colourmap_id(BuiltinColourmap::Viridis);
    let white = material(
        &mut harness,
        Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0)),
    );
    let mut it = item(sphere, white, [0.0, 0.0, 0.0]);
    it.active_attribute = Some(AttributeRef {
        name: "height".to_string(),
        kind: AttributeKind::Vertex,
    });
    it.colourmap_id = Some(viridis);
    it.settings.unlit = true;
    let mut frame = frame(vec![it]);
    // Look from the side so the top and bottom are both in view.
    let mut camera = Camera::default();
    camera.set_center(glam::Vec3::ZERO);
    camera.set_distance(6.0);
    camera.set_orientation(glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    camera.set_aspect_ratio(W as f32, H as f32);
    frame.camera.render_camera = RenderCamera::from_camera(&camera);
    let pixels = render(&mut harness, &frame);

    // Z-up: the top of the sphere is near the top of the frame.
    let top = pixel(&pixels, W / 2, H / 2 - 50);
    let bottom = pixel(&pixels, W / 2, H / 2 + 50);
    assert!(
        top[0] > 150 && top[1] > 150 && top[2] < 120,
        "the top must read yellow, got {top:?}"
    );
    assert!(
        bottom[2] > bottom[1] && bottom[0] > bottom[1],
        "the bottom must read purple, got {bottom:?}"
    );
    check_golden(&harness, "colourmapped_mesh", pixels);
}

/// Replacing a mesh works on the tier, in place and with a new topology, and
/// the replacement draws.
#[test]
fn compatibility_tier_replaces_a_mesh() {
    let Some(mut harness) = forward_harness("compatibility-replace") else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let cube = viewport_lib::geometry::primitives::cube(1.0);
    let resources = harness.renderer.resources_mut();
    let mesh = resources.upload_mesh_data(&harness.device, &cube).unwrap();
    resources
        .replace_mesh_data(&harness.device, &harness.queue, mesh, &cube)
        .expect("an in-place replace on the compatibility tier");
    let sphere = viewport_lib::geometry::primitives::sphere(1.5, 32, 16);
    resources
        .replace_mesh_data(&harness.device, &harness.queue, mesh, &sphere)
        .expect("a replace with a new topology on the compatibility tier");
    let white = material(
        &mut harness,
        Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0)),
    );
    let frame = frame(vec![item(mesh, white, [0.0, 0.0, 0.0])]);
    let pixels = render(&mut harness, &frame);
    let centre = pixel(&pixels, W / 2, H / 2);
    let corner = pixel(&pixels, 2, 2);
    assert_ne!(centre, corner, "the replaced mesh must draw");
}

/// The compatibility tier has no GPU pick pass, so a pick or a box select of any
/// backend finds the surface on the CPU, down to the face under the cursor.
#[test]
fn compatibility_tier_picks_a_surface_on_the_cpu() {
    let Some(mut harness) = forward_harness("compatibility-pick") else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness.renderer.set_cpu_pick_cache(true);
    let sphere = viewport_lib::geometry::primitives::sphere(1.5, 32, 16);
    let mesh = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &sphere)
        .unwrap();
    let white = material(
        &mut harness,
        Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0)),
    );
    let mut it = item(mesh, white, [0.0, 0.0, 0.0]);
    it.settings.pick_id = viewport_lib::PickId(7);
    let frame = frame(vec![it]);
    let _ = render(&mut harness, &frame);
    let centre = glam::Vec2::new(W as f32 / 2.0, H as f32 / 2.0);
    for backend in [
        viewport_lib::PickBackend::Auto,
        viewport_lib::PickBackend::Gpu,
        viewport_lib::PickBackend::Cpu,
    ] {
        let hit = harness.renderer.pick_object(
            backend,
            centre,
            &frame,
            &harness.device,
            &harness.queue,
            viewport_lib::PickMask::OBJECT,
        );
        assert_eq!(hit.map(|h| h.id), Some(7), "{backend:?}");
        // A box around the centre selects it too, and a face pick names a
        // triangle of it.
        let rect = harness.renderer.pick_rect_objects(
            backend,
            centre - glam::Vec2::splat(20.0),
            centre + glam::Vec2::splat(20.0),
            &frame,
            &harness.device,
            &harness.queue,
            viewport_lib::PickMask::OBJECT,
        );
        assert_eq!(rect.objects, vec![7], "{backend:?}");
        let face = harness.renderer.pick_object(
            backend,
            centre,
            &frame,
            &harness.device,
            &harness.queue,
            viewport_lib::PickMask::FACE,
        );
        assert!(
            matches!(
                face.and_then(|h| h.sub_object),
                Some(viewport_lib::SubObjectRef::Face(_))
            ),
            "{backend:?}: {face:?}"
        );
    }
}

/// A selected sphere is outlined in the frame's outline colour, and only when
/// the frame asks for outlines.
#[test]
fn compatibility_tier_outlines_a_selected_mesh() {
    let Some(mut harness) = forward_harness("compatibility-outline") else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let sphere = viewport_lib::geometry::primitives::sphere(1.0, 48, 24);
    let cube = viewport_lib::geometry::primitives::cube(1.2);
    let resources = harness.renderer.resources_mut();
    let sphere = resources
        .upload_mesh_data(&harness.device, &sphere)
        .unwrap();
    let cube = resources.upload_mesh_data(&harness.device, &cube).unwrap();
    let grey = material(
        &mut harness,
        Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.6)),
    );
    let mut selected = item(sphere, grey, [-1.2, 0.0, 0.0]);
    selected.settings.selected = true;
    let mut frame = frame(vec![selected, item(cube, grey, [1.4, 0.0, 0.0])]);
    frame.interaction.outline_colour = Colour::linear_rgb(1.0, 0.5, 0.0);
    frame.interaction.outline_width_px = 3.0;

    frame.interaction.outline_selected = false;
    let plain = render(&mut harness, &frame);
    frame.interaction.outline_selected = true;
    let pixels = render(&mut harness, &frame);

    let orange = |p: &[u8]| p[0] as i32 > p[2] as i32 + 120 && p[1] as i32 > p[2] as i32 + 40;
    let ring = pixels
        .chunks_exact(4)
        .zip(plain.chunks_exact(4))
        .filter(|(a, b)| orange(a) && !orange(b))
        .count();
    assert!(
        ring > 150,
        "the sphere must be outlined: {ring} ring pixels"
    );
    // Only the sphere is selected: nothing changes on the cube's side.
    let right_half_changed = pixels
        .chunks_exact(4)
        .zip(plain.chunks_exact(4))
        .enumerate()
        .filter(|(i, (a, b))| (*i as u32 % W) > W * 3 / 4 && a != b)
        .count();
    assert_eq!(right_half_changed, 0, "the unselected cube is not outlined");
    check_golden(&harness, "outlined_mesh", pixels);
}

/// A wireframe surface draws its mesh's edges in flat grey through the basic
/// mesh path, whether the item asks for it or the frame's global toggle does,
/// and a translucent wireframe item draws its lines opaque, as on the full
/// tier.
#[test]
fn compatibility_tier_draws_mesh_wireframes() {
    let Some(mut harness) = forward_harness("compatibility-mesh-wireframe") else {
        return;
    };
    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == "primitives_trio")
        .expect("primitives_trio is in the catalogue");
    let built = harness.build_scene(&scene);
    let camera = &scene.cameras[0].camera;
    let draw = |harness: &mut Harness, edit: &dyn Fn(&mut FrameData)| {
        let mut frame = frame_for(&built, camera, [W as f32, H as f32]);
        edit(&mut frame);
        let _ = render(harness, &frame);
        render(harness, &frame)
    };
    let surfaces = |frame: &mut FrameData, f: &dyn Fn(&mut SceneRenderItem)| {
        let SurfaceSubmission::Flat(items) = &frame.scene.surfaces else {
            panic!("the catalogue submits flat surfaces");
        };
        let mut items = items.to_vec();
        items.iter_mut().for_each(f);
        frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    };

    let empty = draw(&mut harness, &|f| {
        f.scene.surfaces = SurfaceSubmission::Flat(Vec::new().into())
    });
    let solid = draw(&mut harness, &|_| {});
    let per_item = draw(&mut harness, &|f| {
        surfaces(f, &|i| {
            i.settings.wireframe = true;
            i.settings.opacity = 0.4;
        })
    });
    let global = draw(&mut harness, &|f| f.viewport.wireframe_mode = true);

    let drawn = |pixels: &[u8]| -> Vec<[u8; 3]> {
        pixels
            .chunks_exact(4)
            .zip(empty.chunks_exact(4))
            .filter(|(a, b)| a[..3] != b[..3])
            .map(|(a, _)| [a[0], a[1], a[2]])
            .collect()
    };
    let solid_px = drawn(&solid).len();
    for (label, pixels) in [("per item", &per_item), ("global", &global)] {
        let px = drawn(pixels);
        assert!(
            px.len() > 200 && px.len() < solid_px,
            "{label}: edges draw as lines, {} pixels vs {solid_px} solid",
            px.len()
        );
        let grey = px
            .iter()
            .filter(|p| p[0].abs_diff(p[1]) < 8 && p[1].abs_diff(p[2]) < 8 && p[0] > 150)
            .count();
        assert!(
            grey * 10 > px.len() * 8,
            "{label}: the lines are opaque flat grey, {grey} of {}",
            px.len()
        );
    }
}
