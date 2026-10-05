//! `SurfaceContourItem` draws its lines where the CPU extraction puts them.
//!
//! The same field on the same surface is rendered twice: once with the lines
//! extracted on the CPU by `extract_isolines` and drawn as a polyline, once
//! with the contour item finding them per pixel. Every line pixel of each image must have a line
//! pixel of the other close by.

use glam::{Mat4, Vec3};
use viewport_lib::{
    AttributeData, CameraFrame, FrameData, Material, PolylineItem, SceneFrame, SceneRenderItem,
    extract_isolines, isoline_strips, primitives,
};
use viewport_lib_plugins::item_types::surface_contour::{ContourLevels, SurfaceContourItem};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 240;
const H: u32 = 180;
const LEVELS: [f32; 4] = [-0.6, -0.2, 0.2, 0.6];

fn field(p: Vec3) -> f32 {
    (1.3 * p.x).sin() * (1.1 * p.y).cos() + 0.3 * p.x
}

fn frame(surface: SceneRenderItem) -> FrameData {
    // Oblique, so the lines are foreshortened as they would be in use.
    let camera = orbit_camera(Vec3::ZERO, 7.0, 0.4, 0.9);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        SceneFrame::from_surface_items(vec![surface]),
    );
    fd.viewport.show_axes_indicator = false;
    fd.viewport.show_grid = false;
    // Light, so only a line reads as dark.
    fd.viewport.background_colour = Some([0.7, 0.7, 0.7, 1.0].into());
    fd
}

/// Pixels dark enough to be a line: the surface and background are light.
fn line_mask(pixels: &[u8]) -> Vec<bool> {
    pixels.chunks_exact(4).map(|p| p[0] < 60).collect()
}

/// Share of `a`'s line pixels with a line pixel of `b` within `radius`.
fn covered(a: &[bool], b: &[bool], radius: i32) -> f32 {
    let (w, h) = (W as i32, H as i32);
    let mut total = 0;
    let mut near = 0;
    for y in 0..h {
        for x in 0..w {
            if !a[(y * w + x) as usize] {
                continue;
            }
            total += 1;
            let hit = (-radius..=radius).any(|dy| {
                (-radius..=radius).any(|dx| {
                    let (nx, ny) = (x + dx, y + dy);
                    nx >= 0 && ny >= 0 && nx < w && ny < h && b[(ny * w + nx) as usize]
                })
            });
            near += hit as usize;
        }
    }
    assert!(total > 0, "no line pixels");
    near as f32 / total as f32
}

#[test]
fn contour_lines_coincide_with_extracted_isolines() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut mesh = primitives::grid_plane(4.0, 4.0, 64, 64);
    let scalars: Vec<f32> = mesh
        .positions
        .iter()
        .map(|p| field(Vec3::from(*p)))
        .collect();
    mesh.attributes
        .insert("f".to_string(), AttributeData::Vertex(scalars.clone()));
    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &mesh)
        .unwrap();

    let model = Mat4::IDENTITY;
    let mut surface = SceneRenderItem::default();
    surface.mesh_id = mesh_id;
    surface.model = model.to_cols_array_2d();
    surface.material = Material::from_colour([1.0, 1.0, 1.0]);
    surface.settings.unlit = true;

    let mut extracted = frame(surface.clone());
    // The same small lift off the surface the old per-frame extraction used,
    // so the lines do not fight the surface for depth.
    let lines = extract_isolines(&mesh.positions, &mesh.indices, &scalars, &LEVELS, 0.005);
    let (positions, strip_lengths, _) = isoline_strips(&lines);
    let mut polyline = PolylineItem::default();
    polyline.positions = positions;
    polyline.strip_lengths = strip_lengths;
    polyline.default_colour = [0.0, 0.0, 0.0, 1.0].into();
    polyline.line_width = 1.5;
    polyline.model = model.to_cols_array_2d();
    *extracted.scene.items_mut::<PolylineItem>() = vec![polyline];

    let mut shaded = frame(surface);
    *shaded.scene.items_mut::<SurfaceContourItem>() = vec![SurfaceContourItem::new(
        mesh_id,
        model.to_cols_array_2d(),
        "f",
        ContourLevels::Values(LEVELS.to_vec()),
    )];

    let a = line_mask(&h.render(&extracted, W, H));
    let b = line_mask(&h.render(&shaded, W, H));
    let shaded_near = covered(&b, &a, 2);
    let extracted_near = covered(&a, &b, 2);
    assert!(
        shaded_near >= 0.97 && extracted_near >= 0.97,
        "shaded pixels near an extracted line: {shaded_near}, extracted near a shaded one: \
         {extracted_near}"
    );
}
