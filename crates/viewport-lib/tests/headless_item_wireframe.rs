//! A surface item with `settings.wireframe` draws its edge lines, not its
//! triangles, on both the HDR and the direct paths, whether it sits beside
//! instanced items or alone on the per-object path.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::Colour;
use viewport_lib::wgpu;

mod common;
use common::*;

const SIZE: u32 = 128;

/// Render the given boxes (x offset, wireframe, opacity) on black and return
/// the RGBA pixels.
fn render(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    (mesh, material): (MeshId, MaterialId),
    boxes: &[(f32, bool, f32)],
    mode: viewport_lib::PipelineMode,
) -> Vec<u8> {
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame.effects.display.mode = mode;
    let items: Vec<SceneRenderItem> = boxes
        .iter()
        .map(|&(x, wireframe, opacity)| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
            item.material = material;
            item.settings.wireframe = wireframe;
            item.settings.opacity = opacity;
            item
        })
        .collect();
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    renderer.render_offscreen(device, queue, &frame, SIZE, SIZE)
}

/// How many pixels in columns `x0..x1` are not background.
fn coverage(px: &[u8], x0: u32, x1: u32) -> usize {
    (0..SIZE)
        .flat_map(|y| (x0..x1).map(move |x| ((y * SIZE + x) * 4) as usize))
        .filter(|&i| px[i].max(px[i + 1]).max(px[i + 2]) > 20)
        .count()
}

/// The device, a renderer, and the grey box every test draws.
fn setup() -> Option<(
    wgpu::Device,
    wgpu::Queue,
    ViewportRenderer,
    (MeshId, MaterialId),
)> {
    let (device, queue) = headless_device()?;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let grey = upload_material(
        &mut renderer,
        Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8)),
    );
    Some((device, queue, renderer, (mesh, grey)))
}

const MODES: [viewport_lib::PipelineMode; 2] = [
    viewport_lib::PipelineMode::Hdr,
    viewport_lib::PipelineMode::Direct,
];

/// Beside a solid box (so the frame instances), the wireframe box covers far
/// fewer pixels than the solid one: lines, not a filled silhouette.
#[test]
fn wireframe_item_beside_instanced_items_draws_edges() {
    let Some((device, queue, mut renderer, mesh)) = setup() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for mode in MODES {
        let px = render(
            &mut renderer,
            &device,
            &queue,
            mesh,
            &[(-0.8, false, 1.0), (0.8, true, 1.0)],
            mode,
        );
        let solid = coverage(&px, 0, SIZE / 2);
        let wire = coverage(&px, SIZE / 2, SIZE);
        assert!(
            solid > 200,
            "{mode:?}: the solid box draws, covered {solid}"
        );
        assert!(
            wire > 0 && wire * 2 < solid,
            "{mode:?}: the wireframe box draws edge lines, covered {wire} against the solid {solid}"
        );
    }
}

/// Alone, the wireframe box takes the per-object path, and a transparent one
/// still draws opaque edge lines rather than a blended fill.
#[test]
fn wireframe_item_on_the_per_object_path_draws_edges() {
    let Some((device, queue, mut renderer, mesh)) = setup() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for mode in MODES {
        let solid = coverage(
            &render(
                &mut renderer,
                &device,
                &queue,
                mesh,
                &[(0.0, false, 1.0)],
                mode,
            ),
            0,
            SIZE,
        );
        for opacity in [1.0, 0.5] {
            let wire = coverage(
                &render(
                    &mut renderer,
                    &device,
                    &queue,
                    mesh,
                    &[(0.0, true, opacity)],
                    mode,
                ),
                0,
                SIZE,
            );
            assert!(
                wire > 0 && wire * 2 < solid,
                "{mode:?}, opacity {opacity}: the wireframe box draws edge lines, \
                 covered {wire} against the solid {solid}"
            );
        }
    }
}
