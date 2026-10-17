//! Per-slot texture transforms on the per-object mesh path.
//!
//! The metallic-roughness and emissive maps take their own `UvTransform` (and
//! `uv_set`) rather than the albedo slot's, on every mesh path.

use viewport_lib::wgpu;
use viewport_lib::{Colour, TextureSlot, UvTransform};

mod common;
use common::*;

/// Render one emissive box whose emissive slot reads a single texel of a
/// half-black, half-white strip, chosen by the slot's own offset. Returns the
/// mean brightness over the centre of the frame.
fn emissive_slot_brightness(offset_u: f32) -> Option<f64> {
    let (device, queue) = headless_device()?;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // Texels: black, black, white, white. u = 0.125 is black, u = 0.875 white.
    let strip = [
        [0u8, 0, 0, 255],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255],
    ]
    .concat();
    let emissive_tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            viewport_lib::TextureData::srgb(4, 1, strip),
        )
        .expect("upload emissive texture");

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.lighting.lights = vec![];
    frame.effects.lighting.hemisphere_intensity = 0.0;
    frame.effects.display.exposure = viewport_lib::ExposureSettings::manual(0.0);

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material.shading_model = ShadingModel::Pbr;
    item.material.base_colour = Colour::linear_rgb(0.0, 0.0, 0.0);
    item.material.emissive = Colour::linear_rgb(1.0, 1.0, 1.0);
    item.material.emissive_texture_id = Some(emissive_tex);
    // Collapse the emissive slot's UVs onto one texel; the shared (albedo)
    // transform stays identity, so only a per-slot read can pick the texel.
    item.material = item.material.with_texture_transform(
        TextureSlot::Emissive,
        UvTransform {
            offset: [offset_u, 0.5],
            scale: [0.0, 0.0],
            ..UvTransform::default()
        },
    );
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let (w, h) = (64u32, 64u32);
    let pixels = renderer.render_offscreen(&device, &queue, &frame, w, h);
    let (mut sum, mut n) = (0.0f64, 0u64);
    for y in (h * 3 / 8)..(h * 5 / 8) {
        for x in (w * 3 / 8)..(w * 5 / 8) {
            let i = ((y * w + x) * 4) as usize;
            sum += pixels[i] as f64 + pixels[i + 1] as f64 + pixels[i + 2] as f64;
            n += 1;
        }
    }
    Some(sum / n as f64)
}

/// The emissive map follows its own slot transform on the per-object path: an
/// offset onto the white texel glows, an offset onto the black one does not.
#[test]
fn per_object_emissive_uses_its_own_slot_transform() {
    let Some(white) = emissive_slot_brightness(0.875) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let black = emissive_slot_brightness(0.125).unwrap();
    assert!(
        white > black + 150.0,
        "emissive slot transform ignored: white texel {white:.1}, black texel {black:.1}"
    );
}
