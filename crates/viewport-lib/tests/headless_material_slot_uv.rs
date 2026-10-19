//! Per-slot texture transforms on the per-object mesh paths, opaque and OIT.
//!
//! The metallic-roughness and emissive maps take their own `UvTransform` (and
//! `uv_set`) rather than the albedo slot's, on every mesh path.

use viewport_lib::wgpu;
use viewport_lib::{AlphaMode, Colour, TextureSlot, UvTransform};

mod common;
use common::*;

/// Render one emissive box, unlit apart from its emissive map, with `texture`
/// on the emissive slot under the slot's own `transform`. `Blend` draws it
/// through the OIT shader instead of the opaque one. Returns the brightness
/// (sum of RGB) of each pixel over the centre of the frame.
fn render_emissive_box(
    texture: viewport_lib::TextureData,
    transform: UvTransform,
    alpha_mode: AlphaMode,
) -> Option<Vec<f64>> {
    let (device, queue) = headless_device()?;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::cube(1.0))
        .unwrap();
    let emissive_tex = renderer
        .resources_mut()
        .upload_texture(&device, &queue, texture)
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
    // Large enough that the measured centre of the frame is all cube.
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(4.0)).to_cols_array_2d();
    item.material.shading_model = ShadingModel::Pbr;
    item.material.base_colour = Colour::linear_rgb(0.0, 0.0, 0.0);
    item.material.emissive = Colour::linear_rgb(1.0, 1.0, 1.0);
    item.material.emissive_texture_id = Some(emissive_tex);
    item.material.alpha_mode = alpha_mode;
    if alpha_mode == AlphaMode::Blend {
        item.material.base_colour = Colour::linear(0.0, 0.0, 0.0, 0.8);
    }
    item.material = item
        .material
        .with_texture_transform(TextureSlot::Emissive, transform);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let (w, h) = (64u32, 64u32);
    let pixels = renderer.render_offscreen(&device, &queue, &frame, w, h);
    let mut out = Vec::new();
    for y in (h * 3 / 8)..(h * 5 / 8) {
        for x in (w * 3 / 8)..(w * 5 / 8) {
            let i = ((y * w + x) * 4) as usize;
            out.push(pixels[i] as f64 + pixels[i + 1] as f64 + pixels[i + 2] as f64);
        }
    }
    Some(out)
}

/// Mean and standard deviation of a set of pixel brightnesses.
fn mean_and_spread(px: &[f64]) -> (f64, f64) {
    let n = px.len() as f64;
    let mean = px.iter().sum::<f64>() / n;
    let var = px.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Mean brightness of an emissive box whose emissive slot reads a single texel
/// of a half-black, half-white strip, chosen by the slot's own offset.
fn emissive_slot_brightness(offset_u: f32, alpha_mode: AlphaMode) -> Option<f64> {
    // Texels: black, black, white, white. u = 0.125 is black, u = 0.875 white.
    let strip = [
        [0u8, 0, 0, 255],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255],
    ]
    .concat();
    // Collapse the emissive slot's UVs onto one texel; the shared (albedo)
    // transform stays identity, so only a per-slot read can pick the texel.
    let transform = UvTransform {
        offset: [offset_u, 0.5],
        scale: [0.0, 0.0],
        ..UvTransform::default()
    };
    let px = render_emissive_box(
        viewport_lib::TextureData::srgb(4, 1, strip),
        transform,
        alpha_mode,
    )?;
    Some(mean_and_spread(&px).0)
}

/// Mean and spread of an emissive box whose emissive slot tiles a 64x64
/// checker of 8-texel cells 16 times, while the albedo slot stays identity.
/// The face is minified far enough that the right mip is flat grey; mip 0, or
/// a level taken from the albedo slot's gradients, aliases into noise.
fn minified_emissive_checker(alpha_mode: AlphaMode) -> Option<(f64, f64)> {
    let size = 64u32;
    let mut checker = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let v = if ((x / 8) + (y / 8)) % 2 == 0 {
                255u8
            } else {
                0
            };
            checker.extend_from_slice(&[v, v, v, 255]);
        }
    }
    let transform = UvTransform {
        scale: [64.0, 64.0],
        ..UvTransform::default()
    };
    let px = render_emissive_box(
        viewport_lib::TextureData::srgb(size, size, checker),
        transform,
        alpha_mode,
    )?;
    Some(mean_and_spread(&px))
}

/// The emissive map follows its own slot transform on the per-object path: an
/// offset onto the white texel glows, an offset onto the black one does not.
#[test]
fn per_object_emissive_uses_its_own_slot_transform() {
    let Some(white) = emissive_slot_brightness(0.875, AlphaMode::Opaque) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let black = emissive_slot_brightness(0.125, AlphaMode::Opaque).unwrap();
    assert!(
        white > black + 150.0,
        "emissive slot transform ignored: white texel {white:.1}, black texel {black:.1}"
    );
}

/// The same on the per-object OIT path, which has its own copy of the shader.
#[test]
fn per_object_oit_emissive_uses_its_own_slot_transform() {
    let Some(white) = emissive_slot_brightness(0.875, AlphaMode::Blend) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let black = emissive_slot_brightness(0.125, AlphaMode::Blend).unwrap();
    assert!(
        white > black + 100.0,
        "OIT emissive slot transform ignored: white texel {white:.1}, black texel {black:.1}"
    );
}

/// A minified emissive map is filtered at the mip level of its own slot
/// transform on the per-object path, not sampled at full resolution.
#[test]
fn per_object_emissive_minifies_at_its_own_slot_level() {
    let Some((mean, spread)) = minified_emissive_checker(AlphaMode::Opaque) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(
        spread < 15.0 && mean > 100.0 && mean < 700.0,
        "minified emissive checker not filtered: mean {mean:.1}, spread {spread:.1}"
    );
}

/// The same on the per-object OIT path.
#[test]
fn per_object_oit_emissive_minifies_at_its_own_slot_level() {
    let Some((mean, spread)) = minified_emissive_checker(AlphaMode::Blend) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(
        spread < 15.0 && mean > 50.0,
        "minified OIT emissive checker not filtered: mean {mean:.1}, spread {spread:.1}"
    );
}
