//! Per-material `SamplerKey`: a material's wrap mode changes how a texture is
//! sampled outside `[0, 1]`. A quad whose UVs run 0..2 tiles the texture under
//! the default `Repeat` sampler; a `ClampToEdge` sampler stretches the edge
//! texel instead, so the clamped render has visibly more of the right-edge
//! colour. Proves the per-object path binds the material's own sampler.

mod common;
use common::*;

use viewport_lib::{SamplerKey, TextureSlot};

/// Build a flat quad in the XY plane (facing +Z) whose U coordinate runs 0..2
/// across its width, so the horizontal wrap mode is exercised.
fn tiling_quad() -> MeshData {
    let mut q = MeshData::default();
    q.positions = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    q.normals = vec![[0.0, 0.0, 1.0]; 4];
    // U spans 0..2 left-to-right; V is 0..1.
    q.uvs = Some(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]);
    q.indices = vec![0, 1, 2, 0, 2, 3];
    q
}

/// Count pixels that read clearly blue (the right-edge texel colour).
fn blue_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[2] > 120 && p[2] as i16 - p[0] as i16 > 60)
        .count()
}

#[test]
fn material_sampler_wrap_mode_changes_tiling() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &tiling_quad())
        .unwrap();

    // 2x1 texture: left texel red, right texel blue.
    let tex = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            viewport_lib::TextureData::srgb(
                2,
                1,
                [
                    255, 0, 0, 255, /* red */ 0, 0, 255, 255, /* blue */
                ]
                .to_vec(),
            ),
        )
        .unwrap();

    let render_with = |renderer: &mut ViewportRenderer, sampler: Option<SamplerKey>| -> Vec<u8> {
        let mut frame = FrameData::default();
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0].into());
        let cam = Camera::default();
        frame.camera.render_camera = {
            let mut rc = RenderCamera::from_camera(&cam);
            rc.aspect = 1.0;
            rc
        };
        frame.camera.viewport_size = [96.0, 96.0];

        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.base_colour = [1.0, 1.0, 1.0].into();
        item.material.texture_id = Some(tex);
        item.settings.unlit = true;
        if let Some(key) = sampler {
            item.material = item.material.with_sampler(TextureSlot::Albedo, key);
        }
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        renderer.render_offscreen(&device, &queue, &frame, 96, 96)
    };

    // Default (Repeat): the texture tiles twice across the quad, so red and blue
    // each cover roughly half.
    let repeat_px = render_with(&mut renderer, None);
    let repeat_blue = blue_pixels(&repeat_px);

    // ClampToEdge: everything past U=1 stretches the blue edge texel, so the
    // right half of the quad is solid blue on top of the tiled left half.
    let clamp_px = render_with(&mut renderer, Some(SamplerKey::clamp()));
    let clamp_blue = blue_pixels(&clamp_px);

    assert!(
        repeat_blue > 0,
        "sanity: the repeat render should show some blue ({repeat_blue})"
    );
    assert!(
        clamp_blue > repeat_blue + repeat_blue / 4,
        "a clamp sampler must stretch the blue edge over more of the quad than \
         repeat tiling does (clamp {clamp_blue} vs repeat {repeat_blue})"
    );
}
