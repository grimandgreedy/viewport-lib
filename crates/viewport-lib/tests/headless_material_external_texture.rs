//! `register_texture_view`: an external GPU texture used as a material texture.
//!
//! A colour texture already on the device (a dma-buf import, a video surface, or
//! here a stand-in filled with a known colour) is registered as a material
//! `TextureId` and set on a mesh's `Material::texture_id`. Two things are checked:
//! the register / re-point / free mechanics (external entries are distinct from
//! CPU-uploaded ones), and that a mesh textured with the external view actually
//! samples it (a blue source renders blue, not the fallback).
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

/// Create an sRGB texture of `size` filled with `rgba` and return it with a view.
fn solid_srgb_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
    rgba: [u8; 4],
) -> (wgpu::Texture, wgpu::TextureView) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let extent = wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("external_material_source"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let pixels: Vec<u8> = std::iter::repeat(rgba)
        .take((size * size) as usize)
        .flatten()
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        extent,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// register / re-point / free semantics, and the external-vs-owned distinction.
#[test]
fn register_update_free_semantics() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let res = renderer.resources_mut();

    let (_t1, v1) = solid_srgb_texture(&device, &queue, 4, [10, 20, 30, 255]);
    let (_t2, v2) = solid_srgb_texture(&device, &queue, 4, [40, 50, 60, 255]);

    // Register an external view, then re-point it in place.
    let ext = res.register_texture_view(&device, &v1, 4, 4);
    assert!(
        res.update_texture_view(&device, ext, &v2, 4, 4),
        "update_texture_view should re-point an external id"
    );

    // An owned (CPU-uploaded) texture is rejected by the external update path.
    let owned = res
        .upload_texture(&device, &queue, 1, 1, &[255u8, 255, 255, 255])
        .unwrap();
    assert!(
        !res.update_texture_view(&device, owned, &v1, 4, 4),
        "update_texture_view must reject an owned texture id"
    );

    // A freed external id no longer resolves.
    assert!(res.free_texture(ext), "external id should free");
    assert!(
        !res.update_texture_view(&device, ext, &v1, 4, 4),
        "update_texture_view must reject a freed id"
    );
}

/// A mesh textured with an external blue view renders blue at its centre, proving
/// the material path samples the external view (not the white fallback).
#[test]
fn external_view_is_sampled_by_a_material() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Pure blue source: unmistakable through lighting and tone-map.
    let (_src, view) = solid_srgb_texture(&device, &queue, 4, [0, 0, 255, 255]);
    let tex = renderer
        .resources_mut()
        .register_texture_view(&device, &view, 4, 4);

    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = FrameData::default();
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // Camera-facing light so the front face is lit.
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.0, 0.0, 1.0],
    };
    light.intensity = 3.0;
    frame.effects.lighting.lights = vec![light];

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material.base_colour = [1.0, 1.0, 1.0].into();
    item.material.texture_id = Some(tex);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let cam = Camera::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];

    let px = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let i = ((32 * 64 + 32) * 4) as usize;
    let (r, g, b) = (px[i] as i32, px[i + 1] as i32, px[i + 2] as i32);
    assert!(
        b > r + 20 && b > g + 20,
        "centre should be blue from the external texture, got rgb ({r}, {g}, {b}) \
         (a white fallback would read r ~= g ~= b)"
    );
}
