//! Lightmap replace/AO blend modes and the set/clear lifecycle.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

/// Lightmap consumption, Replace mode: a mesh with a solid-red radiance lightmap and
/// no direct lights must render red, where the default sky/hemisphere ambient
/// would render blue-grey. Exercises the whole path: the UV1 sidecar (binding
/// 16), the lightmap texture (binding 17), the `lightmap_mode` object field, and
/// the `apply_lightmap` blend in the ambient slot.
#[test]
fn lightmap_replace_mode_recolors_object() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // A solid red radiance texture. UV1 is left empty (set_lightmap pads to
    // zero), so every fragment samples the same red texel.
    let red = vec![[255u8, 0, 0, 255]; 16].concat();
    let radiance = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 4, 4, &red)
        .expect("upload lightmap texture");
    renderer
        .resources_mut()
        .set_lightmap(
            &device,
            mesh_id,
            &[],
            viewport_lib::resources::LightmapData::NonDirectional { radiance },
            viewport_lib::resources::LightmapMode::Replace,
        )
        .expect("set lightmap");

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
    // No direct lights: the object colour is purely its indirect (lightmap) term.
    frame.effects.lighting.lights = vec![];

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material.shading_model = ShadingModel::Pbr;
    item.material.base_colour = [1.0, 1.0, 1.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let (w, h) = (64u32, 64u32);
    let pixels = renderer.render_offscreen(&device, &queue, &frame, w, h);

    let mut best = (0u8, 0u8, 0u8);
    for px in pixels.chunks_exact(4) {
        if px[0] > best.0 {
            best = (px[0], px[1], px[2]);
        }
    }
    assert!(
        best.0 as i32 > best.2 as i32 + 40 && best.0 as i32 > best.1 as i32 + 40,
        "lightmapped object should be red: brightest-red pixel was {best:?}"
    );
}

/// Lightmap AmbientOcclusion mode: a near-black occlusion lightmap must darken the
/// object relative to the same scene with no lightmap. Both renders share a plain
/// sky ambient, so the only difference is the occlusion multiply.
#[test]
fn lightmap_ao_mode_darkens_object() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let build_frame = || {
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
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.shading_model = ShadingModel::Pbr;
        item.material.base_colour = [1.0, 1.0, 1.0];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        frame
    };

    let (w, h) = (64u32, 64u32);
    // Mean over the central region, which the box fills, so the AO multiply on
    // the box's ambient is not diluted by the sky background.
    let center_mean = |pixels: &[u8]| -> f64 {
        let (mut sum, mut n) = (0.0f64, 0u64);
        for y in (h / 4)..(3 * h / 4) {
            for x in (w / 4)..(3 * w / 4) {
                let i = ((y * w + x) * 4) as usize;
                sum += pixels[i] as f64 + pixels[i + 1] as f64 + pixels[i + 2] as f64;
                n += 1;
            }
        }
        sum / n as f64
    };

    let baseline = center_mean(&renderer.render_offscreen(&device, &queue, &build_frame(), w, h));

    // Dark occlusion factor (0.1) multiplies the ambient term.
    let dark = vec![[26u8, 26, 26, 255]; 16].concat();
    let occlusion = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 4, 4, &dark)
        .expect("upload occlusion texture");
    renderer
        .resources_mut()
        .set_lightmap(
            &device,
            mesh_id,
            &[],
            viewport_lib::resources::LightmapData::AmbientOcclusion { occlusion },
            viewport_lib::resources::LightmapMode::AmbientOcclusion,
        )
        .expect("set lightmap");

    let occluded = center_mean(&renderer.render_offscreen(&device, &queue, &build_frame(), w, h));

    assert!(
        occluded < baseline * 0.75,
        "AO lightmap should darken the object: baseline {baseline:.1}, occluded {occluded:.1}"
    );
}

/// The `set_lightmap` / `clear_lightmap` lifecycle: setting, clearing, and
/// re-setting all succeed on a live mesh, and a removed mesh reports a stale
/// handle rather than panicking.
#[test]
fn lightmap_set_clear_lifecycle() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let white = vec![[255u8, 255, 255, 255]; 16].concat();
    let radiance = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 4, 4, &white)
        .expect("upload texture");
    let data = viewport_lib::resources::LightmapData::NonDirectional { radiance };

    let res = renderer.resources_mut();
    assert!(
        res.set_lightmap(
            &device,
            mesh_id,
            &[],
            data,
            viewport_lib::resources::LightmapMode::Add
        )
        .is_ok()
    );
    assert!(res.clear_lightmap(mesh_id).is_ok());
    // clear_lightmap on a mesh with none set is a no-op, not an error.
    assert!(res.clear_lightmap(mesh_id).is_ok());
    assert!(
        res.set_lightmap(
            &device,
            mesh_id,
            &[],
            data,
            viewport_lib::resources::LightmapMode::Replace
        )
        .is_ok()
    );

    // Removing the mesh makes further calls report a stale handle.
    renderer.resources_mut().free_mesh(mesh_id);
    let err = renderer.resources_mut().set_lightmap(
        &device,
        mesh_id,
        &[],
        data,
        viewport_lib::resources::LightmapMode::Replace,
    );
    assert!(matches!(err, Err(ViewportError::StaleHandle { .. })));
}

