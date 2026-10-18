//! Materials uploaded once and edited by handle draw what a renderer built
//! fresh with the new values draws, and only edits that move the batch key or
//! the instance record rebuild the batches.

use crate::Colour;
use crate::renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission};
use crate::{Camera, Material, MaterialId, ViewportRenderer};

const SIZE: u32 = 96;
const FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rgba8UnormSrgb;

fn device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
    let instance = crate::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &crate::gpu::RequestAdapterOptions {
            power_preference: crate::gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            #[cfg(wgpu30)]
            apply_limit_buckets: false,
        },
    ))
    .ok()?;
    pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
}

fn cube(renderer: &mut ViewportRenderer, device: &crate::gpu::Device) -> crate::MeshId {
    renderer
        .resources_mut()
        .upload_mesh_data(device, &crate::geometry::primitives::cube(1.0))
        .unwrap()
}

/// Three cubes sharing `material`, one batch. The third carries a NaN colour,
/// so it reads a shading variant of the material's block rather than the
/// block itself.
fn frame(mesh: crate::MeshId, material: MaterialId, colour: Option<Colour>) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.1, 0.1, 0.12, 1.0));
    let items: Vec<SceneRenderItem> = (0..3)
        .map(|i| {
            let mut item = SceneRenderItem::new(mesh, material);
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(i as f32 * 1.3 - 1.3, 0.0, 0.0))
                    .to_cols_array_2d();
            item.settings.colour = colour;
            if i == 2 {
                item.nan_colour = Some(Colour::linear_rgb(0.0, 1.0, 0.0));
            }
            item
        })
        .collect();
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    frame.scene.generation = 1;
    frame
}

/// The frame a fresh renderer draws with `material` uploaded and `colour` on
/// every item.
fn fresh(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    material: Option<Material>,
    colour: Option<Colour>,
) -> Vec<u8> {
    let mut renderer = ViewportRenderer::new(device, FORMAT);
    let mesh = cube(&mut renderer, device);
    let id = match material {
        Some(m) => renderer.resources_mut().upload_material(m).unwrap(),
        None => MaterialId::DEFAULT,
    };
    renderer.render_offscreen(device, queue, &frame(mesh, id, colour), SIZE, SIZE)
}

fn base() -> Material {
    let mut m = Material::pbr(Colour::linear_rgb(0.8, 0.5, 0.3), 0.0, 0.6);
    m.emissive = Colour::linear_rgb(0.0, 0.0, 0.0);
    m
}

#[test]
fn a_scalar_replace_redraws_without_a_rebuild() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    let mesh = cube(&mut renderer, &device);
    let id = renderer.resources_mut().upload_material(base()).unwrap();
    let f = frame(mesh, id, None);
    let _ = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);
    let before = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);

    let mut glowing = base();
    glowing.emissive = Colour::linear_rgb(0.9, 0.2, 0.1);
    glowing.roughness = 0.2;
    let epoch = renderer.resources.resource_view_epoch;
    renderer
        .resources_mut()
        .replace_material(id, glowing)
        .unwrap();
    assert_eq!(
        renderer.resources.resource_view_epoch, epoch,
        "an edit the block alone reads must not force a rebuild"
    );
    let after = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);
    assert!(after != before, "the edit drew nothing new");
    assert!(
        after == fresh(&device, &queue, Some(glowing), None),
        "the edited material, and its variant, differ from a fresh upload"
    );
}

#[test]
fn a_colour_or_texture_replace_rebuilds() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    let mesh = cube(&mut renderer, &device);
    let id = renderer.resources_mut().upload_material(base()).unwrap();
    let f = frame(mesh, id, None);
    let _ = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);

    let mut blue = base();
    blue.base_colour = Colour::linear_rgb(0.1, 0.2, 0.9);
    let epoch = renderer.resources.resource_view_epoch;
    renderer.resources_mut().replace_material(id, blue).unwrap();
    assert_ne!(
        renderer.resources.resource_view_epoch, epoch,
        "base colour rides the instance record, so the batches rebuild"
    );
    let after = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);
    assert!(after == fresh(&device, &queue, Some(blue), None));

    let mut textured = blue;
    textured.texture_id = Some(crate::TextureId::INVALID);
    let epoch = renderer.resources.resource_view_epoch;
    renderer
        .resources_mut()
        .replace_material(id, textured)
        .unwrap();
    assert_ne!(renderer.resources.resource_view_epoch, epoch);
}

#[test]
fn a_freed_material_draws_the_default_and_never_aliases() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    let mesh = cube(&mut renderer, &device);
    let id = renderer.resources_mut().upload_material(base()).unwrap();
    let f = frame(mesh, id, None);
    let _ = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);

    assert!(renderer.resources_mut().free_material(id));
    assert!(!renderer.resources_mut().free_material(id), "already freed");
    assert!(
        !renderer.resources_mut().free_material(MaterialId::DEFAULT),
        "the default is never freed"
    );
    // A later upload reuses the slot; the stale handle must not draw it.
    let mut green = base();
    green.base_colour = Colour::linear_rgb(0.1, 0.9, 0.1);
    let reused = renderer.resources_mut().upload_material(green).unwrap();
    assert_eq!(reused.index(), id.index());
    assert!(renderer.resources().material(id).is_none());

    let after = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);
    assert!(after == fresh(&device, &queue, None, None));
}

#[test]
fn an_item_colour_draws_like_a_material_colour() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let red = Colour::linear_rgb(0.9, 0.1, 0.1);
    let mut red_material = base();
    red_material.base_colour = red;
    assert!(
        fresh(&device, &queue, Some(base()), Some(red))
            == fresh(&device, &queue, Some(red_material), None),
        "an item colour must replace the material's base colour exactly"
    );
}

/// A foreground item interning on frames where the instanced batch cache
/// holds must not take over the block the cached instances name.
#[test]
fn foreground_items_on_cache_hit_frames_keep_the_scene_materials() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    let mesh = cube(&mut renderer, &device);
    let mut scene = base();
    scene.emissive = Colour::linear_rgb(0.0, 0.4, 0.9);
    let scene_id = renderer.resources_mut().upload_material(scene).unwrap();
    let mut gizmo = base();
    gizmo.emissive = Colour::linear_rgb(0.9, 0.9, 0.0);
    gizmo.roughness = 0.1;
    let gizmo_id = renderer.resources_mut().upload_material(gizmo).unwrap();

    let mut f = frame(mesh, scene_id, None);
    let mut fg = SceneRenderItem::new(mesh, gizmo_id);
    fg.model = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::splat(0.3),
        glam::Quat::IDENTITY,
        glam::Vec3::new(0.0, 0.0, 1.5),
    )
    .to_cols_array_2d();
    f.scene.foreground_items = vec![fg];

    let first = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);
    for _ in 0..3 {
        let next = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);
        assert!(next == first, "a cache-hit frame drew differently");
    }
}

struct Flat;
impl crate::MaterialPlugin for Flat {
    fn name(&self) -> &'static str {
        "store_flat"
    }
    fn texture_count(&self) -> u32 {
        1
    }
    fn wgsl_body(&self) -> String {
        "fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> { return vec3<f32>(0.0); }\n\
         fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {\n\
         \x20   return params.colour * textureSampleLevel(material_texture_0, material_sampler, surf.uv, 0.0).rgb;\n\
         }\n"
            .to_string()
    }
    fn param_layout(&self) -> Vec<crate::ParamDesc> {
        vec![crate::ParamDesc::colour("colour", Colour::WHITE)]
    }
}

/// A plugin parameter edit is one block write and draws what a fresh upload
/// of the edited material draws; a plugin texture edit rebuilds.
#[test]
fn a_plugin_parameter_replace_is_a_block_write() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let draw = |edit: Option<Colour>| {
        let mut renderer = ViewportRenderer::new(&device, FORMAT);
        let mesh = cube(&mut renderer, &device);
        let plugin = renderer
            .resources_mut()
            .register_material_plugin(&device, &Flat)
            .unwrap();
        let m = |c| {
            Material::default()
                .with_plugin(plugin)
                .with_param("colour", c)
        };
        let first = edit.map_or(Colour::linear_rgb(0.9, 0.2, 0.1), |_| {
            Colour::linear_rgb(0.1, 0.2, 0.9)
        });
        let id = renderer.resources_mut().upload_material(m(first)).unwrap();
        let f = frame(mesh, id, None);
        let _ = renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE);
        if let Some(c) = edit {
            let epoch = renderer.resources.resource_view_epoch;
            renderer.resources_mut().replace_material(id, m(c)).unwrap();
            assert_eq!(
                renderer.resources.resource_view_epoch, epoch,
                "a parameter edit must not force a rebuild"
            );
            let tex = renderer
                .resources_mut()
                .upload_texture(
                    &device,
                    &queue,
                    crate::TextureData::srgb(1, 1, vec![255; 4]),
                )
                .unwrap();
            let mut textured = m(c);
            textured.plugin_textures[0] = Some(tex);
            let epoch = renderer.resources.resource_view_epoch;
            renderer
                .resources_mut()
                .replace_material(id, textured)
                .unwrap();
            assert_ne!(
                renderer.resources.resource_view_epoch, epoch,
                "a plugin texture edit changes the batch key"
            );
        }
        renderer.render_offscreen(&device, &queue, &f, SIZE, SIZE)
    };
    let fresh = draw(None);
    let edited = draw(Some(Colour::linear_rgb(0.9, 0.2, 0.1)));
    assert!(
        edited == fresh,
        "the edited block draws what a fresh upload does"
    );
}
