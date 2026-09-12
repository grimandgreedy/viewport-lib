//! Parity between the external effect copies and the built-ins, and the
//! stage-stack behaviour.
//!
//! The copies share the built-ins' shaders, formats, sizes, and uniform
//! derivations, so on the same device the composite output must be
//! pixel-identical whichever implementation fills the slot. Each test also
//! guards that the effect actually changed the image (a trivially blank
//! effect would pass parity vacuously).

use viewport_lib::wgpu;
use viewport_lib::{
    Camera, LightKind, LightSource, Material,
    renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, ViewportRenderer},
    resources::MeshData,
};
use viewport_lib_post_effects::{
    BloomEffect, BloomEffectSettings, ContactShadowEffect, ContactShadowEffectSettings, vfx_stack,
};
use viewport_lib_testkit::{DeviceProfile, headless_device_with};

const SIZE: u32 = 96;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(&DeviceProfile::low_power("post-effects-parity"))
}

fn quad_mesh() -> MeshData {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh
}

fn box_mesh() -> MeshData {
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    mesh.normals = vec![
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    mesh.indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    mesh
}

/// A frame with the default orbit camera, square aspect, no chrome, and a
/// flat background.
fn base_frame(background: [f32; 4]) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(background.into());
    frame
}

/// Ground plane + floating box scene lit by one slanted directional light:
/// geometry that produces a screen-space contact shadow on the ground.
fn contact_shadow_frame(light_direction: [f32; 3]) -> FrameData {
    let mut frame = base_frame([0.2, 0.2, 0.2, 1.0]);
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: light_direction,
    };
    frame.effects.lighting.lights = vec![light];
    frame
}

fn contact_shadow_items(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
) -> Vec<SceneRenderItem> {
    let ground = renderer
        .resources_mut()
        .upload_mesh_data(device, &quad_mesh())
        .unwrap();
    let cube = renderer
        .resources_mut()
        .upload_mesh_data(device, &box_mesh())
        .unwrap();
    let mut ground_item = SceneRenderItem::default();
    ground_item.mesh_id = ground;
    ground_item.model = glam::Mat4::from_scale(glam::Vec3::splat(8.0)).to_cols_array_2d();
    ground_item.material = Material::from_colour([0.7, 0.7, 0.7]);
    let mut cube_item = SceneRenderItem::default();
    cube_item.mesh_id = cube;
    cube_item.model = (glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.25))
        * glam::Mat4::from_scale(glam::Vec3::splat(0.4)))
    .to_cols_array_2d();
    cube_item.material = Material::from_colour([0.6, 0.3, 0.2]);
    vec![ground_item, cube_item]
}

/// The external contact-shadow copy renders pixel-identically to the
/// built-in.
#[test]
fn contact_shadow_parity() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let light_direction = [0.5, 0.2, 0.8];
    let settings = ContactShadowEffectSettings {
        enabled: true,
        max_distance: 0.6,
        steps: 16,
        thickness: 0.4,
        light_direction,
    };

    #[derive(Clone, Copy, PartialEq)]
    enum Variant {
        Builtin,
        External,
        Off,
    }
    let render = |variant: Variant| -> Vec<u8> {
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let items = contact_shadow_items(&mut renderer, &device);
        if variant == Variant::External {
            let (effect, _handle) = ContactShadowEffect::new(settings);
            renderer.add_post_effect_producer(Box::new(effect));
        }
        let mut frame = contact_shadow_frame(light_direction);
        let cs = &mut frame.effects.post_process.contact_shadows;
        cs.enabled = variant == Variant::Builtin;
        cs.max_distance = settings.max_distance;
        cs.steps = settings.steps;
        cs.thickness = settings.thickness;
        frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
        renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE)
    };

    let builtin = render(Variant::Builtin);
    let external = render(Variant::External);
    let off = render(Variant::Off);

    // Guard: the effect must actually darken something, or parity is
    // vacuous.
    assert_ne!(
        builtin, off,
        "contact shadows changed nothing in this scene; the parity check proves nothing"
    );
    assert_eq!(
        builtin, external,
        "external contact-shadow copy diverged from the built-in"
    );
}

/// The external bloom copy renders pixel-identically to the built-in.
#[test]
fn bloom_parity() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let settings = BloomEffectSettings {
        enabled: true,
        threshold: 0.7,
        intensity: 2.0,
        max_brightness: 8.0,
    };

    #[derive(Clone, Copy, PartialEq)]
    enum Variant {
        Builtin,
        External,
        Off,
    }
    let render = |variant: Variant| -> Vec<u8> {
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let mesh = renderer
            .resources_mut()
            .upload_mesh_data(&device, &quad_mesh())
            .unwrap();
        if variant == Variant::External {
            let (effect, _handle) = BloomEffect::new(settings);
            renderer.add_post_effect_producer(Box::new(effect));
        }
        let mut frame = base_frame([0.15, 0.15, 0.15, 1.0]);
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::from_scale(glam::Vec3::splat(0.6)).to_cols_array_2d();
        item.material = Material::from_colour([0.02, 0.02, 0.02]);
        item.material.emissive = [6.0, 6.0, 6.0].into();
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        let bloom = &mut frame.effects.post_process.bloom;
        bloom.enabled = variant == Variant::Builtin;
        bloom.threshold = settings.threshold;
        bloom.intensity = settings.intensity;
        bloom.max_brightness = settings.max_brightness;
        renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE)
    };

    let builtin = render(Variant::Builtin);
    let external = render(Variant::External);
    let off = render(Variant::Off);

    assert_ne!(
        builtin, off,
        "bloom changed nothing in this scene; the parity check proves nothing"
    );
    assert_eq!(
        builtin, external,
        "external bloom copy diverged from the built-in"
    );
}

/// The three-stage vfx stack: disabled stages are inert (registered ==
/// unregistered), enabled stages change the image through the chain with
/// no host-side target plumbing.
#[test]
fn vfx_stack_chains() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let render = |register: bool, enable: bool| -> Vec<u8> {
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let mesh = renderer
            .resources_mut()
            .upload_mesh_data(&device, &box_mesh())
            .unwrap();
        if register {
            let (stages, settings) = vfx_stack();
            for (stage, order) in stages {
                renderer.add_post_effect_stage(stage, order);
            }
            {
                let mut s = settings.lock().unwrap();
                s.colour_grade.enabled = enable;
                s.colour_grade.tint = [1.2, 0.9, 0.7];
                s.depth_fog.enabled = enable;
                s.depth_fog.near = 0.0;
                s.depth_fog.far = 1.0;
                s.depth_fog.amount = 0.4;
                s.edge_detect.enabled = enable;
            }
        }
        let mut frame = base_frame([0.25, 0.3, 0.35, 1.0]);
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.material = Material::from_colour([0.6, 0.6, 0.6]);
        item.settings.unlit = true;
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE)
    };

    let baseline = render(false, false);
    let disabled = render(true, false);
    let enabled = render(true, true);

    assert_eq!(
        baseline, disabled,
        "registered-but-disabled stages altered the image"
    );
    assert_ne!(baseline, enabled, "enabled vfx stack changed nothing");

    // The warm grade tint must show: the background's red channel gains on
    // blue relative to the baseline's cool background.
    let idx = ((2 * SIZE + 2) * 4) as usize;
    let (r0, b0) = (baseline[idx] as i32, baseline[idx + 2] as i32);
    let (r1, b1) = (enabled[idx] as i32, enabled[idx + 2] as i32);
    assert!(
        (r1 - b1) > (r0 - b0),
        "warm tint not visible: baseline r-b {}, graded r-b {}",
        r0 - b0,
        r1 - b1
    );
}
