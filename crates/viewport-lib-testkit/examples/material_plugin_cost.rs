//! What material plugins cost: the warm-up call, and what the first frame that
//! draws them still compiles after it.
//!
//! Registers the five shading plugins the eframe showcase uses, warms them the
//! way that showcase does at startup, then draws one sphere per plugin and
//! reads the build log back for each of the first frames.
//!
//! ```bash
//! cargo run --release --example material_plugin_cost
//! ```
//!
//! `VPL_NO_WARM=1` skips the warm-up, so the first frames show the cost of
//! building the sets on demand. `VPL_DETAIL=1` lists every object built.

use std::time::Instant;

use viewport_lib::resources::build_log;
use viewport_lib::{
    Camera, FrameData, LightKind, LightSource, Material, RenderCamera, SceneRenderItem,
    SurfaceSubmission, ViewportRenderer, wgpu,
};
use viewport_lib_testkit::{DeviceProfile, device::headless_device_with_info};

#[path = "../../viewport-lib-examples/eframe/examples/plugins/surface_detail_plugin.rs"]
#[allow(dead_code)]
mod surface_detail_plugin;
#[path = "../../viewport-lib-examples/eframe/examples/plugins/toon_plugin.rs"]
#[allow(dead_code)]
mod toon_plugin;

use surface_detail_plugin::{DetailLayerPlugin, DissolvePlugin, ParallaxPlugin};
use toon_plugin::{RimPlugin, ToonPlugin};

const SIZE: u32 = 512;

fn ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}

fn report(what: &str, took: f32) {
    let builds = build_log::drain();
    let total: f32 = builds.iter().map(|(_, ms)| ms).sum();
    println!(
        "{what:<34} {took:8.2} ms, {} pipelines/modules ({total:.2} ms)",
        builds.len()
    );
    let plugin = |l: &str| l.contains("material_plugin") || l.contains("shade_compose");
    let of_plugins: Vec<&(String, f32)> = builds.iter().filter(|(l, _)| plugin(l)).collect();
    if !of_plugins.is_empty() {
        let plugin_ms: f32 = of_plugins.iter().map(|(_, ms)| ms).sum();
        println!(
            "    of which the plugins' own: {} ({plugin_ms:.2} ms)",
            of_plugins.len()
        );
    }
    if std::env::var_os("VPL_DETAIL").is_some() {
        // Every object, with repeats of a label folded together.
        let mut by_label: Vec<(String, u32, f32)> = Vec::new();
        for (label, ms) in &builds {
            match by_label.iter_mut().find(|(l, _, _)| l == label) {
                Some(entry) => {
                    entry.1 += 1;
                    entry.2 += ms;
                }
                None => by_label.push((label.clone(), 1, *ms)),
            }
        }
        by_label.sort_by(|a, b| b.2.total_cmp(&a.2));
        for (label, count, ms) in &by_label {
            println!("    {count:>3} x {label:<58} {ms:7.3} ms");
        }
    }
}

fn main() {
    build_log::enable();
    let profile =
        DeviceProfile::high_performance("material-plugin-cost").with_recommended_features();
    let (device, queue, info) = headless_device_with_info(&profile).expect("no GPU adapter");
    println!(
        "adapter: {:?} / {} ({:?})",
        info.backend, info.name, info.device_type
    );

    let t = Instant::now();
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    viewport_lib_item_types::install(&mut renderer, &device);
    report("new + install", ms(t));

    let t = Instant::now();
    let resources = renderer.resources_mut();
    let ids: Vec<_> = [
        resources.register_material_plugin(&device, &ToonPlugin),
        resources.register_material_plugin(&device, &RimPlugin),
        resources.register_material_plugin(&device, &DetailLayerPlugin),
        resources.register_material_plugin(&device, &ParallaxPlugin),
        resources.register_material_plugin(&device, &DissolvePlugin),
    ]
    .into_iter()
    .map(|id| id.expect("register plugin"))
    .collect();
    report("register five plugins", ms(t));

    if std::env::var_os("VPL_NO_WARM").is_none() {
        let t = Instant::now();
        renderer
            .resources_mut()
            .warm_material_plugin_pipelines(&device, &ids);
        report("warm five plugins", ms(t));
    }

    // One built-in sphere and one per plugin, in a row, under one sun.
    let mesh = viewport_lib::primitives::sphere(1.0, 48, 24);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .expect("upload sphere");
    let plugins = std::iter::once(None).chain(ids.iter().copied().map(Some));
    let items: Vec<SceneRenderItem> = plugins
        .enumerate()
        .map(|(i, plugin)| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(i as f32 * 3.0 - 7.5, 0.0, 1.0))
                    .to_cols_array_2d();
            item.material = Material::pbr([0.75, 0.3, 0.3], 0.1, 0.55);
            item.material.shading_plugin = plugin;
            item
        })
        .collect();

    let camera = Camera {
        center: glam::Vec3::new(0.0, 0.0, 1.0),
        distance: 15.0,
        orientation: glam::Quat::from_rotation_x(1.1),
        ..Camera::default()
    };
    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [0.5, 0.35, 1.0],
    };
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&camera);
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    frame.effects.lighting.lights = vec![sun];
    let _ = build_log::drain();

    for i in 0..4 {
        let t = Instant::now();
        let _ = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
        report(&format!("frame {i}"), ms(t));
    }
}
