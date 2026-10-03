//! GPU particle systems drawn through the renderer.
//!
//! One file per item type, so a type's coverage travels with it.

use viewport_lib::renderer::SpriteBlend;
use viewport_lib_item_types::*;

mod common;
use common::*;

/// Warming the type builds its compute pipelines and every draw pipeline, so
/// the first frames of an unlit sprite, a lit sprite and a mesh system build
/// nothing in either format.
#[test]
fn a_warmed_particle_type_builds_nothing_on_its_first_frame() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut configs = vec![GpuParticleSystemConfig::default(); 3];
    if let ParticleRender::Sprite { lit, blend, .. } = &mut configs[1].render {
        *lit = true;
        *blend = SpriteBlend::Additive;
    }
    configs[2].render = ParticleRender::Mesh {
        mesh_id,
        texture_id: None,
        blend: SpriteBlend::Premultiplied,
        align: ParticleMeshAlign::Identity,
    };
    let ids: Vec<_> = configs
        .iter_mut()
        .map(|config| {
            config.capacity = 64;
            renderer.create_gpu_particle_system(&device, &queue, config)
        })
        .collect();

    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default().with_item_type::<GpuParticlesPlugin>(),
    );
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for hdr in [true, false] {
        let mut frame = sub_object_pick_frame();
        if !hdr {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        for id in &ids {
            let mut item = GpuParticleSystemItem::new(*id, 0.1);
            item.emitter.rate = 100.0;
            frame.scene.items_mut::<GpuParticleSystemItem>().push(item);
        }
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    }
    let builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| l.starts_with("particle") || l.starts_with("module particle"))
        .collect();
    assert!(
        builds.is_empty(),
        "the first particle frames built pipelines after the warm-up: {builds:?}"
    );
}
