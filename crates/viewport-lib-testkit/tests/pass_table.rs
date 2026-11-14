//! The HDR frame's pass list.
//!
//! Each viewport plans its HDR passes before it draws. These tests pin the
//! passes the plan runs for the default scene and a few effect combinations,
//! and hold the plan against the passes that actually encoded.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use viewport_lib::Colour;
use viewport_lib::plugin_api::post_effect::stage_order;
use viewport_lib::{FrameData, Material, MeshItem, PostEffectSlot, SurfaceSubmission};
use viewport_lib_testkit::fixtures::{
    CallLog, LoggingPostEffectProducer, PassthroughPostEffectStage, probe_frame, probe_quad,
};
use viewport_lib_testkit::{Harness, catalogue, frame_for};

const SIZE: u32 = 64;

/// A frame with one lit quad in the middle of the view.
fn quad_frame(harness: &mut Harness) -> FrameData {
    let mesh = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");
    let mut frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    let mut item = MeshItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(0.8)).into();
    item.material = harness
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8)))
        .expect("material upload");
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame
}

/// Render `frame` and return the passes the plan runs, after checking them
/// against the passes that encoded.
fn passes(harness: &mut Harness, frame: &FrameData) -> Vec<&'static str> {
    harness.render(frame, SIZE, SIZE);
    let planned: Vec<&'static str> = harness
        .renderer
        .viewport_pass_table(0)
        .into_iter()
        .filter(|(_, live)| *live)
        .map(|(name, _)| name)
        .collect();
    let ran = harness.renderer.debug_viewport_passes_ran(0);
    assert_eq!(planned, ran, "the plan must match the passes that encoded");
    planned
}

/// The passes every HDR frame of the quad scene runs. The harness installs
/// the built-in item types, some of which encode at the encoder scopes, but
/// the frame has none of their items, so no scope runs.
const BASE: [&str; 3] = ["scene", "exposure", "tone_map"];

fn with(extra: &[&'static str]) -> Vec<&'static str> {
    let mut list: Vec<&'static str> = Vec::new();
    for name in [
        "scene",
        "hiz_store",
        "ssaa_resolve",
        "surface_mask",
        "scope_on_opaque_surfaces",
        "sub_highlight",
        "depth_read",
        "opaque_copy",
        "transmission",
        "scope_after_opaque",
        "oit",
        "scope_after_transparent",
        "outline_composite",
        "foreground",
        "ssao",
        "contact_shadow",
        "bloom",
        "dof",
        "exposure",
        "external_producers",
        "tone_map",
        "post_stages",
        "upscale",
        "output_depth_blit",
        "foreground_depth_stamp",
        "scene_overlays",
        "overlay",
    ] {
        if BASE.contains(&name) || extra.contains(&name) {
            list.push(name);
        }
    }
    list
}

#[test]
fn default_scene_runs_todays_passes() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let frame = quad_frame(&mut harness);
    let list = passes(&mut harness, &frame);
    eprintln!("default: {list:?}");
    assert_eq!(list, with(&[]));
}

#[test]
fn each_effect_adds_its_own_pass() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut frame = quad_frame(&mut harness);
    let defaults = frame.effects.post_process.clone();
    type Case = (&'static str, fn(&mut FrameData), &'static [&'static str]);
    let cases: [Case; 5] = [
        (
            "bloom",
            |f| f.effects.post_process.bloom.enabled = true,
            &["bloom"],
        ),
        ("ssao", |f| f.effects.post_process.ssao = true, &["ssao"]),
        (
            "dof",
            |f| f.effects.post_process.dof.enabled = true,
            &["dof"],
        ),
        (
            "fxaa",
            |f| f.effects.post_process.fxaa = true,
            &["post_stages"],
        ),
        (
            "bloom+ssao+dof+fxaa",
            |f| {
                let pp = &mut f.effects.post_process;
                pp.bloom.enabled = true;
                pp.ssao = true;
                pp.dof.enabled = true;
                pp.fxaa = true;
            },
            &["ssao", "bloom", "dof", "post_stages"],
        ),
    ];
    for (label, apply, extra) in cases {
        frame.effects.post_process = defaults.clone();
        apply(&mut frame);
        let list = passes(&mut harness, &frame);
        eprintln!("{label}: {list:?}");
        assert_eq!(list, with(extra), "{label}");
    }
    // Back to the default frame: the extra passes stop again.
    frame.effects.post_process = defaults;
    assert_eq!(passes(&mut harness, &frame), with(&[]));
}

/// Contact shadows, the foreground pass and its depth stamp, the grid,
/// supersampling and the Hi-Z store each add their passes.
#[test]
fn scene_features_add_their_passes() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut frame = quad_frame(&mut harness);
    frame.effects.post_process.contact_shadows.enabled = true;
    assert_eq!(passes(&mut harness, &frame), with(&["contact_shadow"]));
    frame.effects.post_process.contact_shadows.enabled = false;

    let item = match &frame.scene.surfaces {
        SurfaceSubmission::Flat(items) => items[0].clone(),
        _ => unreachable!(),
    };
    frame.scene.foreground_items.push(item);
    assert_eq!(
        passes(&mut harness, &frame),
        with(&["foreground", "foreground_depth_stamp"])
    );
    frame.scene.foreground_items.clear();

    frame.viewport.show_grid = true;
    assert_eq!(passes(&mut harness, &frame), with(&["scene_overlays"]));
    frame.viewport.show_grid = false;

    frame.effects.post_process.ssaa_factor = 2;
    assert_eq!(passes(&mut harness, &frame), with(&["ssaa_resolve"]));
    frame.effects.post_process.ssaa_factor = 1;

    harness.renderer.set_occlusion_culling(true);
    assert_eq!(passes(&mut harness, &frame), with(&["hiz_store"]));
}

/// A transmissive material moves its mesh out of the scene pass into the
/// transmission pass, after the opaque scene is copied. Back to an opaque
/// material, both passes stop.
#[test]
fn a_transmissive_material_adds_the_copy_and_transmission_passes() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut frame = quad_frame(&mut harness);
    let mut glass = Material::from_colour(Colour::linear_rgb(0.9, 0.9, 0.9))
        .with_transmission(1.0, 0.5)
        .with_ior(1.5);
    glass.shading_model = viewport_lib::ShadingModel::Pbr;
    glass.roughness = 0.1;
    let glass = harness
        .renderer
        .resources_mut()
        .upload_material(glass)
        .expect("material upload");
    let mut item = match &frame.scene.surfaces {
        SurfaceSubmission::Flat(items) => items[0].clone(),
        _ => unreachable!(),
    };
    let opaque = item.material;
    item.material = glass;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    frame.scene.generation += 1;
    let list = passes(&mut harness, &frame);
    eprintln!("transmission: {list:?}");
    assert_eq!(list, with(&["opaque_copy", "transmission"]));

    item.material = opaque;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame.scene.generation += 1;
    assert_eq!(passes(&mut harness, &frame), with(&[]));
}

#[test]
fn external_producers_and_stages_run_as_side_effects() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    harness
        .renderer
        .add_post_effect_producer(Box::new(LoggingPostEffectProducer::new(
            log.clone(),
            PostEffectSlot::Bloom,
            Arc::new(AtomicBool::new(true)),
        )))
        .unwrap();
    let stage = PassthroughPostEffectStage::new(
        &harness.device,
        log.clone(),
        "stage",
        1.0,
        Arc::new(AtomicBool::new(true)),
    );
    harness
        .renderer
        .add_post_effect_stage(Box::new(stage), stage_order::EXTERNAL_DEFAULT)
        .unwrap();
    let frame = quad_frame(&mut harness);
    let list = passes(&mut harness, &frame);
    eprintln!("producer fixture: {list:?}");
    assert_eq!(list, with(&["external_producers", "post_stages"]));
}

/// Across the golden catalogue, which covers transparency, outlines,
/// overlays, the foreground pass and supersampling, every pass the plan runs
/// encodes and nothing else does.
#[test]
fn the_plan_matches_the_encoded_passes_across_the_catalogue() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut seen: Vec<&'static str> = Vec::new();
    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [400.0, 300.0]);
        for _ in 0..2 {
            harness.render(&frame, 400, 300);
            let planned: Vec<&'static str> = harness
                .renderer
                .viewport_pass_table(0)
                .into_iter()
                .filter(|(_, live)| *live)
                .map(|(name, _)| name)
                .collect();
            let ran = harness.renderer.debug_viewport_passes_ran(0);
            assert_eq!(
                planned, ran,
                "{}: plan and encoded passes differ",
                scene.name
            );
            for name in ran {
                if !seen.contains(&name) {
                    seen.push(name);
                }
            }
        }
    }
    eprintln!("passes seen across the catalogue: {seen:?}");
}
