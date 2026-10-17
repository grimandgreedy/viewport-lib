//! `FrameStats::per_object_by_reason` names one reason per per-object item,
//! the first test it fails, and sums to `per_object_items`.

use viewport_lib::{
    BuiltinMatcap, CameraFrame, Colour, FrameData, Material, PerObjectReasons, SamplerKey,
    SceneFrame, SceneRenderItem, ShadingModel, TextureSlot,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

#[test]
fn per_object_items_are_split_by_reason() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::heightfield(8, 8, 1.0, 0.2))
        .expect("upload");
    let clay = h
        .renderer
        .resources_mut()
        .builtin_matcap_id(BuiltinMatcap::Clay);

    let base = Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.7));
    let mut matcap = base.clone();
    matcap.shading_model = ShadingModel::Matcap(clay);
    let sampler = base
        .clone()
        .with_sampler(TextureSlot::Albedo, SamplerKey::default());
    // A sampler no longer keeps an item off the instanced path, so this counts
    // as a matcap only.
    let matcap_and_sampler = matcap
        .clone()
        .with_sampler(TextureSlot::Albedo, SamplerKey::default());

    let mut items = Vec::new();
    let mut push = |material: &Material, wireframe: bool, hidden: bool, force: bool| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = id;
        item.model =
            glam::Mat4::from_translation(glam::Vec3::new(items.len() as f32 * 1.5, 0.0, 0.0))
                .to_cols_array_2d();
        item.material = material.clone();
        item.settings.wireframe = wireframe;
        item.settings.hidden = hidden;
        item.settings.force_per_object = force;
        items.push(item);
    };
    for _ in 0..4 {
        push(&base, false, false, false);
    }
    for _ in 0..3 {
        push(&matcap, false, false, false);
    }
    for _ in 0..2 {
        push(&sampler, false, false, false);
    }
    push(&matcap_and_sampler, false, false, false);
    push(&base, true, false, false);
    // The test setting is the last reason, so a forced matcap counts as a matcap.
    push(&base, false, false, true);
    push(&matcap, false, false, true);
    // Hidden items draw on neither path and are not counted.
    push(&matcap, false, true, false);

    let camera = orbit_camera(glam::Vec3::new(8.0, 0.0, 0.0), 20.0, 0.6, 1.0);
    let fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(items),
    );
    let stats = h.render_two_frames(&fd, 200, 150);

    assert_eq!(
        stats.per_object_by_reason,
        PerObjectReasons {
            matcap: 5,
            wireframe: 1,
            force_per_object: 1,
            ..Default::default()
        }
    );
    assert_eq!(stats.per_object_items, 7);
    assert_eq!(stats.per_object_by_reason.total(), stats.per_object_items);
}
