//! Pipeline warmup checks: the decal pipelines build at renderer creation
//! and resource uploads compile their pipelines, so the first frame that
//! uses either compiles nothing. `FrameStats::pipelines_built_this_frame`
//! is the observable: it counts lazy pipeline builds since the previous
//! prepare, so a frame that hits a cold pipeline reads non-zero.

use viewport_lib::{
    CameraFrame, DecalItem, FrameData, Material, PointCloudItem, PointCloudRefItem, SceneFrame,
    SceneRenderItem, VolumeItem,
};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

fn mesh_frame(item: SceneRenderItem, size: [f32; 2]) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 6.0, 0.6, 1.0);
    FrameData::new(
        CameraFrame::from_camera(&camera, size),
        SceneFrame::from_surface_items(vec![item]),
    )
}

fn checker_texture(h: &mut Harness) -> viewport_lib::TextureId {
    let tex = viewport_lib_testkit::textures::checker(64, 8, [200, 40, 40], [240, 240, 240]);
    h.renderer
        .resources_mut()
        .upload_texture(
            &h.device,
            &h.queue,
            viewport_lib::TextureData::srgb(tex.width, tex.height, tex.rgba.to_vec()),
        )
        .expect("texture upload")
}

/// The first frame containing a decal must not compile a pipeline: the decal
/// pipelines are built in `ViewportRenderer::new()` precisely because decals
/// tend to first appear mid-session.
#[test]
fn first_decal_frame_builds_no_pipelines() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::stress_sphere(1.0, 3).into())
        .expect("mesh upload");
    let tex_id = checker_texture(&mut h);

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);

    // Settle: the first frames build the frame-one lazy set (HDR shared,
    // instanced pipelines), which is load-time work, not the decal's.
    let base = mesh_frame(item.clone(), [200.0, 150.0]);
    let _ = h.render_two_frames(&base, 200, 150);
    assert_eq!(
        h.stats().pipelines_built_this_frame,
        0,
        "steady frames must not compile pipelines"
    );

    let mut decal = DecalItem::default();
    decal.texture_id = tex_id;
    decal.transform = glam::Mat4::from_scale(glam::Vec3::splat(2.0)).to_cols_array_2d();
    let mut with_decal = mesh_frame(item, [200.0, 150.0]);
    with_decal.scene.decals.push(decal);
    let _ = h.render(&with_decal, 200, 150);
    assert_eq!(
        h.stats().pipelines_built_this_frame,
        0,
        "the first decal frame must not compile a pipeline (decal pipelines are built in new())"
    );
}

/// The point cloud pipelines belong to the point cloud item-type plugin, so
/// `upload_point_cloud` compiles nothing: it writes buffers and builds one bind
/// group against a layout the renderer already holds.
#[test]
fn point_cloud_pipelines_are_owned_by_the_plugin() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::stress_sphere(1.0, 3).into())
        .expect("mesh upload");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    let base = mesh_frame(item.clone(), [200.0, 150.0]);
    let _ = h.render_two_frames(&base, 200, 150);

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
    cloud.point_size = 8.0;
    let source = h
        .renderer
        .resources_mut()
        .upload_point_cloud(&h.device, &h.queue, &cloud);

    let _ = h.render(&base, 200, 150);
    assert_eq!(
        h.stats().pipelines_built_this_frame,
        0,
        "upload_point_cloud writes buffers and compiles nothing"
    );

    // The first frame that draws the cloud builds the plugin's pipelines, but
    // they are the plugin's own and so stay off this counter.
    let mut with_cloud = mesh_frame(item, [200.0, 150.0]);
    with_cloud
        .scene
        .point_cloud_refs
        .push(PointCloudRefItem::new(source));
    let _ = h.render(&with_cloud, 200, 150);
    assert_eq!(
        h.stats().pipelines_built_this_frame,
        0,
        "plugin-owned pipelines are not counted in the lib's pipeline cache"
    );
}

/// The volume ray-march pipelines belong to the volume item-type plugin, which
/// builds them on the first prepare that has a volume to draw. Neither
/// `upload_volume` nor any frame reports them through
/// `pipelines_built_this_frame`: that counter tracks the lib's own pipeline
/// cache, and a plugin owns its pipelines outright.
#[test]
fn volume_pipelines_are_owned_by_the_plugin() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::stress_sphere(1.0, 3).into())
        .expect("mesh upload");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    let base = mesh_frame(item.clone(), [200.0, 150.0]);
    let _ = h.render_two_frames(&base, 200, 150);

    let dims = [8u32, 8, 8];
    let data = vec![0.5f32; 512];
    let volume_id = h
        .renderer
        .resources_mut()
        .upload_volume(&h.device, &h.queue, &data, dims);

    // The upload is a texture upload only; no pipeline work rides along.
    let _ = h.render(&base, 200, 150);
    assert_eq!(
        h.stats().pipelines_built_this_frame,
        0,
        "upload_volume uploads a texture and compiles nothing"
    );

    // The first frame that draws the volume does build the plugin's pipelines,
    // but they are the plugin's own and so stay off this counter.
    let mut volume = VolumeItem::default();
    volume.volume_id = volume_id;
    let mut with_volume = mesh_frame(item, [200.0, 150.0]);
    with_volume.scene.volumes.push(volume);
    let _ = h.render(&with_volume, 200, 150);
    assert_eq!(
        h.stats().pipelines_built_this_frame,
        0,
        "plugin-owned pipelines are not counted in the lib's pipeline cache"
    );
}

/// `prebuild_mesh_debug_sidecars` builds the wireframe and normals buffers at
/// call time; the views then render correctly from the prebuilt data.
#[test]
fn prebuilt_sidecars_render() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let data = meshes::stress_sphere(1.0, 4);
    let expected_tris = (data.indices.len() / 3) as u64;
    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &data.into())
        .expect("mesh upload");
    h.renderer
        .resources_mut()
        .prebuild_mesh_debug_sidecars(&h.device, mesh_id);

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    item.settings.wireframe = true;
    item.show_normals = true;
    let fd = mesh_frame(item, [200.0, 150.0]);
    let stats = h.render_two_frames(&fd, 200, 150);
    assert_eq!(stats.triangles_submitted, expected_tris);
    assert_eq!(stats.pipelines_built_this_frame, 0);
}

/// The curve mesh pipelines belong to the streamtube, tube and ribbon item-type
/// plugins, so neither `upload_streamtube` nor the first frame that draws one
/// reports through `pipelines_built_this_frame`: that counter tracks the lib's
/// own pipeline cache, and a plugin owns its pipelines outright.
#[test]
fn curve_pipelines_are_owned_by_the_plugins() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::stress_sphere(1.0, 3).into())
        .expect("mesh upload");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    let base = mesh_frame(item.clone(), [200.0, 150.0]);
    let _ = h.render_two_frames(&base, 200, 150);

    let mut ribbon = viewport_lib::RibbonItem::default();
    ribbon.positions = vec![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 0.5;
    let source = h
        .renderer
        .resources_mut()
        .upload_ribbon(&h.device, &h.queue, &ribbon);

    let _ = h.render(&base, 200, 150);
    assert_eq!(
        h.stats().pipelines_built_this_frame,
        0,
        "upload_ribbon writes buffers and compiles nothing"
    );

    // The first frame that draws a curve builds the plugins' pipelines, but they
    // are the plugins' own and so stay off this counter.
    let mut with_curves = mesh_frame(item, [200.0, 150.0]);
    with_curves
        .scene
        .ribbon_refs
        .push(viewport_lib::RibbonRefItem::new(source));
    let mut streamtube = viewport_lib::StreamtubeItem::default();
    streamtube.positions = vec![[-1.0, 0.5, 0.0], [0.0, 0.5, 0.0], [1.0, 0.5, 0.0]];
    streamtube.strip_lengths = vec![3];
    with_curves.scene.streamtube_items.push(streamtube);
    let mut tube = viewport_lib::TubeItem::default();
    tube.positions = vec![[-1.0, -0.5, 0.0], [0.0, -0.5, 0.0], [1.0, -0.5, 0.0]];
    tube.strip_lengths = vec![3];
    with_curves.scene.tube_items.push(tube);
    let _ = h.render(&with_curves, 200, 150);
    assert_eq!(
        h.stats().pipelines_built_this_frame,
        0,
        "plugin-owned pipelines are not counted in the lib's pipeline cache"
    );
}
