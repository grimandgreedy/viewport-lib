//! GPU and CPU picking: object id, sub-object refinement, plugin hooks, rect pick.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

#[test]
fn gpu_pick_returns_object_id_under_cursor() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

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

    // A single box at the origin with a nonzero pick id.
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.settings.pick_id = PickId(7);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    // Cursor at the viewport centre lands on the box. This exercises the
    // multi-target pick pass (object id + primitive id + depth).
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(7)));
}

#[test]
fn gpu_pick_async_begin_poll_returns_hit() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

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

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.settings.pick_id = PickId(7);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    // Poll before starting anything: no pick is in flight.
    assert!(matches!(renderer.pick_object_poll(&device), PickPoll::Idle));

    // Begin submits the pass without blocking on the queue.
    let started = renderer.pick_object_begin(
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::all(),
    );
    assert!(started);

    // Poll until the read-back lands. The poll is non-blocking, so drive the
    // device between polls the way a render loop's own submissions would, and
    // bound the loop so a stuck map fails the test instead of hanging.
    let mut hit = None;
    let mut saw_pending = false;
    for _ in 0..1000 {
        match renderer.pick_object_poll(&device) {
            PickPoll::Pending => {
                saw_pending = true;
                let enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                queue.submit(std::iter::once(enc.finish()));
                continue;
            }
            PickPoll::Ready(h) => {
                hit = h;
                break;
            }
            PickPoll::Idle => panic!("poll went Idle while a pick was in flight"),
        }
    }
    // The read-back should not be ready on the very first poll (that would mean
    // the poll blocked); it lands after the device is driven.
    assert!(
        saw_pending,
        "expected at least one Pending before the pick resolved"
    );
    assert_eq!(hit.map(|h| h.id), Some(7));

    // The slot is cleared once read: a further poll is Idle.
    assert!(matches!(renderer.pick_object_poll(&device), PickPoll::Idle));
}

#[test]
fn gpu_pick_async_begin_on_empty_space_reports_no_hit() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

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

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.settings.pick_id = PickId(7);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    // A corner pixel misses the centred box. The pass still runs (there is
    // pickable geometry), so begin submits; the read-back resolves to no hit.
    let started = renderer.pick_object_begin(
        glam::Vec2::new(1.0, 1.0),
        &frame,
        &device,
        &queue,
        PickMask::all(),
    );
    assert!(started);

    let mut resolved = false;
    for _ in 0..1000 {
        match renderer.pick_object_poll(&device) {
            PickPoll::Pending => {
                let enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                queue.submit(std::iter::once(enc.finish()));
                continue;
            }
            PickPoll::Ready(h) => {
                assert!(h.is_none(), "corner pixel should miss the box");
                resolved = true;
                break;
            }
            PickPoll::Idle => panic!("poll went Idle while a pick was in flight"),
        }
    }
    assert!(resolved, "async pick never resolved");
}

#[test]
fn gpu_pick_hits_voxel_volume() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // A fully dense 8^3 volume: every voxel is in-threshold, so the raymarch pick
    // hits on the first sample anywhere the bounding cube covers.
    let dims = [8u32, 8, 8];
    let data = vec![1.0f32; (dims[0] * dims[1] * dims[2]) as usize];
    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &data, dims);

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
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());

    // Centre the bounding cube on the origin so the default camera sees it.
    let mut vol = VolumeItem::default();
    vol.volume_id = volume_id;
    vol.scalar_range = (0.0, 1.0);
    vol.threshold_min = 0.0;
    vol.threshold_max = 1.0;
    vol.bbox_min = [-0.5, -0.5, -0.5];
    vol.bbox_max = [0.5, 0.5, 0.5];
    vol.settings.pick_id = PickId(63);
    frame.scene.volumes = vec![vol];

    // prepare builds the per-volume GPU data (bind group + cube) the pick reuses.
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::all(),
    );
    assert_eq!(hit.map(|h| h.id), Some(63));
}

#[test]
fn gpu_pick_voxel_volume_resolves_voxel() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // A fully dense 8^3 volume centred on the origin: the raymarch pick hits a
    // voxel wherever the bounding cube covers, and the primitive channel carries
    // that voxel's flat index.
    let dims = [8u32, 8, 8];
    let data = vec![1.0f32; (dims[0] * dims[1] * dims[2]) as usize];
    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &data, dims);

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
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());

    let mut vol = VolumeItem::default();
    vol.volume_id = volume_id;
    vol.scalar_range = (0.0, 1.0);
    vol.threshold_min = 0.0;
    vol.threshold_max = 1.0;
    vol.bbox_min = [-0.5, -0.5, -0.5];
    vol.bbox_max = [0.5, 0.5, 0.5];
    vol.settings.pick_id = PickId(63);
    frame.scene.volumes = vec![vol];

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // A VOXEL-masked query (a subset of POINT_LIKE) resolves the hit voxel's
    // flat index, in range for the 8^3 = 512-voxel grid.
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::VOXEL,
        )
        .expect("centre ray should hit the dense volume");
    assert_eq!(hit.id, 63);
    match hit.sub_object {
        Some(viewport_lib::SubObjectRef::Voxel(v)) => {
            assert!(v < 512, "voxel index {v} out of range for 8^3 grid");
        }
        other => panic!("expected a Voxel sub-object, got {other:?}"),
    }

    // An OBJECT-only query still resolves the volume at object level, no voxel.
    let obj = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::OBJECT,
        )
        .expect("centre ray should hit the dense volume");
    assert_eq!(obj.id, 63);
    assert_eq!(obj.sub_object, None);
}

fn scatter_pick_frame() -> FrameData {
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
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

#[test]
fn gpu_pick_hits_box_scatter_volume() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = scatter_pick_frame();

    // A box scatter volume centred on the origin. The pick rasterises the actual
    // box (cube proxy) and reads back its id.
    let aabb = Aabb {
        min: glam::Vec3::splat(-0.5),
        max: glam::Vec3::splat(0.5),
    };
    let mut item = ScatterVolumeItem::new(ScatterVolume::box_uniform(aabb, 1.0, [1.0, 1.0, 1.0]));
    item.settings.pick_id = PickId(41);
    frame.scene.scatter_volumes = vec![item];

    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(41)));
}

#[test]
fn gpu_pick_hits_sphere_scatter_volume() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = scatter_pick_frame();

    // A sphere scatter volume centred on the origin. The pick rasterises the
    // icosphere proxy and reads back its id at the viewport centre.
    let mut item = ScatterVolumeItem::new(ScatterVolume::sphere_uniform(
        [0.0, 0.0, 0.0],
        0.5,
        1.0,
        [1.0, 1.0, 1.0],
    ));
    item.settings.pick_id = PickId(42);
    frame.scene.scatter_volumes = vec![item];

    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(42)));
}

#[test]
fn gpu_pick_hits_decal_box() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());

    // A decal is the unit box [-0.5, 0.5]^3 mapped by `transform`; the default
    // transform places it at the origin. The GPU pick rasterises that box as a
    // proxy and reads back its pick_id.
    let mut decal = DecalItem::default();
    decal.settings.pick_id = PickId(77);
    frame.scene.decals = vec![decal];

    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(77)));
}

#[test]
fn gpu_pick_hits_volume_mesh_boundary() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // The box stands in for an extracted volume-mesh boundary surface.
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

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

    // No surfaces: the only pickable geometry is a volume-mesh boundary, which
    // the pick pass only sees after G3 folds boundaries into the surface draw.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    let mut vm = VolumeMeshItem::new(mesh_idx, vec![]);
    vm.settings.pick_id = PickId(9);
    frame.scene.volume_meshes = vec![vm];

    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(9)));
}

#[test]
fn gpu_pick_object_honors_type_mask() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

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

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.settings.pick_id = PickId(7);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let cursor = glam::Vec2::new(32.0, 32.0);

    // OBJECT mask: the surface draws and its object id comes back, but the GPU
    // path is object-level so there is no sub-object even for a mesh.
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        cursor,
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(7));
    assert_eq!(hit.and_then(|h| h.sub_object), None);

    // INSTANCE mask: surfaces answer no instance-level query, so nothing draws
    // and the pick reads back as a clean miss rather than the surface behind it.
    let miss = renderer.pick_object(
        PickBackend::Gpu,
        cursor,
        &frame,
        &device,
        &queue,
        PickMask::INSTANCE,
    );
    assert!(miss.is_none());
}

#[test]
fn gpu_pick_hits_ribbon() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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

    // A wide ribbon centred on the origin. Ribbons build an owned connected mesh
    // into ribbon_gpu_data during prepare(), so the pick pass only sees it after
    // the prepare path has run.
    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(4242)));
}

#[test]
fn gpu_pick_ribbon_resolves_segment_and_node() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Segment and node resolve from the pick shader variants; no CPU pick cache.
    let mut frame = sub_object_pick_frame();

    // A wide ribbon centred on the origin: 3 control points along X, so two
    // segments with the middle node under the centre pixel.
    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // SEGMENT: the hit resolves to one of the two segments.
    let seg = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::SEGMENT,
        )
        .expect("ribbon should be hit");
    assert_eq!(seg.id, 4242);
    match seg.sub_object {
        Some(viewport_lib::SubObjectRef::Segment(s)) => assert!(s < 2, "segment {s} out of range"),
        other => panic!("expected a Segment sub-object, got {other:?}"),
    }

    // POLY_NODE: a centre click resolves to the middle control point (index 1),
    // the nearer endpoint of whichever segment the ray landed on.
    let node = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::POLY_NODE,
        )
        .expect("ribbon should be hit");
    assert_eq!(node.id, 4242);
    assert_eq!(node.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
}

#[test]
fn gpu_pick_hits_showcase_style_voxel_volume() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Replicate showcase 33's volume: a 16^3 sphere-shaped scalar field, a bbox
    // offset from the origin, an off-origin model, and a 0.15 threshold. This is
    // the configuration reported as not selecting on the GPU backend, so the
    // test pins the object-level behaviour with those exact values.
    let dims = [16u32, 16, 16];
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    let mut data = vec![0.0f32; n];
    let (cx, cy, cz, radius) = (7.5f32, 7.5, 7.5, 7.5);
    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let flat = (ix + iy * dims[0] + iz * dims[0] * dims[1]) as usize;
                let dx = ix as f32 + 0.5 - cx;
                let dy = iy as f32 + 0.5 - cy;
                let dz = iz as f32 + 0.5 - cz;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                data[flat] = (1.0 - dist / radius).max(0.0);
            }
        }
    }
    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &data, dims);

    // Aim a camera at the volume centre in world space: bbox [0,4]^3 translated
    // by (-2,-1,-6) spans (-2,-1,-6)..(2,3,-2), centred at (0,1,-4). View it from
    // an offset that is not along the Z-up axis so the up vector stays valid.
    let target = glam::vec3(0.0, 1.0, -4.0);
    let eye = target + glam::vec3(2.0, -8.0, 3.0);
    let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Z);
    let proj = glam::Mat4::perspective_rh(60_f32.to_radians(), 1.0, 0.1, 100.0);
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::default();
        rc.view = view;
        rc.projection = proj;
        rc.eye_position = eye.to_array();
        rc.forward = (target - eye).normalize().to_array();
        rc.orientation = glam::Quat::IDENTITY;
        rc.near = 0.1;
        rc.far = 100.0;
        rc.distance = (eye - target).length();
        rc.fov = 60_f32.to_radians();
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());

    let mut vol = VolumeItem::default();
    vol.volume_id = volume_id;
    vol.model = glam::Mat4::from_translation(glam::vec3(-2.0, -1.0, -6.0)).to_cols_array_2d();
    vol.bbox_min = [0.0, 0.0, 0.0];
    vol.bbox_max = [4.0, 4.0, 4.0];
    vol.scalar_range = (0.0, 1.0);
    vol.threshold_min = 0.15;
    vol.threshold_max = 1.0;
    vol.settings.pick_id = PickId(20);
    frame.scene.volumes = vec![vol];

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(20));
}

#[test]
fn gpu_pick_hits_glyph_set() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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

    // One large sphere glyph at the origin. Glyph sets are uploaded during
    // prepare(), and the pick pass reuses the render glyph transform, so the
    // prepare path must run before picking.
    let mut glyph = GlyphItem::default();
    glyph.positions = vec![[0.0, 0.0, 0.0]];
    glyph.vectors = vec![[0.0, 0.0, 1.0]];
    glyph.scale = 2.0;
    glyph.scale_by_magnitude = false;
    glyph.glyph_type = GlyphType::Sphere;
    glyph.settings.pick_id = PickId(555);
    frame.scene.glyphs.push(glyph);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(555)));
}

#[test]
fn gpu_pick_hits_sprite_set() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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

    // One large world-space sprite at the origin. Sprite billboards are expanded
    // during the render vertex stage, which the pick pipeline reuses, so prepare
    // must run first to build the sprite buffers.
    let mut sprite = SpriteItem::default();
    sprite.positions = vec![[0.0, 0.0, 0.0]];
    sprite.default_size = 4.0;
    sprite.size_mode = SpriteSizeMode::WorldSpace;
    sprite.settings.pick_id = PickId(777);
    frame.scene.sprite_items.push(sprite);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(777)));
}

#[test]
fn gpu_pick_hits_polyline() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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

    // A thick polyline through the origin. Polylines expand to screen-space
    // ribbons in the render vertex stage, which the pick pipeline reuses, so
    // prepare must run first to build the segment buffer.
    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![2];
    polyline.line_width = 20.0;
    polyline.settings.pick_id = PickId(888);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(888)));
}

// ---------------------------------------------------------------------------
// GPU pick: sub-object identity (G8)
// ---------------------------------------------------------------------------

/// A 64x64 frame with the default orbit camera and no overlays, so world origin
/// projects to the screen centre (32, 32).
fn sub_object_pick_frame() -> FrameData {
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
    frame
}

#[test]
fn gpu_pick_glyph_resolves_instance() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // Three sphere glyphs spread along X. Instance index follows position order,
    // so the centre glyph under the cursor is instance 1. Instance-level picking
    // reads the instance_index channel, which needs no device feature.
    let mut glyph = GlyphItem::default();
    glyph.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    glyph.vectors = vec![[0.0, 0.0, 1.0]; 3];
    glyph.scale = 1.0;
    glyph.scale_by_magnitude = false;
    glyph.glyph_type = GlyphType::Sphere;
    glyph.settings.pick_id = PickId(555);
    frame.scene.glyphs.push(glyph);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::INSTANCE,
    );
    let hit = hit.expect("centre glyph should be hit");
    assert_eq!(hit.id, 555);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
}

#[test]
fn gpu_pick_polyline_resolves_segment() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // Three-node strip: segment 0 spans x in [-2, -1], segment 1 spans [-1, 2].
    // World origin (screen centre) lies on segment 1. Segment picking reads the
    // per-segment instance_index channel, so no device feature is needed.
    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![3];
    polyline.line_width = 20.0;
    polyline.settings.pick_id = PickId(888);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::SEGMENT,
    );
    let hit = hit.expect("polyline should be hit at the centre");
    assert_eq!(hit.id, 888);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Segment(1)));
}

#[test]
fn gpu_pick_polyline_resolves_strip_without_cpu_cache() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Deliberately leave the CPU pick cache OFF: strip resolution must come from
    // the persistent PolylineGpuData::strip_lengths, not pick_polyline_items.
    let mut frame = sub_object_pick_frame();

    // Two strips. Strip 0 is a single off-centre segment (global segment 0).
    // Strip 1 is three nodes whose middle segment (global segment 2) crosses the
    // world origin, which the camera centre projects onto.
    let mut polyline = PolylineItem::default();
    polyline.positions = vec![
        [-5.0, 3.0, 0.0],
        [-4.0, 3.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ];
    polyline.strip_lengths = vec![2, 3];
    polyline.line_width = 20.0;
    polyline.settings.pick_id = PickId(889);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::STRIP,
        )
        .expect("polyline should be hit at the centre");
    assert_eq!(hit.id, 889);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Strip(1)));
}

#[test]
fn gpu_pick_surface_resolves_face_and_vertex() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // No CPU pick cache: face and vertex resolve from the per-object surface
    // metadata copied off the frame plus the mesh geometry in mesh_store.
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.pick_id = PickId(321);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // FACE: the read-back triangle index is a valid face of the 12-triangle box.
    let face_hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::FACE,
        )
        .expect("box should be hit");
    assert_eq!(face_hit.id, 321);
    match face_hit.sub_object {
        Some(viewport_lib::SubObjectRef::Face(f)) => assert!(f < 12, "face {f} out of range"),
        other => panic!("expected a Face sub-object, got {other:?}"),
    }

    // VERTEX: refines the hit face to its nearest corner (a valid box vertex).
    let vertex_hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::VERTEX,
        )
        .expect("box should be hit");
    match vertex_hit.sub_object {
        Some(viewport_lib::SubObjectRef::Vertex(v)) => assert!(v < 8, "vertex {v} out of range"),
        other => panic!("expected a Vertex sub-object, got {other:?}"),
    }
}

#[test]
fn pick_auto_backend_routes_to_gpu_when_feature_present() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Auto must not need the CPU pick cache on a device that can resolve sub-object
    // picking on the GPU.
    let mut frame = sub_object_pick_frame();

    assert!(
        renderer.gpu_sub_object_supported(&device),
        "adapter with SHADER_PRIMITIVE_INDEX should support GPU sub-object picking"
    );

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.pick_id = PickId(777);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // Object-level Auto: routes to the GPU and returns the object.
    let obj = renderer
        .pick_object(
            PickBackend::Auto,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::OBJECT,
        )
        .expect("box should be hit");
    assert_eq!(obj.id, 777);

    // Sub-object Auto: the feature is present, so it stays on the GPU and resolves
    // the vertex without any CPU pick cache being enabled.
    let vertex = renderer
        .pick_object(
            PickBackend::Auto,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::VERTEX,
        )
        .expect("box should be hit");
    assert_eq!(vertex.id, 777);
    match vertex.sub_object {
        Some(viewport_lib::SubObjectRef::Vertex(v)) => assert!(v < 8, "vertex {v} out of range"),
        other => panic!("expected a Vertex sub-object from Auto, got {other:?}"),
    }
}

#[test]
fn gpu_pick_volume_mesh_resolves_cell_without_cpu_cache() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // No CPU pick cache: the boundary-face-to-cell map is copied off the frame
    // into the pick's surface metadata, so cell picking needs no retained clone.
    let mut frame = sub_object_pick_frame();

    // One hexahedron spanning [-0.5, 0.5]^3 in VTK vertex order. Every boundary
    // face belongs to cell 0.
    let mut data = viewport_lib::VolumeMeshData::default();
    data.positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    data.cells = vec![[0, 1, 2, 3, 4, 5, 6, 7]];

    let mut item = renderer
        .resources_mut()
        .upload_volume_mesh(&device, &data)
        .expect("upload volume mesh");
    item.settings.pick_id = PickId(654);
    frame.scene.volume_meshes.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::CELL,
        )
        .expect("volume mesh should be hit at the centre");
    assert_eq!(hit.id, 654);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Cell(0)));
}

#[test]
fn gpu_pick_surface_resolves_edge() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.pick_id = PickId(321);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::EDGE,
        )
        .expect("box should be hit");
    // Edge id is `face * 3 + local_edge`; the box has 12 triangles, 36 local edges.
    match hit.sub_object {
        Some(viewport_lib::SubObjectRef::Edge(e)) => assert!(e < 36, "edge {e} out of range"),
        other => panic!("expected an Edge sub-object, got {other:?}"),
    }
    // The snap position is the closest point on that edge, inside the box bounds.
    let snap = hit
        .sub_object_world_pos
        .expect("edge pick should fill the snap position");
    for c in [snap.x, snap.y, snap.z] {
        assert!(c.abs() <= 0.5 + 1e-4, "edge snap {snap:?} outside the box");
    }
    // The nearest point must lie on the box surface (at least one face coord).
    assert!(
        [snap.x, snap.y, snap.z]
            .iter()
            .any(|c| (c.abs() - 0.5).abs() < 1e-3),
        "edge snap {snap:?} should be on a box face"
    );
}

#[test]
fn gpu_pick_vertex_fills_snap_world_pos() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.pick_id = PickId(321);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::VERTEX,
        )
        .expect("box should be hit");
    assert!(matches!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Vertex(_))
    ));
    // The snap position is the chosen corner of the [-0.5, 0.5]^3 box.
    let snap = hit
        .sub_object_world_pos
        .expect("vertex pick should fill the snap position");
    for c in [snap.x, snap.y, snap.z] {
        assert!(
            (c.abs() - 0.5).abs() < 1e-4,
            "corner coord {c} is not +/-0.5 (snap {snap:?})"
        );
    }
    // The snap position is the corner, distinct from the raw hit point on the face.
    assert!(
        (snap - hit.world_pos).length() > 1e-3,
        "snap should be the corner, not the face hit point"
    );
}

#[test]
fn gpu_pick_face_fills_geometric_normal() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.pick_id = PickId(321);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::FACE,
        )
        .expect("box should be hit");
    assert!(matches!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Face(_))
    ));
    // The normal is a real unit axis of the box (an axis-aligned face), oriented
    // to point back toward the camera rather than the camera-facing stand-in.
    assert!(
        (hit.normal.length() - 1.0).abs() < 1e-4,
        "normal should be unit length, got {:?}",
        hit.normal
    );
    let max_component = hit
        .normal
        .to_array()
        .iter()
        .map(|c| c.abs())
        .fold(0.0f32, f32::max);
    assert!(
        (max_component - 1.0).abs() < 1e-4,
        "expected an axis-aligned box face normal, got {:?}",
        hit.normal
    );
    let eye = frame.camera.render_camera.eye_position;
    let to_eye = glam::Vec3::from(eye) - hit.world_pos;
    assert!(
        hit.normal.dot(to_eye) > 0.0,
        "face normal should point toward the camera, got {:?}",
        hit.normal
    );
}

#[test]
fn gpu_snap_query_snaps_to_surface_vertex() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.pick_id = PickId(321);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // Cursor a couple of pixels off centre, with a tolerance wide enough to reach
    // the box. The window snaps to the nearest box corner even though the cursor
    // is not exactly on it.
    let snap = renderer
        .snap_query(
            glam::Vec2::new(30.0, 30.0),
            16.0,
            &frame,
            &device,
            &queue,
            PickMask::VERTEX,
        )
        .expect("box within the tolerance window should snap");
    assert_eq!(snap.object_id, 321);
    match snap.sub_object {
        Some(viewport_lib::SubObjectRef::Vertex(v)) => assert!(v < 8, "vertex {v} out of range"),
        other => panic!("expected a Vertex sub-object, got {other:?}"),
    }
    // The snap position is a corner of the [-0.5, 0.5]^3 box.
    for c in [snap.world_pos.x, snap.world_pos.y, snap.world_pos.z] {
        assert!(
            (c.abs() - 0.5).abs() < 1e-4,
            "corner coord {c} is not +/-0.5 (snap {:?})",
            snap.world_pos
        );
    }
}

#[test]
fn gpu_snap_query_empty_scene_returns_none() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // No pickable geometry: nothing draws into the window, so the snap misses.
    let frame = sub_object_pick_frame();
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let snap = renderer.snap_query(
        glam::Vec2::new(32.0, 32.0),
        16.0,
        &frame,
        &device,
        &queue,
        PickMask::VERTEX,
    );
    assert!(snap.is_none(), "empty scene should not snap to anything");
}

#[test]
fn gpu_pick_curve_node_fills_snap_world_pos() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::POLY_NODE,
        )
        .expect("ribbon should be hit");
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
    // Node 1 sits at the world origin; the snap position must be that node.
    let snap = hit
        .sub_object_world_pos
        .expect("node pick should fill the snap position");
    assert!(
        snap.length() < 1e-4,
        "node 1 should be at the origin, got {snap:?}"
    );
}

// ---------------------------------------------------------------------------
// GPU pick: non-rasterized surface types (G7c/G7d)
// ---------------------------------------------------------------------------

#[test]
fn gpu_pick_hits_implicit_surface() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // One SDF sphere of radius 1.5 at the origin. The pick pass raymarches the
    // isosurface on a full-screen quad and writes the item's pick id at the hit.
    let prim = viewport_lib::ImplicitPrimitive {
        kind: 1, // sphere
        blend: 0.0,
        _pad: [0.0; 2],
        params: [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        colour: [1.0, 1.0, 1.0, 1.0],
    };
    let mut item = viewport_lib::GpuImplicitItem::default();
    item.primitives.push(prim);
    item.settings.pick_id = PickId(909);
    frame.scene.gpu_implicit.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(909)));
}

#[test]
fn gpu_pick_hits_marching_cubes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // A radial scalar field centred at the origin: the isosurface at value 1.5 is
    // a sphere of radius 1.5. Grid spans roughly [-2.3, 2.3]^3.
    let dims = [24u32, 24, 24];
    let spacing = [0.2f32; 3];
    let origin = [-(23.0 * 0.2) / 2.0; 3];
    let mut data = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * spacing[0];
                let wy = origin[1] + y as f32 * spacing[1];
                let wz = origin[2] + z as f32 * spacing[2];
                let idx = (x + y * dims[0] + z * dims[0] * dims[1]) as usize;
                data[idx] = (wx * wx + wy * wy + wz * wz).sqrt();
            }
        }
    }
    let vol = viewport_lib::VolumeData {
        data,
        dims,
        origin,
        spacing,
    };
    let volume_id = renderer
        .resources_mut()
        .upload_volume_for_mc(&device, &queue, &vol)
        .expect("mc volume upload");

    let mut job = viewport_lib::GpuMarchingCubesJob {
        volume_id,
        isovalue: 1.5,
        material: Material::default(),
        settings: ItemSettings::default(),
        cpu_data: None,
    };
    job.settings.pick_id = PickId(717);
    frame.scene.gpu_mc_jobs.push(job);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(717)));
}

// ---------------------------------------------------------------------------
// GPU pick: item-type plugin hook (render_pick)
// ---------------------------------------------------------------------------

/// Minimal plugin collection carrying a single pickable item's settings.
struct MockPickCollection {
    settings: ItemSettings,
}

impl MockPickCollection {
    fn new(pick_id: PickId) -> Self {
        let mut settings = ItemSettings::default();
        settings.pick_id = pick_id;
        Self { settings }
    }
}

impl PluginItemCollection for MockPickCollection {
    fn len(&self) -> usize {
        1
    }
    fn item_settings(&self, _index: usize) -> &ItemSettings {
        &self.settings
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Vertex stage: a fullscreen triangle that reads its pick id from a group-1
/// uniform and forwards it flat. Concatenated with `SHARED_PICK_WGSL`, whose
/// `viewport_pick_fs` writes the three pick channels.
const MOCK_PICK_VS: &str = r#"
@group(1) @binding(0) var<uniform> mock_pick_id: vec4<u32>;

struct MockVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) pick_id: u32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> MockVsOut {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    var out: MockVsOut;
    out.pos = vec4<f32>(verts[vi], 0.0, 1.0);
    out.pick_id = mock_pick_id.x;
    return out;
}
"#;

/// Plugin that rasterises a fullscreen triangle into the pick pass, tagging it
/// with its item's pick id. Hand-rolls the pipeline the way a real plugin does
/// (plugins get `SharedBindings`, not `DeviceResources`, at `init_gpu`).
struct MockPickPlugin {
    pipeline: Option<wgpu::RenderPipeline>,
    id_bgl: Option<wgpu::BindGroupLayout>,
    id_bg: Option<wgpu::BindGroup>,
}

impl MockPickPlugin {
    fn new() -> Self {
        Self {
            pipeline: None,
            id_bgl: None,
            id_bg: None,
        }
    }
}

impl ItemTypePlugin for MockPickPlugin {
    fn type_name(&self) -> &'static str {
        "mock_pick"
    }

    fn init_gpu(&mut self, device: &wgpu::Device, shared: &SharedBindings<'_>) {
        let id_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mock_pick_id_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let source = format!("{MOCK_PICK_VS}\n{SHARED_PICK_WGSL}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mock_pick_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        // Build through the library's version-portability helpers (see
        // `viewport_lib::wgpu`) so this test stays free of per-wgpu-version cfg.
        let layout = viewport_lib::wgpu::pipeline_layout(
            &device,
            "mock_pick_layout",
            &[shared.group0_layout, &id_bgl],
        );

        let color = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let pipeline = viewport_lib::wgpu::render_pipeline(
            &device,
            viewport_lib::wgpu::RenderPipelineDesc {
                label: "mock_pick_pipeline",
                layout: &layout,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("viewport_pick_fs"),
                    targets: &[
                        color(PICK_COLOR_FORMAT),
                        color(PICK_COLOR_FORMAT),
                        color(PICK_DEPTH_CHANNEL_FORMAT),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(viewport_lib::wgpu::depth_stencil(
                    SCENE_DEPTH_FORMAT,
                    true,
                    wgpu::CompareFunction::LessEqual,
                )),
                multisample: wgpu::MultisampleState::default(),
                cache: None,
            },
        );

        self.id_bgl = Some(id_bgl);
        self.pipeline = Some(pipeline);
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _ctx: &viewport_lib::plugin_api::ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(coll) = items.as_any().downcast_ref::<MockPickCollection>() else {
            return Vec::new();
        };
        let Some(id_bgl) = self.id_bgl.as_ref() else {
            return Vec::new();
        };
        let id = [coll.settings.pick_id.0 as u32, 0, 0, 0];
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mock_pick_id_buf"),
            size: std::mem::size_of_val(&id) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&id));
        self.id_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mock_pick_id_bg"),
            layout: id_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        }));
        Vec::new()
    }

    fn render_pick<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        _ctx: &PickPassContext<'a>,
        items: &'a dyn PluginItemCollection,
    ) {
        let (Some(pipeline), Some(id_bg)) = (self.pipeline.as_ref(), self.id_bg.as_ref()) else {
            return;
        };
        let Some(coll) = items.as_any().downcast_ref::<MockPickCollection>() else {
            return;
        };
        if coll.settings.hidden || coll.settings.pick_id == PickId::NONE {
            return;
        }
        // Group 0 (camera) is already bound by the lib.
        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, id_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn plugin_pick_frame() -> FrameData {
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
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

#[test]
fn gpu_pick_returns_plugin_item_id() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.with_item_type_plugin(&device, Box::new(MockPickPlugin::new()));

    let mut frame = plugin_pick_frame();
    frame
        .scene
        .submit_plugin_items("mock_pick", MockPickCollection::new(PickId(321)));

    // prepare runs the plugin's `prepare` (builds its id bind group) and updates
    // the shared camera bind group the pick pass binds at group 0.
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::all(),
    );
    assert_eq!(hit.map(|h| h.id), Some(321));
}

#[test]
fn gpu_pick_skips_hidden_plugin_item() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.with_item_type_plugin(&device, Box::new(MockPickPlugin::new()));

    let mut frame = plugin_pick_frame();
    let mut coll = MockPickCollection::new(PickId(321));
    coll.settings.hidden = true;
    frame.scene.submit_plugin_items("mock_pick", coll);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::all(),
    );
    assert!(hit.is_none(), "hidden plugin item must not be pickable");
}

// ---------------------------------------------------------------------------
// GPU pick: item-type plugin sub-object refinement (resolve_sub_object)
// ---------------------------------------------------------------------------

/// Vertex stage for the sub-object pick plugin: a screen-covering quad as two
/// clip-space triangles (0-2 below the y = x diagonal, 3-5 above it), pick id
/// from a group-1 uniform. Concatenated with `SHARED_PICK_PRIM_WGSL`, whose
/// `viewport_pick_prim_fs` writes the rasterised triangle index into the
/// primitive channel.
const SUB_PICK_VS: &str = r#"
@group(1) @binding(0) var<uniform> sub_pick_id: vec4<u32>;

struct SubPickVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) pick_id: u32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> SubPickVsOut {
    var verts = array<vec2<f32>, 6>(
        // Triangle 0: below the y = x diagonal.
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        // Triangle 1: above it.
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    var out: SubPickVsOut;
    out.pos = vec4<f32>(verts[vi], 0.0, 1.0);
    out.pick_id = sub_pick_id.x;
    return out;
}
"#;

/// Plugin whose pick draw writes real triangle indices and whose
/// `resolve_sub_object` refines them: FACE maps the triangle index straight to
/// a face, VERTEX returns the hit triangle's first corner (index * 3) and
/// records the world position it was handed so tests can check the depth
/// reconstruction plumbing.
struct SubPickPlugin {
    pipeline: Option<wgpu::RenderPipeline>,
    id_bgl: Option<wgpu::BindGroupLayout>,
    id_bg: Option<wgpu::BindGroup>,
    last_world: std::sync::Arc<std::sync::Mutex<Option<glam::Vec3>>>,
}

impl SubPickPlugin {
    fn new(last_world: std::sync::Arc<std::sync::Mutex<Option<glam::Vec3>>>) -> Self {
        Self {
            pipeline: None,
            id_bgl: None,
            id_bg: None,
            last_world,
        }
    }
}

impl ItemTypePlugin for SubPickPlugin {
    fn type_name(&self) -> &'static str {
        "sub_pick"
    }

    fn init_gpu(&mut self, device: &wgpu::Device, shared: &SharedBindings<'_>) {
        use viewport_lib::plugin_api::shared_wgsl::{PICK_PRIM_ENABLE_WGSL, SHARED_PICK_PRIM_WGSL};

        if !device
            .features()
            .contains(viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE)
        {
            return;
        }

        let id_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sub_pick_id_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let source = format!("{PICK_PRIM_ENABLE_WGSL}{SUB_PICK_VS}\n{SHARED_PICK_PRIM_WGSL}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sub_pick_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let layout = viewport_lib::wgpu::pipeline_layout(
            &device,
            "sub_pick_layout",
            &[shared.group0_layout, &id_bgl],
        );

        let color = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let pipeline = viewport_lib::wgpu::render_pipeline(
            &device,
            viewport_lib::wgpu::RenderPipelineDesc {
                label: "sub_pick_pipeline",
                layout: &layout,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("viewport_pick_prim_fs"),
                    targets: &[
                        color(PICK_COLOR_FORMAT),
                        color(PICK_COLOR_FORMAT),
                        color(PICK_DEPTH_CHANNEL_FORMAT),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(viewport_lib::wgpu::depth_stencil(
                    SCENE_DEPTH_FORMAT,
                    true,
                    wgpu::CompareFunction::LessEqual,
                )),
                multisample: wgpu::MultisampleState::default(),
                cache: None,
            },
        );

        self.id_bgl = Some(id_bgl);
        self.pipeline = Some(pipeline);
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _ctx: &viewport_lib::plugin_api::ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(coll) = items.as_any().downcast_ref::<MockPickCollection>() else {
            return Vec::new();
        };
        let Some(id_bgl) = self.id_bgl.as_ref() else {
            return Vec::new();
        };
        let id = [coll.settings.pick_id.0 as u32, 0, 0, 0];
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sub_pick_id_buf"),
            size: std::mem::size_of_val(&id) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&id));
        self.id_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sub_pick_id_bg"),
            layout: id_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        }));
        Vec::new()
    }

    fn render_pick<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        _ctx: &PickPassContext<'a>,
        items: &'a dyn PluginItemCollection,
    ) {
        let (Some(pipeline), Some(id_bg)) = (self.pipeline.as_ref(), self.id_bg.as_ref()) else {
            return;
        };
        let Some(coll) = items.as_any().downcast_ref::<MockPickCollection>() else {
            return;
        };
        if coll.settings.hidden || coll.settings.pick_id == PickId::NONE {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, id_bg, &[]);
        pass.draw(0..6, 0..1);
    }

    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        primitive_index: u32,
        world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<viewport_lib::SubObjectRef> {
        *self.last_world.lock().unwrap() = Some(world_pos);
        if mask.intersects(PickMask::VERTEX) {
            Some(viewport_lib::SubObjectRef::Vertex(primitive_index * 3))
        } else if mask.intersects(PickMask::FACE) {
            Some(viewport_lib::SubObjectRef::Face(primitive_index))
        } else {
            None
        }
    }
}

fn sub_pick_setup(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pick_id: u64,
) -> (
    ViewportRenderer,
    FrameData,
    std::sync::Arc<std::sync::Mutex<Option<glam::Vec3>>>,
) {
    let last_world = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.with_item_type_plugin(device, Box::new(SubPickPlugin::new(last_world.clone())));
    let mut frame = plugin_pick_frame();
    frame
        .scene
        .submit_plugin_items("sub_pick", MockPickCollection::new(PickId(pick_id)));
    let _ = renderer.pass().prepare(device, queue, &frame);
    (renderer, frame, last_world)
}

#[test]
fn gpu_pick_plugin_resolves_face_via_hook() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no GPU adapter with primitive-index support");
        return;
    };
    let (mut renderer, frame, _) = sub_pick_setup(&device, &queue, 611);

    // (48, 32) is below the quad's y = x diagonal in NDC: triangle 0.
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(48.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT | PickMask::FACE,
    );
    let hit = hit.expect("plugin quad under cursor");
    assert_eq!(hit.id, 611);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Face(0)));

    // (16, 32) is above the diagonal: triangle 1.
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(16.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT | PickMask::FACE,
    );
    assert_eq!(
        hit.and_then(|h| h.sub_object),
        Some(viewport_lib::SubObjectRef::Face(1))
    );
}

#[test]
fn gpu_pick_plugin_vertex_hook_gets_world_position() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no GPU adapter with primitive-index support");
        return;
    };
    let (mut renderer, frame, last_world) = sub_pick_setup(&device, &queue, 612);

    let cursor = glam::Vec2::new(48.0, 32.0);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        cursor,
        &frame,
        &device,
        &queue,
        PickMask::VERTEX,
    );
    assert_eq!(
        hit.and_then(|h| h.sub_object),
        Some(viewport_lib::SubObjectRef::Vertex(0)),
        "triangle 0's first corner via the hook"
    );

    // The hook's world position is the cursor pixel un-projected at the quad's
    // clip-space depth (0.0): same reconstruction to_pick_hit performs.
    let recorded = last_world
        .lock()
        .unwrap()
        .expect("resolve_sub_object was called");
    let view_proj_inv = frame.camera.render_camera.view_proj().inverse();
    let ndc = glam::Vec3::new(
        (cursor.x / 64.0) * 2.0 - 1.0,
        1.0 - (cursor.y / 64.0) * 2.0,
        0.0,
    );
    let expected = view_proj_inv.project_point3(ndc);
    assert!(
        (recorded - expected).length() < 1e-3,
        "hook world_pos {recorded:?} != reconstructed {expected:?}"
    );
}

#[test]
fn gpu_pick_plugin_without_hook_stays_object_level() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no GPU adapter with primitive-index support");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.with_item_type_plugin(&device, Box::new(MockPickPlugin::new()));

    let mut frame = plugin_pick_frame();
    frame
        .scene
        .submit_plugin_items("mock_pick", MockPickCollection::new(PickId(613)));
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // A sub-object-only mask now draws the plugin (it competes in the depth
    // test), but without `resolve_sub_object` the hit stays object-level.
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::FACE,
    );
    let hit = hit.expect("plugin drawn under a FACE-only mask");
    assert_eq!(hit.id, 613);
    assert_eq!(hit.sub_object, None);
}

#[test]
fn gpu_pick_rect_resolves_plugin_faces() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no GPU adapter with primitive-index support");
        return;
    };
    let (mut renderer, frame, _) = sub_pick_setup(&device, &queue, 614);

    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT | PickMask::FACE,
    );
    assert!(result.objects.contains(&614));
    for face in [0u32, 1] {
        assert!(
            result
                .elements
                .contains(&(614, viewport_lib::SubObjectRef::Face(face))),
            "rect elements missing Face({face}): {:?}",
            result.elements
        );
    }
}

// ---------------------------------------------------------------------------
// GPU pick: point clouds and Gaussian splats (G3d), image slices and volume
// surface slices (G3e)
// ---------------------------------------------------------------------------

#[test]
fn gpu_pick_point_cloud_resolves_point() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // Three points spread along X. The centre point (index 1) sits at world
    // origin, under the cursor. CLOUD_POINT picking reads the forwarded
    // instance_index, which needs no device feature.
    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    cloud.point_size = 20.0;
    cloud.settings.pick_id = PickId(444);
    frame.scene.point_clouds.push(cloud);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::CLOUD_POINT,
    );
    let hit = hit.expect("centre point should be hit");
    assert_eq!(hit.id, 444);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
}

#[test]
fn gpu_pick_splat_resolves_splat() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // Three splats spread along X, large enough to cover the centre pixel.
    // The centre splat (index 1) sits at world origin, under the cursor.
    let mut data = GaussianSplatData::default();
    data.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    data.scales = vec![[0.5, 0.5, 0.5]; 3];
    data.rotations = vec![[0.0, 0.0, 0.0, 1.0]; 3];
    data.opacities = vec![1.0; 3];
    data.sh_coefficients = vec![0.0; 9];
    data.sh_degree = ShDegree::Zero;
    let splat_id = renderer
        .resources_mut()
        .upload_gaussian_splat(&device, &queue, &data)
        .expect("upload splat set");

    let mut item = GaussianSplatItem::default();
    item.source = splat_id;
    item.settings.pick_id = PickId(777);
    frame.scene.gaussian_splats.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::SPLAT,
    );
    let hit = hit.expect("centre splat should be hit");
    assert_eq!(hit.id, 777);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Splat(1)));
}

#[test]
fn gpu_pick_hits_image_slice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &[0.5; 8], [2, 2, 2]);

    let mut slice = ImageSliceItem::default();
    slice.volume_id = volume_id;
    slice.axis = SliceAxis::Z;
    slice.offset = 0.5;
    slice.bbox_min = [-1.0, -1.0, -1.0];
    slice.bbox_max = [1.0, 1.0, 1.0];
    slice.settings.pick_id = PickId(222);
    frame.scene.image_slices.push(slice);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(222));
}

#[test]
fn gpu_pick_hits_volume_surface_slice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &[0.5; 8], [2, 2, 2]);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");

    let mut slice = VolumeSurfaceSliceItem::default();
    slice.volume_id = volume_id;
    slice.mesh_id = mesh_id;
    slice.bbox_min = [-1.0, -1.0, -1.0];
    slice.bbox_max = [1.0, 1.0, 1.0];
    slice.settings.pick_id = PickId(333);
    frame.scene.volume_surface_slices.push(slice);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(333));
}

#[test]
fn gpu_pick_rect_returns_unique_object_ids() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.pick_id = PickId(654);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // A rect spanning the whole 64x64 frame should touch the centred box.
    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(result.objects, vec![654]);
    assert!(
        result.elements.is_empty(),
        "GPU rect pick is object-level only"
    );

    // A mask with no OBJECT bit returns no objects, but does resolve sub-objects
    // where the primitive channel names them. Face resolution needs
    // SHADER_PRIMITIVE_INDEX, so elements are either empty (feature absent) or all
    // faces of the box.
    let faces = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::FACE,
    );
    assert!(faces.objects.is_empty(), "no OBJECT bit -> no objects");
    assert!(
        faces
            .elements
            .iter()
            .all(|(id, sub)| *id == 654 && matches!(sub, viewport_lib::SubObjectRef::Face(_))),
        "FACE rect elements must all be faces of the box"
    );
}

#[test]
fn gpu_pick_rect_resolves_point_cloud_elements() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // Three fat points spread across the centre of the view.
    let mut pc = PointCloudItem::default();
    pc.positions = vec![[-1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]];
    pc.point_size = 24.0;
    pc.settings.pick_id = PickId(500);
    frame.scene.point_clouds.push(pc);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // A CLOUD_POINT rect over the whole frame collects point sub-objects and no
    // objects (the mask carries no OBJECT bit). Point sub-objects come from the
    // instance index, so this needs no device feature.
    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::CLOUD_POINT,
    );
    assert!(
        result.objects.is_empty(),
        "CLOUD_POINT mask carries no OBJECT bit"
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect point sub-objects"
    );
    assert!(
        result
            .elements
            .iter()
            .all(|(id, sub)| *id == 500 && matches!(sub, viewport_lib::SubObjectRef::Point(_))),
        "every element must be a point of the cloud"
    );
}

#[test]
fn gpu_pick_rect_resolves_surface_vertex() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // No CPU pick cache: the VERTEX variant writes the nearest corner index per
    // pixel, so a rect reads it straight from the primitive channel.
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.settings.pick_id = PickId(321);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::VERTEX,
    );
    assert!(
        result.objects.is_empty(),
        "VERTEX mask carries no OBJECT bit"
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect vertex sub-objects"
    );
    assert!(
        result.elements.iter().all(|(id, sub)| *id == 321
            && matches!(sub, viewport_lib::SubObjectRef::Vertex(v) if *v < 8)),
        "every element must be a box vertex, got {:?}",
        result.elements
    );
}

#[test]
fn gpu_pick_rect_resolves_curve_node() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // No CPU pick cache: the POLY_NODE variant writes the nearest node index per
    // pixel, so a rect reads it straight from the primitive channel.
    let mut frame = sub_object_pick_frame();

    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::POLY_NODE,
    );
    assert!(
        result.objects.is_empty(),
        "POLY_NODE mask carries no OBJECT bit"
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect node sub-objects"
    );
    assert!(
        result.elements.iter().all(|(id, sub)| *id == 4242
            && matches!(sub, viewport_lib::SubObjectRef::Point(n) if *n < 3)),
        "every element must be a ribbon node, got {:?}",
        result.elements
    );
    // The middle node (index 1) sits under the centre of the rect, so it must be
    // among the collected nodes.
    assert!(
        result
            .elements
            .iter()
            .any(|(_, sub)| *sub == viewport_lib::SubObjectRef::Point(1)),
        "the middle node should be collected, got {:?}",
        result.elements
    );
}

#[test]
fn gpu_pick_hits_top_right_scaled_screen_image() {
    // Mirrors showcase 33's overlay: a TopRight-anchored image with scale > 1,
    // driven through the real pick entry points (point and rect), not the free
    // functions.
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // 8x8 at scale 2 = 16x16 effective, pinned to the top-right of the 64x64
    // viewport: screen rect x in [48, 64], y in [0, 16].
    let mut image = ScreenImageItem::default();
    image.pixels = vec![[255, 255, 255, 255]; 8 * 8];
    image.width = 8;
    image.height = 8;
    image.scale = 2.0;
    image.anchor = ImageAnchor::TopRight;
    image.settings.pick_id = PickId(52);
    frame.scene.screen_images.push(image);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // Point: a click inside the top-right rect selects it.
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(56.0, 8.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(
        hit.map(|h| h.id),
        Some(52),
        "top-right overlay should be hit"
    );

    // Point: a click in the opposite corner misses it.
    let miss = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(4.0, 60.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert!(
        miss.is_none(),
        "bottom-left click should not hit the overlay"
    );

    // Screen overlays are OBJECT-only on both backends: a sub-object-only mask
    // (no OBJECT bit) returns nothing, and the GPU backend agrees with the CPU
    // backend. A consumer that wants the overlay under a sub-object query includes
    // OBJECT in the mask.
    renderer.set_cpu_pick_cache(true);
    let _ = renderer.pass().prepare(&device, &queue, &frame);
    for backend in [PickBackend::Gpu, PickBackend::Cpu] {
        let sub = renderer.pick_object(
            backend,
            glam::Vec2::new(56.0, 8.0),
            &frame,
            &device,
            &queue,
            PickMask::POINT_LIKE,
        );
        assert!(
            sub.is_none(),
            "{backend:?}: overlay is OBJECT-only, POINT_LIKE should not select it"
        );
    }

    // Rect: a rubber band over the top-right corner collects the overlay.
    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(50.0, 0.0),
        glam::Vec2::new(64.0, 14.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert!(
        result.objects.contains(&52),
        "rect over the corner should collect the overlay, got {:?}",
        result.objects
    );
}

#[test]
fn gpu_pick_hits_screen_image() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut image = ScreenImageItem::default();
    image.pixels = vec![[255, 255, 255, 255]; 16 * 16];
    image.width = 16;
    image.height = 16;
    image.anchor = ImageAnchor::Center;
    image.settings.pick_id = PickId(444);
    frame.scene.screen_images.push(image);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // Centre of the 64x64 viewport lands inside the 16x16 centred image.
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(444));

    // Corner of the viewport falls outside the centred image.
    let miss = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(1.0, 1.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert!(miss.is_none());
}

#[test]
fn gpu_pick_rect_hits_screen_image() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut image = ScreenImageItem::default();
    image.pixels = vec![[255, 255, 255, 255]; 16 * 16];
    image.width = 16;
    image.height = 16;
    image.anchor = ImageAnchor::Center;
    image.settings.pick_id = PickId(555);
    frame.scene.screen_images.push(image);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // A rect spanning the whole viewport touches the centred image.
    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(result.objects, vec![555]);

    // A rect confined to a corner, away from the centred image, misses it.
    let miss = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(4.0, 4.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert!(miss.objects.is_empty());
}

// ---------------------------------------------------------------------------
// CPU pick: cached mesh-local TriMesh
// ---------------------------------------------------------------------------

#[test]
fn cpu_pick_hits_mesh_via_cached_trimesh() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.settings.pick_id = PickId(111);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Cpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(111));
}

#[test]
fn cpu_pick_matches_gpu_pick_under_non_uniform_scale() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    // Non-uniform scale: the cached TriMesh is mesh-local, so this exercises
    // the inverse-model ray transform and the inverse-transpose normal fix-up
    // in `renderer.pick()` section 1.
    item.model = glam::Mat4::from_scale(glam::Vec3::new(1.0, 2.0, 3.0)).to_cols_array_2d();
    item.settings.pick_id = PickId(222);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let gpu_hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::OBJECT,
        )
        .expect("scaled box should be hit (gpu)");
    let cpu_hit = renderer
        .pick_object(
            PickBackend::Cpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::OBJECT,
        )
        .expect("scaled box should be hit (cpu)");

    assert_eq!(cpu_hit.id, 222);
    assert_eq!(cpu_hit.id, gpu_hit.id);

    // GPU depth reconstruction reads back a single quantized pixel while the
    // CPU ray is cast through the exact continuous click position, so on an
    // oblique default camera the two land a fraction of a pixel apart even
    // with no bug involved (this reproduces at identity scale too). A loose
    // bound confirms they hit the same face, not a stale or mistransformed one.
    assert!(
        cpu_hit.world_pos.distance(gpu_hit.world_pos) < 0.15,
        "cpu {:?} vs gpu {:?}",
        cpu_hit.world_pos,
        gpu_hit.world_pos
    );

    // Precise check: map the CPU hit back into mesh-local space through the
    // known model matrix. Regardless of the non-uniform scale applied, the
    // true hit point must land exactly on the unit box's surface (one
    // component at +/-0.5). This is what actually exercises correctness of
    // the inverse-model ray transform and the inverse-transpose normal
    // fix-up, without depending on GPU pixel quantization.
    let model = glam::Mat4::from_scale(glam::Vec3::new(1.0, 2.0, 3.0));
    let local_hit = model.inverse().transform_point3(cpu_hit.world_pos);
    let max_component = local_hit
        .x
        .abs()
        .max(local_hit.y.abs())
        .max(local_hit.z.abs());
    assert!(
        (max_component - 0.5).abs() < 1e-4,
        "expected the hit to land on the unit box surface in local space, got {local_hit:?}"
    );
}

#[test]
fn cpu_pick_reuses_cached_trimesh_across_instances() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");

    // Two instances of the same mesh_id at different positions along the ray
    // through screen centre; the nearer one (closer to the camera) should win.
    let mut near_item = SceneRenderItem::default();
    near_item.mesh_id = mesh_id;
    near_item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    near_item.settings.pick_id = PickId(11);

    let mut far_item = SceneRenderItem::default();
    far_item.mesh_id = mesh_id;
    far_item.model =
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -3.0)).to_cols_array_2d();
    far_item.settings.pick_id = PickId(22);

    frame.scene.surfaces = SurfaceSubmission::Flat(vec![near_item, far_item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Cpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(11));
}

#[test]
fn cpu_pick_invalidates_cache_after_replace_mesh_data() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.settings.pick_id = PickId(333);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let first = renderer
        .pick_object(
            PickBackend::Cpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::OBJECT,
        )
        .expect("unit box should be hit");

    // Replace with a much bigger box (unit box scaled 10x baked into vertex
    // data), same PickId, no scale on the item transform. If the cached
    // TriMesh were not invalidated by `content_rev`, this would still read
    // back the old unit-box surface.
    let mut big_box = box_mesh();
    for p in &mut big_box.positions {
        p[0] *= 10.0;
        p[1] *= 10.0;
        p[2] *= 10.0;
    }
    renderer
        .resources_mut()
        .replace_mesh_data(&device, &queue, mesh_id, &big_box)
        .expect("replace mesh data");

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let second = renderer
        .pick_object(
            PickBackend::Cpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::OBJECT,
        )
        .expect("bigger box should be hit");

    assert!(
        second.world_pos.distance(first.world_pos) > 1.0,
        "expected the bigger box surface to be hit further out: first {:?}, second {:?}",
        first.world_pos,
        second.world_pos
    );
}
