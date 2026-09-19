//! Picking for the volume item type: GPU pick-id, voxel-level refinement, and
//! a showcase-shaped field.
//!
//! One file per item type, so a type's coverage travels with it.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

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
fn cpu_pick_hits_voxel_volume() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);

    // The CPU path marches `volume_data`, not the uploaded texture, so the item
    // has to carry the same field it was uploaded with.
    let dims = [8u32, 8, 8];
    let data = vec![1.0f32; (dims[0] * dims[1] * dims[2]) as usize];
    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &data, dims);

    let mut frame = sub_object_pick_frame();
    let mut vol = VolumeItem::default();
    vol.volume_id = volume_id;
    vol.volume_data = Some(std::sync::Arc::new(viewport_lib::VolumeData {
        data,
        dims,
        origin: [-0.5, -0.5, -0.5],
        spacing: [0.125, 0.125, 0.125],
    }));
    vol.scalar_range = (0.0, 1.0);
    vol.threshold_min = 0.0;
    vol.threshold_max = 1.0;
    vol.bbox_min = [-0.5, -0.5, -0.5];
    vol.bbox_max = [0.5, 0.5, 0.5];
    vol.settings.pick_id = PickId(64);
    frame.scene.volumes = vec![vol];

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let vp = glam::Vec2::new(64.0, 64.0);
    let view_proj = frame.camera.render_camera.view_proj();

    // Object level: the centre ray crosses the cube.
    let hit = renderer.pick(glam::Vec2::new(32.0, 32.0), vp, view_proj, PickMask::OBJECT);
    assert_eq!(hit.map(|h| h.id), Some(64));

    // Voxel level: the same ray reports which voxel it stopped in.
    let voxel = renderer.pick(glam::Vec2::new(32.0, 32.0), vp, view_proj, PickMask::VOXEL);
    assert!(
        matches!(
            voxel.and_then(|h| h.sub_object),
            Some(viewport_lib::SubObjectRef::Voxel(_))
        ),
        "VOXEL mask should refine the hit to a voxel"
    );

    // A corner ray misses the cube entirely.
    let miss = renderer.pick(glam::Vec2::new(1.0, 1.0), vp, view_proj, PickMask::OBJECT);
    assert!(miss.is_none(), "corner ray should miss the volume");
}

#[test]
fn rect_pick_hits_voxel_volume() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);

    let dims = [4u32, 4, 4];
    let data = vec![1.0f32; (dims[0] * dims[1] * dims[2]) as usize];
    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &data, dims);

    let mut frame = sub_object_pick_frame();
    let mut vol = VolumeItem::default();
    vol.volume_id = volume_id;
    vol.volume_data = Some(std::sync::Arc::new(viewport_lib::VolumeData {
        data,
        dims,
        origin: [-0.5, -0.5, -0.5],
        spacing: [0.25, 0.25, 0.25],
    }));
    vol.scalar_range = (0.0, 1.0);
    vol.threshold_min = 0.0;
    vol.threshold_max = 1.0;
    vol.bbox_min = [-0.5, -0.5, -0.5];
    vol.bbox_max = [0.5, 0.5, 0.5];
    vol.settings.pick_id = PickId(65);
    frame.scene.volumes = vec![vol];

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let vp = glam::Vec2::new(64.0, 64.0);
    let view_proj = frame.camera.render_camera.view_proj();

    // A full-viewport rect projects every in-threshold voxel centre inside it.
    let result = renderer.pick_rect(
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        vp,
        view_proj,
        PickMask::OBJECT | PickMask::VOXEL,
    );
    assert!(
        result.objects.contains(&65),
        "full-viewport rect should select the volume, got {:?}",
        result.objects
    );
    assert_eq!(
        result.elements.len(),
        64,
        "every voxel centre should be reported at VOXEL level"
    );

    // A rect in the corner covers no voxel centre.
    let away = renderer.pick_rect(
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(2.0, 2.0),
        vp,
        view_proj,
        PickMask::OBJECT | PickMask::VOXEL,
    );
    assert!(!away.objects.contains(&65));
}
