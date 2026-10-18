//! A mesh's sidecar streams and per-mesh draw words live in one record of a
//! shared pool chunk. These cover what the record carries on the per-object
//! paths: the second UV set's base, the override windows on the OIT path, and
//! the streams kept across a record move.

use viewport_lib::wgpu;
use viewport_lib::{AlphaMode, AttributeData, AttributeKind, AttributeRef, Colour};
use viewport_lib::{TextureSlot, UvTransform};

mod common;
use common::*;

const W: u32 = 64;
const H: u32 = 64;

fn frame_of(items: Vec<SceneRenderItem>) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [W as f32, H as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.lighting.lights = vec![];
    frame.effects.lighting.hemisphere_intensity = 0.0;
    frame.effects.display.exposure = viewport_lib::ExposureSettings::passthrough();
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
    frame
}

/// Mean RGB brightness over the centre of the frame.
fn centre_brightness(pixels: &[u8]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0u64);
    for y in (H * 3 / 8)..(H * 5 / 8) {
        for x in (W * 3 / 8)..(W * 5 / 8) {
            let i = ((y * W + x) * 4) as usize;
            sum += pixels[i] as f64 + pixels[i + 1] as f64 + pixels[i + 2] as f64;
            n += 1;
        }
    }
    sum / n as f64
}

/// A mesh that is not the first in its slab chunk reads its own second UV set
/// on the per-object paths: the draw binds a mesh sub-slice, and the record's
/// base recovers the chunk-wide index. The emissive slot samples UV1, which
/// lands on the white middle of a strip; UV0, and the zero UV a lost base
/// reads, land on its black ends.
#[test]
fn a_per_object_draw_reads_its_own_second_uv_set() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // A mesh ahead of it in the slab, with no second UV set, so a lost base
    // reads this one's zero-filled uv1 region instead.
    renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let mut data = box_mesh();
    data.uvs = Some(vec![[0.0, 0.5]; data.positions.len()]);
    data.uvs1 = Some(vec![[0.5, 0.5]; data.positions.len()]);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &data)
        .unwrap();

    let strip = [
        [0u8, 0, 0, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255],
        [0, 0, 0, 255],
    ]
    .concat();
    let emissive = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            viewport_lib::TextureData::srgb(4, 1, strip),
        )
        .expect("upload emissive texture");

    for alpha_mode in [AlphaMode::Opaque, AlphaMode::Blend] {
        let mut m = Material::default();
        m.shading_model = ShadingModel::Pbr;
        m.base_colour = Colour::linear(0.0, 0.0, 0.0, 0.8);
        m.emissive = Colour::linear_rgb(1.0, 1.0, 1.0);
        m.emissive_texture_id = Some(emissive);
        m.alpha_mode = alpha_mode;
        m = m.with_texture_transform(
            TextureSlot::Emissive,
            UvTransform {
                uv_set: 1,
                ..UvTransform::default()
            },
        );
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.material = upload_material(&mut renderer, m);
        item.settings.force_per_object = true;
        let frame = frame_of(vec![item]);
        let lit = centre_brightness(&renderer.render_offscreen(&device, &queue, &frame, W, H));
        assert!(
            lit > 300.0,
            "{alpha_mode:?}: the emissive slot read UV1 from the wrong place ({lit:.1})"
        );
    }
}

/// A transparent mesh draws through the OIT shader, which reads the position
/// override like the opaque one: an override that moves every vertex away
/// leaves the frame empty.
#[test]
fn a_transparent_mesh_follows_its_position_override() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let mut m = Material::default();
    m.base_colour = Colour::linear(1.0, 1.0, 1.0, 0.9);
    m.alpha_mode = AlphaMode::Blend;
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = upload_material(&mut renderer, m);
    item.settings.unlit = true;
    let mut frame = frame_of(vec![item]);

    let before = centre_brightness(&renderer.render_offscreen(&device, &queue, &frame, W, H));
    assert!(before > 400.0, "the transparent box draws ({before:.1})");

    let moved: Vec<f32> = (0..8).flat_map(|_| [0.0_f32, 0.0, -1000.0]).collect();
    let override_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_position_override"),
        size: (moved.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&override_buf, 0, bytemuck::cast_slice(&moved));
    renderer
        .resources_mut()
        .set_position_override_buffer(mesh_id, override_buf)
        .unwrap();
    // Moved off screen, the box leaves only the background behind.
    let after = centre_brightness(&renderer.render_offscreen(&device, &queue, &frame, W, H));
    frame.scene.surfaces = SurfaceSubmission::Flat(Vec::new().into());
    let empty = centre_brightness(&renderer.render_offscreen(&device, &queue, &frame, W, H));
    assert!(
        (after - empty).abs() < 1.0,
        "the transparent box ignored its position override ({after:.1}, background {empty:.1})"
    );
}

/// Setting a lightmap after upload moves the mesh's record to fit the new
/// stream. The scalar attribute moves with it: the colouring is unchanged, and
/// a later in-place replace still reaches it.
#[test]
fn a_lightmap_set_after_upload_keeps_the_mesh_attributes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut data = box_mesh();
    let ramp: Vec<f32> = (0..data.positions.len())
        .map(|i| i as f32 / (data.positions.len() - 1) as f32)
        .collect();
    data.attributes
        .insert("heat".to_string(), AttributeData::Vertex(ramp.clone()));
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &data)
        .unwrap();

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = upload_material(&mut renderer, Material::default());
    item.settings.unlit = true;
    item.settings.force_per_object = true;
    item.active_attribute = Some(AttributeRef {
        name: "heat".to_string(),
        kind: AttributeKind::Vertex,
    });
    item.colourmap_id = Some(
        renderer
            .resources()
            .builtin_colourmap_id(viewport_lib::BuiltinColourmap::Viridis),
    );
    let frame = frame_of(vec![item]);
    let coloured = renderer.render_offscreen(&device, &queue, &frame, W, H);

    // A black radiance added on top changes nothing about the colouring.
    let black = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            viewport_lib::TextureData::linear(1, 1, vec![0u8, 0, 0, 255]),
        )
        .expect("upload lightmap");
    let uv1 = vec![glam::Vec2::new(0.5, 0.5); data.positions.len()];
    renderer
        .resources_mut()
        .set_lightmap(
            &device,
            mesh_id,
            &uv1,
            viewport_lib::resources::LightmapData::NonDirectional { radiance: black },
            viewport_lib::resources::LightmapMode::Add,
        )
        .unwrap();
    let moved = renderer.render_offscreen(&device, &queue, &frame, W, H);
    assert_eq!(
        coloured, moved,
        "the scalar colouring changed when the record moved"
    );

    let flat = vec![0.0f32; data.positions.len()];
    renderer
        .resources_mut()
        .replace_attribute(&queue, mesh_id, "heat", &flat)
        .unwrap();
    let replaced = renderer.render_offscreen(&device, &queue, &frame, W, H);
    assert_ne!(
        moved, replaced,
        "the in-place replace did not reach the moved stream"
    );
}

/// Shadow casters read their position override window from the record too,
/// under directional and point lights. The caster's window sits after a copy of
/// its rest pose in a shared buffer and moves it far away, so only the right
/// window leaves the floor unshadowed.
#[test]
fn a_shadow_caster_follows_its_override_window() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let floor_mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(8.0, 8.0))
        .unwrap();
    let half = 0.3_f32;
    let rest = [
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ];
    let mut quad = MeshData::default();
    quad.positions = rest.to_vec();
    quad.normals = vec![[0.0, 0.0, 1.0]; 4];
    quad.indices = vec![0, 1, 2, 0, 2, 3];
    let caster_mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &quad)
        .unwrap();

    let mut floor = SceneRenderItem::default();
    floor.mesh_id = floor_mesh;
    floor.material = upload_material(
        &mut renderer,
        Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8)),
    );
    // A lone quad casts only when two-sided: the shadow pass culls front faces.
    let mut caster_material = Material::from_colour(Colour::linear_rgb(0.2, 0.2, 0.8));
    caster_material.backface_policy = viewport_lib::BackfacePolicy::Identical;
    let mut caster = SceneRenderItem::default();
    caster.mesh_id = caster_mesh;
    caster.model = glam::Mat4::from_translation(glam::Vec3::new(0.5, 1.0, 1.5)).to_cols_array_2d();
    caster.material = upload_material(&mut renderer, caster_material);

    let mut sun = LightSource::default();
    sun.kind = LightKind::Directional {
        direction: [1.0, 0.0, 1.0],
    };
    let mut bulb = LightSource::default();
    bulb.kind = LightKind::Point {
        position: [3.0, 0.0, 3.0],
        range: 30.0,
        radius: 0.0,
    };
    bulb.intensity = 40.0;
    bulb.cast_shadows = true;

    let mut generation = 0u64;
    let mut shadowed = |renderer: &mut ViewportRenderer, light: &LightSource| -> usize {
        const SIZE: u32 = 64;
        generation += 1;
        let mut frame = FrameData::default();
        frame.scene.generation = generation;
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.effects.display.exposure = viewport_lib::ExposureSettings::passthrough();
        frame.effects.lighting.lights = vec![light.clone()];
        frame.effects.lighting.hemisphere_intensity = 0.05;
        frame.camera.render_camera = {
            let mut cam = Camera::default();
            cam.center = glam::Vec3::ZERO;
            cam.distance = 6.0;
            cam.orientation = glam::Quat::IDENTITY;
            let mut rc = RenderCamera::from_camera(&cam);
            rc.aspect = 1.0;
            rc
        };
        frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![floor.clone(), caster.clone()].into());
        renderer
            .render_offscreen(&device, &queue, &frame, SIZE, SIZE)
            .chunks_exact(4)
            .filter(|p| (p[0] as u32 + p[1] as u32 + p[2] as u32) < 300)
            .count()
    };

    for (label, light) in [("directional", &sun), ("point", &bulb)] {
        assert!(
            shadowed(&mut renderer, light) > 20,
            "{label}: the caster shadows the floor at rest"
        );
    }

    // The rest pose, then the same quad moved far off.
    let pool: Vec<f32> = rest
        .iter()
        .copied()
        .chain(rest.iter().map(|p| [p[0] + 1000.0, p[1], p[2]]))
        .flatten()
        .collect();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_pooled_override"),
        size: (pool.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&pool));
    renderer
        .resources_mut()
        .set_position_override_buffer_sliced(
            caster_mesh,
            buffer,
            viewport_lib::resources::OverrideBufferSlice::new(4, 4),
        )
        .unwrap();
    for (label, light) in [("directional", &sun), ("point", &bulb)] {
        let n = shadowed(&mut renderer, light);
        assert_eq!(n, 0, "{label}: the moved caster still shadows the floor");
    }
}

/// Copies of one lightmapped mesh each sample their own rect of the atlas when
/// they carry their own placement: one upload and one unwrap serve both. The
/// atlas is red on the left and green on the right, and every vertex unwraps
/// to its centre, so the placement alone decides the colour.
#[test]
fn copies_of_a_lightmapped_mesh_sample_their_own_rects() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let quad = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::plane(1.0, 1.0))
        .unwrap();
    let atlas = renderer
        .resources_mut()
        .upload_texture(
            &device,
            &queue,
            viewport_lib::TextureData::linear(2, 1, vec![255u8, 0, 0, 255, 0, 255, 0, 255]),
        )
        .expect("upload atlas");
    let n = viewport_lib::primitives::plane(1.0, 1.0).positions.len();
    renderer
        .resources_mut()
        .set_lightmap(
            &device,
            quad,
            &vec![glam::Vec2::new(0.5, 0.5); n],
            viewport_lib::resources::LightmapData::NonDirectional { radiance: atlas },
            viewport_lib::resources::LightmapMode::Replace,
        )
        .unwrap();
    let white = upload_material(
        &mut renderer,
        Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0)),
    );
    let copy_at = |x: f32, placement: usize| {
        let mut it = SceneRenderItem::default();
        it.mesh_id = quad;
        it.material = white;
        it.model = glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)).to_cols_array_2d();
        it.lightmap_placement = Some(viewport_lib::resources::LightmapPlacementId::from_index(
            placement,
        ));
        it
    };
    let mut frame = frame_of(vec![copy_at(-0.6, 0), copy_at(0.6, 1)]);
    frame.scene.lightmap_placements = vec![
        viewport_lib::resources::LightmapPlacement::new(0, [0.5, 1.0, 0.0, 0.0]),
        viewport_lib::resources::LightmapPlacement::new(0, [0.5, 1.0, 0.5, 0.0]),
    ];
    frame.camera.render_camera = {
        let mut cam = Camera::default();
        cam.orientation = glam::Quat::IDENTITY;
        cam.center = glam::Vec3::ZERO;
        cam.distance = 5.0;
        cam.aspect = 1.0;
        RenderCamera::from_camera(&cam)
    };
    let rc = frame.camera.render_camera.clone();
    let img = renderer.render_offscreen(&device, &queue, &frame, W, H);
    let pixel = |world: glam::Vec3| {
        let clip = rc.view_proj() * world.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let x = (((ndc.x * 0.5 + 0.5) * W as f32) as u32).min(W - 1);
        let y = (((0.5 - ndc.y * 0.5) * H as f32) as u32).min(H - 1);
        let i = ((y * W + x) * 4) as usize;
        [img[i] as i32, img[i + 1] as i32, img[i + 2] as i32]
    };
    let left = pixel(glam::Vec3::new(-0.6, 0.0, 0.0));
    let right = pixel(glam::Vec3::new(0.6, 0.0, 0.0));
    assert!(
        left[0] > left[1] + 60,
        "the left copy samples the red rect, got {left:?}"
    );
    assert!(
        right[1] > right[0] + 60,
        "the right copy samples the green rect, got {right:?}"
    );
}

/// A per-object draw of a mesh with no sidecar record takes the pipeline with
/// the sidecar reads compiled out; a mesh with a record keeps the full one.
#[test]
fn a_mesh_without_a_record_draws_through_the_lean_pipeline() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    viewport_lib::resources::build_log::enable();
    let plain = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let mut data = box_mesh();
    data.attributes.insert(
        "heat".to_string(),
        AttributeData::Vertex(vec![0.5; data.positions.len()]),
    );
    let with_record = renderer
        .resources_mut()
        .upload_mesh_data(&device, &data)
        .unwrap();
    let material = upload_material(&mut renderer, Material::default());
    let built = |renderer: &mut ViewportRenderer, mesh_id| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.material = material;
        item.settings.force_per_object = true;
        let frame = frame_of(vec![item]);
        let _ = viewport_lib::resources::build_log::drain();
        let lit = centre_brightness(&renderer.render_offscreen(&device, &queue, &frame, W, H));
        assert!(lit > 30.0, "the box draws ({lit:.1})");
        viewport_lib::resources::build_log::drain()
            .into_iter()
            .map(|(label, _)| label)
            .filter(|l| l.starts_with("hdr_solid"))
            .collect::<Vec<_>>()
    };
    let lean = built(&mut renderer, plain);
    assert!(
        lean.iter().any(|l| l.contains("nosidecar")),
        "a mesh without a record built {lean:?}"
    );
    let full = built(&mut renderer, with_record);
    assert!(
        full.iter().any(|l| !l.contains("nosidecar")),
        "a mesh with a record built {full:?}"
    );
}

/// A vertex scalar shorter than the mesh is padded with its last value at
/// upload, which draws what clamping the read to the last element did, and an
/// in-place replace still takes the length that was uploaded.
#[test]
fn a_short_vertex_scalar_draws_like_its_last_value() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let upload = |renderer: &mut ViewportRenderer, values: Vec<f32>| {
        let mut data = box_mesh();
        data.attributes
            .insert("heat".to_string(), AttributeData::Vertex(values));
        renderer
            .resources_mut()
            .upload_mesh_data(&device, &data)
            .unwrap()
    };
    let n = box_mesh().positions.len();
    let short = upload(&mut renderer, vec![0.2, 0.9]);
    let full = upload(&mut renderer, {
        let mut v = vec![0.9; n];
        v[0] = 0.2;
        v
    });
    let material = upload_material(&mut renderer, Material::default());
    let viridis = renderer
        .resources()
        .builtin_colourmap_id(viewport_lib::BuiltinColourmap::Viridis);
    let draw = |renderer: &mut ViewportRenderer, mesh_id| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.material = material;
        item.settings.unlit = true;
        item.settings.force_per_object = true;
        item.active_attribute = Some(AttributeRef {
            name: "heat".to_string(),
            kind: AttributeKind::Vertex,
        });
        item.scalar_range = Some((0.0, 1.0));
        item.colourmap_id = Some(viridis);
        renderer.render_offscreen(&device, &queue, &frame_of(vec![item]), W, H)
    };
    let a = draw(&mut renderer, short);
    let b = draw(&mut renderer, full);
    assert_eq!(
        a, b,
        "the short scalar drew differently from its padded form"
    );

    assert!(
        renderer
            .resources_mut()
            .replace_attribute(&queue, short, "heat", &[0.9, 0.2])
            .is_ok(),
        "a replace with the uploaded length is accepted"
    );
    assert!(
        renderer
            .resources_mut()
            .replace_attribute(&queue, short, "heat", &vec![0.5; n])
            .is_err(),
        "a replace with a different length is rejected"
    );
}
