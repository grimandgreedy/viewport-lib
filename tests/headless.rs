//! Headless integration tests for viewport-lib.
//!
//! These tests create a real wgpu device (headless) and exercise the GPU
//! resource APIs. Requires a GPU adapter (software or hardware).

// On the 27 leg the plain `wgpu` dependency is active and `wgpu::` resolves to
// it directly. On the 29 leg that dependency is inactive, so name wgpu through
// the library's re-export instead, which tracks whichever leg is built.
#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

use viewport_lib::{
    Aabb, AlphaMode, BackfacePolicy, Camera, DecalItem, GaussianSplatData, GaussianSplatItem,
    GlyphItem, GlyphType, ImageAnchor, ImageSliceItem, IndirectLightSource, ItemSettings,
    LightKind, LightSource, Material, MeshId, OverrideBufferSlice, PickBackend, PickId, PickMask,
    PickPoll, PointCloudItem, PolylineItem, RibbonItem, ScatterVolume, ScatterVolumeItem, Scene,
    ScreenImageItem, Selection, ShDegree, ShadingModel, SliceAxis, SpriteItem, SpriteSizeMode,
    VolumeItem, VolumeMeshItem, VolumeSurfaceSliceItem,
    error::ViewportError,
    plugin_api::{
        ItemTypePlugin, PickPassContext, PluginItemCollection, SharedBindings,
        shared_wgsl::SHARED_PICK_WGSL,
    },
    renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, ViewportRenderer},
    resources::{MeshData, PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT, SCENE_DEPTH_FORMAT},
};

/// Create a headless wgpu device + queue for testing.
fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = viewport_lib::wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Headless device with `SHADER_PRIMITIVE_INDEX` enabled, or `None` when no
/// adapter is available or the adapter does not support the feature. Used by the
/// GPU sub-object tests that read the pick pass's triangle-index channel.
fn headless_device_with_primitive_index() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = viewport_lib::wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    if !adapter
        .features()
        .contains(viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE)
    {
        return None;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test-primitive-index"),
        required_features: viewport_lib::gpu::PRIMITIVE_INDEX_FEATURE,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Headless device with `max_bind_groups` capped at 2, or `None` when no
/// adapter is available. Mirrors the device iced_wgpu 0.14 requests (it
/// hardcodes this limit for WebGL2 portability and gives a consumer no way to
/// raise it), so any draw site that unconditionally binds group index 2 fails
/// wgpu validation against this device the same way it would against iced's.
/// See `docs/issues/iced-max-bind-groups-2-draw-path-incomplete.md` and
/// `docs/plans/iced-two-bind-group-support-plan.md`.
fn headless_device_limited_bind_groups() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = viewport_lib::wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let mut limits = wgpu::Limits::default();
    limits.max_bind_groups = 2;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test-2-bind-groups"),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Simple unit box mesh data for testing.
fn box_mesh() -> MeshData {
    let positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let normals = vec![
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}

#[test]
fn upload_mesh_data_valid() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let result = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh());
    assert!(result.is_ok());
}

#[test]
fn upload_mesh_data_empty() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let empty = MeshData::default();
    let result = renderer.resources_mut().upload_mesh_data(&device, &empty);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::EmptyMesh { .. }
    ));
}

#[test]
fn upload_mesh_data_length_mismatch() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut bad = MeshData::default();
    bad.positions = vec![[0.0; 3], [1.0; 3]];
    bad.normals = vec![[0.0; 3]]; // mismatched length
    bad.indices = vec![0, 1, 0];
    let result = renderer.resources_mut().upload_mesh_data(&device, &bad);
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::MeshLengthMismatch { .. }
    ));
}

#[test]
fn upload_mesh_data_invalid_index() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut bad = MeshData::default();
    bad.positions = vec![[0.0; 3], [1.0; 3], [2.0; 3]];
    bad.normals = vec![[0.0; 3]; 3];
    bad.indices = vec![0, 1, 99]; // 99 is out of bounds
    let result = renderer.resources_mut().upload_mesh_data(&device, &bad);
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::InvalidVertexIndex {
            vertex_index: 99,
            ..
        }
    ));
}

#[test]
fn replace_mesh_data_bad_index() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let result =
        renderer
            .resources_mut()
            .replace_mesh_data(&device, &queue, MeshId::INVALID, &box_mesh());
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::StaleHandle { .. }
    ));
}

#[test]
fn prepare_empty_scene_no_panic() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [0.0, 0.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    // Should not panic.
    let _ = renderer.pass().prepare(&device, &queue, &frame);
}

#[test]
fn test_remove_mesh_frees_slot() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    assert!(renderer.resources().mesh(idx).is_some());

    let removed = renderer.resources_mut().free_mesh(idx);
    assert!(removed);
    assert!(renderer.resources().mesh(idx).is_none());
}

#[test]
fn test_upload_reuses_freed_slot() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let idx1 = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    renderer.resources_mut().free_mesh(idx1);

    // Next upload should reuse the freed slot, but at a new generation so the
    // old handle no longer matches.
    let idx2 = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    assert_eq!(idx1.index(), idx2.index(), "freed slot should be reused");
    assert_ne!(
        idx1, idx2,
        "reused slot must carry a new generation so the old handle differs"
    );
}

#[test]
fn test_scene_collect_render_items_roundtrip() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut scene = Scene::new();
    let node_id = scene.add(
        Some(mesh_idx),
        glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0)),
        Material::default(),
    );

    let mut sel = Selection::new();
    sel.select_one(node_id);

    let items = scene.collect_render_items(&sel);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].mesh_id, mesh_idx);
    assert!(items[0].settings.selected);
    // Verify position is in the model matrix.
    let pos_x = items[0].model[3][0];
    assert!((pos_x - 1.0).abs() < 1e-5, "model[3][0] = {pos_x}");
}

#[test]
fn render_offscreen_produces_rgba_pixels() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Use Rgba8UnormSrgb so no BGRA swizzle complicates assertions.
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Upload a mesh so the scene is non-trivial.
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
    // Add the box as a scene item.
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.settings.selected = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let width = 64u32;
    let height = 64u32;
    let pixels = renderer.render_offscreen(&device, &queue, &frame, width, height);

    // Must be exactly width * height * 4 RGBA bytes.
    assert_eq!(pixels.len(), (width * height * 4) as usize);

    // At least some pixels should be non-zero (the mesh or background).
    let has_nonzero = pixels.iter().any(|&b| b != 0);
    assert!(has_nonzero, "offscreen render produced all-zero image");
}

/// `capture_hdr` must return linear radiance with the full HDR range intact:
/// a value above 1.0 in the scene has to survive to the CPU, where the
/// tone-mapped LDR path would have clamped it. A box with emissive `[5,5,5]`
/// (emissive is added after the pre-tonemap clamp) is a guaranteed > 1.0 signal
/// independent of the lighting model, so the captured pixels must exceed 1.0.
#[test]
fn capture_hdr_preserves_values_above_one() {
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
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // A very bright directional light on a plain white box. This exercises the
    // *lit* path specifically (not emissive, which is added past the clamp), so
    // the assertion below only holds if the capture actually raised the shader's
    // lit_clamp to the f16 max on the HDR path. A camera-facing light keeps the
    // visible face lit.
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.0, 0.0, 1.0],
    };
    light.intensity = 20.0;
    frame.effects.lighting.lights = vec![light];

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material.base_colour = [1.0, 1.0, 1.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    // Snapshot the fields capture_hdr overrides, to prove they are restored.
    let orig_viewport_size = frame.camera.viewport_size;
    let orig_pp_enabled = frame.effects.post_process.enabled;

    let mut face_cam = RenderCamera::from_camera(&cam);
    face_cam.aspect = 1.0;
    let captured = renderer.capture_hdr(&device, &queue, &mut frame, face_cam, 64);

    assert_eq!(captured.width, 64);
    assert_eq!(captured.height, 64);
    assert_eq!(captured.rgba.len(), 64 * 64 * 4);

    let max_channel = captured
        .rgba
        .iter()
        .copied()
        .fold(0.0f32, |acc, v| acc.max(v));
    assert!(
        max_channel > 1.5,
        "captured lit radiance was clamped: max channel {max_channel} (expected > 1.5; lit_clamp not lifted?)"
    );

    // The override snapshot must be restored: the caller's frame is unchanged.
    assert_eq!(frame.camera.viewport_size, orig_viewport_size);
    assert_eq!(frame.effects.post_process.enabled, orig_pp_enabled);
}

/// A directional (dominant-direction) lightmap must make a normal-mapped surface
/// respond to the baked light direction: tilting the pixel normal toward the
/// baked dominant direction brightens the baked term, tilting away dims it. A
/// flat quad (geometric normal +Z) is lit by a uniform radiance atlas plus a
/// dominant direction of (0.6, 0, 0.8); one render tilts the normal-mapped normal
/// toward +X (into the light), the other toward -X (away).
#[test]
fn directional_lightmap_responds_to_normal() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Quad in the XY plane facing +Z, tangent +X, with UV0 for normal mapping.
    let mut quad = MeshData::default();
    quad.positions = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    quad.normals = vec![[0.0, 0.0, 1.0]; 4];
    quad.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    quad.tangents = Some(vec![[1.0, 0.0, 0.0, 1.0]; 4]);
    quad.indices = vec![0, 1, 2, 0, 2, 3];
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &quad)
        .unwrap();
    let uv1 = vec![glam::Vec2::new(0.5, 0.5); 4];

    // Uniform radiance 1.0, dominant direction (0.6,0,0.8) world, directionality 1.
    let radiance = renderer
        .resources_mut()
        .upload_texture_hdr(&device, &queue, 2, 2, &[1.0f32; 2 * 2 * 4])
        .unwrap();
    let dir_rgba: Vec<f32> = std::iter::repeat([0.6f32, 0.0, 0.8, 1.0])
        .take(2 * 2)
        .flatten()
        .collect();
    let direction = renderer
        .resources_mut()
        .upload_texture_hdr(&device, &queue, 2, 2, &dir_rgba)
        .unwrap();
    renderer
        .resources_mut()
        .set_lightmap(
            &device,
            mesh,
            &uv1,
            viewport_lib::resources::LightmapData::DominantDirection {
                radiance,
                direction,
            },
            viewport_lib::resources::LightmapMode::Replace,
        )
        .unwrap();

    // Tangent-space normal maps: (0.6,0,0.8) tilts world N toward +X (into the
    // light); (-0.6,0,0.8) tilts toward -X. Encoded n*0.5+0.5 in 8-bit.
    let enc = |x: f32, y: f32, z: f32| {
        [
            ((x * 0.5 + 0.5) * 255.0) as u8,
            ((y * 0.5 + 0.5) * 255.0) as u8,
            ((z * 0.5 + 0.5) * 255.0) as u8,
            255,
        ]
    };
    let toward: Vec<u8> = std::iter::repeat(enc(0.6, 0.0, 0.8))
        .take(2 * 2)
        .flatten()
        .collect();
    let away: Vec<u8> = std::iter::repeat(enc(-0.6, 0.0, 0.8))
        .take(2 * 2)
        .flatten()
        .collect();
    let nm_toward = renderer
        .resources_mut()
        .upload_normal_map(&device, &queue, 2, 2, &toward)
        .unwrap();
    let nm_away = renderer
        .resources_mut()
        .upload_normal_map(&device, &queue, 2, 2, &away)
        .unwrap();

    let peak_with = |renderer: &mut ViewportRenderer, nm| -> f32 {
        let cam = Camera::default();
        let mut frame = FrameData::default();
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.effects.lighting.lights = Vec::new();
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.base_colour = [1.0, 1.0, 1.0];
        item.material.normal_map_id = Some(nm);
        item.material.normal_strength = 1.0;
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        let mut face_cam = RenderCamera::from_camera(&cam);
        face_cam.aspect = 1.0;
        let captured = renderer.capture_hdr(&device, &queue, &mut frame, face_cam, 64);
        // Sample the centre pixel (on the quad) rather than the global peak, which
        // would pick up the background clear when the quad is dim.
        let c = ((32 * 64 + 32) * 4) as usize;
        captured.rgba[c]
            .max(captured.rgba[c + 1])
            .max(captured.rgba[c + 2])
    };

    let peak_toward = peak_with(&mut renderer, nm_toward);
    let peak_away = peak_with(&mut renderer, nm_away);
    println!("directional lightmap: toward={peak_toward:.3} away={peak_away:.3}");
    assert!(
        peak_toward > peak_away * 1.8,
        "normal facing the baked light ({peak_toward}) should be much brighter than facing away ({peak_away})"
    );
}

/// A shadowmask attenuates a realtime light's direct contribution per channel: the
/// same quad lit by one directional light reads bright where light 0's shadowmask
/// channel is 1 (lit) and dark where it is 0 (shadowed). Black radiance in Replace
/// mode removes the ambient term, so the readback is the shadowmask-gated direct.
#[test]
fn shadowmask_attenuates_direct_light() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut quad = MeshData::default();
    quad.positions = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    quad.normals = vec![[0.0, 0.0, 1.0]; 4];
    quad.indices = vec![0, 1, 2, 0, 2, 3];
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &quad)
        .unwrap();
    let uv1 = vec![glam::Vec2::new(0.5, 0.5); 4];

    // Black radiance -> Replace zeroes the ambient term, so only the direct light
    // (gated by the shadowmask) reaches the readback.
    let radiance = renderer
        .resources_mut()
        .upload_texture_hdr(&device, &queue, 2, 2, &[0.0f32; 2 * 2 * 4])
        .unwrap();

    let peak_with_vis = |renderer: &mut ViewportRenderer, v: f32| -> f32 {
        // Shadowmask: light 0 -> red channel = v; the other channels stay lit (1).
        let sm: Vec<f32> = std::iter::repeat([v, 1.0, 1.0, 1.0])
            .take(2 * 2)
            .flatten()
            .collect();
        let shadowmask = renderer
            .resources_mut()
            .upload_texture_hdr(&device, &queue, 2, 2, &sm)
            .unwrap();
        renderer
            .resources_mut()
            .set_lightmap(
                &device,
                mesh,
                &uv1,
                viewport_lib::resources::LightmapData::Shadowmask {
                    radiance,
                    shadowmask,
                },
                viewport_lib::resources::LightmapMode::Replace,
            )
            .unwrap();

        let cam = Camera::default();
        let mut frame = FrameData::default();
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        // One directional light (index 0) straight toward the quad's +Z face.
        let mut key = LightSource::default();
        key.kind = LightKind::Directional {
            direction: [0.0, 0.0, 1.0],
        };
        key.colour = [1.0, 1.0, 1.0];
        key.intensity = 1.0;
        frame.effects.lighting.lights = vec![key];
        frame.effects.lighting.hemisphere_intensity = 0.0;
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.base_colour = [1.0, 1.0, 1.0];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        let mut face_cam = RenderCamera::from_camera(&cam);
        face_cam.aspect = 1.0;
        let captured = renderer.capture_hdr(&device, &queue, &mut frame, face_cam, 64);
        let c = ((32 * 64 + 32) * 4) as usize;
        captured.rgba[c]
            .max(captured.rgba[c + 1])
            .max(captured.rgba[c + 2])
    };

    let lit = peak_with_vis(&mut renderer, 1.0);
    let shadowed = peak_with_vis(&mut renderer, 0.0);
    println!("shadowmask: lit={lit:.3} shadowed={shadowed:.3}");
    assert!(
        lit > 0.05,
        "lit texel should receive direct light, got {lit}"
    );
    assert!(
        lit > shadowed * 4.0,
        "shadowmask 0 should strongly darken the direct light: lit={lit} shadowed={shadowed}"
    );
}

/// A baked lightmap with radiance above 1.0 must survive to the HDR render path.
/// The 8-bit `upload_texture` path clamps at upload (sRGB, [0,1]); the
/// `upload_texture_hdr` (`Rgba16Float`) path must not. Both are rendered in
/// Replace mode with no runtime lights, so the captured radiance is the lightmap
/// value straight through: the LDR one saturates near 1.0, the HDR one keeps 4.0.
#[test]
fn hdr_lightmap_survives_above_one() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let vcount = box_mesh().positions.len();
    let uv1 = vec![glam::Vec2::new(0.5, 0.5); vcount];

    // Render the box lit only by a uniform lightmap of value `radiance`, and
    // return the peak captured channel.
    let mut capture_with = |renderer: &mut ViewportRenderer, tex| -> f32 {
        renderer
            .resources_mut()
            .set_lightmap(
                &device,
                mesh,
                &uv1,
                viewport_lib::resources::LightmapData::NonDirectional { radiance: tex },
                viewport_lib::resources::LightmapMode::Replace,
            )
            .unwrap();
        let cam = Camera::default();
        let mut frame = FrameData::default();
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.effects.lighting.lights = Vec::new();
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.base_colour = [1.0, 1.0, 1.0];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        let mut face_cam = RenderCamera::from_camera(&cam);
        face_cam.aspect = 1.0;
        let captured = renderer.capture_hdr(&device, &queue, &mut frame, face_cam, 64);
        captured.rgba.iter().copied().fold(0.0f32, f32::max)
    };

    // LDR upload: value 4.0 -> byte 255 -> ~1.0 after sRGB decode; clamped.
    let ldr = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 4, 4, &[255u8; 4 * 4 * 4])
        .unwrap();
    let ldr_peak = capture_with(&mut renderer, ldr);

    // HDR upload: linear 4.0 kept through Rgba16Float.
    let hdr_rgba: Vec<f32> = std::iter::repeat([4.0f32, 4.0, 4.0, 1.0])
        .take(4 * 4)
        .flatten()
        .collect();
    let hdr = renderer
        .resources_mut()
        .upload_texture_hdr(&device, &queue, 4, 4, &hdr_rgba)
        .unwrap();
    let hdr_peak = capture_with(&mut renderer, hdr);

    assert!(
        ldr_peak <= 1.2,
        "LDR lightmap should clamp near 1.0, got {ldr_peak}"
    );
    assert!(
        hdr_peak > 3.0,
        "HDR lightmap radiance was lost: peak {hdr_peak} (expected ~4.0)"
    );
}

/// A multi-page lightmap must select its atlas layer from the per-vertex page
/// index (UV1.z). One 2-layer HDR array is uploaded (layer 0 dim = 0.25, layer 1
/// bright = 4.0) and the same mesh with the same UV1 is rendered twice, changing
/// only the page assigned to every vertex. Page 0 must read the dim layer and
/// page 1 the bright layer, so the two captures diverge by the layer contents
/// alone. This proves the page index routes to distinct texture-array layers.
#[test]
fn multi_page_lightmap_selects_layer_per_vertex() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let vcount = box_mesh().positions.len();
    let uv1 = vec![glam::Vec2::new(0.5, 0.5); vcount];

    // Two-layer atlas: page 0 = 0.25 everywhere, page 1 = 4.0 everywhere. Data is
    // layer-major (all of page 0, then all of page 1).
    let mut atlas: Vec<f32> = Vec::new();
    atlas.extend(
        std::iter::repeat([0.25f32, 0.25, 0.25, 1.0])
            .take(2 * 2)
            .flatten(),
    );
    atlas.extend(
        std::iter::repeat([4.0f32, 4.0, 4.0, 1.0])
            .take(2 * 2)
            .flatten(),
    );
    let radiance = renderer
        .resources_mut()
        .upload_texture_hdr_layers(&device, &queue, 2, 2, 2, &atlas)
        .unwrap();

    let mut capture_page = |renderer: &mut ViewportRenderer, page: u32| -> f32 {
        let pages = vec![page; vcount];
        renderer
            .resources_mut()
            .set_lightmap_paged(
                &device,
                mesh,
                &uv1,
                &pages,
                viewport_lib::resources::LightmapData::NonDirectional { radiance },
                viewport_lib::resources::LightmapMode::Replace,
            )
            .unwrap();
        let cam = Camera::default();
        let mut frame = FrameData::default();
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.effects.lighting.lights = Vec::new();
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.base_colour = [1.0, 1.0, 1.0];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        let mut face_cam = RenderCamera::from_camera(&cam);
        face_cam.aspect = 1.0;
        let captured = renderer.capture_hdr(&device, &queue, &mut frame, face_cam, 64);
        // Peak over RGB only; the alpha channel is 1.0 and would mask a dim layer.
        captured
            .rgba
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .fold(0.0f32, f32::max)
    };

    let page0_peak = capture_page(&mut renderer, 0);
    let page1_peak = capture_page(&mut renderer, 1);
    println!("multi-page lightmap: page0={page0_peak:.3} page1={page1_peak:.3}");
    assert!(
        page0_peak < 1.0,
        "page 0 should read the dim layer (~0.25), got {page0_peak}"
    );
    assert!(
        page1_peak > 3.0,
        "page 1 should read the bright layer (~4.0), got {page1_peak}"
    );
    assert!(
        page1_peak > page0_peak * 3.0,
        "the two pages must diverge by layer contents: page0={page0_peak} page1={page1_peak}"
    );
}

/// Scene-level atlasing: many objects share one atlas array, each sampling its own
/// page (array layer) and sub-rect via a per-object layer + UV scale/bias. One
/// 2-layer, 4x1 atlas is shared by two meshes: object A reads the dim left half of
/// layer 0, object B reads the bright right half of layer 1. Because both point at
/// the same texture and differ only in their per-object `set_scene_lightmap`
/// placement, this proves the layer and scale/bias route each object to its own
/// region of a shared atlas.
#[test]
fn scene_lightmap_addresses_shared_atlas_per_object() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_a = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let mesh_b = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let vcount = box_mesh().positions.len();
    let uv1 = vec![glam::Vec2::new(0.5, 0.5); vcount];

    // Shared atlas: 4 texels wide, 1 tall, 2 layers. Left half / right half differ
    // within each layer, and the two layers differ, so a wrong layer or a wrong
    // sub-rect reads a clearly different value. Layout is layer-major.
    let row = |left: f32, right: f32| -> Vec<f32> {
        let mut v = Vec::new();
        for &c in &[left, left, right, right] {
            v.extend_from_slice(&[c, c, c, 1.0]);
        }
        v
    };
    let mut atlas = row(0.2, 1.0); // layer 0: left 0.2, right 1.0
    atlas.extend(row(2.0, 4.0)); // layer 1: left 2.0, right 4.0
    let shared = renderer
        .resources_mut()
        .upload_texture_hdr_layers(&device, &queue, 4, 1, 2, &atlas)
        .unwrap();

    // A: layer 0, left half -> lm_u = 0.5*0.5 + 0.0 = 0.25 -> 0.2.
    renderer
        .resources_mut()
        .set_scene_lightmap(
            &device,
            mesh_a,
            &uv1,
            shared,
            0,
            [0.5, 1.0, 0.0, 0.0],
            viewport_lib::resources::LightmapMode::Replace,
        )
        .unwrap();
    // B: layer 1, right half -> lm_u = 0.5*0.5 + 0.5 = 0.75 -> 4.0.
    renderer
        .resources_mut()
        .set_scene_lightmap(
            &device,
            mesh_b,
            &uv1,
            shared,
            1,
            [0.5, 1.0, 0.5, 0.0],
            viewport_lib::resources::LightmapMode::Replace,
        )
        .unwrap();

    let peak_of = |renderer: &mut ViewportRenderer, mesh| -> f32 {
        let cam = Camera::default();
        let mut frame = FrameData::default();
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.effects.lighting.lights = Vec::new();
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.base_colour = [1.0, 1.0, 1.0];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        let mut face_cam = RenderCamera::from_camera(&cam);
        face_cam.aspect = 1.0;
        let captured = renderer.capture_hdr(&device, &queue, &mut frame, face_cam, 64);
        // Peak over RGB only; alpha is 1.0 and would mask the dim object.
        captured
            .rgba
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .fold(0.0f32, f32::max)
    };

    let peak_a = peak_of(&mut renderer, mesh_a);
    let peak_b = peak_of(&mut renderer, mesh_b);
    println!("scene atlas: A(layer0,left)={peak_a:.3} B(layer1,right)={peak_b:.3}");
    assert!(
        peak_a < 1.0,
        "object A should read layer 0's dim left region (~0.2), got {peak_a}"
    );
    assert!(
        peak_b > 3.0,
        "object B should read layer 1's bright right region (~4.0), got {peak_b}"
    );
    assert!(
        peak_b > peak_a * 3.0,
        "the two objects share an atlas but must land in different regions: A={peak_a} B={peak_b}"
    );
}

/// End-to-end scene atlasing: `viewport_lib_lightbake::pack_scene_atlas` assigns each
/// object a rectangle in a shared page, the objects' baked atlases are blitted
/// into those rectangles, and the placement's `scale_bias`/`layer` drive
/// `set_scene_lightmap`. This ties the bake-side packer to the runtime: if the
/// packer's `scale_bias` disagreed with where the blit put each object, an object
/// would sample its neighbour's rectangle. Two objects (dim + bright) are packed
/// into one page; each must read back its own value.
#[test]
fn packed_scene_atlas_round_trips_through_the_packer() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_a = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let mesh_b = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let vcount = box_mesh().positions.len();
    let uv1 = vec![glam::Vec2::new(0.5, 0.5); vcount];

    // Each object's own baked atlas: a solid value so sampling anywhere in its
    // rectangle returns it. A dim, B bright.
    let obj_w = 64u32;
    let obj_h = 64u32;
    let atlas_a = vec![[0.25f32, 0.25, 0.25, 1.0]; (obj_w * obj_h) as usize];
    let atlas_b = vec![[4.0f32, 4.0, 4.0, 1.0]; (obj_w * obj_h) as usize];

    // Pack both into one shared page and blit each into its rectangle.
    let layout = viewport_lib_lightbake::pack_scene_atlas(
        &[
            viewport_lib_lightbake::SceneAtlasItem {
                width: obj_w,
                height: obj_h,
            },
            viewport_lib_lightbake::SceneAtlasItem {
                width: obj_w,
                height: obj_h,
            },
        ],
        256,
        4,
    );
    let page = layout.page_size;
    let mut pages = vec![0.0f32; (page * page * layout.layers) as usize * 4];
    let blit = |pages: &mut [f32], src: &[[f32; 4]], p: &viewport_lib_lightbake::ScenePlacement| {
        for row in 0..p.height {
            for col in 0..p.width {
                let s = (row * p.width + col) as usize;
                let d = ((p.layer * page * page) + (p.y + row) * page + (p.x + col)) as usize * 4;
                pages[d] = src[s][0];
                pages[d + 1] = src[s][1];
                pages[d + 2] = src[s][2];
                pages[d + 3] = src[s][3];
            }
        }
    };
    blit(&mut pages, &atlas_a, &layout.placements[0]);
    blit(&mut pages, &atlas_b, &layout.placements[1]);

    let shared = renderer
        .resources_mut()
        .upload_texture_hdr_layers(&device, &queue, page, page, layout.layers, &pages)
        .unwrap();

    for (mesh, p) in [
        (mesh_a, layout.placements[0]),
        (mesh_b, layout.placements[1]),
    ] {
        renderer
            .resources_mut()
            .set_scene_lightmap(
                &device,
                mesh,
                &uv1,
                shared,
                p.layer,
                p.scale_bias,
                viewport_lib::resources::LightmapMode::Replace,
            )
            .unwrap();
    }

    let peak_of = |renderer: &mut ViewportRenderer, mesh| -> f32 {
        let cam = Camera::default();
        let mut frame = FrameData::default();
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.effects.lighting.lights = Vec::new();
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.base_colour = [1.0, 1.0, 1.0];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        let mut face_cam = RenderCamera::from_camera(&cam);
        face_cam.aspect = 1.0;
        let captured = renderer.capture_hdr(&device, &queue, &mut frame, face_cam, 64);
        captured
            .rgba
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .fold(0.0f32, f32::max)
    };

    let peak_a = peak_of(&mut renderer, mesh_a);
    let peak_b = peak_of(&mut renderer, mesh_b);
    println!(
        "packed scene atlas: A={peak_a:.3} B={peak_b:.3} (layers={})",
        layout.layers
    );
    assert!(
        peak_a < 1.0,
        "packed object A should read its dim rect (~0.25), got {peak_a}"
    );
    assert!(
        peak_b > 3.0,
        "packed object B should read its bright rect (~4.0), got {peak_b}"
    );
}

/// `capture_equirect` must resolve the six faces into a panorama whose
/// direction mapping matches the shader consumer: a bright emissive box placed
/// along +X, viewed from the origin, has to land near the equirect centre
/// (u=0.5 -> phi=0 -> +X, v=0.5 -> theta=0 -> equator). A flipped axis in the
/// resolve would put the brightest texel somewhere else, so this pins the
/// convention end to end. It also confirms the 2:1 aspect and that HDR survives
/// the resolve.
#[test]
fn capture_equirect_maps_direction_like_the_shader() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = FrameData::default();
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    // Place the box along +X so it fills only the +X face from the origin.
    item.model = glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.0, 0.0)).to_cols_array_2d();
    item.material.emissive = [8.0, 8.0, 8.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let eq_h = 64u32;
    let captured =
        renderer.capture_equirect(&device, &queue, &mut frame, [0.0, 0.0, 0.0], 128, eq_h);

    assert_eq!(captured.width, eq_h * 2);
    assert_eq!(captured.height, eq_h);
    assert_eq!(
        captured.rgba.len(),
        (captured.width * captured.height * 4) as usize
    );

    // Find the brightest texel (by luminance-ish sum) and its normalised UV.
    let (mut best_i, mut best_lum) = (0usize, f32::NEG_INFINITY);
    for i in 0..(captured.width * captured.height) as usize {
        let o = i * 4;
        let lum = captured.rgba[o] + captured.rgba[o + 1] + captured.rgba[o + 2];
        if lum > best_lum {
            best_lum = lum;
            best_i = i;
        }
    }
    let px = (best_i as u32 % captured.width) as f32;
    let py = (best_i as u32 / captured.width) as f32;
    let u = (px + 0.5) / captured.width as f32;
    let v = (py + 0.5) / captured.height as f32;

    assert!(
        best_lum > 1.0,
        "HDR emissive did not survive the resolve: {best_lum}"
    );
    assert!(
        (0.42..=0.58).contains(&u) && (0.42..=0.58).contains(&v),
        "+X emissive box resolved to uv ({u:.3}, {v:.3}), expected near (0.5, 0.5)"
    );
}

/// LP-c consumption: an object marked `IndirectLightSource::LightProbe` must
/// take its indirect diffuse from the uploaded SH field. A red-only probe with
/// no direct lights makes a white PBR box render red, where global-IBL /
/// hemisphere ambient (blue-ish sky) would not. This exercises the whole
/// consumption path: the per-object SH prepass, the group-0 storage buffer, the
/// 336-byte ObjectUniform, and `evaluate_sh_probe` in the fragment.
#[test]
fn light_probe_object_is_lit_by_the_probe_field() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // Red-only probe: the DC coefficient r[0] = 1/Y00 makes evaluate_sh return
    // ~[1,0,0] for every normal.
    let mut sh = viewport_lib::resources::SHCoefficients::default();
    sh.r[0] = 1.0 / 0.282095;
    let probes =
        viewport_lib::resources::LightProbeSet::new(vec![viewport_lib::resources::LightProbe {
            position: [0.0, 0.0, 0.0],
            sh,
        }]);
    renderer.set_light_probes(probes);

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
    // No direct lights: the object colour is purely its indirect (probe) term.
    frame.effects.lighting.lights = vec![];

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material.shading_model = ShadingModel::Pbr;
    item.material.base_colour = [1.0, 1.0, 1.0];
    item.indirect_light = IndirectLightSource::LightProbe;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let (w, h) = (64u32, 64u32);
    let pixels = renderer.render_offscreen(&device, &queue, &frame, w, h);

    // The brightest-red pixel must be strongly red (the probe), not grey/blue.
    let mut best = (0u8, 0u8, 0u8);
    for px in pixels.chunks_exact(4) {
        if px[0] > best.0 {
            best = (px[0], px[1], px[2]);
        }
    }
    assert!(
        best.0 as i32 > best.2 as i32 + 40 && best.0 as i32 > best.1 as i32 + 40,
        "probe-lit object should be red: brightest-red pixel was {best:?}"
    );
}

/// End-to-end light-probe bake: a probe baked next to a bright emissive box on
/// its +X side must, when its SH is evaluated, read brighter for a +X-facing
/// normal than a -X-facing one. This exercises the whole LP-g path
/// (capture_equirect -> project_equirect_to_sh) plus the directional convention.
#[test]
fn bake_light_probes_captures_directional_radiance() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = FrameData::default();
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.0)).to_cols_array_2d();
    item.material.emissive = [6.0, 6.0, 6.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let set = renderer.bake_light_probes(&device, &queue, &mut frame, &[[0.0, 0.0, 0.0]], 96, 64);
    assert_eq!(set.probes().len(), 1);

    let sh = set.probes()[0].sh;
    let toward = viewport_lib::resources::evaluate_sh(&sh, [1.0, 0.0, 0.0])[0];
    let away = viewport_lib::resources::evaluate_sh(&sh, [-1.0, 0.0, 0.0])[0];
    assert!(
        toward > away + 0.05,
        "probe should be brighter toward the +X box: toward {toward}, away {away}"
    );

    // blend_sh_at at the probe returns its own SH.
    let blended = set.blend_sh_at([0.0, 0.0, 0.0]);
    assert!((blended.r[0] - sh.r[0]).abs() < 1e-3);
}

/// `capture_reflection_probe` bakes the scene into a fresh environment layer and
/// returns a parallax-enabled zone; the zone then drives a render through the
/// per-fragment parallax path without validation errors.
#[test]
fn capture_reflection_probe_bakes_a_parallax_zone() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = FrameData::default();
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // A bright emissive box off to +X, so the captured probe has structure.
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::from_translation(glam::Vec3::new(4.0, 0.0, 0.0)).to_cols_array_2d();
    item.material.emissive = [5.0, 5.0, 5.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let bounds = viewport_lib::Aabb {
        min: glam::Vec3::splat(-3.0),
        max: glam::Vec3::splat(3.0),
    };
    let zone = renderer
        .capture_reflection_probe(&device, &queue, &mut frame, bounds, 1.0, 64, 48)
        .unwrap();

    assert!(zone.parallax, "a reflection probe is parallax-corrected");
    assert!(
        zone.environment.index() >= 1,
        "the probe takes an extra layer, not the default (0)"
    );

    // Drive a render with the probe active: exercises the parallax + specular
    // occlusion path in the mesh shader without wgpu validation errors.
    renderer.set_environment_zones(&queue, &[zone]);
    let mut lit = SceneRenderItem::default();
    lit.mesh_id = mesh_idx;
    lit.material = viewport_lib::Material::pbr([1.0, 1.0, 1.0], 1.0, 0.15);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![lit].into());
    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(pixels.len(), 64 * 64 * 4);
}

/// A solid-colour equirect panorama (RGBA f32).
fn solid_env(rgb: [f32; 3], w: u32, h: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        v.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
    }
    v
}

/// Regression guard for the `EnvZone` GPU-struct stride: the WGSL struct must be
/// 48 bytes to match the Rust upload. A `vec3<u32>` pad (align 16) rounds it up
/// to 64 and reads every zone after the first at the wrong offset, so only the
/// first zone lights. Here the green environment is the SECOND zone, so it only
/// reaches the sphere when the stride is correct.
#[test]
fn environment_zones_select_the_second_zone() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Default (layer 0) black, plus red (layer 1) and green (layer 2).
    renderer
        .upload_environment_map(&device, &queue, &solid_env([0.0, 0.0, 0.0], 8, 4), 8, 4)
        .unwrap();
    let red = renderer
        .upload_environment(&device, &queue, &solid_env([1.0, 0.0, 0.0], 8, 4), 8, 4)
        .unwrap();
    let green = renderer
        .upload_environment(&device, &queue, &solid_env([0.0, 1.0, 0.0], 8, 4), 8, 4)
        .unwrap();

    // Red zone far away (no coverage); green zone around the origin. Green is the
    // second entry, so a stride mismatch reads it wrong and the sphere loses it.
    let far = viewport_lib::Aabb {
        min: glam::Vec3::splat(-31.0),
        max: glam::Vec3::splat(-29.0),
    };
    let here = viewport_lib::Aabb {
        min: glam::Vec3::splat(-3.0),
        max: glam::Vec3::splat(3.0),
    };
    renderer.set_environment_zones(
        &queue,
        &[
            viewport_lib::EnvironmentZone {
                bounds: far,
                environment: red,
                fade_distance: 0.5,
                parallax: false,
            },
            viewport_lib::EnvironmentZone {
                bounds: here,
                environment: green,
                fade_distance: 0.5,
                parallax: false,
            },
        ],
    );

    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &viewport_lib::primitives::sphere(1.0, 24, 12))
        .unwrap();

    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    // Black background so only the sphere's own (environment-lit) pixels count.
    frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0]);
    // IBL on, no direct or hemisphere light, so the matte sphere shows only the
    // selected environment's irradiance.
    frame.effects.environment = Some(viewport_lib::EnvironmentMap {
        intensity: 1.0,
        rotation: 0.0,
        show_skybox: false,
    });
    frame.effects.lighting.lights = vec![];
    frame.effects.lighting.hemisphere_intensity = 0.0;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material = Material::pbr([1.0, 1.0, 1.0], 0.0, 1.0); // matte white
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let (mut r, mut g) = (0u64, 0u64);
    for px in pixels.chunks_exact(4) {
        r += px[0] as u64;
        g += px[1] as u64;
    }
    assert!(
        g > r * 2 + 1,
        "sphere in the second (green) zone must read green (g {g}), not the \
         garbage a stride mismatch produces (r {r})"
    );
}

/// Regression test for the silent-skip bug where a `set_position_override_buffer`
/// binding would render nothing when the item was routed through the instanced
/// pipeline. `mesh_instanced.wgsl` has no awareness of the override binding,
/// so the consumer's compute output is dropped on the floor. The fix is twofold:
///   1. Items with a bound override are excluded from the instanced batches
///      (forced through the per-object pipeline that does know about overrides).
///   2. The per-item `ObjectUniform` write loop is entered (and the override
///      item is not skipped within it) so `has_position_override = 1` reaches
///      the shader.
///
/// The test renders one red plane with an override displacing every vertex
/// far behind the camera. A second decoy item is added so the visible-item
/// count exceeds `INSTANCING_THRESHOLD = 1` and the instanced pipeline is
/// actually engaged. If the bug returns, the red plane remains visible
/// because the instanced shader ignores the override entirely.
#[test]
fn position_override_takes_effect_through_render_path() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Two simple plane meshes: a red "test" plane that will get the override,
    // and a blue "decoy" plane that exists only to push the visible-item
    // count past INSTANCING_THRESHOLD = 1 so the instanced pipeline engages.
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    let red_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();
    let blue_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
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

    // Red plane (target of the override) at the origin; blue decoy off to the
    // side so it doesn't overdraw the red region we measure.
    let mut red_item = SceneRenderItem::default();
    red_item.mesh_id = red_id;
    red_item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    red_item.material = Material::from_colour([1.0, 0.0, 0.0]);

    let mut blue_item = SceneRenderItem::default();
    blue_item.mesh_id = blue_id;
    blue_item.model =
        glam::Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0)).to_cols_array_2d();
    blue_item.material = Material::from_colour([0.0, 0.0, 1.0]);

    frame.scene.surfaces =
        SurfaceSubmission::Flat(vec![red_item.clone(), blue_item.clone()].into());

    // ---- Render 1: no override. The red plane should be visible. ----
    let baseline = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    // The lit red plane tone-maps to roughly [246, 60, 60]: with the lit
    // clamp open on the HDR path, the Khronos Neutral curve desaturates the
    // above-1.0 highlight, lifting the green/blue channels.
    let count_red = |pixels: &[u8]| -> usize {
        pixels
            .chunks_exact(4)
            .filter(|rgba| rgba[0] > 150 && rgba[1] < 100 && rgba[2] < 100)
            .count()
    };
    let baseline_red = count_red(&baseline);
    assert!(
        baseline_red > 0,
        "baseline render should show the red plane; got {baseline_red} red pixels",
    );

    // ---- Render 2: bind an override on the red plane that pushes every
    // vertex far behind the camera. If the fix is in place, the red plane
    // disappears regardless of whether instancing is active. If the bug
    // returns, the red plane stays put because the instanced shader ignores
    // the override.
    let displaced: Vec<f32> = (0..4).flat_map(|_| [0.0_f32, 0.0, -1000.0]).collect();
    let override_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_position_override"),
        size: (displaced.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&override_buf, 0, bytemuck::cast_slice(&displaced));
    renderer
        .resources_mut()
        .set_position_override_buffer(red_id, override_buf)
        .unwrap();

    let overridden = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let overridden_red = count_red(&overridden);

    assert_eq!(
        overridden_red, 0,
        "with the position override pushing the red plane's vertices off-screen,\n\
         no red pixels should remain. Got {overridden_red} red (baseline had \
         {baseline_red}). If this regresses, the item was routed through the \
         instanced pipeline (`mesh_instanced.wgsl`) which has no awareness of \
         `has_position_override`, OR the per-item ObjectUniform write was \
         skipped so the shader flag stayed at 0.",
    );
}

/// An external scalar source must drive GPU marching cubes per frame: the
/// slab scalar buffers are refreshed from the consumer's buffer before every
/// dispatch. The uploaded CPU volume has no surface at the isovalue; writing
/// a sphere field into the external buffer makes one appear, and writing the
/// empty field again makes it vanish, proving both the copy and its
/// per-frame cadence.
#[test]
fn mc_external_scalar_drives_isosurface() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let dims = [16u32, 16, 16];
    let spacing = [0.3f32; 3];
    let origin = [-(15.0 * 0.3) / 2.0; 3];
    let node_count = (dims[0] * dims[1] * dims[2]) as usize;

    // Uploaded field: uniformly far above the isovalue, no surface anywhere.
    let vol = viewport_lib::VolumeData {
        data: vec![10.0; node_count],
        dims,
        origin,
        spacing,
    };
    let volume_id = renderer
        .resources_mut()
        .upload_volume_for_mc(&device, &queue, &vol)
        .expect("mc volume upload");

    // Sphere field: distance from the origin; isovalue 1.5 is a sphere well
    // inside the grid.
    let mut sphere = vec![0.0f32; node_count];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * spacing[0];
                let wy = origin[1] + y as f32 * spacing[1];
                let wz = origin[2] + z as f32 * spacing[2];
                let idx = (x + y * dims[0] + z * dims[0] * dims[1]) as usize;
                sphere[idx] = (wx * wx + wy * wy + wz * wz).sqrt();
            }
        }
    }

    let scalar_src = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_mc_scalar_src"),
        size: (node_count * 4) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let make_frame = || -> FrameData {
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
            .scene
            .gpu_mc_jobs
            .push(viewport_lib::GpuMarchingCubesJob {
                volume_id,
                isovalue: 1.5,
                material: Material::default(),
                settings: ItemSettings::default(),
                cpu_data: None,
            });
        frame
    };
    let empty_frame = || -> FrameData {
        let mut frame = make_frame();
        frame.scene.gpu_mc_jobs.clear();
        frame
    };
    let diff_count = |a: &[u8], b: &[u8]| -> usize {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(pa, pb)| pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8))
            .count()
    };

    let empty = renderer.render_offscreen(&device, &queue, &empty_frame(), 64, 64);

    // Baseline: uploaded field has no surface.
    let flat = renderer.render_offscreen(&device, &queue, &make_frame(), 64, 64);
    assert_eq!(
        diff_count(&flat, &empty),
        0,
        "the uploaded all-above-iso field must extract no surface"
    );

    // Attach the external source and write the sphere field into it.
    queue.write_buffer(&scalar_src, 0, bytemuck::cast_slice(&sphere));
    renderer
        .resources_mut()
        .set_mc_scalar_source_buffer(volume_id, scalar_src.clone(), 0)
        .unwrap();
    let with_sphere = renderer.render_offscreen(&device, &queue, &make_frame(), 64, 64);
    assert!(
        diff_count(&with_sphere, &empty) > 0,
        "after the external sphere field is copied in, the isosurface must \
         render"
    );

    // Rewrite the buffer with the empty field: the surface must vanish on
    // the next frame, proving the copy happens every dispatch.
    queue.write_buffer(
        &scalar_src,
        0,
        bytemuck::cast_slice(&vec![10.0f32; node_count]),
    );
    let flat_again = renderer.render_offscreen(&device, &queue, &make_frame(), 64, 64);
    assert_eq!(
        diff_count(&flat_again, &empty),
        0,
        "rewriting the external buffer with an all-above-iso field must \
         remove the surface on the next frame"
    );
}

/// An external instance set must draw exactly the window of the consumer's
/// positions buffer selected by the item's instance range. The buffer holds
/// four positions: elements 0..2 far behind the camera, elements 2..4 in
/// front of it. Drawing `first_instance = 2, instance_count = 2` must show
/// geometry; re-pointing the range at 0..2 must show none.
#[test]
fn external_instances_render_with_instance_range_slice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // Elements 0 and 1 behind the camera, 2 and 3 visible near the origin.
    let positions: Vec<f32> = vec![
        0.0, 0.0, -1000.0, // 0
        0.0, 0.0, -1000.0, // 1
        -0.6, 0.0, 0.0, // 2
        0.6, 0.0, 0.0, // 3
    ];
    let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_external_positions"),
        size: (positions.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&positions));

    let set_id = renderer
        .resources_mut()
        .create_external_instance_set(
            &device,
            &viewport_lib::ExternalInstanceSetConfig::new(mesh_id, pos_buf),
        )
        .unwrap();

    let make_frame = |first: u32, count: u32| -> FrameData {
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
        let mut item = viewport_lib::ExternalInstancesItem::new(set_id, count);
        item.first_instance = first;
        item.scale = 0.4;
        item.colour = [1.0, 0.2, 0.2, 1.0];
        frame.scene.external_instances = vec![item];
        frame
    };

    let empty = renderer.render_offscreen(&device, &queue, &make_frame(0, 0), 64, 64);
    let visible = renderer.render_offscreen(&device, &queue, &make_frame(2, 2), 64, 64);
    let hidden = renderer.render_offscreen(&device, &queue, &make_frame(0, 2), 64, 64);

    let diff_count = |a: &[u8], b: &[u8]| -> usize {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(pa, pb)| pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8))
            .count()
    };

    assert!(
        diff_count(&visible, &empty) > 0,
        "instance range 2..4 selects the visible elements; the boxes must \
         render. If nothing shows, instance_index is not honouring the draw \
         call's first_instance or the storage window is wrong.",
    );
    assert_eq!(
        diff_count(&hidden, &empty),
        0,
        "instance range 0..2 selects only the behind-camera elements; the \
         image must match an empty scene.",
    );
}

/// The selection outline mask must rasterise override-driven geometry, not
/// the bind pose. A selected plane's override pushes every vertex far behind
/// the camera: the mesh vanishes, and the halo must vanish with it. If the
/// mask still reads the vertex buffer, a ghost outline of the bind-pose
/// silhouette remains in an otherwise geometry-free image.
#[test]
fn outline_mask_follows_position_override() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();

    let make_frame = |items: Vec<SceneRenderItem>| -> FrameData {
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
        frame.interaction.outline_selected = true;
        frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
        frame
    };

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material = Material::from_colour([1.0, 0.0, 0.0]);
    item.settings.selected = true;
    let frame = make_frame(vec![item]);

    // Empty reference: what the frame looks like with nothing drawn at all.
    let empty_frame = make_frame(Vec::new());
    let empty = renderer.render_offscreen(&device, &queue, &empty_frame, 64, 64);

    let diff_count = |a: &[u8], b: &[u8]| -> usize {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(pa, pb)| pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8))
            .count()
    };

    // Sanity: selected plane with no override differs from empty (mesh +
    // outline are drawn).
    let baseline = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert!(
        diff_count(&baseline, &empty) > 0,
        "baseline render should draw the selected plane and its outline"
    );

    // Override pushes the whole plane far behind the camera.
    let displaced: Vec<f32> = (0..4).flat_map(|_| [0.0_f32, 0.0, -1000.0]).collect();
    let override_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_outline_override"),
        size: (displaced.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&override_buf, 0, bytemuck::cast_slice(&displaced));
    renderer
        .resources_mut()
        .set_position_override_buffer(mesh_id, override_buf)
        .unwrap();

    let overridden = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let ghost = diff_count(&overridden, &empty);
    if ghost != 0 {
        for (i, (pa, pb)) in overridden
            .chunks_exact(4)
            .zip(empty.chunks_exact(4))
            .enumerate()
        {
            if pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8) {
                eprintln!(
                    "diff at ({}, {}): got {:?} empty {:?}",
                    i % 64,
                    i / 64,
                    pa,
                    pb
                );
            }
        }
    }
    assert_eq!(
        ghost, 0,
        "with the override pushing the selected plane off-screen the image \
         must match an empty scene; {ghost} differing pixels indicate the \
         outline mask (or mesh) still rasterises the bind pose.",
    );
}

/// A sliced override binding must read its window out of a pooled buffer, not
/// element 0. The pool holds two "bodies": elements 0..5 push vertices far
/// behind the camera, elements 5..9 hold the plane's real (visible)
/// positions. `base_element = 5` is deliberately not 256-byte aligned
/// (5 * 12 = 60 bytes): the window is applied by the shader via
/// `position_override_base`, not a buffer binding offset.
#[test]
fn position_override_slice_reads_correct_pool_window() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    let red_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
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

    let mut red_item = SceneRenderItem::default();
    red_item.mesh_id = red_id;
    red_item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    red_item.material = Material::from_colour([1.0, 0.0, 0.0]);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![red_item].into());

    // Pool: body A (elements 0..5) far behind the camera, body B (elements
    // 5..9) the plane's own corner positions.
    let mut pool_data: Vec<f32> = Vec::new();
    for _ in 0..5 {
        pool_data.extend_from_slice(&[0.0, 0.0, -1000.0]);
    }
    for p in &mesh.positions {
        pool_data.extend_from_slice(p);
    }
    let pool = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_override_pool"),
        size: (pool_data.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&pool, 0, bytemuck::cast_slice(&pool_data));

    let count_red = |pixels: &[u8]| -> usize {
        pixels
            .chunks_exact(4)
            .filter(|rgba| rgba[0] > 150 && rgba[1] < 100 && rgba[2] < 100)
            .count()
    };

    // Window over body B: the plane renders from the pool's tail.
    renderer
        .resources_mut()
        .set_position_override_buffer_sliced(red_id, pool.clone(), OverrideBufferSlice::new(5, 4))
        .unwrap();
    let body_b = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let body_b_red = count_red(&body_b);
    assert!(
        body_b_red > 0,
        "slice base_element = 5 should read the visible body B positions; \
         got {body_b_red} red pixels. If this regresses, the shader is \
         ignoring position_override_base and reading body A (off-screen) \
         from element 0.",
    );

    // Re-point the window at body A: everything moves behind the camera.
    renderer
        .resources_mut()
        .set_position_override_buffer_sliced(red_id, pool, OverrideBufferSlice::new(0, 4))
        .unwrap();
    let body_a = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let body_a_red = count_red(&body_a);
    assert_eq!(
        body_a_red, 0,
        "slice base_element = 0 covers body A (all vertices at z = -1000); \
         no red pixels should remain (body B window drew {body_b_red}).",
    );
}

/// Exercise the HiZ occlusion-cull path end to end: enabling occlusion culling
/// builds the HiZ pyramid from the depth written by the HDR scene pass (which
/// validates `hiz_copy.wgsl` / `hiz_reduce.wgsl` compile and run), and the next
/// frame's cull samples it (validating the extended `cull.wgsl`). Several boxes
/// are submitted so the instanced cull path engages, and a near box sits
/// directly in front of farther ones along the view direction so there is real
/// front-to-back occlusion to find.
///
/// The test asserts the render does not panic and produces pixels. When the
/// device supports GPU-driven culling, it also checks the cull breakdown is
/// monotonic: total >= frustum-survivors >= drawn.
#[test]
fn occlusion_culling_render_path_runs() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    renderer.set_occlusion_culling(true);
    assert!(renderer.occlusion_culling_enabled());

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
    // HiZ is built in the HDR scene pass, so post-processing must be on.
    frame.effects.post_process.enabled = true;

    // A column of boxes along the view direction (Z-up world, camera looks down
    // -Z by default here): a big near box and several smaller ones behind it.
    let mut items = Vec::new();
    for i in 0..8 {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_idx;
        let z = -(i as f32) * 1.5;
        let s = if i == 0 { 4.0 } else { 1.0 };
        item.model = (glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, z))
            * glam::Mat4::from_scale(glam::Vec3::splat(s)))
        .to_cols_array_2d();
        items.push(item);
    }
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    // Render several frames, nudging the camera each frame so the reprojection
    // uses a non-identity previous-to-current transform (exercises the inverse
    // view-projection path, not just a static-camera no-op). Frame 0 stores the
    // first depth, later frames reproject it and cull; the stats readback lags a
    // couple of frames before it lands.
    let base_view = frame.camera.render_camera.view;
    let mut last_pixels = Vec::new();
    for i in 0..4 {
        let nudge = glam::Mat4::from_translation(glam::Vec3::new(0.05 * i as f32, 0.0, 0.0));
        frame.camera.render_camera.view = base_view * nudge;
        last_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    }
    assert_eq!(last_pixels.len(), 64 * 64 * 4);
    assert!(
        last_pixels.iter().any(|&b| b != 0),
        "occlusion render produced an all-zero image",
    );

    let stats = renderer.last_frame_stats();
    if stats.gpu_culling_active {
        let total = stats.gpu_culled_total.expect("total should be read back");
        let frustum = stats
            .gpu_frustum_visible
            .expect("frustum-survivors should be read back");
        let drawn = stats
            .gpu_visible_instances
            .expect("drawn count should be read back");
        assert!(
            total >= frustum && frustum >= drawn,
            "cull breakdown must be monotonic: total={total} frustum={frustum} drawn={drawn}",
        );
        assert_eq!(total, 8, "all 8 instances should enter the cull");
    }
}

/// Regression test: the HiZ reprojection passes must use 2D dispatches. At a
/// large depth resolution a 1D dispatch over `w * h / 64` exceeds the 65535
/// per-dimension workgroup limit and wgpu raises a validation error. 2048x2048
/// gives 65536 workgroups, one over the limit, so this would panic before the
/// fix. Renders two frames so the reprojection (which needs a stored prior
/// depth) actually runs.
#[test]
fn occlusion_large_viewport_no_dispatch_overflow() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    renderer.set_occlusion_culling(true);

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    let dim = 2048u32;
    frame.camera.viewport_size = [dim as f32, dim as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.post_process.enabled = true;

    let mut items = Vec::new();
    for i in 0..4 {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_idx;
        item.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -(i as f32) * 1.5))
            .to_cols_array_2d();
        items.push(item);
    }
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    // Frame 0 stores depth; frame 1 reprojects it (runs the init/scatter passes
    // that would overflow with a 1D dispatch). No panic == pass.
    for _ in 0..2 {
        let _ = renderer.render_offscreen(&device, &queue, &frame, dim, dim);
    }
}

/// Occlusion culling must run on the LDR path too, not just HDR. With
/// post-processing off, the scene renders through `render_frame_ldr`, which now
/// keeps the scene depth and copies it into the HiZ prev-depth target. This
/// exercises that store path (sampling the LDR depth target, which required
/// adding TEXTURE_BINDING + a depth-only view) across several moving-camera
/// frames; a no-panic run with non-zero output is the pass.
#[test]
fn occlusion_culling_ldr_path_runs() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    renderer.set_occlusion_culling(true);

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
    // LDR path: post-processing OFF (this is the path the fix targets).
    frame.effects.post_process.enabled = false;

    let mut items = Vec::new();
    for i in 0..8 {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_idx;
        let z = -(i as f32) * 1.5;
        let s = if i == 0 { 4.0 } else { 1.0 };
        item.model = (glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, z))
            * glam::Mat4::from_scale(glam::Vec3::splat(s)))
        .to_cols_array_2d();
        items.push(item);
    }
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    let base_view = frame.camera.render_camera.view;
    let mut last_pixels = Vec::new();
    for i in 0..4 {
        let nudge = glam::Mat4::from_translation(glam::Vec3::new(0.05 * i as f32, 0.0, 0.0));
        frame.camera.render_camera.view = base_view * nudge;
        last_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    }
    assert_eq!(last_pixels.len(), 64 * 64 * 4);
    assert!(
        last_pixels.iter().any(|&b| b != 0),
        "LDR occlusion render produced an all-zero image",
    );
}

/// Regression test for the per-object LOD draw path: a LOD item that is culled
/// (below its group's `cull_below` size) must actually be skipped by the paint
/// pass, not just by the stats. The bug was that LOD resolve mutated a throwaway
/// copy while the per-object draw re-read the raw `frame.scene.surfaces`, so a
/// culled non-instanced item still drew at full detail. A `DifferentColour`
/// back-face policy forces the item onto the per-object path (the path the fix
/// targets). A culled item must render identically to an empty scene.
#[test]
fn lod_culled_per_object_item_is_not_drawn() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let full = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let crude = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let group = renderer
        .resources_mut()
        .register_lod_group(&[full, crude], &[0.5, 0.0])
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
    frame.effects.post_process.enabled = true;

    // A single non-instanced (per-object) LOD item filling the view centre. The
    // DifferentColour back-face policy is what forces it onto the per-object path.
    let mut item = SceneRenderItem::default();
    item.mesh_id = full;
    item.lod_group = Some(group);
    item.material.backface_policy = BackfacePolicy::DifferentColour([1.0, 0.2, 0.2]);
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(3.0)).to_cols_array_2d();

    // Empty scene baseline.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    let empty = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    // cull_below defaults to None, so the item draws.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let drawn = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        drawn, empty,
        "control: the per-object LOD item should be drawn when not culled",
    );

    // Force the item below the cull size: it must now contribute nothing.
    renderer
        .resources_mut()
        .set_lod_cull_below(group, Some(100.0))
        .unwrap();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let culled = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(
        culled, empty,
        "a culled per-object LOD item must be skipped in the paint pass",
    );
}

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

/// Toggling DebugVis swaps the lit pipelines between the stripped shader
/// variant (no debug storage write, early-Z friendly) and the full debug
/// variant, and back. Exercises the rebuild path end to end: a validation
/// error in either module variant fails the prepare calls.
#[test]
fn debug_vis_toggle_rebuilds_lit_pipelines() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // Default: stripped variant.
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // On: rebuilds with the debug block present.
    frame.effects.lighting.debug_vis.active = true;
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // Off again: rebuilds stripped.
    frame.effects.lighting.debug_vis.active = false;
    let _ = renderer.pass().prepare(&device, &queue, &frame);
}

/// Regression test for the recurring class of bug where a draw site
/// unconditionally binds wgpu group index 2. iced_wgpu 0.14 requests a device
/// with `max_bind_groups: 2` and gives consumers no way to raise it, so any
/// unconditional `set_bind_group(2, ...)` fails wgpu validation the moment a
/// mesh, shadow, or HDR pass draws on that device. This has been fixed and
/// silently reintroduced by new draw sites multiple times (see
/// `docs/issues/iced-max-bind-groups-2-draw-path-incomplete.md`) because the
/// only thing that previously caught it was manually running the iced
/// example. This test exercises the same crash surface headlessly on every
/// `cargo test`, with no reliance on remembering to gate a new call site.
///
/// wgpu's default (no error-scope) behaviour is to raise the validation
/// error through the uncaptured-error handler, which panics the calling
/// thread; a regression here fails the test the same way it crashed iced.
#[test]
fn two_bind_group_device_renders_without_validation_errors() {
    let Some((device, queue)) = headless_device_limited_bind_groups() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(
        device.limits().max_bind_groups,
        2,
        "adapter did not honour the requested 2-group limit; test would not \
         be exercising the iced crash surface"
    );

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    // Default lighting has shadows_enabled = true, and mesh items default to
    // cast_shadows = true, so the shadow pass (the site that crashed the
    // iced example even after the mesh draw path was gated) runs below.

    // One opaque box, one transparent box so the OIT pipeline family draws
    // too, not just the opaque path.
    let mut opaque = SceneRenderItem::default();
    opaque.mesh_id = mesh_id;
    let mut transparent = SceneRenderItem::default();
    transparent.mesh_id = mesh_id;
    transparent.settings.opacity = 0.5;
    transparent.model =
        glam::Mat4::from_translation(glam::Vec3::new(1.5, 0.0, 0.0)).to_cols_array_2d();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![opaque, transparent].into());

    // LDR path.
    frame.effects.post_process.enabled = false;
    let ldr_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(ldr_pixels.len(), 64 * 64 * 4);

    // HDR path: exercises the deform/OIT binds inside hdr_path.rs and the
    // full tonemap/bloom/SSAO/FXAA post chain.
    frame.effects.post_process.enabled = true;
    let hdr_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(hdr_pixels.len(), 64 * 64 * 4);

    // Wireframe mode is a separate draw path (wireframe pipeline, its own
    // group-2 deform dummy bind) from both the solid and HDR paths above.
    frame.viewport.wireframe_mode = true;
    let wireframe_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(wireframe_pixels.len(), 64 * 64 * 4);
}

/// End-to-end proof of the material-plugin path: a registered plugin with a
/// toon `shade_light` + `shade_ambient` body, selected per material via
/// `Material::shading_plugin`, must change the rendered output on the LDR
/// path, respond to live params writes, and run the HDR and OIT paths
/// without validation errors (wgpu's uncaptured-error handler panics the
/// thread on any validation failure, so completing the renders is itself the
/// assertion).
#[test]
fn material_plugin_changes_rendered_output() {
    struct ToonPlugin;
    impl viewport_lib::MaterialPlugin for ToonPlugin {
        fn name(&self) -> &'static str {
            "toon_test"
        }
        fn wgsl_body(&self) -> String {
            "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    let bands = max(material_params[0].x, 1.0);
    let ndl = max(dot(surf.normal, light.l), 0.0);
    let stepped = ceil(ndl * bands) / bands;
    return surf.base_colour * stepped * light.radiance * light.shadow;
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return surf.base_colour * material_params[0].y * surf.ao;
}
"
            .to_string()
        }
        fn initial_params(&self) -> [[f32; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S] {
            let mut p = [[0.0; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S];
            p[0] = [2.0, 0.25, 0.0, 0.0];
            p
        }
    }

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let plugin_id = renderer
        .resources_mut()
        .register_material_plugin(&device, &ToonPlugin)
        .expect("register material plugin");
    // Idempotent re-registration returns the same id.
    assert_eq!(
        renderer
            .resources_mut()
            .register_material_plugin(&device, &ToonPlugin)
            .unwrap(),
        plugin_id
    );

    // Stats: registered but not yet drawn, so no pipelines are built.
    let stats = renderer.resources().material_plugin_stats();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].name, "toon_test");
    assert_eq!(stats[0].variants, 1);
    assert_eq!(stats[0].texture_count, 0);
    assert_eq!(stats[0].pipelines_built, 0);

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.post_process.enabled = false;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let builtin = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    item.material.shading_plugin = Some(plugin_id);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let toon = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        builtin, toon,
        "selecting the plugin must change the LDR output"
    );

    // Drawing through the plugin lazily built its full pipeline set.
    let stats = renderer.resources().material_plugin_stats();
    assert_eq!(stats[0].pipelines_built, 9);

    // Live params: raising the band count and ambient changes the image.
    let params = renderer
        .resources_mut()
        .material_plugin_params_handle(plugin_id)
        .expect("params handle");
    params.write(&queue, &[[16.0, 0.9, 0.0, 0.0]]);
    let toon_reparam = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        toon, toon_reparam,
        "params writes must be live in the next render"
    );

    // Per-material params: a second variant with different params renders
    // differently from the default variant in the same frame's plugin.
    let variant_b = renderer
        .resources_mut()
        .create_material_plugin_variant(&device, plugin_id, &[[2.0, 0.05, 0.0, 0.0]], &[])
        .expect("variant");
    assert_eq!(variant_b.plugin_index(), plugin_id.plugin_index());
    assert_ne!(variant_b.variant_index(), plugin_id.variant_index());
    // Variants share the plugin's pipeline set; only the variant count grows.
    let stats = renderer.resources().material_plugin_stats();
    assert_eq!(stats[0].variants, 2);
    assert_eq!(stats[0].pipelines_built, 9);
    item.material.shading_plugin = Some(variant_b);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let toon_variant_b = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        toon_reparam, toon_variant_b,
        "a second variant must carry its own params window"
    );
    item.material.shading_plugin = Some(plugin_id);

    // HDR path (and, with a transparent second item, the OIT path).
    let mut transparent = item.clone();
    transparent.settings.opacity = 0.5;
    transparent.model =
        glam::Mat4::from_translation(glam::Vec3::new(1.5, 0.0, 0.0)).to_cols_array_2d();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item, transparent].into());
    frame.effects.post_process.enabled = true;
    let hdr = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(hdr.len(), 64 * 64 * 4);
}

/// A plugin declaring a texture slot samples the per-variant texture: a
/// variant bound to a red texture must render differently from the default
/// variant's 1x1 white fallback.
#[test]
fn material_plugin_texture_variant_renders() {
    struct StripePlugin;
    impl viewport_lib::MaterialPlugin for StripePlugin {
        fn name(&self) -> &'static str {
            "stripe_test"
        }
        fn texture_count(&self) -> u32 {
            1
        }
        fn wgsl_body(&self) -> String {
            "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    let tex = textureSampleGrad(material_texture_0, material_sampler, surf.uv, surf.uv_ddx, surf.uv_ddy).rgb;
    let ndl = max(dot(surf.normal, light.l), 0.0);
    return surf.base_colour * tex * ndl * light.radiance * light.shadow;
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    let tex = textureSampleGrad(material_texture_0, material_sampler, surf.uv, surf.uv_ddx, surf.uv_ddy).rgb;
    return surf.base_colour * tex * 0.3;
}
"
            .to_string()
        }
    }

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let plugin_id = renderer
        .resources_mut()
        .register_material_plugin(&device, &StripePlugin)
        .expect("register");

    let red = vec![[255u8, 0, 0, 255]; 16].concat();
    let red_tex = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 4, 4, &red)
        .expect("upload texture");
    let red_variant = renderer
        .resources_mut()
        .create_material_plugin_variant(&device, plugin_id, &[], &[red_tex])
        .expect("variant");

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.post_process.enabled = false;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material.shading_plugin = Some(plugin_id);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let white_fallback = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    item.material.shading_plugin = Some(red_variant);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    let red_textured = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        white_fallback, red_textured,
        "a variant's texture must reach the hook"
    );
}

/// A plugin with `reads_vertex_attribute` sees `MeshData::extension_attributes`
/// as `surf.attr`; a mesh without the channel reads vec4(0).
#[test]
fn material_plugin_reads_vertex_attribute() {
    struct AttrPlugin;
    impl viewport_lib::MaterialPlugin for AttrPlugin {
        fn name(&self) -> &'static str {
            "attr_test"
        }
        fn reads_vertex_attribute(&self) -> bool {
            true
        }
        fn wgsl_body(&self) -> String {
            "\
fn recolor(surf: ShadingSurface, direct: vec3<f32>, ambient: vec3<f32>) -> vec3<f32> {
    return direct + ambient + surf.attr.rgb;
}
"
            .to_string()
        }
    }

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let plain_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let mut green_data = box_mesh();
    green_data.extension_attributes = Some(vec![[0.0, 0.6, 0.0, 0.0]; green_data.positions.len()]);
    let green_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &green_data)
        .unwrap();
    let plugin_id = renderer
        .resources_mut()
        .register_material_plugin(&device, &AttrPlugin)
        .expect("register");

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.post_process.enabled = false;

    let mut item = SceneRenderItem::default();
    item.mesh_id = plain_id;
    item.material.shading_plugin = Some(plugin_id);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let no_channel = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    item.mesh_id = green_id;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    let with_channel = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        no_channel, with_channel,
        "the extension attribute must reach surf.attr"
    );

    // The channel's green tint should show up in the with-channel image:
    // compare summed green minus red across the frame.
    let bias = |img: &[u8]| -> i64 {
        img.chunks(4)
            .map(|p| p[1] as i64 - p[0] as i64)
            .sum::<i64>()
    };
    assert!(
        bias(&with_channel) > bias(&no_channel),
        "attr (0, 0.6, 0) should bias the image toward green"
    );
}

/// The surface-authoring hook: a plugin that defines only `shade_surface`
/// renders its authored base colour and emissive under stock lighting, and
/// its alpha output is honoured only under Mask/Blend alpha modes.
#[test]
fn material_plugin_surface_hook_authors_surface_and_gated_alpha() {
    struct PaintPlugin;
    impl viewport_lib::MaterialPlugin for PaintPlugin {
        fn name(&self) -> &'static str {
            "paint_test"
        }
        fn wgsl_body(&self) -> String {
            // params[0] = (alpha, emissive_green, 0, 0)
            "\
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {
    var ov: SurfaceOverride;
    ov.base_colour = vec3<f32>(0.9, 0.1, 0.1);
    ov.normal = surf.normal;
    ov.metallic = surf.metallic;
    ov.roughness = surf.roughness;
    ov.emissive = vec3<f32>(0.0, material_params[0].y, 0.0);
    ov.alpha = material_params[0].x;
    return ov;
}
"
            .to_string()
        }
        fn initial_params(&self) -> [[f32; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S] {
            let mut p = [[0.0; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S];
            p[0] = [1.0, 0.0, 0.0, 0.0];
            p
        }
    }

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let plugin_id = renderer
        .resources_mut()
        .register_material_plugin(&device, &PaintPlugin)
        .expect("register");
    let params = renderer
        .resources_mut()
        .material_plugin_params_handle(plugin_id)
        .expect("params handle");

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.post_process.enabled = false;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let builtin = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    // Authored base colour under stock lighting.
    item.material.shading_plugin = Some(plugin_id);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let painted = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(builtin, painted, "surface hook must change the image");
    let red_bias = |img: &[u8]| -> i64 {
        img.chunks(4)
            .map(|p| p[0] as i64 - p[1] as i64)
            .sum::<i64>()
    };
    assert!(
        red_bias(&painted) > red_bias(&builtin),
        "authored red base colour should bias the image red"
    );

    // Hook emissive reaches the image.
    params.write(&queue, &[[1.0, 0.8, 0.0, 0.0]]);
    let emissive = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let green = |img: &[u8]| -> i64 { img.chunks(4).map(|p| p[1] as i64).sum::<i64>() };
    assert!(
        green(&emissive) > green(&painted),
        "hook emissive should raise green"
    );

    // Opaque materials ignore the hook's alpha entirely.
    params.write(&queue, &[[0.05, 0.0, 0.0, 0.0]]);
    let opaque_low_alpha = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(
        painted, opaque_low_alpha,
        "opaque draws must ignore hook alpha"
    );

    // Blend materials take it as output alpha.
    item.material.alpha_mode = viewport_lib::material::AlphaMode::Blend;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    params.write(&queue, &[[0.9, 0.0, 0.0, 0.0]]);
    let blend_high = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    params.write(&queue, &[[0.1, 0.0, 0.0, 0.0]]);
    let blend_low = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(blend_high, blend_low, "blend draws must honour hook alpha");

    // Mask materials re-test the cutoff against the hook's alpha: below it,
    // every fragment discards and the image matches an empty scene.
    item.material.alpha_mode = viewport_lib::material::AlphaMode::Mask(0.5);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    params.write(&queue, &[[0.1, 0.0, 0.0, 0.0]]);
    let mask_discarded = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    let empty = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(
        mask_discarded, empty,
        "mask draws below the hook alpha cutoff must discard"
    );
}

// The reference plugins shipped under examples/plugins/ must stay
// registrable: their WGSL runs through the full composer + wgpu validation
// at registration, so this catches contract or prefixer regressions (e.g.
// a body local named like a ShadingSurface field).
#[path = "../examples/plugins/surface_detail_plugin.rs"]
mod surface_detail_plugin;
#[path = "../examples/plugins/toon_plugin.rs"]
mod toon_plugin;

#[test]
fn example_reference_plugins_register() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let resources = renderer.resources_mut();
    resources
        .register_material_plugin(&device, &toon_plugin::ToonPlugin)
        .expect("toon plugin registers");
    resources
        .register_material_plugin(&device, &toon_plugin::RimPlugin)
        .expect("rim plugin registers");
    resources
        .register_material_plugin(&device, &surface_detail_plugin::DetailLayerPlugin)
        .expect("detail plugin registers");
    resources
        .register_material_plugin(&device, &surface_detail_plugin::ParallaxPlugin)
        .expect("parallax plugin registers");
}

/// A camera-facing unit quad in the XY plane (normal +Z), for the tone-map
/// background-composite tests below.
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

/// A frame looking down -Z at the origin with a flat background colour and the
/// viewport overlays off, sized `size` x `size`.
fn tonemap_frame(size: u32, background: [f32; 4]) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(background);
    frame
}

/// Tone-map composite: transparent content over the *empty* background must be
/// composited over the flat background colour, not replace it with its own dim
/// premultiplied value. Regression test for the fade-to-black artifact where a
/// faint transparent draw over empty scene read darker than the background.
///
/// A dim, semi-transparent, unlit quad (no depth write → stays at the far plane,
/// so the tone-mapper treats it as background) is drawn over a mid-grey
/// background. Its centre must be at least as bright as a background corner.
#[test]
fn transparent_over_empty_background_not_darker() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &quad_mesh())
        .unwrap();

    let size = 64u32;
    let bg = [0.3, 0.3, 0.3, 1.0];
    let mut frame = tonemap_frame(size, bg);

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    // Cover the centre but leave the corners as background.
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(2.0)).to_cols_array_2d();
    item.material = Material::from_colour([0.6, 0.6, 0.6]);
    item.settings.unlit = true;
    item.settings.opacity = 0.3;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let luma = |x: u32, y: u32| {
        let i = ((y * size + x) * 4) as usize;
        px[i] as i32 + px[i + 1] as i32 + px[i + 2] as i32
    };
    let center = luma(size / 2, size / 2);
    let corner = luma(1, 1);

    // Corner must be the pure grey background (guards against the quad covering
    // it, which would invalidate the comparison).
    assert!(
        corner > 420 && corner < 470,
        "corner {corner} is not the expected grey background"
    );
    // Centre (transparent quad over background) must not be darker than it.
    assert!(
        center >= corner,
        "transparent quad centre {center} darker than background {corner}"
    );
}

/// Tone-map composite: bloom must glow into the *empty* background. The
/// tone-mapper's background fast path used to return the flat background wherever
/// nothing drew directly (HDR alpha ~ 0), before bloom was added, clipping a
/// bloom halo hard at the silhouette of whatever cast it.
///
/// A small, bright (emissive) opaque quad is rendered with bloom off, then on. At
/// least one pixel that reads as pure background with bloom off must get brighter
/// with bloom on: the halo glowing past the quad's edge into the background.
#[test]
fn bloom_glows_into_empty_background() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &quad_mesh())
        .unwrap();

    let size = 128u32;
    let bg = [0.25, 0.25, 0.25, 1.0];

    let mut render = |bloom: bool| -> Vec<u8> {
        let mut frame = tonemap_frame(size, bg);
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        // Small, so there is surrounding background for the halo to fall on.
        item.model = glam::Mat4::from_scale(glam::Vec3::splat(0.6)).to_cols_array_2d();
        // Emissive well above the bloom threshold; opaque (sharp alpha edge).
        item.material = Material::from_colour([0.02, 0.02, 0.02]);
        item.material.emissive = [6.0, 6.0, 6.0];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        frame.effects.post_process.enabled = true;
        frame.effects.post_process.bloom = bloom;
        frame.effects.post_process.bloom_threshold = 0.7;
        frame.effects.post_process.bloom_intensity = 2.0;
        renderer.render_offscreen(&device, &queue, &frame, size, size)
    };

    let off = render(false);
    let on = render(true);

    let luma = |px: &[u8], i: usize| px[i] as i32 + px[i + 1] as i32 + px[i + 2] as i32;
    let corner = luma(&off, ((2 * size + 2) * 4) as usize); // pure background

    // Sanity: the emissive quad actually drew something bright with bloom off.
    let max_off = (0..(size * size) as usize)
        .map(|p| luma(&off, p * 4))
        .max()
        .unwrap();
    assert!(
        max_off > corner + 200,
        "emissive quad did not draw (max {max_off}, background {corner})"
    );

    // Biggest brightening, among pixels that are pure background with bloom off,
    // when bloom is turned on. That is the halo glowing into the background,
    // impossible before the fix (those pixels were clamped to the background).
    let max_halo = (0..(size * size) as usize)
        .filter(|&p| (luma(&off, p * 4) - corner).abs() <= 6)
        .map(|p| luma(&on, p * 4) - luma(&off, p * 4))
        .max()
        .unwrap();
    assert!(
        max_halo > 25,
        "bloom did not glow into the empty background (max background brightening {max_halo})"
    );
}

// ---------------------------------------------------------------------------
// Foreground pass
// ---------------------------------------------------------------------------

/// Frame with the default orbit camera, a 64x64 viewport, and no chrome.
fn foreground_frame() -> FrameData {
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

fn centre_pixel(img: &[u8]) -> [u8; 4] {
    let i = ((32 * 64) + 32) * 4;
    [img[i], img[i + 1], img[i + 2], img[i + 3]]
}

fn coloured_item(mesh_id: MeshId, colour: [f32; 3], model: glam::Mat4) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = Material::default();
    item.material.base_colour = colour;
    item.model = model.to_cols_array_2d();
    item
}

/// A foreground item draws over world geometry that would fully occlude it:
/// the world box encloses the foreground box, so without the cleared-depth
/// pass the foreground box could never be visible.
#[test]
fn foreground_item_draws_over_occluding_geometry() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = foreground_frame();
    let occluder = coloured_item(
        mesh_id,
        [0.1, 0.1, 0.9],
        glam::Mat4::from_scale(glam::Vec3::splat(3.0)),
    );
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![occluder].into());
    let without_fg = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    let fg = coloured_item(mesh_id, [0.9, 0.1, 0.1], glam::Mat4::IDENTITY);
    frame.scene.foreground_items.push(fg);
    let with_fg = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    assert_ne!(
        centre_pixel(&without_fg),
        centre_pixel(&with_fg),
        "foreground item enclosed by world geometry must still be visible"
    );
    let c = centre_pixel(&with_fg);
    assert!(
        c[0] > c[2],
        "centre pixel should show the red foreground item, got {c:?}"
    );
}

/// The override projection's small near plane keeps close-held geometry from
/// being clipped; without the override the same item is cut by the scene near
/// plane and contributes nothing.
#[test]
fn foreground_override_projection_avoids_near_clip() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = foreground_frame();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    let empty = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    // A small box held 0.05 world units in front of the eye: inside the
    // scene near plane (0.1 for the default camera).
    let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
    let fwd = glam::Vec3::from(frame.camera.render_camera.forward);
    let model = glam::Mat4::from_translation(eye + fwd * 0.05)
        * glam::Mat4::from_scale(glam::Vec3::splat(0.02));
    let fg = coloured_item(mesh_id, [0.9, 0.1, 0.1], model);
    frame.scene.foreground_items.push(fg);

    let scene_near = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(
        empty, scene_near,
        "item inside the scene near plane must be clipped without an override"
    );

    frame.effects.foreground = Some(viewport_lib::ForegroundPass {
        projection: Some(viewport_lib::ForegroundProjection {
            fov_y: 65_f32.to_radians(),
            near: 0.005,
            far: None,
        }),
    });
    let overridden = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        empty, overridden,
        "the override near plane must make the close-held item visible"
    );
}

/// Foreground items survive the screen-space post effects: with SSAO, bloom,
/// and DOF all enabled the foreground item must still reach the output (DOF
/// redirects the tone-map input, and the coverage mask keeps the item sharp
/// and un-darkened).
#[test]
fn foreground_item_survives_post_effects() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = foreground_frame();
    frame.effects.post_process.ssao = true;
    frame.effects.post_process.bloom = true;
    frame.effects.post_process.dof_enabled = true;
    let occluder = coloured_item(
        mesh_id,
        [0.1, 0.1, 0.9],
        glam::Mat4::from_scale(glam::Vec3::splat(3.0)),
    );
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![occluder].into());
    let without_fg = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    let fg = coloured_item(mesh_id, [0.9, 0.1, 0.1], glam::Mat4::IDENTITY);
    frame.scene.foreground_items.push(fg);
    let with_fg = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    assert_ne!(
        centre_pixel(&without_fg),
        centre_pixel(&with_fg),
        "foreground item must survive SSAO + bloom + DOF"
    );
    let c = centre_pixel(&with_fg);
    assert!(
        c[0] > c[2],
        "centre pixel should stay red under post effects, got {c:?}"
    );
}

/// The LDR path (post-processing disabled) draws foreground items through its
/// own pass.
#[test]
fn foreground_item_visible_on_ldr_path() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = foreground_frame();
    frame.effects.post_process.enabled = false;
    let occluder = coloured_item(
        mesh_id,
        [0.1, 0.1, 0.9],
        glam::Mat4::from_scale(glam::Vec3::splat(3.0)),
    );
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![occluder].into());
    let without_fg = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    let fg = coloured_item(mesh_id, [0.9, 0.1, 0.1], glam::Mat4::IDENTITY);
    frame.scene.foreground_items.push(fg);
    let with_fg = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    assert_ne!(
        centre_pixel(&without_fg),
        centre_pixel(&with_fg),
        "LDR path must draw foreground items"
    );
}

/// Scene clip planes never slice foreground items: the pass binds a
/// clip-disabled group 0.
#[test]
fn foreground_item_ignores_scene_clip_planes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // A plane that clips away everything (keeps only z >= 100).
    let mut clip = viewport_lib::ClipObject::default();
    clip.shape = viewport_lib::ClipShape::Plane {
        normal: [0.0, 0.0, 1.0],
        distance: -100.0,
        cap_colour: None,
        display_center: None,
    };
    clip.colour = None;
    clip.edge_colour = None;

    let mut frame = foreground_frame();
    frame.effects.clip_objects.push(clip);
    let world = coloured_item(
        mesh_id,
        [0.1, 0.1, 0.9],
        glam::Mat4::from_scale(glam::Vec3::splat(3.0)),
    );
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![world].into());
    let clipped_world = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    let empty = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(
        clipped_world, empty,
        "sanity: the clip plane must remove the world geometry entirely"
    );

    let fg = coloured_item(mesh_id, [0.9, 0.1, 0.1], glam::Mat4::IDENTITY);
    frame.scene.foreground_items.push(fg);
    let with_fg = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        empty, with_fg,
        "foreground item must not be sliced by the scene clip plane"
    );
}

/// Foreground transparency is sorted alpha blending inside the pass: a
/// blended foreground item over an opaque foreground item changes the result.
#[test]
fn foreground_transparency_blends_over_opaque_foreground() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut frame = foreground_frame();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    let red = coloured_item(mesh_id, [0.9, 0.1, 0.1], glam::Mat4::IDENTITY);
    frame.scene.foreground_items.push(red.clone());
    let opaque_only = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    let mut glass = coloured_item(
        mesh_id,
        [0.1, 0.1, 0.9],
        glam::Mat4::from_scale(glam::Vec3::splat(1.5)),
    );
    glass.settings.opacity = 0.4;
    frame.scene.foreground_items.push(glass);
    let blended = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    assert_ne!(
        centre_pixel(&opaque_only),
        centre_pixel(&blended),
        "transparent foreground item must blend over the opaque one"
    );
}

/// An item-type plugin with `draws_foreground` draws into the foreground pass
/// through a pipeline built by `build_foreground_pipeline`.
#[test]
fn plugin_paint_foreground_draws_into_pass() {
    struct FgCollection {
        settings: ItemSettings,
    }
    impl PluginItemCollection for FgCollection {
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

    struct FgPlugin {
        pipeline: wgpu::RenderPipeline,
    }
    impl ItemTypePlugin for FgPlugin {
        fn type_name(&self) -> &'static str {
            "fg_test"
        }
        fn draws_foreground(&self) -> bool {
            true
        }
        fn paint_foreground<'a>(
            &'a self,
            pass: &mut wgpu::RenderPass<'a>,
            _ctx: &viewport_lib::plugin_api::PaintContext<'a>,
            _items: &'a dyn PluginItemCollection,
        ) {
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..3, 0..1);
        }
    }

    const FG_WGSL: &str = r#"
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    return vec4<f32>(verts[vi], 0.5, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 2.0, 0.0, 1.0);
}
"#;

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fg_test_shader"),
        source: wgpu::ShaderSource::Wgsl(FG_WGSL.into()),
    });
    let mut opts = viewport_lib::resources::PluginPipelineOpts::new(
        Some("fg_test_pipeline"),
        &shader,
        "vs",
        "fs",
        &[],
    );
    opts.primitive.cull_mode = None;
    let pipeline = renderer
        .resources()
        .build_foreground_pipeline(&device, &opts);
    renderer.with_item_type_plugin(&device, Box::new(FgPlugin { pipeline }));

    let mut frame = foreground_frame();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    let without = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    frame.scene.submit_plugin_items(
        "fg_test",
        FgCollection {
            settings: ItemSettings::default(),
        },
    );
    let with_plugin = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    assert_ne!(
        without, with_plugin,
        "plugin foreground draw must be visible"
    );
    let green = |img: &[u8]| -> i64 { img.chunks(4).map(|p| p[1] as i64).sum::<i64>() };
    assert!(
        green(&with_plugin) > green(&without),
        "plugin's green fullscreen triangle should raise green"
    );
}

/// The lit clamp policy: on the HDR path the lit (pre-emissive) term may
/// exceed 1.0 into the Rgba16Float target, so raising a light's intensity far
/// past the point where the old [0, 1] clamp saturated must still change the
/// tone-mapped output. On the LDR path the historical [0, 1] clamp holds, so
/// the same intensity change above saturation produces identical pixels.
#[test]
fn lit_clamp_hdr_passes_above_one_ldr_saturates() {
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
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(3.0)).to_cols_array_2d();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    // Ambient-only lighting: the hemisphere term is uniform per face, so a
    // large intensity saturates every lit pixel and the comparison is not
    // polluted by grazing-angle pixels that sit below the clamp.
    let mut render_at = |intensity: f32, hdr: bool, renderer: &mut ViewportRenderer| {
        frame.effects.post_process.enabled = hdr;
        frame.effects.lighting.lights = Vec::new();
        frame.effects.lighting.hemisphere_intensity = intensity;
        renderer.render_offscreen(&device, &queue, &frame, 64, 64)
    };

    // Both intensities are far past the old saturation point for every lit
    // pixel, so under a [0, 1] clamp they render identically.
    let hdr_lo = render_at(50.0, true, &mut renderer);
    let hdr_hi = render_at(500.0, true, &mut renderer);
    assert_ne!(
        hdr_lo, hdr_hi,
        "HDR path: lit output must respond to intensity above 1.0 (clamp should not saturate before tone mapping)"
    );

    let ldr_lo = render_at(50.0, false, &mut renderer);
    let ldr_hi = render_at(500.0, false, &mut renderer);
    assert_eq!(
        ldr_lo, ldr_hi,
        "LDR path: the [0, 1] lit clamp must hold (byte-identical output above saturation)"
    );
}

/// Bloom firefly cap: with the lit path unclamped on the HDR path, a single
/// very bright texel (tight specular highlight, hot emissive) must not bloom
/// into a large blob. `bloom_max_brightness` scales each pixel's luminance
/// down before thresholding; the capped render must produce a strictly
/// smaller bright area than an uncapped one, and a bounded one in absolute
/// terms.
#[test]
fn bloom_firefly_cap_bounds_blob_size() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // A small quad with an extremely hot emissive term: a deterministic
    // firefly, independent of specular geometry.
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.25, -0.25, 0.0],
        [0.25, -0.25, 0.0],
        [0.25, 0.25, 0.0],
        [-0.25, 0.25, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
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
    frame.effects.post_process.enabled = true;
    frame.effects.post_process.bloom = true;

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    let mut mat = Material::from_colour([1.0, 1.0, 1.0]);
    mat.emissive = [400.0, 400.0, 400.0];
    item.material = mat;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let bright_pixels = |img: &[u8]| -> usize {
        img.chunks_exact(4)
            .filter(|p| p[0] > 220 && p[1] > 220 && p[2] > 220)
            .count()
    };

    frame.effects.post_process.bloom_max_brightness = f32::MAX;
    let uncapped = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    frame.effects.post_process.bloom_max_brightness = 4.0;
    let capped = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    let (u, c) = (bright_pixels(&uncapped), bright_pixels(&capped));
    assert!(
        u > c,
        "uncapped bloom should spread a hot texel further than capped (uncapped {u} vs capped {c} bright pixels)"
    );
    assert!(
        c < 64 * 64 / 10,
        "capped bloom blob should stay small; got {c} bright pixels"
    );
}

/// Lightmap consumption, Replace mode: a mesh with a solid-red radiance lightmap and
/// no direct lights must render red, where the default sky/hemisphere ambient
/// would render blue-grey. Exercises the whole path: the UV1 sidecar (binding
/// 16), the lightmap texture (binding 17), the `lightmap_mode` object field, and
/// the `apply_lightmap` blend in the ambient slot.
#[test]
fn lightmap_replace_mode_recolors_object() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // A solid red radiance texture. UV1 is left empty (set_lightmap pads to
    // zero), so every fragment samples the same red texel.
    let red = vec![[255u8, 0, 0, 255]; 16].concat();
    let radiance = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 4, 4, &red)
        .expect("upload lightmap texture");
    renderer
        .resources_mut()
        .set_lightmap(
            &device,
            mesh_id,
            &[],
            viewport_lib::resources::LightmapData::NonDirectional { radiance },
            viewport_lib::resources::LightmapMode::Replace,
        )
        .expect("set lightmap");

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
    // No direct lights: the object colour is purely its indirect (lightmap) term.
    frame.effects.lighting.lights = vec![];

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material.shading_model = ShadingModel::Pbr;
    item.material.base_colour = [1.0, 1.0, 1.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let (w, h) = (64u32, 64u32);
    let pixels = renderer.render_offscreen(&device, &queue, &frame, w, h);

    let mut best = (0u8, 0u8, 0u8);
    for px in pixels.chunks_exact(4) {
        if px[0] > best.0 {
            best = (px[0], px[1], px[2]);
        }
    }
    assert!(
        best.0 as i32 > best.2 as i32 + 40 && best.0 as i32 > best.1 as i32 + 40,
        "lightmapped object should be red: brightest-red pixel was {best:?}"
    );
}

/// Lightmap AmbientOcclusion mode: a near-black occlusion lightmap must darken the
/// object relative to the same scene with no lightmap. Both renders share a plain
/// sky ambient, so the only difference is the occlusion multiply.
#[test]
fn lightmap_ao_mode_darkens_object() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let build_frame = || {
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
        frame.effects.lighting.lights = vec![];
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        item.material.shading_model = ShadingModel::Pbr;
        item.material.base_colour = [1.0, 1.0, 1.0];
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        frame
    };

    let (w, h) = (64u32, 64u32);
    // Mean over the central region, which the box fills, so the AO multiply on
    // the box's ambient is not diluted by the sky background.
    let center_mean = |pixels: &[u8]| -> f64 {
        let (mut sum, mut n) = (0.0f64, 0u64);
        for y in (h / 4)..(3 * h / 4) {
            for x in (w / 4)..(3 * w / 4) {
                let i = ((y * w + x) * 4) as usize;
                sum += pixels[i] as f64 + pixels[i + 1] as f64 + pixels[i + 2] as f64;
                n += 1;
            }
        }
        sum / n as f64
    };

    let baseline = center_mean(&renderer.render_offscreen(&device, &queue, &build_frame(), w, h));

    // Dark occlusion factor (0.1) multiplies the ambient term.
    let dark = vec![[26u8, 26, 26, 255]; 16].concat();
    let occlusion = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 4, 4, &dark)
        .expect("upload occlusion texture");
    renderer
        .resources_mut()
        .set_lightmap(
            &device,
            mesh_id,
            &[],
            viewport_lib::resources::LightmapData::AmbientOcclusion { occlusion },
            viewport_lib::resources::LightmapMode::AmbientOcclusion,
        )
        .expect("set lightmap");

    let occluded = center_mean(&renderer.render_offscreen(&device, &queue, &build_frame(), w, h));

    assert!(
        occluded < baseline * 0.75,
        "AO lightmap should darken the object: baseline {baseline:.1}, occluded {occluded:.1}"
    );
}

/// The `set_lightmap` / `clear_lightmap` lifecycle: setting, clearing, and
/// re-setting all succeed on a live mesh, and a removed mesh reports a stale
/// handle rather than panicking.
#[test]
fn lightmap_set_clear_lifecycle() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let white = vec![[255u8, 255, 255, 255]; 16].concat();
    let radiance = renderer
        .resources_mut()
        .upload_texture(&device, &queue, 4, 4, &white)
        .expect("upload texture");
    let data = viewport_lib::resources::LightmapData::NonDirectional { radiance };

    let res = renderer.resources_mut();
    assert!(
        res.set_lightmap(
            &device,
            mesh_id,
            &[],
            data,
            viewport_lib::resources::LightmapMode::Add
        )
        .is_ok()
    );
    assert!(res.clear_lightmap(mesh_id).is_ok());
    // clear_lightmap on a mesh with none set is a no-op, not an error.
    assert!(res.clear_lightmap(mesh_id).is_ok());
    assert!(
        res.set_lightmap(
            &device,
            mesh_id,
            &[],
            data,
            viewport_lib::resources::LightmapMode::Replace
        )
        .is_ok()
    );

    // Removing the mesh makes further calls report a stale handle.
    renderer.resources_mut().free_mesh(mesh_id);
    let err = renderer.resources_mut().set_lightmap(
        &device,
        mesh_id,
        &[],
        data,
        viewport_lib::resources::LightmapMode::Replace,
    );
    assert!(matches!(err, Err(ViewportError::StaleHandle { .. })));
}

/// A big quad in the XY plane split into two submesh ranges (one triangle
/// each). Range materials are assigned per item, so the same upload can be
/// drawn single-material or per-range.
fn two_range_quad(resources: &mut viewport_lib::DeviceResources, device: &wgpu::Device) -> MeshId {
    let mut quad = MeshData::default();
    quad.positions = vec![
        [-5.0, -5.0, 0.0],
        [5.0, -5.0, 0.0],
        [5.0, 5.0, 0.0],
        [-5.0, 5.0, 0.0],
    ];
    quad.normals = vec![[0.0, 0.0, 1.0]; 4];
    quad.indices = vec![0, 1, 2, 0, 2, 3];
    quad.submeshes = vec![
        viewport_lib::SubmeshRange {
            first_index: 0,
            index_count: 3,
        },
        viewport_lib::SubmeshRange {
            first_index: 3,
            index_count: 3,
        },
    ];
    resources.upload_mesh_data(device, &quad).unwrap()
}

fn submesh_frame(mesh_id: MeshId, materials: Option<Vec<Material>>) -> FrameData {
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
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    // Unlit so pixels carry the raw material base colour.
    item.settings.unlit = true;
    item.material.base_colour = [0.0, 0.0, 1.0];
    item.submesh_materials = materials;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame
}

fn count_reddish(pixels: &[u8]) -> usize {
    pixels
        .chunks(4)
        .filter(|p| p[0] > 150 && p[1] < 100 && p[2] < 100)
        .count()
}

fn count_greenish(pixels: &[u8]) -> usize {
    pixels
        .chunks(4)
        .filter(|p| p[1] > 150 && p[0] < 100 && p[2] < 100)
        .count()
}

/// Two ranges with two materials must produce two visibly different regions:
/// one draw per range, each with its own object bind group. Exercises the LDR
/// per-object path.
#[test]
fn submesh_materials_draw_per_range_colours_ldr() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = two_range_quad(renderer.resources_mut(), &device);

    let mut red = Material::default();
    red.base_colour = [1.0, 0.0, 0.0];
    let mut green = Material::default();
    green.base_colour = [0.0, 1.0, 0.0];
    let frame = submesh_frame(mesh_id, Some(vec![red, green]));
    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    assert!(count_reddish(&pixels) > 0, "range 0 (red) did not draw");
    assert!(count_greenish(&pixels) > 0, "range 1 (green) did not draw");
}

/// Without `submesh_materials` (or with a count mismatch) the whole mesh
/// draws with the item material: no per-range colours may appear.
#[test]
fn submesh_materials_fall_back_to_item_material() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = two_range_quad(renderer.resources_mut(), &device);

    // None: single-material draw.
    let frame = submesh_frame(mesh_id, None);
    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(count_reddish(&pixels), 0);
    assert_eq!(count_greenish(&pixels), 0);

    // Length mismatch (3 materials, 2 ranges): falls back, item material only.
    let mut red = Material::default();
    red.base_colour = [1.0, 0.0, 0.0];
    let frame = submesh_frame(
        mesh_id,
        Some(vec![red, Material::default(), Material::default()]),
    );
    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(
        count_reddish(&pixels),
        0,
        "mismatched submesh_materials must fall back to the item material"
    );
}

/// On the HDR path a mixed item splits across passes: opaque ranges draw in
/// the scene pass, blend ranges in OIT. Both must be visible.
#[test]
fn submesh_materials_split_across_hdr_and_oit() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = two_range_quad(renderer.resources_mut(), &device);

    let mut red = Material::default();
    red.base_colour = [1.0, 0.0, 0.0];
    let mut green = Material::default();
    green.base_colour = [0.0, 1.0, 0.0];
    green.alpha_mode = AlphaMode::Blend;
    let mut frame = submesh_frame(mesh_id, Some(vec![red, green]));
    frame.effects.post_process.enabled = true;

    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert!(
        count_reddish(&pixels) > 0,
        "opaque range (red) did not draw in the HDR scene pass"
    );
    assert!(
        count_greenish(&pixels) > 0,
        "blend range (green) did not draw in the OIT pass"
    );
}

/// The `Scene` path must carry per-submesh materials end to end: set on the
/// node via `set_submesh_materials`, populated by `collect_render_items`,
/// and drawn one range per material.
#[test]
fn scene_submesh_materials_reach_render_items_and_draw() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = two_range_quad(renderer.resources_mut(), &device);

    let mut red = Material::default();
    red.base_colour = [1.0, 0.0, 0.0];
    let mut green = Material::default();
    green.base_colour = [0.0, 1.0, 0.0];

    let mut scene = Scene::new();
    let mut base = Material::default();
    base.base_colour = [0.0, 0.0, 1.0];
    let node = scene.add(Some(mesh_id), glam::Mat4::IDENTITY, base);
    let mut settings = ItemSettings::default();
    settings.unlit = true;
    scene.set_appearance(node, settings);
    scene.set_submesh_materials(node, Some(vec![red, green]));
    assert!(scene.node(node).unwrap().submesh_materials().is_some());

    let items = scene.collect_render_items(&Selection::new());
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0].submesh_materials.as_ref().map(|m| m.len()),
        Some(2),
        "collect_render_items must carry the node's submesh materials"
    );

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
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    let pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert!(count_reddish(&pixels) > 0, "range 0 (red) did not draw");
    assert!(count_greenish(&pixels) > 0, "range 1 (green) did not draw");

    // Clearing restores the single-material draw.
    scene.set_submesh_materials(node, None);
    let items = scene.collect_render_items(&Selection::new());
    assert!(items[0].submesh_materials.is_none());
}
