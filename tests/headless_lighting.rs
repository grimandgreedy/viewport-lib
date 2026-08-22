//! Lightmaps, light probes, environment capture, and env zones.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

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
    let orig_pp_enabled = frame.effects.display.mode;

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
    assert_eq!(frame.effects.display.mode, orig_pp_enabled);
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

/// F1-gpu: the on-GPU cube -> equirect resolve must produce the same panorama
/// as the CPU resolve. Both render the same six faces of the same scene; the
/// GPU path keeps them in a texture array and resolves with a fragment pass,
/// the CPU path loops in Rust. Aggregate radiance must agree (f16 rounding and
/// hardware-vs-manual bilinear give small per-texel differences, so this checks
/// the summed relative error, not exact bytes), and the bright emissive lobe
/// must resolve to the same place.
#[test]
fn capture_equirect_gpu_matches_the_cpu_resolve() {
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
    item.model = glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.0, 0.0)).to_cols_array_2d();
    item.material.emissive = [8.0, 8.0, 8.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let eq_h = 64u32;
    let cpu = renderer.capture_equirect(&device, &queue, &mut frame, [0.0, 0.0, 0.0], 128, eq_h);
    let gpu_cap =
        renderer.capture_equirect_gpu(&device, &queue, &mut frame, [0.0, 0.0, 0.0], 128, eq_h);
    let gpu = renderer.read_captured_hdr(&device, &queue, &gpu_cap);

    assert_eq!((gpu.width, gpu.height), (cpu.width, cpu.height));
    assert_eq!((gpu.width, gpu.height), (eq_h * 2, eq_h));
    assert_eq!(gpu.rgba.len(), cpu.rgba.len());

    // Summed relative error over RGB. The bright emissive lobe dominates the
    // sum, so a mismatched resolve (wrong face pick, flipped projection) would
    // blow this up well past a few percent.
    let mut diff = 0.0f64;
    let mut mag = 0.0f64;
    let texels = (cpu.width * cpu.height) as usize;
    let (mut cpu_best, mut cpu_i) = (f32::NEG_INFINITY, 0usize);
    let (mut gpu_best, mut gpu_i) = (f32::NEG_INFINITY, 0usize);
    for i in 0..texels {
        let o = i * 4;
        let mut cl = 0.0f32;
        let mut gl = 0.0f32;
        for k in 0..3 {
            let c = cpu.rgba[o + k];
            let g = gpu.rgba[o + k];
            diff += (c - g).abs() as f64;
            mag += c.abs() as f64;
            cl += c;
            gl += g;
        }
        if cl > cpu_best {
            cpu_best = cl;
            cpu_i = i;
        }
        if gl > gpu_best {
            gpu_best = gl;
            gpu_i = i;
        }
    }
    let rel = diff / mag.max(1e-6);
    assert!(
        rel < 0.05,
        "GPU resolve diverged from the CPU resolve: relative error {rel:.4}"
    );

    // The brightest texel must land on the same equirect location (allow a
    // one-texel slop for the bilinear difference).
    let cx = (cpu_i as u32 % cpu.width) as i32;
    let cy = (cpu_i as u32 / cpu.width) as i32;
    let gx = (gpu_i as u32 % gpu.width) as i32;
    let gy = (gpu_i as u32 / gpu.width) as i32;
    assert!(
        (cx - gx).abs() <= 1 && (cy - gy).abs() <= 1,
        "brightest texel moved: cpu ({cx}, {cy}) vs gpu ({gx}, {gy})"
    );
    assert!(
        gpu_best > 1.0,
        "HDR emissive did not survive the GPU resolve"
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

/// APV: an object marked `IndirectLightSource::ProbeVolume` takes its indirect
/// diffuse from the uploaded volume, sampled per fragment. A volume bright on one
/// X side must light that side of a large object; flipping which side is bright
/// must move the bright region. A per-object (center-sampled) path would render
/// both the same, so the two images differing proves the lookup is per-fragment.
#[test]
fn probe_volume_lights_object_per_fragment() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // Two probes along X (dims 2x1x1), spanning the scaled box's extent.
    let red = {
        let mut sh = viewport_lib::resources::SHCoefficients::default();
        sh.r[0] = 1.0 / 0.282095; // DC red -> evaluate_sh returns ~[1,0,0]
        sh
    };
    let black = viewport_lib::resources::SHCoefficients::default();
    let volume = |bright_plus_x: bool| {
        let cells = if bright_plus_x {
            vec![black, red]
        } else {
            vec![red, black]
        };
        viewport_lib::resources::LightProbeVolume::new(
            [-1.5, -1.5, -1.5],
            [3.0, 3.0, 3.0],
            [2, 1, 1],
            cells,
        )
    };

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
    item.mesh_id = mesh_idx;
    // Scale the unit box up so it spans the volume's X gradient.
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(3.0)).to_cols_array_2d();
    item.material.shading_model = ShadingModel::Pbr;
    item.material.base_colour = [1.0, 1.0, 1.0];
    item.indirect_light = IndirectLightSource::ProbeVolume;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let (w, h) = (64u32, 64u32);
    let max_red = |px: &[u8]| px.chunks_exact(4).map(|p| p[0]).max().unwrap_or(0);

    renderer.set_light_probe_volume(&device, &volume(true));
    let a = renderer.render_offscreen(&device, &queue, &frame, w, h);
    renderer.set_light_probe_volume(&device, &volume(false));
    let b = renderer.render_offscreen(&device, &queue, &frame, w, h);

    // Each render must actually be lit red by the volume (end-to-end path works).
    assert!(
        max_red(&a) > 60 && max_red(&b) > 60,
        "volume-lit object should be red: max red a={}, b={}",
        max_red(&a),
        max_red(&b)
    );

    // Flipping the bright side must change the image: sum the per-pixel red
    // difference. A center-sampled (per-object) path would leave the two
    // identical.
    let red_diff: u64 = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .map(|(pa, pb)| (pa[0] as i32 - pb[0] as i32).unsigned_abs() as u64)
        .sum();
    assert!(
        red_diff > 2000,
        "per-fragment volume sampling should move the bright side; red diff was {red_diff}"
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
    frame.effects.environment = Some(viewport_lib::EnvironmentSettings {
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
