//! Tone-map compositing, bloom, transparency, and the foreground pass.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

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
        frame.effects.display.mode = viewport_lib::PipelineMode::Hdr;
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
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
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
        frame.effects.display.mode = if hdr {
            viewport_lib::PipelineMode::Hdr
        } else {
            viewport_lib::PipelineMode::Direct
        };
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
    frame.effects.display.mode = viewport_lib::PipelineMode::Hdr;
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
