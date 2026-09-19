//! Material plugin shading hooks and reference-plugin registration.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

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
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;

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

    // Drawing through the plugin lazily built its full pipeline set: 4 LDR +
    // 4 HDR opaque (discarding + discard-free, each facedness) + 1 HDR
    // transparent + 2 OIT accumulate.
    let stats = renderer.resources().material_plugin_stats();
    assert_eq!(stats[0].pipelines_built, 11);

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
    assert_eq!(stats[0].pipelines_built, 11);
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
    frame.effects.display.mode = viewport_lib::PipelineMode::Hdr;
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
        .upload_texture(
            &device,
            &queue,
            viewport_lib::TextureData::srgb(4, 4, red.to_vec()),
        )
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
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;

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
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;

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
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;

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

// The reference plugins the examples ship must stay registrable: their WGSL runs
// through the full composer + wgpu validation at registration, so this catches
// contract or prefixer regressions (e.g. a body local named like a
// ShadingSurface field). The sources live with the examples that demonstrate
// them, so this test reaches across to them rather than keeping a second copy
// that could drift.
#[path = "../../viewport-lib-examples/eframe/examples/plugins/surface_detail_plugin.rs"]
mod surface_detail_plugin;
#[path = "../../viewport-lib-examples/eframe/examples/plugins/toon_plugin.rs"]
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

/// A plugin material on several instances of one mesh must draw through the
/// instanced plugin path on the LDR (`Direct`) frame, not fall back to one
/// draw per object. Proves the LDR `emit_draw_calls` plugin sub-loop: the
/// batch stats show the plugin items instanced (`per_object_items == 0`), the
/// render completes without a validation error (wgpu's uncaptured-error
/// handler panics on failure), and the plugin shading changes the output.
#[test]
fn material_plugin_instances_on_ldr_path() {
    struct ToonPlugin;
    impl viewport_lib::MaterialPlugin for ToonPlugin {
        fn name(&self) -> &'static str {
            "toon_ldr_instanced"
        }
        fn wgsl_body(&self) -> String {
            "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    let ndl = max(dot(surf.normal, light.l), 0.0);
    let stepped = ceil(ndl * 3.0) / 3.0;
    return surf.base_colour * stepped * light.radiance * light.shadow;
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return surf.base_colour * 0.2 * surf.ao;
}
"
            .to_string()
        }
    }

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // The low-power test device never enables bindless, so plugin items take
    // the per-batch instanced path (bindless would route them per-object).
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let plugin_id = renderer
        .resources_mut()
        .register_material_plugin(&device, &ToonPlugin)
        .expect("register material plugin");

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
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;

    // Four instances of the one mesh at the origin: they overlap into one
    // visible box but still form a four-instance batch, which is all the
    // routing assertion needs. Matches the single-item test's visible framing.
    let make = |plugin: Option<viewport_lib::MaterialPluginId>| {
        let mut it = SceneRenderItem::default();
        it.mesh_id = mesh_id;
        it.material.shading_plugin = plugin;
        it
    };

    // Built-in shading reference.
    let builtin_items: Vec<_> = (0..4).map(|_| make(None)).collect();
    frame.scene.surfaces = SurfaceSubmission::Flat(builtin_items.into());
    let builtin = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    // Same four instances, now selecting the plugin. Bump the scene generation
    // so the instanced batch cache rebuilds (a material change is a scene
    // change; the cache keys on the generation for exactly this).
    let plugin_items: Vec<_> = (0..4).map(|_| make(Some(plugin_id))).collect();
    frame.scene.surfaces = SurfaceSubmission::Flat(plugin_items.into());
    frame.scene.generation += 1;
    let toon = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    let stats = renderer.last_frame_stats();
    assert_eq!(
        stats.per_object_items, 0,
        "plugin instances must not fall back to the per-object path on LDR"
    );
    assert!(
        stats.instanced_batches >= 1,
        "the four plugin instances must form at least one instanced batch"
    );
    assert_ne!(
        builtin, toon,
        "the plugin shading must change the instanced LDR output"
    );
}

/// A plugin batch rides GPU culling: with culling on, plugin instances draw
/// through the plugin's `vs_main_cull` pipeline from the cull-written indirect
/// args, and for an all-visible scene that renders pixel-identically to the same
/// plugin drawn unculled. Completing both renders is also the validation
/// assertion (the culled draw binds the cull group-1 layout and the indirect
/// args, so a layout or offset mistake panics the uncaptured-error handler).
/// Exercises the per-batch (non-bindless) cull plugin path on the HDR scene pass.
#[test]
fn material_plugin_batch_rides_gpu_culling() {
    struct ToonPlugin;
    impl viewport_lib::MaterialPlugin for ToonPlugin {
        fn name(&self) -> &'static str {
            "toon_cull_instanced"
        }
        fn wgsl_body(&self) -> String {
            "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    let ndl = max(dot(surf.normal, light.l), 0.0);
    return surf.base_colour * ndl * light.radiance * light.shadow;
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return surf.base_colour * 0.2 * surf.ao;
}
"
            .to_string()
        }
    }

    // Needs INDIRECT_FIRST_INSTANCE for the GPU-culled indirect draw; skip where
    // the adapter lacks it. Per-batch textures (not bindless), so this covers the
    // per-batch cull plugin pipeline + `instance_cull_bind_groups` bind path.
    let Some((device, queue)) = headless_device_with_indirect() else {
        eprintln!("skipping: no adapter with INDIRECT_FIRST_INSTANCE");
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
    // The cull plugin path is on the HDR scene pass.
    frame.effects.display.mode = viewport_lib::PipelineMode::Hdr;

    let items: Vec<_> = (0..4)
        .map(|_| {
            let mut it = SceneRenderItem::default();
            it.mesh_id = mesh_id;
            it.material.shading_plugin = Some(plugin_id);
            it
        })
        .collect();
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    // Culled: plugin batches draw from the cull-written indirect args.
    renderer.enable_gpu_driven_culling();
    let culled = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let culled_stats = renderer.last_frame_stats();

    // Unculled reference: same instances drawn directly.
    renderer.disable_gpu_driven_culling();
    let unculled = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    assert_eq!(
        culled_stats.per_object_items, 0,
        "plugin instances must instance, not fall to the per-object path"
    );
    // Guard against a false pass where nothing drew (two blank frames also
    // compare equal): the box must actually be visible.
    let bg = &culled[0..4];
    assert!(
        culled.chunks_exact(4).any(|p| p != bg),
        "the plugin box must be visible, not an empty frame"
    );
    // The whole box is in frustum, so culling removes nothing: the culled plugin
    // path must render the same image as the direct one.
    assert_eq!(
        culled, unculled,
        "the GPU-culled plugin draw must match the unculled draw for an all-visible scene"
    );
}

/// Instanced plugin shading matches per-object plugin shading pixel-for-pixel.
/// The plugin's body is composed onto two different base shaders: `mesh.wgsl`
/// (per-object, reads `ObjectUniform`) and `mesh_instanced.wgsl` (reads
/// `InstanceData`); this is the proof they agree. The instancing threshold is
/// `visible_count > 1`, so one sphere draws per-object and two spheres at the
/// same transform draw instanced (overlapping into the same pixels). Same plugin,
/// same look -> the two renders must be identical. Closes the instanced-vs-
/// per-object plugin parity gate on the M4 (no desktop needed: both paths run
/// here; only the count-multi-draw submission is Metal-dormant, and that draws
/// the same pixels as the per-batch indirect path it collapses).
#[test]
fn material_plugin_instanced_matches_per_object() {
    struct ToonPlugin;
    impl viewport_lib::MaterialPlugin for ToonPlugin {
        fn name(&self) -> &'static str {
            "toon_parity"
        }
        fn wgsl_body(&self) -> String {
            "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    let ndl = max(dot(surf.normal, light.l), 0.0);
    let stepped = ceil(ndl * 4.0) / 4.0;
    return surf.base_colour * stepped * light.radiance * light.shadow;
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return surf.base_colour * 0.2 * surf.ao;
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
        .upload_mesh_data(&device, &viewport_lib::primitives::sphere(1.0, 32, 16))
        .unwrap();
    let plugin_id = renderer
        .resources_mut()
        .register_material_plugin(&device, &ToonPlugin)
        .expect("register material plugin");

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

    let sphere = || {
        let mut it = SceneRenderItem::default();
        it.mesh_id = mesh_id;
        it.material = Material::pbr([0.75, 0.3, 0.3], 0.1, 0.55);
        it.material.shading_plugin = Some(plugin_id);
        it
    };

    // One sphere: visible_count == 1, so it draws through the per-object path.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![sphere()].into());
    let per_object = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let po_stats = renderer.last_frame_stats();

    // Two spheres at the same transform: visible_count > 1, so they draw
    // instanced; overlapping exactly, they cover the same pixels as the one
    // per-object sphere (the second instance fails the depth test behind the
    // first). Bump the generation so the batch list rebuilds for the new scene.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![sphere(), sphere()].into());
    frame.scene.generation += 1;
    let instanced = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let inst_stats = renderer.last_frame_stats();

    // Confirm the A/B actually took the two different paths.
    assert_eq!(
        po_stats.instanced_batches, 0,
        "one sphere should draw per-object (below the instancing threshold)"
    );
    assert!(
        inst_stats.instanced_batches >= 1,
        "two spheres should form an instanced batch"
    );
    // The sphere must be visible, not an empty frame.
    let bg = &per_object[0..4];
    assert!(
        per_object.chunks_exact(4).any(|p| p != bg),
        "the plugin sphere must be visible"
    );
    assert_eq!(
        per_object, instanced,
        "instanced plugin shading must match per-object plugin shading pixel-for-pixel"
    );
}

/// A plugin that reads the per-vertex extension attribute (`surf.attr`) must
/// render the same whatever the item count, i.e. it must stay on the per-object
/// path rather than instance. The instanced mesh path has no per-vertex
/// extension-attribute binding (its `surf.attr` is the per-instance custom-data
/// channel), so instancing such a plugin would feed it zero instead of the vertex
/// data. Two overlapping items cross the instancing threshold, so without the
/// per-object guard they would instance and lose the attribute; with it they stay
/// per-object and match the single-item render.
#[test]
fn material_plugin_reading_vertex_attribute_stays_per_object() {
    struct AttrPlugin;
    impl viewport_lib::MaterialPlugin for AttrPlugin {
        fn name(&self) -> &'static str {
            "attr_per_object"
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
    // A uniform green per-vertex attribute: the plugin adds it to the lit colour,
    // so a green bias in the image proves `surf.attr` reached the hook.
    let mut data = box_mesh();
    data.extension_attributes = Some(vec![[0.0, 0.6, 0.0, 0.0]; data.positions.len()]);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &data)
        .unwrap();
    let plugin_id = renderer
        .resources_mut()
        .register_material_plugin(&device, &AttrPlugin)
        .expect("register material plugin");

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
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;

    let item = || {
        let mut it = SceneRenderItem::default();
        it.mesh_id = mesh_id;
        it.material.shading_plugin = Some(plugin_id);
        it
    };

    // One item: per-object.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item()].into());
    let one = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    // Two overlapping items: crosses the instancing threshold. The guard keeps a
    // vertex-attribute plugin per-object anyway.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item(), item()].into());
    frame.scene.generation += 1;
    let two = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let two_stats = renderer.last_frame_stats();

    assert_eq!(
        two_stats.instanced_batches, 0,
        "a vertex-attribute plugin must not instance (it would lose surf.attr)"
    );
    // The attribute reached the hook: the box is tinted green (G > R on the box).
    let bias = |img: &[u8]| -> i64 { img.chunks_exact(4).map(|p| p[1] as i64 - p[0] as i64).sum() };
    assert!(
        bias(&two) > 0,
        "the per-vertex attribute should tint the box green via surf.attr"
    );
    assert_eq!(
        one, two,
        "a vertex-attribute plugin must render the same at one or two items"
    );
}
