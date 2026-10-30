//! Smoke tests for the family-D fixtures: WGSL bodies spliced into the core
//! shaders.
//!
//! Both seams need the renderer's recommended device limits (a deformer needs
//! three bind groups plus the storage-buffer headroom, a material plugin
//! needs four), which the harness default profile does not request, so these
//! tests build their harness on a recommended-limits device and skip when no
//! adapter offers one.

use viewport_lib::Colour;
use viewport_lib::wgpu;
use viewport_lib::{Material, SceneRenderItem, SurfaceSubmission};
use viewport_lib_testkit::fixtures::{
    ConstantOffsetDeformer, FlatColourMaterialPlugin, constant_offset_deformer, probe_frame,
    probe_quad,
};
use viewport_lib_testkit::{DeviceProfile, Harness};

const SIZE: u32 = 64;

fn recommended_harness(label: &'static str) -> Option<Harness> {
    Harness::with_profile(&DeviceProfile::low_power(label))
}

/// Sum of the RGB bytes at a pixel.
fn luma(pixels: &[u8], x: u32, y: u32) -> i32 {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
}

/// The registered deformer moves the mesh: with a zero offset the quad covers
/// the centre of the frame, and with a large `+X` offset that pixel reads
/// background instead.
///
/// This exercises registration, the shader composition and validation, the
/// per-slot flag gate (the mesh only moves once slot data is attached), and
/// the slot params window, without caring what the maths is.
#[test]
fn deformer_fixture_moves_the_mesh() {
    let Some(mut harness) = recommended_harness("fixture-deformer") else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };

    let mesh = probe_quad();
    let vertex_count = mesh.positions.len();
    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &mesh)
        .expect("upload probe quad");

    let deformer = match harness
        .renderer
        .resources_mut()
        .register_deformer(&harness.device, constant_offset_deformer())
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a deformer: {err}");
            return;
        }
    };
    let slot = deformer.slot();
    ConstantOffsetDeformer::attach(
        harness.renderer.resources_mut(),
        &harness.device,
        mesh_id,
        slot,
        vertex_count,
    );

    let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = harness
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.9, 0.9, 0.9)))
        .expect("material upload");
    item.settings.unlit = true;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    // Offset zero: the quad is where the mesh puts it.
    ConstantOffsetDeformer::set_offset(
        harness.renderer.resources_mut(),
        &harness.queue,
        slot,
        glam::Vec3::ZERO,
    );
    let still = harness.render(&frame, SIZE, SIZE);
    let centre_still = luma(&still, SIZE / 2, SIZE / 2);
    assert!(
        centre_still > 300,
        "the undeformed quad must cover the centre pixel (read {centre_still})"
    );

    // A large offset along +X pushes the quad off the centre pixel.
    ConstantOffsetDeformer::set_offset(
        harness.renderer.resources_mut(),
        &harness.queue,
        slot,
        glam::Vec3::new(4.0, 0.0, 0.0),
    );
    let moved = harness.render(&frame, SIZE, SIZE);
    let centre_moved = luma(&moved, SIZE / 2, SIZE / 2);
    assert!(
        centre_moved < centre_still - 200,
        "the deformer must move the quad off the centre pixel: still {centre_still}, moved {centre_moved}"
    );
}

/// The registered material plugin shades the draws that select it: the hooked
/// quad reads as the plugin's flat colour, which differs from the stock lit
/// result for the same material.
#[test]
fn material_plugin_fixture_changes_shading() {
    let Some(mut harness) = recommended_harness("fixture-material") else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };

    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");

    let plugin = FlatColourMaterialPlugin::new("fixture_flat_colour", [0.0, 0.9, 0.0]);
    let plugin_id = match harness
        .renderer
        .resources_mut()
        .register_material_plugin(&harness.device, &plugin)
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a material plugin: {err}");
            return;
        }
    };

    let render = |harness: &mut Harness, shading: Option<viewport_lib::MaterialPluginId>| {
        let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model = glam::Mat4::from_scale(glam::Vec3::splat(1.5)).to_cols_array_2d();
        let mut red = Material::from_colour(Colour::linear_rgb(0.8, 0.1, 0.1));
        red.shading_plugin = shading;
        item.material = harness
            .renderer
            .resources_mut()
            .upload_material(red)
            .expect("material upload");
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        harness.render(&frame, SIZE, SIZE)
    };

    let stock = render(&mut harness, None);
    let hooked = render(&mut harness, Some(plugin_id));

    let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    assert_ne!(
        &stock[i..i + 3],
        &hooked[i..i + 3],
        "the hooked draw must shade differently from the stock path"
    );
    // The fixture shades flat green, so green must dominate where the stock
    // red material did not.
    assert!(
        hooked[i + 1] > hooked[i] + 40,
        "the hooked draw must read as the plugin's flat colour: got {:?}",
        &hooked[i..i + 4]
    );
}

/// The surface hook's emissive (`k`) and alpha.
const GLOW: f32 = 0.4;
const ALPHA: f32 = 0.5;
/// The grey backdrop the alpha check blends over, linear.
const BACKDROP: f32 = 0.5;
/// Allowed error in linear units: an 8-bit sRGB step near these values is
/// about 0.002, so this leaves room for the OIT accumulate and resolve and
/// for driver rounding, and still fails a missing or doubled term (the
/// smallest such error, alpha at 1.0 instead of 0.5, is 0.05).
const LINEAR_TOLERANCE: f32 = 0.015;

/// sRGB-encoded byte to linear.
fn srgb_to_linear(byte: u8) -> f32 {
    let c = byte as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The Khronos PBR Neutral operator for a grey value, the default tone map
/// the HDR path applies at the passthrough exposure (a multiplier of 1).
fn khronos_neutral_grey(x: f32) -> f32 {
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    let x = x - offset;
    let start = 0.8 - 0.04;
    if x < start {
        return x;
    }
    let d = 1.0 - start;
    let peak = 1.0 - d * d / (x + d - start);
    // A grey stays grey, so the desaturation step changes nothing.
    peak
}

/// What a linear scene value reads as on `mode`'s output.
fn displayed(mode: viewport_lib::PipelineMode, linear: f32) -> f32 {
    match mode {
        viewport_lib::PipelineMode::Hdr => khronos_neutral_grey(linear),
        _ => linear,
    }
}

/// Linear RGB at a pixel of a `SIZE` x `SIZE` sRGB readback.
fn linear_at(pixels: &[u8], (x, y): (u32, u32)) -> [f32; 3] {
    let i = ((y * SIZE + x) * 4) as usize;
    [
        srgb_to_linear(pixels[i]),
        srgb_to_linear(pixels[i + 1]),
        srgb_to_linear(pixels[i + 2]),
    ]
}

fn assert_grey_near(mode: viewport_lib::PipelineMode, what: &str, got: [f32; 3], want: f32) {
    for c in got {
        assert!(
            (c - want).abs() <= LINEAR_TOLERANCE,
            "{mode:?}: {what}: expected linear {want:.3} +- {LINEAR_TOLERANCE} on every channel, got {got:?}"
        );
    }
}

/// A plugin's `shade_surface` emissive and alpha reach the pixel, and do so
/// the same whether its draw is a batch of one or of three, on both colour
/// paths (HDR, where blended draws go through the OIT shader, and Direct).
#[test]
fn material_plugin_surface_emissive_and_alpha_compose() {
    for mode in [
        viewport_lib::PipelineMode::Hdr,
        viewport_lib::PipelineMode::Direct,
    ] {
        surface_emissive_and_alpha(mode);
    }
}

fn surface_emissive_and_alpha(mode: viewport_lib::PipelineMode) {
    use viewport_lib::{AlphaMode, Camera, RenderCamera};
    use viewport_lib_testkit::fixtures::SurfaceGlowMaterialPlugin;

    let Some(mut harness) = recommended_harness("fixture-material-surface") else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let plugin = SurfaceGlowMaterialPlugin::new("fixture_surface_glow", GLOW, ALPHA);
    let plugin_id = match harness
        .renderer
        .resources_mut()
        .register_material_plugin(&harness.device, &plugin)
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a material plugin: {err}");
            return;
        }
    };
    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");
    // One material shared by every item, so the three items can batch
    // together. The base colour is irrelevant: the hook replaces it with black.
    let upload = |harness: &mut Harness, alpha_mode: AlphaMode| {
        let mut material = Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8));
        material.alpha_mode = alpha_mode;
        material.shading_plugin = Some(plugin_id);
        harness
            .renderer
            .resources_mut()
            .upload_material(material)
            .expect("material upload")
    };
    let blended = upload(&mut harness, AlphaMode::Blend);
    let opaque = upload(&mut harness, AlphaMode::Opaque);

    // Straight down `-Z` onto the quads, which face `+Z`.
    let mut camera = Camera::default();
    camera.orientation = glam::Quat::IDENTITY;
    camera.aspect = 1.0;
    let render_camera = RenderCamera::from_camera(&camera);
    let view_proj = render_camera.view_proj();
    let pixel_of = |world: glam::Vec3| {
        let ndc = view_proj.project_point3(world);
        (
            ((ndc.x * 0.5 + 0.5) * SIZE as f32) as u32,
            ((0.5 - ndc.y * 0.5) * SIZE as f32) as u32,
        )
    };
    let centres = [-1.3f32, 0.0, 1.3].map(|x| glam::Vec3::new(x, 0.0, 0.0));
    let probes = centres.map(pixel_of);

    let mut generation = 0u64;
    let mut render = |harness: &mut Harness,
                      material: viewport_lib::MaterialId,
                      xs: &[glam::Vec3],
                      background: f32| {
        let mut frame = probe_frame(SIZE, [background, background, background, 1.0]);
        frame.camera.render_camera = render_camera.clone();
        frame.effects.display.mode = mode;
        let items: Vec<SceneRenderItem> = xs
            .iter()
            .map(|&centre| {
                let mut item = SceneRenderItem::default();
                item.mesh_id = mesh_id;
                item.material = material;
                item.model = glam::Mat4::from_translation(centre).to_cols_array_2d();
                item
            })
            .collect();
        frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
        generation += 1;
        frame.scene.generation = generation;
        let pixels = harness.render(&frame, SIZE, SIZE);
        (pixels, harness.stats().instanced_batches)
    };

    // HDR resolves blended draws through OIT and batches them like opaque
    // ones; Direct blends in draw order, so it draws each blended item as its
    // own batch to sort them. Opaque items batch on both paths.
    let blended_batches = match mode {
        viewport_lib::PipelineMode::Hdr => 1,
        _ => 3,
    };
    let (batch_black, batches) = render(&mut harness, blended, &centres, 0.0);
    assert_eq!(
        batches, blended_batches,
        "test premise ({mode:?}): three blended items sharing a mesh and material"
    );
    let (batch_backdrop, _) = render(&mut harness, blended, &centres, BACKDROP);
    let (single_black, _) = render(&mut harness, blended, &centres[1..2], 0.0);
    let (single_backdrop, _) = render(&mut harness, blended, &centres[1..2], BACKDROP);
    let (opaque_batch, batches) = render(&mut harness, opaque, &centres, BACKDROP);
    assert_eq!(
        batches, 1,
        "test premise ({mode:?}): three opaque items sharing a mesh and material draw as one batch"
    );
    let (opaque_single, _) = render(&mut harness, opaque, &centres[1..2], BACKDROP);

    // The background itself, between the quads, reads as the backdrop on
    // both paths: the HDR tone map leaves pixels no geometry covers alone,
    // so only the surfaces below go through `displayed`.
    let gap = pixel_of(glam::Vec3::new(0.65, 0.0, 0.0));
    assert_grey_near(
        mode,
        "the backdrop between the quads",
        linear_at(&batch_backdrop, gap),
        BACKDROP,
    );

    for (i, &probe) in probes.iter().enumerate() {
        // Lighting is zeroed and the base colour is black, so over black the
        // surface adds only its emissive `k`, weighted by its alpha.
        assert_grey_near(
            mode,
            &format!("blended item {i} over black"),
            linear_at(&batch_black, probe),
            displayed(mode, ALPHA * GLOW),
        );
        // Over the grey backdrop, `1 - alpha` of the backdrop shows through
        // on top of the same weighted emissive.
        assert_grey_near(
            mode,
            &format!("blended item {i} over the backdrop"),
            linear_at(&batch_backdrop, probe),
            displayed(mode, ALPHA * GLOW + (1.0 - ALPHA) * BACKDROP),
        );
        // Opaque ignores the hook's alpha: the pixel is the full emissive `k`
        // and none of the backdrop.
        assert_grey_near(
            mode,
            &format!("opaque item {i} over the backdrop"),
            linear_at(&opaque_batch, probe),
            displayed(mode, GLOW),
        );
    }

    // A batch of one shades the centre item as the batch of three does. The
    // other two items do not overlap it, so the pixel is the same draw's.
    let centre = probes[1];
    let i = ((centre.1 * SIZE + centre.0) * 4) as usize;
    for (what, single, batch) in [
        ("blended over black", &single_black, &batch_black),
        (
            "blended over the backdrop",
            &single_backdrop,
            &batch_backdrop,
        ),
        ("opaque", &opaque_single, &opaque_batch),
    ] {
        let (a, b) = (&single[i..i + 3], &batch[i..i + 3]);
        assert!(
            a.iter().zip(b).all(|(x, y)| x.abs_diff(*y) <= 1),
            "{mode:?}: one {what} item must read as the same item among three \
             (within one 8-bit step): single {a:?}, three {b:?}"
        );
    }
}

/// Register `plugin` on a recommended-limits harness and upload the probe
/// quad, or `None` (after saying why) when the device cannot.
fn plugin_harness(
    label: &'static str,
    plugin: &dyn viewport_lib::MaterialPlugin,
) -> Option<(
    Harness,
    viewport_lib::MaterialPluginId,
    viewport_lib::MeshId,
)> {
    let Some(mut harness) = recommended_harness(label) else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return None;
    };
    let plugin_id = match harness
        .renderer
        .resources_mut()
        .register_material_plugin(&harness.device, plugin)
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skipping: this device cannot register a material plugin: {err}");
            return None;
        }
    };
    let mesh_id = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");
    Some((harness, plugin_id, mesh_id))
}

/// A probe frame drawing the quad, scaled to fill the centre, with `material`.
fn quad_frame(
    harness: &mut Harness,
    mesh_id: viewport_lib::MeshId,
    material: Material,
) -> viewport_lib::FrameData {
    let mut frame = probe_frame(SIZE, [0.0, 0.0, 0.0, 1.0]);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(1.5)).to_cols_array_2d();
    item.material = harness
        .renderer
        .resources_mut()
        .upload_material(material)
        .expect("material upload");
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame
}

/// A hook reading `camera.time`, `camera.viewport` and `surf.vertex_colour`
/// composes, and each reaches the pixel: red follows `FrameData::time`, green
/// the physical viewport width, blue the quad's (absent, so white) vertex
/// colour.
#[test]
fn material_plugin_reads_frame_time_and_viewport() {
    use viewport_lib::FrameTime;
    use viewport_lib_testkit::fixtures::FrameInputsMaterialPlugin;

    let plugin = FrameInputsMaterialPlugin::new("fixture_frame_inputs");
    let Some((mut harness, plugin_id, mesh_id)) = plugin_harness("fixture-frame-inputs", &plugin)
    else {
        return;
    };
    let mut material = Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8));
    material.shading_plugin = Some(plugin_id);
    let mut frame = quad_frame(&mut harness, mesh_id, material);
    frame.effects.display.mode = viewport_lib::PipelineMode::Direct;

    let centre = (SIZE / 2, SIZE / 2);
    for seconds in [0.25, 0.5] {
        frame.time = Some(FrameTime::fixed(seconds));
        let pixels = harness.render(&frame, SIZE, SIZE);
        let got = linear_at(&pixels, centre);
        let want = [seconds as f32, SIZE as f32 / 256.0, 0.5];
        for (c, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= LINEAR_TOLERANCE,
                "channel {c} at t = {seconds}: expected linear {w:.3}, got {got:?}"
            );
        }
    }
}

/// Two renders of one frame at the same `FrameTime` are byte-identical, even
/// with a hook that reads the time; a different time moves the pixel, so the
/// time does reach the shader. An offscreen frame without a time renders at
/// the fixed zero time.
#[test]
fn frame_time_renders_are_byte_identical() {
    use viewport_lib::FrameTime;
    use viewport_lib_testkit::fixtures::FrameInputsMaterialPlugin;

    let plugin = FrameInputsMaterialPlugin::new("fixture_frame_inputs_determinism");
    let Some((mut harness, plugin_id, mesh_id)) =
        plugin_harness("fixture-frame-time-determinism", &plugin)
    else {
        return;
    };
    let mut material = Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8));
    material.shading_plugin = Some(plugin_id);
    let mut frame = quad_frame(&mut harness, mesh_id, material);
    frame.time = Some(FrameTime::new(0.375, 0.016, 42));

    let first = harness.render(&frame, SIZE, SIZE);
    // A frame through the presented path between the two, at another time,
    // must not leak into the second render.
    frame.time = Some(FrameTime::new(0.75, 0.016, 43));
    let other = harness.render(&frame, SIZE, SIZE);
    frame.time = Some(FrameTime::new(0.375, 0.016, 42));
    let second = harness.render(&frame, SIZE, SIZE);

    assert!(
        first == second,
        "same FrameTime must render byte-identical frames"
    );
    assert!(
        first != other,
        "a different FrameTime must change the hooked pixel"
    );

    // An offscreen render of a frame without a time uses the fixed zero time,
    // not the renderer's clock.
    frame.time = None;
    let unset = harness.render(&frame, SIZE, SIZE);
    frame.time = Some(FrameTime::fixed(0.0));
    let fixed = harness.render(&frame, SIZE, SIZE);
    assert!(
        unset == fixed,
        "render_offscreen must render a frame without a time at FrameTime::fixed(0.0)"
    );
}

/// `SURFACE_OVERRIDE_AO` with `ao = 0` takes the ambient term to black; the
/// same `ao = 0` without the flag leaves the draw as the stock PBR path
/// renders it.
#[test]
fn material_plugin_ao_override_darkens_ambient() {
    use viewport_lib::ShadingModel;
    use viewport_lib_testkit::fixtures::AoOverrideMaterialPlugin;

    let flagged = AoOverrideMaterialPlugin::new("fixture_ao_flagged", true);
    let Some((mut harness, flagged_id, mesh_id)) = plugin_harness("fixture-ao-override", &flagged)
    else {
        return;
    };
    let unflagged_id = harness
        .renderer
        .resources_mut()
        .register_material_plugin(
            &harness.device,
            &AoOverrideMaterialPlugin::new("fixture_ao_unflagged", false),
        )
        .expect("register the second plugin");

    // Ambient only: no lights, the hemisphere at full strength.
    let render = |harness: &mut Harness, plugin: Option<viewport_lib::MaterialPluginId>| {
        let mut material = Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8));
        material.shading_model = ShadingModel::Pbr;
        material.shading_plugin = plugin;
        let mut frame = quad_frame(harness, mesh_id, material);
        frame.effects.lighting.lights.clear();
        frame.effects.lighting.hemisphere_intensity = 1.0;
        frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        harness.render(&frame, SIZE, SIZE)
    };
    let stock = render(&mut harness, None);
    let unflagged = render(&mut harness, Some(unflagged_id));
    let flagged = render(&mut harness, Some(flagged_id));

    let centre = (SIZE / 2, SIZE / 2);
    let stock_px = linear_at(&stock, centre);
    assert!(
        stock_px.iter().all(|&c| c > 0.05),
        "test premise: the stock ambient lights the quad, got {stock_px:?}"
    );
    let flagged_px = linear_at(&flagged, centre);
    assert!(
        flagged_px.iter().all(|&c| c <= 0.002),
        "SURFACE_OVERRIDE_AO with ao = 0 must take the ambient to black, got {flagged_px:?} (stock {stock_px:?})"
    );
    let max_diff = stock
        .iter()
        .zip(&unflagged)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(
        max_diff <= 1,
        "without the flag the override's ao must be ignored (max diff {max_diff} from stock)"
    );
}

// Keep `wgpu` named so this binary resolves the same leg as the crate.
const _: Option<wgpu::TextureFormat> = None;
