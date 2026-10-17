//! The per-object and instanced mesh paths render the same material the same
//! way.
//!
//! Each case draws one material twice: alone, which takes the per-object path,
//! and beside a second copy, which batches it onto the instanced path. The
//! centre pixel, covered only by the first copy, must match.

use viewport_lib::wgpu;
use viewport_lib::{
    AlphaMode, Colour, FrameStats, LightKind, LightSource, Material, MeshId, PipelineMode,
};

mod common;
use common::*;

/// A quad in the XY plane facing +Z, so a top-down camera sees its front.
fn quad_mesh(half: f32) -> viewport_lib::MeshData {
    let mut mesh = viewport_lib::MeshData::default();
    mesh.positions = vec![
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.uvs = Some(vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh
}

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: ViewportRenderer,
    mesh: MeshId,
    generation: u64,
}

impl Scene {
    fn new() -> Option<Self> {
        let (device, queue) = headless_device()?;
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let mesh = renderer
            .resources_mut()
            .upload_mesh_data(&device, &quad_mesh(1.0))
            .unwrap();
        Some(Scene {
            device,
            queue,
            renderer,
            mesh,
            generation: 0,
        })
    }

    /// Draw `material` at the centre, plus a second copy off to the side when
    /// `instanced`. Returns the centre pixel and the frame's stats.
    fn render(
        &mut self,
        material: Material,
        opacity: f32,
        instanced: bool,
    ) -> ([u8; 4], FrameStats) {
        const SIZE: u32 = 64;
        self.generation += 1;
        let mut frame = FrameData::default();
        frame.scene.generation = self.generation;
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
        frame.effects.display.mode = PipelineMode::Hdr;
        frame.effects.display.exposure = viewport_lib::ExposureSettings::manual(0.0);
        // A dim, fixed ambient and no direct light, so lit colours stay well
        // inside the displayable range and a factor-of-two error shows.
        frame.effects.lighting.lights = vec![];
        frame.effects.lighting.hemisphere_intensity = 0.5;
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

        let item_at = |x: f32| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = self.mesh;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(x, x, 0.0)).to_cols_array_2d();
            item.material = material.clone();
            item.settings.opacity = opacity;
            item
        };
        let mut items = vec![item_at(0.0)];
        if instanced {
            items.push(item_at(1.7));
        }
        frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

        let px = self
            .renderer
            .render_offscreen(&self.device, &self.queue, &frame, SIZE, SIZE);
        let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
        (
            [px[i], px[i + 1], px[i + 2], px[i + 3]],
            self.renderer.last_frame_stats(),
        )
    }

    /// Render `material` on both paths, check each took the path it was meant
    /// to, and return the two centre pixels (per-object, instanced).
    fn both_paths(&mut self, material: Material, opacity: f32) -> ([u8; 4], [u8; 4]) {
        let (alone, alone_stats) = self.render(material.clone(), opacity, false);
        assert!(
            alone_stats.instanced_batches == 0,
            "a single item should draw per-object"
        );
        let (batched, batched_stats) = self.render(material, opacity, true);
        assert!(
            batched_stats.instanced_batches > 0,
            "two copies of one material should batch onto the instanced path"
        );
        (alone, batched)
    }
}

fn assert_close(label: &str, a: [u8; 4], b: [u8; 4], tol: i32) {
    for c in 0..3 {
        assert!(
            (a[c] as i32 - b[c] as i32).abs() <= tol,
            "{label}: per-object {a:?} vs instanced {b:?}"
        );
    }
}

/// A lit `BlendPremultiplied` material composites the same on both
/// transparent paths: neither multiplies its colour by alpha a second time.
#[test]
fn premultiplied_blend_matches_on_both_paths() {
    let Some(mut scene) = Scene::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut material = Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8));
    material.alpha_mode = AlphaMode::BlendPremultiplied;
    let (alone, batched) = scene.both_paths(material, 0.5);
    assert!(alone[0] > 10, "the surface should render: {alone:?}");
    assert_close("BlendPremultiplied", alone, batched, 4);
}

/// A masked material faded below full opacity draws through the transparent
/// path; fragments under the cutoff are discarded there on both paths rather
/// than composited as a faint quad.
#[test]
fn transparent_alpha_mask_discards_on_both_paths() {
    let Some(mut scene) = Scene::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Albedo alpha 0.2: with opacity 0.8 the combined alpha (0.16) is under
    // the 0.5 cutoff, so the whole quad is cut out.
    let tex = scene
        .renderer
        .resources_mut()
        .upload_texture(
            &scene.device,
            &scene.queue,
            viewport_lib::TextureData::srgb(1, 1, vec![220, 220, 220, 51]),
        )
        .unwrap();
    let mut material = Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0));
    material.texture_id = Some(tex);
    material.alpha_mode = AlphaMode::Mask(0.5);
    let (alone, batched) = scene.both_paths(material, 0.8);
    assert!(
        alone[0] < 4,
        "per-object should cut the quad out: {alone:?}"
    );
    assert_close("transparent Mask", alone, batched, 2);
}

/// A blend material at full opacity still draws through the transparent pass
/// on both paths, so its albedo alpha fades it over the background.
#[test]
fn blend_at_full_opacity_blends_on_both_paths() {
    let Some(mut scene) = Scene::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let tex = scene
        .renderer
        .resources_mut()
        .upload_texture(
            &scene.device,
            &scene.queue,
            viewport_lib::TextureData::srgb(1, 1, vec![220, 220, 220, 51]),
        )
        .unwrap();
    let mut solid = Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0));
    solid.texture_id = Some(tex);
    let (opaque, _) = scene.both_paths(solid.clone(), 1.0);
    let mut material = solid;
    material.alpha_mode = AlphaMode::Blend;
    let (alone, batched) = scene.both_paths(material, 1.0);
    assert!(
        alone[0] > 4 && (alone[0] as i32) < opaque[0] as i32 / 2,
        "per-object should fade the quad: {alone:?} vs opaque {opaque:?}"
    );
    assert_close("Blend at full opacity", alone, batched, 4);
}

/// An opaque material below full opacity fades by the opacity alone on both
/// paths: its texture's alpha is ignored, as it is at full opacity.
#[test]
fn faded_opaque_material_ignores_texture_alpha_on_both_paths() {
    let Some(mut scene) = Scene::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut textured = |alpha: u8| {
        let tex = scene
            .renderer
            .resources_mut()
            .upload_texture(
                &scene.device,
                &scene.queue,
                viewport_lib::TextureData::srgb(1, 1, vec![220, 220, 220, alpha]),
            )
            .unwrap();
        let mut m = Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0));
        m.texture_id = Some(tex);
        m
    };
    let solid = textured(255);
    let see_through = textured(51);
    let (expected, _) = scene.both_paths(solid, 0.8);
    let (alone, batched) = scene.both_paths(see_through, 0.8);
    assert!(expected[0] > 10, "the surface should render: {expected:?}");
    assert_close("faded Opaque, per-object", expected, alone, 2);
    assert_close("faded Opaque, instanced", expected, batched, 2);
}

/// An opaque masked material with no albedo texture takes its alpha from the
/// vertex colours; both paths cut it out.
#[test]
fn untextured_alpha_mask_discards_on_both_paths() {
    let Some(mut scene) = Scene::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut mesh = quad_mesh(1.0);
    mesh.vertex_colours = Some(vec![[1.0, 1.0, 1.0, 0.2]; 4]);
    scene.mesh = scene
        .renderer
        .resources_mut()
        .upload_mesh_data(&scene.device, &mesh)
        .unwrap();
    let mut material = Material::from_colour(Colour::linear_rgb(1.0, 1.0, 1.0));
    material.alpha_mode = AlphaMode::Mask(0.5);
    let (alone, batched) = scene.both_paths(material, 1.0);
    assert!(
        alone[0] < 4,
        "per-object should cut the quad out: {alone:?}"
    );
    assert_close("untextured Mask", alone, batched, 2);
}

/// A material whose emissive texture has been freed falls back to its plain
/// emissive factor on both paths, rather than sampling the empty fallback.
#[test]
fn freed_emissive_texture_falls_back_on_both_paths() {
    let Some(mut scene) = Scene::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let tex = scene
        .renderer
        .resources_mut()
        .upload_texture(
            &scene.device,
            &scene.queue,
            viewport_lib::TextureData::srgb(1, 1, vec![255, 255, 255, 255]),
        )
        .unwrap();
    scene.renderer.resources_mut().free_texture(tex);
    let mut material = Material::from_colour(Colour::linear_rgb(0.0, 0.0, 0.0));
    material.emissive = Colour::linear_rgb(0.5, 0.5, 0.5);
    material.emissive_texture_id = Some(tex);
    let (alone, batched) = scene.both_paths(material, 1.0);
    assert!(
        alone[0] > 20,
        "per-object should keep its emissive factor: {alone:?}"
    );
    assert_close("freed emissive texture", alone, batched, 4);
}

/// A caster masked by vertex-colour alpha casts no shadow where it is cut out,
/// on both paths: the floor under it is as bright as with no caster at all.
#[test]
fn vertex_alpha_mask_cuts_the_shadow_on_both_paths() {
    let Some(mut scene) = Scene::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let floor_mesh = scene
        .renderer
        .resources_mut()
        .upload_mesh_data(&scene.device, &viewport_lib::primitives::plane(8.0, 8.0))
        .unwrap();
    let mut caster_mesh = quad_mesh(1.0);
    caster_mesh.vertex_colours = Some(vec![[1.0, 1.0, 1.0, 0.2]; 4]);
    let caster_mesh = scene
        .renderer
        .resources_mut()
        .upload_mesh_data(&scene.device, &caster_mesh)
        .unwrap();

    let mut floor = SceneRenderItem::default();
    floor.mesh_id = floor_mesh;
    floor.material = Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8));
    let caster_at = |x: f32, per_object: bool| {
        let mut it = SceneRenderItem::default();
        it.mesh_id = caster_mesh;
        it.model = glam::Mat4::from_translation(glam::Vec3::new(x, x, 0.5)).to_cols_array_2d();
        it.material = Material::from_colour(Colour::linear_rgb(0.2, 0.2, 0.8));
        it.material.alpha_mode = AlphaMode::Mask(0.5);
        // A lone quad casts only when two-sided: the shadow pass culls front
        // faces, and this one faces the light.
        it.material.backface_policy = viewport_lib::BackfacePolicy::Identical;
        if per_object {
            // A per-material sampler keeps an item per-object without changing
            // what its shadow pass does.
            it.material = it.material.with_sampler(
                viewport_lib::TextureSlot::Albedo,
                viewport_lib::SamplerKey::default(),
            );
        }
        it
    };

    let mut floor_centre = |items: Vec<SceneRenderItem>| -> i32 {
        const SIZE: u32 = 64;
        scene.generation += 1;
        let mut frame = FrameData::default();
        frame.scene.generation = scene.generation;
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.effects.display.exposure = viewport_lib::ExposureSettings::manual(0.0);
        let mut sun = LightSource::default();
        sun.kind = LightKind::Directional {
            direction: [0.0, 0.0, 1.0],
        };
        frame.effects.lighting.lights = vec![sun];
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
        frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
        let px = scene
            .renderer
            .render_offscreen(&scene.device, &scene.queue, &frame, SIZE, SIZE);
        let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
        px[i] as i32 + px[i + 1] as i32 + px[i + 2] as i32
    };

    let no_caster = floor_centre(vec![floor.clone()]);
    let per_object = floor_centre(vec![floor.clone(), caster_at(0.0, true)]);
    let batched = floor_centre(vec![
        floor.clone(),
        caster_at(0.0, false),
        caster_at(2.5, false),
    ]);
    for (label, lit) in [("per-object", per_object), ("instanced", batched)] {
        assert!(
            (lit - no_caster).abs() < 30,
            "{label}: the floor under a cut-out caster ({lit}) should be lit as with no caster ({no_caster})"
        );
    }
}
