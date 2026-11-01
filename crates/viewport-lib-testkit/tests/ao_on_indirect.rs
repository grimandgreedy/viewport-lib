//! Ambient occlusion and contact shadows darken indirect light only.
//!
//! The scene pass writes the indirect share of each pixel's light, and the
//! composite scales both screen-space occlusion terms by it. A box floats just
//! above a plane under an overhead sun: with the hemisphere fill off every
//! pixel is direct or black, so turning either term on changes nothing; with
//! the fill on, the shadowed plane under the box is lit by the fill alone and
//! darkens. An emissive card in the plane's place, whose light is all direct,
//! is unchanged.
//!
//! The tests read back a half-float target through Reinhard, which inverts in
//! closed form, so a lit pixel's darkening can be bounded by its indirect
//! share.

use viewport_lib::{
    CameraFrame, Colour, FrameData, LightKind, LightSource, LightingSettings, Material, SceneFrame,
    SceneRenderItem, ShadingModel, ToneMapping,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 320;
const H: u32 = 240;

/// The screen-space term a test turns on.
#[derive(Clone, Copy, Debug)]
enum Occlusion {
    Ssao,
    ContactShadows,
}

/// What the plane under the box is.
#[derive(Clone, Copy, Debug)]
enum Ground {
    /// Grey and diffuse.
    Diffuse,
    /// Black, emitting green: every pixel of it is direct light.
    EmissiveCard,
}

struct Scene {
    ssaa_factor: u32,
    plane: viewport_lib::MeshId,
    cube: viewport_lib::MeshId,
    grey: viewport_lib::MaterialId,
    card: viewport_lib::MaterialId,
    generation: u64,
}

impl Scene {
    fn upload(h: &mut Harness, ssaa_factor: u32) -> Self {
        let res = h.renderer.resources_mut();
        let plane = res
            .upload_mesh_data(
                &h.device,
                &viewport_lib::geometry::primitives::plane(12.0, 12.0),
            )
            .expect("plane");
        let cube = res
            .upload_mesh_data(&h.device, &viewport_lib::geometry::primitives::cube(1.0))
            .expect("cube");
        let grey = res
            .upload_material(Material::pbr(Colour::linear_rgb(0.6, 0.6, 0.6), 0.0, 0.8))
            .expect("grey");
        // Blinn-Phong over black reflects nothing, so the card's light is its
        // emission alone.
        let mut emissive = Material::from_colour(Colour::linear_rgb(0.0, 0.0, 0.0));
        emissive.shading_model = ShadingModel::Phong;
        emissive.emissive = Colour::linear_rgb(0.0, 1.0, 0.0);
        emissive.emissive_strength = 0.6;
        emissive.emissive_exposure_weight = 0.0;
        let card = res.upload_material(emissive).expect("card");
        Self {
            ssaa_factor,
            plane,
            cube,
            grey,
            card,
            generation: 0,
        }
    }

    /// The box and plane under an overhead sun, with the hemisphere fill at
    /// `fill` and `occlusion` on when given.
    fn frame(&mut self, ground: Ground, fill: f32, occlusion: Option<Occlusion>) -> FrameData {
        let mut plane = SceneRenderItem::default();
        plane.mesh_id = self.plane;
        plane.material = match ground {
            Ground::Diffuse => self.grey,
            Ground::EmissiveCard => self.card,
        };
        let mut cube = SceneRenderItem::default();
        cube.mesh_id = self.cube;
        cube.material = self.grey;
        // The underside 0.15 above the plane, inside the AO radius.
        cube.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.65)).into();

        self.generation += 1;
        let camera = camera();
        let mut frame = FrameData::new(
            CameraFrame::from_camera(&camera, [W as f32, H as f32]),
            SceneFrame::from_surface_items(vec![plane, cube]).with_generation(self.generation),
        );
        let mut lighting = LightingSettings::default();
        let mut sun = LightSource::default();
        sun.kind = LightKind::Directional {
            direction: [0.0, 0.0, 1.0],
        };
        lighting.lights = vec![sun];
        lighting.hemisphere_intensity = fill;
        frame.effects.lighting = lighting;
        frame.effects.display.operator = ToneMapping::Reinhard;
        let pp = &mut frame.effects.post_process;
        pp.ssaa_factor = self.ssaa_factor;
        match occlusion {
            Some(Occlusion::Ssao) => pp.ssao = true,
            Some(Occlusion::ContactShadows) => pp.contact_shadows.enabled = true,
            None => {}
        }
        frame.viewport.background_colour = Some(Colour::BLACK);
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame
    }
}

fn camera() -> viewport_lib::Camera {
    let mut camera = orbit_camera(glam::Vec3::ZERO, 6.0, 0.4, 0.45);
    camera.aspect = W as f32 / H as f32;
    camera
}

/// The screen rectangle, in pixels, that holds every point within `reach`
/// of the box: the most the occlusion terms can touch.
fn reach_rect(reach: f32) -> [f32; 4] {
    let view_proj = camera().view_proj_matrix();
    let half = 0.5 + reach;
    let mut rect = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for x in [-half, half] {
        for y in [-half, half] {
            for z in [0.0, 1.15 + reach] {
                let clip = view_proj * glam::Vec4::new(x, y, z, 1.0);
                let ndc = clip.truncate() / clip.w;
                let px = (ndc.x * 0.5 + 0.5) * W as f32;
                let py = (0.5 - ndc.y * 0.5) * H as f32;
                rect = [
                    rect[0].min(px),
                    rect[1].min(py),
                    rect[2].max(px),
                    rect[3].max(py),
                ];
            }
        }
    }
    rect
}

fn inside(rect: [f32; 4], i: usize) -> bool {
    let (x, y) = ((i as u32 % W) as f32 + 0.5, (i as u32 / W) as f32 + 0.5);
    x >= rect[0] && x <= rect[2] && y >= rect[1] && y <= rect[3]
}

/// Undo Reinhard on one channel.
fn linear(y: f32) -> f32 {
    let y = y.min(0.999);
    y / (1.0 - y)
}

fn luminance(p: [f32; 4]) -> f32 {
    0.2126 * linear(p[0]) + 0.7152 * linear(p[1]) + 0.0722 * linear(p[2])
}

/// The renders one test compares, all over the same harness.
struct Renders {
    /// Sun only, occlusion off and on.
    sun_off: Vec<[f32; 4]>,
    sun_on: Vec<[f32; 4]>,
    /// Sun and fill, occlusion off and on.
    fill_off: Vec<[f32; 4]>,
    fill_on: Vec<[f32; 4]>,
    /// The emissive card under sun and fill, occlusion off and on.
    card_off: Vec<[f32; 4]>,
    card_on: Vec<[f32; 4]>,
}

const FILL: f32 = 0.55;

fn render(occlusion: Occlusion, ssaa_factor: u32) -> Option<Renders> {
    let mut h = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT)?;
    let mut scene = Scene::upload(&mut h, ssaa_factor);
    let mut pair = |h: &mut Harness, ground, fill| {
        let off = scene.frame(ground, fill, None);
        let off = h.render_float(&off, W, H);
        let on = scene.frame(ground, fill, Some(occlusion));
        // A second frame, so a term that settles over frames has settled.
        h.render_float(&on, W, H);
        let on = scene.frame(ground, fill, Some(occlusion));
        (off, h.render_float(&on, W, H))
    };
    let (sun_off, sun_on) = pair(&mut h, Ground::Diffuse, 0.0);
    let (fill_off, fill_on) = pair(&mut h, Ground::Diffuse, FILL);
    let (card_off, card_on) = pair(&mut h, Ground::EmissiveCard, FILL);
    Some(Renders {
        sun_off,
        sun_on,
        fill_off,
        fill_on,
        card_off,
        card_on,
    })
}

/// Pixels the sun lights.
fn sunlit(r: &Renders) -> Vec<usize> {
    (0..r.sun_off.len())
        .filter(|&i| luminance(r.sun_off[i]) > 0.05)
        .collect()
}

/// Pixels the sun does not reach but the fill does: the shadowed plane and
/// the box's sides.
fn fill_only(r: &Renders) -> Vec<usize> {
    (0..r.sun_off.len())
        .filter(|&i| luminance(r.sun_off[i]) < 1e-4 && luminance(r.fill_off[i]) > 0.02)
        .collect()
}

fn check(occlusion: Occlusion, ssaa_factor: u32) {
    let Some(r) = render(occlusion, ssaa_factor) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let lit = sunlit(&r);
    let shade = fill_only(&r);
    assert!(
        lit.len() > 10_000,
        "{occlusion:?}: only {} sunlit pixels",
        lit.len()
    );
    assert!(
        shade.len() > 300,
        "{occlusion:?}: only {} fill-only pixels",
        shade.len()
    );

    // Sun alone: every pixel is direct light or black, so the term changes
    // nothing, bit for bit.
    let changed = (0..r.sun_off.len())
        .filter(|&i| r.sun_off[i] != r.sun_on[i])
        .count();
    assert_eq!(
        changed, 0,
        "{occlusion:?} changed {changed} pixels lit by the sun alone"
    );

    // Sun and fill: the shadowed pixels, lit by the fill alone, darken.
    let darkened: Vec<usize> = shade
        .iter()
        .copied()
        .filter(|&i| luminance(r.fill_on[i]) < luminance(r.fill_off[i]) * 0.97)
        .collect();
    let deepest = darkened
        .iter()
        .map(|&i| luminance(r.fill_on[i]) / luminance(r.fill_off[i]))
        .fold(1.0f32, f32::min);
    eprintln!(
        "{occlusion:?}: {} of {} fill-only pixels darken, deepest to {deepest:.3}",
        darkened.len(),
        shade.len()
    );
    assert!(
        darkened.len() > 40,
        "{occlusion:?} darkened only {} of {} fill-only pixels",
        darkened.len(),
        shade.len()
    );

    // A sunlit pixel loses at most its indirect share, the fill's part of it,
    // and one beyond the reach of the box is bit-identical.
    let reach = reach_rect(0.6);
    let mut lit_changed = 0usize;
    let mut lit_outside = 0usize;
    let mut worst = 0.0f32;
    for &i in &lit {
        let (off, on, sun) = (r.fill_off[i], r.fill_on[i], r.sun_off[i]);
        if !inside(reach, i) {
            lit_outside += 1;
            assert_eq!(
                off,
                on,
                "{occlusion:?} changed sunlit pixel {} {} beyond the box",
                i as u32 % W,
                i as u32 / W
            );
        }
        if off == on {
            continue;
        }
        lit_changed += 1;
        let indirect_share = (1.0 - luminance(sun) / luminance(off)).clamp(0.0, 1.0);
        let lost = 1.0 - luminance(on) / luminance(off);
        worst = worst.max(lost - indirect_share);
    }
    eprintln!(
        "{occlusion:?}: {lit_changed} of {} sunlit pixels change with the fill on, \
         {lit_outside} beyond the box all unchanged",
        lit.len()
    );
    assert!(
        lit_outside > 10_000,
        "{occlusion:?}: only {lit_outside} sunlit pixels beyond the box"
    );
    assert!(
        worst <= 0.01,
        "{occlusion:?}: a sunlit pixel lost {worst:.3} more than its indirect share"
    );

    // The emissive card: its light is direct, so it is unchanged everywhere,
    // under the box included.
    let card: Vec<usize> = (0..r.card_off.len())
        .filter(|&i| {
            let p = r.card_off[i];
            p[1] > 0.05 && p[0] < 1e-3 && p[2] < 1e-3
        })
        .collect();
    let darkened: std::collections::HashSet<usize> = darkened.into_iter().collect();
    let card_under_box = card.iter().filter(|i| darkened.contains(i)).count();
    eprintln!(
        "{occlusion:?}: {} card pixels, {card_under_box} where the grey plane darkened",
        card.len()
    );
    assert!(
        card.len() > 10_000,
        "{occlusion:?}: only {} card pixels",
        card.len()
    );
    assert!(
        card_under_box > 20,
        "{occlusion:?}: only {card_under_box} card pixels where the grey plane darkened"
    );
    let card_changed = card
        .iter()
        .filter(|&&i| r.card_off[i] != r.card_on[i])
        .count();
    assert_eq!(
        card_changed, 0,
        "{occlusion:?} changed {card_changed} pixels of the emissive card"
    );
}

#[test]
fn ssao_darkens_indirect_light_only() {
    check(Occlusion::Ssao, 1);
}

#[test]
fn contact_shadows_darken_indirect_light_only() {
    check(Occlusion::ContactShadows, 1);
}

/// Under SSAA the scene pass writes a supersampled fraction, which the resolve
/// averages down with the colour.
#[test]
fn ssao_under_ssaa_darkens_indirect_light_only() {
    check(Occlusion::Ssao, 2);
}
