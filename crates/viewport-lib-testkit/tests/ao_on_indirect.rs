//! Ambient occlusion darkens indirect light, contact shadows direct light.
//!
//! The scene pass writes how each pixel's light splits into indirect, direct
//! and emitted, and the composite scales ambient occlusion by the indirect
//! share and contact shadows by the direct share. A box floats just above a
//! plane under an overhead sun.
//!
//! - SSAO: with the hemisphere fill off every pixel is direct or black, so
//!   turning it on changes nothing; with the fill on, the shadowed plane under
//!   the box is lit by the fill alone and darkens.
//! - Contact shadows: the sun casts no shadow map, so the plane under the box
//!   is sunlit and darkens; with the sun off every pixel is indirect and
//!   nothing changes, and with both on the fill-only pixels do not change.
//!
//! In both, an emissive card in the plane's place is unchanged, and a sunlit
//! pixel beyond the box's reach is bit-identical. The tests read back a
//! half-float target through Reinhard, which inverts in closed form, so a
//! pixel's darkening can be bounded by its share of the light.

use viewport_lib::{
    CameraFrame, Colour, FrameData, LightKind, LightSource, LightingSettings, Material, MeshItem,
    SceneFrame, ShadingModel, ToneMapping,
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
    /// Whether the sun casts a shadow map.
    sun_shadows: bool,
    plane: viewport_lib::MeshId,
    cube: viewport_lib::MeshId,
    grey: viewport_lib::MaterialId,
    card: viewport_lib::MaterialId,
    generation: u64,
}

impl Scene {
    fn upload(h: &mut Harness, ssaa_factor: u32, sun_shadows: bool) -> Self {
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
            sun_shadows,
            plane,
            cube,
            grey,
            card,
            generation: 0,
        }
    }

    /// The box and plane under an overhead sun at `sun` times its default
    /// intensity, with the hemisphere fill at `fill` and `occlusion` on when
    /// given.
    fn frame(
        &mut self,
        ground: Ground,
        sun: f32,
        fill: f32,
        occlusion: Option<Occlusion>,
    ) -> FrameData {
        let mut plane = MeshItem::default();
        plane.mesh_id = self.plane;
        plane.material = match ground {
            Ground::Diffuse => self.grey,
            Ground::EmissiveCard => self.card,
        };
        let mut cube = MeshItem::default();
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
        let mut light = LightSource::default();
        light.kind = LightKind::Directional {
            direction: [0.0, 0.0, 1.0],
        };
        light.intensity *= sun;
        light.cast_shadows = self.sun_shadows;
        lighting.lights = if sun > 0.0 { vec![light] } else { Vec::new() };
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

/// One lighting setup rendered with the term off and on.
struct Pair {
    off: Vec<[f32; 4]>,
    on: Vec<[f32; 4]>,
}

impl Pair {
    fn changed(&self, i: usize) -> bool {
        self.off[i] != self.on[i]
    }

    /// The pixels in `of` that darken by more than 3%.
    fn darkened(&self, of: &[usize]) -> Vec<usize> {
        of.iter()
            .copied()
            .filter(|&i| luminance(self.on[i]) < luminance(self.off[i]) * 0.97)
            .collect()
    }
}

/// The renders one test compares, all over the same harness.
struct Renders {
    /// Sun alone.
    sun: Pair,
    /// Fill alone.
    fill: Pair,
    /// Sun and fill.
    both: Pair,
    /// The emissive card under sun and fill.
    card: Pair,
}

const FILL: f32 = 0.55;

fn render(occlusion: Occlusion, ssaa_factor: u32, sun_shadows: bool) -> Option<Renders> {
    let mut h = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT)?;
    let mut scene = Scene::upload(&mut h, ssaa_factor, sun_shadows);
    let mut pair = |h: &mut Harness, ground, sun, fill| {
        let off = scene.frame(ground, sun, fill, None);
        let off = h.render_float(&off, W, H);
        let on = scene.frame(ground, sun, fill, Some(occlusion));
        // A second frame, so a term that settles over frames has settled.
        h.render_float(&on, W, H);
        let on = scene.frame(ground, sun, fill, Some(occlusion));
        Pair {
            off,
            on: h.render_float(&on, W, H),
        }
    };
    Some(Renders {
        sun: pair(&mut h, Ground::Diffuse, 1.0, 0.0),
        fill: pair(&mut h, Ground::Diffuse, 0.0, FILL),
        both: pair(&mut h, Ground::Diffuse, 1.0, FILL),
        card: pair(&mut h, Ground::EmissiveCard, 1.0, FILL),
    })
}

/// Pixels the sun lights.
fn sunlit(r: &Renders) -> Vec<usize> {
    (0..r.sun.off.len())
        .filter(|&i| luminance(r.sun.off[i]) > 0.05)
        .collect()
}

/// Pixels the sun does not reach but the fill does.
fn fill_only(r: &Renders) -> Vec<usize> {
    (0..r.sun.off.len())
        .filter(|&i| luminance(r.sun.off[i]) < 1e-4 && luminance(r.both.off[i]) > 0.02)
        .collect()
}

fn count_changed(p: &Pair) -> usize {
    (0..p.off.len()).filter(|&i| p.changed(i)).count()
}

fn report(label: &str, darkened: &[usize], of: usize, p: &Pair) {
    let deepest = darkened
        .iter()
        .map(|&i| luminance(p.on[i]) / luminance(p.off[i]))
        .fold(1.0f32, f32::min);
    eprintln!(
        "{label}: {} of {of} pixels darken, deepest to {deepest:.3}",
        darkened.len()
    );
}

/// With sun and fill, every sunlit pixel loses at most its share of the light
/// the term applies to (`direct` picks the share) and one beyond the reach of
/// the box is bit-identical. Returns how many changed.
fn bound_sunlit(label: &str, r: &Renders, lit: &[usize], direct: bool) -> usize {
    let reach = reach_rect(0.6);
    let mut changed = 0usize;
    let mut outside = 0usize;
    let mut worst = 0.0f32;
    for &i in lit {
        let (off, on) = (r.both.off[i], r.both.on[i]);
        if !inside(reach, i) {
            outside += 1;
            assert_eq!(
                off,
                on,
                "{label} changed sunlit pixel {} {} beyond the box",
                i as u32 % W,
                i as u32 / W
            );
        }
        if off == on {
            continue;
        }
        changed += 1;
        let direct_share = (luminance(r.sun.off[i]) / luminance(off)).clamp(0.0, 1.0);
        let share = if direct {
            direct_share
        } else {
            1.0 - direct_share
        };
        let lost = 1.0 - luminance(on) / luminance(off);
        worst = worst.max(lost - share);
    }
    eprintln!(
        "{label}: {changed} of {} sunlit pixels change with sun and fill, \
         {outside} beyond the box all unchanged",
        lit.len()
    );
    assert!(
        outside > 10_000,
        "{label}: only {outside} sunlit pixels beyond the box"
    );
    assert!(
        worst <= 0.01,
        "{label}: a sunlit pixel lost {worst:.3} more than its share"
    );
    changed
}

/// The emissive card is unchanged everywhere, including over `darkened`, the
/// pixels where the grey plane darkened.
fn card_unchanged(label: &str, r: &Renders, darkened: &[usize]) {
    let card: Vec<usize> = (0..r.card.off.len())
        .filter(|&i| {
            let p = r.card.off[i];
            p[1] > 0.05 && p[0] < 1e-3 && p[2] < 1e-3
        })
        .collect();
    let darkened: std::collections::HashSet<usize> = darkened.iter().copied().collect();
    let under = card.iter().filter(|i| darkened.contains(i)).count();
    eprintln!(
        "{label}: {} card pixels, {under} where the grey plane darkened",
        card.len()
    );
    assert!(
        card.len() > 10_000,
        "{label}: only {} card pixels",
        card.len()
    );
    assert!(
        under > 20,
        "{label}: only {under} card pixels where the grey plane darkened"
    );
    let changed = card.iter().filter(|&&i| r.card.changed(i)).count();
    assert_eq!(
        changed, 0,
        "{label} changed {changed} pixels of the emissive card"
    );
}

fn check_ssao(ssaa_factor: u32) {
    let label = format!("Ssao x{ssaa_factor}");
    let Some(r) = render(Occlusion::Ssao, ssaa_factor, true) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let lit = sunlit(&r);
    let shade = fill_only(&r);
    assert!(
        lit.len() > 10_000,
        "{label}: only {} sunlit pixels",
        lit.len()
    );
    assert!(
        shade.len() > 300,
        "{label}: only {} fill-only pixels",
        shade.len()
    );

    // Sun alone: every pixel is direct light or black, so nothing changes.
    let changed = count_changed(&r.sun);
    assert_eq!(
        changed, 0,
        "{label} changed {changed} pixels lit by the sun alone"
    );

    // Sun and fill: the shadowed pixels, lit by the fill alone, darken.
    let darkened = r.both.darkened(&shade);
    report(
        &format!("{label} fill-only"),
        &darkened,
        shade.len(),
        &r.both,
    );
    assert!(
        darkened.len() > 40,
        "{label} darkened only {} fill-only pixels",
        darkened.len()
    );
    bound_sunlit(&label, &r, &lit, false);
    card_unchanged(&label, &r, &darkened);
}

fn check_contact_shadows() {
    let label = "ContactShadows";
    // No shadow map, so the plane under the box is sunlit and the contact
    // shadow is the only thing that shades it.
    let Some(r) = render(Occlusion::ContactShadows, 1, false) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let lit = sunlit(&r);
    let shade = fill_only(&r);
    assert!(
        lit.len() > 10_000,
        "{label}: only {} sunlit pixels",
        lit.len()
    );
    assert!(
        shade.len() > 300,
        "{label}: only {} fill-only pixels",
        shade.len()
    );

    // Fill alone: every pixel is indirect light, so nothing changes.
    let changed = count_changed(&r.fill);
    assert_eq!(
        changed, 0,
        "{label} changed {changed} pixels lit by the fill alone"
    );

    // Sun and fill: sunlit pixels under the box darken, fill-only ones do not.
    let darkened = r.both.darkened(&lit);
    report(&format!("{label} sunlit"), &darkened, lit.len(), &r.both);
    assert!(
        darkened.len() > 40,
        "{label} darkened only {} sunlit pixels",
        darkened.len()
    );
    let shade_changed = shade.iter().filter(|&&i| r.both.changed(i)).count();
    assert_eq!(
        shade_changed,
        0,
        "{label} changed {shade_changed} of {} fill-only pixels",
        shade.len()
    );
    bound_sunlit(label, &r, &lit, true);
    card_unchanged(label, &r, &darkened);
}

#[test]
fn ssao_darkens_indirect_light_only() {
    check_ssao(1);
}

#[test]
fn contact_shadows_darken_direct_light_only() {
    check_contact_shadows();
}

/// Under SSAA the scene pass writes a supersampled split, which the resolve
/// averages down with the colour.
#[test]
fn ssao_under_ssaa_darkens_indirect_light_only() {
    check_ssao(2);
}
