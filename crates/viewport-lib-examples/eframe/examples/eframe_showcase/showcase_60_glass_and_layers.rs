//! Showcase 60: Glass and Layered Materials.
//!
//! The transmission and layered-lobe fields of `Material`, under image-based
//! lighting (the gradient environment of showcase 58) and a daylight key,
//! seen through the camera's own physical exposure (`Camera::exposure`).
//!
//! - **Transmission** - a glass sphere, a thick glass slab and a hollow
//!   two-sided bottle holding a liquid at another index of refraction, in
//!   front of a patterned back wall. Sliders drive `transmission`,
//!   `thickness`, `ior`, `attenuation_colour` and `attenuation_distance` on
//!   the selected object; a toggle draws the slab as `AlphaMode::Blend` at
//!   the same opacity instead, which neither refracts nor tints by depth. A
//!   camera preset looks through two panes in a row: a transmissive surface
//!   reads a copy of the opaque scene, so the far pane shows the environment
//!   rather than itself.
//! - **Layers** - a clear-coated metallic-flake sphere (`clearcoat`,
//!   `clearcoat_roughness`), a velvet sphere (`sheen_colour`,
//!   `sheen_roughness`) and a brushed-metal cylinder (`anisotropy`,
//!   `anisotropy_rotation`), each beside a twin with the same base and no
//!   layer, under a key light with a yaw slider.
//!
//! **Trace** builds a `raytrace::RtScene` from the meshes and materials on
//! screen, sets `RtScene::set_exposure` from the camera's multiplier and path
//! traces the view at a chosen sample count, shown beside the raster in the
//! sidebar.

use crate::App;
use crate::eframe::egui;
use crate::shared::{edit_material, upload_material};
use viewport_lib as vpl;
use viewport_lib::Colour;
use vpl::{
    AlphaMode, BackfacePolicy, EnvironmentIntensity, EnvironmentLighting, EnvironmentMapId,
    ExposureSettings, LightSource, LightingSettings, Lux, Material, MaterialId, MeshData,
    ViewportRenderer, scene::Scene,
};

/// Size of the gradient environment, and its brightness in nits.
const ENV_W: u32 = 64;
const ENV_H: u32 = 32;
const ENV_NITS: f32 = 10_000.0;
/// The key light, in lux.
const KEY_LUX: f32 = 100_000.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlassSub {
    Transmission,
    Layers,
}

/// The transmissive object the sliders edit.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlassTarget {
    Sphere,
    Slab,
    Liquid,
}

/// The volume fields the transmission sliders drive.
#[derive(Clone, Copy)]
pub(crate) struct Volume {
    transmission: f32,
    thickness: f32,
    ior: f32,
    attenuation: [f32; 3],
    attenuation_distance: f32,
}

impl Volume {
    fn apply(&self, m: &mut Material) {
        m.transmission = self.transmission;
        m.thickness = self.thickness;
        *m = m.with_ior(self.ior);
        m.attenuation_colour = Colour::from_linear_rgb_array(self.attenuation);
        m.attenuation_distance = self.attenuation_distance;
    }
}

/// A mesh on screen, kept on the CPU so the tracer can rebuild the scene.
struct Part {
    mesh: MeshData,
    model: glam::Mat4,
    material: MaterialId,
}

pub(crate) struct GlassState {
    pub sub: GlassSub,
    built: [bool; 2],
    scenes: [Scene; 2],
    parts: [Vec<Part>; 2],
    env: Option<EnvironmentMapId>,
    env_pixels: Vec<f32>,
    // Transmission.
    target: GlassTarget,
    volumes: [Volume; 3],
    glass: [MaterialId; 3],
    slab_blend: bool,
    // Layers.
    pub key_yaw: f32,
    clearcoat: (f32, f32),
    sheen: ([f32; 3], f32),
    anisotropy: (f32, f32),
    layered: [MaterialId; 3],
    // Trace.
    samples: u32,
    trace: Option<(Vec<u8>, [usize; 2])>,
    trace_texture: Option<egui::TextureHandle>,
}

impl Default for GlassState {
    fn default() -> Self {
        let clear = Volume {
            transmission: 1.0,
            thickness: 1.0,
            ior: 1.5,
            attenuation: [1.0, 1.0, 1.0],
            attenuation_distance: 1.0e6,
        };
        Self {
            sub: GlassSub::Transmission,
            built: [false; 2],
            scenes: [Scene::new(), Scene::new()],
            parts: [Vec::new(), Vec::new()],
            env: None,
            env_pixels: Vec::new(),
            target: GlassTarget::Slab,
            volumes: [
                Volume {
                    thickness: 1.6,
                    ..clear
                },
                Volume {
                    thickness: 0.4,
                    attenuation: [0.55, 0.85, 0.8],
                    attenuation_distance: 0.6,
                    ..clear
                },
                Volume {
                    thickness: 0.8,
                    ior: 1.33,
                    attenuation: [0.95, 0.55, 0.15],
                    attenuation_distance: 0.5,
                    ..clear
                },
            ],
            glass: [MaterialId::DEFAULT; 3],
            slab_blend: false,
            key_yaw: 0.6,
            clearcoat: (1.0, 0.05),
            sheen: ([0.95, 0.55, 0.65], 0.5),
            anisotropy: (0.8, 0.0),
            layered: [MaterialId::DEFAULT; 3],
            samples: 64,
            trace: None,
            trace_texture: None,
        }
    }
}

impl GlassState {
    fn index(&self) -> usize {
        match self.sub {
            GlassSub::Transmission => 0,
            GlassSub::Layers => 1,
        }
    }

    fn lighting(&self) -> LightingSettings {
        let (s, c) = self.key_yaw.sin_cos();
        let mut key = LightSource::directional_lux([c * 0.8, s * 0.8 - 0.4, 1.0], Lux(KEY_LUX));
        key.cast_shadows = true;
        let mut l = LightingSettings::default();
        l.lights = vec![key];
        l.shadows.enabled = true;
        // The environment carries the ambient and the reflections.
        l.hemisphere_intensity = 0.0;
        l.environment_intensity = EnvironmentIntensity::Multiplier(ENV_NITS);
        l
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// A 256 x 256 checker for the back wall, with a grid of thin lines so
/// refraction and blur read.
fn wall_texture() -> vpl::TextureData {
    let n = 256u32;
    let mut px = Vec::with_capacity((n * n * 4) as usize);
    for y in 0..n {
        for x in 0..n {
            let check = ((x / 32) + (y / 32)) % 2 == 0;
            let line = x % 32 < 2 || y % 32 < 2;
            let c = if line {
                [30, 30, 36]
            } else if check {
                [235, 200, 120]
            } else {
                [70, 110, 170]
            };
            px.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    vpl::TextureData::srgb(n, n, px)
}

/// A tangent-space normal map of small random flakes, for the metallic paint.
fn flake_normals() -> vpl::TextureData {
    let n = 128u32;
    let mut seed = 0x9e37_79b9u32;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut px = Vec::with_capacity((n * n * 4) as usize);
    for _ in 0..n * n {
        let (x, y) = (rand() * 0.35, rand() * 0.35);
        let z = (1.0 - x * x - y * y).max(0.0).sqrt();
        let enc = |v: f32| ((v * 0.5 + 0.5) * 255.0).round() as u8;
        px.extend_from_slice(&[enc(x), enc(y), enc(z), 255]);
    }
    vpl::TextureData::normal_map(n, n, px)
}

/// Add `mesh` to scene `index` and keep its CPU copy for the tracer.
fn add_part(
    app: &mut App,
    renderer: &mut ViewportRenderer,
    index: usize,
    name: &str,
    mesh: MeshData,
    model: glam::Mat4,
    material: Material,
) -> MaterialId {
    let id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &mesh)
        .expect("glass mesh upload");
    let mat = upload_material(renderer, material);
    app.glass_state.scenes[index].add_named(name, Some(id), model, mat);
    app.glass_state.parts[index].push(Part {
        mesh,
        model,
        material: mat,
    });
    mat
}

fn ensure_environment(app: &mut App, renderer: &mut ViewportRenderer) {
    if app.glass_state.env.is_some() {
        return;
    }
    let px = crate::showcase_58_physically_based_surfaces::equirect_gradient(
        [0.55, 0.7, 1.0],
        [0.3, 0.26, 0.22],
        ENV_W,
        ENV_H,
    );
    app.glass_state.env_pixels = px.clone();
    let env = renderer
        .upload_environment(
            &app.device,
            &app.queue,
            vpl::TextureData::hdr(ENV_W, ENV_H, px),
            vpl::EnvironmentOptions::default(),
        )
        .expect("environment upload");
    app.glass_state.env = Some(env);
}

fn build_transmission(app: &mut App, renderer: &mut ViewportRenderer) {
    let floor = Material::pbr(Colour::linear_rgb(0.45, 0.45, 0.45), 0.0, 0.8);
    add_part(
        app,
        renderer,
        0,
        "Floor",
        vpl::primitives::cuboid(16.0, 10.0, 0.1),
        glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.5, -0.05)),
        floor,
    );
    let wall_tex = renderer
        .resources_mut()
        .upload_texture(&app.device, &app.queue, wall_texture())
        .expect("wall texture upload");
    let mut wall = Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.7);
    wall.texture_id = Some(wall_tex);
    add_part(
        app,
        renderer,
        0,
        "Patterned wall",
        vpl::primitives::cuboid(14.0, 0.1, 5.0),
        glam::Mat4::from_translation(glam::Vec3::new(1.0, 3.05, 2.5)),
        wall,
    );

    let glass = |v: &Volume| {
        let mut m = Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.04);
        v.apply(&mut m);
        m
    };
    let volumes = app.glass_state.volumes;
    let sphere = add_part(
        app,
        renderer,
        0,
        "Glass sphere",
        vpl::primitives::sphere(0.8, 64, 32),
        glam::Mat4::from_translation(glam::Vec3::new(-2.4, 0.0, 0.8)),
        glass(&volumes[0]),
    );
    let slab = add_part(
        app,
        renderer,
        0,
        "Glass slab",
        vpl::primitives::cuboid(1.6, 0.4, 1.6),
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.8)),
        glass(&volumes[1]),
    );
    // The bottle: a thin two-sided glass wall around a liquid.
    let mut wall_glass = Material::pbr(Colour::linear_rgb(1.0, 1.0, 1.0), 0.0, 0.03)
        .with_transmission(1.0, 0.04)
        .with_ior(1.5);
    wall_glass.backface_policy = BackfacePolicy::Identical;
    add_part(
        app,
        renderer,
        0,
        "Bottle",
        vpl::primitives::cylinder(0.5, 1.6, 64),
        glam::Mat4::from_translation(glam::Vec3::new(2.4, 0.0, 0.8)),
        wall_glass,
    );
    let liquid = add_part(
        app,
        renderer,
        0,
        "Liquid",
        vpl::primitives::cylinder(0.44, 0.9, 64),
        glam::Mat4::from_translation(glam::Vec3::new(2.4, 0.0, 0.47)),
        glass(&volumes[2]),
    );
    // Two panes in a row, for the "glass behind glass" view.
    for (name, y) in [("Near pane", -1.4), ("Far pane", -0.4)] {
        add_part(
            app,
            renderer,
            0,
            name,
            vpl::primitives::cuboid(1.2, 0.04, 1.4),
            glam::Mat4::from_translation(glam::Vec3::new(4.8, y, 0.7)),
            Material::pbr(Colour::linear_rgb(0.85, 1.0, 0.9), 0.0, 0.02)
                .with_transmission(1.0, 0.04),
        );
    }
    app.glass_state.glass = [sphere, slab, liquid];
}

fn build_layers(app: &mut App, renderer: &mut ViewportRenderer) {
    add_part(
        app,
        renderer,
        1,
        "Floor",
        vpl::primitives::cuboid(14.0, 10.0, 0.1),
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.05)),
        Material::pbr(Colour::linear_rgb(0.35, 0.35, 0.35), 0.0, 0.85),
    );
    let flakes = renderer
        .resources_mut()
        .upload_texture(&app.device, &app.queue, flake_normals())
        .expect("flake normal map upload");
    let st = &app.glass_state;
    let (cc, ccr) = st.clearcoat;
    let (sheen, sheen_rough) = st.sheen;
    let (aniso, aniso_rot) = st.anisotropy;

    let mut paint = Material::pbr(Colour::linear_rgb(0.55, 0.04, 0.05), 0.85, 0.45);
    paint.normal_map_id = Some(flakes);
    paint.normal_strength = 0.6;
    let velvet = Material::pbr(Colour::linear_rgb(0.25, 0.03, 0.06), 0.0, 0.9);
    let brushed = Material::pbr(Colour::linear_rgb(0.9, 0.9, 0.92), 1.0, 0.35);
    let sphere = || vpl::primitives::sphere(0.7, 64, 32);
    let rows = [
        (
            "Clear coat",
            sphere(),
            paint,
            paint.with_clearcoat(cc, ccr),
            -3.0,
        ),
        (
            "Sheen",
            sphere(),
            velvet,
            velvet.with_sheen(Colour::from_linear_rgb_array(sheen), sheen_rough),
            0.0,
        ),
        (
            "Anisotropy",
            vpl::primitives::cylinder(0.55, 1.4, 96),
            brushed,
            brushed.with_anisotropy(aniso, aniso_rot),
            3.0,
        ),
    ];
    let mut layered = [MaterialId::DEFAULT; 3];
    for (i, (name, mesh, plain, layer, x)) in rows.into_iter().enumerate() {
        add_part(
            app,
            renderer,
            1,
            &format!("{name}: plain twin"),
            mesh.clone(),
            glam::Mat4::from_translation(glam::Vec3::new(x, 1.0, 0.72)),
            plain,
        );
        layered[i] = add_part(
            app,
            renderer,
            1,
            &format!("{name}: layered"),
            mesh,
            glam::Mat4::from_translation(glam::Vec3::new(x, -1.0, 0.72)),
            layer,
        );
    }
    app.glass_state.layered = layered;
}

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------

/// The tracer's copy of a raster material: the fields the two share.
#[cfg(feature = "showcase")]
fn rt_material(m: &Material) -> vpl::raytrace::RtMaterial {
    let mut r = vpl::raytrace::RtMaterial::default();
    r.base_colour = m.base_colour;
    r.metallic = m.metallic;
    r.roughness = m.roughness;
    r.emissive = Colour::from_linear_rgb_array(m.emissive_nits());
    r.emissive_exposure_weight = m.emissive_exposure_weight;
    r.transmission = m.transmission;
    r.ior = m.ior;
    r.reflectance = m.reflectance;
    r.specular_factor = m.specular_factor;
    r.specular_colour = m.specular_colour;
    r.clearcoat = m.clearcoat;
    r.clearcoat_roughness = m.clearcoat_roughness;
    r.sheen_colour = m.sheen_colour;
    r.sheen_roughness = m.sheen_roughness;
    r.anisotropy = m.anisotropy;
    r.anisotropy_rotation = m.anisotropy_rotation;
    r.attenuation_colour = m.attenuation_colour;
    r.attenuation_distance = m.attenuation_distance;
    r
}

/// Path trace the active scene from the current camera at its exposure.
#[cfg(feature = "showcase")]
fn trace(app: &mut App, renderer: &ViewportRenderer, aspect: f32) {
    use vpl::raytrace::{RtCamera, RtLight, RtScene, RtSettings, Tracer};
    let st = &app.glass_state;
    let mut scene = RtScene::new();
    for part in &st.parts[st.index()] {
        let Some(m) = renderer.resources().material(part.material) else {
            continue;
        };
        let positions: Vec<glam::Vec3> = part.mesh.positions.iter().map(|&p| p.into()).collect();
        let normals: Vec<glam::Vec3> = part.mesh.normals.iter().map(|&n| n.into()).collect();
        let mesh = scene.add_mesh_geometry(
            &positions,
            &part.mesh.indices,
            Some(&normals),
            rt_material(m),
        );
        scene.add_instance(mesh, part.model);
    }
    let lighting = st.lighting();
    if let vpl::LightKind::Directional { direction } = lighting.lights[0].kind {
        scene.add_light(RtLight::Directional {
            direction: glam::Vec3::from(direction).normalize().to_array(),
            colour: Colour::linear_rgb(KEY_LUX, KEY_LUX, KEY_LUX),
        });
    }
    let env: Vec<f32> = st.env_pixels.iter().map(|v| v * ENV_NITS).collect();
    scene.set_environment(&env, ENV_W, ENV_H);
    let ev = physical_ev100(&app.camera.exposure);
    scene.set_exposure(1.0 / (1.2 * ev.exp2()));

    let width = 256u32;
    let height = (width as f32 / aspect.max(0.1)).round().max(1.0) as u32;
    let mut camera = app.camera.clone();
    camera.set_aspect_ratio(width as f32, height as f32);
    let image = Tracer::new(&app.device, &app.queue, &scene).trace(
        &app.device,
        &app.queue,
        &RtCamera {
            inv_view_proj: camera.view_proj_matrix().inverse(),
            position: camera.eye().as_vec3(),
            width,
            height,
        },
        &RtSettings {
            samples: st.samples,
            max_bounces: 8,
            denoise: false,
            seed: 0,
        },
    );
    // Khronos Neutral and an sRGB encode, as the raster's default view transform.
    let srgb = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let e = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (e * 255.0).round() as u8
    };
    let rgba = image
        .rgba
        .chunks_exact(4)
        .flat_map(|p| {
            let c = vpl::vplt::effects::tone_operators::khronos_neutral([p[0], p[1], p[2]]);
            [srgb(c[0]), srgb(c[1]), srgb(c[2]), 255]
        })
        .collect();
    app.glass_state.trace = Some((rgba, [width as usize, height as usize]));
    app.glass_state.trace_texture = None;
}

/// EV100 of a physical camera: `log2(N^2 / t) + log2(100 / ISO)`.
fn physical_ev100(camera: &vpl::PhysicalExposure) -> f32 {
    let n = camera.aperture;
    (n * n / camera.shutter_seconds.max(1e-6)).log2() + (100.0 / camera.iso.max(1.0)).log2()
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

fn controls(app: &mut App, ui: &mut egui::Ui, frame: &crate::eframe::Frame) {
    ui.horizontal(|ui| {
        let st = &mut app.glass_state;
        ui.selectable_value(&mut st.sub, GlassSub::Transmission, "Transmission");
        ui.selectable_value(&mut st.sub, GlassSub::Layers, "Layers");
    });
    ui.separator();
    match app.glass_state.sub {
        GlassSub::Transmission => controls_transmission(app, ui, frame),
        GlassSub::Layers => controls_layers(app, ui, frame),
    }
    ui.separator();
    controls_trace(app, ui, frame);
}

fn controls_transmission(app: &mut App, ui: &mut egui::Ui, frame: &crate::eframe::Frame) {
    ui.label(
        "Transmissive surfaces draw after the opaque scene and read a copy of it, \
         refracted by ior and thickness and tinted by attenuation over distance.",
    );
    let st = &mut app.glass_state;
    ui.horizontal(|ui| {
        ui.label("Edit:");
        ui.selectable_value(&mut st.target, GlassTarget::Sphere, "Sphere");
        ui.selectable_value(&mut st.target, GlassTarget::Slab, "Slab");
        ui.selectable_value(&mut st.target, GlassTarget::Liquid, "Liquid");
    });
    let i = st.target as usize;
    let v = &mut st.volumes[i];
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut v.transmission, 0.0..=1.0).text("transmission"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut v.thickness, 0.0..=2.0).text("thickness"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut v.ior, 1.0..=2.4).text("ior"))
        .changed();
    ui.horizontal(|ui| {
        ui.label("attenuation_colour");
        changed |= ui.color_edit_button_rgb(&mut v.attenuation).changed();
    });
    changed |= ui
        .add(
            egui::Slider::new(&mut v.attenuation_distance, 0.05..=1.0e6)
                .logarithmic(true)
                .text("attenuation_distance"),
        )
        .changed();
    let slab_changed = ui
        .checkbox(
            &mut st.slab_blend,
            "Slab as AlphaMode::Blend (same opacity)",
        )
        .changed();
    if st.slab_blend {
        ui.weak("Blend lays a tinted layer over the wall: no refraction, no depth tint.");
    }
    if changed || slab_changed {
        let volumes = st.volumes;
        let blend = st.slab_blend;
        for (k, id) in st.glass.into_iter().enumerate() {
            edit_material(frame, id, |m| {
                volumes[k].apply(m);
                m.alpha_mode = AlphaMode::Opaque;
                m.base_colour = m.base_colour.with_alpha(1.0);
                if k == GlassTarget::Slab as usize && blend {
                    // The same fraction of the background shows through.
                    let opacity = 1.0 - volumes[k].transmission;
                    m.transmission = 0.0;
                    m.alpha_mode = AlphaMode::Blend;
                    m.base_colour = Colour::from_linear_rgb_array(volumes[k].attenuation)
                        .with_alpha(opacity.max(0.05));
                }
            });
        }
    }
    ui.separator();
    if ui.button("Look through two panes").clicked() {
        app.camera = vpl::Camera::look_at(
            glam::DVec3::new(4.8, -6.0, 0.9),
            glam::DVec3::new(4.8, 3.0, 0.7),
            glam::Vec3::Z,
        );
    }
    ui.weak(
        "The far pane is behind the near one, so through the near pane it shows as the \
         environment: transmission reads the opaque scene only.",
    );
}

fn controls_layers(app: &mut App, ui: &mut egui::Ui, frame: &crate::eframe::Frame) {
    ui.label(
        "Front row: the layer. Back row: the same base with no layer. Turn the key \
         light to move the highlights.",
    );
    let st = &mut app.glass_state;
    ui.add(egui::Slider::new(&mut st.key_yaw, -3.1..=3.1).text("Key light yaw"));
    ui.separator();
    let (cc, ccr) = &mut st.clearcoat;
    let mut changed = [false; 3];
    ui.label("Clear coat over metallic flake paint:");
    changed[0] |= ui
        .add(egui::Slider::new(cc, 0.0..=1.0).text("clearcoat"))
        .changed();
    changed[0] |= ui
        .add(egui::Slider::new(ccr, 0.0..=1.0).text("clearcoat_roughness"))
        .changed();
    ui.label("Sheen (velvet):");
    let (sheen, sheen_rough) = &mut st.sheen;
    ui.horizontal(|ui| {
        ui.label("sheen_colour");
        changed[1] |= ui.color_edit_button_rgb(sheen).changed();
    });
    changed[1] |= ui
        .add(egui::Slider::new(sheen_rough, 0.0..=1.0).text("sheen_roughness"))
        .changed();
    ui.label("Anisotropy (brushed metal):");
    let (aniso, rot) = &mut st.anisotropy;
    changed[2] |= ui
        .add(egui::Slider::new(aniso, -1.0..=1.0).text("anisotropy"))
        .changed();
    changed[2] |= ui
        .add(egui::Slider::new(rot, 0.0..=std::f32::consts::PI).text("anisotropy_rotation"))
        .changed();
    let (cc, ccr) = st.clearcoat;
    let (sheen, sheen_rough) = st.sheen;
    let (aniso, rot) = st.anisotropy;
    if changed[0] {
        edit_material(frame, st.layered[0], |m| {
            m.clearcoat = cc;
            m.clearcoat_roughness = ccr;
        });
    }
    if changed[1] {
        edit_material(frame, st.layered[1], |m| {
            m.sheen_colour = Colour::from_linear_rgb_array(sheen);
            m.sheen_roughness = sheen_rough;
        });
    }
    if changed[2] {
        edit_material(frame, st.layered[2], |m| {
            m.anisotropy = aniso;
            m.anisotropy_rotation = rot;
        });
    }
}

fn controls_trace(app: &mut App, ui: &mut egui::Ui, frame: &crate::eframe::Frame) {
    #[cfg(feature = "showcase")]
    {
        ui.label("Path tracer:");
        ui.horizontal(|ui| {
            ui.label("Samples:");
            for n in [16, 64, 256, 1024] {
                ui.selectable_value(&mut app.glass_state.samples, n, n.to_string());
            }
        });
        if ui.button("Trace").clicked() {
            let aspect = app.camera.aspect;
            if let Some(rs) = frame.wgpu_render_state() {
                let guard = rs.renderer.read();
                if let Some(renderer) = guard.callback_resources.get::<ViewportRenderer>() {
                    trace(app, renderer, aspect);
                }
            }
        }
        let st = &mut app.glass_state;
        if let Some((rgba, size)) = &st.trace {
            let tex = st.trace_texture.get_or_insert_with(|| {
                ui.ctx().load_texture(
                    "glass_trace",
                    egui::ColorImage::from_rgba_unmultiplied(*size, rgba),
                    egui::TextureOptions::LINEAR,
                )
            });
            let w = 220.0;
            let h = w * size[1] as f32 / size[0] as f32;
            ui.image((tex.id(), egui::vec2(w, h)));
            ui.weak(format!(
                "{} samples at EV100 {:.1}, Khronos Neutral. The tracer reads no \
                 textures, so the wall traces in its base colour; it follows rays \
                 through every glass surface, so the liquid shows.",
                st.samples,
                physical_ev100(&app.camera.exposure)
            ));
        }
    }
    #[cfg(not(feature = "showcase"))]
    {
        let _ = (app, frame);
        ui.weak("Build with the `showcase` feature for the path tracer.");
    }
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScGlassAndLayers;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScGlassAndLayers = ScGlassAndLayers;

impl crate::Showcase for ScGlassAndLayers {
    fn needs_build(&self, app: &crate::App) -> bool {
        let st = &app.glass_state;
        !st.built[st.index()]
    }

    fn build(&self, app: &mut crate::App, renderer: &mut ViewportRenderer) {
        ensure_environment(app, renderer);
        let i = app.glass_state.index();
        app.glass_state.scenes[i] = Scene::new();
        app.glass_state.parts[i].clear();
        match app.glass_state.sub {
            GlassSub::Transmission => {
                build_transmission(app, renderer);
                app.camera = vpl::Camera::orbit(
                    glam::DVec3::new(1.0, 0.0, 0.9),
                    11.0,
                    glam::Quat::from_rotation_z(0.25) * glam::Quat::from_rotation_x(1.25),
                );
            }
            GlassSub::Layers => {
                build_layers(app, renderer);
                app.camera = vpl::Camera::orbit(
                    glam::DVec3::new(0.0, 0.0, 0.7),
                    10.0,
                    glam::Quat::from_rotation_z(-0.2) * glam::Quat::from_rotation_x(1.15),
                );
            }
        }
        app.glass_state.built[i] = true;
    }

    fn scene(
        &self,
        app: &mut crate::App,
        _frame: &crate::eframe::Frame,
        _out: &mut crate::SceneOverrides,
    ) -> crate::SceneContents {
        let st = &mut app.glass_state;
        let i = st.index();
        let items = st.scenes[i].collect_render_items(&vpl::Selection::new());
        crate::SceneContents {
            items,
            bg_colour: None,
            lighting: st.lighting(),
            // Distinct per scene, so switching tabs invalidates the batches.
            scene_gen: st.scenes[i].version() * 2 + i as u64,
            sel_gen: 0,
        }
    }

    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, _ctx: &crate::FrameCtx) {
        fd.effects.display.exposure =
            ExposureSettings::from_mode(app.camera.exposure.to_exposure_mode());
        if let Some(env) = app.glass_state.env {
            fd.effects.environment = Some(EnvironmentLighting::new(env));
        }
        let cam = &app.camera;
        let far = (cam.focus_distance() * 4.0).max(60.0);
        fd.camera.render_camera =
            vpl::RenderCamera::from_camera(cam).with_depth_range(cam.depth_range().0, far);
    }

    fn controls(&self, app: &mut crate::App, ui: &mut egui::Ui, frame: &crate::eframe::Frame) {
        controls(app, ui, frame)
    }

    fn tour(&self, app: &mut crate::App, step: u32) -> bool {
        let st = &mut app.glass_state;
        match step {
            0 => {}
            1 => {
                app.camera = vpl::Camera::look_at(
                    glam::DVec3::new(4.8, -6.0, 0.9),
                    glam::DVec3::new(4.8, 3.0, 0.7),
                    glam::Vec3::Z,
                );
            }
            2 => st.sub = GlassSub::Layers,
            3 => st.key_yaw = -1.2,
            _ => return false,
        }
        true
    }
}
