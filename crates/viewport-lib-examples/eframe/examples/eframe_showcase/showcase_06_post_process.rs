//! Showcase 6: View Transform and Post-Processing.
//!
//! How exposed radiance becomes display pixels, on a lit studio scene with
//! content above white: a 2 000 nit emissive strip, a chrome sphere, and a
//! readout strip along the front edge (a 20-stop emissive grey ramp and six
//! saturated hue cards at 4 nits) so the operator's shoulder, desaturation
//! and hue handling read directly.
//!
//! - **Operators** - the six named [`ToneMapping`] operators. `Agx` takes an
//!   [`AgxLook`]; `Generic` takes a [`GenericToneCurve`], plotted in the
//!   sidebar from the CPU port in `tone_operators`.
//! - **Grading** - a procedural teal-orange strip LUT (`TextureData::hdr`) on
//!   [`PostProcessSettings::grade_lut`] at a chosen strength, placed by
//!   [`GradingSpace`]. Under `Scene` the background keeps its colour; under
//!   `Display` it is graded with everything else. "Bake" folds the current
//!   operator, look and grade into one [`ToneMapping::Custom`] LUT with
//!   [`bake_view_transform_lut`] and [`OperatorCurve`], and a toggle swaps
//!   between the analytic and baked results.
//! - **Working space** - [`WorkingSpace::Rec709`] against
//!   [`WorkingSpace::Rec2020`]: per-channel operators compress saturated
//!   colours differently in the wider primaries.
//!
//! The post chain sits under every tab: bloom threshold and intensity, SSAO,
//! FXAA, eye-dome lighting, vignette and depth of field.

use crate::App;
use crate::eframe::egui;
use crate::geometry::make_uv_sphere;
use crate::shared::{two_sided, upload_material};
use viewport_lib as vpl;
use viewport_lib::Colour;
use vpl::vplt::effects::tone_operators;
use vpl::{
    AgxLook, GenericToneCurve, GradingSpace, Material, OperatorCurve, PostProcessSettings,
    TextureData, TextureId, ToneCurve, ToneMapping, VIEW_LUT_SHAPER, ViewportRenderer,
    WorkingSpace, bake_view_transform_lut, scene::Scene,
};

/// Slices of the grading LUT.
const GRADE_LUT_SIZE: u32 = 16;
/// Stops in the grey ramp, from 2^-12 nits upward.
const RAMP_STOPS: i32 = 20;
/// Saturated hue cards, linear Rec.709, shown at [`HUE_NITS`].
const HUES: [[f32; 3]; 6] = [
    [0.8, 0.1, 0.08],
    [0.85, 0.45, 0.08],
    [0.75, 0.75, 0.1],
    [0.12, 0.6, 0.15],
    [0.08, 0.5, 0.75],
    [0.55, 0.15, 0.7],
];
const HUE_NITS: f32 = 4.0;
const STRIP_NITS: f32 = 2_000.0;

/// The named operators, in the order the radio lists them.
const OPERATORS: [(ToneMapping, &str); 6] = [
    (ToneMapping::Reinhard, "Reinhard"),
    (ToneMapping::Aces, "ACES (fit)"),
    (ToneMapping::KhronosNeutral, "Khronos Neutral"),
    (ToneMapping::Agx, "AgX"),
    (ToneMapping::AcesRrt, "ACES RRT"),
    (ToneMapping::Generic, "Generic"),
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PpSub {
    Operators,
    Grading,
    WorkingSpace,
}

/// A view-transform LUT baked from the settings below.
pub(crate) struct BakedLut {
    id: TextureId,
    slices: u32,
    operator: ToneMapping,
    working_space: WorkingSpace,
    grading_space: GradingSpace,
}

pub(crate) struct PostProcessState {
    pub built: bool,
    pub scene: Scene,
    pub sub: PpSub,
    pub point_light_on: bool,
    pub dir_intensity: f32,
    // View transform.
    pub operator: ToneMapping,
    pub agx_look: AgxLook,
    pub generic: GenericToneCurve,
    pub working_space: WorkingSpace,
    // Grading.
    pub grade_on: bool,
    pub grade_strength: f32,
    pub grading_space: GradingSpace,
    pub background: [u8; 3],
    grade_lut: Option<(TextureId, f32)>,
    pub bake_slices: u32,
    baked: Option<BakedLut>,
    pub show_baked: bool,
    // Post chain.
    pub bloom: bool,
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
    pub ssao: bool,
    pub fxaa: bool,
    pub edl: bool,
    pub vignette: bool,
    pub vignette_amount: f32,
    pub dof_enabled: bool,
    pub dof_focal_dist: f32,
    pub dof_focal_range: f32,
    pub dof_max_blur: f32,
}

impl Default for PostProcessState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            sub: PpSub::Operators,
            point_light_on: true,
            dir_intensity: 0.4,
            operator: ToneMapping::KhronosNeutral,
            agx_look: AgxLook::None,
            generic: GenericToneCurve::default(),
            working_space: WorkingSpace::Rec709,
            grade_on: true,
            grade_strength: 1.0,
            grading_space: GradingSpace::Scene,
            background: [56, 72, 110],
            grade_lut: None,
            bake_slices: 32,
            baked: None,
            show_baked: false,
            bloom: true,
            bloom_threshold: 1.0,
            bloom_intensity: 0.1,
            ssao: false,
            fxaa: true,
            edl: false,
            vignette: false,
            vignette_amount: 0.5,
            dof_enabled: false,
            dof_focal_dist: 5.0,
            dof_focal_range: 1.0,
            dof_max_blur: 8.0,
        }
    }
}

impl PostProcessState {
    /// The post settings the analytic view transform reads.
    fn post_settings(&self) -> PostProcessSettings {
        let mut p = PostProcessSettings::default();
        p.agx_look = self.agx_look;
        p.generic_tone_curve = self.generic;
        p.working_space = self.working_space;
        p.grading_space = self.grading_space;
        if self.grade_on {
            p.grade_lut = self.grade_lut.map(|(id, _)| id);
        }
        p.bloom.enabled = self.bloom;
        p.bloom.threshold = self.bloom_threshold;
        p.bloom.intensity = self.bloom_intensity;
        p.ssao = self.ssao;
        p.fxaa = self.fxaa;
        p.edl.enabled = self.edl;
        p.vignette.enabled = self.vignette;
        p.vignette.amount = self.vignette_amount;
        p.dof.enabled = self.dof_enabled;
        p.dof.focal_distance = self.dof_focal_dist;
        p.dof.focal_range = self.dof_focal_range;
        p.dof.max_blur_radius = self.dof_max_blur;
        p
    }
}

// ---------------------------------------------------------------------------
// Grade
// ---------------------------------------------------------------------------

/// A teal-orange grade on shaped values: values a stop or more below
/// mid-grey lean teal, values above it lean orange, and saturation is raised
/// around the shaped luma.
fn teal_orange(s: [f32; 3]) -> [f32; 3] {
    let luma = 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2];
    // Mid-grey (0.18) shapes to about 0.48; 0.1 is 2.4 stops.
    let tone = ((luma - 0.48) / 0.1).clamp(-1.0, 1.0);
    let sat = |c: f32| luma + (c - luma) * 1.3;
    [
        sat(s[0]) + 0.04 * tone,
        sat(s[1]) + 0.01 * tone,
        sat(s[2]) - 0.05 * tone,
    ]
}

/// [`teal_orange`] mixed with the identity at `strength`.
fn graded(s: [f32; 3], strength: f32) -> [f32; 3] {
    let g = teal_orange(s);
    [0, 1, 2].map(|i| s[i] + (g[i] - s[i]) * strength)
}

/// A strip LUT of `n` slices of `n` x `n` (width `n * n`, height `n`), as
/// float RGBA. `f` maps a lattice coordinate in `[0, 1]` to the value stored
/// there.
fn strip_lut(n: u32, f: impl Fn([f32; 3]) -> [f32; 3]) -> Vec<f32> {
    let unit = |v: u32| v as f32 / (n - 1) as f32;
    let mut out = Vec::with_capacity((n * n * n * 4) as usize);
    for g in 0..n {
        for b in 0..n {
            for r in 0..n {
                let c = f([unit(r), unit(g), unit(b)]);
                out.extend_from_slice(&[c[0], c[1], c[2], 1.0]);
            }
        }
    }
    out
}

/// Upload (or re-upload) the grade LUT when its strength moved.
fn sync_grade_lut(app: &mut App, renderer: &mut ViewportRenderer) {
    let strength = app.pp_state.grade_strength;
    if app
        .pp_state
        .grade_lut
        .is_some_and(|(_, s)| (s - strength).abs() < 1e-4)
    {
        return;
    }
    let texels = strip_lut(GRADE_LUT_SIZE, |c| graded(c, strength));
    let resources = renderer.resources_mut();
    if let Some((old, _)) = app.pp_state.grade_lut.take() {
        resources.free_texture(old);
    }
    let id = resources
        .upload_texture(
            &app.device,
            &app.queue,
            TextureData::hdr(GRADE_LUT_SIZE * GRADE_LUT_SIZE, GRADE_LUT_SIZE, texels),
        )
        .expect("grade LUT upload");
    app.pp_state.grade_lut = Some((id, strength));
}

/// Bake the current operator, look and grade into one view-transform LUT.
fn bake(app: &mut App, renderer: &mut ViewportRenderer) {
    let st = &app.pp_state;
    let post = st.post_settings();
    let curve = OperatorCurve::new(st.operator, &post);
    let strength = if st.grade_on { st.grade_strength } else { 0.0 };
    // A scene-referred grade runs on shaped radiance before the curve.
    let scene_grade = move |c: [f32; 3]| {
        let s = graded(c.map(|x| VIEW_LUT_SHAPER.shape(x)), strength);
        s.map(|v| VIEW_LUT_SHAPER.unshape(v))
    };
    // A display-referred grade runs on the curve's display values.
    let display_curve = |c: [f32; 3]| graded(curve.map(c), strength);
    let lut = match st.grading_space {
        GradingSpace::Display => bake_view_transform_lut(&display_curve, None, st.bake_slices),
        _ => bake_view_transform_lut(&curve, Some(&scene_grade), st.bake_slices),
    };
    let baked = BakedLut {
        id: TextureId::INVALID,
        slices: st.bake_slices,
        operator: st.operator,
        working_space: st.working_space,
        grading_space: st.grading_space,
    };
    let resources = renderer.resources_mut();
    if let Some(old) = app.pp_state.baked.take() {
        resources.free_texture(old.id);
    }
    let id = resources
        .upload_texture(&app.device, &app.queue, lut)
        .expect("view LUT upload");
    app.pp_state.baked = Some(BakedLut { id, ..baked });
    app.pp_state.show_baked = true;
}

/// Run `f` with the renderer from the eframe callback resources.
fn with_renderer(frame: &crate::eframe::Frame, f: impl FnOnce(&mut ViewportRenderer)) {
    let Some(rs) = frame.wgpu_render_state() else {
        return;
    };
    let mut guard = rs.renderer.write();
    if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
        f(renderer);
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// A black surface emitting `rgb * nits`: exposed and tone mapped like any
/// light, with no lit term of its own.
fn emitter(rgb: [f32; 3], nits: f32) -> Material {
    let mut m = Material::pbr(Colour::linear_rgb(0.0, 0.0, 0.0), 0.0, 1.0);
    m.specular_factor = 0.0;
    m.emissive = Colour::from_linear_rgb_array(rgb);
    m.emissive_strength = nits;
    m
}

impl App {
    /// Build the studio scene for Showcase 6.
    pub(crate) fn build_pp_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.pp_state.scene = Scene::new();
        let scene = &mut self.pp_state.scene;

        let cube = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &self.box_mesh_data)
            .expect("box mesh upload");
        let add_box = |scene: &mut Scene,
                       renderer: &mut ViewportRenderer,
                       name: &str,
                       at: glam::Mat4,
                       m: Material| {
            let mat = upload_material(renderer, m);
            scene.add_named(name, Some(cube), at, mat);
        };
        add_box(
            scene,
            renderer,
            "Ground",
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(10.0, 10.0, 0.15),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, -0.575),
            ),
            Material::pbr(Colour::linear_rgb(0.8, 0.8, 0.8), 0.0, 0.9),
        );
        for (name, at, colour, metallic, roughness) in [
            ("Gold", [-1.2, -1.2], [1.0, 0.72, 0.06], 0.95, 0.05),
            ("Brushed steel", [1.2, -1.2], [0.82, 0.82, 0.86], 0.75, 0.35),
            ("Ceramic", [1.2, 1.2], [0.82, 0.58, 0.42], 0.0, 0.9),
        ] {
            add_box(
                scene,
                renderer,
                name,
                glam::Mat4::from_translation(glam::Vec3::new(at[0], at[1], 0.0)),
                Material::pbr(Colour::from_linear_rgb_array(colour), metallic, roughness),
            );
        }
        add_box(
            scene,
            renderer,
            "Pillar",
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.5, 0.5, 2.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, 1.0),
            ),
            Material::pbr(Colour::linear_rgb(0.4, 0.4, 0.45), 0.1, 0.7),
        );
        // The emissive strip: a 2 000 nit bar standing behind the boxes.
        add_box(
            scene,
            renderer,
            "Emissive strip (2 000 nits)",
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(3.0, 0.12, 0.12),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 2.8, 1.6),
            ),
            emitter([1.0, 0.92, 0.8], STRIP_NITS),
        );

        let sphere = make_uv_sphere(48, 24, 0.6);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("pp sphere upload");
        for (name, at, m) in [
            (
                "Chrome sphere",
                [-1.2, 1.2],
                Material::pbr(Colour::linear_rgb(0.95, 0.95, 0.97), 1.0, 0.03),
            ),
            (
                "Orange sphere",
                [3.0, 0.0],
                Material::pbr(Colour::linear_rgb(0.75, 0.28, 0.05), 0.0, 0.55),
            ),
        ] {
            let mat = upload_material(renderer, m);
            scene.add_named(
                name,
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(at[0], at[1], 0.1)),
                mat,
            );
        }

        // The readout strip along the front edge: one emissive card per stop
        // of the grey ramp, then the hue cards.
        let card = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &vpl::primitives::plane(1.0, 1.0))
            .expect("card upload");
        let ramp_step = 8.4 / RAMP_STOPS as f32;
        for i in 0..RAMP_STOPS {
            let nits = ((i - 12) as f32).exp2();
            let x = (i as f32 - (RAMP_STOPS as f32 - 1.0) / 2.0) * ramp_step;
            let mat = upload_material(renderer, two_sided(emitter([1.0; 3], nits)));
            scene.add_named(
                &format!("Ramp 2^{} nits", i - 12),
                Some(card),
                glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::new(ramp_step * 0.9, 0.6, 1.0),
                    glam::Quat::IDENTITY,
                    glam::Vec3::new(x, -4.1, -0.49),
                ),
                mat,
            );
        }
        for (i, hue) in HUES.iter().enumerate() {
            let x = (i as f32 - 2.5) * 1.3;
            let mat = upload_material(renderer, two_sided(emitter(*hue, HUE_NITS)));
            scene.add_named(
                &format!("Hue card {i}"),
                Some(card),
                glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::new(1.15, 0.6, 1.0),
                    glam::Quat::IDENTITY,
                    glam::Vec3::new(x, -3.35, -0.49),
                ),
                mat,
            );
        }

        sync_grade_lut(self, renderer);
        self.pp_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

/// Plot the active operator's grey response from the CPU port, with the
/// input in stops around mid-grey.
fn plot_curve(ui: &mut egui::Ui, operator: ToneMapping, post: &PostProcessSettings) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(200.0, 120.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(24));
    let (lo, hi) = (-8.0f32, 8.0f32);
    let to_screen = |stops: f32, y: f32| {
        egui::pos2(
            rect.left() + (stops - lo) / (hi - lo) * rect.width(),
            rect.bottom() - y.clamp(0.0, 1.05) / 1.05 * rect.height(),
        )
    };
    let grid = egui::Stroke::new(1.0, egui::Color32::from_gray(50));
    painter.line_segment([to_screen(0.0, 0.0), to_screen(0.0, 1.05)], grid);
    painter.line_segment([to_screen(lo, 1.0), to_screen(hi, 1.0)], grid);
    let points: Vec<egui::Pos2> = (0..=96)
        .map(|i| {
            let stops = lo + (hi - lo) * i as f32 / 96.0;
            let x = 0.18 * stops.exp2();
            let y = tone_operators::tone_map([x; 3], operator, post)[1];
            to_screen(stops, y)
        })
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(1.5, egui::Color32::from_rgb(240, 200, 90)),
    ));
    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        egui::Align2::LEFT_TOP,
        "display (linear) vs stops from 0.18",
        egui::FontId::proportional(10.0),
        egui::Color32::from_gray(150),
    );
}

pub(crate) fn controls_post_process(
    app: &mut App,
    ui: &mut egui::Ui,
    frame: &crate::eframe::Frame,
) {
    ui.horizontal(|ui| {
        let st = &mut app.pp_state;
        ui.selectable_value(&mut st.sub, PpSub::Operators, "Operators");
        ui.selectable_value(&mut st.sub, PpSub::Grading, "Grading");
        ui.selectable_value(&mut st.sub, PpSub::WorkingSpace, "Working space");
    });
    ui.separator();

    match app.pp_state.sub {
        PpSub::Operators => controls_operators(app, ui),
        PpSub::Grading => controls_grading(app, ui, frame),
        PpSub::WorkingSpace => controls_working_space(app, ui),
    }

    ui.separator();
    controls_post_chain(app, ui);
}

fn operator_radio(ui: &mut egui::Ui, st: &mut PostProcessState) {
    for (op, label) in OPERATORS {
        ui.radio_value(&mut st.operator, op, label);
    }
}

fn controls_operators(app: &mut App, ui: &mut egui::Ui) {
    let st = &mut app.pp_state;
    ui.label(
        "The front strip is a 20-stop emissive grey ramp (2^-12 to 2^7 nits) and six hue \
         cards at 4 nits; the bar behind the boxes is 2 000 nits. Each operator rolls the \
         ramp off to white along its own shoulder.",
    );
    ui.separator();
    operator_radio(ui, st);
    match st.operator {
        ToneMapping::Agx => {
            ui.horizontal(|ui| {
                ui.label("Look:");
                ui.selectable_value(&mut st.agx_look, AgxLook::None, "None");
                ui.selectable_value(&mut st.agx_look, AgxLook::Punchy, "Punchy");
                ui.selectable_value(&mut st.agx_look, AgxLook::Golden, "Golden");
            });
        }
        ToneMapping::Generic => {
            let g = &mut st.generic;
            ui.add(egui::Slider::new(&mut g.contrast, 0.5..=3.0).text("Contrast"));
            ui.add(egui::Slider::new(&mut g.mid_grey_in, 0.05..=0.5).text("Mid-grey in"));
            ui.add(egui::Slider::new(&mut g.mid_grey_out, 0.05..=0.5).text("Mid-grey out"));
            ui.add(
                egui::Slider::new(&mut g.hdr_max, 1.0..=64.0)
                    .logarithmic(true)
                    .text("White point (HDR max)"),
            );
        }
        _ => {}
    }
    let post = st.post_settings();
    plot_curve(ui, st.operator, &post);
}

fn controls_grading(app: &mut App, ui: &mut egui::Ui, frame: &crate::eframe::Frame) {
    let do_bake;
    {
        let st = &mut app.pp_state;
        ui.label(
            "A teal-orange strip LUT. Under Scene it grades exposed radiance before the \
             operator and leaves the background alone; under Display it grades the final \
             image, background included.",
        );
        ui.separator();
        ui.checkbox(&mut st.grade_on, "Grade");
        ui.add_enabled(
            st.grade_on,
            egui::Slider::new(&mut st.grade_strength, 0.0..=1.5).text("Strength"),
        );
        ui.horizontal(|ui| {
            ui.label("Grading space:");
            ui.selectable_value(&mut st.grading_space, GradingSpace::Scene, "Scene");
            ui.selectable_value(&mut st.grading_space, GradingSpace::Display, "Display");
        });
        ui.horizontal(|ui| {
            ui.label("Background:");
            ui.color_edit_button_srgb(&mut st.background);
        });
        ui.separator();
        ui.label("Operator:");
        operator_radio(ui, st);
        ui.separator();
        ui.label("Bake into one LUT (ToneMapping::Custom):");
        ui.horizontal(|ui| {
            ui.label("Slices:");
            for n in [16, 32, 64] {
                ui.selectable_value(&mut st.bake_slices, n, n.to_string());
            }
        });
        do_bake = ui.button("Bake operator + look + grade").clicked();
    }
    if do_bake {
        with_renderer(frame, |r| bake(app, r));
    }
    let st = &mut app.pp_state;
    match &st.baked {
        Some(b) => {
            ui.horizontal(|ui| {
                ui.radio_value(&mut st.show_baked, false, "Analytic");
                ui.radio_value(&mut st.show_baked, true, "Baked");
            });
            ui.label(format!(
                "Baked: {:?}, {} slices ({} x {} texels), {:?} grade.",
                b.operator,
                b.slices,
                b.slices * b.slices,
                b.slices,
                b.grading_space
            ));
            if b.working_space != st.working_space {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 170, 80),
                    "The LUT was baked for another working space; bake again.",
                );
            }
            if b.grading_space == GradingSpace::Display {
                ui.weak("A display grade baked into the operator does not reach the background.");
            }
        }
        None => {
            ui.weak("Nothing baked yet.");
        }
    }
}

fn controls_working_space(app: &mut App, ui: &mut egui::Ui) {
    let st = &mut app.pp_state;
    ui.label(
        "The primaries the scene grade and the operator work in. Per-channel operators \
         (Reinhard, ACES fit, Generic) compress the saturated hue cards differently in \
         Rec.2020; AgX and ACES RRT use their own spaces under either setting.",
    );
    ui.separator();
    ui.horizontal(|ui| {
        ui.selectable_value(&mut st.working_space, WorkingSpace::Rec709, "Rec.709");
        ui.selectable_value(&mut st.working_space, WorkingSpace::Rec2020, "Rec.2020");
    });
    ui.separator();
    ui.label("Operator:");
    operator_radio(ui, st);
}

fn controls_post_chain(app: &mut App, ui: &mut egui::Ui) {
    let st = &mut app.pp_state;
    ui.label("Lighting:");
    ui.add(egui::Slider::new(&mut st.dir_intensity, 0.0..=5.0).text("Sun (lux)"));
    ui.checkbox(&mut st.point_light_on, "Point light");
    ui.separator();
    ui.label("Post chain:");
    ui.checkbox(&mut st.bloom, "Bloom");
    ui.add_enabled_ui(st.bloom, |ui| {
        ui.add(
            egui::Slider::new(&mut st.bloom_threshold, 0.1..=8.0)
                .logarithmic(true)
                .text("Threshold"),
        );
        ui.add(egui::Slider::new(&mut st.bloom_intensity, 0.0..=1.0).text("Intensity"));
    });
    ui.checkbox(&mut st.ssao, "SSAO");
    ui.checkbox(&mut st.fxaa, "FXAA");
    ui.checkbox(&mut st.edl, "Eye-dome lighting");
    ui.checkbox(&mut st.vignette, "Vignette");
    ui.add_enabled(
        st.vignette,
        egui::Slider::new(&mut st.vignette_amount, 0.0..=1.0).text("Amount"),
    );
    ui.checkbox(&mut st.dof_enabled, "Depth of field");
    if st.dof_enabled {
        ui.add(egui::Slider::new(&mut st.dof_focal_dist, 0.5..=30.0).text("Focal distance"));
        ui.add(egui::Slider::new(&mut st.dof_focal_range, 0.1..=10.0).text("Focal range"));
        ui.add(egui::Slider::new(&mut st.dof_max_blur, 1.0..=20.0).text("Max blur (px)"));
    }
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

fn lighting(st: &PostProcessState) -> vpl::LightingSettings {
    let mut sun = vpl::LightSource::default();
    sun.kind = vpl::LightKind::Directional {
        direction: [0.6, 0.4, 1.0],
    };
    sun.intensity = st.dir_intensity;
    let mut lights = vec![sun];
    if st.point_light_on {
        let mut p = vpl::LightSource::default();
        p.kind = vpl::LightKind::Point {
            position: [3.0, 3.0, 3.0].into(),
            range: 15.0,
            radius: 0.1,
        };
        p.colour = Colour::linear_rgb(1.0, 0.9, 0.7);
        p.intensity = 20.0;
        // Warm fill only: one key caster keeps a single clean shadow.
        p.cast_shadows = false;
        lights.push(p);
    }
    let mut l = vpl::LightingSettings::default();
    l.lights = lights;
    l.shadows.enabled = true;
    l.shadows.filter = vpl::ShadowFilter::Pcss;
    l.hemisphere_intensity = 0.4;
    l.sky_colour = Colour::linear_rgb(1.0, 1.0, 1.0);
    l.ground_colour = Colour::linear_rgb(1.0, 1.0, 1.0);
    l
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScPostProcess;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScPostProcess = ScPostProcess;

impl crate::Showcase for ScPostProcess {
    fn needs_build(&self, app: &crate::App) -> bool {
        !app.pp_state.built
    }

    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        app.build_pp_scene(renderer);
        app.camera = vpl::Camera::orbit(
            glam::DVec3::new(0.0, -1.4, 0.0),
            14.0,
            glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1),
        );
    }

    fn scene(
        &self,
        app: &mut crate::App,
        _frame: &crate::eframe::Frame,
        _out: &mut crate::SceneOverrides,
    ) -> crate::SceneContents {
        let items = app
            .pp_state
            .scene
            .collect_render_items(&vpl::Selection::new());
        let bg = Colour::rgb(
            app.pp_state.background[0],
            app.pp_state.background[1],
            app.pp_state.background[2],
        )
        .to_linear_rgba();
        crate::SceneContents {
            items,
            bg_colour: Some(bg),
            lighting: lighting(&app.pp_state),
            scene_gen: app.pp_state.scene.version(),
            sel_gen: 0,
        }
    }

    fn flush_gpu(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        with_renderer(cx.frame, |r| sync_grade_lut(app, r));
    }

    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, _ctx: &crate::FrameCtx) {
        // Cap far plane for better cascade distribution, but track orbit
        // distance so the scene doesn't disappear when zooming out.
        let cam = &app.camera;
        let far = (cam.focus_distance() * 3.0).max(60.0);
        fd.camera.render_camera =
            vpl::RenderCamera::from_camera(cam).with_depth_range(cam.depth_range().0, far);

        let st = &app.pp_state;
        let mut post = st.post_settings();
        fd.effects.display.operator = st.operator;
        if let (true, Some(b)) = (st.show_baked && st.sub == PpSub::Grading, &st.baked) {
            // The baked LUT holds the operator and the grade: no grade LUT.
            fd.effects.display.operator = ToneMapping::Custom { lut: b.id };
            post.grade_lut = None;
        }
        if st.sub == PpSub::Operators {
            post.grade_lut = None;
        }
        fd.effects.post_process = post;
    }

    fn controls(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        frame: &crate::eframe::Frame,
    ) {
        controls_post_process(app, ui, frame)
    }

    fn tour(&self, app: &mut crate::App, step: u32) -> bool {
        let st = &mut app.pp_state;
        match step {
            0 => {}
            1 => st.operator = ToneMapping::Agx,
            2 => st.operator = ToneMapping::Generic,
            3 => {
                st.sub = PpSub::Grading;
                st.operator = ToneMapping::KhronosNeutral;
            }
            4 => st.grading_space = GradingSpace::Display,
            5 => {
                st.sub = PpSub::WorkingSpace;
                st.grading_space = GradingSpace::Scene;
                st.operator = ToneMapping::Reinhard;
                st.working_space = WorkingSpace::Rec2020;
            }
            _ => return false,
        }
        true
    }
}
