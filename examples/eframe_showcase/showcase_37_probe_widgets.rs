//! Showcase 37: Probe Widgets
//!
//! Demonstrates the three interactive 3D widgets from `viewport_lib::interaction::widgets`
//! applied to a point cloud:
//!
//! - **Line Probe**: drag endpoints to reposition; reports distance and point count near the segment.
//! - **Sphere Widget**: drag center/radius handles; select or deselect points inside the sphere.
//! - **Box Widget**: drag center/face handles; select or deselect points inside the box.
//!
//! Selected points are highlighted in orange. Orbit is suppressed while a widget handle is active.

use crate::App;
use eframe::egui;
use viewport_lib::{
    BoxWidget, LineProbeWidget, SphereWidget, WidgetContext, WidgetResult,
    scene::Scene, ViewportRenderer,
};

const CLOUD_N: usize = 20000;
const CLOUD_RANGE: f32 = 3.5;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PwSubMode {
    LineProbe,
    Sphere,
    Box,
}

pub(crate) struct ProbeWidgetState {
    pub sub_mode: PwSubMode,
    pub probe: LineProbeWidget,
    pub sphere: SphereWidget,
    pub bw: BoxWidget,
    pub last_result: WidgetResult,
    pub suppress_orbit: bool,
    pub cloud_positions: Vec<[f32; 3]>,
    pub selected: Vec<bool>,
    pub line_threshold: f32,
}

impl ProbeWidgetState {
    pub fn new() -> Self {
        let handle_color = [0.1, 1.0, 0.2, 1.0];

        let mut probe = LineProbeWidget::new(
            glam::Vec3::new(-2.0, 0.0, 0.0),
            glam::Vec3::new(2.0, 0.0, 0.0),
        );
        probe.line_width = 3.0;
        probe.color = [1.0, 0.9, 0.1, 1.0];
        probe.handle_color = handle_color;

        let mut sphere = SphereWidget::new(glam::Vec3::ZERO, 2.0);
        sphere.color = [1.0, 0.9, 0.1, 0.15];
        sphere.handle_color = handle_color;

        let mut bw = BoxWidget::new(glam::Vec3::ZERO, glam::Vec3::splat(2.0));
        bw.color = [1.0, 0.9, 0.1, 1.0];
        bw.handle_color = handle_color;

        let cloud_positions = generate_cloud(CLOUD_N);
        let selected = vec![false; CLOUD_N];

        Self {
            sub_mode: PwSubMode::Sphere,
            probe,
            sphere,
            bw,
            last_result: WidgetResult::None,
            suppress_orbit: false,
            cloud_positions,
            selected,
            line_threshold: 0.4,
        }
    }

    pub fn selected_count(&self) -> usize {
        self.selected.iter().filter(|&&s| s).count()
    }

    pub fn apply_sphere_selection(&mut self, add: bool) {
        let c = self.sphere.center;
        let r = self.sphere.radius;
        for (i, p) in self.cloud_positions.iter().enumerate() {
            if (glam::Vec3::from(*p) - c).length() <= r {
                self.selected[i] = add;
            }
        }
    }

    pub fn apply_box_selection(&mut self, add: bool) {
        let aabb = self.bw.aabb();
        for (i, p) in self.cloud_positions.iter().enumerate() {
            let p = glam::Vec3::from(*p);
            if p.x >= aabb.min.x && p.x <= aabb.max.x
                && p.y >= aabb.min.y && p.y <= aabb.max.y
                && p.z >= aabb.min.z && p.z <= aabb.max.z
            {
                self.selected[i] = add;
            }
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected.iter_mut().for_each(|s| *s = false);
    }

    /// Count points within `self.line_threshold` world units of the probe segment.
    pub fn points_near_line(&self) -> usize {
        self.near_line_iter().count()
    }

    pub fn apply_line_selection(&mut self, add: bool) {
        let indices: Vec<usize> = self.near_line_iter().collect();
        for i in indices {
            self.selected[i] = add;
        }
    }

    fn near_line_iter(&self) -> impl Iterator<Item = usize> + '_ {
        let start = self.probe.start;
        let end = self.probe.end;
        let dir = end - start;
        let len = dir.length();
        let dir_n = if len > 1e-6 { dir / len } else { glam::Vec3::X };
        let threshold = self.line_threshold;
        self.cloud_positions.iter().enumerate().filter_map(move |(i, p)| {
            let p = glam::Vec3::from(*p);
            let t = ((p - start).dot(dir_n)).clamp(0.0, len);
            if (p - (start + dir_n * t)).length() <= threshold { Some(i) } else { None }
        })
    }
}

/// Deterministic LCG point cloud spread across [-CLOUD_RANGE, CLOUD_RANGE]^3.
fn generate_cloud(n: usize) -> Vec<[f32; 3]> {
    let mut s: u32 = 0xdeadbeef;
    let mut next = move || -> f32 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as f32 / 16_777_216.0
    };
    (0..n)
        .map(|_| {
            [
                next() * CLOUD_RANGE * 2.0 - CLOUD_RANGE,
                next() * CLOUD_RANGE * 2.0 - CLOUD_RANGE,
                next() * CLOUD_RANGE * 2.0 - CLOUD_RANGE,
            ]
        })
        .collect()
}

impl App {
    pub(crate) fn build_probe_widgets_scene(&mut self, _renderer: &mut ViewportRenderer) {
        self.pw_scene = Scene::new();
        self.pw_state = ProbeWidgetState::new();
        self.pw_built = true;
    }

    pub(crate) fn update_probe_widgets(&mut self, ctx_widget: WidgetContext) {
        let state = &mut self.pw_state;
        let result = match state.sub_mode {
            PwSubMode::LineProbe => state.probe.update(&ctx_widget),
            PwSubMode::Sphere => state.sphere.update(&ctx_widget),
            PwSubMode::Box => state.bw.update(&ctx_widget),
        };
        state.last_result = result;
        state.suppress_orbit = match state.sub_mode {
            PwSubMode::LineProbe => state.probe.is_active(),
            PwSubMode::Sphere => state.sphere.is_active(),
            PwSubMode::Box => state.bw.is_active(),
        };
    }

    pub(crate) fn controls_probe_widgets(&mut self, ui: &mut egui::Ui) {
        let state = &mut self.pw_state;

        ui.label("Widget:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut state.sub_mode, PwSubMode::LineProbe, "Line Probe");
            ui.radio_value(&mut state.sub_mode, PwSubMode::Sphere, "Sphere");
            ui.radio_value(&mut state.sub_mode, PwSubMode::Box, "Box");
        });
        ui.separator();

        match state.sub_mode {
            PwSubMode::LineProbe => {
                let s = state.probe.start;
                let e = state.probe.end;
                ui.label(format!("Start:  [{:.2}, {:.2}, {:.2}]", s.x, s.y, s.z));
                ui.label(format!("End:    [{:.2}, {:.2}, {:.2}]", e.x, e.y, e.z));
                ui.label(format!("Length: {:.3}", (e - s).length()));
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(egui::Slider::new(&mut state.line_threshold, 0.05..=3.0).step_by(0.05));
                });
                ui.label(format!(
                    "Points within {:.2}: {}",
                    state.line_threshold,
                    state.points_near_line()
                ));
                ui.separator();
                ui.label(format!(
                    "Selected: {} / {}",
                    state.selected_count(),
                    state.cloud_positions.len()
                ));
                ui.horizontal(|ui| {
                    if ui.button("Select from region").clicked() {
                        state.apply_line_selection(true);
                    }
                    if ui.button("Deselect from region").clicked() {
                        state.apply_line_selection(false);
                    }
                });
                if ui.button("Clear selection").clicked() {
                    state.clear_selection();
                }
            }
            PwSubMode::Sphere => {
                let c = state.sphere.center;
                ui.label(format!("Center: [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z));
                ui.label(format!("Radius: {:.3}", state.sphere.radius));
                ui.separator();
                ui.label(format!(
                    "Selected: {} / {}",
                    state.selected_count(),
                    state.cloud_positions.len()
                ));
                ui.horizontal(|ui| {
                    if ui.button("Select from region").clicked() {
                        state.apply_sphere_selection(true);
                    }
                    if ui.button("Deselect from region").clicked() {
                        state.apply_sphere_selection(false);
                    }
                });
                if ui.button("Clear selection").clicked() {
                    state.clear_selection();
                }
            }
            PwSubMode::Box => {
                let c = state.bw.center;
                let h = state.bw.half_extents;
                ui.label(format!("Center:       [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z));
                ui.label(format!("Half-extents: [{:.2}, {:.2}, {:.2}]", h.x, h.y, h.z));
                ui.separator();
                ui.label(format!(
                    "Selected: {} / {}",
                    state.selected_count(),
                    state.cloud_positions.len()
                ));
                ui.horizontal(|ui| {
                    if ui.button("Select from region").clicked() {
                        state.apply_box_selection(true);
                    }
                    if ui.button("Deselect from region").clicked() {
                        state.apply_box_selection(false);
                    }
                });
                if ui.button("Clear selection").clicked() {
                    state.clear_selection();
                }
            }
        }

        ui.separator();
        if state.suppress_orbit {
            ui.label("(Orbit suppressed: widget active)");
        }
    }
}
