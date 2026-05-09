//! Showcase 37: Probe Widgets
//!
//! Demonstrates the three interactive 3D widgets from `viewport_lib::interaction::widgets`:
//!
//! - **Line Probe**: drag either endpoint to reposition the probe path.
//! - **Sphere Widget**: drag the center handle to move, drag the radius handle to resize.
//! - **Box Widget**: drag the center handle to move, drag any of the six face handles to resize.
//!
//! Suppress orbit while a widget handle is active (same pattern as ManipulationController).

use crate::App;
use eframe::egui;
use viewport_lib::{
    BoxWidget, LineProbeWidget, SphereWidget, WidgetContext, WidgetResult,
    scene::Scene,
    Material, ViewportRenderer,
};

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
}

impl ProbeWidgetState {
    pub fn new() -> Self {
        let mut probe = LineProbeWidget::new(
            glam::Vec3::new(-2.0, 0.0, 0.0),
            glam::Vec3::new(2.0, 0.0, 0.0),
        );
        probe.line_width = 3.0;

        Self {
            sub_mode: PwSubMode::LineProbe,
            probe,
            sphere: SphereWidget::new(glam::Vec3::ZERO, 1.5),
            bw: BoxWidget::new(glam::Vec3::ZERO, glam::Vec3::splat(1.5)),
            last_result: WidgetResult::None,
            suppress_orbit: false,
        }
    }
}

impl App {
    pub(crate) fn build_probe_widgets_scene(&mut self, renderer: &mut ViewportRenderer) {
        // Upload a small reference mesh so the scene is not empty.
        let mesh = self.upload_box(renderer);
        let mut scene = Scene::new();
        scene.add_named(
            "Reference",
            Some(mesh),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(0.3),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, -2.0),
            ),
            {
                let mut m = Material::from_color([0.5, 0.5, 0.55]);
                m.roughness = 0.8;
                m
            },
        );
        self.pw_scene = scene;
        self.pw_state = ProbeWidgetState::new();
        self.pw_built = true;
    }

    /// Update widget state from current input. Called inside the viewport response block.
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
                let len = (e - s).length();
                ui.label(format!("Length: {:.3}", len));
                if let Some(ep) = state.probe.hovered_endpoint() {
                    ui.label(format!("Hover: endpoint {}", ep));
                }
            }
            PwSubMode::Sphere => {
                let c = state.sphere.center;
                ui.label(format!("Center: [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z));
                ui.label(format!("Radius: {:.3}", state.sphere.radius));
            }
            PwSubMode::Box => {
                let c = state.bw.center;
                let h = state.bw.half_extents;
                ui.label(format!("Center:       [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z));
                ui.label(format!("Half-extents: [{:.2}, {:.2}, {:.2}]", h.x, h.y, h.z));
                let size = h * 2.0;
                ui.label(format!("Size:         [{:.2}, {:.2}, {:.2}]", size.x, size.y, size.z));
            }
        }

        ui.separator();
        ui.label("Drag a sphere handle to move it.");
        if state.suppress_orbit {
            ui.label("(Orbit suppressed: widget active)");
        }
    }
}
