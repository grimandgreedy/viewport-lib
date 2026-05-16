//! Showcase 37: Probe Widgets
//!
//! Demonstrates all interactive 3D widgets from `viewport_lib::interaction::widgets`
//! applied to a point cloud:
//!
//! - **Line Probe**: drag endpoints; select points near the segment.
//! - **Sphere Widget**: drag center/radius handles; select points inside.
//! - **Box Widget**: drag center/face/rotation handles; select points inside the OBB.
//! - **Plane Widget**: drag center/normal handles; select points on the positive side.
//! - **Disk Widget**: drag center/normal/radius handles; select points inside the disk region.
//! - **Cylinder Widget**: drag endpoints/radius; select points inside the cylinder.
//! - **Polyline Widget**: N draggable waypoints; select points near the path (double-click to add/remove).
//!
//! Selected points are highlighted in orange. Orbit is suppressed while a widget handle is active.

use crate::App;
use eframe::egui;
use viewport_lib::{
    BoxWidget, CameraFrame, CylinderWidget, DiskWidget, FrameData, LightingSettings,
    LineProbeWidget, PlaneWidget, PointCloudItem, PolylineWidget, SceneRenderItem, SphereWidget,
    ViewportRenderer, WidgetContext, WidgetResult, scene::Scene,
};

const CLOUD_N: usize = 20000;
const CLOUD_RANGE: f32 = 3.5;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PwSubMode {
    LineProbe,
    Sphere,
    Box,
    Plane,
    Disk,
    Cylinder,
    Polyline,
}

pub(crate) struct ProbeWidgetState {
    pub built: bool,
    pub scene: Scene,
    pub sub_mode: PwSubMode,
    // Phase 2 widgets
    pub probe: LineProbeWidget,
    pub sphere: SphereWidget,
    pub bw: BoxWidget,
    // Phase 11 widgets
    pub plane: PlaneWidget,
    pub disk: DiskWidget,
    pub cylinder: CylinderWidget,
    pub polyline: PolylineWidget,
    // Common state
    pub last_result: WidgetResult,
    pub suppress_orbit: bool,
    pub cloud_positions: Vec<[f32; 3]>,
    pub selected: Vec<bool>,
    pub line_threshold: f32,
    pub polyline_threshold: f32,
    pub disk_half_thickness: f32,
}

impl ProbeWidgetState {
    pub fn new() -> Self {
        let handle_colour = [0.1, 1.0, 0.2, 1.0];

        let mut probe = LineProbeWidget::new(
            glam::Vec3::new(-2.0, 0.0, 0.0),
            glam::Vec3::new(2.0, 0.0, 0.0),
        );
        probe.line_width = 3.0;
        probe.colour = [1.0, 0.9, 0.1, 1.0];
        probe.handle_colour = handle_colour;

        let mut sphere = SphereWidget::new(glam::Vec3::ZERO, 2.0);
        sphere.colour = [1.0, 0.9, 0.1, 0.15];
        sphere.handle_colour = handle_colour;

        let mut bw = BoxWidget::new(glam::Vec3::ZERO, glam::Vec3::splat(2.0));
        bw.colour = [1.0, 0.9, 0.1, 1.0];
        bw.handle_colour = handle_colour;

        let mut plane = PlaneWidget::new(glam::Vec3::ZERO, glam::Vec3::Z);
        plane.colour = [0.4, 0.8, 1.0, 1.0];
        plane.handle_colour = handle_colour;

        let mut disk = DiskWidget::new(glam::Vec3::ZERO, glam::Vec3::Z, 2.0);
        disk.colour = [1.0, 0.7, 0.2, 1.0];
        disk.handle_colour = handle_colour;

        let mut cylinder = CylinderWidget::new(
            glam::Vec3::new(0.0, -2.0, 0.0),
            glam::Vec3::new(0.0, 2.0, 0.0),
            1.5,
        );
        cylinder.colour = [0.5, 1.0, 0.5, 1.0];
        cylinder.handle_colour = handle_colour;

        let mut polyline = PolylineWidget::new(vec![
            glam::Vec3::new(-2.0, 0.0, 0.5),
            glam::Vec3::new(-0.5, 1.5, 0.5),
            glam::Vec3::new(0.5, -1.5, 0.5),
            glam::Vec3::new(2.0, 0.0, 0.5),
        ]);
        polyline.colour = [1.0, 0.5, 0.2, 1.0];
        polyline.handle_colour = handle_colour;

        let cloud_positions = generate_cloud(CLOUD_N);
        let selected = vec![false; CLOUD_N];

        Self {
            built: false,
            scene: Scene::new(),
            sub_mode: PwSubMode::Sphere,
            probe,
            sphere,
            bw,
            plane,
            disk,
            cylinder,
            polyline,
            last_result: WidgetResult::None,
            suppress_orbit: false,
            cloud_positions,
            selected,
            line_threshold: 0.4,
            polyline_threshold: 0.5,
            disk_half_thickness: 0.5,
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
        for (i, p) in self.cloud_positions.iter().enumerate() {
            if self.bw.contains_point(glam::Vec3::from(*p)) {
                self.selected[i] = add;
            }
        }
    }

    pub fn apply_plane_selection(&mut self, add: bool) {
        // Select points on the positive-normal side of the plane.
        let c = self.plane.center;
        let n = self.plane.normal;
        for (i, p) in self.cloud_positions.iter().enumerate() {
            if (glam::Vec3::from(*p) - c).dot(n) >= 0.0 {
                self.selected[i] = add;
            }
        }
    }

    pub fn apply_disk_selection(&mut self, add: bool) {
        let c = self.disk.center;
        let n = self.disk.normal;
        let r = self.disk.radius;
        let ht = self.disk_half_thickness;
        for (i, p) in self.cloud_positions.iter().enumerate() {
            let dp = glam::Vec3::from(*p) - c;
            let signed_dist = dp.dot(n);
            if signed_dist.abs() <= ht {
                let in_plane = dp - n * signed_dist;
                if in_plane.length() <= r {
                    self.selected[i] = add;
                }
            }
        }
    }

    pub fn apply_cylinder_selection(&mut self, add: bool) {
        let a = self.cylinder.start;
        let b = self.cylinder.end;
        let axis = b - a;
        let axis_len = axis.length();
        if axis_len < 1e-6 {
            return;
        }
        let axis_dir = axis / axis_len;
        let r = self.cylinder.radius;
        for (i, p) in self.cloud_positions.iter().enumerate() {
            let dp = glam::Vec3::from(*p) - a;
            let t = dp.dot(axis_dir);
            if t >= 0.0 && t <= axis_len {
                let radial = dp - axis_dir * t;
                if radial.length() <= r {
                    self.selected[i] = add;
                }
            }
        }
    }

    pub fn apply_polyline_selection(&mut self, add: bool) {
        let indices: Vec<usize> = self.near_polyline_iter().collect();
        for i in indices {
            self.selected[i] = add;
        }
    }

    fn near_polyline_iter(&self) -> impl Iterator<Item = usize> + '_ {
        let pts = &self.polyline.points;
        let threshold = self.polyline_threshold;
        self.cloud_positions
            .iter()
            .enumerate()
            .filter_map(move |(i, p)| {
                let p = glam::Vec3::from(*p);
                let near = pts.windows(2).any(|w| {
                    let a = w[0];
                    let b = w[1];
                    let seg = b - a;
                    let seg_len = seg.length();
                    if seg_len < 1e-6 {
                        return (p - a).length() <= threshold;
                    }
                    let dir = seg / seg_len;
                    let t = ((p - a).dot(dir)).clamp(0.0, seg_len);
                    (p - (a + dir * t)).length() <= threshold
                });
                if near { Some(i) } else { None }
            })
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
        self.cloud_positions
            .iter()
            .enumerate()
            .filter_map(move |(i, p)| {
                let p = glam::Vec3::from(*p);
                let t = ((p - start).dot(dir_n)).clamp(0.0, len);
                if (p - (start + dir_n * t)).length() <= threshold {
                    Some(i)
                } else {
                    None
                }
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
        self.pw_state = ProbeWidgetState::new();
        self.pw_state.built = true;
    }

    pub(crate) fn update_probe_widgets(&mut self, ctx_widget: WidgetContext) {
        let state = &mut self.pw_state;
        let result = match state.sub_mode {
            PwSubMode::LineProbe => state.probe.update(&ctx_widget),
            PwSubMode::Sphere => state.sphere.update(&ctx_widget),
            PwSubMode::Box => state.bw.update(&ctx_widget),
            PwSubMode::Plane => state.plane.update(&ctx_widget),
            PwSubMode::Disk => state.disk.update(&ctx_widget),
            PwSubMode::Cylinder => state.cylinder.update(&ctx_widget),
            PwSubMode::Polyline => state.polyline.update(&ctx_widget),
        };
        state.last_result = result;
        state.suppress_orbit = match state.sub_mode {
            PwSubMode::LineProbe => state.probe.is_active(),
            PwSubMode::Sphere => state.sphere.is_active(),
            PwSubMode::Box => state.bw.is_active(),
            PwSubMode::Plane => state.plane.is_active(),
            PwSubMode::Disk => state.disk.is_active(),
            PwSubMode::Cylinder => state.cylinder.is_active(),
            PwSubMode::Polyline => state.polyline.is_active(),
        };
    }

    pub(crate) fn controls_probe_widgets(&mut self, ui: &mut egui::Ui) {
        let state = &mut self.pw_state;

        ui.label("Widget:");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut state.sub_mode, PwSubMode::LineProbe, "Line Probe");
            ui.radio_value(&mut state.sub_mode, PwSubMode::Sphere, "Sphere");
            ui.radio_value(&mut state.sub_mode, PwSubMode::Box, "Box (OBB)");
            ui.radio_value(&mut state.sub_mode, PwSubMode::Plane, "Plane");
            ui.radio_value(&mut state.sub_mode, PwSubMode::Disk, "Disk");
            ui.radio_value(&mut state.sub_mode, PwSubMode::Cylinder, "Cylinder");
            ui.radio_value(&mut state.sub_mode, PwSubMode::Polyline, "Polyline");
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
                selection_buttons(ui, state, PwSubMode::LineProbe);
            }
            PwSubMode::Sphere => {
                let c = state.sphere.center;
                ui.label(format!("Center: [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z));
                ui.label(format!("Radius: {:.3}", state.sphere.radius));
                ui.separator();
                selection_buttons(ui, state, PwSubMode::Sphere);
            }
            PwSubMode::Box => {
                let c = state.bw.center;
                let h = state.bw.half_extents;
                let (_, _, rot) = state.bw.obb();
                let (axis, angle) = rot.to_axis_angle();
                ui.label(format!(
                    "Center:       [{:.2}, {:.2}, {:.2}]",
                    c.x, c.y, c.z
                ));
                ui.label(format!(
                    "Half-extents: [{:.2}, {:.2}, {:.2}]",
                    h.x, h.y, h.z
                ));
                ui.label(format!(
                    "Rotation:     {:.1} deg around [{:.2}, {:.2}, {:.2}]",
                    angle.to_degrees(),
                    axis.x,
                    axis.y,
                    axis.z
                ));
                ui.separator();
                ui.label("Drag arc handles (outer circles) to rotate.");
                ui.separator();
                selection_buttons(ui, state, PwSubMode::Box);
            }
            PwSubMode::Plane => {
                let c = state.plane.center;
                let n = state.plane.normal;
                ui.label(format!("Center: [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z));
                ui.label(format!("Normal: [{:.2}, {:.2}, {:.2}]", n.x, n.y, n.z));
                ui.separator();
                ui.label("Select: points on positive-normal side.");
                ui.separator();
                selection_buttons(ui, state, PwSubMode::Plane);
            }
            PwSubMode::Disk => {
                let c = state.disk.center;
                let n = state.disk.normal;
                ui.label(format!("Center: [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z));
                ui.label(format!("Normal: [{:.2}, {:.2}, {:.2}]", n.x, n.y, n.z));
                ui.label(format!("Radius: {:.3}", state.disk.radius));
                ui.horizontal(|ui| {
                    ui.label("Half-thickness:");
                    ui.add(
                        egui::Slider::new(&mut state.disk_half_thickness, 0.05..=2.0).step_by(0.05),
                    );
                });
                ui.separator();
                selection_buttons(ui, state, PwSubMode::Disk);
            }
            PwSubMode::Cylinder => {
                let s = state.cylinder.start;
                let e = state.cylinder.end;
                ui.label(format!("Start:  [{:.2}, {:.2}, {:.2}]", s.x, s.y, s.z));
                ui.label(format!("End:    [{:.2}, {:.2}, {:.2}]", e.x, e.y, e.z));
                ui.label(format!("Radius: {:.3}", state.cylinder.radius));
                ui.label(format!("Length: {:.3}", (e - s).length()));
                ui.separator();
                selection_buttons(ui, state, PwSubMode::Cylinder);
            }
            PwSubMode::Polyline => {
                ui.label(format!("Points: {}", state.polyline.points.len()));
                for (i, p) in state.polyline.points.iter().enumerate() {
                    ui.label(format!("  [{}] [{:.2}, {:.2}, {:.2}]", i, p.x, p.y, p.z));
                }
                ui.horizontal(|ui| {
                    ui.label("Near-path radius:");
                    ui.add(
                        egui::Slider::new(&mut state.polyline_threshold, 0.05..=3.0).step_by(0.05),
                    );
                });
                ui.separator();
                ui.label("Double-click a handle to remove it.");
                ui.label("Double-click a segment to add a point.");
                ui.separator();
                selection_buttons(ui, state, PwSubMode::Polyline);
            }
        }

        ui.separator();
        if state.suppress_orbit {
            ui.label("(Orbit suppressed: widget active)");
        }
    }
}

fn selection_buttons(ui: &mut egui::Ui, state: &mut ProbeWidgetState, mode: PwSubMode) {
    ui.label(format!(
        "Selected: {} / {}",
        state.selected_count(),
        state.cloud_positions.len()
    ));
    ui.horizontal(|ui| {
        if ui.button("Select from region").clicked() {
            match mode {
                PwSubMode::LineProbe => state.apply_line_selection(true),
                PwSubMode::Sphere => state.apply_sphere_selection(true),
                PwSubMode::Box => state.apply_box_selection(true),
                PwSubMode::Plane => state.apply_plane_selection(true),
                PwSubMode::Disk => state.apply_disk_selection(true),
                PwSubMode::Cylinder => state.apply_cylinder_selection(true),
                PwSubMode::Polyline => state.apply_polyline_selection(true),
            }
        }
        if ui.button("Deselect from region").clicked() {
            match mode {
                PwSubMode::LineProbe => state.apply_line_selection(false),
                PwSubMode::Sphere => state.apply_sphere_selection(false),
                PwSubMode::Box => state.apply_box_selection(false),
                PwSubMode::Plane => state.apply_plane_selection(false),
                PwSubMode::Disk => state.apply_disk_selection(false),
                PwSubMode::Cylinder => state.apply_cylinder_selection(false),
                PwSubMode::Polyline => state.apply_polyline_selection(false),
            }
        }
    });
    if ui.button("Clear selection").clicked() {
        state.clear_selection();
    }
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn pw_collect_scene_items(
    app: &mut App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64, u64) {
    let items = app.pw_state.scene.collect_render_items(&viewport_lib::selection::Selection::new());
    let lighting = LightingSettings {
        hemisphere_intensity: 0.6,
        sky_colour: [1.0, 1.0, 1.0],
        ground_colour: [0.8, 0.8, 0.8],
        ..LightingSettings::default()
    };
    let sg = app.pw_state.scene.version();
    (items, lighting, sg, 0)
}

pub(crate) fn submit_pw_items(app: &App, fd: &mut FrameData, w: f32, h: f32) {
    if !app.pw_state.built {
        return;
    }
    let render_cam = CameraFrame::from_camera(&app.camera, [w, h]).render_camera;
    let widget_ctx = WidgetContext {
        camera: render_cam,
        viewport_size: glam::Vec2::new(w, h),
        cursor_viewport: app.interact_state.last_cursor_viewport,
        drag_started: false,
        dragging: false,
        released: false,
        double_clicked: false,
    };
    let state = &app.pw_state;
    match state.sub_mode {
        PwSubMode::LineProbe => {
            fd.scene.polylines.push(state.probe.polyline_item(0));
            fd.scene
                .glyphs
                .push(state.probe.handle_glyphs(100, &widget_ctx));
        }
        PwSubMode::Sphere => {
            fd.effects.clip_objects.push(state.sphere.clip_object());
            fd.scene.polylines.push(state.sphere.wireframe_item(0));
            fd.scene
                .glyphs
                .push(state.sphere.handle_glyphs(100, &widget_ctx));
        }
        PwSubMode::Box => {
            fd.scene.polylines.push(state.bw.wireframe_item(0));
            fd.scene.polylines.push(state.bw.rotation_arcs_item(1));
            fd.scene
                .glyphs
                .push(state.bw.handle_glyphs(100, &widget_ctx));
        }
        PwSubMode::Plane => {
            fd.scene.polylines.push(state.plane.plane_item(0));
            fd.scene
                .glyphs
                .push(state.plane.handle_glyphs(100, &widget_ctx));
        }
        PwSubMode::Disk => {
            fd.scene.polylines.push(state.disk.wireframe_item(0));
            fd.scene
                .glyphs
                .push(state.disk.handle_glyphs(100, &widget_ctx));
        }
        PwSubMode::Cylinder => {
            fd.scene.polylines.push(state.cylinder.wireframe_item(0));
            fd.scene
                .glyphs
                .push(state.cylinder.handle_glyphs(100, &widget_ctx));
        }
        PwSubMode::Polyline => {
            fd.scene.polylines.push(state.polyline.polyline_item(0));
            fd.scene
                .glyphs
                .push(state.polyline.handle_glyphs(100, &widget_ctx));
        }
    }

    // Point cloud: unselected in blue, selected in orange.
    let unsel: Vec<[f32; 3]> = state
        .cloud_positions
        .iter()
        .zip(state.selected.iter())
        .filter(|(_, s)| !**s)
        .map(|(p, _)| *p)
        .collect();
    let sel: Vec<[f32; 3]> = state
        .cloud_positions
        .iter()
        .zip(state.selected.iter())
        .filter(|(_, s)| **s)
        .map(|(p, _)| *p)
        .collect();
    if !unsel.is_empty() {
        let mut pc = PointCloudItem::default();
        pc.positions = unsel;
        pc.default_colour = [0.5, 0.7, 1.0, 1.0];
        pc.gaussian = true;
        pc.point_size = 8.0;
        fd.scene.point_clouds.push(pc);
    }
    if !sel.is_empty() {
        let mut pc = PointCloudItem::default();
        pc.positions = sel;
        pc.default_colour = [1.0, 0.55, 0.1, 1.0];
        pc.gaussian = true;
        pc.point_size = 14.0;
        fd.scene.point_clouds.push(pc);
    }
}
