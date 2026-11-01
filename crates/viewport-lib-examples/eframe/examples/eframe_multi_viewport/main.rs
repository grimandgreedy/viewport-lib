//! Four-viewport layout with gizmo-driven manipulation.
//!
//! Layout (2x2 quad):
//!   TL: Perspective  TR: Top (ortho, -Z)
//!   BL: Front (ortho, -Y)  BR: Right (ortho, -X)
//!
//! Each quadrant has its own orbit camera. Mouse input routes to whichever quad
//! is under the cursor. The gizmo appears in all four viewports simultaneously
//! with its own screen-space scale per quad; hit-testing and drag transforms use
//! the hovered quad's camera.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom
//!
//! Selection:
//!   Left-click object         : select
//!   Left-click empty          : deselect
//!
//! Manipulation (requires a selection):
//!   G / R / S                 : move / rotate / scale
//!   X / Y / Z                 : constrain axis
//!   Shift+X/Y/Z               : exclude axis (constrain to plane)
//!   Tab                       : cycle gizmo mode
//!   Digits while active       : numeric input
//!   Enter / left-click        : confirm   Esc : cancel

use crate::eframe::{egui, wgpu};
use std::collections::HashMap;
use viewport_lib as vpl;
use viewport_lib::Colour;
use viewport_lib::input::adapters::{AdapterCtx, egui as vpl_egui};
pub use viewport_lib_examples_eframe::eframe;
use viewport_lib_gizmo::{
    FrameInput as GizmoInput, GizmoDriver, GizmoStyle, ManipResult, Selection, TransformDelta,
    build_numeric_label, gizmo_centre_for_pivot,
};
use vpl::{
    Action, BackfacePolicy, BindingPreset, ButtonState, Camera, CameraFrame, FrameData,
    InputOwnership, LightingSettings, Material, MaterialId, MeshId, MouseButton,
    OffscreenViewportTarget, OrbitCameraController, SceneFrame, SceneRenderItem, ViewportContext,
    ViewportEvent, ViewportId, ViewportInput, ViewportRenderer, picking::screen_to_ray, primitives,
};

const QUAD_LABELS: [&str; 4] = ["Perspective", "Top", "Front", "Right"];

fn quad_rects(full: egui::Rect) -> [egui::Rect; 4] {
    let mx = full.left() + full.width() * 0.5;
    let my = full.top() + full.height() * 0.5;
    [
        egui::Rect::from_min_max(full.min, egui::pos2(mx, my)),
        egui::Rect::from_min_max(egui::pos2(mx, full.top()), egui::pos2(full.right(), my)),
        egui::Rect::from_min_max(egui::pos2(full.left(), my), egui::pos2(mx, full.bottom())),
        egui::Rect::from_min_max(egui::pos2(mx, my), full.max),
    ]
}

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Multi-Viewport",
        viewport_lib_examples_eframe::native_options([1280.0, 800.0]),
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let device = &rs.device;
            let format = rs.target_format;

            // sRGB render format; the offscreen targets keep the encode through egui.
            let mut renderer =
                ViewportRenderer::new(device, OffscreenViewportTarget::render_format(format));
            let res = renderer.resources_mut();
            let m_sphere = res
                .upload_mesh_data(device, &primitives::sphere(0.6, 24, 12))
                .unwrap();
            let m_cube = res
                .upload_mesh_data(device, &primitives::cube(1.0))
                .unwrap();
            let m_torus = res
                .upload_mesh_data(device, &primitives::torus(0.5, 0.18, 32, 16))
                .unwrap();
            // One material for every object; each item sets its own colour.
            let mut material = Material::default();
            material.backface_policy = BackfacePolicy::Identical;
            let material = res.upload_material(material).unwrap();

            let vp_ids = [
                renderer.create_viewport(device),
                renderer.create_viewport(device),
                renderer.create_viewport(device),
                renderer.create_viewport(device),
            ];

            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App::new(
                vp_ids, material, m_sphere, m_cube, m_torus,
            )))
        }),
    )
}

// ---------------------------------------------------------------------------
// Scene object
// ---------------------------------------------------------------------------

struct SceneObject {
    mesh_id: MeshId,
    model: glam::Mat4,
    colour: [f32; 3],
    // Approximate bounding sphere radius used for CPU ray picking.
    pick_radius: f32,
}

impl SceneObject {
    fn new(mesh_id: MeshId, pos: [f32; 3], colour: [f32; 3], pick_radius: f32) -> Self {
        Self {
            mesh_id,
            model: glam::Mat4::from_translation(glam::Vec3::from(pos)),
            colour,
            pick_radius,
        }
    }

    fn position(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.model.w_axis.x,
            self.model.w_axis.y,
            self.model.w_axis.z,
        )
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    objects: Vec<SceneObject>,
    material: MaterialId,
    snapshots: HashMap<u64, glam::Mat4>,
    selection: Selection,

    cameras: [Camera; 4],
    controllers: [OrbitCameraController; 4],
    /// One resolver per quadrant: each owns its bindings and resolves the events
    /// routed to it. The controllers keep only their sensitivities.
    inputs: [ViewportInput; 4],
    /// The transform gizmo, driven from the hovered quadrant and drawn in all four.
    gizmo: GizmoDriver,
    gizmo_style: GizmoStyle,
    gizmo_centre: Option<glam::Vec3>,

    vp_ids: [ViewportId; 4],

    hovered_quad: usize,
    cursor_local: glam::Vec2,
    left_held: bool,

    /// Offscreen targets for the four quadrants (sRGB dual-view so the tonemap
    /// encode survives egui's sample).
    mv_targets: Vec<Target>,
}

/// An offscreen colour target and its egui texture id, recreated on resize.
struct Target {
    inner: OffscreenViewportTarget,
    id: egui::TextureId,
}

impl App {
    fn new(
        vp_ids: [ViewportId; 4],
        material: MaterialId,
        m_sphere: MeshId,
        m_cube: MeshId,
        m_torus: MeshId,
    ) -> Self {
        use std::f32::consts::FRAC_PI_2;
        let cameras = [
            // Perspective (TL)
            Camera::orbit(
                glam::DVec3::ZERO,
                12.0,
                glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1),
            ),
            // Top (TR): looking down -Z, orthographic
            Camera::orbit(glam::DVec3::ZERO, 10.0, glam::Quat::IDENTITY).orthographic(),
            // Front (BL): looking along -Y, orthographic
            Camera::orbit(
                glam::DVec3::ZERO,
                10.0,
                glam::Quat::from_rotation_x(FRAC_PI_2),
            )
            .orthographic(),
            // Right (BR): looking along -X, orthographic
            Camera::orbit(
                glam::DVec3::ZERO,
                10.0,
                glam::Quat::from_rotation_z(-FRAC_PI_2) * glam::Quat::from_rotation_x(FRAC_PI_2),
            )
            .orthographic(),
        ];

        Self {
            objects: vec![
                SceneObject::new(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2], 0.65),
                SceneObject::new(m_cube, [0.0, 0.0, 0.0], [0.4, 0.6, 0.9], 0.87),
                SceneObject::new(m_torus, [2.5, 0.0, 0.0], [0.3, 0.8, 0.4], 0.70),
            ],
            snapshots: HashMap::new(),
            selection: Selection::new(),
            cameras,
            controllers: [
                OrbitCameraController::new_stateless(),
                OrbitCameraController::new_stateless(),
                OrbitCameraController::new_stateless(),
                OrbitCameraController::new_stateless(),
            ],
            inputs: [
                ViewportInput::from_preset(BindingPreset::Default),
                ViewportInput::from_preset(BindingPreset::Default),
                ViewportInput::from_preset(BindingPreset::Default),
                ViewportInput::from_preset(BindingPreset::Default),
            ],
            gizmo: GizmoDriver::new(),
            gizmo_style: GizmoStyle::default(),
            gizmo_centre: None,
            vp_ids,
            material,
            hovered_quad: 0,
            cursor_local: glam::Vec2::ZERO,
            left_held: false,
            mv_targets: Vec::new(),
        }
    }

    fn position_of(&self, id: u64) -> Option<glam::Vec3> {
        self.objects.get(id as usize).map(|o| o.position())
    }

    fn save_snapshots(&mut self) {
        self.snapshots.clear();
        for &id in self.selection.iter() {
            if let Some(obj) = self.objects.get(id as usize) {
                self.snapshots.insert(id, obj.model);
            }
        }
    }

    fn restore_snapshots(&mut self) {
        for (&id, &snap) in &self.snapshots {
            if let Some(obj) = self.objects.get_mut(id as usize) {
                obj.model = snap;
            }
        }
    }

    /// Apply a `TransformDelta` to the selection. The delta is the whole change
    /// since the session began, so each object starts again from its snapshot,
    /// rotating and scaling about the session's `centre`. Typed values replace
    /// the dragged ones per axis, read relative to the start pose.
    fn apply_delta(&mut self, delta: TransformDelta, centre: glam::Vec3) {
        let pick = |typed: [Option<f32>; 3], dragged: glam::Vec3| {
            glam::Vec3::new(
                typed[0].unwrap_or(dragged.x),
                typed[1].unwrap_or(dragged.y),
                typed[2].unwrap_or(dragged.z),
            )
        };
        let translation = pick(delta.position_override, delta.translation);
        let scale = pick(delta.scale_override, delta.scale);
        let rotation = if delta.rotation_override.iter().any(Option::is_some) {
            let [x, y, z] = delta
                .rotation_override
                .map(|v| v.unwrap_or(0.0).to_radians());
            glam::Quat::from_euler(glam::EulerRot::XYZ, x, y, z)
        } else {
            delta.rotation
        };

        let about_pivot = glam::Mat4::from_translation(translation)
            * glam::Mat4::from_translation(centre)
            * glam::Mat4::from_quat(rotation)
            * glam::Mat4::from_scale(scale)
            * glam::Mat4::from_translation(-centre);
        for (&id, &snap) in &self.snapshots {
            if let Some(obj) = self.objects.get_mut(id as usize) {
                obj.model = about_pivot * snap;
            }
        }
    }

    /// CPU pick via ray-sphere intersection. Returns the closest object index hit.
    fn pick(&self, quad: usize, qw: f32, qh: f32) -> Option<u64> {
        let cam = &self.cameras[quad];
        let vp_inv = (cam.proj_matrix() * cam.view_matrix()).inverse();
        let (ray_origin, ray_dir) =
            screen_to_ray(self.cursor_local, glam::Vec2::new(qw, qh), vp_inv);

        let mut best_t = f32::INFINITY;
        let mut best_id: Option<u64> = None;

        for (i, obj) in self.objects.iter().enumerate() {
            let centre = obj.position();
            let r = obj.pick_radius;
            let oc = ray_origin - centre;
            let b = oc.dot(ray_dir);
            let c = oc.dot(oc) - r * r;
            let discriminant = b * b - c;
            if discriminant >= 0.0 {
                let t = -b - discriminant.sqrt();
                if t > 0.0 && t < best_t {
                    best_t = t;
                    best_id = Some(i as u64);
                }
            }
        }

        best_id
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

// eframe 0.35 replaced `App::update(&Context, ..)` with `App::ui(&mut Ui, ..)`,
// handing the app a margin-free root Ui instead of the Context. On the 0.33 leg a
// frameless central panel makes the same Ui, so one body serves every leg.
#[cfg(feature = "wgpu27")]
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, eframe_frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| self.frame_ui(ui, eframe_frame));
    }
}

#[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, eframe_frame: &mut eframe::Frame) {
        self.frame_ui(ui, eframe_frame);
    }
}

impl App {
    fn frame_ui(&mut self, ui: &mut egui::Ui, eframe_frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let ppp = ctx.pixels_per_point();

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| {
                let (rect, response) = ui.allocate_exact_size(
                    ui.available_rect_before_wrap().size(),
                    egui::Sense::click_and_drag(),
                );

                let quads = quad_rects(rect);

                // Determine hovered quad from cursor position.
                let hover_pos = ui.input(|i| i.pointer.hover_pos());
                let prev_hq = self.hovered_quad;
                if let Some(pos) = hover_pos {
                    if rect.contains(pos) {
                        if let Some(q) = quads.iter().position(|qr| qr.contains(pos)) {
                            self.hovered_quad = q;
                            let qr = quads[q];
                            self.cursor_local =
                                glam::Vec2::new(pos.x - qr.left(), pos.y - qr.top());
                        }
                    }
                }
                let hq = self.hovered_quad;

                // begin_frame for every resolver; the controllers only need the
                // size their pan reads. Events go to the hovered quad alone, so it
                // takes the viewport widget's ownership and the others have none.
                for i in 0..4 {
                    let qr = quads[i];
                    let ownership = if i == hq {
                        vpl_egui::ownership(&response, ui.ctx(), qr, self.gizmo.is_active())
                    } else {
                        InputOwnership::NONE
                    };
                    self.inputs[i]
                        .begin_frame(ViewportContext::new(ownership, [qr.width(), qr.height()]));
                    self.controllers[i].set_viewport_size([qr.width(), qr.height()]);
                }

                // Notify the previously hovered quad on quad switch.
                if hq != prev_hq {
                    self.inputs[prev_hq].push_event(ViewportEvent::PointerLeft);
                }

                // Route input events to the hovered quad's controller, in its local
                // coordinates.
                let actx = AdapterCtx::new(1.0, glam::Vec2::new(quads[hq].left(), quads[hq].top()));
                let mut events = Vec::new();
                let mut press_inside = false;
                ui.input(|i| {
                    press_inside = i.pointer.press_origin().is_some_and(|p| rect.contains(p));
                    vpl_egui::begin_frame(i, &actx, &mut events);
                    for event in &i.events {
                        vpl_egui::translate(event, &actx, &mut events);
                    }
                });
                for ev in events {
                    if let ViewportEvent::MouseButton {
                        button: MouseButton::Left,
                        state,
                    } = ev
                    {
                        let pressed = state == ButtonState::Pressed;
                        // Presses that originate outside the viewport are not held.
                        if !pressed || press_inside {
                            self.left_held = pressed;
                        }
                    }
                    self.inputs[hq].push_event(ev);
                }

                // One resolve for the hovered quadrant; its camera holds still
                // while a gizmo session owns the pointer.
                let action_frame = self.inputs[hq].resolve();
                if !self.gizmo.is_active() {
                    let camera = &mut self.cameras[hq];
                    let dt = ui.input(|i| i.stable_dt);
                    self.controllers[hq].apply(camera, &action_frame, dt);
                }

                // Tab cycles the gizmo mode (ignored while a session runs).
                if action_frame.is_active(Action::CycleGizmoMode) {
                    self.gizmo.cycle_mode();
                }

                // The hovered quadrant drives the gizmo. The press and hold come
                // from egui so a drag keeps going when it crosses into another
                // quadrant.
                let qr = quads[hq];
                let input = GizmoInput {
                    cursor_viewport: Some(self.cursor_local),
                    pointer_delta: ui
                        .input(|i| glam::Vec2::new(i.pointer.delta().x, i.pointer.delta().y)),
                    drag_started: ui.input(|i| i.pointer.any_pressed()),
                    dragging: self.left_held,
                    clicked: ui.input(|i| i.pointer.any_click()),
                    ..GizmoInput::from_action(
                        &action_frame,
                        &self.cameras[hq],
                        glam::Vec2::new(qr.width(), qr.height()),
                        self.gizmo_centre,
                    )
                };
                match self.gizmo.update(input) {
                    ManipResult::Update(delta) => {
                        if let Some(state) = self.gizmo.state() {
                            self.apply_delta(delta, state.centre);
                        }
                    }
                    ManipResult::Cancel | ManipResult::ConstraintChanged => {
                        self.restore_snapshots();
                    }
                    ManipResult::Commit => {
                        self.save_snapshots();
                    }
                    ManipResult::None if !self.gizmo.is_active() => self.save_snapshots(),
                    _ => {}
                }

                // Apply remaining cameras and set aspect ratios.
                for i in 0..4 {
                    if i != hq {
                        let frame = self.inputs[i].resolve();
                        let camera = &mut self.cameras[i];
                        let dt = ui.input(|input| input.stable_dt);
                        self.controllers[i].apply(camera, &frame, dt);
                    }
                    let qr = quads[i];
                    self.cameras[i].set_aspect_ratio(qr.width(), qr.height());
                }

                // Click to select (only when no session is active).
                if ui.input(|i| i.pointer.any_click()) && !self.gizmo.is_active() {
                    let qr = quads[hq];
                    if let Some(id) = self.pick(hq, qr.width(), qr.height()) {
                        self.selection.select_one(id);
                    } else {
                        self.selection.clear();
                    }
                }

                // Update the gizmo centre for the next frame.
                self.gizmo_centre =
                    gizmo_centre_for_pivot(&self.gizmo.pivot_mode(), &self.selection, |id| {
                        self.position_of(id)
                    });

                // Build scene render items.
                let scene_items: Vec<SceneRenderItem> = self
                    .objects
                    .iter()
                    .enumerate()
                    .map(|(i, obj)| {
                        let mut item = SceneRenderItem::new(obj.mesh_id, self.material);
                        item.model = obj.model.into();
                        item.settings.colour = Some(Colour::from_linear_rgb_array(obj.colour));
                        item.settings.selected = self.selection.contains(i as u64);
                        item
                    })
                    .collect();

                let readout = self.gizmo.readout().zip(self.gizmo.gizmo_centre());
                let has_selection = !self.selection.is_empty();
                let sel_gen = self.selection.version();

                // Build one FrameData per viewport.
                let frames: [FrameData; 4] = std::array::from_fn(|i| {
                    let qr = quads[i];
                    let qw = qr.width();
                    let qh = qr.height();
                    let cam_frame = CameraFrame::from_camera(&self.cameras[i], [qw, qh])
                        .with_viewport_id(self.vp_ids[i])
                        .with_pixels_per_point(ppp);
                    let mut fd = FrameData::new(
                        cam_frame,
                        SceneFrame::from_surface_items(scene_items.clone()),
                    );
                    fd.effects.lighting = LightingSettings::default();
                    fd.effects.display.mode = vpl::PipelineMode::Hdr;
                    fd.viewport.show_grid = true;
                    fd.viewport.show_axes_indicator = true;
                    fd.interaction.outline_selected = has_selection;
                    fd.interaction.selection_generation = sel_gen;

                    // Gizmo: drawn in every viewport at its own screen-space scale;
                    // only the driving quadrant shows the hovered handle.
                    let overlays = &mut fd.overlays;
                    self.gizmo.build_view_overlays(
                        &self.gizmo_style,
                        &self.cameras[i],
                        glam::Vec2::new(qw, qh),
                        i == hq,
                        &mut overlays.shapes,
                        &mut overlays.polylines,
                    );
                    if let Some((text, centre)) = readout.as_ref().filter(|_| i == hq) {
                        overlays
                            .labels
                            .push(build_numeric_label(&self.gizmo_style, text, *centre));
                    }

                    fd
                });

                // Render each quadrant into its own offscreen target (sRGB dual-view
                // so the tonemap encode survives egui's sample), then draw them.
                let rs = eframe_frame.wgpu_render_state().expect("wgpu required");
                let quad_px = |qr: &egui::Rect| -> [u32; 2] {
                    [
                        (qr.width() * ppp).round().max(1.0) as u32,
                        (qr.height() * ppp).round().max(1.0) as u32,
                    ]
                };
                let mut guard = rs.renderer.write();
                let needs_rebuild = self.mv_targets.len() != 4
                    || self
                        .mv_targets
                        .iter()
                        .zip(quads.iter())
                        .any(|(t, qr)| t.inner.size() != quad_px(qr));
                if needs_rebuild {
                    for old in self.mv_targets.drain(..) {
                        guard.free_texture(&old.id);
                    }
                    for qr in &quads {
                        let inner =
                            OffscreenViewportTarget::new(&rs.device, rs.target_format, quad_px(qr));
                        let id = guard.register_native_texture(
                            &rs.device,
                            inner.sample_view(),
                            wgpu::FilterMode::Linear,
                        );
                        self.mv_targets.push(Target { inner, id });
                    }
                }
                if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                    let (scene_fx, _) = frames[0].effects.split();
                    let token = renderer
                        .owned()
                        .prepare_scene(&rs.device, &rs.queue, &frames[0], &scene_fx);
                    for i in 0..4 {
                        renderer.owned().prepare_viewport(
                            &rs.device,
                            &rs.queue,
                            &token,
                            self.vp_ids[i],
                            &frames[i],
                        );
                    }
                    for i in 0..4 {
                        let cmd = renderer.owned().render_viewport(
                            &rs.device,
                            &rs.queue,
                            self.mv_targets[i].inner.render_view(),
                            self.vp_ids[i],
                            &frames[i],
                        );
                        rs.queue.submit(std::iter::once(cmd));
                    }
                }
                let ids: [egui::TextureId; 4] = std::array::from_fn(|i| self.mv_targets[i].id);
                drop(guard);
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                for (i, qr) in quads.iter().enumerate() {
                    ui.painter().image(ids[i], *qr, uv, egui::Color32::WHITE);
                }

                // Quad dividers and labels.
                let painter = ui.painter_at(rect);
                let stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(70));
                let mx = rect.center().x;
                let my = rect.center().y;
                painter.line_segment(
                    [egui::pos2(mx, rect.top()), egui::pos2(mx, rect.bottom())],
                    stroke,
                );
                painter.line_segment(
                    [egui::pos2(rect.left(), my), egui::pos2(rect.right(), my)],
                    stroke,
                );
                let label_colour = egui::Color32::from_rgba_unmultiplied(200, 200, 200, 160);
                for (i, qr) in quads.iter().enumerate() {
                    painter.text(
                        qr.min + egui::vec2(6.0, 4.0),
                        egui::Align2::LEFT_TOP,
                        QUAD_LABELS[i],
                        egui::FontId::proportional(11.0),
                        label_colour,
                    );
                }

                // Highlight the active quad with a faint border.
                painter.rect_stroke(
                    quads[hq].shrink(0.5),
                    0.0,
                    egui::Stroke::new(
                        1.5,
                        egui::Color32::from_rgba_unmultiplied(180, 180, 255, 60),
                    ),
                    egui::StrokeKind::Middle,
                );

                if readout.is_some() {
                    ctx.request_repaint();
                }
            });
    }
}
