//! Showcase 13: Multi-Viewport : build, update, and controls.
//!
//! Renders the same scene from four independent cameras in a 2x2 quad layout:
//!
//! ```text
//! +-----------------+-----------------+
//! |  Perspective    |  Top (ortho)    |
//! |  (orbit)        |  looking -Z     |
//! +-----------------+-----------------+
//! |  Front (ortho)  |  Right (ortho)  |
//! |  looking -Y     |  looking -X     |
//! +-----------------+-----------------+
//! ```
//!
//! Each quadrant has its own `OrbitCameraController`. Mouse input is routed to
//! whichever quad is under the cursor. The gizmo appears in all four viewports
//! simultaneously (each with its own screen-space scale), but hit-testing and
//! drag transforms are driven by the hovered quad's camera.

use crate::App;
use crate::eframe;
use crate::eframe::egui;
use crate::scene_gizmo::{self, Snapshots};
use crate::shared::upload_material;
use viewport_lib as vpl;
use viewport_lib::Colour;
use viewport_lib::input::adapters::{AdapterCtx, egui as vpl_egui};
use viewport_lib_gizmo::{
    FrameInput as GizmoInput, GizmoDriver, GizmoMode, GizmoSpace, GizmoStyle, ManipResult,
};
use vpl::{
    Action, BindingPreset, ButtonState, Camera, CameraFrame, FrameData, InputOwnership,
    LightingSettings, Material, MouseButton, OrbitCameraController, PickBackend, PickMask,
    SceneFrame, Selection, ViewportContext, ViewportEvent, ViewportInput, ViewportRenderer,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct MvState {
    pub scene: vpl::scene::Scene,
    pub selection: Selection,
    pub cameras: [Camera; 4],
    pub controllers: [OrbitCameraController; 4],
    /// One resolver per quadrant: each owns its bindings and resolves the
    /// events routed to it. The controllers keep only their sensitivities.
    pub inputs: [ViewportInput; 4],
    pub viewports: Option<[vpl::ViewportId; 4]>,
    pub built: bool,
    /// The transform gizmo, driven from the hovered quadrant and drawn in all four.
    pub gizmo: GizmoDriver,
    pub gizmo_style: GizmoStyle,
    pub hovered_quad: usize,
    pub cursor_local: glam::Vec2,
    /// Deferred pick request `(quad, cursor_local)`, set on a click and consumed
    /// at the render site with that quad's frame via the unified GPU picker.
    pub pending_pick: Option<(usize, glam::Vec2)>,
    /// The selection's poses when the current session began.
    pub snapshots: Snapshots,
    pub left_held: bool,
}

impl Default for MvState {
    fn default() -> Self {
        Self {
            scene: vpl::scene::Scene::new(),
            selection: Selection::new(),
            cameras: [
                Camera::orbit(
                    glam::DVec3::ZERO,
                    12.0,
                    glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1),
                ),
                Camera::orbit(glam::DVec3::ZERO, 10.0, glam::Quat::IDENTITY).orthographic(),
                Camera::orbit(
                    glam::DVec3::ZERO,
                    10.0,
                    glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                )
                .orthographic(),
                Camera::orbit(
                    glam::DVec3::ZERO,
                    10.0,
                    glam::Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)
                        * glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                )
                .orthographic(),
            ],
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
            viewports: None,
            built: false,
            gizmo: GizmoDriver::new(),
            gizmo_style: GizmoStyle::default(),
            hovered_quad: 0,
            cursor_local: glam::Vec2::ZERO,
            pending_pick: None,
            snapshots: Snapshots::default(),
            left_held: false,
        }
    }
}

const QUAD_LABELS: [&str; 4] = ["Perspective", "Top", "Front", "Right"];

// ---------------------------------------------------------------------------
// Scene build
// ---------------------------------------------------------------------------

impl App {
    /// Build the shared scene for showcase 13.
    pub(crate) fn build_mv_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.mv_state.scene = vpl::scene::Scene::new();
        self.mv_state.selection = Selection::new();

        let positions: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let colours: [[f32; 3]; 6] = [
            [0.55, 0.55, 0.55],
            [0.65, 0.09, 0.07],
            [0.10, 0.52, 0.18],
            [0.10, 0.26, 0.68],
            [0.60, 0.50, 0.05],
            [0.75, 0.28, 0.05],
        ];
        let names = ["Centre", "Right", "Left", "Front", "Back", "Top"];

        for (i, ((pos, colour), name)) in positions.iter().zip(&colours).zip(&names).enumerate() {
            let mesh = self.upload_box(renderer);
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material::from_colour(Colour::from_linear_rgb_array(*colour));
            let id = self.mv_state.scene.add_named(
                name,
                Some(mesh),
                transform,
                upload_material(renderer, mat),
            );
            if i == 0 {
                self.mv_state.selection.select_one(id);
            }
        }
        self.mv_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

impl App {
    /// Called from the CentralPanel when `ShowcaseMode::MultiViewport` is active.
    pub(crate) fn update_multi_viewport(
        &mut self,
        _ctx: &egui::Context,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        frame: &eframe::Frame,
    ) {
        let Some(viewports) = self.mv_state.viewports else {
            return;
        };

        let w = rect.width();
        let h = rect.height();
        let half_w = w / 2.0;
        let half_h = h / 2.0;

        // Absolute-screen quad rects (egui logical points).
        let quad_rects: [egui::Rect; 4] = [
            egui::Rect::from_min_size(rect.min, egui::vec2(half_w, half_h)),
            egui::Rect::from_min_size(
                rect.min + egui::vec2(half_w, 0.0),
                egui::vec2(w - half_w, half_h),
            ),
            egui::Rect::from_min_size(
                rect.min + egui::vec2(0.0, half_h),
                egui::vec2(half_w, h - half_h),
            ),
            egui::Rect::from_min_size(
                rect.min + egui::vec2(half_w, half_h),
                egui::vec2(w - half_w, h - half_h),
            ),
        ];

        // Determine hovered quadrant from current pointer position.
        let hover_pos = ui.input(|i| i.pointer.hover_pos());
        let prev_hq = self.mv_state.hovered_quad;
        if let Some(pos) = hover_pos {
            if rect.contains(pos) {
                let new_quad = quad_rects
                    .iter()
                    .position(|qr| qr.contains(pos))
                    .unwrap_or(0);
                self.mv_state.hovered_quad = new_quad;
                let qr = quad_rects[new_quad];
                self.mv_state.cursor_local = glam::Vec2::new(pos.x - qr.left(), pos.y - qr.top());
            }
        }
        let hq = self.mv_state.hovered_quad;

        // begin_frame for all four resolvers; the controllers only need the size
        // their pan reads. Events go to the hovered quad alone, so it takes the
        // viewport widget's ownership and the others have none.
        for i in 0..4 {
            let qr = quad_rects[i];
            let ownership = if i == hq {
                vpl_egui::ownership(response, ui.ctx(), qr, self.mv_state.gizmo.is_active())
            } else {
                InputOwnership::NONE
            };
            self.mv_state.inputs[i]
                .begin_frame(ViewportContext::new(ownership, [qr.width(), qr.height()]));
            self.mv_state.controllers[i].set_viewport_size([qr.width(), qr.height()]);
        }

        // Notify the previously hovered quad that the pointer left.
        if hq != prev_hq {
            self.mv_state.inputs[prev_hq].push_event(ViewportEvent::PointerLeft);
        }

        // Push this frame's input events to the hovered controller, in its local
        // coordinates.
        let qr = quad_rects[hq];
        let actx = AdapterCtx::new(1.0, glam::Vec2::new(qr.left(), qr.top()));
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
            // Track the left button for the gizmo, ignoring presses that
            // originate outside the full viewport.
            if let ViewportEvent::MouseButton {
                button: MouseButton::Left,
                state,
            } = ev
            {
                let pressed = state == ButtonState::Pressed;
                if !pressed || press_inside {
                    self.mv_state.left_held = pressed;
                }
            }
            self.mv_state.inputs[hq].push_event(ev);
        }

        // ----- Gizmo update (G/R/S and handle drags) -----
        {
            let qr = quad_rects[hq];
            let viewport_size = glam::Vec2::new(qr.width(), qr.height());

            // One resolve for the hovered quadrant; its camera holds still while
            // a gizmo session owns the pointer.
            let action_frame = self.mv_state.inputs[hq].resolve();
            if !self.mv_state.gizmo.is_active() {
                let camera = &mut self.mv_state.cameras[hq];
                let dt = ui.input(|i| i.stable_dt);
                self.mv_state.controllers[hq].apply(camera, &action_frame, dt);
            }

            let state = &mut self.mv_state;
            // Tab cycles the gizmo mode (ignored while a session runs).
            if action_frame.is_active(Action::CycleGizmoMode) {
                state.gizmo.cycle_mode();
            }

            // The hovered quadrant drives the gizmo. The press and hold come
            // from egui so a drag keeps going when it crosses into another
            // quadrant.
            let centre = scene_gizmo::selection_centre(&state.selection, &state.scene);
            let orientation = scene_gizmo::selection_orientation(&state.selection, &state.scene);
            let input = GizmoInput {
                cursor_viewport: Some(state.cursor_local),
                pointer_delta: ui
                    .input(|i| glam::Vec2::new(i.pointer.delta().x, i.pointer.delta().y)),
                drag_started: response.drag_started(),
                dragging: state.left_held,
                clicked: response.clicked(),
                ..GizmoInput::from_action(&action_frame, &state.cameras[hq], viewport_size, centre)
            }
            .with_selection_orientation(orientation);
            match state.gizmo.update(input) {
                ManipResult::Update(delta) => {
                    if let Some(session) = state.gizmo.state() {
                        state
                            .snapshots
                            .apply(&delta, session.centre, &mut state.scene);
                    }
                }
                ManipResult::Cancel | ManipResult::ConstraintChanged => {
                    state.snapshots.restore(&mut state.scene);
                }
                _ => {}
            }
            // Idle: keep the snapshots current so G/R/S always starts clean.
            if !state.gizmo.is_active() {
                state.snapshots.save(&state.selection, &state.scene);
            }

            // Apply remaining controllers (non-hq); hq already applied/resolved above.
            for i in 0..4 {
                if i != hq {
                    let frame = self.mv_state.inputs[i].resolve();
                    let camera = &mut self.mv_state.cameras[i];
                    let dt = ui.input(|input| input.stable_dt);
                    self.mv_state.controllers[i].apply(camera, &frame, dt);
                }
                self.mv_state.cameras[i]
                    .set_aspect_ratio(quad_rects[i].width(), quad_rects[i].height());
            }

            // Click-to-select only when no session is active. Resolved at the
            // render site against the clicked quad's frame (unified GPU picker).
            if response.clicked() && !self.mv_state.gizmo.is_active() {
                self.mv_state.pending_pick = Some((hq, self.mv_state.cursor_local));
            }
        }

        // --- Build FrameData ---
        // Collect render items once; clone into each frame.
        let scene_items = self
            .mv_state
            .scene
            .collect_render_items(&self.mv_state.selection);
        let lighting = {
            let mut _t = LightingSettings::default();
            _t.hemisphere_intensity = 0.5;
            _t.sky_colour = Colour::linear_rgb(1.0, 1.0, 1.0);
            _t.ground_colour = Colour::linear_rgb(1.0, 1.0, 1.0);
            _t
        };
        let ppp = ui.ctx().pixels_per_point();
        let scene_gen = self.mv_state.scene.version();
        let sel_gen = self.mv_state.selection.version();
        let has_selection = !self.mv_state.selection.is_empty();
        let gizmo = &self.mv_state.gizmo;
        let gizmo_style = &self.mv_state.gizmo_style;
        let cameras = &self.mv_state.cameras;

        let frames: [FrameData; 4] = std::array::from_fn(|i| {
            let qr = quad_rects[i];
            let qw = qr.width();
            let qh = qr.height();

            let camera_frame = CameraFrame::from_camera(&self.mv_state.cameras[i], [qw, qh])
                .with_viewport_id(viewports[i])
                .with_pixels_per_point(ppp);

            let mut fd = FrameData::new(
                camera_frame,
                SceneFrame::from_surface_items(scene_items.clone()),
            );

            fd.effects.lighting = lighting.clone();
            fd.viewport.show_grid = true;
            fd.viewport.show_axes_indicator = true;
            // Leave background_colour unset so the renderer default applies.
            fd.interaction.outline_selected = has_selection;
            fd.scene.generation = scene_gen;
            fd.interaction.selection_generation = sel_gen;

            // Gizmo: drawn in every viewport at its own screen-space scale; only
            // the driving quadrant shows the hovered handle.
            gizmo.build_view_overlays(
                gizmo_style,
                &cameras[i],
                glam::Vec2::new(qw, qh),
                i == hq,
                &mut fd.overlays.shapes,
                &mut fd.overlays.polylines,
            );

            fd
        });

        // --- egui overlays: divider lines + quad labels ---
        let painter = ui.painter_at(rect);
        let div_colour = egui::Color32::from_gray(90);
        let stroke = egui::Stroke::new(1.0, div_colour);
        // Vertical centre line.
        painter.line_segment(
            [
                rect.min + egui::vec2(half_w, 0.0),
                rect.min + egui::vec2(half_w, h),
            ],
            stroke,
        );
        // Horizontal centre line.
        painter.line_segment(
            [
                rect.min + egui::vec2(0.0, half_h),
                rect.min + egui::vec2(w, half_h),
            ],
            stroke,
        );
        // Quad labels.
        let label_colour = egui::Color32::from_rgba_unmultiplied(200, 200, 200, 160);
        for (i, qr) in quad_rects.iter().enumerate() {
            painter.text(
                qr.min + egui::vec2(6.0, 4.0),
                egui::Align2::LEFT_TOP,
                QUAD_LABELS[i],
                egui::FontId::proportional(11.0),
                label_colour,
            );
        }

        // Highlight the hovered quad with a faint border.
        painter.rect_stroke(
            quad_rects[hq].shrink(0.5),
            0.0,
            egui::Stroke::new(
                1.5,
                egui::Color32::from_rgba_unmultiplied(180, 180, 255, 80),
            ),
            egui::StrokeKind::Middle,
        );

        // Render each quadrant into its own offscreen target (sRGB dual-view so the
        // tonemap encode survives egui's sample), then draw them as egui images.
        let rs = frame.wgpu_render_state().expect("wgpu required");
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
                .zip(quad_rects.iter())
                .any(|(t, qr)| t.inner.size() != quad_px(qr));
        if needs_rebuild {
            for old in self.mv_targets.drain(..) {
                guard.free_texture(&old.id);
            }
            for qr in &quad_rects {
                let inner =
                    vpl::OffscreenViewportTarget::new(&self.device, rs.target_format, quad_px(qr));
                let id = guard.register_native_texture(
                    &self.device,
                    inner.sample_view(),
                    eframe::wgpu::FilterMode::Linear,
                );
                self.mv_targets.push(crate::Target { inner, id });
            }
        }
        if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
            let (scene_fx, _) = frames[0].effects.split();
            let token =
                renderer
                    .owned()
                    .prepare_scene(&self.device, &self.queue, &frames[0], &scene_fx);
            for i in 0..4 {
                renderer.owned().prepare_viewport(
                    &self.device,
                    &self.queue,
                    &token,
                    viewports[i],
                    &frames[i],
                );
            }
            for i in 0..4 {
                let cmd = renderer.owned().render_viewport(
                    &self.device,
                    &self.queue,
                    self.mv_targets[i].inner.render_view(),
                    viewports[i],
                    &frames[i],
                );
                self.queue.submit(std::iter::once(cmd));
            }
            // Resolve a deferred click pick against the clicked quad's frame.
            if let Some((quad, cursor)) = self.mv_state.pending_pick.take() {
                let hit = renderer.pick_object(
                    PickBackend::Gpu,
                    cursor,
                    &frames[quad],
                    &self.device,
                    &self.queue,
                    PickMask::OBJECT,
                );
                match hit {
                    Some(h) => self.mv_state.selection.select_one(h.id),
                    None => self.mv_state.selection.clear(),
                }
            }
        }
        let ids: [egui::TextureId; 4] = std::array::from_fn(|i| self.mv_targets[i].id);
        drop(guard);
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        for (i, qr) in quad_rects.iter().enumerate() {
            ui.painter().image(ids[i], *qr, uv, egui::Color32::WHITE);
        }
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_mv(app: &mut App, ui: &mut egui::Ui) {
    let node_count = app.mv_state.scene.node_count();
    let sel_count = app.mv_state.selection.len();
    ui.label(format!("Objects: {node_count}   Selected: {sel_count}"));
    ui.separator();

    let gizmo = &mut app.mv_state.gizmo;
    ui.label("Gizmo Mode:");
    ui.horizontal(|ui| {
        for (mode, label) in [
            (GizmoMode::Translate, "Move"),
            (GizmoMode::Rotate, "Rotate"),
            (GizmoMode::Scale, "Scale"),
        ] {
            if ui.radio(gizmo.mode() == mode, label).clicked() {
                gizmo.set_mode(mode);
            }
        }
    });

    ui.separator();
    ui.label("Space:");
    ui.horizontal(|ui| {
        for (space, label) in [(GizmoSpace::World, "World"), (GizmoSpace::Local, "Local")] {
            if ui.radio(gizmo.space() == space, label).clicked() {
                gizmo.set_space(space);
            }
        }
    });

    ui.separator();
    if ui.button("Clear Selection").clicked() {
        app.mv_state.selection.clear();
    }

    ui.separator();
    ui.weak("Camera controls");
    ui.label("Middle/Left drag: Orbit");
    ui.label("Right/Shift+drag: Pan");
    ui.label("Scroll: Zoom");
    ui.separator();
    ui.weak("Manipulation");
    ui.label("G / R / S: Move / Rotate / Scale");
    ui.label("X / Y / Z: Constrain axis");
    ui.label("Enter: Confirm   Esc: Cancel");
    ui.label("Tab: Cycle gizmo mode");
    ui.separator();
    ui.weak("Selection");
    ui.label("Click object: Select");
    ui.label("Click empty: Deselect");
    ui.separator();
    ui.weak("Quad layout");
    ui.label("TL : Perspective (orbit)");
    ui.label("TR : Top (ortho, -Z)");
    ui.label("BL : Front (ortho, -Y)");
    ui.label("BR : Right (ortho, -X)");
}

// ---------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.mv_state.built || app.mv_state.viewports.is_none()
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    if app.mv_state.viewports.is_none() {
        let vp0 = renderer.create_viewport(&app.device);
        let vp1 = renderer.create_viewport(&app.device);
        let vp2 = renderer.create_viewport(&app.device);
        let vp3 = renderer.create_viewport(&app.device);
        app.mv_state.viewports = Some([vp0, vp1, vp2, vp3]);
    }
    if !app.mv_state.built {
        app.build_mv_scene(renderer);
    }
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. `_out` carries
/// the few extra frame settings a showcase can set alongside its items.
pub(crate) fn scene(
    _app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    _out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    // This showcase drives all four viewports from `viewport_override`, so the
    // host's single-viewport frame assembly never runs for it.
    unreachable!("multi-viewport is handled before build_frame_data")
}

// ---------------------------------------------------------------------------
// Per-frame frame-data tweaks
// ---------------------------------------------------------------------------

/// Fold this showcase's own contributions into the assembled frame: extra
/// render items, overlays, and effect settings that are re-submitted every
/// frame rather than baked into the scene.

// ---------------------------------------------------------------------------
// Viewport overlay and per-frame tick
// ---------------------------------------------------------------------------

/// Draw this showcase's own egui overlay on top of the rendered viewport:
/// selection rectangles, mode readouts, and in-scene labels.

/// Advance this showcase's animation and ask for another frame. Runs after the
/// viewport has been drawn, so it only affects the next frame.

/// Route a viewport click for this showcase. The host calls this for a plain
/// click that no gizmo or widget has already consumed; `pos` is in viewport
/// pixels.

/// Handle drag gestures this showcase owns, before the camera controller runs.

/// Advance this showcase's own camera animation or object motion for the frame.

/// Update this showcase's interactive widgets for the frame.

/// Flush any per-frame GPU writes this showcase has queued.

/// Take over the whole viewport for this frame: this showcase drives four
/// quadrants itself rather than the host's single-viewport path.
pub(crate) fn viewport_override(
    app: &mut crate::App,
    ui: &mut crate::eframe::egui::Ui,
    cx: &crate::ViewportCtx,
) -> bool {
    app.update_multi_viewport(cx.egui, ui, cx.rect, cx.response, cx.frame);
    true
}

/// Drive the orbit controller for this showcase. Returning false leaves the
/// host to run the usual suppress-or-apply path.
pub(crate) fn drive_camera(_app: &mut crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

/// Whether the orbit controller should resolve without moving the camera this
/// frame. This showcase never suppresses it.
pub(crate) fn suppress_orbit(_app: &crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScMultiViewport;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScMultiViewport = ScMultiViewport;

impl crate::Showcase for ScMultiViewport {
    fn needs_build(&self, app: &crate::App) -> bool {
        needs_build(app)
    }
    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        build(app, renderer)
    }
    fn scene(
        &self,
        app: &mut crate::App,
        frame: &crate::eframe::Frame,
        out: &mut crate::SceneOverrides,
    ) -> crate::SceneContents {
        scene(app, frame, out)
    }
    fn viewport_override(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        cx: &crate::ViewportCtx,
    ) -> bool {
        viewport_override(app, ui, cx)
    }
    fn drive_camera(&self, app: &mut crate::App, cx: &crate::ViewportCtx) -> bool {
        drive_camera(app, cx)
    }
    fn suppress_orbit(&self, app: &crate::App, cx: &crate::ViewportCtx) -> bool {
        suppress_orbit(app, cx)
    }
    fn controls(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        _frame: &crate::eframe::Frame,
    ) {
        controls_mv(app, ui)
    }
}
