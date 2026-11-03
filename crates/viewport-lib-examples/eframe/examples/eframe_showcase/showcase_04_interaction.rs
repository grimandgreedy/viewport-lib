//! Showcase 4: Professional Interaction.

use crate::App;
use crate::eframe::egui;
use crate::scene_gizmo::{self, Snapshots};
use crate::shared::upload_material;
use viewport_lib as vpl;
use viewport_lib::Colour;
use viewport_lib_gizmo::actions;
use viewport_lib_gizmo::{
    FrameInput as GizmoInput, GizmoAxis, GizmoDriver, GizmoMode, GizmoSpace, GizmoStyle,
    ManipResult, build_numeric_label,
};
use vpl::{
    CameraAnimator, CameraFrame, Easing, FrameData, LightingSettings, Material, SceneRenderItem,
    ViewPreset, ViewportRenderer, scene::Scene, selection::Selection,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct InteractState {
    pub scene: Scene,
    pub selection: Selection,
    pub animator: CameraAnimator,
    /// The transform gizmo: hover, mode, space, and the G/R/S session.
    pub gizmo: GizmoDriver,
    pub gizmo_style: GizmoStyle,
    /// The selection's poses when the current session began.
    pub snapshots: Snapshots,
    pub built: bool,
    pub spline: vpl::SplineWidget,
    /// Unit sphere the spline's control-point handles are drawn with.
    pub handle_mesh: vpl::MeshId,
}

impl Default for InteractState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            selection: Selection::new(),
            animator: CameraAnimator::with_default_damping(),
            gizmo: GizmoDriver::new(),
            gizmo_style: GizmoStyle::default(),
            handle_mesh: vpl::MeshId::INVALID,
            snapshots: Snapshots::default(),
            built: false,
            spline: vpl::SplineWidget::new(vec![
                glam::Vec3::new(-2.0, 0.0, 1.5),
                glam::Vec3::new(-0.5, 1.5, 1.5),
                glam::Vec3::new(0.5, -1.5, 1.5),
                glam::Vec3::new(2.0, 0.0, 1.5),
            ]),
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_interact_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.interact_state.scene = Scene::new();
        self.interact_state.selection.clear();

        let handle_sphere = vpl::primitives::icosphere(1.0, 2);
        self.interact_state.handle_mesh = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &handle_sphere)
            .expect("spline handle sphere upload");

        let positions = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
        ];
        let colours = [
            [0.45, 0.45, 0.45],
            [0.65, 0.09, 0.07],
            [0.10, 0.52, 0.18],
            [0.10, 0.26, 0.68],
            [0.60, 0.50, 0.05],
        ];
        let names = ["Centre", "Right", "Left", "Front", "Back"];

        for (i, ((pos, colour), name)) in positions.iter().zip(&colours).zip(&names).enumerate() {
            let mesh = self.upload_box(renderer);
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material::from_colour(Colour::from_linear_rgb_array(*colour));
            let id = self.interact_state.scene.add_named(
                name,
                Some(mesh),
                transform,
                upload_material(renderer, mat),
            );
            if i == 0 {
                self.interact_state.selection.select_one(id);
            }
        }

        self.interact_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Camera framing
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn zoom_to_fit_interact(&mut self) {
        let mut min = glam::Vec3::splat(f32::INFINITY);
        let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;

        let iter_nodes: Vec<_> = if !self.interact_state.selection.is_empty() {
            self.interact_state.selection.iter().copied().collect()
        } else {
            self.interact_state
                .scene
                .walk_depth_first()
                .iter()
                .map(|(id, _)| *id)
                .collect()
        };

        for nid in iter_nodes {
            if let Some(node) = self.interact_state.scene.node(nid) {
                let t = node.world_transform().to_mat4();
                let pos = glam::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z);
                min = min.min(pos - glam::Vec3::splat(0.6));
                max = max.max(pos + glam::Vec3::splat(0.6));
                any = true;
            }
        }

        if any {
            let aabb = vpl::Aabb { min, max };
            let target = self.camera.fit_aabb_target(&aabb);
            self.interact_state.animator.fly_to(
                &self.camera,
                target.centre,
                target.distance,
                target.orientation,
                0.6,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_interaction(app: &mut App, ui: &mut egui::Ui) {
    let gizmo = &mut app.interact_state.gizmo;
    ui.label("Gizmo Mode:");
    ui.horizontal(|ui| {
        for (mode, label) in [
            (GizmoMode::Translate, "Translate"),
            (GizmoMode::Rotate, "Rotate"),
            (GizmoMode::Scale, "Scale"),
        ] {
            if ui.radio(gizmo.mode() == mode, label).clicked() {
                gizmo.set_mode(mode);
            }
        }
    });

    ui.separator();

    ui.label("Gizmo Space:");
    ui.horizontal(|ui| {
        for (space, label) in [(GizmoSpace::World, "World"), (GizmoSpace::Local, "Local")] {
            if ui.radio(gizmo.space() == space, label).clicked() {
                gizmo.set_space(space);
            }
        }
    });

    ui.separator();
    ui.label("Shortcuts: G move | R rotate | S scale");
    ui.label("X / Y / Z : constrain axis  |  Enter / click : confirm  |  Esc : cancel");
    ui.separator();
    ui.label("View presets:");
    egui::Grid::new("view_presets_grid")
        .num_columns(4)
        .show(ui, |ui| {
            for (label, preset) in [
                ("Front", ViewPreset::Front),
                ("Back", ViewPreset::Back),
                ("Left", ViewPreset::Left),
                ("Right", ViewPreset::Right),
                ("Top", ViewPreset::Top),
                ("Bottom", ViewPreset::Bottom),
                ("Iso", ViewPreset::Isometric),
            ] {
                if ui.button(label).clicked() {
                    app.interact_state.animator.fly_to_full(
                        &app.camera,
                        app.camera.target(),
                        app.camera.focus_distance(),
                        preset.orientation(),
                        preset.prefers_orthographic().then(|| {
                            app.camera
                                .projection
                                .to_orthographic(app.camera.focus_distance())
                        }),
                        0.6,
                        Easing::EaseInOutCubic,
                    );
                }
            }
        });

    ui.separator();

    if ui.button("Zoom to Fit").clicked() {
        app.zoom_to_fit_interact();
    }

    ui.separator();

    if ui.button("Clear Selection").clicked() {
        app.interact_state.selection.clear();
    }
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn interact_collect_scene_items(
    app: &mut App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64, u64) {
    let items = app
        .interact_state
        .scene
        .collect_render_items(&app.interact_state.selection);
    let sg = app.interact_state.scene.version();
    let ss = app.interact_state.selection.version();
    let lighting = {
        let mut _t = LightingSettings::default();
        _t.hemisphere_intensity = 0.5;
        _t.sky_colour = Colour::linear_rgb(1.0, 1.0, 1.0);
        _t.ground_colour = Colour::linear_rgb(1.0, 1.0, 1.0);
        _t
    };
    (items, lighting, sg, ss)
}

pub(crate) fn interact_outline_selected(app: &App) -> bool {
    !app.interact_state.selection.is_empty()
}

pub(crate) fn submit_interact_items(app: &App, fd: &mut FrameData, w: f32, h: f32) {
    if !app.interact_state.built {
        return;
    }
    fd.scene
        .items_mut::<viewport_lib::PolylineItem>()
        .push(app.interact_state.spline.polyline_item(9900));
    let render_cam = CameraFrame::from_camera(&app.camera, [w, h]).render_camera;
    let spline_ctx = vpl::WidgetContext {
        camera: render_cam,
        viewport_size: glam::Vec2::new(w, h),
        cursor_viewport: app.cursor_viewport,
        drag_started: false,
        dragging: false,
        released: false,
        double_clicked: false,
    };
    fd.scene.mesh_instances.push(
        app.interact_state
            .spline
            .handle_markers(9901, &spline_ctx)
            .to_mesh_instances(app.interact_state.handle_mesh),
    );
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.interact_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_interact_scene(renderer);
    app.camera = vpl::Camera::orbit(
        glam::DVec3::ZERO,
        12.0,
        glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1),
    );
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. `_out` carries
/// the few extra frame settings a showcase can set alongside its items.
pub(crate) fn scene(
    app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    _out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    let (items, bg_colour, lighting, scene_gen, sel_gen) = {
        let (items, lighting, sg, ss) = interact_collect_scene_items(app);
        (items, None, lighting, sg, ss)
    };
    crate::SceneContents {
        items,
        bg_colour,
        lighting,
        scene_gen,
        sel_gen,
    }
}

// ---------------------------------------------------------------------------
// Per-frame frame-data tweaks
// ---------------------------------------------------------------------------

/// Fold this showcase's own contributions into the assembled frame: extra
/// render items, overlays, and effect settings that are re-submitted every
/// frame rather than baked into the scene.
pub(crate) fn frame(app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
    // Spline widget polyline + handles (Showcase 4) : submitted every frame.
    submit_interact_items(app, &mut *fd, ctx.w, ctx.h);

    // The gizmo and its readout, drawn as overlays.
    let state = &app.interact_state;
    let overlays = &mut fd.overlays;
    state.gizmo.build_view_overlays(
        &state.gizmo_style,
        &app.camera,
        glam::Vec2::new(ctx.w, ctx.h),
        true,
        &mut overlays.shapes,
        &mut overlays.polylines,
    );
    if let (Some(text), Some(centre)) = (state.gizmo.readout(), state.gizmo.gizmo_centre()) {
        overlays
            .labels
            .push(build_numeric_label(&state.gizmo_style, &text, centre));
    }
}

// ---------------------------------------------------------------------------
// Viewport overlay and per-frame tick
// ---------------------------------------------------------------------------

/// Advance this showcase's animation and ask for another frame. Runs after the
/// viewport has been drawn, so it only affects the next frame.
pub(crate) fn tick(app: &mut crate::App, cx: &crate::ViewportCtx) {
    // ----- Continuous repaint for animated camera and a running session -----
    if app.interact_state.animator.is_animating() || app.interact_state.gizmo.is_active() {
        cx.egui.request_repaint();
    }
}

/// Route a viewport click for this showcase. The host calls this for a plain
/// click that no gizmo or widget has already consumed; `pos` is in viewport
/// pixels.
pub(crate) fn on_click(app: &mut crate::App, cx: &crate::ClickCtx) {
    // Object-level selection: defer the pick to the render site, where the
    // renderer and the on-screen `FrameData` are in scope, and resolve it with
    // the unified GPU picker. See `apply_pending_pick`.
    app.pending_pick = Some(cx.pos);
}

/// Handle drag gestures this showcase owns, before the camera controller runs.

/// Advance this showcase's own camera animation or object motion for the frame.
pub(crate) fn advance(app: &mut crate::App, cx: &crate::ViewportCtx) {
    // ----- Advance camera animator (Showcases 4 and 10) -----
    let dt = cx.egui.input(|i| i.stable_dt.min(1.0 / 30.0));
    app.interact_state.animator.update(dt, &mut app.camera);
}

/// Update this showcase's interactive widgets for the frame.
pub(crate) fn widgets(app: &mut crate::App, cx: &crate::ViewportCtx) {
    // ----- Spline widget update (Showcase 4) -----
    if app.interact_state.built {
        let render_cam =
            vpl::CameraFrame::from_camera(&app.camera, [cx.rect.width(), cx.rect.height()])
                .render_camera;
        let widget_ctx = vpl::WidgetContext {
            camera: render_cam,
            viewport_size: glam::Vec2::new(cx.rect.width(), cx.rect.height()),
            cursor_viewport: app.cursor_viewport,
            drag_started: cx.response.drag_started(),
            dragging: cx.response.dragged(),
            released: cx.response.drag_stopped(),
            double_clicked: false,
        };
        app.interact_state.spline.update(&widget_ctx);
    }
}

/// Flush any per-frame GPU writes this showcase has queued.

/// Take over the whole viewport for this frame. Returning false leaves the
/// host's normal single-viewport path in charge.
pub(crate) fn viewport_override(
    _app: &mut crate::App,
    _ui: &mut crate::eframe::egui::Ui,
    _cx: &crate::ViewportCtx,
) -> bool {
    false
}

/// Drive the orbit controller for this showcase. Manipulation needs the
/// `ActionFrame` that driving the camera produces, so that the same frame
/// feeds both the camera and the gizmo; returning true tells the host this
/// showcase has taken care of the camera itself.
pub(crate) fn drive_camera(app: &mut crate::App, cx: &crate::ViewportCtx) -> bool {
    // One resolve per frame; the camera holds still while a gizmo session owns
    // the pointer.
    let action_frame = app.input.resolve();
    if !app.interact_state.gizmo.is_active() {
        let dt = cx.egui.input(|i| i.stable_dt);
        app.controller.apply(&mut app.camera, &action_frame, dt);
    }
    if !app.interact_state.built {
        return true;
    }

    let state = &mut app.interact_state;
    // Tab cycles the gizmo mode (ignored while a session runs).
    if action_frame.is_active(actions::CYCLE_GIZMO_MODE) {
        state.gizmo.cycle_mode();
    }

    let viewport_size = glam::Vec2::new(cx.rect.width(), cx.rect.height());
    let centre = scene_gizmo::selection_centre(&state.selection, &state.scene);
    let orientation = scene_gizmo::selection_orientation(&state.selection, &state.scene);
    let input = GizmoInput::from_action(&action_frame, &app.camera, viewport_size, centre)
        .with_selection_orientation(orientation);
    let mut committed = false;
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
        ManipResult::Commit => committed = true,
        _ => {}
    }
    // Idle: keep the snapshots current so G/R/S always starts clean.
    if !state.gizmo.is_active() {
        state.snapshots.save(&state.selection, &state.scene);
    }

    // Click-to-select, unless the click confirmed a session or a handle is
    // under the cursor.
    if cx.response.clicked()
        && !committed
        && !state.gizmo.is_active()
        && state.gizmo.hovered_axis() == GizmoAxis::None
    {
        let click_cx = crate::ClickCtx {
            frame: cx.frame,
            pos: app.cursor_viewport,
            w: viewport_size.x,
            h: viewport_size.y,
            pixels_per_point: cx.egui.pixels_per_point(),
        };
        app.handle_click_select(&click_cx);
    }
    true
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
pub(crate) struct ScInteraction;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScInteraction = ScInteraction;

impl crate::Showcase for ScInteraction {
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
    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
        frame(app, fd, ctx)
    }
    fn tick(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        tick(app, cx)
    }
    fn on_click(&self, app: &mut crate::App, cx: &crate::ClickCtx) {
        on_click(app, cx)
    }
    fn advance(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        advance(app, cx)
    }
    fn widgets(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        widgets(app, cx)
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
        controls_interaction(app, ui)
    }
}
