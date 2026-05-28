//! Showcase 13: Multi-Viewport : build, update, and controls.
//!
//! Renders the same scene from four independent cameras in a 2×2 quad layout:
//!
//! ```text
//! ┌─────────────────┬─────────────────┐
//! │  Perspective    │  Top (ortho)    │
//! │  (orbit)        │  looking −Z     │
//! ├─────────────────┼─────────────────┤
//! │  Front (ortho)  │  Right (ortho)  │
//! │  looking −Y     │  looking −X     │
//! └─────────────────┴─────────────────┘
//! ```
//!
//! Each quadrant has its own `OrbitCameraController`. Mouse input is routed to
//! whichever quad is under the cursor. The gizmo appears in all four viewports
//! simultaneously (each with its own screen-space scale), but hit-testing and
//! drag transforms are driven by the hovered quad's camera.

use crate::App;
use crate::multi_viewport_callback::MultiViewportCallback;
use eframe::egui;
use viewport_lib::{
    Action, ButtonState, Camera, CameraFrame, FrameData, Gizmo, GizmoAxis, GizmoInfo, GizmoMode,
    GizmoSpace, LightingSettings, ManipResult, ManipulationContext, ManipulationController,
    Material, Modifiers, MouseButton, NodeId, OrbitCameraController, Projection, SceneFrame,
    ScrollUnits, Selection, ViewportContext, ViewportEvent, ViewportRenderer,
    gizmo::{compute_gizmo_scale, gizmo_center_from_selection},
    picking::pick_scene_nodes_cpu,
    picking::screen_to_ray,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct MvState {
    pub scene: viewport_lib::scene::Scene,
    pub selection: Selection,
    pub cameras: [Camera; 4],
    pub controllers: [OrbitCameraController; 4],
    pub viewports: Option<[viewport_lib::ViewportId; 4]>,
    pub built: bool,
    pub gizmo: Gizmo,
    pub hovered_quad: usize,
    pub cursor_local: glam::Vec2,
    pub gizmo_center: Option<glam::Vec3>,
    pub gizmo_scales: [f32; 4],
    pub manip: ManipulationController,
    pub transforms_snapshot: std::collections::HashMap<NodeId, glam::Mat4>,
    pub left_held: bool,
}

impl Default for MvState {
    fn default() -> Self {
        Self {
            scene: viewport_lib::scene::Scene::new(),
            selection: Selection::new(),
            cameras: [
                Camera {
                    center: glam::Vec3::ZERO,
                    distance: 12.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                },
                Camera {
                    center: glam::Vec3::ZERO,
                    distance: 10.0,
                    orientation: glam::Quat::IDENTITY,
                    projection: Projection::Orthographic,
                    ..Camera::default()
                },
                Camera {
                    center: glam::Vec3::ZERO,
                    distance: 10.0,
                    orientation: glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    projection: Projection::Orthographic,
                    ..Camera::default()
                },
                Camera {
                    center: glam::Vec3::ZERO,
                    distance: 10.0,
                    orientation: glam::Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)
                        * glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    projection: Projection::Orthographic,
                    ..Camera::default()
                },
            ],
            controllers: [
                OrbitCameraController::viewport_all(),
                OrbitCameraController::viewport_all(),
                OrbitCameraController::viewport_all(),
                OrbitCameraController::viewport_all(),
            ],
            viewports: None,
            built: false,
            gizmo: Gizmo::new(),
            hovered_quad: 0,
            cursor_local: glam::Vec2::ZERO,
            gizmo_center: None,
            gizmo_scales: [1.0; 4],
            manip: ManipulationController::new(),
            transforms_snapshot: std::collections::HashMap::new(),
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
        self.mv_state.scene = viewport_lib::scene::Scene::new();
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
            [0.85, 0.85, 0.85],
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.3],
            [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3],
            [0.9, 0.5, 0.1],
        ];
        let names = ["Center", "Right", "Left", "Front", "Back", "Top"];

        for (i, ((pos, colour), name)) in positions.iter().zip(&colours).zip(&names).enumerate() {
            let mesh = self.upload_box(renderer);
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material::from_colour(*colour);
            let id = self
                .mv_state
                .scene
                .add_named(name, Some(mesh), transform, mat);
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
        response: egui::Response,
        _frame: &eframe::Frame,
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

        // begin_frame for all four controllers.
        for i in 0..4 {
            let qr = quad_rects[i];
            self.mv_state.controllers[i].begin_frame(ViewportContext {
                hovered: i == hq && response.hovered(),
                focused: i == hq && response.hovered(),
                viewport_size: [qr.width(), qr.height()],
            });
        }

        // Notify the previously hovered quad that the pointer left.
        if hq != prev_hq {
            self.mv_state.controllers[prev_hq].push_event(ViewportEvent::PointerLeft);
        }

        let manip_active_for_text = self.mv_state.manip.is_active();

        // Push this frame's input events to the hovered controller.
        ui.input(|i| {
            let mods = Modifiers {
                alt: i.modifiers.alt,
                shift: i.modifiers.shift,
                ctrl: i.modifiers.command,
            };
            self.mv_state.controllers[hq].push_event(ViewportEvent::ModifiersChanged(mods));
            self.mv_state.controllers[hq].push_event(ViewportEvent::PointerMoved {
                position: self.mv_state.cursor_local,
            });

            for event in &i.events {
                match event {
                    egui::Event::Key {
                        key,
                        pressed,
                        repeat,
                        ..
                    } => {
                        if let Some(kc) = crate::shared::egui_key_to_keycode(*key) {
                            self.mv_state.controllers[hq].push_event(ViewportEvent::Key {
                                key: kc,
                                state: if *pressed {
                                    ButtonState::Pressed
                                } else {
                                    ButtonState::Released
                                },
                                repeat: *repeat,
                            });
                        }
                    }

                    egui::Event::Text(text) if manip_active_for_text => {
                        for c in text.chars() {
                            self.mv_state.controllers[hq].push_event(ViewportEvent::Character(c));
                        }
                    }

                    egui::Event::PointerButton {
                        button,
                        pressed,
                        pos,
                        ..
                    } => {
                        let vp_button = match button {
                            egui::PointerButton::Primary => MouseButton::Left,
                            egui::PointerButton::Secondary => MouseButton::Right,
                            egui::PointerButton::Middle => MouseButton::Middle,
                            _ => continue,
                        };

                        // Ignore presses that originate outside the full viewport.
                        if *pressed && !rect.contains(*pos) {
                            continue;
                        }

                        // Track raw left-button held state for ManipulationContext.
                        if *button == egui::PointerButton::Primary {
                            self.mv_state.left_held = *pressed;
                        }

                        let state = if *pressed {
                            ButtonState::Pressed
                        } else {
                            ButtonState::Released
                        };
                        self.mv_state.controllers[hq].push_event(ViewportEvent::MouseButton {
                            button: vp_button,
                            state,
                        });
                    }

                    egui::Event::MouseWheel { unit, delta, .. } => {
                        let over_vp = hover_pos.map(|p| rect.contains(p)).unwrap_or(false);
                        if over_vp {
                            let units = match unit {
                                egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                                egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                                egui::MouseWheelUnit::Page => ScrollUnits::Pages,
                            };
                            self.mv_state.controllers[hq].push_event(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units,
                            });
                        }
                    }

                    _ => {}
                }
            }
        });

        // ----- ManipulationController update (G/R/S) -----
        {
            let qr = quad_rects[hq];
            let w = qr.width();
            let h = qr.height();
            let viewport_size = glam::Vec2::new(w, h);

            // Per-frame gizmo hover when no session is active.
            if !self.mv_state.manip.is_active() {
                if let Some(center) = self.mv_state.gizmo_center {
                    let cam = &self.mv_state.cameras[hq];
                    let ray_origin = cam.eye_position();
                    let view_proj = cam.proj_matrix() * cam.view_matrix();
                    let cursor = self.mv_state.cursor_local;
                    let ndc_x = (cursor.x / w.max(1.0)) * 2.0 - 1.0;
                    let ndc_y = 1.0 - (cursor.y / h.max(1.0)) * 2.0;
                    let inv_vp = view_proj.inverse();
                    let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
                    let ray_dir = (far - ray_origin).normalize_or_zero();
                    let orient = mv_gizmo_orientation(
                        self.mv_state.gizmo.space,
                        &self.mv_state.selection,
                        &self.mv_state.scene,
                    );
                    self.mv_state.gizmo.hovered_axis = self.mv_state.gizmo.hit_test_oriented(
                        ray_origin,
                        ray_dir,
                        center,
                        self.mv_state.gizmo_scales[hq],
                        orient,
                    );
                } else {
                    self.mv_state.gizmo.hovered_axis = GizmoAxis::None;
                }
            }

            // Build GizmoInfo.
            let orient = mv_gizmo_orientation(
                self.mv_state.gizmo.space,
                &self.mv_state.selection,
                &self.mv_state.scene,
            );
            let gizmo_info = self.mv_state.gizmo_center.map(|center| GizmoInfo {
                center,
                scale: self.mv_state.gizmo_scales[hq],
                orientation: orient,
                mode: self.mv_state.gizmo.mode,
            });

            // Build ManipulationContext.
            let pointer_delta =
                ui.input(|i| glam::Vec2::new(i.pointer.delta().x, i.pointer.delta().y));
            let manip_ctx = ManipulationContext {
                camera: self.mv_state.cameras[hq].clone(),
                viewport_size,
                cursor_viewport: Some(self.mv_state.cursor_local),
                pointer_delta,
                selection_center: self.mv_state.gizmo_center,
                gizmo: gizmo_info,
                drag_started: response.drag_started(),
                dragging: self.mv_state.left_held,
                clicked: response.clicked(),
            };

            // Orbit: resolve hq controller while manipulation is active.
            let action_frame = if self.mv_state.manip.is_active() {
                self.mv_state.controllers[hq].resolve()
            } else {
                self.mv_state.controllers[hq].apply_to_camera(&mut self.mv_state.cameras[hq])
            };

            // Tab cycles gizmo mode when no session is active.
            if !self.mv_state.manip.is_active() && action_frame.is_active(Action::CycleGizmoMode) {
                self.mv_state.gizmo.mode = match self.mv_state.gizmo.mode {
                    GizmoMode::Translate => GizmoMode::Rotate,
                    GizmoMode::Rotate => GizmoMode::Scale,
                    GizmoMode::Scale => GizmoMode::Translate,
                    _ => GizmoMode::Translate,
                };
            }

            match self.mv_state.manip.update(&action_frame, manip_ctx) {
                ManipResult::Update(delta) => {
                    self.apply_mv_delta(delta, hq);
                }
                ManipResult::Cancel | ManipResult::ConstraintChanged => {
                    self.restore_mv_snapshots();
                }
                ManipResult::Commit => {
                    self.save_mv_snapshots();
                }
                ManipResult::None => {
                    if !self.mv_state.manip.is_active() {
                        self.save_mv_snapshots();
                    }
                }
                _ => {}
            }

            // Apply remaining controllers (non-hq); hq already applied/resolved above.
            for i in 0..4 {
                if i != hq {
                    self.mv_state.controllers[i].apply_to_camera(&mut self.mv_state.cameras[i]);
                }
                self.mv_state.cameras[i]
                    .set_aspect_ratio(quad_rects[i].width(), quad_rects[i].height());
            }

            // Click-to-select only when no session is active.
            if response.clicked() && !self.mv_state.manip.is_active() {
                self.handle_mv_click(hq, w, h);
            }
        }

        // Update shared gizmo center and per-quad screen-space scales.
        self.mv_state.gizmo_center = gizmo_center_from_selection(&self.mv_state.selection, |id| {
            self.mv_state.scene.node(id).map(|n| {
                let t = n.world_transform();
                glam::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z)
            })
        });
        for i in 0..4 {
            if let Some(center) = self.mv_state.gizmo_center {
                let qr = quad_rects[i];
                self.mv_state.gizmo_scales[i] = compute_gizmo_scale(
                    center,
                    self.mv_state.cameras[i].eye_position(),
                    self.mv_state.cameras[i].fov_y,
                    qr.height(),
                );
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
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        let ppp = ui.ctx().pixels_per_point();
        let scene_gen = self.mv_state.scene.version();
        let sel_gen = self.mv_state.selection.version();
        let has_selection = !self.mv_state.selection.is_empty();
        let gizmo_center = self.mv_state.gizmo_center;
        let gizmo_scales = self.mv_state.gizmo_scales;
        let gizmo_mode = self.mv_state.gizmo.mode;
        let gizmo_hovered_axis = if let Some(state) = self.mv_state.manip.state() {
            state.axis.unwrap_or(GizmoAxis::None)
        } else {
            self.mv_state.gizmo.hovered_axis
        };
        let gizmo_orient = mv_gizmo_orientation(
            self.mv_state.gizmo.space,
            &self.mv_state.selection,
            &self.mv_state.scene,
        );

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
            fd.viewport.background_colour = Some(crate::BG_COLOUR);
            fd.interaction.outline_selected = has_selection;
            fd.scene.generation = scene_gen;
            fd.interaction.selection_generation = sel_gen;

            // Gizmo : appears in every viewport with its own screen-space scale.
            if let Some(center) = gizmo_center {
                let scale = gizmo_scales[i];
                fd.interaction.gizmo_model = Some(glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::splat(scale),
                    gizmo_orient,
                    center,
                ));
                fd.interaction.gizmo_mode = gizmo_mode;
                fd.interaction.gizmo_space_orientation = gizmo_orient;
                fd.interaction.gizmo_hovered = gizmo_hovered_axis;
            }

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

        // Schedule the multi-viewport paint callback over the full panel.
        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(
                rect,
                MultiViewportCallback { frames, viewports },
            ));
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

    ui.label("Gizmo Mode:");
    ui.horizontal(|ui| {
        if ui
            .radio(app.mv_state.gizmo.mode == GizmoMode::Translate, "Move")
            .clicked()
        {
            app.mv_state.gizmo.mode = GizmoMode::Translate;
        }
        if ui
            .radio(app.mv_state.gizmo.mode == GizmoMode::Rotate, "Rotate")
            .clicked()
        {
            app.mv_state.gizmo.mode = GizmoMode::Rotate;
        }
        if ui
            .radio(app.mv_state.gizmo.mode == GizmoMode::Scale, "Scale")
            .clicked()
        {
            app.mv_state.gizmo.mode = GizmoMode::Scale;
        }
    });

    ui.separator();
    ui.label("Space:");
    ui.horizontal(|ui| {
        if ui
            .radio(app.mv_state.gizmo.space == GizmoSpace::World, "World")
            .clicked()
        {
            app.mv_state.gizmo.space = GizmoSpace::World;
        }
        if ui
            .radio(app.mv_state.gizmo.space == GizmoSpace::Local, "Local")
            .clicked()
        {
            app.mv_state.gizmo.space = GizmoSpace::Local;
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

impl App {
    fn handle_mv_click(&mut self, quad: usize, w: f32, h: f32) {
        let cam = &self.mv_state.cameras[quad];
        let vp_inv = cam.view_proj_matrix().inverse();
        let (ray_origin, ray_dir) =
            screen_to_ray(self.mv_state.cursor_local, glam::Vec2::new(w, h), vp_inv);

        let mut mesh_lookup = std::collections::HashMap::new();
        for node in self.mv_state.scene.nodes() {
            if let Some(mid) = viewport_lib::traits::ViewportObject::mesh_id(node) {
                mesh_lookup.entry(mid).or_insert_with(|| {
                    (
                        self.box_mesh_data.positions.clone(),
                        self.box_mesh_data.indices.clone(),
                    )
                });
            }
        }

        let hit = pick_scene_nodes_cpu(ray_origin, ray_dir, &self.mv_state.scene, &mesh_lookup);
        if let Some(hit) = hit {
            self.mv_state.selection.select_one(hit.id);
        } else {
            self.mv_state.selection.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Manipulation helpers
// ---------------------------------------------------------------------------

impl crate::App {
    /// Apply a [`viewport_lib::TransformDelta`] to all selected mv scene nodes.
    pub(crate) fn apply_mv_delta(&mut self, delta: viewport_lib::TransformDelta, hq: usize) {
        let Some(center) = self.mv_state.gizmo_center else {
            return;
        };

        let has_pos_override = delta.position_override.iter().any(|v| v.is_some());
        let has_scale_override = delta.scale_override.iter().any(|v| v.is_some());

        if has_pos_override || has_scale_override {
            self.restore_mv_snapshots();
        }

        let translation = if has_pos_override {
            glam::Vec3::new(
                delta.position_override[0].unwrap_or(0.0),
                delta.position_override[1].unwrap_or(0.0),
                delta.position_override[2].unwrap_or(0.0),
            )
        } else {
            delta.translation
        };

        let scale = if has_scale_override {
            glam::Vec3::new(
                delta.scale_override[0].unwrap_or(1.0),
                delta.scale_override[1].unwrap_or(1.0),
                delta.scale_override[2].unwrap_or(1.0),
            )
        } else {
            delta.scale
        };

        let _ = hq;
        let rot_mat = glam::Mat4::from_quat(delta.rotation);
        let scale_mat = glam::Mat4::from_scale(scale);
        let translate_mat = glam::Mat4::from_translation(translation);
        let to_pivot = glam::Mat4::from_translation(-center);
        let from_pivot = glam::Mat4::from_translation(center);

        for id in self.mv_state.selection.iter().copied().collect::<Vec<_>>() {
            if let Some(node) = self.mv_state.scene.node(id) {
                let cur = node.local_transform();
                let new_t = translate_mat * from_pivot * rot_mat * scale_mat * to_pivot * cur;
                self.mv_state.scene.set_local_transform(id, new_t);
            }
        }
        self.mv_state.scene.update_transforms();
    }

    /// Snapshot the current local transforms of all selected mv nodes.
    pub(crate) fn save_mv_snapshots(&mut self) {
        self.mv_state.transforms_snapshot.clear();
        for id in self.mv_state.selection.iter().copied().collect::<Vec<_>>() {
            if let Some(node) = self.mv_state.scene.node(id) {
                self.mv_state
                    .transforms_snapshot
                    .insert(id, node.local_transform());
            }
        }
    }

    /// Restore local transforms from the last mv snapshot.
    pub(crate) fn restore_mv_snapshots(&mut self) {
        let ids: Vec<_> = self.mv_state.transforms_snapshot.keys().copied().collect();
        for id in ids {
            if let Some(&t) = self.mv_state.transforms_snapshot.get(&id) {
                self.mv_state.scene.set_local_transform(id, t);
            }
        }
        self.mv_state.scene.update_transforms();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gizmo orientation: identity for World space; first selected node's rotation for Local.
fn mv_gizmo_orientation(
    space: GizmoSpace,
    selection: &viewport_lib::Selection,
    scene: &viewport_lib::scene::Scene,
) -> glam::Quat {
    match space {
        GizmoSpace::World => glam::Quat::IDENTITY,
        GizmoSpace::Local => selection
            .primary()
            .and_then(|id| scene.node(id))
            .map(|n| glam::Quat::from_mat4(&n.world_transform()))
            .unwrap_or(glam::Quat::IDENTITY),
    }
}
