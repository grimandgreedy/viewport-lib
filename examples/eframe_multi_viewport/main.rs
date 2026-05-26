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

mod multi_viewport_callback;

use eframe::egui;
use std::collections::HashMap;
use viewport_lib::{
    Action, BackfacePolicy, ButtonState, Camera, CameraFrame, FrameData, Gizmo, GizmoAxis,
    GizmoInfo, GizmoMode, GizmoSpace, KeyCode, LightingSettings, ManipResult, ManipulationContext,
    ManipulationController, Material, MeshId, Modifiers, MouseButton, OrbitCameraController,
    Projection, SceneFrame, SceneRenderItem, ScrollUnits, Selection,
    ViewportContext, ViewportEvent, ViewportId, ViewportRenderer, gizmo::compute_gizmo_scale,
    gizmo_center_for_pivot, picking::screen_to_ray, primitives,
};

const QUAD_LABELS: [&str; 4] = ["Perspective", "Top", "Front", "Right"];

fn egui_key_to_keycode(key: egui::Key) -> Option<KeyCode> {
    match key {
        egui::Key::G => Some(KeyCode::G),
        egui::Key::R => Some(KeyCode::R),
        egui::Key::S => Some(KeyCode::S),
        egui::Key::X => Some(KeyCode::X),
        egui::Key::Y => Some(KeyCode::Y),
        egui::Key::Z => Some(KeyCode::Z),
        egui::Key::Tab => Some(KeyCode::Tab),
        egui::Key::Enter => Some(KeyCode::Enter),
        egui::Key::Escape => Some(KeyCode::Escape),
        egui::Key::Period => Some(KeyCode::Period),
        egui::Key::Minus => Some(KeyCode::Minus),
        _ => None,
    }
}

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
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let device = &rs.device;
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);
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

            let vp_ids = [
                renderer.create_viewport(device),
                renderer.create_viewport(device),
                renderer.create_viewport(device),
                renderer.create_viewport(device),
            ];

            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App::new(vp_ids, m_sphere, m_cube, m_torus)))
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
        glam::Vec3::new(self.model.w_axis.x, self.model.w_axis.y, self.model.w_axis.z)
    }

    fn rotation(&self) -> glam::Quat {
        glam::Quat::from_mat4(&self.model)
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    objects: Vec<SceneObject>,
    snapshots: HashMap<u64, glam::Mat4>,
    selection: Selection,

    cameras: [Camera; 4],
    controllers: [OrbitCameraController; 4],
    gizmo: Gizmo,
    gizmo_center: Option<glam::Vec3>,
    gizmo_scales: [f32; 4],
    manip: ManipulationController,

    vp_ids: [ViewportId; 4],

    hovered_quad: usize,
    cursor_local: glam::Vec2,
    left_held: bool,
}

impl App {
    fn new(
        vp_ids: [ViewportId; 4],
        m_sphere: MeshId,
        m_cube: MeshId,
        m_torus: MeshId,
    ) -> Self {
        use std::f32::consts::FRAC_PI_2;
        let cameras = [
            // Perspective (TL)
            Camera {
                center: glam::Vec3::ZERO,
                distance: 12.0,
                orientation: glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1),
                ..Camera::default()
            },
            // Top (TR): looking down -Z, orthographic
            Camera {
                center: glam::Vec3::ZERO,
                distance: 10.0,
                orientation: glam::Quat::IDENTITY,
                projection: Projection::Orthographic,
                ..Camera::default()
            },
            // Front (BL): looking along -Y, orthographic
            Camera {
                center: glam::Vec3::ZERO,
                distance: 10.0,
                orientation: glam::Quat::from_rotation_x(FRAC_PI_2),
                projection: Projection::Orthographic,
                ..Camera::default()
            },
            // Right (BR): looking along -X, orthographic
            Camera {
                center: glam::Vec3::ZERO,
                distance: 10.0,
                orientation: glam::Quat::from_rotation_z(-FRAC_PI_2) * glam::Quat::from_rotation_x(FRAC_PI_2),
                projection: Projection::Orthographic,
                ..Camera::default()
            },
        ];

        Self {
            objects: vec![
                SceneObject::new(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2], 0.65),
                SceneObject::new(m_cube,   [ 0.0, 0.0, 0.0], [0.4, 0.6, 0.9], 0.87),
                SceneObject::new(m_torus,  [ 2.5, 0.0, 0.0], [0.3, 0.8, 0.4], 0.70),
            ],
            snapshots: HashMap::new(),
            selection: Selection::new(),
            cameras,
            controllers: [
                OrbitCameraController::viewport_all(),
                OrbitCameraController::viewport_all(),
                OrbitCameraController::viewport_all(),
                OrbitCameraController::viewport_all(),
            ],
            gizmo: Gizmo::new(),
            gizmo_center: None,
            gizmo_scales: [1.0; 4],
            manip: ManipulationController::new(),
            vp_ids,
            hovered_quad: 0,
            cursor_local: glam::Vec2::ZERO,
            left_held: false,
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

    fn apply_delta(&mut self, delta: viewport_lib::TransformDelta) {
        let Some(center) = self.gizmo_center else { return };

        let has_pos_override = delta.position_override.iter().any(|v| v.is_some());
        let has_scale_override = delta.scale_override.iter().any(|v| v.is_some());

        if has_pos_override || has_scale_override {
            self.restore_snapshots();
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

        let rot_mat = glam::Mat4::from_quat(delta.rotation);
        let scale_mat = glam::Mat4::from_scale(scale);
        let translate_mat = glam::Mat4::from_translation(translation);
        let to_pivot = glam::Mat4::from_translation(-center);
        let from_pivot = glam::Mat4::from_translation(center);

        for &id in self.selection.iter().copied().collect::<Vec<_>>().iter() {
            if let Some(obj) = self.objects.get_mut(id as usize) {
                obj.model = translate_mat * from_pivot * rot_mat * scale_mat * to_pivot * obj.model;
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
            let center = obj.position();
            let r = obj.pick_radius;
            let oc = ray_origin - center;
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

    fn gizmo_orientation(&self) -> glam::Quat {
        match self.gizmo.space {
            GizmoSpace::World => glam::Quat::IDENTITY,
            GizmoSpace::Local => self
                .selection
                .primary()
                .and_then(|id| self.objects.get(id as usize))
                .map(|o| o.rotation())
                .unwrap_or(glam::Quat::IDENTITY),
        }
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let ppp = ctx.pixels_per_point();

        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = {
                let r = ui.available_rect_before_wrap();
                let (r, _) = ui.allocate_exact_size(r.size(), egui::Sense::click_and_drag());
                r
            };

            let quads = quad_rects(rect);

            // Determine hovered quad from cursor position.
            let hover_pos = ui.input(|i| i.pointer.hover_pos());
            let prev_hq = self.hovered_quad;
            if let Some(pos) = hover_pos {
                if rect.contains(pos) {
                    if let Some(q) = quads.iter().position(|qr| qr.contains(pos)) {
                        self.hovered_quad = q;
                        let qr = quads[q];
                        self.cursor_local = glam::Vec2::new(pos.x - qr.left(), pos.y - qr.top());
                    }
                }
            }
            let hq = self.hovered_quad;

            // begin_frame for every controller.
            for i in 0..4 {
                let qr = quads[i];
                self.controllers[i].begin_frame(ViewportContext {
                    hovered: i == hq && ui.rect_contains_pointer(rect),
                    focused: i == hq && ui.rect_contains_pointer(rect),
                    viewport_size: [qr.width(), qr.height()],
                });
            }

            // Notify the previously hovered quad on quad switch.
            if hq != prev_hq {
                self.controllers[prev_hq].push_event(ViewportEvent::PointerLeft);
            }

            let manip_active_for_text = self.manip.is_active();

            // Route input events to the hovered quad's controller.
            ui.input(|i| {
                let mods = Modifiers {
                    alt: i.modifiers.alt,
                    shift: i.modifiers.shift,
                    ctrl: i.modifiers.command,
                };
                self.controllers[hq].push_event(ViewportEvent::ModifiersChanged(mods));
                self.controllers[hq].push_event(ViewportEvent::PointerMoved {
                    position: self.cursor_local,
                });

                for event in &i.events {
                    match event {
                        egui::Event::Key {
                            key,
                            pressed,
                            repeat,
                            ..
                        } => {
                            if let Some(kc) = egui_key_to_keycode(*key) {
                                self.controllers[hq].push_event(ViewportEvent::Key {
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
                                self.controllers[hq].push_event(ViewportEvent::Character(c));
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
                            if *pressed && !rect.contains(*pos) {
                                continue;
                            }
                            if *button == egui::PointerButton::Primary {
                                self.left_held = *pressed;
                            }
                            self.controllers[hq].push_event(ViewportEvent::MouseButton {
                                button: vp_button,
                                state: if *pressed {
                                    ButtonState::Pressed
                                } else {
                                    ButtonState::Released
                                },
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
                                self.controllers[hq].push_event(ViewportEvent::Wheel {
                                    delta: glam::Vec2::new(delta.x, delta.y),
                                    units,
                                });
                            }
                        }
                        _ => {}
                    }
                }
            });

            // Gizmo hit test in the hovered quad (only when no session is active).
            if !self.manip.is_active() {
                if let Some(center) = self.gizmo_center {
                    let cam = &self.cameras[hq];
                    let qr = quads[hq];
                    let w = qr.width();
                    let h = qr.height();
                    let ndc_x = (self.cursor_local.x / w.max(1.0)) * 2.0 - 1.0;
                    let ndc_y = 1.0 - (self.cursor_local.y / h.max(1.0)) * 2.0;
                    let vp_inv = (cam.proj_matrix() * cam.view_matrix()).inverse();
                    let far = vp_inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
                    let ray_origin = cam.eye_position();
                    let ray_dir = (far - ray_origin).normalize_or_zero();
                    self.gizmo.hovered_axis = self.gizmo.hit_test(
                        ray_origin,
                        ray_dir,
                        center,
                        self.gizmo_scales[hq],
                    );
                } else {
                    self.gizmo.hovered_axis = GizmoAxis::None;
                }
            }

            let orient = self.gizmo_orientation();
            let gizmo_info = self.gizmo_center.map(|center| GizmoInfo {
                center,
                scale: self.gizmo_scales[hq],
                orientation: orient,
                mode: self.gizmo.mode,
            });

            let pointer_delta =
                ui.input(|i| glam::Vec2::new(i.pointer.delta().x, i.pointer.delta().y));
            let qr = quads[hq];
            let manip_ctx = ManipulationContext {
                camera: self.cameras[hq].clone(),
                viewport_size: glam::Vec2::new(qr.width(), qr.height()),
                cursor_viewport: Some(self.cursor_local),
                pointer_delta,
                selection_center: self.gizmo_center,
                gizmo: gizmo_info,
                drag_started: ui.input(|i| i.pointer.any_pressed()),
                dragging: self.left_held,
                clicked: ui.input(|i| i.pointer.any_click()),
            };

            // Apply hovered controller (or resolve while manipulating).
            let action_frame = if self.manip.is_active() {
                self.controllers[hq].resolve()
            } else {
                self.controllers[hq].apply_to_camera(&mut self.cameras[hq])
            };

            // Tab cycles gizmo mode when no session is active.
            if !self.manip.is_active() && action_frame.is_active(Action::CycleGizmoMode) {
                self.gizmo.mode = match self.gizmo.mode {
                    GizmoMode::Translate => GizmoMode::Rotate,
                    GizmoMode::Rotate => GizmoMode::Scale,
                    GizmoMode::Scale => GizmoMode::Translate,
                    _ => GizmoMode::Translate,
                };
            }

            match self.manip.update(&action_frame, manip_ctx) {
                ManipResult::Update(delta) => {
                    self.apply_delta(delta);
                }
                ManipResult::Cancel | ManipResult::ConstraintChanged => {
                    self.restore_snapshots();
                }
                ManipResult::Commit => {
                    self.save_snapshots();
                }
                ManipResult::None => {
                    if !self.manip.is_active() {
                        self.save_snapshots();
                    }
                }
            }

            // Apply remaining cameras and set aspect ratios.
            for i in 0..4 {
                if i != hq {
                    self.controllers[i].apply_to_camera(&mut self.cameras[i]);
                }
                let qr = quads[i];
                self.cameras[i].set_aspect_ratio(qr.width(), qr.height());
            }

            // Click to select (only when no session is active).
            if ui.input(|i| i.pointer.any_click()) && !self.manip.is_active() {
                let qr = quads[hq];
                if let Some(id) = self.pick(hq, qr.width(), qr.height()) {
                    self.selection.select_one(id);
                } else {
                    self.selection.clear();
                }
            }

            // Update gizmo center and per-viewport gizmo scales.
            self.gizmo_center = gizmo_center_for_pivot(
                &self.gizmo.pivot_mode,
                &self.selection,
                |id| self.position_of(id),
            );
            for i in 0..4 {
                if let Some(center) = self.gizmo_center {
                    let qr = quads[i];
                    self.gizmo_scales[i] = compute_gizmo_scale(
                        center,
                        self.cameras[i].eye_position(),
                        self.cameras[i].fov_y,
                        qr.height(),
                    );
                }
            }

            // Gizmo hovered axis for rendering: use active session axis if one is in progress.
            let gizmo_hovered_axis = if let Some(state) = self.manip.state() {
                state.axis.unwrap_or(GizmoAxis::None)
            } else {
                self.gizmo.hovered_axis
            };

            // Build scene render items.
            let scene_items: Vec<SceneRenderItem> = self
                .objects
                .iter()
                .enumerate()
                .map(|(i, obj)| {
                    let mut item = SceneRenderItem::default();
                    item.mesh_id = obj.mesh_id;
                    item.model = obj.model.to_cols_array_2d();
                    item.material = Material::from_colour(obj.colour);
                    item.material.backface_policy = BackfacePolicy::Identical;
                    item.settings.selected = self.selection.contains(i as u64);
                    item
                })
                .collect();

            let gizmo_center = self.gizmo_center;
            let gizmo_scales = self.gizmo_scales;
            let gizmo_mode = self.gizmo.mode;
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
                fd.effects.post_process.enabled = true;
                fd.viewport.show_grid = true;
                fd.viewport.show_axes_indicator = true;
                fd.interaction.outline_selected = has_selection;
                fd.interaction.selection_generation = sel_gen;

                // Gizmo: rendered in every viewport with its own screen-space scale.
                if let Some(center) = gizmo_center {
                    let scale = gizmo_scales[i];
                    fd.interaction.gizmo_model = Some(glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(scale),
                        orient,
                        center,
                    ));
                    fd.interaction.gizmo_mode = gizmo_mode;
                    fd.interaction.gizmo_space_orientation = orient;
                    fd.interaction.gizmo_hovered = gizmo_hovered_axis;
                }

                fd
            });

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    multi_viewport_callback::MultiViewportCallback {
                        frames,
                        viewports: self.vp_ids,
                    },
                ));

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
                egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(180, 180, 255, 60)),
                egui::StrokeKind::Middle,
            );

            // Manipulation status in the active quad.
            if let Some(ms) = self.manip.state() {
                let kind = match ms.kind {
                    viewport_lib::ManipulationKind::Move => "Move",
                    viewport_lib::ManipulationKind::Rotate => "Rotate",
                    viewport_lib::ManipulationKind::Scale => "Scale",
                };
                let axis = match ms.axis {
                    Some(GizmoAxis::X) => if ms.exclude_axis { " (YZ)" } else { " (X)" },
                    Some(GizmoAxis::Y) => if ms.exclude_axis { " (XZ)" } else { " (Y)" },
                    Some(GizmoAxis::Z) => if ms.exclude_axis { " (XY)" } else { " (Z)" },
                    _ => "",
                };
                let text = if let Some(ref num) = ms.numeric_display {
                    format!("{kind}{axis}: {num}")
                } else {
                    format!("{kind}{axis}")
                };
                let aqr = quads[hq];
                let font = egui::FontId::proportional(13.0);
                let galley = painter.layout_no_wrap(text, font, egui::Color32::WHITE);
                let pos = egui::pos2(
                    aqr.center().x - galley.size().x * 0.5,
                    aqr.max.y - 28.0,
                );
                let bg = egui::Rect::from_min_size(
                    pos - egui::vec2(6.0, 3.0),
                    galley.size() + egui::vec2(12.0, 6.0),
                );
                painter.rect_filled(bg, 3.0, egui::Color32::from_black_alpha(180));
                painter.galley(pos, galley, egui::Color32::WHITE);
                ctx.request_repaint();
            }
        });
    }
}
