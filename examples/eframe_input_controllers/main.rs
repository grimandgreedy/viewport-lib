//! Demonstrates the two built-in input presets.
//!
//! Press **1** : Camera only (`ViewportPrimitives`):
//!   Left/Middle drag -> Orbit  ·  Right drag -> Pan  ·  Scroll -> Zoom
//!   Ctrl+Scroll -> Orbit (two-axis)  ·  Shift+Scroll -> Pan (two-axis)
//!
//! Press **2** : Full controls (`ViewportAll`) + `ManipulationController`:
//!   Ctrl+Scroll -> Orbit  ·  Right drag -> Pan  ·  Scroll -> Zoom
//!   Click -> select  ·  Shift+Click -> multi-select
//!   G move  R rotate  S scale  ·  X/Y/Z constrain  ·  Shift+X/Y/Z exclude
//!   Enter / left-click -> confirm  ·  Esc -> cancel
//!   [ -> cycle pivot forward  ·  ] -> cycle pivot backward

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    Action, ButtonState, Camera, CameraFrame, FrameData, Gizmo, GizmoInfo, InteractionFrame,
    KeyCode, LightingSettings, ManipResult, ManipulationContext, ManipulationController,
    ManipulationKind, Material, OrbitCameraController, PivotMode, SceneFrame, SceneRenderItem,
    ScrollUnits, Selection, ViewportContext, ViewportEvent, ViewportRenderer,
    gizmo_center_for_pivot, primitives,
};

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Input Controllers",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 600.0]),
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
            let m_box = res
                .upload_mesh_data(device, &primitives::cube(1.0))
                .unwrap();
            let m_sphere = res
                .upload_mesh_data(device, &primitives::sphere(0.75, 32, 16))
                .unwrap();

            rs.renderer.write().callback_resources.insert(renderer);
            Ok(Box::new(App::new(m_box, m_sphere)))
        }),
    )
}

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Camera navigation only : ViewportPrimitives preset.
    Primitives,
    /// Full controls + object manipulation : ViewportAll preset.
    All,
}

// ---------------------------------------------------------------------------
// Scene object
// ---------------------------------------------------------------------------

struct Object {
    model: glam::Mat4,
    snapshot: glam::Mat4,
    mesh: usize,
    color: [f32; 3],
}

impl Object {
    fn new(pos: glam::Vec3, mesh: usize, color: [f32; 3]) -> Self {
        let model = glam::Mat4::from_translation(pos);
        Self {
            model,
            snapshot: model,
            mesh,
            color,
        }
    }

    fn position(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.model.w_axis.x,
            self.model.w_axis.y,
            self.model.w_axis.z,
        )
    }

    fn save_snapshot(&mut self) {
        self.snapshot = self.model;
    }

    fn restore_snapshot(&mut self) {
        self.model = self.snapshot;
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    mode: Mode,
    camera: Camera,

    // Mode 1
    ctrl_primitives: OrbitCameraController,

    // Mode 2
    ctrl_all: OrbitCameraController,
    objects: Vec<Object>,
    selection: Selection,
    gizmo: Gizmo,
    gizmo_center: Option<glam::Vec3>,
    gizmo_scale: f32,
    manip: ManipulationController,
    active_manip_kind: ManipulationKind,

    /// Increments whenever any `obj.model` changes. Passed as `SceneFrame::generation`
    /// so the renderer knows to re-upload instance transforms.
    scene_generation: u64,
    /// Increments whenever `selection` changes. Passed as `InteractionFrame::selection_generation`
    /// so the renderer knows to rebuild the selection outline.
    selection_generation: u64,

    // Per-frame pointer state
    cursor_viewport: Option<glam::Vec2>,
    cursor_prev: Option<glam::Vec2>,
    left_held: bool,
    drag_started: bool,
    clicked: bool,
    shift_held: bool,
    press_origin: Option<glam::Vec2>,
}

impl App {
    fn new(m_box: usize, m_sphere: usize) -> Self {
        Self {
            mode: Mode::Primitives,
            camera: Camera {
                distance: 12.0,
                ..Camera::default()
            },
            ctrl_primitives: OrbitCameraController::viewport_primitives(),
            ctrl_all: OrbitCameraController::viewport_all(),
            objects: vec![
                Object::new(glam::Vec3::new(-3.0, 0.0, 0.0), m_box, [0.4, 0.6, 0.9]),
                Object::new(glam::Vec3::new(0.0, 0.0, 0.0), m_sphere, [0.9, 0.5, 0.2]),
                Object::new(glam::Vec3::new(3.0, 0.0, 0.0), m_box, [0.3, 0.8, 0.4]),
            ],
            selection: Selection::new(),
            gizmo: Gizmo::new(),
            gizmo_center: None,
            gizmo_scale: 1.0,
            manip: ManipulationController::new(),
            active_manip_kind: ManipulationKind::Move,
            scene_generation: 1,
            selection_generation: 1,
            cursor_viewport: None,
            cursor_prev: None,
            left_held: false,
            drag_started: false,
            clicked: false,
            shift_held: false,
            press_origin: None,
        }
    }

    /// Apply a `TransformDelta` to all selected objects.
    ///
    /// Rotation and scale pivot depends on the current `PivotMode`:
    /// - `IndividualOrigins` : each object transforms around its own centre.
    /// - Everything else : all objects transform around the shared `gizmo_center`.
    fn apply_delta(&mut self, delta: viewport_lib::TransformDelta) {
        let pivot_mode = self.gizmo.pivot_mode;
        let gizmo_center = self.gizmo_center.unwrap_or(glam::Vec3::ZERO);

        let rot_mat = glam::Mat4::from_quat(delta.rotation);
        let scale_mat = glam::Mat4::from_scale(delta.scale);
        let translate_mat = glam::Mat4::from_translation(delta.translation);

        for id in self.selection.iter().copied().collect::<Vec<_>>() {
            let obj = &mut self.objects[id as usize];

            let pivot = match pivot_mode {
                PivotMode::IndividualOrigins => obj.position(),
                _ => gizmo_center,
            };

            let to_pivot = glam::Mat4::from_translation(-pivot);
            let from_pivot = glam::Mat4::from_translation(pivot);

            // translate · (from_pivot · rot · scale · to_pivot) · old_model
            obj.model = translate_mat * from_pivot * rot_mat * scale_mat * to_pivot * obj.model;
        }
        self.scene_generation += 1;
    }

    fn restore_snapshots(&mut self) {
        for id in self.selection.iter().copied().collect::<Vec<_>>() {
            self.objects[id as usize].restore_snapshot();
        }
        self.scene_generation += 1;
    }

    fn save_snapshots(&mut self) {
        for id in self.selection.iter().copied().collect::<Vec<_>>() {
            self.objects[id as usize].save_snapshot();
        }
    }

    /// Project all object centres to screen and return the index of the one
    /// nearest to `cursor` (within `threshold` pixels), or `None`.
    fn pick_object(
        &self,
        cursor: glam::Vec2,
        view_proj: glam::Mat4,
        viewport_size: glam::Vec2,
        threshold: f32,
    ) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, obj) in self.objects.iter().enumerate() {
            let ndc = view_proj.project_point3(obj.position());
            if ndc.z < 0.0 || ndc.z > 1.0 {
                continue; // behind camera or past far plane
            }
            let screen = glam::Vec2::new(
                (ndc.x + 1.0) * 0.5 * viewport_size.x,
                (1.0 - ndc.y) * 0.5 * viewport_size.y,
            );
            let dist = (screen - cursor).length();
            if dist < threshold && best.map_or(true, |(_, d)| dist < d) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }

    fn recompute_gizmo_center(&mut self) {
        self.gizmo_center = gizmo_center_for_pivot(&self.gizmo.pivot_mode, &self.selection, |id| {
            Some(self.objects[id as usize].position())
        });
    }
}

fn egui_key_to_keycode(key: egui::Key) -> Option<KeyCode> {
    match key {
        egui::Key::A => Some(KeyCode::A),
        egui::Key::D => Some(KeyCode::D),
        egui::Key::E => Some(KeyCode::E),
        egui::Key::F => Some(KeyCode::F),
        egui::Key::G => Some(KeyCode::G),
        egui::Key::Q => Some(KeyCode::Q),
        egui::Key::R => Some(KeyCode::R),
        egui::Key::S => Some(KeyCode::S),
        egui::Key::W => Some(KeyCode::W),
        egui::Key::X => Some(KeyCode::X),
        egui::Key::Y => Some(KeyCode::Y),
        egui::Key::Z => Some(KeyCode::Z),
        egui::Key::Tab => Some(KeyCode::Tab),
        egui::Key::Enter => Some(KeyCode::Enter),
        egui::Key::Escape => Some(KeyCode::Escape),
        egui::Key::Backspace => Some(KeyCode::Backspace),
        egui::Key::Backtick => Some(KeyCode::Backtick),
        egui::Key::Comma => Some(KeyCode::Comma),
        egui::Key::Period => Some(KeyCode::Period),
        egui::Key::OpenBracket => Some(KeyCode::LeftBracket),
        egui::Key::CloseBracket => Some(KeyCode::RightBracket),
        egui::Key::Slash => Some(KeyCode::Slash),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Mode switch: consume 1/2 before any controller sees them.
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Key {
                    key,
                    pressed: true,
                    repeat: false,
                    ..
                } = event
                {
                    let next = match key {
                        egui::Key::Num1 => Some(Mode::Primitives),
                        egui::Key::Num2 => Some(Mode::All),
                        _ => None,
                    };
                    if let Some(m) = next {
                        if m != self.mode {
                            match self.mode {
                                Mode::Primitives => {
                                    self.ctrl_primitives.push_event(ViewportEvent::FocusLost)
                                }
                                Mode::All => self.ctrl_all.push_event(ViewportEvent::FocusLost),
                            }
                            self.mode = m;
                            self.manip.reset();
                        }
                    }
                }
            }
        });

        // ---- Mode bar ----
        egui::TopBottomPanel::top("mode_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, Mode::Primitives, "1 · Camera only");
                ui.selectable_value(&mut self.mode, Mode::All, "2 · Manipulation");
                ui.separator();
                match self.mode {
                    Mode::Primitives => {
                        ui.small("Left/Middle drag: orbit  ·  Right drag: pan  ·  Scroll: zoom");
                    }
                    Mode::All => {
                        ui.small(
                            "Click: select  ·  Shift+Click: multi-select  ·  G/R/S: move/rotate/scale  ·  X/Y/Z: constrain  ·  [: pivot fwd  ·  ]: pivot back",
                        );
                        if !self.selection.is_empty() {
                            ui.separator();
                            ui.small(format!("Pivot: {}", self.gizmo.pivot_mode.label()));
                        }
                    }
                }
            });
        });

        // ---- Viewport ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.drag_started = false;
            self.clicked = false;

            let vp_ctx = ViewportContext {
                hovered: response.hovered(),
                // Treat any hover as focus so keyboard shortcuts work without
                // requiring a separate egui-focus click on the canvas widget.
                focused: response.hovered() || response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            };
            match self.mode {
                Mode::Primitives => self.ctrl_primitives.begin_frame(vp_ctx),
                Mode::All => self.ctrl_all.begin_frame(vp_ctx),
            }

            let mut vp_events: Vec<ViewportEvent> = Vec::new();
            let manip_active = self.manip.is_active();

            ui.input(|i| {
                self.shift_held = i.modifiers.shift;

                vp_events.push(ViewportEvent::ModifiersChanged(viewport_lib::Modifiers {
                    alt: i.modifiers.alt,
                    shift: i.modifiers.shift,
                    ctrl: i.modifiers.command,
                }));

                let local_pos = i
                    .pointer
                    .interact_pos()
                    .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
                self.cursor_prev = self.cursor_viewport;
                self.cursor_viewport = local_pos;
                if let Some(pos) = local_pos {
                    vp_events.push(ViewportEvent::PointerMoved { position: pos });
                }

                for event in &i.events {
                    match event {
                        egui::Event::Key {
                            key,
                            pressed,
                            repeat,
                            ..
                        } => {
                            if matches!(key, egui::Key::Num1 | egui::Key::Num2) {
                                continue;
                            }
                            if let Some(kc) = egui_key_to_keycode(*key) {
                                vp_events.push(ViewportEvent::Key {
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
                        egui::Event::PointerButton {
                            button, pressed, ..
                        } => {
                            let vp_button = match button {
                                egui::PointerButton::Primary => viewport_lib::MouseButton::Left,
                                egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
                                egui::PointerButton::Middle => viewport_lib::MouseButton::Middle,
                                _ => continue,
                            };
                            vp_events.push(ViewportEvent::MouseButton {
                                button: vp_button,
                                state: if *pressed {
                                    ButtonState::Pressed
                                } else {
                                    ButtonState::Released
                                },
                            });
                            if *button == egui::PointerButton::Primary {
                                if *pressed {
                                    self.left_held = true;
                                    self.press_origin = self.cursor_viewport;
                                    self.drag_started = true;
                                } else {
                                    let is_click = self
                                        .press_origin
                                        .zip(self.cursor_viewport)
                                        .map(|(o, c)| (c - o).length() < 5.0)
                                        .unwrap_or(false);
                                    if is_click {
                                        self.clicked = true;
                                    }
                                    self.left_held = false;
                                    self.press_origin = None;
                                }
                            }
                        }
                        egui::Event::MouseWheel { delta, .. } => {
                            vp_events.push(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units: ScrollUnits::Pixels,
                            });
                        }
                        egui::Event::Text(text) if manip_active => {
                            for c in text.chars() {
                                vp_events.push(ViewportEvent::Character(c));
                            }
                        }
                        _ => {}
                    }
                }
            });

            let ctrl = match self.mode {
                Mode::Primitives => &mut self.ctrl_primitives,
                Mode::All => &mut self.ctrl_all,
            };
            for e in vp_events {
                ctrl.push_event(e);
            }

            let w = rect.width();
            let h = rect.height();
            let viewport_size = glam::Vec2::new(w, h);
            let pointer_delta = self
                .cursor_viewport
                .zip(self.cursor_prev)
                .map(|(c, p)| c - p)
                .unwrap_or(glam::Vec2::ZERO);

            // ---------------------------------------------------------------
            // Mode 1: camera only : three static boxes
            // ---------------------------------------------------------------
            let scene_items: Vec<SceneRenderItem> = match self.mode {
                Mode::Primitives => {
                    self.ctrl_primitives.apply_to_camera(&mut self.camera);

                    // Render the same objects as Mode 2 so the scene looks
                    // identical across modes (no confusing shape swaps).
                    self.objects
                        .iter()
                        .map(|obj| {
                            let mut item = SceneRenderItem::default();
                            item.mesh_index = obj.mesh;
                            item.model = obj.model.to_cols_array_2d();
                            item.material = Material::from_color(obj.color);
                            item
                        })
                        .collect()
                }

                // ---------------------------------------------------------------
                // Mode 2: full controls : three selectable objects
                // ---------------------------------------------------------------
                Mode::All => {
                    let camera_view = self.camera.view_matrix();
                    let camera_proj = self.camera.proj_matrix();
                    let view_proj = camera_proj * camera_view;

                    // Single action frame for the whole frame : resolve() or
                    // apply_to_camera() must only be called once per frame.
                    let action_frame = if self.manip.is_active() {
                        self.ctrl_all.resolve()
                    } else {
                        self.ctrl_all.apply_to_camera(&mut self.camera)
                    };

                    // Pivot mode cycling : check the action frame we already have.
                    let cycle_fwd = action_frame.is_active(Action::CyclePivotModeForward);
                    let cycle_bwd = action_frame.is_active(Action::CyclePivotModeBackward);
                    if cycle_fwd || cycle_bwd {
                        if cycle_fwd {
                            self.gizmo.cycle_pivot_forward();
                        } else {
                            self.gizmo.cycle_pivot_backward();
                        }
                        self.recompute_gizmo_center();
                        if self.manip.is_active() {
                            self.manip.reset();
                            self.restore_snapshots();
                            if let Some(center) = self.gizmo_center {
                                self.manip.begin(self.active_manip_kind, center);
                            }
                        }
                    }

                    // Update gizmo hover from the current cursor position.
                    if let (Some(cursor), Some(center)) = (self.cursor_viewport, self.gizmo_center)
                    {
                        if !self.manip.is_active() {
                            let ray_origin = self.camera.eye_position();
                            let ndc_x = (cursor.x / w.max(1.0)) * 2.0 - 1.0;
                            let ndc_y = 1.0 - (cursor.y / h.max(1.0)) * 2.0;
                            let inv_vp = view_proj.inverse();
                            let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
                            let ray_dir = (far - ray_origin).normalize_or(glam::Vec3::NEG_Z);
                            self.gizmo.hovered_axis =
                                self.gizmo
                                    .hit_test(ray_origin, ray_dir, center, self.gizmo_scale);
                        }
                    }

                    // Build ManipulationContext.
                    let gizmo_info = self.gizmo_center.map(|center| GizmoInfo {
                        center,
                        scale: self.gizmo_scale,
                        orientation: glam::Quat::IDENTITY,
                        mode: self.gizmo.mode,
                    });

                    let manip_ctx = ManipulationContext {
                        camera: self.camera.clone(),
                        viewport_size,
                        cursor_viewport: self.cursor_viewport,
                        pointer_delta,
                        selection_center: self.gizmo_center,
                        gizmo: gizmo_info,
                        drag_started: self.drag_started,
                        dragging: self.left_held,
                        clicked: self.clicked,
                    };

                    match self.manip.update(&action_frame, manip_ctx) {
                        ManipResult::Update(delta) => {
                            // Delta is incremental : accumulate, do NOT restore snapshot.
                            self.apply_delta(delta);
                            self.recompute_gizmo_center();
                        }
                        ManipResult::Cancel | ManipResult::ConstraintChanged => {
                            self.restore_snapshots();
                            self.recompute_gizmo_center();
                        }
                        ManipResult::Commit => {
                            self.save_snapshots();
                            self.recompute_gizmo_center();
                            // If a click confirmed the session, consume it so
                            // click-to-select doesn't fire on the same frame.
                            self.clicked = false;
                        }
                        ManipResult::None => {
                            if self.manip.is_active() {
                                // Session just started this frame : record its kind.
                                if let Some(state) = self.manip.state() {
                                    self.active_manip_kind = state.kind;
                                }
                            } else {
                                // Idle : keep snapshot current so G/R/S starts clean.
                                self.save_snapshots();
                            }
                        }
                    }

                    // Click-to-select (not during or right after a manipulation).
                    if self.clicked && !self.manip.is_active() {
                        if let Some(cursor) = self.cursor_viewport {
                            let hit = self.pick_object(cursor, view_proj, viewport_size, 60.0);
                            match hit {
                                Some(idx) => {
                                    if self.shift_held {
                                        if self.selection.contains(idx as u64) {
                                            self.selection.remove(idx as u64);
                                        } else {
                                            self.selection.add(idx as u64);
                                        }
                                    } else {
                                        self.selection.clear();
                                        self.selection.select_one(idx as u64);
                                    }
                                }
                                None => {
                                    if !self.shift_held {
                                        self.selection.clear();
                                    }
                                }
                            }
                            self.recompute_gizmo_center();
                            self.selection_generation += 1;
                        }
                    }

                    // Recompute gizmo scale each frame so it stays screen-size-stable.
                    if let Some(center) = self.gizmo_center {
                        self.gizmo_scale = viewport_lib::gizmo::compute_gizmo_scale(
                            center,
                            self.camera.eye_position(),
                            self.camera.fov_y,
                            h,
                        );
                    }

                    // Build scene render items.
                    self.objects
                        .iter()
                        .enumerate()
                        .map(|(i, obj)| {
                            let mut item = SceneRenderItem::default();
                            item.mesh_index = obj.mesh;
                            item.model = obj.model.to_cols_array_2d();
                            item.material = Material::from_color(obj.color);
                            item.selected = self.selection.contains(i as u64);
                            item
                        })
                        .collect()
                }
            };

            self.camera.set_aspect_ratio(w, h);

            let mut scene_frame = SceneFrame::from_surface_items(scene_items);
            scene_frame.generation = self.scene_generation;

            let mut frame_data =
                FrameData::new(CameraFrame::from_camera(&self.camera, [w, h]), scene_frame);
            frame_data.effects.lighting = LightingSettings::default();
            frame_data.viewport.show_grid = true;
            frame_data.viewport.show_axes_indicator = true;

            // Gizmo rendering (Mode 2 only).
            if self.mode == Mode::All {
                if let Some(center) = self.gizmo_center {
                    let gizmo_model = glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(self.gizmo_scale),
                        glam::Quat::IDENTITY,
                        center,
                    );
                    let mut interaction = InteractionFrame::default();
                    interaction.gizmo_model = Some(gizmo_model);
                    interaction.gizmo_mode = self.gizmo.mode;
                    interaction.gizmo_hovered = self.gizmo.hovered_axis;
                    interaction.gizmo_space_orientation = glam::Quat::IDENTITY;
                    interaction.outline_selected = true;
                    interaction.selection_generation = self.selection_generation;
                    frame_data.interaction = interaction;
                } else {
                    frame_data.interaction.outline_selected = true;
                    frame_data.interaction.selection_generation = self.selection_generation;
                }
            }

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback { frame: frame_data },
                ));

            if response.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}
