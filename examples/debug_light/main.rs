//! Debug example reproducing the lighting setup from
//! hamilton_engine_v2/examples/cloth/cloth_fast_drape.rs.
//!
//! Scene: a flat gray floor plane with a small flat platform (slab)
//! floating above it, plus a handful of translucent tetrahedralized volume
//! meshes (a cube, a wide slab, and a tall box). Lighting is
//! `LightingSettings::default()`.
//!
//! The volume meshes are here to reproduce a shading complaint: viewed from
//! above they read as an even translucent blue, but viewed from below the
//! outer faces collapse toward black. They render through the lit
//! boundary-surface path (`upload_volume_mesh` + `settings.opacity < 1`), so
//! any face whose outward normal points away from the overhead light gets no
//! direct diffuse and only the dark end of the hemisphere ambient.
//!
//! Navigation:
//!   Left drag / Middle drag   : orbit
//!   Right drag                : pan
//!   Scroll                    : zoom

mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    BackfacePolicy, ButtonState, Camera, CameraFrame, FrameData, LightingSettings, Material,
    MeshData, MeshId, Modifiers, MouseButton, OrbitCameraController, SceneFrame, SceneRenderItem,
    ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};

const SLAB_HALF_EXTENTS: [f32; 3] = [0.5, 0.5, 0.005];
const SLAB_HEIGHT: f32 = 0.6;

const FLOOR_COLOUR: [f32; 3] = [0.35, 0.35, 0.38];
const SLAB_COLOUR: [f32; 3] = [0.85, 0.55, 0.35];

// Blue with alpha 0.18, matching `GeometryOptions::tet_albedo` in
// viewport-lib-mesh-assembly (the material_lab tet volume).
const TVM_COLOUR: [f32; 3] = [0.45, 0.75, 0.95];
const TVM_OPACITY: f32 = 0.18;

/// A lattice of tetrahedra filling a cube centered at the origin.
///
/// `n` cells per axis, each split into five tets with orientation alternating
/// by cell parity so shared diagonals match across neighbours. This mirrors
/// `tet_lattice` in viewport-lib-mesh-assembly/examples/material_lab.
fn tet_lattice(n: usize, size: f32) -> (Vec<[f32; 3]>, Vec<[u32; 4]>) {
    let step = size / n as f32;
    let g = n + 1;
    let idx = |i: usize, j: usize, k: usize| ((k * g + j) * g + i) as u32;
    let mut positions = Vec::with_capacity(g * g * g);
    for k in 0..g {
        for j in 0..g {
            for i in 0..g {
                positions.push([
                    i as f32 * step - size * 0.5,
                    j as f32 * step - size * 0.5,
                    k as f32 * step - size * 0.5,
                ]);
            }
        }
    }
    let mut tets = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let c = [
                    idx(i, j, k),
                    idx(i + 1, j, k),
                    idx(i + 1, j + 1, k),
                    idx(i, j + 1, k),
                    idx(i, j, k + 1),
                    idx(i + 1, j, k + 1),
                    idx(i + 1, j + 1, k + 1),
                    idx(i, j + 1, k + 1),
                ];
                if (i + j + k) % 2 == 0 {
                    tets.push([c[0], c[1], c[3], c[4]]);
                    tets.push([c[1], c[2], c[3], c[6]]);
                    tets.push([c[1], c[3], c[4], c[6]]);
                    tets.push([c[1], c[4], c[5], c[6]]);
                    tets.push([c[3], c[4], c[6], c[7]]);
                } else {
                    tets.push([c[0], c[1], c[2], c[5]]);
                    tets.push([c[0], c[2], c[3], c[7]]);
                    tets.push([c[0], c[2], c[5], c[7]]);
                    tets.push([c[0], c[4], c[5], c[7]]);
                    tets.push([c[2], c[5], c[6], c[7]]);
                }
            }
        }
    }
    (positions, tets)
}

/// Face-soup mesh: every one of the four faces of every tet, emitted as an
/// independent triangle with a flat per-face normal. No boundary extraction,
/// so the interior tets stay visible when the mesh is drawn translucent
/// through the OIT pass. This is how the mesh-assembly `TetVolume` renders.
fn tet_soup_mesh(positions: &[[f32; 3]], tets: &[[u32; 4]]) -> MeshData {
    // Same face winding the mesh-assembly renderer uses.
    const FACES: [[usize; 3]; 4] = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
    let mut out_pos = Vec::with_capacity(tets.len() * 4 * 3);
    let mut out_nrm = Vec::with_capacity(tets.len() * 4 * 3);
    let mut indices = Vec::with_capacity(tets.len() * 4 * 3);
    for tet in tets {
        for face in FACES {
            let p: [glam::Vec3; 3] =
                std::array::from_fn(|k| glam::Vec3::from(positions[tet[face[k]] as usize]));
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            let base = out_pos.len() as u32;
            for v in p {
                out_pos.push(v.to_array());
                out_nrm.push(n.to_array());
            }
            indices.extend([base, base + 1, base + 2]);
        }
    }
    let mut mesh = MeshData::default();
    mesh.positions = out_pos;
    mesh.normals = out_nrm;
    mesh.indices = indices;
    mesh
}

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : Debug Light",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 720.0]),
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

            let floor_mesh = res
                .upload_mesh_data(device, &primitives::plane(20.0, 20.0))
                .expect("floor mesh upload");
            let slab_mesh = res
                .upload_mesh_data(
                    device,
                    &primitives::cuboid(
                        2.0 * SLAB_HALF_EXTENTS[0],
                        2.0 * SLAB_HALF_EXTENTS[1],
                        2.0 * SLAB_HALF_EXTENTS[2],
                    ),
                )
                .expect("slab mesh upload");

            // A handful of translucent tet volumes floating above the floor,
            // rendered as face soup (all tet faces, flat normals) through the
            // OIT pass exactly as viewport-lib-mesh-assembly's `TetVolume` does.
            // Each lattice is built centered at the origin and placed via the
            // render item's model matrix.
            let tvm_specs: [(usize, f32, glam::Vec3); 3] = [
                // The cube: the shape in the original report (matches material_lab).
                (3, 1.6, glam::Vec3::new(0.0, 0.0, 1.4)),
                // A finer cube.
                (4, 1.4, glam::Vec3::new(-3.2, 0.0, 1.2)),
                // A tall box (non-cubic via a scaled model matrix).
                (3, 1.2, glam::Vec3::new(3.2, 0.0, 1.6)),
            ];
            let mut tvm_items = Vec::new();
            for (idx, (n, size, translation)) in tvm_specs.into_iter().enumerate() {
                let (positions, tets) = tet_lattice(n, size);
                let mesh = tet_soup_mesh(&positions, &tets);
                let mesh_id = res.upload_mesh_data(device, &mesh).expect("tvm upload");
                let scale = if idx == 2 {
                    glam::Vec3::new(1.0, 1.0, 1.6)
                } else {
                    glam::Vec3::ONE
                };
                let mut item = SceneRenderItem::default();
                item.mesh_id = mesh_id;
                item.model = (glam::Mat4::from_translation(translation)
                    * glam::Mat4::from_scale(scale))
                .to_cols_array_2d();
                item.material = {
                    let mut m = Material::from_colour(TVM_COLOUR);
                    // Show both sides of every tet face, like the face soup.
                    m.backface_policy = BackfacePolicy::Identical;
                    m
                };
                item.settings.opacity = TVM_OPACITY;
                tvm_items.push(item);
            }

            rs.renderer.write().callback_resources.insert(renderer);
            Ok(Box::new(App::new(floor_mesh, slab_mesh, tvm_items)))
        }),
    )
}

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    floor_mesh: MeshId,
    slab_mesh: MeshId,
    tvm_items: Vec<SceneRenderItem>,
}

impl App {
    fn new(floor_mesh: MeshId, slab_mesh: MeshId, tvm_items: Vec<SceneRenderItem>) -> Self {
        Self {
            camera: Camera {
                center: glam::Vec3::new(0.0, 0.0, 1.0),
                distance: 8.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            floor_mesh,
            slab_mesh,
            tvm_items,
        }
    }

    fn build_items(&self) -> Vec<SceneRenderItem> {
        let mut floor = SceneRenderItem::default();
        floor.mesh_id = self.floor_mesh;
        floor.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        floor.material = {
            let mut m = Material::from_colour(FLOOR_COLOUR);
            m.backface_policy = BackfacePolicy::Identical;
            m
        };

        let mut slab = SceneRenderItem::default();
        slab.mesh_id = self.slab_mesh;
        slab.model =
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, SLAB_HEIGHT)).to_cols_array_2d();
        slab.material = Material::from_colour(SLAB_COLOUR);

        let mut items = vec![slab, floor];
        items.extend(self.tvm_items.iter().cloned());
        items
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.controller.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            });

            ui.input(|i| {
                self.controller
                    .push_event(ViewportEvent::ModifiersChanged(Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    }));
                let local = i
                    .pointer
                    .interact_pos()
                    .map(|p| glam::Vec2::new(p.x - rect.left(), p.y - rect.top()));
                if let Some(pos) = local {
                    self.controller
                        .push_event(ViewportEvent::PointerMoved { position: pos });
                }
                for event in &i.events {
                    match event {
                        egui::Event::PointerButton {
                            button, pressed, ..
                        } => {
                            let vp_btn = match button {
                                egui::PointerButton::Primary => MouseButton::Left,
                                egui::PointerButton::Secondary => MouseButton::Right,
                                egui::PointerButton::Middle => MouseButton::Middle,
                                _ => continue,
                            };
                            self.controller.push_event(ViewportEvent::MouseButton {
                                button: vp_btn,
                                state: if *pressed {
                                    ButtonState::Pressed
                                } else {
                                    ButtonState::Released
                                },
                            });
                        }
                        egui::Event::MouseWheel { unit, delta, .. } => {
                            let units = match unit {
                                egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                                egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                                egui::MouseWheelUnit::Page => ScrollUnits::Pages,
                            };
                            self.controller.push_event(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units,
                            });
                        }
                        _ => {}
                    }
                }
            });

            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(rect.width(), rect.height());

            let mut frame_data = FrameData::new(
                CameraFrame::from_camera(&self.camera, [rect.width(), rect.height()])
                    .with_pixels_per_point(ui.ctx().pixels_per_point()),
                SceneFrame::from_surface_items(self.build_items()),
            );
            frame_data.effects.lighting = LightingSettings::default();

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    viewport_callback::ViewportCallback { frame: frame_data },
                ));
        });

        ctx.request_repaint();
    }
}
