//! Retained overlay geometry: compile a scrollable panel once, then scroll it by
//! updating only its per-frame translate (no re-tessellation).
//!
//! This is the payoff of `compile_overlay_geometry` / `compile_overlay_label` /
//! `RetainedOverlay`. Three groups are compiled a single time at startup:
//!
//! - a static panel background: one mixed handle carrying an SDF rounded-rect
//!   plate plus a fixed-local title label (a string laid out once, bundled into the
//!   same handle), submitted every frame at a fixed position,
//! - the panel's scrollable content (polyline "rows", drawn through the text
//!   pipeline), submitted every frame with a `translate` that scrolls it and a
//!   `clip_rect` that clips it to the panel interior, and
//! - a world-anchored label pinned to the top of the cube (`compile_overlay_label`),
//!   whose text is laid out once: the renderer reprojects its anchor every frame so
//!   it tracks the cube as the camera orbits, and hides it when the cube is off
//!   screen, with no re-layout.
//!
//! None is re-tessellated as it scrolls or the camera moves: only the small
//! per-frame `RetainedOverlay` (a handle plus translate/opacity/clip) changes, and
//! the label carries no translate at all. Compare with pushing hundreds of items
//! into `OverlayFrame` every frame.
//!
//! Navigation: left/middle drag orbit, right drag pan, scroll zoom (the cube is
//! just context behind the panel).

use eframe::{egui, wgpu};
use viewport_lib as vpl;
use vpl::input::adapters::from_egui;
use vpl::{
    AnchorY, LabelItem, Material, Modifiers, OffscreenViewportTarget, OrbitCameraController,
    OverlayFill, OverlayGeometryId, OverlayPolylineItem, OverlayShape, OverlayShapeItem,
    RetainedOverlay, ViewportContext, ViewportEvent, ViewportInstance, primitives,
};

/// Panel geometry, in the panel's own logical-pixel space (top-left origin near
/// `[40, 40]`). The content is taller than the panel so it scrolls.
const PANEL_X: f32 = 40.0;
const PANEL_TOP: f32 = 40.0;
const PANEL_W: f32 = 340.0;
const PANEL_H: f32 = 420.0;
const ROW_COUNT: usize = 40;
const ROW_STEP: f32 = 44.0;

/// The static panel background: one SDF rounded-rect (shape stream).
fn panel_background() -> Vec<OverlayShapeItem> {
    vec![
        OverlayShapeItem::new(
            OverlayShape::RoundedRect { radii: [14.0; 4] },
            [PANEL_X, PANEL_TOP],
            [PANEL_W, PANEL_H],
        )
        .with_fill(OverlayFill::Solid([0.11, 0.12, 0.16, 0.96])),
    ]
}

/// The scrollable content: a stack of row separators plus a colour swatch per
/// row, as polylines (text stream). Positions are the un-scrolled layout.
fn panel_content() -> Vec<OverlayPolylineItem> {
    let mut lines = Vec::new();
    for i in 0..ROW_COUNT {
        let y = PANEL_TOP + 16.0 + i as f32 * ROW_STEP;
        // Row separator line.
        let mut sep = OverlayPolylineItem::default();
        sep.points = vec![
            [PANEL_X + 20.0, y + ROW_STEP - 8.0],
            [PANEL_X + PANEL_W - 20.0, y + ROW_STEP - 8.0],
        ];
        sep.thickness = 1.5;
        sep.colour = [0.4, 0.45, 0.6, 0.7];
        lines.push(sep);
        // A small filled swatch, hue cycling down the list.
        let t = i as f32 / ROW_COUNT as f32;
        let mut swatch = OverlayPolylineItem::default();
        swatch.points = vec![
            [PANEL_X + 20.0, y],
            [PANEL_X + 44.0, y],
            [PANEL_X + 44.0, y + 24.0],
            [PANEL_X + 20.0, y + 24.0],
        ];
        swatch.closed = true;
        swatch.fill = Some(OverlayFill::Solid([
            0.9 - t * 0.6,
            0.4 + t * 0.4,
            0.3 + t * 0.5,
            1.0,
        ]));
        lines.push(swatch);
    }
    lines
}

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib : retained overlay (egui)",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 640.0]),
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let mut session = ViewportInstance::new(
                &rs.device,
                OffscreenViewportTarget::render_format(rs.target_format),
            );

            let cube = session
                .resources_mut()
                .upload_mesh_data(&rs.device, &primitives::cube(1.0))
                .unwrap();
            session.scene_mut().add(
                Some(cube),
                glam::Mat4::IDENTITY,
                Material::from_colour([0.2, 0.4, 0.7]),
            );
            session.camera_mut().distance = 6.0;

            Ok(Box::new(App {
                session,
                orbit: OrbitCameraController::viewport_all(),
                target: None,
                background: None,
                content: None,
                label: None,
            }))
        }),
    )
}

struct Target {
    inner: OffscreenViewportTarget,
    id: egui::TextureId,
}

struct App {
    session: ViewportInstance,
    orbit: OrbitCameraController,
    target: Option<Target>,
    /// Compiled once, on the first frame (when `pixels_per_point` is known).
    background: Option<OverlayGeometryId>,
    content: Option<OverlayGeometryId>,
    /// A world-anchored label pinned to the top of the cube. Compiled once; the
    /// renderer reprojects its anchor every frame so it tracks the cube as the
    /// camera orbits, with no re-layout, and hides when the point goes off screen.
    label: Option<OverlayGeometryId>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let rs = frame.wgpu_render_state().expect("wgpu backend required");
        let time = ctx.input(|i| i.time) as f32;

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
                let ppp = ui.ctx().pixels_per_point();
                let size = [
                    (rect.width() * ppp).round().max(1.0) as u32,
                    (rect.height() * ppp).round().max(1.0) as u32,
                ];

                if self
                    .target
                    .as_ref()
                    .map_or(true, |t| t.inner.size() != size)
                {
                    let inner = OffscreenViewportTarget::new(&rs.device, rs.target_format, size);
                    let id = rs.renderer.write().register_native_texture(
                        &rs.device,
                        inner.sample_view(),
                        wgpu::FilterMode::Linear,
                    );
                    self.target = Some(Target { inner, id });
                }
                let target = self.target.as_ref().unwrap();

                self.session.begin_frame(ViewportContext {
                    hovered: response.hovered(),
                    focused: response.has_focus(),
                    viewport_size: [rect.width(), rect.height()],
                });
                self.session.set_pixels_per_point(ppp);
                let origin = glam::Vec2::new(rect.left(), rect.top());
                ui.input(|i| {
                    self.session
                        .handle_event(ViewportEvent::ModifiersChanged(Modifiers {
                            alt: i.modifiers.alt,
                            shift: i.modifiers.shift,
                            ctrl: i.modifiers.command,
                        }));
                    for event in &i.events {
                        if let Some(ev) = from_egui(event, origin) {
                            self.session.handle_event(ev);
                        }
                    }
                });

                // Compile the two panel groups once, now that ppp is known.
                if self.background.is_none() {
                    // The panel background is one mixed handle: the rounded-rect
                    // plate plus a fixed-local title label, laid out once. The
                    // label rides the shape in a single handle (one dirty unit);
                    // its anchor is ignored here, it sits at its local position.
                    let title = LabelItem::new("Layers")
                        .with_screen_anchor([PANEL_X, PANEL_TOP - 8.0])
                        .with_align_y(AnchorY::Bottom)
                        .with_font_size(16.0)
                        .with_colour([0.95, 0.97, 1.0, 1.0])
                        .with_background(true)
                        .with_background_colour([0.2, 0.24, 0.34, 1.0])
                        .with_padding(5.0)
                        .with_border_radius(4.0);
                    let bg = self.session.renderer_mut().compile_overlay_geometry(
                        &rs.device,
                        &rs.queue,
                        &[],
                        &panel_background(),
                        &[],
                        std::slice::from_ref(&title),
                        ppp,
                    );
                    let content = self.session.renderer_mut().compile_overlay_geometry(
                        &rs.device,
                        &rs.queue,
                        &panel_content(),
                        &[],
                        &[],
                        &[],
                        ppp,
                    );
                    // A world-anchored label, laid out once. The renderer resolves
                    // its anchor each frame so it tracks the cube; the "cube" text,
                    // background, and leader line are never re-tessellated.
                    let tag = LabelItem::new("cube")
                        .with_world_anchor([0.0, 0.0, 0.7])
                        .with_leader_line(true)
                        .with_background(true)
                        .with_border_radius(4.0)
                        .with_align_y(AnchorY::Bottom)
                        .with_font_size(15.0);
                    let label = self
                        .session
                        .renderer_mut()
                        .compile_overlay_label(&rs.device, &rs.queue, &tag, ppp);
                    self.background = Some(bg);
                    self.content = Some(content);
                    self.label = Some(label);
                }

                self.session.update_orbit(&mut self.orbit);

                // Scroll the content: a gentle triangle wave over the extra height
                // that does not fit the panel. Only this offset changes per frame.
                let extra = (ROW_COUNT as f32 * ROW_STEP - PANEL_H + 40.0).max(0.0);
                let phase = (time * 0.15).fract();
                let scroll = extra * (1.0 - (2.0 * phase - 1.0).abs());
                let clip = [PANEL_X, PANEL_TOP, PANEL_X + PANEL_W, PANEL_TOP + PANEL_H];

                // Set both groups as retained submissions (assembly clears overlays,
                // so this runs after `update_orbit`).
                self.session.frame_data_mut().overlays.retained = vec![
                    // Static background (shape stream), drawn under the content.
                    RetainedOverlay::new(self.background.unwrap()).with_z_order(0),
                    // Scrolling content (polyline stream), clipped to the panel.
                    RetainedOverlay::new(self.content.unwrap())
                        .with_translate([0.0, -scroll])
                        .with_clip_rect(clip)
                        .with_z_order(1),
                    // World-anchored label: no translate here. The renderer
                    // resolves its baked anchor to the cube's projected position
                    // each frame and hides it when the cube is off screen.
                    RetainedOverlay::new(self.label.unwrap()).with_z_order(2),
                ];

                let cmd = self
                    .session
                    .render(&rs.device, &rs.queue, target.inner.render_view());
                rs.queue.submit(std::iter::once(cmd));
                ui.painter().image(
                    target.id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            });

        ctx.request_repaint();
    }
}
