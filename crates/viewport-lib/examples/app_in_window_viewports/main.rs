//! Two independent, individually-navigable 3D scenes side by side in one OS window.
//!
//! `ViewportAppV2`'s per-window paint hook lets a window host in-window viewports
//! while the consumer owns the layout. Each half is its own `ViewportInstance`
//! rendered into an `OffscreenViewportTarget`, then blitted into its rect of the
//! window surface in the paint hook. The window's own instance is an empty
//! compositor: it just clears the background the two panes sit on.
//!
//! Input is routed to the pane under the cursor: orbit (left/middle drag), pan
//! (right drag), and zoom (scroll) act on whichever half you are over. A drag latches
//! to the pane it started in, so it keeps working if the cursor crosses the seam.
//!
//! This is the in-window counterpart to `app-multi-window` (each viewport in its own
//! OS window) and to `winit-multi-viewport` (the renderer's shared-scene
//! multi-viewport API). Here the two scenes are fully independent.

use std::cell::RefCell;
use std::rc::Rc;

use viewport_lib as vpl;
use vpl::{
    AppConfigV2, BlitTexture, ButtonState, Material, NodeId, OffscreenViewportTarget,
    OrbitCameraController, ViewportAppV2, ViewportContext, ViewportEvent, ViewportInstance,
    WindowConfig, primitives,
};

/// One in-window viewport: its scene, the offscreen target it renders into, a camera
/// controller, and the blit handle used to composite it into the window.
struct Pane {
    session: ViewportInstance,
    target: OffscreenViewportTarget,
    orbit: OrbitCameraController,
    blit: Option<BlitTexture>,
    size: [u32; 2],
}

impl Pane {
    fn new(device: &vpl::wgpu::Device, surface_format: vpl::wgpu::TextureFormat, size: [u32; 2]) -> Self {
        Self {
            // The offscreen instance targets the sRGB render format so the blit into
            // the (sRGB) window surface encodes exactly once.
            session: ViewportInstance::new(device, OffscreenViewportTarget::render_format(surface_format)),
            target: OffscreenViewportTarget::new(device, surface_format, size),
            orbit: OrbitCameraController::viewport_all(),
            blit: None,
            size,
        }
    }

    /// Resize the offscreen target to `size` (physical pixels); drop the stale blit if
    /// the texture was recreated so it is rebuilt against the new view.
    fn resize(&mut self, device: &vpl::wgpu::Device, size: [u32; 2]) {
        if self.target.resize(device, size) {
            self.blit = None;
        }
        self.size = self.target.size();
    }

    /// Render this pane's scene into its offscreen target. When `active`, the pane is
    /// hovered/focused and the (pane-local) input events drive its camera.
    fn render(
        &mut self,
        device: &vpl::wgpu::Device,
        queue: &vpl::wgpu::Queue,
        vp_logical: [f32; 2],
        ppp: f32,
        active: bool,
        events: &[ViewportEvent],
    ) {
        // Events land in the accumulator, which begin_frame resets, so feed them
        // after begin_frame and before update_orbit resolves them.
        self.session.begin_frame(ViewportContext {
            hovered: active,
            focused: active,
            viewport_size: vp_logical,
        });
        self.session.set_viewport_size(vp_logical);
        self.session.set_pixels_per_point(ppp);
        if active {
            for ev in events {
                self.session.handle_event(ev.clone());
            }
        }
        let _ = self.session.update_orbit(&mut self.orbit);
        let cmd = self.session.render(device, queue, self.target.render_view());
        queue.submit(std::iter::once(cmd));
    }
}

#[derive(Default)]
struct Panes {
    left: Option<Pane>,
    right: Option<Pane>,
    cube: Option<NodeId>,
    /// The pane a drag started in (0 = left, 1 = right), so a drag keeps its pane even
    /// if the cursor crosses the seam. `None` when no drag is active.
    drag_pane: Option<usize>,
}

/// Shift a pointer position into a pane's local space by subtracting the pane's left
/// edge (in logical points); other events pass through unchanged.
fn remap(ev: &ViewportEvent, x_offset: f32) -> ViewportEvent {
    match ev {
        ViewportEvent::PointerMoved { position } => ViewportEvent::PointerMoved {
            position: glam::Vec2::new(position.x - x_offset, position.y),
        },
        other => other.clone(),
    }
}

fn main() {
    let state: Rc<RefCell<Panes>> = Rc::new(RefCell::new(Panes::default()));
    let paint_state = state.clone();

    ViewportAppV2::new(AppConfigV2::default())
        .window_with_paint(
            WindowConfig::default()
                .with_title("viewport-lib : two viewports in one window")
                .with_window_size(1200, 700),
            // The window's own instance is an empty compositor (clears the backdrop).
            |_vp, _device| {},
            move |ctx| {
                let device = ctx.device().clone();
                let queue = ctx.queue().clone();
                let [pw, ph] = ctx.surface_size();
                let [wl, hl] = ctx.viewport_size();
                let fmt = ctx.surface_format();
                let ppp = pw as f32 / wl.max(1.0);
                let half_logical = wl / 2.0;
                let left_w = pw / 2;
                let right_w = pw - left_w;

                // The window's resolved cursor (window-logical points), and this
                // frame's raw events, drive whichever pane is active.
                let cursor = ctx.action_frame().pointer.cursor;
                let events: Vec<ViewportEvent> = ctx.events().to_vec();

                let mut panes = state.borrow_mut();

                // Build the two offscreen panes once, now that the surface format and
                // size are known.
                if panes.left.is_none() {
                    let mut left = Pane::new(&device, fmt, [left_w.max(1), ph]);
                    let cube = left
                        .session
                        .resources_mut()
                        .upload_mesh_data(&device, &primitives::cube(1.0))
                        .unwrap();
                    let node = left.session.scene_mut().add(
                        Some(cube),
                        glam::Mat4::IDENTITY,
                        Material::from_colour([0.12, 0.3, 0.7]),
                    );
                    left.session.camera_mut().distance = 5.0;
                    panes.cube = Some(node);
                    panes.left = Some(left);

                    let mut right = Pane::new(&device, fmt, [right_w.max(1), ph]);
                    let torus = right
                        .session
                        .resources_mut()
                        .upload_mesh_data(&device, &primitives::torus(0.6, 0.22, 32, 16))
                        .unwrap();
                    right.session.scene_mut().add(
                        Some(torus),
                        glam::Mat4::IDENTITY,
                        Material::from_colour([0.1, 0.55, 0.2]),
                    );
                    right.session.camera_mut().distance = 5.0;
                    panes.right = Some(right);
                }

                // Which pane does input go to this frame? Latch on a button press so a
                // drag stays with its pane; otherwise follow the hovered half.
                let hovered = cursor.map(|c| if c.x < half_logical { 0 } else { 1 });
                for ev in &events {
                    if let ViewportEvent::MouseButton { state, .. } = ev {
                        match state {
                            ButtonState::Pressed => panes.drag_pane = hovered,
                            ButtonState::Released => panes.drag_pane = None,
                        }
                    }
                }
                let active = panes.drag_pane.or(hovered);
                let pane_vp = [half_logical, hl];

                // Left pane: resize, spin the cube (Z-up), route input if active, render.
                let cube = panes.cube;
                if let Some(left) = panes.left.as_mut() {
                    left.resize(&device, [left_w.max(1), ph]);
                    if let Some(cube) = cube {
                        left.session
                            .scene_mut()
                            .set_local_transform(cube, glam::Mat4::from_rotation_z(ctx.time()));
                    }
                    let left_events: Vec<ViewportEvent> =
                        events.iter().map(|e| remap(e, 0.0)).collect();
                    left.render(&device, &queue, pane_vp, ppp, active == Some(0), &left_events);
                }
                if panes.left.as_ref().is_some_and(|p| p.blit.is_none()) {
                    let blit = ctx
                        .renderer_mut()
                        .create_blit(&device, panes.left.as_ref().unwrap().target.render_view());
                    panes.left.as_mut().unwrap().blit = Some(blit);
                }

                // Right pane: static torus; its events shift into pane-local space.
                if let Some(right) = panes.right.as_mut() {
                    right.resize(&device, [right_w.max(1), ph]);
                    let right_events: Vec<ViewportEvent> =
                        events.iter().map(|e| remap(e, half_logical)).collect();
                    right.render(&device, &queue, pane_vp, ppp, active == Some(1), &right_events);
                }
                if panes.right.as_ref().is_some_and(|p| p.blit.is_none()) {
                    let blit = ctx
                        .renderer_mut()
                        .create_blit(&device, panes.right.as_ref().unwrap().target.render_view());
                    panes.right.as_mut().unwrap().blit = Some(blit);
                }
            },
            move |pctx| {
                let panes = paint_state.borrow();
                let [pw, ph] = pctx.surface_size();
                let left_w = pw / 2;
                if let Some(blit) = panes.left.as_ref().and_then(|p| p.blit.as_ref()) {
                    pctx.blit_rect(blit, 0, 0, left_w, ph);
                }
                if let Some(blit) = panes.right.as_ref().and_then(|p| p.blit.as_ref()) {
                    pctx.blit_rect(blit, left_w, 0, pw - left_w, ph);
                }
            },
        )
        .run();
}
