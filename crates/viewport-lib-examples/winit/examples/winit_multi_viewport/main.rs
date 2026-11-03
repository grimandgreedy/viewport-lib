//! Multi-viewport example : quad-view CAD layout.
//!
//! Shows how to use pass().prepare_hdr_viewport and pass_view().paint_hdr_blit
//! to render four independent viewports into a shared surface texture. Each
//! viewport is rendered into its own HDR intermediate texture in the prepare
//! step, then blitted into its quadrant of the surface via a scissored render
//! pass with no depth attachment.
//!
//! ```text
//! +----------------------------------+
//! |  Perspective  |  Top (ortho)     |
//! |   (orbit)     |  looking -Z      |
//! +----------------------------------+
//! | Front (ortho) | Right (ortho)    |
//! |  looking -Y   |  looking -X      |
//! +----------------------------------+
//! ```
//!
//! The scene is built once and shared across all viewports. Each viewport gets
//! its own camera and OrbitCameraController. Mouse input is routed to
//! whichever quadrant the cursor is currently in, and that quadrant draws a
//! slightly blue background to show the per-viewport background setting.

use std::sync::Arc;
use std::time::Instant;
use viewport_lib as vpl;
use viewport_lib::Colour;

use vpl::PostProcessSettings;
use vpl::input::adapters::{AdapterCtx, winit as vpl_winit};
use vpl::wgpu;
use vpl::{
    BindingPreset, Camera, CameraFrame, FrameData, InputOwnership, LightingSettings, MeshId,
    OrbitCameraController, SceneFrame, SceneRenderItem, ViewportContext, ViewportEvent, ViewportId,
    ViewportInput, ViewportRenderer, primitives,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");
}

// ---------------------------------------------------------------------------
// Quadrant helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Quad {
    TopLeft = 0,
    TopRight = 1,
    BottomLeft = 2,
    BottomRight = 3,
}

/// The quadrants in resolver order.
const QUADS: [Quad; 4] = [
    Quad::TopLeft,
    Quad::TopRight,
    Quad::BottomLeft,
    Quad::BottomRight,
];

fn quad_from_pos(x: f32, y: f32, w: f32, h: f32) -> Quad {
    let half_w = w / 2.0;
    let half_h = h / 2.0;
    match (x >= half_w, y >= half_h) {
        (false, false) => Quad::TopLeft,
        (true, false) => Quad::TopRight,
        (false, true) => Quad::BottomLeft,
        (true, true) => Quad::BottomRight,
    }
}

/// Return (x_offset, y_offset, width, height) for a quadrant in physical pixels.
fn quad_rect(quad: Quad, w: u32, h: u32) -> (u32, u32, u32, u32) {
    let hw = w / 2;
    let hh = h / 2;
    // Use ceiling for the second half to handle odd dimensions.
    let hw2 = w - hw;
    let hh2 = h - hh;
    match quad {
        Quad::TopLeft => (0, 0, hw, hh),
        Quad::TopRight => (hw, 0, hw2, hh),
        Quad::BottomLeft => (0, hh, hw, hh2),
        Quad::BottomRight => (hw, hh, hw2, hh2),
    }
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,

    renderer: ViewportRenderer,
    mesh_id: MeshId,

    /// One camera + controller per quadrant.
    cameras: [Camera; 4],
    controllers: [OrbitCameraController; 4],
    /// When the last frame was drawn, for the frame time fly movement reads.
    last_frame: Instant,
    /// One resolver per quadrant: each owns its bindings and resolves the events
    /// routed to it. The controllers keep only their sensitivities.
    inputs: [ViewportInput; 4],

    /// Viewport handles (one per quadrant, created in order).
    viewports: [ViewportId; 4],

    /// Which quadrant the mouse is currently over.
    hovered_quad: Quad,
    cursor_pos: glam::Vec2,
    /// Whether the window has keyboard focus.
    focused: bool,
}

impl AppState {
    /// Who owns the input in `quad`: the quadrant under the cursor has what the
    /// window has, and the others have nothing, since events go to that one alone.
    fn quad_ownership(&self, quad: Quad) -> InputOwnership {
        if quad == self.hovered_quad {
            vpl_winit::ownership(self.focused, true, false, false, false)
        } else {
            InputOwnership::NONE
        }
    }

    /// Update every quadrant's ownership between frames, when the hovered
    /// quadrant or the window focus changes.
    fn set_ownership(&mut self) {
        for (i, quad) in QUADS.into_iter().enumerate() {
            let ownership = self.quad_ownership(quad);
            self.inputs[i].set_ownership(ownership);
        }
    }

    /// Build the scene items (a small grid of coloured cubes). Shared across all viewports.
    fn build_scene(&self) -> SceneFrame {
        let positions_colours: &[([f32; 3], [f32; 4])] = &[
            ([-2.0, -2.0, 0.0], [0.9, 0.3, 0.3, 1.0]),
            ([2.0, -2.0, 0.0], [0.3, 0.9, 0.3, 1.0]),
            ([-2.0, 2.0, 0.0], [0.3, 0.3, 0.9, 1.0]),
            ([2.0, 2.0, 0.0], [0.9, 0.9, 0.3, 1.0]),
            ([0.0, 0.0, 2.0], [0.9, 0.5, 0.1, 1.0]),
        ];
        let items: Vec<SceneRenderItem> = positions_colours
            .iter()
            .map(|&(pos, colour)| {
                let mut item = SceneRenderItem::default();
                item.mesh_id = self.mesh_id;
                item.model = glam::Mat4::from_translation(glam::Vec3::from(pos)).into();
                item.settings.colour = Some(Colour::linear_rgb(colour[0], colour[1], colour[2]));
                item
            })
            .collect();
        SceneFrame::from_surface_items(items)
    }

    /// Build a FrameData for the given quadrant.
    fn build_frame(&self, quad: Quad, vp_id: ViewportId, w: u32, h: u32) -> FrameData {
        let (ox, oy, qw, qh) = quad_rect(quad, w, h);
        let _ = (ox, oy); // pixel offsets not needed for owned() path

        let cam = &self.cameras[quad as usize];
        let size = [qw as f32, qh as f32];
        let camera_frame = CameraFrame::from_camera(cam, size).with_viewport_id(vp_id);

        let mut fd = FrameData::new(camera_frame, self.build_scene());
        fd.effects.lighting = LightingSettings::default();
        fd.effects.post_process = {
            let mut _t = PostProcessSettings::default();
            _t.bloom.enabled = true;
            _t
        };
        fd.viewport.show_grid = true;
        fd.viewport.show_axes_indicator = true;
        // Each viewport sets its own background: the one under the cursor is
        // tinted slightly blue against the default #3b3b40.
        if quad == self.hovered_quad {
            fd.viewport.background_colour = Some(vpl::Colour::srgb_u8(0x3b, 0x40, 0x4e, 0xff));
        }
        fd
    }
}

// ---------------------------------------------------------------------------
// ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("viewport-lib : Multi-Viewport (quad-view)")
                        .with_inner_size(winit::dpi::LogicalSize::new(1200u32, 900u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No suitable GPU adapter");

        let required_features = ViewportRenderer::recommended_device_features(&adapter);
        if !required_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
            eprintln!("INDIRECT_FIRST_INSTANCE not supported -- GPU culling will be disabled");
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits: ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }))
        .expect("Failed to create device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut renderer = ViewportRenderer::new(&device, format);
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .expect("cube mesh");

        // Create four viewport slots.
        let vp0 = renderer.create_viewport(&device); // TopLeft:     perspective
        let vp1 = renderer.create_viewport(&device); // TopRight:    top (ortho)
        let vp2 = renderer.create_viewport(&device); // BottomLeft:  front (ortho)
        let vp3 = renderer.create_viewport(&device); // BottomRight: right (ortho)

        // Perspective camera : standard orbit view.
        let cam_persp = Camera::orbit(
            glam::DVec3::ZERO,
            12.0,
            glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1),
        );

        // Top view: looking straight down (-Z direction).
        // from_rotation_x(-PI/2) puts the camera directly above looking down.
        let cam_top = Camera::orbit(
            glam::DVec3::ZERO,
            10.0,
            glam::Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        )
        .orthographic();

        // Front view: looking along -Y.
        let cam_front = Camera::orbit(glam::DVec3::ZERO, 10.0, glam::Quat::IDENTITY).orthographic();

        // Right view: looking along -X (rotate 90 degrees around Z then front).
        let cam_right = Camera::orbit(
            glam::DVec3::ZERO,
            10.0,
            glam::Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2),
        )
        .orthographic();

        let make_controller = || {
            let mut c = OrbitCameraController::new_stateless();
            // Pan reads the viewport height off the controller.
            c.set_viewport_size([config.width as f32 / 2.0, config.height as f32 / 2.0]);
            c
        };
        // The window opens focused with the top-left quadrant hovered.
        let make_input = |hovered: bool, size: [f32; 2]| {
            let mut i = ViewportInput::from_preset(BindingPreset::Viewer);
            let ownership = if hovered {
                vpl_winit::ownership(true, true, false, false, false)
            } else {
                InputOwnership::NONE
            };
            i.begin_frame(ViewportContext::new(ownership, size));
            i
        };

        let hw = config.width as f32 / 2.0;
        let hh = config.height as f32 / 2.0;
        let ctrl0 = make_controller();
        let ctrl1 = make_controller();
        let ctrl2 = make_controller();
        let ctrl3 = make_controller();

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_config: config,
            renderer,
            mesh_id,
            cameras: [cam_persp, cam_top, cam_front, cam_right],
            controllers: [ctrl0, ctrl1, ctrl2, ctrl3],
            last_frame: Instant::now(),
            inputs: [
                make_input(true, [hw, hh]),
                make_input(false, [hw, hh]),
                make_input(false, [hw, hh]),
                make_input(false, [hw, hh]),
            ],
            viewports: [vp0, vp1, vp2, vp3],
            hovered_quad: Quad::TopLeft,
            cursor_pos: glam::Vec2::ZERO,
            focused: true,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.surface_config.width = new_size.width;
                    state.surface_config.height = new_size.height;
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    state.window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                let surf_frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                    Err(e) => {
                        eprintln!("Surface error: {e:?}");
                        return;
                    }
                };

                let output_view = surf_frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let w = state.surface_config.width;
                let h = state.surface_config.height;

                // Apply accumulated events to each camera.
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                for (i, ((cam, ctrl), input)) in state
                    .cameras
                    .iter_mut()
                    .zip(state.controllers.iter_mut())
                    .zip(state.inputs.iter_mut())
                    .enumerate()
                {
                    let action_frame = input.resolve();
                    ctrl.apply(cam, &action_frame, dt);
                    let (_, _, qw, qh) = quad_rect(
                        [
                            Quad::TopLeft,
                            Quad::TopRight,
                            Quad::BottomLeft,
                            Quad::BottomRight,
                        ][i],
                        w,
                        h,
                    );
                    cam.set_aspect_ratio(qw as f32, qh as f32);
                }

                // Build per-viewport FrameData.
                let quads = QUADS;
                let frames: [FrameData; 4] =
                    std::array::from_fn(|i| state.build_frame(quads[i], state.viewports[i], w, h));

                // Prepare: scene once (shared), then per-viewport GPU data and HDR intermediate texture.
                let (scene_fx, _) = frames[0].effects.split();
                let token = state.renderer.pass().prepare_scene(
                    &state.device,
                    &state.queue,
                    &frames[0],
                    &scene_fx,
                );
                for (i, frame) in frames.iter().enumerate() {
                    state.renderer.pass().prepare_viewport(
                        &state.device,
                        &state.queue,
                        &token,
                        state.viewports[i],
                        frame,
                    );
                }

                // Render each viewport into its own HDR intermediate texture.
                let hdr_cmds: Vec<wgpu::CommandBuffer> = frames
                    .iter()
                    .enumerate()
                    .map(|(i, frame)| {
                        state.renderer.pass().prepare_hdr_viewport(
                            &state.device,
                            &state.queue,
                            state.viewports[i],
                            frame,
                        )
                    })
                    .collect();
                state.queue.submit(hdr_cmds.into_iter());

                // Blit each viewport's HDR result into its quadrant of the surface.
                // No depth attachment: the blit pass is a fullscreen quad per slot.
                let mut encoder =
                    state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("hdr_blit"),
                        });
                {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("hdr_blit_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &output_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    for (i, frame) in frames.iter().enumerate() {
                        let (qx, qy, qw, qh) = quad_rect(quads[i], w, h);
                        rp.set_viewport(qx as f32, qy as f32, qw as f32, qh as f32, 0.0, 1.0);
                        rp.set_scissor_rect(qx, qy, qw, qh);
                        state
                            .renderer
                            .pass_view()
                            .paint_hdr_blit_no_ds(&mut rp, frame);
                    }
                }
                state.queue.submit(std::iter::once(encoder.finish()));
                surf_frame.present();

                // Begin next frame's event accumulation for each quadrant resolver.
                for i in 0..4 {
                    let quad = quads[i];
                    let (_, _, qw, qh) = quad_rect(quad, w, h);
                    let ownership = state.quad_ownership(quad);
                    state.inputs[i]
                        .begin_frame(ViewportContext::new(ownership, [qw as f32, qh as f32]));
                    // Pan reads the viewport height off the controller.
                    state.controllers[i].set_viewport_size([qw as f32, qh as f32]);
                }
            }

            // Everything else goes through the adapter in window pixels and is
            // routed to the quadrant under the cursor, in that quadrant's local
            // coordinates. Focus loss reaches every quadrant.
            other => {
                if let WindowEvent::Focused(focused) = other {
                    state.focused = focused;
                    state.set_ownership();
                }
                let mut events = Vec::new();
                vpl_winit::translate(&other, &AdapterCtx::default(), &mut events);
                if events.is_empty() {
                    return;
                }
                let (w, h) = (state.surface_config.width, state.surface_config.height);
                for ev in events {
                    match ev {
                        ViewportEvent::FocusLost => {
                            for input in &mut state.inputs {
                                input.push_event(ViewportEvent::FocusLost);
                            }
                            continue;
                        }
                        ViewportEvent::PointerMoved { position } => {
                            let new_quad =
                                quad_from_pos(position.x, position.y, w as f32, h as f32);
                            if new_quad != state.hovered_quad {
                                // Leave old quadrant, and hand the pointer to the new
                                // one before this move reaches it.
                                state.inputs[state.hovered_quad as usize]
                                    .push_event(ViewportEvent::PointerLeft);
                                state.hovered_quad = new_quad;
                                state.set_ownership();
                            }
                        }
                        _ => {}
                    }
                    let (ox, oy, _, _) = quad_rect(state.hovered_quad, w, h);
                    let local = ev.offset_by(glam::Vec2::new(ox as f32, oy as f32));
                    if let ViewportEvent::PointerMoved { position } = local {
                        state.cursor_pos = position;
                    }
                    state.inputs[state.hovered_quad as usize].push_event(local);
                }
                state.window.request_redraw();
            }
        }
    }
}
