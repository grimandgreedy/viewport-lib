//! winit-web: run a viewport in the browser on WebGPU.
//!
//! This is the hand-written winit loop (compare `examples/winit_hdr`), adapted
//! for the web. The viewport-lib parts are identical to desktop: build a
//! `ViewportRenderer`, upload meshes, and call `renderer.owned().render()` each
//! frame. Only the bring-up differs, because a browser cannot do what the
//! desktop runner does:
//!
//!   - the adapter and device requests are awaited, not `block_on`-ed (blocking
//!     the browser main thread is not allowed);
//!   - the window is a `<canvas>` appended to the document;
//!   - the event loop is started with `spawn_app`, which returns immediately,
//!     instead of `run_app`, which would block.
//!
//! The whole file still compiles and runs natively (`cargo run --example
//! winit-web`): the web-only pieces are behind `cfg(target_arch = "wasm32")`
//! and the native path keeps the usual `block_on` + `run_app` shape.
//!
//! Target WebGPU, not WebGL2. viewport-lib's lit mesh path binds storage
//! buffers in the fragment stage and the HDR pipeline uses compute; neither
//! exists on WebGL2. On WebGPU the storage-buffer count sits exactly at the
//! spec baseline (8), which is what the base renderer needs.
//!
//! Build and serve (needs `wasm-bindgen-cli` and any static file server):
//!
//!   rustup target add wasm32-unknown-unknown
//!   cargo build --release --target wasm32-unknown-unknown \
//!       --example winit-web --features wgpu27
//!   wasm-bindgen --target web --no-typescript --out-dir examples/winit_web/pkg \
//!       target/wasm32-unknown-unknown/release/examples/winit-web.wasm
//!   # then serve examples/winit_web/ over http and open index.html in a
//!   # WebGPU-capable browser (recent Chrome/Edge, or Safari/Firefox nightly).

use std::sync::Arc;

use viewport_lib::{
    ButtonState, Camera, CameraFrame, EffectsFrame, FrameData, Material, MeshId,
    OrbitCameraController, PostProcessSettings, SceneFrame, SceneRenderItem, ScrollUnits,
    ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowAttributes, WindowId};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

const CANVAS_W: u32 = 1280;
const CANVAS_H: u32 = 720;

/// Everything a running frame needs. On desktop it is built synchronously in
/// `resumed`; on the web it is built by an async task and delivered back into
/// the event loop as a user event (`App::user_event`).
struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: ViewportRenderer,
    camera: Camera,
    controller: OrbitCameraController,
    scene_items: Vec<SceneRenderItem>,
    /// Index of the item spun about the world up axis each frame.
    spin_item: usize,
    start: Instant,
}

struct App {
    /// Delivers the async-built `State` back into the loop on the web. Unused on
    /// desktop, where `State` is built inline in `resumed`.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    proxy: EventLoopProxy<State>,
    state: Option<State>,
    /// Set once the async build is in flight so `resumed` firing again does not
    /// kick off a second one. Only meaningful on the web.
    building: bool,
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() || self.building {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("viewport-lib : web (WebGPU)")
                        .with_inner_size(winit::dpi::LogicalSize::new(CANVAS_W, CANVAS_H)),
                )
                .expect("window"),
        );

        // On the web the window is a canvas that winit created detached; size it
        // and put it in the document so it is visible and the surface has real
        // dimensions to configure against.
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            // Do not set the canvas width/height here: winit's resize observer
            // owns the backing store, deriving it from the CSS size (fixed to the
            // viewport in index.html) times the device pixel ratio. Setting it by
            // hand fights that and drives an unbounded resize feedback loop.
            let canvas = window.canvas().expect("winit created a canvas");
            let doc = web_sys::window()
                .and_then(|w| w.document())
                .expect("a document");
            let body = doc.body().expect("a document body");
            body.append_child(&canvas).expect("append canvas");
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.building = true;
            let proxy = self.proxy.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let state = build_state(window).await;
                // The loop takes ownership from here; if the receiver is gone the
                // page is closing, so dropping the state is fine.
                let _ = proxy.send_event(state);
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(build_state(window)));
        }
    }

    /// The async GPU build finished (web only): install the state and draw.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, state: State) {
        state.window.request_redraw();
        self.state = Some(state);
        self.building = false;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(sz) => {
                if sz.width > 0 && sz.height > 0 {
                    state.surface_config.width = sz.width;
                    state.surface_config.height = sz.height;
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    state.window.request_redraw();
                }
            }

            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                let vp_button = match button {
                    MouseButton::Left => viewport_lib::MouseButton::Left,
                    MouseButton::Middle => viewport_lib::MouseButton::Middle,
                    MouseButton::Right => viewport_lib::MouseButton::Right,
                    _ => return,
                };
                let vp_state = if btn_state == ElementState::Pressed {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                };
                state.controller.push_event(ViewportEvent::MouseButton {
                    button: vp_button,
                    state: vp_state,
                });
                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let pos = glam::Vec2::new(position.x as f32, position.y as f32);
                state
                    .controller
                    .push_event(ViewportEvent::PointerMoved { position: pos });
                state.window.request_redraw();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let (d, units) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        (glam::Vec2::new(x, y), ScrollUnits::Lines)
                    }
                    MouseScrollDelta::PixelDelta(px) => (
                        glam::Vec2::new(px.x as f32, px.y as f32),
                        ScrollUnits::Pixels,
                    ),
                };
                state
                    .controller
                    .push_event(ViewportEvent::Wheel { delta: d, units });
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                    Err(e) => {
                        log_error(&format!("surface error: {e:?}"));
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let w = state.surface_config.width as f32;
                let h = state.surface_config.height as f32;

                state.controller.apply_to_camera(&mut state.camera);
                state.camera.set_aspect_ratio(w, h);

                // Z-up: spin one item about the world up axis.
                let t = state.start.elapsed().as_secs_f32();
                state.scene_items[state.spin_item].model =
                    glam::Mat4::from_rotation_z(t).to_cols_array_2d();

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&state.camera, [w, h]),
                    SceneFrame::from_surface_items(state.scene_items.clone()),
                );
                let mut effects = EffectsFrame::default();
                effects.post_process = {
                    let mut pp = PostProcessSettings::default();
                    pp.bloom.enabled = true;
                    pp.bloom.threshold = 1.0;
                    pp.bloom.intensity = 0.15;
                    pp
                };
                frame_data.effects = effects;

                // The full HDR pipeline: prepare, shadow pass, HDR scene,
                // post-process, tone map, blit to the surface view.
                let cmd =
                    state
                        .renderer
                        .owned()
                        .render(&state.device, &state.queue, &view, &frame_data);
                state.queue.submit(std::iter::once(cmd));
                frame.present();

                state.controller.begin_frame(ViewportContext {
                    hovered: true,
                    focused: true,
                    viewport_size: [w, h],
                });

                // Keep the loop spinning for the animation.
                state.window.request_redraw();
            }

            _ => {}
        }
    }
}

/// Build the GPU state and initial scene. Async so the web path can await the
/// adapter and device; on desktop it is driven to completion with `block_on`.
async fn build_state(window: Arc<Window>) -> State {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).expect("surface");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .expect("adapter (is WebGPU available in this browser?)");

    let required_features = ViewportRenderer::recommended_device_features(&adapter);
    // WebGPU's default max_texture_dimension_2d is 8192, but the renderer's HDR
    // and post-process targets can exceed the surface size (supersampling), so
    // raise it to whatever the adapter supports to leave headroom on HiDPI.
    let mut required_limits = ViewportRenderer::recommended_device_limits(&adapter);
    required_limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits,
            ..Default::default()
        })
        .await
        .expect("device");

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    // Prefer sRGB: the tone mapper writes linear values and relies on the
    // hardware sRGB conversion to encode gamma.
    let format = caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(caps.formats[0]);
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    let mut renderer = ViewportRenderer::new(&device, format);
    let res = renderer.resources_mut();
    let m_sphere = res
        .upload_mesh_data(&device, &primitives::sphere(0.6, 24, 12))
        .unwrap();
    let m_cube = res
        .upload_mesh_data(&device, &primitives::cube(1.0))
        .unwrap();
    let m_torus = res
        .upload_mesh_data(&device, &primitives::torus(0.5, 0.18, 32, 16))
        .unwrap();

    let make_item = |mesh_id: MeshId, [x, y, z]: [f32; 3], colour: [f32; 3]| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
        item.material = Material::from_colour(colour);
        // A little emissive above 1.0 gives bloom some HDR energy to extract.
        item.material.emissive = [colour[0] * 1.2, colour[1] * 1.2, colour[2] * 1.2];
        item
    };
    let scene_items = vec![
        make_item(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2]),
        make_item(m_cube, [0.0, 0.0, 0.0], [0.4, 0.6, 0.9]),
        make_item(m_torus, [2.5, 0.0, 0.0], [0.3, 0.8, 0.4]),
    ];

    let camera = Camera {
        distance: 10.0,
        ..Camera::default()
    };
    let mut controller = OrbitCameraController::viewport_primitives();
    controller.begin_frame(ViewportContext {
        hovered: true,
        focused: true,
        viewport_size: [surface_config.width as f32, surface_config.height as f32],
    });

    State {
        window,
        surface,
        device,
        queue,
        surface_config,
        renderer,
        camera,
        controller,
        scene_items,
        spin_item: 1,
        start: Instant::now(),
    }
}

fn log_error(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    log::error!("{msg}");
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{msg}");
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Info);
    }

    let event_loop = EventLoop::<State>::with_user_event()
        .build()
        .expect("event loop");
    let proxy = event_loop.create_proxy();
    let app = App {
        proxy,
        state: None,
        building: false,
    };

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).expect("run");
    }
}
