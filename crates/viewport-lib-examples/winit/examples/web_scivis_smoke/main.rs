//! Web smoke check: the CPU paths a scientific-visualisation consumer depends
//! on, run in a browser.
//!
//! The render path on WebGPU was already covered by `winit-web`. What was not
//! covered is everything that happens on the CPU before a frame is drawn:
//! iso-surface extraction, the pick accelerator's BVH build, and a ray query
//! against it. Those use the workspace's parallelism module, which has no
//! threads to use on this target, and they were never exercised here.
//!
//! So this example runs them and reports what happened, on the page rather than
//! only in the console:
//!
//!   1. Build a scalar field large enough that iso-surface extraction takes its
//!      slab-decomposition branch, which is the branch that asks for a thread
//!      pool.
//!   2. Extract an iso-surface from it and count the triangles.
//!   3. Upload that surface, put it in a scene, build a pick accelerator over
//!      it, and fire a ray down the view axis.
//!   4. Draw the result, and report the extraction and pick outcomes.
//!
//! A green line per step means that path works in a browser. Anything else is
//! the thing to go and fix. Click the surface to run a fresh pick at the cursor.
//!
//! Native runs too, which is what keeps it honest: the same code, the same
//! assertions, one target with threads and one without.

use std::collections::HashMap;
use std::sync::Arc;

use web_time::Instant;

use viewport_lib::vplg::marching_cubes::{VolumeData, extract_isosurface};
use viewport_lib::picking::screen_to_ray;
use viewport_lib::{
    Aabb, ButtonState, Camera, CameraFrame, FrameData, Material, MouseButton, OrbitCameraController,
    PickAccelerator, Scene, SceneFrame, SceneRenderItem, ScrollUnits, ViewportContext,
    ViewportEvent, ViewportRenderer, wgpu,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton as WinitMouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowAttributes, WindowId};

/// Grid resolution per axis. Iso-surface extraction splits into slabs above
/// 64^3 cells, so this sits above that: the serial path would otherwise be the
/// one under test, and the parallel path is the one that has never run here.
const GRID: u32 = 80;

/// How many lines the setup phase reports. A click appends one more and the
/// oldest click line is dropped, so the page shows the setup results plus the
/// latest click rather than growing without bound.
const SETUP_LINES: usize = 5;

/// The field is a smooth blob, so the iso-surface is a closed shell a ray can
/// be relied on to hit.
fn scalar_field(dims: u32) -> VolumeData {
    let n = dims as usize;
    let mut data = vec![0.0f32; n * n * n];
    let c = (dims as f32 - 1.0) * 0.5;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let dx = (x as f32 - c) / c;
                let dy = (y as f32 - c) / c;
                let dz = (z as f32 - c) / c;
                // A sphere modulated by a low-frequency ripple, so the surface
                // has enough detail to produce a realistic triangle count.
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let ripple = 0.08 * (6.0 * dx).sin() * (6.0 * dy).sin() * (6.0 * dz).sin();
                data[x + y * n + z * n * n] = r + ripple;
            }
        }
    }
    let spacing = 2.0 / (dims as f32 - 1.0);
    VolumeData {
        data,
        dims: [dims, dims, dims],
        origin: [-1.0, -1.0, -1.0],
        spacing: [spacing, spacing, spacing],
    }
}

/// What each checked step reported, in the order they ran.
#[derive(Default)]
struct Results {
    lines: Vec<(bool, String)>,
}

impl Results {
    fn pass(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        log_line(&format!("ok: {msg}"));
        self.lines.push((true, msg));
    }

    fn fail(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        log_line(&format!("FAILED: {msg}"));
        self.lines.push((false, msg));
    }

    /// Write the report into the page's `#results` list, if there is one.
    #[cfg(target_arch = "wasm32")]
    fn publish(&self) {
        let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };
        let Some(host) = doc.get_element_by_id("results") else {
            return;
        };
        host.set_inner_html("");
        for (ok, msg) in &self.lines {
            let Ok(li) = doc.create_element("li") else {
                continue;
            };
            li.set_class_name(if *ok { "ok" } else { "bad" });
            li.set_text_content(Some(msg));
            let _ = host.append_child(&li);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn publish(&self) {}
}

fn log_line(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    log::info!("{msg}");
    #[cfg(not(target_arch = "wasm32"))]
    println!("{msg}");
}

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
    /// Kept so a click can re-run a pick against the same geometry.
    accelerator: PickAccelerator,
    mesh_lookup: HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    /// Last cursor position, in physical pixels from the viewport top-left.
    cursor: glam::Vec2,
    results: Results,
}

struct App {
    proxy: EventLoopProxy<State>,
    state: Option<State>,
    building: bool,
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() || self.building {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default().with_title("viewport-lib: web scivis"))
                .expect("window"),
        );
        self.building = true;

        #[cfg(target_arch = "wasm32")]
        {
            let proxy = self.proxy.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let state = build_state(window).await;
                let _ = proxy.send_event(state);
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(build_state(window));
            let _ = self.proxy.send_event(state);
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, state: State) {
        self.building = false;
        state.results.publish();
        state.window.request_redraw();
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.surface_config.width = size.width.max(1);
                state.surface_config.height = size.height.max(1);
                state.surface.configure(&state.device, &state.surface_config);
                state.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.cursor = glam::Vec2::new(position.x as f32, position.y as f32);
                state.controller.push_event(ViewportEvent::PointerMoved {
                    position: state.cursor,
                });
                state.window.request_redraw();
            }
            WindowEvent::MouseInput {
                button: WinitMouseButton::Left,
                state: element_state,
                ..
            } => {
                state.controller.push_event(ViewportEvent::MouseButton {
                    button: MouseButton::Left,
                    state: match element_state {
                        ElementState::Pressed => ButtonState::Pressed,
                        ElementState::Released => ButtonState::Released,
                    },
                });
                // A release is the click: run a pick at the cursor and report
                // it, so the CPU pick path is exercised on demand and not only
                // once during setup.
                if element_state == ElementState::Released {
                    state.pick_at_cursor();
                }
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
                state.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

impl State {
    /// Cast a ray through the cursor and report what it hit. The same accelerator
    /// and lookup the setup pick used, so a click re-runs exactly that path.
    fn pick_at_cursor(&mut self) {
        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;
        let view_proj = self.camera.proj_matrix() * self.camera.view_matrix();
        let (origin, dir) = screen_to_ray(self.cursor, glam::Vec2::new(w, h), view_proj.inverse());
        match self.accelerator.pick(origin, dir, &self.mesh_lookup) {
            Some(hit) => self.results.pass(format!(
                "click pick hit at ({:.2}, {:.2}, {:.2})",
                hit.world_pos[0], hit.world_pos[1], hit.world_pos[2]
            )),
            None => self.results.pass("click pick: no hit (pointing past the surface)"),
        }
        // Keep the report to the setup lines plus the most recent click.
        if self.results.lines.len() > SETUP_LINES + 1 {
            self.results.lines.remove(SETUP_LINES);
        }
        self.results.publish();
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                log_line(&format!("surface error: {e:?}"));
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;

        self.controller.apply_to_camera(&mut self.camera);
        self.camera.set_aspect_ratio(w, h);

        let frame_data = FrameData::new(
            CameraFrame::from_camera(&self.camera, [w, h]),
            SceneFrame::from_surface_items(self.scene_items.clone()),
        );
        let cmd = self
            .renderer
            .owned()
            .render(&self.device, &self.queue, &view, &frame_data);
        self.queue.submit(std::iter::once(cmd));
        frame.present();

        self.controller.begin_frame(ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [w, h],
        });
    }
}

async fn build_state(window: Arc<Window>) -> State {
    let mut results = Results::default();

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
    results.pass("WebGPU adapter and device");

    // Step 1 and 2: build the field and extract a surface from it. This is the
    // call that would panic outright if the parallelism module still reached
    // for a thread pool that cannot exist here.
    let volume = scalar_field(GRID);
    let cells = (GRID as usize - 1).pow(3);
    let t0 = Instant::now();
    let mesh = extract_isosurface(&volume, 0.65);
    let extract_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let tris = mesh.indices.len() / 3;
    if tris > 0 {
        results.pass(format!(
            "iso-surface extraction: {tris} triangles from {cells} cells in {extract_ms:.0} ms"
        ));
    } else {
        results.fail("iso-surface extraction produced no triangles");
    }

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
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
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .expect("upload extracted surface");
    results.pass("extracted surface uploaded to the GPU");

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.material = Material::from_colour([0.35, 0.72, 0.95]);
    let scene_items = vec![item];

    // Step 3: the pick accelerator builds a BVH over the scene, which is the
    // other CPU path a consumer hits on the first click.
    let mut scene = Scene::new();
    scene.add(Some(mesh_id), glam::Mat4::IDENTITY, Material::default());
    scene.update_transforms();

    let bounds = Aabb::from_positions(&mesh.positions);
    let t1 = Instant::now();
    let mut accelerator = PickAccelerator::build_from_scene(&scene, |_| Some(bounds));
    let build_ms = t1.elapsed().as_secs_f32() * 1000.0;
    results.pass(format!("pick accelerator built in {build_ms:.1} ms"));

    let mut mesh_lookup = HashMap::new();
    mesh_lookup.insert(
        mesh_id.index() as u64,
        (mesh.positions.clone(), mesh.indices.clone()),
    );

    // Step 4: a ray straight down the view axis, which the blob is centred on.
    let t2 = Instant::now();
    let hit = accelerator.pick(
        glam::Vec3::new(0.0, 0.0, 6.0),
        glam::Vec3::new(0.0, 0.0, -1.0),
        &mesh_lookup,
    );
    let pick_ms = t2.elapsed().as_secs_f32() * 1000.0;
    match hit {
        Some(h) => results.pass(format!(
            "CPU pick hit at ({:.2}, {:.2}, {:.2}) in {pick_ms:.2} ms",
            h.world_pos[0], h.world_pos[1], h.world_pos[2]
        )),
        None => results.fail("CPU pick returned no hit against a surface on the ray"),
    }

    let camera = Camera {
        distance: 4.0,
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
        accelerator,
        mesh_lookup,
        cursor: glam::Vec2::ZERO,
        results,
    }
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
