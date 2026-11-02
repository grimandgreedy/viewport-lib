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
//!   5. Register the built-in item types and draw one item of each type the
//!      tier runs, read back to check it covered pixels (`item_types.rs`), and
//!      below the full tier time the Gaussian splat sort, which runs on the CPU
//!      there.
//!
//! A green line per step means that path works in a browser. Anything else is
//! the thing to go and fix. Click the surface to run a fresh pick at the cursor.
//!
//! On a browser with no WebGPU adapter the page comes up on WebGL2, where the
//! renderer runs at the compatibility tier. The CPU steps run the same; the surface
//! draws through the basic mesh path, and a frame of it with the overlays,
//! drawn under error scopes, checks both on the real GL driver. A tier
//! without a lit group 0 refuses the upload. The first
//! line names the API the page landed on, which is what `scripts/check_web.sh`
//! reads to confirm each run took the path it asked for.
//!
//! Native runs too, which is what keeps it honest: the same code, the same
//! assertions, one target with threads and one without.

use std::collections::HashMap;
use std::sync::Arc;
use viewport_lib::Colour;
use viewport_lib::input::adapters::{AdapterCtx, winit as vpl_winit};

use web_time::Instant;

mod item_types;

use viewport_lib::picking::screen_to_ray;
use viewport_lib::vplg::volume::grid::VolumeData;
use viewport_lib::vplg::volume::marching_cubes::extract_isosurface;
use viewport_lib::{
    Aabb, BindingPreset, ButtonState, Camera, CameraFrame, FrameData, InputOwnership, LabelItem,
    Material, MaterialId, MeshId, MouseButton, OrbitCameraController, OverlayFill, OverlayShape,
    OverlayShapeItem, PickAccelerator, RendererConfig, Scene, SceneFrame, SceneRenderItem, Tier,
    ViewportContext, ViewportError, ViewportEvent, ViewportInput, ViewportRenderer, wgpu,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowAttributes, WindowId};

/// Grid resolution per axis. Iso-surface extraction splits into slabs above
/// 64^3 cells, so this sits above that: the serial path would otherwise be the
/// one under test, and the parallel path is the one that has never run here.
const GRID: u32 = 80;

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
    /// When the last frame was drawn, for the frame time fly movement reads.
    last_frame: Instant,
    /// Owns the bindings and resolves events into an `ActionFrame`. The
    /// controller keeps only its sensitivities and applies the frame.
    input: ViewportInput,
    scene_items: Vec<SceneRenderItem>,
    /// Kept so a click can re-run a pick against the same geometry.
    accelerator: PickAccelerator,
    mesh_lookup: HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    /// Last cursor position, in physical pixels from the viewport top-left.
    cursor: glam::Vec2,
    results: Results,
    /// How many lines the setup phase reported. A click appends one more and
    /// the oldest click line is dropped, so the page shows the setup results
    /// plus the latest click rather than growing without bound.
    setup_lines: usize,
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
                state
                    .surface
                    .configure(&state.device, &state.surface_config);
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            // The viewport fills the canvas, so it owns everything; positions
            // stay in physical pixels, the units the surface size is given in.
            other => {
                let mut events = Vec::new();
                vpl_winit::translate(&other, &AdapterCtx::default(), &mut events);
                if !events.is_empty() {
                    state.window.request_redraw();
                }
                for ev in events {
                    let left_release = matches!(
                        ev,
                        ViewportEvent::MouseButton {
                            button: MouseButton::Left,
                            state: ButtonState::Released,
                        }
                    );
                    if let ViewportEvent::PointerMoved { position } = ev {
                        state.cursor = position;
                    }
                    state.input.push_event(ev);
                    // A release is the click: run a pick at the cursor and report
                    // it, so the CPU pick path is exercised on demand and not only
                    // once during setup.
                    if left_release {
                        state.pick_at_cursor();
                    }
                }
            }
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
            None => self
                .results
                .pass("click pick: no hit (pointing past the surface)"),
        }
        // Keep the report to the setup lines plus the most recent click.
        if self.results.lines.len() > self.setup_lines + 1 {
            self.results.lines.remove(self.setup_lines);
        }
        self.results.publish();
    }

    fn render(&mut self) {
        let frame = match viewport_lib::gpu::acquire_surface(&self.surface) {
            viewport_lib::gpu::SurfaceFrame::Acquired(f) => f,
            viewport_lib::gpu::SurfaceFrame::Recreate => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            viewport_lib::gpu::SurfaceFrame::Skip => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;

        let action_frame = self.input.resolve();
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.controller.apply(&mut self.camera, &action_frame, dt);
        self.camera.set_aspect_ratio(w, h);

        let mut frame_data = FrameData::new(
            CameraFrame::from_camera(&self.camera, [w, h]),
            SceneFrame::from_surface_items(self.scene_items.clone()),
        );
        frame_data.overlays = tier_overlay(self.renderer.tier());
        let cmd = self
            .renderer
            .owned()
            .render(&self.device, &self.queue, &view, &frame_data);
        self.queue.submit(std::iter::once(cmd));
        viewport_lib::gpu::present(&self.queue, frame);

        self.input
            .begin_frame(ViewportContext::new(InputOwnership::full_window(), [w, h]));
        // Pan reads the viewport height off the controller.
        self.controller.set_viewport_size([w, h]);
    }
}

async fn build_state(window: Arc<Window>) -> State {
    let mut results = Results::default();

    // The surface comes from the canvas element rather than the winit window: a
    // window brings its own display handle, a bare canvas does not, and the
    // canvas is the target a host without winit hands over.
    #[cfg(target_arch = "wasm32")]
    let (_instance, surface, adapter) = viewport_lib::gpu::request_adapter_with_gl_fallback(
        wgpu::SurfaceTarget::Canvas(
            winit::platform::web::WindowExtWebSys::canvas(&*window).expect("winit made a canvas"),
        ),
        wgpu::PowerPreference::HighPerformance,
    )
    .await
    .expect("neither WebGPU nor WebGL2 gave an adapter");
    #[cfg(not(target_arch = "wasm32"))]
    let (surface, adapter) = {
        let instance = viewport_lib::gpu::default_instance();
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("adapter");
        (surface, adapter)
    };
    let backend = adapter.get_info().backend;

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
    let api = match backend {
        wgpu::Backend::BrowserWebGpu => "WebGPU".to_string(),
        wgpu::Backend::Gl if cfg!(target_arch = "wasm32") => "WebGL2".to_string(),
        other => format!("{other:?}"),
    };
    results.pass(format!("{api} adapter and device"));

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
    let surface_config = viewport_lib::gpu::runner_surface_config(
        format,
        size.width.max(1),
        size.height.max(1),
        wgpu::PresentMode::AutoVsync,
        caps.alpha_modes[0],
    );
    surface.configure(&device, &surface_config);

    let config = RendererConfig::new(format).with_adapter(&adapter);
    let mut renderer = ViewportRenderer::try_with_config(&device, &config)
        .expect("the renderer refused the device");
    let tier = renderer.tier();
    results.pass(format!("renderer built at the {tier:?} tier"));

    // The full and compatibility tiers draw the surface. Below them the upload is
    // refused, and the pick accelerator, which only needs a key per mesh, runs
    // on a placeholder.
    let mut scene_items = Vec::new();
    let mesh_id = if tier >= Tier::Compatibility {
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &mesh)
            .expect("upload extracted surface");
        results.pass("extracted surface uploaded to the GPU");
        let material = renderer
            .resources_mut()
            .upload_material(Material::from_colour(Colour::linear_rgb(0.35, 0.72, 0.95)))
            .expect("upload material");
        scene_items.push(SceneRenderItem::new(mesh_id, material));
        mesh_id
    } else {
        match renderer.resources_mut().upload_mesh_data(&device, &mesh) {
            Err(ViewportError::TierUnsupported { .. }) => {
                results.pass("mesh upload refused on this tier, as expected")
            }
            Ok(_) => results.fail("mesh upload succeeded on a tier that cannot draw it"),
            Err(e) => results.fail(format!("mesh upload failed with the wrong error: {e}")),
        }
        MeshId::INVALID
    };

    // A frame of the surface and the overlays under error scopes: on WebGL2
    // this is the first time their shaders meet the browser's GL driver. A
    // GLSL compile or link failure is an internal error there, not a
    // validation one, so both are watched.
    let probe = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("overlay probe"),
        size: wgpu::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let probe_view = probe.create_view(&wgpu::TextureViewDescriptor::default());
    let internal = viewport_lib::gpu::push_error_scope(&device, wgpu::ErrorFilter::Internal);
    let validation = viewport_lib::gpu::push_error_scope(&device, wgpu::ErrorFilter::Validation);
    let mut probe_frame = FrameData::new(
        CameraFrame::from_camera(&Camera::default(), [256.0, 128.0]),
        SceneFrame::from_surface_items(scene_items.clone()),
    );
    probe_frame.overlays = tier_overlay(tier);
    let cmd = renderer
        .owned()
        .render(&device, &queue, &probe_view, &probe_frame);
    queue.submit(std::iter::once(cmd));
    let validation = validation.pop().await;
    let internal = internal.pop().await;
    match validation.or(internal) {
        None => results.pass("surface and overlay frame drawn with no GPU error"),
        Some(e) => results.fail(format!("surface and overlay frame: {e}")),
    }

    if tier >= Tier::Compatibility {
        let mut gpu = item_types::Gpu {
            device: &device,
            queue: &queue,
            renderer: &mut renderer,
            format,
        };
        item_types::check(&mut gpu, &mut results).await;
    }

    // Step 3: the pick accelerator builds a BVH over the scene, which is the
    // other CPU path a consumer hits on the first click.
    let mut scene = Scene::new();
    scene.add(Some(mesh_id), glam::Mat4::IDENTITY, MaterialId::DEFAULT);
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

    let camera = Camera::orbit(glam::DVec3::ZERO, 4.0, Camera::default().orientation());
    let controller = OrbitCameraController::new_stateless();
    let mut input = ViewportInput::from_preset(BindingPreset::Viewer);
    input.begin_frame(ViewportContext::new(
        InputOwnership::full_window(),
        [surface_config.width as f32, surface_config.height as f32],
    ));

    State {
        window,
        surface,
        device,
        queue,
        surface_config,
        renderer,
        camera,
        controller,
        last_frame: Instant::now(),
        input,
        scene_items,
        accelerator,
        mesh_lookup,
        cursor: glam::Vec2::ZERO,
        setup_lines: results.lines.len(),
        results,
    }
}

/// A panel and a label naming the tier, so the page shows which path it is on
/// and the overlay shape and text pipelines both draw every frame.
fn tier_overlay(tier: Tier) -> viewport_lib::OverlayFrame {
    let mut overlays = viewport_lib::OverlayFrame::default();
    overlays.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 6.0 },
            [8.0, 8.0],
            [220.0, 34.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.05, 0.05, 0.1, 0.8))),
    );
    overlays.labels.push(
        LabelItem::new(format!("tier: {tier:?}"))
            .with_position([18.0, 16.0])
            .with_font_size(16.0)
            .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
    );
    overlays
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
