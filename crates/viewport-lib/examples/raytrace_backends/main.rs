//! Path tracer with a software / hardware traversal toggle.
//!
//! Opens a window and path-traces a small Z-up scene (a ground plane and three
//! spheres: diffuse, clear glass, and metal) under a procedural HDR environment.
//! It builds two tracers on the same device: one forced to the portable compute
//! traversal, one to the hardware `rayQuery` backend (which needs a Vulkan/DX12
//! GPU with ray query). Press B (or Space) to switch which one renders; the
//! output should be identical, so the toggle is a live A/B check that the
//! hardware traversal matches the software one.
//!
//! On a device without ray query (Metal, the web) the hardware tracer falls back
//! to the compute traversal, and the toggle shows "Software" for both. Run it on
//! a ray-query GPU to exercise the real hardware path.
//!
//! Controls: left-drag orbits, scroll zooms, B / Space toggles the backend.
//!
//! Usage:
//!   cargo run --release --example raytrace-backends \
//!       --features raytrace,raytrace-hardware

use std::sync::Arc;

use glam::{Mat4, Vec3};
use viewport_lib::primitives;
use viewport_lib::raytrace::{
    RtBackend, RtCamera, RtLight, RtMaterial, RtScene, RtSettings, Tracer,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

/// Samples to converge to when the camera is idle.
const TARGET_SPP: u32 = 512;
/// Samples added per idle frame, so each progressive step stays short.
const CHUNK: u32 = 24;

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop error");
}

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

    // Two tracers on the same device and scene: index 0 forced to the software
    // compute traversal, index 1 to the hardware rayQuery backend (or a software
    // fallback when the device has no ray query). `active` selects which renders.
    tracers: [Tracer; 2],
    active: usize,
    // Whether tracer 1 actually got the hardware backend, for honest labelling.
    hardware_available: bool,

    // Blit pipeline: draw the traced texture over a fullscreen triangle.
    blit: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    tex_bind_group: Option<wgpu::BindGroup>,

    // Orbit camera (Z-up).
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,

    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    // Set after a scroll to resume convergence once zooming stops (scroll has no
    // release event to trigger it).
    settle_at: Option<std::time::Instant>,
}

/// A procedural equirect HDR environment: a Z-up sky gradient with a bright sun
/// disc aligned to the scene's directional light. Fills the same projection the
/// tracer samples (longitude around +Z, latitude with +Z at the top).
fn procedural_env(w: u32, h: u32, sun_dir: Vec3) -> Vec<f32> {
    use std::f32::consts::PI;
    let sun = sun_dir.normalize();
    let ground = Vec3::new(0.20, 0.18, 0.16);
    let horizon = Vec3::new(0.72, 0.78, 0.88);
    let zenith = Vec3::new(0.18, 0.40, 0.90);
    let mut px = vec![0.0f32; (w * h * 4) as usize];
    for y in 0..h {
        let v = (y as f32 + 0.5) / h as f32;
        let theta = (0.5 - v) * PI; // +pi/2 at the top (+Z)
        let (st, ct) = theta.sin_cos();
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let phi = (u - 0.5) * 2.0 * PI;
            let dir = Vec3::new(ct * phi.cos(), ct * phi.sin(), st);

            // Sky: ground below the horizon, horizon->zenith above.
            let mut c = if dir.z >= 0.0 {
                horizon.lerp(zenith, dir.z.powf(0.6))
            } else {
                ground.lerp(horizon, (dir.z + 1.0).clamp(0.0, 1.0))
            };

            // Sun: a small bright disc plus a soft glow.
            let d = dir.dot(sun).clamp(-1.0, 1.0);
            if d > 0.9995 {
                c = Vec3::splat(10.0);
            } else {
                c += Vec3::new(1.0, 0.85, 0.6) * (d.max(0.0).powf(64.0)) * 3.0;
            }

            let i = ((y * w + x) * 4) as usize;
            px[i] = c.x;
            px[i + 1] = c.y;
            px[i + 2] = c.z;
            px[i + 3] = 1.0;
        }
    }
    px
}

/// Build the demo scene once.
fn build_scene() -> RtScene {
    let mut scene = RtScene::new();
    let sun_dir = Vec3::new(0.3, -0.4, 0.85);
    scene.set_environment(&procedural_env(512, 256, sun_dir), 512, 256);

    let add =
        |scene: &mut RtScene, mesh: &viewport_lib::MeshData, origin: Vec3, mat: RtMaterial| {
            let positions: Vec<Vec3> = mesh
                .positions
                .iter()
                .map(|p| Vec3::from(*p) + origin)
                .collect();
            let normals: Vec<Vec3> = mesh.normals.iter().map(|n| Vec3::from(*n)).collect();
            scene.add_mesh(&positions, &mesh.indices, Some(&normals), mat);
        };

    let ground = primitives::cuboid(24.0, 24.0, 0.4);
    add(
        &mut scene,
        &ground,
        Vec3::new(0.0, 0.0, -0.2),
        RtMaterial {
            base_colour: [0.6, 0.6, 0.62],
            roughness: 0.9,
            ..RtMaterial::default()
        },
    );

    let sphere = primitives::sphere(1.0, 48, 24);
    add(
        &mut scene,
        &sphere,
        Vec3::new(-2.6, 0.0, 1.0),
        RtMaterial {
            base_colour: [0.75, 0.12, 0.10],
            roughness: 0.6,
            ..RtMaterial::default()
        },
    );
    add(
        &mut scene,
        &sphere,
        Vec3::new(0.0, 0.0, 1.0),
        RtMaterial {
            base_colour: [0.95, 0.98, 1.0],
            roughness: 0.05,
            transmission: 1.0,
            ior: 1.5,
            ..RtMaterial::default()
        },
    );
    add(
        &mut scene,
        &sphere,
        Vec3::new(2.6, 0.0, 1.0),
        RtMaterial {
            base_colour: [0.95, 0.85, 0.55],
            metallic: 1.0,
            roughness: 0.15,
            ..RtMaterial::default()
        },
    );

    scene.add_light(RtLight::Directional {
        direction: [0.3, -0.4, 0.85],
        colour: [3.0, 2.9, 2.7],
    });
    scene
}

impl AppState {
    /// Eye position from the current orbit angles (Z-up).
    fn eye(&self) -> Vec3 {
        let dir = Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        );
        self.target + dir * self.distance
    }

    /// The active tracer's backend, for the window title.
    fn active_label(&self) -> &'static str {
        match self.tracers[self.active].backend() {
            RtBackend::Hardware => "Hardware (rayQuery)",
            RtBackend::Software => {
                if self.active == 1 && !self.hardware_available {
                    "Software (hardware unavailable)"
                } else {
                    "Software (compute)"
                }
            }
        }
    }

    /// Put the active backend and sample count in the window title.
    fn update_title(&self) {
        self.window.set_title(&format!(
            "viewport-lib : path tracer [{}]  {} spp  (B toggles)",
            self.active_label(),
            self.tracers[self.active].accumulated_samples(),
        ));
    }

    /// Switch the active backend, restart its accumulation, and redraw.
    fn toggle_backend(&mut self) {
        self.active ^= 1;
        self.tracers[self.active].reset_accumulation();
        self.update_title();
        self.window.request_redraw();
    }

    /// Reset the accumulation of whichever tracer is active.
    fn reset_active(&mut self) {
        self.tracers[self.active].reset_accumulation();
    }

    /// Render one frame with the active tracer and upload it. While interacting,
    /// traces a fresh cheap denoised preview at reduced resolution. When idle,
    /// progressively accumulates full-resolution samples toward `TARGET_SPP`.
    /// Returns true when more accumulation is pending.
    fn render(&mut self) -> bool {
        let sw = self.surface_config.width.max(1);
        let sh = self.surface_config.height.max(1);
        // Cap the trace resolution; the blit stretches it to fill the window.
        let full_w = sw.min(900);
        let full_h = ((full_w as f32) * (sh as f32) / (sw as f32))
            .round()
            .max(1.0) as u32;

        let eye = self.eye();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Z);

        // "Interacting" covers both an active drag and the brief settle window
        // after a scroll (which has no release event): both want the cheap
        // preview, and convergence resumes once input stops.
        let interacting = self.dragging || self.settle_at.is_some();

        // Drop the resolution while interacting so it stays responsive.
        let (tw, th) = if interacting {
            let dw = full_w.min(480);
            (
                dw,
                ((dw as f32) * (full_h as f32) / (full_w as f32))
                    .round()
                    .max(1.0) as u32,
            )
        } else {
            (full_w, full_h)
        };

        let proj = Mat4::perspective_rh(42f32.to_radians(), tw as f32 / th as f32, 0.1, 100.0);
        let camera = RtCamera {
            inv_view_proj: (proj * view).inverse(),
            position: eye,
            width: tw,
            height: th,
        };

        let tracer = &mut self.tracers[self.active];
        let (img, pending) = if interacting {
            // Fresh denoised low-res preview each frame the camera moves.
            let settings = RtSettings {
                samples: 6,
                max_bounces: 4,
                denoise: true,
                seed: 0,
            };
            (
                tracer.trace(&self.device, &self.queue, &camera, &settings),
                false,
            )
        } else {
            // Idle: keep adding samples to the same image until it converges.
            if tracer.accumulated_samples() >= TARGET_SPP {
                return false; // converged; leave the current texture on screen
            }
            let settings = RtSettings {
                samples: CHUNK,
                max_bounces: 8,
                denoise: false,
                seed: 0,
            };
            let img = tracer.accumulate(&self.device, &self.queue, &camera, &settings);
            let pending = tracer.accumulated_samples() < TARGET_SPP;
            (img, pending)
        };

        self.upload(&img, tw, th);
        self.update_title();
        pending
    }

    /// Tone-map the traced HDR image and upload it as the blit texture.
    fn upload(&mut self, img: &viewport_lib::raytrace::RtImage, tw: u32, th: u32) {
        // Reinhard tone map to linear 8-bit; the sRGB surface applies gamma.
        let mut bytes = vec![0u8; (tw * th * 4) as usize];
        for (i, px) in img.rgba.chunks_exact(4).enumerate() {
            for c in 0..3 {
                let mapped = px[c] / (1.0 + px[c]);
                bytes[i * 4 + c] = (mapped.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            bytes[i * 4 + 3] = 255;
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rt_preview"),
            size: wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tw * 4),
                rows_per_image: Some(th),
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.tex_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_blit_bg"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
    }
}

const BLIT_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let xy = corners[vi];
    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("viewport-lib : path tracer")
                        .with_inner_size(winit::dpi::LogicalSize::new(900u32, 600u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("adapter");

        // Ask for the ray-query feature when the adapter has it, so the hardware
        // tracer can build its acceleration structure and rayQuery kernel. Absent
        // it (Metal, the web), the hardware tracer falls back to the compute path.
        let ray_query = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        let has_ray_query = adapter.features().contains(ray_query);
        let wanted = if has_ray_query {
            ray_query
        } else {
            wgpu::Features::empty()
        };
        // `EXPERIMENTAL_RAY_QUERY` is an experimental feature, so requesting it
        // also needs the acknowledgement token or the device request fails with
        // `ExperimentalFeaturesNotEnabled`. Only opt in when actually asking for it.
        let experimental_features = if has_ray_query {
            // Safety: we only read the acceleration structure through the tracer's
            // own rayQuery kernel, which is the intended use of the feature.
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        // The acceleration-structure limits (max_blas_geometry_count and friends)
        // default to 0, so building a BLAS fails validation unless they are
        // raised; take the adapter's supported limits when using ray query.
        let required_limits = if has_ray_query {
            adapter.limits()
        } else {
            wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wanted,
            required_limits,
            experimental_features,
            ..Default::default()
        }))
        .expect("device");

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt_blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt_blit_layout"),
            bind_group_layouts: &[&blit_layout],
            push_constant_ranges: &[],
        });
        let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rt_blit_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rt_blit_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let scene = build_scene();
        // Build both tracers on the same device and scene. Tracer 1 requests the
        // hardware backend; it silently falls back to software without ray query.
        let sw_tracer = Tracer::new_with_backend(&device, &queue, &scene, RtBackend::Software);
        let hw_tracer = Tracer::new_with_backend(&device, &queue, &scene, RtBackend::Hardware);
        let hardware_available = hw_tracer.backend() == RtBackend::Hardware;
        println!(
            "path tracer backends: {} triangles. left-drag orbit, scroll zoom, B toggles backend.",
            scene.triangle_count()
        );
        if hardware_available {
            println!("hardware rayQuery backend available: B switches software <-> hardware.");
        } else {
            println!(
                "no ray-query device (e.g. Metal): the hardware tracer runs the software fallback."
            );
        }

        let mut state = AppState {
            window,
            surface,
            device,
            queue,
            surface_config: config,
            tracers: [sw_tracer, hw_tracer],
            active: 0,
            hardware_available,
            blit,
            blit_layout,
            sampler,
            tex_bind_group: None,
            yaw: -1.3,
            pitch: 0.42,
            distance: 9.3,
            target: Vec3::new(0.0, 0.0, 1.0),
            dragging: false,
            last_cursor: None,
            settle_at: None,
        };
        state.update_title();
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
                if size.width > 0 && size.height > 0 {
                    state.surface_config.width = size.width;
                    state.surface_config.height = size.height;
                    state
                        .surface
                        .configure(&state.device, &state.surface_config);
                    state.reset_active();
                    state.window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed && !event.repeat;
                let toggle = match &event.logical_key {
                    Key::Named(NamedKey::Space) => true,
                    Key::Character(s) => s.eq_ignore_ascii_case("b"),
                    _ => false,
                };
                if pressed && toggle {
                    state.toggle_backend();
                }
            }

            WindowEvent::MouseInput {
                state: btn_state,
                button: MouseButton::Left,
                ..
            } => {
                state.dragging = btn_state == ElementState::Pressed;
                if !state.dragging {
                    state.last_cursor = None;
                    // Released: resume convergence from a clean slate.
                    state.reset_active();
                    state.window.request_redraw();
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                if state.dragging {
                    if let Some((px, py)) = state.last_cursor {
                        let dx = (x - px) as f32;
                        let dy = (y - py) as f32;
                        state.yaw -= dx * 0.008;
                        state.pitch = (state.pitch + dy * 0.008).clamp(-1.45, 1.45);
                        // Camera moved: the running accumulation is stale.
                        state.reset_active();
                        state.window.request_redraw();
                    }
                }
                state.last_cursor = Some((x, y));
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                };
                state.distance = (state.distance * (1.0 - dy * 0.1)).clamp(2.5, 40.0);
                // Camera moved; push the settle deadline out so a preview renders
                // now and convergence resumes once scrolling stops.
                state.reset_active();
                state.settle_at =
                    Some(std::time::Instant::now() + std::time::Duration::from_millis(160));
                state.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let pending = state.render();
                if pending {
                    // Keep converging: schedule the next accumulation step.
                    state.window.request_redraw();
                }
                let frame = match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => {
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("rt_blit_encoder"),
                        });
                {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("rt_blit_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
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
                    if let Some(bg) = &state.tex_bind_group {
                        rp.set_pipeline(&state.blit);
                        rp.set_bind_group(0, bg, &[]);
                        rp.draw(0..3, 0..1);
                    }
                }
                state.queue.submit(std::iter::once(encoder.finish()));
                frame.present();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        // Drive the post-scroll settle: once no wheel event has arrived for the
        // debounce window, resume convergence. Until then, sleep exactly until the
        // deadline; when idle, wait for the next event.
        match state.settle_at {
            Some(deadline) if std::time::Instant::now() >= deadline => {
                state.settle_at = None;
                state.reset_active();
                state.window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}
