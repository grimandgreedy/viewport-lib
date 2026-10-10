//! Does a pipeline compile on another thread stall the thread that renders?
//!
//! Raw wgpu, no renderer: worker threads build render pipelines from
//! generated shaders while the main thread draws a full-screen triangle each
//! frame and times its encode, submit and poll separately. Every shader
//! carries a per-run nonce, so the driver's shader cache never has it and each
//! compile is cold without clearing the cache.
//!
//! ```bash
//! cargo run --release --example compile_stall_probe
//! ```
//!
//! `VPL_STALL_WORKERS` (default 4) and `VPL_STALL_PIPELINES` (per worker,
//! default 24) size the compile load; `VPL_STALL_LINES` (default 400) is the
//! length of each generated shader. `VPL_STALL_CHURN=1` also has each frame
//! create a buffer, write it, create a bind group and a small shader module,
//! the per-frame resource work a renderer does, timed as `churn`.
//! `VPL_STALL_RENDERER=1` draws a frame of four boxes through a
//! `ViewportRenderer` instead, so the timing covers the library's own
//! per-frame work; its whole frame lands in `encode`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use viewport_lib::wgpu;
use viewport_lib_testkit::{DeviceProfile, device::headless_device_with_info};

const SIZE: u32 = 512;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A fragment shader long enough to take the driver a while, unique to
/// `seed` and this run's `nonce`.
fn heavy_shader(nonce: u64, seed: usize, lines: usize) -> String {
    let mut s = String::from(
        "@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {\n\
         let x = f32(i32(i) / 2) * 4.0 - 1.0; let y = f32(i32(i) % 2) * 4.0 - 1.0;\n\
         return vec4<f32>(x, y, 0.0, 1.0); }\n\
         @fragment fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {\n\
         var v = p.xy * 0.001;\n",
    );
    for k in 0..lines {
        let a = ((nonce as usize ^ (seed * 7919 + k * 104729)) % 10007) as f32 / 10007.0;
        s.push_str(&format!(
            "v = vec2<f32>(sin(v.x * {:.6} + v.y), cos(v.y * {:.6} - v.x)) + vec2<f32>({:.6});\n",
            1.0 + a,
            1.0 + a * 0.5,
            a * 0.01
        ));
    }
    s.push_str("return vec4<f32>(v, 0.0, 1.0); }\n");
    s
}

fn pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::RenderPipeline {
    use viewport_lib::plugin_api::builders;
    let module = builders::wgsl_module(device, label, source);
    let layout = builders::pipeline_layout(device, label, &[]);
    builders::render_pipeline(
        device,
        builders::RenderPipelineDesc {
            label,
            layout: &layout,
            vertex_module: &module,
            vertex_entry: "vs_main",
            vertex_buffers: &[],
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            cache: None,
        },
    )
}

#[derive(Default, Clone, Copy)]
struct Frame {
    churn: f32,
    encode: f32,
    submit: f32,
    poll: f32,
}

impl Frame {
    fn total(&self) -> f32 {
        self.churn + self.encode + self.submit + self.poll
    }
}

fn summary(label: &str, frames: &[Frame]) {
    let pct = |mut v: Vec<f32>, q: f32| {
        v.sort_by(f32::total_cmp);
        v[((v.len() - 1) as f32 * q) as usize]
    };
    let col = |f: fn(&Frame) -> f32| frames.iter().map(f).collect::<Vec<_>>();
    println!("{label} ({} frames)", frames.len());
    for (name, values) in [
        ("frame ", col(Frame::total)),
        ("churn ", col(|f| f.churn)),
        ("encode", col(|f| f.encode)),
        ("submit", col(|f| f.submit)),
        ("poll  ", col(|f| f.poll)),
    ] {
        println!(
            "  {name}  p50 {:7.2} ms  p95 {:7.2} ms  max {:7.2} ms",
            pct(values.clone(), 0.5),
            pct(values.clone(), 0.95),
            pct(values, 1.0)
        );
    }
}

fn main() {
    let profile = DeviceProfile::high_performance("compile-stall-probe");
    let (device, queue, info) = headless_device_with_info(&profile).expect("no adapter");
    println!("adapter: {:?} / {}", info.backend, info.name);
    let workers = env_or("VPL_STALL_WORKERS", 4);
    let per_worker = env_or("VPL_STALL_PIPELINES", 24);
    let lines = env_or("VPL_STALL_LINES", 400);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let target = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("stall_target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    // The main thread's own pipeline, compiled before anything is timed.
    let draw_pipeline = pipeline(&device, "stall_draw", &heavy_shader(nonce, usize::MAX, 8));

    // `1` for all four calls, or one of `buffer`, `write`, `bind_group`,
    // `module` for that call alone (`write` and `bind_group` reuse one
    // buffer made up front).
    let churn = std::env::var("VPL_STALL_CHURN").ok();
    let churn_on = churn.is_some();
    let does = |op: &str| churn.as_deref().is_some_and(|c| c == "1" || c == op);
    let spare_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stall_spare_buf"),
        size: 256,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let churn_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stall_churn_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let churn_frame = std::cell::Cell::new(0u64);
    let draw = |frames: &mut Vec<Frame>| {
        let t = Instant::now();
        if churn_on {
            let made = does("buffer").then(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("stall_churn_buf"),
                    size: 256,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            let buf = made.as_ref().unwrap_or(&spare_buf);
            if does("write") {
                queue.write_buffer(buf, 0, &[1u8; 256]);
            }
            if does("bind_group") {
                let _bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stall_churn_bg"),
                    layout: &churn_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
            }
            churn_frame.set(churn_frame.get() + 1);
        }
        if does("module") {
            let _m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("stall_churn_module"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "const K: f32 = {}.0;\n@compute @workgroup_size(1) fn main() {{}}",
                        churn_frame.get()
                    )
                    .into(),
                ),
            });
        }
        let churn = t.elapsed().as_secs_f32() * 1000.0;
        let t = Instant::now();
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&draw_pipeline);
            pass.draw(0..3, 0..1);
        }
        let cmd = enc.finish();
        let encode = t.elapsed().as_secs_f32() * 1000.0;
        let t = Instant::now();
        queue.submit(std::iter::once(cmd));
        let submit = t.elapsed().as_secs_f32() * 1000.0;
        let t = Instant::now();
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let poll = t.elapsed().as_secs_f32() * 1000.0;
        frames.push(Frame {
            churn,
            encode,
            submit,
            poll,
        });
    };

    let mut renderer = std::env::var_os("VPL_STALL_RENDERER").map(|_| {
        let mut renderer = viewport_lib::ViewportRenderer::new(&device, FORMAT);
        renderer.set_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
        let mesh = renderer
            .resources_mut()
            .upload_mesh_data(&device, &viewport_lib::primitives::cube(1.0))
            .expect("upload cube");
        let mut frame = viewport_lib::FrameData::default();
        frame.camera.render_camera =
            viewport_lib::RenderCamera::from_camera(&viewport_lib::Camera::default());
        frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
        let items: Vec<viewport_lib::SceneRenderItem> = (0..4)
            .map(|i| {
                let mut it = viewport_lib::SceneRenderItem::default();
                it.mesh_id = mesh;
                it.model =
                    glam::Mat4::from_translation(glam::Vec3::new(i as f32 * 2.5 - 4.0, 0.0, 0.0))
                        .to_cols_array_2d();
                it
            })
            .collect();
        frame.scene.surfaces = viewport_lib::SurfaceSubmission::Flat(items.into());
        // Build everything this frame needs before anything is timed.
        for _ in 0..3 {
            renderer.render_to_texture(&device, &queue, &target, &frame);
        }
        (renderer, frame)
    });
    let mut draw_frame = |frames: &mut Vec<Frame>| match renderer.as_mut() {
        Some((renderer, frame)) => {
            let t = Instant::now();
            renderer.render_to_texture(&device, &queue, &target, frame);
            let encode = t.elapsed().as_secs_f32() * 1000.0;
            let t = Instant::now();
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            frames.push(Frame {
                encode,
                poll: t.elapsed().as_secs_f32() * 1000.0,
                ..Default::default()
            });
        }
        None => draw(frames),
    };

    let mut idle = Vec::new();
    for _ in 0..300 {
        draw_frame(&mut idle);
    }
    summary("idle", &idle);

    let done = Arc::new(AtomicBool::new(false));
    let compile_times = Arc::new(Mutex::new(Vec::<f32>::new()));
    let started = Instant::now();
    let handles: Vec<_> = (0..workers)
        .map(|w| {
            let device = device.clone();
            let compile_times = Arc::clone(&compile_times);
            std::thread::Builder::new()
                .name(format!("stall-compile-{w}"))
                .spawn(move || {
                    for p in 0..per_worker {
                        let src = heavy_shader(nonce, w * 1000 + p, lines);
                        let t = Instant::now();
                        let _ = pipeline(&device, "stall_compile", &src);
                        compile_times
                            .lock()
                            .unwrap()
                            .push(t.elapsed().as_secs_f32() * 1000.0);
                    }
                })
                .unwrap()
        })
        .collect();
    let watcher = {
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            for h in handles {
                h.join().unwrap();
            }
            done.store(true, Ordering::SeqCst);
        })
    };
    println!("pid {} compiling from now", std::process::id());
    let mut during = Vec::new();
    while !done.load(Ordering::SeqCst) {
        draw_frame(&mut during);
    }
    watcher.join().unwrap();
    let workers_ms = started.elapsed().as_secs_f32() * 1000.0;
    summary("while compiling", &during);
    let mut c = compile_times.lock().unwrap().clone();
    c.sort_by(f32::total_cmp);
    println!(
        "compiles: {} in {:.0} ms, each p50 {:.1} ms max {:.1} ms",
        c.len(),
        workers_ms,
        c[c.len() / 2],
        c[c.len() - 1]
    );
    // Long enough for a sampler started late to see the end state too.
    std::thread::sleep(Duration::from_millis(10));
}
