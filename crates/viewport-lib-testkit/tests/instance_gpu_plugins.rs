//! GPU plugins driven by a `ViewportInstance`: the runtime's hooks run from
//! the instance's render calls, and the scene colour and depth a frame leaves
//! behind are what a plugin's `post_paint` receives.

use viewport_lib::plugin_api::depth::DEPTH_CLEAR;
use viewport_lib::runtime::gpu_plugin::gpu_phase;
use viewport_lib::wgpu;
use viewport_lib::{GpuFrameContext, GpuPlugin, NoticeDetail, PostPaintTargets, ViewportRuntime};
use viewport_lib::{
    InputOwnership, Material, OffscreenViewportTarget, PipelineMode, RendererConfig,
    ViewportContext, ViewportInstance, primitives,
};
use viewport_lib::{PostEffectContext, PostEffectResizeContext, PostEffectStage};
use viewport_lib_testkit::fixtures::{CallLog, LoggingGpuPlugin};
use viewport_lib_testkit::{headless_device, orbit_camera};

const SIZE: u32 = 64;
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// An instance drawing one cube that covers the centre of the frame.
fn instance(device: &wgpu::Device) -> ViewportInstance {
    let mut config = RendererConfig::new(OffscreenViewportTarget::render_format(SURFACE_FORMAT))
        .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    if device
        .features()
        .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8)
    {
        config = config.with_scene_depth_format(wgpu::TextureFormat::Depth32FloatStencil8);
    }
    let mut instance = ViewportInstance::with_config(device, &config);
    let material = instance
        .resources_mut()
        .upload_material(Material::default())
        .expect("material");
    instance
        .upload_and_add(
            device,
            &primitives::cube(2.0),
            glam::Mat4::IDENTITY,
            material,
        )
        .expect("mesh");
    *instance.camera_mut() = orbit_camera(glam::Vec3::ZERO, 5.0, 0.6, 1.0);
    instance
}

/// Assemble one frame at the test size.
fn assemble(instance: &mut ViewportInstance, mode: PipelineMode) {
    instance.effects_mut().display.mode = mode;
    instance.frame(ViewportContext::new(
        InputOwnership::default(),
        [SIZE as f32, SIZE as f32],
    ));
}

/// Assemble and render two frames into `view`.
fn render(
    instance: &mut ViewportInstance,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    mode: PipelineMode,
) {
    instance.effects_mut().display.mode = mode;
    for _ in 0..2 {
        instance.frame(ViewportContext::new(
            InputOwnership::default(),
            [SIZE as f32, SIZE as f32],
        ));
        let cmd = instance.render(device, queue, view);
        queue.submit(std::iter::once(cmd));
    }
}

/// A compute pass reading the depth at the centre of a depth-aspect view and
/// the largest depth anywhere in it, the way a plugin would sample it.
struct DepthProbe {
    pipeline: wgpu::ComputePipeline,
    storage: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl DepthProbe {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth_probe"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> out: array<f32, 2>;
@compute @workgroup_size(1)
fn main() {
    let dims = textureDimensions(depth_tex);
    out[0] = textureLoad(depth_tex, vec2<i32>(dims / 2u), 0);
    var m = 0.0;
    for (var y = 0u; y < dims.y; y++) {
        for (var x = 0u; x < dims.x; x++) {
            m = max(m, textureLoad(depth_tex, vec2<i32>(i32(x), i32(y)), 0));
        }
    }
    out[1] = m;
}
"#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("depth_probe"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depth_probe_out"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depth_probe_staging"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            storage,
            staging,
        }
    }

    fn encode(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::CommandBuffer {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth_probe"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.storage.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.storage, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.storage, 0, &self.staging, 0, 8);
        encoder.finish()
    }

    /// The last encoded probe's result, once the GPU has run it.
    fn read(&self, device: &wgpu::Device) -> [f32; 2] {
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .expect("probe poll");
        let values = {
            let mapped = viewport_lib::gpu::mapped_range(self.staging.slice(..));
            let read = |i: usize| f32::from_le_bytes(mapped[i * 4..i * 4 + 4].try_into().unwrap());
            [read(0), read(1)]
        };
        self.staging.unmap();
        values
    }
}

/// Probe `view` now.
fn probe_depth(device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView) -> [f32; 2] {
    let probe = DepthProbe::new(device);
    queue.submit(std::iter::once(probe.encode(device, view)));
    probe.read(device)
}

/// After an HDR frame, and after an LDR frame into a sampleable offscreen
/// target, the instance hands out a scene colour and depth, both sampleable.
#[test]
fn scene_targets_are_sampleable_on_hdr_and_ldr() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    for mode in [PipelineMode::Hdr, PipelineMode::Direct] {
        let mut instance = instance(&device);
        render(&mut instance, &device, &queue, target.render_view(), mode);
        let targets = instance
            .scene_target_views(Some(target.render_view()))
            .unwrap_or_else(|| panic!("{mode:?}: no scene targets"));
        for (name, view) in [
            ("colour", targets.color_view),
            ("depth", targets.depth_view),
        ] {
            assert!(
                view.texture()
                    .usage()
                    .contains(wgpu::TextureUsages::TEXTURE_BINDING),
                "{mode:?}: {name} is not sampleable"
            );
        }
        // The probe binds the depth view as a depth texture, so a view that
        // cannot be sampled that way fails validation here.
        let _ = probe_depth(&device, &queue, targets.depth_view);
    }
}

/// An LDR frame into a target without `TEXTURE_BINDING` (a window surface)
/// has no readable colour, so there are no scene targets.
#[test]
fn ldr_frame_into_an_unsampleable_target_has_no_scene_targets() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let texture = surface_like_texture(&device);
    let view = texture.create_view(&Default::default());
    let mut instance = instance(&device);
    render(&mut instance, &device, &queue, &view, PipelineMode::Direct);
    assert!(instance.scene_target_views(Some(&view)).is_none());
}

/// A colour target without `TEXTURE_BINDING`, like a window surface.
fn surface_like_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_like_target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OffscreenViewportTarget::render_format(SURFACE_FORMAT),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

/// How a test drives one frame through the instance.
#[derive(Clone, Copy, Debug)]
enum Path {
    OwnedHdr,
    OwnedLdr,
    PassHdr,
    PassLdr,
}

impl Path {
    fn mode(self) -> PipelineMode {
        match self {
            Path::OwnedHdr | Path::PassHdr => PipelineMode::Hdr,
            Path::OwnedLdr | Path::PassLdr => PipelineMode::Direct,
        }
    }

    /// Assemble and render one frame. The pass paths only prepare: the hooks
    /// run there, and painting needs a host pass the test has no use for.
    fn frame(
        self,
        instance: &mut ViewportInstance,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &OffscreenViewportTarget,
    ) {
        assemble(instance, self.mode());
        match self {
            Path::OwnedHdr | Path::OwnedLdr => {
                let cmd = instance.render(device, queue, target.render_view());
                queue.submit(std::iter::once(cmd));
            }
            Path::PassHdr | Path::PassLdr => {
                let buffers = instance.prepare(device, queue);
                queue.submit(buffers);
            }
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
    }
}

/// On the owned HDR, owned LDR (into an offscreen target) and pass HDR paths,
/// the instance initialises an attached runtime's GPU plugin once, then runs
/// `pre_prepare` before each frame and `post_paint` after it, with the format
/// of the scene colour it hands over.
#[test]
fn instance_runs_both_hooks_where_the_frame_is_readable() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    let hdr = format!("{:?}", wgpu::TextureFormat::Rgba16Float);
    let ldr = format!("{:?}", target.format());
    for (path, format) in [
        (Path::OwnedHdr, &hdr),
        (Path::OwnedLdr, &ldr),
        (Path::PassHdr, &hdr),
    ] {
        let log = CallLog::new();
        let runtime = ViewportRuntime::new()
            .with_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
        let mut instance = instance(&device).with_runtime(runtime);
        for _ in 0..2 {
            path.frame(&mut instance, &device, &queue, &target);
        }
        let entries = log.entries();
        assert_eq!(
            entries,
            [
                "init_gpu".to_string(),
                "pre_prepare:0".to_string(),
                format!("post_paint:0:{format}:pick=false"),
                "pre_prepare:1".to_string(),
                format!("post_paint:1:{format}:pick=false"),
            ],
            "{path:?}"
        );
    }
}

/// Where the frame leaves nothing a plugin can read, `pre_prepare` still runs
/// but `post_paint` is not called, and the renderer says why in a notice: an
/// LDR frame painted into the host's pass, and an LDR frame rendered into a
/// target without `TEXTURE_BINDING` (a window surface).
#[test]
fn post_paint_is_skipped_with_a_notice_where_the_frame_is_unreadable() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    let surface = surface_like_texture(&device);
    let surface_view = surface.create_view(&Default::default());
    for path in ["pass", "surface"] {
        let log = CallLog::new();
        let runtime = ViewportRuntime::new()
            .with_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
        let mut instance = instance(&device).with_runtime(runtime);
        for _ in 0..2 {
            if path == "pass" {
                Path::PassLdr.frame(&mut instance, &device, &queue, &target);
            } else {
                assemble(&mut instance, PipelineMode::Direct);
                let cmd = instance.render(&device, &queue, &surface_view);
                queue.submit(std::iter::once(cmd));
            }
        }
        log.assert_take(&["init_gpu", "pre_prepare:0", "pre_prepare:1"]);
        let skipped: Vec<_> = instance
            .renderer_mut()
            .take_notices()
            .into_iter()
            .filter(|n| matches!(n.detail, NoticeDetail::PostPaintSkipped { .. }))
            .collect();
        assert_eq!(skipped.len(), 1, "{path}: one coalesced notice");
        assert_eq!(skipped[0].count, 2, "{path}: reported on both frames");
    }
}

/// Probes the scene depth it is handed in `post_paint`.
struct DepthProbePlugin {
    probe: std::sync::Arc<DepthProbe>,
}

impl GpuPlugin for DepthProbePlugin {
    fn post_paint(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        targets: &PostPaintTargets<'_>,
        _ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        vec![self.probe.encode(device, targets.depth_view)]
    }
}

/// With GPU plugins attached, the scene depth a plugin samples in
/// `post_paint` holds the frame's geometry, on default settings: on HDR with
/// the axes indicator on (its overlay pass ends the frame on the scene
/// depth), and on LDR without occlusion culling.
#[test]
fn post_paint_sees_the_frames_depth() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    for path in [Path::OwnedHdr, Path::OwnedLdr, Path::PassHdr] {
        let probe = std::sync::Arc::new(DepthProbe::new(&device));
        let runtime = ViewportRuntime::new().with_gpu_plugin(DepthProbePlugin {
            probe: probe.clone(),
        });
        let mut instance = instance(&device).with_runtime(runtime);
        assert!(instance.viewport_frame_mut().show_axes_indicator);
        assert!(!instance.renderer_mut().occlusion_culling_enabled());
        for _ in 0..2 {
            path.frame(&mut instance, &device, &queue, &target);
        }
        assert!(instance.renderer_mut().debug_last_frame_kept_scene_depth());
        let [centre, max] = probe.read(&device);
        assert!(
            centre > DEPTH_CLEAR && max > DEPTH_CLEAR,
            "{path:?}: the depth should hold the cube, got centre {centre}, max {max}"
        );
    }
}

/// The renderer keeps the scene depth only while a runtime with GPU plugins
/// is attached: with no runtime, or an empty one, the frame ends on the
/// usual discard.
#[test]
fn scene_depth_is_kept_only_with_gpu_plugins() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    for path in [Path::OwnedHdr, Path::OwnedLdr, Path::PassHdr] {
        for (label, runtime, kept) in [
            ("no runtime", None, false),
            ("empty runtime", Some(ViewportRuntime::new()), false),
            (
                "GPU plugin",
                Some(
                    ViewportRuntime::new()
                        .with_gpu_plugin(LoggingGpuPlugin::new(CallLog::new(), 0)),
                ),
                true,
            ),
        ] {
            let mut instance = instance(&device);
            if let Some(runtime) = runtime {
                instance = instance.with_runtime(runtime);
            }
            path.frame(&mut instance, &device, &queue, &target);
            assert_eq!(
                instance.renderer_mut().debug_last_frame_kept_scene_depth(),
                kept,
                "{path:?}, {label}"
            );
        }
    }
}

/// A GPU plugin that records the `viewport_id` and `dt` it is called with.
struct ContextPlugin {
    log: CallLog,
}

impl GpuPlugin for ContextPlugin {
    fn pre_prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        self.log.record(format!(
            "pre_prepare:{:?}:{}:{}x{}",
            ctx.viewport_id.map(|id| id.index()),
            ctx.dt,
            ctx.viewport_size.x,
            ctx.viewport_size.y
        ));
        Vec::new()
    }
}

/// The instance calls the hooks with its own viewport id: a plugin scoped to
/// `viewport_id()` runs, one scoped to another viewport does not, and an
/// unscoped one runs once per render. The context carries the `dt` of the
/// last `step_runtime` and the physical viewport size.
#[test]
fn instance_scopes_gpu_plugins_to_its_viewport() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    let mut instance = instance(&device);
    let own = instance.viewport_id();
    // A second viewport on the same renderer, which the instance does not
    // render. The first id `create_viewport` hands out is the instance's own.
    let first = instance.renderer_mut().create_viewport(&device);
    assert_eq!(first, own);
    let other = instance.renderer_mut().create_viewport(&device);
    assert_ne!(own, other);

    let scoped = CallLog::new();
    let elsewhere = CallLog::new();
    let unscoped = CallLog::new();
    let runtime = ViewportRuntime::new()
        .with_gpu_plugin_for_viewport(
            own,
            ContextPlugin {
                log: scoped.clone(),
            },
        )
        .with_gpu_plugin_for_viewport(
            other,
            ContextPlugin {
                log: elsewhere.clone(),
            },
        )
        .with_gpu_plugin(ContextPlugin {
            log: unscoped.clone(),
        });
    let mut instance = instance.with_runtime(runtime);
    instance.step_runtime(0.25);
    Path::OwnedHdr.frame(&mut instance, &device, &queue, &target);
    Path::OwnedHdr.frame(&mut instance, &device, &queue, &target);

    let expected = format!("pre_prepare:Some({}):0.25:{SIZE}x{SIZE}", own.index());
    scoped.assert_take(&[&expected, &expected]);
    unscoped.assert_take(&[&expected, &expected]);
    elsewhere.assert_empty();
}

/// Writes the frame index into a buffer from its `pre_prepare` command
/// buffer.
struct WriterPlugin {
    source: wgpu::Buffer,
    target: wgpu::Buffer,
}

impl GpuPlugin for WriterPlugin {
    fn priority(&self) -> i32 {
        gpu_phase::PRE_PREPARE
    }

    fn pre_prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        let value = (ctx.frame_index as u32 + 1) * 7;
        queue.write_buffer(&self.source, 0, &value.to_le_bytes());
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.source, 0, &self.target, 0, 4);
        vec![encoder.finish()]
    }
}

/// A post-effect stage that copies the plugin's buffer into a readback
/// buffer while the frame's passes are encoded, so the copy sees whatever
/// had reached the buffer by then.
struct ReaderStage {
    plugin_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    inputs: std::collections::HashMap<usize, (wgpu::Texture, wgpu::TextureView)>,
}

impl PostEffectStage for ReaderStage {
    fn type_name(&self) -> &'static str {
        "reader_stage"
    }

    fn enabled(&self) -> bool {
        true
    }

    fn on_viewport_resized(&mut self, device: &wgpu::Device, ctx: &PostEffectResizeContext<'_>) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reader_stage_input"),
            size: wgpu::Extent3d {
                width: ctx.scene_size[0],
                height: ctx.scene_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ctx.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        self.inputs.insert(ctx.viewport_index, (texture, view));
    }

    fn input_view(&self, viewport_index: usize) -> &wgpu::TextureView {
        &self.inputs[&viewport_index].1
    }

    fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        _target: &wgpu::TextureView,
        _ctx: &PostEffectContext<'_>,
    ) {
        encoder.copy_buffer_to_buffer(&self.plugin_buffer, 0, &self.readback, 0, 4);
    }
}

fn read_u32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> u32 {
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .expect("readback poll");
    let value = {
        let mapped = viewport_lib::gpu::mapped_range(buffer.slice(..));
        u32::from_le_bytes(mapped[..4].try_into().unwrap())
    };
    buffer.unmap();
    value
}

/// What a plugin writes in `pre_prepare` is visible to the passes of the
/// same frame, not the next one.
#[test]
fn pre_prepare_writes_reach_the_same_frame() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    let buffer = |label, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 4,
            usage,
            mapped_at_creation: false,
        })
    };
    let source = buffer(
        "writer_source",
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let shared = buffer(
        "writer_target",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let readback = buffer(
        "reader_readback",
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let runtime = ViewportRuntime::new().with_gpu_plugin(WriterPlugin {
        source,
        target: shared.clone(),
    });
    let mut instance = instance(&device).with_runtime(runtime);
    instance
        .renderer_mut()
        .add_post_effect_stage(
            Box::new(ReaderStage {
                plugin_buffer: shared,
                readback: readback.clone(),
                inputs: Default::default(),
            }),
            viewport_lib::plugin_api::post_effect::stage_order::EXTERNAL_DEFAULT,
        )
        .expect("stage");
    for frame in 0..3u32 {
        Path::OwnedHdr.frame(&mut instance, &device, &queue, &target);
        assert_eq!(
            read_u32(&device, &readback),
            (frame + 1) * 7,
            "frame {frame}"
        );
    }
}
