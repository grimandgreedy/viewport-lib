//! An attached runtime costs a `ViewportInstance` render no heap allocations:
//! a frame with no runtime, one with an empty runtime, and one with a GPU
//! plugin that encodes nothing allocate the same.
//!
//! Its own test binary, because it installs a counting global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use viewport_lib::wgpu;
use viewport_lib::{
    GpuFrameContext, GpuPlugin, InputOwnership, Material, OffscreenViewportTarget, PipelineMode,
    RendererConfig, ViewportContext, ViewportInstance, ViewportRuntime, primitives,
};
use viewport_lib_testkit::{headless_device, orbit_camera};

struct Counting;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // `try_with`: the thread-locals are gone while a thread shuts down.
        let _ = COUNTING.try_with(|on| {
            if on.get() {
                let _ = COUNT.try_with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Allocations made on this thread while `f` runs.
fn count_allocations(f: impl FnOnce()) -> u64 {
    COUNT.with(|c| c.set(0));
    COUNTING.with(|on| on.set(true));
    f();
    COUNTING.with(|on| on.set(false));
    COUNT.with(|c| c.get())
}

const SIZE: u32 = 64;
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// A GPU plugin with nothing to encode.
struct IdlePlugin;

impl GpuPlugin for IdlePlugin {
    fn type_name(&self) -> &'static str {
        "idle_gpu_plugin"
    }

    fn pre_prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        Vec::new()
    }
}

/// The fewest allocations one `render` call made over a run of settled
/// frames on `mode`. The minimum, because the renderer's own per-frame
/// allocations vary a little between frames.
fn settled_render_allocations(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    runtime: Option<ViewportRuntime>,
    mode: PipelineMode,
) -> u64 {
    let config = RendererConfig::new(OffscreenViewportTarget::render_format(SURFACE_FORMAT))
        .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    let mut instance = ViewportInstance::with_config(device, &config);
    if let Some(runtime) = runtime {
        instance = instance.with_runtime(runtime);
    }
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
    instance.effects_mut().display.mode = mode;
    let target = OffscreenViewportTarget::new(device, SURFACE_FORMAT, [SIZE, SIZE]);

    let mut fewest = u64::MAX;
    for frame in 0..24 {
        instance.frame(ViewportContext::new(
            InputOwnership::default(),
            [SIZE as f32, SIZE as f32],
        ));
        let mut cmd = None;
        let count = count_allocations(|| {
            cmd = Some(instance.render(device, queue, target.render_view()));
        });
        queue.submit(cmd);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        if frame >= 8 {
            fewest = fewest.min(count);
        }
    }
    fewest
}

#[test]
fn an_attached_runtime_adds_no_allocations_to_a_render() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for mode in [PipelineMode::Hdr, PipelineMode::Direct] {
        let none = settled_render_allocations(&device, &queue, None, mode);
        let empty = settled_render_allocations(&device, &queue, Some(ViewportRuntime::new()), mode);
        let idle = settled_render_allocations(
            &device,
            &queue,
            Some(ViewportRuntime::new().with_gpu_plugin(IdlePlugin)),
            mode,
        );
        eprintln!("{mode:?}: no runtime {none}, empty runtime {empty}, idle GPU plugin {idle}");
        assert_eq!(empty, none, "{mode:?}: an empty runtime allocated");
        assert_eq!(idle, none, "{mode:?}: an idle GPU plugin allocated");
    }
}
