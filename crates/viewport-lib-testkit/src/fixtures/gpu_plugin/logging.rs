//! A GPU plugin that records its lifecycle and encodes one trivial pass.

use crate::fixtures::CallLog;
use viewport_lib::wgpu;
use viewport_lib::{GpuFrameContext, GpuPlugin, PostPaintTargets};

/// Records `init_gpu`, `on_device_recreated`, `pre_prepare` (with the frame
/// index) and `post_paint` (with the colour format it was handed), returns one
/// command buffer from `pre_prepare` that clears a buffer it owns, and reads
/// the post-paint targets without drawing into them.
///
/// The buffer clear is there so the returned command buffer is real work the
/// host's `queue.submit` has to accept: an empty encoder would submit fine
/// even if the plugin's output were being dropped.
pub struct LoggingGpuPlugin {
    log: CallLog,
    priority: i32,
    /// Allocated in `init_gpu`, so a test can assert deferred init ran by
    /// checking the clear submits at all.
    scratch: Option<wgpu::Buffer>,
}

impl LoggingGpuPlugin {
    /// A plugin logging into `log` at `priority` (use the
    /// [`gpu_phase`](viewport_lib::runtime::gpu_plugin::gpu_phase) constants).
    pub fn new(log: CallLog, priority: i32) -> Self {
        Self {
            log,
            priority,
            scratch: None,
        }
    }
}

impl GpuPlugin for LoggingGpuPlugin {
    fn priority(&self) -> i32 {
        self.priority
    }

    fn type_name(&self) -> &'static str {
        "logging_gpu_plugin"
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        self.log.record("init_gpu");
        self.scratch = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logging_gpu_plugin_scratch"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }

    fn on_device_recreated(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        self.log.record("on_device_recreated");
        // Drop the old device's resources and let the following `init_gpu`
        // rebuild them, which is the contract's suggested shape.
        self.scratch = None;
    }

    fn pre_prepare(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        self.log.record(format!("pre_prepare:{}", ctx.frame_index));
        let Some(scratch) = self.scratch.as_ref() else {
            return Vec::new();
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logging_gpu_plugin_pre_prepare"),
        });
        encoder.clear_buffer(scratch, 0, None);
        vec![encoder.finish()]
    }

    fn post_paint(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        targets: &PostPaintTargets<'_>,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        self.log.record(format!(
            "post_paint:{}:{:?}:pick={}",
            ctx.frame_index,
            targets.color_format,
            targets.pick_id_view.is_some()
        ));
        Vec::new()
    }
}
