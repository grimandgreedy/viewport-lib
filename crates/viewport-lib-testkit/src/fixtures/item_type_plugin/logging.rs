//! An item-type plugin that records the dispatch it receives.

use crate::fixtures::CallLog;
use viewport_lib::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickPassContext, PickRay,
    PluginItemCollection, SharedBindings,
};
use viewport_lib::renderer::PickHit;
use viewport_lib::wgpu;

/// Records `init_gpu`, `prepare` (with the submitted item count),
/// `paint`, `outline_mask` and `render_pick`, and draws nothing.
///
/// Registered under the `type_name` given at construction, so two of these
/// can coexist and a test can tell which one the renderer dispatched to. The
/// CPU `pick` reports a hit on the first item at a fixed distance when
/// `pickable` is set, which is enough to prove the ray reaches the plugin.
pub struct LoggingItemTypePlugin {
    log: CallLog,
    type_name: &'static str,
    pickable: bool,
}

impl LoggingItemTypePlugin {
    /// A plugin registered under `type_name`, logging into `log`.
    pub fn new(log: CallLog, type_name: &'static str) -> Self {
        Self {
            log,
            type_name,
            pickable: false,
        }
    }

    /// Report a CPU pick hit on the first item. Off by default.
    pub fn pickable(mut self) -> Self {
        self.pickable = true;
        self
    }
}

impl ItemTypePlugin for LoggingItemTypePlugin {
    fn type_name(&self) -> &'static str {
        self.type_name
    }

    fn init_gpu(&mut self, _device: &wgpu::Device, shared: &SharedBindings<'_>) {
        self.log
            .record(format!("init_gpu:samples={}", shared.sample_count));
    }

    fn on_device_recreated(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        self.log.record("on_device_recreated");
    }

    fn prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        self.log.record(format!(
            "prepare:{}:items={}:vp={}",
            self.type_name,
            items.len(),
            ctx.viewport_index
        ));
        Vec::new()
    }

    fn paint(
        &self,
        _pass: &mut wgpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.log.record(format!(
            "paint:{}:items={}:vp={}",
            self.type_name,
            items.len(),
            ctx.viewport_index
        ));
    }

    fn outline_mask(
        &self,
        _pass: &mut wgpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        self.log.record(format!("outline_mask:{}", self.type_name));
    }

    fn render_pick(
        &self,
        _pass: &mut wgpu::RenderPass<'_>,
        _ctx: &PickPassContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        self.log.record(format!("render_pick:{}", self.type_name));
    }

    fn pick(
        &self,
        ray: &PickRay,
        _ctx: &viewport_lib::plugin_api::PickContext,
    ) -> Option<(f32, PickHit)> {
        self.log.record(format!(
            "pick:{}:dir_z={:.1}",
            self.type_name, ray.direction.z
        ));
        if !self.pickable {
            return None;
        }
        Some((1.0, PickHit::object_hit(1, glam::Vec3::ZERO, glam::Vec3::Z)))
    }
}
