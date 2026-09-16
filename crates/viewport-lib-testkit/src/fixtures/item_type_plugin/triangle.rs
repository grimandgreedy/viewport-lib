//! An item-type plugin that actually draws, through the library's plugin
//! pipeline builders.

use crate::fixtures::CallLog;
use viewport_lib::plugin_api::shared_wgsl::{
    SHARED_BINDINGS_WGSL, SHARED_PICK_WGSL, SHARED_SHADOW_BINDINGS_WGSL,
};
use viewport_lib::plugin_api::{
    EncoderScope, EncoderScopeContext, ItemTypePlugin, PaintContext, PickPassContext,
    PluginItemCollection, ShadowCastContext, SharedBindings,
};
use viewport_lib::resources::{DeviceResources, PluginPipelineOpts};
use viewport_lib::wgpu;

/// A world-space triangle, drawn in the opaque scene pass and the pick pass,
/// with pipelines built by
/// [`build_opaque_pipeline`](DeviceResources::build_opaque_pipeline) and
/// [`build_pick_pipeline`](DeviceResources::build_pick_pipeline).
///
/// The `Logging*` fixtures prove the renderer still calls a plugin. This one
/// proves a plugin can still *render*: it reads the camera from group 0, so
/// the shared bind layout has to be usable from outside the crate, and its
/// three pipelines are built from the published target descriptors, so a
/// change to a format, blend state, or the MSAA sample count breaks it here
/// rather than in a consumer's repo.
///
/// The geometry is generated from `@builtin(vertex_index)` (no vertex
/// buffers) and its world position comes from the constructor, so a test can
/// place it where the camera sees it.
pub struct TriangleItemTypePlugin {
    log: CallLog,
    type_name: &'static str,
    opaque: wgpu::RenderPipeline,
    pick: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    /// Drawn in `encode`, which opens its own pass over the scene colour
    /// rather than drawing into one the lib began.
    encode: wgpu::RenderPipeline,
    pick_id_layout: wgpu::BindGroupLayout,
    pick_id_group: Option<wgpu::BindGroup>,
}

impl TriangleItemTypePlugin {
    /// Build the plugin's pipelines against `resources` and register it under
    /// `type_name`.
    ///
    /// The pipeline builders live on [`DeviceResources`], which
    /// [`ItemTypePlugin::init_gpu`] is not handed (it receives the device and
    /// the shared bind layout only), so a plugin that uses them is
    /// constructed by the host, which has the renderer. This is the same
    /// shape the shipped renderer-side handles use.
    pub fn new(
        resources: &DeviceResources,
        device: &wgpu::Device,
        log: CallLog,
        type_name: &'static str,
        centre: glam::Vec3,
        colour: [f32; 3],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle_item_shader"),
            source: wgpu::ShaderSource::Wgsl(triangle_wgsl(centre, colour).into()),
        });

        // The pick pass needs the item's id; the other two passes take group 0
        // alone. One layout, declared once, listed only for the pick pipeline.
        let pick_id_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("triangle_item_pick_id_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let opaque = resources.build_opaque_pipeline(
            device,
            &PluginPipelineOpts::new(
                Some("triangle_item_opaque"),
                &shader,
                "vs_scene",
                "fs_scene",
                &[],
            ),
        );
        let mut pick_opts = PluginPipelineOpts::new(
            Some("triangle_item_pick"),
            &shader,
            "vs_pick",
            "viewport_pick_fs",
            &[],
        );
        let pick_layouts = [&pick_id_layout];
        pick_opts.extra_bind_group_layouts = &pick_layouts;
        let pick = resources.build_pick_pipeline(device, &pick_opts);

        // The shadow pass binds a different group 0, so its stage lives in its
        // own module with `SHARED_SHADOW_BINDINGS_WGSL` at the top.
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle_item_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(triangle_shadow_wgsl(centre).into()),
        });
        let shadow = resources.build_shadow_pipeline(
            device,
            &PluginPipelineOpts::new(
                Some("triangle_item_shadow"),
                &shadow_shader,
                "vs_shadow",
                "",
                &[],
            ),
        );

        // The encode pass targets the same HDR scene attachments the opaque
        // pass does, so it is built from the same descriptor.
        let encode = resources.build_opaque_pipeline(
            device,
            &PluginPipelineOpts::new(
                Some("triangle_item_encode"),
                &shader,
                "vs_encode",
                "fs_encode",
                &[],
            ),
        );

        Self {
            log,
            type_name,
            opaque,
            pick,
            shadow,
            encode,
            pick_id_layout,
            pick_id_group: None,
        }
    }
}

impl ItemTypePlugin for TriangleItemTypePlugin {
    fn type_name(&self) -> &'static str {
        self.type_name
    }

    fn init_gpu(&mut self, _device: &wgpu::Device, shared: &SharedBindings<'_>) {
        // The pipelines were built in `new`; record the sample count the
        // renderer reports so a test can check it matches what they were
        // built against.
        self.log
            .record(format!("init_gpu:samples={}", shared.sample_count));
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _ctx: &viewport_lib::plugin_api::ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        self.log.record(format!("prepare:items={}", items.len()));
        if items.is_empty() {
            return Vec::new();
        }
        // A uniform buffer's minimum binding size is 16 bytes, so the id is
        // padded out to a vec4<u32>.
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&(items.pick_id(0).0 as u32).to_ne_bytes());
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("triangle_item_pick_id"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        self.pick_id_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("triangle_item_pick_id_bg"),
            layout: &self.pick_id_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        }));
        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.log.record("paint");
        if items.is_empty() || items.item_settings(0).hidden {
            return;
        }
        // Group 0 is bound by the renderer on pass entry.
        pass.set_pipeline(&self.opaque);
        pass.draw(0..3, 0..1);
    }

    fn cast_shadow_pass(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        ctx: &ShadowCastContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.log
            .record(format!("cast_shadow_pass:cascade={}", ctx.cascade_idx));
        if items.is_empty() || items.item_settings(0).hidden {
            return;
        }
        // Group 0 (the cascade's light view-projection, at its dynamic offset)
        // is bound by the renderer on pass entry.
        pass.set_pipeline(&self.shadow);
        pass.draw(0..3, 0..1);
    }

    fn encoder_scopes(&self) -> &[EncoderScope] {
        &[EncoderScope::AfterTransparent]
    }

    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.log.record(format!("encode:scope={:?}", ctx.scope));
        if items.is_empty() || items.item_settings(0).hidden {
            return;
        }
        // Open a pass of the plugin's own over the scene attachments. Both
        // load: everything already drawn has to survive.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("triangle_item_encode_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.scene_colour,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.scene_depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.encode);
        pass.set_bind_group(0, ctx.camera_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn render_pick(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        _ctx: &PickPassContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.log.record("render_pick");
        let Some(group) = self.pick_id_group.as_ref() else {
            return;
        };
        if items.is_empty() || items.item_settings(0).hidden {
            return;
        }
        pass.set_pipeline(&self.pick);
        pass.set_bind_group(1, group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The plugin's WGSL: one vertex stage per pass over the same generated
/// triangle, plus the shared pick fragment helper.
///
/// `SHARED_BINDINGS_WGSL` supplies the group-0 declarations (camera included),
/// so nothing here re-declares them.
fn triangle_wgsl(centre: glam::Vec3, colour: [f32; 3]) -> String {
    let [cx, cy, cz] = centre.to_array();
    let [r, g, b] = colour;
    format!(
        "{SHARED_BINDINGS_WGSL}
{SHARED_PICK_WGSL}

// A triangle in the plane facing +Z, generated from the vertex index so the
// plugin needs no vertex buffer.
fn triangle_world(vertex_index: u32) -> vec3<f32> {{
    let centre = vec3<f32>({cx:?}, {cy:?}, {cz:?});
    var offsets = array<vec2<f32>, 3>(
        vec2<f32>(-0.6, -0.5),
        vec2<f32>(0.6, -0.5),
        vec2<f32>(0.0, 0.7),
    );
    let o = offsets[vertex_index];
    return centre + vec3<f32>(o.x, o.y, 0.0);
}}

@vertex
fn vs_scene(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    return camera.view_proj * vec4<f32>(triangle_world(vi), 1.0);
}}

@fragment
fn fs_scene() -> @location(0) vec4<f32> {{
    return vec4<f32>({r:?}, {g:?}, {b:?}, 1.0);
}}

struct PickId {{
    id: vec4<u32>,
}};
@group(1) @binding(0) var<uniform> pick: PickId;

struct PickVsOut {{
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) pick_id: u32,
}};

@vertex
fn vs_pick(@builtin(vertex_index) vi: u32) -> PickVsOut {{
    var out: PickVsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(triangle_world(vi), 1.0);
    out.pick_id = pick.id.x;
    return out;
}}

// The encode stage draws the same triangle offset along +X, so a test can
// tell the encode pass's pixels from the opaque pass's by position.
@vertex
fn vs_encode(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    let world = triangle_world(vi) + vec3<f32>(1.6, 0.0, 0.0);
    return camera.view_proj * vec4<f32>(world, 1.0);
}}

@fragment
fn fs_encode() -> @location(0) vec4<f32> {{
    return vec4<f32>({b:?}, {r:?}, {g:?}, 1.0);
}}
"
    )
}

/// The shadow-cast stage, in its own module: the shadow pass binds a different
/// group 0 from every other pass, so its declarations come from
/// `SHARED_SHADOW_BINDINGS_WGSL` and this cannot share a module with the rest.
///
/// Depth-only, so there is no fragment stage.
fn triangle_shadow_wgsl(centre: glam::Vec3) -> String {
    let [cx, cy, cz] = centre.to_array();
    format!(
        "{SHARED_SHADOW_BINDINGS_WGSL}

fn triangle_world(vertex_index: u32) -> vec3<f32> {{
    let centre = vec3<f32>({cx:?}, {cy:?}, {cz:?});
    var offsets = array<vec2<f32>, 3>(
        vec2<f32>(-0.6, -0.5),
        vec2<f32>(0.6, -0.5),
        vec2<f32>(0.0, 0.7),
    );
    let o = offsets[vertex_index];
    return centre + vec3<f32>(o.x, o.y, 0.0);
}}

@vertex
fn vs_shadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    return shadow_camera.light_view_proj * vec4<f32>(triangle_world(vi), 1.0);
}}
"
    )
}
