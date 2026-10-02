//! GPU state for the GPU implicit surface item type: the render, pick, and
//! outline-mask pipelines and the per-frame per-item uniform bind groups.

use super::types::{GpuImplicitItem, ImplicitBlendMode, ImplicitPrimitive};
use crate::shader::{lit_shader, scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::renderer::PickId;
use viewport_lib::resources::DeviceResources;

/// Pipelines and layouts, built lazily on the first prepare with items.
pub(super) struct GpuImplicitGpu {
    bgl: gpu::BindGroupLayout,
    pub(super) pipeline: builders::DualPipeline,
    pub(super) mask_pipeline: gpu::RenderPipeline,
    pub(super) surface_mask_pipeline: gpu::RenderPipeline,
    pub(super) pick_pipeline: gpu::RenderPipeline,
    pick_id_bgl: gpu::BindGroupLayout,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct GpuImplicitFrame {
    pub(super) bind_group: gpu::BindGroup,
    pub(super) _uniform_buf: gpu::Buffer,
    /// Object-id uniform + bind group for the GPU pick pass; `None` when the
    /// item is not pickable.
    pub(super) pick: Option<(gpu::Buffer, gpu::BindGroup)>,
    pub(super) selected: bool,
    pub(super) settings: viewport_lib::ItemSettings,
}

/// Flat uniform buffer layout matching the WGSL `ImplicitUniform` struct.
///
/// Total size: 32 header bytes + 16 * 64 primitive bytes = 1056 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ImplicitUniformRaw {
    num_primitives: u32,
    blend_mode: u32,
    max_steps: u32,
    unlit: u32,
    step_scale: f32,
    hit_threshold: f32,
    max_distance: f32,
    opacity: f32,
    primitives: [ImplicitPrimitive; 16],
}

impl GpuImplicitGpu {
    pub(super) fn new(device: &gpu::Device, resources: &DeviceResources) -> Self {
        // Group 1: single uniform buffer containing ImplicitUniformRaw.
        let bgl = builders::uniform_bgl(device, "implicit_bgl", gpu::ShaderStages::FRAGMENT);

        let shader = builders::wgsl_module(
            device,
            "implicit_shader",
            lit_shader(&[], wgsl_source!("implicit")),
        );
        // Group 0 reuses the shared camera layout (CameraUniform + LightsUniform).
        let layout = builders::standard_scene_layout(
            device,
            "implicit_pipeline_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        // depth_write is on so later depth-tested passes occlude against the surface.
        let pipeline = builders::build_dual_pipeline(
            device,
            &builders::DualPipelineDesc {
                label: "implicit_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[],
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: gpu::CompareFunction::LessEqual,
                sample_count: 1,
                ldr_format: resources.target_format(),
            },
        );

        // Outline mask: the same ray-march, writing white on hit and
        // discarding on miss.
        let mask_shader = builders::wgsl_module(
            device,
            "implicit_outline_mask_shader",
            &scene_shader(&[], wgsl_source!("implicit_outline_mask")),
        );
        let mask_opts = viewport_lib::resources::PluginPipelineOpts {
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            extra_bind_group_layouts: &[&bgl],
            ..viewport_lib::resources::PluginPipelineOpts::new(
                Some("implicit_outline_mask_pipeline"),
                &mask_shader,
                "vs_main",
                "fs_main",
                &[],
            )
        };
        let mask_pipeline = resources.build_mask_pipeline(device, &mask_opts);
        // The surface mask marches again, writing the hit depth so the stamp
        // lands only where this surface is the visible one.
        let surface_mask_pipeline = resources.build_surface_mask_pipeline(
            device,
            &viewport_lib::resources::PluginPipelineOpts {
                label: Some("implicit_surface_mask_pipeline"),
                fs_entry: "fs_stamp",
                ..mask_opts
            },
        );

        // Pick: the same ray-march writing the item's object id and the hit
        // depth. Group 1 reuses the render bind group, group 2 is the object
        // id. Laid out against the shared group-0 camera, which the pick pass
        // binds before plugin dispatch; the fragment reads `inv_view_proj` to
        // reconstruct the ray and `view_proj` to project the hit depth.
        let pick_id_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("implicit_pick_id_bgl"),
            entries: &[gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: gpu::ShaderStages::FRAGMENT,
                ty: gpu::BindingType::Buffer {
                    ty: gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pick_shader = builders::wgsl_module(
            device,
            "implicit_pick_shader",
            &scene_shader(&[], wgsl_source!("implicit_pick")),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&bgl, &pick_id_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("implicit_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_pick",
                    &[],
                )
            },
        );

        Self {
            bgl,
            pipeline,
            mask_pipeline,
            surface_mask_pipeline,
            pick_pipeline,
            pick_id_bgl,
        }
    }

    /// Build the per-item uniform + bind group.
    pub(super) fn upload_item(
        &self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &GpuImplicitItem,
    ) -> GpuImplicitFrame {
        use gpu::util::DeviceExt as _;

        let blend_mode_u32 = match item.blend_mode {
            ImplicitBlendMode::Union => 0u32,
            ImplicitBlendMode::SmoothUnion => 1,
            ImplicitBlendMode::Intersection => 2,
        };
        let mut raw = ImplicitUniformRaw {
            num_primitives: item.primitives.len().min(16) as u32,
            blend_mode: blend_mode_u32,
            max_steps: item.march_options.max_steps,
            unlit: item.settings.unlit as u32,
            step_scale: item.march_options.step_scale,
            hit_threshold: item.march_options.hit_threshold,
            max_distance: item.march_options.max_distance,
            opacity: item.settings.opacity,
            primitives: [ImplicitPrimitive::zeroed(); 16],
        };
        for (i, prim) in item.primitives.iter().take(16).enumerate() {
            raw.primitives[i] = *prim;
        }

        let uniform_buf = device.create_buffer_init(&gpu::util::BufferInitDescriptor {
            label: Some("implicit_uniform_buf"),
            contents: bytemuck::bytes_of(&raw),
            usage: gpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("implicit_bind_group"),
            layout: &self.bgl,
            entries: &[gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pick = (item.settings.pick_id != PickId::NONE).then(|| {
            let id_data: [u32; 4] = [item.settings.pick_id.0 as u32, 0, 0, 0];
            let pick_buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("implicit_pick_id_buf"),
                size: std::mem::size_of_val(&id_data) as u64,
                usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pick_buf, 0, bytemuck::cast_slice(&id_data));
            let pick_bg = device.create_bind_group(&gpu::BindGroupDescriptor {
                label: Some("implicit_pick_id_bg"),
                layout: &self.pick_id_bgl,
                entries: &[gpu::BindGroupEntry {
                    binding: 0,
                    resource: pick_buf.as_entire_binding(),
                }],
            });
            (pick_buf, pick_bg)
        });

        GpuImplicitFrame {
            bind_group,
            _uniform_buf: uniform_buf,
            pick,
            selected: item.settings.selected,
            settings: item.settings,
        }
    }
}
