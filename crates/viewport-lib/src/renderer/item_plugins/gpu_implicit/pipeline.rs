//! GPU state for the GPU implicit surface item type: the render, pick, and
//! outline-mask pipelines and the per-frame per-item uniform bind groups.

use crate::renderer::{GpuImplicitItem, ImplicitBlendMode, ImplicitPrimitive, PickId};
use crate::resources::DeviceResources;

/// Pipelines and layouts, built lazily on the first prepare with items.
pub(super) struct GpuImplicitGpu {
    bgl: crate::gpu::BindGroupLayout,
    pub(super) pipeline: crate::resources::DualPipeline,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pick_id_bgl: crate::gpu::BindGroupLayout,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct GpuImplicitFrame {
    pub(super) bind_group: crate::gpu::BindGroup,
    pub(super) _uniform_buf: crate::gpu::Buffer,
    /// Object-id uniform + bind group for the GPU pick pass; `None` when the
    /// item is not pickable.
    pub(super) pick: Option<(crate::gpu::Buffer, crate::gpu::BindGroup)>,
    pub(super) selected: bool,
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
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        // Group 1: single uniform buffer containing ImplicitUniformRaw.
        let bgl = crate::resources::builders::uniform_bgl(
            device,
            "implicit_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "implicit_shader",
            crate::resources::builders::wgsl_source!("implicit"),
        );
        // Group 0 reuses the shared camera layout (CameraUniform + LightsUniform).
        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "implicit_pipeline_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        // depth_write is on so later depth-tested passes occlude against the surface.
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "implicit_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[],
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::LessEqual,
                sample_count: 1,
                ldr_format: resources.target_format,
            },
        );

        // Outline mask: the same ray-march, writing white on hit and
        // discarding on miss.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "implicit_outline_mask_shader",
            crate::resources::builders::wgsl_source!("implicit_outline_mask"),
        );
        let mask_pipeline = resources.build_mask_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&bgl],
                ..crate::resources::PluginPipelineOpts::new(
                    Some("implicit_outline_mask_pipeline"),
                    &mask_shader,
                    "vs_main",
                    "fs_main",
                    &[],
                )
            },
        );

        // Pick: the same ray-march writing the item's object id and the hit
        // depth. Group 1 reuses the render bind group, group 2 is the object
        // id. Laid out against the shared group-0 camera, which the pick pass
        // binds before plugin dispatch; the fragment reads `inv_view_proj` to
        // reconstruct the ray and `view_proj` to project the hit depth.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("implicit_pick_id_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pick_shader = crate::resources::builders::wgsl_module(
            device,
            "implicit_pick_shader",
            crate::resources::builders::wgsl_source!("implicit_pick"),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&bgl, &pick_id_bgl],
                ..crate::resources::PluginPipelineOpts::new(
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
            pick_pipeline,
            pick_id_bgl,
        }
    }

    /// Build the per-item uniform + bind group.
    pub(super) fn upload_item(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &GpuImplicitItem,
    ) -> GpuImplicitFrame {
        use crate::gpu::util::DeviceExt as _;

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

        let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("implicit_uniform_buf"),
            contents: bytemuck::bytes_of(&raw),
            usage: crate::gpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("implicit_bind_group"),
            layout: &self.bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pick = (item.settings.pick_id != PickId::NONE).then(|| {
            let id_data: [u32; 4] = [item.settings.pick_id.0 as u32, 0, 0, 0];
            let pick_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("implicit_pick_id_buf"),
                size: std::mem::size_of_val(&id_data) as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pick_buf, 0, bytemuck::cast_slice(&id_data));
            let pick_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("implicit_pick_id_bg"),
                layout: &self.pick_id_bgl,
                entries: &[crate::gpu::BindGroupEntry {
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
        }
    }
}
