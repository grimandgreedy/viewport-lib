//! GPU state for the volume surface slice item type: the render, pick, and
//! mask pipelines, each built the first time a draw needs it, and the
//! per-frame per-item bind groups. The geometry is a consumer-uploaded mesh,
//! so nothing here owns vertex or index buffers.

use super::types::VolumeSurfaceSliceItem;
use crate::item_types::shader::{scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::renderer::PickId;
use viewport_lib::resources::DeviceResources;

/// Members of [`SlicePipelines`].
pub(super) const COLOUR_LDR: usize = 0;
pub(super) const COLOUR_HDR: usize = 1;
pub(super) const MASK: usize = 2;
pub(super) const SURFACE_MASK: usize = 3;
pub(super) const PICK: usize = 4;

/// What a slice pipeline build reads.
pub(super) struct SliceRecipe {
    device: gpu::Device,
    builder: viewport_lib::plugin_api::PipelineBuilder,
    layout: gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    mask_shader: viewport_lib::plugin_api::LazyModule,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    bgl: gpu::BindGroupLayout,
    pick_id_bgl: gpu::BindGroupLayout,
    sample_count: u32,
    ldr_format: gpu::TextureFormat,
}

/// The slice in both formats, the outline and surface masks and the pick
/// pipeline.
pub(super) type SlicePipelines = viewport_lib::plugin_api::LazyPipelines<SliceRecipe, 5>;

fn build(r: &SliceRecipe, i: usize) -> gpu::RenderPipeline {
    let vertex_buffers = [builders::mesh_vertex_layout()];
    let primitive = gpu::PrimitiveState {
        topology: gpu::PrimitiveTopology::TriangleList,
        cull_mode: None,
        ..Default::default()
    };
    match i {
        COLOUR_LDR | COLOUR_HDR => builders::build_dual_pipeline_variant(
            &r.device,
            &builders::DualPipelineDesc {
                label: "volume_surface_slice_pipeline",
                layout: &r.layout,
                shader: r.shader.get(),
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &vertex_buffers,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: gpu::CompareFunction::LessEqual,
                sample_count: r.sample_count,
                ldr_format: r.ldr_format,
            },
            i == COLOUR_HDR,
        ),
        // Outline mask: the slice mesh transformed by its model matrix. The
        // shared mesh mask pipeline also carries position-override and deform
        // support, neither of which a slice ever uses, so this is the same
        // vertex transform with those branches removed. The surface mask
        // stamps the same geometry into the scene stencil.
        MASK | SURFACE_MASK => {
            let label = if i == MASK {
                "volume_surface_slice_mask_pipeline"
            } else {
                "volume_surface_slice_surface_mask_pipeline"
            };
            let opts = viewport_lib::resources::PluginPipelineOpts {
                primitive,
                extra_bind_group_layouts: &[&r.bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some(label),
                    r.mask_shader.get(),
                    "vs_main",
                    "fs_main",
                    &vertex_buffers,
                )
            };
            if i == MASK {
                r.builder.build_mask_pipeline(&r.device, &opts)
            } else {
                r.builder.build_surface_mask_pipeline(&r.device, &opts)
            }
        }
        // Pick: the same mesh, writing the item's object id. Group 1 reuses the
        // render bind group, group 2 is the object id.
        _ => r.builder.build_pick_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive,
                extra_bind_group_layouts: &[&r.bgl, &r.pick_id_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("volume_surface_slice_pick_pipeline"),
                    r.pick_shader.get(),
                    "vs_main",
                    "fs_main",
                    &vertex_buffers,
                )
            },
        ),
    }
}

/// Pipelines and layouts, made on the first prepare with items.
pub(super) struct SliceGpu {
    bgl: gpu::BindGroupLayout,
    pub(super) pipelines: SlicePipelines,
    pick_id_bgl: gpu::BindGroupLayout,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct SliceFrame {
    pub(super) bind_group: gpu::BindGroup,
    pub(super) _uniform_buf: gpu::Buffer,
    /// The mesh to draw, resolved through the draw hook's mesh handle.
    pub(super) mesh_id: viewport_lib::MeshId,
    /// Object-id uniform + bind group for the GPU pick pass; `None` when the
    /// item is not pickable.
    pub(super) pick: Option<(gpu::Buffer, gpu::BindGroup)>,
    pub(super) selected: bool,
    pub(super) settings: viewport_lib::ItemSettings,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SliceUniform {
    model: [[f32; 4]; 4],
    bbox_min: [f32; 3],
    scalar_min: f32,
    bbox_max: [f32; 3],
    scalar_max: f32,
    opacity: f32,
    _pad: [f32; 3],
}

impl SliceGpu {
    pub(super) fn new(device: &gpu::Device, resources: &DeviceResources) -> Self {
        let filterable = |view_dimension| gpu::BindingType::Texture {
            multisampled: false,
            view_dimension,
            sample_type: gpu::TextureSampleType::Float { filterable: true },
        };
        let sampler = || gpu::BindingType::Sampler(gpu::SamplerBindingType::Filtering);
        let bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("volume_surface_slice_bgl"),
            entries: &[
                // binding 0: SliceUniform
                gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: gpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: the scalar field texture_3d<f32> (filterable:
                // R16Float, or R32Float with FLOAT32_FILTERABLE), so the slice
                // samples it trilinearly instead of nearest-neighbor.
                gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: filterable(gpu::TextureViewDimension::D3),
                    count: None,
                },
                // binding 2: vol_sampler (linear)
                gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: sampler(),
                    count: None,
                },
                // binding 3: lut_tex (colourmap texture_2d)
                gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: filterable(gpu::TextureViewDimension::D2),
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: sampler(),
                    count: None,
                },
            ],
        });

        let shader = resources.lazy_module(
            device,
            "volume_surface_slice_shader",
            &scene_shader(&[], wgsl_source!("volume_surface_slice")),
        );
        let layout = builders::standard_scene_layout(
            device,
            "volume_surface_slice_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        let mask_shader = resources.lazy_module(
            device,
            "volume_surface_slice_mask_shader",
            &scene_shader(&[], wgsl_source!("volume_surface_slice_mask")),
        );
        // Group 2 of the pick pass: the object id.
        let pick_id_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("volume_surface_slice_pick_id_bgl"),
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
        let pick_shader = resources.lazy_module(
            device,
            "volume_surface_slice_pick_shader",
            &scene_shader(&[], wgsl_source!("volume_surface_slice_pick")),
        );
        let pipelines = resources.lazy_pipelines(
            SliceRecipe {
                device: device.clone(),
                builder: resources.pipeline_builder(),
                layout,
                shader,
                mask_shader,
                pick_shader,
                bgl: bgl.clone(),
                pick_id_bgl: pick_id_bgl.clone(),
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
            build,
        );

        Self {
            bgl,
            pipelines,
            pick_id_bgl,
        }
    }

    /// Whether the slice can draw this frame in either format. The mask and
    /// pick passes wait for it, so a slice is never outlined, stamped or
    /// picked before it is drawn.
    pub(super) fn drawn(&self) -> bool {
        self.pipelines.available(COLOUR_LDR) || self.pipelines.available(COLOUR_HDR)
    }

    /// Build the per-item uniform and bind group, or `None` when the item's
    /// volume or mesh is not resident.
    pub(super) fn upload_item(
        &self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        resources: &DeviceResources,
        item: &VolumeSurfaceSliceItem,
    ) -> Option<SliceFrame> {
        let vol_view = resources.volume_view(item.volume_id)?;
        // A stale mesh id draws nothing, so drop the item rather than build
        // state for a mesh that cannot be bound.
        resources.mesh_index_count(item.mesh_id)?;

        let uniform_data = SliceUniform {
            model: item.model,
            bbox_min: item.bbox_min,
            scalar_min: item.scalar_range.0,
            bbox_max: item.bbox_max,
            scalar_max: item.scalar_range.1,
            // ItemSettings.opacity multiplies into the type's own opacity field
            // so consumers can drive transparency through the standard per-item
            // settings without abandoning the existing field.
            opacity: item.opacity * item.settings.opacity,
            _pad: [0.0; 3],
        };
        let uniform_buf = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("volume_surface_slice_uniform"),
            size: std::mem::size_of::<SliceUniform>() as u64,
            usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let vol_sampler =
            builders::clamp_linear_sampler(device, "volume_surface_slice_vol_sampler");

        let lut_view = item
            .colour_lut
            .and_then(|id| resources.colourmap_view(id))
            .unwrap_or_else(|| {
                resources.builtin_colourmap_view(viewport_lib::resources::BuiltinColourmap::Viridis)
            });

        let bind_group = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("volume_surface_slice_bg"),
            layout: &self.bgl,
            entries: &[
                gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu::BindingResource::TextureView(vol_view),
                },
                gpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu::BindingResource::Sampler(&vol_sampler),
                },
                gpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu::BindingResource::TextureView(lut_view),
                },
                gpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu::BindingResource::Sampler(resources.material_sampler()),
                },
            ],
        });

        let pick = (item.settings.pick_id != PickId::NONE).then(|| {
            let id_data: [u32; 4] = [item.settings.pick_id.0 as u32, 0, 0, 0];
            let pick_buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("volume_surface_slice_pick_id_buf"),
                size: std::mem::size_of_val(&id_data) as u64,
                usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pick_buf, 0, bytemuck::cast_slice(&id_data));
            let pick_bg = device.create_bind_group(&gpu::BindGroupDescriptor {
                label: Some("volume_surface_slice_pick_id_bg"),
                layout: &self.pick_id_bgl,
                entries: &[gpu::BindGroupEntry {
                    binding: 0,
                    resource: pick_buf.as_entire_binding(),
                }],
            });
            (pick_buf, pick_bg)
        });

        Some(SliceFrame {
            bind_group,
            _uniform_buf: uniform_buf,
            mesh_id: item.mesh_id,
            pick,
            selected: item.settings.selected,
            settings: item.settings,
        })
    }
}
