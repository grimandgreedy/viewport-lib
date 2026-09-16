//! GPU state for the image slice item type: the render, pick, and
//! outline-mask pipelines and the per-frame per-item bind groups.

use crate::renderer::{ImageSliceItem, PickId};
use crate::resources::DeviceResources;

/// Pipelines and layouts, built lazily on the first prepare with items.
pub(super) struct ImageSliceGpu {
    pub(super) bgl: crate::gpu::BindGroupLayout,
    pub(super) pipeline: crate::resources::DualPipeline,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_id_bgl: crate::gpu::BindGroupLayout,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
    /// Linear clamp sampler for the 3D field; shared by every slice.
    pub(super) vol_sampler: crate::gpu::Sampler,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct ImageSliceFrame {
    pub(super) bind_group: crate::gpu::BindGroup,
    pub(super) _uniform_buf: crate::gpu::Buffer,
    /// Object-id uniform + bind group for the GPU pick pass; `None` when the
    /// item is not pickable.
    pub(super) pick: Option<(crate::gpu::Buffer, crate::gpu::BindGroup)>,
    pub(super) selected: bool,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageSliceUniform {
    bbox_min: [f32; 3],
    axis: u32,
    bbox_max: [f32; 3],
    offset: f32,
    scalar_min: f32,
    scalar_max: f32,
    opacity: f32,
    _pad: f32,
}

impl ImageSliceGpu {
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("image_slice_bgl"),
            entries: &[
                // binding 0: ImageSliceUniform
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: the scalar field texture_3d<f32> (filterable:
                // R16Float, or R32Float with FLOAT32_FILTERABLE), so the slice
                // samples it trilinearly instead of nearest-neighbor.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: crate::gpu::TextureViewDimension::D3,
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 2: vol_sampler (linear)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: lut_tex (colourmap texture_2d)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "image_slice_shader",
            crate::resources::builders::wgsl_source!("image_slice"),
        );
        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "image_slice_pipeline_layout",
            &resources.binds.camera_bgl,
            &bgl,
        );
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "image_slice_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[], // no vertex buffer: generates quad from vertex_index
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: crate::gpu::CompareFunction::LessEqual,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );

        // Pick: same quad expansion (group 1 reuses the render bind group
        // unchanged), object id at group 2, constant-0 primitive channel.
        // Laid out against the shared group-0 camera, which the pick pass
        // binds before plugin dispatch.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("image_slice_pick_id_bgl"),
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
            "image_slice_pick_shader",
            crate::resources::builders::wgsl_source!("image_slice_pick"),
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
                    Some("image_slice_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[],
                )
            },
        );

        // Outline mask: the same quad rasterised into the R8 selection mask.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "image_slice_mask_shader",
            crate::resources::builders::wgsl_source!("image_slice_mask"),
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
                    Some("image_slice_mask_pipeline"),
                    &mask_shader,
                    "vs_main",
                    "fs_main",
                    &[],
                )
            },
        );

        let vol_sampler =
            crate::resources::builders::clamp_linear_sampler(device, "image_slice_vol_sampler");

        Self {
            bgl,
            pipeline,
            pick_pipeline,
            pick_id_bgl,
            mask_pipeline,
            vol_sampler,
        }
    }

    /// Build the per-item uniform + bind group, or `None` when the item's
    /// volume is not resident.
    pub(super) fn upload_item(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &DeviceResources,
        item: &ImageSliceItem,
    ) -> Option<ImageSliceFrame> {
        // Check volume exists before allocating anything.
        let vol_view = &resources.content.volume_textures.get(item.volume_id)?.1;

        let axis_u32 = match item.axis {
            crate::renderer::SliceAxis::X => 0u32,
            crate::renderer::SliceAxis::Y => 1u32,
            crate::renderer::SliceAxis::Z => 2u32,
        };
        let uniform_data = ImageSliceUniform {
            bbox_min: item.bbox_min,
            axis: axis_u32,
            bbox_max: item.bbox_max,
            offset: item.offset.clamp(0.0, 1.0),
            scalar_min: item.scalar_range.0,
            scalar_max: item.scalar_range.1,
            opacity: item.opacity,
            _pad: 0.0,
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("image_slice_uniform_buf"),
            size: std::mem::size_of::<ImageSliceUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // Resolve the LUT: the item's colourmap, defaulting to Viridis, with
        // the 1x1 fallback when the builtin set is not resident.
        let lut_view = resources
            .content
            .builtin_colourmap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colour_lut
                    .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                resources.content.colourmap_views.get(preset_id.0)
            })
            .unwrap_or(&resources.content.fallback_lut_view);

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("image_slice_bg"),
            layout: &self.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(vol_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&self.vol_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::Sampler(resources.material_sampler()),
                },
            ],
        });

        let pick = (item.settings.pick_id != PickId::NONE).then(|| {
            let id_data: [u32; 4] = [item.settings.pick_id.0 as u32, 0, 0, 0];
            let pick_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("image_slice_pick_id_buf"),
                size: 16,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pick_buf, 0, bytemuck::bytes_of(&id_data));
            let pick_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("image_slice_pick_id_bg"),
                layout: &self.pick_id_bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: pick_buf.as_entire_binding(),
                }],
            });
            (pick_buf, pick_bg)
        });

        Some(ImageSliceFrame {
            bind_group,
            _uniform_buf: uniform_buf,
            pick,
            selected: item.settings.selected,
        })
    }
}
