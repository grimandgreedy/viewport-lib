//! GPU state for the image slice item type: the render, pick, and
//! outline-mask pipelines and the per-frame per-item bind groups.

use super::types::ImageSliceItem;
use super::types::SliceAxis;
use crate::shader::{scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::renderer::PickId;
use viewport_lib::resources::DeviceResources;

/// Pipelines and layouts, built lazily on the first prepare with items.
pub(super) struct ImageSliceGpu {
    pub(super) bgl: gpu::BindGroupLayout,
    pub(super) pipeline: builders::DualPipeline,
    pub(super) pick_pipeline: gpu::RenderPipeline,
    pub(super) pick_id_bgl: gpu::BindGroupLayout,
    pub(super) mask_pipeline: gpu::RenderPipeline,
    /// Linear clamp sampler for the 3D field; shared by every slice.
    pub(super) vol_sampler: gpu::Sampler,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct ImageSliceFrame {
    pub(super) bind_group: gpu::BindGroup,
    pub(super) _uniform_buf: gpu::Buffer,
    /// Object-id uniform + bind group for the GPU pick pass; `None` when the
    /// item is not pickable.
    pub(super) pick: Option<(gpu::Buffer, gpu::BindGroup)>,
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
    pub(super) fn new(device: &gpu::Device, resources: &DeviceResources) -> Self {
        let bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("image_slice_bgl"),
            entries: &[
                // binding 0: ImageSliceUniform
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
                    ty: gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: gpu::TextureViewDimension::D3,
                        sample_type: gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 2: vol_sampler (linear)
                gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Sampler(gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: lut_tex (colourmap texture_2d)
                gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: gpu::TextureViewDimension::D2,
                        sample_type: gpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Sampler(gpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = builders::wgsl_module(
            device,
            "image_slice_shader",
            &scene_shader(&[], wgsl_source!("image_slice")),
        );
        let layout = builders::standard_scene_layout(
            device,
            "image_slice_pipeline_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        let pipeline = builders::build_dual_pipeline(
            device,
            &builders::DualPipelineDesc {
                label: "image_slice_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[], // no vertex buffer: generates quad from vertex_index
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: gpu::CompareFunction::LessEqual,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
        );

        // Pick: same quad expansion (group 1 reuses the render bind group
        // unchanged), object id at group 2, constant-0 primitive channel.
        // Laid out against the shared group-0 camera, which the pick pass
        // binds before plugin dispatch.
        let pick_id_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("image_slice_pick_id_bgl"),
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
            "image_slice_pick_shader",
            &scene_shader(&[], wgsl_source!("image_slice_pick")),
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
                    Some("image_slice_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[],
                )
            },
        );

        // Outline mask: the same quad rasterised into the R8 selection mask.
        let mask_shader = builders::wgsl_module(
            device,
            "image_slice_mask_shader",
            &scene_shader(&[], wgsl_source!("image_slice_mask")),
        );
        let mask_pipeline = resources.build_mask_pipeline(
            device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("image_slice_mask_pipeline"),
                    &mask_shader,
                    "vs_main",
                    "fs_main",
                    &[],
                )
            },
        );

        let vol_sampler = builders::clamp_linear_sampler(device, "image_slice_vol_sampler");

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
        device: &gpu::Device,
        queue: &gpu::Queue,
        resources: &DeviceResources,
        item: &ImageSliceItem,
    ) -> Option<ImageSliceFrame> {
        // Check volume exists before allocating anything.
        let vol_view = resources.volume_view(item.volume_id)?;

        let axis_u32 = match item.axis {
            SliceAxis::X => 0u32,
            SliceAxis::Y => 1u32,
            SliceAxis::Z => 2u32,
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
        let uniform_buf = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("image_slice_uniform_buf"),
            size: std::mem::size_of::<ImageSliceUniform>() as u64,
            usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // Resolve the LUT: the item's colourmap, defaulting to Viridis, with
        // the 1x1 fallback when the builtin set is not resident.
        // An item that names a colourmap gets that one or the neutral fallback;
        // a stale id does not silently fall back to the default preset.
        let lut_view = item
            .colour_lut
            .and_then(|id| resources.colourmap_view(id))
            .unwrap_or_else(|| {
                resources.builtin_colourmap_view(viewport_lib::resources::BuiltinColourmap::Viridis)
            });

        let bind_group = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("image_slice_bg"),
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
                    resource: gpu::BindingResource::Sampler(&self.vol_sampler),
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
                label: Some("image_slice_pick_id_buf"),
                size: 16,
                usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pick_buf, 0, bytemuck::bytes_of(&id_data));
            let pick_bg = device.create_bind_group(&gpu::BindGroupDescriptor {
                label: Some("image_slice_pick_id_bg"),
                layout: &self.pick_id_bgl,
                entries: &[gpu::BindGroupEntry {
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
