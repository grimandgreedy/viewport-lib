//! GPU state for the volume surface slice item type: the render, pick, and
//! outline-mask pipelines and the per-frame per-item bind groups. The geometry
//! is a consumer-uploaded mesh, so nothing here owns vertex or index buffers.

use crate::renderer::{PickId, VolumeSurfaceSliceItem};
use crate::resources::{DeviceResources, Vertex, VertexBufferLayoutExt as _};

/// Pipelines and layouts, built lazily on the first prepare with items.
pub(super) struct SliceGpu {
    bgl: crate::gpu::BindGroupLayout,
    pub(super) pipeline: crate::resources::DualPipeline,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pick_id_bgl: crate::gpu::BindGroupLayout,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct SliceFrame {
    pub(super) bind_group: crate::gpu::BindGroup,
    pub(super) _uniform_buf: crate::gpu::Buffer,
    /// The mesh to draw, resolved through the draw hook's mesh handle.
    pub(super) mesh_id: crate::MeshId,
    /// Object-id uniform + bind group for the GPU pick pass; `None` when the
    /// item is not pickable.
    pub(super) pick: Option<(crate::gpu::Buffer, crate::gpu::BindGroup)>,
    pub(super) selected: bool,
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
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        let filterable = |view_dimension| crate::gpu::BindingType::Texture {
            multisampled: false,
            view_dimension,
            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
        };
        let sampler =
            || crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering);
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_surface_slice_bgl"),
            entries: &[
                // binding 0: SliceUniform
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
                    ty: filterable(crate::gpu::TextureViewDimension::D3),
                    count: None,
                },
                // binding 2: vol_sampler (linear)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: sampler(),
                    count: None,
                },
                // binding 3: lut_tex (colourmap texture_2d)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: filterable(crate::gpu::TextureViewDimension::D2),
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: sampler(),
                    count: None,
                },
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "volume_surface_slice_shader",
            crate::resources::builders::wgsl_source!("volume_surface_slice"),
        );
        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "volume_surface_slice_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "volume_surface_slice_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[Vertex::buffer_layout()],
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::LessEqual,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );

        // Outline mask: the slice mesh transformed by its model matrix. The
        // shared mesh mask pipeline also carries position-override and deform
        // support, neither of which a slice ever uses, so this is the same
        // vertex transform with those branches removed.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "volume_surface_slice_mask_shader",
            crate::resources::builders::wgsl_source!("volume_surface_slice_mask"),
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
                    Some("volume_surface_slice_mask_pipeline"),
                    &mask_shader,
                    "vs_main",
                    "fs_main",
                    &[Vertex::buffer_layout()],
                )
            },
        );

        // Pick: the same mesh, writing the item's object id. Group 1 reuses the
        // render bind group, group 2 is the object id.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_surface_slice_pick_id_bgl"),
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
            "volume_surface_slice_pick_shader",
            crate::resources::builders::wgsl_source!("volume_surface_slice_pick"),
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
                    Some("volume_surface_slice_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[Vertex::buffer_layout()],
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

    /// Build the per-item uniform and bind group, or `None` when the item's
    /// volume or mesh is not resident.
    pub(super) fn upload_item(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
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
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_surface_slice_uniform"),
            size: std::mem::size_of::<SliceUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let vol_sampler = crate::resources::builders::clamp_linear_sampler(
            device,
            "volume_surface_slice_vol_sampler",
        );

        let lut_view = item
            .colour_lut
            .and_then(|id| resources.colourmap_view(id))
            .unwrap_or_else(|| {
                resources.builtin_colourmap_view(crate::resources::BuiltinColourmap::Viridis)
            });

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("volume_surface_slice_bg"),
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
                    resource: crate::gpu::BindingResource::Sampler(&vol_sampler),
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
                label: Some("volume_surface_slice_pick_id_buf"),
                size: std::mem::size_of_val(&id_data) as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pick_buf, 0, bytemuck::cast_slice(&id_data));
            let pick_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("volume_surface_slice_pick_id_bg"),
                layout: &self.pick_id_bgl,
                entries: &[crate::gpu::BindGroupEntry {
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
        })
    }
}
