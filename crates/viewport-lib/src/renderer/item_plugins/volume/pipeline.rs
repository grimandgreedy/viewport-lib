//! GPU state for the volume item type: the render, pick, and outline-mask
//! pipelines, the shared default opacity LUT, and the per-frame per-item bind
//! groups and cube proxy buffers.

use crate::renderer::{ClipObject, ClipShape, PickId, VolumeItem};
use crate::resources::DeviceResources;

/// Pipelines, layouts, and the default opacity ramp, built lazily on the first
/// prepare with items.
pub(super) struct VolumeGpu {
    bgl: crate::gpu::BindGroupLayout,
    pub(super) pipeline: crate::resources::DualPipeline,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pick_id_bgl: crate::gpu::BindGroupLayout,
    /// Linear ramp opacity LUT (256x1, R8Unorm), bound whenever an item does
    /// not name one of its own. Kept alive by the view it backs.
    _default_opacity_lut: crate::gpu::Texture,
    default_opacity_lut_view: crate::gpu::TextureView,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct VolumeFrame {
    pub(super) bind_group: crate::gpu::BindGroup,
    pub(super) vertex_buffer: crate::gpu::Buffer,
    pub(super) index_buffer: crate::gpu::Buffer,
    pub(super) _uniform_buf: crate::gpu::Buffer,
    /// Object-id uniform + bind group for the GPU pick pass; `None` when the
    /// item is not pickable.
    pub(super) pick: Option<(crate::gpu::Buffer, crate::gpu::BindGroup)>,
    /// When true the ray-march draw is skipped; an OBB wireframe polyline is
    /// rendered by the core line substrate instead.
    pub(super) wireframe: bool,
    pub(super) selected: bool,
}

/// The unit cube the ray-march rasterises, in the bounding box's local space.
#[rustfmt::skip]
const CUBE_VERTICES: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
];
#[rustfmt::skip]
const CUBE_INDICES: [u32; 36] = [
    0,2,1, 0,3,2, 4,5,6, 4,6,7,
    0,4,7, 0,7,3, 1,2,6, 1,6,5,
    0,1,5, 0,5,4, 3,7,6, 3,6,2,
];

/// Position-only vertex layout shared by the render, mask, and pick pipelines.
const CUBE_VERTEX_LAYOUT: crate::gpu::VertexBufferLayout<'static> =
    crate::gpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: crate::gpu::VertexStepMode::Vertex,
        attributes: &[crate::gpu::VertexAttribute {
            format: crate::gpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    };

impl VolumeGpu {
    pub(super) fn new(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &DeviceResources,
    ) -> Self {
        let filterable = |view_dimension| crate::gpu::BindingType::Texture {
            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        };
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Filterable so the ray-march reconstructs the field with
                // trilinear interpolation. The bound texture is R16Float
                // (baseline filterable) or R32Float (with FLOAT32_FILTERABLE),
                // chosen by `volume_texture_format` to keep this valid.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: filterable(crate::gpu::TextureViewDimension::D3),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: filterable(crate::gpu::TextureViewDimension::D2),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: filterable(crate::gpu::TextureViewDimension::D2),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "volume_shader",
            crate::resources::builders::wgsl_source!("volume"),
        );
        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "volume_pipeline_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "volume_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[CUBE_VERTEX_LAYOUT],
                blend: Some(crate::gpu::BlendState {
                    color: crate::gpu::BlendComponent {
                        src_factor: crate::gpu::BlendFactor::SrcAlpha,
                        dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                        operation: crate::gpu::BlendOperation::Add,
                    },
                    alpha: crate::gpu::BlendComponent {
                        src_factor: crate::gpu::BlendFactor::One,
                        dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                        operation: crate::gpu::BlendOperation::Add,
                    },
                }),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );

        // Outline mask: the same ray-march into the R8 mask, so the outline
        // hugs the marched silhouette rather than the bounding cube.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "volume_outline_mask_shader",
            crate::resources::builders::wgsl_source!("volume_outline_mask"),
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
                    Some("volume_outline_mask_pipeline"),
                    &mask_shader,
                    "vs_main",
                    "fs_main",
                    &[CUBE_VERTEX_LAYOUT],
                )
            },
        );

        // Pick: the same cube, marched to the first in-threshold voxel, writing
        // the object id and that voxel's flat index. Group 1 reuses the render
        // bind group, group 2 is the object id. Both cube faces are drawn so
        // the volume stays pickable with the camera inside the box.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_pick_id_bgl"),
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
            "volume_pick_shader",
            crate::resources::builders::wgsl_source!("volume_pick"),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&bgl, &pick_id_bgl],
                ..crate::resources::PluginPipelineOpts::new(
                    Some("volume_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_pick",
                    &[CUBE_VERTEX_LAYOUT],
                )
            },
        );

        let (lut, lut_view) = default_opacity_lut(device, queue);

        Self {
            bgl,
            pipeline,
            mask_pipeline,
            pick_pipeline,
            pick_id_bgl,
            _default_opacity_lut: lut,
            default_opacity_lut_view: lut_view,
        }
    }

    /// Build the per-item uniform, bind group, and cube proxy buffers.
    pub(super) fn upload_item(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &DeviceResources,
        item: &VolumeItem,
        clip_objects: &[ClipObject],
        step_scale_multiplier: f32,
        wireframe: bool,
    ) -> VolumeFrame {
        let dims = resources.volume_dims(item.volume_id).unwrap_or_else(|| {
            let vol_id = item.volume_id;
            panic!("invalid VolumeId: {vol_id:?}")
        });

        let item_model = glam::Mat4::from_cols_array_2d(&item.model);
        let bbox_min = glam::Vec3::from(item.bbox_min);
        let bbox_max = glam::Vec3::from(item.bbox_max);
        let extent = bbox_max - bbox_min;
        let bbox_model = glam::Mat4::from_translation(bbox_min) * glam::Mat4::from_scale(extent);
        let model = item_model * bbox_model;
        let inv_model = model.inverse();

        let max_dim = dims[0].max(dims[1]).max(dims[2]) as f32;
        let step_size = (item.step_scale * step_scale_multiplier) / max_dim.max(1.0);

        let mut clip_plane_data = [[0.0f32; 4]; 6];
        let mut num_clip = 0u32;
        for obj in clip_objects.iter().filter(|o| o.enabled) {
            if num_clip >= 6 {
                break;
            }
            if let ClipShape::Plane {
                normal, distance, ..
            } = obj.shape
            {
                clip_plane_data[num_clip as usize] = [normal[0], normal[1], normal[2], distance];
                num_clip += 1;
            }
        }

        let mut uniform_data = [0u8; 304];
        {
            let mut offset = 0usize;
            let mut put = |bytes: &[u8]| {
                uniform_data[offset..offset + bytes.len()].copy_from_slice(bytes);
                offset += bytes.len();
            };
            put(bytemuck::bytes_of(&model.to_cols_array()));
            put(bytemuck::bytes_of(&inv_model.to_cols_array()));
            put(bytemuck::bytes_of(&item.bbox_min));
            put(bytemuck::bytes_of(&step_size));
            put(bytemuck::bytes_of(&item.bbox_max));
            put(bytemuck::bytes_of(&item.opacity_scale));
            put(bytemuck::bytes_of(&item.scalar_range.0));
            put(bytemuck::bytes_of(&item.scalar_range.1));
            put(bytemuck::bytes_of(&item.threshold_min));
            put(bytemuck::bytes_of(&item.threshold_max));
            // ItemSettings.unlit forces gradient shading off regardless of the
            // per-item enable_shading toggle. The ray-marcher's only lighting
            // path is gradient Phong, gated by VolumeUniform.enable_shading;
            // ORing unlit here gives consumers a uniform way to disable
            // lighting across all item types.
            let shading_u32: u32 = u32::from(item.enable_shading && !item.settings.unlit);
            put(bytemuck::bytes_of(&shading_u32));
            put(bytemuck::bytes_of(&num_clip));
            let use_nan_colour_u32: u32 = u32::from(item.nan_colour.is_some());
            put(bytemuck::bytes_of(&use_nan_colour_u32));
            put(&[0u8; 4]);
            let nan_colour = item
                .nan_colour
                .map(|c| c.to_linear_rgba())
                .unwrap_or([0.0f32; 4]);
            put(bytemuck::bytes_of(&nan_colour));
            for cp in &clip_plane_data {
                put(bytemuck::bytes_of(cp));
            }
            debug_assert_eq!(offset, 304);
        }

        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_uniform_buf"),
            size: 304,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(uniform_buf.slice(..), &uniform_data);
        uniform_buf.unmap();

        let volume_view = resources
            .volume_view(item.volume_id)
            .expect("VolumeId validated above");

        let colour_lut_view = match item.colour_lut {
            Some(id) => resources.colourmap_view(id),
            None => resources.builtin_colourmap_view(crate::resources::BuiltinColourmap::Viridis),
        }
        .unwrap_or_else(|| resources.fallback_colourmap_view());

        let opacity_lut_view = item
            .opacity_lut
            .and_then(|id| resources.colourmap_view(id))
            .unwrap_or(&self.default_opacity_lut_view);

        // Trilinear sampling of the scalar field: the volume texture is
        // filterable (R16Float, or R32Float with FLOAT32_FILTERABLE), so a
        // linear clamp sampler reconstructs it smoothly instead of
        // nearest-neighbor.
        let volume_sampler =
            crate::resources::builders::clamp_linear_sampler(device, "volume_sampler");
        let lut_sampler =
            crate::resources::builders::clamp_linear_mip_sampler(device, "volume_lut_sampler");

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("volume_bind_group"),
            layout: &self.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(volume_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&volume_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(colour_lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(opacity_lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::Sampler(&lut_sampler),
                },
            ],
        });

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_cube_vb_frame"),
            size: std::mem::size_of_val(&CUBE_VERTICES) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            vertex_buffer.slice(..),
            bytemuck::cast_slice(&CUBE_VERTICES),
        );
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("volume_cube_ib_frame"),
            size: std::mem::size_of_val(&CUBE_INDICES) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            index_buffer.slice(..),
            bytemuck::cast_slice(&CUBE_INDICES),
        );
        index_buffer.unmap();

        let pick = (item.settings.pick_id != PickId::NONE).then(|| {
            let id_data: [u32; 4] = [item.settings.pick_id.0 as u32, 0, 0, 0];
            let pick_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("volume_pick_id_buf"),
                size: std::mem::size_of_val(&id_data) as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pick_buf, 0, bytemuck::cast_slice(&id_data));
            let pick_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("volume_pick_id_bg"),
                layout: &self.pick_id_bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: pick_buf.as_entire_binding(),
                }],
            });
            (pick_buf, pick_bg)
        });

        VolumeFrame {
            bind_group,
            vertex_buffer,
            index_buffer,
            _uniform_buf: uniform_buf,
            pick,
            wireframe,
            selected: item.settings.selected,
        }
    }
}

/// Build the 256x1 linear ramp bound as the opacity transfer function for any
/// item that does not name one.
fn default_opacity_lut(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
) -> (crate::gpu::Texture, crate::gpu::TextureView) {
    let mut data = [0u8; 256];
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as u8;
    }
    let size = crate::gpu::Extent3d {
        width: 256,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some("volume_default_opacity_lut"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::R8Unorm,
        usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        crate::gpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: crate::gpu::Origin3d::ZERO,
            aspect: crate::gpu::TextureAspect::All,
        },
        &data,
        crate::gpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: Some(1),
        },
        size,
    );
    let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
    (texture, view)
}
