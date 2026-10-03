//! GPU state for the volume item type: the render, pick, and outline-mask
//! pipelines, each built the first time a draw needs it, the shared default
//! opacity LUT, and the per-frame per-item bind groups and cube proxy buffers.

use super::types::VolumeItem;
use viewport_lib::renderer::{ClipObject, ClipShape, PickId};
use viewport_lib::resources::DeviceResources;

/// Members of [`VolumePipelines`].
pub(super) const COLOUR_LDR: usize = 0;
pub(super) const COLOUR_HDR: usize = 1;
pub(super) const MASK: usize = 2;
pub(super) const PICK: usize = 3;

/// What a volume pipeline build reads.
pub(super) struct VolumeRecipe {
    device: viewport_lib::gpu::Device,
    builder: viewport_lib::plugin_api::PipelineBuilder,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::gpu::ShaderModule,
    mask_shader: viewport_lib::gpu::ShaderModule,
    pick_shader: viewport_lib::gpu::ShaderModule,
    bgl: viewport_lib::gpu::BindGroupLayout,
    pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
}

/// The ray-march in both formats, the outline mask and the pick pipeline.
pub(super) type VolumePipelines = viewport_lib::plugin_api::LazyPipelines<VolumeRecipe, 4>;

fn build(r: &VolumeRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    match i {
        COLOUR_LDR | COLOUR_HDR => viewport_lib::plugin_api::builders::build_dual_pipeline_variant(
            &r.device,
            &viewport_lib::plugin_api::builders::DualPipelineDesc {
                label: "volume_pipeline",
                layout: &r.layout,
                shader: &r.shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[CUBE_VERTEX_LAYOUT],
                blend: Some(viewport_lib::gpu::BlendState {
                    color: viewport_lib::gpu::BlendComponent {
                        src_factor: viewport_lib::gpu::BlendFactor::SrcAlpha,
                        dst_factor: viewport_lib::gpu::BlendFactor::OneMinusSrcAlpha,
                        operation: viewport_lib::gpu::BlendOperation::Add,
                    },
                    alpha: viewport_lib::gpu::BlendComponent {
                        src_factor: viewport_lib::gpu::BlendFactor::One,
                        dst_factor: viewport_lib::gpu::BlendFactor::OneMinusSrcAlpha,
                        operation: viewport_lib::gpu::BlendOperation::Add,
                    },
                }),
                topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: viewport_lib::gpu::CompareFunction::Less,
                sample_count: r.sample_count,
                ldr_format: r.ldr_format,
            },
            i == COLOUR_HDR,
        ),
        // Outline mask: the same ray-march into the R8 mask, so the outline
        // hugs the marched silhouette rather than the bounding cube.
        MASK => r.builder.build_mask_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&r.bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("volume_outline_mask_pipeline"),
                    &r.mask_shader,
                    "vs_main",
                    "fs_main",
                    &[CUBE_VERTEX_LAYOUT],
                )
            },
        ),
        // Pick: the same cube, marched to the first in-threshold voxel, writing
        // the object id and that voxel's flat index. Group 1 reuses the render
        // bind group, group 2 is the object id. Both cube faces are drawn so
        // the volume stays pickable with the camera inside the box.
        _ => r.builder.build_pick_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    front_face: viewport_lib::gpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&r.bgl, &r.pick_id_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("volume_pick_pipeline"),
                    &r.pick_shader,
                    "vs_main",
                    "fs_pick",
                    &[CUBE_VERTEX_LAYOUT],
                )
            },
        ),
    }
}

/// Pipelines and layouts, made on the first prepare with items.
pub(super) struct VolumeGpu {
    bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) pipelines: VolumePipelines,
    pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
}

/// Linear ramp opacity LUT (256x1, R8Unorm), bound whenever an item does not
/// name one of its own. Made on the first prepare, which has the queue its
/// upload needs.
pub(super) struct DefaultOpacityLut {
    _texture: viewport_lib::gpu::Texture,
    view: viewport_lib::gpu::TextureView,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct VolumeFrame {
    pub(super) bind_group: viewport_lib::gpu::BindGroup,
    pub(super) vertex_buffer: viewport_lib::gpu::Buffer,
    pub(super) index_buffer: viewport_lib::gpu::Buffer,
    pub(super) _uniform_buf: viewport_lib::gpu::Buffer,
    /// Object-id uniform + bind group for the GPU pick pass; `None` when the
    /// item is not pickable.
    pub(super) pick: Option<(viewport_lib::gpu::Buffer, viewport_lib::gpu::BindGroup)>,
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
const CUBE_VERTEX_LAYOUT: viewport_lib::gpu::VertexBufferLayout<'static> =
    viewport_lib::gpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: viewport_lib::gpu::VertexStepMode::Vertex,
        attributes: &[viewport_lib::gpu::VertexAttribute {
            format: viewport_lib::gpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    };

impl VolumeGpu {
    pub(super) fn new(device: &viewport_lib::gpu::Device, resources: &DeviceResources) -> Self {
        let filterable = |view_dimension| viewport_lib::gpu::BindingType::Texture {
            sample_type: viewport_lib::gpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        };
        let bgl = device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_bgl"),
            entries: &[
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: viewport_lib::gpu::ShaderStages::VERTEX
                        | viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Buffer {
                        ty: viewport_lib::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Filterable so the ray-march reconstructs the field with
                // trilinear interpolation. The bound texture is R16Float
                // (baseline filterable) or R32Float (with FLOAT32_FILTERABLE),
                // chosen by `volume_texture_format` to keep this valid.
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: filterable(viewport_lib::gpu::TextureViewDimension::D3),
                    count: None,
                },
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Sampler(
                        viewport_lib::gpu::SamplerBindingType::Filtering,
                    ),
                    count: None,
                },
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: filterable(viewport_lib::gpu::TextureViewDimension::D2),
                    count: None,
                },
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: filterable(viewport_lib::gpu::TextureViewDimension::D2),
                    count: None,
                },
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Sampler(
                        viewport_lib::gpu::SamplerBindingType::Filtering,
                    ),
                    count: None,
                },
            ],
        });

        let shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "volume_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("volume"),
            ),
        );
        let layout = viewport_lib::plugin_api::builders::standard_scene_layout(
            device,
            "volume_pipeline_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        let mask_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "volume_outline_mask_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("volume_outline_mask"),
            ),
        );
        // Group 2 of the pick pass: the object id.
        let pick_id_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("volume_pick_id_bgl"),
                entries: &[viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Buffer {
                        ty: viewport_lib::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pick_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "volume_pick_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("volume_pick"),
            ),
        );
        let pipelines = resources.lazy_pipelines(
            VolumeRecipe {
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

    /// Whether the ray-march can draw this frame in either format. The mask
    /// and pick passes wait for it, so a volume is never outlined or picked
    /// before it is drawn.
    pub(super) fn drawn(&self) -> bool {
        self.pipelines.available(COLOUR_LDR) || self.pipelines.available(COLOUR_HDR)
    }

    /// Build the per-item uniform, bind group, and cube proxy buffers.
    pub(super) fn upload_item(
        &self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &DeviceResources,
        default_lut: &DefaultOpacityLut,
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

        let uniform_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("volume_uniform_buf"),
            size: 304,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        viewport_lib::plugin_api::builders::write_mapped(uniform_buf.slice(..), &uniform_data);
        uniform_buf.unmap();

        let volume_view = resources
            .volume_view(item.volume_id)
            .expect("VolumeId validated above");

        let colour_lut_view = item
            .colour_lut
            .and_then(|id| resources.colourmap_view(id))
            .unwrap_or_else(|| {
                resources.builtin_colourmap_view(viewport_lib::resources::BuiltinColourmap::Viridis)
            });

        let opacity_lut_view = item
            .opacity_lut
            .and_then(|id| resources.colourmap_view(id))
            .unwrap_or(&default_lut.view);

        // Trilinear sampling of the scalar field: the volume texture is
        // filterable (R16Float, or R32Float with FLOAT32_FILTERABLE), so a
        // linear clamp sampler reconstructs it smoothly instead of
        // nearest-neighbor.
        let volume_sampler =
            viewport_lib::plugin_api::builders::clamp_linear_sampler(device, "volume_sampler");
        let lut_sampler = viewport_lib::plugin_api::builders::clamp_linear_mip_sampler(
            device,
            "volume_lut_sampler",
        );

        let bind_group = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("volume_bind_group"),
            layout: &self.bgl,
            entries: &[
                viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_lib::gpu::BindingResource::TextureView(volume_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 2,
                    resource: viewport_lib::gpu::BindingResource::Sampler(&volume_sampler),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 3,
                    resource: viewport_lib::gpu::BindingResource::TextureView(colour_lut_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 4,
                    resource: viewport_lib::gpu::BindingResource::TextureView(opacity_lut_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 5,
                    resource: viewport_lib::gpu::BindingResource::Sampler(&lut_sampler),
                },
            ],
        });

        let vertex_buffer = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("volume_cube_vb_frame"),
            size: std::mem::size_of_val(&CUBE_VERTICES) as u64,
            usage: viewport_lib::gpu::BufferUsages::VERTEX
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        viewport_lib::plugin_api::builders::write_mapped(
            vertex_buffer.slice(..),
            bytemuck::cast_slice(&CUBE_VERTICES),
        );
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("volume_cube_ib_frame"),
            size: std::mem::size_of_val(&CUBE_INDICES) as u64,
            usage: viewport_lib::gpu::BufferUsages::INDEX
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        viewport_lib::plugin_api::builders::write_mapped(
            index_buffer.slice(..),
            bytemuck::cast_slice(&CUBE_INDICES),
        );
        index_buffer.unmap();

        let pick = (item.settings.pick_id != PickId::NONE).then(|| {
            let id_data: [u32; 4] = [item.settings.pick_id.0 as u32, 0, 0, 0];
            let pick_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                label: Some("volume_pick_id_buf"),
                size: std::mem::size_of_val(&id_data) as u64,
                usage: viewport_lib::gpu::BufferUsages::UNIFORM
                    | viewport_lib::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pick_buf, 0, bytemuck::cast_slice(&id_data));
            let pick_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
                label: Some("volume_pick_id_bg"),
                layout: &self.pick_id_bgl,
                entries: &[viewport_lib::gpu::BindGroupEntry {
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

impl DefaultOpacityLut {
    /// Build the 256x1 linear ramp bound as the opacity transfer function for
    /// any item that does not name one.
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
    ) -> Self {
        let mut data = [0u8; 256];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as u8;
        }
        let size = viewport_lib::gpu::Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&viewport_lib::gpu::TextureDescriptor {
            label: Some("volume_default_opacity_lut"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: viewport_lib::gpu::TextureDimension::D2,
            format: viewport_lib::gpu::TextureFormat::R8Unorm,
            usage: viewport_lib::gpu::TextureUsages::TEXTURE_BINDING
                | viewport_lib::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            viewport_lib::gpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: viewport_lib::gpu::Origin3d::ZERO,
                aspect: viewport_lib::gpu::TextureAspect::All,
            },
            &data,
            viewport_lib::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
            size,
        );
        let view = texture.create_view(&viewport_lib::gpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}
