//! Group 0 on the forward tier: the camera, lights and clip uniforms every
//! lit item type binds, without the storage buffers the full tier's group 0
//! carries (light list, clusters, indirect light, material block).
//!
//! Binding numbers and types match the full tier's layout for every binding
//! kept, so the shared WGSL declarations work unchanged on both tiers. The one
//! difference a shader sees is the light list at binding 13: a uniform array
//! of [`FORWARD_MAX_LIGHTS`] instead of a storage array, which
//! `builders::adapt_wgsl` swaps in. Shadows and image-based lighting are off
//! on this tier, so their bindings hold 1x1 placeholders.

use crate::gpu::ShaderStages;
use crate::resources::builders::{LoggedAlloc, sampler_entry, texture_entry, uniform_entry};
use crate::resources::{LightsUniform, SingleLightUniform};

/// Lights the forward tier's light list holds. Lights past it are dropped in
/// rank order and counted in `FrameStats::lights_dropped`.
pub const FORWARD_MAX_LIGHTS: usize = 16;

/// The resources behind group 0 on the forward tier.
pub(crate) struct ForwardTier {
    pub(crate) camera_bgl: crate::gpu::BindGroupLayout,
    /// The shadow pass's group 0, the same layout as the full tier's. No
    /// shadow pass runs on this tier; it exists so a plugin can build its
    /// shadow pipeline against it in `init_gpu`.
    pub(crate) shadow_camera_bgl: crate::gpu::BindGroupLayout,
    /// The lights header (binding 3).
    pub(crate) lights_header_buf: crate::gpu::Buffer,
    /// The light list (binding 13), [`FORWARD_MAX_LIGHTS`] entries.
    pub(crate) light_list_buf: crate::gpu::Buffer,
    shadow_sampler: crate::gpu::Sampler,
    ibl_sampler: crate::gpu::Sampler,
    shadow_atlas_view: crate::gpu::TextureView,
    point_shadow_view: crate::gpu::TextureView,
    ibl_array_view: crate::gpu::TextureView,
    flat_view: crate::gpu::TextureView,
}

impl ForwardTier {
    pub(crate) fn new(
        device: &crate::gpu::Device,
        caps: &crate::capabilities::Capabilities,
    ) -> Self {
        let both = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let fragment = ShaderStages::FRAGMENT;
        let float_array = |binding| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: crate::gpu::BindingType::Texture {
                sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                view_dimension: crate::gpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let depth = |binding, view_dimension| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: crate::gpu::BindingType::Texture {
                sample_type: crate::gpu::TextureSampleType::Depth,
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let camera_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("forward_camera_bgl"),
            entries: &[
                uniform_entry(0, both),
                depth(1, crate::gpu::TextureViewDimension::D2),
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: fragment,
                    ty: crate::gpu::BindingType::Sampler(
                        crate::gpu::SamplerBindingType::Comparison,
                    ),
                    count: None,
                },
                uniform_entry(3, fragment),
                uniform_entry(4, both),
                uniform_entry(5, fragment),
                uniform_entry(6, fragment),
                float_array(7),
                float_array(8),
                texture_entry(9, fragment),
                sampler_entry(10, fragment),
                texture_entry(11, fragment),
                uniform_entry(13, fragment),
                depth(
                    17,
                    crate::resources::shadow::point_shadow_view_dimension(caps.cube_array_textures),
                ),
            ],
        });
        let shadow_camera_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("forward_shadow_camera_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    uniform_entry(23, ShaderStages::VERTEX),
                ],
            });

        let uniform = |label, size| {
            device.logged_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let lights_header_buf = uniform(
            "forward_lights_header",
            std::mem::size_of::<LightsUniform>() as u64,
        );
        let light_list_buf = uniform(
            "forward_light_list",
            (FORWARD_MAX_LIGHTS * std::mem::size_of::<SingleLightUniform>()) as u64,
        );

        let placeholder = |label, format, layers, dimension| {
            device
                .logged_texture(&crate::gpu::TextureDescriptor {
                    label: Some(label),
                    size: crate::gpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: layers,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: crate::gpu::TextureDimension::D2,
                    format,
                    usage: crate::gpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&crate::gpu::TextureViewDescriptor {
                    dimension: Some(dimension),
                    ..Default::default()
                })
        };
        let point_dimension =
            crate::resources::shadow::point_shadow_view_dimension(caps.cube_array_textures);
        let point_layers = if caps.cube_array_textures { 6 } else { 1 };
        Self {
            camera_bgl,
            shadow_camera_bgl,
            lights_header_buf,
            light_list_buf,
            shadow_sampler: crate::resources::builders::comparison_sampler(
                device,
                "forward_shadow_sampler",
                crate::gpu::CompareFunction::LessEqual,
            ),
            ibl_sampler: device.create_sampler(&crate::gpu::SamplerDescriptor {
                label: Some("forward_ibl_sampler"),
                mag_filter: crate::gpu::FilterMode::Linear,
                min_filter: crate::gpu::FilterMode::Linear,
                ..Default::default()
            }),
            shadow_atlas_view: placeholder(
                "forward_shadow_atlas",
                crate::gpu::TextureFormat::Depth32Float,
                1,
                crate::gpu::TextureViewDimension::D2,
            ),
            point_shadow_view: placeholder(
                "forward_point_shadow",
                crate::gpu::TextureFormat::Depth32Float,
                point_layers,
                point_dimension,
            ),
            ibl_array_view: placeholder(
                "forward_ibl",
                crate::gpu::TextureFormat::Rgba8Unorm,
                1,
                crate::gpu::TextureViewDimension::D2Array,
            ),
            flat_view: placeholder(
                "forward_flat",
                crate::gpu::TextureFormat::Rgba8Unorm,
                1,
                crate::gpu::TextureViewDimension::D2,
            ),
        }
    }

    /// Group 0 for a viewport, over its own camera, clip and shadow-info
    /// buffers.
    pub(crate) fn camera_bind_group(
        &self,
        device: &crate::gpu::Device,
        camera_buf: &crate::gpu::Buffer,
        clip_planes_buf: &crate::gpu::Buffer,
        shadow_info_buf: &crate::gpu::Buffer,
        clip_volume_buf: &crate::gpu::Buffer,
        label: &str,
    ) -> crate::gpu::BindGroup {
        use crate::gpu::{BindGroupEntry, BindingResource};
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.camera_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.shadow_atlas_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.shadow_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.lights_header_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: clip_planes_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: shadow_info_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: clip_volume_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(&self.ibl_array_view),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::TextureView(&self.ibl_array_view),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: BindingResource::TextureView(&self.flat_view),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: BindingResource::Sampler(&self.ibl_sampler),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: BindingResource::TextureView(&self.flat_view),
                },
                BindGroupEntry {
                    binding: 13,
                    resource: self.light_list_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 17,
                    resource: BindingResource::TextureView(&self.point_shadow_view),
                },
            ],
        })
    }
}
