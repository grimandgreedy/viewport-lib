//! The resources behind group 0 on the forward tier: its own lights header
//! and light list, and 1x1 placeholders for the shadow and IBL bindings,
//! which are off on this tier. The layout and the bind group come from the
//! tier's `Group0Profile` (`resources::group0`), like the full tier's.

use crate::resources::builders::LoggedAlloc;
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
    /// The samplers plugins share, the same configuration as the full tier's.
    pub(crate) material_sampler: crate::gpu::Sampler,
    pub(crate) lut_sampler: crate::gpu::Sampler,
    pub(crate) shadow_sampler: crate::gpu::Sampler,
    ibl_sampler: crate::gpu::Sampler,
    shadow_atlas_view: crate::gpu::TextureView,
    point_shadow_view: crate::gpu::TextureView,
    ibl_array_view: crate::gpu::TextureView,
    flat_view: crate::gpu::TextureView,
}

impl ForwardTier {
    pub(crate) fn new(
        device: &crate::gpu::Device,
        profile: &crate::resources::group0::Group0Profile,
    ) -> Self {
        let camera_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("forward_camera_bgl"),
            entries: &profile.layout_entries(),
        });
        let shadow_camera_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("forward_shadow_camera_bgl"),
                entries: &crate::resources::group0::shadow_layout_entries(),
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
        let lights = profile.uniform_light_list.unwrap_or(FORWARD_MAX_LIGHTS);
        let light_list_buf = uniform(
            "forward_light_list",
            (lights * std::mem::size_of::<SingleLightUniform>()) as u64,
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
        let point_dimension = profile.point_shadow_dimension;
        let point_layers = match point_dimension {
            crate::gpu::TextureViewDimension::CubeArray => 6,
            _ => 1,
        };
        Self {
            camera_bgl,
            shadow_camera_bgl,
            lights_header_buf,
            light_list_buf,
            material_sampler: crate::resources::builders::repeat_linear_sampler(
                device,
                "forward_material_sampler",
                crate::gpu::FilterMode::Nearest,
            ),
            lut_sampler: crate::resources::builders::clamp_linear_sampler(
                device,
                "forward_lut_sampler",
            ),
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

    /// What this tier's group 0 binds, around a viewport's own buffers.
    pub(crate) fn group0_resources<'a>(
        &'a self,
        per_viewport: crate::resources::group0::PerViewport<'a>,
    ) -> crate::resources::group0::Group0Resources<'a> {
        crate::resources::group0::Group0Resources {
            per_viewport,
            shadow_atlas: &self.shadow_atlas_view,
            shadow_sampler: &self.shadow_sampler,
            lights_header: &self.lights_header_buf,
            ibl_irradiance: &self.ibl_array_view,
            ibl_specular: &self.ibl_array_view,
            brdf_lut: &self.flat_view,
            ibl_sampler: &self.ibl_sampler,
            skybox: &self.flat_view,
            light_list: &self.light_list_buf,
            point_shadows: &self.point_shadow_view,
            clusters: None,
            mesh_family: None,
        }
    }
}
