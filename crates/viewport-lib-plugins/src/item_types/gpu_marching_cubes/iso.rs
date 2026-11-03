//! The marching cubes item type on a device with no compute stage or storage
//! buffers (the compatibility tier). The field is uploaded as a 3D texture
//! and the isosurface is ray-marched in a fragment shader
//! (`mc_isosurface.wgsl`), so a new isovalue costs a uniform write rather
//! than an extraction. There are no triangles here: wireframe items draw as
//! surfaces, and nothing casts shadows.

use super::pipeline::McSurfaceRaw;
use super::store::McVolumeGpuData;
use super::types::GpuMarchingCubesItem;
use crate::item_types::shader::{lit_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::gpu::util::DeviceExt as _;
use viewport_lib::plugin_api::{builders, shared_wgsl};
use viewport_lib::resources::DeviceResources;
use viewport_lib_geometry::volume::grid::VolumeData;

/// The ray-marched isosurface's shader: the shared shading, then the march.
pub(crate) fn isosurface_source() -> String {
    lit_shader(
        &[
            shared_wgsl::SHARED_CSM_WGSL,
            shared_wgsl::SHARED_SHADOW_ATLAS_WGSL,
            shared_wgsl::SHARED_CLIP_WGSL,
        ],
        &[wgsl_source!("mc_shading"), wgsl_source!("mc_isosurface")].join("\n"),
    )
}

/// An uploaded field as the ray march reads it: one `R32Float` texel per node.
pub(crate) struct McField {
    view: gpu::TextureView,
    dims: [u32; 3],
    origin: [f32; 3],
    spacing: [f32; 3],
}

impl McField {
    pub(super) fn bytes(&self) -> u64 {
        self.dims.iter().map(|&d| d as u64).product::<u64>() * 4
    }
}

/// Upload a field as a 3D texture. Refused when a side is past the device's
/// 3D texture limit (256 on WebGL2).
pub(super) fn build_field(
    device: &gpu::Device,
    queue: &gpu::Queue,
    vol: &VolumeData,
) -> viewport_lib::error::ViewportResult<McVolumeGpuData> {
    let [nx, ny, nz] = vol.dims;
    let max = device.limits().max_texture_dimension_3d;
    if nx.max(ny).max(nz) > max {
        return Err(viewport_lib::error::ViewportError::DeviceUnsupported {
            reason: format!(
                "a {nx}x{ny}x{nz} marching cubes grid is past this device's 3D texture limit of {max} per side"
            ),
        });
    }
    let texture = device.create_texture_with_data(
        queue,
        &gpu::TextureDescriptor {
            label: Some("mc_field"),
            size: gpu::Extent3d {
                width: nx,
                height: ny,
                depth_or_array_layers: nz,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D3,
            format: gpu::TextureFormat::R32Float,
            usage: gpu::TextureUsages::TEXTURE_BINDING | gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        gpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&vol.data),
    );
    Ok(McVolumeGpuData {
        slabs: Vec::new(),
        dims: vol.dims,
        external_scalar: None,
        field: Some(McField {
            view: texture.create_view(&gpu::TextureViewDescriptor::default()),
            dims: vol.dims,
            origin: vol.origin,
            spacing: vol.spacing,
        }),
    })
}

/// `McIsoUniform` in mc_isosurface.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct McIsoRaw {
    origin: [f32; 3],
    isovalue: f32,
    spacing: [f32; 3],
    step: f32,
    dims: [u32; 3],
    max_steps: u32,
}

impl McIsoRaw {
    /// `origin` is the frame's world origin; the box is placed relative to
    /// it.
    fn new(field: &McField, isovalue: f32, origin: glam::DVec3) -> Self {
        // Half a cell per step, so a crossing as thin as a cell is not
        // stepped over, and enough steps for the box's diagonal.
        let step = 0.5 * field.spacing.iter().copied().fold(f32::INFINITY, f32::min);
        let cells = glam::Vec3::from(field.dims.map(|d| d.saturating_sub(1) as f32));
        let max_steps = ((cells.length() * 2.0).ceil() as u32 + 2).min(4096);
        Self {
            origin: crate::item_types::origin::rel_point(field.origin, origin),
            isovalue,
            spacing: field.spacing,
            step,
            dims: field.dims,
            max_steps,
        }
    }
}

/// Members of [`McIsoPipelines`].
pub(super) const COLOUR: usize = 0;
pub(super) const MASK: usize = 1;

/// What an isosurface pipeline build reads.
pub(super) struct McIsoRecipe {
    device: gpu::Device,
    builder: viewport_lib::plugin_api::PipelineBuilder,
    layout: gpu::PipelineLayout,
    bgl: gpu::BindGroupLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    ldr_format: gpu::TextureFormat,
    depth_format: gpu::TextureFormat,
}

/// The LDR colour pipeline and the selection mask.
pub(super) type McIsoPipelines = viewport_lib::plugin_api::LazyPipelines<McIsoRecipe, 2>;

fn build(r: &McIsoRecipe, i: usize) -> gpu::RenderPipeline {
    // Back faces only: the ray starts at the box entry, or at the eye when it
    // is inside the box.
    match i {
        COLOUR => builders::build_dual_pipeline_variant(
            &r.device,
            &builders::DualPipelineDesc {
                label: "mc_isosurface_pipeline",
                layout: &r.layout,
                shader: r.shader.get(),
                vertex_entry: "vs_iso",
                fragment_entry: "fs_iso",
                vertex_buffers: &[],
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(gpu::Face::Front),
                depth_write: true,
                depth_compare: viewport_lib::plugin_api::depth::DEPTH_COMPARE,
                sample_count: 1,
                ldr_format: r.ldr_format,
                depth_format: r.depth_format,
                indirect_fraction_entry: None,
            },
            false,
        ),
        _ => r.builder.build_mask_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(gpu::Face::Front),
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&r.bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("mc_isosurface_mask_pipeline"),
                    r.shader.get(),
                    "vs_iso",
                    "fs_iso_mask",
                    &[],
                )
            },
        ),
    }
}

/// Pipelines and the group-1 layout, made on the first prepare with items.
pub(super) struct McIsoGpu {
    pub(super) pipelines: McIsoPipelines,
    bgl: gpu::BindGroupLayout,
}

/// One item's draw state for this frame.
pub(super) struct McIsoFrame {
    pub(super) bind_group: gpu::BindGroup,
    pub(super) selected: bool,
}

impl McIsoGpu {
    pub(super) fn new(device: &gpu::Device, resources: &DeviceResources) -> Self {
        let bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("mc_isosurface_bgl"),
            entries: &[
                builders::uniform_entry(0, gpu::ShaderStages::FRAGMENT),
                gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Texture {
                        sample_type: gpu::TextureSampleType::Float { filterable: false },
                        view_dimension: gpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                builders::uniform_entry(2, gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT),
            ],
        });
        let layout = builders::standard_scene_layout(
            device,
            "mc_isosurface_layout",
            resources.group0_layout(),
            &bgl,
        );
        let shader = resources.lazy_module(device, "mc_isosurface_shader", isosurface_source());
        let pipelines = resources.lazy_pipelines(
            McIsoRecipe {
                device: device.clone(),
                builder: resources.pipeline_builder(),
                layout,
                bgl: bgl.clone(),
                shader,
                ldr_format: resources.target_format(),
                depth_format: resources.scene_depth_format(),
            },
            build,
        );
        Self { pipelines, bgl }
    }

    /// Whether the colour pipeline can draw this frame. The mask waits for
    /// it, so a surface is never outlined before it is drawn.
    pub(super) fn drawn(&self) -> bool {
        self.pipelines.available(COLOUR)
    }

    /// The draw state of every visible item whose volume resolves.
    pub(super) fn frame(
        &self,
        device: &gpu::Device,
        volumes: &super::store::McVolumeStore,
        jobs: &[GpuMarchingCubesItem],
        origin: glam::DVec3,
    ) -> Vec<McIsoFrame> {
        jobs.iter()
            .filter(|job| !job.settings.hidden)
            .filter_map(|job| {
                let field = volumes.get(job.volume_id)?.field.as_ref()?;
                let material = device.create_buffer_init(&gpu::util::BufferInitDescriptor {
                    label: Some("mc_isosurface_material"),
                    contents: bytemuck::bytes_of(&McSurfaceRaw::for_item(job)),
                    usage: gpu::BufferUsages::UNIFORM,
                });
                let iso = device.create_buffer_init(&gpu::util::BufferInitDescriptor {
                    label: Some("mc_isosurface_uniform"),
                    contents: bytemuck::bytes_of(&McIsoRaw::new(field, job.isovalue, origin)),
                    usage: gpu::BufferUsages::UNIFORM,
                });
                let bind_group = device.create_bind_group(&gpu::BindGroupDescriptor {
                    label: Some("mc_isosurface_bg"),
                    layout: &self.bgl,
                    entries: &[
                        gpu::BindGroupEntry {
                            binding: 0,
                            resource: material.as_entire_binding(),
                        },
                        gpu::BindGroupEntry {
                            binding: 1,
                            resource: gpu::BindingResource::TextureView(&field.view),
                        },
                        gpu::BindGroupEntry {
                            binding: 2,
                            resource: iso.as_entire_binding(),
                        },
                    ],
                });
                Some(McIsoFrame {
                    bind_group,
                    selected: job.settings.selected,
                })
            })
            .collect()
    }
}
