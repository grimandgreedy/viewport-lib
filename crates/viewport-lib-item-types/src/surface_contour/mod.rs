//! Contour lines on a mesh as an [`ItemTypePlugin`]: lines where a per-vertex
//! scalar attribute crosses a set of levels. Consumers submit
//! [`SurfaceContourItem`]s with `SceneFrame::items_mut`, beside the surface
//! item that draws the mesh.
//!
//! The lines are found per pixel in a fragment shader, so a static field costs
//! one extra draw of the mesh and a changing one costs only the attribute
//! upload. The mesh is drawn again from [`ItemTypePlugin::paint`], inside the
//! scene pass after the opaque geometry, depth-tested against what the
//! surface wrote.

use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, PaintContext, PluginItem, builders,
};
use viewport_lib::resources::mesh::mesh_store::MeshId;
use viewport_lib::{Colour, ItemSettings};

use crate::shader::{scene_shader, wgsl_source};

pub const TYPE_NAME: &str = "vpl.surface_contour";

/// Most levels a [`ContourLevels::Values`] list carries. The rest of a longer
/// list are dropped without a warning; [`ContourLevels::Spaced`] has no limit.
pub const MAX_CONTOUR_LEVELS: usize = 32;

fn shader_source() -> String {
    scene_shader(&[], wgsl_source!("surface_contour"))
}

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    vec![("surface_contour.wgsl", shader_source())]
}

/// Where the contour lines of a [`SurfaceContourItem`] fall.
#[derive(Debug, Clone, PartialEq)]
pub enum ContourLevels {
    /// A line at each of these values. Only the first [`MAX_CONTOUR_LEVELS`]
    /// are drawn; split a longer list over several items.
    Values(Vec<f32>),
    /// A line at `origin + k * interval` for every integer `k`.
    Spaced { origin: f32, interval: f32 },
}

/// Contour lines of a scalar field on one mesh.
///
/// The item draws only the lines. Submit it beside the `SceneRenderItem` that
/// draws the same mesh, with the same `model`: the lines are depth-tested
/// against that surface, so they show where it is the visible one. The field
/// that is contoured need not be the one that colours the surface.
///
/// Lines are drawn where the field is changing. Where it is flat, a plateau
/// sitting on a level draws no line, as an extracted contour would not.
///
/// Limits: a mesh deformed or displaced on the GPU is contoured in its
/// undeformed shape; a transparent surface writes no depth, so the lines on
/// the far side of a closed transparent mesh show through.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SurfaceContourItem {
    /// The mesh the field lies on.
    pub mesh_id: MeshId,
    /// Local to world transform.
    pub model: [[f32; 4]; 4],
    /// Name of a per-vertex scalar attribute the mesh was uploaded with:
    /// `AttributeData::Vertex`, or `Cell` or `Edge`, which are averaged to the
    /// vertices. An item naming an attribute the mesh lacks draws nothing.
    pub scalar_attribute: String,
    /// Where the lines fall.
    pub levels: ContourLevels,
    /// Line colour. Its alpha scales the line's coverage. Default: black.
    pub colour: Colour,
    /// Line width in logical pixels. Default: 1.5.
    pub width: f32,
    /// Shared per-item settings. Only `hidden` is read.
    pub settings: ItemSettings,
}

impl SurfaceContourItem {
    /// Black lines along `scalar_attribute` on `mesh_id`, at `levels`.
    pub fn new(
        mesh_id: MeshId,
        model: [[f32; 4]; 4],
        scalar_attribute: impl Into<String>,
        levels: ContourLevels,
    ) -> Self {
        Self {
            mesh_id,
            model,
            scalar_attribute: scalar_attribute.into(),
            levels,
            colour: [0.0, 0.0, 0.0, 1.0].into(),
            width: 1.5,
            settings: ItemSettings::default(),
        }
    }
}

impl PluginItem for SurfaceContourItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &ItemSettings {
        &self.settings
    }
}

/// One item's uniform block. Matches `Contour` in the shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ContourUniform {
    model: [[f32; 4]; 4],
    colour: [f32; 4],
    half_width: f32,
    viewport_width: f32,
    mode: u32,
    count: u32,
    origin: f32,
    interval: f32,
    spacing: f32,
    _pad: f32,
    levels: [f32; MAX_CONTOUR_LEVELS],
}

impl ContourUniform {
    /// The block for `item`, or `None` when its levels can draw nothing.
    fn new(item: &SurfaceContourItem, viewport_width: f32) -> Option<Self> {
        let mut block = Self {
            model: item.model,
            colour: item.colour.to_linear_rgba(),
            half_width: item.width.max(0.0) * 0.5,
            viewport_width,
            mode: 0,
            count: 0,
            origin: 0.0,
            interval: 0.0,
            spacing: 0.0,
            _pad: 0.0,
            levels: [0.0; MAX_CONTOUR_LEVELS],
        };
        match &item.levels {
            ContourLevels::Spaced { origin, interval } => {
                if !(interval.is_finite() && *interval > 0.0 && origin.is_finite()) {
                    return None;
                }
                block.mode = 1;
                block.origin = *origin;
                block.interval = *interval;
                block.spacing = *interval;
            }
            ContourLevels::Values(values) => {
                let mut levels: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
                if levels.is_empty() {
                    return None;
                }
                levels.truncate(MAX_CONTOUR_LEVELS);
                block.count = levels.len() as u32;
                block.levels[..levels.len()].copy_from_slice(&levels);
                block.spacing = level_spacing(&mut levels);
            }
        }
        Some(block)
    }
}

/// Smallest gap between distinct levels, or the magnitude of a lone level
/// (at least 1), as the scale a flat region is judged against.
fn level_spacing(levels: &mut [f32]) -> f32 {
    levels.sort_by(f32::total_cmp);
    levels
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|gap| *gap > 0.0)
        .fold(None, |min: Option<f32>, gap| Some(min.map_or(gap, |m| m.min(gap))))
        .unwrap_or_else(|| levels[0].abs().max(1.0))
}

/// What a pipeline build needs, held by the lazy set.
struct ContourRecipe {
    device: viewport_lib::gpu::Device,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::gpu::ShaderModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
}

/// The colour pipeline in the LDR (member 0) and HDR (member 1) formats.
type ContourPipelines = viewport_lib::plugin_api::LazyPipelines<ContourRecipe, 2>;

fn build_contour(r: &ContourRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let format = if i == 1 {
        viewport_lib::resources::HDR_COLOR_FORMAT
    } else {
        r.ldr_format
    };
    // Buffer 0 is the shared mesh vertex, read for its position alone.
    let position_layout = viewport_lib::gpu::VertexBufferLayout {
        array_stride: builders::mesh_vertex_layout().array_stride,
        step_mode: viewport_lib::gpu::VertexStepMode::Vertex,
        attributes: &viewport_lib::gpu::vertex_attr_array![0 => Float32x3],
    };
    // Tested against the surface's own depth. The bias absorbs the rounding
    // between this draw and the colour draw of the same triangles.
    let mut depth =
        builders::scene_depth_stencil(false, viewport_lib::gpu::CompareFunction::LessEqual);
    depth.bias.constant = -2;
    builders::render_pipeline(
        &r.device,
        builders::RenderPipelineDesc {
            label: "surface_contour_pipeline",
            layout: &r.layout,
            vertex_module: &r.shader,
            vertex_entry: "vs_main",
            vertex_buffers: &[position_layout, builders::scalar_attribute_layout(1)],
            fragment: Some(viewport_lib::gpu::FragmentState {
                module: &r.shader,
                entry_point: Some("fs_main"),
                targets: &[Some(viewport_lib::gpu::ColorTargetState {
                    format,
                    blend: Some(viewport_lib::gpu::BlendState::ALPHA_BLENDING),
                    write_mask: viewport_lib::gpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            // Both sides: a two-sided surface shows its lines from behind, and
            // on a closed one the depth test removes the far side.
            primitive: viewport_lib::gpu::PrimitiveState {
                topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth),
            multisample: viewport_lib::gpu::MultisampleState {
                count: r.sample_count,
                ..Default::default()
            },
            cache: None,
        },
    )
}

/// Pipelines and the layout the per-item uniform is bound with.
struct ContourGpu {
    pipelines: ContourPipelines,
    bgl: viewport_lib::gpu::BindGroupLayout,
    /// Distance between item blocks in the uniform buffer.
    stride: u64,
}

impl ContourGpu {
    fn new(
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::resources::DeviceResources,
    ) -> Self {
        let bgl = device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
            label: Some("surface_contour_bgl"),
            entries: &[viewport_lib::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: viewport_lib::gpu::ShaderStages::VERTEX
                    | viewport_lib::gpu::ShaderStages::FRAGMENT,
                ty: viewport_lib::gpu::BindingType::Buffer {
                    ty: viewport_lib::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ContourUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let layout = builders::pipeline_layout(
            device,
            "surface_contour_layout",
            &[resources.shared_bindings().group0_layout, &bgl],
        );
        let shader = builders::wgsl_module(device, "surface_contour", &shader_source());
        let pipelines = resources.lazy_pipelines(
            ContourRecipe {
                device: device.clone(),
                layout,
                shader,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
            },
            build_contour,
        );
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = (std::mem::size_of::<ContourUniform>() as u64).div_ceil(align) * align;
        Self {
            pipelines,
            bgl,
            stride,
        }
    }
}

/// One item to draw this frame.
struct ContourDraw {
    mesh_id: MeshId,
    scalar_attribute: String,
    /// Byte offset of the item's block in the uniform buffer.
    offset: u32,
}

/// The surface contour item type. Register it with
/// [`ViewportRenderer::with_item_type_plugin`](viewport_lib::renderer::ViewportRenderer::with_item_type_plugin),
/// or through [`install`](crate::install) with the rest of this crate.
#[derive(Default)]
pub struct SurfaceContourPlugin {
    gpu: Option<ContourGpu>,
    /// Every drawn item's block, its bind group, and how many blocks it holds.
    uniforms: Option<(
        viewport_lib::gpu::Buffer,
        viewport_lib::gpu::BindGroup,
        usize,
    )>,
    /// This frame's draw list, built in `prepare`.
    draws: Vec<ContourDraw>,
}

impl ItemTypePlugin for SurfaceContourPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn warm(
        &mut self,
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::resources::DeviceResources,
    ) {
        self.gpu
            .get_or_insert_with(|| ContourGpu::new(device, resources))
            .pipelines
            .request_all();
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        self.draws.clear();
        let viewport_width = ctx.viewport_size.x.max(1.0);
        let visible: Vec<(&SurfaceContourItem, ContourUniform)> = items
            .of::<SurfaceContourItem>()
            .iter()
            .filter(|item| !item.settings.hidden && !item.scalar_attribute.is_empty())
            .filter_map(|item| Some((item, ContourUniform::new(item, viewport_width)?)))
            .collect();
        if visible.is_empty() {
            return Vec::new();
        }

        let gpu = self
            .gpu
            .get_or_insert_with(|| ContourGpu::new(device, ctx.resources));
        if self
            .uniforms
            .as_ref()
            .is_none_or(|(_, _, capacity)| *capacity < visible.len())
        {
            let capacity = visible.len().next_power_of_two();
            let buffer = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                label: Some("surface_contour_uniforms"),
                size: capacity as u64 * gpu.stride,
                usage: viewport_lib::gpu::BufferUsages::UNIFORM
                    | viewport_lib::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
                label: Some("surface_contour_bg"),
                layout: &gpu.bgl,
                entries: &[viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: viewport_lib::gpu::BindingResource::Buffer(
                        viewport_lib::gpu::BufferBinding {
                            buffer: &buffer,
                            offset: 0,
                            size: std::num::NonZeroU64::new(
                                std::mem::size_of::<ContourUniform>() as u64,
                            ),
                        },
                    ),
                }],
            });
            self.uniforms = Some((buffer, bind_group, capacity));
        }

        let mut bytes = vec![0u8; visible.len() * gpu.stride as usize];
        for (i, (item, block)) in visible.iter().enumerate() {
            let at = i * gpu.stride as usize;
            bytes[at..at + std::mem::size_of::<ContourUniform>()]
                .copy_from_slice(bytemuck::bytes_of(block));
            self.draws.push(ContourDraw {
                mesh_id: item.mesh_id,
                scalar_attribute: item.scalar_attribute.clone(),
                offset: (i as u64 * gpu.stride) as u32,
            });
        }
        if let Some((buffer, _, _)) = &self.uniforms {
            queue.write_buffer(buffer, 0, &bytes);
        }
        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let (Some(gpu), Some((_, bind_group, _))) = (self.gpu.as_ref(), self.uniforms.as_ref())
        else {
            return;
        };
        if self.draws.is_empty() {
            return;
        }
        let hdr = ctx.target_format == viewport_lib::resources::HDR_COLOR_FORMAT;
        // Still compiling: the lines draw from the frame it is ready.
        let Some(pipeline) = gpu.pipelines.get(hdr as usize) else {
            return;
        };
        pass.set_pipeline(pipeline);
        for draw in &self.draws {
            pass.set_bind_group(1, bind_group, &[draw.offset]);
            if ctx
                .meshes
                .bind_scalar_attribute(pass, 1, draw.mesh_id, &draw.scalar_attribute)
            {
                ctx.meshes.draw_indexed(pass, draw.mesh_id);
            }
        }
    }

    fn draws_ldr(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(levels: ContourLevels) -> SurfaceContourItem {
        SurfaceContourItem::new(MeshId::from_index(0), [[0.0; 4]; 4], "f", levels)
    }

    #[test]
    fn uniform_matches_the_shader_layout() {
        // model 64, colour 16, eight scalars 32, 32 levels 128.
        assert_eq!(std::mem::size_of::<ContourUniform>(), 240);
        assert_eq!(std::mem::offset_of!(ContourUniform, levels), 112);
    }

    #[test]
    fn spacing_is_the_smallest_gap_between_levels() {
        let block = ContourUniform::new(&item(ContourLevels::Values(vec![3.0, 1.0, 1.5])), 100.0)
            .unwrap();
        assert_eq!(block.count, 3);
        assert_eq!(block.spacing, 0.5);
        let lone = ContourUniform::new(&item(ContourLevels::Values(vec![-4.0])), 100.0).unwrap();
        assert_eq!(lone.spacing, 4.0);
    }

    #[test]
    fn levels_that_draw_nothing_are_skipped() {
        for levels in [
            ContourLevels::Values(vec![]),
            ContourLevels::Values(vec![f32::NAN]),
            ContourLevels::Spaced {
                origin: 0.0,
                interval: 0.0,
            },
            ContourLevels::Spaced {
                origin: 0.0,
                interval: -1.0,
            },
        ] {
            assert!(ContourUniform::new(&item(levels), 100.0).is_none());
        }
    }

    #[test]
    fn long_lists_are_truncated() {
        let values = (0..40).map(|i| i as f32).collect();
        let block = ContourUniform::new(&item(ContourLevels::Values(values)), 100.0).unwrap();
        assert_eq!(block.count as usize, MAX_CONTOUR_LEVELS);
        assert_eq!(block.levels[MAX_CONTOUR_LEVELS - 1], 31.0);
    }
}
