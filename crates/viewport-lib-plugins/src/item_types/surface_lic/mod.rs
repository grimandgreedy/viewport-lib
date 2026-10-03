//! Surface line integral convolution as an [`ItemTypePlugin`]: streaks drawn
//! along a vector field on a mesh. Consumers submit [`SurfaceLicItem`]s with
//! `SceneFrame::items_mut`, beside the surface item that draws the mesh.
//!
//! Both passes are encoded from [`ItemTypePlugin::encode`] at
//! [`EncoderScope::OnOpaqueSurfaces`]: each flow surface into a vector target
//! the plugin owns, depth-tested against the scene, then one fullscreen draw
//! that advects noise along those vectors and multiplies the scene colour by
//! the result.

use std::collections::HashMap;

use viewport_lib::ItemSettings;
use viewport_lib::plugin_api::{
    EncoderScope, EncoderScopeContext, ItemCollections, ItemFrameContext, ItemTypePlugin,
    PluginItem, builders,
};
use viewport_lib::resources::mesh::mesh_store::MeshId;

use crate::item_types::shader::{scene_shader, wgsl_source};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.surface_lic";

/// Highest strength the vector target's blue channel carries. The vector pass
/// writes `strength / STRENGTH_MAX` and the advect pass scales it back, so
/// strengths above this clamp. Matches `STRENGTH_MAX` in the advect shader.
const STRENGTH_MAX: f32 = 4.0;

/// The vector target: packed screen direction, strength, coverage.
const VECTOR_FORMAT: viewport_lib::gpu::TextureFormat =
    viewport_lib::gpu::TextureFormat::Rgba8Unorm;

fn vector_source() -> String {
    scene_shader(&[], wgsl_source!("surface_lic_vector"))
}

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    vec![
        ("surface_lic_vector.wgsl", vector_source()),
        (
            "surface_lic_advect.wgsl",
            wgsl_source!("surface_lic_advect").to_string(),
        ),
    ]
}

/// Advection settings for surface LIC.
///
/// The noise is one independent value per screen pixel, and the kernel reaches
/// `steps * step_size` pixels each way along the flow. Longer kernels give
/// smoother streaks; shorter ones give more contrast for less GPU time.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceLicConfig {
    /// Advection steps taken in each direction from every pixel. Default: 20.
    pub steps: u32,
    /// Distance advanced per step, in screen pixels. Default: 1.5.
    pub step_size: f32,
    /// How strongly the streaks modulate the surface colour. At 0 there is no
    /// effect; at 1 the colour ranges from black to twice as bright. Values up
    /// to 4 raise the contrast further. Default: 1.0.
    pub strength: f32,
}

impl Default for SurfaceLicConfig {
    fn default() -> Self {
        Self {
            steps: 20,
            step_size: 1.5,
            strength: 1.0,
        }
    }
}

/// Flow streaks on one mesh.
///
/// The item draws no surface of its own: it modulates whatever colour is on
/// screen where the mesh is the visible surface. Submit it beside the
/// `SceneRenderItem` that draws the same mesh, with the same `model`.
///
/// `strength` is per item. `steps` and `step_size` are read from the first
/// visible item of the frame, because every flow surface is advected in one
/// pass.
///
/// Only the HDR render path draws it.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SurfaceLicItem {
    /// The mesh the flow lies on.
    pub mesh_id: MeshId,
    /// Local to world transform. Applied to the positions and the vectors.
    pub model: [[f32; 4]; 4],
    /// Name of the `AttributeData::VertexVector` attribute the mesh was
    /// uploaded with. An item naming an attribute the mesh lacks draws nothing.
    pub vector_attribute: String,
    /// Advection settings.
    pub config: SurfaceLicConfig,
    /// Shared per-item settings. Only `hidden` is read.
    pub settings: ItemSettings,
}

impl SurfaceLicItem {
    /// Streaks along `vector_attribute` on `mesh_id`, with the default config.
    pub fn new(mesh_id: MeshId, model: [[f32; 4]; 4], vector_attribute: impl Into<String>) -> Self {
        Self {
            mesh_id,
            model,
            vector_attribute: vector_attribute.into(),
            config: SurfaceLicConfig::default(),
            settings: ItemSettings::default(),
        }
    }
}

impl PluginItem for SurfaceLicItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &ItemSettings {
        &self.settings
    }
}

/// One item's instance record: the model matrix, then the normalised strength
/// in the first lane of a vec4.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LicInstance {
    model: [[f32; 4]; 4],
    params: [f32; 4],
}

const INSTANCE_ATTRIBUTES: [viewport_lib::gpu::VertexAttribute; 5] = viewport_lib::gpu::vertex_attr_array![
    2 => Float32x4,
    3 => Float32x4,
    4 => Float32x4,
    5 => Float32x4,
    6 => Float32x4,
];

/// Uniform of the advect pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AdvectParams {
    steps: u32,
    step_size: f32,
    _pad: [f32; 2],
}

/// Members of [`LicPipelines`].
const VECTOR: usize = 0;
const ADVECT: usize = 1;

/// What a pipeline build reads.
struct LicRecipe {
    device: viewport_lib::gpu::Device,
    vector_layout: viewport_lib::gpu::PipelineLayout,
    vector_shader: viewport_lib::gpu::ShaderModule,
    advect_layout: viewport_lib::gpu::PipelineLayout,
    advect_shader: viewport_lib::gpu::ShaderModule,
}

/// The vector pass and the advect pass, each built the first time a frame
/// draws streaks.
type LicPipelines = viewport_lib::plugin_api::LazyPipelines<LicRecipe, 2>;

fn build(r: &LicRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    if i == ADVECT {
        // src * dst + dst * src: the scene colour times twice the modulation.
        // Alpha keeps what the scene wrote.
        let modulate = viewport_lib::gpu::BlendState {
            color: viewport_lib::gpu::BlendComponent {
                src_factor: viewport_lib::gpu::BlendFactor::Dst,
                dst_factor: viewport_lib::gpu::BlendFactor::Src,
                operation: viewport_lib::gpu::BlendOperation::Add,
            },
            alpha: viewport_lib::gpu::BlendComponent {
                src_factor: viewport_lib::gpu::BlendFactor::Zero,
                dst_factor: viewport_lib::gpu::BlendFactor::One,
                operation: viewport_lib::gpu::BlendOperation::Add,
            },
        };
        return builders::build_fullscreen_pipeline(
            &r.device,
            "surface_lic_advect_pipeline",
            &r.advect_layout,
            &r.advect_shader,
            viewport_lib::resources::HDR_COLOR_FORMAT,
            Some(modulate),
        );
    }
    // Buffer 0 is the shared mesh vertex, read for its position alone.
    let position_layout = viewport_lib::gpu::VertexBufferLayout {
        array_stride: builders::mesh_vertex_layout().array_stride,
        step_mode: viewport_lib::gpu::VertexStepMode::Vertex,
        attributes: &viewport_lib::gpu::vertex_attr_array![0 => Float32x3],
    };
    let instance_layout = viewport_lib::gpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<LicInstance>() as u64,
        step_mode: viewport_lib::gpu::VertexStepMode::Instance,
        attributes: &INSTANCE_ATTRIBUTES,
    };
    // Tested against the scene depth so a flow surface writes vectors only
    // where it is the visible one. The bias absorbs the rounding between
    // this draw and the colour draw of the same triangles.
    let mut depth =
        builders::scene_depth_stencil(false, viewport_lib::gpu::CompareFunction::LessEqual);
    depth.bias.constant = -2;
    builders::render_pipeline(
        &r.device,
        builders::RenderPipelineDesc {
            label: "surface_lic_vector_pipeline",
            layout: &r.vector_layout,
            vertex_module: &r.vector_shader,
            vertex_entry: "vs_main",
            vertex_buffers: &[
                position_layout,
                builders::vector_attribute_layout(1),
                instance_layout,
            ],
            fragment: Some(viewport_lib::gpu::FragmentState {
                module: &r.vector_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(viewport_lib::gpu::ColorTargetState {
                    format: VECTOR_FORMAT,
                    blend: None,
                    write_mask: viewport_lib::gpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: viewport_lib::gpu::PrimitiveState {
                topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth),
            multisample: viewport_lib::gpu::MultisampleState::default(),
            cache: None,
        },
    )
}

/// Pipelines and the resources every viewport shares.
struct LicGpu {
    pipelines: LicPipelines,
    advect_bgl: viewport_lib::gpu::BindGroupLayout,
    sampler: viewport_lib::gpu::Sampler,
    params_buf: viewport_lib::gpu::Buffer,
}

impl LicGpu {
    fn new(
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::resources::DeviceResources,
    ) -> Self {
        let vector_shader = builders::wgsl_module(device, "surface_lic_vector", &vector_source());
        let vector_layout = builders::pipeline_layout(
            device,
            "surface_lic_vector_layout",
            &[resources.shared_bindings().group0_layout],
        );

        let advect_shader = builders::wgsl_module(
            device,
            "surface_lic_advect",
            wgsl_source!("surface_lic_advect"),
        );
        let fragment = viewport_lib::gpu::ShaderStages::FRAGMENT;
        let advect_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("surface_lic_advect_bgl"),
                entries: &[
                    builders::uniform_entry(0, fragment),
                    builders::texture_entry(1, fragment),
                    builders::texture_entry(2, fragment),
                    builders::sampler_entry(3, fragment),
                ],
            });
        let advect_layout =
            builders::pipeline_layout(device, "surface_lic_advect_layout", &[&advect_bgl]);
        let pipelines = resources.lazy_pipelines(
            LicRecipe {
                device: device.clone(),
                vector_layout,
                vector_shader,
                advect_layout,
                advect_shader,
            },
            build,
        );

        let params_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("surface_lic_advect_params"),
            size: std::mem::size_of::<AdvectParams>() as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipelines,
            advect_bgl,
            sampler: builders::clamp_linear_sampler(device, "surface_lic_sampler"),
            params_buf,
        }
    }
}

/// One viewport's vector target and noise, at the scene size they were built
/// for.
struct LicTargets {
    size: [u32; 2],
    vector_view: viewport_lib::gpu::TextureView,
    bind_group: viewport_lib::gpu::BindGroup,
    _vector_tex: viewport_lib::gpu::Texture,
    _noise_tex: viewport_lib::gpu::Texture,
}

impl LicTargets {
    fn new(
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        gpu: &LicGpu,
        size: [u32; 2],
    ) -> Self {
        let extent = viewport_lib::gpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let texture = |label, format, usage| {
            device.create_texture(&viewport_lib::gpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: viewport_lib::gpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let vector_tex = texture(
            "surface_lic_vector",
            VECTOR_FORMAT,
            viewport_lib::gpu::TextureUsages::RENDER_ATTACHMENT
                | viewport_lib::gpu::TextureUsages::TEXTURE_BINDING,
        );
        let noise_tex = texture(
            "surface_lic_noise",
            viewport_lib::gpu::TextureFormat::R8Unorm,
            viewport_lib::gpu::TextureUsages::TEXTURE_BINDING
                | viewport_lib::gpu::TextureUsages::COPY_DST,
        );
        queue.write_texture(
            viewport_lib::gpu::TexelCopyTextureInfo {
                texture: &noise_tex,
                mip_level: 0,
                origin: viewport_lib::gpu::Origin3d::ZERO,
                aspect: viewport_lib::gpu::TextureAspect::All,
            },
            &white_noise(size[0] * size[1]),
            viewport_lib::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size[0]),
                rows_per_image: Some(size[1]),
            },
            extent,
        );

        let vector_view = vector_tex.create_view(&Default::default());
        let noise_view = noise_tex.create_view(&Default::default());
        let bind_group = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("surface_lic_advect_bg"),
            layout: &gpu.advect_bgl,
            entries: &[
                viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.params_buf.as_entire_binding(),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_lib::gpu::BindingResource::TextureView(&vector_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 2,
                    resource: viewport_lib::gpu::BindingResource::TextureView(&noise_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 3,
                    resource: viewport_lib::gpu::BindingResource::Sampler(&gpu.sampler),
                },
            ],
        });
        Self {
            size,
            vector_view,
            bind_group,
            _vector_tex: vector_tex,
            _noise_tex: noise_tex,
        }
    }
}

/// `count` bytes of white noise, the same for every run: a xorshift mix of
/// the pixel index.
fn white_noise(count: u32) -> Vec<u8> {
    (0..count)
        .map(|i| {
            let mut v = i.wrapping_add(1).wrapping_mul(2246822519);
            v ^= v >> 13;
            v ^= v << 17;
            v ^= v >> 5;
            v as u8
        })
        .collect()
}

/// One flow surface to draw this frame.
struct LicDraw {
    mesh_id: MeshId,
    vector_attribute: String,
    /// Index of the item's record in the instance buffer.
    instance: u32,
}

/// The surface LIC item type. Register it with
/// [`ViewportRenderer::with_item_type_plugin`](viewport_lib::renderer::ViewportRenderer::with_item_type_plugin),
/// or through [`install`](crate::item_types::install) with the other item types. It
/// belongs after the decal type, so a decal on a flow surface takes the
/// streaks too.
#[derive(Default)]
pub struct SurfaceLicPlugin {
    gpu: Option<LicGpu>,
    /// Every visible item's instance record, and how many the buffer holds.
    instances: Option<(viewport_lib::gpu::Buffer, usize)>,
    /// This frame's draw list, built in `prepare`.
    draws: Vec<LicDraw>,
    /// Vector target and noise per viewport, rebuilt when the scene size
    /// changes. Behind a lock because `encode` runs from a shared borrow.
    targets: std::sync::Mutex<HashMap<usize, LicTargets>>,
}

impl ItemTypePlugin for SurfaceLicPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn warm(
        &mut self,
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::resources::DeviceResources,
    ) {
        self.gpu
            .get_or_insert_with(|| LicGpu::new(device, resources))
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
        let visible: Vec<&SurfaceLicItem> = items
            .of::<SurfaceLicItem>()
            .iter()
            .filter(|item| !item.settings.hidden && !item.vector_attribute.is_empty())
            .collect();
        let Some(first) = visible.first() else {
            // Nothing to draw: give the per-viewport targets back.
            self.targets.lock().unwrap().clear();
            return Vec::new();
        };

        let gpu = self
            .gpu
            .get_or_insert_with(|| LicGpu::new(device, ctx.resources));
        let params = AdvectParams {
            steps: first.config.steps,
            step_size: first.config.step_size,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&gpu.params_buf, 0, bytemuck::bytes_of(&params));

        let records: Vec<LicInstance> = visible
            .iter()
            .map(|item| LicInstance {
                model: item.model,
                params: [
                    (item.config.strength.max(0.0) / STRENGTH_MAX).min(1.0),
                    0.0,
                    0.0,
                    0.0,
                ],
            })
            .collect();
        if self
            .instances
            .as_ref()
            .is_none_or(|(_, capacity)| *capacity < records.len())
        {
            let capacity = records.len().next_power_of_two();
            let buffer = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                label: Some("surface_lic_instances"),
                size: (capacity * std::mem::size_of::<LicInstance>()) as u64,
                usage: viewport_lib::gpu::BufferUsages::VERTEX
                    | viewport_lib::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances = Some((buffer, capacity));
        }
        if let Some((buffer, _)) = &self.instances {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&records));
        }

        self.draws
            .extend(visible.iter().enumerate().map(|(i, item)| LicDraw {
                mesh_id: item.mesh_id,
                vector_attribute: item.vector_attribute.clone(),
                instance: i as u32,
            }));
        Vec::new()
    }

    fn encoder_scopes(&self) -> &[EncoderScope] {
        // The streaks are part of how the surface looks, so they sit under
        // transparency and the selection affordances.
        &[EncoderScope::OnOpaqueSurfaces]
    }

    fn encode(
        &self,
        encoder: &mut viewport_lib::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let (Some(gpu), Some((instances, _))) = (self.gpu.as_ref(), self.instances.as_ref()) else {
            return;
        };
        if self.draws.is_empty() || ctx.scene_size[0] == 0 || ctx.scene_size[1] == 0 {
            return;
        }
        // Still compiling: the streaks draw from the frame both are ready.
        let (Some(vector_pipeline), Some(advect_pipeline)) =
            (gpu.pipelines.get(VECTOR), gpu.pipelines.get(ADVECT))
        else {
            return;
        };

        let mut targets = self.targets.lock().unwrap();
        if targets
            .get(&ctx.viewport_index)
            .is_none_or(|t| t.size != ctx.scene_size)
        {
            targets.insert(
                ctx.viewport_index,
                LicTargets::new(ctx.device, ctx.queue, gpu, ctx.scene_size),
            );
        }
        let targets = &targets[&ctx.viewport_index];

        let mut drew = false;
        {
            let mut pass = encoder.begin_render_pass(&viewport_lib::gpu::RenderPassDescriptor {
                #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
                multiview_mask: None,
                label: Some("surface_lic_vector_pass"),
                color_attachments: &[Some(viewport_lib::gpu::RenderPassColorAttachment {
                    view: &targets.vector_view,
                    resolve_target: None,
                    ops: viewport_lib::gpu::Operations {
                        load: viewport_lib::gpu::LoadOp::Clear(
                            viewport_lib::gpu::Color::TRANSPARENT,
                        ),
                        store: viewport_lib::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(
                    viewport_lib::gpu::RenderPassDepthStencilAttachment {
                        view: ctx.scene_depth,
                        depth_ops: Some(viewport_lib::gpu::Operations {
                            load: viewport_lib::gpu::LoadOp::Load,
                            store: viewport_lib::gpu::StoreOp::Store,
                        }),
                        stencil_ops: Some(viewport_lib::gpu::Operations {
                            load: viewport_lib::gpu::LoadOp::Load,
                            store: viewport_lib::gpu::StoreOp::Store,
                        }),
                    },
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(vector_pipeline);
            pass.set_bind_group(0, ctx.camera_bind_group, &[]);
            pass.set_vertex_buffer(2, instances.slice(..));
            for draw in &self.draws {
                if ctx.meshes.bind_vector_attribute(
                    &mut pass,
                    1,
                    draw.mesh_id,
                    &draw.vector_attribute,
                ) {
                    drew |= ctx.meshes.draw_indexed_instance_range(
                        &mut pass,
                        draw.mesh_id,
                        draw.instance..draw.instance + 1,
                    );
                }
            }
        }
        // Every item named a missing mesh or attribute: nothing to advect.
        if !drew {
            return;
        }

        let mut pass = encoder.begin_render_pass(&viewport_lib::gpu::RenderPassDescriptor {
            #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
            multiview_mask: None,
            label: Some("surface_lic_advect_pass"),
            color_attachments: &[Some(viewport_lib::gpu::RenderPassColorAttachment {
                view: ctx.scene_colour,
                resolve_target: None,
                ops: viewport_lib::gpu::Operations {
                    load: viewport_lib::gpu::LoadOp::Load,
                    store: viewport_lib::gpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(advect_pipeline);
        pass.set_bind_group(0, &targets.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_repeatable_and_spread() {
        let a = white_noise(4096);
        assert_eq!(a, white_noise(4096));
        let mean = a.iter().map(|&v| v as f32).sum::<f32>() / a.len() as f32;
        assert!((mean - 127.5).abs() < 8.0, "mean {mean}");
    }

    #[test]
    fn instance_record_matches_its_layout() {
        assert_eq!(std::mem::size_of::<LicInstance>(), 80);
        assert_eq!(INSTANCE_ATTRIBUTES[4].offset, 64);
    }
}
