//! The sprite mask item type: [`SpriteMaskItem`]s drawn into the renderer's
//! painter mask target before the painter sequence, so painter content that
//! tests a slot is cut by them.
//!
//! Each item's quads are an instance-stepped vertex stream and its block a
//! uniform, so masks draw on every tier.

use super::types::{MaskSource, SpriteItem, SpriteMaskItem, SpriteSizeMode};
use super::{MASK_TYPE_NAME, SpriteOrientation};
use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, PainterMask, PainterMaskContext,
    PerViewport, PluginItem,
};

impl PluginItem for SpriteMaskItem {
    const TYPE_NAME: &'static str = MASK_TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

/// The per-quad stream: position, size, rotation, pivot, uv rect, tint and
/// velocity, 72 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaskQuad {
    position: [f32; 3],
    size: [f32; 2],
    rotation: f32,
    pivot: [f32; 2],
    uv_rect: [f32; 4],
    colour: [f32; 4],
    velocity: [f32; 3],
}

const QUAD_ATTRS: [viewport_lib::gpu::VertexAttribute; 7] = viewport_lib::gpu::vertex_attr_array![
    0 => Float32x3,
    1 => Float32x2,
    2 => Float32,
    3 => Float32x2,
    4 => Float32x4,
    5 => Float32x4,
    6 => Float32x3,
];

/// The batch uniform, laid out as `SpriteUniform` in `sprite_common.wgsl`;
/// only the placement fields and `has_texture` are read.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaskSpriteUniform {
    model: [[f32; 4]; 4],
    world_space: u32,
    has_texture: u32,
    soft_particle_distance: f32,
    orientation: u32,
    axis: [f32; 3],
    unused: [u32; 7],
    flags: u32,
    pad: u32,
    mask: [u32; 4],
}

/// `SpriteMaskDraw` in `sprite_mask.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaskDrawUniform {
    mode: u32,
    threshold: f32,
    slot: u32,
    shape: u32,
    params: [f32; 4],
}

/// One mask drawn this frame.
struct MaskDraw {
    slot: u8,
    layers: std::ops::RangeInclusive<u32>,
    quads: viewport_lib::gpu::Buffer,
    count: u32,
    bind_group: viewport_lib::gpu::BindGroup,
}

/// The pipeline and layout, made on the first frame with a mask.
struct MaskGpu {
    bgl: viewport_lib::gpu::BindGroupLayout,
    pipeline: viewport_lib::gpu::RenderPipeline,
}

/// Draws [`SpriteMaskItem`]s into the painter mask target. Registered by
/// `item_types::install` after the sprite type.
#[derive(Default)]
pub struct SpriteMaskPlugin {
    gpu: Option<MaskGpu>,
    views: PerViewport<Vec<MaskDraw>>,
}

/// The shader the mask pipeline compiles, shared sections spliced in.
pub(crate) fn mask_shader_source() -> String {
    crate::item_types::shader::scene_shader(
        &[
            viewport_lib::plugin_api::shared_wgsl::SHARED_PAINTER_MASK_WGSL,
            super::SPRITE_COMMON,
        ],
        crate::item_types::shader::wgsl_source!("sprite_mask"),
    )
}

impl MaskGpu {
    fn new(device: &viewport_lib::gpu::Device, resources: &viewport_lib::DeviceResources) -> Self {
        use viewport_lib::gpu::{
            BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, ShaderStages,
            TextureSampleType, TextureViewDimension,
        };
        let uniform = |binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
            label: Some("sprite_mask_bgl"),
            entries: &[
                uniform(0),
                uniform(1),
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_mask_pipeline_layout",
            &[resources.group0_layout(), &bgl],
        );
        let shader = resources.lazy_module(device, "sprite_mask_shader", mask_shader_source());
        let module = shader.get();
        let buffers = [viewport_lib::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MaskQuad>() as u64,
            step_mode: viewport_lib::gpu::VertexStepMode::Instance,
            attributes: &QUAD_ATTRS,
        }];
        let pipeline = viewport_lib::plugin_api::builders::render_pipeline(
            device,
            viewport_lib::plugin_api::builders::RenderPipelineDesc {
                label: "sprite_mask_pipeline",
                layout: &layout,
                vertex_module: module,
                vertex_entry: "vs_main",
                vertex_buffers: &buffers,
                fragment: Some(viewport_lib::gpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(viewport_lib::gpu::ColorTargetState {
                        format: viewport_lib::resources::PAINTER_MASK_FORMAT,
                        blend: Some(viewport_lib::resources::PAINTER_MASK_BLEND),
                        write_mask: viewport_lib::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: viewport_lib::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: viewport_lib::gpu::MultisampleState::default(),
                cache: None,
            },
        );
        Self { bgl, pipeline }
    }
}

/// The mask block for `source`.
fn draw_block(source: &MaskSource, slot: u8) -> MaskDrawUniform {
    use viewport_lib_types::overlay::OverlayShape;
    let mut block = MaskDrawUniform {
        mode: 0,
        threshold: 0.0,
        slot: (slot & 3) as u32,
        shape: 6,
        params: [0.0; 4],
    };
    match source {
        MaskSource::Alpha { threshold, .. } => block.threshold = *threshold,
        MaskSource::Shape { shape, .. } => {
            block.mode = 1;
            (block.shape, block.params) = match shape {
                OverlayShape::Rect { corner_radius } => (0, [*corner_radius, 0.0, 0.0, 0.0]),
                // Overlay order top-left, top-right, bottom-right, bottom-left.
                OverlayShape::RoundedRect { radii } => {
                    (1, [radii[1], radii[2], radii[0], radii[3]])
                }
                OverlayShape::Circle => (2, [0.0; 4]),
                OverlayShape::Ellipse => (3, [0.0; 4]),
                OverlayShape::Capsule => (4, [0.0; 4]),
                OverlayShape::Ring { inner_radius_frac } => {
                    (5, [*inner_radius_frac, 0.0, 0.0, 0.0])
                }
                _ => (6, [0.0; 4]),
            };
        }
    }
    block
}

/// The sprites a source places its quads by.
fn source_sprite(source: &MaskSource) -> &SpriteItem {
    match source {
        MaskSource::Alpha { sprite, .. } | MaskSource::Shape { sprite, .. } => sprite,
    }
}

impl ItemTypePlugin for SpriteMaskPlugin {
    fn type_name(&self) -> &'static str {
        MASK_TYPE_NAME
    }

    fn requires(&self) -> viewport_lib::Requirements {
        viewport_lib::Requirements::compatibility_tier()
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        _items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        let draws = self.views.get_mut(ctx.viewport_index);
        draws.clear();
        let items = ctx.items_of::<SpriteMaskItem>();
        if items.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| MaskGpu::new(device, ctx.resources));
        let origin = ctx.world_origin();
        for item in items {
            let sprite = source_sprite(&item.source);
            if item.settings.hidden || sprite.positions.is_empty() {
                continue;
            }
            let alpha = matches!(item.source, MaskSource::Alpha { .. });
            let texture = sprite
                .texture_id
                .filter(|_| alpha)
                .and_then(|id| ctx.resources.texture_view(id));
            let quads: Vec<MaskQuad> = (0..sprite.positions.len())
                .map(|i| MaskQuad {
                    position: sprite.positions[i],
                    size: sprite
                        .sizes
                        .get(i)
                        .copied()
                        .unwrap_or([sprite.default_size; 2]),
                    rotation: sprite.rotations.get(i).copied().unwrap_or(0.0),
                    pivot: sprite
                        .pivots
                        .get(i)
                        .copied()
                        .unwrap_or(sprite.default_pivot),
                    uv_rect: sprite.flips.get(i).copied().unwrap_or_default().apply(
                        sprite
                            .uv_rects
                            .get(i)
                            .copied()
                            .unwrap_or([0.0, 0.0, 1.0, 1.0]),
                    ),
                    colour: sprite
                        .colours
                        .get(i)
                        .unwrap_or(&sprite.default_colour)
                        .to_linear_rgba(),
                    velocity: sprite.velocities.get(i).copied().unwrap_or([0.0; 3]),
                })
                .collect();
            let quad_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                label: Some("sprite_mask_quads"),
                size: (quads.len() * std::mem::size_of::<MaskQuad>()) as u64,
                usage: viewport_lib::gpu::BufferUsages::VERTEX
                    | viewport_lib::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&quad_buf, 0, bytemuck::cast_slice(&quads));
            let placement = MaskSpriteUniform {
                model: crate::item_types::origin::rel_model(sprite.model, origin),
                world_space: (sprite.size_mode == SpriteSizeMode::WorldSpace) as u32,
                has_texture: texture.is_some() as u32,
                soft_particle_distance: 0.0,
                orientation: match sprite.orientation {
                    SpriteOrientation::CameraFacing => 0,
                    SpriteOrientation::VelocityStretched => 1,
                    SpriteOrientation::AxisLocked => 2,
                },
                axis: sprite.axis,
                unused: [0; 7],
                flags: sprite.pixel_snap as u32,
                pad: 0,
                mask: [0; 4],
            };
            let block = draw_block(&item.source, item.slot);
            let uniform = |label, bytes: &[u8]| {
                let buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes.len() as u64,
                    usage: viewport_lib::gpu::BufferUsages::UNIFORM
                        | viewport_lib::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, bytes);
                buf
            };
            let placement_buf =
                uniform("sprite_mask_sprite_uniform", bytemuck::bytes_of(&placement));
            let block_buf = uniform("sprite_mask_draw_uniform", bytemuck::bytes_of(&block));
            let view = texture.unwrap_or_else(|| ctx.resources.fallback_colourmap_view());
            let sampler = match sprite.filter {
                viewport_lib::TextureFilter::Linear => ctx.resources.material_sampler().clone(),
                viewport_lib::TextureFilter::Nearest => ctx.resources.sampler_for(
                    device,
                    viewport_lib::SamplerKey {
                        filter: viewport_lib::TextureFilter::Nearest,
                        ..Default::default()
                    },
                ),
            };
            let bind_group = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
                label: Some("sprite_mask_bg"),
                layout: &gpu.bgl,
                entries: &[
                    viewport_lib::gpu::BindGroupEntry {
                        binding: 0,
                        resource: placement_buf.as_entire_binding(),
                    },
                    viewport_lib::gpu::BindGroupEntry {
                        binding: 1,
                        resource: block_buf.as_entire_binding(),
                    },
                    viewport_lib::gpu::BindGroupEntry {
                        binding: 2,
                        resource: viewport_lib::gpu::BindingResource::TextureView(view),
                    },
                    viewport_lib::gpu::BindGroupEntry {
                        binding: 3,
                        resource: viewport_lib::gpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            draws.push(MaskDraw {
                slot: item.slot,
                layers: item.layers.clone(),
                quads: quad_buf,
                count: quads.len() as u32,
                bind_group,
            });
        }
        Vec::new()
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn painter_masks(&self, viewport_index: usize, out: &mut Vec<PainterMask>) {
        if let Some(draws) = self.views.get(viewport_index) {
            out.extend(
                draws
                    .iter()
                    .map(|d| PainterMask::new(d.slot, d.layers.clone())),
            );
        }
    }

    fn paint_painter_masks(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PainterMaskContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let (Some(gpu), Some(draws)) = (&self.gpu, self.views.get(ctx.viewport_index)) else {
            return;
        };
        pass.set_pipeline(&gpu.pipeline);
        for d in draws {
            pass.set_bind_group(1, &d.bind_group, &[]);
            pass.set_vertex_buffer(0, d.quads.slice(..));
            pass.draw(0..6, 0..d.count);
        }
    }
}
