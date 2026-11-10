//! The 2D lighting plugin: the per-frame light list and edge stream, and the
//! two passes it encodes after the scene.

use super::*;
use viewport_lib::plugin_api::{
    EncoderScope, EncoderScopeContext, ItemCollections, ItemFrameContext, ItemTypePlugin,
    PluginItem,
};

/// This type's shaders, as the pipelines compile them.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::item_types::shader::wgsl_source;
    vec![
        ("lighting_2d.wgsl", wgsl_source!("lighting_2d").to_string()),
        (
            "lighting_2d_compose.wgsl",
            wgsl_source!("lighting_2d_compose").to_string(),
        ),
    ]
}

impl PluginItem for Lighting2dItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

/// What the last frame's light layer drew.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lighting2dStats {
    /// Lights drawn.
    pub lights: u32,
    /// Occluder edges each light's shadow pass drew.
    pub edges: u32,
    /// Render passes encoded: two with lights, none without.
    pub passes: u32,
}

/// One light's block, matching `LightBlock` in `lighting_2d.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightBlock {
    view_proj: [[f32; 4]; 4],
    origin: [f32; 4],
    x_axis: [f32; 4],
    y_axis: [f32; 4],
    light: [f32; 4],
    colour: [f32; 4],
}

/// Distance between consecutive light blocks: the largest uniform offset
/// alignment any backend asks for.
const BLOCK_STRIDE: u64 = 256;

/// One occluder edge in the vertex stream, matching `EdgeIn`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEdge {
    ends: [f32; 4],
    one_sided: u32,
}

/// The format of the light layer: the scene's HDR colour format, so a light
/// can be brighter than one.
const LIGHT_FORMAT: viewport_lib::gpu::TextureFormat = viewport_lib::resources::HDR_COLOR_FORMAT;
/// The stencil the shadows stamp. The depth half is unused.
const STENCIL_FORMAT: viewport_lib::gpu::TextureFormat =
    viewport_lib::gpu::TextureFormat::Depth24PlusStencil8;

/// A light projected onto the plane.
#[derive(Clone, Copy, Debug)]
struct PlaneLight {
    position: Vec2,
    radius: f32,
    colour: [f32; 3],
}

/// What the frame's light layer draws, built in `prepare`.
#[derive(Default)]
struct FrameLayer {
    lights: Vec<PlaneLight>,
    edge_count: u32,
    ambient: [f32; 3],
    plane_origin: glam::DVec3,
    x_axis: glam::Vec3,
    y_axis: glam::Vec3,
}

struct Pipelines {
    light_bgl: viewport_lib::gpu::BindGroupLayout,
    compose_bgl: viewport_lib::gpu::BindGroupLayout,
    shadow: viewport_lib::gpu::RenderPipeline,
    light: viewport_lib::gpu::RenderPipeline,
    compose: viewport_lib::gpu::RenderPipeline,
}

/// A viewport's light layer, stencil and light blocks, reallocated when the
/// scene size or the light count outgrows them.
struct ViewportLayer {
    size: [u32; 2],
    light_view: viewport_lib::gpu::TextureView,
    stencil_view: viewport_lib::gpu::TextureView,
    blocks: viewport_lib::gpu::Buffer,
    block_capacity: u32,
    light_bg: viewport_lib::gpu::BindGroup,
    compose_bg: viewport_lib::gpu::BindGroup,
}

/// Draws [`Lighting2dItem`]s. Register it with
/// `renderer.with_item_type_plugin(device, Box::new(Lighting2dPlugin::default()))`
/// or through `item_types::install`.
#[derive(Default)]
pub struct Lighting2dPlugin {
    pipelines: Option<Pipelines>,
    frame: FrameLayer,
    edges: Option<viewport_lib::gpu::Buffer>,
    edge_capacity: u32,
    viewports: std::sync::Mutex<Vec<Option<ViewportLayer>>>,
    stats: std::sync::Mutex<Lighting2dStats>,
    /// Draws encoded since the last prepare, across every viewport: each
    /// light's shadow edges and quad, and the compose.
    issued: std::sync::atomic::AtomicU32,
}

const SCOPES: &[EncoderScope] = &[EncoderScope::AfterTransparent];

impl Lighting2dPlugin {
    /// What the light layer drew in the last frame that submitted a
    /// [`Lighting2dItem`]: zero lights and zero passes when it had no lights.
    pub fn last_stats(&self) -> Lighting2dStats {
        *self.stats.lock().unwrap()
    }

    fn pipelines(device: &viewport_lib::gpu::Device) -> Pipelines {
        use viewport_lib::gpu;
        use viewport_lib::plugin_api::builders;
        let light_bgl =
            device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
                label: Some("lighting_2d_light_bgl"),
                entries: &[gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: gpu::BufferSize::new(
                            std::mem::size_of::<LightBlock>() as u64
                        ),
                    },
                    count: None,
                }],
            });
        let compose_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("lighting_2d_compose_bgl"),
            entries: &[gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: gpu::ShaderStages::FRAGMENT,
                ty: gpu::BindingType::Texture {
                    sample_type: gpu::TextureSampleType::Float { filterable: false },
                    view_dimension: gpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let shader = builders::wgsl_module(
            device,
            "lighting_2d_shader",
            crate::item_types::shader::wgsl_source!("lighting_2d"),
        );
        let compose_shader = builders::wgsl_module(
            device,
            "lighting_2d_compose_shader",
            crate::item_types::shader::wgsl_source!("lighting_2d_compose"),
        );
        let layout = builders::pipeline_layout(device, "lighting_2d_layout", &[&light_bgl]);
        let compose_layout =
            builders::pipeline_layout(device, "lighting_2d_compose_layout", &[&compose_bgl]);

        let stencil = |compare, pass_op| {
            let face = gpu::StencilFaceState {
                compare,
                fail_op: gpu::StencilOperation::Keep,
                depth_fail_op: gpu::StencilOperation::Keep,
                pass_op,
            };
            let mut state =
                builders::depth_stencil(STENCIL_FORMAT, false, gpu::CompareFunction::Always);
            state.stencil = gpu::StencilState {
                front: face,
                back: face,
                read_mask: 0xff,
                write_mask: 0xff,
            };
            state
        };
        let edge_attrs = [
            gpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: gpu::VertexFormat::Float32x4,
            },
            gpu::VertexAttribute {
                offset: 16,
                shader_location: 1,
                format: gpu::VertexFormat::Uint32,
            },
        ];
        let edge_layout = [gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuEdge>() as u64,
            step_mode: gpu::VertexStepMode::Instance,
            attributes: &edge_attrs,
        }];
        let pipeline = |label,
                        layout: &gpu::PipelineLayout,
                        module: &gpu::ShaderModule,
                        vs,
                        fs,
                        buffers: &[gpu::VertexBufferLayout<'_>],
                        target: gpu::ColorTargetState,
                        depth_stencil| {
            builders::render_pipeline(
                device,
                builders::RenderPipelineDesc {
                    label,
                    layout,
                    vertex_module: module,
                    vertex_entry: vs,
                    vertex_buffers: buffers,
                    fragment: Some(gpu::FragmentState {
                        module,
                        entry_point: Some(fs),
                        targets: &[Some(target)],
                        compilation_options: gpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: gpu::PrimitiveState {
                        topology: gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil,
                    multisample: gpu::MultisampleState::default(),
                    cache: None,
                },
            )
        };
        // The shadows touch only the stencil: the light's index where an edge
        // casts.
        let shadow = pipeline(
            "lighting_2d_shadow_pipeline",
            &layout,
            &shader,
            "vs_shadow",
            "fs_shadow",
            &edge_layout,
            gpu::ColorTargetState {
                format: LIGHT_FORMAT,
                blend: None,
                write_mask: gpu::ColorWrites::empty(),
            },
            Some(stencil(
                gpu::CompareFunction::Always,
                gpu::StencilOperation::Replace,
            )),
        );
        // The light adds where the stencil does not hold its index.
        let light = pipeline(
            "lighting_2d_light_pipeline",
            &layout,
            &shader,
            "vs_light",
            "fs_light",
            &[],
            gpu::ColorTargetState {
                format: LIGHT_FORMAT,
                blend: Some(builders::ADDITIVE_BLEND),
                write_mask: gpu::ColorWrites::COLOR,
            },
            Some(stencil(
                gpu::CompareFunction::NotEqual,
                gpu::StencilOperation::Keep,
            )),
        );
        // `dst * src`: the layer multiplied over the scene, alpha untouched.
        let multiply = gpu::BlendState {
            color: gpu::BlendComponent {
                src_factor: gpu::BlendFactor::Zero,
                dst_factor: gpu::BlendFactor::Src,
                operation: gpu::BlendOperation::Add,
            },
            alpha: gpu::BlendComponent {
                src_factor: gpu::BlendFactor::Zero,
                dst_factor: gpu::BlendFactor::One,
                operation: gpu::BlendOperation::Add,
            },
        };
        let compose = pipeline(
            "lighting_2d_compose_pipeline",
            &compose_layout,
            &compose_shader,
            "vs_main",
            "fs_main",
            &[],
            gpu::ColorTargetState {
                format: LIGHT_FORMAT,
                blend: Some(multiply),
                write_mask: gpu::ColorWrites::ALL,
            },
            None,
        );
        Pipelines {
            light_bgl,
            compose_bgl,
            shadow,
            light,
            compose,
        }
    }

    /// The viewport's layer, allocated or reallocated for `size` and `lights`.
    fn viewport_layer<'a>(
        &self,
        cache: &'a mut Vec<Option<ViewportLayer>>,
        device: &viewport_lib::gpu::Device,
        pipelines: &Pipelines,
        index: usize,
        size: [u32; 2],
        lights: u32,
    ) -> &'a ViewportLayer {
        use viewport_lib::gpu;
        while cache.len() <= index {
            cache.push(None);
        }
        let stale = cache[index]
            .as_ref()
            .is_none_or(|l| l.size != size || l.block_capacity < lights);
        if stale {
            let texture = |label, format, usage| {
                device.create_texture(&gpu::TextureDescriptor {
                    label: Some(label),
                    size: gpu::Extent3d {
                        width: size[0].max(1),
                        height: size[1].max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: gpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            let light_view = texture(
                "lighting_2d_layer",
                LIGHT_FORMAT,
                gpu::TextureUsages::RENDER_ATTACHMENT | gpu::TextureUsages::TEXTURE_BINDING,
            )
            .create_view(&gpu::TextureViewDescriptor::default());
            let stencil_view = texture(
                "lighting_2d_stencil",
                STENCIL_FORMAT,
                gpu::TextureUsages::RENDER_ATTACHMENT,
            )
            .create_view(&gpu::TextureViewDescriptor::default());
            let block_capacity = lights.next_power_of_two().max(8);
            let blocks = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("lighting_2d_blocks"),
                size: BLOCK_STRIDE * block_capacity as u64,
                usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let light_bg = device.create_bind_group(&gpu::BindGroupDescriptor {
                label: Some("lighting_2d_light_bg"),
                layout: &pipelines.light_bgl,
                entries: &[gpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu::BindingResource::Buffer(gpu::BufferBinding {
                        buffer: &blocks,
                        offset: 0,
                        size: gpu::BufferSize::new(std::mem::size_of::<LightBlock>() as u64),
                    }),
                }],
            });
            let compose_bg = device.create_bind_group(&gpu::BindGroupDescriptor {
                label: Some("lighting_2d_compose_bg"),
                layout: &pipelines.compose_bgl,
                entries: &[gpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu::BindingResource::TextureView(&light_view),
                }],
            });
            cache[index] = Some(ViewportLayer {
                size,
                light_view,
                stencil_view,
                blocks,
                block_capacity,
                light_bg,
                compose_bg,
            });
        }
        cache[index].as_ref().expect("just allocated")
    }
}

/// The frame's point lights on the plane, in the order the frame lists them,
/// at most `max`.
fn plane_lights(
    lights: &[viewport_lib::LightSource],
    origin: glam::DVec3,
    x_axis: glam::Vec3,
    y_axis: glam::Vec3,
    intensity: f32,
    max: u32,
) -> Vec<PlaneLight> {
    lights
        .iter()
        .filter_map(|light| match light.kind {
            viewport_lib::LightKind::Point {
                position, range, ..
            } if range > 0.0 => {
                let rel = position - origin;
                let [r, g, b, _] = light.colour.to_linear_rgba();
                let k = light.intensity * intensity;
                Some(PlaneLight {
                    position: Vec2::new(
                        rel.dot(x_axis.as_dvec3()) as f32,
                        rel.dot(y_axis.as_dvec3()) as f32,
                    ),
                    radius: range,
                    colour: [r * k, g * k, b * k],
                })
            }
            _ => None,
        })
        .take(max.min(MAX_LIGHTS_2D) as usize)
        .collect()
}

impl ItemTypePlugin for Lighting2dPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    /// The full tier: the layer draws after the scene on the HDR path, which
    /// only the full tier has.
    fn requires(&self) -> viewport_lib::Requirements {
        viewport_lib::Requirements::full_tier()
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        self.frame = FrameLayer::default();
        *self.stats.get_mut().unwrap() = Lighting2dStats::default();
        *self.issued.get_mut() = 0;
        let items: Vec<&Lighting2dItem> = items
            .of::<Lighting2dItem>()
            .iter()
            .filter(|i| !i.settings.hidden)
            .collect();
        let Some(first) = items.first() else {
            return Vec::new();
        };
        let plane_origin = first
            .plane
            .to_world(glam::DVec2::ZERO, first.elevation as f64);
        let (x_axis, y_axis) = (first.plane.x_axis(), first.plane.y_axis());
        let lights = plane_lights(
            ctx.lights,
            plane_origin,
            x_axis,
            y_axis,
            first.intensity,
            first.max_lights,
        );
        if lights.is_empty() {
            return Vec::new();
        }
        if self.pipelines.is_none() {
            self.pipelines = Some(Self::pipelines(device));
        }

        let mut edges = Vec::new();
        if first.shadows {
            for item in &items {
                for occluder in &item.occluders {
                    occluder.edges(&mut edges);
                }
            }
        }
        let gpu_edges: Vec<GpuEdge> = edges
            .iter()
            .map(|e| GpuEdge {
                ends: [e.a.x, e.a.y, e.b.x, e.b.y],
                one_sided: e.one_sided as u32,
            })
            .collect();
        if !gpu_edges.is_empty() {
            let count = gpu_edges.len() as u32;
            if self.edges.is_none() || self.edge_capacity < count {
                self.edge_capacity = count.next_power_of_two().max(64);
                self.edges = Some(device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                    label: Some("lighting_2d_edges"),
                    size: self.edge_capacity as u64 * std::mem::size_of::<GpuEdge>() as u64,
                    usage: viewport_lib::gpu::BufferUsages::VERTEX
                        | viewport_lib::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            if let Some(buf) = &self.edges {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&gpu_edges));
            }
        }

        let [r, g, b, _] = first.ambient.to_linear_rgba();
        self.frame = FrameLayer {
            lights,
            edge_count: gpu_edges.len() as u32,
            ambient: [r, g, b],
            plane_origin,
            x_axis,
            y_axis,
        };
        Vec::new()
    }

    fn draw_calls(&self) -> u32 {
        self.issued.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn encoder_scopes(&self) -> &[EncoderScope] {
        if self.frame.lights.is_empty() {
            &[]
        } else {
            SCOPES
        }
    }

    fn encode(
        &self,
        encoder: &mut viewport_lib::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        use viewport_lib::gpu;
        let frame = &self.frame;
        let (Some(pipelines), false) = (&self.pipelines, frame.lights.is_empty()) else {
            *self.stats.lock().unwrap() = Lighting2dStats::default();
            return;
        };
        let count = frame.lights.len() as u32;
        let mut draws = 0u32;
        let mut cache = self.viewports.lock().unwrap();
        let layer = self.viewport_layer(
            &mut cache,
            ctx.device,
            pipelines,
            ctx.viewport_index,
            ctx.scene_size,
            count,
        );

        // Everything relative to the camera eye, so far-off scenes keep their
        // precision.
        let eye = ctx.camera.eye_position();
        let view_proj = ctx.camera.projection()
            * ctx.camera.view()
            * glam::Mat4::from_translation(eye.as_vec3());
        let origin = (frame.plane_origin - eye).as_vec3();
        for (i, light) in frame.lights.iter().enumerate() {
            let block = LightBlock {
                view_proj: view_proj.to_cols_array_2d(),
                origin: origin.extend(0.0).to_array(),
                x_axis: frame.x_axis.extend(0.0).to_array(),
                y_axis: frame.y_axis.extend(0.0).to_array(),
                light: [light.position.x, light.position.y, light.radius, 0.0],
                colour: [light.colour[0], light.colour[1], light.colour[2], 0.0],
            };
            ctx.queue.write_buffer(
                &layer.blocks,
                i as u64 * BLOCK_STRIDE,
                bytemuck::bytes_of(&block),
            );
        }

        {
            let [r, g, b] = frame.ambient;
            let mut pass = encoder.begin_render_pass(&gpu::RenderPassDescriptor {
                #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
                multiview_mask: None,
                label: Some("lighting_2d_lights"),
                color_attachments: &[Some(gpu::RenderPassColorAttachment {
                    view: &layer.light_view,
                    resolve_target: None,
                    ops: gpu::Operations {
                        load: gpu::LoadOp::Clear(gpu::Color {
                            r: r as f64,
                            g: g as f64,
                            b: b as f64,
                            a: 1.0,
                        }),
                        store: gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(gpu::RenderPassDepthStencilAttachment {
                    view: &layer.stencil_view,
                    depth_ops: Some(gpu::Operations {
                        load: gpu::LoadOp::Clear(viewport_lib::plugin_api::depth::DEPTH_CLEAR),
                        store: gpu::StoreOp::Discard,
                    }),
                    stencil_ops: Some(gpu::Operations {
                        load: gpu::LoadOp::Clear(0),
                        store: gpu::StoreOp::Discard,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            for i in 0..count {
                // Each light stamps its own value, so no light's shadow needs
                // clearing before the next.
                pass.set_bind_group(0, &layer.light_bg, &[i * BLOCK_STRIDE as u32]);
                pass.set_stencil_reference(i + 1);
                if let (Some(edges), true) = (&self.edges, frame.edge_count > 0) {
                    pass.set_pipeline(&pipelines.shadow);
                    pass.set_vertex_buffer(0, edges.slice(..));
                    pass.draw(0..6, 0..frame.edge_count);
                    draws += 1;
                }
                pass.set_pipeline(&pipelines.light);
                pass.draw(0..6, 0..1);
                draws += 1;
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&gpu::RenderPassDescriptor {
                #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
                multiview_mask: None,
                label: Some("lighting_2d_compose"),
                color_attachments: &[Some(gpu::RenderPassColorAttachment {
                    view: ctx.scene_colour,
                    resolve_target: None,
                    ops: gpu::Operations {
                        load: gpu::LoadOp::Load,
                        store: gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipelines.compose);
            pass.set_bind_group(0, &layer.compose_bg, &[]);
            pass.draw(0..3, 0..1);
            draws += 1;
        }
        self.issued
            .fetch_add(draws, std::sync::atomic::Ordering::Relaxed);

        *self.stats.lock().unwrap() = Lighting2dStats {
            lights: count,
            edges: frame.edge_count,
            passes: 2,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_lights_project_onto_the_plane_and_the_rest_are_skipped() {
        let lights = vec![
            viewport_lib::LightSource::default(),
            viewport_lib::LightSource::point_candela(
                glam::DVec3::new(3.0, 4.0, 9.0),
                viewport_lib::Candela(2.0),
                10.0,
                0.0,
            ),
        ];
        let out = plane_lights(
            &lights,
            glam::DVec3::ZERO,
            glam::Vec3::X,
            glam::Vec3::Y,
            1.0,
            8,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].position, Vec2::new(3.0, 4.0));
        assert_eq!(out[0].radius, 10.0);
        let capped = plane_lights(
            &lights,
            glam::DVec3::ZERO,
            glam::Vec3::X,
            glam::Vec3::Y,
            1.0,
            0,
        );
        assert!(capped.is_empty());
    }

    #[test]
    fn block_fits_its_stride() {
        assert!(std::mem::size_of::<LightBlock>() as u64 <= BLOCK_STRIDE);
        assert_eq!(std::mem::size_of::<GpuEdge>(), 20);
    }
}
