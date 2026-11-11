//! The 2D lighting plugin: each item's light list and edge stream, and the
//! light layers it accumulates before the frame for lit content to sample.

use super::*;
use viewport_lib::plugin_api::{
    ItemCollections, ItemFrameContext, ItemTypePlugin, PainterLightLayers, PerViewport, PluginItem,
};

/// This type's shaders, as the pipelines compile them.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::item_types::shader::wgsl_source;
    vec![("lighting_2d.wgsl", wgsl_source!("lighting_2d").to_string())]
}

impl PluginItem for Lighting2dItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

/// What the last frame's light layers drew.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lighting2dStats {
    /// Light draws, summed over the light layers: a light that lights two
    /// parts of a range counts twice.
    pub lights: u32,
    /// Occluder edges in the shadow streams, summed over the items.
    pub edges: u32,
    /// Render passes encoded before the frame, one per light layer.
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
    params: [f32; 4],
    bounds: [f32; 4],
    flags: [u32; 4],
    points: [[f32; 4]; MAX_POLYGON_LIGHT_POINTS / 2],
}

/// Distance between consecutive light blocks: a multiple of the largest
/// uniform offset alignment any backend asks for.
const BLOCK_STRIDE: u64 = 768;

/// One occluder edge in the vertex stream, matching `EdgeIn`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEdge {
    ends: [f32; 4],
    one_sided: u32,
}

/// The light layer's format where the HDR path exists: the scene's own, so a
/// light can be brighter than one.
const FLOAT_LAYER_FORMAT: viewport_lib::gpu::TextureFormat =
    viewport_lib::resources::HDR_COLOR_FORMAT;
/// The light layer's format on the compatibility tier, which every tier can
/// render to. It holds half the light, and the content sampling it doubles
/// it, so a light can still brighten what it lands on up to twice.
const UNORM_LAYER_FORMAT: viewport_lib::gpu::TextureFormat =
    viewport_lib::gpu::TextureFormat::Rgba8Unorm;
/// The stencil the shadows stamp. The depth half is unused.
const STENCIL_FORMAT: viewport_lib::gpu::TextureFormat =
    viewport_lib::gpu::TextureFormat::Depth24PlusStencil8;

const KIND_POINT: u32 = 0;
const KIND_CONE: u32 = 1;
const KIND_DIRECTIONAL: u32 = 2;
const KIND_GLOBAL: u32 = 3;
const KIND_TEXTURED: u32 = 4;
const KIND_POLYGON: u32 = 5;

/// A light resolved onto the plane, ready to become a block.
#[derive(Clone, Debug)]
struct PlaneLight {
    kind: u32,
    position: Vec2,
    /// How far its shadows reach.
    reach: f32,
    radius: f32,
    colour: [f32; 3],
    falloff: (u32, f32),
    params: [f32; 4],
    /// The plane rectangle the shape covers: centre and half size.
    bounds: [f32; 4],
    points: Vec<[f32; 2]>,
    texture: Option<TextureId>,
    casts: bool,
    layers: RangeInclusive<u32>,
}

impl PlaneLight {
    fn block(&self, frame: &BlockFrame, scale: f32, directional_reach: f32) -> LightBlock {
        let mut points = [[0.0f32; 4]; MAX_POLYGON_LIGHT_POINTS / 2];
        for (i, p) in self.points.iter().enumerate() {
            points[i / 2][(i % 2) * 2] = p[0];
            points[i / 2][(i % 2) * 2 + 1] = p[1];
        }
        let reach = if self.kind == KIND_DIRECTIONAL {
            directional_reach
        } else {
            self.reach
        };
        LightBlock {
            view_proj: frame.view_proj.to_cols_array_2d(),
            origin: frame.origin.extend(0.0).to_array(),
            x_axis: frame.x_axis.extend(0.0).to_array(),
            y_axis: frame.y_axis.extend(0.0).to_array(),
            light: [self.position.x, self.position.y, reach, self.radius],
            colour: [
                self.colour[0] * scale,
                self.colour[1] * scale,
                self.colour[2] * scale,
                self.falloff.1,
            ],
            params: self.params,
            bounds: self.bounds,
            flags: [self.kind, self.falloff.0, self.points.len() as u32, 0],
            points,
        }
    }
}

/// The parts of a light block common to an item's lights.
struct BlockFrame {
    view_proj: glam::Mat4,
    origin: glam::Vec3,
    x_axis: glam::Vec3,
    y_axis: glam::Vec3,
}

fn falloff_code(falloff: Falloff) -> (u32, f32) {
    match falloff {
        Falloff::Linear => (0, 1.0),
        Falloff::Smooth => (1, 1.0),
        Falloff::InverseSquare => (2, 1.0),
        Falloff::Exponent(e) => (3, e),
    }
}

/// `light` on the plane with its brightness times `intensity`, or `None`
/// when it lights nothing: a point, cone or polygon of no size, a texture the
/// renderer does not hold.
fn resolve(
    light: &Light2d,
    intensity: f32,
    has_texture: impl Fn(TextureId) -> bool,
) -> Option<PlaneLight> {
    let [r, g, b, _] = light.colour.to_linear_rgba();
    let k = light.intensity * intensity;
    let position = Vec2::from(light.position);
    let radius = light.radius.max(0.0);
    let around = |reach: f32| [position.x, position.y, reach, reach];
    let mut out = PlaneLight {
        kind: KIND_POINT,
        position,
        reach: radius,
        radius,
        colour: [r * k, g * k, b * k],
        falloff: falloff_code(light.falloff),
        params: [0.0; 4],
        bounds: around(radius),
        points: Vec::new(),
        texture: None,
        casts: true,
        layers: light.layers.clone(),
    };
    match &light.kind {
        Light2dKind::Point => {
            if radius <= 0.0 {
                return None;
            }
        }
        Light2dKind::Cone {
            direction,
            inner,
            outer,
        } => {
            let d = Vec2::from(*direction);
            if radius <= 0.0 || d.length_squared() < 1e-12 {
                return None;
            }
            let d = d.normalize();
            let outer = outer.clamp(0.0, std::f32::consts::PI);
            let inner = inner.clamp(0.0, outer);
            let cos_inner = inner.cos();
            let cos_outer = outer.cos().min(cos_inner - 1e-4);
            out.kind = KIND_CONE;
            out.params = [d.x, d.y, cos_inner, cos_outer];
        }
        Light2dKind::Directional { direction } => {
            let d = Vec2::from(*direction);
            if d.length_squared() < 1e-12 {
                return None;
            }
            let d = d.normalize();
            out.kind = KIND_DIRECTIONAL;
            out.params = [d.x, d.y, 0.0, 0.0];
        }
        Light2dKind::Global => {
            out.kind = KIND_GLOBAL;
            out.casts = false;
        }
        Light2dKind::Textured {
            texture,
            size,
            rotation,
        } => {
            let half = Vec2::from(*size) * 0.5;
            if half.x <= 0.0 || half.y <= 0.0 || !has_texture(*texture) {
                return None;
            }
            let (s, c) = rotation.sin_cos();
            out.kind = KIND_TEXTURED;
            out.params = [c, s, half.x, half.y];
            out.reach = half.length();
            out.bounds = around(out.reach);
            out.texture = Some(*texture);
        }
        Light2dKind::Polygon { points } => {
            if points.len() < 3 {
                return None;
            }
            let points: Vec<[f32; 2]> = points
                .iter()
                .take(MAX_POLYGON_LIGHT_POINTS)
                .copied()
                .collect();
            let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            let mut far = 0.0f32;
            for p in &points {
                let p = Vec2::from(*p);
                lo = lo.min(p);
                hi = hi.max(p);
                far = far.max(p.length());
            }
            out.kind = KIND_POLYGON;
            out.reach = (far + radius).max(1e-3);
            let (centre, half) = ((lo + hi) * 0.5, (hi - lo) * 0.5 + Vec2::splat(radius));
            out.bounds = [position.x + centre.x, position.y + centre.y, half.x, half.y];
            out.points = points;
        }
    }
    Some(out)
}

/// The frame's point and spot lights projected onto the plane through
/// `origin` with axes `x_axis` and `y_axis`, as point and cone lights over
/// every layer.
fn frame_lights(
    lights: &[viewport_lib::LightSource],
    origin: glam::DVec3,
    x_axis: glam::Vec3,
    y_axis: glam::Vec3,
) -> Vec<Light2d> {
    let normal = x_axis.cross(y_axis).as_dvec3();
    let on_plane = |position: glam::DVec3| {
        let rel = position - origin;
        (
            [
                rel.dot(x_axis.as_dvec3()) as f32,
                rel.dot(y_axis.as_dvec3()) as f32,
            ],
            rel.dot(normal) as f32,
        )
    };
    lights
        .iter()
        .filter_map(|light| {
            let light2d = match light.kind {
                viewport_lib::LightKind::Point {
                    position, range, ..
                } if range > 0.0 => {
                    let (p, h) = on_plane(position);
                    Light2d::point(p, range).with_height(h)
                }
                viewport_lib::LightKind::Spot {
                    position,
                    direction,
                    range,
                    inner_angle,
                    outer_angle,
                    ..
                } if range > 0.0 => {
                    let (p, h) = on_plane(position);
                    let d = glam::Vec3::from(direction);
                    let d2 = [d.dot(x_axis), d.dot(y_axis)];
                    // A spot facing straight into the plane lights a disc.
                    if Vec2::from(d2).length() < 1e-3 {
                        Light2d::point(p, range).with_height(h)
                    } else {
                        Light2d::cone(p, range, d2, inner_angle, outer_angle).with_height(h)
                    }
                }
                _ => return None,
            };
            Some(
                light2d
                    .with_colour(light.colour)
                    .with_intensity(light.intensity),
            )
        })
        .collect()
}

/// How far a directional light's shadows must reach to leave the view: the
/// furthest the view or an edge end lies from the plane origin, twice over.
fn directional_reach(
    inv_view_proj: glam::Mat4,
    origin: glam::Vec3,
    x_axis: glam::Vec3,
    y_axis: glam::Vec3,
    edges: &[GpuEdge],
) -> f32 {
    let normal = x_axis.cross(y_axis);
    let mut far = 0.0f32;
    for e in edges {
        far = far
            .max(Vec2::new(e.ends[0], e.ends[1]).length())
            .max(Vec2::new(e.ends[2], e.ends[3]).length());
    }
    let unproject = |x: f32, y: f32, z: f32| {
        let p = inv_view_proj * glam::Vec4::new(x, y, z, 1.0);
        (p.w.abs() > 1e-12).then(|| p.truncate() / p.w)
    };
    let mut view = 0.0f32;
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let hit = unproject(x, y, 0.9)
            .zip(unproject(x, y, 0.5))
            .and_then(|(a, b)| {
                let dir = b - a;
                let den = normal.dot(dir);
                (den.abs() > 1e-9).then(|| a + dir * (normal.dot(origin - a) / den))
            });
        match hit {
            Some(h) => {
                let rel = h - origin;
                view = view.max(Vec2::new(rel.dot(x_axis), rel.dot(y_axis)).length());
            }
            // A corner ray along the plane: the plane runs to the horizon.
            None => view = view.max(1.0e6),
        }
    }
    2.0 * (far + view.min(1.0e6)) + 1.0
}

struct Layouts {
    light_bgl: viewport_lib::gpu::BindGroupLayout,
    texture_bgl: viewport_lib::gpu::BindGroupLayout,
}

/// The accumulation pipelines for one light layer format.
struct Accumulate {
    format: viewport_lib::gpu::TextureFormat,
    shadow: viewport_lib::gpu::RenderPipeline,
    light: viewport_lib::gpu::RenderPipeline,
}

/// A viewport's light layers, one array layer per part, and the stencil they
/// share, kept across frames and reallocated when the scene size, the layer
/// format or the part count outgrows them.
struct LayerTargets {
    size: [u32; 2],
    format: viewport_lib::gpu::TextureFormat,
    /// The whole array, as lit content samples it.
    array_view: viewport_lib::gpu::TextureView,
    /// One view per array layer, to render into.
    slice_views: Vec<viewport_lib::gpu::TextureView>,
    stencil_view: viewport_lib::gpu::TextureView,
}

/// A viewport's light layers, built in `prepare` for that viewport.
#[derive(Default)]
struct FrameLayer {
    /// The painter layers each array layer lights, array layer 0 first.
    parts: Vec<RangeInclusive<u32>>,
    targets: Option<LayerTargets>,
    /// One shadow stream per item, by its place among the kept items.
    edges: Vec<(viewport_lib::gpu::Buffer, u32)>,
    blocks: Option<(viewport_lib::gpu::Buffer, u32, viewport_lib::gpu::BindGroup)>,
    /// Whether the layers hold the full light (a float layer).
    float: bool,
}

/// Draws [`Lighting2dItem`]s. Register it with
/// `renderer.with_item_type_plugin(device, Box::new(Lighting2dPlugin::default()))`
/// or through `item_types::install`.
#[derive(Default)]
pub struct Lighting2dPlugin {
    layouts: Option<Layouts>,
    accumulate: Vec<Accumulate>,
    /// A white texel and its sampler, bound for lights without a texture.
    white: Option<viewport_lib::gpu::BindGroup>,
    sampler: Option<viewport_lib::gpu::Sampler>,
    frame: PerViewport<FrameLayer>,
    stats: std::sync::Mutex<Lighting2dStats>,
    /// The shadow and light draws the last prepare encoded.
    issued: u32,
}

impl Lighting2dPlugin {
    /// What the light layers drew in the last frame that submitted a
    /// [`Lighting2dItem`].
    pub fn last_stats(&self) -> Lighting2dStats {
        *self.stats.lock().unwrap()
    }

    fn layouts(device: &viewport_lib::gpu::Device) -> Layouts {
        use viewport_lib::gpu;
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
        let texture_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("lighting_2d_texture_bgl"),
            entries: &[
                gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Texture {
                        sample_type: gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Sampler(gpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        Layouts {
            light_bgl,
            texture_bgl,
        }
    }

    fn build_accumulate(
        device: &viewport_lib::gpu::Device,
        layouts: &Layouts,
        format: viewport_lib::gpu::TextureFormat,
    ) -> Accumulate {
        use viewport_lib::gpu;
        use viewport_lib::plugin_api::builders;
        let shader = builders::wgsl_module(
            device,
            "lighting_2d_shader",
            crate::item_types::shader::wgsl_source!("lighting_2d"),
        );
        let layout = builders::pipeline_layout(
            device,
            "lighting_2d_layout",
            &[&layouts.light_bgl, &layouts.texture_bgl],
        );
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
                        vs,
                        fs,
                        buffers: &[gpu::VertexBufferLayout<'_>],
                        target: gpu::ColorTargetState,
                        depth_stencil| {
            builders::render_pipeline(
                device,
                builders::RenderPipelineDesc {
                    label,
                    layout: &layout,
                    vertex_module: &shader,
                    vertex_entry: vs,
                    vertex_buffers: buffers,
                    fragment: Some(gpu::FragmentState {
                        module: &shader,
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
            "vs_shadow",
            "fs_shadow",
            &edge_layout,
            gpu::ColorTargetState {
                format,
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
            "vs_light",
            "fs_light",
            &[],
            gpu::ColorTargetState {
                format,
                blend: Some(builders::ADDITIVE_BLEND),
                write_mask: gpu::ColorWrites::COLOR,
            },
            Some(stencil(
                gpu::CompareFunction::NotEqual,
                gpu::StencilOperation::Keep,
            )),
        );
        Accumulate {
            format,
            shadow,
            light,
        }
    }

    /// The white texel's bind group and the sampler for light textures,
    /// built once.
    fn ensure_white(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
    ) {
        use viewport_lib::gpu;
        if self.white.is_some() {
            return;
        }
        let Some(layouts) = &self.layouts else {
            return;
        };
        let sampler = resources.sampler_for(
            device,
            viewport_lib::SamplerKey {
                wrap_u: viewport_lib::WrapMode::ClampToEdge,
                wrap_v: viewport_lib::WrapMode::ClampToEdge,
                ..Default::default()
            },
        );
        let extent = gpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&gpu::TextureDescriptor {
            label: Some("lighting_2d_white"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            format: gpu::TextureFormat::Rgba8Unorm,
            usage: gpu::TextureUsages::TEXTURE_BINDING | gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            gpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: gpu::Origin3d::ZERO,
                aspect: gpu::TextureAspect::All,
            },
            &[255u8; 4],
            gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            extent,
        );
        let view = texture.create_view(&gpu::TextureViewDescriptor::default());
        self.white = Some(texture_bind_group(
            device,
            &layouts.texture_bgl,
            &view,
            &sampler,
        ));
        self.sampler = Some(sampler);
    }
}

fn texture_bind_group(
    device: &viewport_lib::gpu::Device,
    layout: &viewport_lib::gpu::BindGroupLayout,
    view: &viewport_lib::gpu::TextureView,
    sampler: &viewport_lib::gpu::Sampler,
) -> viewport_lib::gpu::BindGroup {
    use viewport_lib::gpu;
    device.create_bind_group(&gpu::BindGroupDescriptor {
        label: Some("lighting_2d_texture_bg"),
        layout,
        entries: &[
            gpu::BindGroupEntry {
                binding: 0,
                resource: gpu::BindingResource::TextureView(view),
            },
            gpu::BindGroupEntry {
                binding: 1,
                resource: gpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// The light layers in `targets`, allocated or reallocated for `size`,
/// `format` and at least `count` array layers.
fn layer_targets<'a>(
    targets: &'a mut Option<LayerTargets>,
    device: &viewport_lib::gpu::Device,
    count: usize,
    size: [u32; 2],
    format: viewport_lib::gpu::TextureFormat,
) -> &'a LayerTargets {
    use viewport_lib::gpu;
    let fresh = targets
        .as_ref()
        .is_some_and(|t| t.size == size && t.format == format && t.slice_views.len() >= count);
    if !fresh {
        // At least two array layers: GL views a one-layer texture as 2D only.
        let layers = count.next_power_of_two().max(2) as u32;
        let extent = |layers| gpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: layers,
        };
        let light = device.create_texture(&gpu::TextureDescriptor {
            label: Some("lighting_2d_layers"),
            size: extent(layers),
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            format,
            usage: gpu::TextureUsages::RENDER_ATTACHMENT | gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = light.create_view(&gpu::TextureViewDescriptor {
            dimension: Some(gpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let slice_views = (0..layers)
            .map(|k| {
                light.create_view(&gpu::TextureViewDescriptor {
                    dimension: Some(gpu::TextureViewDimension::D2),
                    base_array_layer: k,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let stencil_view = device
            .create_texture(&gpu::TextureDescriptor {
                label: Some("lighting_2d_stencil"),
                size: extent(1),
                mip_level_count: 1,
                sample_count: 1,
                dimension: gpu::TextureDimension::D2,
                format: STENCIL_FORMAT,
                usage: gpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&gpu::TextureViewDescriptor::default());
        *targets = Some(LayerTargets {
            size,
            format,
            array_view,
            slice_views,
            stencil_view,
        });
    }
    targets.as_ref().expect("made above")
}

/// Whether two layer ranges share a layer.
fn ranges_overlap(a: &RangeInclusive<u32>, b: &RangeInclusive<u32>) -> bool {
    a.start() <= b.end() && b.start() <= a.end()
}

/// One kept item, resolved for the frame.
struct ItemPlan {
    frame: BlockFrame,
    ambient: [f32; 3],
    lights: Vec<PlaneLight>,
    edges: Vec<GpuEdge>,
    shadows: bool,
    parts: Vec<(RangeInclusive<u32>, Vec<usize>)>,
}

impl ItemTypePlugin for Lighting2dPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    /// Every tier: uniforms, vertex streams, a stencil and a texture array,
    /// drawn before the frame and sampled by the lit content on either path.
    fn requires(&self) -> viewport_lib::Requirements {
        viewport_lib::Requirements::compatibility_tier()
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        use viewport_lib::gpu;
        *self.stats.get_mut().unwrap() = Lighting2dStats::default();
        self.issued = 0;
        self.frame.get_mut(ctx.viewport_index).parts.clear();

        // The items to draw: visible, with a range, and not overlapping an
        // earlier one.
        let mut kept: Vec<(usize, &Lighting2dItem)> = Vec::new();
        for (i, item) in items.of::<Lighting2dItem>().iter().enumerate() {
            if item.settings.hidden || item.layers.is_empty() {
                continue;
            }
            if let Some(&(k, _)) = kept
                .iter()
                .find(|(_, o)| ranges_overlap(&o.layers, &item.layers))
            {
                ctx.report_detail(
                    viewport_lib::Severity::Warning,
                    "overlapping_light_ranges",
                    || viewport_lib::NoticeDetail::OverlappingLightRanges {
                        ignored: i as u32,
                        kept: k as u32,
                    },
                );
                continue;
            }
            kept.push((i, item));
        }
        if kept.is_empty() {
            return Vec::new();
        }

        if self.layouts.is_none() {
            self.layouts = Some(Self::layouts(device));
        }
        self.ensure_white(device, queue, ctx.resources);
        let float = ctx.resources.has_hdr_path();
        let format = if float {
            FLOAT_LAYER_FORMAT
        } else {
            UNORM_LAYER_FORMAT
        };
        if !self.accumulate.iter().any(|a| a.format == format) {
            let built = Self::build_accumulate(device, self.layouts.as_ref().unwrap(), format);
            self.accumulate.push(built);
        }
        let scale = if float { 1.0 } else { 0.5 };
        let size = [
            (ctx.viewport_size.x * ctx.pixels_per_point)
                .round()
                .max(1.0) as u32,
            (ctx.viewport_size.y * ctx.pixels_per_point)
                .round()
                .max(1.0) as u32,
        ];

        // Everything relative to the camera eye, so far-off scenes keep their
        // precision.
        let eye = ctx.camera.eye_position();
        let view_proj = ctx.camera.projection()
            * ctx.camera.view()
            * glam::Mat4::from_translation(eye.as_vec3());
        let inv_view_proj = view_proj.inverse();

        let resources = ctx.resources;
        let mut plans: Vec<ItemPlan> = kept
            .iter()
            .map(|(_, item)| {
                let plane_origin = item
                    .plane
                    .to_world(glam::DVec2::ZERO, item.elevation as f64);
                let (x_axis, y_axis) = (item.plane.x_axis(), item.plane.y_axis());
                let mut sources: Vec<Light2d> = item.lights.clone();
                if item.use_frame_lights {
                    sources.extend(frame_lights(ctx.lights, plane_origin, x_axis, y_axis));
                }
                let lights: Vec<PlaneLight> = sources
                    .iter()
                    .filter_map(|l| resolve(l, item.intensity, |id| resources.has_texture(id)))
                    .take(item.max_lights.min(MAX_LIGHTS_2D) as usize)
                    .collect();
                let mut edges = Vec::new();
                if item.shadows {
                    let mut shadow_edges = Vec::new();
                    for occluder in &item.occluders {
                        occluder.edges(&mut shadow_edges);
                    }
                    edges = shadow_edges
                        .iter()
                        .map(|e| GpuEdge {
                            ends: [e.a.x, e.a.y, e.b.x, e.b.y],
                            one_sided: e.one_sided as u32,
                        })
                        .collect();
                }
                let light_layers: Vec<RangeInclusive<u32>> =
                    lights.iter().map(|l| l.layers.clone()).collect();
                let [r, g, b, _] = item.ambient.to_linear_rgba();
                ItemPlan {
                    frame: BlockFrame {
                        view_proj,
                        origin: (plane_origin - eye).as_vec3(),
                        x_axis,
                        y_axis,
                    },
                    ambient: [r, g, b],
                    parts: light_parts(&item.layers, &light_layers),
                    lights,
                    edges,
                    shadows: item.shadows,
                }
            })
            .collect();

        // The parts past the most the renderer binds are not lit.
        let mut room = viewport_lib::resources::MAX_PAINTER_LIGHT_RANGES;
        let mut dropped = 0usize;
        for plan in &mut plans {
            let keep = plan.parts.len().min(room);
            dropped += plan.parts.len() - keep;
            plan.parts.truncate(keep);
            room -= keep;
        }
        if dropped > 0 {
            ctx.report_notice(viewport_lib::Severity::Warning, "too_many_light_ranges", || {
                format!(
                    "{dropped} 2D light layer ranges past the {} the renderer binds were not lit; \
                     use fewer lighting items or per-light layer ranges",
                    viewport_lib::resources::MAX_PAINTER_LIGHT_RANGES
                )
            });
        }
        let part_count: usize = plans.iter().map(|p| p.parts.len()).sum();

        let Self {
            layouts,
            accumulate,
            white,
            sampler,
            frame,
            stats,
            issued,
        } = self;
        let (Some(layouts), Some(white), Some(sampler)) = (&*layouts, &*white, &*sampler) else {
            return Vec::new();
        };
        let accumulate = accumulate
            .iter()
            .find(|a| a.format == format)
            .expect("just built");
        let frame = frame.get_mut(ctx.viewport_index);
        frame.float = float;

        // The light blocks, every part's lights in order.
        let block_count: usize = plans
            .iter()
            .map(|p| p.parts.iter().map(|(_, l)| l.len()).sum::<usize>())
            .sum();
        let needed = (block_count as u32).max(1);
        if frame
            .blocks
            .as_ref()
            .is_none_or(|(_, cap, _)| *cap < needed)
        {
            let capacity = needed.next_power_of_two().max(8);
            let buffer = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("lighting_2d_blocks"),
                size: BLOCK_STRIDE * capacity as u64,
                usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&gpu::BindGroupDescriptor {
                label: Some("lighting_2d_light_bg"),
                layout: &layouts.light_bgl,
                entries: &[gpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu::BindingResource::Buffer(gpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: gpu::BufferSize::new(std::mem::size_of::<LightBlock>() as u64),
                    }),
                }],
            });
            frame.blocks = Some((buffer, capacity, bg));
        }
        let (blocks, _, light_bg) = frame.blocks.as_ref().expect("just allocated");
        let mut block_data = vec![0u8; block_count * BLOCK_STRIDE as usize];
        let mut slot = 0usize;
        for plan in &plans {
            let reach = directional_reach(
                inv_view_proj,
                plan.frame.origin,
                plan.frame.x_axis,
                plan.frame.y_axis,
                &plan.edges,
            );
            for (_, lit) in &plan.parts {
                for &l in lit {
                    let block = plan.lights[l].block(&plan.frame, scale, reach);
                    let at = slot * BLOCK_STRIDE as usize;
                    block_data[at..at + std::mem::size_of::<LightBlock>()]
                        .copy_from_slice(bytemuck::bytes_of(&block));
                    slot += 1;
                }
            }
        }
        if !block_data.is_empty() {
            queue.write_buffer(blocks, 0, &block_data);
        }

        // The textured lights' groups, one per texture.
        let mut texture_bgs: Vec<(TextureId, gpu::BindGroup)> = Vec::new();
        for plan in &plans {
            for light in &plan.lights {
                let Some(id) = light.texture else { continue };
                if texture_bgs.iter().any(|(t, _)| *t == id) {
                    continue;
                }
                if let Some(view) = resources.texture_view(id) {
                    texture_bgs.push((
                        id,
                        texture_bind_group(device, &layouts.texture_bgl, view, sampler),
                    ));
                }
            }
        }

        // The shadow streams.
        while frame.edges.len() < plans.len() {
            frame.edges.push((
                device.create_buffer(&gpu::BufferDescriptor {
                    label: Some("lighting_2d_edges"),
                    size: 64 * std::mem::size_of::<GpuEdge>() as u64,
                    usage: gpu::BufferUsages::VERTEX | gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                64,
            ));
        }
        for (i, plan) in plans.iter().enumerate() {
            let count = plan.edges.len() as u32;
            if count == 0 {
                continue;
            }
            if frame.edges[i].1 < count {
                let capacity = count.next_power_of_two().max(64);
                frame.edges[i] = (
                    device.create_buffer(&gpu::BufferDescriptor {
                        label: Some("lighting_2d_edges"),
                        size: capacity as u64 * std::mem::size_of::<GpuEdge>() as u64,
                        usage: gpu::BufferUsages::VERTEX | gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    capacity,
                );
            }
            queue.write_buffer(&frame.edges[i].0, 0, bytemuck::cast_slice(&plan.edges));
        }

        let mut encoder = device.create_command_encoder(&gpu::CommandEncoderDescriptor {
            label: Some("lighting_2d_encoder"),
        });
        let targets = layer_targets(&mut frame.targets, device, part_count, size, format);
        let mut totals = Lighting2dStats::default();
        let mut draws = 0u32;
        let mut slot = 0u32;
        let mut part_index = 0usize;
        for (i, plan) in plans.iter().enumerate() {
            totals.edges += plan.edges.len() as u32;
            for (layers, lit) in &plan.parts {
                let [r, g, b] = plan.ambient.map(|c| c * scale);
                {
                    let mut pass = encoder.begin_render_pass(&gpu::RenderPassDescriptor {
                        #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
                        multiview_mask: None,
                        label: Some("lighting_2d_lights"),
                        color_attachments: &[Some(gpu::RenderPassColorAttachment {
                            view: &targets.slice_views[part_index],
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
                            view: &targets.stencil_view,
                            depth_ops: Some(gpu::Operations {
                                load: gpu::LoadOp::Clear(
                                    viewport_lib::plugin_api::depth::DEPTH_CLEAR,
                                ),
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
                    let (edges, edge_count) = (&frame.edges[i].0, plan.edges.len() as u32);
                    for (j, &l) in lit.iter().enumerate() {
                        let light = &plan.lights[l];
                        // Each light stamps its own value, so no light's
                        // shadow needs clearing before the next.
                        pass.set_bind_group(0, light_bg, &[slot * BLOCK_STRIDE as u32]);
                        let texture_bg = light
                            .texture
                            .and_then(|id| texture_bgs.iter().find(|(t, _)| *t == id))
                            .map_or(white, |(_, bg)| bg);
                        pass.set_bind_group(1, texture_bg, &[]);
                        pass.set_stencil_reference(j as u32 + 1);
                        if plan.shadows && light.casts && edge_count > 0 {
                            pass.set_pipeline(&accumulate.shadow);
                            pass.set_vertex_buffer(0, edges.slice(..));
                            pass.draw(0..6, 0..edge_count);
                            draws += 1;
                        }
                        pass.set_pipeline(&accumulate.light);
                        pass.draw(0..6, 0..1);
                        draws += 1;
                        slot += 1;
                    }
                }
                totals.lights += lit.len() as u32;
                totals.passes += 1;
                frame.parts.push(layers.clone());
                part_index += 1;
            }
        }
        *issued = draws;
        *stats.get_mut().unwrap() = totals;
        vec![encoder.finish()]
    }

    fn draw_calls(&self) -> u32 {
        self.issued
    }

    /// The light layers serve the LDR path as well, though the plugin draws
    /// nothing in its passes.
    fn draws_ldr(&self) -> bool {
        true
    }

    /// One array layer per part of each kept item's range.
    fn painter_light_layers(&self, viewport_index: usize) -> Option<PainterLightLayers> {
        let frame = self.frame.get(viewport_index)?;
        if frame.parts.is_empty() {
            return None;
        }
        let targets = frame.targets.as_ref()?;
        Some(PainterLightLayers::new(
            targets.array_view.clone(),
            frame.parts.clone(),
            if frame.float { 1.0 } else { 2.0 },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_point_and_spot_lights_project_onto_the_plane_and_the_rest_are_skipped() {
        let mut spot = viewport_lib::LightSource::default();
        spot.kind = viewport_lib::LightKind::Spot {
            position: glam::DVec3::new(1.0, 2.0, 4.0),
            direction: [1.0, 0.0, -1.0],
            range: 20.0,
            inner_angle: 0.2,
            outer_angle: 0.4,
            radius: 0.0,
        };
        let lights = vec![
            viewport_lib::LightSource::default(),
            viewport_lib::LightSource::point_candela(
                glam::DVec3::new(3.0, 4.0, 9.0),
                viewport_lib::Candela(2.0),
                10.0,
                0.0,
            ),
            spot,
        ];
        let out = frame_lights(&lights, glam::DVec3::ZERO, glam::Vec3::X, glam::Vec3::Y);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].kind, Light2dKind::Point);
        assert_eq!(out[0].position, [3.0, 4.0]);
        assert_eq!(out[0].radius, 10.0);
        assert_eq!(out[0].height, 9.0);
        assert_eq!(
            out[1].kind,
            Light2dKind::Cone {
                direction: [1.0, 0.0],
                inner: 0.2,
                outer: 0.4
            }
        );
    }

    #[test]
    fn lights_of_no_size_light_nothing() {
        let any = |_| true;
        assert!(resolve(&Light2d::point([0.0; 2], 0.0), 1.0, any).is_none());
        assert!(resolve(&Light2d::directional([0.0, 0.0]), 1.0, any).is_none());
        assert!(
            resolve(
                &Light2d::polygon([0.0; 2], vec![[0.0; 2]; 2], 1.0),
                1.0,
                any
            )
            .is_none()
        );
        let textured = Light2d::textured(TextureId::from_raw(3), [0.0; 2], [4.0, 2.0], 0.0);
        assert!(resolve(&textured, 1.0, |_| false).is_none());
        let t = resolve(&textured, 1.0, any).expect("held texture");
        assert_eq!(t.params, [1.0, 0.0, 2.0, 1.0]);
        assert!(!resolve(&Light2d::global(), 1.0, any).unwrap().casts);
    }

    #[test]
    fn block_fits_its_stride() {
        assert!(std::mem::size_of::<LightBlock>() as u64 <= BLOCK_STRIDE);
        assert_eq!(BLOCK_STRIDE % 256, 0);
        assert_eq!(std::mem::size_of::<GpuEdge>(), 20);
    }
}
