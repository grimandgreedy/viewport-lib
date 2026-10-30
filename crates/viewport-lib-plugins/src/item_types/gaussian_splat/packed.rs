//! Gaussian splats on a device without compute or vertex storage (the
//! compatibility tier).
//!
//! Each set is packed into `Rgba32Uint` data textures by a render pass that
//! reads its channel buffers as vertex attributes, one point per splat
//! (`gaussian_splat_pack.wgsl`), and repacked only when the set changes. The
//! draw order is sorted back to front on the CPU from the positions the
//! plugin keeps for picking, per set and viewport, and only when the eye, the
//! model matrix or the set has changed; it reaches the draw as an
//! instance-stepped index stream (`gaussian_splat_packed.wgsl`).

use super::store::{GaussianSplatGpuSet, ShDegree};
use crate::item_types::shader::{scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::resources::DeviceResources;

/// The packed path's draw shader: the common splat code, then the vertex
/// stage reading the data textures.
pub(crate) fn draw_source() -> String {
    scene_shader(
        &[],
        &[
            wgsl_source!("gaussian_splat_common"),
            wgsl_source!("gaussian_splat_packed"),
        ]
        .join("\n"),
    )
}

/// The widest a set's data textures are made; a set takes as many rows as it
/// needs below the device's limit.
const MAX_WIDTH: u32 = 2048;

/// The data texture extent for `capacity` splats, or `None` past the device's
/// 2D texture limit.
pub(super) fn texture_extent(capacity: u32, max_dimension: u32) -> Option<(u32, u32)> {
    let width = capacity.clamp(1, MAX_WIDTH.min(max_dimension));
    let height = capacity.div_ceil(width).max(1);
    (height <= max_dimension).then_some((width, height))
}

/// Refuse a set too large to pack on this device.
pub(super) fn check_capacity(
    device: &gpu::Device,
    capacity: u32,
) -> viewport_lib::error::ViewportResult<()> {
    let max = device.limits().max_texture_dimension_2d;
    match texture_extent(capacity, max) {
        Some(_) => Ok(()),
        None => Err(viewport_lib::error::ViewportError::DeviceUnsupported {
            reason: format!(
                "a set of {capacity} Gaussian splats is past what this device's {max} x {max} texture limit packs"
            ),
        }),
    }
}

/// Members of [`PackedPipelines`].
pub(super) const DRAW: usize = 0;
pub(super) const MASK: usize = 1;
/// The base pack passes, a pair per SH stride (degree 0, 1, 3): the first
/// writes texels 0 and 1, the second texel 2.
const PACK_BASE: usize = 2;
/// The SH pack passes, a pair per stride of the degrees that have
/// view-dependent terms (1, 3).
const PACK_SH: usize = 8;
const PIPELINE_COUNT: usize = 12;

/// Which of the per-stride pack pipelines a degree takes.
fn degree_slot(degree: ShDegree) -> usize {
    match degree {
        ShDegree::Zero => 0,
        ShDegree::One => 1,
        ShDegree::Three => 2,
    }
}

const STRIDES: [u64; 3] = [12, 48, 192];

/// What a packed-path pipeline build reads.
pub(super) struct PackedRecipe {
    device: gpu::Device,
    draw_layout: gpu::PipelineLayout,
    draw_shader: viewport_lib::plugin_api::LazyModule,
    mask_layout: gpu::PipelineLayout,
    mask_shader: viewport_lib::plugin_api::LazyModule,
    pack_layout: gpu::PipelineLayout,
    pack_shader: viewport_lib::plugin_api::LazyModule,
    ldr_format: gpu::TextureFormat,
    depth_format: gpu::TextureFormat,
}

/// The draw, the outline mask, and the pack passes.
pub(super) type PackedPipelines =
    viewport_lib::plugin_api::LazyPipelines<PackedRecipe, PIPELINE_COUNT>;

const SORTED_INDEX_ATTRS: [gpu::VertexAttribute; 1] = gpu::vertex_attr_array![0 => Uint32];
const POSITION_ATTRS: [gpu::VertexAttribute; 1] = gpu::vertex_attr_array![0 => Float32x4];
const SCALE_ATTRS: [gpu::VertexAttribute; 1] = gpu::vertex_attr_array![1 => Float32x4];
const ROTATION_ATTRS: [gpu::VertexAttribute; 1] = gpu::vertex_attr_array![2 => Float32x4];
const OPACITY_ATTRS: [gpu::VertexAttribute; 1] = gpu::vertex_attr_array![3 => Float32];
const DC_ATTRS: [gpu::VertexAttribute; 1] = gpu::vertex_attr_array![4 => Float32x3];
const SH_ATTRS: [gpu::VertexAttribute; 4] =
    gpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32x3];

fn per_splat(
    stride: u64,
    attributes: &'static [gpu::VertexAttribute],
) -> gpu::VertexBufferLayout<'static> {
    gpu::VertexBufferLayout {
        array_stride: stride,
        step_mode: gpu::VertexStepMode::Vertex,
        attributes,
    }
}

fn build(r: &PackedRecipe, i: usize) -> gpu::RenderPipeline {
    match i {
        DRAW => builders::build_dual_pipeline_variant(
            &r.device,
            &builders::DualPipelineDesc {
                label: "gaussian_splat_packed_pipeline",
                layout: &r.draw_layout,
                shader: r.draw_shader.get(),
                vertex_entry: "vs_packed",
                fragment_entry: "fs_main",
                vertex_buffers: &[gpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: gpu::VertexStepMode::Instance,
                    attributes: &SORTED_INDEX_ATTRS,
                }],
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
                sample_count: 1,
                ldr_format: r.ldr_format,
                depth_format: r.depth_format,
            },
            false,
        ),
        MASK => super::pipeline::disc_mask_pipeline(
            &r.device,
            &r.mask_layout,
            r.mask_shader.get(),
            r.depth_format,
        ),
        _ => {
            let pair = (i - PACK_BASE) % 2 == 0;
            let (entry, buffers) = if i < PACK_SH {
                let stride = STRIDES[(i - PACK_BASE) / 2];
                (
                    "vs_pack_base",
                    vec![
                        per_splat(16, &POSITION_ATTRS),
                        per_splat(16, &SCALE_ATTRS),
                        per_splat(16, &ROTATION_ATTRS),
                        per_splat(4, &OPACITY_ATTRS),
                        per_splat(stride, &DC_ATTRS),
                    ],
                )
            } else {
                // Degrees 1 and 3, the strides past degree 0.
                let stride = STRIDES[(i - PACK_SH) / 2 + 1];
                ("vs_pack_sh", vec![per_splat(stride, &SH_ATTRS)])
            };
            let target = Some(gpu::ColorTargetState {
                format: gpu::TextureFormat::Rgba32Uint,
                blend: None,
                write_mask: gpu::ColorWrites::ALL,
            });
            let pair_targets = [target.clone(), target.clone()];
            let last_target = [target];
            builders::render_pipeline(
                &r.device,
                builders::RenderPipelineDesc {
                    label: "gaussian_splat_pack_pipeline",
                    layout: &r.pack_layout,
                    vertex_module: r.pack_shader.get(),
                    vertex_entry: entry,
                    vertex_buffers: &buffers,
                    fragment: Some(gpu::FragmentState {
                        module: r.pack_shader.get(),
                        entry_point: Some(if pair { "fs_pack_pair" } else { "fs_pack_last" }),
                        targets: if pair { &pair_targets } else { &last_target },
                        compilation_options: gpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: gpu::PrimitiveState {
                        topology: gpu::PrimitiveTopology::PointList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: gpu::MultisampleState::default(),
                    cache: None,
                },
            )
        }
    }
}

/// Pipelines and layouts, made on the first prepare with items.
pub(super) struct PackedGpu {
    pub(super) pipelines: PackedPipelines,
    draw_bgl: gpu::BindGroupLayout,
    pack_bgl: gpu::BindGroupLayout,
    /// Group 1 of the outline mask pipeline: the single uniform
    /// `point_disc_mask.wgsl` reads.
    pub(super) mask_bgl: gpu::BindGroupLayout,
}

impl PackedGpu {
    pub(super) fn new(device: &gpu::Device, resources: &DeviceResources) -> Self {
        let texture_entry = |binding| gpu::BindGroupLayoutEntry {
            binding,
            visibility: gpu::ShaderStages::VERTEX,
            ty: gpu::BindingType::Texture {
                sample_type: gpu::TextureSampleType::Uint,
                view_dimension: gpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let draw_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_packed_bgl"),
            entries: &[
                // The fragment stage reads `ignore_clip` for the clip test.
                builders::uniform_entry(0, gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                texture_entry(5),
                texture_entry(6),
            ],
        });
        let pack_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_pack_bgl"),
            entries: &[builders::uniform_entry(0, gpu::ShaderStages::VERTEX)],
        });
        let mask_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_mask_bgl"),
            entries: &[builders::uniform_entry(
                0,
                gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT,
            )],
        });
        let pipelines = resources.lazy_pipelines(
            PackedRecipe {
                device: device.clone(),
                draw_layout: builders::standard_scene_layout(
                    device,
                    "gaussian_splat_packed_layout",
                    resources.group0_layout(),
                    &draw_bgl,
                ),
                draw_shader: resources.lazy_module(
                    device,
                    "gaussian_splat_packed_shader",
                    &draw_source(),
                ),
                mask_layout: builders::pipeline_layout(
                    device,
                    "gaussian_splat_mask_layout",
                    &[resources.group0_layout(), &mask_bgl],
                ),
                mask_shader: resources.lazy_module(
                    device,
                    "point_disc_mask_shader",
                    &scene_shader(&[], wgsl_source!("point_disc_mask")),
                ),
                pack_layout: builders::pipeline_layout(
                    device,
                    "gaussian_splat_pack_layout",
                    &[&pack_bgl],
                ),
                pack_shader: resources.lazy_module(
                    device,
                    "gaussian_splat_pack_shader",
                    wgsl_source!("gaussian_splat_pack"),
                ),
                ldr_format: resources.target_format(),
                depth_format: resources.scene_depth_format(),
            },
            build,
        );
        Self {
            pipelines,
            draw_bgl,
            pack_bgl,
            mask_bgl,
        }
    }

    /// Whether the draw pipeline is built. The mask waits for it, so a set is
    /// never outlined before it is drawn.
    pub(super) fn drawn(&self) -> bool {
        self.pipelines.available(DRAW)
    }
}

/// One set's data textures and the bookkeeping that says when to repack.
pub(super) struct PackedSet {
    /// The store revision the textures were made for: a replace or a grow
    /// moves it, and the textures are remade at the new capacity.
    pub(super) revision: u64,
    /// The set generation last packed, `None` before the first pack.
    packed: Option<u64>,
    /// Base textures (centre, covariance, colour), then the SH textures
    /// (1 x 1 placeholders for a degree-0 set).
    views: [gpu::TextureView; 6],
    pack_uniform: gpu::Buffer,
    /// Per viewport index, the sort and the draw bind group.
    pub(super) viewports: Vec<Option<PackedViewport>>,
}

/// The sort order of one set in one viewport, and what the draw binds.
pub(super) struct PackedViewport {
    uniform_buf: gpu::Buffer,
    pub(super) bind_group: gpu::BindGroup,
    pub(super) index_buf: gpu::Buffer,
    /// What the current order was sorted for.
    sorted_for: Option<SortKey>,
    scratch: SortScratch,
}

/// Everything a sort's result depends on.
#[derive(Clone, Copy, PartialEq)]
struct SortKey {
    eye: [f32; 3],
    model: [[f32; 4]; 4],
    count: u32,
    generation: u64,
}

impl PackedSet {
    pub(super) fn new(
        device: &gpu::Device,
        queue: &gpu::Queue,
        set: &GaussianSplatGpuSet,
        revision: u64,
    ) -> Self {
        let max = device.limits().max_texture_dimension_2d;
        let (width, height) = texture_extent(set.positions.capacity(), max).unwrap_or((1, 1));
        let sh = set.sh_degree != ShDegree::Zero;
        let texture = |label, w, h| {
            device
                .create_texture(&gpu::TextureDescriptor {
                    label: Some(label),
                    size: gpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: gpu::TextureDimension::D2,
                    format: gpu::TextureFormat::Rgba32Uint,
                    usage: gpu::TextureUsages::TEXTURE_BINDING
                        | gpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&gpu::TextureViewDescriptor::default())
        };
        let (sw, sh_h) = if sh { (width, height) } else { (1, 1) };
        let views = [
            texture("gaussian_splat_base0", width, height),
            texture("gaussian_splat_base1", width, height),
            texture("gaussian_splat_base2", width, height),
            texture("gaussian_splat_sh0", sw, sh_h),
            texture("gaussian_splat_sh1", sw, sh_h),
            texture("gaussian_splat_sh2", sw, sh_h),
        ];
        let pack_uniform = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("gaussian_splat_pack_uniform"),
            size: 16,
            usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &pack_uniform,
            0,
            bytemuck::cast_slice(&[width, height, 0, 0]),
        );
        Self {
            revision,
            packed: None,
            views,
            pack_uniform,
            viewports: Vec::new(),
        }
    }

    /// Repack the set into its textures if it changed since the last pack.
    /// `false` while a pack pipeline is still compiling, when the set has
    /// never been packed and so cannot draw yet.
    pub(super) fn pack_if_changed(
        &mut self,
        device: &gpu::Device,
        gpu: &PackedGpu,
        encoder: &mut gpu::CommandEncoder,
        set: &GaussianSplatGpuSet,
    ) -> bool {
        if self.packed == Some(set.generation) {
            return true;
        }
        let slot = degree_slot(set.sh_degree);
        let base = PACK_BASE + slot * 2;
        let sh = (slot > 0).then(|| PACK_SH + (slot - 1) * 2);
        // Ask for every pass this set needs before giving up on any, so they
        // compile together.
        let needed: Vec<usize> = [Some(base), Some(base + 1), sh, sh.map(|i| i + 1)]
            .into_iter()
            .flatten()
            .collect();
        let built: Vec<_> = needed.iter().map(|&i| gpu.pipelines.get(i)).collect();
        if built.iter().any(Option::is_none) {
            return self.packed.is_some();
        }
        let built: Vec<&gpu::RenderPipeline> = built.into_iter().flatten().collect();
        let bind_group = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("gaussian_splat_pack_bg"),
            layout: &gpu.pack_bgl,
            entries: &[gpu::BindGroupEntry {
                binding: 0,
                resource: self.pack_uniform.as_entire_binding(),
            }],
        });
        let count = set.count();
        let mut pass = |pipeline: &gpu::RenderPipeline,
                        views: &[gpu::TextureView],
                        buffers: &[&gpu::Buffer]| {
            let attachment = |view| {
                Some(gpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: gpu::Operations {
                        load: gpu::LoadOp::Clear(gpu::Color::TRANSPARENT),
                        store: gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })
            };
            let attachments: Vec<_> = views.iter().map(attachment).collect();
            let mut pass = encoder.begin_render_pass(&gpu::RenderPassDescriptor {
                #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
                multiview_mask: None,
                label: Some("gaussian_splat_pack"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            for (slot, buffer) in buffers.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buffer.slice(..));
            }
            pass.draw(0..count, 0..1);
        };
        let base_buffers = [
            set.positions.buffer(),
            set.scales.buffer(),
            set.rotations.buffer(),
            set.opacities.buffer(),
            set.sh.buffer(),
        ];
        pass(built[0], &self.views[0..2], &base_buffers);
        pass(built[1], &self.views[2..3], &base_buffers);
        if sh.is_some() {
            pass(built[2], &self.views[3..5], &[set.sh.buffer()]);
            pass(built[3], &self.views[5..6], &[set.sh.buffer()]);
        }
        self.packed = Some(set.generation);
        true
    }

    /// The viewport's sort state, sorted for this frame's eye and model:
    /// re-sorted and re-uploaded only when they or the set changed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sorted_viewport(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        gpu: &PackedGpu,
        set: &GaussianSplatGpuSet,
        viewport: usize,
        eye: [f32; 3],
        model: [[f32; 4]; 4],
        viewport_size: [f32; 2],
        ignore_clip: bool,
    ) -> &PackedViewport {
        while self.viewports.len() <= viewport {
            self.viewports.push(None);
        }
        let count = set.count();
        let views = &self.views;
        let state = self.viewports[viewport].get_or_insert_with(|| {
            let uniform_buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("gaussian_splat_packed_uniform"),
                size: std::mem::size_of::<super::pipeline::SplatUniform>() as u64,
                usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut entries = vec![gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }];
            entries.extend(
                views
                    .iter()
                    .enumerate()
                    .map(|(i, view)| gpu::BindGroupEntry {
                        binding: i as u32 + 1,
                        resource: gpu::BindingResource::TextureView(view),
                    }),
            );
            let bind_group = device.create_bind_group(&gpu::BindGroupDescriptor {
                label: Some("gaussian_splat_packed_bg"),
                layout: &gpu.draw_bgl,
                entries: &entries,
            });
            PackedViewport {
                uniform_buf,
                bind_group,
                index_buf: index_buffer(device, set.positions.capacity()),
                sorted_for: None,
                scratch: SortScratch::default(),
            }
        });

        let uniform = super::pipeline::SplatUniform {
            model,
            viewport_w: viewport_size[0],
            viewport_h: viewport_size[1],
            sh_degree: match set.sh_degree {
                ShDegree::Zero => 0,
                ShDegree::One => 1,
                ShDegree::Three => 3,
            },
            count,
            ignore_clip: ignore_clip as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&state.uniform_buf, 0, bytemuck::bytes_of(&uniform));

        let key = SortKey {
            eye,
            model,
            count,
            generation: set.generation,
        };
        if state.sorted_for != Some(key) {
            let positions = &set.cpu_positions[..(count as usize).min(set.cpu_positions.len())];
            let order = state.scratch.back_to_front(
                positions,
                glam::Mat4::from_cols_array_2d(&model),
                glam::Vec3::from(eye),
            );
            if state.index_buf.size() < (order.len() as u64 * 4).max(4) {
                state.index_buf = index_buffer(device, order.len() as u32);
            }
            if !order.is_empty() {
                queue.write_buffer(&state.index_buf, 0, bytemuck::cast_slice(order));
            }
            state.sorted_for = Some(key);
        }
        state
    }
}

fn index_buffer(device: &gpu::Device, splats: u32) -> gpu::Buffer {
    device.create_buffer(&gpu::BufferDescriptor {
        label: Some("gaussian_splat_sorted_indices"),
        size: (splats as u64 * 4).max(4),
        usage: gpu::BufferUsages::VERTEX | gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Reused buffers for the CPU depth sort.
#[derive(Default)]
pub(super) struct SortScratch {
    distances: Vec<f32>,
    keys: Vec<u16>,
    counts: Vec<u32>,
    order: Vec<u32>,
}

impl SortScratch {
    /// The splat indices far to near from `eye`: a counting sort of each
    /// splat's distance quantised to 16 bits across the set's own range, one
    /// pass over the splats.
    pub(super) fn back_to_front(
        &mut self,
        positions: &[[f32; 3]],
        model: glam::Mat4,
        eye: glam::Vec3,
    ) -> &[u32] {
        let n = positions.len();
        self.distances.clear();
        self.distances.extend(
            positions
                .iter()
                .map(|p| model.transform_point3(glam::Vec3::from(*p)).distance(eye)),
        );
        let (near, far) = self
            .distances
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &d| (lo.min(d), hi.max(d)));
        let scale = if far > near {
            65535.0 / (far - near)
        } else {
            0.0
        };
        self.keys.clear();
        // Far first: the farthest splat gets key 0.
        self.keys.extend(
            self.distances
                .iter()
                .map(|&d| 65535 - ((d - near) * scale) as u16),
        );
        self.counts.clear();
        self.counts.resize(1 << 16, 0);
        for &k in &self.keys {
            self.counts[k as usize] += 1;
        }
        let mut sum = 0;
        for c in &mut self.counts {
            let here = *c;
            *c = sum;
            sum += here;
        }
        self.order.clear();
        self.order.resize(n, 0);
        for (i, &k) in self.keys.iter().enumerate() {
            let at = &mut self.counts[k as usize];
            self.order[*at as usize] = i as u32;
            *at += 1;
        }
        &self.order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sort_orders_splats_far_to_near_and_is_stable() {
        let positions = [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 9.0],
            [0.0, 0.0, 1.0],
        ];
        let mut scratch = SortScratch::default();
        let order = scratch.back_to_front(&positions, glam::Mat4::IDENTITY, glam::Vec3::ZERO);
        assert_eq!(order, &[2, 0, 1, 3]);
        // The model matrix moves the splats before their distance is taken.
        let flipped = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -10.0));
        let order = scratch.back_to_front(&positions, flipped, glam::Vec3::ZERO);
        assert_eq!(order, &[1, 3, 0, 2]);
    }

    #[test]
    fn the_texture_extent_fits_the_device_or_refuses() {
        assert_eq!(texture_extent(10, 2048), Some((10, 1)));
        assert_eq!(texture_extent(5000, 2048), Some((2048, 3)));
        assert_eq!(texture_extent(2048 * 2048, 2048), Some((2048, 2048)));
        assert_eq!(texture_extent(2048 * 2048 + 1, 2048), None);
    }
}
