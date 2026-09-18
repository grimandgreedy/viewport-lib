//! The ribbon item type as an [`ItemTypePlugin`]: polyline strips swept into
//! flat, two-sided quad strips. Consumers submit [`RibbonItem`]s on
//! `SceneFrame::ribbon_items`, or [`RibbonRefItem`]s on
//! `SceneFrame::ribbon_refs` to draw a ribbon uploaded once through
//! `upload_ribbon`; the renderer routes both fields to this plugin.
//!
//! Ribbons carry more of the render surface than the other two curve types:
//! the blend mode selects a pipeline variant, transparent ribbons route through
//! the OIT pass, and they cast shadows.

use super::cpu_pick::{self, CurveLevels, RectAccumulator, strips_or_single};
use super::draw::{
    build_frame_with, outline_mask_curve_mesh, radius_in_pixels, render_pick_curve_mesh,
    resolve_curve_sub_object,
};
use super::pipeline::{CurveFrame, CurvePickGpu, draw_mesh, draw_solid_indexed};
use super::types::RibbonId;
use crate::plugin_api::pick_helpers::{project_to_screen, ray_triangle, segment_in_rect};
use crate::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, RectPickContext, ShadowCastContext,
};
use crate::renderer::{
    PickHit, PickId, PickMask, PickRectResult, RibbonItem, RibbonRefItem, SpriteBlend, SubObjectRef,
};
use crate::resources::{
    DeviceResources, DualPipeline, HDR_COLOR_FORMAT, Vertex, VertexBufferLayoutExt,
};

pub(crate) const TYPE_NAME: &str = "viewport.ribbon";

/// Ribbon pipeline variant axes: blend mode and thin-wireframe vs solid-triangle
/// geometry. Both axes select the same shader and bind group layout : only the
/// blend state, depth-write flag, and topology change : so this is closer in
/// shape to the mesh family's own `PipelineKey` than to `PolylineKey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct RibbonKey {
    pub blend: SpriteBlend,
    pub wireframe: bool,
    /// Whether the variant writes depth. An `AlphaBlend` or `Premultiplied`
    /// ribbon that does draws with the opaque scene; one that does not is
    /// routed to OIT instead and never reaches this set. `Additive` never
    /// writes depth whatever the item asks for, since accumulating is the
    /// point of that blend.
    pub depth_write: bool,
}

impl RibbonKey {
    fn blend_index(self) -> usize {
        match self.blend {
            SpriteBlend::AlphaBlend => 0,
            SpriteBlend::Additive => 1,
            SpriteBlend::Premultiplied => 2,
        }
    }

    /// Every axis combination, for eager cross-product construction
    /// (`RibbonVariantSet::build`).
    pub fn all() -> impl Iterator<Item = RibbonKey> {
        [
            SpriteBlend::AlphaBlend,
            SpriteBlend::Additive,
            SpriteBlend::Premultiplied,
        ]
        .into_iter()
        .flat_map(|blend| {
            [false, true].into_iter().flat_map(move |wireframe| {
                [false, true].into_iter().map(move |depth_write| RibbonKey {
                    blend,
                    wireframe,
                    depth_write,
                })
            })
        })
    }

    fn slot(self) -> usize {
        self.blend_index() + 3 * (self.wireframe as usize) + 6 * (self.depth_write as usize)
    }
}

/// Build one value per [`RibbonKey`] and place each at its own
/// [`slot`](RibbonKey::slot), which is *not* the order `all()` yields them in.
/// Collecting in iteration order instead puts variants under the wrong keys, so
/// `get` hands back a pipeline belonging to a different blend or topology.
fn place_by_slot<T>(mut build: impl FnMut(RibbonKey) -> T) -> Vec<T> {
    let mut slots: Vec<Option<T>> = (0..12).map(|_| None).collect();
    for key in RibbonKey::all() {
        slots[key.slot()] = Some(build(key));
    }
    slots
        .into_iter()
        .map(|v| v.unwrap_or_else(|| unreachable!("every slot is covered by all()")))
        .collect()
}

/// A `DualPipeline` built for every reachable [`RibbonKey`], indexed for a
/// hash-free draw-time lookup (`get`).
pub(super) struct RibbonVariantSet {
    variants: [DualPipeline; 12],
}

impl RibbonVariantSet {
    pub fn build(build: impl FnMut(RibbonKey) -> DualPipeline) -> Self {
        Self {
            variants: place_by_slot(build)
                .try_into()
                .unwrap_or_else(|_| unreachable!("RibbonKey::all() yields exactly 12 keys")),
        }
    }

    pub fn get(&self, key: RibbonKey) -> &DualPipeline {
        &self.variants[key.slot()]
    }
}

/// The ribbon render, OIT, shadow, pick and mask pipelines.
struct RibbonGpu {
    pipelines: RibbonVariantSet,
    /// Weighted-blended OIT pipeline, straight alpha. HDR-only.
    oit_pipeline: crate::gpu::RenderPipeline,
    /// The same pipeline with the premultiplied fragment entry. The two share
    /// one shader module and pipeline layout, differing only in entry point:
    /// straight versus premultiplied is not a separate GPU blend state here.
    oit_pipeline_premultiplied: crate::gpu::RenderPipeline,
    /// Depth-only shadow-cast pipeline. One pipeline covers every ribbon:
    /// geometry is always the thin, two-sided expanded quad strip, so there is
    /// no cutout or two-sided axis to key on the way the mesh family has.
    /// Group 0 is the shadow pass's own dynamic-offset camera; group 1 reuses
    /// the ribbon bind group built for the solid draw.
    shadow_pipeline: crate::gpu::RenderPipeline,
    pick: CurvePickGpu,
}

impl RibbonGpu {
    fn new(
        device: &crate::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::RibbonResources,
    ) -> Self {
        use crate::resources::builders::{
            DualPipelineDesc, build_dual_pipeline, standard_scene_layout, wgsl_module, wgsl_source,
        };

        let shader = wgsl_module(device, "ribbon_shader", wgsl_source!("ribbon"));
        let layout = standard_scene_layout(
            device,
            "ribbon_pipeline_layout",
            resources.shared_bindings().group0_layout,
            &layouts.bgl,
        );

        let additive_blend = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::One,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::One,
                operation: crate::gpu::BlendOperation::Add,
            },
        };
        let premultiplied_blend = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::One,
                dst_factor: crate::gpu::BlendFactor::OneMinusSrcAlpha,
                operation: crate::gpu::BlendOperation::Add,
            },
        };

        // Additive and premultiplied ribbons are typically used for emissive
        // trails; depth write is disabled so successive segments accumulate
        // rather than clipping each other when they overlap.
        let pipelines = RibbonVariantSet::build(|key| {
            let blend = match key.blend {
                SpriteBlend::AlphaBlend => crate::gpu::BlendState::ALPHA_BLENDING,
                SpriteBlend::Additive => additive_blend,
                SpriteBlend::Premultiplied => premultiplied_blend,
            };
            // Additive never writes depth: successive segments accumulate
            // rather than clipping each other where they overlap.
            let depth_write = key.depth_write && !matches!(key.blend, SpriteBlend::Additive);
            build_dual_pipeline(
                device,
                &DualPipelineDesc {
                    label: "ribbon_pipeline_variant",
                    layout: &layout,
                    shader: &shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &[Vertex::buffer_layout()],
                    blend: Some(blend),
                    topology: if key.wireframe {
                        crate::gpu::PrimitiveTopology::LineList
                    } else {
                        crate::gpu::PrimitiveTopology::TriangleList
                    },
                    cull_mode: None,
                    depth_write,
                    depth_compare: crate::gpu::CompareFunction::Less,
                    sample_count: resources.sample_count,
                    ldr_format: resources.target_format,
                },
            )
        });

        // OIT: the same layout as the solid draw, so the same bind group is
        // bound again here; only the accum / reveal targets differ.
        let oit_shader = wgsl_module(device, "ribbon_oit_shader", wgsl_source!("ribbon_oit"));
        let make_oit = |entry: &str, label: &str| {
            crate::resources::builders::render_pipeline(
                device,
                crate::resources::builders::RenderPipelineDesc {
                    label,
                    layout: &layout,
                    vertex_module: &oit_shader,
                    vertex_entry: "vs_main",
                    vertex_buffers: &[Vertex::buffer_layout()],
                    fragment: Some(crate::gpu::FragmentState {
                        module: &oit_shader,
                        entry_point: Some(entry),
                        targets: &[
                            Some(crate::gpu::ColorTargetState {
                                format: crate::gpu::TextureFormat::Rgba16Float,
                                blend: Some(crate::plugin_api::target_desc::OIT_ACCUM_BLEND),
                                write_mask: crate::gpu::ColorWrites::ALL,
                            }),
                            Some(crate::gpu::ColorTargetState {
                                format: crate::gpu::TextureFormat::R8Unorm,
                                blend: Some(crate::plugin_api::target_desc::OIT_REVEAL_BLEND),
                                write_mask: crate::gpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: crate::gpu::PrimitiveState {
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                        false,
                        crate::gpu::CompareFunction::LessEqual,
                    )),
                    multisample: crate::gpu::MultisampleState {
                        count: resources.sample_count,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };

        let shadow_shader = wgsl_module(
            device,
            "ribbon_shadow_shader",
            wgsl_source!("ribbon_shadow"),
        );
        let shadow_vertex_layouts = [super::pipeline::position_only_layout()];
        let mut shadow_opts = crate::resources::PluginPipelineOpts::new(
            Some("ribbon_shadow_pipeline"),
            &shadow_shader,
            "vs_main",
            "",
            &shadow_vertex_layouts,
        );
        let shadow_extra: [&crate::gpu::BindGroupLayout; 1] = [&layouts.bgl];
        shadow_opts.extra_bind_group_layouts = &shadow_extra;
        shadow_opts.primitive.cull_mode = None;
        shadow_opts.depth_compare = crate::gpu::CompareFunction::Less;
        // A ribbon is a thin open surface, so it self-shadows badly under the
        // mild default. Same bias the lib uses where the shadow pass does not
        // cull.
        shadow_opts.depth_bias =
            Some(crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS_TWO_SIDED);

        Self {
            pipelines,
            oit_pipeline: make_oit("fs_oit", "ribbon_oit_pipeline"),
            oit_pipeline_premultiplied: make_oit(
                "fs_oit_premultiplied",
                "ribbon_oit_pipeline_premultiplied",
            ),
            shadow_pipeline: resources.build_shadow_pipeline(device, &shadow_opts),
            pick: CurvePickGpu::new(device, resources, "ribbon", true),
        }
    }
}

#[derive(Default)]
pub(crate) struct RibbonPlugin {
    /// The pre-uploaded ribbons, owned by the type that draws them.
    stored: super::store::RibbonStore,
    /// The group-1 layout every upload builds its bind group against. Created
    /// on registration, because an upload can arrive before the first frame.
    layouts: Option<super::store::RibbonResources>,
    /// Resource epochs the store was last revalidated against. A stored ribbon
    /// that draws with a streak texture holds its view in a bind group, so a
    /// free or a replace since the last frame means it has to be rebound.
    deps_gate: crate::resources::resource_deps::DepsGate,
    gpu: Option<RibbonGpu>,
    /// Per drawn item, rebuilt each prepare: the inline items first, then the
    /// references.
    frame: Vec<CurveFrame>,
    /// Every inline item from the last prepared frame. Reference items are not
    /// here: their geometry lives on the GPU, so they answer the GPU pick only.
    pick_items: Vec<RibbonItem>,
}

impl RibbonPlugin {
    /// Rebind stored ribbons whose streak texture was freed or swapped since
    /// the last frame.
    ///
    /// A ribbon is the host's content, held until the host drops the handle, so
    /// the answer to a freed texture is to rebind against the fallback, never
    /// to discard the ribbon: the geometry stays, it stops being textured.
    /// Leaving it alone instead would keep the freed texture alive through the
    /// bind group and go on sampling it, so the memory the host asked to
    /// release is never released.
    ///
    /// A replace swaps the view behind a live id, which no per-ribbon liveness
    /// check can see, so every stored ribbon that names a texture is rebound.
    fn revalidate_store(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
    ) {
        use crate::resources::resource_deps::Revalidate;
        let action = self.deps_gate.poll(resources);
        if action == Revalidate::Valid {
            return;
        }
        let Some(layouts) = self.layouts.as_ref() else {
            return;
        };
        for (_, gpu) in self.stored.iter_mut() {
            let Some(texture_id) = gpu.rebind.as_ref().map(|r| r.texture_id) else {
                continue;
            };
            let live = resources.has_texture(texture_id);
            if action == Revalidate::CheckEach && live {
                continue;
            }
            let binds = super::store::resolve_ribbon_texture(resources, layouts, Some(texture_id));
            super::store::rebind_ribbon(device, queue, &binds, gpu);
            // A freed id never comes back: ids are generational, so whatever
            // takes the slot next resolves through a different one. Forget the
            // ribbon's rebind record, and it stops being re-checked on every
            // later free.
            if !live {
                gpu.rebind = None;
            }
        }
    }
}

impl ItemTypePlugin for RibbonPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn init_gpu(
        &mut self,
        device: &crate::gpu::Device,
        _shared: &crate::plugin_api::SharedBindings<'_>,
    ) {
        self.layouts = Some(super::store::RibbonResources::new(device));
    }

    fn resident_bytes(&self) -> u64 {
        self.stored.allocated_bytes()
    }

    fn on_device_recreated(&mut self, device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.layouts = Some(super::store::RibbonResources::new(device));
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        self.revalidate_store(device, queue, ctx.resources);
        let items = items
            .as_any()
            .downcast_ref::<Vec<RibbonItem>>()
            .expect("ribbon collection is the SceneFrame field");
        let refs = ctx.refs_of::<RibbonRefItem>();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && refs.is_empty() {
            return Vec::new();
        }
        let layouts = self
            .layouts
            .get_or_insert_with(|| super::store::RibbonResources::new(device));
        let stored = &self.stored;
        let gpu = self
            .gpu
            .get_or_insert_with(|| RibbonGpu::new(device, ctx.resources, layouts));

        for item in items {
            if item.settings.hidden || item.positions.is_empty() || item.strip_lengths.is_empty() {
                continue;
            }
            let wireframe = ctx.wireframe_mode || item.settings.wireframe;
            let binds = super::store::resolve_ribbon_bindings(ctx.resources, layouts, item);
            let mut gpu_data = super::store::build_ribbon(device, queue, &binds, item, wireframe);
            if gpu_data.index_count == 0 {
                continue;
            }
            gpu_data.pick_id = item.settings.pick_id;
            gpu_data.model = item.model;
            gpu_data.cast_shadows = item.settings.cast_shadows;
            self.frame.push(build_frame_with(
                device,
                queue,
                &gpu.pick,
                gpu_data,
                ctx.outline_selected && item.settings.selected,
            ));
        }

        // Pre-uploaded references: the payload lives in the store, the model
        // matrix and pick id come from the reference, so one stored ribbon can
        // be drawn twice at two places under two ids.
        for ref_item in refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = stored.get(ref_item.source) else {
                continue;
            };
            let mut gpu_data = entry.clone();
            if gpu_data.index_count == 0 {
                continue;
            }
            queue.write_buffer(
                &gpu_data._uniform_buf,
                0,
                bytemuck::bytes_of(&ref_item.model),
            );
            gpu_data.pick_id = ref_item.settings.pick_id;
            gpu_data.model = ref_item.model;
            gpu_data.wireframe = ctx.wireframe_mode || ref_item.settings.wireframe;
            gpu_data.cast_shadows = ref_item.settings.cast_shadows;
            self.frame.push(build_frame_with(
                device,
                queue,
                &gpu.pick,
                gpu_data,
                ctx.outline_selected && ref_item.settings.selected,
            ));
        }
        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        let is_hdr = ctx.target_format == HDR_COLOR_FORMAT;
        for entry in &self.frame {
            let gd = &entry.gpu;
            if gd.index_count == 0 && gd.edge_index_count == 0 {
                continue;
            }
            // OIT-eligible ribbons draw in `paint_transparent` instead. That
            // pass is HDR-only, so on the LDR path they still draw here: there
            // is no OIT pass to route them to.
            if is_hdr && gd.oit_eligible {
                continue;
            }
            let key = RibbonKey {
                blend: gd.blend,
                wireframe: gd.wireframe,
                depth_write: gd.depth_write,
            };
            pass.set_pipeline(gpu.pipelines.get(key).for_format(is_hdr));
            draw_mesh(pass, gd);
        }
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn paint_transparent(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        for entry in self.frame.iter().filter(|f| f.gpu.oit_eligible) {
            let pipeline = match entry.gpu.blend {
                SpriteBlend::Premultiplied => &gpu.oit_pipeline_premultiplied,
                _ => &gpu.oit_pipeline,
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, &entry.gpu.uniform_bind_group, &[]);
            draw_solid_indexed(pass, &entry.gpu);
        }
    }

    fn cast_shadow_pass(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &ShadowCastContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        // No per-item cascade-frustum cull: the uploaded data carries no world
        // AABB, so every visible ribbon casts into every cascade its distance
        // would put it in. Group 1 reuses each ribbon's own solid-draw bind
        // group; only the leading `model` field is read by `ribbon_shadow.wgsl`.
        let mut bound = false;
        for entry in &self.frame {
            if !entry.gpu.cast_shadows || entry.gpu.index_count == 0 {
                continue;
            }
            if !bound {
                pass.set_pipeline(&gpu.shadow_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &entry.gpu.uniform_bind_group, &[]);
            draw_solid_indexed(pass, &entry.gpu);
        }
    }

    fn outline_mask(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        outline_mask_curve_mesh(pass, self.gpu.as_ref().map(|g| &g.pick), &self.frame);
    }

    /// Node proximity plus a ray test against the reconstructed swept quads, so
    /// a click anywhere on the ribbon surface registers, not only near its
    /// centre line.
    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        let levels = CurveLevels::from_mask(ctx.mask);
        if levels.is_empty() {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        let mut consider = |toi: f32, hit: PickHit| {
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                best = Some((toi, hit));
            }
        };
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            if levels.node || levels.strip || levels.object {
                let radius_px = radius_in_pixels(&item.positions, item.width * 0.5, ctx);
                if let Some(hit) = cpu_pick::node_hit(
                    levels,
                    ctx.click_pos,
                    item.settings.pick_id,
                    &item.positions,
                    &item.strip_lengths,
                    ctx.view_proj,
                    ctx.viewport_size,
                    radius_px.max(8.0),
                ) {
                    let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
                    consider(toi, hit);
                }
            }
            if levels.segment || levels.strip || levels.object {
                let mut best_t = f32::MAX;
                let mut best_seg: Option<(u32, glam::Vec3)> = None;
                for_each_quad(item, |seg_idx, c0, c1, c2, c3| {
                    // Both triangles, both faces: a ribbon is flat and has no
                    // front side.
                    let t = ray_triangle(ray.origin, ray.direction, c0, c1, c2)
                        .or_else(|| ray_triangle(ray.origin, ray.direction, c1, c3, c2))
                        .or_else(|| ray_triangle(ray.origin, ray.direction, c2, c1, c0))
                        .or_else(|| ray_triangle(ray.origin, ray.direction, c2, c3, c1));
                    if let Some(t) = t {
                        if t < best_t {
                            best_t = t;
                            best_seg = Some((seg_idx, ray.origin + ray.direction * t));
                        }
                    }
                    true
                });
                if let Some((seg_idx, world_pos)) = best_seg {
                    consider(
                        best_t,
                        cpu_pick::hit_for_segment(
                            levels,
                            item.settings.pick_id,
                            seg_idx,
                            world_pos,
                            &item.strip_lengths,
                        ),
                    );
                }
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> PickRectResult {
        let levels = CurveLevels::from_mask(ctx.mask);
        let mut result = PickRectResult::default();
        if levels.is_empty() {
            return result;
        }
        let in_rect = |p: glam::Vec2| {
            p.x >= ctx.rect_min.x
                && p.x <= ctx.rect_max.x
                && p.y >= ctx.rect_min.y
                && p.y <= ctx.rect_max.y
        };
        let proj = |p: glam::Vec3| project_to_screen(p, ctx.view_proj, ctx.viewport_size);
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let mut acc = RectAccumulator::new(levels, item.settings.pick_id);

            if levels.node || levels.strip || levels.object {
                for (node_idx, pos) in item.positions.iter().enumerate() {
                    let inside = proj(glam::Vec3::from(*pos)).is_some_and(in_rect);
                    if inside && !acc.node(&mut result, node_idx as u32, &item.strip_lengths) {
                        break;
                    }
                }
            }

            // Every quad edge against the rectangle, which also catches a quad
            // corner inside it through `segment_in_rect`'s endpoint test.
            if levels.segment || levels.strip || levels.object {
                let edge_hit = |a: Option<glam::Vec2>, b: Option<glam::Vec2>| match (a, b) {
                    (Some(a), Some(b)) => segment_in_rect(a, b, ctx.rect_min, ctx.rect_max),
                    (Some(a), None) => in_rect(a),
                    (None, Some(b)) => in_rect(b),
                    (None, None) => false,
                };
                for_each_quad(item, |seg_idx, c0, c1, c2, c3| {
                    let (s0, s1, s2, s3) = (proj(c0), proj(c1), proj(c2), proj(c3));
                    let hit = edge_hit(s0, s1)
                        || edge_hit(s2, s3)
                        || edge_hit(s0, s2)
                        || edge_hit(s1, s3);
                    if !hit {
                        return true;
                    }
                    acc.segment(&mut result, seg_idx, &item.strip_lengths)
                });
            }

            acc.finish(&mut result);
        }
        result
    }

    fn render_pick(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        render_pick_curve_mesh(pass, ctx, self.gpu.as_ref().map(|g| &g.pick), &self.frame);
    }

    fn resolve_sub_object(
        &self,
        pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        resolve_curve_sub_object(&self.frame, pick_id, primitive_index, mask)
    }
    fn sub_object_position(
        &self,
        items: &dyn PluginItemCollection,
        pick_id: PickId,
        sub_object: SubObjectRef,
    ) -> Option<glam::Vec3> {
        crate::renderer::picking::helpers::inline_point_position(
            items,
            pick_id,
            sub_object,
            |item: &RibbonItem| (item.settings.pick_id, &item.positions, &item.model),
        )
    }
}

/// Walk the swept quads of a ribbon, calling `f(segment, c0, c1, c2, c3)` with
/// the corners of each: `c0`/`c1` left and right at the segment start, `c2`/`c3`
/// at its end. Stops early when `f` returns `false`.
fn for_each_quad(
    item: &RibbonItem,
    mut f: impl FnMut(u32, glam::Vec3, glam::Vec3, glam::Vec3, glam::Vec3) -> bool,
) {
    let frames = lateral_frames(
        &item.positions,
        &item.strip_lengths,
        item.width,
        item.width_attribute.as_deref(),
        item.twist_attribute.as_deref(),
    );
    let strips = strips_or_single(&item.positions, &item.strip_lengths);
    let mut node_off = 0usize;
    let mut seg_off = 0u32;
    for &slen in &strips {
        let slen = slen as usize;
        for k in 0..slen.saturating_sub(1) {
            let (ia, ib) = (node_off + k, node_off + k + 1);
            let pa = glam::Vec3::from(item.positions[ia]);
            let pb = glam::Vec3::from(item.positions[ib]);
            let (ua, wa) = frames[ia];
            let (ub, wb) = frames[ib];
            if !f(
                seg_off + k as u32,
                pa + ua * wa,
                pa - ua * wa,
                pb + ub * wb,
                pb - ub * wb,
            ) {
                return;
            }
        }
        seg_off += slen.saturating_sub(1) as u32;
        node_off += slen;
    }
}

/// Reconstruct per-vertex (lateral direction, half-width) for a ribbon.
///
/// Replicates the parallel-transport frame the upload builds, so click and rect
/// picking test the actual swept quad rather than a midpoint proxy.
fn lateral_frames(
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    width: f32,
    width_attribute: Option<&[f32]>,
    twist_attribute: Option<&[[f32; 3]]>,
) -> Vec<(glam::Vec3, f32)> {
    // Initialise with a sentinel so any unvisited vertex has zero width.
    let mut frames: Vec<(glam::Vec3, f32)> = vec![(glam::Vec3::X, 0.0); positions.len()];
    let strips = strips_or_single(positions, strip_lengths);

    let mut node_off = 0usize;
    for &slen in &strips {
        let slen = slen as usize;
        if slen < 2 {
            node_off += slen;
            continue;
        }

        let pts: Vec<glam::Vec3> = positions[node_off..node_off + slen]
            .iter()
            .map(|&p| glam::Vec3::from(p))
            .collect();

        let t0 = (pts[1] - pts[0]).normalize_or_zero();
        if t0.length_squared() < 1e-10 {
            node_off += slen;
            continue;
        }
        let ref_v = if t0.x.abs() < 0.9 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let mut u = t0.cross(ref_v).normalize();

        for k in 0..slen {
            let tangent = if k + 1 < slen {
                (pts[k + 1] - pts[k]).normalize_or_zero()
            } else {
                (pts[k] - pts[k - 1]).normalize_or_zero()
            };

            // Parallel transport: rotate u to stay perpendicular to the new
            // tangent.
            if k > 0 {
                let t_prev = (pts[k] - pts[k - 1]).normalize_or_zero();
                let axis = t_prev.cross(tangent);
                let sin_a = axis.length().min(1.0);
                if sin_a > 1e-6 {
                    let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                    let ax = axis / sin_a;
                    u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                    u = u.normalize_or_zero();
                }
            }

            // Apply per-point twist if supplied.
            let mut lateral = u;
            if let Some(twist) = twist_attribute {
                if let Some(&tv) = twist.get(node_off + k) {
                    let tv = glam::Vec3::from(tv);
                    let proj = tv - tangent * tangent.dot(tv);
                    if proj.length_squared() > 1e-10 {
                        lateral = proj.normalize();
                    }
                }
            }

            let half_w = width_attribute
                .and_then(|w| w.get(node_off + k).copied())
                .unwrap_or(width)
                * 0.5;
            frames[node_off + k] = (lateral, half_w);
        }
        node_off += slen;
    }
    frames
}

impl RibbonPlugin {
    /// Build one curve's GPU data against the plugin's layout.
    fn build(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: &RibbonItem,
    ) -> super::store::StreamtubeGpuData {
        let layouts = self
            .layouts
            .get_or_insert_with(|| super::store::RibbonResources::new(device));
        let binds = super::store::resolve_ribbon_bindings(resources, layouts, item);
        super::store::build_ribbon(device, queue, &binds, item, false)
    }

    /// Pre-upload a curve and return its handle.
    pub(crate) fn upload(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: &RibbonItem,
    ) -> RibbonId {
        let gpu = self.build(device, queue, resources, item);
        self.stored.insert_sized(gpu)
    }

    /// Drop a stored curve. `false` when the handle does not resolve.
    pub(crate) fn drop_stored(&mut self, id: RibbonId) -> bool {
        self.stored.remove(id).is_some()
    }

    /// Replace the geometry behind a live handle, keeping the handle.
    pub(crate) fn replace(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        id: RibbonId,
        item: &RibbonItem,
    ) -> bool {
        if !self.stored.contains(id) {
            return false;
        }
        let gpu = self.build(device, queue, resources, item);
        self.stored.replace_sized(id, gpu).is_some()
    }

    /// Sweep the curve mesh on a worker thread. The handle is minted when
    /// [`take_upload_result`](Self::take_upload_result) collects the job.
    ///
    /// The layout and the colourmap the upload binds are resolved here and
    /// cloned into the worker: they are the renderer's and a worker has no
    /// `DeviceResources` borrow.
    pub(crate) fn begin_upload(
        &mut self,
        jobs: &crate::resources::Jobs<'_>,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        item: RibbonItem,
    ) -> crate::resources::JobId {
        let layouts = self
            .layouts
            .get_or_insert_with(|| super::store::RibbonResources::new(device));
        let item = item;
        let binds = super::store::resolve_ribbon_bindings(resources, layouts, &&item);
        let device = device.clone();
        let queue = queue.clone();
        jobs.submit_cpu(move || super::store::build_ribbon(&device, &queue, &binds, &item, false))
    }

    /// Store the curve a finished job built and hand back its handle.
    pub(crate) fn take_upload_result(
        &mut self,
        jobs: &crate::resources::Jobs<'_>,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<RibbonId> {
        match jobs.status(id) {
            crate::resources::UploadStatus::Pending { .. } => {
                Err(crate::error::ViewportError::JobNotReady)
            }
            crate::resources::UploadStatus::Failed(e) => Err(e),
            _ => match jobs.take::<super::store::StreamtubeGpuData>(id) {
                Some(gpu) => Ok(self.stored.insert_sized(gpu)),
                None => Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                }),
            },
        }
    }

    /// Number of items the last `prepare` produced draw data for.
    #[cfg(test)]
    pub(crate) fn drawn_count(&self) -> usize {
        self.frame.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same completeness guarantee as the mesh-family `PipelineVariantSet`
    /// tests: every key in `RibbonKey::all()` must land in its own slot, so a
    /// built set resolves each through `get()` without aliasing. Covers the
    /// `blend x wireframe x depth_write` cross product.
    #[test]
    fn all_keys_are_distinct_and_densely_slotted() {
        let keys: Vec<RibbonKey> = RibbonKey::all().collect();
        assert_eq!(
            keys.len(),
            12,
            "RibbonKey has blend(3) x wireframe x depth_write = 12 keys"
        );

        let mut seen_keys = std::collections::HashSet::new();
        let mut seen_slots = std::collections::HashSet::new();
        for key in keys {
            assert!(
                seen_keys.insert(key),
                "all() yielded {key:?} more than once"
            );
            let slot = key.slot();
            assert!(slot < 12, "{key:?} slotted out of range: {slot}");
            assert!(
                seen_slots.insert(slot),
                "{key:?} collided with another key at slot {slot}"
            );
        }
    }

    /// `build` must place each variant at its own `slot`, not in iteration
    /// order. The two orderings differ, so pushing in iteration order hands out
    /// a pipeline belonging to a different key at draw time: an opaque ribbon
    /// drew through the additive wireframe pipeline and rendered as a zigzag of
    /// lines. The slot-distinctness test above does not catch it, because the
    /// slots are fine; it is the placement that was wrong.
    #[test]
    fn build_places_each_variant_at_its_own_slot() {
        // The same placement `RibbonVariantSet::build` uses, standing the
        // pipelines in for the key each slot was built from.
        let placed = place_by_slot(|key| key);
        for key in RibbonKey::all() {
            assert_eq!(
                placed[key.slot()],
                key,
                "slot {} does not hold the variant built for {key:?}",
                key.slot()
            );
        }
    }

    /// The reason the placement matters: `all()` order and `slot()` order are
    /// genuinely different, so collecting in iteration order is wrong rather
    /// than merely unidiomatic. If this ever stops being true the placement is
    /// still correct, but the hazard it guards has gone.
    #[test]
    fn iteration_order_differs_from_slot_order() {
        let iteration: Vec<usize> = RibbonKey::all().map(|k| k.slot()).collect();
        let dense: Vec<usize> = (0..12).collect();
        assert_ne!(
            iteration, dense,
            "all() now yields keys in slot order; place_by_slot still correct"
        );
    }
}
