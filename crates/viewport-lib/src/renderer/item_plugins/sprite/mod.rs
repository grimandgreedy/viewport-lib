//! The sprite item type as an [`ItemTypePlugin`]: camera-facing textured
//! billboards, optionally lit, with soft-particle fade against scene depth and
//! optional refractive distortion. Consumers submit [`SpriteItem`]s on
//! `SceneFrame::sprite_items`, or reference a pre-uploaded batch with
//! [`SpriteSetRefItem`] on `SceneFrame::sprite_set_refs` or
//! [`SpriteInstanceSetRefItem`] on `SceneFrame::sprite_instance_set_refs`; the
//! renderer routes all three fields to this plugin.
//!
//! A sprite batch draws in one of four places, chosen by its own fields rather
//! than by a separate setting:
//!
//! - writes depth: the opaque scene pass, through `paint`.
//! - refractive: its own pass over a copy of the scene colour, through
//!   `encode` at [`EncoderScope::AfterOpaque`].
//! - blended with no soft fade and no refraction: the OIT pass, through
//!   `paint_transparent`.
//! - anything else transparent: the read-only-depth pass, through
//!   `paint_depth_read`, which is the only one of the four that can sample
//!   scene depth for the soft fade.

mod pipeline;

use crate::plugin_api::{
    DepthReadContext, EncoderScope, EncoderScopeContext, ItemFrameContext, ItemTypePlugin,
    OutlineMaskContext, PaintContext, PickContext, PickPassContext, PickRay, PluginItemCollection,
    RectPickContext,
};
use crate::renderer::{
    PickHit, PickId, PickMask, PickRectResult, SpriteInstanceSetRefItem, SpriteItem,
    SpriteSetRefItem, SpriteSizeMode, SubObjectRef,
};
use crate::resources::SpriteGpuData;

pub(crate) const TYPE_NAME: &str = "viewport.sprite";

impl PluginItemCollection for Vec<SpriteItem> {
    fn len(&self) -> usize {
        self.len()
    }
    fn item_settings(&self, index: usize) -> &crate::scene::material::ItemSettings {
        &self[index].settings
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl PluginItemCollection for Vec<SpriteSetRefItem> {
    fn len(&self) -> usize {
        self.len()
    }
    fn item_settings(&self, index: usize) -> &crate::scene::material::ItemSettings {
        &self[index].settings
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl PluginItemCollection for Vec<SpriteInstanceSetRefItem> {
    fn len(&self) -> usize {
        self.len()
    }
    fn item_settings(&self, index: usize) -> &crate::scene::material::ItemSettings {
        &self[index].settings
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// One selected batch's outline coverage: an index into `frame` plus, for a
/// sub-object selection, the instances to draw rather than the whole batch.
struct SpriteOutline {
    frame_index: usize,
    instances: Option<Vec<u32>>,
}

#[derive(Default)]
pub(crate) struct SpritePlugin {
    gpu: Option<pipeline::SpriteGpu>,
    /// Per drawn batch, rebuilt each prepare: inline items first, then the two
    /// reference forms, the order the upload loop used before they met here.
    frame: Vec<SpriteGpuData>,
    /// Every inline item from the last prepared frame, for the CPU pick. The
    /// reference forms are not here: their positions live on the GPU, so they
    /// answer the GPU pick only.
    pick_items: Vec<SpriteItem>,
    /// Selection-outline coverage for this frame.
    outlines: Vec<SpriteOutline>,
    /// Per-batch group-2 pick-id bind groups, parallel to `frame`.
    pick_bind_groups: Vec<Option<crate::gpu::BindGroup>>,
    /// Scene-colour resolve targets for the refractive pass, one per viewport.
    /// Behind a lock because `encode` takes `&self` but the target has to be
    /// allocated the first time a viewport draws a refractive sprite, and
    /// reallocated when the scene size changes.
    refraction: std::sync::Mutex<Vec<Option<RefractionResolve>>>,
}

/// A viewport's scene-colour copy, allocated the first frame that viewport
/// draws a refractive sprite and reallocated when the scene size changes.
struct RefractionResolve {
    texture: crate::gpu::Texture,
    view: crate::gpu::TextureView,
    size: (u32, u32),
}

const SCOPES: &[EncoderScope] = &[EncoderScope::AfterOpaque];

impl SpritePlugin {
    /// Draw one bucket through the keyed variant set, binding each pipeline
    /// once and only for the keys the frame actually uses.
    fn draw_bucket(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        hdr: bool,
        group2: &crate::gpu::BindGroup,
        wanted: impl Fn(&SpriteGpuData) -> bool,
    ) {
        let Some(gpu) = &self.gpu else { return };
        for key in pipeline::SpriteKey::all() {
            let dual = gpu.pipelines.get(key);
            let mut bound = false;
            for sprite in self.frame.iter().filter(|s| wanted(s)) {
                if sprite.depth_write != key.depth_write
                    || sprite.blend != key.blend
                    || sprite.lit != key.lit
                {
                    continue;
                }
                if !bound {
                    pass.set_pipeline(dual.for_format(hdr));
                    pass.set_bind_group(2, group2, &[]);
                    bound = true;
                }
                pass.set_bind_group(1, &sprite.bind_group, &[]);
                if key.lit {
                    let normal_bg = sprite
                        .lit_normal_bg
                        .as_ref()
                        .unwrap_or(&gpu.lit_fallback_bg);
                    pass.set_bind_group(3, normal_bg, &[]);
                }
                pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
                pass.draw(0..6, 0..sprite.sprite_count);
            }
        }
    }
}

impl ItemTypePlugin for SpritePlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.gpu = None;
        self.frame.clear();
        self.outlines.clear();
        self.pick_bind_groups.clear();
        self.refraction.lock().unwrap().clear();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        self.outlines.clear();
        self.pick_bind_groups.clear();
        let items = items
            .as_any()
            .downcast_ref::<Vec<SpriteItem>>()
            .expect("sprite collection is the SceneFrame field");
        let set_refs = ctx.refs_of::<SpriteSetRefItem>();
        let instance_refs = ctx.refs_of::<SpriteInstanceSetRefItem>();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(items);
        if items.is_empty() && set_refs.is_empty() && instance_refs.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::SpriteGpu::new(device, ctx.resources));

        // Inline items. `outlines` indexes into `frame`, not into the
        // submitted items, so a hidden batch earlier in the frame cannot shift
        // a later batch's outline onto the wrong draw data.
        for item in items {
            if item.settings.hidden || item.positions.is_empty() {
                continue;
            }
            let mut gd = ctx.resources.upload_sprite(device, queue, item);
            gd.wireframe = ctx.wireframe_mode || item.settings.wireframe;
            let frame_index = self.frame.len();
            self.frame.push(gd);
            if ctx.outline_selected {
                if item.settings.selected {
                    self.outlines.push(SpriteOutline {
                        frame_index,
                        instances: None,
                    });
                } else if item.settings.pick_id != PickId::NONE {
                    let instances: Vec<u32> = ctx
                        .sub_selection
                        .iter()
                        .flat_map(|s| s.items.iter())
                        .filter_map(|(node_id, sub)| match sub {
                            SubObjectRef::Instance(idx) if *node_id == item.settings.pick_id.0 => {
                                Some(*idx)
                            }
                            _ => None,
                        })
                        .collect();
                    if !instances.is_empty() {
                        self.outlines.push(SpriteOutline {
                            frame_index,
                            instances: Some(instances),
                        });
                    }
                }
            }
        }

        // Pre-uploaded sprite sets. The stored entry already carries its own
        // pick id and bind group; the reference only re-states visibility.
        for ref_item in set_refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = ctx.resources.content.sprite_set_store.get(ref_item.source) else {
                continue;
            };
            let mut gd = entry.clone();
            gd.wireframe = ctx.wireframe_mode || ref_item.settings.wireframe;
            self.frame.push(gd);
        }

        // Pre-uploaded instance sets.
        for ref_item in instance_refs {
            if ref_item.settings.hidden {
                continue;
            }
            let Some(entry) = ctx
                .resources
                .content
                .sprite_instance_set_store
                .get(ref_item.source)
            else {
                continue;
            };
            let mut gd = entry.clone();
            gd.wireframe = ctx.wireframe_mode || ref_item.settings.wireframe;
            self.frame.push(gd);
        }

        // Group-2 pick-id bind groups, one per pickable batch.
        self.pick_bind_groups = self
            .frame
            .iter()
            .map(|s| {
                (s.pick_id != PickId::NONE && s.sprite_count > 0)
                    .then(|| gpu.pick_bind_group(device, queue, s.pick_id))
            })
            .collect();

        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if !self.frame.iter().any(is_opaque) {
            return;
        }
        // The depth attachment is writable in this pass and is the same buffer
        // the soft fade would sample, so group 2 gets the 1x1 fallback. A
        // depth-writing batch has no soft fade to lose by it.
        let hdr = ctx.target_format == crate::resources::HDR_COLOR_FORMAT;
        self.draw_bucket(pass, hdr, &gpu.soft_fallback_bg, is_opaque);
    }

    fn draws_ldr(&self) -> bool {
        true
    }

    fn draws_depth_read(&self) -> bool {
        self.frame.iter().any(is_soft)
    }

    fn paint_depth_read(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &DepthReadContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        if self.gpu.is_none() {
            return;
        }
        // The depth attachment is read-only here, so the live scene depth can
        // be sampled for the soft fade. The ready-made group matches the layout
        // these pipelines were built against.
        self.draw_bucket(pass, true, ctx.scene_depth_bind_group, is_soft);
    }

    fn paint_transparent(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        for sprite in self.frame.iter().filter(|s| s.oit_eligible && !s.wireframe) {
            let pipeline = match (sprite.lit, sprite.blend) {
                (true, crate::renderer::SpriteBlend::Premultiplied) => {
                    &gpu.oit_lit_pipeline_premultiplied
                }
                (true, _) => &gpu.oit_lit_pipeline,
                (false, crate::renderer::SpriteBlend::Premultiplied) => {
                    &gpu.oit_pipeline_premultiplied
                }
                (false, _) => &gpu.oit_pipeline,
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, &sprite.bind_group, &[]);
            if sprite.lit {
                let normal_bg = sprite
                    .lit_normal_bg
                    .as_ref()
                    .unwrap_or(&gpu.lit_fallback_bg);
                pass.set_bind_group(2, normal_bg, &[]);
            }
            pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
            pass.draw(0..6, 0..sprite.sprite_count);
        }
    }

    fn outline_mask(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if self.outlines.is_empty() {
            return;
        }
        // The billboard quads themselves go into the mask, so the outline
        // follows each sprite's shape and per-instance size.
        pass.set_pipeline(&gpu.outline_mask_pipeline);
        for outline in &self.outlines {
            let Some(sprite) = self.frame.get(outline.frame_index) else {
                continue;
            };
            pass.set_bind_group(1, &sprite.bind_group, &[]);
            pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
            match &outline.instances {
                None => pass.draw(0..6, 0..sprite.sprite_count),
                Some(indices) => {
                    for &i in indices {
                        pass.draw(0..6, i..i + 1);
                    }
                }
            }
        }
    }

    fn encoder_scopes(&self) -> &[EncoderScope] {
        SCOPES
    }

    fn encode(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if !self
            .frame
            .iter()
            .any(|s| s.refraction_strength > 0.0 && !s.wireframe)
        {
            return;
        }
        // A pass cannot sample the colour target it is drawing into, so the
        // scene colour is copied aside first and the refractive sprites sample
        // the copy.
        let mut cache = self.refraction.lock().unwrap();
        while cache.len() <= ctx.viewport_index {
            cache.push(None);
        }
        let [w, h] = ctx.scene_size;
        let stale = cache[ctx.viewport_index]
            .as_ref()
            .is_none_or(|r| r.size != (w, h));
        if stale {
            let texture = ctx.device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("sprite_refraction_resolve"),
                size: crate::gpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: crate::resources::HDR_COLOR_FORMAT,
                usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                    | crate::gpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
            cache[ctx.viewport_index] = Some(RefractionResolve {
                texture,
                view,
                size: (w, h),
            });
        }
        let resolve = cache[ctx.viewport_index].as_ref().expect("just allocated");
        encoder.copy_texture_to_texture(
            ctx.scene_colour_texture.as_image_copy(),
            resolve.texture.as_image_copy(),
            crate::gpu::Extent3d {
                width: resolve.size.0,
                height: resolve.size.1,
                depth_or_array_layers: 1,
            },
        );
        let bind_group = ctx
            .device
            .create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("sprite_refraction_bg"),
                layout: &gpu.refraction_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(&resolve.view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(&gpu.refraction_sampler),
                    },
                ],
            });
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("sprite_refraction_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: ctx.scene_colour,
                resolve_target: None,
                ops: crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: ctx.scene_depth,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&gpu.refraction_pipeline);
        pass.set_bind_group(0, ctx.camera_bind_group, &[]);
        pass.set_bind_group(2, &bind_group, &[]);
        for sprite in &self.frame {
            if sprite.refraction_strength <= 0.0 || sprite.wireframe {
                continue;
            }
            pass.set_bind_group(1, &sprite.bind_group, &[]);
            pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
            pass.draw(0..6, 0..sprite.sprite_count);
        }
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        let wants_instance = ctx.mask.intersects(PickMask::INSTANCE);
        if !wants_instance && !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            let radius_px = sprite_radius_px(item, &model, ctx.view_proj, ctx.viewport_size);
            let Some(mut hit) = crate::interaction::query::picking::pick_gaussian_splat_cpu(
                ctx.click_pos,
                item.settings.pick_id.0,
                &item.positions,
                model,
                ctx.view_proj,
                ctx.viewport_size,
                radius_px,
            ) else {
                continue;
            };
            if !wants_instance {
                hit.sub_object = None;
            }
            let toi = (hit.world_pos - ray.origin).dot(ray.direction).max(0.0);
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                best = Some((toi, hit));
            }
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> PickRectResult {
        let mut result = PickRectResult::default();
        let wants_instance = ctx.mask.intersects(PickMask::INSTANCE);
        let wants_object = ctx.mask.intersects(PickMask::OBJECT);
        if !wants_instance && !wants_object {
            return result;
        }
        for item in &self.pick_items {
            if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.model);
            let id = item.settings.pick_id.0;
            let mut item_hit = false;
            for (index, pos) in item.positions.iter().enumerate() {
                let world = model.transform_point3(glam::Vec3::from(*pos));
                let Some(p) = crate::plugin_api::pick_helpers::project_to_screen(
                    world,
                    ctx.view_proj,
                    ctx.viewport_size,
                ) else {
                    continue;
                };
                if p.x < ctx.rect_min.x
                    || p.x > ctx.rect_max.x
                    || p.y < ctx.rect_min.y
                    || p.y > ctx.rect_max.y
                {
                    continue;
                }
                if wants_instance {
                    result
                        .elements
                        .push((id, SubObjectRef::Instance(index as u32)));
                }
                item_hit = true;
            }
            if wants_object && item_hit {
                result.objects.push(id);
            }
        }
        result
    }

    fn render_pick(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        if !ctx.mask.intersects(PickMask::OBJECT | PickMask::INSTANCE) {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let mut bound = false;
        for (sprite, pick_bg) in self.frame.iter().zip(self.pick_bind_groups.iter()) {
            let Some(pick_bg) = pick_bg else { continue };
            if !bound {
                pass.set_pipeline(&gpu.pick_pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &sprite.bind_group, &[]);
            pass.set_bind_group(2, pick_bg, &[]);
            pass.set_vertex_buffer(0, sprite.vertex_buffer.slice(..));
            pass.draw(0..6, 0..sprite.sprite_count);
        }
    }

    /// A quad outline per billboard, showing where each sprite sits and how
    /// big it is on screen: sub-structure rather than bounds.
    ///
    /// Capped for the same reason as the splat rings, and past the cap a batch
    /// falls back to one axis-aligned box around every sprite in it. That box
    /// is a weak stand-in and is kept only because it is what the batch has
    /// always drawn.
    fn wireframe_polylines(
        &self,
        items: &dyn PluginItemCollection,
        ctx: &ItemFrameContext<'_>,
    ) -> Vec<crate::renderer::PolylineItem> {
        /// Above this many sprites a quad each stops being readable.
        const MAX_OUTLINED_SPRITES: usize = 100;

        let Some(sprites) = items.as_any().downcast_ref::<Vec<SpriteItem>>() else {
            return Vec::new();
        };
        sprites
            .iter()
            .filter(|item| !item.settings.hidden && (ctx.wireframe_mode || item.settings.wireframe))
            .filter(|item| !item.positions.is_empty())
            .map(|item| {
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                if item.positions.len() <= MAX_OUTLINED_SPRITES {
                    sprite_quad_outlines_polyline(item, ctx.camera, ctx.viewport_size, model)
                } else {
                    sprite_bounds_polyline(item, model)
                }
            })
            .collect()
    }

    fn resolve_sub_object(
        &self,
        _pick_id: PickId,
        primitive_index: u32,
        _world_pos: glam::Vec3,
        mask: PickMask,
    ) -> Option<SubObjectRef> {
        // The pick fragment writes the sprite's instance index into the
        // primitive channel; no device feature involved.
        mask.intersects(PickMask::INSTANCE)
            .then_some(SubObjectRef::Instance(primitive_index))
    }
}

/// Batches that draw in the opaque scene pass.
fn is_opaque(s: &SpriteGpuData) -> bool {
    !s.wireframe && s.depth_write && s.refraction_strength <= 0.0
}

/// Batches that draw in the read-only-depth pass: transparent, needing the
/// live scene depth for the soft fade, so neither OIT nor refraction.
fn is_soft(s: &SpriteGpuData) -> bool {
    !s.wireframe && !s.depth_write && s.refraction_strength <= 0.0 && !s.oit_eligible
}

/// Screen-space pick radius for one batch, matching how the vertex shader
/// sizes the billboard: screen-space batches are already in pixels, world-space
/// batches are measured by projecting the batch centroid.
fn sprite_radius_px(
    item: &SpriteItem,
    model: &glam::Mat4,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> f32 {
    match item.size_mode {
        SpriteSizeMode::ScreenSpace => (item.default_size * 0.5).max(4.0),
        SpriteSizeMode::WorldSpace => {
            let n = item.positions.len() as f32;
            let centroid = model.transform_point3(
                item.positions
                    .iter()
                    .map(|p| glam::Vec3::from(*p))
                    .sum::<glam::Vec3>()
                    / n,
            );
            // Measure at the batch centroid rather than the model origin, so a
            // batch whose sprites sit far from its origin still gets a pixel
            // radius that matches what is on screen.
            let world_r = (item.default_size * 0.5).max(0.01);
            let p0 = view_proj * centroid.extend(1.0);
            let p1 = view_proj * (centroid + glam::Vec3::X * world_r).extend(1.0);
            if p0.w.abs() > 1e-6 && p1.w.abs() > 1e-6 {
                let n0 = glam::Vec2::new(p0.x, p0.y) / p0.w;
                let n1 = glam::Vec2::new(p1.x, p1.y) / p1.w;
                ((n1 - n0).length() * 0.5 * viewport_size.x.max(viewport_size.y)).max(4.0)
            } else {
                (world_r * 100.0_f32).max(4.0)
            }
        }
    }
}

/// One axis-aligned box around every sprite position in the batch, for a batch
/// too large to outline individually.
fn sprite_bounds_polyline(item: &SpriteItem, model: glam::Mat4) -> crate::renderer::PolylineItem {
    let mut mn = glam::Vec3::splat(f32::INFINITY);
    let mut mx = glam::Vec3::splat(f32::NEG_INFINITY);
    for pos in &item.positions {
        let wp = model.transform_point3(glam::Vec3::from(*pos));
        mn = mn.min(wp);
        mx = mx.max(wp);
    }
    let corners: [[f32; 3]; 8] = [
        [mn.x, mn.y, mn.z],
        [mx.x, mn.y, mn.z],
        [mn.x, mx.y, mn.z],
        [mx.x, mx.y, mn.z],
        [mn.x, mn.y, mx.z],
        [mx.x, mn.y, mx.z],
        [mn.x, mx.y, mx.z],
        [mx.x, mx.y, mx.z],
    ];
    crate::renderer::obb_wireframe_polyline(&corners, [0.75, 0.75, 0.75, 1.0])
}

/// Generate 4-edge quad outlines for each sprite in a batch.
///
/// Mirrors the sprite vertex shader corner computation:
/// - WorldSpace sprites: expand along camera right/up by half-size in world units.
/// - ScreenSpace sprites: convert NDC corners back to world space via inv_view_proj.
fn sprite_quad_outlines_polyline(
    item: &SpriteItem,
    camera: &crate::RenderCamera,
    viewport_size: glam::Vec2,
    model: glam::Mat4,
) -> crate::renderer::PolylineItem {
    let view = &camera.view;
    // Row 0 of the view matrix = camera right in world space.
    // Row 1 of the view matrix = camera up in world space.
    // glam Mat4 is column-major: view[col][row], matching view[0][0]/view[1][0]/view[2][0] in WGSL.
    let cam_right = glam::Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
    let cam_up = glam::Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);

    let view_proj = camera.view_proj();
    let inv_view_proj = view_proj.inverse();
    let [vw, vh] = [viewport_size.x, viewport_size.y];
    let is_world_space = matches!(
        item.size_mode,
        crate::renderer::types::SpriteSizeMode::WorldSpace
    );

    // BL -> BR -> TR -> TL -> BL: a closed rectangle (4 edges, 5 positions per strip).
    const CORNERS: [(f32, f32); 5] = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ];

    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();

    for i in 0..item.positions.len() {
        let world_pos = model.transform_point3(glam::Vec3::from(item.positions[i]));
        let size = if i < item.sizes.len() {
            item.sizes[i]
        } else {
            item.default_size
        };
        let rotation = if i < item.rotations.len() {
            item.rotations[i]
        } else {
            0.0
        };
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();
        let half = size * 0.5;

        let mut pts: Vec<[f32; 3]> = Vec::with_capacity(5);
        let mut ok = true;

        if is_world_space {
            for (cx, cy) in CORNERS {
                let rx = cos_r * cx - sin_r * cy;
                let ry = sin_r * cx + cos_r * cy;
                let p = world_pos + cam_right * (rx * half) + cam_up * (ry * half);
                pts.push(p.to_array());
            }
        } else {
            let clip_center = view_proj * world_pos.extend(1.0);
            if clip_center.w <= 0.0 {
                // Behind camera -- skip this sprite.
                ok = false;
            } else {
                let ndc_center =
                    glam::Vec3::new(clip_center.x, clip_center.y, clip_center.z) / clip_center.w;
                for (cx, cy) in CORNERS {
                    let rx = cos_r * cx - sin_r * cy;
                    let ry = sin_r * cx + cos_r * cy;
                    let ndc = glam::Vec3::new(
                        ndc_center.x + rx * half / vw,
                        ndc_center.y + ry * half / vh,
                        ndc_center.z,
                    );
                    let world_h = inv_view_proj * ndc.extend(1.0);
                    if world_h.w.abs() < 1e-7 {
                        ok = false;
                        break;
                    }
                    pts.push(
                        (glam::Vec3::new(world_h.x, world_h.y, world_h.z) / world_h.w).to_array(),
                    );
                }
            }
        }

        if ok && pts.len() == 5 {
            all_positions.extend_from_slice(&pts);
            strip_lengths.push(5);
        }
    }

    crate::renderer::PolylineItem {
        positions: all_positions,
        strip_lengths,
        default_colour: [0.75, 0.75, 0.75, 1.0].into(),
        line_width: 1.0,
        ..crate::renderer::PolylineItem::default()
    }
}
