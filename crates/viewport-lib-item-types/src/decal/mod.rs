//! The screen-space decal item type as an [`ItemTypePlugin`]: textures
//! projected onto opaque surfaces along a box. A decal reads the surface mask
//! to stay off surfaces which opted out. Consumers submit [`DecalItem`]s with
//! `SceneFrame::items_mut`.
//!
//! All three passes are encoded from [`ItemTypePlugin::encode`] at
//! [`EncoderScope::OnOpaqueSurfaces`], in the order they have to run: the
//! colour projection, then the outline mask and edge trace when a decal is
//! selected.

mod live;
mod pipeline;
mod types;

pub use live::{DecalHandle, LiveDecal, LiveDecals};
pub use types::{CylindricalFacing, DecalAnimation, DecalBlendMode, DecalItem, DecalProjection};
use viewport_lib::plugin_api::pick_helpers::{ray_unit_box_toi, segment_in_rect};
use viewport_lib::plugin_api::{
    EncoderScope, EncoderScopeContext, ItemCollections, ItemFrameContext, ItemTypePlugin,
    PickContext, PickPassContext, PickRay, PluginItem, RectPickContext,
};
use viewport_lib::renderer::{PickHit, PickId, PickMask};
use viewport_lib::resources::{ResourceGate, Revalidate, TextureId};

pub const TYPE_NAME: &str = "vpl.decal";

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    vec![
        ("decal.wgsl", pipeline::decal_source()),
        ("decal_outline_mask.wgsl", pipeline::outline_mask_source()),
        ("decal_pick.wgsl", pipeline::pick_source()),
    ]
}

impl PluginItem for DecalItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

/// The unit box corners in (x, y, z) bit order, and the twelve edges joining
/// corners that differ in exactly one axis. Shared by the rect pick.
const CORNERS: [[f32; 3]; 8] = [
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
];
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
];

/// The textures one cached decal's bind group names, so the cache can tell
/// when a free has left it pointing at a view that is gone.
type BoundTextures = [Option<TextureId>; 5];

/// The decal item type. Register it with
/// [`ViewportRenderer::with_item_type_plugin`](viewport_lib::renderer::ViewportRenderer::with_item_type_plugin),
/// or through [`install`](crate::install) with the rest of this crate.
#[derive(Default)]
pub struct DecalPlugin {
    /// Layouts and pipelines, made by `warm` or the first prepare with a
    /// decal. Stays `None` on a device that cannot build the decal pass.
    gpu: Option<pipeline::DecalGpu>,
    /// This frame's draw list, in `sort_key` order, built in `prepare`.
    draws: Vec<pipeline::DecalGpuItem>,
    /// GPU resources cached across frames, keyed by decal content hash, so an
    /// unchanged decal rebuilds no uniform buffer and no bind group.
    cache: std::collections::HashMap<u64, (pipeline::DecalGpuItem, BoundTextures)>,
    /// Resource epochs the cache was last validated against.
    deps_gate: ResourceGate,
    /// Items retained from `prepare` for the out-of-band CPU pick answers.
    pick_items: Vec<DecalItem>,
    /// Cache miss and hit counts from the last prepare.
    uploads: u32,
    reused: u32,
    /// The outline mask target and its edge bind group, keyed by target size.
    /// Behind a lock because `encode` runs from a shared borrow but the target
    /// set is rebuilt when the viewport resizes.
    outline_targets: std::sync::Mutex<Option<pipeline::DecalOutlineTargets>>,
}

impl DecalPlugin {
    /// How the cross-frame resource cache did on the last prepared frame, as
    /// `(uploads, reused)`.
    ///
    /// `uploads` counts decals whose uniform buffer and bind group were built
    /// that frame, `reused` the ones served from the cache. With static decals
    /// `uploads` is zero after the first frame; a value near the decal count
    /// every frame means the cache is missing.
    pub fn cache_stats(&self) -> (u32, u32) {
        (self.uploads, self.reused)
    }
}

impl ItemTypePlugin for DecalPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    /// Request every decal pipeline ahead of the first frame that submits a
    /// decal. Decals tend to appear mid-session (impact marks, scorches), so
    /// without a warm-up that frame would wait on the compile or go without
    /// its decals.
    fn warm(
        &mut self,
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::DeviceResources,
    ) {
        if self.gpu.is_none() {
            self.gpu = pipeline::DecalGpu::new(device, resources);
        }
        if let Some(gpu) = &self.gpu {
            gpu.pipelines.request_all();
        }
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        _queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        let decals = items.of::<DecalItem>();
        let res = ctx.resources;

        self.draws.clear();
        self.pick_items.clear();
        let mut uploads = 0u32;
        let mut reused = 0u32;

        if !decals.is_empty() && self.gpu.is_none() {
            self.gpu = pipeline::DecalGpu::new(device, res);
        }
        if decals.is_empty() {
            // No decals this frame: drop any cached GPU resources.
            self.cache.clear();
        } else if let Some(decal_gpu) = &self.gpu {
            // Cached entries hold bind groups over texture views, so a free or
            // a replace since the last frame invalidates them: a free drops
            // only the entries whose deps no longer resolve, a replace drops
            // everything, since a view swapped behind a live id cannot be
            // detected per entry.
            match self.deps_gate.poll(res) {
                Revalidate::RebuildAll => self.cache.clear(),
                Revalidate::CheckEach => {
                    self.cache.retain(|_, (_, bound)| {
                        bound.iter().flatten().all(|id| res.has_texture(*id))
                    });
                }
                Revalidate::Valid => {}
            }
            // Stable sort so equal-key decals stay in submission order.
            let mut sorted: Vec<&DecalItem> = decals.iter().collect();
            sorted.sort_by_key(|d| d.sort_key);
            let mut seen: std::collections::HashSet<u64> =
                std::collections::HashSet::with_capacity(sorted.len());
            for item in sorted {
                if item.settings.hidden || item.settings.opacity <= 0.0 {
                    continue;
                }
                // Apply appearance.opacity on top of the item's own alpha.
                let mut effective = item.clone();
                effective.alpha *= item.settings.opacity;
                let key = pipeline::hash_decal_item(&effective, &|id| res.has_texture(id));
                match self.cache.entry(key) {
                    std::collections::hash_map::Entry::Occupied(e) => {
                        // `selected` is not part of the cache key, so refresh it
                        // on the reused clone to reflect this frame's selection.
                        let mut gpu = e.get().0.clone();
                        gpu.selected = effective.settings.selected;
                        self.draws.push(gpu);
                        reused += 1;
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        use viewport_lib::resources::TextureSlot;
                        res.check_texture_slot(
                            Some(effective.texture_id),
                            TextureSlot::DecalAlbedo,
                        );
                        res.check_texture_slot(
                            effective.normal_texture_id,
                            TextureSlot::DecalNormalMap,
                        );
                        res.check_texture_slot(
                            effective.roughness_texture_id,
                            TextureSlot::DecalRoughness,
                        );
                        res.check_texture_slot(
                            effective.metallic_texture_id,
                            TextureSlot::DecalMetallic,
                        );
                        res.check_texture_slot(
                            effective.emissive_texture_id,
                            TextureSlot::DecalEmissive,
                        );
                        let gpu = decal_gpu.upload_item(device, res, &effective);
                        let bound = [
                            Some(effective.texture_id),
                            effective.normal_texture_id,
                            effective.roughness_texture_id,
                            effective.metallic_texture_id,
                            effective.emissive_texture_id,
                        ];
                        self.draws.push(gpu.clone());
                        e.insert((gpu, bound));
                        uploads += 1;
                    }
                }
                seen.insert(key);
            }
            // Evict decals that were not part of this frame's submission.
            self.cache.retain(|k, _| seen.contains(k));
            self.pick_items.extend(decals.iter().cloned());
        }
        self.uploads = uploads;
        self.reused = reused;

        Vec::new()
    }

    fn surface_mask_readers(&self, items: &ItemCollections<'_>, out: &mut Vec<u32>) {
        for decal in items.of::<DecalItem>() {
            if decal.settings.hidden || decal.settings.opacity <= 0.0 {
                continue;
            }
            let mask = viewport_lib::plugin_api::surface_mask_bits(decal.channel_mask);
            if !out.contains(&mask) {
                out.push(mask);
            }
        }
    }

    fn encoder_scopes(&self) -> &[EncoderScope] {
        // Decals are part of how an opaque surface looks, so they stamp before
        // the selection sub-highlight and the depth-read pass rather than over
        // them.
        &[EncoderScope::OnOpaqueSurfaces]
    }

    fn encode(
        &self,
        encoder: &mut viewport_lib::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if self.draws.is_empty() {
            return;
        }

        // Group 1 of both the colour and outline-mask passes: the depth aspect
        // to reconstruct the receiving surface, and the stencil aspect, which
        // holds the surface mask. Rebuilt per frame because the
        // HDR attachments can be reallocated at the same size, which would
        // leave a cached bind group pointing at a dead view.
        let depth_bg =
            gpu.create_depth_bg(ctx.device, ctx.scene_depth_only, ctx.scene_stencil_only);

        self.encode_colour(gpu, encoder, ctx, &depth_bg);
        self.encode_outline(gpu, encoder, ctx, &depth_bg);
    }

    /// Rasterise each pickable decal's projection box into the shared id
    /// pass. The box can extend past the shaded footprint into empty space,
    /// which is deliberate and matches the CPU decal pick: a click near a
    /// decal but off its receiver still selects it.
    fn render_pick(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PickPassContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return;
        }
        let Some(gpu) = &self.gpu else { return };
        let (vbuf, ibuf) = &gpu.pick_cube;
        let mut bound = false;
        for entry in &self.draws {
            let Some(pick) = entry.pick.as_ref() else {
                continue;
            };
            if !gpu.drawn(entry.blend_mode) {
                continue;
            }
            // Degenerate transforms have no box to rasterise; the CPU pick
            // skips them the same way.
            if entry.model.determinant().abs() < 1e-12 {
                continue;
            }
            if !bound {
                let Some(pl) = gpu.pipelines.get(pipeline::PICK) else {
                    return;
                };
                pass.set_pipeline(pl);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.set_index_buffer(ibuf.slice(..), viewport_lib::gpu::IndexFormat::Uint32);
                bound = true;
            }
            pass.set_bind_group(1, &pick.bind_group, &[]);
            pass.draw_indexed(0..36, 0, 0..1);
        }
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        // Ray versus the decal projection box. A decal is the unit box
        // [-0.5, 0.5]^3 mapped to world by `transform`. The box front face
        // typically hugs the receiver surface, so a decal that straddles a
        // surface wins over that surface by `toi`, letting a click select the
        // decal itself. A decal whose box floats in empty space is still
        // pickable wherever the ray passes through the volume.
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.transform);
            if model.determinant().abs() < 1e-12 {
                continue;
            }
            let inv = model.inverse();
            let local_origin = inv.transform_point3(ray.origin);
            let local_dir = inv.transform_vector3(ray.direction);
            let Some(toi) = ray_unit_box_toi(local_origin, local_dir) else {
                continue;
            };
            if best.as_ref().is_some_and(|(b, _)| toi >= *b) {
                continue;
            }
            let hit = PickHit::object_hit(
                item.settings.pick_id.0,
                ray.origin + ray.direction * toi,
                -ray.direction.normalize_or_zero(),
            );
            best = Some((toi, hit));
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> viewport_lib::renderer::PickRectResult {
        let mut result = viewport_lib::renderer::PickRectResult::default();
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return result;
        }
        // Project the box and test its corners and edges against the selection
        // rect, mirroring the ray-versus-box test the click pick uses so
        // box-select and click agree.
        let in_rect = |x: f32, y: f32| {
            x >= ctx.rect_min.x && x <= ctx.rect_max.x && y >= ctx.rect_min.y && y <= ctx.rect_max.y
        };
        for item in &self.pick_items {
            if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                continue;
            }
            let model = glam::Mat4::from_cols_array_2d(&item.transform);
            if model.determinant().abs() < 1e-12 {
                continue;
            }
            let mvp = ctx.view_proj * model;
            let sc: [Option<glam::Vec2>; 8] = std::array::from_fn(|i| {
                viewport_lib::plugin_api::pick_helpers::project_to_screen(
                    glam::Vec3::from(CORNERS[i]),
                    mvp,
                    ctx.viewport_size,
                )
            });
            let hit = sc.iter().any(|p| p.is_some_and(|p| in_rect(p.x, p.y)))
                || EDGES.iter().any(|&(a, b)| match (sc[a], sc[b]) {
                    (Some(a), Some(b)) => segment_in_rect(a, b, ctx.rect_min, ctx.rect_max),
                    (Some(a), None) => in_rect(a.x, a.y),
                    (None, Some(b)) => in_rect(b.x, b.y),
                    (None, None) => false,
                });
            if hit {
                result.objects.push(item.settings.pick_id.0);
            }
        }
        result
    }
}

impl DecalPlugin {
    /// Project each decal texture onto opaque surfaces. Reads scene depth as a
    /// texture, so the pass has no depth attachment of its own.
    fn encode_colour(
        &self,
        decal_gpu: &pipeline::DecalGpu,
        encoder: &mut viewport_lib::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        depth_bg: &viewport_lib::gpu::BindGroup,
    ) {
        if self.draws.is_empty() {
            return;
        }
        let [target_w, target_h] = ctx.scene_size;
        let mut pass = encoder.begin_render_pass(&viewport_lib::gpu::RenderPassDescriptor {
            #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
            multiview_mask: None,
            label: Some("decal_pass"),
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
        pass.set_bind_group(0, ctx.camera_bind_group, &[]);
        pass.set_bind_group(1, depth_bg, &[]);
        let view_proj = ctx.camera.view_proj();
        for gpu in &self.draws {
            // Still compiling: this decal draws next frame.
            let Some(pl) = decal_gpu
                .pipelines
                .get(pipeline::colour_index(gpu.blend_mode))
            else {
                continue;
            };
            // Confine each decal's fullscreen quad to its screen footprint to
            // avoid fullscreen overdraw per decal.
            match pipeline::decal_scissor(&gpu.model, &view_proj, target_w, target_h) {
                pipeline::DecalScissor::Skip => continue,
                pipeline::DecalScissor::Full => pass.set_scissor_rect(0, 0, target_w, target_h),
                pipeline::DecalScissor::Rect(x, y, w, h) => pass.set_scissor_rect(x, y, w, h),
            }
            pass.set_pipeline(pl);
            pass.set_bind_group(2, &gpu.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

impl DecalPlugin {
    /// Trace an anti-aliased ring around the footprint of selected decals.
    ///
    /// Runs after the colour pass so the scene depth the decal projects against
    /// already exists. Selected decals are stamped into a transient R8 mask
    /// (reusing the colour pass's coverage maths and bind groups), then a
    /// fullscreen edge-detect blends the ring onto the target. Does nothing
    /// when no decal is selected, so the common case pays no cost.
    fn encode_outline(
        &self,
        decal_gpu: &pipeline::DecalGpu,
        encoder: &mut viewport_lib::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        depth_bg: &viewport_lib::gpu::BindGroup,
    ) {
        if !self
            .draws
            .iter()
            .any(|g| g.selected && decal_gpu.drawn(g.blend_mode))
        {
            return;
        }
        let (Some(mask_pl), Some(edge_pl)) = (
            decal_gpu.pipelines.get(pipeline::OUTLINE_MASK),
            decal_gpu.pipelines.get(pipeline::OUTLINE_EDGE),
        ) else {
            return;
        };
        let [target_w, target_h] = ctx.scene_size;
        let (target_w, target_h) = (target_w.max(1), target_h.max(1));

        let mut targets = self.outline_targets.lock().unwrap();
        decal_gpu.ensure_outline_targets(ctx.device, &mut targets, target_w, target_h);
        let Some(targets) = targets.as_ref() else {
            return;
        };

        // Refresh the edge uniform in place; the target set (mask texture,
        // view, buffer, bind group) is reused frame to frame and rebuilt only
        // when the viewport size changes, so the pass allocates nothing per
        // frame.
        let edge_uniform = viewport_lib::resources::OutlineEdgeUniform {
            colour: ctx.outline_colour.to_linear_rgba(),
            radius: ctx.outline_width_px,
            viewport_w: target_w as f32,
            viewport_h: target_h as f32,
            _pad: 0.0,
        };
        ctx.queue.write_buffer(
            &targets.edge_uniform_buf,
            0,
            bytemuck::cast_slice(&[edge_uniform]),
        );

        // Accumulate the selected decals' screen AABB while stamping the mask,
        // so the edge pass runs only over that region instead of the whole
        // frame.
        let mut union: Option<(i32, i32, i32, i32)> = None;
        let mut any_full = false;

        {
            let mut pass = encoder.begin_render_pass(&viewport_lib::gpu::RenderPassDescriptor {
                #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
                multiview_mask: None,
                label: Some("decal_outline_mask_pass"),
                color_attachments: &[Some(viewport_lib::gpu::RenderPassColorAttachment {
                    view: &targets.mask_view,
                    resolve_target: None,
                    ops: viewport_lib::gpu::Operations {
                        load: viewport_lib::gpu::LoadOp::Clear(
                            viewport_lib::gpu::Color::TRANSPARENT,
                        ),
                        store: viewport_lib::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(mask_pl);
            pass.set_bind_group(0, ctx.camera_bind_group, &[]);
            pass.set_bind_group(1, depth_bg, &[]);
            let view_proj = ctx.camera.view_proj();
            for gpu in &self.draws {
                if !gpu.selected || !decal_gpu.drawn(gpu.blend_mode) {
                    continue;
                }
                match pipeline::decal_scissor(&gpu.model, &view_proj, target_w, target_h) {
                    pipeline::DecalScissor::Skip => continue,
                    pipeline::DecalScissor::Full => {
                        any_full = true;
                        pass.set_scissor_rect(0, 0, target_w, target_h);
                    }
                    pipeline::DecalScissor::Rect(x, y, w, h) => {
                        let (x0, y0, x1, y1) = (x as i32, y as i32, (x + w) as i32, (y + h) as i32);
                        union = Some(match union {
                            Some((ux0, uy0, ux1, uy1)) => {
                                (ux0.min(x0), uy0.min(y0), ux1.max(x1), uy1.max(y1))
                            }
                            None => (x0, y0, x1, y1),
                        });
                        pass.set_scissor_rect(x, y, w, h);
                    }
                }
                pass.set_bind_group(2, &gpu.bind_group, &[]);
                pass.draw(0..6, 0..1);
            }
        }

        // Every selected decal projected off screen: nothing to outline.
        if !any_full && union.is_none() {
            return;
        }
        // Bound the edge pass to the decals' screen AABB, expanded by the
        // outline width. A decal straddling the near plane (Full) falls back
        // to fullscreen.
        let edge_rect = if any_full {
            None
        } else {
            union.map(|(x0, y0, x1, y1)| {
                let m = ctx.outline_width_px.ceil() as i32 + 2;
                let cx0 = (x0 - m).clamp(0, target_w as i32);
                let cy0 = (y0 - m).clamp(0, target_h as i32);
                let cx1 = (x1 + m).clamp(0, target_w as i32);
                let cy1 = (y1 + m).clamp(0, target_h as i32);
                (
                    cx0 as u32,
                    cy0 as u32,
                    (cx1 - cx0).max(0) as u32,
                    (cy1 - cy0).max(0) as u32,
                )
            })
        };

        let mut pass = encoder.begin_render_pass(&viewport_lib::gpu::RenderPassDescriptor {
            #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
            multiview_mask: None,
            label: Some("decal_outline_edge_pass"),
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
        pass.set_pipeline(edge_pl);
        pass.set_bind_group(0, &targets.edge_bind_group, &[]);
        if let Some((x, y, w, h)) = edge_rect {
            if w == 0 || h == 0 {
                return;
            }
            pass.set_scissor_rect(x, y, w, h);
        }
        pass.draw(0..3, 0..1);
    }
}
