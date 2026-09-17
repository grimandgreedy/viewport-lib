//! The screen-space decal item type as an [`ItemTypePlugin`]: textures
//! projected onto opaque surfaces along a box, with a stencil mask that keeps
//! them off surfaces which opted out. Consumers submit [`DecalItem`]s on
//! `SceneFrame::decals`; the renderer routes that field to this plugin.
//!
//! All four passes are encoded from [`ItemTypePlugin::encode`] at
//! [`EncoderScope::OnOpaqueSurfaces`], in the order they have to run: the
//! stencil exclude pass, the colour projection, then the outline mask and
//! edge trace when a decal is selected.

mod pipeline;
pub(crate) mod types;

use crate::plugin_api::{
    EncoderScope, EncoderScopeContext, ItemFrameContext, ItemTypePlugin, PickContext, PickRay,
    PluginItemCollection, RectPickContext,
};
use crate::renderer::picking::helpers::{ray_unit_box_toi, segment_in_rect};
use crate::renderer::{DecalBlendMode, DecalItem, PickHit, PickId, PickMask};

pub(crate) const TYPE_NAME: &str = "viewport.decal";

impl PluginItemCollection for Vec<DecalItem> {
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

pub(crate) struct DecalPlugin {
    gpu: pipeline::DecalGpu,
    /// This frame's draw list, in `sort_key` order, built in `prepare`.
    draws: Vec<pipeline::DecalGpuItem>,
    /// GPU resources cached across frames, keyed by decal content hash, so an
    /// unchanged decal rebuilds no uniform buffer and no bind group.
    cache: std::collections::HashMap<
        u64,
        (
            pipeline::DecalGpuItem,
            crate::resources::resource_deps::ResourceDeps,
        ),
    >,
    /// Resource epochs the cache was last validated against.
    deps_gate: crate::resources::resource_deps::DepsGate,
    /// This frame's stencil-exclude draws, one per surface that opted out.
    exclude_draws: Vec<pipeline::DecalExcludeGpuItem>,
    /// Items retained from `prepare` for the out-of-band CPU pick answers.
    pick_items: Vec<DecalItem>,
    /// Cache hit / miss counts for this frame, read back into `FrameStats`.
    stats: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// The outline mask target and its edge bind group, keyed by target size.
    /// Behind a lock because `encode` runs from a shared borrow but the target
    /// set is rebuilt when the viewport resizes.
    outline_targets: std::sync::Mutex<Option<pipeline::DecalOutlineTargets>>,
}

impl DecalPlugin {
    /// `stats` is the renderer's handle on this frame's cache tallies, packed
    /// `(uploads << 32) | reused`, which it reads back into `FrameStats`.
    pub(crate) fn new(stats: std::sync::Arc<std::sync::atomic::AtomicU64>) -> Self {
        Self {
            gpu: pipeline::DecalGpu::default(),
            draws: Vec::new(),
            cache: std::collections::HashMap::new(),
            deps_gate: crate::resources::resource_deps::DepsGate::default(),
            exclude_draws: Vec::new(),
            pick_items: Vec::new(),
            stats,
            outline_targets: std::sync::Mutex::new(None),
        }
    }
}

impl ItemTypePlugin for DecalPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    /// Build the projection pipelines at registration rather than on the first
    /// frame that submits a decal. Decals tend to appear mid-session (impact
    /// marks, scorches), so a lazy build would stall that frame by the compile
    /// cost (~8 ms measured on a desktop GPU). The exclude and outline
    /// pipelines stay lazy: they are only needed by scenes that opt a surface
    /// out or select a decal.
    fn init_gpu(
        &mut self,
        device: &crate::gpu::Device,
        shared: &crate::plugin_api::SharedBindings<'_>,
    ) {
        self.gpu.ensure_shared(device);
        self.gpu.ensure_pipeline(device, shared.group0_layout);
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        _queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        let decals = items
            .as_any()
            .downcast_ref::<Vec<DecalItem>>()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let res = ctx.resources;

        self.draws.clear();
        self.pick_items.clear();
        let mut uploads = 0u32;
        let mut reused = 0u32;

        if decals.is_empty() {
            // No decals this frame: drop any cached GPU resources.
            self.cache.clear();
        } else {
            self.gpu.ensure_shared(device);
            self.gpu.ensure_pipeline(device, &res.binds.camera_bgl);
            // Cached entries hold bind groups over texture views, so a free or
            // a replace since the last frame invalidates them: a free drops
            // only the entries whose deps no longer resolve, a replace drops
            // everything, since a view swapped behind a live id cannot be
            // detected per entry.
            match self.deps_gate.poll(res) {
                crate::resources::resource_deps::Revalidate::RebuildAll => self.cache.clear(),
                crate::resources::resource_deps::Revalidate::CheckEach => {
                    self.cache.retain(|_, (_, deps)| deps.resolves(res));
                }
                crate::resources::resource_deps::Revalidate::Valid => {}
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
                let key = pipeline::hash_decal_item(&effective, &res.content.textures);
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
                        use crate::resources::TextureSlot;
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
                        let gpu = self.gpu.upload_item(device, res, &effective);
                        let deps = crate::resources::resource_deps::ResourceDeps::textures([
                            Some(effective.texture_id),
                            effective.normal_texture_id,
                            effective.roughness_texture_id,
                            effective.metallic_texture_id,
                            effective.emissive_texture_id,
                        ]);
                        self.draws.push(gpu.clone());
                        e.insert((gpu, deps));
                        uploads += 1;
                    }
                }
                seen.insert(key);
            }
            // Evict decals that were not part of this frame's submission.
            self.cache.retain(|k, _| seen.contains(k));
            self.pick_items.extend(decals.iter().cloned());
        }
        self.stats.store(
            ((uploads as u64) << 32) | reused as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // The surfaces that opted out of decal projection. `receives_decals`
        // lives on mesh items, so the lib resolves it and hands the result
        // over; turning it into stencil draws is this type's business.
        self.exclude_draws.clear();
        if !ctx.decal_excluded_surfaces.is_empty() {
            self.gpu
                .ensure_exclude_pipeline(device, &res.binds.camera_bgl);
            for &(mesh_id, model) in ctx.decal_excluded_surfaces {
                self.exclude_draws
                    .push(self.gpu.upload_exclude_item(device, mesh_id, model));
            }
        }

        // The outline pipelines are cheap to hold and the outline pass runs
        // from a shared borrow, so build them here the first frame a decal is
        // selected rather than inside `encode`.
        if self.draws.iter().any(|g| g.selected) {
            self.gpu
                .ensure_outline_pipelines(device, &res.binds.camera_bgl);
        }

        Vec::new()
    }

    fn encoder_scopes(&self) -> &[EncoderScope] {
        // Decals are part of how an opaque surface looks, so they stamp before
        // the selection sub-highlight and the depth-read pass rather than over
        // them.
        &[EncoderScope::OnOpaqueSurfaces]
    }

    fn encode(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        if self.draws.is_empty() && self.exclude_draws.is_empty() {
            return;
        }

        // Group 1 of both the colour and outline-mask passes: the depth aspect
        // to reconstruct the receiving surface, and the stencil aspect to skip
        // the surfaces the exclude pass marked. Rebuilt per frame because the
        // HDR attachments can be reallocated at the same size, which would
        // leave a cached bind group pointing at a dead view.
        let depth_bg =
            self.gpu
                .create_depth_bg(ctx.device, ctx.scene_depth_only, ctx.scene_stencil_only);

        self.encode_exclude(encoder, ctx);
        self.encode_colour(encoder, ctx, &depth_bg);
        self.encode_outline(encoder, ctx, &depth_bg);
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
            #[allow(deprecated)]
            let hit = PickHit {
                id: item.settings.pick_id.0,
                sub_object: None,
                world_pos: ray.origin + ray.direction * toi,
                normal: -ray.direction.normalize_or_zero(),
                scalar_value: None,
                sub_object_world_pos: None,
            };
            best = Some((toi, hit));
        }
        best
    }

    fn pick_rect(&self, ctx: &RectPickContext<'_>) -> crate::renderer::PickRectResult {
        let mut result = crate::renderer::PickRectResult::default();
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
                crate::renderer::picking::helpers::project_to_screen(
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
    /// Stamp stencil = 0 on the surfaces that opted out, so the colour pass
    /// skips those pixels. Depth-only pass, no colour attachment.
    fn encode_exclude(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
    ) {
        if self.exclude_draws.is_empty() {
            return;
        }
        let Some(exclude_pl) = self.gpu.exclude_pipeline.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("decal_exclude_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                view: ctx.scene_depth,
                depth_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                }),
                stencil_ops: Some(crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(exclude_pl);
        pass.set_stencil_reference(0);
        pass.set_bind_group(0, ctx.camera_bind_group, &[]);
        for item in &self.exclude_draws {
            pass.set_bind_group(1, &item.bind_group, &[]);
            ctx.meshes.draw_indexed(&mut pass, item.mesh_id);
        }
    }

    /// Project each decal texture onto opaque surfaces. Reads scene depth as a
    /// texture, so the pass has no depth attachment of its own.
    fn encode_colour(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        depth_bg: &crate::gpu::BindGroup,
    ) {
        if self.draws.is_empty() {
            return;
        }
        let replace_pipeline = self.gpu.replace_pipeline.as_ref();
        let multiply_pipeline = self.gpu.multiply_pipeline.as_ref();
        let additive_pipeline = self.gpu.additive_pipeline.as_ref();
        if replace_pipeline.is_none() && multiply_pipeline.is_none() && additive_pipeline.is_none()
        {
            return;
        }
        let [target_w, target_h] = ctx.scene_size;
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("decal_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: ctx.scene_colour,
                resolve_target: None,
                ops: crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
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
            let pipeline = match gpu.blend_mode {
                DecalBlendMode::Replace => replace_pipeline,
                DecalBlendMode::Multiply => multiply_pipeline,
                DecalBlendMode::Additive => additive_pipeline,
            };
            let Some(pl) = pipeline else { continue };
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
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        depth_bg: &crate::gpu::BindGroup,
    ) {
        if !self.draws.iter().any(|g| g.selected) {
            return;
        }
        let [target_w, target_h] = ctx.scene_size;
        let (target_w, target_h) = (target_w.max(1), target_h.max(1));

        let mut targets = self.outline_targets.lock().unwrap();
        self.gpu
            .ensure_outline_targets(ctx.device, &mut targets, target_w, target_h);
        let (Some(mask_pl), Some(edge_pl), Some(targets)) = (
            self.gpu.outline_mask_pipeline.as_ref(),
            self.gpu.outline_edge_pipeline.as_ref(),
            targets.as_ref(),
        ) else {
            return;
        };

        // Refresh the edge uniform in place; the target set (mask texture,
        // view, buffer, bind group) is reused frame to frame and rebuilt only
        // when the viewport size changes, so the pass allocates nothing per
        // frame.
        let edge_uniform = crate::resources::OutlineEdgeUniform {
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
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("decal_outline_mask_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: &targets.mask_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                        store: crate::gpu::StoreOp::Store,
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
                if !gpu.selected {
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

        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("decal_outline_edge_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: ctx.scene_colour,
                resolve_target: None,
                ops: crate::gpu::Operations {
                    load: crate::gpu::LoadOp::Load,
                    store: crate::gpu::StoreOp::Store,
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
