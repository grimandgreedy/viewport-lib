//! The scatter-volume item type as an [`ItemTypePlugin`]: participating media
//! (fog, smoke, fire) ray-marched through a box or sphere and composited over
//! the lit scene. Consumers submit [`ScatterVolumeItem`]s on
//! `SceneFrame::scatter_volumes`; the renderer routes that field to this
//! plugin.
//!
//! Every pass is encoded from [`ItemTypePlugin::encode`] at
//! [`EncoderScope::AfterTransparent`], in the order they have to run: the
//! optional refraction distortion, the per-volume ray-march into an
//! accumulation target, the optional temporal blend against last frame's
//! history, and the composite onto the scene colour.
//!
//! Two things a scatter volume does are deliberately not here, because they
//! are not rendering. An emissive volume contributes a virtual point light to
//! the lighting prepare, and a selected volume gets a bounds wireframe through
//! the line substrate. Both are other parts of the lib reading the item list,
//! and both stay where they are.

mod pipeline;

use crate::plugin_api::{
    EncoderScope, EncoderScopeContext, ItemFrameContext, ItemTypePlugin, PickContext, PickRay,
    PluginItemCollection,
};
use crate::renderer::{PickHit, PickId, PickMask, ScatterVolumeItem};
use crate::scene::scatter_volume::{ScatterShape, ScatterVolume};

pub(crate) const TYPE_NAME: &str = "viewport.scatter_volume";

/// The scatter intermediates and the scene colour they composite onto are both
/// HDR; the pass has no LDR form.
const TARGET_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rgba16Float;

impl PluginItemCollection for Vec<ScatterVolumeItem> {
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

#[derive(Default)]
pub(crate) struct ScatterVolumePlugin {
    gpu: pipeline::ScatterGpu,
    /// This frame's visible volumes with their per-item opacity and flags, in
    /// the back-to-front order the per-volume draws composite in.
    draws: Vec<(ScatterVolume, f32, u32)>,
    /// The subset of `draws` that asked for refraction, in the same order.
    refraction_draws: Vec<(ScatterVolume, f32)>,
    /// Group 2 bind groups, one per entry in `draws`, resolved in `prepare`
    /// because that is where the texture stores are reachable.
    per_volume_tex_bgs: Vec<crate::gpu::BindGroup>,
    /// Items retained from `prepare` for the out-of-band CPU pick answers.
    pick_items: Vec<ScatterVolumeItem>,
    /// Per-viewport accumulation and history targets. Behind a lock because
    /// `encode` runs from a shared borrow but allocates on resize and advances
    /// the history ping-pong.
    viewports: std::sync::Mutex<Vec<Option<pipeline::ScatterViewportState>>>,
}

impl ItemTypePlugin for ScatterVolumePlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        let volumes = items
            .as_any()
            .downcast_ref::<Vec<ScatterVolumeItem>>()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        self.draws.clear();
        self.refraction_draws.clear();
        self.per_volume_tex_bgs.clear();
        self.pick_items.clear();
        self.pick_items.extend_from_slice(volumes);

        if volumes.is_empty() {
            return Vec::new();
        }

        for item in volumes {
            // A wireframe volume draws its bounds outline through the line
            // substrate instead of ray-marching.
            if item.settings.hidden || item.settings.wireframe || ctx.wireframe_mode {
                continue;
            }
            let mut flags: u32 = 0;
            if item.settings.unlit {
                flags |= crate::scene::scatter_volume::SCATTER_FLAG_UNLIT;
            }
            if item.settings.receive_shadows {
                flags |= crate::scene::scatter_volume::SCATTER_FLAG_RECEIVE_SHADOWS;
            }
            self.draws
                .push((item.volume.clone(), item.settings.opacity, flags));
            if item.volume.refraction.is_some() {
                self.refraction_draws
                    .push((item.volume.clone(), item.settings.opacity));
            }
        }

        if self.draws.is_empty() {
            return Vec::new();
        }

        sort_back_to_front(&mut self.draws, ctx.camera.eye_position);

        let res = ctx.resources;
        self.gpu
            .ensure_pipeline(device, &res.binds.camera_bgl, TARGET_FORMAT);
        self.gpu.ensure_composite_pipeline(device, TARGET_FORMAT);
        self.gpu.ensure_temporal_resolve_pipeline(device);
        self.gpu.ensure_frame_uniform_buffer(device);
        self.gpu.ensure_temporal_uniform_buffer(device);

        let n = self.gpu.write_per_volume_buffer(device, queue, &self.draws);

        // Group 2 binds the volume's colourmap LUT and density texture, both of
        // which live in the shared content store. The cache is rebuilt each
        // frame so a texture freed or replaced since the last frame cannot
        // leave a bind group pointing at a dead view.
        self.gpu.clear_per_volume_tex_cache();
        for (volume, _, _) in self.draws.iter().take(n as usize) {
            let (lut_id, density_id) = pipeline::ScatterGpu::volume_tex_ids(volume);
            let bg = self
                .gpu
                .ensure_per_volume_tex_bg(device, queue, res, lut_id, density_id);
            self.per_volume_tex_bgs.push(bg);
        }
        self.draws.truncate(n as usize);

        if !self.refraction_draws.is_empty() {
            self.gpu
                .ensure_refraction_pipeline(device, &res.binds.camera_bgl, TARGET_FORMAT);
            self.gpu
                .ensure_refraction_blit_pipeline(device, TARGET_FORMAT);
            // Sized here; filled at encode time, where the frame's animation
            // clock is readable.
            self.gpu
                .ensure_refraction_per_volume_buffer(device, self.refraction_draws.len());
        }

        Vec::new()
    }

    fn encoder_scopes(&self) -> &[EncoderScope] {
        // Participating media absorbs and adds to everything behind it, which
        // includes resolved transparency, so it composites after the scene is
        // otherwise finished.
        &[EncoderScope::AfterTransparent]
    }

    fn encode(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        if self.draws.is_empty() {
            return;
        }
        let settings = &ctx.effects.scatter;
        let [sw, sh] = ctx.scene_size;
        let size = if settings.downsample {
            [(sw / 2).max(1), (sh / 2).max(1)]
        } else {
            [sw.max(1), sh.max(1)]
        };

        let Ok(mut viewports) = self.viewports.lock() else {
            return;
        };
        while viewports.len() <= ctx.viewport_index {
            viewports.push(None);
        }
        let stale = match viewports[ctx.viewport_index].as_ref() {
            None => true,
            Some(s) => s.size != size || s.downsampled != settings.downsample,
        };
        if stale {
            viewports[ctx.viewport_index] = Some(pipeline::ScatterViewportState::new(
                ctx.device,
                &self.gpu,
                size,
                settings.downsample,
            ));
        }
        let state = viewports[ctx.viewport_index].as_mut().unwrap();

        // Refraction first: it shimmers the scene colour behind each refractive
        // volume so the absorption and in-scattering below land on top of the
        // distorted image rather than under it.
        self.encode_refraction(encoder, ctx, state);

        self.gpu.write_frame_uniform(
            ctx.queue,
            settings.time_seconds,
            settings.quality.default_steps(),
            settings.blue_noise_jitter,
            ctx.frame_index,
        );
        let frame_bg = self.gpu.make_frame_bg(ctx.device, ctx.scene_depth_only);

        self.encode_march(encoder, ctx, state, &frame_bg);

        let composite_source = if settings.temporal {
            self.gpu.write_temporal_uniform(
                ctx.queue,
                state.prev_view_proj,
                settings.temporal_blend,
                state.history_valid,
            );
            self.encode_temporal_resolve(encoder, ctx, state)
        } else {
            &state.composite_bg_raw
        };
        self.encode_composite(encoder, ctx, composite_source);

        state.prev_view_proj = ctx.camera.view_proj().to_cols_array_2d();
        state.parity = 1 - state.parity;
        state.history_valid = settings.temporal;
    }

    /// The volume's bounds: a box wireframe or three sphere great circles.
    ///
    /// Participating media has no surface for the outline ring to trace, so
    /// this is a scatter volume's only selection feedback, and it is drawn
    /// whether or not `outline_selected` is set for the frame: that flag gates
    /// the surface-mesh outline, which is a different affordance.
    fn wireframe_polylines(
        &self,
        items: &dyn PluginItemCollection,
        ctx: &ItemFrameContext<'_>,
    ) -> Vec<crate::renderer::PolylineItem> {
        let Some(volumes) = items.as_any().downcast_ref::<Vec<ScatterVolumeItem>>() else {
            return Vec::new();
        };
        volumes
            .iter()
            .filter(|item| !item.settings.hidden)
            .filter(|item| item.settings.selected || item.settings.wireframe || ctx.wireframe_mode)
            .map(|item| {
                let colour = if item.settings.selected {
                    [1.0_f32, 0.9, 0.2, 1.0]
                } else {
                    [0.8_f32, 0.85, 0.95, 1.0]
                };
                let mut polyline = match item.volume.shape {
                    ScatterShape::Box(b) => crate::renderer::aabb_wireframe_polyline(&b, colour),
                    ScatterShape::Sphere { center, radius } => {
                        crate::renderer::sphere_wireframe_polyline(center, radius, 48, colour)
                    }
                };
                // Thin single-pixel lines, as the bounds outlines have always
                // drawn: a thick screen-space line reads as geometry.
                polyline.settings.wireframe = true;
                polyline
            })
            .collect()
    }

    fn pick(&self, ray: &PickRay, ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        if !ctx.mask.intersects(PickMask::OBJECT) {
            return None;
        }
        // Ray versus the volume's shape. Participating media has no sub-object
        // level, so the entry point is the whole answer.
        let mut best: Option<(f32, PickHit)> = None;
        for item in &self.pick_items {
            if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                continue;
            }
            let Some((toi, _)) = crate::scene::scatter_volume::ray_intersect(
                &item.volume.shape,
                ray.origin,
                ray.direction,
            ) else {
                continue;
            };
            if best.as_ref().is_some_and(|(b, _)| toi >= *b) {
                continue;
            }
            let world_pos = ray.origin + ray.direction * toi;
            let centre = match item.volume.shape {
                ScatterShape::Box(b) => (b.min + b.max) * 0.5,
                ScatterShape::Sphere { center, .. } => glam::Vec3::from(center),
            };
            let normal = (world_pos - centre)
                .try_normalize()
                .unwrap_or(glam::Vec3::Z);
            best = Some((
                toi,
                PickHit::object_hit(item.settings.pick_id.0, world_pos, normal),
            ));
        }
        best
    }
}

/// Order the per-volume draws back to front for the alpha-over composite.
///
/// The metric is the maximum corner distance of the volume's world AABB from
/// the eye, descending. Centroid distance flips order when one volume contains
/// another (a huge fog containing a small fire): the fire centroid can land on
/// either side of the fog centroid as the camera orbits, which swaps the
/// composite visibly. Sorting by far-corner distance keeps a container, whose
/// far corner is much further away, strictly behind what it contains.
fn sort_back_to_front(draws: &mut [(ScatterVolume, f32, u32)], eye: [f32; 3]) {
    let far_corner = |volume: &ScatterVolume| -> f32 {
        let aabb = volume.world_aabb();
        let pick = |min: f32, max: f32, e: f32| {
            if (min - e).abs() > (max - e).abs() {
                min
            } else {
                max
            }
        };
        let cx = pick(aabb.min.x, aabb.max.x, eye[0]);
        let cy = pick(aabb.min.y, aabb.max.y, eye[1]);
        let cz = pick(aabb.min.z, aabb.max.z, eye[2]);
        (cx - eye[0]).powi(2) + (cy - eye[1]).powi(2) + (cz - eye[2]).powi(2)
    };
    draws.sort_by(|a, b| {
        far_corner(&b.0)
            .partial_cmp(&far_corner(&a.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

impl ScatterVolumePlugin {
    /// Copy the scene colour aside, then write a noise-driven distortion of it
    /// back over each refractive volume's screen footprint.
    fn encode_refraction(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        state: &mut pipeline::ScatterViewportState,
    ) {
        if self.refraction_draws.is_empty() {
            return;
        }
        // Pack the refraction params at this frame's animation clock. A volume
        // whose strength is zero packs nothing, so the count of slots actually
        // written is what the draw loop below runs over.
        let n_ref = self.gpu.write_refraction_per_volume_buffer(
            ctx.queue,
            &self.refraction_draws,
            ctx.effects.scatter.time_seconds,
        );
        if n_ref == 0 {
            return;
        }
        // The copy is at scene resolution, not at the scatter intermediates'
        // resolution, because it stands in for the scene colour itself.
        state.ensure_refraction_source(ctx.device, ctx.scene_size);
        let Some(source_view) = state.refraction_source_view.as_ref() else {
            return;
        };
        let blit_bg = self.gpu.make_composite_bg(ctx.device, ctx.scene_colour);
        let source_bg =
            self.gpu
                .make_refraction_source_bg(ctx.device, source_view, ctx.scene_depth_only);

        if let Some(blit_pipeline) = self.gpu.refraction_blit_pipeline.as_ref() {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("scatter_refraction_blit_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: source_view,
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
            pass.set_pipeline(blit_pipeline);
            pass.set_bind_group(0, &blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if let (Some(pipeline), Some(per_vol_bg)) = (
            self.gpu.refraction_pipeline.as_ref(),
            self.gpu.refraction_per_volume_bg.as_ref(),
        ) {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("scatter_refraction_pass"),
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
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, ctx.camera_bind_group, &[]);
            pass.set_bind_group(2, &source_bg, &[]);
            for i in 0..n_ref {
                pass.set_bind_group(1, per_vol_bg, &[i * self.gpu.refraction_per_volume_stride]);
                pass.draw(0..6, 0..1);
            }
        }
    }

    /// Ray-march each volume into the accumulation target. One draw per volume,
    /// back to front, each covering only that volume's projected footprint.
    fn encode_march(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        state: &pipeline::ScatterViewportState,
        frame_bg: &crate::gpu::BindGroup,
    ) {
        let (Some(pipeline), Some(per_vol_bg)) =
            (self.gpu.pipeline.as_ref(), self.gpu.per_volume_bg.as_ref())
        else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("scatter_volume_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: &state.raw_current_view,
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, ctx.camera_bind_group, &[]);
        pass.set_bind_group(3, frame_bg, &[]);
        for (i, tex_bg) in self.per_volume_tex_bgs.iter().enumerate() {
            pass.set_bind_group(1, per_vol_bg, &[i as u32 * self.gpu.per_volume_stride]);
            pass.set_bind_group(2, tex_bg, &[]);
            pass.draw(0..6, 0..1);
        }
    }

    /// Blend this frame's accumulation against the previous frame's history
    /// into the other history slot, and return the composite source to read.
    fn encode_temporal_resolve<'s>(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        state: &'s pipeline::ScatterViewportState,
    ) -> &'s crate::gpu::BindGroup {
        // `parity` names the slot to write next, so the other slot holds the
        // previous frame.
        let (history_view, previous_view, source) = if state.parity == 0 {
            (
                &state.history_a_view,
                &state.history_b_view,
                &state.composite_bg_history_a,
            )
        } else {
            (
                &state.history_b_view,
                &state.history_a_view,
                &state.composite_bg_history_b,
            )
        };
        if let Some(resolve_pipeline) = self.gpu.temporal_resolve_pipeline.as_ref() {
            let resolve_bg = self.gpu.make_temporal_resolve_bg(
                ctx.device,
                &state.raw_current_view,
                previous_view,
                ctx.scene_depth_only,
            );
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("scatter_temporal_resolve_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: history_view,
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
            pass.set_pipeline(resolve_pipeline);
            pass.set_bind_group(0, &resolve_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        source
    }

    /// Composite the resolved scatter onto the scene colour with premultiplied
    /// alpha-over, upscaling when the intermediates are half-resolution.
    fn encode_composite(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &EncoderScopeContext<'_>,
        source: &crate::gpu::BindGroup,
    ) {
        let Some(composite_pipeline) = self.gpu.composite_pipeline.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(any(wgpu29, wgpu30))]
            multiview_mask: None,
            label: Some("scatter_composite_pass"),
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
        pass.set_pipeline(composite_pipeline);
        pass.set_bind_group(0, source, &[]);
        pass.draw(0..3, 0..1);
    }
}
