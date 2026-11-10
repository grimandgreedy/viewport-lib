//! The HDR frame's pass list.
//!
//! [`PassTable`] declares every pass the HDR path can encode, in encode order,
//! with the targets each reads and writes and whether it runs this frame. The
//! frame plans it once per viewport before its targets are ensured: the
//! target groups the frame needs and the composite's inputs both follow from
//! the passes that run.

use std::sync::atomic::{AtomicU32, Ordering};

use super::*;
use crate::resources::postprocess::targets::{TargetId, TargetSet};

/// Capacity of the table: every pass the HDR path declares.
pub(crate) const MAX_PASSES: usize = 32;

/// One pass of the HDR frame, in the order the frame encodes them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HdrPass {
    Scene,
    HizStore,
    SsaaResolve,
    SurfaceMask,
    ScopeOnOpaqueSurfaces,
    SubHighlight,
    DepthRead,
    OpaqueCopy,
    Transmission,
    ScopeAfterOpaque,
    Oit,
    ScopeAfterTransparent,
    OutlineComposite,
    Foreground,
    /// A built-in composite-input producer, by its index in
    /// `DeviceResources::post_producers`.
    Producer(u8),
    ExternalProducers,
    ToneMap,
    PostStages,
    Upscale,
    OutputDepthBlit,
    ForegroundStamp,
    SceneOverlays,
    Overlay,
}

/// One declared pass: what it reads and writes, and whether it runs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PassDecl {
    pub(crate) pass: HdrPass,
    pub(crate) name: &'static str,
    pub(crate) reads: TargetSet,
    pub(crate) writes: TargetSet,
    /// The pass does work nothing later in the frame reads (plugin encoder
    /// scopes, external producers, next frame's Hi-Z).
    pub(crate) side_effect: bool,
    pub(crate) live: bool,
}

/// The HDR passes one viewport runs this frame. Kept on the viewport and
/// refilled every frame, so planning allocates nothing after the first.
#[derive(Default)]
pub(crate) struct PassTable {
    passes: Vec<PassDecl>,
    /// Passes that encoded this frame, one bit per declaration index. Set
    /// by the passes themselves, so a test can hold the plan against what
    /// ran.
    ran: AtomicU32,
}

impl PassTable {
    /// Empty the table for a new frame. The first call sizes it for every
    /// pass, so planning never allocates after that.
    fn clear(&mut self) {
        self.passes.clear();
        self.passes.reserve_exact(MAX_PASSES);
        *self.ran.get_mut() = 0;
    }

    fn declare(
        &mut self,
        pass: HdrPass,
        name: &'static str,
        live: bool,
        reads: TargetSet,
        writes: TargetSet,
    ) {
        debug_assert!(self.passes.len() < MAX_PASSES);
        self.passes.push(PassDecl {
            pass,
            name,
            reads,
            writes,
            side_effect: false,
            live,
        });
    }

    fn declare_side_effect(&mut self, pass: HdrPass, name: &'static str, live: bool) {
        self.declare(pass, name, live, scene_targets(), TargetSet::empty());
        if let Some(last) = self.passes.last_mut() {
            last.side_effect = true;
        }
    }

    /// The passes that run, in encode order.
    pub(crate) fn live_passes(&self) -> impl Iterator<Item = HdrPass> + '_ {
        self.passes.iter().filter(|d| d.live).map(|d| d.pass)
    }

    /// Every declared pass by name, with whether it runs this frame.
    pub(crate) fn names(&self) -> Vec<(&'static str, bool)> {
        self.passes.iter().map(|d| (d.name, d.live)).collect()
    }

    /// Record that `pass` encoded this frame.
    pub(crate) fn mark_ran(&self, pass: HdrPass) {
        if let Some(i) = self.passes.iter().position(|d| d.pass == pass) {
            self.ran.fetch_or(1 << i, Ordering::Relaxed);
        }
    }

    /// The names of the passes that encoded this frame, in encode order.
    pub(crate) fn ran_names(&self) -> Vec<&'static str> {
        let ran = self.ran.load(Ordering::Relaxed);
        self.passes
            .iter()
            .enumerate()
            .filter(|(i, _)| ran & (1 << i) != 0)
            .map(|(_, d)| d.name)
            .collect()
    }

    /// Drop the passes whose output nothing reads. Walks back from the
    /// frame's output: a pass survives when it has side effects or writes a
    /// target a surviving later pass reads, and then what it reads is needed
    /// in turn. Every pass loads what it writes, so a write never satisfies
    /// an earlier pass's need.
    fn cull(&mut self) {
        let mut needed = TargetSet::of(TargetId::Output);
        for decl in self.passes.iter_mut().rev() {
            if !decl.live {
                continue;
            }
            if decl.side_effect || decl.writes.intersects(needed) {
                needed = needed | decl.reads;
            } else {
                decl.live = false;
            }
        }
    }

    /// The targets the running passes write.
    fn live_writes(&self) -> TargetSet {
        self.passes
            .iter()
            .filter(|d| d.live)
            .fold(TargetSet::empty(), |set, d| set | d.writes)
    }

    /// The composite's inputs: an effect's input is bound only when a pass
    /// that writes it runs this frame. The grade and view LUTs are not pass
    /// outputs and come from the caller.
    pub(crate) fn composite_inputs(
        &self,
        grade_lut: Option<crate::resources::TextureId>,
        view_lut: Option<crate::resources::TextureId>,
    ) -> crate::resources::CompositeInputs {
        let writes = self.live_writes();
        crate::resources::CompositeInputs {
            bloom: writes.contains(TargetId::Bloom),
            ssao: writes.contains(TargetId::Ssao),
            contact_shadows: writes.contains(TargetId::ContactShadow),
            dof: writes.contains(TargetId::Dof),
            foreground: writes.contains(TargetId::ForegroundDepth),
            grade_lut,
            view_lut,
        }
    }

    /// Whether a pass that runs this frame reads the indirect fraction. The
    /// scene pass always writes it, and stores it only then.
    pub(crate) fn reads_indirect_fraction(&self) -> bool {
        self.passes
            .iter()
            .any(|d| d.live && d.reads.contains(TargetId::IndirectFraction))
    }

    /// The target groups the running passes read or write.
    pub(crate) fn groups(&self) -> crate::resources::TargetGroups {
        self.passes
            .iter()
            .filter(|d| d.live)
            .fold(TargetSet::empty(), |set, d| set | d.reads | d.writes)
            .groups()
    }
}

/// The scene colour and depth: what a pass drawing over the scene reads.
fn scene_targets() -> TargetSet {
    TargetSet::of(TargetId::HdrColour).with(TargetId::HdrDepth)
}

/// The scene resolution for an output of `w` x `h` at `render_scale`.
pub(crate) fn scene_size_for(w: u32, h: u32, render_scale: f32) -> [u32; 2] {
    let scale = render_scale.clamp(0.1, 1.0);
    [
        (((w as f32) * scale).round() as u32).max(1),
        (((h as f32) * scale).round() as u32).max(1),
    ]
}

impl ViewportRenderer {
    /// Record that `pass` encoded for viewport `vp_idx` this frame.
    pub(super) fn mark_pass_ran(&self, vp_idx: usize, pass: HdrPass) {
        self.viewport_slots[vp_idx].pass_table.mark_ran(pass);
    }

    /// Whether the OIT pass has anything to draw this frame.
    pub(super) fn hdr_has_transparent(
        &self,
        frame: &FrameData,
        scene_items: &[SceneRenderItem],
    ) -> bool {
        (if self.instancing.use_instancing && !self.instancing.batches.is_empty() {
            // Every transparent item is in a blended batch, which goes through OIT.
            self.instancing.batches.iter().any(|b| b.is_transparent)
        } else {
            scene_items.iter().any(|i| {
                !i.settings.hidden
                    && crate::renderer::prepare::has_transparent_draws(i, &self.resources)
            })
        }) || frame
            .scene
            .volume_meshes
            .iter()
            .any(|i| !i.settings.hidden && i.transparency.is_some())
            // Item-type plugins draw into the OIT pass through
            // `paint_transparent`.
            || self.any_plugin_draws_transparent(frame)
    }

    /// Whether any item-type plugin encodes at `scope`.
    pub(super) fn any_plugin_encodes_at(&self, scope: crate::plugin_api::EncoderScope) -> bool {
        self.item_type_plugins
            .values()
            .any(|p| p.encoder_scopes().contains(&scope))
    }

    /// Fill viewport `vp_idx`'s pass table for `frame`. The slot must exist.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_hdr_passes(
        &mut self,
        frame: &FrameData,
        scene_items: &[SceneRenderItem],
        vp_idx: usize,
        w: u32,
        h: u32,
        ssaa_factor: u32,
    ) {
        use TargetId as T;
        let mut table = std::mem::take(&mut self.viewport_slots[vp_idx].pass_table);
        table.clear();
        let pp = &frame.effects.post_process;
        let slot = &self.viewport_slots[vp_idx];
        let ssaa = ssaa_factor > 1;
        let foreground = self.foreground_active(frame);
        let scaled = scene_size_for(w, h, self.current_render_scale) != [w, h];
        let throttled = self.degradation_effects_throttled;
        let scene = scene_targets();

        table.declare(
            HdrPass::Scene,
            "scene",
            true,
            TargetSet::empty(),
            if ssaa {
                TargetSet::of(T::SsaaColour)
            } else {
                scene
            }
            .with(T::IndirectFraction),
        );
        table.declare(
            HdrPass::HizStore,
            "hiz_store",
            self.resources.occlusion_culling_enabled() && self.render_advances_state(),
            TargetSet::of(if ssaa { T::SsaaColour } else { T::HdrDepth }),
            TargetSet::of(T::HiZ),
        );
        if let Some(last) = table.passes.last_mut() {
            last.side_effect = true;
        }
        table.declare(
            HdrPass::SsaaResolve,
            "ssaa_resolve",
            ssaa,
            TargetSet::of(T::SsaaColour),
            scene.with(T::IndirectFraction),
        );
        table.declare(
            HdrPass::SurfaceMask,
            "surface_mask",
            self.surface_mask.active(),
            TargetSet::of(T::HdrDepth),
            TargetSet::of(T::HdrDepth).with(T::SurfaceStencil),
        );
        table.declare_side_effect(
            HdrPass::ScopeOnOpaqueSurfaces,
            "scope_on_opaque_surfaces",
            self.any_plugin_encodes_at(crate::plugin_api::EncoderScope::OnOpaqueSurfaces),
        );
        table.declare(
            HdrPass::SubHighlight,
            "sub_highlight",
            slot.sub_highlight.is_some(),
            scene,
            TargetSet::of(T::HdrColour),
        );
        table.declare(
            HdrPass::DepthRead,
            "depth_read",
            self.hdr_depth_read_active(frame),
            scene.with(T::SurfaceStencil),
            TargetSet::of(T::HdrColour),
        );
        // Transmissive meshes read a copy of the opaque scene taken here, so
        // they see the sub-highlight and depth-read output but not one
        // another, and draw before the plugins that sample the scene.
        let transmission = self.has_transmissive_batches()
            && self.resources.transmission.mip_pipelines().is_some();
        table.declare(
            HdrPass::OpaqueCopy,
            "opaque_copy",
            transmission,
            scene,
            TargetSet::of(T::OpaqueCopy),
        );
        table.declare(
            HdrPass::Transmission,
            "transmission",
            transmission,
            scene.with(T::OpaqueCopy),
            TargetSet::of(T::HdrColour),
        );
        table.declare_side_effect(
            HdrPass::ScopeAfterOpaque,
            "scope_after_opaque",
            self.any_plugin_encodes_at(crate::plugin_api::EncoderScope::AfterOpaque),
        );
        table.declare(
            HdrPass::Oit,
            "oit",
            self.hdr_has_transparent(frame, scene_items),
            scene.with(T::SurfaceStencil),
            TargetSet::of(T::HdrColour)
                .with(T::OitAccum)
                .with(T::OitReveal),
        );
        table.declare_side_effect(
            HdrPass::ScopeAfterTransparent,
            "scope_after_transparent",
            self.any_plugin_encodes_at(crate::plugin_api::EncoderScope::AfterTransparent),
        );
        table.declare(
            HdrPass::OutlineComposite,
            "outline_composite",
            !slot.selection_outlines.outline_object_buffers.is_empty()
                || slot.selection_outlines.plugin_outline_present,
            scene.with(T::Outline),
            TargetSet::of(T::HdrColour),
        );
        table.declare(
            HdrPass::Foreground,
            "foreground",
            foreground,
            TargetSet::of(T::HdrColour),
            TargetSet::of(T::HdrColour).with(T::ForegroundDepth),
        );

        // The built-in composite-input producers, in their encode order.
        let inputs = crate::resources::ProducerFrameInputs {
            post: pp,
            proj: frame.camera.render_camera.projection(),
            view: self.origin.camera(&frame.camera.render_camera).view(),
            near: frame.camera.render_camera.depth_range().0,
            far: frame.camera.render_camera.depth_range().1,
            first_light: frame.effects.lighting.lights.first(),
            foreground_active: foreground,
            exposure: frame.effects.display.exposure,
            presented: self.render_advances_state(),
        };
        // The composite reads what each enabled producer writes, except
        // while the effects throttle skips that producer: its target would
        // hold an older frame. A producer the composite does not read is
        // culled below.
        let mut composite_reads = scene.with_if(T::ForegroundDepth, foreground);
        for (i, producer) in self.resources.post_producers().into_iter().enumerate() {
            let live = producer.enabled(&inputs);
            let writes = producer.writes();
            if live && (!throttled || !producer.throttleable()) {
                composite_reads = composite_reads | writes;
            }
            table.declare(
                HdrPass::Producer(i as u8),
                producer.name(),
                live,
                producer.reads(&inputs),
                writes,
            );
        }
        let external_producers = !self.post_effect_producers.is_empty() && !throttled;
        table.declare_side_effect(
            HdrPass::ExternalProducers,
            "external_producers",
            external_producers,
        );
        // Ambient occlusion and contact shadows scale by the indirect
        // fraction. An external producer may stand in for either.
        let occludes = composite_reads.contains(T::Ssao)
            || composite_reads.contains(T::ContactShadow)
            || external_producers;
        let composite_reads = composite_reads.with_if(T::IndirectFraction, occludes);

        let stages = pp.fxaa || self.post_effect_stages.iter().any(|e| e.stage.enabled());
        table.declare(
            HdrPass::ToneMap,
            "tone_map",
            true,
            composite_reads,
            TargetSet::of(T::Output).with_if(T::FxaaInput, pp.fxaa),
        );
        table.declare(
            HdrPass::PostStages,
            "post_stages",
            stages,
            TargetSet::of(T::Output).with_if(T::FxaaInput, pp.fxaa),
            TargetSet::of(T::Output),
        );
        table.declare(
            HdrPass::Upscale,
            "upscale",
            scaled,
            TargetSet::of(T::Output),
            TargetSet::of(T::Output),
        );
        table.declare(
            HdrPass::OutputDepthBlit,
            "output_depth_blit",
            scaled,
            TargetSet::of(T::HdrDepth),
            TargetSet::of(T::Output),
        );
        table.declare(
            HdrPass::ForegroundStamp,
            "foreground_depth_stamp",
            foreground,
            TargetSet::of(T::ForegroundDepth),
            TargetSet::of(T::Output),
        );
        let editor_overlays =
            !slot.constraint_line_buffers.is_empty() || !slot.xray_object_buffers.is_empty();
        table.declare(
            HdrPass::SceneOverlays,
            "scene_overlays",
            frame.viewport.show_grid
                || !matches!(
                    frame.effects.ground_plane.mode,
                    crate::renderer::types::GroundPlaneMode::None
                )
                || editor_overlays,
            TargetSet::of(T::HdrDepth),
            TargetSet::of(T::Output),
        );
        table.declare(
            HdrPass::Overlay,
            "overlay",
            slot.overlay.overlay_shape_gpu_data.is_some() || slot.overlay.label_gpu_data.is_some(),
            TargetSet::of(T::HdrDepth),
            TargetSet::of(T::Output),
        );

        table.cull();
        self.viewport_slots[vp_idx].pass_table = table;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(table: &mut PassTable, live: bool) {
        table.clear();
        for i in 0..MAX_PASSES {
            table.declare(
                HdrPass::Producer(i as u8),
                "p",
                live,
                TargetSet::of(TargetId::HdrColour),
                TargetSet::of(TargetId::Output),
            );
        }
    }

    /// A full table refills in place frame after frame.
    #[test]
    fn replanning_reuses_the_table() {
        let mut table = PassTable::default();
        fill(&mut table, true);
        let buffer = table.passes.as_ptr();
        for frame in 0..4 {
            fill(&mut table, frame % 2 == 0);
            assert_eq!(table.passes.as_ptr(), buffer);
        }
    }

    /// A pass whose output nothing reads is culled; one that feeds a
    /// surviving pass, or has side effects, stays.
    #[test]
    fn cull_keeps_only_what_reaches_the_output() {
        let mut table = PassTable::default();
        table.clear();
        let colour = TargetSet::of(TargetId::HdrColour);
        table.declare(HdrPass::Scene, "scene", true, TargetSet::empty(), colour);
        table.declare(
            HdrPass::Producer(0),
            "bloom",
            true,
            colour,
            TargetSet::of(TargetId::Bloom),
        );
        table.declare_side_effect(HdrPass::ExternalProducers, "external", true);
        table.declare(
            HdrPass::ToneMap,
            "tone_map",
            true,
            colour,
            TargetSet::of(TargetId::Output),
        );
        table.cull();
        assert_eq!(
            table.names(),
            vec![
                ("scene", true),
                ("bloom", false),
                ("external", true),
                ("tone_map", true)
            ]
        );
        assert!(!table.composite_inputs(None, None).bloom);
        assert_eq!(table.groups(), crate::resources::TargetGroups::SCENE);
    }
}
