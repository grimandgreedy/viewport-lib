//! Whether a mesh item's or batch's colour pipeline is built yet.
//!
//! Under `PipelineCompilation::Background` the colour pass skips whatever is
//! still compiling. The shadow, outline and pick passes draw the same items
//! with other pipelines, and skip them too while their colour pipeline is on
//! a worker: a shadow with no caster, or an outline around nothing, would be
//! wrong for the frames the compile takes. These checks never start a
//! compile; the colour pass does that.

use crate::renderer::pipeline_key::PipelineKey;
use crate::renderer::types::InstancedBatch;
use crate::renderer::{FrameData, SceneRenderItem};
use crate::resources::DeviceResources;

impl DeviceResources {
    /// Whether `frame` has clip geometry, which disables the discard-free
    /// early-Z twin of the opaque pipelines.
    pub(crate) fn clipping_active(frame: &FrameData) -> bool {
        frame
            .effects
            .clip
            .objects
            .iter()
            .any(|o| o.enabled && o.clip_geometry)
    }

    /// Whether the opaque colour pipeline a per-object draw of `item` would
    /// bind is built. Always true under `Blocking`, where the colour pass
    /// builds it on the spot. `hdr` is the colour family the frame draws
    /// with. Transparent items have no shadow or outline to withhold, so the
    /// opaque key is the one that matters.
    pub(crate) fn item_colour_ready(
        &self,
        item: &SceneRenderItem,
        hdr: bool,
        clipping_active: bool,
    ) -> bool {
        if self.pipeline_compiler.policy() == crate::resources::PipelineCompilation::Blocking {
            return true;
        }
        // The colour pass binds the discard-free twin when the early-Z gate
        // allows it and the discarding pipeline otherwise; which of the two a
        // given draw site picks differs, so either counts as ready.
        let two_sided = item.material.is_two_sided();
        let may_skip_discard = hdr
            && !clipping_active
            && !self.force_po_discard
            && matches!(
                item.material.alpha_mode,
                crate::scene::material::AlphaMode::Opaque
            )
            && item.active_attribute.is_none()
            && item.submesh_materials.is_none();
        let keys = [
            PipelineKey::two_sided(two_sided),
            PipelineKey {
                two_sided,
                no_discard_eligible: may_skip_discard,
                ..PipelineKey::default()
            },
        ];
        match self.material_plugin_draw(item.material.shading_plugin) {
            Some((set, _)) => keys.iter().any(|&k| set.opaque_ready(hdr, k)),
            // A plugin item whose set is not composed yet draws nothing.
            None if item.material.shading_plugin.is_some() => false,
            None => keys.iter().any(|&k| self.scene.opaque_ready(hdr, k)),
        }
    }

    /// Whether the opaque colour pipeline an instanced `batch` would bind is
    /// built: the direct-draw solid or its GPU-culled twin, whichever the
    /// colour pass has. Always true under `Blocking`.
    pub(crate) fn batch_colour_ready(
        &self,
        batch: &InstancedBatch,
        hdr: bool,
        clipping_active: bool,
    ) -> bool {
        if self.pipeline_compiler.policy() == crate::resources::PipelineCompilation::Blocking {
            return true;
        }
        let key = PipelineKey {
            two_sided: batch.two_sided,
            no_discard_eligible: !clipping_active && !batch.has_alpha_mask,
            ..PipelineKey::default()
        };
        match self.material_plugin_instanced_draw(batch.shading_plugin) {
            Some((set, _)) => set.opaque_ready(hdr, key),
            None if batch.shading_plugin.is_some() => false,
            None => self.instancing.opaque_ready(hdr, key) || (hdr && self.cull.opaque_ready(key)),
        }
    }
}
