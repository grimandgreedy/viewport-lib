//! Cached shadow state carried across frames: the cascade, atlas, and
//! point-shadow bookkeeping read and updated each frame. Transient per-frame
//! data such as the cascade matrices and point-shadow faces is not stored here;
//! it flows through `LightingFrame` instead.

use super::point_shadow_pool::PointShadowPool;

pub(crate) struct ShadowState {
    /// Cascade-0 light-space matrix (cached for the ground plane ShadowOnly mode).
    pub(crate) last_cascade0_shadow_mat: glam::Mat4,
    /// LRU allocator for point-light shadow atlas slots.
    pub(crate) point_shadow_pool: PointShadowPool,
    /// Monotonic frame counter for point-shadow LRU bookkeeping.
    pub(crate) point_shadow_frame: u64,
    /// Cascade count from the last prepare.
    pub(crate) last_cascade_count: u32,
    /// Cascade split distances from the last prepare.
    pub(crate) last_cascade_splits: [f32; 4],
    /// Shadow frustum half-extent from the last prepare.
    pub(crate) last_shadow_extent: f32,
    /// Shadow atlas resolution from the last prepare.
    pub(crate) last_shadow_atlas_resolution: u32,
    /// Whether contact shadows were active on the last prepare.
    pub(crate) last_contact_shadow_active: bool,
    /// Cascade splits from the last tracing-log emission (to detect changes).
    pub(crate) last_logged_cascade_splits: [f32; 4],
    /// Shadow atlas uniform from the last prepare.
    pub(crate) last_shadow_atlas_uniform: crate::resources::ShadowAtlasUniform,
    /// Whether the directional atlas currently holds nothing but the cleared
    /// depth value, so a frame that draws no casters can skip re-clearing it.
    /// Cleared by the branch that renders casters into the atlas.
    pub(crate) atlas_cleared: bool,
    /// Per-slot content hash of the cubemap rendered into each point-shadow
    /// pool slot (light position/range plus every in-range caster's mesh
    /// identity, content revision, and model matrix). A slot whose hash is
    /// unchanged skips its six face passes; `None` means never rendered or
    /// invalidated. Indexed by pool slot.
    pub(crate) point_shadow_slot_hashes: Vec<Option<u64>>,
    /// What the lighting prepare last uploaded to each per-frame lighting
    /// buffer, so a frame whose lighting did not change writes nothing.
    pub(crate) written: LightingWrites,
}

/// The contents last written to the lighting and shadow uniforms. `None`
/// until the first write; see [`LightingWrites::changed`].
#[derive(Default)]
pub(crate) struct LightingWrites {
    pub(crate) lights: Option<crate::resources::LightsUniform>,
    pub(crate) light_storage: Vec<u8>,
    pub(crate) cluster_grid: Option<crate::resources::gpu::clustered::ClusterGridUniform>,
    pub(crate) active_lights: Vec<u8>,
    pub(crate) shadow_info: Option<crate::resources::ShadowAtlasUniform>,
    pub(crate) cascades: [Option<[[f32; 4]; 4]>; 4],
}

impl LightingWrites {
    /// Whether `now` differs from what `last` holds, recording it if so.
    pub(crate) fn changed<T: bytemuck::Pod>(last: &mut Option<T>, now: T) -> bool {
        let differs = last
            .as_ref()
            .is_none_or(|l| bytemuck::bytes_of(l) != bytemuck::bytes_of(&now));
        if differs {
            *last = Some(now);
        }
        differs
    }

    /// [`changed`](Self::changed) for a slice of values.
    pub(crate) fn slice_changed<T: bytemuck::Pod>(last: &mut Vec<u8>, now: &[T]) -> bool {
        let bytes: &[u8] = bytemuck::cast_slice(now);
        let differs = last.as_slice() != bytes;
        if differs {
            last.clear();
            last.extend_from_slice(bytes);
        }
        differs
    }
}

impl ShadowState {
    pub(crate) fn new() -> Self {
        Self {
            last_cascade0_shadow_mat: glam::Mat4::IDENTITY,
            point_shadow_pool: PointShadowPool::new(),
            point_shadow_frame: 0,
            last_cascade_count: 0,
            last_cascade_splits: [0.0; 4],
            last_shadow_extent: 20.0,
            last_shadow_atlas_resolution: 4096,
            last_contact_shadow_active: false,
            last_logged_cascade_splits: [f32::MAX; 4],
            last_shadow_atlas_uniform: bytemuck::Zeroable::zeroed(),
            atlas_cleared: false,
            point_shadow_slot_hashes: vec![
                None;
                crate::renderer::types::MAX_POINT_SHADOW_LIGHTS as usize
            ],
            written: LightingWrites::default(),
        }
    }

    /// Forget all cached cubemap contents so every active slot re-renders on
    /// the next frame. Called from `force_dirty` and whenever the pool's
    /// textures are recreated.
    pub(crate) fn invalidate_point_shadow_cache(&mut self) {
        self.point_shadow_slot_hashes.fill(None);
    }
}
