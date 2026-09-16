//! One answer to "did a resource this cached GPU binding holds get freed".
//!
//! A cached bind group (or batch list, or uniform block) names meshes and
//! textures by id. Two events can invalidate it after the fact:
//!
//! - a **free** removes the id, so the cache holds a resource that is gone.
//!   Ids are generational, so a slot freed and reused since resolves through a
//!   different id and is correctly seen as gone. Detectable per entry.
//! - a **replace** swaps the view behind a live id, which no per-entry liveness
//!   check can see. The only safe response is to rebuild everything that might
//!   name the id.
//!
//! [`DepsGate`] turns the two global epochs into the cheap three-way decision
//! every cache makes per frame, and [`ResourceDeps`] is the per-entry record of
//! what a cached binding actually names, so a free only discards the entries
//! it touched.

use crate::resources::DeviceResources;
use crate::resources::TextureId;
use crate::resources::mesh::mesh_store::MeshId;

/// The freeable resources one cached GPU binding names.
///
/// Covers the shapes that exist today: at most one mesh and up to five
/// texture slots (albedo, normal, AO, metallic-roughness, emissive). A site
/// with fewer slots leaves the rest `None`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ResourceDeps {
    /// Mesh the binding draws, when it draws one.
    pub(crate) mesh_id: Option<MeshId>,
    /// Texture ids baked into the binding, in no particular slot order.
    pub(crate) texture_ids: [Option<TextureId>; 5],
}

impl ResourceDeps {
    /// Deps for a binding that names textures only.
    pub(crate) fn textures(ids: [Option<TextureId>; 5]) -> Self {
        Self {
            mesh_id: None,
            texture_ids: ids,
        }
    }

    /// Whether every resource this binding names is still resident.
    ///
    /// `true` means the cached binding still describes exactly what it did
    /// when it was built (modulo replaces, which the view epoch owns).
    pub(crate) fn resolves(&self, resources: &DeviceResources) -> bool {
        self.mesh_id
            .is_none_or(|id| resources.mesh_store.contains(id))
            && self
                .texture_ids
                .iter()
                .flatten()
                .all(|id| resources.content.textures.get(*id).is_some())
    }
}

/// What a cache should do with its entries this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Revalidate {
    /// Nothing was freed or replaced since the last poll: every entry is
    /// still valid.
    Valid,
    /// Something was freed. Keep each entry whose [`ResourceDeps`] still
    /// resolve; drop or rebuild the rest.
    CheckEach,
    /// A view was swapped behind a live id. No per-entry check can tell which
    /// entries are affected: rebuild them all.
    RebuildAll,
}

/// Per-cache record of the resource epochs it was last validated against.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DepsGate {
    free_epoch: u64,
    view_epoch: u64,
}

impl DepsGate {
    /// Compare against the current epochs, catch up, and say what the cache
    /// has to do. Call once per frame before using cached entries.
    pub(crate) fn poll(&mut self, resources: &DeviceResources) -> Revalidate {
        self.poll_epochs(resources.resource_free_epoch, resources.resource_view_epoch)
    }

    /// [`poll`](Self::poll) for gates stored inside `DeviceResources` itself,
    /// where borrowing the whole struct is not possible: pass the two epochs.
    pub(crate) fn poll_epochs(&mut self, free_epoch: u64, view_epoch: u64) -> Revalidate {
        if self.view_epoch != view_epoch {
            self.view_epoch = view_epoch;
            self.free_epoch = free_epoch;
            Revalidate::RebuildAll
        } else if self.free_epoch != free_epoch {
            self.free_epoch = free_epoch;
            Revalidate::CheckEach
        } else {
            Revalidate::Valid
        }
    }
}
