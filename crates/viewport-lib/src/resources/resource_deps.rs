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
//! [`ResourceGate`] turns the two global epochs into the cheap three-way
//! decision every cache makes per frame, and [`ResourceDeps`] is the per-entry
//! record of what a cached binding actually names, so a free only discards the
//! entries it touched.

use crate::resources::DeviceResources;
use crate::resources::TextureId;
use crate::resources::mesh::mesh_store::MeshId;

/// The freeable resources one cached GPU binding names.
///
/// Covers the shapes that exist today: at most one mesh and up to five
/// texture slots (albedo, normal, AO, metallic-roughness, emissive). A site
/// with fewer slots leaves the rest `None`.
///
/// Crate-internal on purpose. A plugin already knows which ids it baked into
/// each entry of its own store and checks them with
/// [`has_texture`](DeviceResources::has_texture) and
/// [`mesh_index_count`](DeviceResources::mesh_index_count); what it cannot
/// work out for itself is *when* to look, which is [`ResourceGate`].
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

/// What a cache of GPU bindings should do with its entries this frame.
///
/// The verdict from [`ResourceGate::poll`].
///
/// Matched exhaustively on purpose: a fourth verdict would be a new thing a
/// cache has to do, and a `_` arm would quietly treat it as "nothing to do".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Revalidate {
    /// Nothing was freed or replaced since the last poll: every entry is
    /// still valid.
    Valid,
    /// Something was freed. Keep each entry whose ids still resolve; rebuild
    /// or null out the rest.
    CheckEach,
    /// A view was swapped behind a live id. No per-entry check can tell which
    /// entries are affected: rebuild them all.
    RebuildAll,
}

/// Per-cache record of the resource epochs it was last validated against.
///
/// An item type that bakes a `TextureView` into a bind group holds that view
/// for as long as it keeps the bind group. When the host frees the texture, the
/// bind group keeps it alive and the draw keeps sampling it; when the host
/// replaces the pixels behind a live id, the swap never arrives. Neither is
/// visible from the ids alone, because a free is a global event and a replace
/// does not change the id at all.
///
/// Keep one gate beside the store, poll it once at the top of `prepare`, and
/// act on the verdict:
///
/// ```no_run
/// # use viewport_lib::plugin_api::{ItemFrameContext, PluginItemCollection};
/// # use viewport_lib::resources::{ResourceGate, Revalidate, TextureId};
/// # use viewport_lib::wgpu;
/// # struct Entry { texture: Option<TextureId>, bind_group: wgpu::BindGroup }
/// # struct MyPlugin { gate: ResourceGate, entries: Vec<Entry> }
/// # impl MyPlugin {
/// # fn rebind(&mut self, _index: usize, _ctx: &ItemFrameContext<'_>) {}
/// fn revalidate(&mut self, ctx: &ItemFrameContext<'_>) {
///     let verdict = self.gate.poll(ctx.resources);
///     for index in 0..self.entries.len() {
///         let stale = match verdict {
///             Revalidate::Valid => false,
///             Revalidate::RebuildAll => true,
///             Revalidate::CheckEach => self.entries[index]
///                 .texture
///                 .is_some_and(|id| !ctx.resources.has_texture(id)),
///         };
///         if stale {
///             self.rebind(index, ctx);
///         }
///     }
/// }
/// # }
/// ```
///
/// Rebind rather than discard. A stored entry belongs to the host, which holds
/// a handle to it and did not ask for it to go away; the right response to a
/// freed texture is to rebuild the binding against the fallback view and forget
/// the dead id, so the entry keeps drawing and picks up a replacement if one
/// arrives later.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResourceGate {
    free_epoch: u64,
    view_epoch: u64,
}

impl ResourceGate {
    /// Compare against the current epochs, catch up, and say what the cache
    /// has to do. Call once per frame before using cached entries.
    ///
    /// A fresh gate reports [`Revalidate::Valid`] on its first poll against an
    /// untouched `DeviceResources`, so a store built this frame is not made to
    /// rebuild itself immediately.
    pub fn poll(&mut self, resources: &DeviceResources) -> Revalidate {
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
