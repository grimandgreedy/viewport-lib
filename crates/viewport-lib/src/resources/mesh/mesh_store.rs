//! Slotted mesh storage with generational handles.
//!
//! `MeshStore` manages GPU mesh lifetimes using a slot-based approach: removed
//! meshes leave empty slots that are reused by subsequent inserts. Each slot
//! carries a generation counter that is bumped on removal, and a [`MeshId`]
//! captures the generation it was issued against. A lookup with a handle whose
//! generation no longer matches the slot returns `None`, so a stale handle held
//! across a remove-then-reinsert cannot silently alias the new mesh.

use crate::resources::GpuMesh;
use crate::resources::handle::SlotStore;

pub use viewport_lib_types::ids::MeshId;

/// Slotted storage for GPU meshes with generational handles, a free list, and a
/// maintained resident-byte total. An entry's byte charge is its
/// [`GpuMesh::gpu_byte_size`].
pub(crate) struct MeshStore {
    store: SlotStore<GpuMesh, MeshId>,
}

impl MeshStore {
    /// Create an empty mesh store.
    pub fn new() -> Self {
        Self {
            store: SlotStore::default(),
        }
    }

    /// Insert a mesh, reusing a free slot if available. Returns the assigned
    /// `MeshId` carrying the slot's current generation.
    pub fn insert(&mut self, mesh: GpuMesh) -> MeshId {
        let bytes = mesh.gpu_byte_size();
        self.store.insert(mesh, bytes)
    }

    /// Get a reference to the mesh at the given ID, or `None` if the slot is
    /// empty, out of range, or the handle is stale.
    pub fn get(&self, id: MeshId) -> Option<&GpuMesh> {
        self.store.get(id)
    }

    /// Get a mutable reference to the mesh at the given ID.
    pub fn get_mut(&mut self, id: MeshId) -> Option<&mut GpuMesh> {
        self.store.get_mut(id)
    }

    /// Replace the mesh at the given ID with a new one.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::SlotEmpty`] if the slot is empty, out of
    /// bounds, or the handle's generation is stale.
    pub fn replace(&mut self, id: MeshId, mesh: GpuMesh) -> crate::error::ViewportResult<()> {
        let bytes = mesh.gpu_byte_size();
        match self.store.replace(id, mesh, bytes) {
            Some(_) => Ok(()),
            None => Err(crate::error::ViewportError::SlotEmpty { index: id.index() }),
        }
    }

    /// Remove a mesh, dropping its GPU buffers, bumping the slot's generation,
    /// and pushing the slot to the free list.
    ///
    /// Returns `true` if a mesh was actually removed, `false` if the slot was
    /// already empty, out of range, or the handle was stale.
    pub fn remove(&mut self, id: MeshId) -> bool {
        self.store.remove(id).is_some()
    }

    /// Number of occupied (non-empty) slots.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Total number of slots (occupied + free).
    pub fn slot_count(&self) -> usize {
        self.store.slot_count()
    }

    /// Total GPU buffer bytes across every resident mesh.
    pub fn allocated_bytes(&self) -> u64 {
        self.store.allocated_bytes()
    }

    /// Whether the slot for the given ID contains a live mesh.
    pub fn contains(&self, id: MeshId) -> bool {
        self.store.contains(id)
    }

    /// Mutably iterate every live mesh with its handle. Used by texture release
    /// to invalidate the object bind groups that referenced a freed texture.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (MeshId, &mut GpuMesh)> {
        self.store.iter_mut()
    }
}
