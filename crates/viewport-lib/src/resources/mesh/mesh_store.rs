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
    /// Running total of the host memory held by the retained CPU geometry
    /// copies, maintained alongside the store's GPU byte total so both are
    /// cheap to poll. See [`GpuMesh::cpu_byte_size`].
    cpu_bytes: u64,
}

impl MeshStore {
    /// Create an empty mesh store.
    pub fn new() -> Self {
        Self {
            store: SlotStore::default(),
            cpu_bytes: 0,
        }
    }

    /// Insert a mesh, reusing a free slot if available. Returns the assigned
    /// `MeshId` carrying the slot's current generation.
    pub fn insert(&mut self, mesh: GpuMesh) -> MeshId {
        let bytes = mesh.gpu_byte_size();
        self.cpu_bytes += mesh.cpu_byte_size();
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
        let cpu_bytes = mesh.cpu_byte_size();
        match self.store.replace(id, mesh, bytes) {
            Some(old) => {
                self.cpu_bytes = self.cpu_bytes.saturating_sub(old.cpu_byte_size()) + cpu_bytes;
                Ok(())
            }
            None => Err(crate::error::ViewportError::SlotEmpty { index: id.index() }),
        }
    }

    /// Remove a mesh, dropping its GPU buffers, bumping the slot's generation,
    /// and pushing the slot to the free list.
    ///
    /// Returns `true` if a mesh was actually removed, `false` if the slot was
    /// already empty, out of range, or the handle was stale.
    pub fn remove(&mut self, id: MeshId) -> bool {
        match self.store.remove(id) {
            Some(mesh) => {
                self.cpu_bytes = self.cpu_bytes.saturating_sub(mesh.cpu_byte_size());
                true
            }
            None => false,
        }
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

    /// Total host memory bytes across every resident mesh's retained CPU
    /// geometry copies.
    pub fn cpu_allocated_bytes(&self) -> u64 {
        self.cpu_bytes
    }

    /// Drop the CPU geometry copies retained on one mesh. Returns the host bytes
    /// released, or `None` for a stale or empty handle.
    pub fn release_cpu_geometry(&mut self, id: MeshId) -> Option<u64> {
        let released = self.store.get_mut(id)?.release_cpu_geometry();
        self.cpu_bytes = self.cpu_bytes.saturating_sub(released);
        Some(released)
    }

    /// Drop the CPU geometry copies retained on every resident mesh. Returns the
    /// total host bytes released.
    pub fn release_all_cpu_geometry(&mut self) -> u64 {
        let mut released = 0;
        for (_, mesh) in self.store.iter_mut() {
            released += mesh.release_cpu_geometry();
        }
        self.cpu_bytes = self.cpu_bytes.saturating_sub(released);
        released
    }

    /// Re-derive the host byte charge for one slot after its CPU geometry
    /// copies were replaced in place, given the charge they carried before.
    pub fn recharge_cpu_bytes(&mut self, id: MeshId, previous: u64) {
        let current = self.store.get(id).map_or(0, |m| m.cpu_byte_size());
        self.cpu_bytes = self.cpu_bytes.saturating_sub(previous) + current;
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
