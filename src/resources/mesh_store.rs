//! Slotted mesh storage with free-list removal.
//!
//! `MeshStore` manages GPU mesh lifetimes using a slot-based approach:
//! removed meshes leave `None` slots that are reused by subsequent inserts.
//! This avoids invalidating existing `MeshId` handles when meshes are removed.

use crate::resources::GpuMesh;

/// Opaque handle to a mesh in the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub(crate) usize);

impl MeshId {
    /// Create a `MeshId` from a raw index.
    pub fn from_index(index: usize) -> Self {
        Self(index)
    }

    /// The raw index into the slot array.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Slotted storage for GPU meshes with a free list for slot reuse.
pub(crate) struct MeshStore {
    slots: Vec<Option<GpuMesh>>,
    free_list: Vec<usize>,
}

impl MeshStore {
    /// Create an empty mesh store.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Insert a mesh, reusing a free slot if available. Returns the assigned `MeshId`.
    pub fn insert(&mut self, mesh: GpuMesh) -> MeshId {
        if let Some(idx) = self.free_list.pop() {
            self.slots[idx] = Some(mesh);
            MeshId(idx)
        } else {
            let idx = self.slots.len();
            self.slots.push(Some(mesh));
            MeshId(idx)
        }
    }

    /// Get a reference to the mesh at the given ID, or `None` if the slot is empty/invalid.
    pub fn get(&self, id: MeshId) -> Option<&GpuMesh> {
        self.slots.get(id.0)?.as_ref()
    }

    /// Get a mutable reference to the mesh at the given ID.
    pub fn get_mut(&mut self, id: MeshId) -> Option<&mut GpuMesh> {
        self.slots.get_mut(id.0)?.as_mut()
    }

    /// Replace the mesh at the given ID with a new one.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::MeshSlotEmpty`] if the slot is empty or out of bounds.
    pub fn replace(&mut self, id: MeshId, mesh: GpuMesh) -> crate::error::ViewportResult<()> {
        match self.slots.get_mut(id.0) {
            Some(slot) if slot.is_some() => {
                *slot = Some(mesh);
                Ok(())
            }
            _ => Err(crate::error::ViewportError::MeshSlotEmpty { index: id.0 }),
        }
    }

    /// Remove a mesh, dropping its GPU buffers and pushing the slot to the free list.
    ///
    /// Returns `true` if a mesh was actually removed, `false` if the slot was already empty.
    pub fn remove(&mut self, id: MeshId) -> bool {
        if let Some(slot) = self.slots.get_mut(id.0) {
            if slot.is_some() {
                *slot = None;
                self.free_list.push(id.0);
                return true;
            }
        }
        false
    }

    /// Number of occupied (non-empty) slots.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Total number of slots (occupied + free).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Whether the slot for the given ID contains a mesh.
    pub fn contains(&self, id: MeshId) -> bool {
        self.get(id).is_some()
    }
}
