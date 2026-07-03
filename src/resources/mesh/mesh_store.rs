//! Slotted mesh storage with generational handles.
//!
//! `MeshStore` manages GPU mesh lifetimes using a slot-based approach: removed
//! meshes leave empty slots that are reused by subsequent inserts. Each slot
//! carries a generation counter that is bumped on removal, and a [`MeshId`]
//! captures the generation it was issued against. A lookup with a handle whose
//! generation no longer matches the slot returns `None`, so a stale handle held
//! across a remove-then-reinsert cannot silently alias the new mesh.

use crate::resources::GpuMesh;

crate::resources::handle::slot_handle! {
    /// Handle to a mesh in the store.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. A handle whose generation is stale (its slot was freed and
    /// reused) resolves to `None` on lookup rather than aliasing a different
    /// mesh.
    pub struct MeshId;
}

/// One mesh slot: the mesh (when occupied), the slot's current generation, and
/// the GPU byte size charged for it (for resident-bytes accounting).
struct Slot {
    mesh: Option<GpuMesh>,
    generation: u32,
    bytes: u64,
}

/// Slotted storage for GPU meshes with generational handles, a free list, and a
/// maintained resident-byte total.
pub(crate) struct MeshStore {
    slots: Vec<Slot>,
    free_list: Vec<usize>,
    allocated_bytes: u64,
}

impl MeshStore {
    /// Create an empty mesh store.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            allocated_bytes: 0,
        }
    }

    /// Insert a mesh, reusing a free slot if available. Returns the assigned
    /// `MeshId` carrying the slot's current generation.
    pub fn insert(&mut self, mesh: GpuMesh) -> MeshId {
        let bytes = mesh.gpu_byte_size();
        self.allocated_bytes += bytes;
        if let Some(idx) = self.free_list.pop() {
            let slot = &mut self.slots[idx];
            slot.mesh = Some(mesh);
            slot.bytes = bytes;
            MeshId::new(idx as u32, slot.generation)
        } else {
            let idx = self.slots.len();
            self.slots.push(Slot {
                mesh: Some(mesh),
                generation: 0,
                bytes,
            });
            MeshId::new(idx as u32, 0)
        }
    }

    /// Look up the live slot for `id`, or `None` if the index is out of range,
    /// the slot is empty, or the handle's generation is stale.
    fn live_slot(&self, id: MeshId) -> Option<&Slot> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.mesh.is_some().then_some(slot)
    }

    /// Get a reference to the mesh at the given ID, or `None` if the slot is
    /// empty, out of range, or the handle is stale.
    pub fn get(&self, id: MeshId) -> Option<&GpuMesh> {
        self.live_slot(id)?.mesh.as_ref()
    }

    /// Get a mutable reference to the mesh at the given ID.
    pub fn get_mut(&mut self, id: MeshId) -> Option<&mut GpuMesh> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.mesh.as_mut()
    }

    /// Replace the mesh at the given ID with a new one.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::MeshSlotEmpty`] if the slot is empty, out of
    /// bounds, or the handle's generation is stale.
    pub fn replace(&mut self, id: MeshId, mesh: GpuMesh) -> crate::error::ViewportResult<()> {
        let bytes = mesh.gpu_byte_size();
        match self.slots.get_mut(id.index as usize) {
            Some(slot) if slot.generation == id.generation && slot.mesh.is_some() => {
                self.allocated_bytes = self.allocated_bytes - slot.bytes + bytes;
                slot.bytes = bytes;
                slot.mesh = Some(mesh);
                Ok(())
            }
            _ => Err(crate::error::ViewportError::MeshSlotEmpty {
                index: id.index as usize,
            }),
        }
    }

    /// Remove a mesh, dropping its GPU buffers, bumping the slot's generation,
    /// and pushing the slot to the free list.
    ///
    /// Returns `true` if a mesh was actually removed, `false` if the slot was
    /// already empty, out of range, or the handle was stale.
    pub fn remove(&mut self, id: MeshId) -> bool {
        if let Some(slot) = self.slots.get_mut(id.index as usize) {
            if slot.generation == id.generation && slot.mesh.is_some() {
                slot.mesh = None;
                self.allocated_bytes = self.allocated_bytes.saturating_sub(slot.bytes);
                slot.bytes = 0;
                // Bump so the just-removed handle no longer matches this slot.
                slot.generation = slot.generation.wrapping_add(1);
                self.free_list.push(id.index as usize);
                return true;
            }
        }
        false
    }

    /// Number of occupied (non-empty) slots.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.mesh.is_some()).count()
    }

    /// Total number of slots (occupied + free).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Total GPU buffer bytes across every resident mesh.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Whether the slot for the given ID contains a live mesh.
    pub fn contains(&self, id: MeshId) -> bool {
        self.get(id).is_some()
    }

    /// Mutably iterate every live mesh with its handle. Used by texture release
    /// to invalidate the object bind groups that referenced a freed texture.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (MeshId, &mut GpuMesh)> {
        self.slots.iter_mut().enumerate().filter_map(|(idx, slot)| {
            let generation = slot.generation;
            slot.mesh
                .as_mut()
                .map(|mesh| (MeshId::new(idx as u32, generation), mesh))
        })
    }
}
