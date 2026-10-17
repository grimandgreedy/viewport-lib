//! Slotted material storage with generational ids.
//!
//! A material is uploaded once, keeps its `MaterialId` for its lifetime, and is
//! edited in place. Each stored material owns one block in the material table
//! (`MaterialTable`), the GPU form the mesh shaders index by `material_id`; the
//! source `Material` stays here for the CPU decisions that read it (batch keys,
//! bind groups, the instance record).

use crate::resources::handle::SlotStore;
use crate::scene::material::Material;

pub use viewport_lib_types::ids::MaterialId;

/// One stored material: the source values and the table block they built.
pub(crate) struct StoredMaterial {
    pub(crate) material: Material,
    pub(crate) block: u32,
}

/// Slotted storage for uploaded materials. Slot 0 holds the default material
/// for the store's lifetime, so [`MaterialId::DEFAULT`] always resolves.
pub(crate) struct MaterialStore {
    store: SlotStore<StoredMaterial, MaterialId>,
}

impl MaterialStore {
    /// A store holding only the default material, in table block 0.
    pub(crate) fn new() -> Self {
        let mut store = SlotStore::default();
        let id = store.insert(
            StoredMaterial {
                material: Material::default(),
                block: 0,
            },
            0,
        );
        debug_assert_eq!(id, MaterialId::DEFAULT);
        Self { store }
    }

    pub(crate) fn insert(&mut self, material: Material, block: u32) -> MaterialId {
        self.store.insert(StoredMaterial { material, block }, 0)
    }

    pub(crate) fn get(&self, id: MaterialId) -> Option<&StoredMaterial> {
        self.store.get(id)
    }

    pub(crate) fn get_mut(&mut self, id: MaterialId) -> Option<&mut StoredMaterial> {
        self.store.get_mut(id)
    }

    /// The error for a stale `id`.
    pub(crate) fn stale(&self, id: MaterialId) -> crate::error::ViewportError {
        self.store.stale(id)
    }

    pub(crate) fn remove(&mut self, id: MaterialId) -> Option<StoredMaterial> {
        self.store.remove(id)
    }

    /// The material `id` names, or the default when the handle is stale.
    pub(crate) fn material_or_default(&self, id: MaterialId) -> &Material {
        match self.store.get(id) {
            Some(s) => &s.material,
            None => {
                &self
                    .store
                    .get(MaterialId::DEFAULT)
                    .expect("default material is never freed")
                    .material
            }
        }
    }

    /// The table block `id` names, or the default material's block when stale.
    #[cfg(test)]
    pub(crate) fn block_or_default(&self, id: MaterialId) -> u32 {
        self.store.get(id).map_or(0, |s| s.block)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (MaterialId, &StoredMaterial)> {
        self.store.iter()
    }
}

/// Whether `a` and `b` differ only in fields that nothing but the material
/// block reads, so an edit between them needs no batch rebuild.
///
/// Copies `b`'s block-only fields onto `a` and compares. A field not listed
/// here counts as one the batch key or the instance record reads, so a field
/// added to `Material` later rebuilds by default rather than going stale.
pub(crate) fn differs_only_in_block(a: &Material, b: &Material) -> bool {
    let mut probe = *a;
    probe.ambient = b.ambient;
    probe.diffuse = b.diffuse;
    probe.specular = b.specular;
    probe.shininess = b.shininess;
    probe.metallic = b.metallic;
    probe.roughness = b.roughness;
    probe.normal_strength = b.normal_strength;
    probe.emissive = b.emissive;
    probe.emissive_strength = b.emissive_strength;
    probe.uv_offset = b.uv_offset;
    probe.uv_scale = b.uv_scale;
    probe.uv_rotation = b.uv_rotation;
    probe.texture_transforms = b.texture_transforms;
    probe.ao_range = b.ao_range;
    probe.metallic_range = b.metallic_range;
    probe.roughness_range = b.roughness_range;
    probe == *b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Colour;

    #[test]
    fn default_material_is_slot_zero_and_resolves() {
        let store = MaterialStore::new();
        assert_eq!(store.block_or_default(MaterialId::DEFAULT), 0);
        assert_eq!(
            *store.material_or_default(MaterialId::DEFAULT),
            Material::default()
        );
    }

    #[test]
    fn stale_handle_reads_the_default() {
        let mut store = MaterialStore::new();
        let mut m = Material::default();
        m.roughness = 0.1;
        let id = store.insert(m, 7);
        assert_eq!(store.block_or_default(id), 7);
        store.remove(id);
        assert_eq!(store.block_or_default(id), 0);
        assert_eq!(
            store.material_or_default(id).roughness,
            Material::default().roughness
        );
    }

    #[test]
    fn scalar_edits_are_block_only_and_colour_is_not() {
        let a = Material::default();
        let mut b = a;
        b.roughness = 0.2;
        b.metallic = 0.9;
        b.emissive_strength = 3.0;
        assert!(differs_only_in_block(&a, &b));
        let mut c = a;
        c.base_colour = Colour::linear_rgb(1.0, 0.0, 0.0);
        assert!(
            !differs_only_in_block(&a, &c),
            "base colour rides the record"
        );
        let mut d = a;
        d.texture_id = Some(crate::resources::TextureId::INVALID);
        assert!(
            !differs_only_in_block(&a, &d),
            "textures are in the batch key"
        );
    }
}
