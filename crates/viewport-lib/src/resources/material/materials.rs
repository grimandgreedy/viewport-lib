use crate::resources::DeviceResources;
use crate::resources::material::material_store::{MaterialId, differs_only_in_block};
use crate::resources::material_gpu::ItemShading;
use crate::scene::material::Material;

impl DeviceResources {
    /// Upload a material and return its id.
    ///
    /// Set the id on `SceneRenderItem::material` (or a `Scene` node). Items
    /// sharing an id share one material block and can share a batch. Each call
    /// makes a distinct material, even for equal values, so editing one with
    /// [`replace_material`](Self::replace_material) never changes another.
    ///
    /// A texture the material names may be missing or freed later; that slot
    /// then renders as though it were unset.
    ///
    /// # Errors
    ///
    /// [`ViewportError::TooManyMaterials`](crate::error::ViewportError::TooManyMaterials)
    /// when the table holds as many blocks as one storage-buffer binding of the
    /// device can.
    pub fn upload_material(
        &mut self,
        material: Material,
    ) -> crate::error::ViewportResult<MaterialId> {
        let block = self.build_material_block(&material, &ItemShading::for_material(&material));
        let Some(index) = self.material_table.alloc(block, self.material_gpu_limit) else {
            return Err(crate::error::ViewportError::TooManyMaterials {
                max: self.material_gpu_limit,
            });
        };
        Ok(self.content.materials.insert(material, index))
    }

    /// Replace a material's values in place, keeping its id.
    ///
    /// Every item holding `id` draws the new values on the next frame, with no
    /// reassignment. An edit to the lighting scalars (roughness, metallic,
    /// emissive, the ranges and UV transforms) is one block write; an edit to
    /// anything that decides how items batch or what their instance records
    /// hold (textures, base colour, alpha mode, shading model, back-face policy)
    /// also rebuilds the batches on the next frame, as a scene change does.
    ///
    /// The edit applies to every item sharing the id. To change one item, give
    /// it its own material, or set its `ItemSettings::colour`.
    ///
    /// # Errors
    ///
    /// [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `id` does not resolve to a live material.
    pub fn replace_material(
        &mut self,
        id: MaterialId,
        material: Material,
    ) -> crate::error::ViewportResult<()> {
        let Some(stored) = self.content.materials.get(id) else {
            return Err(self.content.materials.stale(id));
        };
        let (old, block) = (stored.material, stored.block);
        let base = self.build_material_block(&material, &ItemShading::for_material(&material));
        self.material_table.write(block, base);
        for (variant, shading) in self.material_table.variants_of(id) {
            let built = self.build_material_block(&material, &shading);
            self.material_table.write(variant, built);
        }
        if let Some(stored) = self.content.materials.get_mut(id) {
            stored.material = material;
        }
        if !differs_only_in_block(&old, &material) {
            // A live id whose batching or record inputs moved: no liveness
            // check can see it, so force the rebuild.
            self.resource_view_epoch += 1;
        }
        Ok(())
    }

    /// Free a material, reclaiming its block.
    ///
    /// Items still holding `id` draw [`MaterialId::DEFAULT`] from the next
    /// frame. Returns `true` if a material was freed, `false` if `id` did not
    /// resolve to a live material. The default material cannot be freed: the
    /// call returns `false` and logs a warning.
    pub fn free_material(&mut self, id: MaterialId) -> bool {
        if id == MaterialId::DEFAULT {
            tracing::warn!("free_material refused: the default material is never freed");
            return false;
        }
        let Some(stored) = self.content.materials.remove(id) else {
            return false;
        };
        self.material_table.release(stored.block);
        self.material_table.release_variants_of(id);
        // Packed records name the freed block, which a later upload may reuse.
        self.resource_free_epoch += 1;
        self.resource_view_epoch += 1;
        true
    }

    /// The values of an uploaded material, or `None` for a stale id.
    pub fn material(&self, id: MaterialId) -> Option<&Material> {
        self.content.materials.get(id).map(|s| &s.material)
    }

    /// The material `id` names, or the default for a stale id.
    pub(crate) fn material_or_default(&self, id: MaterialId) -> &Material {
        self.content.materials.material_or_default(id)
    }

    /// The block index an item with `shading` reads for material `id`: the
    /// material's own block, or a variant when the item's shading facts differ
    /// from the material's.
    pub(crate) fn material_block(&mut self, id: MaterialId, shading: &ItemShading) -> u32 {
        let Some(stored) = self.content.materials.get(id) else {
            return 0;
        };
        let (base, m) = (stored.block, stored.material);
        if *shading == ItemShading::for_material(&m) {
            return base;
        }
        let resolved = self.resolve_material_slots(&m);
        let limit = self.material_gpu_limit;
        self.material_table
            .variant(id, shading, base, limit, |t| t.build(&m, resolved, shading))
    }

    fn build_material_block(
        &self,
        m: &Material,
        shading: &ItemShading,
    ) -> crate::resources::material_gpu::MaterialGpu {
        self.material_table
            .build(m, self.resolve_material_slots(m), shading)
    }

    /// Rebuild the blocks of every material `pick` selects, and their variants,
    /// against the current texture store. Run when a texture they name is
    /// freed (its slot now reads as unset) and when the texture binding mode is
    /// set.
    pub(crate) fn rebuild_material_blocks(&mut self, pick: impl Fn(&Material) -> bool) {
        let picked: Vec<(MaterialId, Material, u32)> = self
            .content
            .materials
            .iter()
            .filter(|(_, s)| pick(&s.material))
            .map(|(id, s)| (id, s.material, s.block))
            .collect();
        for (id, m, block) in picked {
            let base = self.build_material_block(&m, &ItemShading::for_material(&m));
            self.material_table.write(block, base);
            for (variant, shading) in self.material_table.variants_of(id) {
                let built = self.build_material_block(&m, &shading);
                self.material_table.write(variant, built);
            }
        }
    }
}

/// Whether `m` names texture `id` in any slot.
pub(crate) fn names_texture(m: &Material, id: crate::resources::TextureId) -> bool {
    [
        m.texture_id,
        m.normal_map_id,
        m.ao_map_id,
        m.metallic_roughness_texture_id,
        m.emissive_texture_id,
    ]
    .contains(&Some(id))
}
