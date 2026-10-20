use crate::resources::DeviceResources;
use crate::resources::material::material_store::{MaterialId, differs_only_in_block};
use crate::resources::material_gpu::ItemShading;
use crate::scene::material::Material;
use viewport_lib_types::plugin::material::MATERIAL_PLUGIN_MAX_PARAMS;

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
    /// [`ViewportError::InvalidMaterialParams`](crate::error::ViewportError::InvalidMaterialParams)
    /// when the plugin values do not match the plugin's declaration.
    pub fn upload_material(
        &mut self,
        material: Material,
    ) -> crate::error::ViewportResult<MaterialId> {
        self.validate_plugin_values(&material)?;
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
    /// Plugin parameters are block-only; plugin textures rebuild.
    ///
    /// The edit applies to every item sharing the id. To change one item, give
    /// it its own material, or set its `ItemSettings::colour`.
    ///
    /// # Errors
    ///
    /// [`ViewportError::StaleHandle`](crate::error::ViewportError::StaleHandle)
    /// if `id` does not resolve to a live material;
    /// [`ViewportError::InvalidMaterialParams`](crate::error::ViewportError::InvalidMaterialParams)
    /// as for [`upload_material`](Self::upload_material).
    pub fn replace_material(
        &mut self,
        id: MaterialId,
        material: Material,
    ) -> crate::error::ViewportResult<()> {
        let Some(stored) = self.content.materials.get(id) else {
            return Err(self.content.materials.stale(id));
        };
        self.validate_plugin_values(&material)?;
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
        let params = self.plugin_param_window(&m);
        let limit = self.material_gpu_limit;
        self.material_table.variant(id, shading, base, limit, |t| {
            let mut block = t.build(&m, resolved, shading);
            block.params = params;
            block
        })
    }

    fn build_material_block(
        &self,
        m: &Material,
        shading: &ItemShading,
    ) -> crate::resources::material_gpu::MaterialGpu {
        let mut block = self
            .material_table
            .build(m, self.resolve_material_slots(m), shading);
        block.params = self.plugin_param_window(m);
        block
    }

    /// The parameter words a material's block carries: its plugin's declared
    /// parameters in order, each the material's value or the declared default.
    /// Zero without a registered plugin.
    fn plugin_param_window(&self, m: &Material) -> [[f32; 4]; MATERIAL_PLUGIN_MAX_PARAMS] {
        let mut window = [[0.0; 4]; MATERIAL_PLUGIN_MAX_PARAMS];
        let Some(gpu) = m
            .shading_plugin
            .and_then(|p| self.material_plugins.get(&p.plugin_index()))
        else {
            return window;
        };
        for (word, (desc, key)) in window
            .iter_mut()
            .zip(gpu.params.iter().zip(gpu.param_keys.iter()))
        {
            let value = m
                .plugin_params
                .iter()
                .find(|(k, v)| k == key && v.fits(desc.kind))
                .map_or(desc.default, |(_, v)| v);
            *word = value.to_vec4();
        }
        window
    }

    /// Check a material's plugin values against what its plugin declares.
    /// Without a plugin they are ignored.
    fn validate_plugin_values(&self, m: &Material) -> crate::error::ViewportResult<()> {
        let Some(plugin) = m.shading_plugin else {
            return Ok(());
        };
        let invalid =
            |reason: String| crate::error::ViewportError::InvalidMaterialParams { reason };
        let names_textures = m.plugin_textures.iter().any(Option::is_some);
        let Some(gpu) = self.material_plugins.get(&plugin.plugin_index()) else {
            if m.plugin_params.is_empty() && !names_textures {
                return Ok(());
            }
            return Err(invalid(format!(
                "material plugin {} is not registered, so its values cannot be checked",
                plugin.plugin_index()
            )));
        };
        let name = self.shade_hooks[plugin.plugin_index() as usize].desc.name;
        if m.plugin_params.overflowed() {
            return Err(invalid(format!(
                "more than {MATERIAL_PLUGIN_MAX_PARAMS} parameters set for '{name}'"
            )));
        }
        for (key, value) in m.plugin_params.iter() {
            let Some(i) = gpu.param_keys.iter().position(|k| *k == key) else {
                let declared: Vec<&str> = gpu.params.iter().map(|d| d.name.as_str()).collect();
                return Err(invalid(format!(
                    "a parameter '{name}' does not declare (it declares: {})",
                    declared.join(", ")
                )));
            };
            let desc = &gpu.params[i];
            if !value.fits(desc.kind) {
                return Err(invalid(format!(
                    "'{name}' parameter '{}' is {}, not {value:?}",
                    desc.name,
                    desc.kind.wgsl()
                )));
            }
        }
        if let Some(slot) = m.plugin_textures.iter().rposition(Option::is_some)
            && slot as u32 >= gpu.texture_count
        {
            return Err(invalid(format!(
                "plugin texture slot {slot} set, but '{name}' declares {} textures",
                gpu.texture_count
            )));
        }
        Ok(())
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

/// Whether `m` names texture `id` in one of its built-in slots. Plugin
/// textures are not in the block, so they do not count.
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
