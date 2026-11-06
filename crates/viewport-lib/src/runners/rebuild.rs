//! Moving a [`ViewportInstance`] to a new device after the old one was lost.

use super::ViewportInstance;
use crate::plugin_api::{ItemTypeId, PostEffectProducerId, PostEffectSlot, PostEffectStageId};
use crate::renderer::ViewportRenderer;
use crate::scene::scene::Scene;
use crate::{DeformerId, MaterialId, MaterialPluginId, MeshId, ShadingHookId, TextureId};

/// The handles an instance's scene still names after
/// [`ViewportInstance::rebuild`], and the plugins the host had registered on
/// the old renderer.
///
/// Each handle belonged to the old renderer and draws nothing on the new one:
/// upload the content again and point the nodes at the new handles
/// (`scene_mut().set_mesh(..)`, `set_material(..)`). Each plugin is gone from
/// the new renderer: register it again, under the name listed, and replace
/// the old id wherever the host kept it.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct StaleHandles {
    meshes: Vec<MeshId>,
    materials: Vec<MaterialId>,
    textures: Vec<TextureId>,
    deformers: Vec<StaleDeformer>,
    material_plugins: Vec<StaleMaterialPlugin>,
    shading_hooks: Vec<StaleShadingHook>,
    item_types: Vec<StaleItemType>,
    post_effect_producers: Vec<StalePostEffectProducer>,
    post_effect_stages: Vec<StalePostEffectStage>,
}

/// A deformer registered with
/// [`register_deformer`](crate::resources::DeviceResources::register_deformer)
/// on the old renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StaleDeformer {
    /// The id the old renderer returned.
    pub id: DeformerId,
    /// The descriptor's name.
    pub name: String,
}

/// A material plugin registered with
/// [`register_material_plugin`](crate::resources::DeviceResources::register_material_plugin)
/// on the old renderer. Materials that select it carry `id` in
/// [`shading_plugin`](crate::Material::shading_plugin).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StaleMaterialPlugin {
    /// The id the old renderer returned.
    pub id: MaterialPluginId,
    /// The plugin's name.
    pub name: String,
}

/// A shading hook registered with
/// [`register_shading_hook`](crate::resources::DeviceResources::register_shading_hook)
/// on the old renderer. Material plugins are listed on their own, not here.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StaleShadingHook {
    /// The id the old renderer returned.
    pub id: ShadingHookId,
    /// The descriptor's name.
    pub name: String,
}

/// An item type registered with
/// [`with_item_type_plugin`](crate::ViewportRenderer::with_item_type_plugin)
/// on the old renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StaleItemType {
    /// The id the old renderer returned.
    pub id: ItemTypeId,
    /// The plugin's type name.
    pub type_name: &'static str,
}

/// A post-effect producer registered with
/// [`add_post_effect_producer`](crate::ViewportRenderer::add_post_effect_producer)
/// on the old renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StalePostEffectProducer {
    /// The id the old renderer returned.
    pub id: PostEffectProducerId,
    /// The producer's type name.
    pub type_name: &'static str,
    /// The composite slot it filled.
    pub slot: PostEffectSlot,
}

/// A post-effect stage registered with
/// [`add_post_effect_stage`](crate::ViewportRenderer::add_post_effect_stage)
/// on the old renderer. Two stages of one type share a type name; `order`
/// and the position in the list tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StalePostEffectStage {
    /// The id the old renderer returned.
    pub id: PostEffectStageId,
    /// The stage's type name.
    pub type_name: &'static str,
    /// The order key it was registered at.
    pub order: i32,
}

impl StaleHandles {
    /// The meshes scene nodes name, each once, in node order.
    pub fn meshes(&self) -> &[MeshId] {
        &self.meshes
    }

    /// The materials scene nodes name, each once, in node order. The default
    /// material is left out: every renderer has it.
    pub fn materials(&self) -> &[MaterialId] {
        &self.materials
    }

    /// The textures those materials name, each once.
    pub fn textures(&self) -> &[TextureId] {
        &self.textures
    }

    /// The deformers the host registered, in registration order. The
    /// skinning deformer is not listed: installing skinning again brings it
    /// back.
    pub fn deformers(&self) -> &[StaleDeformer] {
        &self.deformers
    }

    /// The material plugins, in registration slot order.
    pub fn material_plugins(&self) -> &[StaleMaterialPlugin] {
        &self.material_plugins
    }

    /// The shading hooks registered directly, in registration slot order.
    pub fn shading_hooks(&self) -> &[StaleShadingHook] {
        &self.shading_hooks
    }

    /// The item types the host registered, in registration order. The types
    /// the renderer installs itself are not listed: the new renderer has them.
    pub fn item_types(&self) -> &[StaleItemType] {
        &self.item_types
    }

    /// The post-effect producers, in registration order.
    pub fn post_effect_producers(&self) -> &[StalePostEffectProducer] {
        &self.post_effect_producers
    }

    /// The post-effect stages, in registration order.
    pub fn post_effect_stages(&self) -> &[StalePostEffectStage] {
        &self.post_effect_stages
    }

    /// Read the handles off `scene`, the textures off the materials the
    /// renderer holds for them, and the host's plugin registrations off the
    /// renderer.
    pub(super) fn collect(scene: &Scene, renderer: &ViewportRenderer) -> Self {
        let resources = renderer.resources();
        fn push_once<T: PartialEq>(list: &mut Vec<T>, value: T) {
            if !list.contains(&value) {
                list.push(value);
            }
        }
        let mut stale = Self::default();
        for node in scene.nodes() {
            if let Some(mesh) = node.mesh_id() {
                push_once(&mut stale.meshes, mesh);
            }
            let material = node.material();
            if material != MaterialId::DEFAULT {
                push_once(&mut stale.materials, material);
            }
        }
        for &id in &stale.materials {
            let Some(material) = resources.material(id) else {
                continue;
            };
            let slots = crate::TextureSlot::ALL
                .iter()
                .filter_map(|&slot| material.texture(slot));
            for texture in slots.chain(material.plugin_textures.iter().flatten().copied()) {
                push_once(&mut stale.textures, texture);
            }
        }
        stale.deformers = resources
            .host_deformers()
            .map(|(id, name)| StaleDeformer {
                id,
                name: name.to_owned(),
            })
            .collect();
        stale.material_plugins = resources
            .material_plugin_registrations()
            .map(|(id, name)| StaleMaterialPlugin {
                id,
                name: name.to_owned(),
            })
            .collect();
        stale.shading_hooks = resources
            .raw_shading_hooks()
            .map(|(id, name)| StaleShadingHook {
                id,
                name: name.to_owned(),
            })
            .collect();
        stale.item_types = renderer
            .host_item_types()
            .map(|(id, type_name)| StaleItemType { id, type_name })
            .collect();
        stale.post_effect_producers = renderer
            .post_effect_producer_registrations()
            .map(|(id, type_name, slot)| StalePostEffectProducer {
                id,
                type_name,
                slot,
            })
            .collect();
        stale.post_effect_stages = renderer
            .post_effect_stage_registrations()
            .map(|(id, type_name, order)| StalePostEffectStage {
                id,
                type_name,
                order,
            })
            .collect();
        stale
    }
}

impl ViewportInstance {
    /// Build a new renderer on `device`, with the config this instance was
    /// built with, and drop the old one. For recovery after the old device was
    /// lost (see [`ViewportRenderer`](crate::ViewportRenderer#device-loss)).
    ///
    /// The scene graph, selection, camera, input bindings and frame settings
    /// stay. What does not carry over: every mesh, texture, material and
    /// environment the old renderer held, every plugin registered on it
    /// (deformers, material plugins, shading hooks, item types, post-effect
    /// producers and stages), and the retained items added with
    /// [`add_item`](Self::add_item), which are cleared. The returned
    /// [`StaleHandles`] lists the handles the scene still names and each
    /// plugin with the id and name it was registered under. Register the
    /// plugins again, upload the content again and rebind the nodes. An
    /// attached runtime stays attached, and its GPU plugins run `init_gpu`
    /// again on the new device in the next [`render`](Self::render) or
    /// [`prepare`](Self::prepare), before their `pre_prepare`.
    ///
    /// A feature installed with [`install_plugin`](Self::install_plugin) comes
    /// back with [`reattach_plugin`](Self::reattach_plugin), not another
    /// `install_plugin`: its runtime pieces are still on the attached runtime,
    /// and installing again would run them twice.
    ///
    /// A material that selects a material plugin carries the old plugin id in
    /// [`shading_plugin`](crate::Material::shading_plugin); swap it for the new
    /// one before uploading the material again:
    ///
    /// ```rust,ignore
    /// let stale = instance.rebuild(&device)?;
    /// let mut plugin_ids = HashMap::new();
    /// for old in stale.material_plugins() {
    ///     let plugin = my_plugins.by_name(&old.name);
    ///     let new = instance.resources_mut().register_material_plugin(&device, plugin)?;
    ///     plugin_ids.insert(old.id, new);
    /// }
    /// for &old in stale.materials() {
    ///     let mut material = my_materials[&old].clone();
    ///     material.shading_plugin = material
    ///         .shading_plugin
    ///         .and_then(|id| plugin_ids.get(&id).copied());
    ///     let new = instance.resources_mut().upload_material(material)?;
    ///     // point the nodes that named `old` at `new`
    /// }
    /// ```
    ///
    /// Renderer-side plugin handles such as
    /// [`SkinningPlugin`](crate::plugins::skinning::SkinningPlugin) are not
    /// listed: install them again to get a new handle.
    ///
    /// A device-lost watcher or shared pipeline cache in the original config
    /// belongs to the old device, so the new renderer installs its own watcher
    /// and makes its own cache from the config's `pipeline_cache_data`; pass
    /// new ones with [`rebuild_with_config`](Self::rebuild_with_config).
    pub fn rebuild(
        &mut self,
        device: &crate::gpu::Device,
    ) -> Result<StaleHandles, crate::ViewportError> {
        let mut config = self.config.clone();
        config.device_lost_watcher = None;
        config.pipeline_cache = None;
        self.rebuild_with_config(device, &config)
    }

    /// As [`rebuild`](Self::rebuild), with a new config. Later rebuilds use
    /// it too.
    pub fn rebuild_with_config(
        &mut self,
        device: &crate::gpu::Device,
        config: &crate::RendererConfig,
    ) -> Result<StaleHandles, crate::ViewportError> {
        let mut renderer = crate::ViewportRenderer::try_with_config(device, config)?;
        renderer.set_cpu_pick_cache(true);
        let stale = StaleHandles::collect(&self.scene, &self.renderer);
        self.renderer = renderer;
        self.config = config.clone();
        self.extras.clear();
        if let Some(runtime) = self.runtime.as_mut() {
            runtime.reset_gpu_init();
        }
        Ok(stale)
    }
}
