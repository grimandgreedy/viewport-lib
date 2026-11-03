//! Moving a [`ViewportInstance`] to a new device after the old one was lost.

use super::ViewportInstance;
use crate::resources::DeviceResources;
use crate::scene::scene::Scene;
use crate::{MaterialId, MeshId, TextureId};

/// The handles an instance's scene still names after
/// [`ViewportInstance::rebuild`]. Each one belonged to the old renderer and
/// draws nothing on the new one: upload the content again and point the
/// nodes at the new handles (`scene_mut().set_mesh(..)`,
/// `set_material(..)`).
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct StaleHandles {
    meshes: Vec<MeshId>,
    materials: Vec<MaterialId>,
    textures: Vec<TextureId>,
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

    /// Read the handles off `scene`, and the textures off the materials
    /// `resources` holds for them.
    pub(super) fn collect(scene: &Scene, resources: &DeviceResources) -> Self {
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
    /// environment the old renderer held, the item-type plugins and post
    /// effects registered on it, and the retained items added with
    /// [`add_item`](Self::add_item), which are cleared. Register the plugins
    /// again, upload the content again and rebind the nodes to the returned
    /// [`StaleHandles`]. An attached runtime runs its GPU plugins' `init_gpu`
    /// again on the new device before their next `pre_prepare`.
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
        let stale = StaleHandles::collect(&self.scene, self.renderer.resources());
        self.renderer = renderer;
        self.config = config.clone();
        self.extras.clear();
        if let Some(runtime) = self.runtime.as_mut() {
            runtime.reset_gpu_init();
        }
        Ok(stale)
    }
}
