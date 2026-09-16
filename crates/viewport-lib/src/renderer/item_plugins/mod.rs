//! Internal item types implemented on [`ItemTypePlugin`], one directory per
//! type: the trait impl, the type's pipeline state, and its WGSL live
//! together, the same shape an external item-type crate has.
//!
//! Internal plugins register at renderer construction under a
//! `viewport.`-prefixed type name and read their items from the matching
//! `SceneFrame` field via [`plugin_items_for`], so the consumer-facing
//! submission surface is unchanged: consumers keep filling the field, and
//! the dispatchers route it to the plugin as if it had been submitted under
//! the plugin's name.

pub(crate) mod gaussian_splat;
pub(crate) mod gpu_implicit;
pub(crate) mod gpu_marching_cubes;
pub(crate) mod image_slice;
pub(crate) mod volume;
pub(crate) mod volume_surface_slice;

use crate::plugin_api::PluginItemCollection;
use crate::renderer::types::FrameData;

/// The per-frame collection for a registered plugin: internal item types
/// read their `SceneFrame` field, external plugins read the collection
/// submitted under their type name.
pub(crate) fn plugin_items_for<'f>(
    frame: &'f FrameData,
    name: &str,
) -> Option<&'f dyn PluginItemCollection> {
    match name {
        gaussian_splat::TYPE_NAME => Some(&frame.scene.gaussian_splats),
        gpu_implicit::TYPE_NAME => Some(&frame.scene.gpu_implicit),
        gpu_marching_cubes::TYPE_NAME => Some(&frame.scene.gpu_mc_items),
        image_slice::TYPE_NAME => Some(&frame.scene.image_slices),
        volume::TYPE_NAME => Some(&frame.scene.volumes),
        volume_surface_slice::TYPE_NAME => Some(&frame.scene.volume_surface_slices),
        _ => frame
            .scene
            .plugin_items
            .get(name)
            .map(|items| items.as_ref()),
    }
}

impl crate::renderer::ViewportRenderer {
    /// Register the internal item-type plugins. Called once at construction;
    /// external registration through
    /// [`with_item_type_plugin`](Self::with_item_type_plugin) is unaffected.
    pub(crate) fn register_internal_item_plugins(&mut self, device: &crate::gpu::Device) {
        self.with_item_type_plugin(
            device,
            Box::new(gaussian_splat::GaussianSplatPlugin::default()),
        );
        self.with_item_type_plugin(device, Box::new(gpu_implicit::GpuImplicitPlugin::default()));
        self.with_item_type_plugin(
            device,
            Box::new(gpu_marching_cubes::GpuMarchingCubesPlugin::default()),
        );
        self.with_item_type_plugin(device, Box::new(image_slice::ImageSlicePlugin::default()));
        self.with_item_type_plugin(device, Box::new(volume::VolumePlugin::default()));
        self.with_item_type_plugin(
            device,
            Box::new(volume_surface_slice::VolumeSurfaceSlicePlugin::default()),
        );
    }
}
