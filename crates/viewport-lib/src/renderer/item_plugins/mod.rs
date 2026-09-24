//! Internal item types implemented on [`ItemTypePlugin`], one directory per
//! type: the trait impl, the type's pipeline state, and its WGSL live
//! together, the same shape an external item-type crate has.
//!
//! Internal plugins register at renderer construction under a
//! `vpl.`-prefixed type name and read their items from the matching
//! `SceneFrame` field via [`plugin_items_for`], so the consumer-facing
//! submission surface is unchanged: consumers keep filling the field, and
//! the dispatchers route it to the plugin as if it had been submitted under
//! the plugin's name.

pub(crate) mod registry;

pub(crate) mod curves;
pub(crate) mod decal;
pub(crate) mod external_instances;
pub(crate) mod gaussian_splat;
pub(crate) mod glyph;
pub(crate) mod gpu_implicit;
pub(crate) mod gpu_marching_cubes;
pub(crate) mod gpu_particles;
pub(crate) mod image_slice;
pub(crate) mod polyline;
pub(crate) mod scatter_volume;
pub(crate) mod sprite;
pub(crate) mod tensor_glyph;
pub(crate) mod volume;
pub(crate) mod volume_surface_slice;

use crate::plugin_api::PluginItemCollection;
use crate::renderer::types::FrameData;

/// The first collection submitted under `name` this frame.
///
/// Kept for the `items` argument the draw hooks still take. A type with
/// several item forms reads each by type with
/// [`ItemFrameContext::items_of`](crate::plugin_api::ItemFrameContext::items_of)
/// instead, which is what every built-in type does; an external plugin
/// submitting one collection sees no difference.
pub(crate) fn plugin_items_for<'f>(
    frame: &'f FrameData,
    name: &str,
) -> Option<&'f dyn PluginItemCollection> {
    frame
        .scene
        .plugin_items
        .get(name)
        .and_then(|slot| slot.first())
        .map(|boxed| boxed.as_ref())
}

/// Every collection submitted under `name` this frame, in submission order.
pub(crate) fn plugin_collections_slice<'f>(
    frame: &'f FrameData,
    name: &str,
) -> &'f [Box<dyn PluginItemCollection>] {
    frame
        .scene
        .plugin_items
        .get(name)
        .map(|slot| slot.as_slice())
        .unwrap_or(&[])
}

/// Every collection submitted for `name` this frame. Queries that ask "did
/// this plugin get anything" or "what pick ids does it own" walk this rather
/// than the first collection alone, so a frame carrying only reference items
/// is not mistaken for an empty one.
pub(crate) fn plugin_collections_for<'f>(
    frame: &'f FrameData,
    name: &str,
) -> impl Iterator<Item = &'f dyn PluginItemCollection> {
    plugin_collections_slice(frame, name)
        .iter()
        .map(|boxed| boxed.as_ref())
}

impl crate::renderer::ViewportRenderer {
    /// Register the internal item-type plugins. Called once at construction;
    /// external registration through
    /// [`with_item_type_plugin`](Self::with_item_type_plugin) is unaffected.
    pub(crate) fn register_internal_item_plugins(&mut self, device: &crate::gpu::Device) {
        // Registration order is draw order. The scivis types keep the order the
        // shared draw loop gave them, so a migrated type keeps blending against
        // its neighbours the way it always has.
        // First the types that came off the shared scivis draw loop, in the
        // order that loop drew them.
        self.install_item_type_plugin(device, Box::new(glyph::GlyphPlugin::default()));
        self.install_item_type_plugin(device, Box::new(polyline::PolylinePlugin::default()));
        self.install_item_type_plugin(device, Box::new(volume::VolumePlugin::default()));
        self.install_item_type_plugin(device, Box::new(curves::StreamtubePlugin::default()));
        self.install_item_type_plugin(device, Box::new(curves::TubePlugin::default()));
        self.install_item_type_plugin(device, Box::new(image_slice::ImageSlicePlugin::default()));
        self.install_item_type_plugin(device, Box::new(tensor_glyph::TensorGlyphPlugin::default()));
        self.install_item_type_plugin(
            device,
            Box::new(volume_surface_slice::VolumeSurfaceSlicePlugin::default()),
        );
        self.install_item_type_plugin(device, Box::new(curves::RibbonPlugin::default()));
        // Then the types that always had a draw site of their own.
        self.install_item_type_plugin(
            device,
            Box::new(gaussian_splat::GaussianSplatPlugin::default()),
        );
        self.install_item_type_plugin(device, Box::new(gpu_implicit::GpuImplicitPlugin::default()));
        self.install_item_type_plugin(
            device,
            Box::new(gpu_marching_cubes::GpuMarchingCubesPlugin::default()),
        );
        // Sprites drew after every other non-mesh type, so they register last,
        // with the particles that shared their pass right behind them.
        self.install_item_type_plugin(device, Box::new(sprite::SpritePlugin::default()));
        self.install_item_type_plugin(
            device,
            Box::new(gpu_particles::GpuParticlesPlugin::default()),
        );
        // External instances drew after every plugin paint when they had a pass
        // of their own, so they register after the last type that paints.
        self.install_item_type_plugin(
            device,
            Box::new(external_instances::ExternalInstancesPlugin::default()),
        );
        self.install_item_type_plugin(
            device,
            Box::new(decal::DecalPlugin::new(self.decal_cache_stats.clone())),
        );
        // Scatter composites over the finished scene, so it registers after
        // every type whose pixels it absorbs.
        self.install_item_type_plugin(
            device,
            Box::new(scatter_volume::ScatterVolumePlugin::default()),
        );
    }
}
