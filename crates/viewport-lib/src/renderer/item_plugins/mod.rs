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
pub(crate) mod point_cloud;
pub(crate) mod polyline;
pub(crate) mod scatter_volume;
pub(crate) mod sprite;
pub(crate) mod tensor_glyph;
pub(crate) mod volume;
pub(crate) mod volume_surface_slice;

use crate::plugin_api::PluginItemCollection;
use crate::plugin_api::item_type::MAX_REF_COLLECTIONS;
use crate::renderer::types::FrameData;

/// The per-frame collection for a registered plugin: internal item types
/// read their `SceneFrame` field, external plugins read the collection
/// submitted under their type name.
pub(crate) fn plugin_items_for<'f>(
    frame: &'f FrameData,
    name: &str,
) -> Option<&'f dyn PluginItemCollection> {
    match name {
        curves::RIBBON_TYPE_NAME => Some(&frame.scene.ribbon_items),
        decal::TYPE_NAME => Some(&frame.scene.decals),
        external_instances::TYPE_NAME => Some(&frame.scene.external_instances),
        curves::STREAMTUBE_TYPE_NAME => Some(&frame.scene.streamtube_items),
        curves::TUBE_TYPE_NAME => Some(&frame.scene.tube_items),
        gaussian_splat::TYPE_NAME => Some(&frame.scene.gaussian_splats),
        gpu_implicit::TYPE_NAME => Some(&frame.scene.gpu_implicit),
        gpu_marching_cubes::TYPE_NAME => Some(&frame.scene.gpu_mc_items),
        gpu_particles::TYPE_NAME => Some(&frame.scene.gpu_particle_systems),
        image_slice::TYPE_NAME => Some(&frame.scene.image_slices),
        glyph::TYPE_NAME => Some(&frame.scene.glyphs),
        point_cloud::TYPE_NAME => Some(&frame.scene.point_clouds),
        polyline::TYPE_NAME => Some(&frame.scene.polylines),
        scatter_volume::TYPE_NAME => Some(&frame.scene.scatter_volumes),
        sprite::TYPE_NAME => Some(&frame.scene.sprite_items),
        tensor_glyph::TYPE_NAME => Some(&frame.scene.tensor_glyphs),
        volume::TYPE_NAME => Some(&frame.scene.volumes),
        volume_surface_slice::TYPE_NAME => Some(&frame.scene.volume_surface_slices),
        _ => frame
            .scene
            .plugin_items
            .get(name)
            .map(|items| items.as_ref()),
    }
}

/// The per-frame reference collection for a registered plugin, for the
/// built-in types that have a reference form: a `*_refs` field whose items
/// name a pre-uploaded payload in an upload store rather than carrying the
/// data inline. `None` for every other type, including every external plugin.
pub(crate) fn plugin_ref_items_for<'f>(
    frame: &'f FrameData,
    name: &str,
) -> [Option<&'f dyn PluginItemCollection>; MAX_REF_COLLECTIONS] {
    let one = |c: &'f dyn PluginItemCollection| [Some(c), None];
    match name {
        curves::RIBBON_TYPE_NAME => one(&frame.scene.ribbon_refs),
        curves::STREAMTUBE_TYPE_NAME => one(&frame.scene.streamtube_refs),
        curves::TUBE_TYPE_NAME => one(&frame.scene.tube_refs),
        glyph::TYPE_NAME => one(&frame.scene.glyph_set_refs),
        point_cloud::TYPE_NAME => one(&frame.scene.point_cloud_refs),
        polyline::TYPE_NAME => one(&frame.scene.polyline_refs),
        // Sprite is the one type with two reference forms: a stored batch and a
        // stored instance set.
        sprite::TYPE_NAME => [
            Some(&frame.scene.sprite_set_refs),
            Some(&frame.scene.sprite_instance_set_refs),
        ],
        tensor_glyph::TYPE_NAME => one(&frame.scene.tensor_glyph_set_refs),
        _ => [None, None],
    }
}

/// Every collection submitted for `name` this frame: the type's own items and
/// its reference items. Queries that ask "did this plugin get anything" or
/// "what pick ids does it own" walk this rather than the items alone, so a
/// frame carrying only reference items is not mistaken for an empty one.
pub(crate) fn plugin_collections_for<'f>(
    frame: &'f FrameData,
    name: &str,
) -> impl Iterator<Item = &'f dyn PluginItemCollection> {
    std::iter::once(plugin_items_for(frame, name))
        .chain(plugin_ref_items_for(frame, name))
        .flatten()
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
        self.with_item_type_plugin(device, Box::new(point_cloud::PointCloudPlugin::default()));
        self.with_item_type_plugin(device, Box::new(glyph::GlyphPlugin::default()));
        self.with_item_type_plugin(device, Box::new(polyline::PolylinePlugin::default()));
        self.with_item_type_plugin(device, Box::new(volume::VolumePlugin::default()));
        self.with_item_type_plugin(device, Box::new(curves::StreamtubePlugin::default()));
        self.with_item_type_plugin(device, Box::new(curves::TubePlugin::default()));
        self.with_item_type_plugin(device, Box::new(image_slice::ImageSlicePlugin::default()));
        self.with_item_type_plugin(device, Box::new(tensor_glyph::TensorGlyphPlugin::default()));
        self.with_item_type_plugin(
            device,
            Box::new(volume_surface_slice::VolumeSurfaceSlicePlugin::default()),
        );
        self.with_item_type_plugin(device, Box::new(curves::RibbonPlugin::default()));
        // Then the types that always had a draw site of their own.
        self.with_item_type_plugin(
            device,
            Box::new(gaussian_splat::GaussianSplatPlugin::default()),
        );
        self.with_item_type_plugin(device, Box::new(gpu_implicit::GpuImplicitPlugin::default()));
        self.with_item_type_plugin(
            device,
            Box::new(external_instances::ExternalInstancesPlugin::default()),
        );
        self.with_item_type_plugin(
            device,
            Box::new(gpu_marching_cubes::GpuMarchingCubesPlugin::default()),
        );
        // Sprites drew after every other non-mesh type, so they register last,
        // with the particles that shared their pass right behind them.
        self.with_item_type_plugin(device, Box::new(sprite::SpritePlugin::default()));
        self.with_item_type_plugin(
            device,
            Box::new(gpu_particles::GpuParticlesPlugin::default()),
        );
        self.with_item_type_plugin(
            device,
            Box::new(decal::DecalPlugin::new(self.decal_cache_stats.clone())),
        );
        // Scatter composites over the finished scene, so it registers after
        // every type whose pixels it absorbs.
        self.with_item_type_plugin(
            device,
            Box::new(scatter_volume::ScatterVolumePlugin::default()),
        );
    }
}
