//! The per-type upload calls for the item types that hold their own content.
//!
//! Each of these forwards to the item type that owns the content, so they live
//! on [`ViewportRenderer`] rather than on `DeviceResources`: the renderer owns
//! the registered item types, and an upload that has to reach a type's own
//! storage can only be reached from this level.
//!
//! The private `*_host` / `*_plugin_mut` accessors interleaved with them turn a
//! built-in type name into that type's concrete plugin, which is the lookup
//! plus downcast every call here needs. One per type, rather than one per
//! method.
//!
//! Those lookups cannot fail. A type the renderer installs registers at
//! construction, there is no call that unregisters one, and
//! [`with_item_type_plugin`](crate::renderer::ViewportRenderer::with_item_type_plugin)
//! refuses a plugin claiming one of their names, so the name resolves and the
//! downcast is to the type that put itself there.

use super::*;

impl ViewportRenderer {
    /// The registered polyline item type, which holds the uploaded curves.
    fn polyline_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::polyline::PolylinePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::polyline::TYPE_NAME)
            .expect(
                "the built-in polyline item type registers at construction, under a name nothing else can take",
            )
    }

    /// The registered glyph item type, which holds the uploaded sets.
    fn glyph_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::glyph::GlyphPlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::glyph::TYPE_NAME)
            .expect(
                "the built-in glyph item type registers at construction, under a name nothing else can take",
            )
    }
}

// ---------------------------------------------------------------------------
// The standard upload surface, one implementation per item type with a store
// ---------------------------------------------------------------------------

/// Implement [`Uploads`](crate::plugin_api::Uploads) for one item type whose
/// plugin exposes the five standard store calls under their usual names.
macro_rules! standard_uploads {
    ($item:ty, $id:ty, $host:ident) => {
        impl crate::plugin_api::Uploads<$item> for ViewportRenderer {
            type Id = $id;

            fn upload(
                &mut self,
                device: &crate::gpu::Device,
                queue: &crate::gpu::Queue,
                item: &$item,
            ) -> crate::error::ViewportResult<$id> {
                let host = self.$host();
                Ok(host.plugin.upload(device, queue, host.resources, item))
            }

            fn begin_upload(
                &mut self,
                device: &crate::gpu::Device,
                queue: &crate::gpu::Queue,
                item: $item,
            ) -> crate::error::ViewportResult<crate::resources::JobId> {
                let host = self.$host();
                Ok(host
                    .plugin
                    .begin_upload(&host.jobs, device, queue, host.resources, item))
            }

            fn replace(
                &mut self,
                device: &crate::gpu::Device,
                queue: &crate::gpu::Queue,
                id: $id,
                item: &$item,
            ) -> crate::error::ViewportResult<()> {
                let host = self.$host();
                host.plugin.replace(device, queue, host.resources, id, item)
            }
        }

        impl crate::plugin_api::Handles<$id> for ViewportRenderer {
            fn upload_result(
                &mut self,
                job: crate::resources::JobId,
            ) -> crate::error::ViewportResult<$id> {
                let host = self.$host();
                host.plugin.take_upload_result(&host.jobs, job)
            }

            fn release(&mut self, id: $id) -> bool {
                self.$host().plugin.drop_stored(id)
            }
        }
    };
}

standard_uploads!(
    crate::renderer::PolylineItem,
    crate::resources::PolylineId,
    polyline_host
);
standard_uploads!(
    crate::renderer::GlyphItem,
    crate::resources::GlyphSetId,
    glyph_host
);
