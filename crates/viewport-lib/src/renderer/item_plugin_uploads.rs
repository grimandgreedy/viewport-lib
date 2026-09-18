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

use super::*;

impl ViewportRenderer {
    /// Upload a Gaussian splat set to the GPU.
    ///
    /// Call once per splat set at startup or when it changes. The returned
    /// [`GaussianSplatId`] is valid until [`free_gaussian_splat`](Self::free_gaussian_splat) is called.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// if `data.positions` is empty or if `positions`, `scales`, `rotations`, and `opacities`
    /// differ in length.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use viewport_lib::error::ViewportError;
    /// # use viewport_lib::renderer::{GaussianSplatData, ViewportRenderer};
    /// # fn demo(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device, queue: &viewport_lib::wgpu::Queue) {
    /// let result = renderer.upload_gaussian_splat(device, queue, &GaussianSplatData::default());
    /// assert!(matches!(result, Err(ViewportError::InvalidGaussianSplatData { .. })));
    /// # }
    /// ```
    pub fn upload_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &GaussianSplatData,
    ) -> crate::error::ViewportResult<GaussianSplatId> {
        self.gaussian_splat_plugin_mut()?
            .upload(device, queue, data)
    }

    /// Replace the splats behind a live [`GaussianSplatId`], keeping the handle.
    ///
    /// Items already holding the handle pick up the new set on the next frame.
    /// Use this for content that changes over time, such as a streamed or
    /// re-trained splat set.
    ///
    /// # Errors
    ///
    /// [`InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// when `data` is empty or its per-attribute vectors disagree in length, or
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` no
    /// longer resolves to a live set.
    pub fn replace_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: GaussianSplatId,
        data: &GaussianSplatData,
    ) -> crate::error::ViewportResult<()> {
        self.gaussian_splat_plugin_mut()?
            .replace(device, queue, id, data)
    }

    /// Remove an uploaded Gaussian splat set by handle.
    ///
    /// After this call the `id` is invalid and must not be submitted in `SceneFrame`.
    pub fn free_gaussian_splat(&mut self, id: GaussianSplatId) {
        if let Ok(plugin) = self.gaussian_splat_plugin_mut() {
            plugin.free(id);
        }
    }

    /// The registered Gaussian splat item type, which holds the uploaded sets.
    fn gaussian_splat_plugin_mut(
        &mut self,
    ) -> crate::error::ViewportResult<
        &mut crate::renderer::item_plugins::gaussian_splat::GaussianSplatPlugin,
    > {
        let name = crate::renderer::item_plugins::gaussian_splat::TYPE_NAME;
        self.item_type_plugin_mut(name)
            .ok_or(crate::error::ViewportError::ItemTypePluginMissing { type_name: name })
    }

    /// Upload a polyline for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::PolylineItem,
    ) -> crate::resources::PolylineId {
        let host = self.polyline_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a polyline. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_polyline`](Self::upload_result_polyline).
    pub fn begin_upload_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::PolylineItem,
    ) -> crate::resources::JobId {
        let host = self.polyline_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_polyline`](Self::begin_upload_polyline) job.
    pub fn upload_result_polyline(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::PolylineId> {
        let host = self.polyline_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a polyline handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::PolylineId,
        item: &crate::renderer::PolylineItem,
    ) -> bool {
        let host = self.polyline_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a polyline. `false` if the handle does not resolve.
    pub fn drop_polyline(&mut self, id: crate::resources::PolylineId) -> bool {
        self.polyline_host().plugin.drop_stored(id)
    }

    /// Upload a streamtube for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::StreamtubeItem,
    ) -> crate::resources::StreamtubeId {
        let host = self.streamtube_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a streamtube. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_streamtube`](Self::upload_result_streamtube).
    pub fn begin_upload_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::StreamtubeItem,
    ) -> crate::resources::JobId {
        let host = self.streamtube_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_streamtube`](Self::begin_upload_streamtube) job.
    pub fn upload_result_streamtube(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::StreamtubeId> {
        let host = self.streamtube_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a streamtube handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::StreamtubeId,
        item: &crate::renderer::StreamtubeItem,
    ) -> bool {
        let host = self.streamtube_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a streamtube. `false` if the handle does not resolve.
    pub fn drop_streamtube(&mut self, id: crate::resources::StreamtubeId) -> bool {
        self.streamtube_host().plugin.drop_stored(id)
    }

    /// Upload a tube for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::TubeItem,
    ) -> crate::resources::TubeId {
        let host = self.tube_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a tube. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_tube`](Self::upload_result_tube).
    pub fn begin_upload_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::TubeItem,
    ) -> crate::resources::JobId {
        let host = self.tube_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_tube`](Self::begin_upload_tube) job.
    pub fn upload_result_tube(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TubeId> {
        let host = self.tube_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a tube handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::TubeId,
        item: &crate::renderer::TubeItem,
    ) -> bool {
        let host = self.tube_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a tube. `false` if the handle does not resolve.
    pub fn drop_tube(&mut self, id: crate::resources::TubeId) -> bool {
        self.tube_host().plugin.drop_stored(id)
    }

    /// Upload a ribbon for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::RibbonItem,
    ) -> crate::resources::RibbonId {
        let host = self.ribbon_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a ribbon. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_ribbon`](Self::upload_result_ribbon).
    pub fn begin_upload_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::RibbonItem,
    ) -> crate::resources::JobId {
        let host = self.ribbon_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_ribbon`](Self::begin_upload_ribbon) job.
    pub fn upload_result_ribbon(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::RibbonId> {
        let host = self.ribbon_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a ribbon handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::RibbonId,
        item: &crate::renderer::RibbonItem,
    ) -> bool {
        let host = self.ribbon_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a ribbon. `false` if the handle does not resolve.
    pub fn drop_ribbon(&mut self, id: crate::resources::RibbonId) -> bool {
        self.ribbon_host().plugin.drop_stored(id)
    }

    /// Upload a point cloud for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::PointCloudItem,
    ) -> crate::resources::PointCloudId {
        let host = self.point_cloud_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a point cloud. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_point_cloud`](Self::upload_result_point_cloud).
    pub fn begin_upload_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::PointCloudItem,
    ) -> crate::resources::JobId {
        let host = self.point_cloud_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_point_cloud`](Self::begin_upload_point_cloud) job.
    pub fn upload_result_point_cloud(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::PointCloudId> {
        let host = self.point_cloud_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a point cloud handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::PointCloudId,
        item: &crate::renderer::PointCloudItem,
    ) -> bool {
        let host = self.point_cloud_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a point cloud. `false` if the handle does not resolve.
    pub fn drop_point_cloud(&mut self, id: crate::resources::PointCloudId) -> bool {
        self.point_cloud_host().plugin.drop_stored(id)
    }

    /// The registered polyline item type, which holds the uploaded curves.
    fn polyline_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::polyline::PolylinePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::polyline::TYPE_NAME)
            .expect("the built-in polyline item type is registered at construction")
    }

    /// The registered streamtube item type, which holds the uploaded curves.
    fn streamtube_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::curves::StreamtubePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::curves::STREAMTUBE_TYPE_NAME)
            .expect("the built-in streamtube item type is registered at construction")
    }

    /// The registered tube item type, which holds the uploaded curves.
    fn tube_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::curves::TubePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::curves::TUBE_TYPE_NAME)
            .expect("the built-in tube item type is registered at construction")
    }

    /// The registered ribbon item type, which holds the uploaded curves.
    fn ribbon_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::curves::RibbonPlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::curves::RIBBON_TYPE_NAME)
            .expect("the built-in ribbon item type is registered at construction")
    }

    /// The registered glyph item type, which holds the uploaded sets.
    fn glyph_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::glyph::GlyphPlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::glyph::TYPE_NAME)
            .expect("the built-in glyph item type is registered at construction")
    }

    /// The registered tensor glyph item type, which holds the uploaded sets.
    fn tensor_glyph_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<
        '_,
        crate::renderer::item_plugins::tensor_glyph::TensorGlyphPlugin,
    > {
        self.item_type_plugin_host(crate::renderer::item_plugins::tensor_glyph::TYPE_NAME)
            .expect("the built-in tensor glyph item type is registered at construction")
    }

    /// The registered sprite item type, which holds the uploaded batches.
    fn sprite_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::sprite::SpritePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::sprite::TYPE_NAME)
            .expect("the built-in sprite item type is registered at construction")
    }

    /// The registered point cloud item type, which holds the uploaded clouds.
    fn point_cloud_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<
        '_,
        crate::renderer::item_plugins::point_cloud::PointCloudPlugin,
    > {
        self.item_type_plugin_host(crate::renderer::item_plugins::point_cloud::TYPE_NAME)
            .expect("the built-in point cloud item type is registered at construction")
    }

    /// Upload a glyph set for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::GlyphItem,
    ) -> crate::resources::GlyphSetId {
        let host = self.glyph_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a glyph set. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_glyph_set`](Self::upload_result_glyph_set).
    pub fn begin_upload_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::GlyphItem,
    ) -> crate::resources::JobId {
        let host = self.glyph_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_glyph_set`](Self::begin_upload_glyph_set) job.
    pub fn upload_result_glyph_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::GlyphSetId> {
        let host = self.glyph_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a glyph set handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::GlyphSetId,
        item: &crate::renderer::GlyphItem,
    ) -> bool {
        let host = self.glyph_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a glyph set. `false` if the handle does not resolve.
    pub fn drop_glyph_set(&mut self, id: crate::resources::GlyphSetId) -> bool {
        self.glyph_host().plugin.drop_stored(id)
    }

    /// Upload a tensor glyph set for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::TensorGlyphItem,
    ) -> crate::resources::TensorGlyphSetId {
        let host = self.tensor_glyph_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a tensor glyph set. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_tensor_glyph_set`](Self::upload_result_tensor_glyph_set).
    pub fn begin_upload_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::TensorGlyphItem,
    ) -> crate::resources::JobId {
        let host = self.tensor_glyph_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_tensor_glyph_set`](Self::begin_upload_tensor_glyph_set) job.
    pub fn upload_result_tensor_glyph_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TensorGlyphSetId> {
        let host = self.tensor_glyph_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a tensor glyph set handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::TensorGlyphSetId,
        item: &crate::renderer::TensorGlyphItem,
    ) -> bool {
        let host = self.tensor_glyph_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a tensor glyph set. `false` if the handle does not resolve.
    pub fn drop_tensor_glyph_set(&mut self, id: crate::resources::TensorGlyphSetId) -> bool {
        self.tensor_glyph_host().plugin.drop_stored(id)
    }

    /// Upload a sprite set for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::SpriteItem,
    ) -> crate::resources::SpriteSetId {
        let host = self.sprite_host();
        host.plugin.upload_set(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a sprite set. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_sprite_set`](Self::upload_result_sprite_set).
    pub fn begin_upload_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::SpriteItem,
    ) -> crate::resources::JobId {
        let host = self.sprite_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_sprite_set`](Self::begin_upload_sprite_set) job.
    pub fn upload_result_sprite_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::SpriteSetId> {
        let host = self.sprite_host();
        host.plugin.take_set_result(&host.jobs, id)
    }

    /// Replace the geometry behind a sprite set handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::SpriteSetId,
        item: &crate::renderer::SpriteItem,
    ) -> bool {
        let host = self.sprite_host();
        host.plugin
            .replace_set(device, queue, host.resources, id, item)
    }

    /// Release a sprite set. `false` if the handle does not resolve.
    pub fn drop_sprite_set(&mut self, id: crate::resources::SpriteSetId) -> bool {
        self.sprite_host().plugin.drop_set(id)
    }

    /// Upload a sprite instance set for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::SpriteItem,
    ) -> crate::resources::SpriteInstanceSetId {
        let host = self.sprite_host();
        host.plugin
            .upload_instance_set(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a sprite instance set. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_sprite_instance_set`](Self::upload_result_sprite_instance_set).
    pub fn begin_upload_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::SpriteItem,
    ) -> crate::resources::JobId {
        let host = self.sprite_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_sprite_instance_set`](Self::begin_upload_sprite_instance_set) job.
    pub fn upload_result_sprite_instance_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::SpriteInstanceSetId> {
        let host = self.sprite_host();
        host.plugin.take_instance_set_result(&host.jobs, id)
    }

    /// Replace the geometry behind a sprite instance set handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::SpriteInstanceSetId,
        item: &crate::renderer::SpriteItem,
    ) -> bool {
        let host = self.sprite_host();
        host.plugin
            .replace_instance_set(device, queue, host.resources, id, item)
    }

    /// Release a sprite instance set. `false` if the handle does not resolve.
    pub fn drop_sprite_instance_set(&mut self, id: crate::resources::SpriteInstanceSetId) -> bool {
        self.sprite_host().plugin.drop_instance_set(id)
    }

    /// Upload a scalar volume for GPU marching cubes, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_volume_for_mc(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vol: &crate::geometry::marching_cubes::VolumeData,
    ) -> crate::ViewportResult<crate::resources::McVolumeId> {
        self.gpu_marching_cubes_plugin_mut()?
            .upload(device, queue, vol)
    }

    /// Release a marching-cubes volume and its slab buffers.
    ///
    /// Dropping the buffers takes the volume out of
    /// [`resident_bytes`](Self::resident_bytes) immediately; wgpu defers the
    /// real GPU free until in-flight commands referencing them complete. The
    /// emptied slot is reused by a later upload, at a new generation, so the
    /// freed handle cannot alias its successor.
    pub fn free_mc_volume(&mut self, id: crate::resources::McVolumeId) {
        if let Ok(plugin) = self.gpu_marching_cubes_plugin_mut() {
            plugin.free(id);
        }
    }

    /// Feed a marching-cubes volume from a caller-supplied buffer, refreshed
    /// before every dispatch so the isosurface tracks it with no CPU upload.
    ///
    /// The buffer holds one `f32` per volume node in x-fastest order
    /// (`index = x + y * nx + z * nx * ny`), matching `VolumeData::data`,
    /// starting at `offset_bytes`. It needs `COPY_SRC` usage and
    /// `offset_bytes` must be a multiple of 4. The renderer keeps a clone of
    /// the buffer handle; if the consumer reallocates it, call this again with
    /// the new buffer.
    ///
    /// # Errors
    ///
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` does
    /// not resolve to a live volume,
    /// [`ExternalBufferUsageMissing`](crate::error::ViewportError::ExternalBufferUsageMissing)
    /// if the buffer lacks `COPY_SRC`, or
    /// [`McScalarSourceMismatch`](crate::error::ViewportError::McScalarSourceMismatch)
    /// if the offset is misaligned or the volume's scalars do not fit in the
    /// buffer past `offset_bytes`.
    pub fn set_mc_scalar_source_buffer(
        &mut self,
        id: crate::resources::McVolumeId,
        buffer: crate::gpu::Buffer,
        offset_bytes: u64,
    ) -> crate::ViewportResult<()> {
        self.gpu_marching_cubes_plugin_mut()?
            .set_scalar_source(id, buffer, offset_bytes)
    }

    /// Detach the external scalar source, freezing the isosurface at the last
    /// field copied in.
    ///
    /// # Errors
    ///
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` does
    /// not resolve to a live volume.
    pub fn clear_mc_scalar_source(
        &mut self,
        id: crate::resources::McVolumeId,
    ) -> crate::ViewportResult<()> {
        self.gpu_marching_cubes_plugin_mut()?
            .clear_scalar_source(id)
    }

    /// The registered GPU marching cubes item type, which holds the uploaded
    /// volumes.
    fn gpu_marching_cubes_plugin_mut(
        &mut self,
    ) -> crate::error::ViewportResult<
        &mut crate::renderer::item_plugins::gpu_marching_cubes::GpuMarchingCubesPlugin,
    > {
        let name = crate::renderer::item_plugins::gpu_marching_cubes::TYPE_NAME;
        self.item_type_plugin_mut(name)
            .ok_or(crate::error::ViewportError::ItemTypePluginMissing { type_name: name })
    }

    /// Create a persistent GPU particle system, returning its handle.
    pub fn create_gpu_particle_system(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        config: &crate::resources::GpuParticleSystemConfig,
    ) -> crate::resources::GpuParticleSystemId {
        let host = self.gpu_particles_host();
        host.plugin
            .create_system(device, queue, host.resources, config)
    }

    /// Release a GPU particle system. The handle stops resolving and its
    /// buffers are freed.
    pub fn drop_gpu_particle_system(&mut self, id: crate::resources::GpuParticleSystemId) {
        self.gpu_particles_host().plugin.drop_system(id)
    }

    /// The registered GPU particle item type, which holds the live systems.
    fn gpu_particles_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<
        '_,
        crate::renderer::item_plugins::gpu_particles::GpuParticlesPlugin,
    > {
        self.item_type_plugin_host(crate::renderer::item_plugins::gpu_particles::TYPE_NAME)
            .expect("the built-in GPU particle item type is registered at construction")
    }

    /// Create an instance set drawn from a caller-owned positions buffer.
    pub fn create_external_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        config: &crate::resources::ExternalInstanceSetConfig,
    ) -> crate::error::ViewportResult<crate::resources::ExternalInstanceSetId> {
        let host = self.external_instances_host();
        host.plugin.create_set(device, host.resources, config)
    }

    /// Release an external instance set. Items still naming it are skipped.
    pub fn drop_external_instance_set(&mut self, id: crate::resources::ExternalInstanceSetId) {
        self.external_instances_host().plugin.drop_set(id)
    }

    /// Re-point an external instance set at a different positions buffer.
    pub fn set_external_instance_set_buffer(
        &mut self,
        id: crate::resources::ExternalInstanceSetId,
        positions: crate::gpu::Buffer,
    ) -> crate::error::ViewportResult<()> {
        self.external_instances_host()
            .plugin
            .set_buffer(id, positions)
    }

    /// The registered external instances item type, which holds the sets.
    fn external_instances_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<
        '_,
        crate::renderer::item_plugins::external_instances::ExternalInstancesPlugin,
    > {
        self.item_type_plugin_host(crate::renderer::item_plugins::external_instances::TYPE_NAME)
            .expect("the built-in external instances item type is registered at construction")
    }
}
