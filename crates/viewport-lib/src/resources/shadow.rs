//! Shadow-map GPU resources: the cascade atlas, point-light shadow cube array,
//! their depth passes, and the atlas debug viewer.
//!
//! Grouped off `DeviceResources` as a plain data holder. Construction stays in
//! `resources::init`; the shadow pass in `renderer::prepare::shadow_pass` and the
//! lit pass in `renderer::prepare::lighting` read these fields each frame.

/// Directional-cascade and point-light shadow resources.
pub(crate) struct ShadowResources {
    /// Shadow atlas depth texture (Depth32Float, atlas_size x atlas_size, 2x2 tile grid).
    /// Kept alive for `map_view`; not read directly outside construction.
    #[allow(dead_code)]
    pub(crate) map_texture: crate::gpu::Texture,
    /// Depth texture view for binding as a shader resource (sampling).
    pub(crate) map_view: crate::gpu::TextureView,
    /// Comparison sampler for PCF shadow filtering.
    pub(crate) sampler: crate::gpu::Sampler,
    /// Cubemap-array depth texture for point-light shadows. Layered as
    /// `MAX_POINT_SHADOW_LIGHTS * 6` faces of `POINT_SHADOW_FACE_SIZE` px.
    #[allow(dead_code)]
    pub(crate) point_cube_texture: crate::gpu::Texture,
    /// `texture_depth_cube_array` view bound to the lit-pass bind group.
    pub(crate) point_cube_view: crate::gpu::TextureView,
    /// One 2D-array view per face, used as the depth attachment during the
    /// shadow render pass. `len() == MAX_POINT_SHADOW_LIGHTS * 6`, indexed
    /// as `slot * 6 + face`.
    pub(crate) point_face_views: Vec<crate::gpu::TextureView>,
    /// Render pipeline for the point-shadow depth pass. Same vertex layout
    /// as the cascade shadow pipeline; writes linear distance-to-light.
    pub(crate) point_pipeline: crate::gpu::RenderPipeline,
    /// Bind group layout for the point-shadow per-face uniform (group 0
    /// of the point shadow pass). Kept for pipeline rebuilds.
    #[allow(dead_code)]
    pub(crate) point_face_bgl: crate::gpu::BindGroupLayout,
    /// Per-face uniform buffer holding `view_proj`, `light_pos`, `range`
    /// for every (slot, face) of the point shadow array. Sized as
    /// `MAX_POINT_SHADOW_LIGHTS * 6 * 256` bytes (256-byte dynamic-offset
    /// stride).
    pub(crate) point_face_buf: crate::gpu::Buffer,
    /// Bind group for the point-shadow per-face uniform. Stride is 256;
    /// the per-face render pass sets a dynamic offset.
    pub(crate) point_face_bind_group: crate::gpu::BindGroup,
    /// Shadow depth-pass pipelines, keyed by facedness and cutout (the
    /// no-discard axis is not a real distinction for a depth-only pass, so
    /// both its values map to the same pipeline). Culling front faces (the
    /// `two_sided: false` pipelines) means closed solids cast shadow from
    /// their back face, so a solid's own front face is never compared
    /// against itself in the shadow map; `two_sided: true` uses
    /// `cull_mode: None` and a larger caster-side bias
    /// (`CSM_SHADOW_BIAS_TWO_SIDED`) so both sides of a two-sided mesh
    /// rasterise without self-shadowing. `cutout: true` selects a pipeline
    /// with a fragment stage that samples the caster's albedo alpha and
    /// discards below its cutoff, punching holes for an `AlphaMode::Mask`
    /// material instead of casting a solid silhouette.
    pub(crate) pipeline: crate::renderer::pipeline_key::PipelineVariantSet,
    /// Bind group layout for the shadow camera uniform (group 0 of the
    /// shadow pass). Kept so `register_deformer` can rebuild the shadow
    /// pipeline from a freshly composed shader module.
    pub(crate) camera_bgl: crate::gpu::BindGroupLayout,
    /// Uniform buffer holding the per-cascade light-space view-projection matrix (64 bytes).
    pub(crate) uniform_buf: crate::gpu::Buffer,
    /// Bind group for the shadow pass (group 0: light uniform).
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Uniform buffer for the ShadowAtlasUniform (binding 5 of camera_bgl, 416 bytes).
    pub(crate) info_buf: crate::gpu::Buffer,
    /// Current shadow atlas texture size. Used to detect when atlas needs recreation.
    #[allow(dead_code)]
    pub(crate) atlas_size: u32,
    /// Non-comparison sampler for reading depth values as float (atlas viewer).
    #[allow(dead_code)]
    pub(crate) atlas_depth_sampler: crate::gpu::Sampler,
    /// Pipeline for the shadow atlas corner overlay.
    pub(crate) atlas_viewer_pipeline: crate::gpu::RenderPipeline,
    /// Bind group for the atlas viewer (uniform + depth texture + sampler).
    pub(crate) atlas_viewer_bg: crate::gpu::BindGroup,
    /// Uniform buffer: NDC rect of the atlas viewer quad.
    pub(crate) atlas_viewer_buf: crate::gpu::Buffer,
}

#[cfg(test)]
mod tests {
    /// A freshly constructed `DeviceResources` wires every shadow resource: the
    /// cascade atlas is a square depth texture at the recorded `atlas_size`, and
    /// the point-shadow face-view array has one entry per cube face. Guards the
    /// grouping against a dropped or mis-mapped field in the init assembly.
    #[test]
    fn shadow_resources_are_wired() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let shadow = &res.shadow;
        assert!(shadow.atlas_size > 0, "atlas size must be recorded");
        assert_eq!(
            shadow.map_texture.width(),
            shadow.map_texture.height(),
            "cascade atlas is square"
        );
        assert_eq!(
            shadow.map_texture.width(),
            shadow.atlas_size,
            "atlas texture matches the recorded size"
        );
        assert_eq!(
            shadow.map_texture.format(),
            crate::gpu::TextureFormat::Depth32Float,
            "cascade atlas is a depth texture"
        );
        assert!(
            !shadow.point_face_views.is_empty(),
            "point-shadow face views are allocated"
        );
        assert_eq!(
            shadow.point_face_views.len() % 6,
            0,
            "point-shadow face views cover whole cubes (6 faces each)"
        );
    }

    /// Same completeness guarantee as `scene_pipelines::hdr_opaque_resolves_every_key_once_built`
    /// and `postprocess::oit::tests::oit_pipeline_resolves_every_key_once_built`: every key in
    /// `PipelineKey::all()` must resolve through `get()` without panicking, including the
    /// `no_discard_eligible` combinations this family ignores (a depth-only pass has no
    /// discard-free early-Z distinction). Built eagerly at construction (unlike the HDR/OIT
    /// families, there is no lazy first-use gate), so a fresh renderer already has it.
    #[test]
    fn shadow_pipeline_resolves_every_key_once_built() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            let _ = res.shadow.pipeline.get(key);
        }
    }
}
