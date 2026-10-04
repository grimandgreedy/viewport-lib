//! Shadow-map GPU resources: the cascade atlas, point-light shadow cube array,
//! their depth passes, and the atlas debug viewer.
//!
//! Grouped off `DeviceResources` as a plain data holder. Construction stays in
//! `resources::init`; the shadow pass in `renderer::prepare::shadow_pass` and the
//! lit pass in `renderer::prepare::lighting` read these fields each frame.

use crate::renderer::pipeline_key::PipelineKey;
use crate::resources::pipeline_slot::LazyFamily;

use crate::resources::builders::LoggedAlloc;

/// Directional-cascade and point-light shadow resources.
pub(crate) struct ShadowResources {
    /// Shadow atlas depth texture (Depth32Float, atlas_size x atlas_size, 2x2
    /// tile grid). Read for its dimensions by `shadow_allocation_bytes`.
    pub(crate) map_texture: crate::gpu::Texture,
    /// Depth texture view for binding as a shader resource (sampling).
    pub(crate) map_view: crate::gpu::TextureView,
    /// Whether [`map_texture`](Self::map_texture) is the full-size atlas.
    ///
    /// It starts as a 1x1 placeholder. The atlas is 64 MB at
    /// `SHADOW_ATLAS_SIZE`, and a scene that casts no cascade shadow never
    /// samples it, so the allocation waits until a frame needs it.
    /// [`DeviceResources::ensure_shadow_atlas`](crate::resources::DeviceResources::ensure_shadow_atlas)
    /// promotes it and rebuilds the bind groups that sample it.
    pub(crate) atlas_allocated: bool,
    /// Comparison sampler for PCF shadow filtering.
    pub(crate) sampler: crate::gpu::Sampler,
    /// Cubemap-array depth texture for point-light shadows. Layered as
    /// `MAX_POINT_SHADOW_LIGHTS * 6` faces of `POINT_SHADOW_FACE_SIZE` px.
    pub(crate) point_cube_texture: crate::gpu::Texture,
    /// `texture_depth_cube_array` view bound to the lit-pass bind group.
    pub(crate) point_cube_view: crate::gpu::TextureView,
    /// Whether [`point_cube_texture`](Self::point_cube_texture) is the
    /// full-size cube array.
    ///
    /// It starts as a single 1x1 cube. The full array is 192 MB, and a scene
    /// whose point lights do not cast (or which has no point lights, the
    /// common case) never samples it.
    /// [`DeviceResources::ensure_point_shadow_cubes`](crate::resources::DeviceResources::ensure_point_shadow_cubes)
    /// promotes it.
    pub(crate) point_cubes_allocated: bool,
    /// One 2D-array view per face, used as the depth attachment during the
    /// shadow render pass. `len() == MAX_POINT_SHADOW_LIGHTS * 6`, indexed
    /// as `slot * 6 + face`.
    pub(crate) point_face_views: Vec<crate::gpu::TextureView>,
    /// Render pipeline for the point-shadow depth pass. Same vertex layout
    /// as the cascade shadow pipeline; writes linear distance-to-light.
    /// `None` until a frame has point-shadow faces to draw; see
    /// [`DeviceResources::ensure_point_shadow_pipeline`](crate::resources::DeviceResources::ensure_point_shadow_pipeline).
    pub(crate) point_pipeline: Option<LazyFamily<ShadowRecipe, 1>>,
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
    /// `None` until a frame rasterises casters into the atlas; see
    /// [`DeviceResources::ensure_cascade_shadow_pipelines`](crate::resources::DeviceResources::ensure_cascade_shadow_pipelines).
    pub(crate) pipeline: Option<LazyFamily<ShadowRecipe, 4>>,
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
    pub(crate) atlas_viewer_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group for the atlas viewer (uniform + depth texture + sampler).
    pub(crate) atlas_viewer_bg: crate::gpu::BindGroup,
    /// Layout of `atlas_viewer_bg`, kept so the group can be rebuilt when the
    /// atlas texture is promoted from its placeholder.
    pub(crate) atlas_viewer_bgl: crate::gpu::BindGroupLayout,
    /// Uniform buffer: NDC rect of the atlas viewer quad.
    pub(crate) atlas_viewer_buf: crate::gpu::Buffer,
}

/// Create the cascade atlas depth texture and its sampling view at `size`
/// square. Called twice: once in `init` at 1x1 for the placeholder, once from
/// `ensure_shadow_atlas` at `SHADOW_ATLAS_SIZE`.
pub(crate) fn create_atlas_texture(
    device: &crate::gpu::Device,
    size: u32,
) -> (crate::gpu::Texture, crate::gpu::TextureView) {
    let texture = device.logged_texture(&crate::gpu::TextureDescriptor {
        label: Some("shadow_atlas"),
        size: crate::gpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::Depth32Float,
        usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
            | crate::gpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the point-light cube array for `lights` cubes of `face_size` square,
/// with the `CubeArray` sampling view and one 2D view per face for the depth
/// passes. Called at 1x1x1 cube for the placeholder and at
/// `POINT_SHADOW_FACE_SIZE` / `MAX_POINT_SHADOW_LIGHTS` when promoted.
pub(crate) fn create_point_cube_array(
    device: &crate::gpu::Device,
    face_size: u32,
    lights: u32,
) -> (
    crate::gpu::Texture,
    crate::gpu::TextureView,
    Vec<crate::gpu::TextureView>,
) {
    let layers = lights * 6;
    let texture = device.logged_texture(&crate::gpu::TextureDescriptor {
        label: Some("point_shadow_cube_array"),
        size: crate::gpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::Depth32Float,
        usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
            | crate::gpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let cube_view = texture.create_view(&crate::gpu::TextureViewDescriptor {
        label: Some("point_shadow_cube_view"),
        // iOS Metal does not support CubeArray views. Use D2Array instead;
        // the shader is patched at build time to match.
        dimension: Some(if cfg!(target_os = "ios") {
            crate::gpu::TextureViewDimension::D2Array
        } else {
            crate::gpu::TextureViewDimension::CubeArray
        }),
        aspect: crate::gpu::TextureAspect::DepthOnly,
        base_array_layer: 0,
        array_layer_count: Some(layers),
        base_mip_level: 0,
        mip_level_count: Some(1),
        format: Some(crate::gpu::TextureFormat::Depth32Float),
        usage: None,
    });
    let face_views = (0..layers)
        .map(|layer| {
            texture.create_view(&crate::gpu::TextureViewDescriptor {
                label: Some("point_shadow_face_view"),
                dimension: Some(crate::gpu::TextureViewDimension::D2),
                aspect: crate::gpu::TextureAspect::DepthOnly,
                base_array_layer: layer,
                array_layer_count: Some(1),
                base_mip_level: 0,
                mip_level_count: Some(1),
                format: Some(crate::gpu::TextureFormat::Depth32Float),
                usage: None,
            })
        })
        .collect();
    (texture, cube_view, face_views)
}

/// What a per-object shadow pipeline build reads.
pub(crate) struct ShadowRecipe {
    pub(crate) device: crate::gpu::Device,
    pub(crate) layout: crate::gpu::PipelineLayout,
    pub(crate) shader: crate::gpu::ShaderModule,
}

/// Slot index of the cascade pipeline for `key`: bit 0 two-sided, bit 1 cutout.
fn cascade_index(key: PipelineKey) -> usize {
    key.two_sided as usize + 2 * key.cutout as usize
}

pub(crate) fn build_cascade(r: &ShadowRecipe, i: usize) -> crate::gpu::RenderPipeline {
    let two_sided = i & 1 != 0;
    let cutout = i & 2 != 0;
    crate::resources::mesh::mesh_pipelines::build_shadow_pipeline(
        &r.device,
        &r.layout,
        &r.shader,
        (!two_sided).then_some(crate::gpu::Face::Front),
        cutout,
        None,
    )
}

pub(crate) fn build_point(r: &ShadowRecipe, _i: usize) -> crate::gpu::RenderPipeline {
    crate::resources::mesh::mesh_pipelines::build_shadow_point_pipeline(
        &r.device, &r.layout, &r.shader, None,
    )
}

impl ShadowResources {
    /// The cascade depth pipeline for `key`'s facedness and cutout, or `None`
    /// while a worker has it (or before the family is composed by
    /// [`DeviceResources::ensure_cascade_shadow_pipelines`](crate::resources::DeviceResources::ensure_cascade_shadow_pipelines)).
    pub(crate) fn cascade(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.pipeline.as_ref()?.get(cascade_index(key))
    }

    /// The point-shadow depth pipeline, composed alongside the cube array.
    pub(crate) fn point(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.point_pipeline.as_ref()?.get(0)
    }
}

#[cfg(test)]
mod tests {
    /// A freshly constructed `DeviceResources` wires every shadow resource, and
    /// the two big depth textures start as placeholders rather than at full
    /// size: the atlas is 64 MB and the point cube array 192 MB, and a viewport
    /// that never casts a shadow must not pay either. Guards the grouping
    /// against a dropped field and the placeholder contract against a
    /// regression back to eager allocation.
    #[test]
    fn shadow_resources_are_wired_and_start_as_placeholders() {
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
            shadow.map_texture.format(),
            crate::gpu::TextureFormat::Depth32Float,
            "cascade atlas is a depth texture"
        );
        assert!(
            !shadow.atlas_allocated,
            "the cascade atlas starts as a placeholder"
        );
        assert!(
            shadow.map_texture.width() < shadow.atlas_size,
            "the placeholder is smaller than the atlas it stands in for"
        );
        assert!(
            !shadow.point_cubes_allocated,
            "the point cube array starts as a placeholder"
        );
        assert!(
            shadow.point_face_views.len() < (crate::renderer::MAX_POINT_SHADOW_LIGHTS * 6) as usize,
            "the placeholder covers fewer faces than the full array"
        );
        assert_eq!(
            shadow.point_face_views.len() % 6,
            0,
            "point-shadow face views cover whole cubes (6 faces each)"
        );
    }

    /// Promotion is what the first shadow-casting frame does: both textures
    /// reach full size, the face views cover every slot, and a second call is a
    /// no-op so a steady-state frame does not reallocate.
    #[test]
    fn promotion_allocates_full_size_once() {
        let Some((device, _queue, mut res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        assert!(res.ensure_shadow_atlas(&device), "first call allocates");
        assert!(!res.ensure_shadow_atlas(&device), "second call is a no-op");
        assert_eq!(res.shadow.map_texture.width(), res.shadow.atlas_size);
        assert!(res.shadow.atlas_allocated);

        assert!(
            res.ensure_point_shadow_cubes(&device),
            "first call allocates"
        );
        assert!(
            !res.ensure_point_shadow_cubes(&device),
            "second call is a no-op"
        );
        assert_eq!(
            res.shadow.point_face_views.len(),
            (crate::renderer::MAX_POINT_SHADOW_LIGHTS * 6) as usize
        );
        assert_eq!(
            res.shadow.point_cube_texture.width(),
            crate::renderer::POINT_SHADOW_FACE_SIZE
        );
        assert!(res.shadow.point_cubes_allocated);
    }

    /// Same completeness guarantee as `scene_pipelines::hdr_opaque_resolves_every_key_once_built`
    /// and `postprocess::oit::tests::oit_pipeline_resolves_every_key_once_built`: every key in
    /// `PipelineKey::all()` must resolve through `get()` without panicking, including the
    /// `no_discard_eligible` combinations this family ignores (a depth-only pass has no
    /// discard-free early-Z distinction).
    #[test]
    fn shadow_pipeline_resolves_every_key_once_built() {
        let Some((device, _queue, mut res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        res.pipeline_compiler
            .set_policy(crate::resources::PipelineCompilation::Blocking);
        res.ensure_cascade_shadow_pipelines(&device);
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            assert!(res.shadow.cascade(key).is_some());
        }
    }
}
