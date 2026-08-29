//! Fallback material textures, shared samplers, and the texture-group layout.
//!
//! Every lit draw binds a full material texture set; meshes without a given map
//! bind these 1x1 neutral fallbacks instead, so the bind group layout is always
//! satisfied. Grouped off `DeviceResources` as a plain data holder; construction
//! and pixel upload stay in `resources::init` / the material upload paths.

/// Neutral fallback textures and the shared material samplers.
pub(crate) struct MaterialFallbacks {
    /// Bind group layout for texture group (group 2: albedo + sampler + normal_map + ao_map).
    pub(crate) texture_bgl: crate::gpu::BindGroupLayout,
    /// Fallback 1x1 white texture used when `material.texture_id` is None.
    pub(crate) texture: crate::resources::GpuTexture,
    /// D2Array view of `texture`, bound at lightmap bindings 17/18 for meshes
    /// without a lightmap. Those bindings are `texture_2d_array` so a multi-page
    /// lightmap can select an atlas layer per vertex.
    pub(crate) texture_array_view: crate::gpu::TextureView,
    /// Fallback 1x1 flat normal map [128,128,255,255] (tangent-space neutral).
    pub(crate) normal_map: crate::gpu::Texture,
    pub(crate) normal_map_view: crate::gpu::TextureView,
    /// Fallback 1x1 AO map [255,255,255,255] (no occlusion).
    pub(crate) ao_map: crate::gpu::Texture,
    pub(crate) ao_map_view: crate::gpu::TextureView,
    /// Fallback 1x1 metallic-roughness texture [0, 255, 255, 255].
    /// G=1.0 and B=1.0 so scalar factors pass through unchanged when no ORM texture is set.
    /// Kept alive for `metallic_roughness_view`; not read directly after construction.
    #[allow(dead_code)]
    pub(crate) metallic_roughness: crate::gpu::Texture,
    pub(crate) metallic_roughness_view: crate::gpu::TextureView,
    /// Fallback 1x1 emissive texture [0, 0, 0, 255] (no emission).
    /// Kept alive for `emissive_view`; not read directly after construction.
    #[allow(dead_code)]
    pub(crate) emissive: crate::gpu::Texture,
    pub(crate) emissive_view: crate::gpu::TextureView,
    /// Shared linear-repeat sampler for material textures.
    pub(crate) sampler: crate::gpu::Sampler,
    /// Shared linear-clamp sampler for colourmap LUT lookups.
    pub(crate) lut_sampler: crate::gpu::Sampler,
    /// Non-filtering clamp sampler for the read-only-depth plugin pass. Bound
    /// with the scene depth-only view so plugin shaders sample scene depth.
    pub(crate) depth_read_sampler: crate::gpu::Sampler,
    /// Bind group layout for the read-only-depth plugin pass (binding 0: depth
    /// texture, binding 1: non-filtering sampler). A convenience for plugins
    /// with a spare bind group; the renderer builds the matching bind group
    /// each frame from the depth-only view. Plugins already using all four
    /// groups fold the two bindings into an existing group instead.
    pub(crate) depth_read_bgl: crate::gpu::BindGroupLayout,
    /// Whether the fallback normal map / AO map pixels have been uploaded.
    pub(crate) uploaded: bool,
}

#[cfg(test)]
mod tests {
    /// The neutral fallbacks are 1x1 and present at construction, and the upload
    /// flag starts clear. Guards the init-assembly grouping.
    #[test]
    fn material_fallbacks_are_wired() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let m = &res.material;
        assert!(
            !m.uploaded,
            "fallback pixels are uploaded lazily, not at new()"
        );
        for (t, name) in [
            (&m.normal_map, "normal"),
            (&m.ao_map, "ao"),
            (&m.metallic_roughness, "metallic_roughness"),
            (&m.emissive, "emissive"),
        ] {
            assert_eq!(t.width(), 1, "{name} fallback is 1x1");
            assert_eq!(t.height(), 1, "{name} fallback is 1x1");
        }
    }
}
