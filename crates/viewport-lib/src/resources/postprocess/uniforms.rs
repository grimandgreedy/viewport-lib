//! GPU uniform structs for the post-processing passes.

/// Tone mapping uniform.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ToneMapUniform {
    pub(crate) exposure: f32,
    pub(crate) mode: u32, // 0=Reinhard, 1=ACES, 2=KhronosNeutral
    pub(crate) bloom_enabled: u32,
    pub(crate) ssao_enabled: u32,
    pub(crate) contact_shadows_enabled: u32,
    pub(crate) edl_enabled: u32,
    pub(crate) edl_radius: f32,
    pub(crate) edl_strength: f32,
    pub(crate) background_colour: [f32; 4],
    /// Near, and the far distance EDL normalises view distance against.
    pub(crate) near_plane: f32,
    pub(crate) far_plane: f32,
    /// Non-zero for an orthographic projection.
    pub(crate) orthographic: u32,
    /// The projection's far plane, 0 when it has none.
    pub(crate) projection_far: f32,
    /// Non-zero when the foreground pass ran this frame. Gates the
    /// foreground-coverage test that skips SSAO/contact-shadow/EDL on
    /// pixels covered by foreground geometry.
    pub(crate) foreground_enabled: u32,
    /// Vignette darkening strength at full falloff (0 = off).
    pub(crate) vignette_amount: f32,
    /// Centre-to-corner fraction where the vignette falloff starts.
    pub(crate) vignette_radius: f32,
    /// Width of the vignette falloff band past the radius.
    pub(crate) vignette_softness: f32,
    /// Non-zero when a grade LUT is bound at the composite's LUT slot.
    pub(crate) grade_enabled: u32,
    /// Slice count / height of the strip LUT (its texture height).
    pub(crate) grade_lut_size: f32,
    pub(crate) _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<ToneMapUniform>() == 96);

/// Bloom pass uniform (16 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BloomUniform {
    pub(crate) threshold: f32,
    pub(crate) intensity: f32,
    pub(crate) horizontal: u32, // 1=horizontal pass, 0=vertical
    /// Firefly cap: luminance ceiling applied per pixel in the threshold
    /// pass, before thresholding. Unused in the blur passes.
    pub(crate) max_brightness: f32,
}

/// SSAO uniform (144 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SsaoUniform {
    pub(crate) inv_proj: [[f32; 4]; 4], // 64 bytes : NDC->view
    pub(crate) proj: [[f32; 4]; 4],     // 64 bytes : view->clip (for re-projection)
    pub(crate) radius: f32,
    pub(crate) bias: f32,
    pub(crate) _pad: [f32; 2],
}

/// Depth of field uniform (48 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DofUniform {
    pub(crate) focal_distance: f32,
    pub(crate) focal_range: f32,
    pub(crate) max_blur_radius: f32,
    /// Near and far of the projection; far is 0 when it has none.
    pub(crate) near_plane: f32,
    pub(crate) far_plane: f32,
    pub(crate) viewport_width: f32,
    pub(crate) viewport_height: f32,
    /// Non-zero when the foreground pass ran this frame. Pixels covered by
    /// foreground geometry pass through unblurred.
    pub(crate) foreground_enabled: f32,
    /// Non-zero for an orthographic projection.
    pub(crate) orthographic: f32,
    pub(crate) _pad: [f32; 3],
}

const _: () = assert!(std::mem::size_of::<DofUniform>() == 48);

/// Shadow atlas uniform (368 bytes, bound at group 0 binding 5).
///
/// Contains per-cascade view-projection matrices, split distances, the
/// distance fade, atlas layout, and the PCSS sun size. Used by the fragment
/// shader for CSM cascade selection and shadow sampling.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ShadowAtlasUniform {
    /// 4 cascade view-projection matrices (each 64 bytes = 4x4 f32). Flattened.
    pub(crate) cascade_view_proj: [[f32; 4]; 16], // 256 bytes
    /// Distance-based split values for cascade selection (eye-to-fragment distance).
    pub(crate) cascade_splits: [f32; 4], //  16 bytes
    /// Number of active cascades (1-4).
    pub(crate) cascade_count: u32, //   4 bytes
    /// Backing atlas texture size in pixels: the resolved
    /// `ShadowSettings::atlas_resolution`.
    ///
    /// Shaders derive everything from this plus the atlas rects: one texel in
    /// UV space is `1.0 / atlas_size`, and a tile's pixel width is
    /// `atlas_size * (rect.z - rect.x)`, half the atlas.
    pub(crate) atlas_size: f32, //   4 bytes
    /// Shadow filter mode: 0=PCF, 1=PCSS.
    pub(crate) shadow_filter: u32, //   4 bytes
    /// `tan(ShadowSettings::sun_angular_radius)`: world penumbra radius per
    /// world unit between blocker and receiver, for the PCSS tiers.
    pub(crate) tan_sun: f32, //   4 bytes
    /// `[far, fade_fraction, reserved, r_max_world]`: `far` is where the
    /// cascades end along the view (0 = no distance cut), `fade_fraction`
    /// the share of `far` over which shadows fade out before it.
    /// `r_max_world` caps the PCSS search and filter radius in world units,
    /// 24 texels of cascade 0, so the widest penumbra is the same in every
    /// cascade.
    pub(crate) range: [f32; 4], //  16 bytes
    /// Per-cascade atlas UV rects: [uv_min.x, uv_min.y, uv_max.x, uv_max.y].
    pub(crate) atlas_rects: [[f32; 4]; 4], //  64 bytes
}
// Total: 256 + 16 + 16 + 16 + 64 = 368 bytes
const _: () = assert!(std::mem::size_of::<ShadowAtlasUniform>() == 368);

/// Uniform for the shadow atlas blit overlay. 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AtlasBlitUniform {
    /// NDC rect [xmin, ymin, xmax, ymax].
    pub(crate) rect: [f32; 4],
}

/// Contact shadow uniform (176 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ContactShadowUniform {
    pub(crate) inv_proj: [[f32; 4]; 4],  // 64 bytes
    pub(crate) proj: [[f32; 4]; 4],      // 64 bytes
    pub(crate) light_dir_view: [f32; 4], // 16 bytes
    pub(crate) world_up_view: [f32; 4],  // 16 bytes
    pub(crate) params: [f32; 4],         // 16 bytes: [max_distance, steps, thickness, pad]
}
