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
    pub(crate) near_plane: f32,
    pub(crate) far_plane: f32,
    pub(crate) lic_enabled: u32,
    pub(crate) lic_strength: f32,
    /// Non-zero when the foreground pass ran this frame. Gates the
    /// foreground-coverage test that skips SSAO/contact-shadow/EDL/LIC on
    /// pixels covered by foreground geometry.
    pub(crate) foreground_enabled: u32,
    pub(crate) _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<ToneMapUniform>() == 80);

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

/// Depth of field uniform (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DofUniform {
    pub(crate) focal_distance: f32,
    pub(crate) focal_range: f32,
    pub(crate) max_blur_radius: f32,
    pub(crate) near_plane: f32,
    pub(crate) far_plane: f32,
    pub(crate) viewport_width: f32,
    pub(crate) viewport_height: f32,
    /// Non-zero when the foreground pass ran this frame. Pixels covered by
    /// foreground geometry pass through unblurred.
    pub(crate) foreground_enabled: f32,
}

/// Shadow atlas uniform (416 bytes, bound at group 0 binding 5).
///
/// Contains per-cascade view-projection matrices, split distances, atlas layout,
/// and PCSS parameters. Used by the fragment shader for CSM cascade selection
/// and shadow sampling.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ShadowAtlasUniform {
    /// 4 cascade view-projection matrices (each 64 bytes = 4x4 f32). Flattened.
    pub(crate) cascade_view_proj: [[f32; 4]; 16], // 256 bytes
    /// Distance-based split values for cascade selection (eye-to-fragment distance).
    pub(crate) cascade_splits: [f32; 4], //  16 bytes
    /// Number of active cascades (1-4).
    pub(crate) cascade_count: u32, //   4 bytes
    /// Backing atlas texture size in pixels (always `SHADOW_ATLAS_SIZE`).
    ///
    /// Shaders derive everything from this plus the atlas rects: one texel in
    /// UV space is `1.0 / atlas_size`, and a tile's pixel width is
    /// `atlas_size * (rect.z - rect.x)`. When the requested shadow resolution
    /// is below `SHADOW_ATLAS_SIZE`, the rects shrink but this value does not.
    pub(crate) atlas_size: f32, //   4 bytes
    /// Shadow filter mode: 0=PCF, 1=PCSS.
    pub(crate) shadow_filter: u32, //   4 bytes
    /// PCSS light source radius in UV space.
    pub(crate) pcss_light_radius: f32, //   4 bytes
    /// Per-slot atlas UV rects: [uv_min.x, uv_min.y, uv_max.x, uv_max.y] x 8 slots.
    pub(crate) atlas_rects: [[f32; 4]; 8], // 128 bytes
}
// Total: 256 + 16 + 16 + 128 = 416 bytes

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
