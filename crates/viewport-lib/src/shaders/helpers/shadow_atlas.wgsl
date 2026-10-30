// The cascade shadow atlas uniform the lit shaders bind as `shadow_atlas`
// (368 bytes). Layout matches the renderer's `ShadowAtlasUniform`.
//
// `atlas_size` is the side of the backing depth texture in texels. Cascade i
// renders into `atlas_rects[i]` (uv min xy, uv max zw) of that texture.
//
// `range` is (far, fade_fraction, tan_sun, r_max_world). `far` is where the
// cascades end along the camera's forward axis, 0 when there is no distance
// cut; shadows fade out over the last `fade_fraction * far` before it.
// `tan_sun` and `r_max_world` are reserved for physically sized soft
// shadows and are 0.
struct ShadowAtlas {
    cascade_vp:        array<mat4x4<f32>, 4>, // 256 bytes
    cascade_splits:    vec4<f32>,             //  16 bytes
    cascade_count:     u32,                   //   4 bytes
    atlas_size:        f32,                   //   4 bytes
    shadow_filter:     u32,                   //   4 bytes (tier, see csm.wgsl)
    pcss_light_radius: f32,                   //   4 bytes
    range:             vec4<f32>,             //  16 bytes
    atlas_rects:       array<vec4<f32>, 4>,   //  64 bytes
};
