// The cascade shadow atlas uniform the lit shaders bind as `shadow_atlas`
// (416 bytes). Layout matches the renderer's `ShadowAtlasUniform`.
//
// `atlas_size` is the side of the backing depth texture in texels. Cascade i
// renders into `atlas_rects[i]` (uv min xy, uv max zw) of that texture.
struct ShadowAtlas {
    cascade_vp:        array<mat4x4<f32>, 4>, // 256 bytes
    cascade_splits:    vec4<f32>,             //  16 bytes
    cascade_count:     u32,                   //   4 bytes
    atlas_size:        f32,                   //   4 bytes
    shadow_filter:     u32,                   //   4 bytes (tier, see csm.wgsl)
    pcss_light_radius: f32,                   //   4 bytes
    atlas_rects:       array<vec4<f32>, 8>,   // 128 bytes
};
