// The cascade shadow atlas uniform the lit shaders bind as `shadow_atlas`
// (368 bytes). Layout matches the renderer's `ShadowAtlasUniform`.
//
// `atlas_size` is the side of the backing depth texture in texels. Cascade i
// renders into `atlas_rects[i]` (uv min xy, uv max zw) of that texture.
//
// `tan_sun` is the tangent of the sun's angular radius: the PCSS penumbra
// radius per world unit between blocker and receiver.
//
// `range` is (far, fade_fraction, reserved, r_max_world). `far` is where the
// cascades end along the camera's forward axis, 0 when there is no distance
// cut; shadows fade out over the last `fade_fraction * far` before it.
// `r_max_world` caps the PCSS search and filter radius in world units.
struct ShadowAtlas {
    cascade_vp:        array<mat4x4<f32>, 4>, // 256 bytes
    cascade_splits:    vec4<f32>,             //  16 bytes
    cascade_count:     u32,                   //   4 bytes
    atlas_size:        f32,                   //   4 bytes
    shadow_filter:     u32,                   //   4 bytes (tier, see csm.wgsl)
    tan_sun:           f32,                   //   4 bytes
    range:             vec4<f32>,             //  16 bytes
    atlas_rects:       array<vec4<f32>, 4>,   //  64 bytes
};
