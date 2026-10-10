
// @viewport-wgsl-version: 1
// Scene depth reconstruction for the read-only-depth pass. Requires
// SHARED_BINDINGS_WGSL (uses `camera`). Binding-agnostic: the plugin samples
// its own depth texture (declared in a group it owns) and passes the value in.

// Positive linear view-space depth (distance in front of the camera) of a
// world-space point.
fn viewport_view_z(world_pos: vec3<f32>) -> f32 {
    return -(camera.view * vec4<f32>(world_pos, 1.0)).z;
}

// Positive linear view-space depth of the opaque scene surface under
// `screen_uv`, given its raw non-linear depth `scene_ndc_z` (the value from
// textureSample on the scene depth texture).
fn viewport_scene_view_z_from_ndc(screen_uv: vec2<f32>, scene_ndc_z: f32) -> f32 {
    let ndc = vec4<f32>(
        screen_uv.x * 2.0 - 1.0,
        1.0 - screen_uv.y * 2.0,
        scene_ndc_z,
        1.0,
    );
    let world_h = camera.inv_view_proj * ndc;
    let scene_world = world_h.xyz / world_h.w;
    return -(camera.view * vec4<f32>(scene_world, 1.0)).z;
}

// Soft-particle style fade: 1.0 in open space, ramping to 0.0 as `world_pos`
// meets the sampled scene surface over `soft_dist` world units. `scene_ndc_z`
// is the raw depth sampled under `screen_uv`. Matches the built-in Soft sprite
// sub-mode.
fn viewport_soft_fade_from_ndc(
    world_pos: vec3<f32>,
    screen_uv: vec2<f32>,
    scene_ndc_z: f32,
    soft_dist: f32,
) -> f32 {
    if soft_dist <= 0.0 {
        return 1.0;
    }
    let scene_view_z = viewport_scene_view_z_from_ndc(screen_uv, scene_ndc_z);
    return smoothstep(0.0, soft_dist, scene_view_z - viewport_view_z(world_pos));
}
