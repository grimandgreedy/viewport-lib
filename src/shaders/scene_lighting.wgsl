// Shared scene-lighting evaluation for non-mesh pipelines.
//
// Defines the canonical `SingleLight` and `Lights` struct layouts (matching
// `src/resources/types.rs::LightsUniform`) and one helper:
//
//   apply_scene_lighting(N, base_colour, two_sided, lights) -> vec3<f32>
//
// The helper returns `base_colour * (hemisphere_ambient + directional_sum)`
// where:
//   - hemisphere_ambient = mix(ground_colour, sky_colour, (N.z * 0.5 + 0.5))
//                          * hemisphere_intensity
//   - directional_sum    = sum over `lights[0..min(count, 8)]` of light_type==0
//                          entries of `colour * intensity * weight(N, L)`
// Where `weight` is `max(dot(N, L), 0.0)` when `two_sided` is `false` (one-sided
// surfaces, e.g. streamtube, ribbon, implicit) or `abs(dot(N, L))` when `true`
// (instanced geometry viewed from any side, e.g. glyph).
//
// Point and spot lights are currently skipped; they will be added in the same
// helper once the renderer's clustered lighting pass (scene-lights-plan L3) is
// available. See `docs/plans/lighting-shading-consistency-plan.md` C10.
//
// Consumers must:
//   - Bind a `Lights` uniform at `@group(0) @binding(3)` (every existing scivis
//     pipeline already does).
//   - Remove their local `SingleLight` and `Lights` struct definitions and
//     `// #include "scene_lighting.wgsl"` near the top of the file instead.

struct SingleLight {
    light_view_proj: mat4x4<f32>,
    pos_or_dir:      vec3<f32>,
    light_type:      u32,
    colour:           vec3<f32>,
    intensity:       f32,
    range:           f32,
    inner_angle:     f32,
    outer_angle:     f32,
    spot_direction:  vec3<f32>,
    _pad:            vec2<f32>,
};

struct Lights {
    count:                u32,
    shadow_bias:          f32,
    shadows_enabled:      u32,
    debug_vis_mode:       u32,
    sky_colour:            vec3<f32>,
    hemisphere_intensity: f32,
    ground_colour:         vec3<f32>,
    debug_vis_scale:      f32,
    lights:               array<SingleLight, 8>,
    ibl_enabled:          u32,
    ibl_intensity:        f32,
    ibl_rotation:         f32,
    show_skybox:          u32,
    debug_vis_split_x:    f32,
    _pad_dbg_a:           u32,
    _pad_dbg_b:           u32,
    _pad_dbg_c:           u32,
};

fn apply_scene_lighting(
    normal: vec3<f32>,
    base_colour: vec3<f32>,
    two_sided: bool,
    lights: Lights,
) -> vec3<f32> {
    // Hemisphere ambient. Z-up world: N.z=+1 reads pure sky, N.z=-1 reads pure
    // ground. Scaled by `hemisphere_intensity` so `hemisphere_intensity = 0`
    // disables the term and items rely entirely on directional lights.
    let up_weight = clamp(normal.z * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(lights.ground_colour, lights.sky_colour, up_weight)
                  * lights.hemisphere_intensity;

    // Directional sum.
    var direct = vec3<f32>(0.0);
    let n_lights = min(lights.count, 8u);
    for (var i: u32 = 0u; i < n_lights; i = i + 1u) {
        let l = lights.lights[i];
        if l.light_type != 0u {
            continue; // point / spot deferred to a future phase
        }
        // `pos_or_dir` is the surface-to-light direction (matches mesh.wgsl).
        let L = normalize(l.pos_or_dir);
        let raw = dot(normal, L);
        let n_dot_l = select(max(raw, 0.0), abs(raw), two_sided);
        direct = direct + l.colour * l.intensity * n_dot_l;
    }

    return base_colour * (ambient + direct);
}
