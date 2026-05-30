// Shared scene-lighting evaluation for non-mesh pipelines.
//
// Defines the canonical `SingleLight` and `Lights` struct layouts (matching
// `src/resources/types.rs::LightsUniform`) and one helper:
//
//   apply_scene_lighting(N, base_colour, two_sided, world_pos, lights) -> vec3<f32>
//
// The helper returns `base_colour * (hemisphere_ambient + light_sum)` where
// hemisphere_ambient is mix(ground_colour, sky_colour, N.z*0.5+0.5) * intensity,
// and light_sum accumulates all active light types:
//   - Directional: Lambert weight on N·L (or abs(N·L) when two_sided)
//   - Point: distance falloff, no angular term
//   - Spot: distance falloff * inner/outer cone interpolation
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
    world_pos: vec3<f32>,
    lights: Lights,
) -> vec3<f32> {
    // Hemisphere ambient. Z-up world: N.z=+1 reads pure sky, N.z=-1 reads pure
    // ground. Scaled by `hemisphere_intensity` so `hemisphere_intensity = 0`
    // disables the term and items rely entirely on direct lights.
    let up_weight = clamp(normal.z * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(lights.ground_colour, lights.sky_colour, up_weight)
                  * lights.hemisphere_intensity;

    var direct = vec3<f32>(0.0);
    let n_lights = min(lights.count, 8u);
    for (var i: u32 = 0u; i < n_lights; i = i + 1u) {
        let l = lights.lights[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;

        if l.light_type == 0u {
            // Directional: pos_or_dir is the surface-to-light direction.
            L = normalize(l.pos_or_dir);
            radiance = l.colour * l.intensity;
        } else if l.light_type == 1u {
            // Point: inverse-square-ish falloff clamped at range.
            let to_light = l.pos_or_dir - world_pos;
            let dist = length(to_light);
            L = to_light / max(dist, 0.0001);
            let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
            radiance = l.colour * l.intensity * falloff * falloff;
        } else {
            // Spot: distance falloff * inner/outer cone attenuation.
            let to_light = l.pos_or_dir - world_pos;
            let dist = length(to_light);
            L = to_light / max(dist, 0.0001);
            let dist_falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
            let spot_dir = normalize(l.spot_direction);
            let cos_angle = dot(-L, spot_dir);
            let cos_outer = cos(l.outer_angle);
            let cos_inner = cos(l.inner_angle);
            let cone_att = clamp(
                (cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
                0.0, 1.0,
            );
            radiance = l.colour * l.intensity * dist_falloff * dist_falloff * cone_att;
        }

        let raw = dot(normal, L);
        let n_dot_l = select(max(raw, 0.0), abs(raw), two_sided);
        direct = direct + radiance * n_dot_l;
    }

    return base_colour * (ambient + direct);
}
