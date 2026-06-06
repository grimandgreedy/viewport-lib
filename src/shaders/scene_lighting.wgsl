// Shared scene-lighting evaluation for every lit pipeline.
//
// Defines the canonical `SingleLight` struct, the `Lights` header struct
// (binding 3 of group 0), the dynamically-sized storage buffer of lights
// (binding 13 of group 0), the cluster grid bindings (14, 15, 16), and the
// per-fragment helper:
//
//   apply_scene_lighting(N, base_colour, two_sided, world_pos, lights) -> vec3<f32>
//
// The helper returns `base_colour * (hemisphere_ambient + light_sum)` where
// `hemisphere_ambient` is the sky/ground blend on `N.z` and `light_sum`
// accumulates direct contribution from the per-cluster light list. When the
// active light count is small enough that cluster overhead would dominate,
// the host sets `cluster_grid_uniform.screen.z != 0` and the helper iterates
// the full `lights_storage` array directly.
//
// The world-to-view matrix is carried inside the cluster grid uniform so the
// helper can derive the view-space fragment position itself; consumers don't
// have to extend their per-shader `Camera` struct to expose it.
//
// Consumers must:
//   - Bind the `Lights` header uniform at `@group(0) @binding(3)`.
//   - Bind the per-light storage buffer at `@group(0) @binding(13)`.
//   - Bind the cluster grid uniform / cells / index list at bindings 14, 15,
//     and 16 (all on group 0).
//   - Remove any local `SingleLight` and `Lights` struct definitions and
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
    ibl_enabled:          u32,
    ibl_intensity:        f32,
    ibl_rotation:         f32,
    show_skybox:          u32,
    debug_vis_split_x:    f32,
    _pad_dbg_a:           u32,
    _pad_dbg_b:           u32,
    _pad_dbg_c:           u32,
};

struct ClusterGrid {
    dimensions: vec4<u32>,
    depth:      vec4<f32>,
    screen:     vec4<f32>,
    proj_scale: vec4<f32>,
    view:       mat4x4<f32>,
};

struct ClusterCell {
    offset: u32,
    count:  u32,
};

@group(0) @binding(13) var<storage, read> lights_storage:        array<SingleLight>;
@group(0) @binding(14) var<uniform>       cluster_grid_uniform:  ClusterGrid;
@group(0) @binding(15) var<storage, read> cluster_cells:         array<ClusterCell>;
@group(0) @binding(16) var<storage, read> cluster_light_indices: array<u32>;

fn cluster_index_for(view_pos: vec3<f32>) -> u32 {
    let grid = cluster_grid_uniform;
    let near = grid.depth.x;
    let far  = grid.depth.y;
    let log_ratio = grid.depth.z;

    // View-space z is negative looking down -Z. Distance from camera.
    let z_abs = clamp(-view_pos.z, near, far);
    let nz = f32(grid.dimensions.z);
    let zi = u32(clamp(log(z_abs / near) / log_ratio * nz, 0.0, nz - 1.0));

    // Perspective-divide x, y into NDC space.
    let denom = max(z_abs, 1e-5);
    let x_ndc = view_pos.x / (denom * grid.proj_scale.x);
    let y_ndc = view_pos.y / (denom * grid.proj_scale.y);
    let nx = f32(grid.dimensions.x);
    let ny = f32(grid.dimensions.y);
    let xi = u32(clamp((x_ndc + 1.0) * 0.5 * nx, 0.0, nx - 1.0));
    let yi = u32(clamp((y_ndc + 1.0) * 0.5 * ny, 0.0, ny - 1.0));

    return zi * (grid.dimensions.x * grid.dimensions.y)
         + yi *  grid.dimensions.x
         + xi;
}

// Iteration bounds for the per-fragment light loop. `fallback` is true when
// the host short-circuited the cluster build : in that case `start` is 0 and
// `count` is the total active light count, and `cluster_light_global` returns
// the loop index directly.
struct LightRange {
    fallback: bool,
    start:    u32,
    count:    u32,
};

fn cluster_light_range(world_pos: vec3<f32>, total_count: u32) -> LightRange {
    var r: LightRange;
    r.fallback = cluster_grid_uniform.screen.z != 0.0;
    if r.fallback {
        r.start = 0u;
        r.count = total_count;
    } else {
        let view_pos = (cluster_grid_uniform.view * vec4<f32>(world_pos, 1.0)).xyz;
        let cid = cluster_index_for(view_pos);
        let cell = cluster_cells[cid];
        r.start = cell.offset;
        r.count = cell.count;
    }
    return r;
}

fn cluster_light_global(range: LightRange, j: u32) -> u32 {
    if range.fallback {
        return j;
    }
    return cluster_light_indices[range.start + j];
}

fn evaluate_light(l: SingleLight, world_pos: vec3<f32>) -> array<vec3<f32>, 2> {
    var L: vec3<f32>;
    var radiance: vec3<f32>;
    if l.light_type == 0u {
        L = normalize(l.pos_or_dir);
        radiance = l.colour * l.intensity;
    } else if l.light_type == 1u {
        let to_light = l.pos_or_dir - world_pos;
        let dist = length(to_light);
        L = to_light / max(dist, 0.0001);
        let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
        radiance = l.colour * l.intensity * falloff * falloff;
    } else {
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
    return array<vec3<f32>, 2>(L, radiance);
}

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
    let range = cluster_light_range(world_pos, lights.count);
    for (var j: u32 = 0u; j < range.count; j = j + 1u) {
        let idx = cluster_light_global(range, j);
        let r = evaluate_light(lights_storage[idx], world_pos);
        let raw = dot(normal, r[0]);
        let n_dot_l = select(max(raw, 0.0), abs(raw), two_sided);
        direct = direct + r[1] * n_dot_l;
    }
    return base_colour * (ambient + direct);
}
