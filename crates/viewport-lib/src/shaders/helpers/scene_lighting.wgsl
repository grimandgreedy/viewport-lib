// Shared scene-lighting evaluation for every lit pipeline.
//
// Defines the canonical `SingleLight` struct, the `Lights` header struct
// (binding 3 of group 0), the dynamically-sized storage buffer of lights
// (binding 13 of group 0), the cluster grid bindings (14, 15, 16), the
// per-light helper `eval_light(light, world_pos) -> LightEval` (direction,
// unshadowed radiance, range flag) used by every lit mesh loop, and the
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

// 1/pi. The Lambert diffuse normalisation, shared by every lit path so Phong
// and PBR reflect the same brightness for a given light.
const INV_PI: f32 = 0.31830989;

struct SingleLight {
    light_view_proj:   mat4x4<f32>,
    pos_or_dir:        vec3<f32>,
    light_type:        u32,
    colour:            vec3<f32>,
    intensity:         f32,
    range:             f32,
    inner_angle:       f32,
    outer_angle:       f32,
    spot_direction:    vec3<f32>,
    point_shadow_slot: i32,
    point_shadow_near: f32,
    radius:            f32,
    _pad1:             f32,
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
    env_zone_count:       u32,
    _pad_dbg_b:           u32,
    _pad_dbg_c:           u32,
};

// One environment-selection zone: a world-space box (center + half_extents) that
// selects array layer `layer`, with `fade` the outer falloff band. `parallax`
// is 1 for a local reflection probe (box-project the reflection against this box)
// and 0 for a distant environment. See env_zone_weight / ibl_ambient_zoned in
// ambient.wgsl.
//
// 48 bytes, matching `EnvZoneGpu` in environment.rs. The trailing pad is three
// SCALAR u32s, not a vec3<u32>: a vec3 has 16-byte alignment, which would round
// the struct up to 64 and break the array stride against the 48-byte Rust upload.
// The pad reserves room for a future separate parallax proxy box.
struct EnvZone {
    center:       vec3<f32>,   // offset 0
    layer:        u32,         // offset 12
    half_extents: vec3<f32>,   // offset 16
    fade:         f32,         // offset 28
    parallax:     u32,         // offset 32
    _pad_a:       u32,         // offset 36
    _pad_b:       u32,         // offset 40
    _pad_c:       u32,         // offset 44 (struct size 48)
};

struct ClusterGrid {
    dimensions: vec4<u32>,
    depth:      vec4<f32>,
    screen:     vec4<f32>,
    proj_scale: vec4<f32>,
    view:       mat4x4<f32>,
};

struct ClusterCell {
    offset:          u32,
    count:           u32,
    punctual_count:  u32,
    punctual_demand: u32,
};

@group(0) @binding(13) var<storage, read> lights_storage:        array<SingleLight>;
@group(0) @binding(14) var<uniform>       cluster_grid_uniform:  ClusterGrid;
@group(0) @binding(15) var<storage, read> cluster_cells:         array<ClusterCell>;
@group(0) @binding(16) var<storage, read> cluster_light_indices: array<u32>;
@group(0) @binding(17) var                point_shadow_cube_tex: texture_depth_cube_array;
// Indirect-lighting data (binding 18). Two fixed-size regions share one storage
// buffer to keep the fragment stage within the per-stage storage-buffer budget:
//   [0, ENV_ZONE_BASE)   : per-object light-probe SH, 9 vec4 per light-probe-lit
//                          object (rgb in xyz), selected by object.light_probe_index
//                          (see evaluate_sh_probe).
//   [ENV_ZONE_BASE, end) : environment-selection zones, 3 vec4 per zone, read via
//                          load_env_zone (see ibl_ambient_zoned).
@group(0) @binding(18) var<storage, read> indirect_light_data:   array<vec4<f32>>;
// Start of the env-zone region, in vec4 elements. Must equal
// resources::light_probes::MAX_LIGHT_PROBE_OBJECTS * 9 on the CPU side (a
// const_assert in resources::material::environment guards this).
const ENV_ZONE_BASE: u32 = 36864u;
// Adaptive probe volume: a 3-vec4 header (grid box + dims) then 9 vec4 of SH per
// grid cell, trilinearly sampled by sample_probe_volume in ambient.wgsl. A
// disabled header (element 0 w == 0) is bound when no volume is uploaded.
@group(0) @binding(20) var<storage, read> probe_volume:          array<vec4<f32>>;

// Reconstruct environment zone `i` from the packed vec4 region of
// indirect_light_data. Mirrors EnvZoneGpu (resources::material::environment):
// 3 vec4 per zone, with layer/parallax bit-cast from the vec4 lanes.
fn load_env_zone(i: u32) -> EnvZone {
    let base = ENV_ZONE_BASE + i * 3u;
    let a = indirect_light_data[base + 0u];
    let b = indirect_light_data[base + 1u];
    let c = indirect_light_data[base + 2u];
    return EnvZone(a.xyz, bitcast<u32>(a.w), b.xyz, b.w, bitcast<u32>(c.x), 0u, 0u, 0u);
}


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

// Per-light geometry and unshadowed radiance for one fragment. `radiance`
// folds in colour * intensity * distance/spot attenuation; it carries no
// shadow factor and no N.L term. `in_range` is false when a punctual light is
// at or beyond its range (radiance would be zero); callers skip such lights
// outright and so avoid shadow taps and BRDF work. This is the one canonical
// copy of the per-light attenuation math for every lit mesh loop.
struct LightEval {
    l:        vec3<f32>,
    radiance: vec3<f32>,
    in_range: bool,
};

fn eval_light(light: SingleLight, world_pos: vec3<f32>) -> LightEval {
    var ev: LightEval;
    if light.light_type == 0u {
        ev.l = normalize(light.pos_or_dir);
        ev.radiance = light.colour * light.intensity;
        ev.in_range = true;
        return ev;
    }
    let to_light = light.pos_or_dir - world_pos;
    let dist = length(to_light);
    ev.l = to_light / max(dist, 0.0001);
    ev.in_range = dist < light.range;
    if !ev.in_range {
        ev.radiance = vec3<f32>(0.0);
        return ev;
    }
    // Physical inverse-square falloff, clamped by the source radius so it does
    // not blow up near the light (radius 0 clamps to a tiny epsilon, matching
    // the path tracer's near-clamp). A quartic window fades it smoothly to zero
    // at `range`; `range` bounds reach only, it does not scale brightness. This
    // is the same formula the path tracer uses in raytrace.wgsl::direct_light.
    let r2 = max(light.radius * light.radius, 1.0e-4);
    let inv_sq = 1.0 / max(dist * dist, r2);
    let win = clamp(1.0 - pow(dist / light.range, 4.0), 0.0, 1.0);
    let atten = inv_sq * win * win;
    if light.light_type == 1u {
        ev.radiance = light.colour * light.intensity * atten;
    } else {
        let spot_dir = normalize(light.spot_direction);
        let cos_angle = dot(-ev.l, spot_dir);
        let cos_outer = cos(light.outer_angle);
        let cos_inner = cos(light.inner_angle);
        let cone_att = clamp(
            (cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
            0.0, 1.0,
        );
        ev.radiance = light.colour * light.intensity * atten * cone_att;
    }
    return ev;
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
        let ev = eval_light(lights_storage[idx], world_pos);
        if !ev.in_range { continue; }
        let raw = dot(normal, ev.l);
        let n_dot_l = select(max(raw, 0.0), abs(raw), two_sided);
        // Lambert diffuse carries the 1/pi factor so this simple lit path
        // matches the PBR and Phong diffuse energy for the same light.
        direct = direct + ev.radiance * n_dot_l * INV_PI;
    }
    return base_colour * (ambient + direct);
}
