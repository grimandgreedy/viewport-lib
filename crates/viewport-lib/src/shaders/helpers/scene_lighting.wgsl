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
// and `scene_lighting_terms`, which returns its two terms apart.
//
// The helper returns `base_colour * (hemisphere_ambient + light_sum)` where
// `hemisphere_ambient` is the sky/ground blend on `N.z` and `light_sum`
// accumulates direct contribution from the per-cluster light list. When the
// active light count is small enough that cluster overhead would dominate,
// the host sets `cluster_grid_uniform.screen.z != 0` and the helper iterates
// the full `lights_storage` array directly. A cell whose list overflowed its
// fixed slice iterates the full array too, so no light is ever dropped.
//
// The world-to-view matrix is carried inside the cluster grid uniform so the
// helper can derive the view-space fragment position itself. The radiance
// `eval_light` and `apply_scene_lighting` return is exposed: both multiply by
// `camera.exposure`, the lane after `forward` in the group-0 camera uniform.
//
// The BEGIN_/END_ regions carry an alternative on `// ALT(NAME):` lines, which
// `builders::adapt_wgsl` swaps in where the device's group 0 needs it: the
// light list as a uniform array of COMPATIBILITY_MAX_LIGHTS (UNIFORM), and the
// cluster lookups walking every light where there are no clusters
// (UNCLUSTERED).
//
// Consumers must:
//   - Bind the `Lights` header uniform at `@group(0) @binding(3)`.
//   - Bind the per-light storage buffer at `@group(0) @binding(13)`.
//   - Bind the cluster grid uniform / cells / index list at bindings 14, 15,
//     and 16 (all on group 0).
//   - Declare `camera` at `@group(0) @binding(0)` with the uniform's layout
//     through `exposure` (view_proj, eye_pos, a f32, forward, exposure).
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
    // 0 none, 1 cascade atlas, 2 point light layers, 3 spot light layer.
    shadow_kind:       u32,
    spot_direction:    vec3<f32>,
    // Base layer in the punctual shadow array, -1 none.
    shadow_layer:      i32,
    point_shadow_near: f32,
    radius:            f32,
    channel_mask:      u32,
    _pad1:             u32,
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
    ibl_diffuse_scale:    f32,
    debug_vis_split_x:    f32,
    env_zone_count:       u32,
    ibl_specular_scale:   f32,
    // Start of the custom-data region of indirect_light_data, in vec4 elements.
    custom_data_base:     u32,
    // Material AO options: AO_FLAG_MULTI_BOUNCE, AO_FLAG_MICRO_SHADOW.
    ao_flags:             u32,
    _pad_ao_a:            u32,
    _pad_ao_b:            u32,
    _pad_ao_c:            u32,
};

// Lights::ao_flags bits. Match the constants in renderer/prepare/lighting.rs.
const AO_FLAG_MULTI_BOUNCE: u32 = 1u;
const AO_FLAG_MICRO_SHADOW: u32 = 2u;

// One environment-selection zone: a world-space box (centre + half_extents) that
// selects array layer `layer`, with `fade` the outer falloff band. `parallax`
// is 1 for a local reflection probe (box-project the reflection against this box)
// and 0 for a distant environment. `rotation` (radians about +Z) and `intensity`
// (the resolved multiplier) are the zone's own, not the lighting environment's.
// See env_zone_weight / ibl_ambient_zoned in ambient.wgsl.
//
// 48 bytes, matching `EnvZoneGpu` in environment.rs. The fields after
// `parallax` are SCALARS, not a vec3: a vec3 has 16-byte alignment, which would
// round the struct up to 64 and break the array stride against the 48-byte
// Rust upload.
struct EnvZone {
    centre:       vec3<f32>,   // offset 0
    layer:        u32,         // offset 12
    half_extents: vec3<f32>,   // offset 16
    fade:         f32,         // offset 28
    parallax:     u32,         // offset 32
    rotation:     f32,         // offset 36
    intensity:    f32,         // offset 40
    _pad_c:       u32,         // offset 44 (struct size 48)
};

struct ClusterGrid {
    dimensions: vec4<u32>,
    depth:      vec4<f32>,
    screen:     vec4<f32>,
    proj_scale: vec4<f32>,
    view:       mat4x4<f32>,
};

// BEGIN_LIGHT_LIST
@group(0) @binding(13) var<storage, read> lights_storage:        array<SingleLight>;
// ALT(UNIFORM): @group(0) @binding(13) var<uniform> lights_storage: array<SingleLight, 16>;
// END_LIGHT_LIST
@group(0) @binding(14) var<uniform>       cluster_grid_uniform:  ClusterGrid;
// The cluster cells, four words each (offset, count, punctual_count,
// punctual_demand), then each cell's light bitmask from CLUSTER_INDEX_BASE.
// One buffer so the pair costs one fragment storage binding.
@group(0) @binding(15) var<storage, read> cluster_data:          array<u32>;
// Start of the light index list in cluster_data, in u32 words. Must equal
// resources::gpu::clustered::CLUSTER_INDEX_BASE_BYTES / 4 on the CPU side.
const CLUSTER_INDEX_BASE: u32 = 13824u;
// Point and spot light shadows: a point light's six cube faces from its
// shadow_layer, a spot light's map at its shadow_layer. See punctual_shadow.wgsl.
@group(0) @binding(17) var                punctual_shadow_tex:   texture_depth_2d_array;
// Indirect-lighting data (binding 18). Three regions share one storage buffer
// to keep the fragment stage within the per-stage storage-buffer budget:
//   [0, ENV_ZONE_BASE)   : per-object light-probe SH, 9 vec4 per light-probe-lit
//                          object (rgb in xyz), selected by object.light_probe_index
//                          (see evaluate_sh_probe).
//   [ENV_ZONE_BASE, PROBE_VOLUME_BASE) : environment-selection zones, 3 vec4
//                          per zone, read via load_env_zone (see
//                          ibl_ambient_zoned).
//   [PROBE_VOLUME_BASE, end) : the adaptive probe volume (below).
@group(0) @binding(18) var<storage, read> indirect_light_data:   array<vec4<f32>>;
// Start of the env-zone region, in vec4 elements. Must equal
// resources::light_probes::MAX_LIGHT_PROBE_OBJECTS * 9 on the CPU side (a
// const_assert in resources::material::environment guards this).
const ENV_ZONE_BASE: u32 = 36864u;
// Start of the adaptive probe volume in indirect_light_data, in vec4 elements:
// a 3-vec4 header (grid box + dims) then 9 vec4 of SH per grid cell,
// trilinearly sampled by sample_probe_volume in ambient.wgsl. A disabled header
// (element 0 w == 0) sits there when no volume is uploaded. Must equal
// resources::lighting::PROBE_VOLUME_REGION_OFFSET_BYTES / 16 on the CPU side.
const PROBE_VOLUME_BASE: u32 = 37056u;

// Per-instance custom-data block `id`, the last region of indirect_light_data
// (two vec4 each) starting at `base` (the lights uniform's custom_data_base):
// data0 adds to emissive (nits, .w pad), data1 is the raw material-plugin
// channel surfaced to a hook as `surf.attr`. Block 0 is zero.
fn custom_data0(base: u32, id: u32) -> vec4<f32> {
    return indirect_light_data[base + 2u * id];
}

fn custom_data1(base: u32, id: u32) -> vec4<f32> {
    return indirect_light_data[base + 2u * id + 1u];
}

// Reconstruct environment zone `i` from the packed vec4 region of
// indirect_light_data. Mirrors EnvZoneGpu (resources::material::environment):
// 3 vec4 per zone, with layer/parallax bit-cast from the vec4 lanes and the
// zone's rotation and intensity in c.y, c.z.
fn load_env_zone(i: u32) -> EnvZone {
    let base = ENV_ZONE_BASE + i * 3u;
    let a = indirect_light_data[base + 0u];
    let b = indirect_light_data[base + 1u];
    let c = indirect_light_data[base + 2u];
    return EnvZone(a.xyz, bitcast<u32>(a.w), b.xyz, b.w, bitcast<u32>(c.x), c.y, c.z, 0u);
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

// The lights that reach a fragment, walked one at a time:
//
//     var lights = cluster_lights(world_pos, lights_uniform.count);
//     loop {
//         let i = cluster_next_light(&lights);
//         if i == CLUSTER_LIGHTS_END { break; }
//         // lights_storage[i] ...
//     }
//
// A cell holds a bitmask over the light array (CLUSTER_MASK_WORDS words), so
// a crowded cell walks only the lights that reach it, in light-index order,
// and none is dropped. When the host short-circuited the cluster build
// (`cluster_grid_uniform.screen.z != 0`) the cursor walks every light.
const CLUSTER_LIGHTS_END: u32 = 0xffffffffu;
const CLUSTER_MASK_WORDS: u32 = 16u;

struct ClusterLights {
    fallback: bool,
    // The cell's first mask word in the list at CLUSTER_INDEX_BASE.
    base:     u32,
    // Mask words that can hold a set bit: those covering `count` lights.
    words:    u32,
    word:     u32,
    bits:     u32,
    next:     u32,
    count:    u32,
};

fn cluster_lights(world_pos: vec3<f32>, total_count: u32) -> ClusterLights {
    var c: ClusterLights;
    c.fallback = true;
    c.base = 0u;
    c.words = min(CLUSTER_MASK_WORDS, (total_count + 31u) / 32u);
    c.word = 0u;
    c.bits = 0u;
    c.next = 0u;
    c.count = total_count;
    // BEGIN_CLUSTER_LOOKUP
    c.fallback = cluster_grid_uniform.screen.z != 0.0;
    if !c.fallback && c.words > 0u {
        let view_pos = (cluster_grid_uniform.view * vec4<f32>(world_pos, 1.0)).xyz;
        let cid = cluster_index_for(view_pos);
        c.base = cluster_data[cid * 4u];
        c.bits = cluster_data[CLUSTER_INDEX_BASE + c.base];
    }
    // ALT(UNCLUSTERED): c.fallback = true;
    // END_CLUSTER_LOOKUP
    return c;
}

// The next light index the cursor reaches, or CLUSTER_LIGHTS_END.
fn cluster_next_light(c: ptr<function, ClusterLights>) -> u32 {
    if (*c).fallback {
        if (*c).next >= (*c).count {
            return CLUSTER_LIGHTS_END;
        }
        let i = (*c).next;
        (*c).next = i + 1u;
        return i;
    }
    loop {
        if (*c).bits != 0u {
            let b = firstTrailingBit((*c).bits);
            (*c).bits = (*c).bits & ((*c).bits - 1u);
            return (*c).word * 32u + b;
        }
        (*c).word = (*c).word + 1u;
        if (*c).word >= (*c).words {
            return CLUSTER_LIGHTS_END;
        }
        // BEGIN_CLUSTER_INDEX
        (*c).bits = cluster_data[CLUSTER_INDEX_BASE + (*c).base + (*c).word];
        // ALT(UNCLUSTERED): (*c).bits = 0u;
        // END_CLUSTER_INDEX
    }
    return CLUSTER_LIGHTS_END;
}

// Per-light geometry and unshadowed radiance for one fragment. `radiance`
// folds in colour * intensity * exposure * distance/spot attenuation; it
// carries no shadow factor and no N.L term. `in_range` is false when a punctual light is
// at or beyond its range (radiance would be zero); callers skip such lights
// outright and so avoid shadow taps and BRDF work. This is the one canonical
// copy of the per-light attenuation math for every lit mesh loop.
struct LightEval {
    l:        vec3<f32>,
    radiance: vec3<f32>,
    in_range: bool,
};

// Layer test for light channels: a light contributes to an object only when
// their masks intersect. `object_mask == !0` (the default) passes every light,
// so the channel feature is inert unless a consumer narrows either mask. Called
// at each lit mesh loop right after fetching the light.
fn light_in_channel(light: SingleLight, object_mask: u32) -> bool {
    return (light.channel_mask & object_mask) != 0u;
}

fn eval_light(light: SingleLight, world_pos: vec3<f32>) -> LightEval {
    var ev: LightEval;
    // The light's exposed intensity: exposure applies where radiance is read.
    let intensity = light.intensity * camera.exposure;
    if light.light_type == 0u {
        ev.l = normalize(light.pos_or_dir);
        ev.radiance = light.colour * intensity;
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
        ev.radiance = light.colour * intensity * atten;
    } else {
        let spot_dir = normalize(light.spot_direction);
        let cos_angle = dot(-ev.l, spot_dir);
        let cos_outer = cos(light.outer_angle);
        let cos_inner = cos(light.inner_angle);
        let cone_att = clamp(
            (cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
            0.0, 1.0,
        );
        ev.radiance = light.colour * intensity * atten * cone_att;
    }
    return ev;
}

// The two terms `apply_scene_lighting` sums, before the base colour: the
// hemisphere fill (indirect) and the light loop's sum (direct), both exposed.
// The scene pass's light split is `scene_light_split(base * indirect,
// base * direct, emitted)`.
struct SceneLightingTerms {
    indirect: vec3<f32>,
    direct:   vec3<f32>,
};

fn scene_lighting_terms(
    normal: vec3<f32>,
    two_sided: bool,
    world_pos: vec3<f32>,
    lights: Lights,
) -> SceneLightingTerms {
    // Hemisphere ambient. Z-up world: N.z=+1 reads pure sky, N.z=-1 reads pure
    // ground. Scaled by `hemisphere_intensity` so `hemisphere_intensity = 0`
    // disables the term and items rely entirely on direct lights.
    let up_weight = clamp(normal.z * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(lights.ground_colour, lights.sky_colour, up_weight)
                  * (lights.hemisphere_intensity * camera.exposure);

    var direct = vec3<f32>(0.0);
    var range = cluster_lights(world_pos, lights.count);
    loop {
        let idx = cluster_next_light(&range);
        if idx == CLUSTER_LIGHTS_END { break; }
        let ev = eval_light(lights_storage[idx], world_pos);
        if !ev.in_range { continue; }
        let raw = dot(normal, ev.l);
        let n_dot_l = select(max(raw, 0.0), abs(raw), two_sided);
        // Lambert diffuse carries the 1/pi factor so this simple lit path
        // matches the PBR and Phong diffuse energy for the same light.
        direct = direct + ev.radiance * n_dot_l * INV_PI;
    }
    return SceneLightingTerms(ambient, direct);
}

fn apply_scene_lighting(
    normal: vec3<f32>,
    base_colour: vec3<f32>,
    two_sided: bool,
    world_pos: vec3<f32>,
    lights: Lights,
) -> vec3<f32> {
    let terms = scene_lighting_terms(normal, two_sided, world_pos, lights);
    return base_colour * (terms.indirect + terms.direct);
}
