//! WGSL helper catalog.
//!
//! Each helper is a `&'static str` of WGSL that a plugin shader prefixes (or
//! concatenates into its own shader source) to gain access to the lib's
//! shared bindings, shading helpers, and target conventions.
//!
//! Versioning: each helper carries a `// @viewport-wgsl-version: N`
//! comment. The version is bumped whenever a function signature, struct
//! field, or binding number changes. Plugins compare against
//! [`WGSL_VERSION`] at build time to detect breakage early. Function bodies
//! and private fields are not part of the contract and may change between
//! patch releases.
//!
//! Composition: plugin shaders typically build their source as:
//!
//! ```ignore
//! let src = format!(
//!     "{bindings}\n{pbr}\n{this_pipeline_specific_wgsl}",
//!     bindings = viewport_lib::plugin_api::shared_wgsl::SHARED_BINDINGS_WGSL,
//!     pbr      = viewport_lib::plugin_api::shared_wgsl::SHARED_PBR_WGSL,
//!     this_pipeline_specific_wgsl = include_str!("my_shader.wgsl"),
//! );
//! ```

/// Catalog version. Bumped on any breaking change to a helper signature,
/// struct field, or binding number. Plugins should assert against this at
/// build time:
///
/// ```ignore
/// const _: () = assert!(viewport_lib::plugin_api::shared_wgsl::WGSL_VERSION == 1);
/// ```
pub const WGSL_VERSION: u32 = 6;

/// Group-0 bind declarations and shared scene-data structs.
///
/// Declares every binding in the lib's camera/lights/shadows/clip/IBL group,
/// matching the layout exposed via
/// [`SharedBindings`](super::SharedBindings). Plugin shaders include this
/// once and must not re-declare these bindings.
///
/// Bindings exposed:
///
/// | Binding | Resource | WGSL identifier |
/// |---------|----------|----------------|
/// | 0  | `Camera` uniform | `camera` |
/// | 1  | shadow atlas texture | `shadow_atlas_tex` |
/// | 2  | shadow comparison sampler | `shadow_atlas_sampler` |
/// | 3  | `Lights` header uniform | `lights` |
/// | 4  | `ClipPlanes` uniform | `clip_planes` |
/// | 5  | *(internal: CSM uniform; route through `viewport_sample_csm`)* | *(opaque)* |
/// | 6  | `ClipVolumes` uniform | `clip_volumes` |
/// | 7  | IBL irradiance equirect | `ibl_irradiance_tex` |
/// | 8  | IBL specular equirect | `ibl_specular_tex` |
/// | 9  | BRDF integration LUT | `ibl_brdf_lut` |
/// | 10 | IBL sampler | `ibl_sampler` |
/// | 11 | Skybox equirect | `skybox_tex` |
/// | 12 | debug fragment storage buffer | `debug_frag` |
/// | 13 | per-light array | `lights_storage` |
/// | 17 | point-light shadow cubemap array | `point_shadow_cube` |
pub const SHARED_BINDINGS_WGSL: &str = r#"
// @viewport-wgsl-version: 1
// Shared group-0 declarations. Do not re-declare these bindings in plugin
// shaders.

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad0:         f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

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
    _pad0:             f32,
    _pad1:             f32,
};

struct Lights {
    count:                u32,
    shadow_bias:          f32,
    shadows_enabled:      u32,
    debug_vis_mode:       u32,
    sky_colour:           vec3<f32>,
    hemisphere_intensity: f32,
    ground_colour:        vec3<f32>,
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

struct ClipPlanes {
    planes:          array<vec4<f32>, 6>,
    count:           u32,
    _pad0:           u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct ClipVolumeEntry {
    volume_type:  u32,
    _pad_a:       u32,
    _pad_b:       u32,
    _pad_c:       u32,
    center:       vec3<f32>,
    radius:       f32,
    half_extents: vec3<f32>,
    _pad1:        f32,
    col0:         vec3<f32>,
    _pad2:        f32,
    col1:         vec3<f32>,
    _pad3:        f32,
    col2:         vec3<f32>,
    _pad4:        f32,
};

struct ClipVolumeUB {
    count:    u32,
    _pad_a:   u32,
    _pad_b:   u32,
    _pad_c:   u32,
    volumes:  array<ClipVolumeEntry, 4>,
};

@group(0) @binding(0)  var<uniform> camera:               Camera;
@group(0) @binding(1)  var          shadow_atlas_tex:     texture_depth_2d;
@group(0) @binding(2)  var          shadow_atlas_sampler: sampler_comparison;
@group(0) @binding(3)  var<uniform> lights:               Lights;
@group(0) @binding(4)  var<uniform> clip_planes:          ClipPlanes;
@group(0) @binding(6)  var<uniform> clip_volume:          ClipVolumeUB;
@group(0) @binding(7)  var          ibl_irradiance_tex:   texture_2d<f32>;
@group(0) @binding(8)  var          ibl_specular_tex:     texture_2d<f32>;
@group(0) @binding(9)  var          ibl_brdf_lut:         texture_2d<f32>;
@group(0) @binding(10) var          ibl_sampler:          sampler;
@group(0) @binding(11) var          skybox_tex:           texture_2d<f32>;
@group(0) @binding(13) var<storage, read> lights_storage: array<SingleLight>;
@group(0) @binding(17) var          point_shadow_cube:    texture_depth_cube_array;

// Section-view clip planes: returns false when `world_pos` is on the
// clipped side of any active plane. Plugin fragment shaders call this and
// `discard` when it returns false to match the lib's clipping behaviour.
fn viewport_pass_clip_planes(world_pos: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(world_pos, plane.xyz) + plane.w < 0.0 {
            return false;
        }
    }
    return true;
}

// Composable clip volumes (box / sphere / cylinder): returns true when
// `world_pos` is inside every active clip volume. Returns true when no
// volumes are active.
fn viewport_pass_clip_volumes(world_pos: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_volume.count; i = i + 1u) {
        let e = clip_volume.volumes[i];
        if e.volume_type == 2u {
            let d = world_pos - e.center;
            let local = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
            if abs(local.x) > e.half_extents.x
                || abs(local.y) > e.half_extents.y
                || abs(local.z) > e.half_extents.z {
                return false;
            }
        } else if e.volume_type == 3u {
            let ds = world_pos - e.center;
            if dot(ds, ds) > e.radius * e.radius { return false; }
        } else if e.volume_type == 4u {
            let axis = e.col0;
            let d = world_pos - e.center;
            let along = dot(d, axis);
            if abs(along) > e.half_extents.x { return false; }
            let radial = d - axis * along;
            if dot(radial, radial) > e.radius * e.radius { return false; }
        }
    }
    return true;
}

// Combined clip test. Returns true when the fragment should be kept,
// false when it should be discarded. Plugin fragment shaders typically:
//
//   if !viewport_clip_test(in.world_pos) { discard; }
fn viewport_clip_test(world_pos: vec3<f32>) -> bool {
    return viewport_pass_clip_planes(world_pos)
        && viewport_pass_clip_volumes(world_pos);
}

// The shadow-info uniform (binding 5) is declared inside SHARED_PBR_WGSL
// for the sole use of `viewport_sample_csm`. Its struct layout and field
// set are intentionally not part of the published contract and may change
// between catalog versions. Plugins must not redeclare binding 5 and must
// not read the uniform directly; route shadow queries through
// `viewport_sample_csm`.
"#;

/// Shared PBR shading helper.
///
/// Provides:
///
/// ```ignore
/// fn viewport_pbr_shade(inp: PbrInputs) -> vec3<f32>;
/// fn viewport_sample_csm(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32;
/// fn viewport_apply_scene_lighting(N, base_colour, two_sided, world_pos) -> vec3<f32>;
/// ```
///
/// `viewport_pbr_shade` returns the final lit colour for a fragment given a
/// `PbrInputs` populated with albedo / normal / metallic / roughness / AO /
/// emissive. It applies the lib's standard hemisphere ambient + light loop
/// and attenuates the primary light's contribution by the CSM shadow factor
/// when `lights.shadows_enabled != 0`. Plugins that compose this helper get
/// shadows automatically; do not multiply by `viewport_sample_csm` again.
/// Future revisions may add IBL and SSAO sampling inside this function;
/// consumers should rebuild their shaders when the catalog version bumps to
/// pick up the upgrade.
///
/// `viewport_sample_csm` returns a 0..1 shadow factor for `world_pos`.
/// Returns 1.0 (fully lit) when shadows are disabled or the position is
/// outside every cascade. The cascade scheme, filter kernel, and bias
/// strategy are internal details and may change between catalog versions;
/// the function signature and return-value semantics are the contract.
///
/// `viewport_apply_scene_lighting` is the simpler Lambert helper used by
/// non-PBR pipelines (glyphs, tubes, ribbons). Use it when a plugin wants
/// scene-light parity with those built-in items.
pub const SHARED_PBR_WGSL: &str = r#"
// @viewport-wgsl-version: 2
// Shared PBR / lit-shading helpers. Requires SHARED_BINDINGS_WGSL to be
// included first.
//
// Internal binding: the CSM uniform at @group(0) @binding(5) is declared
// here for `viewport_sample_csm`'s use. Its struct layout is an internal
// detail of this catalog; plugin shaders must not reference
// `_viewport_csm` directly or assume the field set is stable.

struct _ViewportCsm {
    cascade_vp:        array<mat4x4<f32>, 4>,
    cascade_splits:    vec4<f32>,
    cascade_count:     u32,
    atlas_size:        f32,
    shadow_filter:     u32,
    pcss_light_radius: f32,
    atlas_rects:       array<vec4<f32>, 8>,
};

@group(0) @binding(5) var<uniform> _viewport_csm: _ViewportCsm;

const _VIEWPORT_POISSON_DISK: array<vec2<f32>, 32> = array<vec2<f32>, 32>(
    vec2<f32>(-0.94201624, -0.39906216), vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870), vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432), vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845), vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554), vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023), vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507), vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367), vec2<f32>( 0.14383161, -0.14100790),
    vec2<f32>(-0.44451570,  0.67055830), vec2<f32>( 0.70509040, -0.15854630),
    vec2<f32>( 0.07130650, -0.64599580), vec2<f32>( 0.39881030,  0.55789810),
    vec2<f32>(-0.60554040, -0.34964830), vec2<f32>( 0.85095100,  0.47178830),
    vec2<f32>(-0.47994860,  0.08443340), vec2<f32>(-0.12494190, -0.76098760),
    vec2<f32>( 0.64839320,  0.74738240), vec2<f32>(-0.96815740, -0.12345680),
    vec2<f32>( 0.27682050, -0.80927180), vec2<f32>(-0.73016460,  0.18344200),
    vec2<f32>( 0.54754660,  0.06234570), vec2<f32>(-0.30967360, -0.61021430),
    vec2<f32>(-0.57774330,  0.80459740), vec2<f32>( 0.18238670, -0.37596540),
);

struct PbrInputs {
    world_pos:  vec3<f32>,
    world_n:    vec3<f32>,
    view_dir:   vec3<f32>,
    albedo:     vec3<f32>,
    metallic:   f32,
    roughness:  f32,
    ao:         f32,
    emissive:   vec3<f32>,
};

// Forward declaration: defined below; referenced by the lighting helpers.
// The full definition appears after the lighting helpers for readability.
// (WGSL allows module-scope identifiers to be used anywhere in the module
// regardless of source order.)

fn viewport_apply_scene_lighting(
    normal: vec3<f32>,
    base_colour: vec3<f32>,
    two_sided: bool,
    world_pos: vec3<f32>,
) -> vec3<f32> {
    let up_weight = clamp(normal.z * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(lights.ground_colour, lights.sky_colour, up_weight)
                  * lights.hemisphere_intensity;

    var direct = vec3<f32>(0.0);
    let n_lights = lights.count;
    for (var i: u32 = 0u; i < n_lights; i = i + 1u) {
        let l = lights_storage[i];
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
        let raw = dot(normal, L);
        let n_dot_l = select(max(raw, 0.0), abs(raw), two_sided);
        var shadow_factor = 1.0;
        if i == 0u {
            shadow_factor = viewport_sample_csm(world_pos, normal);
        }
        direct = direct + radiance * n_dot_l * shadow_factor;
    }
    return base_colour * (ambient + direct);
}

// Cascade selection + atlas tap. Mirrors the bias scheme and filter
// kernel used by the lib's built-in mesh pipeline so plugin items composite
// consistently in the same scene. Implementation details (cascade count,
// filter, bias) are internal and may change between catalog versions.
fn viewport_sample_csm(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if lights.shadows_enabled == 0u || lights.count == 0u {
        return 1.0;
    }

    let primary = lights_storage[0];
    var light_dir: vec3<f32>;
    if primary.light_type == 0u {
        light_dir = normalize(primary.pos_or_dir);
    } else {
        let to_light = primary.pos_or_dir - world_pos;
        light_dir = to_light / max(length(to_light), 0.0001);
    }

    let eye_pos = camera.eye_pos;
    let dist = dot(world_pos - eye_pos, camera.forward);

    var cascade_idx = 0u;
    for (var i = 0u; i < _viewport_csm.cascade_count; i = i + 1u) {
        if dist > _viewport_csm.cascade_splits[i] {
            cascade_idx = i + 1u;
        }
    }
    cascade_idx = min(cascade_idx, _viewport_csm.cascade_count - 1u);

    let light_clip = _viewport_csm.cascade_vp[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let tile_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    let rect = _viewport_csm.atlas_rects[cascade_idx];
    let atlas_uv = vec2<f32>(
        mix(rect.x, rect.z, tile_uv.x),
        mix(rect.y, rect.w, tile_uv.y),
    );

    let n_dot_l = dot(world_normal, light_dir);
    let offset_sign = select(-1.0, 1.0, n_dot_l >= 0.0);
    let vp = _viewport_csm.cascade_vp[cascade_idx];
    let vp_row0 = vec3<f32>(vp[0][0], vp[1][0], vp[2][0]);
    let vp_row1 = vec3<f32>(vp[0][1], vp[1][1], vp[2][1]);
    let vp_row2 = vec3<f32>(vp[0][2], vp[1][2], vp[2][2]);
    let texel_world = 2.0 / (length(vp_row0) * _viewport_csm.atlas_size * (rect.z - rect.x));

    let primary_light_type = primary.light_type;
    var offset_world: vec3<f32>;
    if primary_light_type == 0u {
        let normal_bias = texel_world * 1.5;
        offset_world = world_pos - light_dir * normal_bias;
    } else {
        let normal_bias = texel_world * mix(1.5, 0.0, clamp(abs(n_dot_l), 0.0, 1.0));
        offset_world = world_pos + world_normal * (offset_sign * normal_bias);
    }
    let offset_clip = _viewport_csm.cascade_vp[cascade_idx] * vec4<f32>(offset_world, 1.0);
    let biased_depth = (offset_clip.xyz / offset_clip.w).z - lights.shadow_bias;

    if tile_uv.x < 0.0 || tile_uv.x > 1.0 || tile_uv.y < 0.0 || tile_uv.y > 1.0 ||
       ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let n_ndc = vec3<f32>(
        dot(vp_row0, world_normal) / dot(vp_row0, vp_row0),
        dot(vp_row1, world_normal) / dot(vp_row1, vp_row1),
        dot(vp_row2, world_normal) / dot(vp_row2, vp_row2),
    );
    let nz_sign = select(-1.0, 1.0, n_ndc.z >= 0.0);
    let nz = nz_sign * max(abs(n_ndc.z), 1e-4);
    let rp_gate = select(0.0, 1.0, primary_light_type == 0u);
    let depth_grad = vec2<f32>(
        -n_ndc.x / nz * 2.0 / (rect.z - rect.x),
         n_ndc.y / nz * 2.0 / (rect.w - rect.y),
    ) * rp_gate;

    let texel_size = 1.0 / _viewport_csm.atlas_size;
    let noise = fract(52.9829189 * fract(dot(world_pos.xz, vec2<f32>(0.06711056, 0.00583715))));
    let rot = noise * 6.28318530;
    let sin_r = sin(rot);
    let cos_r = cos(rot);

    if _viewport_csm.shadow_filter == 1u {
        let search_radius = _viewport_csm.pcss_light_radius * 16.0 * texel_size;
        var blocker_sum = 0.0;
        var blocker_count = 0.0;
        for (var i = 0u; i < 16u; i = i + 1u) {
            let d = _VIEWPORT_POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * search_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let coords = vec2<i32>(clamped_uv * _viewport_csm.atlas_size);
            let raw_depth = textureLoad(shadow_atlas_tex, coords, 0);
            if raw_depth < ndc.z {
                blocker_sum = blocker_sum + raw_depth;
                blocker_count = blocker_count + 1.0;
            }
        }
        if blocker_count < 1.0 {
            return 1.0;
        }
        let avg_blocker = blocker_sum / blocker_count;
        let penumbra_width = _viewport_csm.pcss_light_radius * (biased_depth - avg_blocker) / max(avg_blocker, 0.001);
        let filter_radius = max(penumbra_width * 16.0 * texel_size, texel_size);
        var shadow = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let d = _VIEWPORT_POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * filter_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let tap_depth = biased_depth
                + clamp(dot(depth_grad, clamped_uv - atlas_uv), -0.005, 0.005);
            shadow = shadow + textureSampleCompare(shadow_atlas_tex, shadow_atlas_sampler, clamped_uv, tap_depth);
        }
        return shadow / 32.0;
    } else {
        let pcf_radius = select(4.0, 1.5, primary_light_type == 0u) * texel_size;
        var shadow = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let d = _VIEWPORT_POISSON_DISK[i];
            let rd = vec2<f32>(d.x * cos_r - d.y * sin_r, d.x * sin_r + d.y * cos_r);
            let sample_uv = atlas_uv + rd * pcf_radius;
            let clamped_uv = clamp(sample_uv, rect.xy, rect.zw);
            let tap_depth = biased_depth
                + clamp(dot(depth_grad, clamped_uv - atlas_uv), -0.005, 0.005);
            shadow = shadow + textureSampleCompare(shadow_atlas_tex, shadow_atlas_sampler, clamped_uv, tap_depth);
        }
        return shadow / 32.0;
    }
}

// Rotate a direction into the environment map's frame and project it to
// equirectangular UV. Matches the mapping used by the built-in mesh
// pipelines so plugins reflect the same environment.
fn viewport_dir_to_equirect_uv(dir: vec3<f32>, rotation: f32) -> vec2<f32> {
    let s = sin(rotation);
    let c = cos(rotation);
    let d = vec3<f32>(c * dir.x - s * dir.y, s * dir.x + c * dir.y, dir.z);
    return vec2<f32>(
        0.5 + atan2(d.y, d.x) / (2.0 * 3.14159265),
        0.5 - asin(clamp(d.z, -1.0, 1.0)) / 3.14159265,
    );
}

// Geometric specular anti-aliasing: widen perceptual roughness by the
// screen-space variance of the shading normal so detailed normal maps do
// not alias into per-pixel glints. Matches the built-in mesh pipelines.
fn viewport_specular_aa_roughness(n: vec3<f32>, roughness: f32) -> f32 {
    let du = dpdx(n);
    let dv = dpdy(n);
    let kernel = min(0.5 * (dot(du, du) + dot(dv, dv)), 0.18);
    let alpha = roughness * roughness;
    return sqrt(sqrt(clamp(alpha * alpha + kernel, 0.0, 1.0)));
}

fn viewport_fresnel_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let fr = max(vec3<f32>(1.0 - roughness), f0);
    return f0 + (fr - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Image-based ambient: diffuse irradiance plus prefiltered specular
// reflection from the environment, combined through the split-sum BRDF
// LUT. This is what lets metallic and smooth surfaces reflect the sky
// instead of going dark under hemisphere ambient alone.
fn viewport_ibl_ambient(
    n: vec3<f32>,
    v: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
    ao: f32,
) -> vec3<f32> {
    let n_dot_v = max(dot(n, v), 0.001);
    let f = viewport_fresnel_roughness(n_dot_v, f0, roughness);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let rotation = lights.ibl_rotation;
    let irradiance =
        textureSampleLevel(ibl_irradiance_tex, ibl_sampler, viewport_dir_to_equirect_uv(n, rotation), 0.0).rgb;
    let r = reflect(-v, n);
    // Mip floored by the screen-space footprint of r, matching the built-in
    // mesh pipelines: magnified reflections on detailed normals would
    // otherwise point-sample hot mip-0 texels and read as speckle.
    let dr = max(length(dpdx(r)), length(dpdy(r)));
    let footprint_mip = clamp(log2(max(dr * 128.0 / (2.0 * 3.14159265), 1.0)), 0.0, 4.0);
    let prefiltered = textureSampleLevel(
        ibl_specular_tex,
        ibl_sampler,
        viewport_dir_to_equirect_uv(r, rotation),
        max(roughness * 4.0, footprint_mip),
    ).rgb;
    let brdf = textureSampleLevel(ibl_brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
    let diffuse = kd * irradiance * albedo * ao;
    let specular = prefiltered * (f * brdf.x + brdf.y) * ao;
    return (diffuse + specular) * lights.ibl_intensity;
}

// PBR shading. Cook-Torrance specular with GGX NDF + Smith G + Schlick
// Fresnel, Lambert diffuse weighted by (1 - metallic). Integrates against
// every active scene light; IBL contribution is added when
// `lights.ibl_enabled != 0`.
fn viewport_pbr_shade(inp: PbrInputs) -> vec3<f32> {
    let N = normalize(inp.world_n);
    let V = normalize(inp.view_dir);
    let roughness = viewport_specular_aa_roughness(N, max(inp.roughness, 0.04));
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let f0 = mix(vec3<f32>(0.04), inp.albedo, inp.metallic);

    // Environment reflection when an IBL probe is active, otherwise a
    // cheap hemisphere ambient. The IBL path gives metallic and smooth
    // surfaces something bright to reflect instead of reading as black.
    var ambient: vec3<f32>;
    if lights.ibl_enabled != 0u {
        ambient = viewport_ibl_ambient(N, V, inp.albedo, inp.metallic, roughness, f0, inp.ao);
    } else {
        let up_weight = clamp(N.z * 0.5 + 0.5, 0.0, 1.0);
        ambient = mix(lights.ground_colour, lights.sky_colour, up_weight)
                  * lights.hemisphere_intensity * inp.albedo * inp.ao;
    }

    var lo = vec3<f32>(0.0);
    let n_lights = lights.count;
    for (var i: u32 = 0u; i < n_lights; i = i + 1u) {
        let l = lights_storage[i];
        var L: vec3<f32>;
        var radiance: vec3<f32>;
        if l.light_type == 0u {
            L = normalize(l.pos_or_dir);
            radiance = l.colour * l.intensity;
        } else if l.light_type == 1u {
            let to_light = l.pos_or_dir - inp.world_pos;
            let dist = length(to_light);
            L = to_light / max(dist, 0.0001);
            let falloff = clamp(1.0 - dist / l.range, 0.0, 1.0);
            radiance = l.colour * l.intensity * falloff * falloff;
        } else {
            let to_light = l.pos_or_dir - inp.world_pos;
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

        let H = normalize(V + L);
        let n_dot_l = max(dot(N, L), 0.0);
        let n_dot_v = max(dot(N, V), 0.0001);
        let n_dot_h = max(dot(N, H), 0.0);
        let v_dot_h = max(dot(V, H), 0.0);

        let denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
        let D = alpha2 / max(3.14159265 * denom * denom, 1e-6);
        let k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
        let G1v = n_dot_v / (n_dot_v * (1.0 - k) + k);
        let G1l = n_dot_l / max(n_dot_l * (1.0 - k) + k, 1e-6);
        let G = G1v * G1l;
        let F = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
        let spec = (D * G) * F / max(4.0 * n_dot_v * n_dot_l, 1e-6);

        let kd = (vec3<f32>(1.0) - F) * (1.0 - inp.metallic);
        let diff = kd * inp.albedo / 3.14159265;
        var shadow_factor = 1.0;
        if i == 0u {
            shadow_factor = viewport_sample_csm(inp.world_pos, N);
        }
        lo = lo + (diff + spec) * radiance * n_dot_l * shadow_factor;
    }

    return ambient + lo + inp.emissive;
}
"#;

/// Fragment-output struct and packing helper for the OIT pass.
///
/// A transparent plugin fragment shader returns [`OitOutput`] from its
/// `fs_main`. Use `viewport_oit_pack(color_premul, alpha, view_z)` to
/// build the struct; the weight function matches the lib's built-in OIT
/// pipelines so plugin transparents composite consistently with native
/// transparents in the same pass.
pub const SHARED_OIT_WGSL: &str = r#"
// @viewport-wgsl-version: 1
// OIT MRT output struct and pack helper. Requires SHARED_BINDINGS_WGSL.
//
// Use as:
//   @fragment
//   fn fs_main(...) -> OitOutput {
//       return viewport_oit_pack(color_rgb, alpha, in.view_z);
//   }
//
// `view_z` is the view-space Z coordinate (negative in front of the
// camera). The weight function biases nearer fragments toward higher
// contribution, matching the weight curve in mesh_oit.wgsl.

struct OitOutput {
    @location(0) accum:  vec4<f32>,
    @location(1) reveal: f32,
};

fn viewport_oit_weight(view_z: f32, alpha: f32) -> f32 {
    // Weight curve from McGuire & Bavoil 2013, equation 7. Tuned for the
    // lib's typical scene depth range.
    let z = abs(view_z);
    let w = alpha * clamp(10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0)), 1e-2, 3e3);
    return w;
}

fn viewport_oit_pack(color: vec3<f32>, alpha: f32, view_z: f32) -> OitOutput {
    let w = viewport_oit_weight(view_z, alpha);
    var out: OitOutput;
    out.accum  = vec4<f32>(color * alpha * w, alpha * w);
    out.reveal = alpha;
    return out;
}
"#;

/// Fragment helper for the outline mask pass.
///
/// A plugin's mask pipeline reuses its scene-pass vertex stage and uses
/// `fs_mask` (or any function returning `@location(0) f32 = 1.0`). The
/// composite reads any non-zero value as "this pixel belongs to a selected
/// item." Depth state must match the mask pass: depth test on, depth write
/// off.
pub const SHARED_MASK_WGSL: &str = r#"
// @viewport-wgsl-version: 1
// Outline-mask fragment helper. Returns a single R8 value of 1.0 for any
// covered pixel; the composite reads the mask and draws the outline edge.

@fragment
fn viewport_mask_fs() -> @location(0) f32 {
    return 1.0;
}
"#;

/// Fragment helper for the pick-id pass.
///
/// A plugin's pick pipeline reuses its scene-pass vertex stage (extended to
/// pass a flat-interpolated `pick_id: u32`) and uses `fs_pick`. The
/// renderer reads back the R32U pixel under the cursor to resolve which
/// item was clicked.
pub const SHARED_PICK_WGSL: &str = r#"
// @viewport-wgsl-version: 1
// Pick-id fragment helper for the three-target pick pass. The vertex stage must
// provide a flat-interpolated pick_id at @location(0) of the fragment input; see
// your pipeline's vertex shader for the matching declaration.
//
// The pass targets: @location(0) R32Uint object id, @location(1) R32Uint
// sub-object index (0 here; use viewport_pick_prim_fs from
// SHARED_PICK_PRIM_WGSL to write the triangle index for sub-object picking),
// and @location(2) R32Float depth (the framebuffer z, so the renderer can
// reconstruct world position on read-back). A pipeline built with
// build_pick_pipeline must write all three, which this helper does.

struct ViewportPickOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
};

@fragment
fn viewport_pick_fs(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) @interpolate(flat) pick_id: u32,
) -> ViewportPickOut {
    var out: ViewportPickOut;
    out.object_id = pick_id;
    out.primitive_id = 0u;
    out.depth = frag_pos.z;
    return out;
}
"#;

/// Module directive required by [`SHARED_PICK_PRIM_WGSL`], per wgpu leg.
///
/// naga 29 only accepts `@builtin(primitive_index)` in a module that starts
/// with `enable primitive_index;`; naga 27 rejects the directive. Prepend this
/// constant at the very top of the shader source (before any declaration, so
/// ahead of the vertex stage too) and it resolves to the right text for the
/// active leg.
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
pub const PICK_PRIM_ENABLE_WGSL: &str = "";
/// Module directive required by [`SHARED_PICK_PRIM_WGSL`], per wgpu leg.
///
/// naga 29 only accepts `@builtin(primitive_index)` in a module that starts
/// with `enable primitive_index;`; naga 27 rejects the directive. Prepend this
/// constant at the very top of the shader source (before any declaration, so
/// ahead of the vertex stage too) and it resolves to the right text for the
/// active leg.
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
pub const PICK_PRIM_ENABLE_WGSL: &str = "enable primitive_index;\n";

/// Fragment helper for the pick-id pass that reports the hit triangle.
///
/// Same contract as [`SHARED_PICK_WGSL`]'s `viewport_pick_fs`, except the
/// primitive channel carries `@builtin(primitive_index)` : the index of the
/// rasterised triangle within the draw call : instead of `0u`. With this, a
/// hit on the item can be refined to a sub-object: the renderer hands the
/// read-back index to the plugin's
/// [`resolve_sub_object`](crate::plugin_api::ItemTypePlugin::resolve_sub_object).
///
/// Requirements:
/// - The device must have
///   [`PRIMITIVE_INDEX_FEATURE`](crate::gpu::PRIMITIVE_INDEX_FEATURE); a
///   module using the builtin fails validation without it. Check
///   `device.features()` at `init_gpu` and fall back to `viewport_pick_fs`
///   (picking then stays object-level, matching the built-in surfaces).
/// - The module source must start with [`PICK_PRIM_ENABLE_WGSL`], before any
///   other code:
///   `format!("{PICK_PRIM_ENABLE_WGSL}{MY_VS}{SHARED_PICK_PRIM_WGSL}")`.
pub const SHARED_PICK_PRIM_WGSL: &str = r#"
// @viewport-wgsl-version: 1
// Pick-id fragment helper for the three-target pick pass, primitive-index
// variant. The vertex stage must provide a flat-interpolated pick_id at
// @location(0) of the fragment input, exactly as for viewport_pick_fs.
//
// Targets: @location(0) R32Uint object id, @location(1) R32Uint triangle
// index from @builtin(primitive_index), @location(2) R32Float depth.

struct ViewportPickPrimOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
};

@fragment
fn viewport_pick_prim_fs(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(primitive_index) prim_index: u32,
    @location(0) @interpolate(flat) pick_id: u32,
) -> ViewportPickPrimOut {
    var out: ViewportPickPrimOut;
    out.object_id = pick_id;
    out.primitive_id = prim_index;
    out.depth = frag_pos.z;
    return out;
}
"#;
