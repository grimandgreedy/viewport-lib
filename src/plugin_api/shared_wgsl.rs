//! WGSL helper catalog.
//!
//! Each helper is a `&'static str` of WGSL that a plugin shader prefixes (or
//! concatenates into its own shader source) to gain access to the lib's
//! shared bindings, shading helpers, and target conventions.
//!
//! **Versioning.** Each helper carries a `// @viewport-wgsl-version: N`
//! comment. The version is bumped whenever a function signature, struct
//! field, or binding number changes. Plugins compare against
//! [`WGSL_VERSION`] at build time to detect breakage early. Function bodies
//! and private fields are not part of the contract and may change between
//! patch releases.
//!
//! **Composition.** Plugin shaders typically build their source as:
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
pub const WGSL_VERSION: u32 = 1;

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
/// | 5  | `ShadowInfo` uniform (CSM matrices, splits) | `shadow_info` |
/// | 6  | `ClipVolumes` uniform | `clip_volumes` |
/// | 7  | IBL irradiance equirect | `ibl_irradiance_tex` |
/// | 8  | IBL specular equirect | `ibl_specular_tex` |
/// | 9  | BRDF integration LUT | `ibl_brdf_lut` |
/// | 10 | IBL sampler | `ibl_sampler` |
/// | 11 | Skybox equirect | `skybox_tex` |
/// | 12 | debug fragment storage buffer | `debug_frag` |
/// | 13 | per-light array | `lights_storage` |
pub const SHARED_BINDINGS_WGSL: &str = r#"
// @viewport-wgsl-version: 1
// Shared group-0 declarations. Do not re-declare these bindings in plugin
// shaders.

struct Camera {
    view_proj:        mat4x4<f32>,
    view:             mat4x4<f32>,
    proj:             mat4x4<f32>,
    inv_view:         mat4x4<f32>,
    inv_proj:         mat4x4<f32>,
    eye_pos:          vec3<f32>,
    _pad0:            f32,
    viewport_size:    vec2<f32>,
    _pad1:            vec2<f32>,
};

struct SingleLight {
    light_view_proj: mat4x4<f32>,
    pos_or_dir:      vec3<f32>,
    light_type:      u32,
    colour:          vec3<f32>,
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

@group(0) @binding(0)  var<uniform> camera:               Camera;
@group(0) @binding(1)  var          shadow_atlas_tex:     texture_depth_2d;
@group(0) @binding(2)  var          shadow_atlas_sampler: sampler_comparison;
@group(0) @binding(3)  var<uniform> lights:               Lights;
@group(0) @binding(7)  var          ibl_irradiance_tex:   texture_2d<f32>;
@group(0) @binding(8)  var          ibl_specular_tex:     texture_2d<f32>;
@group(0) @binding(9)  var          ibl_brdf_lut:         texture_2d<f32>;
@group(0) @binding(10) var          ibl_sampler:          sampler;
@group(0) @binding(11) var          skybox_tex:           texture_2d<f32>;
@group(0) @binding(13) var<storage, read> lights_storage: array<SingleLight>;

// Clip and shadow-info uniforms are opaque to plugins; access via the
// helper functions in SHARED_PBR_WGSL (viewport_apply_clip_volumes,
// viewport_sample_csm) rather than reading the raw uniforms directly.
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
/// emissive. It applies the lib's standard hemisphere ambient + light loop +
/// (optionally) shadow attenuation. Future revisions may add IBL and SSAO
/// sampling inside this function; consumers should rebuild their shaders
/// when the catalog version bumps to pick up the upgrade.
///
/// `viewport_sample_csm` returns a 0..1 shadow factor for `world_pos`.
/// Returns 1.0 (fully lit) when shadows are disabled or the position is
/// outside every cascade.
///
/// `viewport_apply_scene_lighting` is the simpler Lambert helper used by
/// non-PBR pipelines (glyphs, tubes, ribbons). Use it when a plugin wants
/// scene-light parity with those built-in items.
pub const SHARED_PBR_WGSL: &str = r#"
// @viewport-wgsl-version: 1
// Shared PBR / lit-shading helpers. Requires SHARED_BINDINGS_WGSL to be
// included first.

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
        direct = direct + radiance * n_dot_l;
    }
    return base_colour * (ambient + direct);
}

// Placeholder shadow sampler. The full CSM tap requires the shadow_info
// uniform layout, which is not yet part of the published group-0 contract;
// while it stabilises this function returns 1.0 (fully lit). A future
// catalog version will wire this to the atlas tap once the layout is
// frozen.
fn viewport_sample_csm(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    return 1.0;
}

// PBR shading. Cook-Torrance specular with GGX NDF + Smith G + Schlick
// Fresnel, Lambert diffuse weighted by (1 - metallic). Integrates against
// every active scene light; IBL contribution is added when
// `lights.ibl_enabled != 0`.
fn viewport_pbr_shade(inp: PbrInputs) -> vec3<f32> {
    let N = normalize(inp.world_n);
    let V = normalize(inp.view_dir);
    let roughness = max(inp.roughness, 0.04);
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let f0 = mix(vec3<f32>(0.04), inp.albedo, inp.metallic);

    // Hemisphere ambient (kept for parity with non-PBR pipelines when IBL
    // is disabled).
    let up_weight = clamp(N.z * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(lights.ground_colour, lights.sky_colour, up_weight)
                  * lights.hemisphere_intensity * inp.albedo * inp.ao;

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
        lo = lo + (diff + spec) * radiance * n_dot_l;
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
// Pick-id fragment helper. The vertex stage must provide a flat-interpolated
// pick_id at @location(0) of the fragment input; see your pipeline's vertex
// shader for the matching declaration.

@fragment
fn viewport_pick_fs(
    @location(0) @interpolate(flat) pick_id: u32,
) -> @location(0) u32 {
    return pick_id;
}
"#;
