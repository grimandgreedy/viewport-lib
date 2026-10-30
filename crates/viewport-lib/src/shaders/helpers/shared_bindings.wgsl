
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
    // Source radius: clamps the inverse-square falloff near the light.
    radius:            f32,
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
    ibl_diffuse_scale:    f32,
    debug_vis_split_x:    f32,
    env_zone_count:       u32,
    ibl_specular_scale:   f32,
    _pad_dbg_c:           u32,
};

// The clip uniforms (bindings 4 and 6) and the clip tests.
// #include "clip_bindings.wgsl"

@group(0) @binding(0)  var<uniform> camera:               Camera;
@group(0) @binding(1)  var          shadow_atlas_tex:     texture_depth_2d;
@group(0) @binding(2)  var          shadow_atlas_sampler: sampler_comparison;
@group(0) @binding(3)  var<uniform> lights:               Lights;
@group(0) @binding(7)  var          ibl_irradiance_tex:   texture_2d_array<f32>;
@group(0) @binding(8)  var          ibl_specular_tex:     texture_2d_array<f32>;
@group(0) @binding(9)  var          ibl_brdf_lut:         texture_2d<f32>;
@group(0) @binding(10) var          ibl_sampler:          sampler;
@group(0) @binding(11) var          skybox_tex:           texture_2d<f32>;
// BEGIN_LIGHT_LIST
@group(0) @binding(13) var<storage, read> lights_storage: array<SingleLight>;
// ALT(UNIFORM): @group(0) @binding(13) var<uniform> lights_storage: array<SingleLight, 16>;
// END_LIGHT_LIST
@group(0) @binding(17) var          point_shadow_cube:    texture_depth_cube_array;

// The shadow-info uniform (binding 5) is declared inside SHARED_PBR_WGSL
// for the sole use of `viewport_sample_csm`. Its struct layout and field
// set are intentionally not part of the published contract and may change
// between catalog versions. Plugins must not redeclare binding 5 and must
// not read the uniform directly; route shadow queries through
// `viewport_sample_csm`.
