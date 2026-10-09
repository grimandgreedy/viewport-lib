
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
    env_zone_count:       u32,
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
@group(0) @binding(7)  var          ibl_irradiance_tex:   texture_2d_array<f32>;
@group(0) @binding(8)  var          ibl_specular_tex:     texture_2d_array<f32>;
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
