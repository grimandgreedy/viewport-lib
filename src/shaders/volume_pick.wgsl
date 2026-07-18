// Volume object-pick shader. Rasterises the volume's bounding cube and marches
// the 3D texture to the first in-threshold voxel, writing the volume's pick id
// and that voxel's depth into the three pick targets. Fragments whose ray finds
// no in-threshold sample are discarded, so the pick selects the volume's actual
// rendered content rather than its bounding box. Mirrors the ray setup, clip
// handling, and threshold test of `volume.wgsl`.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct VolumeUniform {
    model:           mat4x4<f32>,  // unit cube -> world space
    inv_model:       mat4x4<f32>,  // world space -> unit cube [0,1]^3
    bbox_min:        vec3<f32>,
    step_size:       f32,
    bbox_max:        vec3<f32>,
    opacity_scale:   f32,
    scalar_min:      f32,
    scalar_max:      f32,
    threshold_min:   f32,
    threshold_max:   f32,
    enable_shading:  u32,
    num_clip_planes: u32,
    use_nan_colour:  u32,
    _pad0:           u32,
    nan_colour:      vec4<f32>,
    clip_planes:     array<vec4<f32>, 6>,
};

struct ClipVolumeEntry {
    volume_type: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    center: vec3<f32>,
    radius: f32,
    half_extents: vec3<f32>,
    _pad1: f32,
    col0: vec3<f32>,
    _pad2: f32,
    col1: vec3<f32>,
    _pad3: f32,
    col2: vec3<f32>,
    _pad4: f32,
}

struct ClipVolumeUB {
    count: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    volumes: array<ClipVolumeEntry, 4>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

@group(1) @binding(0) var<uniform> volume: VolumeUniform;
@group(1) @binding(1) var volume_tex: texture_3d<f32>;
@group(1) @binding(2) var volume_nearest_sampler: sampler;

@group(2) @binding(0) var<uniform> pick_id: vec4<u32>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VertexOutput {
    let world = volume.model * vec4<f32>(pos, 1.0);
    var out: VertexOutput;
    out.clip_pos = camera.view_proj * world;
    out.world_pos = world.xyz;
    return out;
}

fn intersect_unit_box(ray_origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    let t0 = (vec3<f32>(0.0) - ray_origin) * inv_dir;
    let t1 = (vec3<f32>(1.0) - ray_origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

fn clip_ray_plane(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    plane_normal: vec3<f32>,
    plane_d: f32,
    t_near: f32,
    t_far: f32,
) -> vec2<f32> {
    let denom = dot(plane_normal, ray_dir);
    let num = -(dot(plane_normal, ray_origin) + plane_d);
    if abs(denom) < 1e-8 {
        if num < 0.0 {
            return vec2<f32>(t_far + 1.0, t_far);
        }
        return vec2<f32>(t_near, t_far);
    }
    let t = num / denom;
    if denom > 0.0 {
        return vec2<f32>(max(t_near, t), t_far);
    } else {
        return vec2<f32>(t_near, min(t_far, t));
    }
}

fn clip_ray_box(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    e: ClipVolumeEntry,
    t_near_in: f32,
    t_far_in: f32,
) -> vec2<f32> {
    let d = ray_origin - e.center;
    let local_origin = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
    let local_dir = vec3<f32>(dot(ray_dir, e.col0), dot(ray_dir, e.col1), dot(ray_dir, e.col2));
    let he = e.half_extents;
    var t_near = t_near_in;
    var t_far  = t_far_in;
    for (var ax = 0u; ax < 3u; ax = ax + 1u) {
        var lo: f32; var la: f32;
        if ax == 0u { lo = local_origin.x; la = local_dir.x; }
        else if ax == 1u { lo = local_origin.y; la = local_dir.y; }
        else { lo = local_origin.z; la = local_dir.z; }
        var he_ax: f32;
        if ax == 0u { he_ax = he.x; }
        else if ax == 1u { he_ax = he.y; }
        else { he_ax = he.z; }
        if abs(la) < 1e-8 {
            if abs(lo) > he_ax { return vec2<f32>(t_far + 1.0, t_far); }
        } else {
            let inv = 1.0 / la;
            var t0 = (-he_ax - lo) * inv;
            var t1 = ( he_ax - lo) * inv;
            if t0 > t1 { let tmp = t0; t0 = t1; t1 = tmp; }
            t_near = max(t_near, t0);
            t_far  = min(t_far,  t1);
        }
    }
    return vec2<f32>(t_near, t_far);
}

fn clip_ray_sphere(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    e: ClipVolumeEntry,
    t_near_in: f32,
    t_far_in: f32,
) -> vec2<f32> {
    let oc = ray_origin - e.center;
    let b  = dot(oc, ray_dir);
    let c  = dot(oc, oc) - e.radius * e.radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return vec2<f32>(t_far_in + 1.0, t_far_in);
    }
    let sqrt_d = sqrt(discriminant);
    let t0 = -b - sqrt_d;
    let t1 = -b + sqrt_d;
    return vec2<f32>(max(t_near_in, t0), min(t_far_in, t1));
}

fn clip_ray_cylinder(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    e: ClipVolumeEntry,
    t_near_in: f32,
    t_far_in: f32,
) -> vec2<f32> {
    let axis     = e.col0;
    let half_len = e.half_extents.x;
    let oc       = ray_origin - e.center;
    let d_dot_a  = dot(ray_dir, axis);
    let oc_dot_a = dot(oc, axis);
    let d_perp   = ray_dir - axis * d_dot_a;
    let oc_perp  = oc      - axis * oc_dot_a;
    let qa = dot(d_perp, d_perp);
    let qb = dot(oc_perp, d_perp);
    let qc = dot(oc_perp, oc_perp) - e.radius * e.radius;
    var t_near = t_near_in;
    var t_far  = t_far_in;
    if qa < 1e-12 {
        if qc > 0.0 { return vec2<f32>(t_far + 1.0, t_far); }
    } else {
        let discriminant = qb * qb - qa * qc;
        if discriminant < 0.0 { return vec2<f32>(t_far + 1.0, t_far); }
        let sqrt_d = sqrt(discriminant);
        let inv_a  = 1.0 / qa;
        t_near = max(t_near, (-qb - sqrt_d) * inv_a);
        t_far  = min(t_far,  (-qb + sqrt_d) * inv_a);
    }
    if abs(d_dot_a) < 1e-8 {
        if abs(oc_dot_a) > half_len { return vec2<f32>(t_far + 1.0, t_far); }
    } else {
        let inv_da  = 1.0 / d_dot_a;
        let t_cap0  = (-half_len - oc_dot_a) * inv_da;
        let t_cap1  = ( half_len - oc_dot_a) * inv_da;
        t_near = max(t_near, min(t_cap0, t_cap1));
        t_far  = min(t_far,  max(t_cap0, t_cap1));
    }
    return vec2<f32>(t_near, t_far);
}

struct PickOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
    @builtin(frag_depth) frag_depth: f32,
};

@fragment
fn fs_pick(in: VertexOutput) -> PickOut {
    // Ray in unit-cube [0,1]^3 model space (matches volume.wgsl fs_main).
    let eye_model = (volume.inv_model * vec4<f32>(camera.eye_pos, 1.0)).xyz;
    let frag_model = (volume.inv_model * vec4<f32>(in.world_pos, 1.0)).xyz;
    let ray_dir = normalize(frag_model - eye_model);
    let inv_dir = 1.0 / ray_dir;

    var t_range = intersect_unit_box(eye_model, inv_dir);
    var t_near = max(t_range.x, 0.0);
    var t_far = t_range.y;

    // Per-volume clip planes (model space).
    let inv_model_t = transpose(volume.model);
    for (var i = 0u; i < volume.num_clip_planes; i = i + 1u) {
        let plane_world = volume.clip_planes[i];
        let n_world = plane_world.xyz;
        let d_world = plane_world.w;
        let n_model_raw = (inv_model_t * vec4<f32>(n_world, 0.0)).xyz;
        let n_model_len = length(n_model_raw);
        if n_model_len < 1e-8 { continue; }
        let n_model = n_model_raw / n_model_len;
        let p_on_plane_world = n_world * (-d_world);
        let p_on_plane_model = (volume.inv_model * vec4<f32>(p_on_plane_world, 1.0)).xyz;
        let d_model = -dot(n_model, p_on_plane_model);
        let clipped = clip_ray_plane(eye_model, ray_dir, n_model, d_model, t_near, t_far);
        t_near = clipped.x;
        t_far = clipped.y;
    }

    // Global section clip volumes (world space).
    if clip_volume.count > 0u {
        let model3 = mat3x3<f32>(volume.model[0].xyz, volume.model[1].xyz, volume.model[2].xyz);
        let dir_world_raw = model3 * ray_dir;
        let dir_world_len = length(dir_world_raw);
        if dir_world_len > 1e-8 {
            let dir_world = dir_world_raw / dir_world_len;
            let scale = dir_world_len;
            var tw_near = t_near * scale;
            var tw_far  = t_far  * scale;
            for (var i = 0u; i < clip_volume.count; i = i + 1u) {
                let e = clip_volume.volumes[i];
                if e.volume_type == 2u {
                    let r = clip_ray_box(camera.eye_pos, dir_world, e, tw_near, tw_far);
                    tw_near = r.x; tw_far = r.y;
                } else if e.volume_type == 3u {
                    let r = clip_ray_sphere(camera.eye_pos, dir_world, e, tw_near, tw_far);
                    tw_near = r.x; tw_far = r.y;
                } else if e.volume_type == 4u {
                    let r = clip_ray_cylinder(camera.eye_pos, dir_world, e, tw_near, tw_far);
                    tw_near = r.x; tw_far = r.y;
                }
            }
            t_near = tw_near / scale;
            t_far  = tw_far  / scale;
        }
    }

    if t_near >= t_far {
        discard;
    }

    // March to the first non-NaN, in-threshold voxel: that is the hit.
    let step = volume.step_size;
    let max_steps = 512u;
    var t = max(t_near, 0.0);
    for (var i = 0u; i < max_steps; i = i + 1u) {
        if t >= t_far { break; }
        let sample_pos = eye_model + ray_dir * t;
        let raw_scalar = textureSampleLevel(volume_tex, volume_nearest_sampler, sample_pos, 0.0).r;
        let is_nan = raw_scalar != raw_scalar;
        if is_nan { t = t + step; continue; }
        if raw_scalar < volume.threshold_min || raw_scalar > volume.threshold_max {
            t = t + step;
            continue;
        }
        let world_hit = (volume.model * vec4<f32>(sample_pos, 1.0)).xyz;
        let clip = camera.view_proj * vec4<f32>(world_hit, 1.0);
        let ndc_z = clip.z / clip.w;
        // Flat index of the hit voxel, matching the CPU voxel pick and the
        // highlight decode: flat = ix + iy*nx + iz*nx*ny. sample_pos is in the
        // unit cube [0,1]^3, so scale by the texture grid dimensions.
        let dims = vec3<f32>(textureDimensions(volume_tex, 0));
        let vox = clamp(floor(sample_pos * dims), vec3<f32>(0.0), dims - 1.0);
        let flat = u32(vox.x) + u32(vox.y) * u32(dims.x) + u32(vox.z) * u32(dims.x) * u32(dims.y);
        var out: PickOut;
        out.object_id = pick_id.x;
        out.primitive_id = flat;
        out.depth = ndc_z;
        out.frag_depth = ndc_z;
        return out;
    }

    // Ray passed through with no in-threshold sample: not a hit.
    discard;
    var miss: PickOut;
    miss.object_id = 0u;
    miss.primitive_id = 0u;
    miss.depth = 1.0;
    miss.frag_depth = 1.0;
    return miss;
}
