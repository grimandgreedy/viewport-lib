// Volume ray-march shader for the 3D viewport.
//
// Group 0: Camera uniform (view-projection, eye position)
//          + bindings 1-5 matching camera_bgl (unused here except clip planes).
// Group 1: Volume-specific bindings:
//          binding 0: VolumeUniform (per-volume parameters)
//          binding 1: 3D scalar texture (R32Float, non-filterable)
//          binding 2: nearest sampler for 3D texture (non-filtering)
//          binding 3: colour LUT texture (256x1, Rgba8Unorm)
//          binding 4: opacity LUT texture (256x1, R8Unorm)
//          binding 5: linear sampler for LUT textures (filtering)
//
// Vertex input: unit cube [0,1]^3 positions (location 0).
// The model matrix maps the unit cube to the world-space bounding box.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:  u32,
    _pad0:  u32,
    viewport_width:  f32,
    viewport_height: f32,
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
    threshold_min:   f32,  // raw scalar value; samples below are discarded
    threshold_max:   f32,  // raw scalar value; samples above are discarded
    enable_shading:  u32,
    num_clip_planes: u32,
    use_nan_colour:   u32,  // 1 = render NaN voxels with nan_colour; 0 = skip them
    _pad0:           u32,
    nan_colour:       vec4<f32>,  // RGBA colour for NaN voxels (used when use_nan_colour == 1)
    clip_planes:     array<vec4<f32>, 6>,  // normal.xyz + distance
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
@group(0) @binding(4) var<uniform> clip_planes_ub: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

fn clip_volume_test(p: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_volume.count; i = i + 1u) {
        let e = clip_volume.volumes[i];
        if e.volume_type == 2u {
            let d = p - e.center;
            let local = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
            if abs(local.x) > e.half_extents.x
                || abs(local.y) > e.half_extents.y
                || abs(local.z) > e.half_extents.z {
                return false;
            }
        } else if e.volume_type == 3u {
            let ds = p - e.center;
            if dot(ds, ds) > e.radius * e.radius { return false; }
        } else if e.volume_type == 4u {
            let axis = e.col0;
            let d = p - e.center;
            let along = dot(d, axis);
            if abs(along) > e.half_extents.x { return false; }
            let radial = d - axis * along;
            if dot(radial, radial) > e.radius * e.radius { return false; }
        }
    }
    return true;
}

@group(1) @binding(0) var<uniform> volume: VolumeUniform;
@group(1) @binding(1) var volume_tex: texture_3d<f32>;
@group(1) @binding(2) var volume_nearest_sampler: sampler;
@group(1) @binding(3) var colour_lut: texture_2d<f32>;
@group(1) @binding(4) var opacity_lut: texture_2d<f32>;
@group(1) @binding(5) var lut_sampler: sampler;

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

// Ray-AABB intersection in unit cube space [0,1]^3.
fn intersect_unit_box(ray_origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    let t0 = (vec3<f32>(0.0) - ray_origin) * inv_dir;
    let t1 = (vec3<f32>(1.0) - ray_origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

// Clip ray against a single half-space plane in model space.
// Plane: dot(normal, p) + d >= 0 is the visible half-space.
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
        // Ray parallel to plane.
        if num < 0.0 {
            // Origin outside plane : no intersection.
            return vec2<f32>(t_far + 1.0, t_far);
        }
        return vec2<f32>(t_near, t_far);
    }
    let t = num / denom;
    if denom > 0.0 {
        // Entering the visible half-space.
        return vec2<f32>(max(t_near, t), t_far);
    } else {
        // Exiting the visible half-space.
        return vec2<f32>(t_near, min(t_far, t));
    }
}

// Clip ray against an oriented box in world space.
// Box is defined by center + half_extents + orientation columns (world-space local axes).
// Transforms the ray and box to box-local space, clips against [-half, +half] AABB.
// Returns (t_near, t_far) with t_near > t_far indicating no intersection.
fn clip_ray_box(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    e: ClipVolumeEntry,
    t_near_in: f32,
    t_far_in: f32,
) -> vec2<f32> {
    // Transform ray to box-local space.
    let d = ray_origin - e.center;
    let local_origin = vec3<f32>(
        dot(d, e.col0),
        dot(d, e.col1),
        dot(d, e.col2),
    );
    let local_dir = vec3<f32>(
        dot(ray_dir, e.col0),
        dot(ray_dir, e.col1),
        dot(ray_dir, e.col2),
    );
    let he = e.half_extents;
    // AABB slab test against [-he, +he].
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
            // Ray parallel to slab.
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

// Clip ray against a sphere (world space).
// Returns (t_near, t_far) with t_near > t_far indicating no intersection.
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
        // No intersection with sphere.
        return vec2<f32>(t_far_in + 1.0, t_far_in);
    }
    let sqrt_d = sqrt(discriminant);
    let t0 = -b - sqrt_d;
    let t1 = -b + sqrt_d;
    return vec2<f32>(max(t_near_in, t0), min(t_far_in, t1));
}

// Clip ray against a finite cylinder (world space).
// Cylinder axis: e.col0 (unit), center: e.center, radius: e.radius,
// half-length: e.half_extents.x.
// Returns (t_near, t_far) with t_near > t_far indicating no intersection.
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

    // Components perpendicular to the cylinder axis.
    let d_dot_a  = dot(ray_dir, axis);
    let oc_dot_a = dot(oc, axis);
    let d_perp   = ray_dir - axis * d_dot_a;
    let oc_perp  = oc      - axis * oc_dot_a;

    // Quadratic coefficients for the infinite cylinder.
    let qa = dot(d_perp, d_perp);
    let qb = dot(oc_perp, d_perp);
    let qc = dot(oc_perp, oc_perp) - e.radius * e.radius;

    var t_near = t_near_in;
    var t_far  = t_far_in;

    if qa < 1e-12 {
        // Ray parallel to axis: outside the radial extent means no hit.
        if qc > 0.0 { return vec2<f32>(t_far + 1.0, t_far); }
        // Otherwise the ray is inside the infinite cylinder; skip to cap clip.
    } else {
        let discriminant = qb * qb - qa * qc;
        if discriminant < 0.0 { return vec2<f32>(t_far + 1.0, t_far); }
        let sqrt_d = sqrt(discriminant);
        let inv_a  = 1.0 / qa;
        t_near = max(t_near, (-qb - sqrt_d) * inv_a);
        t_far  = min(t_far,  (-qb + sqrt_d) * inv_a);
    }

    // Axial slab: along(t) = oc_dot_a + t * d_dot_a must be in [-half_len, half_len].
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Transform ray to model space (unit cube [0,1]^3).
    let eye_model = (volume.inv_model * vec4<f32>(camera.eye_pos, 1.0)).xyz;
    let frag_model = (volume.inv_model * vec4<f32>(in.world_pos, 1.0)).xyz;
    let ray_dir = normalize(frag_model - eye_model);
    let inv_dir = 1.0 / ray_dir;

    // Intersect ray with unit box.
    var t_range = intersect_unit_box(eye_model, inv_dir);
    var t_near = max(t_range.x, 0.0);  // Don't start behind the camera.
    var t_far = t_range.y;

    // Clip against active clip planes (transform planes to model space).
    let inv_model_t = transpose(volume.model);  // For transforming plane normals.
    for (var i = 0u; i < volume.num_clip_planes; i = i + 1u) {
        let plane_world = volume.clip_planes[i];
        // Transform plane to model space: normal' = M^T * normal, d' adjusted.
        let n_world = plane_world.xyz;
        let d_world = plane_world.w;
        // Transform normal with M^T (not M^-1) to get model-space normal.
        let n_model_raw = (inv_model_t * vec4<f32>(n_world, 0.0)).xyz;
        let n_model_len = length(n_model_raw);
        if n_model_len < 1e-8 {
            continue;
        }
        let n_model = n_model_raw / n_model_len;
        // Adjust distance: pick any point on the world plane, transform to model space.
        let p_on_plane_world = n_world * (-d_world);
        let p_on_plane_model = (volume.inv_model * vec4<f32>(p_on_plane_world, 1.0)).xyz;
        let d_model = -dot(n_model, p_on_plane_model);

        let clipped = clip_ray_plane(eye_model, ray_dir, n_model, d_model, t_near, t_far);
        t_near = clipped.x;
        t_far = clipped.y;
    }

    // Clip against active clip volumes (box and sphere), working in world space.
    // Reconstruct the world-space ray once, then loop over all active entries.
    if clip_volume.count > 0u {
        let eye_world = camera.eye_pos;
        // Transform model-space ray_dir back to world space.
        let model3 = mat3x3<f32>(volume.model[0].xyz, volume.model[1].xyz, volume.model[2].xyz);
        let dir_world_raw = model3 * ray_dir;
        let dir_world_len = length(dir_world_raw);
        // t_world = t_model * dir_world_len (scale factor between parameter spaces).
        // We clip in world space then convert t values back.
        if dir_world_len > 1e-8 {
            let dir_world = dir_world_raw / dir_world_len;
            let scale = dir_world_len;  // t_model = t_world / scale
            var tw_near = t_near * scale;
            var tw_far  = t_far  * scale;
            for (var i = 0u; i < clip_volume.count; i = i + 1u) {
                let e = clip_volume.volumes[i];
                if e.volume_type == 2u {
                    let r = clip_ray_box(eye_world, dir_world, e, tw_near, tw_far);
                    tw_near = r.x;
                    tw_far  = r.y;
                } else if e.volume_type == 3u {
                    let r = clip_ray_sphere(eye_world, dir_world, e, tw_near, tw_far);
                    tw_near = r.x;
                    tw_far  = r.y;
                } else if e.volume_type == 4u {
                    let r = clip_ray_cylinder(eye_world, dir_world, e, tw_near, tw_far);
                    tw_near = r.x;
                    tw_far  = r.y;
                }
            }
            t_near = tw_near / scale;
            t_far  = tw_far  / scale;
        }
    }

    if t_near >= t_far {
        discard;
    }

    // Ray march parameters.
    let step = volume.step_size;
    let max_steps = 512u;
    let scalar_range = volume.scalar_max - volume.scalar_min;
    let inv_scalar_range = select(1.0 / scalar_range, 1.0, abs(scalar_range) < 1e-10);

    var colour_accum = vec3<f32>(0.0);
    var alpha_accum = 0.0;
    var t = t_near;

    for (var i = 0u; i < max_steps; i = i + 1u) {
        if t >= t_far {
            break;
        }

        let sample_pos = eye_model + ray_dir * t;

        // Sample 3D texture (R32Float). UVW = sample_pos (already in [0,1]^3).
        let raw_scalar = textureSampleLevel(volume_tex, volume_nearest_sampler, sample_pos, 0.0).r;

        // Detect NaN using the IEEE 754 property: NaN != NaN.
        let is_nan = raw_scalar != raw_scalar;
        if is_nan {
            if volume.use_nan_colour != 0u {
                // Composite NaN voxel with the configured NaN colour.
                let nan_alpha = volume.nan_colour.a * volume.opacity_scale;
                let w = (1.0 - alpha_accum) * nan_alpha;
                colour_accum = colour_accum + volume.nan_colour.rgb * w;
                alpha_accum = alpha_accum + w;
                if alpha_accum > 0.99 {
                    break;
                }
            }
            // If use_nan_colour == 0, skip NaN samples (discard).
            t = t + step;
            continue;
        }

        // Threshold: skip samples outside [threshold_min, threshold_max].
        if raw_scalar < volume.threshold_min || raw_scalar > volume.threshold_max {
            t = t + step;
            continue;
        }

        // Normalize to [0,1].
        let normalized = clamp((raw_scalar - volume.scalar_min) * inv_scalar_range, 0.0, 1.0);

        // Sample colour LUT (256x1 Rgba8Unorm texture, sample at u=normalized).
        let lut_coord = vec2<f32>(normalized, 0.5);
        let sample_colour = textureSampleLevel(colour_lut, lut_sampler, lut_coord, 0.0).rgb;

        // Sample opacity LUT (256x1 R8Unorm texture).
        let sample_opacity = textureSampleLevel(opacity_lut, lut_sampler, lut_coord, 0.0).r
                             * volume.opacity_scale;

        var final_colour = sample_colour;

        // Optional gradient-based Phong shading.
        if volume.enable_shading != 0u {
            // Central-difference gradient (in texture space).
            let texel = 1.0 / vec3<f32>(textureDimensions(volume_tex));
            let gx = textureSampleLevel(volume_tex, volume_nearest_sampler, sample_pos + vec3<f32>(texel.x, 0.0, 0.0), 0.0).r
                   - textureSampleLevel(volume_tex, volume_nearest_sampler, sample_pos - vec3<f32>(texel.x, 0.0, 0.0), 0.0).r;
            let gy = textureSampleLevel(volume_tex, volume_nearest_sampler, sample_pos + vec3<f32>(0.0, texel.y, 0.0), 0.0).r
                   - textureSampleLevel(volume_tex, volume_nearest_sampler, sample_pos - vec3<f32>(0.0, texel.y, 0.0), 0.0).r;
            let gz = textureSampleLevel(volume_tex, volume_nearest_sampler, sample_pos + vec3<f32>(0.0, 0.0, texel.z), 0.0).r
                   - textureSampleLevel(volume_tex, volume_nearest_sampler, sample_pos - vec3<f32>(0.0, 0.0, texel.z), 0.0).r;
            let grad = vec3<f32>(gx, gy, gz);
            let grad_len = length(grad);
            if grad_len > 1e-6 {
                let normal = grad / grad_len;
                // Transform sample position to world space for lighting.
                let world_sample = (volume.model * vec4<f32>(sample_pos, 1.0)).xyz;
                let light_dir = normalize(camera.eye_pos - world_sample);
                // Transform normal to world space (using inverse-transpose = transpose(inv_model)).
                let world_normal_raw = (transpose(volume.inv_model) * vec4<f32>(normal, 0.0)).xyz;
                let world_normal = normalize(world_normal_raw);
                let ndotl = max(abs(dot(world_normal, light_dir)), 0.0);
                let ambient = 0.5;
                let diffuse = 0.6;
                final_colour = sample_colour * (ambient + diffuse * ndotl);
            }
        }

        // Front-to-back compositing.
        let w = (1.0 - alpha_accum) * sample_opacity;
        colour_accum = colour_accum + final_colour * w;
        alpha_accum = alpha_accum + w;

        // Early termination.
        if alpha_accum > 0.99 {
            break;
        }

        t = t + step;
    }

    if alpha_accum < 0.001 {
        discard;
    }

    return vec4<f32>(colour_accum, alpha_accum);
}
