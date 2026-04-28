// GPU implicit surface shader (Phase 16).
//
// Renders descriptor-driven SDFs via ray-marching in the fragment stage.
// Outputs both colour and frag_depth so the result depth-composites against
// scene geometry already in the depth buffer.
//
// Group 0 : camera_bgl (shared with all other scene pipelines).
//   binding 0 : CameraUniform  (view_proj, eye_pos, forward, inv_view_proj, view)
//   binding 3 : LightsUniform  (first directional light used for shading)
// Group 1 : implicit-specific.
//   binding 0 : ImplicitUniform (primitive array + march parameters)
//
// Vertex stage : full-screen quad (two triangles, no vertex buffer).
// Fragment stage : reconstruct world-space ray -> sphere-march -> shade -> write depth.

// ---------------------------------------------------------------------------
// Group 0 : camera + lights
// ---------------------------------------------------------------------------

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct SingleLight {
    light_view_proj: mat4x4<f32>,
    pos_or_dir:      vec3<f32>,
    light_type:      u32,
    color:           vec3<f32>,
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
    _pad:                 u32,
    sky_color:            vec3<f32>,
    hemisphere_intensity: f32,
    ground_color:         vec3<f32>,
    _pad2:                f32,
    lights:               array<SingleLight, 8>,
    ibl_enabled:          u32,
    ibl_intensity:        f32,
    ibl_rotation:         f32,
    show_skybox:          u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(3) var<uniform> lights: Lights;

// ---------------------------------------------------------------------------
// Group 1 : implicit-specific uniform
// ---------------------------------------------------------------------------

// Matches ImplicitPrimitive in src/resources/implicit.rs (64 bytes each).
struct ImplicitPrimitive {
    kind:    u32,        // 1=sphere 2=box 3=plane 4=capsule
    blend:   f32,        // smooth-min blend radius (0 = hard union)
    _pad0:   f32,
    _pad1:   f32,
    params0: vec4<f32>,  // sphere: (cx,cy,cz,r)  box: (cx,cy,cz,_)  plane: (nx,ny,nz,d)  capsule: (ax,ay,az,r)
    params1: vec4<f32>,  // box: (hx,hy,hz,_)     capsule: (bx,by,bz,_)
    color:   vec4<f32>,  // linear RGBA per primitive
};

// Matches ImplicitUniformRaw in src/resources/implicit.rs.
struct ImplicitUniform {
    num_primitives: u32,
    blend_mode:     u32,   // 0=union  1=smooth_union  2=intersection
    max_steps:      u32,
    _pad0:          u32,
    step_scale:     f32,
    hit_threshold:  f32,
    max_distance:   f32,
    _pad1:          f32,
    primitives:     array<ImplicitPrimitive, 16>,
};

@group(1) @binding(0) var<uniform> u: ImplicitUniform;

// ---------------------------------------------------------------------------
// Vertex stage : full-screen quad (identical to screen_image.wgsl)
// ---------------------------------------------------------------------------

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
};

// Six vertices forming two triangles that cover NDC [-1,1]^2.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var x: f32;
    var y: f32;
    switch vi {
        case 0u: { x = -1.0; y = -1.0; }
        case 1u: { x =  1.0; y = -1.0; }
        case 2u: { x = -1.0; y =  1.0; }
        case 3u: { x = -1.0; y =  1.0; }
        case 4u: { x =  1.0; y = -1.0; }
        default: { x =  1.0; y =  1.0; }
    }
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.ndc_xy   = vec2<f32>(x, y);
    return out;
}

// ---------------------------------------------------------------------------
// SDF primitives
// ---------------------------------------------------------------------------

fn sdf_sphere(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
    return length(p - prim.params0.xyz) - prim.params0.w;
}

fn sdf_box(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
    let q = abs(p - prim.params0.xyz) - prim.params1.xyz;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_plane(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
    // plane normal in params0.xyz, offset in params0.w
    return dot(p, normalize(prim.params0.xyz)) + prim.params0.w;
}

fn sdf_capsule(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
    // segment a=params0.xyz, b=params1.xyz, radius=params0.w
    let a  = prim.params0.xyz;
    let b  = prim.params1.xyz;
    let r  = prim.params0.w;
    let pa = p - a;
    let ba = b - a;
    let h  = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

fn eval_primitive(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
    switch prim.kind {
        case 1u: { return sdf_sphere(p, prim); }
        case 2u: { return sdf_box(p, prim); }
        case 3u: { return sdf_plane(p, prim); }
        case 4u: { return sdf_capsule(p, prim); }
        default: { return u.max_distance; }
    }
}

// Inigo Quilez polynomial smooth-min.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return a * h + b * (1.0 - h) - k * h * (1.0 - h);
}

// Evaluate the combined SDF for all primitives.
fn scene_sdf(p: vec3<f32>) -> f32 {
    var d = u.max_distance;
    for (var i: u32 = 0u; i < u.num_primitives; i++) {
        let prim = u.primitives[i];
        let pd = eval_primitive(p, prim);
        if u.blend_mode == 1u {
            // SmoothUnion: smooth-min against accumulated distance.
            let k = select(1e-5, prim.blend, prim.blend > 0.0);
            d = smin(d, pd, k);
        } else if u.blend_mode == 2u {
            // Intersection: max of all.
            if i == 0u { d = pd; } else { d = max(d, pd); }
        } else {
            // Union (default): min.
            d = min(d, pd);
        }
    }
    return d;
}

// Proximity-weighted color blend across all primitives (matches blob_color logic).
fn scene_color(p: vec3<f32>) -> vec4<f32> {
    var total_w = 0.0;
    var col = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < u.num_primitives; i++) {
        let prim = u.primitives[i];
        let d = eval_primitive(p, prim);
        let blend = select(0.9, prim.blend, prim.blend > 0.0);
        let w = max(-d + blend, 0.0);
        col   += prim.color * w;
        total_w += w;
    }
    if total_w < 1e-5 {
        return u.primitives[0].color;
    }
    return col / total_w;
}

// Central-difference normal (6 SDF evaluations).
fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 1e-3;
    return normalize(vec3<f32>(
        scene_sdf(p + vec3<f32>(e, 0.0, 0.0)) - scene_sdf(p - vec3<f32>(e, 0.0, 0.0)),
        scene_sdf(p + vec3<f32>(0.0, e, 0.0)) - scene_sdf(p - vec3<f32>(0.0, e, 0.0)),
        scene_sdf(p + vec3<f32>(0.0, 0.0, e)) - scene_sdf(p - vec3<f32>(0.0, 0.0, e)),
    ));
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

struct FragOutput {
    @location(0)         color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    // Reconstruct world-space ray from NDC position.
    // Unproject two clip-space points at Z=0 (near) and Z=1 (far).
    let near_clip = vec4<f32>(in.ndc_xy, 0.0, 1.0);
    let far_clip  = vec4<f32>(in.ndc_xy, 1.0, 1.0);

    let near_world_h = camera.inv_view_proj * near_clip;
    let far_world_h  = camera.inv_view_proj * far_clip;

    let near_world = near_world_h.xyz / near_world_h.w;
    let far_world  = far_world_h.xyz  / far_world_h.w;

    let ray_origin = near_world;
    let ray_dir    = normalize(far_world - near_world);

    // Sphere-march.
    var t       = 0.0;
    var hit     = false;
    var hit_pos = ray_origin;

    for (var step: u32 = 0u; step < u.max_steps; step++) {
        hit_pos  = ray_origin + ray_dir * t;
        let d    = scene_sdf(hit_pos);
        if d < u.hit_threshold {
            hit = true;
            break;
        }
        t += d * u.step_scale;
        if t > u.max_distance {
            break;
        }
    }

    if !hit {
        discard;
    }

    // Normal and shading.
    let normal     = estimate_normal(hit_pos);
    let base_color = scene_color(hit_pos);

    // First directional light (or hardcoded fallback).
    var light_dir: vec3<f32>;
    var light_rgb: vec3<f32>;
    if lights.count > 0u && lights.lights[0].light_type == 0u {
        // Directional: pos_or_dir is the light direction (pointing away from surface).
        light_dir = normalize(-lights.lights[0].pos_or_dir);
        light_rgb = lights.lights[0].color * lights.lights[0].intensity;
    } else {
        light_dir = normalize(vec3<f32>(0.577, 0.577, 0.577));
        light_rgb = vec3<f32>(1.0);
    }

    const AMBIENT: f32 = 0.25;
    let diffuse   = max(dot(normal, light_dir), 0.0);
    let shade_fac = AMBIENT + (1.0 - AMBIENT) * diffuse;
    let shaded    = vec4<f32>(base_color.rgb * light_rgb * shade_fac, base_color.a);

    // Compute NDC depth of the hit point so the hardware depth test fires correctly.
    let clip_hit = camera.view_proj * vec4<f32>(hit_pos, 1.0);
    let ndc_depth = clip_hit.z / clip_hit.w;

    var out: FragOutput;
    out.color = shaded;
    out.depth = ndc_depth;
    return out;
}
