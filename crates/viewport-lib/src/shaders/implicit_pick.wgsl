// GPU implicit surface object-pick shader.
//
// Mirrors implicit.wgsl: a full-screen quad reconstructs a world-space ray per
// pixel and sphere-marches the combined SDF. On a hit it writes the item's pick
// id and the hit point's depth into the three pick targets; a ray that finds no
// isosurface is discarded, so the pick selects the actual rendered surface
// rather than any bounding proxy. Object-level: the primitive channel is 0.
//
// Group 0: camera (binding 0), the full scene camera layout so `inv_view_proj`
//          is available for the ray reconstruction. Lighting is not needed.
// Group 1: the implicit render uniform (binding 0), reused unchanged.
// Group 2: per-item pick id (binding 0).

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

// Matches ImplicitPrimitive in implicit.wgsl / src/resources/implicit.rs.
struct ImplicitPrimitive {
    kind:    u32,
    blend:   f32,
    _pad0:   f32,
    _pad1:   f32,
    params0: vec4<f32>,
    params1: vec4<f32>,
    colour:  vec4<f32>,
};

struct ImplicitUniform {
    num_primitives: u32,
    blend_mode:     u32,
    max_steps:      u32,
    unlit:          u32,
    step_scale:     f32,
    hit_threshold:  f32,
    max_distance:   f32,
    opacity:        f32,
    primitives:     array<ImplicitPrimitive, 16>,
};

@group(1) @binding(0) var<uniform> u: ImplicitUniform;

@group(2) @binding(0) var<uniform> pick_id: vec4<u32>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
};

// Full-screen quad, identical to implicit.wgsl vs_main.
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

fn sdf_sphere(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
    return length(p - prim.params0.xyz) - prim.params0.w;
}

fn sdf_box(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
    let q = abs(p - prim.params0.xyz) - prim.params1.xyz;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_plane(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
    return dot(p, normalize(prim.params0.xyz)) + prim.params0.w;
}

fn sdf_capsule(p: vec3<f32>, prim: ImplicitPrimitive) -> f32 {
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

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return a * h + b * (1.0 - h) - k * h * (1.0 - h);
}

fn scene_sdf(p: vec3<f32>) -> f32 {
    var d = u.max_distance;
    for (var i: u32 = 0u; i < u.num_primitives; i++) {
        let prim = u.primitives[i];
        let pd = eval_primitive(p, prim);
        if u.blend_mode == 1u {
            let k = select(1e-5, prim.blend, prim.blend > 0.0);
            d = smin(d, pd, k);
        } else if u.blend_mode == 2u {
            if i == 0u { d = pd; } else { d = max(d, pd); }
        } else {
            d = min(d, pd);
        }
    }
    return d;
}

struct PickOut {
    @location(0) object_id: u32,
    @location(1) primitive_id: u32,
    @location(2) depth: f32,
    @builtin(frag_depth) frag_depth: f32,
};

@fragment
fn fs_pick(in: VertexOutput) -> PickOut {
    // Reconstruct the world-space ray, matching implicit.wgsl fs_main.
    let near_clip = vec4<f32>(in.ndc_xy, 0.0, 1.0);
    let far_clip  = vec4<f32>(in.ndc_xy, 1.0, 1.0);
    let near_world_h = camera.inv_view_proj * near_clip;
    let far_world_h  = camera.inv_view_proj * far_clip;
    let near_world = near_world_h.xyz / near_world_h.w;
    let far_world  = far_world_h.xyz  / far_world_h.w;
    let ray_origin = near_world;
    let ray_dir    = normalize(far_world - near_world);

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

    let clip = camera.view_proj * vec4<f32>(hit_pos, 1.0);
    let ndc_z = clip.z / clip.w;
    var out: PickOut;
    out.object_id = pick_id.x;
    out.primitive_id = 0u;
    out.depth = ndc_z;
    out.frag_depth = ndc_z;
    return out;
}
