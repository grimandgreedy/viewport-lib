// implicit_outline_mask.wgsl: renders selected GPU implicit surfaces into the R8 outline mask.
//
// Same ray-march as implicit.wgsl but the fragment stage outputs 1.0 (white) on
// hit and discards on miss. No lighting computation. No depth write.
//
// `fs_stamp` is the surface-mask stage: the same march, writing the hit's
// depth and no colour, so the stamp is depth-tested against the scene.
//
// Group 0 : the shared scene bindings, prefixed from SHARED_BINDINGS_WGSL.
//           Only camera.inv_view_proj and camera.view_proj are read.
// Group 1 : implicit-specific (binding 0: ImplicitUniform).

// ---------------------------------------------------------------------------
// Group 1 : implicit-specific uniform
// ---------------------------------------------------------------------------

struct ImplicitPrimitive {
    kind:    u32,
    blend:   f32,
    _pad0:   f32,
    _pad1:   f32,
    params0: vec4<f32>,
    params1: vec4<f32>,
    colour:   vec4<f32>,
};

struct ImplicitUniform {
    num_primitives: u32,
    blend_mode:     u32,
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
// Vertex stage : full-screen quad
// ---------------------------------------------------------------------------

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
};

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
// SDF helpers (identical to implicit.wgsl)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Fragment stages
// ---------------------------------------------------------------------------

struct MarchHit {
    pos: vec3<f32>,
    hit: bool,
}

// March the pixel's ray to the surface.
fn march(ndc_xy: vec2<f32>) -> MarchHit {
    let near_clip = vec4<f32>(ndc_xy, 0.0, 1.0);
    let far_clip  = vec4<f32>(ndc_xy, 1.0, 1.0);

    let near_world_h = camera.inv_view_proj * near_clip;
    let far_world_h  = camera.inv_view_proj * far_clip;

    let near_world = near_world_h.xyz / near_world_h.w;
    let far_world  = far_world_h.xyz  / far_world_h.w;

    let ray_origin = near_world;
    let ray_dir    = normalize(far_world - near_world);

    var t   = 0.0;
    var hit = false;
    var p   = ray_origin;

    for (var step: u32 = 0u; step < u.max_steps; step++) {
        p = ray_origin + ray_dir * t;
        let d = scene_sdf(p);
        if d < u.hit_threshold {
            hit = true;
            break;
        }
        t += d * u.step_scale;
        if t > u.max_distance {
            break;
        }
    }

    return MarchHit(p, hit);
}

// Outline mask: 1.0 on hit, discard on miss.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let m = march(in.ndc_xy);
    if !m.hit {
        discard;
    }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

// Surface mask: the hit's depth, nudged towards the camera so it passes the
// depth test against the copy the colour pass wrote.
@fragment
fn fs_stamp(in: VertexOutput) -> @builtin(frag_depth) f32 {
    let m = march(in.ndc_xy);
    if !m.hit {
        discard;
    }
    let clip = camera.view_proj * vec4<f32>(m.pos, 1.0);
    return max(clip.z / clip.w - 1e-6, 0.0);
}
