// ---------------------------------------------------------------------------
// Path tracer megakernel.
//
// One invocation per pixel traces a full path in registers. Compute only, no
// hardware ray query. World-space triangle BVH; next-event estimation to
// analytic lights plus a hemisphere sky on miss; cosine-weighted diffuse and
// GGX-sampled specular lobes; Russian-roulette termination; samples accumulate
// into a storage buffer.
//
// The direct BRDF is the shared helpers/brdf.wgsl, the same one the mesh
// shaders use, so the tracer and the rasteriser produce the same shading.
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265;
const EPS: f32 = 1.0e-4;
const T_MAX: f32 = 1.0e30;

struct Node {
    aabb_min: vec3<f32>,
    left_first: u32,   // interior (count==0): index of right child, left = self+1.
    aabb_max: vec3<f32>,
    count: u32,        // >0: leaf with `count` triangles from left_first.
}

struct Triangle {
    p0: vec3<f32>, mat: u32,
    p1: vec3<f32>, _p1: u32,
    p2: vec3<f32>, _p2: u32,
    n0: vec3<f32>, _n0: u32,
    n1: vec3<f32>, _n1: u32,
    n2: vec3<f32>, _n2: u32,
}

struct Material {
    base: vec3<f32>,
    metallic: f32,
    emissive: vec3<f32>,
    roughness: f32,
}

// data.xyz = direction (toward light, normalised) for kind 0, or position for
// kind 1; data.w = kind (0 = directional, 1 = point). colour.rgb = radiance;
// colour.w = point-light range (0 = no falloff).
struct Light {
    data: vec4<f32>,
    colour: vec4<f32>,
}

struct Frame {
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sky_top: vec4<f32>,
    sky_bottom: vec4<f32>,
    dims: vec4<u32>,    // width, height, num_lights, max_bounces
    params: vec4<u32>,  // samples_per_frame, sample_base, frame_seed, _
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> tris: array<Triangle>;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read> lights: array<Light>;
@group(0) @binding(5) var<storage, read_write> accum: array<vec4<f32>>;

// #include "helpers/brdf.wgsl"

// ----- RNG (PCG hash) -----

fn pcg(state: ptr<function, u32>) -> u32 {
    let s = *state;
    *state = s * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rand(state: ptr<function, u32>) -> f32 {
    return f32(pcg(state)) * (1.0 / 4294967296.0);
}

fn init_rng(pixel: u32, seed: u32) -> u32 {
    var r = pixel * 9781u + seed * 2654435761u + 1u;
    // One hash round to decorrelate neighbouring pixels/seeds.
    r = pcg(&r);
    return r;
}

// ----- Intersection -----

struct Hit {
    t: f32,
    pos: vec3<f32>,
    ns: vec3<f32>,   // shading normal (interpolated, faced toward the ray)
    ng: vec3<f32>,   // geometric normal (faced toward the ray)
    mat: u32,
    hit: bool,
}

// Slab test; returns the near entry distance or T_MAX on miss. Clamps to 0 so a
// ray starting inside the box still enters.
fn ray_box(o: vec3<f32>, inv_d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, tmax: f32) -> f32 {
    let t0 = (bmin - o) * inv_d;
    let t1 = (bmax - o) * inv_d;
    let tsmall = min(t0, t1);
    let tbig = max(t0, t1);
    let tnear = max(max(tsmall.x, tsmall.y), tsmall.z);
    let tfar = min(min(tbig.x, tbig.y), tbig.z);
    if tfar >= max(tnear, 0.0) && tnear < tmax {
        return max(tnear, 0.0);
    }
    return T_MAX;
}

// Moller-Trumbore. Writes t and barycentrics on hit.
fn ray_triangle(o: vec3<f32>, d: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>,
    t_out: ptr<function, f32>, uv_out: ptr<function, vec2<f32>>) -> bool {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let pv = cross(d, e2);
    let det = dot(e1, pv);
    if abs(det) < 1.0e-12 { return false; }
    let inv_det = 1.0 / det;
    let tv = o - p0;
    let u = dot(tv, pv) * inv_det;
    if u < 0.0 || u > 1.0 { return false; }
    let qv = cross(tv, e1);
    let v = dot(d, qv) * inv_det;
    if v < 0.0 || u + v > 1.0 { return false; }
    let t = dot(e2, qv) * inv_det;
    if t <= EPS { return false; }
    *t_out = t;
    *uv_out = vec2<f32>(u, v);
    return true;
}

fn closest_hit(o: vec3<f32>, d: vec3<f32>) -> Hit {
    var h: Hit;
    h.hit = false;
    h.t = T_MAX;
    let inv_d = 1.0 / d;

    var best_t = T_MAX;
    var best_tri = 0u;
    var best_uv = vec2<f32>(0.0);

    var stack: array<u32, 32>;
    var sp = 0;
    stack[0] = 0u;
    sp = 1;

    loop {
        if sp <= 0 { break; }
        sp = sp - 1;
        let ni = stack[sp];
        let node = nodes[ni];

        if ray_box(o, inv_d, node.aabb_min, node.aabb_max, best_t) >= best_t {
            continue;
        }

        if node.count > 0u {
            // Leaf.
            for (var i = 0u; i < node.count; i = i + 1u) {
                let ti = node.left_first + i;
                let tri = tris[ti];
                var tt: f32;
                var uv: vec2<f32>;
                if ray_triangle(o, d, tri.p0, tri.p1, tri.p2, &tt, &uv) {
                    if tt < best_t {
                        best_t = tt;
                        best_tri = ti;
                        best_uv = uv;
                    }
                }
            }
        } else {
            // Interior: push both children, nearer one last so it pops first.
            let left = ni + 1u;
            let right = node.left_first;
            let dl = ray_box(o, inv_d, nodes[left].aabb_min, nodes[left].aabb_max, best_t);
            let dr = ray_box(o, inv_d, nodes[right].aabb_min, nodes[right].aabb_max, best_t);
            if dl <= dr {
                if dr < best_t && sp < 31 { stack[sp] = right; sp = sp + 1; }
                if dl < best_t && sp < 31 { stack[sp] = left; sp = sp + 1; }
            } else {
                if dl < best_t && sp < 31 { stack[sp] = left; sp = sp + 1; }
                if dr < best_t && sp < 31 { stack[sp] = right; sp = sp + 1; }
            }
        }
    }

    if best_t < T_MAX {
        let tri = tris[best_tri];
        let w = 1.0 - best_uv.x - best_uv.y;
        let ns = normalize(tri.n0 * w + tri.n1 * best_uv.x + tri.n2 * best_uv.y);
        let ng = normalize(cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
        h.hit = true;
        h.t = best_t;
        h.pos = o + d * best_t;
        // Face both normals toward the incoming ray.
        let vd = -d;
        h.ng = select(-ng, ng, dot(ng, vd) > 0.0);
        h.ns = select(-ns, ns, dot(h.ng, ns) > 0.0);
        h.mat = tri.mat;
    }
    return h;
}

fn any_hit(o: vec3<f32>, d: vec3<f32>, max_t: f32) -> bool {
    let inv_d = 1.0 / d;
    var stack: array<u32, 32>;
    var sp = 0;
    stack[0] = 0u;
    sp = 1;
    loop {
        if sp <= 0 { break; }
        sp = sp - 1;
        let node = nodes[stack[sp]];
        if ray_box(o, inv_d, node.aabb_min, node.aabb_max, max_t) >= max_t {
            continue;
        }
        if node.count > 0u {
            for (var i = 0u; i < node.count; i = i + 1u) {
                let tri = tris[node.left_first + i];
                var tt: f32;
                var uv: vec2<f32>;
                if ray_triangle(o, d, tri.p0, tri.p1, tri.p2, &tt, &uv) {
                    if tt < max_t { return true; }
                }
            }
        } else {
            let ni = stack[sp];
            if sp < 31 { stack[sp] = ni + 1u; sp = sp + 1; }
            if sp < 31 { stack[sp] = node.left_first; sp = sp + 1; }
        }
    }
    return false;
}

// ----- Environment -----

fn sky(dir: vec3<f32>) -> vec3<f32> {
    // Z-up hemisphere gradient.
    let t = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);
    return mix(frame.sky_bottom.rgb, frame.sky_top.rgb, t);
}

// ----- Sampling -----

fn build_onb(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn cosine_sample(n: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand(rng);
    let r2 = rand(rng);
    let phi = 2.0 * PI * r1;
    let r = sqrt(r2);
    let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - r2)));
    return normalize(build_onb(n) * local);
}

// GGX NDF half-vector sample.
fn ggx_sample_h(n: vec3<f32>, roughness: f32, rng: ptr<function, u32>) -> vec3<f32> {
    let a = roughness * roughness;
    let r1 = rand(rng);
    let r2 = rand(rng);
    let phi = 2.0 * PI * r1;
    let cos_t = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let local = vec3<f32>(sin_t * cos(phi), sin_t * sin(phi), cos_t);
    return normalize(build_onb(n) * local);
}

// ----- Direct lighting (NEE to analytic delta lights) -----

fn direct_light(pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, m: Material, f0: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let count = frame.dims.z;
    for (var i = 0u; i < count; i = i + 1u) {
        let lt = lights[i];
        var l: vec3<f32>;
        var radiance = lt.colour.rgb;
        var max_t = T_MAX;
        if lt.data.w < 0.5 {
            // Directional.
            l = normalize(lt.data.xyz);
        } else {
            // Point.
            let to = lt.data.xyz - pos;
            let dist = length(to);
            l = to / dist;
            let range = lt.colour.w;
            var atten = 1.0 / max(dist * dist, 1.0e-4);
            if range > 0.0 {
                let f = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
                atten = atten * f * f;
            }
            radiance = radiance * atten;
            max_t = dist - EPS;
        }
        if dot(n, l) <= 0.0 { continue; }
        if any_hit(pos + n * EPS, l, max_t) { continue; }
        sum = sum + pbr_light_contrib(n, v, l, radiance, m.base, m.metallic, m.roughness, f0);
    }
    return sum;
}

// ----- Path integrator -----

fn trace_path(ro_in: vec3<f32>, rd_in: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    var ro = ro_in;
    var rd = rd_in;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    let max_bounces = frame.dims.w;

    for (var bounce = 0u; bounce < max_bounces; bounce = bounce + 1u) {
        let h = closest_hit(ro, rd);
        if !h.hit {
            radiance = radiance + throughput * sky(rd);
            break;
        }

        let m = materials[h.mat];
        // Emissive surfaces are found by BSDF sampling only (not NEE'd), so add
        // once with no double counting.
        radiance = radiance + throughput * m.emissive;

        let n = h.ns;
        let v = -rd;
        let f0 = mix(vec3<f32>(0.04), m.base, m.metallic);
        let rough = clamp(m.roughness, 0.04, 1.0);

        // Next-event estimation to analytic lights.
        radiance = radiance + throughput * direct_light(h.pos, n, v, m, f0);

        // Sample a new direction: pick specular vs diffuse lobe.
        let ndotv = max(dot(n, v), 1.0e-3);
        let fr = F_Schlick(ndotv, f0);
        var p_spec = clamp(max(fr.r, max(fr.g, fr.b)), 0.1, 0.9);
        p_spec = mix(p_spec, 1.0, m.metallic * 0.5);

        if rand(rng) < p_spec {
            // Specular (GGX) lobe.
            let hvec = ggx_sample_h(n, rough, rng);
            let l = reflect(-v, hvec);
            let ndotl = dot(n, l);
            if ndotl <= 0.0 { break; }
            let ndoth = max(dot(n, hvec), 1.0e-4);
            let vdoth = max(dot(v, hvec), 1.0e-4);
            let g = G_Smith(ndotv, ndotl, rough);
            let f = F_Schlick(vdoth, f0);
            // weight = brdf * ndotl / pdf = G F VdotH / (NdotV NdotH), then / p_spec.
            let weight = f * (g * vdoth / (ndotv * ndoth));
            throughput = throughput * weight / p_spec;
            ro = h.pos + n * EPS;
            rd = l;
        } else {
            // Diffuse (cosine) lobe. bsdf/pdf reduces to the diffuse albedo.
            let albedo = m.base * (1.0 - m.metallic);
            throughput = throughput * albedo / (1.0 - p_spec);
            ro = h.pos + n * EPS;
            rd = cosine_sample(n, rng);
        }

        // Russian roulette.
        if bounce >= 3u {
            let q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
            if rand(rng) > q { break; }
            throughput = throughput / q;
        }
    }
    return radiance;
}

fn camera_ray(px: vec2<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let w = f32(frame.dims.x);
    let h = f32(frame.dims.y);
    let jitter = vec2<f32>(rand(rng), rand(rng));
    let uv = (px + jitter) / vec2<f32>(w, h);
    var ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    let far = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far_point = far.xyz / far.w;
    return normalize(far_point - frame.cam_pos.xyz);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = frame.dims.x;
    let h = frame.dims.y;
    if gid.x >= w || gid.y >= h { return; }
    let pidx = gid.y * w + gid.x;

    let spp = frame.params.x;
    let sample_base = frame.params.y;
    var rng = init_rng(pidx, frame.params.z);

    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < spp; s = s + 1u) {
        let rd = camera_ray(vec2<f32>(f32(gid.x), f32(gid.y)), &rng);
        let c = trace_path(frame.cam_pos.xyz, rd, &rng);
        if all(c == c) {  // reject NaN
            sum = sum + c;
        }
    }

    let mean_new = sum / f32(spp);
    var result = mean_new;
    if sample_base > 0u {
        let prev = accum[pidx].rgb;
        let total = f32(sample_base) + f32(spp);
        result = (prev * f32(sample_base) + sum) / total;
    }
    accum[pidx] = vec4<f32>(result, 1.0);
}
