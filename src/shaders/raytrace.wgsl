// ---------------------------------------------------------------------------
// Path tracer megakernel.
//
// One invocation per pixel traces a full path in registers. The traversal
// (closest_hit / any_hit, marked with rt-traversal below) is a compute walk of
// a world-space triangle BVH by default; the raytrace-hardware backend swaps
// that region for a rayQuery kernel. Next-event estimation to
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
    transmission: f32,
    ior: f32,
    _pad0: f32,
    _pad1: f32,
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
    dims: vec4<u32>,      // width, height, num_lights, max_bounces
    params: vec4<u32>,    // samples_per_frame, sample_base, frame_seed, has_env
    env_dist: vec4<f32>,  // env-IS: width, height, integral, enabled
    emit: vec4<u32>,      // area-light NEE: count, offset into env_tables, _, _
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> tris: array<Triangle>;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read> lights: array<Light>;
@group(0) @binding(5) var<storage, read_write> accum: array<vec4<f32>>;
// First-hit guide buffers for the denoiser: primary-surface albedo (rgb, with
// w = 1 on a hit / 0 on a sky miss) and world normal (xyz, w unused).
@group(0) @binding(6) var<storage, read_write> gbuf_albedo: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> gbuf_normal: array<vec4<f32>>;
// Equirect HDR environment sampled on ray miss when frame.params.w != 0.
@group(0) @binding(8) var env_tex: texture_2d<f32>;
@group(0) @binding(9) var env_samp: sampler;
// Per-texel surfaces for the lightmap bake (bake_main only), packed as two
// contiguous halves of one buffer to leave a storage-buffer slot for the env
// tables: positions in texel_surf[0 .. w*h] (xyz, w = coverage; w <= 0 is an
// empty texel), then normals in texel_surf[w*h .. 2*w*h] (xyz). These replace the
// camera as the primary-ray source: one bake invocation shoots its hemisphere
// from the surface point behind its atlas texel.
@group(0) @binding(10) var<storage, read> texel_surf: array<vec4<f32>>;
// Dominant-direction accumulator for the bake: the running mean of the
// luminance-weighted incoming-light direction (xyz), coverage in w. A later
// encode stage normalises it into a dominant direction + directionality.
@group(0) @binding(12) var<storage, read_write> accum_dir: array<vec4<f32>>;
// Environment importance-sampling tables (camera path only), packed into one
// buffer to stay under the storage-buffer limit: the sin-weighted luminance
// func[0 .. w*h], then the per-row conditional CDFs (each width+1 long) at
// w*h .. w*h + h*(w+1), then the marginal CDF over rows (height+1) after that.
// Built on the CPU by EnvDistribution; sliced by env_func_at / env_cond_at /
// env_marg_at below.
@group(0) @binding(14) var<storage, read> env_tables: array<f32>;

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
    front: bool,     // true if the ray struck the outward (front) face
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

// closest_hit + any_hit walk the compute BVH below. The hardware backend
// (raytrace-hardware) replaces this whole region, up to the matching close
// marker, with a rayQuery kernel over an acceleration structure, keeping the
// same Hit layout so all shading downstream is untouched.
// <rt-traversal>
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
        // Face both normals toward the incoming ray, and record whether the ray
        // hit the outward (front) face : needed to orient refraction.
        let vd = -d;
        let front = dot(ng, vd) > 0.0;
        h.front = front;
        h.ng = select(-ng, ng, front);
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
// </rt-traversal>

// ----- Environment -----

// viewport-lib is Z-up: longitude around +Z, latitude with +Z at the top. Must
// match dir_to_equirect_uv in the IBL shaders so the tracer and rasteriser read
// the same environment.
fn dir_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let phi = atan2(dir.y, dir.x);
    let theta = asin(clamp(dir.z, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), 0.5 - theta / PI);
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
    if frame.params.w != 0u {
        // Equirect environment (image-based lighting).
        let uv = dir_to_equirect_uv(dir);
        return textureSampleLevel(env_tex, env_samp, uv, 0.0).rgb;
    }
    // Z-up hemisphere gradient.
    let t = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);
    return mix(frame.sky_bottom.rgb, frame.sky_top.rgb, t);
}

// ----- Environment importance sampling -----
//
// The tables in env_func / env_cond_cdf / env_marg_cdf let the integrator send
// shadow rays toward bright parts of the environment (next-event estimation)
// instead of finding them only by chance through BSDF sampling, which is what
// makes pure-IBL scenes converge slowly. This mirrors the CPU EnvDistribution
// (raytrace/env_dist.rs): the same marginal/conditional inverse-CDF sampling and
// the same func/integral density, with the equirect solid-angle Jacobian applied
// here so the pdf is per unit solid angle.

struct EnvSample {
    dir: vec3<f32>,
    pdf: f32,   // solid-angle pdf; 0 when the environment is not importance-sampled
}

// True when an importance-sampling distribution is available (an equirect
// environment was set). The hemisphere sky has none, so env NEE is skipped.
fn env_is_enabled() -> bool {
    return frame.env_dist.w != 0.0;
}

// Slice the packed env_tables buffer: func, then conditional CDFs, then marginal.
fn env_func_at(i: u32) -> f32 {
    return env_tables[i];
}
fn env_cond_at(i: u32) -> f32 {
    let w = u32(frame.env_dist.x);
    let h = u32(frame.env_dist.y);
    return env_tables[w * h + i];
}
fn env_marg_at(i: u32) -> f32 {
    let w = u32(frame.env_dist.x);
    let h = u32(frame.env_dist.y);
    return env_tables[w * h + h * (w + 1u) + i];
}

// Direction for an equirect (u, v): longitude around +Z, +Z at v = 0. Inverse of
// dir_to_equirect_uv.
fn equirect_uv_to_dir(u: f32, v: f32) -> vec3<f32> {
    let phi = (u - 0.5) * 2.0 * PI;
    let theta = (0.5 - v) * PI;       // latitude; +pi/2 at the top (+Z)
    let ct = cos(theta);
    return vec3<f32>(ct * cos(phi), ct * sin(phi), sin(theta));
}

// Largest i in [0, n-2] with cdf[off + i] <= x. Binary search over a storage CDF
// with a leading 0 and trailing 1; matches EnvDistribution::find_interval.
fn cond_find(off: u32, n: u32, x: f32) -> u32 {
    var lo = 0u;
    var hi = n - 1u;   // candidate interval indices are [0, n-2]
    loop {
        if lo + 1u >= hi { break; }
        let mid = (lo + hi) / 2u;
        if env_cond_at(off + mid + 1u) <= x { lo = mid; } else { hi = mid; }
    }
    return min(lo, n - 2u);
}

fn marg_find(n: u32, x: f32) -> u32 {
    var lo = 0u;
    var hi = n - 1u;
    loop {
        if lo + 1u >= hi { break; }
        let mid = (lo + hi) / 2u;
        if env_marg_at(mid + 1u) <= x { lo = mid; } else { hi = mid; }
    }
    return min(lo, n - 2u);
}

// Solid-angle pdf of sampling direction `dir` from the environment distribution.
// Must agree with sample_env's returned pdf for MIS to be unbiased.
fn env_pdf(dir: vec3<f32>) -> f32 {
    if !env_is_enabled() { return 0.0; }
    let w = u32(frame.env_dist.x);
    let h = u32(frame.env_dist.y);
    let uv = dir_to_equirect_uv(dir);
    let x = min(u32(uv.x * f32(w)), w - 1u);
    let y = min(u32(uv.y * f32(h)), h - 1u);
    let sin_t = sin(PI * uv.y);
    if sin_t <= 0.0 { return 0.0; }
    let p_uv = env_func_at(y * w + x) / frame.env_dist.z;
    return p_uv / (2.0 * PI * PI * sin_t);
}

// Sample a direction toward the environment, returning it with its solid-angle
// pdf. Returns pdf 0 when there is no distribution.
fn sample_env(u1: f32, u2: f32) -> EnvSample {
    var out: EnvSample;
    out.dir = vec3<f32>(0.0, 0.0, 1.0);
    out.pdf = 0.0;
    if !env_is_enabled() { return out; }

    let w = u32(frame.env_dist.x);
    let h = u32(frame.env_dist.y);

    // Marginal: pick a row, then interpolate within its CDF bin for a continuous v.
    let y = marg_find(h + 1u, u2);
    let m0 = env_marg_at(y);
    let m1 = env_marg_at(y + 1u);
    let dm = max(m1 - m0, 1.0e-12);
    let v = (f32(y) + (u2 - m0) / dm) / f32(h);

    // Conditional: pick a column within that row.
    let base = y * (w + 1u);
    let x = cond_find(base, w + 1u, u1);
    let c0 = env_cond_at(base + x);
    let c1 = env_cond_at(base + x + 1u);
    let dc = max(c1 - c0, 1.0e-12);
    let u = (f32(x) + (u1 - c0) / dc) / f32(w);

    let sin_t = sin(PI * v);
    if sin_t <= 0.0 { return out; }
    let p_uv = env_func_at(y * w + x) / frame.env_dist.z;
    out.dir = equirect_uv_to_dir(u, v);
    out.pdf = p_uv / (2.0 * PI * PI * sin_t);
    return out;
}

// Power heuristic (beta = 2) for two sampling strategies with pdfs a and b.
fn power_heuristic(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    let b2 = b * b;
    let denom = a2 + b2;
    if denom <= 0.0 { return 0.0; }
    return a2 / denom;
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

// Unpolarised Fresnel reflectance at a dielectric interface. `cos_i` is the
// cosine of the incident angle against the microfacet normal; `n1`/`n2` are the
// refractive indices on the incident and transmitted sides. Returns 1.0 on
// total internal reflection.
fn fresnel_dielectric(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let ci = clamp(cos_i, 0.0, 1.0);
    let sin_t = (n1 / n2) * sqrt(max(0.0, 1.0 - ci * ci));
    if sin_t >= 1.0 {
        return 1.0;
    }
    let ct = sqrt(max(0.0, 1.0 - sin_t * sin_t));
    let rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    let rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
    return 0.5 * (rs * rs + rp * rp);
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

// Pdf (per solid angle) of the opaque-lobe sampler below for direction `l`,
// matching its diffuse/specular mixture. Needed to MIS environment NEE against
// BSDF sampling. Returns 0 for directions at or below the surface.
fn bsdf_pdf(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, metallic: f32, roughness: f32, f0: vec3<f32>) -> f32 {
    let ndotl = dot(n, l);
    if ndotl <= 0.0 { return 0.0; }
    let ndotv = max(dot(n, v), 1.0e-3);
    let fr = F_Schlick(ndotv, f0);
    var p_spec = clamp(max(fr.r, max(fr.g, fr.b)), 0.1, 0.9);
    p_spec = mix(p_spec, 1.0, metallic * 0.5);
    let rough = clamp(roughness, 0.04, 1.0);
    let pdf_diff = ndotl / PI;
    let hvec = normalize(v + l);
    let ndoth = max(dot(n, hvec), 1.0e-4);
    let vdoth = max(dot(v, hvec), 1.0e-4);
    let d = D_GGX(ndoth, rough);
    let pdf_spec = d * ndoth / (4.0 * vdoth);
    return p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff;
}

// One environment next-event-estimation sample at an opaque surface: sample a
// direction toward the environment, shadow-test it, and return the MIS-weighted
// (power heuristic) radiance for unit throughput. The matching BSDF-sampled
// environment hit is weighted on ray miss in trace_path, so the two strategies
// combine without double counting.
fn env_nee(pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, m: Material, f0: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    if !env_is_enabled() { return vec3<f32>(0.0); }
    let es = sample_env(rand(rng), rand(rng));
    if es.pdf <= 0.0 || dot(n, es.dir) <= 0.0 { return vec3<f32>(0.0); }
    if any_hit(pos + n * EPS, es.dir, T_MAX) { return vec3<f32>(0.0); }
    let env_rad = sky(es.dir);
    let bsdf_cos = pbr_light_contrib(n, v, es.dir, vec3<f32>(1.0), m.base, m.metallic, m.roughness, f0);
    let pdf_b = bsdf_pdf(n, v, es.dir, m.metallic, m.roughness, f0);
    let w = power_heuristic(es.pdf, pdf_b);
    return bsdf_cos * env_rad * (w / es.pdf);
}

// The triangle index of the `i`-th emissive triangle, read from the tail of the
// env_tables buffer (frame.emit.y is the offset; indices are stored as their u32
// bit pattern reinterpreted as f32, so bitcast recovers them exactly).
fn emitter_tri(i: u32) -> u32 {
    return bitcast<u32>(env_tables[frame.emit.y + i]);
}

// One area-light next-event-estimation sample at an opaque surface: pick an
// emissive triangle uniformly, sample a point on it, shadow-test the segment, and
// return the radiance for unit throughput. Emissive geometry is otherwise found
// only by chance BSDF bounces, so small bright emitters are noisy without this;
// the matching BSDF-sampled emitter hit is dropped in trace_path (counted only on
// delta/camera arrivals), so the two do not double count. Emitters are treated as
// two-sided. Returns zero when the scene has no emitters.
fn sample_emitters(pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, m: Material, f0: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let count = frame.emit.x;
    if count == 0u { return vec3<f32>(0.0); }

    let ei = min(u32(rand(rng) * f32(count)), count - 1u);
    let tri = tris[emitter_tri(ei)];

    // Uniform point on the triangle in barycentric coordinates.
    let su = sqrt(rand(rng));
    let b1 = 1.0 - su;
    let b2 = rand(rng) * su;
    let lp = tri.p0 * (1.0 - b1 - b2) + tri.p1 * b1 + tri.p2 * b2;

    let e1 = tri.p1 - tri.p0;
    let e2 = tri.p2 - tri.p0;
    let cr = cross(e1, e2);
    let area2 = length(cr);
    if area2 < 1.0e-12 { return vec3<f32>(0.0); }
    let ng_l = cr / area2;

    var to = lp - pos;
    let dist2 = dot(to, to);
    let dist = sqrt(dist2);
    if dist < 1.0e-4 { return vec3<f32>(0.0); }
    let wi = to / dist;
    if dot(n, wi) <= 0.0 { return vec3<f32>(0.0); }
    let cos_l = abs(dot(ng_l, wi));
    if cos_l < 1.0e-4 { return vec3<f32>(0.0); }
    if any_hit(pos + n * EPS, wi, dist - 1.0e-3) { return vec3<f32>(0.0); }

    let le = materials[tri.mat].emissive;
    // Estimator: (f * cos_surface * Le) / pdf_sa, with the solid-angle pdf of
    // uniform-over-emitters, uniform-on-triangle sampling
    //   pdf_sa = dist^2 / (count * area * cos_light),  area = area2 / 2.
    let bsdf_cos = pbr_light_contrib(n, v, wi, le, m.base, m.metallic, m.roughness, f0);
    let inv_pdf = f32(count) * (0.5 * area2) * cos_l / dist2;
    return bsdf_cos * inv_pdf;
}

// ----- Path integrator -----

fn trace_path(ro_in: vec3<f32>, rd_in: vec3<f32>, rng: ptr<function, u32>, mis0: f32) -> vec3<f32> {
    var ro = ro_in;
    var rd = rd_in;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    let max_bounces = frame.dims.w;
    // Pdf the last bounce sampled `rd` with, for MIS-weighting an environment hit
    // reached by BSDF sampling and for gating emitter emission that area-light NEE
    // already sampled. Negative marks a delta / non-NEE'd bounce (a pinhole camera
    // ray, or a transmission event): those take emission and the environment at
    // full weight. `mis0` sets the value for the primary ray: -1 for the camera
    // (a delta ray, so a directly-viewed emitter counts), 0 for the bake (its
    // primary is a cosine BSDF sample whose emitter hit is covered by the texel's
    // emitter NEE, so it must be gated).
    var mis_bsdf_pdf = mis0;

    for (var bounce = 0u; bounce < max_bounces; bounce = bounce + 1u) {
        let h = closest_hit(ro, rd);
        if !h.hit {
            // Environment reached by BSDF sampling. When env NEE also sampled it
            // (an opaque bounce with a distribution), weight this strategy by the
            // power heuristic; otherwise take it at full weight.
            var w = 1.0;
            if mis_bsdf_pdf >= 0.0 && env_is_enabled() {
                w = power_heuristic(mis_bsdf_pdf, env_pdf(rd));
            }
            radiance = radiance + throughput * sky(rd) * w;
            break;
        }

        let m = materials[h.mat];
        // Emissive surfaces: counted on a delta or camera arrival (mis_bsdf_pdf
        // < 0), where no area-light NEE sampled them. Opaque bounces reach
        // emitters through sample_emitters below instead, so counting the
        // BSDF-sampled hit here too would double count. With no emitters in the
        // scene m.emissive is zero and the gate falls back to counting always, so
        // scenes without emissive geometry are unchanged.
        if mis_bsdf_pdf < 0.0 || frame.emit.x == 0u {
            radiance = radiance + throughput * m.emissive;
        }

        let n = h.ns;
        let v = -rd;
        let f0 = mix(vec3<f32>(0.04), m.base, m.metallic);
        let rough = clamp(m.roughness, 0.04, 1.0);

        // Next-event estimation to analytic lights, scaled down by the
        // transmission fraction so clear glass is not diffusely lit.
        let opacity = 1.0 - select(0.0, m.transmission, m.metallic < 0.5);
        radiance = radiance + throughput * direct_light(h.pos, n, v, m, f0) * opacity;
        // Environment NEE (importance-sampled), same opaque-only scaling.
        radiance = radiance + throughput * env_nee(h.pos, n, v, m, f0, rng) * opacity;
        // Area-light NEE toward emissive geometry, same opaque-only scaling. Paired
        // with the delta-gated emission above so an emitter is counted once.
        radiance = radiance + throughput * sample_emitters(h.pos, n, v, m, f0, rng) * opacity;

        // Choose the interaction: a dielectric transmission event through a
        // sampled microfacet, or the opaque reflection lobes. Metals never
        // transmit.
        let p_trans = 1.0 - opacity;
        if p_trans > 0.0 && rand(rng) < p_trans {
            throughput = throughput / p_trans;
            // Transmission is treated as a delta bounce for MIS: env NEE did not
            // sample through it, so the next environment hit takes full weight.
            mis_bsdf_pdf = -1.0;
            // Rough-dielectric interface: reflect or refract through a sampled
            // microfacet normal, split stochastically by the Fresnel term.
            let hvec = ggx_sample_h(n, rough, rng);
            let n1 = select(m.ior, 1.0, h.front);
            let n2 = select(1.0, m.ior, h.front);
            let cos_i = clamp(dot(v, hvec), 1.0e-4, 1.0);
            let fr = fresnel_dielectric(cos_i, n1, n2);
            let ndotv = max(dot(n, v), 1.0e-4);
            let ndoth = max(dot(n, hvec), 1.0e-4);
            if rand(rng) < fr {
                // Reflect. Fresnel is carried by the split, so the weight is the
                // colourless microfacet-reflection term.
                let l = reflect(-v, hvec);
                if dot(h.ng, l) <= 0.0 { break; }
                let ndotl = max(dot(n, l), 1.0e-4);
                let g = G_Smith(ndotv, ndotl, rough);
                throughput = throughput * (g * cos_i / (ndotv * ndoth));
                ro = h.pos + h.ng * EPS;
                rd = l;
            } else {
                // Refract to the far side, tinted by the glass colour.
                let l = refract(-v, hvec, n1 / n2);
                if dot(l, l) < 1.0e-6 { break; }
                let ln = normalize(l);
                let ndotl = max(abs(dot(n, ln)), 1.0e-4);
                let g = G_Smith(ndotv, ndotl, rough);
                throughput = throughput * m.base * (g * cos_i / (ndotv * max(abs(dot(n, hvec)), 1.0e-4)));
                ro = h.pos - h.ng * EPS;
                rd = ln;
            }
        } else {
            throughput = throughput / opacity;
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
            // Record the pdf of the sampled direction so a following environment
            // hit is MIS-weighted against the env NEE done above.
            mis_bsdf_pdf = bsdf_pdf(n, v, rd, m.metallic, m.roughness, f0);
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

fn ray_through(uv: vec2<f32>) -> vec3<f32> {
    var ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    let far = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far_point = far.xyz / far.w;
    return normalize(far_point - frame.cam_pos.xyz);
}

fn camera_ray(px: vec2<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let dims = vec2<f32>(f32(frame.dims.x), f32(frame.dims.y));
    let jitter = vec2<f32>(rand(rng), rand(rng));
    return ray_through((px + jitter) / dims);
}

fn camera_ray_centre(px: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(f32(frame.dims.x), f32(frame.dims.y));
    return ray_through((px + vec2<f32>(0.5)) / dims);
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

    // First-hit guide for the denoiser: an unjittered primary ray gives a stable
    // per-pixel albedo and normal to demodulate and edge-stop against. Written
    // once on the first accumulation batch; identical for every later batch.
    if sample_base == 0u {
        let prd = camera_ray_centre(vec2<f32>(f32(gid.x), f32(gid.y)));
        let ph = closest_hit(frame.cam_pos.xyz, prd);
        if ph.hit {
            let pm = materials[ph.mat];
            gbuf_albedo[pidx] = vec4<f32>(max(pm.base, vec3<f32>(0.02)), 1.0);
            gbuf_normal[pidx] = vec4<f32>(ph.ns, 0.0);
        } else {
            gbuf_albedo[pidx] = vec4<f32>(1.0, 1.0, 1.0, 0.0);
            gbuf_normal[pidx] = vec4<f32>(0.0);
        }
    }

    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < spp; s = s + 1u) {
        let rd = camera_ray(vec2<f32>(f32(gid.x), f32(gid.y)), &rng);
        let c = trace_path(frame.cam_pos.xyz, rd, &rng, -1.0);
        // Reject non-finite samples (NaN via self-inequality, Inf via the bound):
        // a degenerate path from a pdf underflow carries no real energy but would
        // otherwise poison the pixel's running mean.
        if all(c == c) && max(c.x, max(c.y, c.z)) < 1.0e30 {
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

// ----- Lightmap bake -----

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Direct lighting a texel receives from the analytic delta lights: the geometric
// irradiance (radiance times cosine, visibility-tested, no BRDF) in `e`, and the
// luminance-weighted sum of the light directions in `d` for the dominant-
// direction encode. A hemisphere ray almost never hits a delta light, so these
// are sampled explicitly; the environment and bounced surfaces are left to the
// hemisphere integral in bake_main.
struct DirectLighting {
    e: vec3<f32>,
    d: vec3<f32>,
}

fn texel_direct(pos: vec3<f32>, n: vec3<f32>) -> DirectLighting {
    var e = vec3<f32>(0.0);
    var d = vec3<f32>(0.0);
    let count = frame.dims.z;
    for (var i = 0u; i < count; i = i + 1u) {
        let lt = lights[i];
        var l: vec3<f32>;
        var radiance = lt.colour.rgb;
        var max_t = T_MAX;
        if lt.data.w < 0.5 {
            l = normalize(lt.data.xyz);
        } else {
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
        let ndl = dot(n, l);
        if ndl <= 0.0 { continue; }
        if any_hit(pos + n * EPS, l, max_t) { continue; }
        let contrib = radiance * ndl;
        e = e + contrib;
        d = d + l * luminance(contrib);
    }
    return DirectLighting(e, d);
}

// One area-light NEE sample for the bake: like sample_emitters, but accumulates
// incident irradiance (no BRDF, since the bake stores material-independent
// irradiance and a later stage applies albedo). Sampling an emitter directly from
// the texel keeps small bright emitters low-noise; the bake's primary cosine ray
// is gated (mis0 = 0 in trace_path) so its emitter hit does not also count.
fn texel_emitter_nee(pos: vec3<f32>, n: vec3<f32>, rng: ptr<function, u32>) -> DirectLighting {
    let count = frame.emit.x;
    if count == 0u { return DirectLighting(vec3<f32>(0.0), vec3<f32>(0.0)); }

    let ei = min(u32(rand(rng) * f32(count)), count - 1u);
    let tri = tris[emitter_tri(ei)];

    let su = sqrt(rand(rng));
    let b1 = 1.0 - su;
    let b2 = rand(rng) * su;
    let lp = tri.p0 * (1.0 - b1 - b2) + tri.p1 * b1 + tri.p2 * b2;

    let e1 = tri.p1 - tri.p0;
    let e2 = tri.p2 - tri.p0;
    let cr = cross(e1, e2);
    let area2 = length(cr);
    if area2 < 1.0e-12 { return DirectLighting(vec3<f32>(0.0), vec3<f32>(0.0)); }
    let ng_l = cr / area2;

    var to = lp - pos;
    let dist2 = dot(to, to);
    let dist = sqrt(dist2);
    if dist < 1.0e-4 { return DirectLighting(vec3<f32>(0.0), vec3<f32>(0.0)); }
    let wi = to / dist;
    let ndl = dot(n, wi);
    if ndl <= 0.0 { return DirectLighting(vec3<f32>(0.0), vec3<f32>(0.0)); }
    let cos_l = abs(dot(ng_l, wi));
    if cos_l < 1.0e-4 { return DirectLighting(vec3<f32>(0.0), vec3<f32>(0.0)); }
    if any_hit(pos + n * EPS, wi, dist - 1.0e-3) { return DirectLighting(vec3<f32>(0.0), vec3<f32>(0.0)); }

    // E = Le * cos_surface / pdf_sa, pdf_sa = dist^2 / (count * area * cos_light).
    let le = materials[tri.mat].emissive;
    let inv_pdf = f32(count) * (0.5 * area2) * cos_l / dist2;
    let contrib = le * ndl * inv_pdf;
    return DirectLighting(contrib, wi * luminance(contrib));
}

// One invocation per atlas texel. Reads the surface point behind the texel from
// the G-buffer, sums direct irradiance from the analytic lights, and estimates
// indirect + environment irradiance by cosine-sampling the hemisphere: for a
// cosine pdf the estimator of the irradiance integral is PI times the mean of
// the radiance each ray gathers (trace_path handles further bounces, emissive
// surfaces, and the sky/environment on miss). Stores incident irradiance, which
// is material-independent; a later encode stage applies albedo.
@compute @workgroup_size(8, 8, 1)
fn bake_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = frame.dims.x;
    let h = frame.dims.y;
    if gid.x >= w || gid.y >= h { return; }
    let idx = gid.y * w + gid.x;

    let sample_base = frame.params.y;
    let cov = texel_surf[idx].w;
    if cov <= 0.0 {
        // Empty texel: no surface behind it. Clear once on the first batch.
        if sample_base == 0u {
            accum[idx] = vec4<f32>(0.0);
            accum_dir[idx] = vec4<f32>(0.0);
        }
        return;
    }

    let pos = texel_surf[idx].xyz;
    let n = normalize(texel_surf[w * h + idx].xyz);
    let spp = frame.params.x;
    var rng = init_rng(idx, frame.params.z);

    // Direct lighting is deterministic per texel; evaluate it once and add it to
    // every sample so the cross-batch mean below stays exact.
    let direct = texel_direct(pos, n);

    var sum = vec3<f32>(0.0);
    var dir_sum = vec3<f32>(0.0);
    for (var s = 0u; s < spp; s = s + 1u) {
        // Area-light NEE toward emissive geometry, sampled per iteration. The
        // primary ray below is gated (mis0 = 0) so a cosine ray that happens to
        // hit the same emitter does not double count.
        let en = texel_emitter_nee(pos, n, &rng);
        let dir = cosine_sample(n, &rng);
        let li = trace_path(pos + n * EPS, dir, &rng, 0.0);
        let indirect = PI * li;
        let c = direct.e + en.e + indirect;
        // Reject non-finite samples (NaN and Inf), as in the camera kernel.
        if all(c == c) && max(c.x, max(c.y, c.z)) < 1.0e30 {
            sum = sum + c;
            dir_sum = dir_sum + direct.d + en.d + dir * luminance(indirect);
        }
    }

    let inv = 1.0 / f32(spp);
    var result = sum * inv;
    var dir_result = dir_sum * inv;
    if sample_base > 0u {
        let total = f32(sample_base) + f32(spp);
        result = (accum[idx].rgb * f32(sample_base) + sum) / total;
        dir_result = (accum_dir[idx].xyz * f32(sample_base) + dir_sum) / total;
    }
    // w carries coverage so a consumer can tell baked texels from empty ones.
    accum[idx] = vec4<f32>(result, cov);
    accum_dir[idx] = vec4<f32>(dir_result, cov);
}
