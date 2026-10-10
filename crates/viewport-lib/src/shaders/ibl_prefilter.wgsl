// GGX-prefiltered specular convolution of an equirectangular environment.
//
// One invocation per destination texel for a single mip level. The source is
// the bake's filter pyramid. Each importance sample reads the pyramid level
// whose texels match the solid angle the sample stands for (filtered importance
// sampling), so a small bright sun spreads over the lobe instead of lighting a
// few texels and missing their neighbours. Roughness 0 is a single read of the
// pyramid level the size of the destination (`min_lod`).
//
// Reusable across both IBL environment prefilter and reflection-probe
// prefilter; both are GGX roughness convolutions; only the source texture
// dimensions differ.

const PI: f32 = 3.14159265358979;
const NUM_SAMPLES: u32 = 2048u;

struct PrefilterParams {
    roughness: f32,
    // The pyramid level a roughness-0 texel reads.
    min_lod: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: PrefilterParams;

fn radical_inverse_vdc(b: u32) -> f32 {
    var bits = b;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, a: f32) -> vec3<f32> {
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let h_ts = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    var up: vec3<f32>;
    if (abs(n.z) < 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(h_ts.x * tangent + h_ts.y * bitangent + h_ts.z * n);
}

// viewport-lib is Z-up: longitude around +Z, latitude with +Z polar.
fn dir_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let phi = atan2(dir.y, dir.x);
    let theta = asin(clamp(dir.z, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), 0.5 - theta / PI);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Texel centres.
    let u = (f32(gid.x) + 0.5) / f32(dims.x);
    let v = (f32(gid.y) + 0.5) / f32(dims.y);
    let theta_n = PI * (0.5 - v);
    let phi_n = 2.0 * PI * (u - 0.5);

    let st = sin(theta_n);
    let ct = cos(theta_n);
    let sp = sin(phi_n);
    let cp = cos(phi_n);
    // Z-up: latitude theta drives Z, longitude phi spins around Z in the XY plane.
    let n = vec3<f32>(ct * cp, ct * sp, st);

    // For split-sum approximation, view direction = reflection direction = normal.
    let v_dir = n;

    let a = params.roughness * params.roughness;
    var color = vec3<f32>(0.0);
    var total_weight: f32 = 0.0;

    let base = textureDimensions(src_tex, 0);
    let max_lod = f32(textureNumLevels(src_tex) - 1u);
    // Solid angle of a base-level texel at the equator.
    let texel_omega = (2.0 * PI / f32(base.x)) * (PI / f32(base.y));
    let a2 = a * a;
    if (a == 0.0) {
        // A mirror lobe is the one direction: read it at the output's size.
        let c = textureSampleLevel(src_tex, src_sampler, dir_to_equirect_uv(n), params.min_lod).rgb;
        textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(c, 1.0));
        return;
    }

    for (var i: u32 = 0u; i < NUM_SAMPLES; i = i + 1u) {
        let xi = hammersley(i, NUM_SAMPLES);
        let h = importance_sample_ggx(xi, n, a);
        let l = normalize(2.0 * dot(v_dir, h) * h - v_dir);
        let n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l > 0.0) {
            // With the view along the normal, the sample's pdf is D / 4. The
            // level whose texels cover the solid angle the sample stands for,
            // one level blurrier as the published method biases it.
            let n_dot_h = max(dot(n, h), 0.0);
            let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
            let pdf = a2 / (PI * denom * denom) * 0.25;
            let sample_omega = 1.0 / (f32(NUM_SAMPLES) * pdf + 1e-6);
            // Texels shrink towards the poles.
            let cos_lat = max(sqrt(1.0 - l.z * l.z), 1e-3);
            let lod = max(0.5 * log2(sample_omega / (texel_omega * cos_lat)) + 1.0, 0.0);
            let uv = dir_to_equirect_uv(l);
            let c = textureSampleLevel(src_tex, src_sampler, uv, min(lod, max_lod)).rgb;
            color += c * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    color /= max(total_weight, 0.001);
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
