// BRDF integration LUT for split-sum specular approximation.
//
// Output is a 128x128 Rgba16Float texture. R = a (scale on F0), G = b (bias on F0).
// One invocation per destination texel; 1024 Hammersley samples each.
//
// This is the GPU equivalent of `generate_brdf_lut` in environment.rs. The LUT
// is scene-independent so this dispatch only ever runs once per renderer.

const PI: f32 = 3.14159265358979;
const NUM_SAMPLES: u32 = 1024u;

@group(0) @binding(0) var dst_tex: texture_storage_2d<rgba16float, write>;

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

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let g1v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g1l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return g1v * g1l;
}

fn integrate_brdf(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);
    let a = roughness * roughness;

    var a_out: f32 = 0.0;
    var b_out: f32 = 0.0;

    for (var i: u32 = 0u; i < NUM_SAMPLES; i = i + 1u) {
        let xi = hammersley(i, NUM_SAMPLES);
        let h = importance_sample_ggx(xi, n, a);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let n_dot_l = max(l.z, 0.0);
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);

        if (n_dot_l > 0.0) {
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / max(n_dot_h * n_dot_v, 0.001);
            let fc = pow(1.0 - v_dot_h, 5.0);
            a_out += (1.0 - fc) * g_vis;
            b_out += fc * g_vis;
        }
    }
    let inv = 1.0 / f32(NUM_SAMPLES);
    return vec2<f32>(a_out * inv, b_out * inv);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let n_dot_v = max((f32(gid.x) + 0.5) / f32(dims.x), 0.001);
    let roughness = max((f32(gid.y) + 0.5) / f32(dims.y), 0.01);

    let ab = integrate_brdf(n_dot_v, roughness);
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(ab, 0.0, 1.0));
}
