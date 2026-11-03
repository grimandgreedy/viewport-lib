// Diffuse irradiance of an equirectangular environment, summed exactly.
//
// One invocation per destination texel. The source is the 128x64 level of the
// bake's filter pyramid, so every texel of the environment contributes through
// its block average: a small bright sun is never missed or over-counted, as it
// is when a few thousand point samples of the full-resolution image land on it
// or not. Each block weighs in by its radiance, its solid angle and its cosine
// to the normal. The result is stored divided by pi, ready to multiply by
// albedo.

const PI: f32 = 3.14159265358979;

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Z-up: latitude drives Z, longitude spins around Z. Texel centres.
    let lat_n = PI * (0.5 - (f32(gid.y) + 0.5) / f32(dims.y));
    let lon_n = 2.0 * PI * ((f32(gid.x) + 0.5) / f32(dims.x) - 0.5);
    let normal = vec3<f32>(cos(lat_n) * cos(lon_n), cos(lat_n) * sin(lon_n), sin(lat_n));

    let src = textureDimensions(src_tex);
    let d_lon = 2.0 * PI / f32(src.x);
    var irr = vec3<f32>(0.0);
    for (var y = 0u; y < src.y; y = y + 1u) {
        let top = PI * (0.5 - f32(y) / f32(src.y));
        let bottom = PI * (0.5 - f32(y + 1u) / f32(src.y));
        // Exact solid angle of one texel of this row.
        let omega = (sin(top) - sin(bottom)) * d_lon;
        let lat = 0.5 * (top + bottom);
        for (var x = 0u; x < src.x; x = x + 1u) {
            let lon = 2.0 * PI * ((f32(x) + 0.5) / f32(src.x) - 0.5);
            let dir = vec3<f32>(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));
            let c = dot(normal, dir);
            if (c > 0.0) {
                irr += textureLoad(src_tex, vec2<u32>(x, y), 0).rgb * (omega * c);
            }
        }
    }

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(irr / PI, 1.0));
}
