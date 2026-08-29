// Diffuse irradiance convolution of an equirectangular HDR skybox.
//
// One invocation per destination texel. Each invocation walks the upper
// hemisphere with cos-weighted hemisphere sampling, reading the source
// skybox by 2D equirect projection.

const PI: f32 = 3.14159265358979;

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, write>;

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

    let u = f32(gid.x) / f32(dims.x);
    let v = f32(gid.y) / f32(dims.y);
    let theta_n = PI * (0.5 - v);
    let phi_n = 2.0 * PI * (u - 0.5);

    let st = sin(theta_n);
    let ct = cos(theta_n);
    let sp = sin(phi_n);
    let cp = cos(phi_n);
    // Z-up: latitude theta drives Z, longitude phi spins around Z in the XY plane.
    let normal = vec3<f32>(ct * cp, ct * sp, st);

    // Pick a non-collinear reference axis to build the tangent frame.
    var up: vec3<f32>;
    if (abs(normal.z) < 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    var irr = vec3<f32>(0.0);
    var sample_count: f32 = 0.0;
    let sample_delta: f32 = 0.05;

    var s_phi: f32 = 0.0;
    loop {
        if (s_phi >= 2.0 * PI) { break; }
        var s_theta: f32 = 0.0;
        loop {
            if (s_theta >= 0.5 * PI) { break; }
            let sst = sin(s_theta);
            let sct = cos(s_theta);
            let ssp = sin(s_phi);
            let scp = cos(s_phi);
            let ts = vec3<f32>(sst * scp, sst * ssp, sct);
            let dir = normalize(ts.x * tangent + ts.y * bitangent + ts.z * normal);
            let uv = dir_to_equirect_uv(dir);
            let c = textureSampleLevel(src_tex, src_sampler, uv, 0.0).rgb;
            let w = sct * sst;
            irr += c * w;
            sample_count += 1.0;
            s_theta += sample_delta;
        }
        s_phi += sample_delta;
    }

    let scale = PI / sample_count;
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(irr * scale, 1.0));
}
