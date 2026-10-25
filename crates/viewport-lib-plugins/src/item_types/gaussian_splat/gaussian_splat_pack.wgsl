// gaussian_splat_pack.wgsl
//
// Packs a splat set into the data textures gaussian_splat_packed.wgsl reads,
// on devices without vertex storage. One point per splat, drawn at the texel
// that splat owns (texel i is at (i % width, i / width)), reads the set's
// channel buffers as vertex attributes and writes Rgba32Uint texels:
//
//   vs_pack_base, three texels:
//     0: centre xyz, opacity
//     1: covariance 00, 01, 02, 11
//     2: covariance 12, 22, then the degree-0 colour as two pack2x16float words
//   vs_pack_sh, three texels: the first twelve SH coefficients (DC and the
//     degree-1 terms, RGB per coefficient), four floats a texel.
//
// Floats are stored as their bits. WebGL2 allows 32 bytes of colour targets a
// sample, two of these texels, so each pack runs as two passes over the same
// vertex stage: fs_pack_pair writes texels 0 and 1, fs_pack_last texel 2.
//
// Group 0: PackUniform.

// Padded to 16 bytes: wgpu's GL backend requires every uniform binding to be
// a multiple of 16 (std140 rounds a block up to it).
struct PackUniform {
    width:  u32,
    height: u32,
    _pad0:  u32,
    _pad1:  u32,
};

@group(0) @binding(0) var<uniform> pack_u: PackUniform;

// The clip position of texel `i`'s centre.
fn texel_position(i: u32) -> vec4<f32> {
    let x = f32(i % pack_u.width) + 0.5;
    let y = f32(i / pack_u.width) + 0.5;
    let ndc = vec2<f32>(
        x / f32(pack_u.width) * 2.0 - 1.0,
        1.0 - y / f32(pack_u.height) * 2.0,
    );
    return vec4<f32>(ndc, 0.0, 1.0);
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let x2 = x * x; let y2 = y * y; let z2 = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y2+z2), 2.0*(xy+wz),     2.0*(xz-wy)),
        vec3<f32>(2.0*(xy-wz),        1.0-2.0*(x2+z2), 2.0*(yz+wx)),
        vec3<f32>(2.0*(xz+wy),        2.0*(yz-wx),     1.0-2.0*(x2+y2)),
    );
}

struct PackOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) t0: vec4<u32>,
    @location(1) @interpolate(flat) t1: vec4<u32>,
    @location(2) @interpolate(flat) t2: vec4<u32>,
};

struct PackPair {
    @location(0) t0: vec4<u32>,
    @location(1) t1: vec4<u32>,
};

@vertex
fn vs_pack_base(
    @builtin(vertex_index) i: u32,
    @location(0) position: vec4<f32>,
    @location(1) scale:    vec4<f32>,
    @location(2) rotation: vec4<f32>,
    @location(3) opacity:  f32,
    @location(4) dc:       vec3<f32>,
) -> PackOut {
    // The same covariance gaussian_splat_common.wgsl's splat_covariance builds.
    let R = quat_to_mat3(rotation);
    let M = mat3x3<f32>(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    let S = M * transpose(M);
    // The degree-0 colour, as eval_sh0 evaluates it.
    let colour = clamp(vec3<f32>(0.5) + 0.28209479177 * dc, vec3<f32>(0.0), vec3<f32>(1.0));

    var out: PackOut;
    out.clip_pos = texel_position(i);
    out.t0 = bitcast<vec4<u32>>(vec4<f32>(position.xyz, opacity));
    out.t1 = bitcast<vec4<u32>>(vec4<f32>(S[0][0], S[0][1], S[0][2], S[1][1]));
    out.t2 = vec4<u32>(
        bitcast<u32>(S[1][2]),
        bitcast<u32>(S[2][2]),
        pack2x16float(colour.rg),
        pack2x16float(vec2<f32>(colour.b, 0.0)),
    );
    return out;
}

@vertex
fn vs_pack_sh(
    @builtin(vertex_index) i: u32,
    @location(0) c0: vec3<f32>,
    @location(1) c1: vec3<f32>,
    @location(2) c2: vec3<f32>,
    @location(3) c3: vec3<f32>,
) -> PackOut {
    var out: PackOut;
    out.clip_pos = texel_position(i);
    out.t0 = bitcast<vec4<u32>>(vec4<f32>(c0, c1.x));
    out.t1 = bitcast<vec4<u32>>(vec4<f32>(c1.yz, c2.xy));
    out.t2 = bitcast<vec4<u32>>(vec4<f32>(c2.z, c3));
    return out;
}

@fragment
fn fs_pack_pair(in: PackOut) -> PackPair {
    return PackPair(in.t0, in.t1);
}

@fragment
fn fs_pack_last(in: PackOut) -> @location(0) vec4<u32> {
    return in.t2;
}
