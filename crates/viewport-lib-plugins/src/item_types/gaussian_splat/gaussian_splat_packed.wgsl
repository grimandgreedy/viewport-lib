// gaussian_splat_packed.wgsl
//
// Draws a splat set on devices without vertex storage. The set lives in the
// Rgba32Uint data textures gaussian_splat_pack.wgsl writes, and the draw order
// arrives as an instance-stepped index stream the CPU sorts back to front.
// Appended to gaussian_splat_common.wgsl, whose projection and fragment stage
// this shares with the storage path.
//
// Group 1: SplatUniform (binding 0, common), then
//          binding 1-3: centre and opacity, covariance, covariance and colour
//          binding 4-6: the first twelve SH coefficients (degree 1 and up)

@group(1) @binding(1) var splat_base0: texture_2d<u32>;
@group(1) @binding(2) var splat_base1: texture_2d<u32>;
@group(1) @binding(3) var splat_base2: texture_2d<u32>;
@group(1) @binding(4) var splat_sh0:   texture_2d<u32>;
@group(1) @binding(5) var splat_sh1:   texture_2d<u32>;
@group(1) @binding(6) var splat_sh2:   texture_2d<u32>;

@vertex
fn vs_packed(
    @builtin(vertex_index) vi:        u32,
    @location(0)           splat_idx: u32,
) -> VsOut {
    let width = textureDimensions(splat_base0).x;
    let at = vec2<i32>(i32(splat_idx % width), i32(splat_idx / width));
    let t0 = bitcast<vec4<f32>>(textureLoad(splat_base0, at, 0));
    let t1 = bitcast<vec4<f32>>(textureLoad(splat_base1, at, 0));
    let t2 = textureLoad(splat_base2, at, 0);
    let pos_obj = t0.xyz;
    let s12 = bitcast<f32>(t2.x);
    let s22 = bitcast<f32>(t2.y);
    let Sigma3d = mat3x3<f32>(
        vec3<f32>(t1.x, t1.y, t1.z),
        vec3<f32>(t1.y, t1.w, s12),
        vec3<f32>(t1.z, s12,  s22),
    );

    var colour = vec3<f32>(unpack2x16float(t2.z), unpack2x16float(t2.w).x);
    if splat_u.sh_degree != 0u {
        let a = bitcast<vec4<f32>>(textureLoad(splat_sh0, at, 0));
        let b = bitcast<vec4<f32>>(textureLoad(splat_sh1, at, 0));
        let c = bitcast<vec4<f32>>(textureLoad(splat_sh2, at, 0));
        colour = eval_sh1(
            a.xyz,
            vec3<f32>(a.w, b.xy),
            vec3<f32>(b.zw, c.x),
            c.yzw,
            splat_view_dir(pos_obj),
        );
    }
    return project_splat(pos_obj, Sigma3d, colour, t0.w, vi);
}
