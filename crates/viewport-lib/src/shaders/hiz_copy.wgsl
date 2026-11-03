// HiZ pyramid: copy the scene depth buffer into mip 0 of the R32Float HiZ
// texture. mip 0 is the same size as the depth target, so this is a 1:1 copy
// from the sampled depth aspect into a storage texture the reduction kernel
// and the cull shader can read.
//
// Depth is stored as-is (reversed NDC depth: 1 at the near plane, 0 far). The
// pyramid keeps the farthest depth in each region, a bound that is safe for a
// conservative occlusion test.

@group(0) @binding(0) var depth_src: texture_2d<f32>;
@group(0) @binding(1) var mip_dst:   texture_storage_2d<r32float, write>;
@group(0) @binding(2) var depth_sampler: sampler;

// #include "helpers/depth_texel.wgsl"

@compute @workgroup_size(8, 8)
fn copy_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(mip_dst);
    if gid.x >= dim.x || gid.y >= dim.y {
        return;
    }
    let d = depth_texel(depth_src, depth_sampler, vec2<i32>(gid.xy));
    textureStore(mip_dst, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}
