// Copy the reprojected depth (scattered into a u32 buffer with atomic min)
// into mip 0 of the HiZ pyramid texture. Runs after the scatter pass; the
// max-reduction then builds the rest of the chain from this mip.

@group(0) @binding(0) var<storage, read> src:  array<u32>;
@group(0) @binding(1) var                mip0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn to_texture(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(mip0);
    if gid.x >= dim.x || gid.y >= dim.y {
        return;
    }
    let idx = gid.y * dim.x + gid.x;
    let d = bitcast<f32>(src[idx]);
    textureStore(mip0, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}
