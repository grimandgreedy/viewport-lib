// Box-downsample an equirectangular environment into one level of the bake's
// filter pyramid.
//
// One invocation per destination texel. Each averages the source texels its
// footprint covers, weighted by their solid angle (the cosine of the row's
// latitude), so the energy of a small bright sun survives the reduction. A
// destination larger than the source repeats the nearest texel.

const PI: f32 = 3.14159265358979;

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = textureDimensions(dst_tex);
    if (gid.x >= dst.x || gid.y >= dst.y) {
        return;
    }
    let src = textureDimensions(src_tex);
    let x0 = gid.x * src.x / dst.x;
    let x1 = max((gid.x + 1u) * src.x / dst.x, x0 + 1u);
    let y0 = gid.y * src.y / dst.y;
    let y1 = max((gid.y + 1u) * src.y / dst.y, y0 + 1u);

    var sum = vec3<f32>(0.0);
    var weight: f32 = 0.0;
    for (var y = y0; y < y1; y = y + 1u) {
        let w = cos(PI * (0.5 - (f32(y) + 0.5) / f32(src.y)));
        for (var x = x0; x < x1; x = x + 1u) {
            sum += textureLoad(src_tex, vec2<u32>(x, y), 0).rgb * w;
        }
        weight += w * f32(x1 - x0);
    }
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(sum / max(weight, 1e-8), 1.0));
}
