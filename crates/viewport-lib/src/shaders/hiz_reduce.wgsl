// HiZ pyramid: build mip N+1 as the max of the 2x2 block below it in mip N.
// Max (not min) so each texel reports the farthest depth in its region, which
// keeps the occlusion test conservative: an instance is only culled when it is
// behind the farthest occluder in the box it covers.
//
// Odd parent dimensions are handled by folding the extra row/column into the
// max, so the result stays an upper bound on depth over the whole footprint.

@group(0) @binding(0) var mip_src: texture_2d<f32>;
@group(0) @binding(1) var mip_dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dim = textureDimensions(mip_dst);
    if gid.x >= dst_dim.x || gid.y >= dst_dim.y {
        return;
    }

    let src_dim = vec2<i32>(textureDimensions(mip_src));
    let last = src_dim - vec2<i32>(1, 1);
    let base = vec2<i32>(gid.xy) * 2;

    var m = textureLoad(mip_src, base, 0).r;
    m = max(m, textureLoad(mip_src, min(base + vec2<i32>(1, 0), last), 0).r);
    m = max(m, textureLoad(mip_src, min(base + vec2<i32>(0, 1), last), 0).r);
    m = max(m, textureLoad(mip_src, min(base + vec2<i32>(1, 1), last), 0).r);

    // Odd source width: this destination texel also covers the column at base+2.
    let odd_x = (src_dim.x & 1) == 1;
    let odd_y = (src_dim.y & 1) == 1;
    if odd_x {
        let ex = min(base.x + 2, last.x);
        m = max(m, textureLoad(mip_src, vec2<i32>(ex, base.y), 0).r);
        m = max(m, textureLoad(mip_src, vec2<i32>(ex, min(base.y + 1, last.y)), 0).r);
    }
    if odd_y {
        let ey = min(base.y + 2, last.y);
        m = max(m, textureLoad(mip_src, vec2<i32>(base.x, ey), 0).r);
        m = max(m, textureLoad(mip_src, vec2<i32>(min(base.x + 1, last.x), ey), 0).r);
        if odd_x {
            m = max(m, textureLoad(mip_src, vec2<i32>(min(base.x + 2, last.x), ey), 0).r);
        }
    }

    textureStore(mip_dst, vec2<i32>(gid.xy), vec4<f32>(m, 0.0, 0.0, 0.0));
}
