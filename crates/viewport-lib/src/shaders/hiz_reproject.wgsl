// Reproject last frame's scene depth into this frame's camera, producing the
// depth that mip 0 of the HiZ pyramid is built from. The cull runs before this
// frame's scene pass, so there is no current depth yet; reprojecting the prior
// frame's depth is the cheapest source that tracks camera motion.
//
// Forward scatter: each source texel is unprojected to world space with the
// previous view-projection, reprojected with the current one, and its depth is
// written to the destination pixel with an atomic min (the nearest surface that
// lands on a pixel wins). The destination starts at the far value, so texels no
// source reaches (revealed this frame) stay far and read as "no occluder",
// which keeps the cull from dropping anything that just became visible.

struct ReprojUniform {
    // Inverse of the previous frame's view-projection (NDC -> world).
    inv_prev_vp: mat4x4<f32>,
    // Current frame's view-projection (world -> NDC).
    cur_vp:      mat4x4<f32>,
    dims:        vec2<u32>,
    _pad:        vec2<u32>,
}

@group(0) @binding(0) var<uniform>             u:          ReprojUniform;
@group(0) @binding(1) var                      prev_depth: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> dst:        array<atomic<u32>>;

// Bit pattern of 1.0f. For depths in [0,1] the float bit pattern is monotonic,
// so atomicMin over the bits is atomicMin over the depth.
const FAR_BITS: u32 = 0x3f800000u;

// 2D dispatch (not 1D over w*h) so neither workgroup dimension exceeds the
// 65535 limit at large or supersampled resolutions.
@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = u.dims;
    if gid.x >= dim.x || gid.y >= dim.y {
        return;
    }
    atomicStore(&dst[gid.y * dim.x + gid.x], FAR_BITS);
}

@compute @workgroup_size(8, 8)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = u.dims;
    if gid.x >= dim.x || gid.y >= dim.y {
        return;
    }

    let d = textureLoad(prev_depth, vec2<i32>(gid.xy), 0).r;
    // Background carries no occluder to reproject.
    if d >= 1.0 {
        return;
    }

    let dimf = vec2<f32>(dim);
    let uv = (vec2<f32>(gid.xy) + 0.5) / dimf;
    let ndc_prev = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, d);

    let world_h = u.inv_prev_vp * vec4<f32>(ndc_prev, 1.0);
    if world_h.w == 0.0 {
        return;
    }
    let world = world_h.xyz / world_h.w;

    let clip = u.cur_vp * vec4<f32>(world, 1.0);
    if clip.w <= 0.0 {
        return;
    }
    let ndc = clip.xyz / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return;
    }

    let du = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    let px = vec2<u32>(du * dimf);
    let cx = min(px.x, dim.x - 1u);
    let cy = min(px.y, dim.y - 1u);
    let idx = cy * dim.x + cx;
    atomicMin(&dst[idx], bitcast<u32>(ndc.z));
}
