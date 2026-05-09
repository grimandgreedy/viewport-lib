// ssao.wgsl : screen-space ambient occlusion using hemisphere sampling.
// Reconstructs view-space position from depth, samples a rotated hemisphere kernel,
// and estimates per-pixel ambient occlusion.
//
// Point cloud fallback: point cloud billboards render all pixels with the same
// depth (constant clip-space Z from the billboard center). The hemisphere kernel
// produces unreliable results for these fragments because samples land on
// background or distant other billboards. When the 8-neighbor depth neighborhood
// shows a constant-depth region (billboard interior), a depth-cavity test is used
// instead: darken if the pixel is significantly deeper than its surroundings.

struct SsaoUniform {
    inv_proj: mat4x4<f32>,  // 64 bytes : NDC+depth -> view-space position (unproject)
    proj:     mat4x4<f32>,  // 64 bytes : view-space -> clip (re-project samples)
    radius:   f32,          //  4 bytes : hemisphere sample radius in view units
    bias:     f32,          //  4 bytes : depth comparison bias (avoids self-occlusion)
    _pad:     vec2<f32>,    //  8 bytes : alignment
}

@group(0) @binding(0) var depth_tex:  texture_depth_2d;
@group(0) @binding(1) var depth_samp: sampler;
@group(0) @binding(2) var noise_tex:  texture_2d<f32>;
@group(0) @binding(3) var noise_samp: sampler;
@group(0) @binding(4) var<storage, read> kernel: array<vec4<f32>>;  // 64 hemisphere samples
@group(0) @binding(5) var<uniform> params: SsaoUniform;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    let uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv);
}

// Reconstruct view-space position from a UV and a depth value.
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), depth, 1.0);
    let vp = params.inv_proj * ndc;
    return vp.xyz / vp.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(depth_tex));
    let dims_i = vec2<i32>(textureDimensions(depth_tex));
    let pixel = vec2<i32>(i32(in.pos.x), i32(in.pos.y));

    // Load raw depth and bail on background pixels.
    let depth = textureLoad(depth_tex, pixel, 0);
    if depth >= 0.9999 {
        return vec4<f32>(1.0);
    }

    // Detect point cloud billboard pixels: sample 8 immediate neighbors and
    // count how many have the same depth as the center within a tight tolerance.
    // Billboard quads render all pixels at the same clip-space Z, so interior
    // pixels have exactly equal depth. Mesh surfaces have smoothly varying depth.
    let depth_eps = 0.000005;
    var same_depth_count: u32 = 0u;
    var nonbg_count: u32 = 0u;
    let n_offsets = array<vec2<i32>, 8>(
        vec2<i32>( 1,  0), vec2<i32>(-1,  0),
        vec2<i32>( 0,  1), vec2<i32>( 0, -1),
        vec2<i32>( 1,  1), vec2<i32>(-1,  1),
        vec2<i32>( 1, -1), vec2<i32>(-1, -1),
    );
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let np = clamp(pixel + n_offsets[k], vec2<i32>(0), dims_i - vec2<i32>(1));
        let nd = textureLoad(depth_tex, np, 0);
        if nd < 0.9999 {
            nonbg_count = nonbg_count + 1u;
            if abs(nd - depth) < depth_eps {
                same_depth_count = same_depth_count + 1u;
            }
        }
    }
    // If all non-background neighbors share the same depth, this is a billboard pixel.
    let is_billboard = nonbg_count > 0u && same_depth_count >= nonbg_count;

    if is_billboard {
        // Depth-cavity fallback for point cloud fragments.
        // Linearize depth: z_eye = near / (depth * (far/near - 1) + 1) (reversed-Z not used,
        // standard NDC depth).  We only need relative comparison so use a simple proxy:
        // shallower depth value (smaller number) = closer to camera.
        //
        // Sample 8 neighbors at a wider radius (~3 px). If the center is notably
        // deeper (further) than most of its neighbors, it sits in a cavity between
        // point clusters and should be darkened.
        let ring = 3;
        var closer_count: f32 = 0.0;
        var valid_count: f32 = 0.0;
        let cavity_thresh = 0.0002;
        for (var k: i32 = 0; k < 8; k = k + 1) {
            let np = clamp(pixel + n_offsets[k] * ring, vec2<i32>(0), dims_i - vec2<i32>(1));
            let nd = textureLoad(depth_tex, np, 0);
            if nd < 0.9999 {
                valid_count = valid_count + 1.0;
                // Smaller depth value = closer; if neighbor is shallower than center
                // by more than threshold, the center is in a cavity.
                if nd < depth - cavity_thresh {
                    closer_count = closer_count + 1.0;
                }
            }
        }
        // Fraction of neighbors that are shallower (center is deeper = in cavity).
        let cavity_ratio = select(0.0, closer_count / valid_count, valid_count > 0.0);
        let ao = 1.0 - cavity_ratio;
        return vec4<f32>(ao, ao, ao, 1.0);
    }

    // Standard hemisphere SSAO for mesh geometry.

    // Reconstruct the current pixel's view-space position.
    let pos_v = view_pos_from_depth(in.uv, depth);

    // Reconstruct view-space normal from position derivatives.
    let pos_dx = dpdx(pos_v);
    let pos_dy = dpdy(pos_v);
    // Swap order: cross(pos_dx, pos_dy) points into the surface in wgpu screen-space
    // (screen Y increases downward, so dpdy points in -view-Y). Swapping gives +Z (toward camera).
    let normal = normalize(cross(pos_dy, pos_dx));

    // Random rotation tangent from a tiled 4x4 noise texture.
    let noise_uv = in.uv * (dim / 4.0);
    let rnd_xy   = textureSample(noise_tex, noise_samp, noise_uv).xy * 2.0 - 1.0;
    let rnd      = vec3<f32>(rnd_xy, 0.0);
    let tangent  = normalize(rnd - normal * dot(rnd, normal));
    let bitan    = cross(normal, tangent);
    let tbn      = mat3x3<f32>(tangent, bitan, normal);

    // Accumulate occlusion from 64 hemisphere samples.
    var occlusion: f32 = 0.0;
    for (var i: i32 = 0; i < 64; i = i + 1) {
        // Rotate sample into view space.
        let sample_v = pos_v + (tbn * kernel[i].xyz) * params.radius;

        // Project sample back to screen UV.
        let sample_clip = params.proj * vec4<f32>(sample_v, 1.0);
        let sample_ndc  = sample_clip.xyz / sample_clip.w;
        let sample_uv   = vec2<f32>(
            sample_ndc.x *  0.5 + 0.5,
            sample_ndc.y * -0.5 + 0.5,
        );

        // Discard out-of-screen samples.
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            continue;
        }

        // Load scene depth at the projected sample position.
        let sp = vec2<i32>(
            i32(clamp(sample_uv.x * dim.x, 0.0, dim.x - 1.0)),
            i32(clamp(sample_uv.y * dim.y, 0.0, dim.y - 1.0)),
        );
        let scene_depth = textureLoad(depth_tex, sp, 0);
        let scene_v     = view_pos_from_depth(sample_uv, scene_depth);

        // Ranged occlusion check.
        let range_check = smoothstep(0.0, 1.0, params.radius / abs(pos_v.z - scene_v.z));
        if scene_v.z >= sample_v.z + params.bias {
            occlusion = occlusion + range_check;
        }
    }

    let ao = 1.0 - (occlusion / 64.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
