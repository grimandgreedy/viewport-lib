// Depth-composited screen-space image overlay shader (Phase 12).
//
// Identical quad geometry to screen_image.wgsl, but the fragment shader:
//   1. samples the per-pixel image depth from a separate R32Float texture,
//   2. outputs that depth as frag_depth so the hardware depth test compares it
//      against the scene depth buffer already populated by 3D geometry.
//
// Pipeline uses depth_compare: LessEqual and depth_write_enabled: false.
// Fragments behind scene geometry are discarded; those in front are alpha-blended
// into the colour attachment.
//
// Depth values in `depth_img` must be in wgpu NDC depth range [0, 1] where
// 0 = near plane, 1 = far plane.

struct ScreenImageUniform {
    // Bottom-left corner in NDC space.
    ndc_min: vec2<f32>,
    // Top-right corner in NDC space.
    ndc_max: vec2<f32>,
    // Overall opacity multiplier (applied on top of per-pixel alpha).
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: ScreenImageUniform;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var img_sampler: sampler;
// R32Float depth image, one f32 per pixel in [0, 1] NDC depth.
@group(0) @binding(3) var depth_img: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

struct FragOutput {
    @location(0)         color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// Triangle 0: verts 0-2, Triangle 1: verts 3-5.
// Same layout as screen_image.wgsl.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var ndc_x: f32;
    var ndc_y: f32;
    var uv_x: f32;
    var uv_y: f32;

    switch vi {
        case 0u: { ndc_x = u.ndc_min.x; ndc_y = u.ndc_min.y; uv_x = 0.0; uv_y = 1.0; }
        case 1u: { ndc_x = u.ndc_max.x; ndc_y = u.ndc_min.y; uv_x = 1.0; uv_y = 1.0; }
        case 2u: { ndc_x = u.ndc_min.x; ndc_y = u.ndc_max.y; uv_x = 0.0; uv_y = 0.0; }
        case 3u: { ndc_x = u.ndc_min.x; ndc_y = u.ndc_max.y; uv_x = 0.0; uv_y = 0.0; }
        case 4u: { ndc_x = u.ndc_max.x; ndc_y = u.ndc_min.y; uv_x = 1.0; uv_y = 1.0; }
        default: { ndc_x = u.ndc_max.x; ndc_y = u.ndc_max.y; uv_x = 1.0; uv_y = 0.0; }
    }

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv       = vec2<f32>(uv_x, uv_y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    // Sample colour from the RGBA image.
    var color = textureSample(img, img_sampler, in.uv);
    color.a *= u.alpha;

    // Load depth from the R32Float depth image using nearest-pixel lookup so
    // depth values are not blended across pixel boundaries.
    let dims = vec2<i32>(textureDimensions(depth_img));
    let px_x = clamp(i32(in.uv.x * f32(dims.x)), 0, dims.x - 1);
    let px_y = clamp(i32(in.uv.y * f32(dims.y)), 0, dims.y - 1);
    let img_depth = textureLoad(depth_img, vec2<i32>(px_x, px_y), 0).r;

    var out: FragOutput;
    out.color = color;
    // Hardware depth test (LessEqual) will discard this fragment if img_depth
    // is greater than the scene depth already in the depth buffer.
    out.depth = img_depth;
    return out;
}
