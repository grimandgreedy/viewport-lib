// Screen-space decal projection shader.
//
// Group 0: camera_bgl (CameraUniform, provides inv_view_proj for world reconstruction)
// Group 1: per-viewport scene depth texture (depth-only aspect view)
// Group 2: per-decal uniform + albedo texture + sampler
//
// Vertex: full-screen quad (6 vertices, no vertex buffer, identical to implicit.wgsl)
// Fragment: load scene depth, reconstruct world position, project into decal local
//           space, sample texture, output blended colour.

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

// Scene depth written by the opaque pass (depth-only aspect, Depth24PlusStencil8).
@group(1) @binding(0) var scene_depth: texture_depth_2d;

struct DecalUniform {
    // Inverse of the decal model matrix: transforms world -> decal local space.
    inv_transform: mat4x4<f32>,
    blend_mode:    u32,   // 0 = Replace, 1 = Multiply
    alpha:         f32,
    _pad:          vec2<f32>,
};

@group(2) @binding(0) var<uniform> u:             DecalUniform;
@group(2) @binding(1) var          decal_tex:     texture_2d<f32>;
@group(2) @binding(2) var          decal_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc_xy:   vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var x: f32;
    var y: f32;
    switch vi {
        case 0u: { x = -1.0; y = -1.0; }
        case 1u: { x =  1.0; y = -1.0; }
        case 2u: { x = -1.0; y =  1.0; }
        case 3u: { x = -1.0; y =  1.0; }
        case 4u: { x =  1.0; y = -1.0; }
        default: { x =  1.0; y =  1.0; }
    }
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.ndc_xy   = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Load scene depth at the current pixel.  The depth texture is at the same
    // resolution as the render target (scene_size), so clip_pos.xy indexes it directly.
    let pix   = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
    let depth = textureLoad(scene_depth, pix, 0);

    // Depth == 1.0 means the skybox or the cleared background -- no surface here.
    if depth >= 1.0 {
        discard;
    }

    // Reconstruct world-space position from NDC.
    let ndc     = vec4<f32>(in.ndc_xy, depth, 1.0);
    let world_h = camera.inv_view_proj * ndc;
    let world   = world_h.xyz / world_h.w;

    // Transform into decal local space.  The decal's projection volume is
    // [-0.5, 0.5]^3 in local coordinates.
    let local_h = u.inv_transform * vec4<f32>(world, 1.0);
    let local   = local_h.xyz;

    // Reject fragments outside the projection box.
    if any(local < vec3<f32>(-0.5)) || any(local > vec3<f32>(0.5)) {
        discard;
    }

    // Map local XY from [-0.5, 0.5] to UV [0, 1].
    let uv      = local.xy + vec2<f32>(0.5);
    let tex_col = textureSample(decal_tex, decal_sampler, uv);
    let alpha   = tex_col.a * u.alpha;

    if alpha < 0.001 {
        discard;
    }

    return vec4<f32>(tex_col.rgb, alpha);
}
