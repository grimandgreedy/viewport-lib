// Decal outline coverage mask.
//
// Writes the visible footprint of a selected decal into a single-channel mask.
// The coverage math (depth reconstruction + projection-box / cylinder test +
// facing checks) is the same as decal.wgsl's colour pass, and it samples the
// same albedo alpha and applies the same edge fade so the mask traces the
// sticker the viewer sees (e.g. a circular cutout), not the raw box. Where the
// colour shader would shade the decal this writes 1.0; where it discards, or
// where coverage falls below the silhouette threshold, this discards.
//
// Group 0: camera_bgl (CameraUniform at binding 0; other bindings unused here)
// Group 1: per-viewport scene depth + stencil (same as the decal colour pass)
// Group 2: per-decal bind group (same as the colour pass); the DecalUniform at
//          binding 0 and the albedo texture/sampler at bindings 1/2 are read

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

@group(1) @binding(0) var scene_depth:   texture_depth_2d;
@group(1) @binding(1) var scene_stencil: texture_2d<u32>;

struct DecalUniform {
    inv_transform:         mat4x4<f32>,
    blend_mode:            u32,
    alpha:                 f32,
    normal_blend_strength: f32,
    has_normal:            u32,
    roughness:             f32,
    metallic:              f32,
    has_roughness_tex:     u32,
    has_metallic_tex:      u32,
    uv_offset:             vec2<f32>,
    uv_scale:              vec2<f32>,
    emissive:              f32,
    has_emissive_tex:      u32,
    edge_fade:             f32,
    _pad:                  u32,
    projection:            u32,
    tri_blend_sharpness:   f32,
    _pad2:                 u32,
    _pad3:                 u32,
};

@group(2) @binding(0) var<uniform> u: DecalUniform;
@group(2) @binding(1) var decal_tex:  texture_2d<f32>;
@group(2) @binding(2) var decal_samp: sampler;

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
    let pix   = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
    let depth = textureLoad(scene_depth, pix, 0);

    let stencil = textureLoad(scene_stencil, pix, 0).r;
    if stencil == 0u { discard; }

    if depth >= 1.0 {
        discard;
    }

    let ndc     = vec4<f32>(in.ndc_xy, depth, 1.0);
    let world_h = camera.inv_view_proj * ndc;
    let world   = world_h.xyz / world_h.w;

    let local_h = u.inv_transform * vec4<f32>(world, 1.0);
    let local   = local_h.xyz;

    if u.projection == 2u || u.projection == 3u {
        let r2 = local.x * local.x + local.y * local.y;
        if r2 > 0.25 || abs(local.z) > 0.5 { discard; }
    } else {
        if any(local < vec3<f32>(-0.5)) || any(local > vec3<f32>(0.5)) {
            discard;
        }
    }

    // Receiver surface normal from world-position screen derivatives, matching
    // the colour shader's facing checks.
    let ddx_w    = dpdx(world);
    let ddy_w    = dpdy(world);
    let n_raw    = normalize(cross(ddx_w, ddy_w));
    let view_dir = normalize(camera.eye_pos - world);
    let N_recv   = select(-n_raw, n_raw, dot(n_raw, view_dir) > 0.0);

    let decal_Z = normalize(vec3<f32>(u.inv_transform[0][2], u.inv_transform[1][2], u.inv_transform[2][2]));

    if u.projection == 0u {
        if dot(N_recv, decal_Z) < 0.1 { discard; }
        if dot(view_dir, decal_Z) < 0.05 { discard; }
    }

    if u.projection == 2u || u.projection == 3u {
        let local_normal = normalize((u.inv_transform * vec4<f32>(N_recv, 0.0)).xyz);
        let r = sqrt(local.x * local.x + local.y * local.y);
        if r > 0.001 {
            let radial_dir = vec2<f32>(local.x, local.y) / r;
            let radial_dot = dot(local_normal.xy, radial_dir);
            if u.projection == 2u {
                if radial_dot < 0.1 { discard; }
            } else {
                if radial_dot > -0.1 { discard; }
            }
        }
    }

    // Trace the visible sticker footprint rather than the raw box: apply the
    // same edge fade and albedo alpha the colour pass uses, so the outline
    // follows the sticker edge (a circular cutout, a faded border) instead of
    // the square projection box.
    var edge_alpha = 1.0;
    if u.edge_fade > 0.0 {
        if u.projection == 2u || u.projection == 3u {
            let r  = sqrt(local.x * local.x + local.y * local.y);
            let fr = smoothstep(0.0, u.edge_fade, 0.5 - r);
            let fz = smoothstep(0.0, u.edge_fade, 0.5 - abs(local.z));
            edge_alpha = fr * fz;
        } else {
            let fx = smoothstep(0.0, u.edge_fade, 0.5 - abs(local.x));
            let fy = smoothstep(0.0, u.edge_fade, 0.5 - abs(local.y));
            let fz = smoothstep(0.0, u.edge_fade, 0.5 - abs(local.z));
            edge_alpha = fx * fy * fz;
        }
    }

    var tex_alpha: f32;
    if u.projection == 0u {
        let base_uv = local.xy + vec2<f32>(0.5);
        tex_alpha = textureSample(decal_tex, decal_samp, u.uv_offset + u.uv_scale * base_uv).a;
    } else if u.projection == 2u || u.projection == 3u {
        let angle   = atan2(local.y, local.x);
        let base_uv = vec2<f32>(angle / (2.0 * 3.14159265) + 0.5, local.z + 0.5);
        tex_alpha = textureSample(decal_tex, decal_samp, u.uv_offset + u.uv_scale * base_uv).a;
    } else {
        // Tri-planar: blend the three axis samples' alpha by the surface normal.
        let local_normal = normalize((u.inv_transform * vec4<f32>(N_recv, 0.0)).xyz);
        var w = pow(abs(local_normal), vec3<f32>(u.tri_blend_sharpness));
        w = w / (w.x + w.y + w.z);
        let a_xy = textureSample(decal_tex, decal_samp, u.uv_offset + u.uv_scale * (local.xy + vec2<f32>(0.5))).a;
        let a_xz = textureSample(decal_tex, decal_samp, u.uv_offset + u.uv_scale * (local.xz + vec2<f32>(0.5))).a;
        let a_yz = textureSample(decal_tex, decal_samp, u.uv_offset + u.uv_scale * (local.yz + vec2<f32>(0.5))).a;
        tex_alpha = a_xy * w.z + a_xz * w.y + a_yz * w.x;
    }

    // Silhouette at the half-alpha contour. `u.alpha` (overall opacity) is
    // intentionally excluded so a faint decal still outlines its full shape.
    if tex_alpha * edge_alpha < 0.5 {
        discard;
    }

    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
