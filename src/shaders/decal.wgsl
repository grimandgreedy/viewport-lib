// Screen-space decal projection shader (D1 + D2 + D3 + D4).
//
// Group 0: camera_bgl (CameraUniform)
// Group 1: per-viewport scene depth texture (depth-only aspect view)
// Group 2: per-decal uniform + albedo + sampler + normal + roughness + metallic
//
// Vertex: full-screen quad (6 vertices, no vertex buffer)
// Fragment: load scene depth, reconstruct world position, project into decal local
//           space, sample texture, optionally perturb shading via a normal map,
//           and apply roughness/metallic specular approximation.

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
    inv_transform:         mat4x4<f32>,
    blend_mode:            u32,   // 0 = Replace, 1 = Multiply
    alpha:                 f32,
    normal_blend_strength: f32,   // D2: [0, 1], 0 = no effect
    has_normal:            u32,   // D2: 1 when a normal map is bound
    // D3
    roughness:             f32,   // [0, 1]: 0 = mirror-smooth, 1 = fully matte
    metallic:              f32,   // [0, 1]: 0 = dielectric, 1 = metal
    has_roughness_tex:     u32,   // 1 when roughness_tex is bound
    has_metallic_tex:      u32,   // 1 when metallic_tex is bound
    // D4
    uv_offset:             vec2<f32>,  // added to final UV before sampling
    uv_scale:              vec2<f32>,  // scales final UV before offset (sprite sheet / scroll)
};

@group(2) @binding(0) var<uniform> u:             DecalUniform;
@group(2) @binding(1) var          decal_tex:     texture_2d<f32>;
@group(2) @binding(2) var          decal_samp:    sampler;
@group(2) @binding(3) var          decal_normal:  texture_2d<f32>;  // D2
@group(2) @binding(4) var          roughness_tex: texture_2d<f32>;  // D3
@group(2) @binding(5) var          metallic_tex:  texture_2d<f32>;  // D3

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
    // Load scene depth at the current pixel.
    let pix   = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
    let depth = textureLoad(scene_depth, pix, 0);

    // depth == 1.0 means background -- no surface.
    if depth >= 1.0 {
        discard;
    }

    // Reconstruct world-space position from NDC.
    let ndc     = vec4<f32>(in.ndc_xy, depth, 1.0);
    let world_h = camera.inv_view_proj * ndc;
    let world   = world_h.xyz / world_h.w;

    // Transform into decal local space. Projection volume is [-0.5, 0.5]^3.
    let local_h = u.inv_transform * vec4<f32>(world, 1.0);
    let local   = local_h.xyz;

    // Reject fragments outside the projection box.
    if any(local < vec3<f32>(-0.5)) || any(local > vec3<f32>(0.5)) {
        discard;
    }

    // Estimate the receiver surface normal from world-position screen derivatives.
    // Used for grazing-angle rejection, D2 normal-map shading, and D3 specular.
    let ddx_w    = dpdx(world);
    let ddy_w    = dpdy(world);
    let n_raw    = normalize(cross(ddx_w, ddy_w));
    let view_dir = normalize(camera.eye_pos - world);
    let N_recv   = select(-n_raw, n_raw, dot(n_raw, view_dir) > 0.0);

    // Decal projection axis (row 2 of inv_transform = world-space local Z).
    let decal_Z = normalize(vec3<f32>(u.inv_transform[0][2], u.inv_transform[1][2], u.inv_transform[2][2]));

    // Reject fragments on surfaces nearly perpendicular to the projection axis.
    // Prevents the decal from stretching at grazing angles.
    if abs(dot(N_recv, decal_Z)) < 0.25 {
        discard;
    }

    // D4: apply UV scale + offset (sprite sheet / scroll animation).
    // Base UV maps local XY from [-0.5, 0.5] to [0, 1].
    let base_uv = local.xy + vec2<f32>(0.5);
    let uv      = u.uv_offset + u.uv_scale * base_uv;

    let tex_col = textureSample(decal_tex, decal_samp, uv);
    let alpha   = tex_col.a * u.alpha;

    if alpha < 0.001 {
        discard;
    }

    var out_rgb = tex_col.rgb;

    // D2: Normal map shading perturbation.
    // Approximates the receiver normal from world-position screen derivatives,
    // applies the decal's tangent-space normal map, and scales the output colour
    // by the ratio of new-to-old N.V to mimic the change in surface lighting.
    if u.has_normal != 0u {
        // Derive the decal's world-space tangent frame from inv_transform.
        // Row i of inv_transform is proportional to world-space decal axis i
        // (the inverse-transpose property for normal/direction transforms).
        let decal_T = normalize(vec3<f32>(u.inv_transform[0][0], u.inv_transform[1][0], u.inv_transform[2][0]));
        let decal_B = normalize(vec3<f32>(u.inv_transform[0][1], u.inv_transform[1][1], u.inv_transform[2][1]));

        // Sample and decode the tangent-space normal map.
        let nmap_raw = textureSample(decal_normal, decal_samp, uv);
        let nmap     = normalize(nmap_raw.xyz * 2.0 - 1.0);

        // Rotate tangent-space normal into world space using the TBN frame
        // (N_recv acts as the TBN normal axis).
        let N_decal = normalize(decal_T * nmap.x + decal_B * nmap.y + N_recv * nmap.z);

        // Blend between receiver normal and decal normal.
        let N_final = normalize(mix(N_recv, N_decal, u.normal_blend_strength));

        // Modulate output colour by the ratio of N.V terms so the decal reads
        // as a lit surface indent/emboss rather than a flat sticker.
        // View direction serves as a proxy for the dominant scene light.
        let old_nv = max(dot(N_recv,  view_dir), 0.05);
        let new_nv = max(dot(N_final, view_dir), 0.05);
        out_rgb = out_rgb * clamp(new_nv / old_nv, 0.0, 4.0);
    }

    // D3: Roughness and metallic specular approximation.
    // Accurate PBR requires scene light data that is unavailable post-opaque.
    // This uses N.V as a retroreflection proxy: at low roughness a tight highlight
    // appears at near-normal incidence; metallic tints the highlight by albedo.
    let roughness_val = select(u.roughness,
                               textureSample(roughness_tex, decal_samp, uv).r,
                               u.has_roughness_tex != 0u);
    let metallic_val  = select(u.metallic,
                               textureSample(metallic_tex,  decal_samp, uv).r,
                               u.has_metallic_tex  != 0u);

    let gloss = 1.0 - roughness_val;
    if gloss > 0.01 {
        let NoV = max(dot(N_recv, view_dir), 0.0);
        // Phong exponent: increases quadratically with gloss for tight highlights.
        let spec_exp       = max(gloss * gloss * 128.0, 1.0);
        let spec_intensity = pow(NoV, spec_exp) * gloss;
        // Dielectric: white highlight. Metal: highlight tinted by albedo colour.
        let spec_color = mix(vec3<f32>(0.95), out_rgb, metallic_val);
        out_rgb = out_rgb + spec_color * spec_intensity;
    }

    return vec4<f32>(out_rgb, alpha);
}
