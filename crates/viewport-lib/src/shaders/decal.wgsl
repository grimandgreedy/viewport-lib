// Screen-space decal projection shader (D1 + D2 + D3 + D4 + D6 + D7 + D8).
//
// Group 0: camera_bgl (CameraUniform)
// Group 1: per-viewport scene depth texture (depth-only aspect view)
// Group 2: per-decal uniform + albedo + sampler + normal + roughness + metallic
//
// Vertex: full-screen quad (6 vertices, no vertex buffer)
// Fragment: load scene depth, reconstruct world position, project into decal local
//           space, sample texture, optionally perturb the normal with a normal
//           map, and light the result with the scene's lights and shadows.

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

// Scene lights, shared with the opaque pass (same group 0 layout). The decal
// runs the same lighting model the opaque pass runs, so a decal sits in the
// scene's light instead of reading as a sticker pasted over it.
// #include "helpers/scene_lighting.wgsl"
@group(0) @binding(3) var<uniform> lights_uniform: Lights;

// Cascaded shadow map for the primary directional light, same bindings the
// mesh shaders use.
@group(0) @binding(1) var shadow_map:     texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

struct ShadowAtlas {
    cascade_vp: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    cascade_count: u32,
    atlas_size: f32,
    shadow_filter: u32,
    pcss_light_radius: f32,
    atlas_rects: array<vec4<f32>, 8>,
};
@group(0) @binding(5) var<uniform> shadow_atlas: ShadowAtlas;

// Scene depth written by the opaque pass (depth-only aspect, Depth24PlusStencil8).
@group(1) @binding(0) var scene_depth:   texture_depth_2d;
// Scene stencil written by the opaque pass (stencil-only aspect, Depth24PlusStencil8).
// Value 1 = receives decals, 0 = excluded.
@group(1) @binding(1) var scene_stencil: texture_2d<u32>;

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
    // D6
    emissive:              f32,   // emissive intensity multiplier
    has_emissive_tex:      u32,   // 1 when emissive_tex is bound
    // D7
    edge_fade:             f32,   // [0, 0.5]: fraction of half-extent over which alpha fades
    ambient:               f32,   // constant ambient coefficient, matching Material::ambient
    // D8
    projection:            u32,   // 0 = Planar, 1 = TriPlanar
    tri_blend_sharpness:   f32,
    _pad2:                 u32,
    _pad3:                 u32,
};

@group(2) @binding(0) var<uniform> u:             DecalUniform;
@group(2) @binding(1) var          decal_tex:     texture_2d<f32>;
@group(2) @binding(2) var          decal_samp:    sampler;
@group(2) @binding(3) var          decal_normal:  texture_2d<f32>;  // D2
@group(2) @binding(4) var          roughness_tex: texture_2d<f32>;  // D3
@group(2) @binding(5) var          metallic_tex:  texture_2d<f32>;  // D3
@group(2) @binding(6) var          emissive_tex:  texture_2d<f32>;  // D6

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

// Cascade selection and filtering for the primary light's shadow, and the
// Cook-Torrance BRDF. Both are the copies the mesh shaders use, so a decal and
// the surface it lands on are lit and shadowed by the same code.
// #include "helpers/csm.wgsl"
// #include "helpers/brdf.wgsl"

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Load scene depth at the current pixel.
    let pix   = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
    let depth = textureLoad(scene_depth, pix, 0);

    // D5: stencil 0 means this surface is marked non-receiver -- skip it.
    let stencil = textureLoad(scene_stencil, pix, 0).r;
    if stencil == 0u { discard; }

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

    // Reject fragments outside the projection volume.
    // Cylindrical: radial distance in XY <= 0.5, Z within [-0.5, 0.5].
    // Planar / TriPlanar: box test.
    if u.projection == 2u || u.projection == 3u {
        let r2 = local.x * local.x + local.y * local.y;
        if r2 > 0.25 || abs(local.z) > 0.5 { discard; }
    } else {
        if any(local < vec3<f32>(-0.5)) || any(local > vec3<f32>(0.5)) {
            discard;
        }
    }

    // D7: compute edge fade from local-space coordinates.
    // Cylindrical: fade by radial distance and Z.
    // Planar / TriPlanar: fade by each box face.
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

    // Estimate the receiver surface normal from world-position screen derivatives.
    // Used for D2 normal-map shading and D3 specular -- not for the facing check.
    let ddx_w    = dpdx(world);
    let ddy_w    = dpdy(world);
    let n_raw    = normalize(cross(ddx_w, ddy_w));
    let view_dir = normalize(camera.eye_pos - world);
    let N_recv   = select(-n_raw, n_raw, dot(n_raw, view_dir) > 0.0);

    // Specular anti-aliasing kernel, taken in uniform control flow before any
    // of the shading branches. The receiver normal here comes from depth
    // derivatives, which step in bands as depth precision runs out; a tight
    // highlight lands on those steps and reads as striping. Widening the lobe
    // where the normal changes fast across the pixel absorbs it.
    let n_du = dpdx(N_recv);
    let n_dv = dpdy(N_recv);
    let saa_kernel = min(0.5 * (dot(n_du, n_du) + dot(n_dv, n_dv)), 0.18);

    // Decal projection axis (column 2 of model matrix = world-space local Z,
    // extracted from rows of inv_transform via inverse-transpose).
    let decal_Z = normalize(vec3<f32>(u.inv_transform[0][2], u.inv_transform[1][2], u.inv_transform[2][2]));

    // Planar-only: reject fragments where the surface or camera doesn't face
    // the decal projection axis.  Skipped for tri-planar (all axes are used).
    //
    // The surface-normal check (N_recv . decal_Z) keeps the projection off
    // geometry that is perpendicular to the projection direction, where the UV
    // collapses to a constant coordinate and the decal smears into a uniform
    // stripe instead of showing its texture.
    // A threshold of 0.1 rejects surfaces more than ~84 degrees off-axis while
    // allowing slightly curved or low-angle receivers.
    if u.projection == 0u {
        if dot(N_recv, decal_Z) < 0.1 { discard; }
        if dot(view_dir, decal_Z) < 0.05 { discard; }
    }

    // D9: Cylindrical facing check.
    // Transform receiver normal into decal local space and test its XY radial
    // component against the surface position to verify the surface faces the
    // correct side of the cylinder.
    if u.projection == 2u || u.projection == 3u {
        let local_normal = normalize((u.inv_transform * vec4<f32>(N_recv, 0.0)).xyz);
        let r = sqrt(local.x * local.x + local.y * local.y);
        if r > 0.001 {
            let radial_dir = vec2<f32>(local.x, local.y) / r;
            let radial_dot = dot(local_normal.xy, radial_dir);
            if u.projection == 2u {
                // Outward: normal should point away from axis.
                if radial_dot < 0.1 { discard; }
            } else {
                // Inward: normal should point toward axis.
                if radial_dot > -0.1 { discard; }
            }
        }
    }

    // D8/D9: sample texture.
    // Planar: standard XY projection.
    // TriPlanar: blend three orthogonal projections weighted by the surface normal.
    // Cylindrical: angle around Z axis -> UV.x; position along Z -> UV.y.
    var uv:      vec2<f32>;
    var tex_col: vec4<f32>;

    if u.projection == 0u {
        // D4: apply UV scale + offset. Base UV maps local XY from [-0.5, 0.5] to [0, 1].
        let base_uv = local.xy + vec2<f32>(0.5);
        uv      = u.uv_offset + u.uv_scale * base_uv;
        tex_col = textureSample(decal_tex, decal_samp, uv);
    } else if u.projection == 2u || u.projection == 3u {
        // D9: cylindrical -- angle around local Z axis, length along local Z.
        let angle   = atan2(local.y, local.x);
        let base_uv = vec2<f32>(angle / (2.0 * 3.14159265) + 0.5, local.z + 0.5);
        uv      = u.uv_offset + u.uv_scale * base_uv;
        tex_col = textureSample(decal_tex, decal_samp, uv);
    } else {
        // Transform the world-space receiver normal into decal local space.
        // Applying inv_transform as a direction transform (not a proper normal
        // transform) is correct when the decal matrix has no shear; normalization
        // handles non-uniform scale.
        let local_normal = normalize((u.inv_transform * vec4<f32>(N_recv, 0.0)).xyz);

        // Blend weights: each axis's weight is proportional to how much the
        // surface faces that axis.
        var w = pow(abs(local_normal), vec3<f32>(u.tri_blend_sharpness));
        w = w / (w.x + w.y + w.z);

        let uv_xy = u.uv_offset + u.uv_scale * (local.xy + vec2<f32>(0.5));
        let uv_xz = u.uv_offset + u.uv_scale * (local.xz + vec2<f32>(0.5));
        let uv_yz = u.uv_offset + u.uv_scale * (local.yz + vec2<f32>(0.5));

        let c_xy = textureSample(decal_tex, decal_samp, uv_xy);
        let c_xz = textureSample(decal_tex, decal_samp, uv_xz);
        let c_yz = textureSample(decal_tex, decal_samp, uv_yz);

        tex_col = c_xy * w.z + c_xz * w.y + c_yz * w.x;
        uv = uv_xy; // fallback UV for other texture slots (normal, roughness, emissive)
    }

    let alpha = tex_col.a * u.alpha * edge_alpha;

    if alpha < 0.001 {
        discard;
    }

    var out_rgb = tex_col.rgb;

    // Perturbed shading normal: the receiver normal bent by the decal normal
    // map when one is bound, else the receiver normal itself.
    var N_final = N_recv;
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
        N_final = normalize(mix(N_recv, N_decal, u.normal_blend_strength));
    }

    let roughness_val = select(u.roughness,
                               textureSample(roughness_tex, decal_samp, uv).r,
                               u.has_roughness_tex != 0u);
    let metallic_val  = select(u.metallic,
                               textureSample(metallic_tex,  decal_samp, uv).r,
                               u.has_metallic_tex  != 0u);

    // Light the decal.
    //
    // The decal texture holds albedo, and it composites over a receiver that the
    // opaque pass has already lit. Writing the albedo straight out puts the two
    // on different scales: the decal ignores how bright the surface under it is,
    // so it neither darkens in shadow nor follows the key light, and a decal
    // authored against one rig reads wrong under any other. Running it through
    // the same Cook-Torrance loop the opaque pass runs puts the decal's own
    // colour in radiance units before the blend.
    //
    // Only `Replace` is lit. `Multiply` and `Additive` do not carry an albedo:
    // multiply scales a receiver colour that already has the lighting in it, and
    // additive adds emitted light. Lighting either would apply the scene's light
    // twice.
    if u.blend_mode == 0u {
        let albedo   = tex_col.rgb;
        let rough    = specular_aa_roughness_kernel(clamp(roughness_val, 0.04, 1.0), saa_kernel);
        let metallic = clamp(metallic_val, 0.0, 1.0);
        let F0       = mix(vec3<f32>(0.04), albedo, metallic);

        var lo = vec3<f32>(0.0);
        let range = cluster_light_range(world, lights_uniform.count);
        for (var j = 0u; j < range.count; j = j + 1u) {
            let idx = cluster_light_global(range, j);
            let light = lights_storage[idx];
            let ev = eval_light(light, world);
            if !ev.in_range { continue; }
            // pbr_light_contrib returns zero on the back hemisphere; skipping
            // early also skips the shadow taps.
            if dot(N_final, ev.l) <= 0.0 { continue; }
            var radiance = ev.radiance;
            // Shadowing follows the primary directional light only. Point-light
            // shadow cubes are not sampled here, so a decal inside a point
            // light's shadow still takes that light.
            if lights_uniform.shadows_enabled != 0u && idx == 0u && light.light_type != 1u {
                // Bias against the geometric receiver normal, not the
                // normal-mapped one: the depth buffer the decal reads was
                // written by the geometry.
                let s = sample_shadow_csm(world, camera.eye_pos, N_recv, ev.l, 0u);
                radiance = radiance * s.factor;
            }
            lo = lo + pbr_light_contrib(
                N_final, view_dir, ev.l, radiance, albedo, metallic, rough, F0,
            );
        }

        // Hemisphere ambient, matching the opaque pass's non-IBL branch. Image
        // based lighting is not sampled here, so in an IBL-lit scene a decal
        // takes the hemisphere fill where the surface under it takes the
        // environment.
        let hemi_t = clamp(N_final.z * 0.5 + 0.5, 0.0, 1.0);
        let hemi   = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t)
                     * lights_uniform.hemisphere_intensity;
        let ambient = (vec3<f32>(u.ambient) + hemi)
                      * (albedo * (1.0 - metallic) + F0 * metallic);

        out_rgb = lo + ambient;
    }

    // D6: emissive contribution -- always additive on top of the blend result.
    if u.emissive > 0.0 {
        let emissive_col = select(out_rgb,
                                  textureSample(emissive_tex, decal_samp, uv).rgb,
                                  u.has_emissive_tex != 0u);
        out_rgb = out_rgb + emissive_col * u.emissive;
    }

    return vec4<f32>(out_rgb, alpha);
}
