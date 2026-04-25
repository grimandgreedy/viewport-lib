// Ground plane shader — full-screen ray-march approach (no vertex buffer).
//
// Intersects the camera ray with a horizontal plane at a configurable Z height,
// then outputs one of four modes:
//   0 = None        (never reached — draw call is skipped)
//   1 = ShadowOnly  (invisible plane; shows shadow tint only where shadowed)
//   2 = Tile        (procedural checker pattern)
//   3 = SolidColor  (flat solid color)
//
// Uses @builtin(frag_depth) for correct depth-tested occlusion against scene geometry.
// Alpha blending is always enabled; ShadowOnly uses partial alpha, others use alpha=1.

struct GroundPlaneUniform {
    view_proj:      mat4x4<f32>,  // offset   0, 64 bytes — for clip-space depth output
    cam_right:      vec4<f32>,    // offset  64, 16 bytes — camera right axis (world space)
    cam_up:         vec4<f32>,    // offset  80, 16 bytes — camera up axis (world space)
    cam_back:       vec4<f32>,    // offset  96, 16 bytes — camera back axis (world space, = -forward)
    eye_pos:        vec3<f32>,    // offset 112, 12 bytes
    height:         f32,          // offset 124,  4 bytes — ground plane Z coordinate
    color:          vec4<f32>,    // offset 128, 16 bytes — Tile / SolidColor output color
    shadow_color:   vec4<f32>,    // offset 144, 16 bytes — ShadowOnly tint color
    light_vp:       mat4x4<f32>, // offset 160, 64 bytes — cascade 0 light-space view-proj
    tan_half_fov:   f32,          // offset 224,  4 bytes
    aspect:         f32,          // offset 228,  4 bytes
    tile_size:      f32,          // offset 232,  4 bytes
    shadow_bias:    f32,          // offset 236,  4 bytes
    mode:           u32,          // offset 240,  4 bytes
    shadow_opacity: f32,          // offset 244,  4 bytes
    _pad:           vec2<f32>,    // offset 248,  8 bytes
} // total 256 bytes

// Shadow atlas uniform — matches the mesh shader's ShadowAtlas struct exactly.
struct ShadowAtlas {
    cascade_vp:       array<mat4x4<f32>, 4>,  // 256 bytes
    cascade_splits:   vec4<f32>,               //  16 bytes
    cascade_count:    u32,                     //   4 bytes
    atlas_size:       f32,                     //   4 bytes
    shadow_filter:    u32,                     //   4 bytes
    pcss_light_radius: f32,                    //   4 bytes
    atlas_rects:      array<vec4<f32>, 8>,     // 128 bytes
} // total 416 bytes

@group(0) @binding(0) var<uniform> gp: GroundPlaneUniform;
@group(0) @binding(1) var shadow_tex: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform> shadow_atlas: ShadowAtlas;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc:      vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), p);
}

struct FragOut {
    @location(0)         color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    // Never render from below the ground plane — the plane is one-sided.
    if gp.eye_pos.z < gp.height { discard; }

    // Reconstruct world-space ray direction from NDC.
    let dir_cam = vec3<f32>(
        in.ndc.x * gp.aspect * gp.tan_half_fov,
        in.ndc.y * gp.tan_half_fov,
        -1.0,
    );
    // Rotate to world space using camera basis columns (right, up, back).
    let ray_dir = gp.cam_right.xyz * dir_cam.x
                + gp.cam_up.xyz    * dir_cam.y
                + gp.cam_back.xyz  * dir_cam.z;

    // Intersect with z = height plane.
    if abs(ray_dir.z) < 1e-6 { discard; }
    let t = (gp.height - gp.eye_pos.z) / ray_dir.z;
    if t <= 0.0 { discard; }

    let hit = gp.eye_pos + ray_dir * t;

    // Clip-space depth for the hit point.
    let hit_clip = gp.view_proj * vec4<f32>(hit, 1.0);
    let frag_depth = clamp(hit_clip.z / hit_clip.w, 0.0, 1.0);

    // Horizon fade — prevents razor-thin aliasing at grazing angles.
    let angle_sin = abs(ray_dir.z) / length(ray_dir);
    let fade = smoothstep(0.01, 0.08, angle_sin);
    if fade < 0.001 { discard; }

    var out_color: vec4<f32>;

    if gp.mode == 2u {
        // ------------------------------------------------------------------
        // Tile — checkerboard in world-space XY with screen-space LOD.
        // ------------------------------------------------------------------
        // Compute a uniform pixel_world estimate based on the camera nadir (the point directly
        // below the camera).  fwidth(in.ndc) is constant across the screen because NDC is a
        // linear mapping of pixel coordinates.  Multiplying by the eye height and fov projects
        // that into world-space units, giving the same LOD level for the whole plane at once.
        let ndc_fw         = fwidth(in.ndc);
        let eye_height     = max(gp.eye_pos.z - gp.height, 0.001);
        let nadir_px_world = max(ndc_fw.x * gp.aspect, ndc_fw.y) * eye_height * gp.tan_half_fov;

        // Smooth power-of-2 LOD: target ~64 screen pixels per tile.
        // scale is based purely on screen-space zoom — tile_size is NOT in the denominator.
        // n is therefore zoom-driven only; multiplying by tile_size afterwards means a
        // tile_size=2 plane always has tiles 2× larger on screen than tile_size=1,
        // at every zoom level, instead of converging to the same apparent size.
        let scale  = max(nadir_px_world * 64.0, 1.0);
        let log_s  = log2(scale);
        let n      = floor(log_s);
        let frac   = log_s - n;

        let fine_tile   = gp.tile_size * exp2(n);
        let coarse_tile = fine_tile * 2.0;

        // Integer parity checker at each LOD level.
        let uv_f = hit.xy / fine_tile;
        let uv_c = hit.xy / coarse_tile;
        let p_f = (i32(floor(uv_f.x)) + i32(floor(uv_f.y))) & 1;
        let p_c = (i32(floor(uv_c.x)) + i32(floor(uv_c.y))) & 1;

        // Standard 2-tone fine checker.
        let b_fine_std = mix(0.6, 1.0, f32(p_f));
        // 4-tone fine checker: coarse parity pre-emphasises cells that survive the next
        // LOD step — analogous to major grid lines being brighter than minor ones.
        let b_fine_emph = 0.55 + f32(p_f) * 0.30 + f32(p_c) * 0.15;
        // Coarse level: standard 2-tone.
        let b_coarse = mix(0.6, 1.0, f32(p_c));

        // Two separate curves, both late in the octave:
        //   emphasis_blend — fades the fine pattern from 2-tone to 4-tone.
        //   lod_blend      — then quickly crossfades fine to coarse.
        let emphasis_blend = smoothstep(0.6, 0.82, frac);
        let lod_blend      = smoothstep(0.82, 0.97, frac);

        let b_fine     = mix(b_fine_std, b_fine_emph, emphasis_blend);
        let brightness = mix(b_fine, b_coarse, lod_blend);

        let base = gp.color.rgb * brightness;
        out_color = vec4<f32>(base, gp.color.a * fade);

    } else if gp.mode == 3u {
        // ------------------------------------------------------------------
        // SolidColor.
        // ------------------------------------------------------------------
        out_color = vec4<f32>(gp.color.rgb, gp.color.a * fade);

    } else if gp.mode == 1u {
        // ------------------------------------------------------------------
        // ShadowOnly — sample shadow atlas with correct cascade selection.
        //
        // Mirrors the mesh shader's cascade-selection logic exactly:
        //   1. Camera-forward distance selects cascade index.
        //   2. Project hit into that cascade's clip space.
        //   3. Map tile UV through the cascade's atlas rect.
        //   4. 3×3 PCF.
        // ------------------------------------------------------------------

        // Camera-forward distance to the hit point (same metric the mesh shader uses).
        let cam_forward = -gp.cam_back.xyz;
        let dist = dot(hit - gp.eye_pos, cam_forward);

        var cascade_idx = 0u;
        for (var i = 0u; i < shadow_atlas.cascade_count; i++) {
            if dist > shadow_atlas.cascade_splits[i] {
                cascade_idx = i + 1u;
            }
        }
        cascade_idx = min(cascade_idx, shadow_atlas.cascade_count - 1u);

        // Project hit into the selected cascade's clip space.
        let light_clip = shadow_atlas.cascade_vp[cascade_idx] * vec4<f32>(hit, 1.0);
        let ndc        = light_clip.xyz / light_clip.w;

        // NDC → tile UV [0,1]  (Y negated: NDC Y+ up, texture V=0 at top).
        let tile_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

        // Out-of-range check (same as mesh shader).
        if tile_uv.x < 0.0 || tile_uv.x > 1.0 ||
           tile_uv.y < 0.0 || tile_uv.y > 1.0 ||
           ndc.z < 0.0 || ndc.z > 1.0 {
            discard;
        }

        // Map tile UV through the cascade's atlas rect.
        let rect       = shadow_atlas.atlas_rects[cascade_idx];
        let shadow_uv  = vec2<f32>(
            mix(rect.x, rect.z, tile_uv.x),
            mix(rect.y, rect.w, tile_uv.y),
        );
        let shadow_depth = ndc.z - gp.shadow_bias;

        // 3×3 PCF filter.
        let texel_size = 1.0 / shadow_atlas.atlas_size;
        var shadow_sum = 0.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let offset = vec2<f32>(f32(dx), f32(dy)) * texel_size;
                let clamped = clamp(shadow_uv + offset, rect.xy, rect.zw);
                shadow_sum += textureSampleCompare(
                    shadow_tex, shadow_sampler,
                    clamped,
                    shadow_depth,
                );
            }
        }
        let shadow_factor = shadow_sum / 9.0;
        // shadow_factor == 1.0 → lit, 0.0 → fully in shadow.
        let shadow_alpha = (1.0 - shadow_factor) * gp.shadow_opacity * fade;
        if shadow_alpha < 0.005 { discard; }
        out_color = vec4<f32>(gp.shadow_color.rgb, shadow_alpha);

    } else {
        discard;
    }

    return FragOut(out_color, frag_depth);
}
