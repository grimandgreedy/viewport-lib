// Debug visualization pass.
//
// Pasted inline in fs_main of each mesh shader variant (before the final return).
// Variables available in scope: N, final_rgb, in.world_pos, lights_uniform,
// shadow_atlas, camera, obj_colour.
// Modifies final_rgb in place when active (no early return, works for OIT too).

if lights_uniform.debug_vis_mode != 0u {
    let dbg_mode = (lights_uniform.debug_vis_mode >> 15u) & 0x3u;
    let dbg_r = lights_uniform.debug_vis_mode & 0x1fu;
    let dbg_g = (lights_uniform.debug_vis_mode >> 5u) & 0x1fu;
    let dbg_b = (lights_uniform.debug_vis_mode >> 10u) & 0x1fu;
    let dbg_scale = lights_uniform.debug_vis_scale;

    // Compute light direction for shadow quantities (use first light).
    var dbg_L = vec3<f32>(0.0, 0.0, 1.0);
    if lights_uniform.count > 0u {
        let l0 = lights_uniform.lights[0];
        if l0.light_type == 0u {
            dbg_L = normalize(l0.pos_or_dir);
        } else {
            dbg_L = normalize(l0.pos_or_dir - in.world_pos);
        }
    }

    // Cascade selection by view-space depth.
    let dbg_dist = dot(in.world_pos - camera.eye_pos, camera.forward);
    var dbg_casc = 0u;
    for (var ci = 0u; ci < shadow_atlas.cascade_count; ci++) {
        if dbg_dist > shadow_atlas.cascade_splits[ci] { dbg_casc = ci + 1u; }
    }
    dbg_casc = min(dbg_casc, shadow_atlas.cascade_count - 1u);

    // Project surface point into shadow space.
    let dbg_lclip = shadow_atlas.cascade_vp[dbg_casc] * vec4<f32>(in.world_pos, 1.0);
    let dbg_lndc = dbg_lclip.xyz / dbg_lclip.w;
    let dbg_tile_uv = vec2<f32>(dbg_lndc.x * 0.5 + 0.5, -dbg_lndc.y * 0.5 + 0.5);
    let dbg_rect = shadow_atlas.atlas_rects[dbg_casc];
    let dbg_atlas_uv = vec2<f32>(
        mix(dbg_rect.x, dbg_rect.z, dbg_tile_uv.x),
        mix(dbg_rect.y, dbg_rect.w, dbg_tile_uv.y),
    );

    // Normal bias and biased depth.
    let dbg_ndl_raw = dot(N, dbg_L);
    let dbg_rect_w = max(dbg_rect.z - dbg_rect.x, 0.0001);
    let dbg_texel_w = 2.0 / (shadow_atlas.cascade_vp[dbg_casc][0][0]
        * shadow_atlas.atlas_size * dbg_rect_w);
    let dbg_nbias = dbg_texel_w * mix(1.5, 0.0, clamp(abs(dbg_ndl_raw), 0.0, 1.0));
    let dbg_offset_w = in.world_pos + N * (select(-1.0, 1.0, dbg_ndl_raw >= 0.0) * dbg_nbias);
    let dbg_oclip = shadow_atlas.cascade_vp[dbg_casc] * vec4<f32>(dbg_offset_w, 1.0);
    let dbg_biased_d = (dbg_oclip.xyz / dbg_oclip.w).z - lights_uniform.shadow_bias;

    // Shadow factor (recomputed; only runs when debug mode is active).
    let dbg_shadow = sample_shadow_csm(in.world_pos, camera.eye_pos, N, dbg_L);
    let dbg_ndotl = clamp(dbg_ndl_raw, 0.0, 1.0);

    // Cascade index as a normalised 0..1 scalar.
    let dbg_ci_f = f32(dbg_casc) / max(f32(shadow_atlas.cascade_count) - 1.0, 1.0);

    // All quantities indexed by DebugQuantity enum value.
    // Index 4 (ContactShadowFactor) and 16/17 (Roughness/Metallic) are stubs.
    var dbg_vals: array<f32, 18>;
    dbg_vals[0]  = 0.0;                           // Zero
    dbg_vals[1]  = 1.0;                           // One
    dbg_vals[2]  = dbg_ci_f;                      // CascadeIndex (scalar; hue handled below)
    dbg_vals[3]  = dbg_shadow;                    // ShadowFactor
    dbg_vals[4]  = 0.5;                           // ContactShadowFactor (stub)
    dbg_vals[5]  = dbg_ndotl;                     // NdotL
    dbg_vals[6]  = dbg_nbias * dbg_scale;         // NormalBiasMagnitude
    dbg_vals[7]  = dbg_atlas_uv.x;               // AtlasUvX
    dbg_vals[8]  = dbg_atlas_uv.y;               // AtlasUvY
    dbg_vals[9]  = dbg_tile_uv.x;                // TileUvX
    dbg_vals[10] = dbg_tile_uv.y;                // TileUvY
    dbg_vals[11] = dbg_biased_d;                  // BiasedDepth
    dbg_vals[12] = dbg_lndc.z;                    // SurfaceDepth
    dbg_vals[13] = N.x * 0.5 + 0.5;              // WorldNormalX
    dbg_vals[14] = N.y * 0.5 + 0.5;              // WorldNormalY
    dbg_vals[15] = N.z * 0.5 + 0.5;              // WorldNormalZ
    dbg_vals[16] = 0.5;                           // Roughness (stub, D4)
    dbg_vals[17] = 0.5;                           // Metallic (stub, D4)

    // CascadeIndex (idx 2) outputs a distinct hue per cascade.
    let dbg_casc_hues = array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.2, 0.2),
        vec3<f32>(0.2, 1.0, 0.2),
        vec3<f32>(0.2, 0.4, 1.0),
        vec3<f32>(1.0, 1.0, 0.2),
    );
    let dbg_casc_hue = dbg_casc_hues[dbg_casc];

    let dbg_r_val = select(dbg_vals[dbg_r], dbg_casc_hue.r, dbg_r == 2u);
    let dbg_g_val = select(dbg_vals[dbg_g], dbg_casc_hue.g, dbg_g == 2u);
    let dbg_b_val = select(dbg_vals[dbg_b], dbg_casc_hue.b, dbg_b == 2u);

    let dbg_rgb = vec3<f32>(dbg_r_val, dbg_g_val, dbg_b_val);

    if dbg_mode == 1u {
        // TintOverlay: blend 50/50 with the normal render.
        final_rgb = mix(final_rgb, dbg_rgb, 0.5);
    } else {
        // Replace: override entirely.
        final_rgb = dbg_rgb;
    }
}
