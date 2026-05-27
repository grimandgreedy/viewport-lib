// Debug visualization pass.
// Pasted inline in fs_main after lighting and emissive.
// Reads last_shadow_sample (filled by sample_shadow_csm in the light loop).
// Modifies final_rgb in place; no early return so OIT shaders work too.

if lights_uniform.debug_vis_mode != 0u {
    let dbg_mode = (lights_uniform.debug_vis_mode >> 15u) & 0x3u;
    let dbg_r = lights_uniform.debug_vis_mode & 0x1fu;
    let dbg_g = (lights_uniform.debug_vis_mode >> 5u) & 0x1fu;
    let dbg_b = (lights_uniform.debug_vis_mode >> 10u) & 0x1fu;
    let dbg_scale = lights_uniform.debug_vis_scale;

    // Recompute light direction for quantities not in ShadowSample (NdotL uses N).
    var dbg_L = vec3<f32>(0.0, 0.0, 1.0);
    if lights_uniform.count > 0u {
        let l0 = lights_uniform.lights[0];
        if l0.light_type == 0u {
            dbg_L = normalize(l0.pos_or_dir);
        } else {
            dbg_L = normalize(l0.pos_or_dir - in.world_pos);
        }
    }
    let dbg_ndotl = clamp(dot(N, dbg_L), 0.0, 1.0);

    // Read shadow internals directly from the struct captured during the light loop.
    let dbg_shadow      = last_shadow_sample.factor;
    let dbg_casc        = last_shadow_sample.cascade_idx;
    let dbg_atlas_uv    = last_shadow_sample.atlas_uv;
    let dbg_tile_uv     = last_shadow_sample.tile_uv;
    let dbg_biased_d    = last_shadow_sample.biased_depth;
    let dbg_surface_d   = last_shadow_sample.surface_depth;
    let dbg_nbias       = last_shadow_sample.normal_bias_world;

    let dbg_ci_f = f32(dbg_casc) / max(f32(shadow_atlas.cascade_count) - 1.0, 1.0);

    var dbg_vals: array<f32, 24>;
    dbg_vals[0]  = 0.0;
    dbg_vals[1]  = 1.0;
    dbg_vals[2]  = dbg_ci_f;                          // CascadeIndex (scalar; hue handled below)
    dbg_vals[3]  = dbg_shadow;                        // ShadowFactor
    dbg_vals[4]  = 0.5;                               // ContactShadowFactor stub
    dbg_vals[5]  = dbg_ndotl;                         // NdotL
    dbg_vals[6]  = dbg_nbias * dbg_scale;             // NormalBiasMagnitude
    dbg_vals[7]  = dbg_atlas_uv.x;                   // AtlasUvX
    dbg_vals[8]  = dbg_atlas_uv.y;                   // AtlasUvY
    dbg_vals[9]  = dbg_tile_uv.x;                    // TileUvX
    dbg_vals[10] = dbg_tile_uv.y;                    // TileUvY
    dbg_vals[11] = dbg_biased_d;                      // BiasedDepth
    dbg_vals[12] = dbg_surface_d;                     // SurfaceDepth
    dbg_vals[13] = N.x * 0.5 + 0.5;                  // WorldNormalX
    dbg_vals[14] = N.y * 0.5 + 0.5;                  // WorldNormalY
    dbg_vals[15] = N.z * 0.5 + 0.5;                  // WorldNormalZ
    dbg_vals[16] = dbg_roughness;                     // Roughness
    dbg_vals[17] = dbg_metallic;                      // Metallic
    dbg_vals[18] = ao_factor;                         // AoFactor
    dbg_vals[19] = dbg_direct_lum * dbg_scale;        // DirectLightLuminance
    dbg_vals[20] = dbg_ambient_lum * dbg_scale;       // AmbientLuminance
    dbg_vals[21] = dbg_ibl_diff_lum * dbg_scale;     // IblDiffuseLuminance
    dbg_vals[22] = dbg_ibl_spec_lum * dbg_scale;     // IblSpecularLuminance
    dbg_vals[23] = dbg_emissive_lum * dbg_scale;     // EmissiveLuminance

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
        final_rgb = mix(final_rgb, dbg_rgb, 0.5);
    } else if dbg_mode == 2u {
        let split_px = lights_uniform.debug_vis_split_x * clip_planes.viewport_width;
        let px = in.clip_pos.x;
        if abs(px - split_px) < 1.5 {
            final_rgb = vec3<f32>(1.0);
        } else if px > split_px {
            final_rgb = dbg_rgb;
        }
        // left of split: final_rgb unchanged
    } else {
        final_rgb = dbg_rgb;
    }

    // Write to per-fragment storage buffer for pixel inspector readback.
    // Uses the window-space position from @builtin(position).
    let dbg_px = u32(in.clip_pos.x);
    let dbg_py = u32(in.clip_pos.y);
    let dbg_buf_stride = u32(clip_planes.viewport_width);
    let dbg_buf_idx = dbg_py * dbg_buf_stride + dbg_px;
    let dbg_buf_len = arrayLength(&debug_frag_buf);
    if dbg_buf_idx < dbg_buf_len {
        debug_frag_buf[dbg_buf_idx] = vec4<f32>(dbg_rgb, 1.0);
    }
}
