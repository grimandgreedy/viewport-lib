// ---------------------------------------------------------------------------
// The PBR surface model the lit mesh shaders (mesh, mesh_oit) shade with.
//
// `pbr_resolve` turns the material inputs into a `PbrPixel`, `pbr_direct`
// evaluates one light on it and `pbr_ambient` gives the indirect term. The
// light loops, their shadow logic and the shade-slot markers stay in each
// shader; only the maths lives here.
//
// Each mesh shader includes this file itself, next to brdf.wgsl and
// ambient.wgsl. `pbr_ambient` reads the including shader's `lights_uniform`,
// `object`, `lightmap_tex`, `obj_sampler`, `FEAT_LIGHTMAP`, `FEAT_LIGHT_PROBE`
// and `lightmap_directional_factor`; the layer helpers read `FEAT_CLEARCOAT`,
// `FEAT_SHEEN` and `FEAT_ANISOTROPY`. Nothing here takes a screen-space
// derivative: the caller supplies the specular AA kernel and the reflection
// footprint from uniform control flow.
// ---------------------------------------------------------------------------

// Floor on perceptual roughness, so a mirror-smooth material keeps a finite
// GGX lobe.
const MIN_PERCEPTUAL_ROUGHNESS: f32 = 0.04;

// One fragment's resolved PBR surface.
struct PbrPixel {
    // The base colour the diffuse lobe scales. The (1 - metallic) weight is
    // applied with the Fresnel term, not folded in here.
    diffuse_colour: vec3<f32>,
    // Specular reflectance at normal and grazing incidence.
    f0: vec3<f32>,
    f90: f32,
    // The material's dielectric reflectance at normal and grazing incidence,
    // before the metallic mix: f0 and f90 are recomputed from them when a
    // hook overrides metallic.
    dielectric_f0: vec3<f32>,
    dielectric_f90: f32,
    metallic: f32,
    // Roughness after the floor and specular anti-aliasing.
    perceptual_roughness: f32,
    // GGX alpha: perceptual_roughness squared.
    alpha: f32,
    // Split-sum DFG terms (scale, bias): the BRDF LUT at (n_dot_v,
    // perceptual_roughness).
    dfg: vec2<f32>,
    // Multiple-scattering energy compensation for the specular lobe, from f0
    // and dfg.
    energy_comp: vec3<f32>,
    // max(N.V, 1e-3) for the resolved normal.
    n_dot_v: f32,
    // Material ambient occlusion: the diffuse visibility, and the specular
    // occlusion derived from it, N.V and alpha (spec_occlusion).
    diffuse_ao: f32,
    spec_ao: f32,
    // The clear coat, sheen and anisotropy over the base; zero flags without
    // them. Filled by pbr_resolve_layers.
    layers: ViewportBrdfLayers,
};

// Resolve the material inputs: clamp metallic, floor roughness, apply the
// metallic-roughness map's remap when `has_mr_map` (G = roughness,
// B = metallic, each mixed across its range before the scalar factor), widen
// the roughness by the specular AA kernel, then derive F0 and look up the
// split-sum terms and the energy compensation. `N` and `V` are the shading
// normal and the unit vector toward the eye; `dielectric_f0` and
// `dielectric_f90` the material's dielectric reflectance at normal and grazing
// incidence, which metallic mixes toward the base colour and 1.
fn pbr_resolve(
    N: vec3<f32>,
    V: vec3<f32>,
    base_colour: vec3<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    has_mr_map: bool,
    mr_sample: vec2<f32>,
    metallic_range: vec2<f32>,
    roughness_range: vec2<f32>,
    saa_kernel: f32,
    ao: f32,
    dielectric_f0: vec3<f32>,
    dielectric_f90: f32,
) -> PbrPixel {
    var metallic = clamp(metallic_factor, 0.0, 1.0);
    var roughness = max(roughness_factor, MIN_PERCEPTUAL_ROUGHNESS);
    if has_mr_map {
        let m_remapped = mix(metallic_range.x, metallic_range.y, mr_sample.y);
        let r_remapped = mix(roughness_range.x, roughness_range.y, mr_sample.x);
        metallic = clamp(m_remapped * metallic, 0.0, 1.0);
        roughness = max(r_remapped * roughness, MIN_PERCEPTUAL_ROUGHNESS);
    }
    roughness = viewport_brdf_specular_aa_roughness_kernel(roughness, saa_kernel);

    var px: PbrPixel;
    px.diffuse_colour = base_colour;
    px.metallic = metallic;
    px.perceptual_roughness = roughness;
    px.alpha = roughness * roughness;
    px.dielectric_f0 = dielectric_f0;
    px.dielectric_f90 = dielectric_f90;
    px.f0 = mix(dielectric_f0, base_colour, metallic);
    px.f90 = pbr_f90(dielectric_f90, metallic);
    px.n_dot_v = max(dot(N, V), 1e-3);
    px.dfg = sample_brdf_lut(px.n_dot_v, roughness);
    px.energy_comp = viewport_brdf_energy_compensation(px.f0, px.dfg);
    px.diffuse_ao = ao;
    px.spec_ao = spec_occlusion(px.n_dot_v, ao, px.alpha);
    return px;
}

// Grazing reflectance: the dielectric value mixed toward 1 by metallic. Written
// out rather than with mix() so a dielectric F90 of 1 stays exactly 1.
fn pbr_f90(dielectric_f90: f32, metallic: f32) -> f32 {
    return dielectric_f90 + (1.0 - dielectric_f90) * metallic;
}

// Replace the surface terms of a resolved pixel with authored ones (a shading
// hook's `shade_surface` output). No specular AA: the author owns the value.
// `SURFACE_OVERRIDE_AO` in `ov.flags` replaces the material AO with `ov.ao`
// and `SURFACE_OVERRIDE_F0` the dielectric F0 with `ov.f0`; without them
// both are kept. The split-sum terms and the specular occlusion are derived
// again at the resolved N.V.
fn pbr_apply_override(px: PbrPixel, ov: SurfaceOverride) -> PbrPixel {
    var out = px;
    out.diffuse_colour = ov.base_colour;
    out.metallic = clamp(ov.metallic, 0.0, 1.0);
    out.perceptual_roughness = max(ov.roughness, MIN_PERCEPTUAL_ROUGHNESS);
    out.alpha = out.perceptual_roughness * out.perceptual_roughness;
    if (ov.flags & SURFACE_OVERRIDE_AO) != 0u {
        out.diffuse_ao = clamp(ov.ao, 0.0, 1.0);
    }
    if (ov.flags & SURFACE_OVERRIDE_F0) != 0u {
        out.dielectric_f0 = ov.f0;
    }
    out.f0 = mix(out.dielectric_f0, ov.base_colour, out.metallic);
    out.f90 = pbr_f90(px.dielectric_f90, out.metallic);
    out.dfg = sample_brdf_lut(out.n_dot_v, out.perceptual_roughness);
    out.energy_comp = viewport_brdf_energy_compensation(out.f0, out.dfg);
    out.spec_ao = spec_occlusion(out.n_dot_v, out.diffuse_ao, out.alpha);
    return out;
}

// The layer bits this pipeline compiles in.
fn pbr_layer_features() -> u32 {
    return select(0u, VIEWPORT_BRDF_LAYER_CLEARCOAT, FEAT_CLEARCOAT)
         | select(0u, VIEWPORT_BRDF_LAYER_SHEEN, FEAT_SHEEN)
         | select(0u, VIEWPORT_BRDF_LAYER_ANISOTROPY, FEAT_ANISOTROPY);
}

// Whether this pipeline compiles any layer in.
fn pbr_layers_compiled() -> bool {
    return FEAT_CLEARCOAT || FEAT_SHEEN || FEAT_ANISOTROPY;
}

// Read the material's layers onto a resolved pixel. `layers0` and `layers1`
// are the material block's lanes and `item_flags` its flag word (layer bits
// 8..10). The coat raises the base roughness toward its own and the
// split-sum terms are derived again; the sheen takes its albedo from the
// BRDF LUT's blue lane; anisotropy turns the tangent frame from
// `world_tangent` (a frame built from N when the mesh has no tangent) by
// its rotation. Runs after any surface hook, so an authored roughness is
// coated too.
fn pbr_resolve_layers(
    px: PbrPixel,
    N: vec3<f32>,
    world_tangent: vec4<f32>,
    layers0: vec4<f32>,
    layers1: vec4<f32>,
    item_flags: u32,
    saa_kernel: f32,
) -> PbrPixel {
    var out = px;
    let flags = (item_flags >> 8u) & pbr_layer_features();
    out.layers.flags = flags;
    if (flags & VIEWPORT_BRDF_LAYER_CLEARCOAT) != 0u {
        let coat_r = viewport_brdf_specular_aa_roughness_kernel(
            max(layers0.y, MIN_PERCEPTUAL_ROUGHNESS), saa_kernel);
        out.layers.clearcoat = layers0.x;
        out.layers.clearcoat_roughness = coat_r;
        let r = mix(px.perceptual_roughness, max(px.perceptual_roughness, coat_r), layers0.x);
        out.perceptual_roughness = r;
        out.alpha = r * r;
        out.dfg = sample_brdf_lut(out.n_dot_v, r);
        out.energy_comp = viewport_brdf_energy_compensation(out.f0, out.dfg);
        out.spec_ao = spec_occlusion(out.n_dot_v, out.diffuse_ao, out.alpha);
    }
    if (flags & VIEWPORT_BRDF_LAYER_SHEEN) != 0u {
        let sr = max(layers1.w, MIN_PERCEPTUAL_ROUGHNESS);
        out.layers.sheen_colour = layers1.xyz;
        out.layers.sheen_roughness = sr;
        out.layers.sheen_dfg = sample_brdf_lut_sheen(out.n_dot_v, sr);
        out.layers.sheen_scaling = viewport_brdf_sheen_scaling(layers1.xyz, out.layers.sheen_dfg);
    }
    if (flags & VIEWPORT_BRDF_LAYER_ANISOTROPY) != 0u {
        var t = world_tangent.xyz - dot(world_tangent.xyz, N) * N;
        let t_len = length(t);
        if t_len > 1e-5 {
            t = t / t_len;
        } else {
            let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(N.z) < 0.9);
            t = normalize(cross(up, N));
        }
        let handedness = select(world_tangent.w, 1.0, world_tangent.w == 0.0);
        let b = cross(N, t) * handedness;
        let c = cos(layers0.w);
        let s = sin(layers0.w);
        out.layers.anisotropy = layers0.z;
        out.layers.anisotropy_t = t * c + b * s;
        out.layers.anisotropy_b = b * c - t * s;
    }
    return out;
}

// The normal the environment's reflection is taken about: N, or for an
// anisotropic surface N bent toward the plane across the stretch (Filament's
// bent reflection vector).
fn pbr_reflection_normal(px: PbrPixel, N: vec3<f32>, V: vec3<f32>) -> vec3<f32> {
    if !FEAT_ANISOTROPY || (px.layers.flags & VIEWPORT_BRDF_LAYER_ANISOTROPY) == 0u {
        return N;
    }
    let a = px.layers.anisotropy;
    let dir = select(px.layers.anisotropy_t, px.layers.anisotropy_b, a >= 0.0);
    let at = cross(dir, V);
    let an = cross(at, dir);
    let bend = abs(a) * clamp(5.0 * px.perceptual_roughness, 0.0, 1.0);
    return normalize(mix(N, an, bend));
}

// Micro-shadowing (Chan 2018): material AO read as a cone of visibility, so a
// cavity is shadowed where the light grazes it. Multiplies a directional
// light's shadow factor; 1 where N.L clears the cone.
fn pbr_micro_shadow(n_dot_l: f32, ao: f32) -> f32 {
    let aperture = inverseSqrt(1.0 - min(ao, 0.9999));
    let ms = clamp(n_dot_l * aperture, 0.0, 1.0);
    return ms * ms;
}

// One light's direct contribution. `radiance` is the light's colour times its
// attenuation and shadow. The specular lobe carries the energy compensation.
// With layers, the sheen and the coat cover the base. Exactly zero when
// N.L <= 0.
fn pbr_direct(px: PbrPixel, N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    if pbr_layers_compiled() && px.layers.flags != 0u {
        var ly = px.layers;
        ly.flags = ly.flags & pbr_layer_features();
        return viewport_brdf_light_contrib_layered(N, V, L, radiance, px.diffuse_colour,
                                                   px.metallic, px.perceptual_roughness,
                                                   px.f0, px.f90, px.energy_comp, ly);
    }
    return viewport_brdf_light_contrib_f90(N, V, L, radiance, px.diffuse_colour, px.metallic,
                                           px.perceptual_roughness, px.f0, px.f90,
                                           px.energy_comp);
}

struct AmbientOut {
    // The whole indirect term.
    ambient: vec3<f32>,
    // The environment's diffuse and specular (zero without an environment).
    // A light probe or lightmap that replaces the diffuse keeps the specular.
    ibl_diffuse: vec3<f32>,
    ibl_specular: vec3<f32>,
    // The weight the environment's diffuse gets: (1 - F) * (1 - metallic).
    ambient_kd: vec3<f32>,
    // Luminance of the indirect term for the debug overlay. With only the
    // environment this is the diffuse and specular luminances summed.
    ambient_lum: f32,
};

// The indirect term, exposed: the environment (zoned when zones exist) or the
// hemisphere ambient, then a light probe replacing the diffuse, then the
// baked lightmap. `vertex_normal` is the interpolated, unnormalised vertex
// normal, `refl_dr` the reflection footprint and `lightmap_shadow` the main
// light's shadow factor, which darkens a shadow-occluded bake.
fn pbr_ambient(
    px: PbrPixel,
    N: vec3<f32>,
    V: vec3<f32>,
    vertex_normal: vec3<f32>,
    world_pos: vec3<f32>,
    lightmap_uv: vec2<f32>,
    lightmap_page: f32,
    refl_dr: f32,
    lightmap_shadow: f32,
) -> AmbientOut {
    let lum_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    let base_colour = px.diffuse_colour;
    let metallic = px.metallic;
    let roughness = px.perceptual_roughness;
    let F0 = px.f0;
    // Diffuse terms take the visibility, the environment's specular the
    // specular occlusion; each through the multi-bounce fit when enabled,
    // over the diffuse albedo and F0 respectively.
    let multi_bounce = (lights_uniform.ao_flags & AO_FLAG_MULTI_BOUNCE) != 0u;
    let diffuse_occ = ao_occlusion(px.diffuse_ao, base_colour * (1.0 - metallic), multi_bounce);
    let spec_occ = ao_occlusion(px.spec_ao, F0, multi_bounce);

    var out: AmbientOut;
    var ambient: vec3<f32>;
    var ibl_specular = vec3<f32>(0.0);
    var ibl_diffuse = vec3<f32>(0.0);
    var ambient_lum = 0.0;
    let ambient_kd = (vec3<f32>(1.0) - F_Schlick_roughness_f90(max(dot(N, V), 0.001), F0, px.f90,
                                                                roughness))
        * (1.0 - metallic);
    let Nr = pbr_reflection_normal(px, N, V);
    if lights_uniform.ibl_enabled != 0u {
        var ibl: IblContrib;
        if lights_uniform.env_zone_count != 0u {
            ibl = ibl_ambient_zoned(N, Nr, V, base_colour, metallic, roughness, F0, px.f90,
                                    diffuse_occ, spec_occ, lights_uniform.ibl_intensity,
                                    lights_uniform.ibl_rotation, refl_dr, world_pos,
                                    lights_uniform.env_zone_count, px.dfg, px.energy_comp);
        } else {
            ibl = ibl_ambient_grad(N, Nr, V, base_colour, metallic, roughness, F0, px.f90,
                                   diffuse_occ, spec_occ, lights_uniform.ibl_intensity,
                                   lights_uniform.ibl_rotation, refl_dr, px.dfg, px.energy_comp);
        }
        // Exposure applies once to the environment, zones included.
        ibl.diffuse *= lights_uniform.ibl_diffuse_scale * camera.exposure;
        ibl.specular *= lights_uniform.ibl_specular_scale * camera.exposure;
        ibl_diffuse = ibl.diffuse;
        ibl_specular = ibl.specular;
        ambient = ibl.diffuse + ibl.specular;
        ambient_lum = dot(ibl.diffuse, lum_weights) + dot(ibl.specular, lum_weights);
    } else {
        let hemi_t = clamp(vertex_normal.z * 0.5 + 0.5, 0.0, 1.0);
        let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
        let ambient_scale = (vec3<f32>(object.ambient) + hemi_colour * lights_uniform.hemisphere_intensity)
            * camera.exposure;
        ambient = ambient_scale * (base_colour * (1.0 - metallic) + F0 * metallic) * diffuse_occ;
        ambient_lum = dot(ambient, lum_weights);
    }
    // A light-probe object takes its indirect diffuse from the SH field at its
    // position. SH probes carry diffuse only: with an environment, the probe
    // takes the diffuse weight and the environment's reflection stays.
    if FEAT_LIGHT_PROBE && object.has_light_probe != 0u {
        let probe = evaluate_object_indirect(object.light_probe_index, world_pos, N) * base_colour * diffuse_occ;
        if lights_uniform.ibl_enabled != 0u {
            ambient = probe * ambient_kd + ibl_specular;
        } else {
            ambient = probe;
        }
        ambient_lum = dot(ambient, lum_weights);
    }
    // Baked lightmap: replace, add, or occlude the ambient term. Sample the
    // base mip explicitly: a lightmap packs charts tightly, so UV derivatives
    // across a chart are large and derivative-based mip selection would snap
    // to a coarse level and show the atlas texels as blocky bands. The
    // object's sub-rect and base layer are already folded into the UV and page.
    if FEAT_LIGHTMAP && object.lightmap_mode != 0u {
        let lm_page = i32(round(lightmap_page));
        var lm = textureSampleLevel(lightmap_tex, obj_sampler, lightmap_uv, lm_page, 0.0);
        if object.lightmap_directional == 1u {
            let f = lightmap_directional_factor(lightmap_uv, lm_page, N, vertex_normal);
            lm = vec4<f32>(lm.rgb * f, lm.a);
        }
        ambient = apply_lightmap(ambient, base_colour, diffuse_occ, lm, object.lightmap_mode, lightmap_shadow);
        // Replace and Subtractive bake the diffuse only: weight it like the
        // environment's diffuse and keep the environment's reflection.
        if lights_uniform.ibl_enabled != 0u
            && (object.lightmap_mode == 1u || object.lightmap_mode == 4u) {
            ambient = ambient * ambient_kd + ibl_specular;
        }
        ambient_lum = dot(ambient, lum_weights);
    }

    if pbr_layers_compiled() && px.layers.flags != 0u {
        ambient = pbr_ambient_layers(px, ambient, N, V, refl_dr);
    }

    out.ambient = ambient;
    out.ibl_diffuse = ibl_diffuse;
    out.ibl_specular = ibl_specular;
    out.ambient_kd = ambient_kd;
    out.ambient_lum = ambient_lum;
    return out;
}

// The environment's exposed radiance from direction `R`, prefiltered for
// perceptual `roughness`: the environment (default layer) when one is bound,
// else the hemisphere ambient.
fn pbr_layer_radiance(R: vec3<f32>, roughness: f32, refl_dr: f32) -> vec3<f32> {
    if lights_uniform.ibl_enabled != 0u {
        return sample_ibl_prefiltered_grad(R, roughness, lights_uniform.ibl_rotation, refl_dr)
            * (lights_uniform.ibl_intensity * camera.exposure) * lights_uniform.ibl_specular_scale;
    }
    let hemi_t = clamp(R.z * 0.5 + 0.5, 0.0, 1.0);
    let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
    return (vec3<f32>(object.ambient) + hemi_colour * lights_uniform.hemisphere_intensity)
        * camera.exposure;
}

// Light refracted through a transmissive surface, before the scene behind it
// is known. `pbr_transmission` fills it; the caller adds `fallback` (the
// environment along `direction`) and the transmission pass swaps it for the
// scene seen at `exit_pos`.
struct PbrTransmission {
    enabled: bool,
    // The share of the diffuse lobe given to transmission:
    // transmission * (1 - metallic).
    amount: f32,
    // Radiance from behind is scaled by this: the base colour tint, the
    // light the specular lobe did not reflect (1 - the split-sum specular
    // albedo) and the Beer-Lambert absorption over the thickness.
    weight: vec3<f32>,
    // Where the refracted ray leaves the volume and the direction it leaves
    // in. A thin wall (zero thickness) does not bend the ray.
    exit_pos: vec3<f32>,
    direction: vec3<f32>,
    // The environment radiance along `direction` times `weight`.
    fallback: vec3<f32>,
    // Perceptual roughness, which blurs the scene seen through the surface.
    roughness: f32,
};

// Resolve the transmission of `px` at `world_pos` from the material's volume
// lanes: `volume0` (transmission, thickness, ior, 0) and `volume1`
// (absorption coefficient rgb, 0). A thick surface refracts into a sphere of
// the given thickness and out of its far side.
fn pbr_transmission(
    px: PbrPixel,
    N: vec3<f32>,
    V: vec3<f32>,
    world_pos: vec3<f32>,
    volume0: vec4<f32>,
    volume1: vec4<f32>,
    refl_dr: f32,
) -> PbrTransmission {
    var t: PbrTransmission;
    t.enabled = true;
    t.amount = clamp(volume0.x, 0.0, 1.0) * (1.0 - px.metallic);
    let thickness = max(volume0.y, 0.0);
    let eta = 1.0 / max(volume0.z, 1.0);
    t.exit_pos = world_pos;
    t.direction = -V;
    if thickness > 0.0 {
        let r = refract(-V, N, eta);
        if dot(r, r) > 1e-6 {
            let n_dot_r = dot(N, r);
            t.exit_pos = world_pos + r * (thickness * -n_dot_r);
            let n_out = normalize(n_dot_r * r - N * 0.5);
            let out_dir = refract(r, n_out, 1.0 / eta);
            t.direction = select(r, normalize(out_dir), dot(out_dir, out_dir) > 1e-6);
        }
    }
    let specular_albedo = px.f0 * px.dfg.x + vec3<f32>(px.f90 * px.dfg.y);
    let absorption = exp(-max(volume1.xyz, vec3<f32>(0.0)) * thickness);
    t.weight = px.diffuse_colour * clamp(vec3<f32>(1.0) - specular_albedo, vec3<f32>(0.0), vec3<f32>(1.0))
        * absorption * t.amount;
    t.roughness = px.perceptual_roughness;
    t.fallback = t.weight * pbr_layer_radiance(t.direction, t.roughness, refl_dr);
    return t;
}

// The layers over the indirect term: the sheen dims the base and adds its
// own reflection, then the coat dims both and adds its reflection, each
// taking the specular occlusion. Environment zones and probes light the
// base only; the layers read the default environment.
fn pbr_ambient_layers(
    px: PbrPixel,
    ambient: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
    refl_dr: f32,
) -> vec3<f32> {
    let flags = px.layers.flags & pbr_layer_features();
    let R = reflect(-V, N);
    var a = ambient;
    if (flags & VIEWPORT_BRDF_LAYER_SHEEN) != 0u {
        let ly = px.layers;
        a = a * ly.sheen_scaling
            + ly.sheen_colour * ly.sheen_dfg * pbr_layer_radiance(R, ly.sheen_roughness, refl_dr)
              * px.spec_ao;
    }
    if (flags & VIEWPORT_BRDF_LAYER_CLEARCOAT) != 0u {
        let fc = viewport_brdf_coat_fresnel(px.n_dot_v, px.layers.clearcoat);
        a = a * (1.0 - fc)
            + pbr_layer_radiance(R, px.layers.clearcoat_roughness, refl_dr) * (fc * px.spec_ao);
    }
    return a;
}
