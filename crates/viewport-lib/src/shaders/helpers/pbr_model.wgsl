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
// and `lightmap_directional_factor`. Nothing here takes a screen-space
// derivative: the caller supplies the specular AA kernel and the reflection
// footprint from uniform control flow.
// ---------------------------------------------------------------------------

// Floor on perceptual roughness, so a mirror-smooth material keeps a finite
// GGX lobe.
const MIN_PERCEPTUAL_ROUGHNESS: f32 = 0.04;
// Reflectance at normal incidence of a dielectric (about 4%, an IOR of 1.5).
const DEFAULT_DIELECTRIC_F0: f32 = 0.04;

// One fragment's resolved PBR surface.
struct PbrPixel {
    // The base colour the diffuse lobe scales. The (1 - metallic) weight is
    // applied with the Fresnel term, not folded in here.
    diffuse_colour: vec3<f32>,
    // Specular reflectance at normal and grazing incidence.
    f0: vec3<f32>,
    f90: f32,
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
    diffuse_ao: f32,
    spec_ao: f32,
};

// Resolve the material inputs: clamp metallic, floor roughness, apply the
// metallic-roughness map's remap when `has_mr_map` (G = roughness,
// B = metallic, each mixed across its range before the scalar factor), widen
// the roughness by the specular AA kernel, then derive F0 and look up the
// split-sum terms and the energy compensation. `N` and `V` are the shading
// normal and the unit vector toward the eye.
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
    px.f0 = mix(vec3<f32>(DEFAULT_DIELECTRIC_F0), base_colour, metallic);
    px.f90 = 1.0;
    px.n_dot_v = max(dot(N, V), 1e-3);
    px.dfg = sample_brdf_lut(px.n_dot_v, roughness);
    px.energy_comp = viewport_brdf_energy_compensation(px.f0, px.dfg);
    px.diffuse_ao = ao;
    px.spec_ao = ao;
    return px;
}

// Replace the surface terms of a resolved pixel with authored ones (a shading
// hook's `shade_surface` output). No specular AA: the author owns the value.
// The split-sum terms are looked up again at the resolved N.V.
fn pbr_apply_override(
    px: PbrPixel,
    base_colour: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> PbrPixel {
    var out = px;
    out.diffuse_colour = base_colour;
    out.metallic = clamp(metallic, 0.0, 1.0);
    out.perceptual_roughness = max(roughness, MIN_PERCEPTUAL_ROUGHNESS);
    out.alpha = out.perceptual_roughness * out.perceptual_roughness;
    out.f0 = mix(vec3<f32>(DEFAULT_DIELECTRIC_F0), base_colour, out.metallic);
    out.dfg = sample_brdf_lut(out.n_dot_v, out.perceptual_roughness);
    out.energy_comp = viewport_brdf_energy_compensation(out.f0, out.dfg);
    return out;
}

// One light's direct contribution. `radiance` is the light's colour times its
// attenuation and shadow. The specular lobe carries the energy compensation.
// Exactly zero when N.L <= 0.
fn pbr_direct(px: PbrPixel, N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    return viewport_brdf_light_contrib_compensated(N, V, L, radiance, px.diffuse_colour,
                                                   px.metallic, px.perceptual_roughness, px.f0,
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

// The indirect term: the environment (zoned when zones exist) or the
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
    let ao_factor = px.diffuse_ao;

    var out: AmbientOut;
    var ambient: vec3<f32>;
    var ibl_specular = vec3<f32>(0.0);
    var ibl_diffuse = vec3<f32>(0.0);
    var ambient_lum = 0.0;
    let ambient_kd = (vec3<f32>(1.0) - F_Schlick_roughness(max(dot(N, V), 0.001), F0, roughness))
        * (1.0 - metallic);
    if lights_uniform.ibl_enabled != 0u {
        var ibl: IblContrib;
        if lights_uniform.env_zone_count != 0u {
            ibl = ibl_ambient_zoned(N, V, base_colour, metallic, roughness, F0,
                                    ao_factor, lights_uniform.ibl_intensity,
                                    lights_uniform.ibl_rotation, refl_dr, world_pos,
                                    lights_uniform.env_zone_count, px.dfg, px.energy_comp);
        } else {
            ibl = ibl_ambient_grad(N, V, base_colour, metallic, roughness, F0,
                                   ao_factor, lights_uniform.ibl_intensity,
                                   lights_uniform.ibl_rotation, refl_dr, px.dfg, px.energy_comp);
        }
        ibl.diffuse *= lights_uniform.ibl_diffuse_scale;
        ibl.specular *= lights_uniform.ibl_specular_scale;
        ibl_diffuse = ibl.diffuse;
        ibl_specular = ibl.specular;
        ambient = ibl.diffuse + ibl.specular;
        ambient_lum = dot(ibl.diffuse, lum_weights) + dot(ibl.specular, lum_weights);
    } else {
        let hemi_t = clamp(vertex_normal.z * 0.5 + 0.5, 0.0, 1.0);
        let hemi_colour = mix(lights_uniform.ground_colour, lights_uniform.sky_colour, hemi_t);
        let ambient_scale = vec3<f32>(object.ambient) + hemi_colour * lights_uniform.hemisphere_intensity;
        ambient = ambient_scale * (base_colour * (1.0 - metallic) + F0 * metallic) * ao_factor;
        ambient_lum = dot(ambient, lum_weights);
    }
    // A light-probe object takes its indirect diffuse from the SH field at its
    // position. SH probes carry diffuse only: with an environment, the probe
    // takes the diffuse weight and the environment's reflection stays.
    if FEAT_LIGHT_PROBE && object.has_light_probe != 0u {
        let probe = evaluate_object_indirect(object.light_probe_index, world_pos, N) * base_colour * ao_factor;
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
        ambient = apply_lightmap(ambient, base_colour, ao_factor, lm, object.lightmap_mode, lightmap_shadow);
        // Replace and Subtractive bake the diffuse only: weight it like the
        // environment's diffuse and keep the environment's reflection.
        if lights_uniform.ibl_enabled != 0u
            && (object.lightmap_mode == 1u || object.lightmap_mode == 4u) {
            ambient = ambient * ambient_kd + ibl_specular;
        }
        ambient_lum = dot(ambient, lum_weights);
    }

    out.ambient = ambient;
    out.ibl_diffuse = ibl_diffuse;
    out.ibl_specular = ibl_specular;
    out.ambient_kd = ambient_kd;
    out.ambient_lum = ambient_lum;
    return out;
}
