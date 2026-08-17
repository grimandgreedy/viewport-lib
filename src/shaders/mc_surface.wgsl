// mc_surface.wgsl - GPU marching cubes: lightweight Phong surface shader.
//
// Consumes the 24-byte McVertex buffer produced by mc_generate.wgsl:
//   location 0 : position  vec3<f32>  (bytes  0-11)
//   location 1 : normal    vec3<f32>  (bytes 12-23)
//
// Group 0 : camera_bgl (shared with all scene pipelines)
//   binding 0 : Camera uniform
//   binding 3 : Lights uniform
//
// Group 1 : per-draw material
//   binding 0 : McSurfaceUniform (base_colour vec3, roughness f32)
//
// Uses the canonical `SingleLight` / `Lights` structs from `scene_lighting.wgsl`
// but keeps an inline per-light loop because it adds a Blinn-Phong specular term
// that the shared helper does not cover.

// #include "helpers/scene_lighting.wgsl"

// ---------------------------------------------------------------------------
// Group 0: camera + lights
// ---------------------------------------------------------------------------

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

// `SingleLight` and `Lights` come from the included `scene_lighting.wgsl`.

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(3) var<uniform> lights: Lights;

// ---------------------------------------------------------------------------
// Group 1: per-draw material
// ---------------------------------------------------------------------------

struct McSurfaceUniform {
    base_colour: vec3<f32>,
    roughness:  f32,
    unlit:      u32,
    opacity:    f32,
    // Per-material ambient scalar (`Material::ambient`). Folded into the
    // hemisphere ambient so the MC path matches the regular mesh shader's
    // Blinn-Phong ambient term, which otherwise leaves the MC dark side
    // visibly darker than an equivalent regular mesh.
    ambient:    f32,
    _pad1:      u32,
};

@group(1) @binding(0) var<uniform> material: McSurfaceUniform;

// ---------------------------------------------------------------------------
// Vertex stage
// ---------------------------------------------------------------------------

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       world_pos:  vec3<f32>,
    @location(1)       world_norm: vec3<f32>,
};

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos   = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.world_pos  = v.position;
    out.world_norm = v.normal;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Unlit early-out: return base colour with no lighting.
    if material.unlit != 0u {
        return vec4<f32>(material.base_colour, material.opacity);
    }

    let N = normalize(in.world_norm);
    let V = normalize(camera.eye_pos - in.world_pos);

    // Hemisphere ambient (Z-up world) plus the per-material ambient scalar
    // the regular mesh shader's Blinn-Phong path also adds. Without the
    // material term the MC path reads darker than an equivalent regular
    // mesh on the side facing away from the light.
    let up_dot = clamp(N.z * 0.5 + 0.5, 0.0, 1.0);
    let hemi_ambient = mix(
        lights.ground_colour * lights.hemisphere_intensity,
        lights.sky_colour    * lights.hemisphere_intensity,
        up_dot,
    );
    let ambient = hemi_ambient + vec3<f32>(material.ambient);

    // Accumulate all light types. Iterate the cluster's light list when the
    // build pass is active, or the full active-light array under the small-N
    // fallback : see `scene_lighting.wgsl`.
    var diffuse  = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    let range = cluster_light_range(in.world_pos, lights.count);
    for (var j: u32 = 0u; j < range.count; j = j + 1u) {
        let i = cluster_light_global(range, j);
        let light = lights_storage[i];
        var L: vec3<f32>;
        var light_rgb: vec3<f32>;

        if light.light_type == 0u {
            // Directional: pos_or_dir is the surface-to-light direction.
            L = normalize(light.pos_or_dir);
            light_rgb = light.colour * light.intensity;
        } else if light.light_type == 1u {
            // Point: physical inverse-square falloff, radius-clamped and windowed
            // to zero at `range`. Matches scene_lighting.wgsl::eval_light.
            let to_light = light.pos_or_dir - in.world_pos;
            let dist = length(to_light);
            L = to_light / max(dist, 0.0001);
            let r2 = max(light.radius * light.radius, 1.0e-4);
            let win = clamp(1.0 - pow(dist / light.range, 4.0), 0.0, 1.0);
            let atten = win * win / max(dist * dist, r2);
            light_rgb = light.colour * light.intensity * atten;
        } else {
            // Spot: inverse-square falloff * cone attenuation.
            let to_light = light.pos_or_dir - in.world_pos;
            let dist = length(to_light);
            L = to_light / max(dist, 0.0001);
            let r2 = max(light.radius * light.radius, 1.0e-4);
            let win = clamp(1.0 - pow(dist / light.range, 4.0), 0.0, 1.0);
            let atten = win * win / max(dist * dist, r2);
            let spot_dir = normalize(light.spot_direction);
            let cos_angle = dot(-L, spot_dir);
            let cos_outer = cos(light.outer_angle);
            let cos_inner = cos(light.inner_angle);
            let cone_att = clamp(
                (cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
                0.0, 1.0,
            );
            light_rgb = light.colour * light.intensity * atten * cone_att;
        }

        let H = normalize(L + V);
        let diff  = max(dot(N, L), 0.0);
        // Blinn-Phong specular; map roughness [0,1] -> shininess [2, 128].
        let shine = mix(128.0, 2.0, material.roughness);
        let spec  = pow(max(dot(N, H), 0.0), shine) * (1.0 - material.roughness) * 0.3;

        diffuse  += light_rgb * diff;
        specular += light_rgb * spec;
    }

    let final_colour = material.base_colour * (ambient + diffuse) + specular;
    return vec4<f32>(final_colour, material.opacity);
}
