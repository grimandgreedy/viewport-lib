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

// #include "scene_lighting.wgsl"

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
    _pad1:      u32,
    _pad2:      u32,
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

    // Hemisphere ambient (Z-up world).
    let up_dot = clamp(N.z * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(
        lights.ground_colour * lights.hemisphere_intensity,
        lights.sky_colour    * lights.hemisphere_intensity,
        up_dot,
    );

    // Accumulate directional lights.
    var diffuse  = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < min(lights.count, 8u); i++) {
        let light = lights.lights[i];
        if light.light_type != 0u { continue; } // directional only

        // `pos_or_dir` is the surface-to-light direction (matches mesh.wgsl).
        let L     = normalize(light.pos_or_dir);
        let H     = normalize(L + V);
        let light_rgb = light.colour * light.intensity;

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
