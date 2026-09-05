// Ribbon shader : flat swept quad strips.
//
// Identical to streamtube.wgsl except cull_mode is None on the pipeline side
// and the fragment shader flips the normal for back-facing triangles so both
// sides of the ribbon shade correctly under the shared lighting helper.
//
// Group 0: Camera uniform (view-projection, eye position, forward, inverse
// view-projection, view) + shadow map/sampler/atlas + Lights + ClipPlanes +
// ClipVolume -- the same slots mesh.wgsl and sprite_lit.wgsl bind, since all
// three share the scene `camera_bgl`.
// Group 1: StreamtubeUniform : colour (vec4) + radius (f32) + use_vertex_colour (u32).

// #include "helpers/scene_lighting.wgsl"

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct ShadowAtlas {
    cascade_vp:        array<mat4x4<f32>, 4>,
    cascade_splits:    vec4<f32>,
    cascade_count:     u32,
    atlas_size:        f32,
    shadow_filter:     u32,
    pcss_light_radius: f32,
    atlas_rects:       array<vec4<f32>, 8>,
};

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:  u32,
    _pad0:  u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct StreamtubeUniform {
    model:             mat4x4<f32>,
    colour:            vec4<f32>,
    radius:           f32,
    use_vertex_colour: u32,
    unlit:            u32,
    opacity:          f32,
    wireframe:        u32,
    has_texture:      u32,
    receive_shadows:  u32,
    _pad:             f32,
};

struct ClipVolumeEntry {
    volume_type: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    center: vec3<f32>,
    radius: f32,
    half_extents: vec3<f32>,
    _pad1: f32,
    col0: vec3<f32>,
    _pad2: f32,
    col1: vec3<f32>,
    _pad3: f32,
    col2: vec3<f32>,
    _pad4: f32,
}

struct ClipVolumeUB {
    count: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    volumes: array<ClipVolumeEntry, 4>,
};

// `SingleLight` and `Lights` come from the included `scene_lighting.wgsl`.

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var                shadow_map:    texture_depth_2d;
@group(0) @binding(2) var                shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform>       lights_uniform: Lights;
@group(0) @binding(4) var<uniform>       clip_planes:   ClipPlanes;
@group(0) @binding(5) var<uniform>       shadow_atlas:  ShadowAtlas;
@group(0) @binding(6) var<uniform>       clip_volume:   ClipVolumeUB;
@group(1) @binding(0) var<uniform>       tube:           StreamtubeUniform;
@group(1) @binding(1) var                ribbon_texture: texture_2d<f32>;
@group(1) @binding(2) var                ribbon_sampler: sampler;

// #include "helpers/clip_volume_test.wgsl"

// Cascaded shadow map sampling: cascade selection, receiver bias, and the
// PCF/PCSS/hard filter tiers, shared with the mesh shader family.
// #include "helpers/csm.wgsl"

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) colour:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       world_nrm: vec3<f32>,
    @location(2)       vert_col:  vec4<f32>,
    @location(3)       uv:        vec2<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world = (tube.model * vec4<f32>(in.position, 1.0)).xyz;
    let nrm   = (tube.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.clip_pos  = camera.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.world_nrm = normalize(nrm);
    out.vert_col  = in.colour;
    out.uv        = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    // Section-plane clipping.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }

    if tube.wireframe != 0u {
        return vec4<f32>(0.75, 0.75, 0.75, 1.0);
    }

    // Resolve base colour. When a streak texture is bound it is multiplied in
    // on top of the per-vertex resolved colour so trail / beam textures can
    // ride the existing colour and alpha ramps.
    var base_colour = select(tube.colour, in.vert_col, tube.use_vertex_colour != 0u);
    if tube.has_texture != 0u {
        base_colour = base_colour * textureSample(ribbon_texture, ribbon_sampler, in.uv);
    }
    let alpha = base_colour.a * tube.opacity;

    // Unlit early-out: skip lighting entirely and return the resolved colour.
    if tube.unlit != 0u {
        return vec4<f32>(base_colour.rgb, alpha);
    }

    // Flip normal for back faces so both sides shade correctly.
    let n_raw = normalize(in.world_nrm);
    let n     = select(-n_raw, n_raw, is_front);

    // Hemisphere ambient + directional lights via the shared helper. Ribbon is
    // already flipped per-face above, so the helper sees a forward-facing normal
    // and one-sided weighting is enough.
    var shaded = apply_scene_lighting(n, base_colour.rgb, false, in.world_pos, lights_uniform);

    // When `receive_shadows` is set, attenuate the lit result by the cascade
    // shadow map. Mirrors sprite_lit.wgsl's pattern: mix the fully-lit colour
    // toward an ambient-only colour by the primary directional light's shadow
    // factor, rather than re-deriving a per-light shadowed sum (the shared
    // helper folds every light into one value with no per-light breakdown to
    // attenuate individually).
    if tube.receive_shadows != 0u
        && lights_uniform.shadows_enabled != 0u
        && lights_uniform.count > 0u {
        let l0 = lights_storage[0];
        if l0.light_type == 0u {
            let light_dir = normalize(l0.pos_or_dir);
            let shadow_factor = sample_shadow_csm(in.world_pos, camera.eye_pos, n, light_dir, 0u).factor;
            let up_weight = clamp(n.z * 0.5 + 0.5, 0.0, 1.0);
            let ambient = mix(
                lights_uniform.ground_colour,
                lights_uniform.sky_colour,
                up_weight,
            ) * lights_uniform.hemisphere_intensity;
            let ambient_rgb = base_colour.rgb * ambient;
            shaded = mix(ambient_rgb, shaded, shadow_factor);
        }
    }

    return vec4<f32>(shaded, alpha);
}
