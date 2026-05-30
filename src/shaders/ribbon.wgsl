// Ribbon shader : flat swept quad strips.
//
// Identical to streamtube.wgsl except cull_mode is None on the pipeline side
// and the fragment shader flips the normal for back-facing triangles so both
// sides of the ribbon shade correctly under the shared lighting helper.
//
// Group 0: Camera uniform (view-projection, eye position) + Lights + ClipPlanes + ClipVolume.
// Group 1: StreamtubeUniform : colour (vec4) + radius (f32) + use_vertex_colour (u32).

// #include "scene_lighting.wgsl"

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:  u32,
    _pad0:  u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct StreamtubeUniform {
    colour:            vec4<f32>,
    radius:           f32,
    use_vertex_colour: u32,
    unlit:            u32,
    opacity:          f32,
    wireframe:        u32,
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

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(3) var<uniform> lights:      Lights;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;
@group(1) @binding(0) var<uniform> tube:        StreamtubeUniform;

fn clip_volume_test(p: vec3<f32>) -> bool {
    for (var i = 0u; i < clip_volume.count; i = i + 1u) {
        let e = clip_volume.volumes[i];
        if e.volume_type == 2u {
            let d = p - e.center;
            let local = vec3<f32>(dot(d, e.col0), dot(d, e.col1), dot(d, e.col2));
            if abs(local.x) > e.half_extents.x
                || abs(local.y) > e.half_extents.y
                || abs(local.z) > e.half_extents.z {
                return false;
            }
        } else if e.volume_type == 3u {
            let ds = p - e.center;
            if dot(ds, ds) > e.radius * e.radius { return false; }
        } else if e.volume_type == 4u {
            let axis = e.col0;
            let d = p - e.center;
            let along = dot(d, axis);
            if abs(along) > e.half_extents.x { return false; }
            let radial = d - axis * along;
            if dot(radial, radial) > e.radius * e.radius { return false; }
        }
    }
    return true;
}

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
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos  = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_nrm = normalize(in.normal);
    out.vert_col  = in.colour;
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

    // Resolve base colour.
    let base_colour = select(tube.colour, in.vert_col, tube.use_vertex_colour != 0u);
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
    let shaded = apply_scene_lighting(n, base_colour.rgb, false, in.world_pos, lights);

    return vec4<f32>(shaded, alpha);
}
