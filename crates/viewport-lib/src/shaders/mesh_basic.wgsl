// mesh_basic.wgsl : basic-lit mesh shader for devices without storage buffers.
//
// Per-object data arrives as an instance-stepped vertex stream instead of a
// storage array, and the lighting is the shared hemisphere-plus-lights loop
// (`apply_scene_lighting`). Base colour, vertex colour, an albedo texture,
// alpha mask, scalar colouring through a LUT, unlit, two-sided and section
// clipping follow the full mesh shader's rules; shadows, IBL, PBR and the
// material maps are absent.
//
// Group 0: the compatibility group 0. Only the camera, the lights header, the light
//   list and the clip uniforms are read.
// Group 1: albedo texture + sampler, colourmap LUT + sampler.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    // Upper bound on the lit (pre-emissive) colour: 1.0 on the LDR path.
    lit_clamp: f32,
    forward: vec3<f32>,
    _pad1: f32,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    // View -> clip and its inverse.
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    // Viewport in physical pixels: width, height, 1 / width, 1 / height.
    viewport: vec4<f32>,
    // Near, far, 1 when orthographic (0 when perspective), reserved.
    clip: vec4<f32>,
    // Frame time: seconds as f32, the remainder f32 drops from the f64
    // seconds, delta seconds, frame index modulo 2^24.
    time: vec4<f32>,
    // The world origin positions are relative to, rounded to f32, and a
    // counter bumped each time it moves.
    world_origin: vec3<f32>,
    origin_epoch: u32,
};

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count: u32,
    _pad0: u32,
    viewport_width: f32,
    viewport_height: f32,
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

// #include "helpers/scene_lighting.wgsl"

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(3) var<uniform> lights_uniform: Lights;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

// #include "helpers/clip_volume_test.wgsl"

@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_sampler: sampler;
@group(1) @binding(2) var lut_tex: texture_2d<f32>;
@group(1) @binding(3) var lut_sampler: sampler;

// Instance flags (`params.x`).
const BASIC_UNLIT: u32 = 1u;
const BASIC_TWO_SIDED: u32 = 2u;
const BASIC_ALPHA_MASK: u32 = 4u;
const BASIC_HAS_SCALAR: u32 = 8u;
const BASIC_USE_NAN_COLOUR: u32 = 16u;
const BASIC_IGNORE_CLIP: u32 = 32u;
const BASIC_WIREFRAME: u32 = 64u;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(5) model0: vec4<f32>,
    @location(6) model1: vec4<f32>,
    @location(7) model2: vec4<f32>,
    @location(8) model3: vec4<f32>,
    // Linear RGB and opacity.
    @location(9) base_colour: vec4<f32>,
    // Emissive RGB (scaled), alpha cutoff.
    @location(10) emissive: vec4<f32>,
    // Metallic, roughness, scalar range min, scalar range max.
    @location(11) material: vec4<f32>,
    // Flags, NaN colour (RGBA8), unused, unused.
    @location(12) params: vec4<u32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) emissive: vec4<f32>,
    @location(5) scalar_t: f32,
    @location(6) is_nan: f32,
    @location(7) @interpolate(flat) flags: u32,
    @location(8) @interpolate(flat) nan_colour: u32,
};

fn vertex_common(in: VertexIn, scalar: f32, has_scalar: bool) -> VertexOut {
    let model = mat4x4<f32>(in.model0, in.model1, in.model2, in.model3);
    let world = model * vec4<f32>(in.position, 1.0);
    // Cofactor matrix: the inverse transpose up to scale, so normals stay
    // perpendicular under non-uniform scale.
    let c0 = in.model0.xyz;
    let c1 = in.model1.xyz;
    let c2 = in.model2.xyz;
    let normal_matrix = mat3x3<f32>(cross(c1, c2), cross(c2, c0), cross(c0, c1));

    var out: VertexOut;
    out.clip_pos = camera.view_proj * world;
    out.world_pos = world.xyz;
    out.world_normal = normal_matrix * in.normal;
    out.colour = in.base_colour * in.colour;
    out.uv = in.uv;
    out.emissive = in.emissive;
    out.flags = in.params.x;
    out.nan_colour = in.params.y;

    // NaN is detected before interpolation can corrupt the bit pattern.
    let bits = bitcast<u32>(scalar);
    let nan = has_scalar && (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
    out.is_nan = select(0.0, 1.0, nan);
    let range = in.material.w - in.material.z;
    out.scalar_t = clamp(
        select(0.0, (scalar - in.material.z) / range, range > 0.0001),
        0.0,
        1.0,
    );
    return out;
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    return vertex_common(in, 0.0, false);
}

@vertex
fn vs_scalar(in: VertexIn, @location(13) scalar: f32) -> VertexOut {
    return vertex_common(in, scalar, true);
}

fn unpack_rgba8(v: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(v & 0xFFu),
        f32((v >> 8u) & 0xFFu),
        f32((v >> 16u) & 0xFFu),
        f32((v >> 24u) & 0xFFu),
    ) / 255.0;
}

@fragment
fn fs_main(in: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    // Both samples come first, while control flow is still uniform.
    let tex_colour = textureSample(albedo_tex, albedo_sampler, in.uv);
    let lut_colour = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(in.scalar_t, 0.5), 0.0);

    if (in.flags & BASIC_IGNORE_CLIP) == 0u {
        for (var i = 0u; i < clip_planes.count; i++) {
            let plane = clip_planes.planes[i];
            if dot(in.world_pos, plane.xyz) + plane.w < 0.0 {
                discard;
            }
        }
        if !clip_volume_test(in.world_pos) {
            discard;
        }
    }

    // Wireframe: the mesh's edges in flat grey, unlit, as the full tier
    // draws them.
    if (in.flags & BASIC_WIREFRAME) != 0u {
        return vec4<f32>(0.75, 0.75, 0.75, 1.0);
    }

    let colour = in.colour * tex_colour;
    if (in.flags & BASIC_ALPHA_MASK) != 0u && colour.a < in.emissive.w {
        discard;
    }

    var base_colour = colour.rgb;
    if (in.flags & BASIC_HAS_SCALAR) != 0u {
        if in.is_nan > 0.5 {
            if (in.flags & BASIC_USE_NAN_COLOUR) == 0u {
                discard;
            }
            return unpack_rgba8(in.nan_colour);
        }
        base_colour = lut_colour.rgb;
    }

    if (in.flags & BASIC_UNLIT) != 0u {
        return vec4<f32>(base_colour, colour.a);
    }

    let two_sided = (in.flags & BASIC_TWO_SIDED) != 0u;
    var n = normalize(in.world_normal);
    if two_sided && !is_front {
        n = -n;
    }
    let lit = apply_scene_lighting(n, base_colour, two_sided, in.world_pos, lights_uniform);
    let rgb = min(lit, vec3<f32>(camera.lit_clamp)) + in.emissive.rgb;
    return vec4<f32>(rgb, colour.a);
}
