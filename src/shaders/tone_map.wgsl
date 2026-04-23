// tone_map.wgsl — fullscreen post-process composite: tone mapping, bloom, SSAO, contact shadows.
// Renders a fullscreen triangle (no vertex buffer) using vertex_index.

struct ToneMapUniform {
    exposure:                f32,
    mode:                    u32,  // 0=Reinhard, 1=ACES, 2=KhronosNeutral
    bloom_enabled:           u32,
    ssao_enabled:            u32,
    contact_shadows_enabled: u32,
    _pad0:                   u32,
    _pad1:                   u32,
    _pad2:                   u32,
    background_color:        vec4<f32>,
}

@group(0) @binding(0) var hdr_texture:  texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler:  sampler;
@group(0) @binding(2) var<uniform> params: ToneMapUniform;
@group(0) @binding(3) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var ao_texture:    texture_2d<f32>;
@group(0) @binding(5) var cs_texture:    texture_2d<f32>;
@group(0) @binding(6) var depth_texture: texture_depth_2d;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];
    let uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return VertexOutput(vec4<f32>(p, 0.0, 1.0), uv);
}

fn reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (x + vec3<f32>(1.0));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn khronos_neutral(x: vec3<f32>) -> vec3<f32> {
    let a: f32 = 0.15;
    let b: f32 = 0.50;
    let c: f32 = 0.10;
    let d: f32 = 0.20;
    let e: f32 = 0.02;
    let f: f32 = 0.30;
    let w: f32 = 11.2;
    let curve = (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f;
    let white = (w * (a * w + c * b) + d * e) / (w * (a * w + b) + d * f) - e / f;
    return clamp(curve / white, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth_dims = textureDimensions(depth_texture);
    let depth_uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(0.99999994));
    let depth_coord = vec2<i32>(vec2<u32>(depth_uv * vec2<f32>(depth_dims)));
    let depth = textureLoad(depth_texture, depth_coord, 0);
    if depth >= 0.999999 {
        return params.background_color;
    }

    var color = textureSample(hdr_texture, hdr_sampler, in.uv).rgb;

    // Add bloom additively before tone mapping.
    if params.bloom_enabled != 0u {
        let bloom = textureSample(bloom_texture, hdr_sampler, in.uv).rgb;
        color = color + bloom;
    }

    // Multiply by AO before tone mapping.
    if params.ssao_enabled != 0u {
        let ao = textureSample(ao_texture, hdr_sampler, in.uv).r;
        color = color * ao;
    }

    // Multiply by contact shadow factor before tone mapping.
    if params.contact_shadows_enabled != 0u {
        let cs = textureSample(cs_texture, hdr_sampler, in.uv).r;
        color = color * cs;
    }

    // Pre-tone-mapping exposure.
    color = color * params.exposure;

    // Tone mapping.
    if params.mode == 0u {
        color = reinhard(color);
    } else if params.mode == 1u {
        color = aces(color);
    } else {
        color = khronos_neutral(color);
    }

    // Gamma correction (linear -> sRGB approximation).
    color = pow(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
