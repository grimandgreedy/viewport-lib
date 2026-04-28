// Point cloud shader for the 3D viewport.
//
// Group 0: Camera uniform (view-projection, eye position)
//          + shadow atlas texture + comparison sampler
//          + Lights uniform
//          + ClipPlanes uniform (up to 6 half-space clipping planes)
//          + ShadowAtlas uniform (unused here, but layout must match camera_bgl).
// Group 1: PointCloud uniform (model matrix, point_size, scalar mapping params,
//          default_color, has_scalars, has_colors)
//          + LUT texture (256x1, Rgba8Unorm)
//          + LUT sampler
//          + scalar storage buffer (f32 per point)
//          + color storage buffer (vec4 per point)
//
// Vertex input: position vec3 (location 0).
//
// The shader reads per-point color or scalar data from storage buffers,
// mapping through the LUT when has_scalars != 0.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

// Clip planes uniform : must match mesh.wgsl group 0 binding 4.
struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count:  u32,
    _pad0:  u32,
    viewport_width:  f32,
    viewport_height: f32,
};

// Point cloud per-item uniform : 112 bytes.
struct PointCloudUniform {
    model:            mat4x4<f32>,   // 64 bytes
    default_color:    vec4<f32>,     // 16 bytes
    point_size:       f32,           //  4 bytes
    has_scalars:      u32,           //  4 bytes (1 = use scalar buffer + LUT)
    scalar_min:       f32,           //  4 bytes
    scalar_max:       f32,           //  4 bytes
    has_colors:       u32,           //  4 bytes (1 = use color buffer)
    has_radius:       u32,           //  4 bytes (1 = per-point radius from radius_buffer)
    has_transparency: u32,           //  4 bytes (1 = per-point alpha from transparency_buffer)
    _pad:             u32,           //  4 bytes : total 112 bytes
};

struct ClipVolumeUB {
    volume_type: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    plane_normal: vec3<f32>,
    plane_dist: f32,
    box_center: vec3<f32>,
    _padB0: f32,
    box_half_extents: vec3<f32>,
    _padB1: f32,
    box_col0: vec3<f32>,
    _padB2: f32,
    box_col1: vec3<f32>,
    _padB3: f32,
    box_col2: vec3<f32>,
    _padB4: f32,
    sphere_center: vec3<f32>,
    sphere_radius: f32,
};

@group(0) @binding(0) var<uniform> camera:     Camera;
// Bindings 1-5 of group 0 are shadow/light uniforms present in the layout but unused here.
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

fn clip_volume_test(p: vec3<f32>) -> bool {
    if clip_volume.volume_type == 0u { return true; }
    if clip_volume.volume_type == 1u {
        return dot(p, clip_volume.plane_normal) + clip_volume.plane_dist >= 0.0;
    }
    if clip_volume.volume_type == 2u {
        let d = p - clip_volume.box_center;
        let local = vec3<f32>(
            dot(d, clip_volume.box_col0),
            dot(d, clip_volume.box_col1),
            dot(d, clip_volume.box_col2),
        );
        return abs(local.x) <= clip_volume.box_half_extents.x
            && abs(local.y) <= clip_volume.box_half_extents.y
            && abs(local.z) <= clip_volume.box_half_extents.z;
    }
    let ds = p - clip_volume.sphere_center;
    return dot(ds, ds) <= clip_volume.sphere_radius * clip_volume.sphere_radius;
}

@group(1) @binding(0) var<uniform>            pc_uniform:           PointCloudUniform;
@group(1) @binding(1) var                     lut_texture:          texture_2d<f32>;
@group(1) @binding(2) var                     lut_sampler:          sampler;
@group(1) @binding(3) var<storage, read>      scalar_buffer:        array<f32>;
@group(1) @binding(4) var<storage, read>      color_buffer:         array<vec4<f32>>;
@group(1) @binding(5) var<storage, read>      radius_buffer:        array<f32>;
@group(1) @binding(6) var<storage, read>      transparency_buffer:  array<f32>;

// Each point is rendered as an instanced billboard quad (6 vertices = 2 triangles).
// The position attribute is per-instance (step_mode = Instance in Rust).
// vertex_index (0-5) selects the quad corner; instance_index is the point index.
struct VertexIn {
    @location(0)             position:       vec3<f32>,
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       color:     vec4<f32>,
    @location(1)       world_pos: vec3<f32>,
};

// Unit quad corners for a billboard (two CCW triangles).
fn quad_corner(vi: u32) -> vec2<f32> {
    switch vi {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0,  1.0); }
        case 4u: { return vec2<f32>( 1.0, -1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let world_pos = (pc_uniform.model * vec4<f32>(in.position, 1.0)).xyz;
    let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // Determine color : indexed by instance (= point index), not vertex.
    let idx = in.instance_index;

    // Expand to a screen-space quad. corner is in [-1,1]^2, mapped to pixels via
    // half_size. The NDC offset is scaled by w so the division in the rasteriser
    // produces the correct pixel-space result.
    let point_size  = select(pc_uniform.point_size, radius_buffer[idx], pc_uniform.has_radius != 0u);
    let half_size   = point_size * 0.5;
    let corner      = quad_corner(in.vertex_index);
    let ndc_offset  = corner * half_size
                      / vec2<f32>(clip_planes.viewport_width, clip_planes.viewport_height);
    out.clip_pos  = vec4<f32>(
        center.x + ndc_offset.x * center.w,
        center.y + ndc_offset.y * center.w,
        center.z,
        center.w,
    );
    out.world_pos = world_pos;

    if pc_uniform.has_scalars != 0u {
        let raw   = scalar_buffer[idx];
        let range = pc_uniform.scalar_max - pc_uniform.scalar_min;
        let t     = select(0.0, (raw - pc_uniform.scalar_min) / range, range > 0.0);
        let u     = clamp(t, 0.0, 1.0);
        out.color = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(u, 0.5), 0.0);
    } else if pc_uniform.has_colors != 0u {
        out.color = color_buffer[idx];
    } else {
        out.color = pc_uniform.default_color;
    }

    // Apply per-point transparency (multiplies the alpha channel).
    if pc_uniform.has_transparency != 0u {
        out.color.a = out.color.a * clamp(transparency_buffer[idx], 0.0, 1.0);
    }

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Clip-plane culling (section views).
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }
    if !clip_volume_test(in.world_pos) { discard; }
    return in.color;
}
