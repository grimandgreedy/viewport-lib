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

// Point cloud per-item uniform : 128 bytes.
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
    gaussian:         u32,           //  4 bytes (1 = gaussian splat falloff; implies circle clip)
    render_mode:      u32,           //  4 bytes (0 = ScreenSpaceCircle, 1 = Sphere)
    _pad0:            u32,           //  4 bytes padding
    _pad1:            u32,           //  4 bytes padding
    _pad2:            u32,           //  4 bytes padding
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

@group(0) @binding(0) var<uniform> camera:     Camera;
// Bindings 1-5 of group 0 are shadow/light uniforms present in the layout but unused here.
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;

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
    @location(2)       uv:        vec2<f32>,
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
    out.uv        = corner;

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

    // Apply global opacity: default_color.a acts as a uniform opacity multiplier
    // for all color modes (scalar LUT, per-point color, and solid color).
    // For solid color mode it is already baked into default_color.a above.
    if pc_uniform.has_scalars != 0u || pc_uniform.has_colors != 0u {
        out.color.a = out.color.a * pc_uniform.default_color.a;
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

    // All modes clip to a circle. uv is in [-1,1]^2; d2=1 is the quad edge.
    let d2 = dot(in.uv, in.uv);
    if d2 > 1.0 { discard; }

    var color = in.color;

    if pc_uniform.gaussian != 0u {
        // Soft Gaussian splat: alpha falls off as exp(-3*d²).
        color.a = color.a * exp(-3.0 * d2);
    } else if pc_uniform.render_mode == 1u {
        // Sphere shading: reconstruct a hemisphere normal from the billboard UV.
        // uv.xy lie in [-1,1]; z is the front-facing hemisphere depth.
        let nz  = sqrt(1.0 - d2); // always >= 0 since d2 <= 1
        let n   = vec3<f32>(in.uv.x, in.uv.y, nz); // already unit length (d2+nz²=1)

        // Fixed view-space light direction (upper-left, slightly in front).
        let light = normalize(vec3<f32>(-0.4, 0.6, 1.0));

        let ambient  = 0.25;
        let diffuse  = max(dot(n, light), 0.0);

        // Blinn-Phong specular: half-vector between light and view direction (0,0,1).
        let h        = normalize(light + vec3<f32>(0.0, 0.0, 1.0));
        let specular = pow(max(dot(n, h), 0.0), 32.0) * 0.4;

        let brightness = ambient + (1.0 - ambient) * diffuse + specular;
        color = vec4<f32>(color.rgb * brightness, color.a);
    }
    // render_mode == 0 (ScreenSpaceCircle): flat disc, no shading beyond the circle clip above.

    return color;
}
