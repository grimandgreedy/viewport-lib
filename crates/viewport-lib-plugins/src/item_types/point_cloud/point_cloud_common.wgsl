// Point cloud shading shared by the two ways a point's channels reach the
// vertex stage: storage buffers indexed by instance (point_cloud.wgsl) and
// instance-stepped vertex streams on devices without vertex storage (the
// stream entry points the plugin generates).
//
// Group 0: the shared scene bindings, prefixed from SHARED_BINDINGS_WGSL.
//          Only camera.view_proj, clip_planes and clip_volume are read.
// Group 1: PointCloud uniform (model matrix, point_size, scalar mapping params,
//          default_colour, channel presence flags) + LUT texture (256x1) +
//          LUT sampler. The storage path adds its channel buffers after them.
//
// Each point is rendered as an instanced billboard quad (6 vertices = 2
// triangles); vertex_index (0-5) selects the corner.

// Point cloud per-item uniform : 128 bytes.
struct PointCloudUniform {
    model:            mat4x4<f32>,   // 64 bytes
    default_colour:    vec4<f32>,     // 16 bytes
    point_size:       f32,           //  4 bytes
    has_scalars:      u32,           //  4 bytes (1 = use scalar buffer + LUT)
    scalar_min:       f32,           //  4 bytes
    scalar_max:       f32,           //  4 bytes
    has_colours:       u32,           //  4 bytes (1 = use colour buffer)
    has_radius:       u32,           //  4 bytes (1 = per-point radius from radius_buffer)
    has_transparency: u32,           //  4 bytes (1 = per-point alpha from transparency_buffer)
    gaussian:         u32,           //  4 bytes (1 = gaussian splat falloff; implies circle clip)
    render_mode:      u32,           //  4 bytes (0 = ScreenSpaceCircle, 1 = Sphere)
    _pad0:            u32,           //  4 bytes padding
    _pad1:            u32,           //  4 bytes padding
    _pad2:            u32,           //  4 bytes padding
};

@group(1) @binding(0) var<uniform>            pc_uniform:           PointCloudUniform;
@group(1) @binding(1) var                     lut_texture:          texture_2d<f32>;
@group(1) @binding(2) var                     lut_sampler:          sampler;

// One point's channels. A channel the item does not carry holds a placeholder,
// and the uniform's has_* flag keeps it from being read.
struct PointInputs {
    position:     vec3<f32>,
    scalar:       f32,
    colour:       vec4<f32>,
    radius:       f32,
    transparency: f32,
};

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       colour:     vec4<f32>,
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

fn expand_point(p: PointInputs, vertex_index: u32) -> VertexOut {
    var out: VertexOut;

    let world_pos = (pc_uniform.model * vec4<f32>(p.position, 1.0)).xyz;
    let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // Expand to a screen-space quad. corner is in [-1,1]^2, mapped to pixels via
    // half_size. The NDC offset is scaled by w so the division in the rasteriser
    // produces the correct pixel-space result.
    let point_size  = select(pc_uniform.point_size, p.radius, pc_uniform.has_radius != 0u);
    let half_size   = point_size * 0.5;
    let corner      = quad_corner(vertex_index);
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
        let raw   = p.scalar;
        let range = pc_uniform.scalar_max - pc_uniform.scalar_min;
        let t     = select(0.0, (raw - pc_uniform.scalar_min) / range, range > 0.0);
        let u     = clamp(t, 0.0, 1.0);
        out.colour = textureSampleLevel(lut_texture, lut_sampler, vec2<f32>(u, 0.5), 0.0);
    } else if pc_uniform.has_colours != 0u {
        out.colour = p.colour;
    } else {
        out.colour = pc_uniform.default_colour;
    }

    // Apply per-point transparency (multiplies the alpha channel).
    if pc_uniform.has_transparency != 0u {
        out.colour.a = out.colour.a * clamp(p.transparency, 0.0, 1.0);
    }

    // Apply global opacity: default_colour.a acts as a uniform opacity multiplier
    // for all colour modes (scalar LUT, per-point colour, and solid colour).
    // For solid colour mode it is already baked into default_colour.a above.
    if pc_uniform.has_scalars != 0u || pc_uniform.has_colours != 0u {
        out.colour.a = out.colour.a * pc_uniform.default_colour.a;
    }

    return out;
}

fn fs_main_body(in: VertexOut) -> vec4<f32> {
    // Section-view clipping: planes and volumes together.
    if !viewport_clip_test(in.world_pos) { discard; }

    // All modes clip to a circle. uv is in [-1,1]^2; d2=1 is the quad edge.
    let d2 = dot(in.uv, in.uv);
    if d2 > 1.0 { discard; }

    var colour = in.colour;

    if pc_uniform.gaussian != 0u {
        // Soft Gaussian splat: alpha falls off as exp(-3*d^2).
        colour.a = colour.a * exp(-3.0 * d2);
    } else if pc_uniform.render_mode == 1u {
        // Sphere shading: reconstruct a hemisphere normal from the billboard UV.
        // uv.xy lie in [-1,1]; z is the front-facing hemisphere depth.
        let nz  = sqrt(1.0 - d2); // always >= 0 since d2 <= 1
        let n   = vec3<f32>(in.uv.x, in.uv.y, nz); // already unit length (d2+nz^2=1)

        // Fixed view-space light direction (upper-left, slightly in front).
        let light = normalize(vec3<f32>(-0.4, 0.6, 1.0));

        let ambient  = 0.25;
        let diffuse  = max(dot(n, light), 0.0);

        // Blinn-Phong specular: half-vector between light and view direction (0,0,1).
        let h        = normalize(light + vec3<f32>(0.0, 0.0, 1.0));
        let specular = pow(max(dot(n, h), 0.0), 32.0) * 0.4;

        let brightness = ambient + (1.0 - ambient) * diffuse + specular;
        scene_indirect_fraction = indirect_fraction(
            vec3<f32>(ambient),
            vec3<f32>((1.0 - ambient) * diffuse + specular),
        );
        colour = vec4<f32>(colour.rgb * brightness, colour.a);
    }
    // render_mode == 0 (ScreenSpaceCircle): flat disc, no shading beyond the circle clip above.

    return colour;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return fs_main_body(in);
}

// `fs_main` for the HDR scene pass, with the indirect fraction.
@fragment
fn fs_scene(in: VertexOut) -> SceneFragment {
    let colour = fs_main_body(in);
    return scene_fragment(colour, scene_indirect_fraction);
}
