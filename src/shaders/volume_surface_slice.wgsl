// Volume surface slice shader: samples a 3D volume at each fragment's world position.
//
// The slice surface is any mesh passed as a vertex + index buffer. The fragment shader
// converts the fragment's world position to volume UVW coordinates and samples the
// volume texture, discarding fragments that fall outside the volume bbox.
//
// Group 0: Camera uniform (same layout as all other shaders).
// Group 1: VolumeSurfaceSliceUniform + volume texture + nearest sampler
//          + colormap LUT texture + linear sampler.

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

struct ClipPlanes {
    planes:          array<vec4<f32>, 6>,
    count:           u32,
    _pad0:           u32,
    viewport_width:  f32,
    viewport_height: f32,
};

struct VolumeSurfaceSliceUniform {
    model:      mat4x4<f32>,  // offset   0, size 64
    bbox_min:   vec3<f32>,    // offset  64, size 12  (align 16, fits at 64)
    scalar_min: f32,           // offset  76, size  4
    bbox_max:   vec3<f32>,    // offset  80, size 12  (align 16, fits at 80)
    scalar_max: f32,           // offset  92, size  4
    opacity:    f32,           // offset  96, size  4
    // struct size = roundUp(100, 16) = 112 -- matches the Rust repr(C) layout
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;

@group(1) @binding(0) var<uniform> slice_ub:    VolumeSurfaceSliceUniform;
@group(1) @binding(1) var          vol_tex:     texture_3d<f32>;
@group(1) @binding(2) var          vol_sampler: sampler;
@group(1) @binding(3) var          lut_tex:     texture_2d<f32>;
@group(1) @binding(4) var          lut_sampler: sampler;

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOut {
    var out: VertexOut;
    let world_pos = (slice_ub.model * vec4<f32>(position, 1.0)).xyz;
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Section-plane clipping.
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if dot(vec4<f32>(in.world_pos, 1.0), plane) < 0.0 {
            discard;
        }
    }

    // Convert world position to volume UVW [0, 1]^3.
    let extent = slice_ub.bbox_max - slice_ub.bbox_min;
    let uvw = (in.world_pos - slice_ub.bbox_min) / max(extent, vec3<f32>(1e-6));

    // Discard fragments outside the volume bounding box.
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        discard;
    }

    // Load scalar from the 3D volume texture (R32Float is non-filterable).
    let dims   = vec3<f32>(textureDimensions(vol_tex));
    let texel  = vec3<i32>(clamp(uvw * dims, vec3<f32>(0.0), dims - vec3<f32>(1.0)));
    let scalar = textureLoad(vol_tex, texel, 0).r;

    // Normalize scalar and map through the LUT.
    let range = slice_ub.scalar_max - slice_ub.scalar_min;
    let t     = select(0.0, (scalar - slice_ub.scalar_min) / range, range > 0.0);
    let u     = clamp(t, 0.0, 1.0);
    var color = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(u, 0.5), 0.0);
    color.a   = color.a * slice_ub.opacity;
    return color;
}
