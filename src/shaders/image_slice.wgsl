// Image slice shader: renders a single axis-aligned slice of a 3D volume as a textured quad.
//
// No vertex buffer: the vertex shader generates 6 vertices (2 triangles) from vertex_index.
// The quad corners are computed from the slice axis, offset, and bounding box in the uniform.
//
// Group 0: Camera uniform.
// Group 1: ImageSliceUniform + 3D volume texture (non-filtered) + nearest sampler
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

// Axis: 0=X, 1=Y, 2=Z
struct ImageSliceUniform {
    bbox_min:    vec3<f32>,
    axis:        u32,        // 0=X, 1=Y, 2=Z
    bbox_max:    vec3<f32>,
    offset:      f32,        // normalized [0,1] position along the axis
    scalar_min:  f32,
    scalar_max:  f32,
    opacity:     f32,
    _pad:        f32,
};

@group(0) @binding(0) var<uniform> camera:      Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;

@group(1) @binding(0) var<uniform> slice_ub:      ImageSliceUniform;
@group(1) @binding(1) var          vol_tex:        texture_3d<f32>;
@group(1) @binding(2) var          vol_sampler:    sampler;
@group(1) @binding(3) var          lut_tex:        texture_2d<f32>;
@group(1) @binding(4) var          lut_sampler:    sampler;

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       uvw:       vec3<f32>,
};

// Generate quad corners for the slice in world space.
// axis=0(X): quad in YZ plane at X=pos; axis=1(Y): XZ plane; axis=2(Z): XY plane.
fn quad_world(vi: u32) -> vec3<f32> {
    let bmin = slice_ub.bbox_min;
    let bmax = slice_ub.bbox_max;
    let t    = slice_ub.offset;
    let axis = slice_ub.axis;

    // Compute the slice position along the axis.
    var pos: vec3<f32>;
    var uvw: vec3<f32>;

    // Two triangles: indices 0,1,2 and 0,2,3 -> vi maps to corner (0,1,2,2,3,0)
    let corners = array<u32, 6>(0u, 1u, 2u, 2u, 3u, 0u);
    let c = corners[vi];

    // Corner (cx, cy) in [0,1] for the two non-axis dimensions.
    let cx = array<f32, 4>(0.0, 1.0, 1.0, 0.0);
    let cy = array<f32, 4>(0.0, 0.0, 1.0, 1.0);
    let s  = cx[c];
    let r  = cy[c];

    if axis == 0u {
        // X slice: quad in YZ plane
        let x = bmin.x + t * (bmax.x - bmin.x);
        let y = bmin.y + s * (bmax.y - bmin.y);
        let z = bmin.z + r * (bmax.z - bmin.z);
        pos = vec3<f32>(x, y, z);
        uvw = vec3<f32>(t, s, r);
    } else if axis == 1u {
        // Y slice: quad in XZ plane
        let x = bmin.x + s * (bmax.x - bmin.x);
        let y = bmin.y + t * (bmax.y - bmin.y);
        let z = bmin.z + r * (bmax.z - bmin.z);
        pos = vec3<f32>(x, y, z);
        uvw = vec3<f32>(s, t, r);
    } else {
        // Z slice: quad in XY plane
        let x = bmin.x + s * (bmax.x - bmin.x);
        let y = bmin.y + r * (bmax.y - bmin.y);
        let z = bmin.z + t * (bmax.z - bmin.z);
        pos = vec3<f32>(x, y, z);
        uvw = vec3<f32>(s, r, t);
    }

    return pos;
}

fn quad_uvw(vi: u32) -> vec3<f32> {
    let bmin = slice_ub.bbox_min;
    let bmax = slice_ub.bbox_max;
    let t    = slice_ub.offset;
    let axis = slice_ub.axis;

    let corners = array<u32, 6>(0u, 1u, 2u, 2u, 3u, 0u);
    let c = corners[vi];
    let cx = array<f32, 4>(0.0, 1.0, 1.0, 0.0);
    let cy = array<f32, 4>(0.0, 0.0, 1.0, 1.0);
    let s = cx[c];
    let r = cy[c];

    if axis == 0u {
        return vec3<f32>(t, s, r);
    } else if axis == 1u {
        return vec3<f32>(s, t, r);
    } else {
        return vec3<f32>(s, r, t);
    }
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    var out: VertexOut;
    let world_pos = quad_world(vi);
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.uvw       = quad_uvw(vi);
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

    // Load from the 3D volume texture. R32Float is non-filterable so we use
    // textureLoad with integer texel coordinates instead of textureSampleLevel.
    let dims   = vec3<f32>(textureDimensions(vol_tex));
    let texel  = vec3<i32>(clamp(in.uvw * dims, vec3<f32>(0.0), dims - vec3<f32>(1.0)));
    let scalar = textureLoad(vol_tex, texel, 0).r;

    // Normalize and map through the LUT.
    let range = slice_ub.scalar_max - slice_ub.scalar_min;
    let t     = select(0.0, (scalar - slice_ub.scalar_min) / range, range > 0.0);
    let u     = clamp(t, 0.0, 1.0);
    var color = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(u, 0.5), 0.0);
    color.a   = color.a * slice_ub.opacity;
    return color;
}
