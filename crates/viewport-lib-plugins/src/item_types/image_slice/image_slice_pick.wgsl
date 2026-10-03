// GPU object-ID pick shader for image slices.
//
// The vertex stage mirrors image_slice.wgsl's vs_main exactly: it generates
// the same axis-aligned quad from vertex_index using the same ImageSliceUniform
// (group 1, reused unchanged from the render bind group), so the pick quad
// matches the rendered slice. Object-level only: the primitive channel is a
// constant 0, since a slice is a single flat quad with no sub-object identity.
//
// Group 0: the shared scene bindings, prefixed from SHARED_BINDINGS_WGSL.
// Group 1: ImageSliceUniform (binding 0), reused unchanged from the render
//          bind group. Bindings 1-4 (volume texture, samplers, LUT) are
//          present but unused here.
// Group 2: per-item pick id (binding 0).

struct ImageSliceUniform {
    bbox_min:    vec3<f32>,
    axis:        u32,
    bbox_max:    vec3<f32>,
    offset:      f32,
    scalar_min:  f32,
    scalar_max:  f32,
    opacity:     f32,
    _pad:        f32,
};

// Per-item pick id.
struct PickId {
    object_id: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

@group(1) @binding(0) var<uniform> slice_ub: ImageSliceUniform;

@group(2) @binding(0) var<uniform> pick: PickId;

struct VertexOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

// Identical to image_slice.wgsl's quad_world.
fn quad_world(vi: u32) -> vec3<f32> {
    let bmin = slice_ub.bbox_min;
    let bmax = slice_ub.bbox_max;
    let t    = slice_ub.offset;
    let axis = slice_ub.axis;

    let corners = array<u32, 6>(0u, 1u, 2u, 2u, 3u, 0u);
    let c = corners[vi];
    let cx = array<f32, 4>(0.0, 1.0, 1.0, 0.0);
    let cy = array<f32, 4>(0.0, 0.0, 1.0, 1.0);
    let s  = cx[c];
    let r  = cy[c];

    if axis == 0u {
        let x = bmin.x + t * (bmax.x - bmin.x);
        let y = bmin.y + s * (bmax.y - bmin.y);
        let z = bmin.z + r * (bmax.z - bmin.z);
        return vec3<f32>(x, y, z);
    } else if axis == 1u {
        let x = bmin.x + s * (bmax.x - bmin.x);
        let y = bmin.y + t * (bmax.y - bmin.y);
        let z = bmin.z + r * (bmax.z - bmin.z);
        return vec3<f32>(x, y, z);
    } else {
        let x = bmin.x + s * (bmax.x - bmin.x);
        let y = bmin.y + r * (bmax.y - bmin.y);
        let z = bmin.z + t * (bmax.z - bmin.z);
        return vec3<f32>(x, y, z);
    }
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    var out: VertexOut;
    let world_pos = quad_world(vi);
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    return out;
}

struct FragOut {
    @location(0) object_id:    u32,
    @location(1) primitive_id: u32,
    @location(2) depth:        f32,
};

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    if !viewport_pass_clip_volumes(in.world_pos) { discard; }
    var out: FragOut;
    out.object_id    = pick.object_id;
    out.primitive_id = 0u;
    out.depth        = in.clip_pos.z;
    return out;
}
