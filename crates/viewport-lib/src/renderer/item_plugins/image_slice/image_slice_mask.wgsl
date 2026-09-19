// Image slice outline mask: rasterises the slice quad into the R8 selection
// mask so the outline composite draws the edge around a selected slice.
//
// The quad generation matches image_slice.wgsl exactly; only the fragment
// stage differs (a constant 1.0 coverage value).
//
// Group 0: Camera uniform (shared scene layout).
// Group 1: ImageSliceUniform (the render bind group, extra bindings unused).

struct Camera {
    view_proj:     mat4x4<f32>,
    eye_pos:       vec3<f32>,
    _pad:          f32,
    forward:       vec3<f32>,
    _pad1:         f32,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
};

// Axis: 0=X, 1=Y, 2=Z
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

@group(0) @binding(0) var<uniform> camera:   Camera;
@group(1) @binding(0) var<uniform> slice_ub: ImageSliceUniform;

// Generate quad corners for the slice in world space; must match
// image_slice.wgsl's quad_world so the mask covers the drawn pixels.
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
        return vec3<f32>(x, bmin.y + s * (bmax.y - bmin.y), bmin.z + r * (bmax.z - bmin.z));
    } else if axis == 1u {
        let y = bmin.y + t * (bmax.y - bmin.y);
        return vec3<f32>(bmin.x + s * (bmax.x - bmin.x), y, bmin.z + r * (bmax.z - bmin.z));
    } else {
        let z = bmin.z + t * (bmax.z - bmin.z);
        return vec3<f32>(bmin.x + s * (bmax.x - bmin.x), bmin.y + r * (bmax.y - bmin.y), z);
    }
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(quad_world(vi), 1.0);
}

@fragment
fn fs_main() -> @location(0) f32 {
    return 1.0;
}
