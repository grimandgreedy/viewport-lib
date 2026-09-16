// volume_surface_slice_mask.wgsl : renders selected volume surface slices as
// white (r=1.0) into the R8 selection mask, which the edge pass reads to draw
// the outline.
//
// The slice is a plain uploaded mesh with a model transform: no per-vertex
// position override and no deformers, so the transform is the whole vertex
// stage.
//
// Group 0: camera bind group (only view_proj is read).
// Group 1: the slice's render bind group (only the uniform's model is read).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

struct SliceUniform {
    model:      mat4x4<f32>,  // offset   0, size 64
    bbox_min:   vec3<f32>,    // offset  64, size 12  (align 16, fits at 64)
    scalar_min: f32,          // offset  76, size  4
    bbox_max:   vec3<f32>,    // offset  80, size 12  (align 16, fits at 80)
    scalar_max: f32,          // offset  92, size  4
    opacity:    f32,          // offset  96, size  4
    // struct size = roundUp(100, 16) = 112 -- matches the Rust repr(C) layout
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> slice: SliceUniform;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * slice.model * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
