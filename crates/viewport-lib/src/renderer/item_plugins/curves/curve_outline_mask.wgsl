// Selection outline mask for the curve mesh item types.
//
// Draws the item's connected triangle mesh and writes a solid mask value, so
// the outline composite traces the tube or ribbon silhouette rather than a
// bounding proxy. Group 1 is the same per-draw model + id uniform the pick
// pipeline uses; only the model matrix is read here.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct PickInstance {
    model_c0: vec4<f32>,
    model_c1: vec4<f32>,
    model_c2: vec4<f32>,
    model_c3: vec4<f32>,
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> pick:   PickInstance;

struct VertexIn {
    @location(0) position: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(pick.model_c0, pick.model_c1, pick.model_c2, pick.model_c3);
    return camera.view_proj * model * vec4<f32>(in.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) f32 {
    return 1.0;
}
