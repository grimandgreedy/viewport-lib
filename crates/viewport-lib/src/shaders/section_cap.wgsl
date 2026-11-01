// Section cap fill: the flat cross-section a clip plane leaves through a mesh.
//
// Group 0: the mesh camera layout; this shader reads the camera (binding 0)
//          and the clip planes and clip volumes (bindings 4 and 6).
// Group 1: CapUniform (model matrix, RGBA colour, and the index of the clip
//          plane the cap lies on).
//
// A cap is clipped by every other active clip plane and by the clip volumes,
// so where two planes cross each cap stops at the other plane. Its own plane
// is skipped: the cap lies on it, and testing it would discard at random.

// #include "helpers/clip_bindings.wgsl"
// #include "helpers/scene_fragment.wgsl"

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct CapUniform {
    model:     mat4x4<f32>,
    colour:    vec4<f32>,
    own_plane: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};
@group(1) @binding(0) var<uniform> cap: CapUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = cap.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world;
    out.world_pos = world.xyz;
    return out;
}

fn fs_main_body(in: VertexOutput) -> vec4<f32> {
    for (var i = 0u; i < clip_planes.count; i = i + 1u) {
        let plane = clip_planes.planes[i];
        if i != cap.own_plane && dot(in.world_pos, plane.xyz) + plane.w < 0.0 {
            discard;
        }
    }
    if !viewport_pass_clip_volumes(in.world_pos) {
        discard;
    }
    return cap.colour;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return fs_main_body(in);
}

// `fs_main` for the HDR scene pass, with the indirect fraction.
@fragment
fn fs_scene(in: VertexOutput) -> SceneFragment {
    let colour = fs_main_body(in);
    return scene_fragment(colour, scene_indirect_fraction);
}
