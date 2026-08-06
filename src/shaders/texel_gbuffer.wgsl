// Texel G-buffer: rasterise mesh triangles into UV1 (atlas) space so every atlas
// texel a chart covers gets the world position and world normal of the surface
// point it represents. A lightmap solve reads these as ray origins and normals.
//
// The vertex stage places each vertex at its UV1 coordinate in clip space
// instead of projecting it through a camera, so the triangle lands in the atlas
// rect its UVs address. The fragment stage writes the interpolated world
// position (with a validity flag in .w) and the interpolated world normal.

struct Uniforms {
    model: mat4x4<f32>,
    // Model inverse-transpose (3x3 packed into the upper-left of a mat4) so
    // normals stay correct under non-uniform scale.
    normal_mat: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv1: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    // UV (0,0) is the atlas top-left. Map to clip space with Y flipped so the
    // target's top-left origin matches the UV that samples it at shade time.
    let ndc = vec2<f32>(in.uv1.x * 2.0 - 1.0, 1.0 - in.uv1.y * 2.0);
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.world_pos = (u.model * vec4<f32>(in.position, 1.0)).xyz;
    out.world_normal = (u.normal_mat * vec4<f32>(in.normal, 0.0)).xyz;
    return out;
}

struct FsOut {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var out: FsOut;
    // .w = 1 marks a covered texel; the targets clear to 0, so empty texels stay
    // invalid.
    out.position = vec4<f32>(in.world_pos, 1.0);
    out.normal = vec4<f32>(normalize(in.world_normal), 0.0);
    return out;
}
