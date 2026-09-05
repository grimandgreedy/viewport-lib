// Depth-only shadow caster for GPU marching cubes surfaces.
//
// MC vertices are already world-space: the compute extraction pass writes
// world positions directly and there is no per-item model matrix anywhere in
// the MC path (see `mc_surface.wgsl`'s `vs_main`, which passes `v.position`
// straight through). So this shader needs no group 1 at all -- just the
// shadow pass's own camera uniform at group 0.

struct Light {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> light: Light;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return light.view_proj * vec4<f32>(position, 1.0);
}
