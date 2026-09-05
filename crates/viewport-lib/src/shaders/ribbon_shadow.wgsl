// Depth-only shadow caster for Ribbon items.
//
// Ribbon geometry is already fully expanded (per-vertex width applied) on
// the CPU side, so the vertex shader only needs to apply the item's model
// transform and project into light clip space -- unlike mesh.wgsl's shadow
// caster, there is no per-instance storage array to index.
//
// Group 0: the shadow pass's own dedicated camera uniform (light
// view-projection), a single dynamic-offset binding -- see shadow.wgsl for
// the same contract.
// Group 1: reuses ribbon.wgsl's `StreamtubeUniform` bind group layout
// (uniform + streak texture + sampler) so the same `uniform_bind_group`
// built for the solid draw is bound again here; only the leading `model`
// field is read.

struct Light {
    view_proj: mat4x4<f32>,
};

struct RibbonShadowUniform {
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> light: Light;
@group(1) @binding(0) var<uniform> tube: RibbonShadowUniform;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    let world = (tube.model * vec4<f32>(position, 1.0)).xyz;
    return light.view_proj * vec4<f32>(world, 1.0);
}
