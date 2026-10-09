
// @viewport-wgsl-version: 1
// Shared group-0 declarations for the shadow-cast pass. Do not re-declare
// these bindings in plugin shaders, and do not mix this with
// SHARED_BINDINGS_WGSL: the two describe different group-0 layouts.

struct ViewportShadowCamera {
    light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> shadow_camera: ViewportShadowCamera;
