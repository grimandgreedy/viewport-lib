// outline_mask_ndc.wgsl: renders a screen-space rect into the R8 outline mask.
//
// No camera transform needed. Coordinates are directly in NDC space.
// Used for ScreenImageItem outlines.
//
// Group 0 : binding 0 = NdcRectUniform

struct NdcRectUniform {
    ndc_min: vec2<f32>,
    ndc_max: vec2<f32>,
}

@group(0) @binding(0) var<uniform> rect: NdcRectUniform;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var x: f32;
    var y: f32;
    switch vi {
        case 0u: { x = rect.ndc_min.x; y = rect.ndc_min.y; }
        case 1u: { x = rect.ndc_max.x; y = rect.ndc_min.y; }
        case 2u: { x = rect.ndc_min.x; y = rect.ndc_max.y; }
        case 3u: { x = rect.ndc_min.x; y = rect.ndc_max.y; }
        case 4u: { x = rect.ndc_max.x; y = rect.ndc_min.y; }
        default: { x = rect.ndc_max.x; y = rect.ndc_max.y; }
    }
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
