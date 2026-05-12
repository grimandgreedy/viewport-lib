// splat_outline_mask.wgsl : renders selected Gaussian splat positions as
// screen-space discs into the R8 mask texture used by the outline edge-
// detection pass.  The resulting mask is identical in format to the one
// produced by outline_mask.wgsl, so the same edge-detection and composite
// passes handle both mesh and splat outlines without modification.
//
// Group 0: Camera bind group (only view_proj is used).
// Group 1: SplatOutlineMaskUniform (model matrix, viewport dims, pixel radius).
//
// Each splat position is one instance.  The vertex shader expands it to a
// screen-space quad of size pixel_radius*2 x pixel_radius*2.  The fragment
// shader discards corners to produce a disc.

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

// 80 bytes, matches SplatOutlineMaskUniform in Rust.
struct SplatOutlineMaskUniform {
    model:        mat4x4<f32>, // 64 bytes
    viewport_w:   f32,         //  4 bytes
    viewport_h:   f32,         //  4 bytes
    pixel_radius: f32,         //  4 bytes - half-side of the billboard quad in pixels
    _pad:         f32,         //  4 bytes
};

@group(0) @binding(0) var<uniform> camera:  Camera;
@group(1) @binding(0) var<uniform> u:       SplatOutlineMaskUniform;

// Six vertices per instance (two CCW triangles = one billboard quad).
fn quad_corner(vi: u32) -> vec2<f32> {
    switch vi {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0,  1.0); }
        case 4u: { return vec2<f32>( 1.0, -1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>, // corner in [-1,1]^2
};

@vertex
fn vs_main(
    @location(0)           position:     vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> VsOut {
    var out: VsOut;

    let world_pos = (u.model * vec4<f32>(position, 1.0)).xyz;
    let center    = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // Expand to a screen-space quad.  ndc_offset is scaled by w so the
    // perspective divide in the rasteriser produces the correct pixel offset.
    let corner      = quad_corner(vertex_index);
    let ndc_offset  = corner * u.pixel_radius / vec2<f32>(u.viewport_w, u.viewport_h);
    out.clip_pos = vec4<f32>(
        center.x + ndc_offset.x * center.w,
        center.y + ndc_offset.y * center.w,
        center.z,
        center.w,
    );
    out.uv = corner;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Discard the quad corners outside the unit disc.
    if dot(in.uv, in.uv) > 1.0 { discard; }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
