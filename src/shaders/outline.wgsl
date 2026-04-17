// Outline shader for two-pass stencil selection effect.
//
// Pass 1 (stencil_write_pipeline): renders the object normally; writes stencil=1
//   wherever the depth test passes. Uses the standard mesh pipeline for this.
//
// Pass 2 (outline_pipeline): renders a slightly expanded version of the object;
//   stencil test = NotEqual(1) so only the "ring" outside the original silhouette
//   is drawn. Depth compare = Always so the ring always renders on top.
//
// Group 0: Camera bind group (same layout as mesh.wgsl, bindings 0–7).
// Group 1: OutlineUniform (model matrix, outline color, pixel_offset).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    _pad: f32,
};

struct ClipPlanes {
    planes: array<vec4<f32>, 6>,
    count: u32,
    _pad0: u32,
    viewport_width: f32,
    viewport_height: f32,
};

struct OutlineUniform {
    model: mat4x4<f32>,
    color: vec4<f32>,
    pixel_offset: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct ClipVolumeUB {
    volume_type: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
    plane_normal: vec3<f32>,
    plane_dist: f32,
    box_center: vec3<f32>,
    _padB0: f32,
    box_half_extents: vec3<f32>,
    _padB1: f32,
    box_col0: vec3<f32>,
    _padB2: f32,
    box_col1: vec3<f32>,
    _padB3: f32,
    box_col2: vec3<f32>,
    _padB4: f32,
    sphere_center: vec3<f32>,
    sphere_radius: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(4) var<uniform> clip_planes: ClipPlanes;
@group(0) @binding(6) var<uniform> clip_volume: ClipVolumeUB;
@group(1) @binding(0) var<uniform> outline: OutlineUniform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
    let world_pos = outline.model * vec4<f32>(in.position, 1.0);
    let clip_pos  = camera.view_proj * world_pos;

    // Perform perspective divide to get NDC position.
    let ndc = clip_pos.xy / clip_pos.w;

    // Compute the object center in NDC for a stable expansion direction.
    let center_world = outline.model * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let center_clip  = camera.view_proj * center_world;
    let center_ndc   = center_clip.xy / center_clip.w;

    // Direction from object center to this vertex in NDC (screen space).
    let dir = ndc - center_ndc;
    let len = length(dir);

    // Expand outward by pixel_offset pixels.
    // One pixel in NDC = 2.0 / viewport_dimension.
    let vp_w = max(clip_planes.viewport_width, 1.0);
    let vp_h = max(clip_planes.viewport_height, 1.0);
    let px = vec2<f32>(2.0 / vp_w, 2.0 / vp_h);

    // If the vertex is at the center (degenerate), fall back to no expansion.
    var offset_ndc = vec2<f32>(0.0);
    if len > 0.0001 {
        let norm_dir = dir / len;
        offset_ndc = norm_dir * outline.pixel_offset * px;
    }

    // Apply the offset in clip space (multiply back by w).
    return vec4<f32>(
        clip_pos.x + offset_ndc.x * clip_pos.w,
        clip_pos.y + offset_ndc.y * clip_pos.w,
        clip_pos.z,
        clip_pos.w,
    );
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return outline.color;
}
