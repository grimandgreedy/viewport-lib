// mc_surface.wgsl - GPU marching cubes: the extracted triangles, shaded by
// `shade_mc` (mc_shading.wgsl, prepended).
//
// Consumes the 24-byte McVertex buffer produced by mc_generate.wgsl:
//   location 0 : position  vec3<f32>  (bytes  0-11)
//   location 1 : normal    vec3<f32>  (bytes 12-23)

// ---------------------------------------------------------------------------
// Vertex stage
// ---------------------------------------------------------------------------

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       world_pos:  vec3<f32>,
    @location(1)       world_norm: vec3<f32>,
};

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos   = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.world_pos  = v.position;
    out.world_norm = v.normal;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if !mc_clip_keep(in.world_pos) {
        discard;
    }
    return shade_mc(in.world_pos, in.world_norm);
}
