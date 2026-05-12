// tensor_glyph_outline_mask.wgsl : renders selected tensor glyph instances
// as solid geometry into the R8 outline mask texture.  Uses the same bind
// group layout and vertex transform as tensor_glyph.wgsl but outputs a
// flat mask value instead of lit/colored fragments.
//
// Group 0: Camera (view_proj).
// Group 1: TensorGlyphUniform + LUT texture + sampler (unused, layout must match).
// Group 2: Per-instance storage buffer (TensorInstance with pre-computed model).

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad:      f32,
};

struct TensorGlyphUniform {
    has_scalars: u32,
    scalar_min:  f32,
    scalar_max:  f32,
    _pad0:       f32,
    _pad1:       array<vec4<f32>, 3>,
};

struct TensorInstance {
    model_col0:    vec4<f32>,
    model_col1:    vec4<f32>,
    model_col2:    vec4<f32>,
    model_col3:    vec4<f32>,
    normal_col0:   vec4<f32>,
    normal_col1:   vec4<f32>,
    normal_col2:   vec4<f32>,
    scalar:        f32,
    _pad0:         f32,
    _pad1:         f32,
    _pad2:         f32,
};

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(1) @binding(0) var<uniform>       tg_uniform:  TensorGlyphUniform;
@group(1) @binding(1) var               lut_texture:  texture_2d<f32>;
@group(1) @binding(2) var               lut_sampler:  sampler;
@group(2) @binding(0) var<storage, read> instances:   array<TensorInstance>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) tangent:  vec4<f32>,
    @builtin(instance_index) instance_index: u32,
};

@vertex
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
    let inst  = instances[in.instance_index];
    let model = mat4x4<f32>(inst.model_col0, inst.model_col1, inst.model_col2, inst.model_col3);
    let world_pos = model * vec4<f32>(in.position, 1.0);
    return camera.view_proj * world_pos;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
