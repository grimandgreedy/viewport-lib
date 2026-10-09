
// @viewport-wgsl-version: 1
// OIT MRT output struct and pack helper. Requires SHARED_BINDINGS_WGSL.
//
// Use as:
//   @fragment
//   fn fs_main(...) -> OitOutput {
//       return viewport_oit_pack(color_rgb, alpha, in.view_z);
//   }
//
// `view_z` is the view-space Z coordinate (negative in front of the
// camera). The weight function biases nearer fragments toward higher
// contribution, matching the weight curve in mesh_oit.wgsl.

struct OitOutput {
    @location(0) accum:  vec4<f32>,
    @location(1) reveal: f32,
};

fn viewport_oit_weight(view_z: f32, alpha: f32) -> f32 {
    // Weight curve from McGuire & Bavoil 2013, equation 7. Tuned for the
    // lib's typical scene depth range.
    let z = abs(view_z);
    let w = alpha * clamp(10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0)), 1e-2, 3e3);
    return w;
}

fn viewport_oit_pack(color: vec3<f32>, alpha: f32, view_z: f32) -> OitOutput {
    let w = viewport_oit_weight(view_z, alpha);
    var out: OitOutput;
    out.accum  = vec4<f32>(color * alpha * w, alpha * w);
    out.reveal = alpha;
    return out;
}
