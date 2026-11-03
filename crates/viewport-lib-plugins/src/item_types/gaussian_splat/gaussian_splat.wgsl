// gaussian_splat.wgsl
//
// Renders uploaded Gaussian splat sets sorted back-to-front via a per-viewport
// radix-sorted index buffer. Each splat is a billboard quad (6 vertices, 2 triangles)
// with per-fragment Gaussian alpha falloff. Appended to gaussian_splat_common.wgsl.
//
// Group 1: SplatUniform (binding 0, common), sorted_indices, positions, scales,
//           rotations, opacities, sh_coefficients.

@group(1) @binding(1) var<storage, read>  sorted_indices:  array<u32>;
@group(1) @binding(2) var<storage, read>  positions:       array<vec4<f32>>;
@group(1) @binding(3) var<storage, read>  scales:          array<vec4<f32>>;
@group(1) @binding(4) var<storage, read>  rotations:       array<vec4<f32>>;
@group(1) @binding(5) var<storage, read>  opacities:       array<f32>;
@group(1) @binding(6) var<storage, read>  sh_coefficients: array<f32>;

fn sh_rgb(base: u32) -> vec3<f32> {
    return vec3<f32>(sh_coefficients[base], sh_coefficients[base + 1u], sh_coefficients[base + 2u]);
}

// The degree 2 and 3 terms of a degree-3 set, coefficients 4 to 15 of the
// splat whose block starts at `base`, with the standard 3DGS constants.
fn eval_sh23(base: u32, d: vec3<f32>) -> vec3<f32> {
    let x = d.x; let y = d.y; let z = d.z;
    let xx = x * x; let yy = y * y; let zz = z * z;
    var col = vec3<f32>(0.0);
    col +=  1.0925484305920792 * x * y               * sh_rgb(base + 12u);
    col += -1.0925484305920792 * y * z               * sh_rgb(base + 15u);
    col +=  0.31539156525252005 * (2.0 * zz - xx - yy) * sh_rgb(base + 18u);
    col += -1.0925484305920792 * x * z               * sh_rgb(base + 21u);
    col +=  0.5462742152960396 * (xx - yy)           * sh_rgb(base + 24u);
    col += -0.5900435899266435 * y * (3.0 * xx - yy)  * sh_rgb(base + 27u);
    col +=  2.890611442640554 * x * y * z            * sh_rgb(base + 30u);
    col += -0.4570457994644658 * y * (4.0 * zz - xx - yy) * sh_rgb(base + 33u);
    col +=  0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_rgb(base + 36u);
    col += -0.4570457994644658 * x * (4.0 * zz - xx - yy) * sh_rgb(base + 39u);
    col +=  1.445305721320277 * z * (xx - yy)        * sh_rgb(base + 42u);
    col += -0.5900435899266435 * x * (xx - 3.0 * yy)  * sh_rgb(base + 45u);
    return col;
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vi:         u32,
    @builtin(instance_index) instance_i: u32,
) -> VsOut {
    let splat_idx = sorted_indices[instance_i];
    let pos_obj   = positions[splat_idx].xyz;
    let Sigma3d   = splat_covariance(scales[splat_idx].xyz, rotations[splat_idx]);

    // -- SH colour --
    var colour: vec3<f32>;
    if splat_u.sh_degree == 0u {
        colour = eval_sh0(sh_rgb(splat_idx * 3u));
    } else {
        // Degree 1 holds 12 coefficients a splat, degree 3 holds 48.
        let degree3 = splat_u.sh_degree == 3u;
        let base = splat_idx * select(12u, 48u, degree3);
        let view_dir = splat_view_dir(pos_obj);
        var higher = vec3<f32>(0.0);
        if degree3 {
            higher = eval_sh23(base, view_dir);
        }
        colour = eval_sh1(
            sh_rgb(base),
            sh_rgb(base + 3u),
            sh_rgb(base + 6u),
            sh_rgb(base + 9u),
            view_dir,
            higher,
        );
    }
    return project_splat(pos_obj, Sigma3d, colour, opacities[splat_idx], vi);
}
