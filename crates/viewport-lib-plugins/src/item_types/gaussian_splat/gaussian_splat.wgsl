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
        let base = splat_idx * 12u;
        colour = eval_sh1(
            sh_rgb(base),
            sh_rgb(base + 3u),
            sh_rgb(base + 6u),
            sh_rgb(base + 9u),
            splat_view_dir(pos_obj),
        );
    }
    return project_splat(pos_obj, Sigma3d, colour, opacities[splat_idx], vi);
}
