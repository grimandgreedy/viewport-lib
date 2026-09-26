//! Pieces more than one item type draws through.
//!
//! A type's own shaders and uniforms live in its own directory. What lands
//! here is what two or more of them share, so neither has to reach sideways
//! into the other: the screen-space disc mask the two point-set types outline
//! with, and anything later that ends up in the same position.
//!
//! Shader names stay unique across the crate regardless of directory, because
//! `build.rs` flattens every `.wgsl` into one `OUT_DIR`.

pub(crate) mod point_disc_mask;

/// The shared shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::shader::{scene_shader, wgsl_source};
    vec![(
        "point_disc_mask.wgsl",
        scene_shader(&[], wgsl_source!("point_disc_mask")),
    )]
}
