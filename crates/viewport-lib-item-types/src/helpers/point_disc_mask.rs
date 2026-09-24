//! The screen-space disc mask both point-set item types draw their selection
//! outline through.
//!
//! A selected point becomes one instance of a billboard quad clipped to a
//! disc, written into the R8 mask texture the outline edge-detection pass
//! reads. Point clouds and gaussian splats produce the same coverage, so they
//! share the shader and the uniform behind it.

/// Group-1 uniform of `point_disc_mask.wgsl`: the model matrix, the viewport
/// size the screen-space expansion divides by, and the disc radius in pixels.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PointDiscMaskUniform {
    pub model: [[f32; 4]; 4],
    pub viewport_w: f32,
    pub viewport_h: f32,
    pub pixel_radius: f32,
    pub _pad: [f32; 9],
}
