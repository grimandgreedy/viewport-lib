//! The Gaussian splat set's writable channels.
//!
//! One marker per array a consumer can write part of, all five addressed in
//! splats. A splat set is the awkward case for this API and it is worth knowing
//! why, because the same two shapes recur:
//!
//! - **Three channels are padded on the way in.** Centres, scales and rotations
//!   are `vec4` on the GPU and `[f32; 3]` or `[f32; 4]` in hand, so the write
//!   builds the padded records. That is the channel's job, not the caller's.
//! - **The SH channel's stride is a runtime property of the set.** A degree-3 set
//!   holds 48 coefficients per splat and a degree-0 set holds 3, which no
//!   associated type can carry. So the input stays `f32` and the *addressing*
//!   stays in splats: a write starts at a splat index and must carry a whole
//!   number of splats' worth of coefficients. `sh_coefficients_per_splat` reports
//!   how many that is.
//!
//! Only the SH channel can be absent, when a set was uploaded with none.
//!
//! Writing centres or scales updates the CPU mirror the proximity pick and the
//! wireframe rings read, so picking keeps agreeing with the picture.

use super::types::GaussianSplatId;
use viewport_lib::plugin_api::Channel;

/// Object-space centres, one `[f32; 3]` per splat, padded to `vec4` on the way
/// in. Its live count is the set's draw count.
#[derive(Copy, Clone, Debug, Default)]
pub struct Positions;

/// Per-splat scales in world metres, padded to `vec4` on the way in.
#[derive(Copy, Clone, Debug, Default)]
pub struct Scales;

/// Per-splat unit quaternion rotations, `[x, y, z, w]`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Rotations;

/// Per-splat opacity in `[0, 1]`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Opacities;

/// Spherical-harmonic coefficients, addressed in splats rather than in
/// coefficients.
///
/// `first_element` is a splat index and the data must be a whole number of
/// splats' worth: `sh_coefficients_per_splat` for the set, which follows its
/// degree. Anything else is
/// [`ContentBufferWriteOutOfRange`](viewport_lib::error::ViewportError::ContentBufferWriteOutOfRange),
/// so a caller cannot silently shift every splat's colour by one coefficient.
///
/// Pass them exactly as the trainer produced them: the evaluated colour is
/// sRGB-referred and the shader decodes it.
#[derive(Copy, Clone, Debug, Default)]
pub struct ShCoefficients;

impl Channel for Positions {
    type Id = GaussianSplatId;
    type Input = [f32; 3];
    const NAME: &'static str = "positions";
}

impl Channel for Scales {
    type Id = GaussianSplatId;
    type Input = [f32; 3];
    const NAME: &'static str = "scales";
}

impl Channel for Rotations {
    type Id = GaussianSplatId;
    type Input = [f32; 4];
    const NAME: &'static str = "rotations";
}

impl Channel for Opacities {
    type Id = GaussianSplatId;
    type Input = f32;
    const NAME: &'static str = "opacities";
}

impl Channel for ShCoefficients {
    type Id = GaussianSplatId;
    type Input = f32;
    const NAME: &'static str = "sh_coefficients";
}
