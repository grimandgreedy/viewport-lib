//! The scene depth convention: which way depth compares, what it clears to,
//! and which end is near.
//!
//! Every pipeline that tests or writes scene depth takes its compare function
//! from here, and every pass that clears scene depth clears to [`DEPTH_CLEAR`].
//! Shaders get the same convention from
//! [`SHARED_DEPTH_WGSL`](crate::plugin_api::shared_wgsl::SHARED_DEPTH_WGSL).
//!
//! Shadow maps keep their own convention, [`SHADOW_DEPTH_COMPARE`] and
//! [`SHADOW_DEPTH_CLEAR`]: light projections are separate from the camera's.

use crate::gpu::CompareFunction;

/// Depth compare for a pipeline that draws scene geometry: passes when the
/// fragment is at least as near as what the depth buffer holds.
pub const DEPTH_COMPARE: CompareFunction = CompareFunction::LessEqual;

/// Strict form of [`DEPTH_COMPARE`]: passes only when the fragment is nearer.
pub const DEPTH_COMPARE_STRICT: CompareFunction = CompareFunction::Less;

/// The value scene depth clears to: the far end, where nothing has drawn.
pub const DEPTH_CLEAR: f32 = 1.0;

/// Normalised device depth at the near plane.
pub const DEPTH_NEAR: f32 = 0.0;

/// Normalised device depth at the far plane, or at infinity for a projection
/// with no far plane.
pub const DEPTH_FAR: f32 = 1.0;

/// Depth compare for a shadow-map caster pipeline. Light projections are
/// standard 0..1 depth with the near plane at 0.
pub const SHADOW_DEPTH_COMPARE: CompareFunction = CompareFunction::LessEqual;

/// Strict form of [`SHADOW_DEPTH_COMPARE`].
pub const SHADOW_DEPTH_COMPARE_STRICT: CompareFunction = CompareFunction::Less;

/// The value a shadow map clears to: its far end.
pub const SHADOW_DEPTH_CLEAR: f32 = 1.0;

/// Whether depth `a` is nearer the camera than depth `b` under this
/// convention.
pub fn depth_nearer(a: f32, b: f32) -> bool {
    if DEPTH_NEAR < DEPTH_FAR { a < b } else { a > b }
}

/// Whether a depth value read back from the scene depth buffer is the clear
/// value, where no geometry drew.
pub fn depth_is_background(d: f32) -> bool {
    if DEPTH_NEAR < DEPTH_FAR {
        d >= DEPTH_FAR
    } else {
        d <= DEPTH_FAR
    }
}

/// A depth bias for a pipeline drawing into scene depth, in camera terms:
/// positive `constant` and `slope_scale` push the fragment away from the
/// camera, negative values pull it toward the camera. `clamp` is 0.
pub fn scene_depth_bias(constant: i32, slope_scale: f32) -> crate::gpu::DepthBiasState {
    let sign = if DEPTH_NEAR < DEPTH_FAR { 1 } else { -1 };
    crate::gpu::DepthBiasState {
        constant: constant * sign,
        slope_scale: slope_scale * sign as f32,
        clamp: 0.0,
    }
}

/// Translate a compare function written for scene depth into the shadow-map
/// convention. A compare that names neither convention's near-to-far order
/// (`Always`, `Equal`) passes through.
pub fn scene_to_shadow_compare(compare: CompareFunction) -> CompareFunction {
    if compare == DEPTH_COMPARE {
        SHADOW_DEPTH_COMPARE
    } else if compare == DEPTH_COMPARE_STRICT {
        SHADOW_DEPTH_COMPARE_STRICT
    } else {
        compare
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_is_the_far_end() {
        assert_eq!(DEPTH_CLEAR, DEPTH_FAR);
        assert!(depth_is_background(DEPTH_CLEAR));
        assert!(!depth_is_background(0.5));
        assert!(depth_nearer(DEPTH_NEAR, DEPTH_FAR));
    }

    #[test]
    fn shadow_compare_maps_the_scene_convention() {
        assert_eq!(scene_to_shadow_compare(DEPTH_COMPARE), SHADOW_DEPTH_COMPARE);
        assert_eq!(
            scene_to_shadow_compare(DEPTH_COMPARE_STRICT),
            SHADOW_DEPTH_COMPARE_STRICT
        );
        assert_eq!(
            scene_to_shadow_compare(CompareFunction::Always),
            CompareFunction::Always
        );
    }

    /// The WGSL helper's constants match the ones here.
    #[test]
    fn wgsl_constants_match() {
        let src: String = crate::plugin_api::shared_wgsl::SHARED_DEPTH_WGSL
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        assert!(
            src.contains(&format!("constVIEWPORT_DEPTH_NEAR:f32={DEPTH_NEAR:?};")),
            "VIEWPORT_DEPTH_NEAR differs from DEPTH_NEAR"
        );
        assert!(
            src.contains(&format!("constVIEWPORT_DEPTH_FAR:f32={DEPTH_FAR:?};")),
            "VIEWPORT_DEPTH_FAR differs from DEPTH_FAR"
        );
    }
}
