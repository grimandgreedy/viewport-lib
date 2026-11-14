//! Custom overlay fills: WGSL fill functions an `OverlayFill::Custom` names.
//!
//! Register one with
//! [`DeviceResources::register_overlay_fill`](crate::resources::DeviceResources::register_overlay_fill),
//! which returns the [`CustomFillId`](crate::CustomFillId) an item's fill
//! carries. The function is spliced into the overlay fill module every overlay
//! pipeline shares, so it draws on every overlay family that draws fills:
//! shapes, vector paths, closed polylines, meshes and text.

/// A fill function to register for `OverlayFill::Custom`.
///
/// `wgsl_body` defines
///
/// ```text
/// fn fill(local: vec2<f32>, params: OverlayFillParams) -> vec4<f32>
/// ```
///
/// plus any helpers it needs. `local` is the pixel's offset from the centre of
/// the box the fill spans, in the item's own logical pixels before its
/// transform, and `params` carries that box's half-extents and the item's
/// sixteen parameters:
///
/// ```text
/// struct OverlayFillParams {
///     half_size: vec2<f32>,
///     p0: vec4<f32>, // params[0..4]
///     p1: vec4<f32>, // params[4..8]
///     p2: vec4<f32>, // params[8..12]
///     p3: vec4<f32>, // params[12..16]
/// };
/// ```
///
/// The function returns linear RGBA with straight alpha; the item's opacity
/// and tint multiply it, and the item's coverage and clip are applied after.
/// Every top-level identifier in the body is prefixed with `<name>__` when it
/// is spliced in, so two fills cannot collide.
///
/// The function runs in non-uniform control flow and on devices without
/// storage buffers: it must not sample textures, take derivatives or declare
/// bindings. Keep it to arithmetic on its arguments.
#[derive(Clone, Debug)]
pub struct OverlayFillDesc {
    /// The fill's name. A WGSL identifier, unique among registered fills.
    pub name: std::borrow::Cow<'static, str>,
    /// WGSL defining `fn fill(local: vec2<f32>, params: OverlayFillParams) ->
    /// vec4<f32>` and its helpers.
    pub wgsl_body: String,
}

impl OverlayFillDesc {
    /// A fill named `name` whose body is `wgsl_body`.
    pub fn new(
        name: impl Into<std::borrow::Cow<'static, str>>,
        wgsl_body: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            wgsl_body: wgsl_body.into(),
        }
    }
}
