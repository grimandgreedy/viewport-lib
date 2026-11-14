//! How a viewport's frame reaches its output: at the viewport's own
//! resolution, or drawn at a fixed reference size and scaled up to fill it.

use crate::colour::Colour;

/// How a viewport presents its frame, set on `ViewportFrame::presentation`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Presentation {
    /// The scene draws at the viewport's own size and pixels per point.
    #[default]
    Native,
    /// The scene draws into a target of exactly `size` pixels, which is then
    /// drawn into the viewport with nearest sampling, placed by `fit`.
    ///
    /// The scene sees a viewport of `size` at one pixel per point: sprites,
    /// line widths and world overlay layers are sized in reference pixels.
    /// The camera's vertical extent is kept and its aspect should be
    /// `size[0] / size[1]`. Dynamic resolution and `RenderTuning::render_scale`
    /// do not apply to the reference target.
    Reference {
        /// The reference target's size in pixels.
        size: [u32; 2],
        /// How the target is scaled into the viewport.
        fit: ReferenceFit,
        /// Where the frame's screen overlays draw.
        overlays: OverlayResolution,
    },
}

impl Presentation {
    /// Draw the scene at `size` and scale it into the viewport by `fit`, with
    /// screen overlays at the viewport's own resolution.
    pub fn reference(size: [u32; 2], fit: ReferenceFit) -> Self {
        Self::Reference {
            size,
            fit,
            overlays: OverlayResolution::Native,
        }
    }

    /// Whether this is a [`Presentation::Reference`].
    pub fn is_reference(&self) -> bool {
        matches!(self, Self::Reference { .. })
    }
}

/// How a reference target is scaled into the viewport.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ReferenceFit {
    /// The largest whole multiple of the reference size that fits, centred,
    /// with the rest of the viewport filled with `colour`. A viewport smaller
    /// than the reference size shows it at one device pixel per reference
    /// pixel, centred and cropped.
    IntegerBars {
        /// The colour of the bars.
        colour: Colour,
    },
    /// The smallest whole multiple of the reference size that covers the
    /// viewport, centred, with what falls outside the viewport cropped.
    IntegerCrop,
    /// Stretched over the whole viewport, whatever its aspect.
    Stretch,
}

/// Where a frame's screen overlays draw under a reference presentation.
///
/// World overlay layers are part of the scene and always draw into the
/// reference target.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OverlayResolution {
    /// After the upscale, over the whole viewport at its own resolution, laid
    /// out in the viewport's logical pixels.
    #[default]
    Native,
    /// Into the reference target before the upscale, laid out in reference
    /// pixels.
    Reference,
}

/// Where a reference target lands in a viewport of a given size.
///
/// Positions in the viewport are logical pixels from its top-left corner, as
/// picks take them; positions in the reference target are its pixels from its
/// top-left corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferencePlacement {
    /// The reference target's size in pixels.
    pub reference_size: [u32; 2],
    /// The viewport's size in device pixels.
    pub viewport_px: [u32; 2],
    /// Device pixels per logical pixel of the viewport.
    pub pixels_per_point: f32,
    /// The scaled target's top-left corner in the viewport's device pixels.
    /// Negative where the target is cropped.
    pub origin_px: [f32; 2],
    /// Device pixels per reference pixel, along each axis.
    pub scale: [f32; 2],
}

impl ReferencePlacement {
    /// Place a target of `reference_size` by `fit` in a viewport of
    /// `viewport_size` logical pixels at `pixels_per_point`.
    pub fn new(
        reference_size: [u32; 2],
        fit: ReferenceFit,
        viewport_size: [f32; 2],
        pixels_per_point: f32,
    ) -> Self {
        let rw = reference_size[0].max(1);
        let rh = reference_size[1].max(1);
        let vw = ((viewport_size[0] * pixels_per_point).round() as u32).max(1);
        let vh = ((viewport_size[1] * pixels_per_point).round() as u32).max(1);
        let scale = match fit {
            ReferenceFit::IntegerBars { .. } => {
                let s = (vw / rw).min(vh / rh).max(1) as f32;
                [s, s]
            }
            ReferenceFit::IntegerCrop => {
                let s = vw.div_ceil(rw).max(vh.div_ceil(rh)).max(1) as f32;
                [s, s]
            }
            ReferenceFit::Stretch => [vw as f32 / rw as f32, vh as f32 / rh as f32],
        };
        // A stretched target starts at the corner; a whole-scale one is
        // centred on a whole device pixel.
        let centre = |v: u32, r: u32, s: f32| match fit {
            ReferenceFit::Stretch => 0.0,
            _ => (v as i64 - (r as i64 * s as i64)).div_euclid(2) as f32,
        };
        Self {
            reference_size: [rw, rh],
            viewport_px: [vw, vh],
            pixels_per_point: pixels_per_point.max(1.0e-6),
            origin_px: [centre(vw, rw, scale[0]), centre(vh, rh, scale[1])],
            scale,
        }
    }

    /// The reference pixel position under the viewport position `point`, or
    /// `None` when `point` lies in the bars around the target or outside the
    /// viewport.
    pub fn window_to_reference(&self, point: glam::Vec2) -> Option<glam::Vec2> {
        let px = point * self.pixels_per_point;
        let [vw, vh] = self.viewport_px;
        if px.x < 0.0 || px.y < 0.0 || px.x >= vw as f32 || px.y >= vh as f32 {
            return None;
        }
        let r = (px - glam::Vec2::from(self.origin_px)) / glam::Vec2::from(self.scale);
        let [rw, rh] = self.reference_size;
        (r.x >= 0.0 && r.y >= 0.0 && r.x < rw as f32 && r.y < rh as f32).then_some(r)
    }

    /// The viewport position of the reference pixel position `point`, or
    /// `None` when `point` lies outside the target or where it is cropped out
    /// of the viewport.
    pub fn reference_to_window(&self, point: glam::Vec2) -> Option<glam::Vec2> {
        let [rw, rh] = self.reference_size;
        if point.x < 0.0 || point.y < 0.0 || point.x >= rw as f32 || point.y >= rh as f32 {
            return None;
        }
        let px = glam::Vec2::from(self.origin_px) + point * glam::Vec2::from(self.scale);
        let [vw, vh] = self.viewport_px;
        if px.x < 0.0 || px.y < 0.0 || px.x >= vw as f32 || px.y >= vh as f32 {
            return None;
        }
        Some(px / self.pixels_per_point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    const BARS: ReferenceFit = ReferenceFit::IntegerBars {
        colour: Colour::BLACK,
    };

    #[test]
    fn integer_bars_take_the_largest_whole_scale_and_centre() {
        let p = ReferencePlacement::new([320, 180], BARS, [1920.0, 1080.0], 1.0);
        assert_eq!(p.scale, [6.0, 6.0]);
        assert_eq!(p.origin_px, [0.0, 0.0]);
        let p = ReferencePlacement::new([320, 180], BARS, [1366.0, 768.0], 1.0);
        assert_eq!(p.scale, [4.0, 4.0]);
        assert_eq!(p.origin_px, [43.0, 24.0]);
    }

    #[test]
    fn integer_crop_covers_the_viewport() {
        let p =
            ReferencePlacement::new([320, 180], ReferenceFit::IntegerCrop, [1366.0, 768.0], 1.0);
        assert_eq!(p.scale, [5.0, 5.0]);
        assert_eq!(p.origin_px, [-117.0, -66.0]);
        let p =
            ReferencePlacement::new([320, 180], ReferenceFit::IntegerCrop, [1920.0, 1080.0], 1.0);
        assert_eq!(p.scale, [6.0, 6.0]);
    }

    #[test]
    fn a_viewport_smaller_than_the_reference_keeps_one_pixel_per_texel() {
        let p = ReferencePlacement::new([320, 180], BARS, [200.0, 100.0], 1.0);
        assert_eq!(p.scale, [1.0, 1.0]);
        assert_eq!(p.origin_px, [-60.0, -40.0]);
    }

    #[test]
    fn round_trips_at_corners_and_centre_for_every_fit() {
        for fit in [BARS, ReferenceFit::IntegerCrop, ReferenceFit::Stretch] {
            for (size, ppp) in [([1366.0, 768.0], 1.0), ([960.0, 540.0], 2.0)] {
                let p = ReferencePlacement::new([320, 180], fit, size, ppp);
                // The viewport's own corners and centre, kept just inside.
                let e = 0.25;
                for w in [
                    Vec2::new(e, e),
                    Vec2::new(size[0] - e, e),
                    Vec2::new(e, size[1] - e),
                    Vec2::new(size[0] - e, size[1] - e),
                    Vec2::new(size[0] / 2.0, size[1] / 2.0),
                ] {
                    if let Some(r) = p.window_to_reference(w) {
                        let back = p.reference_to_window(r).unwrap();
                        assert!((back - w).length() < 1.0e-3, "{fit:?} {w} -> {r} -> {back}");
                    } else {
                        assert!(
                            matches!(fit, ReferenceFit::IntegerBars { .. }),
                            "{fit:?} {size:?} {w}"
                        );
                    }
                }
                // The reference target's corners and centre.
                for r in [
                    Vec2::new(0.0, 0.0),
                    Vec2::new(319.5, 0.0),
                    Vec2::new(0.0, 179.5),
                    Vec2::new(319.5, 179.5),
                    Vec2::new(160.0, 90.0),
                ] {
                    if let Some(w) = p.reference_to_window(r) {
                        let back = p.window_to_reference(w).unwrap();
                        assert!((back - r).length() < 1.0e-3, "{fit:?} {r} -> {w} -> {back}");
                    } else {
                        assert_eq!(fit, ReferenceFit::IntegerCrop);
                    }
                }
            }
        }
    }

    #[test]
    fn a_point_in_the_bars_maps_to_nothing() {
        let p = ReferencePlacement::new([320, 180], BARS, [1366.0, 768.0], 1.0);
        assert_eq!(p.window_to_reference(Vec2::new(10.0, 300.0)), None);
        assert_eq!(p.window_to_reference(Vec2::new(600.0, 10.0)), None);
        assert_eq!(p.window_to_reference(Vec2::new(1360.0, 300.0)), None);
        assert_eq!(
            p.window_to_reference(Vec2::new(43.0, 24.0)),
            Some(Vec2::ZERO)
        );
    }

    #[test]
    fn stretch_fills_the_viewport() {
        let p = ReferencePlacement::new([320, 180], ReferenceFit::Stretch, [1000.0, 1000.0], 1.0);
        assert_eq!(p.origin_px, [0.0, 0.0]);
        assert_eq!(
            p.window_to_reference(Vec2::new(500.0, 500.0)),
            Some(Vec2::new(160.0, 90.0))
        );
    }
}
