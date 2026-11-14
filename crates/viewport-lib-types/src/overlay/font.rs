//! Font handle for overlay text items.

/// Handle to an uploaded font, used by overlay label and glyph-run items.
///
/// Obtain one from `DeviceResources::upload_font`. Pass `None` (or omit the
/// field) on overlay items to use the built-in default font; pass
/// `Some(handle)` to use a user-supplied TTF font.
///
/// This is a plain index into the renderer's font store: the store itself (and
/// the font rasterisation behind it) lives in `viewport-lib`, but the handle a
/// consumer names on an overlay item is pure data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontHandle(pub usize);

/// The typography of an overlay text item: which font, at what size.
///
/// Shared by [`LabelItem`](crate::overlay::LabelItem) and
/// [`GlyphRunItem`](crate::overlay::GlyphRunItem), which is the point: the two
/// families differ in what they are given (a string to lay out, or glyphs
/// already positioned) and not in how the glyphs are rasterised.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TextStyle {
    /// Font to rasterise with. `None` uses the built-in default font.
    pub font: Option<FontHandle>,
    /// Size in logical pixels, or in world units when `sizing` is
    /// [`TextSizing::World`].
    pub size: f32,
    /// What `size` is measured in. Default [`TextSizing::Screen`].
    pub sizing: TextSizing,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font: None,
            size: 14.0,
            sizing: TextSizing::Screen,
        }
    }
}

impl TextStyle {
    /// The default font at `size` logical pixels.
    pub fn new(size: f32) -> Self {
        Self {
            font: None,
            size,
            sizing: TextSizing::Screen,
        }
    }

    /// Set the font.
    pub fn with_font(mut self, font: FontHandle) -> Self {
        self.font = Some(font);
        self
    }

    /// Set the size in logical pixels.
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Set what the size is measured in.
    pub fn with_sizing(mut self, sizing: TextSizing) -> Self {
        self.sizing = sizing;
        self
    }

    /// The size and opacity text in this style draws at on screen, hanging
    /// from `anchor`, under a camera with `view` and `proj` drawn relative to
    /// `world_origin` over a `viewport` logical pixels across at
    /// `pixels_per_point`.
    ///
    /// With [`TextSizing::Screen`] this is `size` at full opacity. With
    /// [`TextSizing::World`] it is [`WorldTextSize::fit`] at the logical
    /// pixels one world unit covers on the plane through the anchor, with the
    /// device-pixel size rounded to [`TextSizing::step_px`], which is the
    /// size the renderer draws a screen overlay label at. `None` when the
    /// plane is seen edge-on or the anchor is behind the camera.
    ///
    /// A backing panel measured at this size lands where the text does.
    pub fn resolve_screen_size_at(
        &self,
        anchor: &crate::overlay::OverlayOrigin,
        viewport: [f32; 2],
        pixels_per_point: f32,
        view: &glam::Mat4,
        proj: &glam::Mat4,
        world_origin: glam::DVec3,
    ) -> Option<TextFit> {
        let TextSizing::World(world) = self.sizing else {
            return Some(TextFit {
                size: self.size,
                opacity: 1.0,
            });
        };
        let view_proj = *proj * *view;
        let point = match anchor {
            crate::overlay::OverlayOrigin::World(p) => (*p - world_origin).as_vec3(),
            crate::overlay::OverlayOrigin::Viewport(a) => {
                let at = [a.x.coord(viewport[0]), a.y.coord(viewport[1])];
                screen_point_on_plane(&view_proj, at, viewport, world, world_origin)?
            }
        };
        let per_unit = plane_px_per_unit(&view_proj, world.plane, point, viewport)?;
        let fit = world.fit(self.size, per_unit)?;
        let ppp = pixels_per_point.max(1.0e-6);
        Some(TextFit {
            size: TextSizing::step_px(fit.size * ppp) / ppp,
            opacity: fit.opacity,
        })
    }
}

/// What a text item's [`TextStyle::size`] is measured in.
///
/// A [`LabelItem`](crate::overlay::LabelItem) or
/// [`GlyphRunItem`](crate::overlay::GlyphRunItem) on the screen is sized in
/// logical pixels and keeps its size as the camera zooms. A world-sized one
/// gives its height in world units and grows and shrinks with the camera.
///
/// There are two ways to put world-sized text in a scene:
///
/// - In a [`WorldOverlayLayer`](crate::overlay::WorldOverlayLayer), where
///   every length is already in world units: the text lies on the layer's
///   plane and turns, tilts and scales with it. Use this for signage and
///   labels that belong to the world. `Screen` sizing there already measures
///   `size` in world units; `World` sizing adds the screen-pixel clamps and
///   fade of [`WorldTextSize`] and rasterises at stepped sizes, so a zoom
///   reuses a bounded set of glyphs. Its plane and elevation are the
///   layer's, not the ones given here.
/// - As a screen overlay item with a world anchor and `World` sizing: the
///   text always faces the camera and draws over the 3D content, at the size
///   `size` world units cover at the anchor. Use this for name tags, damage
///   numbers and callouts over 3D objects.
///
/// Screen overlay text with `World` sizing draws at a device-pixel size
/// rounded to [`step_px`](Self::step_px), within 2% of the exact size, so it
/// rasterises at its drawn size and snaps to the pixel grid like other
/// screen text while a zoom reuses a bounded set of glyph rasters. Text
/// above 256 device pixels draws a 256 pixel raster magnified. The anchor
/// stays where it projects and the text grows about it, placed by
/// `anchoring.align`. Shadow lengths and `max_width` are in world units like
/// `size`, so they scale with the text, and a glyph run's positions are in
/// the same units as its `size`; `transform.translate` stays in logical
/// pixels.
///
/// Retained overlay groups draw their text at `size` in the group's own
/// units: world sizing applies to a frame's own labels and glyph runs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum TextSizing {
    /// `size` is in the units of the space the item draws in: logical pixels
    /// on the screen, world units in a world overlay layer.
    #[default]
    Screen,
    /// `size` is a height in world units, measured on a plane.
    World(WorldTextSize),
}

impl TextSizing {
    /// Ratio between neighbouring steps of [`step_px`](Self::step_px).
    pub const STEP: f32 = 1.04;

    /// The step `px` device pixels rounds to: the nearest `48 * 1.04^k` for
    /// a whole `k`. These are the sizes the glyph atlas rasterises text
    /// above 48 device pixels at, extended below 48.
    pub fn step_px(px: f32) -> f32 {
        if !(px > 0.0) || !px.is_finite() {
            return px.max(0.0);
        }
        let steps = ((px / 48.0).ln() / Self::STEP.ln()).round();
        48.0 * Self::STEP.powf(steps)
    }
}

/// How world-sized text is measured and kept legible.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct WorldTextSize {
    /// The plane `size` is measured on. Default
    /// [`WorldPlane::Xy`](crate::overlay::WorldPlane::Xy).
    pub plane: crate::overlay::WorldPlane,
    /// Height of the plane above the world origin, along its normal, for an
    /// item with a viewport anchor: its size is measured where the ray
    /// through the anchor meets the plane. A world anchor is measured at the
    /// anchor itself. Default `0.0`.
    pub elevation: f32,
    /// Smallest on-screen size in logical pixels: the text stops shrinking
    /// here. Default `None`.
    pub min_px: Option<f32>,
    /// Largest on-screen size in logical pixels: the text stops growing here.
    /// Default `None`.
    pub max_px: Option<f32>,
    /// `[gone, full]`, in logical pixels of the unclamped size: the text is
    /// hidden at or below `gone`, fully opaque at or above `full`, and fades
    /// linearly between. Default `None`.
    pub fade_px: Option<[f32; 2]>,
}

impl Default for WorldTextSize {
    fn default() -> Self {
        Self {
            plane: crate::overlay::WorldPlane::Xy,
            elevation: 0.0,
            min_px: None,
            max_px: None,
            fade_px: None,
        }
    }
}

/// The on-screen size and opacity of a text item this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TextFit {
    /// Size in logical pixels.
    pub size: f32,
    /// Opacity multiplier from the fade, `0.0..=1.0`.
    pub opacity: f32,
}

impl WorldTextSize {
    /// Measured on `plane`, with no clamps or fade.
    pub fn new(plane: crate::overlay::WorldPlane) -> Self {
        Self {
            plane,
            ..Self::default()
        }
    }

    /// Builder: set the plane's elevation.
    pub fn with_elevation(mut self, elevation: f32) -> Self {
        self.elevation = elevation;
        self
    }

    /// Builder: never draw smaller than `px` logical pixels.
    pub fn with_min_px(mut self, px: f32) -> Self {
        self.min_px = Some(px);
        self
    }

    /// Builder: never draw larger than `px` logical pixels.
    pub fn with_max_px(mut self, px: f32) -> Self {
        self.max_px = Some(px);
        self
    }

    /// Builder: fade out between `full` and `gone` logical pixels.
    pub fn with_fade_px(mut self, gone: f32, full: f32) -> Self {
        self.fade_px = Some([gone, full]);
        self
    }

    /// The on-screen size of text `height` world units tall where one world
    /// unit covers `px_per_unit` logical pixels: `height * px_per_unit`,
    /// clamped to `min_px` and `max_px`, with the fade of `fade_px` taken on
    /// the unclamped size. `None` when the size is not finite.
    pub fn fit(&self, height: f32, px_per_unit: f32) -> Option<TextFit> {
        let raw = height * px_per_unit;
        if !raw.is_finite() {
            return None;
        }
        let mut size = raw;
        if let Some(max) = self.max_px {
            size = size.min(max);
        }
        if let Some(min) = self.min_px {
            size = size.max(min);
        }
        let opacity = match self.fade_px {
            Some([gone, full]) if full > gone => ((raw - gone) / (full - gone)).clamp(0.0, 1.0),
            Some([gone, _]) => {
                if raw > gone {
                    1.0
                } else {
                    0.0
                }
            }
            None => 1.0,
        };
        Some(TextFit { size, opacity })
    }
}

/// Logical pixels one world unit covers on `plane` at `point` (relative to
/// the world origin the camera is drawn at), seen through `view_proj` over a
/// `viewport` logical pixels across.
///
/// The square root of the area one unit square of the plane covers there:
/// exact for an orthographic camera looking straight at the plane, and the
/// mean of the two axes' scales under a tilted or perspective view. `None`
/// when the plane is seen edge-on or the point is behind the camera.
pub fn plane_px_per_unit(
    view_proj: &glam::Mat4,
    plane: crate::overlay::WorldPlane,
    point: glam::Vec3,
    viewport: [f32; 2],
) -> Option<f32> {
    let c = *view_proj * point.extend(1.0);
    if !(c.w > 1.0e-12) {
        return None;
    }
    // d(clip / w) along each plane axis at the point, then NDC to pixels.
    let d = |axis: glam::Vec3| {
        let a = *view_proj * axis.extend(0.0);
        glam::Vec2::new(
            (a.x * c.w - c.x * a.w) / (c.w * c.w) * viewport[0] * 0.5,
            (a.y * c.w - c.y * a.w) / (c.w * c.w) * viewport[1] * 0.5,
        )
    };
    let du = d(plane.x_axis());
    let dv = d(plane.y_axis());
    let area = (du.x * dv.y - du.y * dv.x).abs();
    let per_unit = area.sqrt();
    (per_unit.is_finite() && per_unit > 0.0).then_some(per_unit)
}

/// Where the ray through screen point `at` (logical pixels) meets the plane
/// of `world`, relative to `world_origin`.
fn screen_point_on_plane(
    view_proj: &glam::Mat4,
    at: [f32; 2],
    viewport: [f32; 2],
    world: WorldTextSize,
    world_origin: glam::DVec3,
) -> Option<glam::Vec3> {
    let inv = view_proj.inverse();
    let ndc = glam::Vec2::new(
        at[0] / viewport[0].max(1.0e-6) * 2.0 - 1.0,
        1.0 - at[1] / viewport[1].max(1.0e-6) * 2.0,
    );
    // Two depths inside the clip range under both depth conventions.
    let a = inv.project_point3(ndc.extend(0.3));
    let b = inv.project_point3(ndc.extend(0.7));
    let dir = b - a;
    let n = world.plane.normal();
    let denom = dir.dot(n);
    if denom.abs() < 1.0e-12 {
        return None;
    }
    // The plane is at `elevation` in absolute world coordinates.
    let offset = world.elevation - world_origin.dot(n.as_dvec3()) as f32;
    let t = (offset - a.dot(n)) / denom;
    let p = a + dir * t;
    p.is_finite().then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::{OverlayOrigin, WorldPlane};

    /// An orthographic camera looking down -Z, `upp` world units per logical
    /// pixel over `px`.
    fn ortho(upp: f32, px: [f32; 2]) -> (glam::Mat4, glam::Mat4) {
        let hw = px[0] * 0.5 * upp;
        let hh = px[1] * 0.5 * upp;
        let proj = glam::Mat4::orthographic_rh(-hw, hw, -hh, hh, 0.1, 100.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(1.0, 2.0, 10.0),
            glam::Vec3::new(1.0, 2.0, 0.0),
            glam::Vec3::Y,
        );
        (view, proj)
    }

    #[test]
    fn world_text_is_its_height_times_the_pixels_per_unit_and_clamped() {
        let px = [640.0, 360.0];
        let style = TextStyle::new(2.0).with_sizing(TextSizing::World(
            WorldTextSize::new(WorldPlane::Xy)
                .with_min_px(10.0)
                .with_max_px(48.0),
        ));
        let anchor = OverlayOrigin::World(glam::DVec3::new(1.5, 2.5, 0.0));
        // 4, 12 and 40 px per unit: 8 px clamps to 10, 24 px passes, 80 px
        // clamps to 48.
        for (per_unit, expect) in [(4.0_f32, 10.0_f32), (12.0, 24.0), (40.0, 48.0)] {
            let (view, proj) = ortho(1.0 / per_unit, px);
            let measured = plane_px_per_unit(
                &(proj * view),
                WorldPlane::Xy,
                glam::Vec3::new(1.5, 2.5, 0.0),
                px,
            )
            .unwrap();
            assert!(
                (measured - per_unit).abs() / per_unit < 1.0e-4,
                "{measured}"
            );
            let exact = match style.sizing {
                TextSizing::World(w) => w.fit(style.size, measured).unwrap(),
                TextSizing::Screen => unreachable!(),
            };
            assert!(
                (exact.size - expect).abs() < 1.0e-3,
                "{per_unit}: {exact:?}"
            );
            let drawn = style
                .resolve_screen_size_at(&anchor, px, 2.0, &view, &proj, glam::DVec3::ZERO)
                .unwrap();
            // Rounded to a step, within 2% of the exact size.
            assert!((drawn.size - expect).abs() <= expect * 0.021, "{drawn:?}");
            let device = drawn.size * 2.0;
            assert!((TextSizing::step_px(device) - device).abs() < 1.0e-3);
        }
        // Unclamped, the size is the height times the pixels per unit.
        let free = WorldTextSize::new(WorldPlane::Xy);
        for per_unit in [3.0_f32, 17.0, 90.0] {
            assert_eq!(free.fit(2.0, per_unit).unwrap().size, 2.0 * per_unit);
        }
    }

    #[test]
    fn fade_runs_on_the_unclamped_size() {
        let w = WorldTextSize::new(WorldPlane::Xy)
            .with_min_px(10.0)
            .with_fade_px(4.0, 8.0);
        assert_eq!(
            w.fit(1.0, 2.0).unwrap(),
            TextFit {
                size: 10.0,
                opacity: 0.0
            }
        );
        assert_eq!(
            w.fit(1.0, 6.0).unwrap(),
            TextFit {
                size: 10.0,
                opacity: 0.5
            }
        );
        assert_eq!(
            w.fit(1.0, 20.0).unwrap(),
            TextFit {
                size: 20.0,
                opacity: 1.0
            }
        );
    }

    #[test]
    fn steps_are_a_ratio_ladder_through_48() {
        assert_eq!(TextSizing::step_px(48.0), 48.0);
        assert!((TextSizing::step_px(49.0) - 48.0 * 1.04).abs() < 1.0e-3);
        let mut px = 6.0_f32;
        while px < 400.0 {
            let s = TextSizing::step_px(px);
            assert!((s - px).abs() <= px * 0.0201, "{px} -> {s}");
            px += 0.37;
        }
    }

    #[test]
    fn a_viewport_anchor_measures_where_its_ray_meets_the_plane() {
        let px = [400.0, 300.0];
        let proj = glam::Mat4::perspective_rh(0.8, px[0] / px[1], 0.1, 100.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, -6.0, 6.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
        );
        let world = WorldTextSize::new(WorldPlane::Xy);
        let centre = OverlayOrigin::Viewport(crate::overlay::Alignment::new(
            crate::overlay::AnchorX::Middle,
            crate::overlay::AnchorY::Middle,
        ));
        let style = TextStyle::new(1.0).with_sizing(TextSizing::World(world));
        let a = style
            .resolve_screen_size_at(&centre, px, 1.0, &view, &proj, glam::DVec3::ZERO)
            .unwrap();
        let b = style
            .resolve_screen_size_at(
                &OverlayOrigin::World(glam::DVec3::ZERO),
                px,
                1.0,
                &view,
                &proj,
                glam::DVec3::ZERO,
            )
            .unwrap();
        assert!((a.size - b.size).abs() < 1.0e-2, "{a:?} {b:?}");
    }
}
