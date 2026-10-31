//! The world origin the renderer draws relative to.
//!
//! Positions reach the GPU as offsets from an origin point chosen once per
//! scene prepare, so a scene far from the world origin keeps the precision
//! of one near it. The subtraction runs on the CPU in f64, and only the small
//! difference is cast to f32.

use glam::{DVec3, Mat4, Vec3};

/// Where the coordinates the renderer sends to the GPU are measured from.
///
/// The default is [`snapped`](WorldOrigin::snapped). Set it on [`RendererConfig::world_origin`](crate::RendererConfig::world_origin),
/// or per frame with [`SceneFrame::with_world_origin`](crate::SceneFrame::with_world_origin).
/// The renderer picks the origin once per scene prepare from the frame's
/// camera, so every viewport over one scene shares it; a viewport far from
/// the scene camera keeps only the precision its distance from the origin
/// allows, so give widely separated viewports a [`Fixed`](Self::Fixed)
/// origin between them or one each.
///
/// Item types that write world-space positions into their own buffers
/// subtract [`PrepareContext::world_origin`](crate::plugin_api::PrepareContext::world_origin)
/// before the cast to f32.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum WorldOrigin {
    /// Draw in world coordinates. Precision falls away from the world origin:
    /// at 1,000 km an f32 step is 6 cm.
    Zero,
    /// Draw relative to a fixed point.
    Fixed(DVec3),
    /// Follow the camera on a grid: the origin moves to the grid point
    /// nearest the eye when the eye strays more than `cell * (0.5 + hysteresis)`
    /// from it along any axis. Build with [`snapped`](WorldOrigin::snapped).
    #[non_exhaustive]
    Snapped {
        /// Grid spacing in world units. `None` sizes it from the camera when it
        /// snaps: a tenth of the culling frustum's width at the far plane (or
        /// of an orthographic view's width), rounded up to a power of two so
        /// the subtraction stays exact.
        cell: Option<f64>,
        /// Extra distance, as a fraction of `cell`, the eye may stray past
        /// half a cell before the origin moves. Keeps the origin still while
        /// the camera hovers near a cell boundary.
        hysteresis: f64,
    },
}

impl Default for WorldOrigin {
    /// [`WorldOrigin::snapped`]: the origin follows the camera.
    fn default() -> Self {
        Self::snapped()
    }
}

impl WorldOrigin {
    /// An origin that follows the camera, with an automatic cell and a
    /// hysteresis of a quarter cell.
    pub fn snapped() -> Self {
        Self::Snapped {
            cell: None,
            hysteresis: 0.25,
        }
    }

    /// Set a snapped origin's grid spacing. Other origins are returned as they
    /// are.
    pub fn with_cell(self, cell: f64) -> Self {
        match self {
            Self::Snapped { hysteresis, .. } => Self::Snapped {
                cell: Some(cell),
                hysteresis,
            },
            other => other,
        }
    }

    /// Set a snapped origin's hysteresis. Other origins are returned as they
    /// are.
    pub fn with_hysteresis(self, hysteresis: f64) -> Self {
        match self {
            Self::Snapped { cell, .. } => Self::Snapped {
                cell,
                hysteresis: hysteresis.max(0.0),
            },
            other => other,
        }
    }
}

/// The origin the renderer is drawing relative to this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OriginState {
    origin: DVec3,
    /// Moves every time `origin` does. Caches holding positions relative to
    /// the origin key on it.
    epoch: u32,
    /// The snapped grid spacing last used, refreshed only on a snap.
    cell: f64,
}

impl Default for OriginState {
    fn default() -> Self {
        Self {
            origin: DVec3::ZERO,
            epoch: 0,
            cell: 0.0,
        }
    }
}

impl OriginState {
    /// Choose this frame's origin for `mode` from `camera`, the world-space
    /// scene camera. Returns true when the origin moved.
    pub(crate) fn resolve(
        &mut self,
        mode: WorldOrigin,
        camera: &crate::renderer::RenderCamera,
    ) -> bool {
        let next = match mode {
            WorldOrigin::Zero => DVec3::ZERO,
            WorldOrigin::Fixed(p) => p,
            WorldOrigin::Snapped { cell, hysteresis } => {
                let eye = camera.eye_position();
                let limit = self.cell * (0.5 + hysteresis.max(0.0));
                let strayed = self.cell <= 0.0
                    || cell.is_some_and(|c| c != self.cell)
                    || (eye - self.origin).abs().max_element() > limit;
                if strayed {
                    self.cell = match cell {
                        Some(c) if c.is_finite() && c > 0.0 => c,
                        _ => auto_cell(camera),
                    };
                    (eye / self.cell).round() * self.cell
                } else {
                    self.origin
                }
            }
        };
        if !matches!(mode, WorldOrigin::Snapped { .. }) {
            self.cell = 0.0;
        }
        if next == self.origin {
            return false;
        }
        self.origin = next;
        self.epoch = self.epoch.wrapping_add(1);
        true
    }

    /// The origin, in world coordinates.
    pub(crate) fn origin(&self) -> DVec3 {
        self.origin
    }

    /// Bumped each time the origin moves.
    pub(crate) fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Whether positions are drawn as they are.
    pub(crate) fn is_zero(&self) -> bool {
        self.origin == DVec3::ZERO
    }

    /// `p` relative to the origin. The subtraction is exact in f64 for an f32
    /// input; only the result is rounded, and it is small near the camera.
    pub(crate) fn rel(&self, p: Vec3) -> Vec3 {
        rel(p, self.origin)
    }

    /// A model matrix with its translation made relative to the origin.
    /// Rotation and scale are unchanged.
    pub(crate) fn rel_model(&self, m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        rel_model(m, self.origin)
    }

    /// An item's placement as an f32 model matrix relative to the origin.
    pub(crate) fn rel_transform(&self, t: &crate::Transform) -> [[f32; 4]; 4] {
        t.to_mat4_relative(self.origin).to_cols_array_2d()
    }

    /// The camera the GPU draws with: `camera` with its view and eye relative
    /// to the origin.
    pub(crate) fn camera(
        &self,
        camera: &crate::renderer::RenderCamera,
    ) -> crate::renderer::RenderCamera {
        camera.relative_to(self.origin)
    }

    /// A world-space plane `dot(n, p) + d = 0` moved into relative
    /// coordinates: returns the new `d`.
    pub(crate) fn rel_plane_d(&self, normal: Vec3, d: f32) -> f32 {
        if self.is_zero() {
            return d;
        }
        (f64::from(d) + normal.as_dvec3().dot(self.origin)) as f32
    }
}

/// `p - origin` in f64, cast to f32.
pub(crate) fn rel(p: Vec3, origin: DVec3) -> Vec3 {
    if origin == DVec3::ZERO {
        return p;
    }
    (p.as_dvec3() - origin).as_vec3()
}

/// `m` with its translation column made relative to `origin`.
pub(crate) fn rel_model(mut m: [[f32; 4]; 4], origin: DVec3) -> [[f32; 4]; 4] {
    if origin == DVec3::ZERO {
        return m;
    }
    let t = rel(Vec3::new(m[3][0], m[3][1], m[3][2]), origin);
    m[3][0] = t.x;
    m[3][1] = t.y;
    m[3][2] = t.z;
    m
}

/// A world-to-view matrix for points relative to `origin`: the rotation of
/// `view` with the translation rebuilt in f64 from the eye, as
/// `-R (eye - origin)`, rather than `view * translate(origin)` in f32.
pub(crate) fn rel_view(view: Mat4, eye: DVec3, origin: DVec3) -> Mat4 {
    if origin == DVec3::ZERO {
        return view;
    }
    let r = glam::DMat3::from_cols(
        view.x_axis.truncate().as_dvec3(),
        view.y_axis.truncate().as_dvec3(),
        view.z_axis.truncate().as_dvec3(),
    );
    let t = -(r * (eye - origin));
    let mut out = view;
    out.w_axis = t.as_vec3().extend(view.w_axis.w);
    out
}

/// The automatic snap cell for `camera`: a tenth of the culling frustum's
/// width at the far plane, or of an orthographic view's width, rounded up to
/// a power of two.
fn auto_cell(camera: &crate::renderer::RenderCamera) -> f64 {
    let p = camera.culling_projection();
    let width = if camera.is_orthographic() {
        2.0 / f64::from(p.x_axis.x)
    } else {
        let far = f64::from(camera.depth_range().1);
        2.0 * far / f64::from(p.x_axis.x)
    };
    let cell = 0.1 * width;
    if cell.is_finite() && cell > 0.0 {
        2f64.powf(cell.log2().ceil())
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::RenderCamera;

    fn camera_at(eye: Vec3) -> RenderCamera {
        RenderCamera::from_matrices(
            Mat4::look_at_rh(eye, eye + Vec3::Y, Vec3::Z),
            Mat4::perspective_infinite_reverse_rh(1.0, 1.0, 0.1),
        )
        .with_depth_range(0.1, 100.0)
    }

    #[test]
    fn zero_leaves_values_untouched() {
        let mut s = OriginState::default();
        assert!(!s.resolve(WorldOrigin::Zero, &camera_at(Vec3::splat(1.0e6))));
        let p = Vec3::new(1.0e6 + 0.0625, -3.5, 7.25);
        assert_eq!(s.rel(p).to_array(), p.to_array());
        let view = Mat4::look_at_rh(p, Vec3::ZERO, Vec3::Z);
        assert_eq!(rel_view(view, p.as_dvec3(), DVec3::ZERO), view);
        assert_eq!(s.epoch(), 0);
    }

    #[test]
    fn rel_is_exact_for_f32_inputs() {
        let origin = DVec3::new(1.0e6, 0.0, 0.0);
        let p = Vec3::new(1_000_000.0625, 0.0, 0.0);
        assert_eq!(rel(p, origin).x, 0.0625);
    }

    #[test]
    fn snapped_moves_only_past_the_hysteresis() {
        let mode = WorldOrigin::snapped().with_cell(16.0);
        let mut s = OriginState::default();
        assert!(s.resolve(mode, &camera_at(Vec3::new(1.0e6 + 3.0, 0.0, 0.0))));
        assert_eq!(s.origin(), DVec3::new(1.0e6, 0.0, 0.0));
        let epoch = s.epoch();
        // Inside 0.75 of a cell: no snap.
        assert!(!s.resolve(mode, &camera_at(Vec3::new(1.0e6 + 11.0, 0.0, 0.0))));
        assert_eq!(s.epoch(), epoch);
        // Past it: snap to the nearest grid point.
        assert!(s.resolve(mode, &camera_at(Vec3::new(1.0e6 + 13.0, 0.0, 0.0))));
        assert_eq!(s.origin(), DVec3::new(1.0e6 + 16.0, 0.0, 0.0));
        assert_eq!(s.epoch(), epoch + 1);
    }

    #[test]
    fn automatic_cell_is_a_power_of_two() {
        let cell = auto_cell(&camera_at(Vec3::ZERO));
        assert_eq!(cell.log2().fract(), 0.0);
        // A tenth of 2 * 100 / x_scale, with x_scale = 1 / tan(0.5).
        let expect = 0.1 * 2.0 * 100.0 * 0.5f64.tan();
        assert!(cell >= expect && cell < 2.0 * expect);
    }

    #[test]
    fn rel_view_matches_view_on_relative_points() {
        let eye = Vec3::new(1.0e6, 2.0e5, 30.0);
        let target = eye + Vec3::new(3.0, 4.0, -1.0);
        let view = Mat4::look_at_rh(eye, target, Vec3::Z);
        let origin = DVec3::new(1.0e6 - 8.0, 2.0e5 + 4.0, 0.0);
        let rv = rel_view(view, eye.as_dvec3(), origin);
        // The eye maps to the view-space origin.
        let e = rv.transform_point3(rel(eye, origin));
        assert!(e.length() < 1e-5, "{e}");
        let t = rv.transform_point3(rel(target, origin));
        assert!((t.length() - (target - eye).length()).abs() < 1e-3);
    }
}
