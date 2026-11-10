//! A flat orthographic camera for 2D views, and the controller that drives it.
//!
//! [`Camera2d`] describes a view in 2D terms (a centre on a world plane, a zoom
//! as world units per pixel, a turn about the view axis) and produces the
//! orthographic [`Camera`](crate::Camera) the renderer draws with. It looks straight at its
//! plane and never orbits.
//!
//! [`Camera2dController`] applies drag-to-pan and zoom-about-the-cursor from an
//! [`ActionFrame`](crate::input::ActionFrame) to a `Camera2d`. Pair it with
//! [`BindingPreset::Planar`](crate::input::BindingPreset::Planar).
//!
//! Screen positions are viewport-local logical pixels with the origin at the
//! top left and y down, the same space as
//! [`PointerFrame::cursor`](crate::input::PointerFrame::cursor). Plane
//! positions are `Vec2`s in world units along the plane's two axes; see
//! [`Plane2d`].

use glam::{DMat3, DQuat, DVec3, Vec2, Vec3};
use viewport_lib_types::camera::{Pose, Projection};

use crate::Camera;
use crate::input::action_frame::ActionFrame;
use crate::input::context::ViewportContext;
use crate::input::event::ViewportEvent;
use crate::input::preset::planar_bindings;
use crate::input::viewport_binding::ViewportBinding;
use crate::input::viewport_input::ViewportInput;

/// The world plane a [`Camera2d`] looks at.
///
/// Both choices put plane x on world +X (screen right at rotation 0) and plane
/// y on the world axis that should read as up. They match
/// `viewport_lib::WorldPlane` (the plane of a world overlay layer) and the
/// tilemap's `TilemapPlane`, axis for axis, and convert to `WorldPlane` with
/// `From`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Plane2d {
    /// The world XY plane seen from above, looking along -Z. Plane y is world
    /// +Y. The plane for a top-down view or a document.
    #[default]
    Xy,
    /// The world XZ plane seen from the front, looking along +Y. Plane y is
    /// world +Z, so up on screen is up in the Z-up world: the plane for a side
    /// view.
    Xz,
}

impl From<Plane2d> for viewport_lib_types::overlay::WorldPlane {
    fn from(plane: Plane2d) -> Self {
        match plane {
            Plane2d::Xy => Self::Xy,
            Plane2d::Xz => Self::Xz,
        }
    }
}

impl Plane2d {
    /// World direction of plane +x.
    pub fn x_axis(self) -> Vec3 {
        Vec3::X
    }

    /// World direction of plane +y.
    pub fn y_axis(self) -> Vec3 {
        match self {
            Plane2d::Xy => Vec3::Y,
            Plane2d::Xz => Vec3::Z,
        }
    }

    /// The direction the camera looks: into the plane.
    pub fn view_direction(self) -> Vec3 {
        match self {
            Plane2d::Xy => Vec3::NEG_Z,
            Plane2d::Xz => Vec3::Y,
        }
    }

    /// The world point at plane position `p`, on the plane through the origin.
    pub fn to_world(self, p: Vec2) -> DVec3 {
        (self.x_axis() * p.x + self.y_axis() * p.y).as_dvec3()
    }

    /// The plane position of a world point, dropping its depth along the view
    /// direction.
    pub fn from_world(self, p: DVec3) -> Vec2 {
        match self {
            Plane2d::Xy => Vec2::new(p.x as f32, p.y as f32),
            Plane2d::Xz => Vec2::new(p.x as f32, p.z as f32),
        }
    }
}

/// An axis-aligned rectangle, in plane units or in pixels depending on use.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rect2d {
    /// The corner with the smaller coordinates.
    pub min: Vec2,
    /// The corner with the larger coordinates.
    pub max: Vec2,
}

impl Rect2d {
    /// The rectangle spanning two corners, in either order.
    pub fn new(a: Vec2, b: Vec2) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// The rectangle `size` across, centred on `centre`.
    pub fn from_centre_size(centre: Vec2, size: Vec2) -> Self {
        let half = size.abs() * 0.5;
        Self {
            min: centre - half,
            max: centre + half,
        }
    }

    /// The middle of the rectangle.
    pub fn centre(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Width and height.
    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// Whether `p` lies inside or on the edge.
    pub fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

/// A decaying screen shake in progress.
#[derive(Clone, Copy, Debug, Default)]
struct Shake {
    amplitude: f32,
    duration: f32,
    elapsed: f32,
    /// The current offset in pixels, x right and y down.
    offset: Vec2,
}

impl Shake {
    fn remaining_amplitude(&self) -> f32 {
        if self.duration <= 0.0 || self.elapsed >= self.duration {
            0.0
        } else {
            let left = 1.0 - self.elapsed / self.duration;
            self.amplitude * left * left
        }
    }
}

/// A flat orthographic camera described in 2D terms.
///
/// The view is a `centre` on a [`Plane2d`], a zoom as `units_per_pixel` (world
/// units across one logical pixel), and a `rotation` about the view axis.
/// [`camera`](Self::camera) produces the orthographic [`Camera`] for the
/// current viewport size; give the renderer that each frame.
///
/// On top of the view sit a follow target with damping and a dead zone,
/// optional world bounds the visible area is kept inside, a decaying screen
/// shake, and pixel snapping for pixel art. [`update`](Self::update) advances
/// the follow and the shake once per frame.
///
/// [`screen_to_world`](Self::screen_to_world),
/// [`world_to_screen`](Self::world_to_screen) and
/// [`visible_rect`](Self::visible_rect) describe the view itself, without the
/// shake offset or pixel snapping that [`camera`](Self::camera) adds.
#[derive(Clone, Debug)]
pub struct Camera2d {
    /// The plane the camera looks at.
    pub plane: Plane2d,
    /// How far the eye sits in front of the plane, in world units. Content
    /// between the eye and the far plane draws, so keep it larger than the
    /// highest layer above the plane. Defaults to `100.0`.
    pub distance: f32,
    /// Physical pixels per logical pixel, for pixel snapping and
    /// [`snap_zoom_to_integer_scale`](Self::snap_zoom_to_integer_scale).
    /// Defaults to `1.0`.
    pub pixels_per_point: f32,
    /// Round the produced camera position so the plane origin lands on a whole
    /// device pixel, which keeps pixel art from shimmering as the camera moves.
    /// Off by default.
    pub snap_to_pixels: bool,
    /// The smallest units per pixel zooming reaches (the closest zoom).
    pub min_units_per_pixel: f32,
    /// The largest units per pixel zooming reaches (the furthest zoom).
    pub max_units_per_pixel: f32,
    /// How quickly the camera catches up with its follow target: the time in
    /// seconds for the distance left outside the dead zone to fall to about
    /// 37%. Zero or less snaps straight to it. Defaults to `0.25`.
    pub follow_time_constant: f32,
    /// The area around the screen centre the follow target can move in without
    /// the camera moving, in pixel offsets from the centre with x right and y
    /// down. Defaults to an empty rectangle at the centre, so the camera keeps
    /// the target centred.
    pub dead_zone: Rect2d,
    centre: Vec2,
    units_per_pixel: f32,
    rotation: f32,
    viewport_size: Vec2,
    bounds: Option<Rect2d>,
    follow_target: Option<Vec2>,
    shake: Shake,
    rng: u32,
}

impl Default for Camera2d {
    fn default() -> Self {
        Self::new(Plane2d::default())
    }
}

impl Camera2d {
    /// Default distance from the eye to the plane.
    pub const DEFAULT_DISTANCE: f32 = 100.0;
    /// Default zoom: 100 logical pixels per world unit.
    pub const DEFAULT_UNITS_PER_PIXEL: f32 = 0.01;
    /// Default follow time constant, in seconds.
    pub const DEFAULT_FOLLOW_TIME_CONSTANT: f32 = 0.25;

    /// A camera on `plane`, centred on the origin at the default zoom, for a
    /// 1 by 1 pixel viewport: call [`set_viewport_size`](Self::set_viewport_size)
    /// before producing a camera.
    pub fn new(plane: Plane2d) -> Self {
        Self {
            plane,
            distance: Self::DEFAULT_DISTANCE,
            pixels_per_point: 1.0,
            snap_to_pixels: false,
            min_units_per_pixel: 1.0e-6,
            max_units_per_pixel: 1.0e6,
            follow_time_constant: Self::DEFAULT_FOLLOW_TIME_CONSTANT,
            dead_zone: Rect2d::default(),
            centre: Vec2::ZERO,
            units_per_pixel: Self::DEFAULT_UNITS_PER_PIXEL,
            rotation: 0.0,
            viewport_size: Vec2::ONE,
            bounds: None,
            follow_target: None,
            shake: Shake::default(),
            rng: 0x9E37_79B9,
        }
    }

    /// Builder: set the centre; see [`set_centre`](Self::set_centre).
    pub fn with_centre(mut self, centre: Vec2) -> Self {
        self.set_centre(centre);
        self
    }

    /// Builder: set the zoom; see [`set_units_per_pixel`](Self::set_units_per_pixel).
    pub fn with_units_per_pixel(mut self, units_per_pixel: f32) -> Self {
        self.set_units_per_pixel(units_per_pixel);
        self
    }

    /// Builder: set the viewport size in logical pixels.
    pub fn with_viewport_size(mut self, size: Vec2) -> Self {
        self.set_viewport_size(size);
        self
    }

    /// Builder: set the rotation; see [`set_rotation`](Self::set_rotation).
    pub fn with_rotation(mut self, rotation: f32) -> Self {
        self.set_rotation(rotation);
        self
    }

    /// Builder: keep the visible area inside `bounds`; see
    /// [`set_bounds`](Self::set_bounds).
    pub fn with_bounds(mut self, bounds: Rect2d) -> Self {
        self.set_bounds(Some(bounds));
        self
    }

    // -----------------------------------------------------------------------
    // View state
    // -----------------------------------------------------------------------

    /// The plane position at the middle of the screen.
    pub fn centre(&self) -> Vec2 {
        self.centre
    }

    /// Move the middle of the screen to `centre`, then apply the bounds.
    pub fn set_centre(&mut self, centre: Vec2) {
        self.centre = centre;
        self.clamp_to_bounds();
    }

    /// World units across one logical pixel. Smaller is closer.
    pub fn units_per_pixel(&self) -> f32 {
        self.units_per_pixel
    }

    /// Set the zoom about the screen centre, clamped to
    /// `[min_units_per_pixel, max_units_per_pixel]`, then apply the bounds.
    pub fn set_units_per_pixel(&mut self, units_per_pixel: f32) {
        self.units_per_pixel = self.clamp_zoom(units_per_pixel);
        self.clamp_to_bounds();
    }

    /// The turn of the view about the view axis, in radians. Positive turns
    /// the camera counter-clockwise on the plane, so the content appears to
    /// turn clockwise.
    pub fn rotation(&self) -> f32 {
        self.rotation
    }

    /// Set the turn about the view axis, then apply the bounds.
    pub fn set_rotation(&mut self, rotation: f32) {
        self.rotation = rotation;
        self.clamp_to_bounds();
    }

    /// The viewport size in logical pixels.
    pub fn viewport_size(&self) -> Vec2 {
        self.viewport_size
    }

    /// Set the viewport size in logical pixels, keeping the centre and zoom,
    /// then apply the bounds. Call it whenever the viewport resizes.
    pub fn set_viewport_size(&mut self, size: Vec2) {
        self.viewport_size = size.max(Vec2::ZERO);
        self.clamp_to_bounds();
    }

    /// The area the visible rectangle is kept inside, if any.
    pub fn bounds(&self) -> Option<Rect2d> {
        self.bounds
    }

    /// Keep the visible area inside `bounds` (in plane units), or remove the
    /// limit with `None`. Along an axis where the view is wider than the
    /// bounds, the view centres on them instead. The shake offset is not
    /// limited.
    pub fn set_bounds(&mut self, bounds: Option<Rect2d>) {
        self.bounds = bounds;
        self.clamp_to_bounds();
    }

    /// The screen's right and up directions on the plane.
    fn screen_axes(&self) -> (Vec2, Vec2) {
        let (s, c) = self.rotation.sin_cos();
        (Vec2::new(c, s), Vec2::new(-s, c))
    }

    /// A screen offset in pixels (x right, y down) as a plane vector.
    fn offset_to_plane(&self, offset: Vec2) -> Vec2 {
        let (right, up) = self.screen_axes();
        (right * offset.x - up * offset.y) * self.units_per_pixel
    }

    /// A plane vector as a screen offset in pixels (x right, y down).
    fn plane_to_offset(&self, v: Vec2) -> Vec2 {
        let (right, up) = self.screen_axes();
        Vec2::new(v.dot(right), -v.dot(up)) / self.units_per_pixel
    }

    /// The plane position under a screen position.
    pub fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        self.centre + self.offset_to_plane(screen - self.viewport_size * 0.5)
    }

    /// The screen position of a plane position.
    pub fn world_to_screen(&self, world: Vec2) -> Vec2 {
        self.viewport_size * 0.5 + self.plane_to_offset(world - self.centre)
    }

    /// The plane area the viewport shows: exact at rotation 0, and the
    /// bounding rectangle of the turned view otherwise.
    pub fn visible_rect(&self) -> Rect2d {
        Rect2d {
            min: self.centre - self.visible_half_extent(),
            max: self.centre + self.visible_half_extent(),
        }
    }

    fn visible_half_extent(&self) -> Vec2 {
        let (right, up) = self.screen_axes();
        let half = self.viewport_size * 0.5 * self.units_per_pixel;
        right.abs() * half.x + up.abs() * half.y
    }

    fn clamp_zoom(&self, units_per_pixel: f32) -> f32 {
        let lo = self.min_units_per_pixel.max(f32::MIN_POSITIVE);
        let hi = self.max_units_per_pixel.max(lo);
        if units_per_pixel.is_finite() {
            units_per_pixel.clamp(lo, hi)
        } else {
            self.units_per_pixel
        }
    }

    fn clamp_to_bounds(&mut self) {
        let Some(bounds) = self.bounds else {
            return;
        };
        let half = self.visible_half_extent();
        let size = bounds.size();
        for axis in 0..2 {
            self.centre[axis] = if 2.0 * half[axis] >= size[axis] {
                bounds.centre()[axis]
            } else {
                self.centre[axis]
                    .clamp(bounds.min[axis] + half[axis], bounds.max[axis] - half[axis])
            };
        }
    }

    // -----------------------------------------------------------------------
    // Navigation
    // -----------------------------------------------------------------------

    /// Move the view as if the content were dragged by `delta_pixels` (x right,
    /// y down): the centre moves the other way by the delta times
    /// [`units_per_pixel`](Self::units_per_pixel).
    pub fn pan(&mut self, delta_pixels: Vec2) {
        if delta_pixels != Vec2::ZERO {
            self.centre -= self.offset_to_plane(delta_pixels);
            self.clamp_to_bounds();
        }
    }

    /// Zoom by `factor` about a screen position: above 1 zooms in, below 1
    /// zooms out, and the plane point under `cursor_pixels` stays under it
    /// (unless the zoom limits or the bounds move it).
    pub fn zoom_at(&mut self, cursor_pixels: Vec2, factor: f32) {
        if !(factor > 0.0) || factor == 1.0 {
            return;
        }
        let anchor = self.screen_to_world(cursor_pixels);
        self.units_per_pixel = self.clamp_zoom(self.units_per_pixel / factor);
        self.centre = anchor - self.offset_to_plane(cursor_pixels - self.viewport_size * 0.5);
        self.clamp_to_bounds();
    }

    /// Set the zoom so that one texel of art drawn at `units_per_texel` world
    /// units covers a whole number of device pixels, the nearest to the
    /// current zoom and at least one. Returns that number. The centre stays.
    pub fn snap_zoom_to_integer_scale(&mut self, units_per_texel: f32) -> u32 {
        let ppp = self.pixels_per_point.max(1e-3);
        let device_per_texel = units_per_texel * ppp / self.units_per_pixel;
        let scale = device_per_texel.round().max(1.0);
        self.set_units_per_pixel(units_per_texel * ppp / scale);
        scale as u32
    }

    // -----------------------------------------------------------------------
    // Follow and shake
    // -----------------------------------------------------------------------

    /// Follow `target`, a plane position. Call it each frame the target
    /// moves; [`update`](Self::update) moves the camera toward it once it
    /// leaves the [`dead_zone`](Self::dead_zone).
    pub fn follow(&mut self, target: Vec2) {
        self.follow_target = Some(target);
    }

    /// Stop following. The camera stays where it is.
    pub fn stop_following(&mut self) {
        self.follow_target = None;
    }

    /// The position being followed, if any.
    pub fn follow_target(&self) -> Option<Vec2> {
        self.follow_target
    }

    /// Start a shake: a random offset of up to `amplitude_pixels` that decays
    /// to nothing over `duration` seconds. It moves only the produced camera,
    /// never the centre. A weaker shake does not cut short a stronger one
    /// still running.
    pub fn shake(&mut self, amplitude_pixels: f32, duration: f32) {
        if amplitude_pixels <= 0.0 || duration <= 0.0 {
            return;
        }
        if self.shake.remaining_amplitude() > amplitude_pixels {
            return;
        }
        self.shake = Shake {
            amplitude: amplitude_pixels,
            duration,
            elapsed: 0.0,
            offset: Vec2::ZERO,
        };
        self.shake.offset = self.random_offset() * amplitude_pixels;
    }

    /// The current shake offset in pixels, x right and y down.
    pub fn shake_offset(&self) -> Vec2 {
        self.shake.offset
    }

    /// Advance the follow and the shake by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        if let Some(target) = self.follow_target {
            let offset = self.plane_to_offset(target - self.centre);
            let zone = Rect2d::new(self.dead_zone.min, self.dead_zone.max);
            let excess = offset - offset.clamp(zone.min, zone.max);
            if excess != Vec2::ZERO {
                let k = if self.follow_time_constant > 0.0 {
                    1.0 - (-dt / self.follow_time_constant).exp()
                } else {
                    1.0
                };
                self.centre += self.offset_to_plane(excess) * k;
                self.clamp_to_bounds();
            }
        }
        if self.shake.duration > 0.0 {
            self.shake.elapsed += dt;
            let amplitude = self.shake.remaining_amplitude();
            if amplitude > 0.0 {
                self.shake.offset = self.random_offset() * amplitude;
            } else {
                self.shake = Shake::default();
            }
        }
    }

    /// A point in the unit square about the origin, from a small xorshift
    /// generator so a shake is the same on every run.
    fn random_offset(&mut self) -> Vec2 {
        let mut next = || {
            let mut x = self.rng;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.rng = x;
            (x >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        Vec2::new(next(), next())
    }

    // -----------------------------------------------------------------------
    // The produced camera
    // -----------------------------------------------------------------------

    /// The orthographic camera for this view: looking along the plane's view
    /// direction from [`distance`](Self::distance) in front of it, as tall as
    /// the viewport height times [`units_per_pixel`](Self::units_per_pixel),
    /// with the shake offset added and, with
    /// [`snap_to_pixels`](Self::snap_to_pixels), the position snapped to whole
    /// device pixels.
    pub fn camera(&self) -> Camera {
        let (right, up) = self.screen_axes();
        let mut centre = self.centre + self.offset_to_plane(self.shake.offset);
        if self.snap_to_pixels {
            let ppp = self.pixels_per_point.max(1e-3);
            let per_device = self.units_per_pixel / ppp;
            let device = self.viewport_size * ppp;
            // The plane origin lands on a whole device pixel when the left
            // and top edges sit a whole number of device pixels from it.
            let r = centre.dot(right) / per_device;
            let u = centre.dot(up) / per_device;
            let r = (r - device.x * 0.5).round() + device.x * 0.5;
            let u = (u + device.y * 0.5).round() - device.y * 0.5;
            centre = (right * r + up * u) * per_device;
        }

        let plane = self.plane;
        let world_right = plane.x_axis() * right.x + plane.y_axis() * right.y;
        let world_up = plane.x_axis() * up.x + plane.y_axis() * up.y;
        let back = -plane.view_direction();
        let basis = DMat3::from_cols(world_right.as_dvec3(), world_up.as_dvec3(), back.as_dvec3());
        let orientation = DQuat::from_mat3(&basis).normalize();
        let distance = self.distance.max(1e-3);
        let eye = plane.to_world(centre) + back.as_dvec3() * f64::from(distance);

        let mut camera = Camera::new(Pose::new(eye, orientation)).with_projection(
            Projection::orthographic(self.viewport_size.y * self.units_per_pixel),
        );
        camera.set_aspect_ratio(self.viewport_size.x, self.viewport_size.y);
        camera.set_focus_distance(distance);
        camera
    }
}

/// Drag-to-pan and zoom-about-the-cursor for a [`Camera2d`].
///
/// Two ways to drive it, as with the other controllers:
///
/// - The host owns a [`ViewportInput`] (or a `ViewportInstance`) with
///   [`planar_bindings`] and hands each resolved frame to
///   [`apply`](Self::apply). Create the controller with
///   [`new_stateless`](Self::new_stateless).
/// - The controller owns the resolver: create it with [`new`](Self::new), call
///   [`begin_frame`](Self::begin_frame) with the frame's
///   [`ViewportContext`], push every translated event with
///   [`push_event`](Self::push_event), and call
///   [`apply_to_camera`](Self::apply_to_camera) once per frame. Events are
///   routed by the context's input ownership, so a drag that starts over the
///   application's UI does not pan.
///
/// ```text
/// controller.begin_frame(ViewportContext::new(ownership, viewport_size));
/// for event in events { controller.push_event(event); }
/// controller.apply_to_camera(&mut camera2d, dt);
/// frame_camera = camera2d.camera();
/// ```
///
/// Both paths also advance the camera's follow and shake, so do not call
/// [`Camera2d::update`] as well.
pub struct Camera2dController {
    input: ViewportInput,
    /// Multiplier on pan deltas. Defaults to `1.0`, which keeps the content
    /// under the pointer while dragging.
    pub pan_sensitivity: f32,
    /// Zoom per pixel of wheel or pinch input: each frame zooms by
    /// `exp(zoom * zoom_sensitivity)`. Defaults to
    /// [`DEFAULT_ZOOM_SENSITIVITY`](Self::DEFAULT_ZOOM_SENSITIVITY).
    pub zoom_sensitivity: f32,
    viewport_size: Vec2,
}

impl Default for Camera2dController {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera2dController {
    /// Default zoom sensitivity: a wheel notch zooms by about 6%.
    pub const DEFAULT_ZOOM_SENSITIVITY: f32 = 0.002;

    /// A controller with its own resolver holding [`planar_bindings`].
    pub fn new() -> Self {
        Self::from_bindings(planar_bindings())
    }

    /// A controller with its own resolver holding `bindings`.
    pub fn from_bindings(bindings: Vec<ViewportBinding>) -> Self {
        Self {
            input: ViewportInput::new(bindings),
            pan_sensitivity: 1.0,
            zoom_sensitivity: Self::DEFAULT_ZOOM_SENSITIVITY,
            viewport_size: Vec2::ZERO,
        }
    }

    /// A controller with no bindings, for a host that resolves its own frames
    /// and calls [`apply`](Self::apply).
    pub fn new_stateless() -> Self {
        Self::from_bindings(Vec::new())
    }

    /// The controller's own resolver, for touch settings or cancelling a
    /// gesture.
    pub fn input_mut(&mut self) -> &mut ViewportInput {
        &mut self.input
    }

    /// Begin a frame: clear the per-frame input and record the context (input
    /// ownership and viewport size). Call it before pushing the frame's events.
    pub fn begin_frame(&mut self, ctx: ViewportContext) {
        self.viewport_size = Vec2::from(ctx.viewport_size);
        self.input.begin_frame(ctx);
    }

    /// [`begin_frame`](Self::begin_frame) with the host's clock, which double
    /// tap and long press need.
    pub fn begin_frame_at(&mut self, ctx: ViewportContext, time_seconds: f32) {
        self.viewport_size = Vec2::from(ctx.viewport_size);
        self.input.begin_frame_at(ctx, time_seconds);
    }

    /// Push one translated event. Routing follows the ownership given to the
    /// last [`begin_frame`](Self::begin_frame).
    pub fn push_event(&mut self, event: ViewportEvent) {
        self.input.push_event(event);
    }

    /// Resolve the events pushed this frame without moving any camera.
    pub fn resolve(&self) -> ActionFrame {
        self.input.resolve()
    }

    /// Resolve this frame's events, set the camera's viewport size from the
    /// last [`begin_frame`](Self::begin_frame), apply the input and advance
    /// follow and shake by `dt`. Returns the resolved frame.
    pub fn apply_to_camera(&mut self, camera: &mut Camera2d, dt: f32) -> ActionFrame {
        let frame = self.input.resolve();
        if self.viewport_size != Vec2::ZERO && self.viewport_size != camera.viewport_size() {
            camera.set_viewport_size(self.viewport_size);
        }
        self.apply(camera, &frame, dt);
        frame
    }

    /// Apply a resolved frame: pan by `navigation.pan`, zoom by
    /// `navigation.zoom` about the cursor (the touch point during a pinch, the
    /// screen centre with neither), then advance follow and shake by `dt`.
    /// Orbit and twist are ignored. The camera's viewport size must already
    /// be current.
    pub fn apply(&mut self, camera: &mut Camera2d, frame: &ActionFrame, dt: f32) {
        let nav = &frame.navigation;
        if nav.pan != Vec2::ZERO {
            camera.pan(nav.pan * self.pan_sensitivity);
        }
        if nav.zoom != 0.0 {
            let pointer = &frame.pointer;
            let anchor = if pointer.contacts >= 2 {
                pointer.touch_position.or(pointer.cursor)
            } else {
                pointer.cursor
            }
            .unwrap_or(camera.viewport_size() * 0.5);
            camera.zoom_at(anchor, (nav.zoom * self.zoom_sensitivity).exp());
        }
        camera.update(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::binding::MouseButton;
    use crate::input::context::InputOwnership;
    use crate::input::event::{ButtonState, ScrollUnits};

    const PLANES: [Plane2d; 2] = [Plane2d::Xy, Plane2d::Xz];

    fn close(a: Vec2, b: Vec2, eps: f32) -> bool {
        (a - b).abs().max_element() <= eps
    }

    fn cam(plane: Plane2d) -> Camera2d {
        Camera2d::new(plane)
            .with_viewport_size(Vec2::new(800.0, 600.0))
            .with_units_per_pixel(0.05)
            .with_centre(Vec2::new(3.0, -2.0))
    }

    /// Where a plane point lands on screen through the produced camera, in
    /// device pixels.
    fn project(c: &Camera2d, world: Vec2) -> Vec2 {
        let camera = c.camera();
        let ndc = camera
            .view_proj_matrix()
            .project_point3(c.plane.to_world(world).as_vec3());
        let device = c.viewport_size() * c.pixels_per_point;
        Vec2::new(
            (ndc.x + 1.0) * 0.5 * device.x,
            (1.0 - ndc.y) * 0.5 * device.y,
        )
    }

    #[test]
    fn zoom_at_cursor_keeps_the_point_under_it() {
        for plane in PLANES {
            for rotation in [0.0, 0.4] {
                let mut c = cam(plane).with_rotation(rotation);
                let cursor = Vec2::new(620.0, 140.0);
                let before = c.screen_to_world(cursor);
                c.zoom_at(cursor, 1.7);
                assert!((c.units_per_pixel() - 0.05 / 1.7).abs() < 1e-7);
                assert!(close(c.screen_to_world(cursor), before, 1e-4));
                assert!(close(project(&c, before), cursor, 0.05), "{plane:?}");
                c.zoom_at(cursor, 0.3);
                assert!(close(c.screen_to_world(cursor), before, 1e-4));
            }
        }
    }

    #[test]
    fn pan_moves_by_pixels_times_units_per_pixel() {
        let mut c = cam(Plane2d::Xy);
        let p = Vec2::new(1.0, 1.5);
        let on_screen = c.world_to_screen(p);
        c.pan(Vec2::new(10.0, -4.0));
        // Dragging right moves the centre left; dragging up (y down) moves it down.
        assert!(close(c.centre(), Vec2::new(3.0 - 0.5, -2.0 - 0.2), 1e-5));
        assert!(close(
            c.world_to_screen(p),
            on_screen + Vec2::new(10.0, -4.0),
            1e-3
        ));
        // Turned a quarter, the same drag moves along the turned axes.
        let mut r = cam(Plane2d::Xz).with_rotation(std::f32::consts::FRAC_PI_2);
        r.pan(Vec2::new(10.0, 0.0));
        assert!(close(r.centre(), Vec2::new(3.0, -2.0 - 0.5), 1e-5));
    }

    #[test]
    fn follow_waits_for_the_dead_zone_then_converges() {
        let mut c = Camera2d::new(Plane2d::Xy)
            .with_viewport_size(Vec2::new(400.0, 300.0))
            .with_units_per_pixel(0.1);
        c.dead_zone = Rect2d::from_centre_size(Vec2::ZERO, Vec2::new(100.0, 100.0));
        c.follow_time_constant = 0.5;

        // 30 px right and 40 px up: inside the 50 px half-size.
        c.follow(Vec2::new(3.0, 4.0));
        for _ in 0..30 {
            c.update(1.0 / 60.0);
        }
        assert_eq!(c.centre(), Vec2::ZERO);

        // 100 px right: 50 px (5 units) outside the zone.
        c.follow(Vec2::new(10.0, 0.0));
        for _ in 0..10 {
            c.update(0.05);
        }
        let left = 10.0 - 5.0 - c.centre().x;
        let expected = 5.0 * (-1.0f32).exp();
        assert!((left - expected).abs() < 1e-4, "{left} vs {expected}");
        assert_eq!(c.centre().y, 0.0);

        for _ in 0..200 {
            c.update(0.05);
        }
        // Settles with the target on the zone's edge, not the centre.
        assert!(close(c.centre(), Vec2::new(5.0, 0.0), 1e-4));
    }

    #[test]
    fn bounds_clamp_and_centre_on_small_bounds() {
        // 200 x 100 px at 0.1 shows 20 x 10 units.
        let mut c = Camera2d::new(Plane2d::Xy)
            .with_viewport_size(Vec2::new(200.0, 100.0))
            .with_units_per_pixel(0.1)
            .with_bounds(Rect2d::new(Vec2::ZERO, Vec2::new(100.0, 50.0)));
        c.set_centre(Vec2::new(-50.0, 70.0));
        assert!(close(c.centre(), Vec2::new(10.0, 45.0), 1e-5));
        let v = c.visible_rect();
        assert!(v.min.x >= -1e-5 && v.max.y <= 50.0 + 1e-5);
        c.pan(Vec2::new(-1.0e5, 0.0));
        assert!(close(c.centre(), Vec2::new(90.0, 45.0), 1e-4));

        // Bounds narrower than the view along x only: centred on x, clamped on y.
        c.set_bounds(Some(Rect2d::new(Vec2::ZERO, Vec2::new(10.0, 50.0))));
        assert!(close(c.centre(), Vec2::new(5.0, 45.0), 1e-5));

        // Smaller than the view on both axes: centred on them.
        c.set_bounds(Some(Rect2d::new(Vec2::ZERO, Vec2::new(10.0, 5.0))));
        assert!(close(c.centre(), Vec2::new(5.0, 2.5), 1e-5));
        c.zoom_at(Vec2::new(10.0, 10.0), 0.5);
        assert!(close(c.centre(), Vec2::new(5.0, 2.5), 1e-5));
    }

    #[test]
    fn shake_decays_to_zero_and_leaves_the_centre() {
        let mut c = cam(Plane2d::Xz);
        let centre = c.centre();
        let rest = c.camera().target();
        c.shake(12.0, 0.5);
        assert!(c.shake_offset() != Vec2::ZERO);
        assert!(c.camera().target() != rest);
        let mut t = 0.0f32;
        let mut moved = false;
        while t < 0.6 {
            c.update(1.0 / 60.0);
            t += 1.0 / 60.0;
            let limit = if t < 0.5 {
                12.0 * (1.0 - t / 0.5).powi(2) + 1e-4
            } else {
                0.0
            };
            assert!(c.shake_offset().abs().max_element() <= limit);
            moved |= c.shake_offset() != Vec2::ZERO;
            assert_eq!(c.centre(), centre);
        }
        assert!(moved);
        assert_eq!(c.shake_offset(), Vec2::ZERO);
        assert_eq!(c.camera().target(), rest);
    }

    #[test]
    fn pixel_snapping_lands_on_whole_device_pixels() {
        for (size, ppp) in [
            (Vec2::new(201.0, 101.0), 1.0),
            (Vec2::new(320.0, 180.0), 2.0),
        ] {
            for plane in PLANES {
                let mut c = Camera2d::new(plane)
                    .with_viewport_size(size)
                    .with_units_per_pixel(0.013)
                    .with_centre(Vec2::new(0.3337, -1.2345));
                c.pixels_per_point = ppp;
                let frac = |p: Vec2| (p - p.round()).abs().max_element();
                assert!(
                    frac(project(&c, Vec2::ZERO)) > 0.01,
                    "unsnapped is off-grid"
                );
                c.snap_to_pixels = true;
                let per_device = 0.013 / ppp;
                for k in [0.0, 3.0, -7.0] {
                    let p = project(&c, Vec2::splat(k * per_device));
                    assert!(frac(p) < 2e-3, "{plane:?} {size} {ppp}: {p}");
                }
            }
        }
    }

    #[test]
    fn integer_zoom_snaps_to_a_whole_scale() {
        let mut c = cam(Plane2d::Xy);
        c.pixels_per_point = 2.0;
        // One texel is 1/16 unit; at 0.05 units per pixel that is 2.5 device px.
        let scale = c.snap_zoom_to_integer_scale(1.0 / 16.0);
        assert!(scale == 2 || scale == 3);
        let device_per_texel = (1.0 / 16.0) * 2.0 / c.units_per_pixel();
        assert!((device_per_texel - scale as f32).abs() < 1e-4);
        assert!(close(c.centre(), Vec2::new(3.0, -2.0), 1e-6));
    }

    #[test]
    fn screen_and_world_round_trip() {
        for plane in PLANES {
            let c = cam(plane).with_rotation(-1.1);
            for p in [Vec2::ZERO, Vec2::new(12.5, -3.25), Vec2::new(-40.0, 17.0)] {
                assert!(close(c.screen_to_world(c.world_to_screen(p)), p, 1e-4));
            }
        }
    }

    #[test]
    fn projection_extent_matches_the_visible_rect() {
        for plane in PLANES {
            let c = cam(plane);
            let camera = c.camera();
            assert!(camera.is_orthographic());
            let rect = c.visible_rect();
            assert!(close(rect.size(), Vec2::new(40.0, 30.0), 1e-4));

            // Rays through the NDC corners cross the plane at the rect corners.
            let corner = |ndc: Vec2| {
                let (origin, dir) = camera.screen_ray(ndc);
                assert!((dir - plane.view_direction()).length() < 1e-5);
                plane.from_world(origin.as_dvec3())
            };
            assert!(
                close(corner(Vec2::new(-1.0, -1.0)), rect.min, 1e-3),
                "{plane:?}"
            );
            assert!(
                close(corner(Vec2::new(1.0, 1.0)), rect.max, 1e-3),
                "{plane:?}"
            );

            // And the view-projection maps the rect corners to the NDC corners.
            let vp = camera.view_proj_matrix();
            let ndc = |p: Vec2| {
                let v = vp.project_point3(plane.to_world(p).as_vec3());
                assert!(v.z > 0.0 && v.z < 1.0);
                Vec2::new(v.x, v.y)
            };
            assert!(close(ndc(rect.min), Vec2::new(-1.0, -1.0), 1e-5));
            assert!(close(ndc(rect.max), Vec2::new(1.0, 1.0), 1e-5));
            // Screen up is plane +y on both planes.
            assert!((camera.up() - plane.y_axis()).length() < 1e-6);
            assert!((camera.right() - Vec3::X).length() < 1e-6);
        }
    }

    #[test]
    fn controller_drags_to_pan_and_wheels_about_the_cursor() {
        let mut ctrl = Camera2dController::new();
        let mut c = Camera2d::new(Plane2d::Xy).with_units_per_pixel(0.1);
        let ctx = ViewportContext::new(InputOwnership::full_window(), [800.0, 600.0]);

        ctrl.begin_frame(ctx);
        ctrl.push_event(ViewportEvent::PointerMoved {
            position: Vec2::new(100.0, 100.0),
        });
        ctrl.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
        });
        ctrl.push_event(ViewportEvent::PointerMoved {
            position: Vec2::new(130.0, 80.0),
        });
        let frame = ctrl.apply_to_camera(&mut c, 0.0);
        assert_eq!(frame.navigation.orbit, Vec2::ZERO);
        assert_eq!(c.viewport_size(), Vec2::new(800.0, 600.0));
        assert!(close(c.centre(), Vec2::new(-3.0, -2.0), 1e-4));

        ctrl.begin_frame(ctx);
        ctrl.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Released,
        });
        let under = c.screen_to_world(Vec2::new(130.0, 80.0));
        ctrl.push_event(ViewportEvent::Wheel {
            delta: Vec2::new(0.0, 3.0),
            units: ScrollUnits::Lines,
        });
        ctrl.apply_to_camera(&mut c, 0.0);
        assert!(c.units_per_pixel() < 0.1);
        assert!(close(
            c.screen_to_world(Vec2::new(130.0, 80.0)),
            under,
            1e-4
        ));
    }

    #[test]
    fn controller_ignores_a_drag_the_viewport_does_not_own() {
        let mut ctrl = Camera2dController::new();
        let mut c = Camera2d::new(Plane2d::Xy);
        let ctx = ViewportContext::new(InputOwnership::default(), [800.0, 600.0]);
        ctrl.begin_frame(ctx);
        ctrl.push_event(ViewportEvent::PointerMoved {
            position: Vec2::new(100.0, 100.0),
        });
        ctrl.push_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
        });
        ctrl.push_event(ViewportEvent::PointerMoved {
            position: Vec2::new(160.0, 100.0),
        });
        ctrl.apply_to_camera(&mut c, 0.0);
        assert_eq!(c.centre(), Vec2::ZERO);
    }
}
