use super::*;
use crate::Colour;
use crate::interaction::manipulation::gizmo::{GizmoAxis, GizmoMode};
use crate::interaction::query::snap::ConstraintOverlay;
use crate::renderer::SubSelectionRef;
use crate::resources::CameraUniform;

// ---------------------------------------------------------------------------
// 0.2.0 grouped frame API types
// ---------------------------------------------------------------------------

/// Renderer-facing camera: a view matrix and a projection matrix.
///
/// Build one with [`from_camera`](Self::from_camera) from an app-side
/// [`Camera`](crate::camera::Camera), or with
/// [`from_matrices`](Self::from_matrices) when the host computes its own
/// matrices. Read the matrices with [`view`](Self::view) and
/// [`projection`](Self::projection) and replace them with
/// [`set_view`](Self::set_view) and [`set_projection`](Self::set_projection).
/// The eye position, basis, depth range, field of view and aspect are derived
/// from the matrices whenever one is set, so they cannot fall out of step.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RenderCamera {
    view: glam::Mat4,
    projection: glam::Mat4,
    culling_projection: glam::Mat4,
    pose: CameraPose,
    lens: CameraLens,
    /// Near and far read from `culling_projection`.
    culling_range: (f32, f32),
    depth_override: Option<(f32, f32)>,
    /// World-to-view rotation built in f64 from an app-side camera's pose.
    /// `view`'s rotation is built from f32 world positions and loses
    /// precision far from the world origin; a relative camera uses this one.
    exact_rotation: Option<glam::Mat3>,
}

/// Camera placement derived from the view matrix.
#[derive(Debug, Clone, Copy)]
struct CameraPose {
    eye: glam::DVec3,
    orientation: glam::Quat,
    right: glam::Vec3,
    up: glam::Vec3,
    forward: glam::Vec3,
}

impl CameraPose {
    fn from_view(view: glam::Mat4) -> Self {
        let m = view.inverse();
        Self {
            eye: m.w_axis.truncate().as_dvec3(),
            orientation: glam::Quat::from_mat4(&m).normalize(),
            right: m.x_axis.truncate(),
            up: m.y_axis.truncate(),
            forward: -m.z_axis.truncate(),
        }
    }
}

/// Projection parameters derived from the projection matrix.
#[derive(Debug, Clone, Copy)]
struct CameraLens {
    orthographic: bool,
    /// Depth runs from 1 at the near plane to 0 at the far plane.
    reversed: bool,
    near: f32,
    /// `f32::INFINITY` for a perspective projection with no far plane.
    far: f32,
    aspect: f32,
    vertical_fov: Option<f32>,
    /// Angle for the renderer's size and frustum maths. See `fov_y`.
    fov_y: f32,
}

impl CameraLens {
    fn from_projection(p: glam::Mat4) -> Self {
        let orthographic = p.w_axis.w != 0.0;
        let (a, b) = (p.z_axis.z, p.w_axis.z);
        // Standard depth has a negative z scale, reversed depth a positive
        // one (zero for an infinite reversed perspective).
        let reversed = a >= 0.0;
        let (near, far) = match (orthographic, reversed) {
            // a = 1 / (near - far), b = near / (near - far)
            (true, false) => {
                let near = b / a;
                (near, near - 1.0 / a)
            }
            // a = 1 / (far - near), b = far / (far - near)
            (true, true) => ((b - 1.0) / a, b / a),
            // a = far / (near - far), b = near * far / (near - far)
            (false, false) => (b / a, b / (a + 1.0)),
            // a = near / (far - near), b = near * far / (far - near); a = 0
            // and b = near with no far plane.
            (false, true) => {
                let far = if a == 0.0 { f32::INFINITY } else { b / a };
                (b / (a + 1.0), far)
            }
        };
        let vertical_fov = (!orthographic).then(|| 2.0 * (1.0 / p.y_axis.y).atan());
        Self {
            orthographic,
            reversed,
            near,
            far,
            aspect: p.y_axis.y / p.x_axis.x,
            vertical_fov,
            fov_y: vertical_fov.unwrap_or(std::f32::consts::FRAC_PI_4),
        }
    }
}

/// Far distance assumed for a projection with no far plane when the host
/// gives no range: this many times the near distance.
const INFINITE_FAR_RATIO: f32 = 10_000.0;

/// `projection` with its depth row replaced by standard 0..1 depth between
/// `near` and `far`, keeping the x, y and w rows.
fn standard_depth_projection(projection: glam::Mat4, near: f32, far: f32) -> glam::Mat4 {
    let orthographic = projection.w_axis.w != 0.0;
    let (a, b) = if orthographic {
        (1.0 / (near - far), near / (near - far))
    } else {
        (far / (near - far), near * far / (near - far))
    };
    let mut p = projection;
    p.x_axis.z = 0.0;
    p.y_axis.z = 0.0;
    p.z_axis.z = a;
    p.w_axis.z = b;
    p
}

impl RenderCamera {
    /// Build a camera from a world-to-view matrix and a view-to-clip
    /// projection matrix.
    ///
    /// The projection must be right-handed with reversed 0..1 depth: 1 at the
    /// near plane and 0 at the far plane, which the renderer's depth test and
    /// clear assume. glam builds one with `perspective_infinite_reverse_rh`,
    /// with `perspective_rh` or `orthographic_rh` given the far distance
    /// before the near one, or from a standard projection with
    /// [`with_projection_reversed`](Self::with_projection_reversed). Its near
    /// and far planes are read back for [`depth_range`](Self::depth_range);
    /// for a projection with no far plane or an oblique near plane, follow
    /// with [`with_depth_range`](Self::with_depth_range).
    pub fn from_matrices(view: glam::Mat4, projection: glam::Mat4) -> Self {
        let mut cam = Self {
            view,
            projection,
            culling_projection: projection,
            pose: CameraPose::from_view(view),
            lens: CameraLens::from_projection(projection),
            culling_range: (0.0, 0.0),
            depth_override: None,
            exact_rotation: None,
        };
        cam.derive_culling();
        cam
    }

    /// Rebuild the culling projection from the projection and the depth
    /// range. A finite standard-depth projection with no range set is its
    /// own culling projection.
    fn derive_culling(&mut self) {
        let lens = self.lens;
        let finite = lens.far.is_finite();
        self.culling_projection = match self.depth_override {
            None if !lens.reversed && finite => self.projection,
            Some((near, far)) => standard_depth_projection(self.projection, near, far),
            None => {
                let far = if finite {
                    lens.far
                } else {
                    lens.near * INFINITE_FAR_RATIO
                };
                standard_depth_projection(self.projection, lens.near, far)
            }
        };
        let culling = CameraLens::from_projection(self.culling_projection);
        self.culling_range = (culling.near, culling.far);
    }

    /// Build a `RenderCamera` from an app-side [`Camera`](crate::camera::Camera):
    /// [`from_matrices`](Self::from_matrices) over the camera's view and
    /// projection matrices, with the camera's culling range.
    ///
    /// The camera's own eye, view direction, f64 rotation and framing angle
    /// replace the values read back from the matrices, which differ in the
    /// last bits (an orthographic matrix carries no angle at all), and the
    /// culling projection is the camera's own. Resolve the camera to a
    /// `RenderCamera` once per frame and pass it through `CameraFrame`.
    pub fn from_camera(cam: &crate::camera::Camera) -> Self {
        let rc = Self::from_matrices(cam.view_matrix(), cam.proj_matrix());
        let mut rc = match cam.projection {
            crate::camera::Projection::Matrix {
                culling_range: (near, far),
                ..
            } => rc.with_depth_range(near, far),
            _ => {
                let mut rc = rc;
                rc.culling_projection = cam.culling_proj_matrix();
                let culling = CameraLens::from_projection(rc.culling_projection);
                rc.culling_range = (culling.near, culling.far);
                rc.lens.fov_y = cam.fov_y();
                rc
            }
        };
        let eye = cam.eye();
        let target = cam.target();
        rc.pose.eye = eye;
        if target == eye {
            // No focus distance to look along: take the axes from the pose.
            rc.pose.forward = cam.forward();
            let rotation = glam::DMat3::from_quat(cam.pose.orientation).transpose();
            rc.exact_rotation = Some(rotation.as_mat3());
            return rc;
        }
        rc.pose.forward = (target.as_vec3() - eye.as_vec3()).normalize_or_zero();
        let up = (cam.orientation() * glam::Vec3::Y).as_dvec3();
        let exact = glam::DMat4::look_at_rh(eye, target, up);
        rc.exact_rotation = Some(glam::DMat3::from_mat4(exact).as_mat3());
        rc
    }

    /// Report `near` and `far` from [`depth_range`](Self::depth_range) instead
    /// of the values read from the projection.
    ///
    /// For a projection the read-back cannot handle, such as an oblique near
    /// plane, or to give a projection with no far plane a far distance. The
    /// projection itself is not changed; the range sets the
    /// [`culling_projection`](Self::culling_projection) and feeds shadow
    /// cascade fitting, clustered light slicing and depth picking. A later
    /// [`set_projection`](Self::set_projection) clears it.
    pub fn with_depth_range(mut self, near: f32, far: f32) -> Self {
        self.depth_override = Some((near, far));
        self.derive_culling();
        self
    }

    /// World-to-view transform matrix.
    pub fn view(&self) -> glam::Mat4 {
        self.view
    }

    /// View-to-clip (projection) matrix.
    pub fn projection(&self) -> glam::Mat4 {
        self.projection
    }

    /// Replace the world-to-view matrix.
    pub fn set_view(&mut self, view: glam::Mat4) {
        self.view = view;
        self.pose = CameraPose::from_view(view);
        self.exact_rotation = None;
    }

    /// Replace the projection matrix, which must use reversed depth as
    /// [`from_matrices`](Self::from_matrices) describes. Clears any range set
    /// with [`with_depth_range`](Self::with_depth_range).
    pub fn set_projection(&mut self, projection: glam::Mat4) {
        self.projection = projection;
        self.lens = CameraLens::from_projection(projection);
        self.depth_override = None;
        self.derive_culling();
    }

    /// Replace the projection with `standard`, a right-handed projection with
    /// standard 0..1 depth (near at 0), converted to the reversed depth the
    /// renderer draws with. For a host that builds its projection with glam's
    /// `perspective_rh` or `orthographic_rh`, or takes one from another
    /// engine. Clears any range set with
    /// [`with_depth_range`](Self::with_depth_range).
    pub fn with_projection_reversed(mut self, standard: glam::Mat4) -> Self {
        // z' = w - z: depth d becomes 1 - d.
        let mut p = standard;
        p.x_axis.z = standard.x_axis.w - standard.x_axis.z;
        p.y_axis.z = standard.y_axis.w - standard.y_axis.z;
        p.z_axis.z = standard.z_axis.w - standard.z_axis.z;
        p.w_axis.z = standard.w_axis.w - standard.w_axis.z;
        self.set_projection(p);
        self
    }

    /// A finite projection with standard 0..1 depth over the same view as
    /// [`projection`](Self::projection), between the near and far of
    /// [`depth_range`](Self::depth_range).
    ///
    /// For CPU work that needs both a near and a far plane: frustum planes,
    /// culling and cascade fitting. Do not draw with it.
    pub fn culling_projection(&self) -> glam::Mat4 {
        self.culling_projection
    }

    /// [`culling_projection`](Self::culling_projection) times the view
    /// matrix. Build CPU frustums from this rather than
    /// [`view_proj`](Self::view_proj).
    pub fn culling_view_proj(&self) -> glam::Mat4 {
        self.culling_projection * self.view
    }

    /// The frustum of [`culling_view_proj`](Self::culling_view_proj) in world
    /// space, extracted relative to the eye so it holds far from the world
    /// origin; see
    /// [`Frustum::from_eye_relative`](crate::camera::frustum::Frustum::from_eye_relative).
    pub fn culling_frustum(&self) -> crate::camera::frustum::Frustum {
        let eye = self.pose.eye;
        crate::camera::frustum::Frustum::from_eye_relative(
            &self.relative_to(eye).culling_view_proj(),
            eye,
        )
    }

    /// The world-space ray through a point on screen, as `(origin, direction)`
    /// with a unit direction.
    ///
    /// `ndc` is the point in normalised device coordinates: -1..1 left to
    /// right and bottom to top. A perspective ray starts at the eye; an
    /// orthographic ray starts on the near plane and runs along the view
    /// direction. Works for any projection, including one with no far plane.
    pub fn screen_ray(&self, ndc: glam::Vec2) -> (glam::Vec3, glam::Vec3) {
        let near_ndc = if self.lens.reversed { 1.0 } else { 0.0 };
        let on_near = self
            .projection
            .inverse()
            .project_point3(glam::Vec3::new(ndc.x, ndc.y, near_ndc));
        let inv_view = self.view.inverse();
        if self.is_orthographic() {
            let dir = (-inv_view.z_axis.truncate()).normalize();
            (inv_view.transform_point3(on_near), dir)
        } else {
            let dir = inv_view.transform_vector3(on_near).normalize();
            (self.pose.eye.as_vec3(), dir)
        }
    }

    /// This camera for points relative to `origin`: the view translation is
    /// rebuilt in f64 from the eye and the eye moved by `-origin`; the
    /// orientation and projection are unchanged. A zero origin returns an
    /// identical copy.
    pub(crate) fn relative_to(&self, origin: glam::DVec3) -> RenderCamera {
        let mut cam = self.clone();
        if origin != glam::DVec3::ZERO {
            let view = match self.exact_rotation {
                Some(r) => {
                    cam.pose.forward = -r.row(2);
                    glam::Mat4::from_mat3(r)
                }
                None => self.view,
            };
            cam.view = crate::renderer::origin::rel_view(view, self.pose.eye, origin);
            cam.pose.eye = self.pose.eye - origin;
        }
        cam
    }

    /// Build the GPU-facing camera uniform from this camera's state.
    pub(crate) fn camera_uniform(&self) -> CameraUniform {
        let vp = self.view_proj();
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            eye_pos: self.pose.eye.as_vec3().to_array(),
            // LDR default; the per-viewport upload raises this on the HDR path.
            lit_clamp: 1.0,
            forward: self.pose.forward.to_array(),
            // No exposure; the per-viewport upload writes the frame's.
            exposure: 1.0,
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            view: self.view.to_cols_array_2d(),
            proj: self.projection.to_cols_array_2d(),
            inv_proj: self.projection.inverse().to_cols_array_2d(),
            // Filled per frame by `CameraFrame::camera_uniform`.
            viewport: [0.0; 4],
            clip: {
                let (near, far) = self.depth_range();
                crate::plugin_api::depth::clip_lane(self.projection, near, far)
            },
            time: [0.0; 4],
            // Filled per frame by `CameraFrame::camera_uniform`.
            world_origin: [0.0; 3],
            origin_epoch: 0,
        }
    }

    /// Combined view-projection matrix (projection * view).
    pub fn view_proj(&self) -> glam::Mat4 {
        self.projection * self.view
    }

    /// Camera eye position in world space, in f64: the translation of
    /// `view.inverse()`, or the camera's own eye for a camera built with
    /// [`from_camera`](Self::from_camera).
    pub fn eye_position(&self) -> glam::DVec3 {
        self.pose.eye
    }

    /// Camera orientation: the rotation of `view.inverse()` as a quaternion.
    ///
    /// The camera looks down its local `-Z` with `+Y` up.
    pub fn orientation(&self) -> glam::Quat {
        self.pose.orientation
    }

    /// Camera forward direction in world space (the local `-Z` axis).
    pub fn forward(&self) -> glam::Vec3 {
        self.pose.forward
    }

    /// Camera right direction in world space (the local `+X` axis).
    pub fn right(&self) -> glam::Vec3 {
        self.pose.right
    }

    /// Camera up direction in world space (the local `+Y` axis).
    pub fn up(&self) -> glam::Vec3 {
        self.pose.up
    }

    /// Whether the projection is orthographic (`w_axis.w == 1`) rather than
    /// perspective (`w_axis.w == 0`).
    pub fn is_orthographic(&self) -> bool {
        self.lens.orthographic
    }

    /// Near and far clip distances: the range set with
    /// [`with_depth_range`](Self::with_depth_range), or else the planes read
    /// from the projection.
    ///
    /// The read-back takes a right-handed projection with standard or
    /// reversed 0..1 depth. For a projection with no far plane, far is
    /// 10,000 times near unless a range is set. A camera built with
    /// [`from_camera`](Self::from_camera) reports the camera's own near and
    /// far. A perspective far plane read from a matrix keeps only about
    /// `f32::EPSILON * far / near` of relative precision.
    pub fn depth_range(&self) -> (f32, f32) {
        if let Some(range) = self.depth_override {
            return range;
        }
        let (near, far) = self.culling_range;
        debug_assert!(
            self.lens.orthographic || (near > 0.0 && near < far),
            "perspective depth range is not 0 < near < far: ({near}, {far}); \
             set it with RenderCamera::with_depth_range"
        );
        (near, far)
    }

    /// Vertical field of view in radians, or `None` for an orthographic
    /// projection.
    pub fn vertical_fov(&self) -> Option<f32> {
        self.lens.vertical_fov
    }

    /// World-space height the viewport spans at view depth `depth`.
    ///
    /// For an orthographic projection the height is the same at every depth.
    pub fn view_height_at(&self, depth: f32) -> f32 {
        let h = self.projection.y_axis.y;
        if self.is_orthographic() {
            2.0 / h
        } else {
            2.0 * depth / h
        }
    }

    /// Aspect ratio (width / height) read from the projection.
    pub fn aspect(&self) -> f32 {
        self.lens.aspect
    }

    /// Vertical field of view for the renderer's size and frustum maths
    /// (grid spacing, LOD screen size, cascade fitting), for both projections.
    ///
    /// [`from_camera`](Self::from_camera) carries the camera's own `fov_y`,
    /// which the angle read back from a perspective matrix differs from in
    /// the last bits (enough to move the shadow cascade fit) and which an
    /// orthographic matrix does not carry at all. A camera built from matrices
    /// uses the angle read from a perspective projection, and `PI / 4` for an
    /// orthographic one.
    pub(crate) fn fov_y(&self) -> f32 {
        self.lens.fov_y
    }

    /// The camera the foreground pass renders with: this camera's view
    /// transform, with the projection replaced by the pass override when one
    /// is set. With no override (or no pass config) the scene camera is used
    /// unchanged.
    ///
    /// The override is always a perspective projection, including over an
    /// orthographic scene camera.
    pub(crate) fn foreground_camera(&self, pass: Option<&ForegroundPass>) -> RenderCamera {
        let mut cam = self.clone();
        if let Some(p) = pass.and_then(|p| p.projection.as_ref()) {
            let near = p.near.max(1e-4);
            let far = p.far.unwrap_or(self.depth_range().1).max(near * 2.0);
            // Reversed depth: far before near.
            cam.set_projection(glam::Mat4::perspective_rh(
                p.fov_y,
                self.aspect(),
                far,
                near,
            ));
            cam.lens.fov_y = p.fov_y;
        }
        cam
    }
}

impl Default for RenderCamera {
    /// A perspective camera at `(0, 0, 5)` looking down `-Z` with `+Y` up:
    /// 45 degree vertical field of view, aspect 1.333, depth range 0.1..1000.
    fn default() -> Self {
        Self::from_matrices(
            glam::Mat4::look_at_rh(
                glam::Vec3::new(0.0, 0.0, 5.0),
                glam::Vec3::ZERO,
                glam::Vec3::Y,
            ),
            glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.333, 1000.0, 0.1),
        )
    }
}

/// Camera submission state for one frame.
///
/// Groups the canonical render camera with viewport sizing and multi-viewport
/// slot index. This is the single owner of all camera-derived state submitted
/// to the renderer each frame.
#[non_exhaustive]
pub struct CameraFrame {
    /// Canonical renderer-facing camera state.
    pub render_camera: RenderCamera,
    /// Viewport size in logical pixels (egui points) (width, height). Default: [800.0, 600.0].
    pub viewport_size: [f32; 2],
    /// Physical pixels per logical pixel. Set from the window or egui context
    /// (e.g. `ui.ctx().pixels_per_point()`). Default: 1.0.
    ///
    /// Screen-space math (projection, picking, widget layout) and overlay
    /// positions/sizes are all in logical pixels and use `viewport_size`
    /// directly. `pixels_per_point` sizes the physical render target and
    /// rasterises overlay text at display resolution, so overlay labels stay
    /// crisp on HiDPI displays. At 1.0 the physical and logical spaces coincide.
    pub pixels_per_point: f32,
    /// Multi-viewport slot index. Default: 0 (single-viewport mode).
    pub viewport_index: usize,
    /// Layers this camera draws, as a 32-bit mask. Default `!0` (draw every
    /// layer). An item is culled on this camera when its
    /// `ItemSettings::visibility_mask` shares no bit with this mask
    /// (`visibility_mask & cull_mask == 0`). This is a CPU-side cull applied at
    /// scene collect, before batching, so it costs nothing at the default,
    /// needs no GPU carrier, and an item it drops never reaches the GPU cull;
    /// it is the per-viewport layer filter for quad-view and editor layouts.
    pub cull_mask: u32,
}

impl Default for CameraFrame {
    fn default() -> Self {
        Self {
            render_camera: RenderCamera::default(),
            viewport_size: [800.0, 600.0],
            pixels_per_point: 1.0,
            viewport_index: 0,
            cull_mask: !0,
        }
    }
}

impl CameraFrame {
    /// The GPU camera uniform for this frame rendered at `time`: the render
    /// camera's matrices relative to `origin`, plus the physical viewport size
    /// and the frame time.
    pub(crate) fn camera_uniform(
        &self,
        time: &FrameTime,
        origin: &crate::renderer::origin::OriginState,
    ) -> CameraUniform {
        let mut u = origin.camera(&self.render_camera).camera_uniform();
        u.world_origin = origin.origin().as_vec3().to_array();
        u.origin_epoch = origin.epoch();
        let ppp = self.pixels_per_point;
        let w = (self.viewport_size[0] * ppp).round().max(1.0);
        let h = (self.viewport_size[1] * ppp).round().max(1.0);
        u.viewport = [w, h, 1.0 / w, 1.0 / h];
        u.time = crate::renderer::frame_clock::pack_time(time);
        u
    }

    /// Build a camera frame from a render camera and viewport size.
    pub fn new(render_camera: RenderCamera, viewport_size: [f32; 2]) -> Self {
        Self {
            render_camera,
            viewport_size,
            pixels_per_point: 1.0,
            viewport_index: 0,
            cull_mask: !0,
        }
    }

    /// Restrict this camera to the layers whose bits are set in `mask`. Items
    /// whose `ItemSettings::visibility_mask` shares no bit with `mask` are
    /// culled CPU-side for this camera. Default is `!0` (every layer).
    pub fn with_cull_mask(mut self, mask: u32) -> Self {
        self.cull_mask = mask;
        self
    }

    /// Build a camera frame from an app-side camera and viewport size.
    pub fn from_camera(cam: &crate::camera::Camera, viewport_size: [f32; 2]) -> Self {
        Self::new(RenderCamera::from_camera(cam), viewport_size)
    }

    /// Set the physical pixels per logical pixel for this camera frame.
    ///
    /// Pass `ui.ctx().pixels_per_point()` from egui, or the equivalent scale
    /// factor from your window system. Sizes the physical render target and
    /// rasterises overlay text at display resolution; all screen-space math and
    /// overlay authoring stay in logical `viewport_size` units.
    pub fn with_pixels_per_point(mut self, pixels_per_point: f32) -> Self {
        self.pixels_per_point = pixels_per_point.max(0.001);
        self
    }

    /// Set the multi-viewport slot index for this camera frame.
    pub fn with_viewport_index(mut self, viewport_index: usize) -> Self {
        self.viewport_index = viewport_index;
        self
    }

    /// Set the multi-viewport slot from a ViewportId returned by ViewportRenderer::create_viewport.
    /// Prefer this over with_viewport_index when you have a ViewportId.
    pub fn with_viewport_id(mut self, id: crate::renderer::ViewportId) -> Self {
        self.viewport_index = id.0;
        self
    }
}

/// Surface submission type for world-space geometry.
///
/// For 0.2.0, only `Flat` submission is supported. This enum leaves room for
/// future large-scene or chunked submission without changing the `SceneFrame`
/// public type.
#[non_exhaustive]
pub enum SurfaceSubmission {
    /// A flat, reference-counted list of scene render items.
    ///
    /// Holding an `Arc<[SceneRenderItem]>` instead of a `Vec` means the per-frame
    /// submission cost is a single atomic refcount increment rather than a full deep
    /// copy of all items. Use [`SceneFrame::from_surface_items`] or
    /// [`SceneFrame::from_scene`] to construct this variant.
    Flat(std::sync::Arc<[SceneRenderItem]>),
}

impl Default for SurfaceSubmission {
    fn default() -> Self {
        SurfaceSubmission::Flat(std::sync::Arc::from([]))
    }
}

/// World-space scene content for one frame.
///
/// Groups all renderable world-space content submitted to the renderer.
/// Surfaces are submitted through [`SurfaceSubmission`]; scientific
/// visualization primitives sit alongside them.
#[non_exhaustive]
pub struct SceneFrame {
    /// Scene version counter from `Scene::version()`. Default: 0 (triggers rebuild on first frame).
    ///
    /// The renderer uses this to skip batch rebuild and GPU upload when scene content
    /// has not changed since the previous frame.
    pub generation: u64,
    /// Surface geometry submission (opaque and transparent meshes).
    pub surfaces: SurfaceSubmission,
    /// Unstructured volume meshes submitted this frame.
    ///
    /// Each [`VolumeMeshItem`] renders either as a boundary surface (default,
    /// through the standard mesh pipeline) or as a volumetric draw through
    /// projected tetrahedra (when [`VolumeMeshItem::transparency`] is set and
    /// the item was uploaded via
    /// [`upload_volume_mesh_with_transparency`](crate::resources::DeviceResources::upload_volume_mesh_with_transparency)).
    /// Cell-level picking, selection outlines, and wireframe overlays are
    /// driven from this collection regardless of mode.
    pub volume_meshes: Vec<VolumeMeshItem>,
    /// Mesh-instance batches to render this frame (mesh-based particles).
    pub mesh_instances: Vec<MeshInstanceItem>,
    /// Scene-graph light sources to union with `EffectsFrame::lighting.lights`.
    ///
    /// Populate via [`crate::scene::scene::Scene::collect_lights`]. The renderer
    /// chains these with the frame-data lights before building the GPU uniform;
    /// consumers that use only `EffectsFrame::lighting` leave this empty.
    pub lights: Vec<LightSource>,
    /// Items drawn by the foreground pass, on top of the finished scene.
    ///
    /// Foreground items render after the world, against a freshly cleared
    /// depth buffer, so they are neither occluded by scene geometry nor clip
    /// into it. They skip instancing, shadow casting, frustum culling,
    /// picking, and selection outlines. Skinning, deformers, and materials
    /// work as for normal surface items. Configure the pass (optional
    /// override projection) via `EffectsFrame::foreground`.
    ///
    /// Empty by default; an empty list adds no pass and no allocations.
    pub foreground_items: Vec<SceneRenderItem>,
    /// Lightmap placements the frame's items name through
    /// [`SceneRenderItem::lightmap_placement`]. Kept here rather than on each
    /// item so an item stays small; empty by default.
    pub lightmap_placements: Vec<crate::resources::LightmapPlacement>,
    /// Plugin item collections, keyed by
    /// [`ItemTypePlugin::type_name`](crate::plugin_api::ItemTypePlugin::type_name).
    ///
    /// Populated via [`Self::submit_plugin_items`]. Empty by default; the
    /// renderer iterates this map during `prepare` / `paint` and dispatches
    /// to the matching registered plugin. Entries whose `type_name` is not
    /// registered on the renderer are silently ignored.
    pub plugin_items: std::collections::HashMap<
        &'static str,
        Vec<Box<dyn crate::plugin_api::PluginItemCollection>>,
    >,
    /// Replaces [`RendererConfig::world_origin`](crate::RendererConfig::world_origin)
    /// for this frame. Default `None`.
    pub world_origin: Option<crate::renderer::WorldOrigin>,
}

impl Default for SceneFrame {
    fn default() -> Self {
        Self {
            generation: 0,
            surfaces: SurfaceSubmission::default(),
            volume_meshes: Vec::new(),
            mesh_instances: Vec::new(),
            lights: Vec::new(),
            foreground_items: Vec::new(),
            lightmap_placements: Vec::new(),
            plugin_items: std::collections::HashMap::new(),
            world_origin: None,
        }
    }
}

impl SceneFrame {
    /// Build a scene frame from a surface submission.
    pub fn new(surfaces: SurfaceSubmission) -> Self {
        Self {
            surfaces,
            ..Self::default()
        }
    }

    /// Build a scene frame from a flat list of surface render items.
    ///
    /// The `Vec` is converted to an `Arc<[SceneRenderItem]>` so callers that
    /// submit the same list repeatedly can cheaply clone the `Arc` instead of
    /// cloning the underlying data.
    ///
    /// The frame's `generation` defaults to `0`. If items change every frame
    /// (e.g. from physics writeback or snapshot interpolation), chain
    /// [`with_generation`](Self::with_generation) to supply a value that
    /// changes with the content -- otherwise the renderer's instance-buffer
    /// cache will treat every frame as identical after the first upload:
    ///
    /// ```rust,ignore
    /// // Static scene -- generation 0 is fine, items never change.
    /// SceneFrame::from_surface_items(items)
    ///
    /// // Dynamic scene -- pass a value that increments when content changes.
    /// // scene.version() is the right source when writeback drives the items.
    /// SceneFrame::from_surface_items(items).with_generation(scene.version())
    /// ```
    pub fn from_surface_items(items: Vec<SceneRenderItem>) -> Self {
        Self::new(SurfaceSubmission::Flat(items.into()))
    }

    /// Override the generation stamp on a scene frame.
    ///
    /// The renderer compares `generation` against the previous frame to decide
    /// whether to rebuild and re-upload the instance buffer. A generation of
    /// `0` (the default for [`from_surface_items`](Self::from_surface_items))
    /// causes the cache to hit on every frame after the first, freezing
    /// rendered objects even if item transforms have changed.
    ///
    /// Pass any value that changes when content changes. `scene.version()` is
    /// the natural choice when items are derived from scene writeback.
    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    /// Draw this frame relative to `origin` instead of the renderer's
    /// [`RendererConfig::world_origin`](crate::RendererConfig::world_origin).
    pub fn with_world_origin(mut self, origin: crate::renderer::WorldOrigin) -> Self {
        self.world_origin = Some(origin);
        self
    }

    /// Set the lightmap placements the items index (see
    /// [`SceneRenderItem::lightmap_placement`]).
    pub fn with_lightmap_placements(
        mut self,
        placements: Vec<crate::resources::LightmapPlacement>,
    ) -> Self {
        self.lightmap_placements = placements;
        self
    }

    /// Submit a plugin item collection under `type_name`.
    ///
    /// The renderer dispatches per-frame work and draw calls to the
    /// matching [`ItemTypePlugin`](crate::plugin_api::ItemTypePlugin)
    /// registered with the same `type_name`. Submitting under a name
    /// that has no registered plugin is a no-op.
    ///
    /// Calling this twice with the same name replaces the previous
    /// submission.
    pub fn submit_plugin_items<C: crate::plugin_api::PluginItemCollection>(
        &mut self,
        type_name: &'static str,
        items: C,
    ) {
        self.plugin_items
            .entry(type_name)
            .or_default()
            .push(Box::new(items));
    }

    /// The submitted items of type `T`, creating an empty collection if this
    /// frame has none yet.
    ///
    /// This is the typed submission path: `frame.scene.items_mut::<crate::PointCloudItem>().push(item)`
    /// reaches the same collection the plugin reads back with
    /// [`ItemFrameContext::items_of`](crate::plugin_api::ItemFrameContext::items_of).
    /// An item type with several forms keeps one collection per form, all
    /// under [`PluginItem::TYPE_NAME`](crate::plugin_api::PluginItem::TYPE_NAME).
    pub fn items_mut<T: crate::plugin_api::PluginItem>(&mut self) -> &mut Vec<T> {
        let slot = self.plugin_items.entry(T::TYPE_NAME).or_default();
        if let Some(index) = slot
            .iter()
            .position(|c| c.as_any().downcast_ref::<Vec<T>>().is_some())
        {
            return slot[index]
                .as_any_mut()
                .downcast_mut::<Vec<T>>()
                .expect("position() just matched this collection's concrete type");
        }
        slot.push(Box::new(Vec::<T>::new()));
        slot.last_mut()
            .expect("just pushed")
            .as_any_mut()
            .downcast_mut::<Vec<T>>()
            .expect("just pushed a Vec<T>")
    }

    /// The submitted items of type `T`, or an empty slice if this frame
    /// carries none. Read-only counterpart to [`items_mut`](Self::items_mut).
    pub fn items_of<T: crate::plugin_api::PluginItem>(&self) -> &[T] {
        self.plugin_items
            .get(T::TYPE_NAME)
            .and_then(|slot| {
                slot.iter()
                    .find_map(|c| c.as_any().downcast_ref::<Vec<T>>())
            })
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Replace the submitted items of type `T` with `items`.
    pub fn submit<T: crate::plugin_api::PluginItem>(&mut self, items: Vec<T>) {
        *self.items_mut::<T>() = items;
    }

    /// Build a scene frame from an already-allocated shared slice.
    ///
    /// Use this variant when you cache render items across frames:
    ///
    /// ```rust,ignore
    /// // First frame / on change: rebuild the Arc once.
    /// self.items_arc = Arc::from(scene.collect_render_items(&sel));
    ///
    /// // Every frame: zero-cost clone.
    /// SceneFrame::from_shared_items(Arc::clone(&self.items_arc), scene.version())
    /// ```
    pub fn from_shared_items(items: std::sync::Arc<[SceneRenderItem]>, generation: u64) -> Self {
        Self {
            generation,
            surfaces: SurfaceSubmission::Flat(items),
            ..Self::default()
        }
    }

    /// Build a scene frame by collecting render items from a [`Scene`](crate::scene::scene::Scene).
    ///
    /// Calls [`Scene::collect_render_items`](crate::scene::scene::Scene::collect_render_items)
    /// and stamps `generation` with the current scene version so the renderer can
    /// skip batch rebuilds on unchanged frames.
    ///
    /// This is the preferred constructor for the common single-viewport path.
    /// Use [`SceneFrame::from_surface_items`] when you need to assemble render items manually.
    pub fn from_scene(
        scene: &mut crate::scene::scene::Scene,
        selection: &crate::interaction::select::selection::Selection,
    ) -> Self {
        let items = scene.collect_render_items(selection);
        let lights = scene.collect_lights();
        // The influence-volume outlines for a selected light need no mesh, so
        // they come through here. The indicator bodies do need one, so they
        // are opt-in: see `scene::build_light_indicators` and
        // `LightIndicators::to_mesh_instances`.
        let indicators = crate::scene::build_light_indicators(scene, selection);
        let mut frame = Self {
            generation: scene.version(),
            surfaces: SurfaceSubmission::Flat(items.into()),
            lights,
            ..Self::default()
        };
        *frame.items_mut::<crate::PolylineItem>() = indicators.outlines;
        frame
    }
}

/// Viewport presentation settings for one frame.
///
/// Groups background, grid, and axes indicator : the viewport chrome that is
/// independent of world-space content.
#[non_exhaustive]
pub struct ViewportFrame {
    /// The background the scene is composited over. `None` = renderer default.
    ///
    /// Treated as premultiplied: the output is `scene over background`, so the
    /// alpha decides what the background *is* rather than only tinting it.
    ///
    /// - Alpha 1 (including [`Colour::BLACK`](crate::Colour::BLACK) and any
    ///   `Colour::srgb_rgb` value) is an opaque background, which is what a
    ///   viewport drawn into a window wants.
    /// - [`Colour::TRANSPARENT`](crate::Colour::TRANSPARENT) is no background at
    ///   all. The render carries only what was drawn, with alpha as coverage, so
    ///   it can be composited over something this renderer knows nothing about:
    ///   a UI plate, a document, another render.
    /// - In between is a translucent plate the scene sits on.
    ///
    /// The output is **premultiplied**, on both the HDR and `Direct` paths, so
    /// composite it with `One` / `OneMinusSrcAlpha` rather than
    /// `SrcAlpha` / `OneMinusSrcAlpha`. Bloom and other glow deliberately carry
    /// no coverage of their own, which under that blend reads as the additive
    /// glow it is instead of occluding whatever is behind the viewport.
    pub background_colour: Option<crate::Colour>,
    /// Whether to render the scene in wireframe mode. Default: false.
    pub wireframe_mode: bool,
    /// Whether to render the ground-plane grid. Default: false.
    pub show_grid: bool,
    /// Grid cell size in world units. Zero = camera-distance-based adaptive spacing.
    pub grid_cell_size: f32,
    /// Half-extent of the grid in world units. Zero = 1000 (effectively infinite).
    pub grid_half_extent: f32,
    /// World-space Z coordinate of the grid plane (3D mode only, Z-up). Default: 0.0.
    pub grid_z: f32,
    /// RGB colour for the grid lines. None = renderer default (mid-grey).
    pub grid_colour: Option<crate::Colour>,
    /// Whether to draw the axes orientation indicator overlay. Default: true.
    pub show_axes_indicator: bool,
    /// What this viewport draws behind the scene. Default: the lighting
    /// environment, or `background_colour` when there is none. Drawn only on
    /// the HDR pipeline.
    pub environment_background: EnvironmentBackground,
}

impl Default for ViewportFrame {
    fn default() -> Self {
        Self {
            background_colour: None,
            wireframe_mode: false,
            show_grid: false,
            grid_cell_size: 0.0,
            grid_half_extent: 0.0,
            grid_z: 0.0,
            grid_colour: None,
            show_axes_indicator: true,
            environment_background: EnvironmentBackground::default(),
        }
    }
}

/// Interaction and selection visualization state for one frame.
///
/// Groups the gizmo, selection overlays, constraint guides, outline, and
/// x-ray state : everything that communicates selection and interaction
/// feedback to the user.
#[non_exhaustive]
pub struct InteractionFrame {
    /// Selection version counter from `Selection::version()`. Default: 0.
    ///
    /// The renderer uses this to skip batch rebuild and GPU upload when selection
    /// state has not changed since the previous frame.
    pub selection_generation: u64,
    /// Gizmo model matrix. Some = selected object exists and gizmo should render.
    pub gizmo_model: Option<glam::Mat4>,
    /// Current gizmo interaction mode.
    pub gizmo_mode: GizmoMode,
    /// Current hovered gizmo axis.
    pub gizmo_hovered: GizmoAxis,
    /// Orientation for gizmo space (identity for world, object orientation for local).
    pub gizmo_space_orientation: glam::Quat,
    /// Constraint guide lines to render this frame.
    pub constraint_overlays: Vec<ConstraintOverlay>,
    /// Draw a stencil-outline ring around selected objects. Default: false.
    pub outline_selected: bool,
    /// RGBA colour of the selection outline ring. Default: white [1.0, 1.0, 1.0, 1.0].
    pub outline_colour: crate::Colour,
    /// Width of the outline ring in pixels. Default: 2.0.
    pub outline_width_px: f32,
    /// Render selected objects as a semi-transparent x-ray overlay. Default: false.
    pub xray_selected: bool,
    /// RGBA colour of the x-ray tint (should have alpha < 1). Default: [0.3, 0.7, 1.0, 0.25].
    pub xray_colour: crate::Colour,

    // --- Sub-object highlight ---
    /// Sub-object selection to highlight this frame.
    ///
    /// `None` = no sub-object highlights drawn. When `Some`, the renderer
    /// builds face fill, edge outline, and vertex/point sprite geometry from
    /// the snapshot and draws them after the opaque scene pass.
    pub sub_selection: Option<SubSelectionRef>,
    /// Fill colour (RGBA) for selected faces. The alpha component controls
    /// fill opacity. Default: translucent yellow `[1.0, 0.85, 0.0, 0.25]`.
    pub sub_highlight_face_fill_colour: crate::Colour,
    /// Edge colour (RGBA) for selected face outlines. Default: opaque yellow
    /// `[1.0, 0.85, 0.0, 1.0]`.
    pub sub_highlight_edge_colour: crate::Colour,
    /// Line width in pixels for face edge outlines. Default: `2.0`.
    pub sub_highlight_edge_width_px: f32,
    /// Point sprite size in pixels for selected vertices and point cloud
    /// points. Default: `10.0`.
    pub sub_highlight_vertex_size_px: f32,
}

impl Default for InteractionFrame {
    fn default() -> Self {
        Self {
            selection_generation: 0,
            gizmo_model: None,
            gizmo_mode: GizmoMode::Translate,
            gizmo_hovered: GizmoAxis::None,
            gizmo_space_orientation: glam::Quat::IDENTITY,
            constraint_overlays: Vec::new(),
            outline_selected: false,
            outline_colour: Colour::linear(1.0, 1.0, 1.0, 1.0),
            outline_width_px: 2.0,
            xray_selected: false,
            xray_colour: Colour::linear(0.3, 0.7, 1.0, 0.25),
            sub_selection: None,
            sub_highlight_face_fill_colour: Colour::linear(1.0, 0.85, 0.0, 0.25),
            sub_highlight_edge_colour: Colour::linear(1.0, 0.85, 0.0, 1.0),
            sub_highlight_edge_width_px: 2.0,
            sub_highlight_vertex_size_px: 10.0,
        }
    }
}

impl InteractionFrame {
    /// Build an interaction frame stamped with the current selection version.
    ///
    /// Sets `selection_generation` from [`Selection::version`](crate::interaction::select::selection::Selection::version)
    /// so the renderer can skip overlay rebuilds on unchanged frames.
    /// All other fields remain at their defaults.
    pub fn from_selection(selection: &crate::interaction::select::selection::Selection) -> Self {
        Self {
            selection_generation: selection.version(),
            ..Self::default()
        }
    }
}

// Ground-plane config (`GroundPlaneMode`, `GroundPlane`) lives in
// viewport-lib-types; re-exported so `crate::renderer::types::*` paths hold.
pub use viewport_lib_types::effects::ground::{GroundPlane, GroundPlaneMode};

// Environment settings live in viewport-lib-types; re-exported so
// `crate::renderer::types::*` paths hold.
pub use viewport_lib_types::effects::environment::{
    BackgroundSource, EnvironmentBackground, EnvironmentIntensity, EnvironmentLighting,
};

// Scatter-volume pass config (`ScatterQuality`, `ScatterSettings`) lives in
// viewport-lib-types; re-exported so `crate::renderer::types::*` paths hold.
pub use viewport_lib_types::effects::scatter::{ScatterQuality, ScatterSettings};

/// Configuration for the foreground pass.
///
/// The foreground pass draws `SceneFrame::foreground_items` (and any
/// item-type plugin implementing `paint_foreground`) over the finished
/// scene, against a cleared depth buffer. Setting this to `Some` is only
/// needed to override the projection; the pass itself runs whenever
/// foreground items are submitted.
///
/// Notes:
/// - Foreground items are never sliced by scene clip planes.
/// - Transparent foreground items use sorted back-to-front alpha blending,
///   not the OIT pass.
/// - The pass runs in the HDR path and in the owned-encoder LDR path. It is
///   not available through `paint_to` style host-owned render passes, where
///   a cleared depth attachment cannot exist; submitting foreground items
///   there logs a warning once and draws nothing.
/// - In the LDR path, 2D overlays are drawn inside the scene pass, so
///   foreground items paint over them. The HDR path keeps overlays on top.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForegroundPass {
    /// Projection override for the pass. `None` reuses the scene projection
    /// (always-on-top gizmos, x-ray parts, HUD props). `Some` replaces the
    /// projection while keeping the scene view transform: a first-person
    /// item gets its own field of view and a near plane small enough for
    /// close-held geometry without changing world depth precision.
    ///
    /// The override is always a perspective projection built from
    /// [`ForegroundProjection::fov_y`] and the scene camera's aspect, so over
    /// an orthographic scene camera the foreground items render in
    /// perspective.
    pub projection: Option<ForegroundProjection>,
}

/// Override projection for the foreground pass.
///
/// Shares the main camera's view transform; only the projection differs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForegroundProjection {
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance. `None` uses the scene camera's far plane.
    pub far: Option<f32>,
}

/// Global rendering effects and modifiers for one frame.
///
/// Groups lighting, clipping, post-processing, and clip
/// volumes : effects that apply globally across the scene rather than to
/// individual objects.
#[non_exhaustive]
pub struct EffectsFrame {
    /// Per-frame lighting configuration.
    pub lighting: LightingSettings,
    /// Participating-media (scatter-volume) quality settings.
    pub scatter: ScatterSettings,
    /// Clip objects (planes/box/sphere) and cap-fill toggle. Default: no clipping.
    pub clip: ClipSettings,
    /// Display transform: pipeline mode (HDR / Direct), exposure, and tone-map
    /// operator. Default: HDR pipeline, passthrough exposure, KhronosNeutral.
    pub display: DisplaySettings,
    /// Post-processing effects (bloom, SSAO, DOF, ...). Default: all off. Every
    /// effect requires `display.mode == PipelineMode::Hdr`.
    pub post_process: PostProcessSettings,
    /// Foreground pass configuration (projection override). Default: None.
    /// The pass itself is driven by `SceneFrame::foreground_items`; this only
    /// carries pass-wide settings.
    pub foreground: Option<ForegroundPass>,
    /// The environment that lights the scene. Default: None. What each viewport
    /// draws behind the scene is `ViewportFrame::environment_background`.
    pub environment: Option<EnvironmentLighting>,
    /// Ground plane configuration. Default: mode = None (not drawn, zero overhead).
    pub ground_plane: GroundPlane,
    /// Debug overlays (shadow-atlas viewer). Default: off.
    pub debug: EffectsDebug,
}

impl Default for EffectsFrame {
    fn default() -> Self {
        Self {
            lighting: LightingSettings::default(),
            scatter: ScatterSettings::default(),
            clip: ClipSettings::default(),
            display: DisplaySettings::default(),
            post_process: PostProcessSettings::default(),
            foreground: None,
            environment: None,
            ground_plane: GroundPlane::default(),
            debug: EffectsDebug::default(),
        }
    }
}

/// Clip geometry for one frame: active clip objects plus the cross-section
/// cap-fill toggle. Grouped on [`EffectsFrame::clip`].
#[non_exhaustive]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClipSettings {
    /// Active clip objects (planes, boxes, spheres). Max 6 planes + 1 box/sphere.
    /// Default: empty (no clipping).
    pub objects: Vec<ClipObject>,
    /// Whether to render filled caps at clip plane cross-sections. Default: true.
    pub cap_fill_enabled: bool,
}

impl Default for ClipSettings {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            cap_fill_enabled: true,
        }
    }
}

// `EffectsDebug` lives in viewport-lib-types (effects::debug); re-exported
// so `crate::renderer::types::*` paths keep resolving.
pub use viewport_lib_types::effects::debug::EffectsDebug;

/// Scene-global effects for one frame, consumed by
/// [`PassPath::prepare_scene`](crate::PassPath::prepare_scene) and
/// [`OwnedPath::prepare_scene`](crate::OwnedPath::prepare_scene).
///
/// Groups the lighting, environment, and scatter configuration that applies
/// to the whole scene (not per-viewport). Construct directly or obtain via
/// [`EffectsFrame::split`].
///
/// # Multi-viewport usage
/// Call `prepare_scene` once per frame with this struct. Each viewport's
/// per-viewport effects are passed separately via [`ViewportEffects`] in
/// `prepare_viewport`.
pub struct SceneEffects<'a> {
    /// Per-frame lighting configuration (drives the shadow pass and light uniform).
    pub lighting: &'a LightingSettings,
    /// The environment that lights the scene.
    pub environment: &'a Option<EnvironmentLighting>,
    /// Participating-media quality settings (scene-global).
    pub scatter: &'a ScatterSettings,
}

/// Per-viewport effects for one frame, consumed by
/// [`PassPath::prepare_viewport`](crate::PassPath::prepare_viewport) and
/// [`OwnedPath::prepare_viewport`](crate::OwnedPath::prepare_viewport).
///
/// Groups the clip objects and post-processing settings that differ
/// per viewport. Construct directly or obtain via [`EffectsFrame::split`].
///
/// # Multi-viewport usage
/// Pass one `ViewportEffects` per viewport to `prepare_viewport`. Scene-global
/// effects are passed once via [`SceneEffects`] in `prepare_scene`.
pub struct ViewportEffects<'a> {
    /// Clip objects and cap-fill toggle.
    pub clip: &'a ClipSettings,
    /// Display transform: pipeline mode, exposure, tone-map operator.
    pub display: &'a DisplaySettings,
    /// Post-processing effects (bloom, SSAO, DOF, ...).
    pub post_process: &'a PostProcessSettings,
    /// Foreground pass configuration (projection override).
    pub foreground: &'a Option<ForegroundPass>,
    /// Ground plane configuration for this viewport.
    pub ground_plane: &'a GroundPlane,
    /// Debug overlays (shadow-atlas viewer).
    pub debug: &'a EffectsDebug,
}

/// A coherent lighting + exposure starting point.
///
/// Whether a scene is "faithful" or "physical daylight" is not one field - it
/// emerges from the light magnitudes in [`EffectsFrame::lighting`] agreeing with
/// the exposure in [`EffectsFrame::display`]. Setting one without the other is the
/// common footgun: real-lux lights under the neutral default exposure clip to
/// white, and an adaptive daylight camera over nominal magnitudes silently loses
/// the "colour is data" look. `LightingPosture` sets both halves together via
/// [`EffectsFrame::with_posture`] so they cannot disagree.
///
/// This governs lighting and exposure only. The per-material
/// [`ShadingModel`](crate::ShadingModel) is chosen independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LightingPosture {
    /// "Colour is data": nominal light magnitudes shown at a fixed neutral
    /// exposure with no camera adaptation, so authored colours read on screen
    /// roughly as-is. The default posture, matching [`EffectsFrame::default`].
    #[default]
    Faithful,
    /// Physically-scaled daylight: real photometric magnitudes
    /// ([`LightingSettings::daylight`]) mapped down by an adaptive camera
    /// ([`ExposureSettings::automatic`]) so the sun does not clip to white.
    /// The environment intensity is a lux target matched to the daylight fill,
    /// so an HDRI sky lights the scene beside the sun whatever its stored
    /// brightness.
    PhysicalDaylight,
}

impl EffectsFrame {
    /// Apply a [`LightingPosture`], setting [`EffectsFrame::lighting`] and the
    /// exposure on [`EffectsFrame::display`] as a matched pair.
    ///
    /// Pick the intent in one call instead of hand-pairing light magnitudes with
    /// an exposure mode - the two must agree or the scene clips (real lux at
    /// neutral exposure) or loses the colour-is-data look (nominal magnitudes
    /// under an adaptive camera). Call it early, then author your own lights
    /// (push onto `effects.lighting.lights`) and adjust other display fields:
    ///
    /// ```no_run
    /// # use viewport_lib::{EffectsFrame, LightingPosture};
    /// let effects = EffectsFrame::default().with_posture(LightingPosture::PhysicalDaylight);
    /// ```
    ///
    /// The environment intensity is part of [`LightingSettings`], so it follows
    /// the posture too: a multiplier of 1.0 for `Faithful`, a lux target for
    /// `PhysicalDaylight`. Set the environment itself afterwards; it is not
    /// touched. Leaves other display fields (pipeline mode, tone-map operator)
    /// and the per-material [`ShadingModel`](crate::ShadingModel) untouched.
    pub fn with_posture(mut self, posture: LightingPosture) -> Self {
        match posture {
            LightingPosture::Faithful => {
                self.lighting = LightingSettings::default();
                self.display.exposure = ExposureSettings::default();
            }
            LightingPosture::PhysicalDaylight => {
                self.lighting = LightingSettings::daylight();
                self.display.exposure = ExposureSettings::automatic();
            }
        }
        self
    }

    /// Decompose into scene-global and per-viewport effect references.
    ///
    /// Both halves borrow from `self` and cannot outlive the `EffectsFrame`.
    /// The scene half is passed to
    /// [`PassPath::prepare_scene`](crate::PassPath::prepare_scene); the
    /// viewport half is passed to
    /// [`PassPath::prepare_viewport`](crate::PassPath::prepare_viewport) (or the
    /// [`OwnedPath`](crate::OwnedPath) equivalents).
    ///
    /// Single-viewport callers can continue using
    /// [`PassPath::prepare`](crate::PassPath::prepare) directly without calling
    /// `split()`.
    pub fn split(&self) -> (SceneEffects<'_>, ViewportEffects<'_>) {
        (
            SceneEffects {
                lighting: &self.lighting,
                environment: &self.environment,
                scatter: &self.scatter,
            },
            ViewportEffects {
                clip: &self.clip,
                display: &self.display,
                post_process: &self.post_process,
                foreground: &self.foreground,
                ground_plane: &self.ground_plane,
                debug: &self.debug,
            },
        )
    }
}

#[cfg(test)]
#[path = "render_camera_tests.rs"]
mod render_camera_tests;
