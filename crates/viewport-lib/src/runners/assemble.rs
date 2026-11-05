//! Frame assembly and the camera-driving entry points.

use super::ViewportInstance;
use crate::Colour;
use crate::interaction::input::ViewportContext;
use crate::runtime::RuntimeFrameContext;
use crate::{
    CameraFrame, FrameData, InteractionFrame, OrbitCameraController, OverlayFrame, SceneFrame,
};

impl ViewportInstance {
    /// Resolve input, drive the camera with `orbit`, and assemble the frame.
    /// Returns the assembled [`FrameData`].
    ///
    /// `orbit` is borrowed so the application keeps its own handle for tuning.
    /// `dt` is the frame time in seconds, passed to
    /// [`OrbitCameraController::apply`] for fly movement. A host running a tool
    /// that takes the pointer (a transform gizmo, say) drives the camera from
    /// its own resolver instead and calls [`frame`](Self::frame).
    pub fn update_orbit(&mut self, orbit: &mut OrbitCameraController, dt: f32) -> &FrameData {
        self.action = self.input.resolve();
        orbit.set_viewport_size(self.viewport_size);
        orbit.apply(&mut self.camera, &self.action, dt);
        self.assemble();
        &self.frame
    }

    /// Assemble the frame without touching the camera. Use this when a controller
    /// other than orbit (or your own input pipeline) has already moved the camera
    /// this frame; see [`camera_mut`](Self::camera_mut) and [`resolve`](Self::resolve).
    pub fn frame(&mut self, ctx: ViewportContext) -> &FrameData {
        self.viewport_size = ctx.viewport_size;
        self.action = self.input.resolve();
        self.assemble();
        &self.frame
    }

    /// Like [`update_orbit`](Self::update_orbit), then run `inject` against the
    /// assembled frame before returning it. This is the ordering-safe way to add
    /// per-frame overlays and non-mesh items: assembly rebuilds the scene
    /// sub-frame and clears overlays, so pushing through
    /// [`frame_data_mut`](Self::frame_data_mut) only lands if it happens after
    /// the update; this closure runs at that exact point.
    ///
    /// ```rust,ignore
    /// session.update_orbit_with(&mut orbit, dt, |frame| {
    ///     frame.scene.items_mut::<crate::PointCloudItem>().push(cloud);
    ///     frame.overlays.labels.push(label);
    /// });
    /// ```
    pub fn update_orbit_with(
        &mut self,
        orbit: &mut OrbitCameraController,
        dt: f32,
        inject: impl FnOnce(&mut FrameData),
    ) -> &FrameData {
        self.update_orbit(orbit, dt);
        inject(&mut self.frame);
        &self.frame
    }

    /// Like [`frame`](Self::frame), then run `inject` against the assembled frame.
    /// The bring-your-own-camera counterpart of
    /// [`update_orbit_with`](Self::update_orbit_with).
    pub fn frame_with(
        &mut self,
        ctx: ViewportContext,
        inject: impl FnOnce(&mut FrameData),
    ) -> &FrameData {
        self.frame(ctx);
        inject(&mut self.frame);
        &self.frame
    }

    /// Step the attached runtime by `dt` seconds, writing simulated transforms
    /// back into the scene. No-op when no runtime is attached. Call before the
    /// per-frame camera update so assembly reads the stepped scene. `dt` is
    /// also what the runtime's GPU plugins get as `GpuFrameContext::dt` on
    /// the next render (0 until this is first called).
    ///
    /// The per-frame drive lives here rather than on
    /// [`runtime_mut`](Self::runtime_mut) because `step` needs the scene and
    /// selection at the same time as the runtime, which only works as disjoint
    /// borrows of the session's own fields.
    pub fn step_runtime(&mut self, dt: f32) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        self.runtime_dt = dt;
        let pointer = self.action.pointer;
        let mut ctx = RuntimeFrameContext {
            dt,
            camera: self.camera.clone(),
            viewport_size: glam::Vec2::from(self.viewport_size),
            input: self.action.clone(),
            pointer_delta: pointer.delta,
            cursor_viewport: pointer.cursor,
            clicked: pointer.clicked,
            drag_started: pointer.pressed,
            dragging: pointer.dragging,
            shift_held: self.input.modifiers().shift,
            ..Default::default()
        };
        ctx.pick_hit = None;
        let _output = runtime.step(&mut self.scene, &mut self.selection, &ctx);
    }

    /// Rebuild the per-frame sub-frames of the retained [`FrameData`].
    ///
    /// `camera`, `scene`, and the derived half of `interaction` are rebuilt from
    /// the current camera/scene/selection each frame; `effects` and `viewport`
    /// persist as settings and are left untouched; the outline styling is
    /// re-stamped; and `overlays` are cleared so the host re-injects per-frame
    /// overlays and direct items after this call, before rendering.
    fn assemble(&mut self) {
        let [w, h] = self.viewport_size;
        self.camera.set_aspect_ratio(w, h);
        self.frame.camera = CameraFrame::from_camera(&self.camera, self.viewport_size)
            .with_pixels_per_point(self.pixels_per_point);
        self.frame.scene = SceneFrame::from_scene(&mut self.scene, &self.selection);
        self.frame.interaction = InteractionFrame::from_selection(&self.selection);
        self.frame.interaction.outline_selected = self.outline_selected;
        self.frame.interaction.outline_colour = Colour::from_linear_array(self.outline_colour);
        self.frame.interaction.outline_width_px = self.outline_width_px;
        self.frame.overlays = OverlayFrame::default();
        self.inject_extras();
    }
}
