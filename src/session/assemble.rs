//! Frame assembly and the camera-driving entry points.

use super::ViewportInstance;
use crate::interaction::input::ViewportContext;
use crate::interaction::manipulation::gizmo::{GizmoAxis, GizmoMode, compute_gizmo_scale};
use crate::interaction::manipulation::{ManipResult, ManipulationContext, ManipulationKind};
use crate::runtime::RuntimeFrameContext;
use crate::{
    CameraFrame, FrameData, InteractionFrame, OrbitCameraController, OverlayFrame, SceneFrame,
};

impl ViewportInstance {
    /// Resolve input, drive the camera with `orbit`, run manipulation, and
    /// assemble the frame. Returns the assembled [`FrameData`].
    ///
    /// Orbit is suppressed while a manipulation session is active so pointer
    /// motion drives the gizmo, not the camera. `orbit` is borrowed so the
    /// application keeps its own handle for tuning.
    pub fn update_orbit(&mut self, orbit: &mut OrbitCameraController) -> &FrameData {
        self.action = self.input.resolve();
        orbit.set_viewport_size(self.viewport_size);

        let manip_active = self.manip.as_ref().is_some_and(|m| m.is_active());
        if !manip_active {
            orbit.apply(&mut self.camera, &self.action);
        }

        self.run_manip();
        self.assemble();
        &self.frame
    }

    /// Assemble the frame without touching the camera. Use this when a controller
    /// other than orbit (or your own input pipeline) has already moved the camera
    /// this frame; see [`camera_mut`](Self::camera_mut) and [`resolve`](Self::resolve).
    pub fn frame(&mut self, ctx: ViewportContext) -> &FrameData {
        self.viewport_size = ctx.viewport_size;
        self.action = self.input.resolve();
        self.run_manip();
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
    /// session.update_orbit_with(&mut orbit, |frame| {
    ///     frame.scene.point_clouds.push(cloud);
    ///     frame.overlays.labels.push(label);
    /// });
    /// ```
    pub fn update_orbit_with(
        &mut self,
        orbit: &mut OrbitCameraController,
        inject: impl FnOnce(&mut FrameData),
    ) -> &FrameData {
        self.update_orbit(orbit);
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
    /// per-frame camera update so assembly reads the stepped scene.
    ///
    /// The per-frame drive lives here rather than on
    /// [`runtime_mut`](Self::runtime_mut) because `step` needs the scene and
    /// selection at the same time as the runtime, which only works as disjoint
    /// borrows of the session's own fields.
    pub fn step_runtime(&mut self, dt: f32) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let pointer = self.action.pointer;
        let mut ctx = RuntimeFrameContext {
            dt,
            camera: self.camera.clone(),
            viewport_size: glam::Vec2::from(self.viewport_size),
            input: self.action.clone(),
            pointer_delta: pointer.delta,
            cursor_viewport: pointer.cursor,
            clicked: pointer.clicked,
            drag_started: pointer.drag_started,
            dragging: pointer.dragging,
            shift_held: self.input.modifiers().shift,
            ..Default::default()
        };
        ctx.pick_hit = None;
        let _output = runtime.step(&mut self.scene, &mut self.selection, &ctx);
    }

    /// Run the manipulation controller for this frame, if one is attached.
    fn run_manip(&mut self) {
        if self.manip.is_none() {
            self.last_manip = ManipResult::None;
            return;
        }
        let selection_center = self.selection_center();
        let pointer = self.action.pointer;
        let ctx = ManipulationContext {
            camera: self.camera.clone(),
            viewport_size: glam::Vec2::from(self.viewport_size),
            cursor_viewport: pointer.cursor,
            pointer_delta: pointer.delta,
            selection_center,
            gizmo: None,
            drag_started: pointer.drag_started,
            dragging: pointer.dragging,
            clicked: pointer.clicked,
        };
        // Disjoint field borrows: `manip` mutable, `action` shared.
        self.last_manip = self.manip.as_mut().unwrap().update(&self.action, ctx);
    }

    /// World-space centroid of the selected nodes, or `None` when nothing is
    /// selected. Feeds `ManipulationContext::selection_center`, which gates G/R/S.
    fn selection_center(&self) -> Option<glam::Vec3> {
        if self.selection.is_empty() {
            return None;
        }
        let mut sum = glam::Vec3::ZERO;
        let mut count = 0.0f32;
        for &id in self.selection.iter() {
            if let Some(node) = self.scene.node(id) {
                sum += node.world_transform().col(3).truncate();
                count += 1.0;
            }
        }
        (count > 0.0).then(|| sum / count)
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
        self.frame.interaction.outline_colour = self.outline_colour;
        self.frame.interaction.outline_width_px = self.outline_width_px;
        self.stamp_gizmo();
        self.frame.overlays = OverlayFrame::default();
        self.inject_extras();
    }

    /// Draw the transform gizmo whenever something is selected: at rest it sits
    /// at the selection centroid in translate mode; during a G/R/S session it
    /// reflects the active kind and constrained axis. Positioned at the live
    /// centroid so it follows a translating object. Nothing selected ->
    /// `gizmo_model` stays `None` and no gizmo is drawn.
    fn stamp_gizmo(&mut self) {
        let Some(center) = self.selection_center() else {
            return;
        };
        let scale = compute_gizmo_scale(
            center,
            self.camera.eye_position(),
            self.camera.fov_y,
            self.viewport_size[1],
        );
        self.frame.interaction.gizmo_model = Some(glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(scale),
            glam::Quat::IDENTITY,
            center,
        ));
        let state = self.manip.as_ref().and_then(|m| m.state());
        self.frame.interaction.gizmo_mode = match state.as_ref().map(|s| s.kind) {
            Some(ManipulationKind::Rotate) => GizmoMode::Rotate,
            Some(ManipulationKind::Scale) => GizmoMode::Scale,
            Some(ManipulationKind::Move) | None => GizmoMode::Translate,
        };
        self.frame.interaction.gizmo_hovered =
            state.and_then(|s| s.axis).unwrap_or(GizmoAxis::None);
    }
}
