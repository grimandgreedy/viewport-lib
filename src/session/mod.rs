//! [`ViewportSession`]: a host object that owns the per-frame wiring.
//!
//! The library deliberately stops at [`FrameData`]: the host owns the window,
//! the event loop, and tool state. That leaves every consumer writing the same
//! setup: translate native events, resolve input, drive the camera, assemble a
//! [`FrameData`], render. `ViewportSession` bundles that wiring behind one object
//! while leaving the advanced surface reachable through accessors
//! ([`resources_mut`](ViewportSession::resources_mut),
//! [`runtime_mut`](ViewportSession::runtime_mut),
//! [`renderer_mut`](ViewportSession::renderer_mut)), so a program can start small
//! and grow into plugins and custom passes without a rewrite.
//!
//! The session owns the renderer, scene, selection, camera, and the input
//! resolver. It does not own the `device`/`queue` (the host framework creates
//! those, and the wgpu version must match the host's), and it does not weld in a
//! camera-motion policy: [`update_orbit`](ViewportSession::update_orbit) is the
//! batteries-included path, and [`resolve`](ViewportSession::resolve) +
//! [`camera_mut`](ViewportSession::camera_mut) + [`frame`](ViewportSession::frame)
//! let any controller drive the camera instead.

mod assemble;
mod extras;
mod render;
mod settings;

pub use extras::ExtraId;

mod orbit;
pub use orbit::OrbitSession;

/// Host integrations that own the window and event loop (feature-gated).
pub mod hosts;

use crate::camera::Camera;
use crate::interaction::input::{ActionFrame, BindingPreset, ViewportContext, ViewportEvent};
use crate::interaction::manipulation::{ManipResult, ManipulationController};
use crate::interaction::select::selection::Selection;
use crate::resources::DeviceResources;
use crate::runtime::ViewportRuntime;
use crate::scene::scene::Scene;
use crate::{FrameData, InteractionFrame, ViewportInput, ViewportRenderer};

/// Owns the per-frame viewport wiring: renderer, scene, camera, input, and the
/// assembled [`FrameData`].
///
/// See the [module docs](self) for the design. The short version: construct one
/// with [`new`](Self::new), feed it native events translated to [`ViewportEvent`]
/// via [`handle_event`](Self::handle_event), drive it once per frame with
/// [`update_orbit`](Self::update_orbit) (or the manual camera path), then submit
/// with [`render`](Self::render) (owned surface) or
/// [`prepare`](Self::prepare) + [`paint`](Self::paint) (render-pass host).
pub struct ViewportSession {
    renderer: ViewportRenderer,
    scene: Scene,
    selection: Selection,
    camera: Camera,
    input: ViewportInput,
    manip: Option<ManipulationController>,
    runtime: Option<ViewportRuntime>,

    /// Retained frame. Its `effects` and `viewport` sub-frames hold persistent
    /// settings (never overwritten by assembly); `camera`, `scene`, `interaction`,
    /// and `overlays` are rebuilt each frame. See [`assemble`](Self::assemble).
    frame: FrameData,
    /// Resolved input for the current frame. Set by [`resolve`](Self::resolve).
    action: ActionFrame,
    /// Result of the last manipulation update, surfaced via [`last_manip`](Self::last_manip).
    last_manip: ManipResult,

    /// Viewport size in logical pixels, recorded by [`begin_frame`](Self::begin_frame).
    viewport_size: [f32; 2],
    /// Physical pixels per logical pixel. Sizes the physical render target and
    /// keeps overlays/axes crisp on HiDPI. Default 1.0 (physical == logical).
    pixels_per_point: f32,

    // Persistent selection-outline styling, re-stamped onto the interaction
    // sub-frame each assembly (the settings half of `InteractionFrame`).
    outline_selected: bool,
    outline_colour: [f32; 4],
    outline_width_px: f32,

    // Retained non-mesh items, re-injected into the scene sub-frame each
    // assembly so static point clouds/glyphs/volumes/splats are added once.
    extras: Vec<(ExtraId, extras::SceneExtra)>,
    next_extra_id: u64,
}

impl ViewportSession {
    /// Create a session for a renderer targeting `target_format`.
    ///
    /// The `device` is used to build the renderer's pipelines; the session does
    /// not keep it. CPU picking is enabled so [`pick`](Self::pick) works after
    /// the first frame.
    pub fn new(device: &crate::gpu::Device, target_format: crate::gpu::TextureFormat) -> Self {
        // Embedded hosts create the device before the session, so a missing
        // feature otherwise degrades silently (e.g. mesh sub-object picking).
        // Warn about the ones the caller could still enable at device creation.
        fn warn_missing_device_features(device: &crate::gpu::Device) {
            let have = device.features();
            let mut missing: Vec<&str> = Vec::new();
            if !have.contains(crate::gpu::PRIMITIVE_INDEX_FEATURE) {
                missing.push(
                    "SHADER_PRIMITIVE_INDEX (mesh face/vertex/edge picking; \
                     falls back to object-level without it)",
                );
            }
            if !have.contains(crate::gpu::Features::FLOAT32_FILTERABLE) {
                missing.push("FLOAT32_FILTERABLE (linear filtering of float textures)");
            }
            if !have.contains(crate::gpu::Features::TIMESTAMP_QUERY) {
                missing.push("TIMESTAMP_QUERY (GPU frame timings)");
            }
            if !have.contains(crate::gpu::Features::INDIRECT_FIRST_INSTANCE) {
                missing.push("INDIRECT_FIRST_INSTANCE (GPU-driven culling)");
            }
            if !missing.is_empty() {
                tracing::warn!(
                    "ViewportSession: device is missing recommended features: {}. If your \
                     adapter supports them, request \
                     ViewportRenderer::recommended_device_features(adapter) in your device \
                     descriptor.",
                    missing.join("; ")
                );
            }
        }
        warn_missing_device_features(device);

        let mut renderer = ViewportRenderer::new(device, target_format);
        renderer.set_cpu_pick_cache(true);
        let defaults = InteractionFrame::default();
        Self {
            renderer,
            scene: Scene::new(),
            selection: Selection::new(),
            camera: Camera::default(),
            // ViewportAll carries the manipulation keybindings (G/R/S, axis
            // constraints) as well as camera navigation, so a session with a
            // ManipulationController resolves them without extra setup.
            input: ViewportInput::from_preset(BindingPreset::ViewportAll),
            manip: None,
            runtime: None,
            frame: FrameData::default(),
            action: ActionFrame::default(),
            last_manip: ManipResult::None,
            viewport_size: [1.0, 1.0],
            pixels_per_point: 1.0,
            outline_selected: defaults.outline_selected,
            outline_colour: defaults.outline_colour,
            outline_width_px: defaults.outline_width_px,
            extras: Vec::new(),
            next_extra_id: 0,
        }
    }

    /// Attach a [`ViewportRuntime`] for physics, animation, and GPU plugins.
    ///
    /// Drive it each frame with [`step_runtime`](Self::step_runtime). Register
    /// plugins after construction through [`runtime_mut`](Self::runtime_mut)
    /// using the `add_*` methods.
    pub fn with_runtime(mut self, runtime: ViewportRuntime) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Attach a [`ManipulationController`] for move/rotate/scale sessions.
    ///
    /// Without it, [`update_orbit`](Self::update_orbit) skips the manipulation
    /// step entirely. With it, the controller runs each frame and its result is
    /// available from [`last_manip`](Self::last_manip); applying that result to
    /// scene nodes stays with the application.
    pub fn with_manipulation(mut self, manip: ManipulationController) -> Self {
        self.manip = Some(manip);
        self
    }

    // ---- per-frame input (universal, all hosts) --------------------------------

    /// Begin a frame: record the viewport context and reset the input
    /// accumulator. Call once per frame around the batch of `handle_event` calls,
    /// following the host's event delivery (start of frame for egui/iced, end of
    /// frame after present for winit).
    pub fn begin_frame(&mut self, ctx: ViewportContext) {
        self.viewport_size = ctx.viewport_size;
        self.input.begin_frame(ctx);
    }

    /// Feed one native event, already translated to a [`ViewportEvent`].
    pub fn handle_event(&mut self, event: ViewportEvent) {
        self.input.push_event(event);
    }

    /// Update the viewport size without resetting the input accumulator.
    ///
    /// The size drives camera aspect and the renderer's internal target sizes, so
    /// it must match the surface being rendered into. Call this when the surface
    /// resizes between [`begin_frame`](Self::begin_frame) calls (a host that
    /// begins its frame at the end of the render loop would otherwise assemble
    /// with a stale size, mismatching the freshly resized swapchain texture).
    pub fn set_viewport_size(&mut self, size: [f32; 2]) {
        self.viewport_size = size;
    }

    /// Set the physical pixels per logical pixel (e.g. `ui.ctx().pixels_per_point()`).
    ///
    /// [`viewport_size`](Self::set_viewport_size) stays in logical units; this
    /// only sizes the physical render target and keeps overlays and the axes
    /// indicator crisp on HiDPI displays. The host that renders into an offscreen
    /// texture must size it `viewport_size * pixels_per_point`. Default 1.0.
    pub fn set_pixels_per_point(&mut self, pixels_per_point: f32) {
        self.pixels_per_point = pixels_per_point.max(0.001);
    }

    /// Resolve accumulated events into an [`ActionFrame`] and cache it.
    ///
    /// [`update_orbit`](Self::update_orbit) and [`frame`](Self::frame) call this
    /// internally; call it directly when driving the camera yourself so
    /// [`action_frame`](Self::action_frame) reflects this frame's input before
    /// you apply your own controller.
    pub fn resolve(&mut self) -> &ActionFrame {
        self.action = self.input.resolve();
        &self.action
    }

    /// The most recently resolved [`ActionFrame`].
    pub fn action_frame(&self) -> &ActionFrame {
        &self.action
    }

    /// The result of the last manipulation update. [`ManipResult::None`] when no
    /// controller is attached or nothing happened this frame.
    pub fn last_manip(&self) -> ManipResult {
        self.last_manip
    }

    /// Whether a move/rotate/scale session is currently active. A host driving
    /// its own camera (first-person, fly) uses this to suppress camera motion
    /// while a manipulation drag owns the pointer, the way
    /// [`update_orbit`](Self::update_orbit) suppresses orbit internally.
    pub fn is_manipulating(&self) -> bool {
        self.manip.as_ref().is_some_and(|m| m.is_active())
    }

    /// Snapshot of the active manipulation session, or `None` when idle. The
    /// `center` field is the pivot to rotate and scale about when applying a
    /// [`last_manip`](Self::last_manip) delta to the selection.
    pub fn manip_state(&self) -> Option<crate::interaction::manipulation::ManipulationState> {
        self.manip.as_ref().and_then(|m| m.state())
    }

    // ---- camera ---------------------------------------------------------------

    /// The current camera.
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// The current camera, for controllers that drive it directly.
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    // ---- accessors that keep the full library reachable -----------------------

    /// The scene graph.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// The scene graph, for adding and transforming nodes.
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// The selection set.
    pub fn selection_mut(&mut self) -> &mut Selection {
        &mut self.selection
    }

    /// The GPU resource registry: mesh uploads, plugins, deformers, and mesh
    /// sidecars (skinning, displacement, scalar fields).
    pub fn resources_mut(&mut self) -> &mut DeviceResources {
        self.renderer.resources_mut()
    }

    /// The runtime, when one is attached: register physics/animation/GPU plugins
    /// with its `add_*` methods. Per-frame driving is
    /// [`step_runtime`](Self::step_runtime), not this accessor.
    pub fn runtime_mut(&mut self) -> Option<&mut ViewportRuntime> {
        self.runtime.as_mut()
    }

    /// The underlying renderer, for advanced use the session does not wrap.
    pub fn renderer_mut(&mut self) -> &mut ViewportRenderer {
        &mut self.renderer
    }

    /// Forward a host device recreation to the renderer and runtime so their
    /// GPU state re-initialises. Call after the host rebuilds the device.
    pub fn notify_device_recreated(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        self.renderer.notify_device_recreated(device, queue);
        if let Some(runtime) = self.runtime.as_mut() {
            runtime.notify_device_recreated(device, queue);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::input::{ButtonState, MouseButton};
    use crate::{Material, OrbitCameraController, PointCloudItem, primitives};

    fn headless_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
                label: Some("session_tests"),
                ..Default::default()
            }))
            .ok()?;
        Some((device, queue))
    }

    fn ctx() -> ViewportContext {
        ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [256.0, 256.0],
        }
    }

    #[test]
    fn assemble_and_render_offscreen() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping assemble_and_render_offscreen: no GPU adapter");
            return;
        };
        let format = crate::gpu::TextureFormat::Bgra8UnormSrgb;
        let mut session = ViewportSession::new(&device, format);

        let cube = session
            .resources_mut()
            .upload_mesh_data(&device, &primitives::cube(1.0))
            .unwrap();
        session.scene_mut().add(
            Some(cube),
            glam::Mat4::IDENTITY,
            Material::from_colour([0.6, 0.6, 0.9]),
        );
        session.camera_mut().distance = 6.0;

        session.begin_frame(ctx());
        let mut orbit = OrbitCameraController::viewport_all();
        let frame = session.update_orbit(&mut orbit);
        // Assembly collected the scene node and stamped a non-default generation.
        assert!(
            frame.scene.generation != 0,
            "scene generation should be stamped from Scene::version"
        );

        // Render into an offscreen target: exercises the owned HDR path end to end.
        let tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("session_test_target"),
            size: crate::gpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let cmd = session.render(&device, &queue, &view);
        queue.submit(std::iter::once(cmd));
    }

    #[test]
    fn pointer_state_drives_click_detection() {
        // Verifies step 0: a press then release without movement is a click, and
        // the session surfaces it on the resolved frame's pointer state.
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping pointer_state_drives_click_detection: no GPU adapter");
            return;
        };
        let mut session = ViewportSession::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        session.begin_frame(ctx());
        session.handle_event(ViewportEvent::PointerMoved {
            position: glam::Vec2::new(40.0, 40.0),
        });
        session.handle_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
        });
        session.handle_event(ViewportEvent::MouseButton {
            button: MouseButton::Left,
            state: ButtonState::Released,
        });
        let action = session.resolve();
        assert!(
            action.pointer.clicked,
            "release without drag should be a click"
        );
        assert!(
            action.pointer.drag_started,
            "press should mark drag_started"
        );
        assert_eq!(action.pointer.cursor, Some(glam::Vec2::new(40.0, 40.0)));
    }

    #[test]
    fn retained_extras_and_injection() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping retained_extras_and_injection: no GPU adapter");
            return;
        };
        let mut session = ViewportSession::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        session.begin_frame(ctx());
        let mut orbit = OrbitCameraController::viewport_all();

        // A retained extra is re-injected into the scene every frame.
        let mut pc = PointCloudItem::default();
        pc.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let id = session.add_point_cloud(pc);
        let frame = session.update_orbit(&mut orbit);
        assert_eq!(frame.scene.point_clouds.len(), 1, "retained extra injected");

        // The injection closure runs after assembly, so per-frame items land.
        let frame = session.update_orbit_with(&mut orbit, |f| {
            f.scene.point_clouds.push(PointCloudItem::default());
        });
        assert_eq!(
            frame.scene.point_clouds.len(),
            2,
            "retained + per-frame injected item"
        );

        // Removing the retained extra drops it from later frames.
        assert!(session.remove_extra(id));
        let frame = session.update_orbit(&mut orbit);
        assert_eq!(frame.scene.point_clouds.len(), 0, "removed extra gone");
    }
}
