//! [`OrbitSession`]: a [`ViewportSession`] that owns a default orbit controller.

use std::ops::{Deref, DerefMut};

use super::ViewportSession;
use crate::{FrameData, OrbitCameraController};

/// A [`ViewportSession`] bundled with its own [`OrbitCameraController`], for the
/// shortest programs: one [`update`](Self::update) call per frame drives the
/// camera with no controller handling in user code.
///
/// It derefs to the inner [`ViewportSession`], so every session accessor
/// (`scene_mut`, `resources_mut`, `pick`, ...) is reachable directly.
pub struct OrbitSession {
    session: ViewportSession,
    orbit: OrbitCameraController,
}

impl OrbitSession {
    /// Create an orbit session for a renderer targeting `target_format`.
    pub fn new(device: &crate::gpu::Device, target_format: crate::gpu::TextureFormat) -> Self {
        Self {
            session: ViewportSession::new(device, target_format),
            orbit: OrbitCameraController::viewport_all(),
        }
    }

    /// Drive the camera with the bundled orbit controller and assemble the frame.
    pub fn update(&mut self) -> &FrameData {
        self.session.update_orbit(&mut self.orbit)
    }

    /// Drive the camera and run `inject` against the assembled frame, the
    /// ordering-safe way to add per-frame overlays and non-mesh items. See
    /// [`ViewportSession::update_orbit_with`].
    pub fn update_with(&mut self, inject: impl FnOnce(&mut FrameData)) -> &FrameData {
        self.session.update_orbit_with(&mut self.orbit, inject)
    }

    /// The bundled orbit controller, for tuning sensitivity or navigation mode.
    pub fn orbit_mut(&mut self) -> &mut OrbitCameraController {
        &mut self.orbit
    }

    /// The inner session.
    pub fn session_mut(&mut self) -> &mut ViewportSession {
        &mut self.session
    }
}

impl Deref for OrbitSession {
    type Target = ViewportSession;

    fn deref(&self) -> &ViewportSession {
        &self.session
    }
}

impl DerefMut for OrbitSession {
    fn deref_mut(&mut self) -> &mut ViewportSession {
        &mut self.session
    }
}
