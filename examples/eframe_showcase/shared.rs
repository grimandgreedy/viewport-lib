//! Shared helpers used across multiple showcase modules.

use eframe::egui;
use viewport_lib::{Camera, KeyCode, MeshId, ViewportRenderer};

use crate::{App, ShowcaseMode};

// ---------------------------------------------------------------------------
// Mesh upload
// ---------------------------------------------------------------------------

impl App {
    /// Upload a new box mesh slot into the renderer. Used by showcase build methods.
    pub(crate) fn upload_box(&self, renderer: &mut ViewportRenderer) -> MeshId {
        renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &self.box_mesh_data)
            .expect("box mesh upload")
    }
}

// ---------------------------------------------------------------------------
// Per-mode opening camera
// ---------------------------------------------------------------------------

/// Returns a camera to apply when switching to `mode`, if the mode needs a specific starting view.
/// Returns `None` for modes that should keep the current camera.
pub(crate) fn opening_camera(mode: ShowcaseMode) -> Option<Camera> {
    match mode {
        ShowcaseMode::PickLevels => Some(Camera {
            center: glam::Vec3::ZERO,
            distance: 14.0,
            orientation: glam::Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            ..Camera::default()
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Input mapping
// ---------------------------------------------------------------------------

/// Translate an egui key into a viewport-lib `KeyCode`.
pub(crate) fn egui_key_to_keycode(key: egui::Key) -> Option<KeyCode> {
    match key {
        egui::Key::A => Some(KeyCode::A),
        egui::Key::D => Some(KeyCode::D),
        egui::Key::E => Some(KeyCode::E),
        egui::Key::F => Some(KeyCode::F),
        egui::Key::G => Some(KeyCode::G),
        egui::Key::Q => Some(KeyCode::Q),
        egui::Key::R => Some(KeyCode::R),
        egui::Key::S => Some(KeyCode::S),
        egui::Key::W => Some(KeyCode::W),
        egui::Key::X => Some(KeyCode::X),
        egui::Key::Y => Some(KeyCode::Y),
        egui::Key::Z => Some(KeyCode::Z),
        egui::Key::Tab => Some(KeyCode::Tab),
        egui::Key::Enter => Some(KeyCode::Enter),
        egui::Key::Escape => Some(KeyCode::Escape),
        egui::Key::Backspace => Some(KeyCode::Backspace),
        egui::Key::Backtick => Some(KeyCode::Backtick),
        egui::Key::Comma => Some(KeyCode::Comma),
        egui::Key::Period => Some(KeyCode::Period),
        egui::Key::OpenBracket => Some(KeyCode::LeftBracket),
        egui::Key::CloseBracket => Some(KeyCode::RightBracket),
        egui::Key::Slash => Some(KeyCode::Slash),
        _ => None,
    }
}
