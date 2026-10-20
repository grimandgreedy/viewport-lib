//! Shared helpers used across multiple showcase modules.

use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{Camera, KeyCode, Material, MaterialId, MeshId, ViewportRenderer};

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

/// Upload a material and return its id. Call once at build time and keep the id.
pub(crate) fn upload_material(renderer: &mut ViewportRenderer, material: Material) -> MaterialId {
    renderer
        .resources_mut()
        .upload_material(material)
        .expect("material upload")
}

/// Edit an uploaded material in place from a controls panel. `edit` gets the
/// current values; if it changes them, every item holding `id` draws the new
/// values next frame.
pub(crate) fn edit_material(
    frame: &crate::eframe::Frame,
    id: MaterialId,
    edit: impl FnOnce(&mut Material),
) {
    // A showcase holds the default id until it is built; never edit the
    // default material through it.
    if id == MaterialId::DEFAULT {
        return;
    }
    let Some(rs) = frame.wgpu_render_state() else {
        return;
    };
    let mut guard = rs.renderer.write();
    let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() else {
        return;
    };
    let resources = renderer.resources_mut();
    let Some(old) = resources.material(id).copied() else {
        return;
    };
    let mut mat = old;
    edit(&mut mat);
    if mat != old {
        resources
            .replace_material(id, mat)
            .expect("material id is live");
    }
}

// ---------------------------------------------------------------------------
// Per-mode opening camera
// ---------------------------------------------------------------------------

/// Returns a camera to apply when switching to `mode`, if the mode needs a specific starting view.
/// Returns `None` for modes that should keep the current camera.
pub(crate) fn opening_camera(mode: ShowcaseMode) -> Option<Camera> {
    match mode {
        // Eye at +Y looking along -Y, Z up on screen.
        ShowcaseMode::PickLevels => Some(Camera {
            center: glam::Vec3::ZERO,
            distance: 14.0,
            orientation: glam::Quat::from_axis_angle(
                glam::Vec3::new(0.0, 1.0, 1.0).normalize(),
                std::f32::consts::PI,
            ),
            ..Camera::default()
        }),
        // Z-up scene: wall front at y=0 (+Y normal), ground top at z=0 (+Z normal).
        // 180-degree rotation around (0,1,1)/sqrt(2): maps orbit-Z -> +Y (eye
        // in front of wall) and orbit-Y -> +Z (Z appears as screen-up).
        ShowcaseMode::Decals => Some(Camera {
            center: glam::Vec3::new(0.0, 0.0, 1.5),
            distance: 10.0,
            orientation: glam::Quat::from_axis_angle(
                glam::Vec3::new(0.0, 1.0, 1.0).normalize(),
                std::f32::consts::PI,
            ),
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
