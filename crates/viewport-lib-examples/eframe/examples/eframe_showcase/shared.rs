//! Shared helpers used across multiple showcase modules.

use viewport_lib as vpl;
use vpl::{Camera, Material, MaterialId, MeshId, ViewportRenderer};

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

/// `m` with back faces drawn and lit from the viewer's side, for open
/// surfaces (planes, cards) the camera can see from behind or below.
pub(crate) fn two_sided(mut m: Material) -> Material {
    m.backface_policy = vpl::BackfacePolicy::Identical;
    m
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
        ShowcaseMode::PickLevels => Some(Camera::orbit(
            glam::DVec3::ZERO,
            14.0,
            glam::Quat::from_axis_angle(
                glam::Vec3::new(0.0, 1.0, 1.0).normalize(),
                std::f32::consts::PI,
            ),
        )),
        // Z-up scene: wall front at y=0 (+Y normal), ground top at z=0 (+Z normal).
        // 180-degree rotation around (0,1,1)/sqrt(2): maps orbit-Z -> +Y (eye
        // in front of wall) and orbit-Y -> +Z (Z appears as screen-up).
        ShowcaseMode::Decals => Some(Camera::orbit(
            glam::DVec3::new(0.0, 0.0, 1.5),
            10.0,
            glam::Quat::from_axis_angle(
                glam::Vec3::new(0.0, 1.0, 1.0).normalize(),
                std::f32::consts::PI,
            ),
        )),
        _ => None,
    }
}
