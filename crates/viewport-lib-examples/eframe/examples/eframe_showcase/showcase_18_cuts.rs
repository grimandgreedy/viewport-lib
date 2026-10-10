//! The Cuts tab of showcase 18 (Clips and Cuts).
//!
//! `viewport_lib_plugins::deformers::cut` removes part of a mesh per item: a
//! plane, a box, a sphere, or a range of a per-vertex scalar. It is a
//! deformer, so the cut holds in every pass: the removed part casts no shadow,
//! gets no selection outline and is not picked.
//!
//! A torus over a floor, with a column standing in its hole. The torus is
//! selected and the light casts shadows, so the cut can be checked against
//! all three. The height range reads the torus's own `height` attribute,
//! with no second upload.

use crate::App;
use crate::eframe::egui;
use viewport_lib as vpl;
use viewport_lib::Colour;
use viewport_lib_plugins::deformers::cut::{Cut, CutDeformer};
use vpl::{
    AttributeKind, AttributeRef, BuiltinColourmap, ColourmapId, LightingSettings, Material, MeshId,
    SceneRenderItem,
};

/// Which shape the torus is cut by.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum CutShape {
    Plane,
    Box,
    Sphere,
    HeightRange,
}

/// The deform instance the torus selects its cut with.
const TORUS_INSTANCE: u32 = 1;

/// Height of the torus centre above the floor.
const TORUS_Z: f32 = 1.2;

pub(crate) struct CutViewsState {
    pub(crate) built: bool,
    /// `None` when the device cannot run deformers.
    cut: Option<CutDeformer>,
    torus: Option<MeshId>,
    column: Option<MeshId>,
    floor: Option<MeshId>,
    shape: CutShape,
    flipped: bool,
    /// Plane offset, box half-size, sphere radius or range top, by shape.
    amount: f32,
    outline: bool,
    dirty: bool,
}

impl Default for CutViewsState {
    fn default() -> Self {
        Self {
            built: false,
            cut: None,
            torus: None,
            column: None,
            floor: None,
            shape: CutShape::Plane,
            flipped: false,
            amount: 0.3,
            outline: true,
            dirty: false,
        }
    }
}

impl CutViewsState {
    fn current_cut(&self) -> Cut {
        let cut = match self.shape {
            CutShape::Plane => Cut::plane([0.6, -0.8, 0.0], self.amount),
            CutShape::Box => {
                let h = 0.4 + self.amount.abs();
                Cut::aabb([-h, -h, TORUS_Z - 2.0], [h, h, TORUS_Z + 2.0])
            }
            CutShape::Sphere => Cut::sphere([1.4, 0.0, TORUS_Z], 0.3 + self.amount.abs()),
            CutShape::HeightRange => Cut::range(-1.0, self.amount),
        };
        if self.flipped { cut.flipped() } else { cut }
    }
}

pub(crate) fn build(app: &mut App, renderer: &mut vpl::ViewportRenderer) {
    let state = &mut app.cut_state;
    let resources = renderer.resources_mut();
    state.cut = CutDeformer::install(resources, &app.device).ok();

    let mut torus = vpl::primitives::torus(1.4, 0.55, 96, 48);
    let heights = torus.positions.iter().map(|p| p[2]).collect();
    torus
        .attributes
        .insert("height".to_string(), vpl::AttributeData::Vertex(heights));
    let torus = resources
        .upload_mesh_data(&app.device, &torus)
        .expect("torus");
    let column = resources
        .upload_mesh_data(&app.device, &vpl::primitives::cylinder(0.35, 2.6, 48))
        .expect("column");
    let floor = resources
        .upload_mesh_data(&app.device, &vpl::primitives::plane(10.0, 10.0))
        .expect("floor");
    if let Some(cut) = state.cut {
        // The range tests the scalar the torus already carries.
        let _ = cut.set_field_from_attribute(resources, &app.device, torus, "height");
    }
    state.torus = Some(torus);
    state.column = Some(column);
    state.floor = Some(floor);
    state.dirty = true;
    state.built = true;

    app.camera = vpl::Camera {
        center: glam::Vec3::new(0.0, 0.0, 0.9),
        distance: 8.5,
        orientation: glam::Quat::from_rotation_z(0.5) * glam::Quat::from_rotation_x(0.95),
        ..vpl::Camera::default()
    };
}

pub(crate) fn scene_items(app: &App) -> Vec<SceneRenderItem> {
    let state = &app.cut_state;
    let (Some(torus), Some(column), Some(floor)) = (state.torus, state.column, state.floor) else {
        return Vec::new();
    };
    let mut items = Vec::new();

    let mut floor_item = SceneRenderItem::default();
    floor_item.mesh_id = floor;
    floor_item.material = Material::from_colour(Colour::linear_rgb(0.75, 0.75, 0.72));
    items.push(floor_item);

    let mut column_item = SceneRenderItem::default();
    column_item.mesh_id = column;
    column_item.model =
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 1.3)).to_cols_array_2d();
    column_item.material = Material::from_colour(Colour::linear_rgb(0.85, 0.55, 0.2));
    items.push(column_item);

    let mut torus_item = SceneRenderItem::default();
    torus_item.mesh_id = torus;
    torus_item.model =
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, TORUS_Z)).to_cols_array_2d();
    torus_item.material = Material::from_colour(Colour::linear_rgb(0.35, 0.5, 0.8));
    torus_item.material.backface_policy = vpl::BackfacePolicy::Identical;
    torus_item.deform_instance = Some(TORUS_INSTANCE);
    torus_item.settings.selected = state.outline;
    if state.shape == CutShape::HeightRange {
        torus_item.active_attribute = Some(AttributeRef {
            name: "height".to_string(),
            kind: AttributeKind::Vertex,
        });
        torus_item.colourmap_id = Some(ColourmapId(BuiltinColourmap::Viridis as usize));
    }
    items.push(torus_item);
    items
}

pub(crate) fn controls(app: &mut App, ui: &mut egui::Ui) {
    let state = &mut app.cut_state;
    if state.cut.is_none() {
        ui.label("This device cannot run deformers, so nothing is cut.");
        return;
    }
    ui.label("Cut the torus by:");
    let before = (state.shape, state.flipped, state.amount);
    ui.horizontal(|ui| {
        ui.radio_value(&mut state.shape, CutShape::Plane, "Plane");
        ui.radio_value(&mut state.shape, CutShape::Box, "Box");
        ui.radio_value(&mut state.shape, CutShape::Sphere, "Sphere");
        ui.radio_value(&mut state.shape, CutShape::HeightRange, "Height");
    });
    let label = match state.shape {
        CutShape::Plane => "Offset",
        CutShape::Box => "Box size",
        CutShape::Sphere => "Radius",
        CutShape::HeightRange => "Highest kept height",
    };
    ui.add(egui::Slider::new(&mut state.amount, -1.0..=1.0).text(label));
    ui.checkbox(&mut state.flipped, "Keep the other side");
    ui.checkbox(&mut state.outline, "Selected");
    if (state.shape, state.flipped, state.amount) != before {
        state.dirty = true;
    }
    ui.separator();
    ui.label(
        "The removed part casts no shadow, gets no outline and cannot be clicked: \
         the column behind it shows through.",
    );
}

pub(crate) fn flush_gpu(app: &mut App, cx: &crate::ViewportCtx) {
    let state = &mut app.cut_state;
    if !state.dirty {
        return;
    }
    state.dirty = false;
    let (Some(cut), Some(torus)) = (state.cut, state.torus) else {
        return;
    };
    let rs = cx.frame.wgpu_render_state().expect("wgpu required");
    let mut guard = rs.renderer.write();
    if let Some(renderer) = guard.callback_resources.get_mut::<vpl::ViewportRenderer>() {
        cut.set(
            renderer.resources_mut(),
            &app.device,
            &app.queue,
            torus,
            TORUS_INSTANCE,
            &[state.current_cut()],
        );
    }
}

/// The lighting for the Cuts tab: the default sun, which casts the shadows the
/// cut has to agree with.
pub(crate) fn lighting() -> LightingSettings {
    let mut lighting = LightingSettings::default();
    lighting.hemisphere_intensity = 0.4;
    lighting
}

/// Per-frame settings for the Cuts tab.
pub(crate) fn frame(app: &mut App, fd: &mut vpl::FrameData) {
    fd.interaction.outline_selected = app.cut_state.outline;
}
