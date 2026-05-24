//! Showcase 44: Runtime Interaction
//!
//! Demonstrates SelectionSystem and ManipulationSystem built into ViewportRuntime.
//! Nine boxes are arranged in a 3x3 grid. Click to select, shift-click to multi-select.
//! G / R / S start move / rotate / scale sessions. X / Y / Z constrain the axis.
//! Enter or click confirms; Esc cancels.
//!
//! The runtime owns SelectionSystem and ManipulationSystem; the app just fills in
//! RuntimeFrameContext each frame and reads back RuntimeOutput.

use std::collections::HashMap;

use eframe::egui;
use viewport_lib::{
    ActionFrame, Aabb, GizmoAxis, GizmoMode, Material, MeshId, PickAccelerator,
    RuntimeFrameContext, ViewportRuntime,
    scene::Scene,
    selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct RtInteractState {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub runtime: ViewportRuntime,
    pub pick_acc: Option<PickAccelerator>,
    pub mesh_id: Option<MeshId>,
    /// AABB of the box mesh in local space, cached so the BVH can be rebuilt
    /// after manipulation without needing access to the renderer.
    pub mesh_aabb: Option<Aabb>,
    /// Last known cursor position in viewport-local pixels.
    pub last_cursor: glam::Vec2,
    /// True while the primary button is held.
    pub left_held: bool,
    /// Tracks manipulation active state from the previous frame to detect commit.
    was_manipulating: bool,
}

impl Default for RtInteractState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            runtime: ViewportRuntime::new()
                .with_selection_system()
                .with_manipulation_system(),
            pick_acc: None,
            mesh_id: None,
            mesh_aabb: None,
            last_cursor: glam::Vec2::ZERO,
            left_held: false,
            was_manipulating: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene construction
// ---------------------------------------------------------------------------

pub(crate) fn build_rt_interact_scene(
    app: &mut App,
    renderer: &mut viewport_lib::ViewportRenderer,
) {
    let box_mesh = viewport_lib::primitives::cube(1.0);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &box_mesh)
        .expect("rt interact box upload");
    app.rt_interact_state.mesh_id = Some(mesh_id);

    // 3x3 grid of boxes in the XY plane.
    let colours: [[f32; 3]; 9] = [
        [0.85, 0.35, 0.35],
        [0.35, 0.75, 0.35],
        [0.35, 0.35, 0.85],
        [0.75, 0.75, 0.35],
        [0.35, 0.75, 0.75],
        [0.75, 0.35, 0.75],
        [0.85, 0.55, 0.35],
        [0.55, 0.85, 0.55],
        [0.65, 0.65, 0.85],
    ];

    app.rt_interact_state.scene = Scene::new();
    let scene = &mut app.rt_interact_state.scene;

    for (i, colour) in colours.iter().enumerate() {
        let col = (i % 3) as f32;
        let row = (i / 3) as f32;
        let pos = glam::Vec3::new((col - 1.0) * 2.5, (row - 1.0) * 2.5, 0.0);
        let transform = glam::Mat4::from_translation(pos);
        let mat = Material::from_colour(*colour);
        scene.add(Some(mesh_id), transform, mat);
    }

    // Build pick accelerator.
    let mesh_aabb = renderer
        .resources()
        .mesh(mesh_id)
        .map(|m| m.aabb);
    app.rt_interact_state.mesh_aabb = mesh_aabb;
    app.rt_interact_state.pick_acc =
        Some(PickAccelerator::build_from_scene(scene, |mid| {
            if mid == mesh_id { mesh_aabb } else { None }
        }));

    app.rt_interact_state.built = true;
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

/// Call once per frame while ShowcaseMode::SceneRuntimeInteract is active.
///
/// `clicked` is true on the frame a primary click was detected without drag.
/// `drag_started` is true on the frame a drag began.
/// `dragging` is true while the primary button is held.
pub(crate) fn update_rt_interact(
    app: &mut App,
    dt: f32,
    cursor: glam::Vec2,
    viewport_size: glam::Vec2,
    action_frame: &ActionFrame,
    clicked: bool,
    drag_started: bool,
    dragging: bool,
    pointer_delta: glam::Vec2,
    shift_held: bool,
) {
    app.rt_interact_state.last_cursor = cursor;

    // Run CPU pick on click frames so SelectionSystem can use the result.
    let pick_hit = if clicked {
        if let Some(ref mut acc) = app.rt_interact_state.pick_acc {
            let w = viewport_size.x.max(1.0);
            let h = viewport_size.y.max(1.0);
            let vp_inv = app.camera.view_proj_matrix().inverse();
            let (ray_origin, ray_dir) = viewport_lib::picking::screen_to_ray(
                cursor,
                glam::Vec2::new(w, h),
                vp_inv,
            );
            let mut mesh_lookup: HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)> = HashMap::new();
            if let Some(mid) = app.rt_interact_state.mesh_id {
                mesh_lookup.insert(
                    mid.index() as u64,
                    (
                        app.box_mesh_data.positions.clone(),
                        app.box_mesh_data.indices.clone(),
                    ),
                );
            }
            viewport_lib::bvh::pick_scene_accelerated_cpu(ray_origin, ray_dir, acc, &mesh_lookup)
        } else {
            None
        }
    } else {
        None
    };

    let camera = app.camera.clone();
    let mut frame_ctx = RuntimeFrameContext::default();
    frame_ctx.dt = dt;
    frame_ctx.camera = camera.clone();
    frame_ctx.viewport_size = viewport_size;
    frame_ctx.input = action_frame.clone();
    frame_ctx.pick_hit = pick_hit;
    frame_ctx.clicked = clicked;
    frame_ctx.drag_started = drag_started;
    frame_ctx.dragging = dragging;
    frame_ctx.pointer_delta = pointer_delta;
    frame_ctx.cursor_viewport = Some(cursor);
    frame_ctx.shift_held = shift_held;

    let was = app.rt_interact_state.was_manipulating;
    app.rt_interact_state.runtime.step(
        &mut app.rt_interact_state.scene,
        &mut app.rt_interact_state.selection,
        &frame_ctx,
    );
    let now = app.rt_interact_state.runtime.is_manipulating();
    app.rt_interact_state.was_manipulating = now;

    // Rebuild the pick BVH when a manipulation session ends so transforms
    // are reflected in subsequent picks.
    if was && !now {
        if let Some(mid) = app.rt_interact_state.mesh_id {
            let mesh_aabb = app.rt_interact_state.mesh_aabb;
            let scene = &app.rt_interact_state.scene;
            app.rt_interact_state.pick_acc =
                Some(PickAccelerator::build_from_scene(scene, |m| {
                    if m == mid { mesh_aabb } else { None }
                }));
        }
    }
}

// ---------------------------------------------------------------------------
// Scene items
// ---------------------------------------------------------------------------

pub(crate) fn rt_interact_scene_items(app: &mut App) -> Vec<viewport_lib::SceneRenderItem> {
    app.rt_interact_state
        .scene
        .collect_render_items(&app.rt_interact_state.selection)
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_rt_interact(app: &mut App, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label("Click to select. Shift-click to multi-select.");
        ui.label("G move   R rotate   S scale");
        ui.label("X / Y / Z : constrain axis");
        ui.label("Enter or click : confirm   Esc : cancel");
        ui.add_space(6.0);

        ui.separator();

        // Gizmo mode readout.
        if let Some(ms) = app.rt_interact_state.runtime.manipulation_system() {
            ui.label(format!(
                "Gizmo mode : {:?}",
                ms.gizmo_mode()
            ));
            ui.label(format!(
                "Manipulating : {}",
                ms.is_active()
            ));
        }

        ui.add_space(4.0);

        ui.label(format!(
            "Step index : {}",
            app.rt_interact_state.runtime.step_index()
        ));

        ui.add_space(4.0);
        ui.separator();
        ui.label("What this shows:");
        ui.label("- SelectionSystem: click-to-select built into the runtime.");
        ui.label("- ManipulationSystem: G/R/S sessions driven by ActionFrame.");
        ui.label("- TransformWriteback: scene transform updates via the runtime.");
    });
}

// ---------------------------------------------------------------------------
// Gizmo model for InteractionFrame
// ---------------------------------------------------------------------------

/// Returns the gizmo model matrix, mode, and hovered axis for InteractionFrame,
/// reading from the built-in ManipulationSystem if available.
pub(crate) fn rt_interact_gizmo(
    app: &App,
) -> (Option<glam::Mat4>, GizmoMode, glam::Quat, GizmoAxis) {
    if let Some(ms) = app.rt_interact_state.runtime.manipulation_system() {
        let model = ms.gizmo_center().map(|c| {
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(ms.gizmo_scale()),
                glam::Quat::IDENTITY,
                c,
            )
        });
        (model, ms.gizmo_mode(), glam::Quat::IDENTITY, ms.gizmo_hovered())
    } else {
        (None, GizmoMode::Translate, glam::Quat::IDENTITY, GizmoAxis::None)
    }
}
