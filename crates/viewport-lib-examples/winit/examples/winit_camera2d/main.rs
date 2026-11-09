//! A 2D view under `Camera2d` with the built-in winit runner.
//!
//! A flat grid of coloured quads on the world XY plane, seen from above through
//! the orthographic camera `Camera2d` produces. `Camera2dController` applies the
//! planar bindings: left or middle drag pans, the wheel zooms about the cursor,
//! and a pinch zooms on touch. A marker circles the grid; `F` toggles following
//! it (with a dead zone, so the camera only moves once the marker nears the
//! edge of the middle of the screen), `Space` shakes the camera, and `P` toggles
//! pixel snapping. The view is kept inside the grid's bounds.
//!
//! The window takes its input through `window_with_input`, so the runner does
//! not drive its orbit camera: the frame callback writes `Camera2d::camera()`
//! into the instance each frame.

use std::cell::Cell;
use std::rc::Rc;

use glam::Vec2;

use viewport_lib as vpl;
use vpl::{
    Action, AppConfig, Camera2d, Camera2dController, Colour, ItemSettings, KeyCode, Material,
    Modifiers, ModifiersMatch, NodeId, Plane2d, Rect2d, ViewportApp, ViewportBinding,
    ViewportGesture, WindowConfig, planar_bindings, primitives,
};

const TOGGLE_FOLLOW: Action = Action::Custom(1);
const SHAKE: Action = Action::Custom(2);
const TOGGLE_SNAP: Action = Action::Custom(3);

/// Quads per side of the grid, one world unit apart.
const GRID: i32 = 24;

fn key(action: Action, key: KeyCode) -> ViewportBinding {
    ViewportBinding::new(
        action,
        ViewportGesture::KeyPress {
            key,
            modifiers: ModifiersMatch::Exact(Modifiers::NONE),
        },
    )
}

fn unlit() -> ItemSettings {
    let mut settings = ItemSettings::default();
    settings.unlit = true;
    settings
}

fn main() {
    let marker_id: Rc<Cell<Option<NodeId>>> = Rc::new(Cell::new(None));
    let setup_id = marker_id.clone();

    let half = GRID as f32 * 0.5;
    let bounds = Rect2d::new(Vec2::splat(-half - 1.0), Vec2::splat(half + 1.0));
    let mut camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(0.02)
        .with_bounds(bounds);
    camera.dead_zone = Rect2d::from_centre_size(Vec2::ZERO, Vec2::new(240.0, 160.0));
    camera.follow_time_constant = 0.3;
    let mut controller = Camera2dController::new_stateless();
    let mut following = true;

    ViewportApp::new(AppConfig::default())
        .window_with_input(
            WindowConfig::default()
                .with_title("viewport-lib : Camera2d")
                .with_window_size(1280, 720),
            move |setup| {
                let device = setup.device().clone();
                let quad = setup
                    .resources_mut()
                    .upload_mesh_data(&device, &primitives::plane(0.9, 0.9))
                    .unwrap();
                let disk = setup
                    .resources_mut()
                    .upload_mesh_data(&device, &primitives::disk(0.35, 32))
                    .unwrap();

                let palette = [
                    Colour::linear_rgb(0.08, 0.25, 0.55),
                    Colour::linear_rgb(0.1, 0.45, 0.3),
                    Colour::linear_rgb(0.5, 0.42, 0.08),
                    Colour::linear_rgb(0.45, 0.12, 0.3),
                ];
                let resources = setup.resources_mut();
                let materials: Vec<_> = palette
                    .iter()
                    .map(|c| {
                        resources
                            .upload_material(Material::from_colour(*c))
                            .unwrap()
                    })
                    .collect();
                let white = resources
                    .upload_material(Material::from_colour(Colour::linear_rgb(0.95, 0.9, 0.85)))
                    .unwrap();

                // Z-up world: the grid lies on the XY plane and the camera looks down -Z.
                let scene = setup.scene_mut();
                for j in 0..GRID {
                    for i in 0..GRID {
                        let x = i as f32 - half + 0.5;
                        let y = j as f32 - half + 0.5;
                        let material = materials[((i / 3 + j / 3) % 4) as usize];
                        let id = scene.add(
                            Some(quad),
                            glam::Mat4::from_translation(glam::Vec3::new(x, y, 0.0)),
                            material,
                        );
                        scene.set_appearance(id, unlit());
                    }
                }
                let marker = scene.add(Some(disk), glam::Mat4::IDENTITY, white);
                scene.set_appearance(marker, unlit());
                setup_id.set(Some(marker));

                // The runner's resolver takes the planar scheme plus three keys.
                let mut bindings = planar_bindings();
                bindings.push(key(TOGGLE_FOLLOW, KeyCode::F));
                bindings.push(key(SHAKE, KeyCode::Space));
                bindings.push(key(TOGGLE_SNAP, KeyCode::P));
                setup.set_bindings(bindings);
            },
            // Everything goes to the viewport: there is no UI over it.
            |input| {
                for event in input.events().to_vec() {
                    input.forward(event);
                }
            },
            move |ctx| {
                let size = ctx.viewport_size();
                camera.set_viewport_size(Vec2::from(size));
                camera.pixels_per_point = ctx.surface_size()[0] as f32 / size[0].max(1.0);

                let frame = ctx.action_frame().clone();
                if frame.is_active(TOGGLE_FOLLOW) {
                    following = !following;
                }
                if frame.is_active(SHAKE) {
                    camera.shake(18.0, 0.45);
                }
                if frame.is_active(TOGGLE_SNAP) {
                    camera.snap_to_pixels = !camera.snap_to_pixels;
                }

                let t = ctx.time() * 0.6;
                let target = Vec2::new(t.cos(), t.sin()) * (half - 3.0);
                if following {
                    camera.follow(target);
                } else {
                    camera.stop_following();
                }
                if let Some(id) = marker_id.get() {
                    let at = glam::Vec3::new(target.x, target.y, 0.05);
                    ctx.scene_mut()
                        .set_local_transform(id, glam::Mat4::from_translation(at));
                }

                // Pan, zoom about the cursor, then follow and shake.
                controller.apply(&mut camera, &frame, ctx.dt());
                *ctx.camera_mut() = camera.camera();
            },
        )
        .run();
}
