//! Split viewport: viewport-lib on the left, Bevy's own 3D renderer on the right.
//!
//! A boid swarm is simulated once in Bevy's ECS. The left half is viewport-lib
//! rendering the boids into a Bevy GPU texture on Bevy's own wgpu device (no CPU
//! copy), shown through a `bevy_ui` `ImageNode`. The right half is a normal Bevy
//! 3D camera drawing the *same* boids as PBR spheres. Both views share one
//! camera and the same background colour, so it reads as a side-by-side of the
//! two renderers on identical data.
//!
//! Bevy 0.19 pins wgpu 29.0.3 and viewport-lib's `wgpu29` feature compiles
//! against the same crate, so `wgpu::Device` / `Queue` / `TextureView` are the
//! same types on both sides and unify with no conversion.
//!
//! Build (heavy first compile: Bevy default features):
//!   cargo run --release --example bevy-swarm --no-default-features --features wgpu29
//!
//! Controls:
//!   Left / Middle drag : orbit        (both views move together)
//!   Right drag         : pan
//!   Scroll             : zoom
//!   Left click a boid  : inspect it (position / velocity in the panel)

use bevy::camera::{ClearColorConfig, Viewport};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::math::Vec3 as BVec3;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::TextureFormat;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::ui::IsDefaultUiCamera;
use bevy::window::PrimaryWindow;

use glam::{Mat4, Vec2, Vec3};
use viewport_lib::{
    ButtonState, Camera as VplCamera, CameraFrame, FrameData, LightingSettings, Material, MeshId,
    OrbitCameraController, PostProcessSettings, SceneFrame, SceneRenderItem, ScrollUnits,
    ViewportContext, ViewportEvent, ViewportRenderer, primitives,
};

// Flocking / world tuning.
const NUM_BOIDS: usize = 600;
const NEIGHBOR_RADIUS: f32 = 6.0;
const SEPARATION_RADIUS: f32 = 2.5;
const SEPARATION_WEIGHT: f32 = 8.0;
const ALIGNMENT_WEIGHT: f32 = 3.5;
const COHESION_WEIGHT: f32 = 2.5;
const BOUND_RADIUS: f32 = 22.0;
const BOUND_WEIGHT: f32 = 10.0;
const MIN_SPEED: f32 = 4.0;
const MAX_SPEED: f32 = 13.0;
const SPHERE_RADIUS: f32 = 0.45;
const PICK_RADIUS: f32 = 0.75;
const SELECTED_SCALE: f32 = 1.7;
const FOV_Y: f32 = 0.85;

/// Shared background, in sRGB. Bevy clears to this directly; viewport-lib's
/// background is the linear form of it (its tone-map writes the background to an
/// sRGB target), so both halves show the same colour.
const BG_SRGB: [f32; 3] = [0.11, 0.12, 0.14];

// The texture viewport-lib renders into. sRGB matches the winit/eframe examples:
// viewport-lib tone-maps to display-ready values and the hardware handles the
// encode/decode roundtrip when bevy_ui samples it.
const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "viewport-lib (left) vs Bevy 3D (right) : boid swarm".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(VplSwarmPlugin)
        .run();
}

// ---------------------------------------------------------------------------
// ECS data
// ---------------------------------------------------------------------------

#[derive(Component)]
struct Boid {
    pos: Vec3,
    vel: Vec3,
    color: [f32; 3],
}

/// Marker for the left-half Bevy 3D camera.
#[derive(Component)]
struct MainCam3d;

/// Orbit camera state, driven from Bevy input. Shared by both views.
#[derive(Resource)]
struct CamState {
    camera: VplCamera,
    controller: OrbitCameraController,
}

/// Currently inspected boid, if any.
#[derive(Resource, Default)]
struct Selected(Option<Entity>);

/// One instance handed to viewport-lib for rendering.
#[derive(Clone)]
struct Instance {
    model: [[f32; 4]; 4],
    color: [f32; 3],
    selected: bool,
}

/// Everything the render world needs to draw a frame. Cloned into the render
/// world each frame by `ExtractResourcePlugin`.
#[derive(Resource, Clone, ExtractResource)]
struct SwarmRenderData {
    target: Handle<Image>,
    width: u32,
    height: u32,
    camera: VplCamera,
    instances: Vec<Instance>,
    /// Bumped every frame. viewport-lib re-uploads the instance buffer only when
    /// this changes, so a static value freezes the swarm.
    generation: u64,
}

/// Marker for the info-panel text entity.
#[derive(Component)]
struct InfoText;

// ---------------------------------------------------------------------------
// Plugin wiring
// ---------------------------------------------------------------------------

struct VplSwarmPlugin;

impl Plugin for VplSwarmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Selected>()
            .add_plugins(ExtractResourcePlugin::<SwarmRenderData>::default())
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    sync_layout,
                    simulate_boids,
                    camera_input,
                    sync_camera3d,
                    sync_boid_transforms,
                    pick_boid,
                    collect_render_data,
                    update_info_panel,
                )
                    .chain(),
            );

        // The render sub-app holds the viewport-lib renderer (it needs Bevy's
        // wgpu device) and submits one command buffer per frame.
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<VplRenderer>()
            .add_systems(Render, render_swarm.in_set(RenderSystems::Render));
    }
}

// ---------------------------------------------------------------------------
// Main-world setup and simulation
// ---------------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let (w, h) = windows
        .single()
        .map(|win| (win.physical_width().max(2), win.physical_height().max(1)))
        .unwrap_or((1280, 720));
    let half_w = (w / 2).max(1);

    // Match both halves to the same background colour.
    commands.insert_resource(ClearColor(Color::srgb(BG_SRGB[0], BG_SRGB[1], BG_SRGB[2])));

    // The shared render target for the left half. viewport-lib renders into
    // this; bevy_ui shows it.
    let target = images.add(Image::new_target_texture(half_w, h, TARGET_FORMAT, None));

    // One sphere mesh shared by every Bevy-side boid.
    let sphere_mesh = meshes.add(Sphere::new(SPHERE_RADIUS).mesh().uv(16, 10));

    // Spawn the swarm. Each boid is a Bevy entity with a PBR sphere; the same
    // entities feed viewport-lib. Colour is fixed per boid so both renderers
    // draw identical colours.
    for i in 0..NUM_BOIDS {
        let idx = i as u32;
        let pos = Vec3::new(
            (hash01(idx * 3) - 0.5) * 28.0,
            (hash01(idx * 3 + 1) - 0.5) * 28.0,
            (hash01(idx * 3 + 2) - 0.5) * 28.0,
        );
        let dir = Vec3::new(
            hash01(idx * 5 + 11) - 0.5,
            hash01(idx * 5 + 12) - 0.5,
            hash01(idx * 5 + 13) - 0.5,
        )
        .normalize_or_zero();
        let color = [
            0.30 + 0.70 * hash01(idx * 9 + 1),
            0.30 + 0.70 * hash01(idx * 9 + 2),
            0.40 + 0.60 * hash01(idx * 9 + 3),
        ];
        let material = materials.add(StandardMaterial {
            base_color: Color::srgb(color[0], color[1], color[2]),
            perceptual_roughness: 0.6,
            ..default()
        });
        commands.spawn((
            Boid {
                pos,
                vel: dir * (MIN_SPEED + hash01(idx * 7 + 3) * (MAX_SPEED - MIN_SPEED)),
                color,
            },
            Mesh3d(sphere_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(BVec3::from_array(pos.to_array())),
        ));
    }

    // Lighting for the Bevy 3D view: a key light and a dimmer fill from the
    // opposite side so boids read from every angle without shadow setup.
    commands.spawn((
        DirectionalLight {
            illuminance: 6000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_translation(BVec3::new(25.0, 20.0, 40.0)).looking_at(BVec3::ZERO, BVec3::Z),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 2500.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_translation(BVec3::new(-30.0, -20.0, -10.0))
            .looking_at(BVec3::ZERO, BVec3::Z),
    ));

    // Orbit camera: Z-up, looking at the swarm centre from a 3/4 angle. Shared
    // by both views.
    let mut camera = VplCamera {
        distance: 55.0,
        ..VplCamera::default()
    };
    camera.set_center(Vec3::ZERO);
    camera.set_fov_y(FOV_Y);
    camera.orbit(0.7, 0.45);

    let mut controller = OrbitCameraController::viewport_primitives();
    controller.begin_frame(ViewportContext {
        hovered: true,
        focused: true,
        viewport_size: [half_w as f32, h as f32],
    });

    commands.insert_resource(CamState { camera, controller });
    commands.insert_resource(SwarmRenderData {
        target: target.clone(),
        width: half_w,
        height: h,
        camera: VplCamera::default(),
        instances: Vec::new(),
        generation: 0,
    });

    // Right half: a normal Bevy 3D camera restricted to the right viewport.
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            viewport: Some(Viewport {
                physical_position: UVec2::new(half_w, 0),
                physical_size: UVec2::new((w - half_w).max(1), h),
                depth: 0.0..1.0,
            }),
            ..default()
        },
        Projection::from(PerspectiveProjection {
            fov: FOV_Y,
            ..default()
        }),
        Transform::from_translation(BVec3::new(0.0, -55.0, 20.0)).looking_at(BVec3::ZERO, BVec3::Z),
        AmbientLight {
            brightness: 180.0,
            ..default()
        },
        MainCam3d,
    ));

    // A 2D camera for the UI overlay. It must not clear, or it would wipe the
    // 3D view. It owns the UI, so mark it the default UI camera.
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        IsDefaultUiCamera,
    ));

    // Left half: the viewport-lib render target.
    commands.spawn((
        ImageNode::new(target),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(0.0),
            width: Val::Percent(50.0),
            height: Val::Percent(100.0),
            ..default()
        },
    ));

    // A title at the top of each half.
    commands.spawn((
        Text::new("VPL"),
        TextFont {
            font_size: FontSize::Px(20.0),
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.85, 0.55)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(14.0),
            ..default()
        },
    ));
    commands.spawn((
        Text::new("Bevy 3D"),
        TextFont {
            font_size: FontSize::Px(18.0),
            ..default()
        },
        TextColor(Color::srgb(0.75, 0.85, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            right: Val::Px(14.0),
            ..default()
        },
    ));

    // Info panel, bottom-left.
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(Color::srgb(0.85, 0.92, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(14.0),
            ..default()
        },
        InfoText,
    ));
}

/// Single source of truth for the split. Both renderers must use the exact same
/// half dimensions or they frame the swarm differently. Every frame this sets
/// the viewport-lib camera aspect and the Bevy 3D viewport from one window read,
/// and recreates the render target only when the size actually changes.
fn sync_layout(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut data: ResMut<SwarmRenderData>,
    mut cam: ResMut<CamState>,
    mut cam3d: Query<&mut Camera, With<MainCam3d>>,
) {
    let Ok(win) = windows.single() else {
        return;
    };
    let w = win.physical_width().max(2);
    let h = win.physical_height().max(1);
    let half_w = (w / 2).max(1);

    // Recreate the render target only on a real size change (it is expensive).
    if half_w != data.width || h != data.height {
        let _ = images.insert(
            data.target.id(),
            Image::new_target_texture(half_w, h, TARGET_FORMAT, None),
        );
        data.width = half_w;
        data.height = h;
    }

    // Every frame: keep the viewport-lib camera aspect and the Bevy viewport on
    // the identical half dimensions so both halves match.
    cam.camera.set_aspect_ratio(half_w as f32, h as f32);
    if let Ok(mut c) = cam3d.single_mut()
        && let Some(vp) = c.viewport.as_mut()
    {
        vp.physical_position = UVec2::new(half_w, 0);
        vp.physical_size = UVec2::new(half_w, h);
    }
}

fn simulate_boids(time: Res<Time>, mut boids: Query<&mut Boid>) {
    let dt = time.delta_secs().min(0.05);

    // Snapshot start-of-frame states so every boid sees the same neighborhood.
    let snapshot: Vec<(Vec3, Vec3)> = boids.iter().map(|b| (b.pos, b.vel)).collect();

    for (i, mut boid) in boids.iter_mut().enumerate() {
        let mut separation = Vec3::ZERO;
        let mut alignment = Vec3::ZERO;
        let mut cohesion = Vec3::ZERO;
        let mut count = 0u32;

        for (j, (other_pos, other_vel)) in snapshot.iter().enumerate() {
            if i == j {
                continue;
            }
            let offset = boid.pos - *other_pos;
            let dist = offset.length();
            if dist < NEIGHBOR_RADIUS && dist > 1e-4 {
                if dist < SEPARATION_RADIUS {
                    separation += offset / dist;
                }
                alignment += *other_vel;
                cohesion += *other_pos;
                count += 1;
            }
        }

        if count > 0 {
            let n = count as f32;
            let align = alignment / n - boid.vel;
            let cohere = cohesion / n - boid.pos;
            boid.vel += (separation * SEPARATION_WEIGHT
                + align * ALIGNMENT_WEIGHT
                + cohere * COHESION_WEIGHT)
                * dt;
        }

        // Steer back toward the centre when leaving the sphere of interest.
        let radius = boid.pos.length();
        if radius > BOUND_RADIUS {
            let pull = -boid.pos / radius;
            boid.vel += pull * BOUND_WEIGHT * dt;
        }

        // Clamp to a sensible speed band.
        let speed = boid.vel.length();
        if speed > MAX_SPEED {
            boid.vel = boid.vel / speed * MAX_SPEED;
        } else if speed < MIN_SPEED && speed > 1e-4 {
            boid.vel = boid.vel / speed * MIN_SPEED;
        }

        let vel = boid.vel;
        boid.pos += vel * dt;
    }
}

fn camera_input(
    mut cam: ResMut<CamState>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut wheel: MessageReader<MouseWheel>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let Ok(win) = windows.single() else {
        return;
    };
    let (vw, h) = (
        (win.physical_width() / 2).max(1),
        win.physical_height().max(1),
    );

    let cam = &mut *cam;
    cam.controller.begin_frame(ViewportContext {
        hovered: true,
        focused: true,
        viewport_size: [vw as f32, h as f32],
    });

    for (bevy_btn, vpl_btn) in [
        (MouseButton::Left, viewport_lib::MouseButton::Left),
        (MouseButton::Middle, viewport_lib::MouseButton::Middle),
        (MouseButton::Right, viewport_lib::MouseButton::Right),
    ] {
        if buttons.just_pressed(bevy_btn) {
            cam.controller.push_event(ViewportEvent::MouseButton {
                button: vpl_btn,
                state: ButtonState::Pressed,
            });
        }
        if buttons.just_released(bevy_btn) {
            cam.controller.push_event(ViewportEvent::MouseButton {
                button: vpl_btn,
                state: ButtonState::Released,
            });
        }
    }

    if let Some(p) = win.cursor_position() {
        cam.controller.push_event(ViewportEvent::PointerMoved {
            position: Vec2::from_array(p.to_array()),
        });
    } else {
        cam.controller.push_event(ViewportEvent::PointerLeft);
    }

    for ev in wheel.read() {
        let units = match ev.unit {
            MouseScrollUnit::Line => ScrollUnits::Lines,
            MouseScrollUnit::Pixel => ScrollUnits::Pixels,
        };
        cam.controller.push_event(ViewportEvent::Wheel {
            delta: Vec2::new(ev.x, ev.y),
            units,
        });
    }

    // Aspect is set in sync_layout so it always matches the Bevy viewport.
    let _ = cam.controller.apply_to_camera(&mut cam.camera);
}

/// Drive the Bevy 3D camera from the shared orbit camera so both views match.
fn sync_camera3d(cam: Res<CamState>, mut q: Query<&mut Transform, With<MainCam3d>>) {
    let Ok(mut transform) = q.single_mut() else {
        return;
    };
    let eye = BVec3::from_array(cam.camera.eye_position().to_array());
    let center = BVec3::from_array(cam.camera.center().to_array());
    // Use viewport-lib's actual camera up (orientation * Y), not world Z, so the
    // orbit orientation matches exactly.
    let up = BVec3::from_array(cam.camera.up().to_array());
    *transform = Transform::from_translation(eye).looking_at(center, up);
}

/// Push simulated positions into the Bevy transforms used by the 3D view.
fn sync_boid_transforms(selected: Res<Selected>, mut q: Query<(Entity, &Boid, &mut Transform)>) {
    for (entity, boid, mut transform) in &mut q {
        transform.translation = BVec3::from_array(boid.pos.to_array());
        let scale = if selected.0 == Some(entity) {
            SELECTED_SCALE
        } else {
            1.0
        };
        transform.scale = BVec3::splat(scale);
    }
}

fn pick_boid(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cam: Res<CamState>,
    boids: Query<(Entity, &Boid)>,
    mut selected: ResMut<Selected>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok(win) = windows.single() else {
        return;
    };
    let Some(cursor) = win.cursor_position() else {
        return;
    };
    let (w, h) = (win.width(), win.height());
    let half_w = w * 0.5;

    // Both halves show the same camera, so map the cursor into whichever half it
    // is over and build a ray from there.
    let local_x = if cursor.x >= half_w {
        (cursor.x - half_w) / half_w
    } else {
        cursor.x / half_w
    };
    let ndc = Vec2::new(local_x * 2.0 - 1.0, 1.0 - cursor.y / h * 2.0);
    let inv = cam.camera.view_proj_matrix().inverse();
    let origin = cam.camera.eye_position();
    let far = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
    let dir = (far - origin).normalize_or_zero();

    let mut best: Option<(Entity, f32)> = None;
    for (entity, boid) in &boids {
        let oc = boid.pos - origin;
        let t = oc.dot(dir);
        if t < 0.0 {
            continue;
        }
        let closest = origin + dir * t;
        if (closest - boid.pos).length_squared() < PICK_RADIUS * PICK_RADIUS
            && best.map(|(_, bt)| t < bt).unwrap_or(true)
        {
            best = Some((entity, t));
        }
    }
    selected.0 = best.map(|(e, _)| e);
}

fn collect_render_data(
    boids: Query<(Entity, &Boid)>,
    cam: Res<CamState>,
    selected: Res<Selected>,
    mut data: ResMut<SwarmRenderData>,
) {
    data.camera = cam.camera.clone();
    // The swarm moves every frame, so bump the generation so viewport-lib
    // re-uploads the instance buffer instead of reusing the cached one.
    data.generation = data.generation.wrapping_add(1);
    data.instances.clear();
    for (entity, boid) in &boids {
        let is_selected = selected.0 == Some(entity);
        let scale = if is_selected { SELECTED_SCALE } else { 1.0 };
        let model = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            glam::Quat::IDENTITY,
            boid.pos,
        )
        .to_cols_array_2d();
        data.instances.push(Instance {
            model,
            color: boid.color,
            selected: is_selected,
        });
    }
}

fn update_info_panel(
    selected: Res<Selected>,
    boids: Query<&Boid>,
    mut text: Query<&mut Text, With<InfoText>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let body = match selected.0.and_then(|e| boids.get(e).ok()) {
        Some(boid) => format!(
            "boid selected\n  pos  {:6.1} {:6.1} {:6.1}\n  vel  {:6.1} {:6.1} {:6.1}\n  speed {:5.1}",
            boid.pos.x,
            boid.pos.y,
            boid.pos.z,
            boid.vel.x,
            boid.vel.y,
            boid.vel.z,
            boid.vel.length(),
        ),
        None => "click a boid to inspect it".to_string(),
    };
    *text = Text::new(format!(
        "{NUM_BOIDS} boids  .  same scene, two renderers\ndrag orbit  .  scroll zoom  .  click to pick\n\n{body}"
    ));
}

// ---------------------------------------------------------------------------
// Render world: viewport-lib draws into Bevy's texture on Bevy's device
// ---------------------------------------------------------------------------

/// Holds the viewport-lib renderer in the render world. Built lazily on the
/// first frame, once Bevy's `RenderDevice` is available.
#[derive(Resource, Default)]
struct VplRenderer {
    inner: Option<ViewportRenderer>,
    sphere: Option<MeshId>,
}

fn render_swarm(
    data: Option<Res<SwarmRenderData>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut renderer: ResMut<VplRenderer>,
) {
    let Some(data) = data else {
        return;
    };
    let Some(gpu_image) = gpu_images.get(&data.target) else {
        return;
    };
    let wgpu_device = device.wgpu_device();

    if renderer.inner.is_none() {
        let mut r = ViewportRenderer::new(wgpu_device, TARGET_FORMAT);
        let sphere = r
            .resources_mut()
            .upload_mesh_data(wgpu_device, &primitives::sphere(SPHERE_RADIUS, 16, 10))
            .expect("upload sphere mesh");
        renderer.inner = Some(r);
        renderer.sphere = Some(sphere);
    }
    let sphere = renderer.sphere.expect("sphere mesh id");

    let items: Vec<SceneRenderItem> = data
        .instances
        .iter()
        .map(|inst| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = sphere;
            item.model = inst.model;
            item.material = Material::from_colour(inst.color);
            if inst.selected {
                item.material.emissive = [1.6, 1.1, 0.25];
            }
            item
        })
        .collect();

    let (w, h) = (data.width as f32, data.height as f32);
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&data.camera, [w, h]),
        SceneFrame::from_surface_items(items).with_generation(data.generation),
    );
    frame.viewport.background_colour = Some([
        srgb_to_linear(BG_SRGB[0]),
        srgb_to_linear(BG_SRGB[1]),
        srgb_to_linear(BG_SRGB[2]),
        1.0,
    ]);
    frame.effects.lighting = LightingSettings::default();
    let mut post = PostProcessSettings::default();
    post.enabled = true;
    post.bloom = true;
    post.bloom_threshold = 1.0;
    post.bloom_intensity = 0.2;
    frame.effects.post_process = post;

    // The full HDR pipeline into Bevy's texture, then submit on Bevy's queue.
    let inner = renderer.inner.as_mut().expect("renderer");
    let cmd = inner
        .owned()
        .render(wgpu_device, &queue, &gpu_image.texture_view, &frame);
    queue.submit(std::iter::once(cmd));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert one sRGB channel to linear, matching the encode wgpu applies when
/// writing to an sRGB target.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Cheap integer hash to a float in [0, 1); avoids pulling in an RNG crate.
fn hash01(mut x: u32) -> f32 {
    x = x.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    let out = (word >> 22) ^ word;
    (out & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}
