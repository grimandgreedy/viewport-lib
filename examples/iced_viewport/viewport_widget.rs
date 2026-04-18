//! Iced shader-widget bridge for `viewport-lib`.
//!
//! `ViewportState` owns the camera and input tracking so `shader::Program::update()`
//! can translate Iced events directly into orbit, pan, and zoom changes.

use std::collections::HashMap;

use iced::event::Event;
use iced::widget::shader;
use iced::{Element, Fill, Point, Rectangle, mouse};
use viewport_lib::{
    Camera, FrameData, LightingSettings, MeshData, RenderCamera, SceneRenderItem,
    SurfaceSubmission, ViewportRenderer,
};

use crate::Message;

// ---------------------------------------------------------------------------
// Camera control constants shared with the other viewport examples.
// ---------------------------------------------------------------------------

const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.001;
const MIN_DISTANCE: f32 = 0.1;

// ---------------------------------------------------------------------------
// Snapshot types (passed from App::view each frame)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SceneSnapshot {
    pub objects: Vec<ObjSnapshot>,
}

#[derive(Debug, Clone)]
pub struct ObjSnapshot {
    pub id: u64,
    pub position: [f32; 3],
}

// ---------------------------------------------------------------------------
// ViewportState - persists across frames, tracks mouse + camera
// ---------------------------------------------------------------------------

/// Iced shader widget state. Tracks mouse button/position state and owns the
/// camera so that input events in `update()` directly drive orbit/pan/zoom.
pub struct ViewportState {
    pub camera: Camera,
    /// Whether the left mouse button is currently pressed.
    left_pressed: bool,
    /// Whether the middle mouse button is currently pressed.
    middle_pressed: bool,
    /// Whether the right mouse button is currently pressed.
    right_pressed: bool,
    /// Whether shift is held (used to switch middle-drag from orbit to pan).
    shift_held: bool,
    /// Last known cursor position (for computing deltas on drag).
    last_pos: Point,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            camera: Camera {
                center: glam::Vec3::ZERO,
                distance: 12.0,
                orientation: glam::Quat::from_rotation_y(0.6) * glam::Quat::from_rotation_x(-0.4),
                ..Camera::default()
            },
            left_pressed: false,
            middle_pressed: false,
            right_pressed: false,
            shift_held: false,
            last_pos: Point::ORIGIN,
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive - carries per-frame scene data + camera snapshot
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ViewportPrimitive {
    objects: Vec<ObjSnapshot>,
    /// Snapshot of the camera at draw time (so prepare/render use consistent state).
    camera_snapshot: CameraSnapshot,
}

#[derive(Debug, Clone)]
struct CameraSnapshot {
    render_camera: RenderCamera,
}

// ---------------------------------------------------------------------------
// Pipeline - wraps ViewportRenderer + depth texture
// ---------------------------------------------------------------------------

pub struct ViewportPipeline {
    renderer: ViewportRenderer,
    /// Track which object ids have been uploaded -> mesh_index.
    uploaded: HashMap<u64, usize>,
    /// Current depth texture + view (recreated on resize).
    depth_view: wgpu::TextureView,
    depth_w: u32,
    depth_h: u32,
    _target_format: wgpu::TextureFormat,
}

impl ViewportPipeline {
    fn ensure_depth(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced_viewport_depth"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

impl shader::Pipeline for ViewportPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let renderer = ViewportRenderer::new(device, format);
        let depth_view =
            Self::ensure_depth(device, 256, 256, wgpu::TextureFormat::Depth24PlusStencil8);

        Self {
            renderer,
            uploaded: HashMap::new(),
            depth_view,
            depth_w: 256,
            depth_h: 256,
            _target_format: format,
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive impl
// ---------------------------------------------------------------------------

impl shader::Primitive for ViewportPrimitive {
    type Pipeline = ViewportPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        let w = viewport.physical_width();
        let h = viewport.physical_height();

        if w != pipeline.depth_w || h != pipeline.depth_h {
            pipeline.depth_view = ViewportPipeline::ensure_depth(
                device,
                w,
                h,
                wgpu::TextureFormat::Depth24PlusStencil8,
            );
            pipeline.depth_w = w;
            pipeline.depth_h = h;
        }

        let box_mesh = unit_box_mesh();
        for obj in &self.objects {
            if !pipeline.uploaded.contains_key(&obj.id) {
                let idx = pipeline
                    .renderer
                    .resources_mut()
                    .upload_mesh_data(device, &box_mesh)
                    .expect("built-in mesh");
                pipeline.uploaded.insert(obj.id, idx);
            }
        }

        let scene_items: Vec<SceneRenderItem> = self
            .objects
            .iter()
            .filter_map(|obj| {
                let mesh_index = *pipeline.uploaded.get(&obj.id)?;
                let model = glam::Mat4::from_translation(glam::Vec3::from(obj.position));
                let mut item = SceneRenderItem::default();
                item.mesh_index = mesh_index;
                item.model = model.to_cols_array_2d();
                Some(item)
            })
            .collect();

        let frame_data = {
            let mut fd = FrameData::default();
            fd.camera.render_camera = self.camera_snapshot.render_camera.clone();
            fd.camera.viewport_size = [bounds.width, bounds.height];
            fd.effects.lighting = LightingSettings::default();
            fd.scene.surfaces = SurfaceSubmission::Flat(scene_items);
            fd.viewport.show_grid = true;
            fd.viewport.grid_y = -0.5; // bottom face of unit boxes
            fd.viewport.show_axes_indicator = true;
            fd
        };

        pipeline.renderer.prepare(device, queue, &frame_data);
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let scene_items: Vec<SceneRenderItem> = self
            .objects
            .iter()
            .filter_map(|obj| {
                let mesh_index = *pipeline.uploaded.get(&obj.id)?;
                let model = glam::Mat4::from_translation(glam::Vec3::from(obj.position));
                let mut item = SceneRenderItem::default();
                item.mesh_index = mesh_index;
                item.model = model.to_cols_array_2d();
                Some(item)
            })
            .collect();

        let frame_data = {
            let mut fd = FrameData::default();
            fd.camera.render_camera = self.camera_snapshot.render_camera.clone();
            fd.camera.viewport_size = [clip_bounds.width as f32, clip_bounds.height as f32];
            fd.effects.lighting = LightingSettings::default();
            fd.scene.surfaces = SurfaceSubmission::Flat(scene_items);
            fd.viewport.show_grid = true;
            fd.viewport.grid_y = -0.5; // bottom face of unit boxes
            fd.viewport.show_axes_indicator = true;
            fd
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iced_viewport_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.12,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &pipeline.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_viewport(
            clip_bounds.x as f32,
            clip_bounds.y as f32,
            clip_bounds.width as f32,
            clip_bounds.height as f32,
            0.0,
            1.0,
        );

        pipeline.renderer.paint_to(&mut render_pass, &frame_data);
    }
}

// ---------------------------------------------------------------------------
// Program impl on SceneSnapshot
// ---------------------------------------------------------------------------

impl shader::Program<Message> for SceneSnapshot {
    type State = ViewportState;
    type Primitive = ViewportPrimitive;

    /// Translate Iced events into camera updates.
    ///
    /// This is the Iced-specific input adapter. Each UI framework needs its own
    /// version of this translation layer:
    ///   - Track which mouse buttons are pressed (for drag detection)
    ///   - Compute cursor deltas between frames
    ///   - Map button+modifier combinations to orbit/pan/zoom actions
    ///   - Apply the resulting deltas to the Camera
    ///
    /// Controls:
    ///   - Left-drag or Middle-drag:  Orbit (rotate camera around center)
    ///   - Right-drag or Shift+Middle-drag:  Pan (translate camera center)
    ///   - Scroll wheel:  Zoom (adjust camera distance)
    fn update(
        &self,
        state: &mut Self::State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<iced::widget::shader::Action<Message>> {
        // Only handle events when cursor is over the viewport.
        let pos = cursor.position_in(bounds)?;

        match event {
            // --- Track modifier keys ---
            Event::Keyboard(iced::keyboard::Event::ModifiersChanged(mods)) => {
                state.shift_held = mods.shift();
                None
            }

            // --- Mouse button press: record position for delta tracking ---
            Event::Mouse(mouse::Event::ButtonPressed(button)) => {
                match button {
                    mouse::Button::Left => state.left_pressed = true,
                    mouse::Button::Middle => state.middle_pressed = true,
                    mouse::Button::Right => state.right_pressed = true,
                    _ => return None,
                }
                state.last_pos = pos;
                Some(iced::widget::shader::Action::request_redraw().and_capture())
            }

            // --- Mouse button release ---
            Event::Mouse(mouse::Event::ButtonReleased(button)) => {
                match button {
                    mouse::Button::Left => state.left_pressed = false,
                    mouse::Button::Middle => state.middle_pressed = false,
                    mouse::Button::Right => state.right_pressed = false,
                    _ => return None,
                }
                None
            }

            // --- Mouse move: apply orbit or pan based on which buttons are held ---
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let dx = pos.x - state.last_pos.x;
                let dy = pos.y - state.last_pos.y;
                state.last_pos = pos;

                let any_drag = state.left_pressed || state.middle_pressed || state.right_pressed;
                if !any_drag || (dx.abs() < 0.001 && dy.abs() < 0.001) {
                    return None;
                }

                // Pan: right-drag, or shift+middle-drag
                let is_pan = state.right_pressed || (state.middle_pressed && state.shift_held);

                if is_pan {
                    let cam = &mut state.camera;
                    let pan_scale = 2.0 * cam.distance * (cam.fov_y / 2.0).tan() / bounds.height;
                    let right = cam.right();
                    let up = cam.up();
                    cam.center -= right * dx * pan_scale;
                    cam.center += up * dy * pan_scale;
                } else {
                    // Orbit: left-drag or middle-drag (without shift)
                    let cam = &mut state.camera;
                    let q_yaw = glam::Quat::from_rotation_y(-dx * ORBIT_SENSITIVITY);
                    let q_pitch = glam::Quat::from_rotation_x(-dy * ORBIT_SENSITIVITY);
                    cam.orientation = (q_yaw * cam.orientation * q_pitch).normalize();
                }

                Some(iced::widget::shader::Action::request_redraw().and_capture())
            }

            // --- Scroll: zoom ---
            Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                let scroll_y = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y * 28.0,
                    mouse::ScrollDelta::Pixels { y, .. } => *y,
                };
                let cam = &mut state.camera;
                cam.distance =
                    (cam.distance * (1.0 - scroll_y * ZOOM_SENSITIVITY)).max(MIN_DISTANCE);
                Some(iced::widget::shader::Action::request_redraw().and_capture())
            }

            _ => None,
        }
    }

    fn draw(
        &self,
        state: &Self::State,
        _cursor: iced::mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        // Snapshot the camera with the current aspect ratio.
        let mut cam = state.camera.clone();
        cam.aspect = if bounds.height > 0.0 {
            bounds.width / bounds.height
        } else {
            1.0
        };

        ViewportPrimitive {
            objects: self.objects.clone(),
            camera_snapshot: CameraSnapshot {
                render_camera: RenderCamera::from_camera(&cam),
            },
        }
    }

    fn mouse_interaction(
        &self,
        state: &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.left_pressed || state.middle_pressed || state.right_pressed {
            mouse::Interaction::Grabbing
        } else if cursor.is_over(bounds) {
            mouse::Interaction::Grab
        } else {
            mouse::Interaction::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Public helper: creates the iced shader widget
// ---------------------------------------------------------------------------

pub fn viewport_shader(scene: SceneSnapshot) -> Element<'static, Message> {
    shader(scene).width(Fill).height(Fill).into()
}

// ---------------------------------------------------------------------------
// Box mesh helper
// ---------------------------------------------------------------------------

fn unit_box_mesh() -> MeshData {
    #[rustfmt::skip]
    let positions: Vec<[f32; 3]> = vec![
        // Front face
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        // Back face
        [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        // Top face
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        // Bottom face
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
        // Right face
        [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
        // Left face
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
    ];

    #[rustfmt::skip]
    let normals: Vec<[f32; 3]> = vec![
        // Front
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        // Back
        [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
        // Top
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        // Bottom
        [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
        // Right
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        // Left
        [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    ];

    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0,  1,  2,  0,  2,  3,   // Front
        4,  5,  6,  4,  6,  7,   // Back
        8,  9,  10, 8,  10, 11,  // Top
        12, 13, 14, 12, 14, 15,  // Bottom
        16, 17, 18, 16, 18, 19,  // Right
        20, 21, 22, 20, 22, 23,  // Left
    ];

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}
