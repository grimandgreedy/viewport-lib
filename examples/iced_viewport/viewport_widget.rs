//! Iced shader-widget bridge for `viewport-lib`.
//!
//! `ViewportState` owns the camera and input tracking so `shader::Program::update()`
//! can translate Iced events directly into orbit, pan, and zoom changes.

use std::collections::HashMap;

use iced::event::Event;
use iced::widget::shader;
use iced::{Element, Fill, Rectangle, mouse};
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, LightingSettings, Modifiers, MouseButton,
    OrbitCameraController, RenderCamera, SceneFrame, SceneRenderItem, ScrollUnits, ViewportContext,
    ViewportEvent, ViewportRenderer, primitives,
};

use crate::Message;

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

/// Iced shader widget state. Owns the camera and controller so that input
/// events in `update()` are forwarded to `OrbitCameraController` via `push_event`.
pub struct ViewportState {
    pub camera: Camera,
    controller: OrbitCameraController,
    /// Track dragging state for cursor interaction display.
    any_pressed: bool,
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
            controller: OrbitCameraController::viewport_primitives(),
            any_pressed: false,
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

        for obj in &self.objects {
            if !pipeline.uploaded.contains_key(&obj.id) {
                let idx = pipeline
                    .renderer
                    .resources_mut()
                    .upload_mesh_data(device, &primitives::cube(1.0))
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

        let mut frame_data = FrameData::new(
            CameraFrame::new(
                self.camera_snapshot.render_camera.clone(),
                [bounds.width, bounds.height],
            ),
            SceneFrame::from_surface_items(scene_items),
        );
        frame_data.effects.lighting = LightingSettings::default();
        frame_data.viewport.show_grid = true;
        frame_data.viewport.grid_z = -0.5; // bottom face of unit boxes
        frame_data.viewport.show_axes_indicator = true;

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

        let mut frame_data = FrameData::new(
            CameraFrame::new(
                self.camera_snapshot.render_camera.clone(),
                [clip_bounds.width as f32, clip_bounds.height as f32],
            ),
            SceneFrame::from_surface_items(scene_items),
        );
        frame_data.effects.lighting = LightingSettings::default();
        frame_data.viewport.show_grid = true;
        frame_data.viewport.grid_z = -0.5; // bottom face of unit boxes
        frame_data.viewport.show_axes_indicator = true;

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

    /// Translate Iced events into `ViewportEvent`s, forward them to
    /// `OrbitCameraController`, and apply the result to the camera immediately.
    ///
    /// Because iced delivers one event per `update()` call (no explicit frame
    /// boundary), we call `begin_frame` + `apply_to_camera` around each event.
    /// `begin_frame` only resets the per-frame drag/wheel accumulators — it
    /// preserves `pointer_pos` and `button_held` — so delta computation remains
    /// correct across consecutive `PointerMoved` events.
    fn update(
        &self,
        state: &mut Self::State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<iced::widget::shader::Action<Message>> {
        let vp_ctx = ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: glam::vec2(bounds.width, bounds.height).into(),
        };

        match event {
            Event::Keyboard(iced::keyboard::Event::ModifiersChanged(mods)) => {
                state.controller.begin_frame(vp_ctx);
                state
                    .controller
                    .push_event(ViewportEvent::ModifiersChanged(if mods.shift() {
                        Modifiers::SHIFT
                    } else {
                        Modifiers::NONE
                    }));
                state.controller.apply_to_camera(&mut state.camera);
                None
            }

            Event::Mouse(mouse::Event::ButtonPressed(button)) => {
                let pos = cursor.position_in(bounds)?;
                let vp_btn = match button {
                    mouse::Button::Left => MouseButton::Left,
                    mouse::Button::Right => MouseButton::Right,
                    mouse::Button::Middle => MouseButton::Middle,
                    _ => return None,
                };
                state.controller.begin_frame(vp_ctx);
                state.controller.push_event(ViewportEvent::PointerMoved {
                    position: glam::vec2(pos.x, pos.y),
                });
                state.controller.push_event(ViewportEvent::MouseButton {
                    button: vp_btn,
                    state: ButtonState::Pressed,
                });
                state.controller.apply_to_camera(&mut state.camera);
                state.any_pressed = true;
                Some(iced::widget::shader::Action::request_redraw().and_capture())
            }

            Event::Mouse(mouse::Event::ButtonReleased(button)) => {
                let vp_btn = match button {
                    mouse::Button::Left => MouseButton::Left,
                    mouse::Button::Right => MouseButton::Right,
                    mouse::Button::Middle => MouseButton::Middle,
                    _ => return None,
                };
                state.controller.begin_frame(vp_ctx);
                state.controller.push_event(ViewportEvent::MouseButton {
                    button: vp_btn,
                    state: ButtonState::Released,
                });
                state.controller.apply_to_camera(&mut state.camera);
                state.any_pressed = false;
                None
            }

            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let pos = cursor.position_in(bounds)?;
                state.controller.begin_frame(vp_ctx);
                state.controller.push_event(ViewportEvent::PointerMoved {
                    position: glam::vec2(pos.x, pos.y),
                });
                state.controller.apply_to_camera(&mut state.camera);
                if state.any_pressed {
                    Some(iced::widget::shader::Action::request_redraw().and_capture())
                } else {
                    None
                }
            }

            Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                let _ = cursor.position_in(bounds)?;
                let scroll_y = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y * 28.0,
                    mouse::ScrollDelta::Pixels { y, .. } => *y,
                };
                state.controller.begin_frame(vp_ctx);
                state.controller.push_event(ViewportEvent::Wheel {
                    delta: glam::vec2(0.0, scroll_y),
                    units: ScrollUnits::Pixels,
                });
                state.controller.apply_to_camera(&mut state.camera);
                Some(iced::widget::shader::Action::request_redraw().and_capture())
            }

            Event::Mouse(mouse::Event::CursorLeft) => {
                state.controller.begin_frame(vp_ctx);
                state.controller.push_event(ViewportEvent::PointerLeft);
                state.controller.apply_to_camera(&mut state.camera);
                state.any_pressed = false;
                None
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
        // Camera is updated in update() via apply_to_camera.
        let mut cam = state.camera.clone();
        cam.set_aspect_ratio(bounds.width, bounds.height);

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
        if state.any_pressed {
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
