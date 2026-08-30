//! Iced shader-widget bridge for `viewport-lib`.
//!
//! `ViewportState` owns the camera and input tracking so `shader::Program::update()`
//! can translate Iced events directly into orbit, pan, and zoom changes.

use std::collections::HashMap;
use viewport_lib as vpl;

use iced::event::Event;
use iced::widget::shader;
use iced::{Element, Fill, Rectangle, mouse};
use vpl::{
    ButtonState, Camera, CameraFrame, FrameData, LightingSettings, MeshId, Modifiers, MouseButton,
    OffscreenViewportTarget, OrbitCameraController, RenderCamera, SceneFrame, SceneRenderItem,
    ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
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
                orientation: glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1),
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
    uploaded: HashMap<u64, MeshId>,
    /// Offscreen sRGB target the scene renders into, recreated on resize. iced
    /// hands the shader widget a non-sRGB surface view (`Bgra8Unorm` here), so
    /// rendering straight into it would skip the tonemap's linear->sRGB encode
    /// and come out too dark. Instead we render into this sRGB target (encode
    /// happens on write) and blit the encoded bytes into iced's target.
    offscreen: Option<OffscreenViewportTarget>,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    /// Bind group over the offscreen sample view, rebuilt when it is recreated.
    blit_bind: Option<wgpu::BindGroup>,
    /// The format iced renders to (the blit's output target format).
    target_format: wgpu::TextureFormat,
}

/// Fullscreen-triangle blit: samples the offscreen (non-sRGB view, so the
/// sRGB-encoded bytes are read verbatim) and writes them to iced's target.
const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = xy[vi];
    var out: VOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_smp, in.uv);
}
"#;

fn create_blit(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Sampler) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("iced_viewport_blit_shader"),
        source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("iced_viewport_blit_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("iced_viewport_blit_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("iced_viewport_blit_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("iced_viewport_blit_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (pipeline, bgl, sampler)
}

impl shader::Pipeline for ViewportPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // Render into an sRGB target so the tonemap encode happens; iced's own
        // target (`format`) is non-sRGB and receives the encoded bytes via blit.
        let renderer =
            ViewportRenderer::new(device, OffscreenViewportTarget::render_format(format));
        let (blit_pipeline, blit_bgl, blit_sampler) = create_blit(device, format);

        Self {
            renderer,
            uploaded: HashMap::new(),
            offscreen: None,
            blit_pipeline,
            blit_bgl,
            blit_sampler,
            blit_bind: None,
            target_format: format,
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
        let scale = viewport.scale_factor() as f32;
        // Size the offscreen to the widget's physical pixels (the blit maps it
        // across the same rect). Recreate on resize and drop the stale bind group.
        let size = [
            (bounds.width * scale).round().max(1.0) as u32,
            (bounds.height * scale).round().max(1.0) as u32,
        ];
        if pipeline.offscreen.as_ref().map(|o| o.size()) != Some(size) {
            pipeline.offscreen = Some(OffscreenViewportTarget::new(
                device,
                pipeline.target_format,
                size,
            ));
            pipeline.blit_bind = None;
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
                let mesh_id = *pipeline.uploaded.get(&obj.id)?;
                let model = glam::Mat4::from_translation(glam::Vec3::from(obj.position));
                let mut item = SceneRenderItem::default();
                item.mesh_id = mesh_id;
                item.model = model.to_cols_array_2d();
                Some(item)
            })
            .collect();

        let mut frame_data = FrameData::new(
            CameraFrame::new(
                self.camera_snapshot.render_camera.clone(),
                [bounds.width, bounds.height],
            )
            .with_pixels_per_point(viewport.scale_factor() as f32),
            SceneFrame::from_surface_items(scene_items),
        );
        frame_data.effects.lighting = LightingSettings::default();
        frame_data.viewport.show_grid = true;
        frame_data.viewport.grid_z = -0.5;
        frame_data.viewport.show_axes_indicator = true;

        // Render the whole frame (prepare + paint, depth managed internally) into
        // the offscreen sRGB target. `render` then just blits the result.
        let offscreen = pipeline
            .offscreen
            .as_ref()
            .expect("offscreen created above");
        pipeline
            .renderer
            .render_to_texture(device, queue, offscreen.render_view(), &frame_data);

        // Build the blit bind group once per offscreen texture (its sample view
        // is stable until the next resize).
        if pipeline.blit_bind.is_none() {
            let offscreen = pipeline.offscreen.as_ref().unwrap();
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iced_viewport_blit_bind"),
                layout: &pipeline.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(offscreen.sample_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&pipeline.blit_sampler),
                    },
                ],
            });
            pipeline.blit_bind = Some(bind);
        }
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        // The scene was rendered into the offscreen target in `prepare`; here we
        // just blit its (sRGB-encoded) bytes into iced's target over the widget
        // rect. The blit covers every pixel it touches, so a Load is fine.
        let Some(bind) = pipeline.blit_bind.as_ref() else {
            return;
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iced_viewport_blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
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
        render_pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        render_pass.set_pipeline(&pipeline.blit_pipeline);
        render_pass.set_bind_group(0, bind, &[]);
        render_pass.draw(0..3, 0..1);
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
    /// `begin_frame` only resets the per-frame drag/wheel accumulators : it
    /// preserves `pointer_pos` and `button_held` : so delta computation remains
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
                let (scroll_y, units) = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => (*y, ScrollUnits::Lines),
                    mouse::ScrollDelta::Pixels { y, .. } => (*y, ScrollUnits::Pixels),
                };
                state.controller.begin_frame(vp_ctx);
                state.controller.push_event(ViewportEvent::Wheel {
                    delta: glam::vec2(0.0, scroll_y),
                    units,
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
