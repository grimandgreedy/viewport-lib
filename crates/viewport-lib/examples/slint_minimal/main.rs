//! Minimal viewport-lib example using Slint on the wgpu 29 leg.
//!
//! Slint pairs with wgpu 29 through its `unstable-wgpu-29` feature. The
//! integration shape differs from the eframe examples: instead of painting
//! into the host's render pass, we hand Slint our wgpu instance/device/queue
//! up front (`BackendSelector::require_wgpu_29`), render the scene into our
//! own texture each frame with `render_to_texture`, and import that texture
//! into the UI as a `slint::Image`. Build with
//! `--no-default-features --features wgpu29`.
//!
//! Navigation:
//!   Left drag   : orbit
//!   Middle drag : orbit
//!   Right drag  : pan
//!   Scroll      : zoom

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use viewport_lib as vpl;

use slint::wgpu_29::{self, wgpu};
use vpl::{
    ButtonState, Camera, CameraFrame, FrameData, LightingSettings, Material, MouseButton,
    OrbitCameraController, SceneFrame, SceneRenderItem, ScrollUnits, ViewportContext,
    ViewportEvent, ViewportRenderer, primitives,
};

slint::slint! {
    export component MainWindow inherits Window {
        title: "viewport-lib : Slint minimal (wgpu 29)";
        preferred-width: 1280px;
        preferred-height: 720px;

        in property <image> viewport-texture;
        out property <length> viewport-width: self.width;
        out property <length> viewport-height: self.height;

        callback pointer-moved(float, float);
        // button: 0 = left, 1 = right, 2 = middle; pressed: down vs up.
        callback mouse-button(int, bool);
        callback wheel(float, float);

        Image {
            width: 100%;
            height: 100%;
            source: root.viewport-texture;
        }
        TouchArea {
            pointer-event(ev) => {
                if (ev.kind == PointerEventKind.move) {
                    root.pointer-moved(self.mouse-x / 1px, self.mouse-y / 1px);
                } else if (ev.kind == PointerEventKind.down || ev.kind == PointerEventKind.up) {
                    root.mouse-button(
                        ev.button == PointerEventButton.left ? 0
                            : ev.button == PointerEventButton.right ? 1
                            : 2,
                        ev.kind == PointerEventKind.down);
                }
            }
            scroll-event(ev) => {
                root.wheel(ev.delta-x / 1px, ev.delta-y / 1px);
                accept
            }
        }
    }
}

/// Everything the per-frame tick mutates, shared between the timer closure
/// and the input callbacks.
struct State {
    renderer: ViewportRenderer,
    camera: Camera,
    controller: OrbitCameraController,
    scene_items: Vec<SceneRenderItem>,
    events: Vec<ViewportEvent>,
    /// Offscreen target the scene renders into; recreated on resize.
    target: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
}

// The scene renders through an sRGB view so the tonemap's linear output is
// sRGB-encoded on write. Slint's femtovg renderer, though, samples an imported
// texture with a default view (the texture's own format) and does no colour
// management: it blits the sampled bytes straight into its Rgba8Unorm canvas. If
// the texture were sRGB, that sample would decode the encoded bytes back to
// linear and the image would come out too dark. So the texture's base format is
// plain Rgba8Unorm (femtovg reads the encoded bytes verbatim) and we render into
// an Rgba8UnormSrgb view of the same texture.
const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const IMPORT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn main() -> Result<(), slint::PlatformError> {
    // Create the wgpu stack ourselves so viewport-lib and Slint share one
    // device and queue, then hand it to Slint before any window exists.
    // No window exists yet, so the instance is created without a display
    // handle; Slint creates its surfaces from this instance later.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("slint-minimal"),
        required_features: ViewportRenderer::recommended_device_features(&adapter),
        required_limits: ViewportRenderer::recommended_device_limits(&adapter),
        ..Default::default()
    }))
    .expect("device request failed");

    slint::BackendSelector::new()
        .require_wgpu_29(wgpu_29::WGPUConfiguration::Manual {
            instance: instance.clone(),
            adapter: adapter.clone(),
            device: device.clone(),
            queue: queue.clone(),
        })
        .select()?;

    let mut renderer = ViewportRenderer::new(&device, RENDER_FORMAT);
    let res = renderer.resources_mut();
    let m_sphere = res
        .upload_mesh_data(&device, &primitives::sphere(0.6, 24, 12))
        .unwrap();
    let m_cube = res
        .upload_mesh_data(&device, &primitives::cube(1.0))
        .unwrap();
    let m_torus = res
        .upload_mesh_data(&device, &primitives::torus(0.5, 0.18, 32, 16))
        .unwrap();

    let make = |mesh_id, [x, y, z]: [f32; 3], colour: [f32; 3]| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, z)).to_cols_array_2d();
        item.material = Material::from_colour(colour);
        item.material.backface_policy = vpl::BackfacePolicy::Identical;
        item
    };

    let state = Rc::new(RefCell::new(State {
        renderer,
        camera: Camera {
            distance: 10.0,
            ..Camera::default()
        },
        controller: OrbitCameraController::viewport_primitives(),
        scene_items: vec![
            make(m_sphere, [-2.5, 0.0, 0.0], [0.9, 0.5, 0.2]),
            make(m_cube, [0.0, 0.0, 0.0], [0.4, 0.6, 0.9]),
            make(m_torus, [2.5, 0.0, 0.0], [0.3, 0.8, 0.4]),
        ],
        events: Vec::new(),
        target: None,
    }));

    let ui = MainWindow::new()?;

    // Input callbacks queue ViewportEvents; the tick drains them.
    {
        let state = state.clone();
        ui.on_pointer_moved(move |x, y| {
            state.borrow_mut().events.push(ViewportEvent::PointerMoved {
                position: glam::Vec2::new(x, y),
            });
        });
    }
    {
        let state = state.clone();
        ui.on_mouse_button(move |button, pressed| {
            let button = match button {
                0 => MouseButton::Left,
                1 => MouseButton::Right,
                _ => MouseButton::Middle,
            };
            state.borrow_mut().events.push(ViewportEvent::MouseButton {
                button,
                state: if pressed {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                },
            });
        });
    }
    {
        let state = state.clone();
        ui.on_wheel(move |dx, dy| {
            state.borrow_mut().events.push(ViewportEvent::Wheel {
                delta: glam::Vec2::new(dx, dy),
                units: ScrollUnits::Pixels,
            });
        });
    }

    // Drive rendering with a repeating timer: render the scene into our
    // texture, import it as a slint::Image, and let Slint composite it.
    let timer = slint::Timer::default();
    {
        let ui_weak = ui.as_weak();
        let state = state.clone();
        timer.start(
            slint::TimerMode::Repeated,
            Duration::from_millis(16),
            move || {
                let Some(ui) = ui_weak.upgrade() else { return };
                let mut s = state.borrow_mut();
                let s = &mut *s;

                let w = ui.get_viewport_width().max(1.0);
                let h = ui.get_viewport_height().max(1.0);
                let sf = ui.window().scale_factor();
                let pw = ((w * sf) as u32).max(1);
                let ph = ((h * sf) as u32).max(1);

                if s.target.as_ref().map(|t| (t.2, t.3)) != Some((pw, ph)) {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("slint-minimal-target"),
                        size: wgpu::Extent3d {
                            width: pw,
                            height: ph,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: IMPORT_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        // The sRGB view we render into reinterprets the same bytes.
                        view_formats: &[RENDER_FORMAT],
                    });
                    // Render through an sRGB view (encode on write); Slint imports
                    // the base Rgba8Unorm texture and reads the bytes verbatim.
                    let view = texture.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(RENDER_FORMAT),
                        ..Default::default()
                    });
                    s.target = Some((texture, view, pw, ph));
                }

                s.controller.begin_frame(ViewportContext {
                    hovered: true,
                    focused: true,
                    viewport_size: [w, h],
                });
                for event in s.events.drain(..) {
                    s.controller.push_event(event);
                }
                s.controller.apply_to_camera(&mut s.camera);
                s.camera.set_aspect_ratio(w, h);

                let mut frame = FrameData::new(
                    CameraFrame::from_camera(&s.camera, [w, h]).with_pixels_per_point(sf),
                    SceneFrame::from_surface_items(s.scene_items.clone()),
                );
                frame.effects.lighting = LightingSettings::default();

                let (texture, view, _, _) = s.target.as_ref().unwrap();
                s.renderer.render_to_texture(&device, &queue, view, &frame);

                let image = slint::Image::try_from(texture.clone())
                    .expect("import rendered texture into Slint");
                ui.set_viewport_texture(image);
            },
        );
    }

    ui.run()
}
