//! Slint integration for `viewport-lib`.
//!
//! Slint does not share a compatible `wgpu` device here, so the example uses
//! an offscreen renderer and copies the result back into a `slint::Image`.

mod viewport_bridge;

use std::cell::RefCell;
use std::rc::Rc;

use slint::Model;
use viewport_bridge::SceneRenderer;

slint::slint! {
    struct ObjectEntry {
        id: int,
        name: string,
    }

    export component App inherits Window {
        title: "viewport-lib - Slint Example";
        preferred-width: 900px;
        preferred-height: 600px;
        background: #1a1a1e;

        in property <image> viewport-texture;
        out property <int> requested-texture-width: viewport-area.width / 1phx;
        out property <int> requested-texture-height: viewport-area.height / 1phx;
        in-out property <[ObjectEntry]> objects;

        callback add-object();
        callback remove-object(int);

        // Viewport input callbacks - Slint tracks mouse state and sends deltas
        // to Rust, which applies them to the Camera.
        //   orbit(dx, dy):  pixel delta for orbit rotation
        //   pan(dx, dy):    pixel delta for camera panning
        //   zoom(delta):    scroll amount for zoom
        callback orbit(float, float);
        callback pan(float, float);
        callback zoom(float);

        HorizontalLayout {
            // --- Left panel: object list ---
            VerticalLayout {
                width: 200px;
                padding: 10px;
                spacing: 8px;

                Text {
                    text: "Objects";
                    font-size: 16px;
                    color: #e0e0e0;
                }

                Rectangle {
                    height: 36px;
                    border-radius: 4px;
                    background: #3a7cff;

                    Text {
                        text: "+ Add Box";
                        color: white;
                        vertical-alignment: center;
                        horizontal-alignment: center;
                    }

                    TouchArea {
                        clicked => { root.add-object(); }
                    }
                }

                Flickable {
                    VerticalLayout {
                        spacing: 4px;

                        for obj in root.objects: Rectangle {
                            height: 30px;
                            border-radius: 3px;
                            background: #2a2a2e;

                            HorizontalLayout {
                                padding-left: 8px;
                                padding-right: 4px;

                                Text {
                                    text: obj.name;
                                    color: #c0c0c0;
                                    vertical-alignment: center;
                                }

                                Rectangle {
                                    width: 24px;
                                    height: 24px;
                                    border-radius: 3px;
                                    background: #ff4444;

                                    Text {
                                        text: "x";
                                        color: white;
                                        vertical-alignment: center;
                                        horizontal-alignment: center;
                                    }

                                    TouchArea {
                                        clicked => { root.remove-object(obj.id); }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // --- Right side: viewport with input handling ---
            //
            // The TouchArea wraps the viewport image and captures all mouse
            // interactions. It tracks which button was pressed, computes pixel
            // deltas on `moved`, and dispatches to the appropriate callback:
            //   - Left-drag or Middle-drag (no shift): orbit
            //   - Right-drag or Shift+Middle-drag: pan
            //   - Scroll wheel: zoom
            Rectangle {
                viewport-area := Image {
                    source: root.viewport-texture;
                    width: 100%;
                    height: 100%;
                }

                // Track which button initiated the current drag.
                // 0 = none, 1 = left, 2 = right, 3 = middle
                property <int> active-button: 0;
                property <bool> shift-held: false;
                property <float> last-x: 0;
                property <float> last-y: 0;

                touch := TouchArea {
                    pointer-event(event) => {
                        if (event.kind == PointerEventKind.down) {
                            parent.last-x = self.mouse-x / 1phx;
                            parent.last-y = self.mouse-y / 1phx;
                            parent.shift-held = event.modifiers.shift;
                            if (event.button == PointerEventButton.left) {
                                parent.active-button = 1;
                            } else if (event.button == PointerEventButton.right) {
                                parent.active-button = 2;
                            } else if (event.button == PointerEventButton.middle) {
                                parent.active-button = 3;
                            }
                        } else if (event.kind == PointerEventKind.up) {
                            parent.active-button = 0;
                        }
                    }

                    moved => {
                        if (parent.active-button != 0) {
                            if (parent.active-button == 2 ||
                                (parent.active-button == 3 && parent.shift-held)) {
                                root.pan(self.mouse-x / 1phx - parent.last-x,
                                         self.mouse-y / 1phx - parent.last-y);
                            } else {
                                root.orbit(self.mouse-x / 1phx - parent.last-x,
                                           self.mouse-y / 1phx - parent.last-y);
                            }
                            parent.last-x = self.mouse-x / 1phx;
                            parent.last-y = self.mouse-y / 1phx;
                        }
                    }

                    scroll-event(event) => {
                        root.zoom(event.delta-y / 1phx);
                        return accept;
                    }
                }
            }
        }
    }
}

/// Camera control constants shared with the other viewport examples.
const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.001;
const MIN_DISTANCE: f32 = 0.1;

fn main() {
    let app = App::new().unwrap();

    // Create our own wgpu 27 device for the viewport renderer.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("Failed to find a suitable GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("slint_viewport_device"),
        ..Default::default()
    }))
    .expect("Failed to create wgpu device");

    let scene_renderer = Rc::new(RefCell::new(SceneRenderer::new(&device, &queue)));
    let device = Rc::new(device);
    let queue = Rc::new(queue);

    // Scene state.
    let next_id = Rc::new(RefCell::new(1i32));

    // --- Add / Remove callbacks ---
    {
        let app_weak = app.as_weak();
        let next_id = next_id.clone();
        app.on_add_object(move || {
            let app = app_weak.unwrap();
            let id = {
                let mut n = next_id.borrow_mut();
                let id = *n;
                *n += 1;
                id
            };
            let mut objects: Vec<ObjectEntry> = app.get_objects().iter().collect();
            objects.push(ObjectEntry {
                id,
                name: format!("Box {id}").into(),
            });
            let model = Rc::new(slint::VecModel::from(objects));
            app.set_objects(model.into());
        });
    }
    {
        let app_weak = app.as_weak();
        app.on_remove_object(move |id| {
            let app = app_weak.unwrap();
            let objects: Vec<ObjectEntry> =
                app.get_objects().iter().filter(|o| o.id != id).collect();
            let model = Rc::new(slint::VecModel::from(objects));
            app.set_objects(model.into());
        });
    }

    // --- Viewport input callbacks ---
    //
    // These receive pixel deltas from the Slint TouchArea and apply them
    // to the Camera inside the SceneRenderer. This is the Slint-specific
    // input adapter - each framework needs its own version of this
    // translation, but the camera math is identical.
    {
        let renderer = scene_renderer.clone();
        app.on_orbit(move |dx, dy| {
            let mut r = renderer.borrow_mut();
            let cam = r.camera_mut();
            let q_yaw = glam::Quat::from_rotation_y(-dx * ORBIT_SENSITIVITY);
            let q_pitch = glam::Quat::from_rotation_x(-dy * ORBIT_SENSITIVITY);
            cam.orientation = (q_yaw * cam.orientation * q_pitch).normalize();
        });
    }
    {
        let renderer = scene_renderer.clone();
        let app_weak = app.as_weak();
        app.on_pan(move |dx, dy| {
            let app = app_weak.unwrap();
            let viewport_h = app.get_requested_texture_height().max(1) as f32;
            let mut r = renderer.borrow_mut();
            let cam = r.camera_mut();
            let pan_scale = 2.0 * cam.distance * (cam.fov_y / 2.0).tan() / viewport_h;
            let right = cam.right();
            let up = cam.up();
            cam.center -= right * dx * pan_scale;
            cam.center += up * dy * pan_scale;
        });
    }
    {
        let renderer = scene_renderer.clone();
        app.on_zoom(move |delta| {
            let mut r = renderer.borrow_mut();
            let cam = r.camera_mut();
            cam.distance = (cam.distance * (1.0 - delta * ZOOM_SENSITIVITY)).max(MIN_DISTANCE);
        });
    }

    // --- Rendering notifier: render viewport and import texture each frame ---
    {
        let app_weak = app.as_weak();
        let scene_renderer = scene_renderer.clone();
        let device = device.clone();
        let queue = queue.clone();
        app.window()
            .set_rendering_notifier(move |state, _graphics_api| {
                if !matches!(state, slint::RenderingState::BeforeRendering) {
                    return;
                }
                let Some(app) = app_weak.upgrade() else {
                    return;
                };

                let w = app.get_requested_texture_width().max(1) as u32;
                let h = app.get_requested_texture_height().max(1) as u32;

                // Collect object positions from the UI model.
                let objects: Vec<(u64, [f32; 3])> = app
                    .get_objects()
                    .iter()
                    .enumerate()
                    .map(|(i, obj)| {
                        let n = i as f32;
                        let x = (n % 4.0) * 2.0 - 3.0;
                        let z = (n / 4.0).floor() * 2.0 - 3.0;
                        (obj.id as u64, [x, 0.0, z])
                    })
                    .collect();

                let image = scene_renderer
                    .borrow_mut()
                    .render(&device, &queue, w, h, &objects);

                app.set_viewport_texture(image);
                app.window().request_redraw();
            })
            .expect("Failed to set rendering notifier");
    }

    app.run().unwrap();
}
