//! Slint integration for `viewport-lib`.
//!
//! Slint does not share a compatible `wgpu` device here, so the example uses
//! an offscreen renderer and copies the result back into a `slint::Image`.

mod viewport_bridge;

use std::cell::RefCell;
use std::rc::Rc;

use slint::Model;
use viewport_bridge::SceneRenderer;
use viewport_lib::{ButtonState, MouseButton, Modifiers, ScrollUnits, ViewportEvent};

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

        // Viewport input callbacks — raw input forwarded to OrbitCameraController.
        callback pointer-pressed(float, float, int);   // x, y, button (1=left,2=right,3=middle)
        callback pointer-released(int);                // button
        callback pointer-moved(float, float);          // x, y (viewport-local)
        callback scrolled(float);                      // scroll delta y (pixels)
        callback modifiers-changed(bool);              // shift held

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
            // The TouchArea forwards raw pointer/scroll events to Rust, which
            // passes them to OrbitCameraController via push_event().
            Rectangle {
                viewport-area := Image {
                    source: root.viewport-texture;
                    width: 100%;
                    height: 100%;
                }

                touch := TouchArea {
                    pointer-event(event) => {
                        if (event.kind == PointerEventKind.down) {
                            root.modifiers-changed(event.modifiers.shift);
                            if (event.button == PointerEventButton.left) {
                                root.pointer-pressed(self.mouse-x / 1phx, self.mouse-y / 1phx, 1);
                            } else if (event.button == PointerEventButton.right) {
                                root.pointer-pressed(self.mouse-x / 1phx, self.mouse-y / 1phx, 2);
                            } else if (event.button == PointerEventButton.middle) {
                                root.pointer-pressed(self.mouse-x / 1phx, self.mouse-y / 1phx, 3);
                            }
                        } else if (event.kind == PointerEventKind.up) {
                            if (event.button == PointerEventButton.left) {
                                root.pointer-released(1);
                            } else if (event.button == PointerEventButton.right) {
                                root.pointer-released(2);
                            } else if (event.button == PointerEventButton.middle) {
                                root.pointer-released(3);
                            }
                        }
                    }

                    moved => {
                        root.pointer-moved(self.mouse-x / 1phx, self.mouse-y / 1phx);
                    }

                    scroll-event(event) => {
                        root.scrolled(event.delta-y / 1phx);
                        return accept;
                    }
                }
            }
        }
    }
}

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

    // --- Viewport input callbacks — forward raw events to OrbitCameraController ---
    {
        let renderer = scene_renderer.clone();
        app.on_pointer_pressed(move |x, y, btn| {
            let button = match btn {
                1 => MouseButton::Left,
                2 => MouseButton::Right,
                _ => MouseButton::Middle,
            };
            renderer.borrow_mut().push_event(ViewportEvent::PointerMoved {
                position: glam::vec2(x, y),
            });
            renderer.borrow_mut().push_event(ViewportEvent::MouseButton {
                button,
                state: ButtonState::Pressed,
            });
        });
    }
    {
        let renderer = scene_renderer.clone();
        app.on_pointer_released(move |btn| {
            let button = match btn {
                1 => MouseButton::Left,
                2 => MouseButton::Right,
                _ => MouseButton::Middle,
            };
            renderer.borrow_mut().push_event(ViewportEvent::MouseButton {
                button,
                state: ButtonState::Released,
            });
        });
    }
    {
        let renderer = scene_renderer.clone();
        app.on_pointer_moved(move |x, y| {
            renderer.borrow_mut().push_event(ViewportEvent::PointerMoved {
                position: glam::vec2(x, y),
            });
        });
    }
    {
        let renderer = scene_renderer.clone();
        app.on_scrolled(move |dy| {
            renderer.borrow_mut().push_event(ViewportEvent::Wheel {
                delta: glam::vec2(0.0, dy),
                units: ScrollUnits::Pixels,
            });
        });
    }
    {
        let renderer = scene_renderer.clone();
        app.on_modifiers_changed(move |shift| {
            renderer.borrow_mut().push_event(ViewportEvent::ModifiersChanged(
                if shift { Modifiers::SHIFT } else { Modifiers::NONE },
            ));
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
