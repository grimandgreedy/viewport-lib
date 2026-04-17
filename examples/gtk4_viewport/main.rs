//! GTK4 integration for `viewport-lib`.
//!
//! This example renders offscreen and displays the result through
//! `gdk::MemoryTexture`, which keeps it usable even though GTK4 does not expose
//! a shareable `wgpu` device here.

mod viewport_bridge;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;
use viewport_bridge::SceneRenderer;

/// Camera control constants shared with the other viewport examples.
const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.001;
const MIN_DISTANCE: f32 = 0.1;

/// Shared application state behind `Rc<RefCell<...>>` so GTK callbacks can
/// mutate it.
struct AppState {
    scene_renderer: SceneRenderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    objects: Vec<(u64, [f32; 3])>,
    next_id: u64,
    dirty: bool,
    tex_w: u32,
    tex_h: u32,
}

fn main() {
    let app = gtk4::Application::builder()
        .application_id("com.grimandgreedy.viewportlib.gtk4-example")
        .build();

    app.connect_activate(build_ui);
    app.run();
}

fn build_ui(app: &gtk4::Application) {
    // Create our own wgpu device (separate from GTK's rendering).
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("Failed to find a suitable GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gtk4_viewport_device"),
        ..Default::default()
    }))
    .expect("Failed to create wgpu device");

    let scene_renderer = SceneRenderer::new(&device, &queue);

    let state = Rc::new(RefCell::new(AppState {
        scene_renderer,
        device,
        queue,
        objects: Vec::new(),
        next_id: 1,
        dirty: true,
        tex_w: 0,
        tex_h: 0,
    }));

    // --- Build UI ---

    let window = gtk4::ApplicationWindow::builder()
        .application(app)
        .title("viewport-lib - GTK4 Example")
        .default_width(900)
        .default_height(600)
        .build();

    let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);

    // --- Left panel: object list ---
    let sidebar = gtk4::Box::new(gtk4::Orientation::Vertical, 8);
    sidebar.set_width_request(200);
    sidebar.set_margin_top(10);
    sidebar.set_margin_bottom(10);
    sidebar.set_margin_start(10);
    sidebar.set_margin_end(10);

    let heading = gtk4::Label::new(Some("Objects"));
    heading.set_halign(gtk4::Align::Start);
    heading.add_css_class("heading");
    sidebar.append(&heading);

    let add_btn = gtk4::Button::with_label("+ Add Box");
    sidebar.append(&add_btn);

    let scrolled = gtk4::ScrolledWindow::new();
    scrolled.set_vexpand(true);
    let list_box = gtk4::ListBox::new();
    list_box.set_selection_mode(gtk4::SelectionMode::None);
    scrolled.set_child(Some(&list_box));
    sidebar.append(&scrolled);

    hbox.append(&sidebar);

    // --- Right side: viewport ---
    let picture = gtk4::Picture::new();
    picture.set_hexpand(true);
    picture.set_vexpand(true);
    picture.set_halign(gtk4::Align::Fill);
    picture.set_valign(gtk4::Align::Fill);
    hbox.append(&picture);

    window.set_child(Some(&hbox));

    // --- Add / Remove callbacks ---
    {
        let state = state.clone();
        let list_box = list_box.clone();
        let picture = picture.clone();
        add_btn.connect_clicked(move |_| {
            let (id, name) = {
                let mut s = state.borrow_mut();
                let id = s.next_id;
                s.next_id += 1;
                let n = s.objects.len() as f32;
                let x = (n % 4.0) * 2.0 - 3.0;
                let z = (n / 4.0).floor() * 2.0 - 3.0;
                s.objects.push((id, [x, 0.0, z]));
                s.dirty = true;
                (id, format!("Box {id}"))
            };

            // Add a row to the list with a remove button.
            let row = gtk4::ListBoxRow::new();
            let row_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 4);
            row_box.set_margin_top(2);
            row_box.set_margin_bottom(2);

            let label = gtk4::Label::new(Some(&name));
            label.set_hexpand(true);
            label.set_halign(gtk4::Align::Start);
            row_box.append(&label);

            let remove_btn = gtk4::Button::with_label("x");
            row_box.append(&remove_btn);

            row.set_child(Some(&row_box));
            list_box.append(&row);

            {
                let state = state.clone();
                let list_box = list_box.clone();
                let row = row.clone();
                let picture = picture.clone();
                remove_btn.connect_clicked(move |_| {
                    state.borrow_mut().objects.retain(|(oid, _)| *oid != id);
                    state.borrow_mut().dirty = true;
                    list_box.remove(&row);
                    picture.queue_draw();
                });
            }

            picture.queue_draw();
        });
    }

    // --- Viewport input handling ---
    //
    // GTK4 uses gesture controllers attached to widgets. We create separate
    // GestureDrag instances for each mouse button and an EventControllerScroll
    // for zoom. Each gesture tracks the cumulative drag offset and computes
    // frame-to-frame deltas by remembering the previous offset.

    // Left-drag: orbit
    attach_drag_gesture(&picture, 1, false, state.clone());
    // Middle-drag: orbit (or pan with shift)
    attach_drag_gesture(&picture, 2, false, state.clone());
    // Right-drag: pan
    attach_drag_gesture(&picture, 3, true, state.clone());

    // Scroll: zoom
    {
        let state = state.clone();
        let picture_ref = picture.clone();
        let scroll_ctrl =
            gtk4::EventControllerScroll::new(gtk4::EventControllerScrollFlags::VERTICAL);
        scroll_ctrl.connect_scroll(move |_ctrl, _dx, dy| {
            let scroll_y = dy as f32 * -28.0; // Invert and scale line delta
            let mut s = state.borrow_mut();
            let cam = s.scene_renderer.camera_mut();
            cam.distance = (cam.distance * (1.0 - scroll_y * ZOOM_SENSITIVITY)).max(MIN_DISTANCE);
            s.dirty = true;
            drop(s);
            picture_ref.queue_draw();
            glib::Propagation::Stop
        });
        picture.add_controller(scroll_ctrl);
    }

    // --- Render loop via tick callback ---
    //
    // The tick callback fires once per frame (synced to the display). If the
    // scene is dirty or the viewport size changed, we re-render and update
    // the Picture's texture.
    {
        let state = state.clone();
        picture.add_tick_callback(move |widget, _clock| {
            let w = widget.width() as u32;
            let h = widget.height() as u32;
            if w == 0 || h == 0 {
                return glib::ControlFlow::Continue;
            }

            let needs_render = {
                let s = state.borrow();
                s.dirty || s.tex_w != w || s.tex_h != h
            };

            if needs_render {
                let pixels = {
                    let mut s = state.borrow_mut();
                    let objects = s.objects.clone();
                    let device = &s.device as *const wgpu::Device;
                    let queue = &s.queue as *const wgpu::Queue;
                    // Safety: device and queue are not modified by render().
                    let pixels = s.scene_renderer.render(
                        unsafe { &*device },
                        unsafe { &*queue },
                        w,
                        h,
                        &objects,
                    );
                    s.tex_w = w;
                    s.tex_h = h;
                    s.dirty = false;
                    pixels
                };

                let bytes = glib::Bytes::from_owned(pixels);
                let texture = gdk::MemoryTexture::new(
                    w as i32,
                    h as i32,
                    gdk::MemoryFormat::R8g8b8a8,
                    &bytes,
                    (w * 4) as usize,
                );
                widget
                    .downcast_ref::<gtk4::Picture>()
                    .unwrap()
                    .set_paintable(Some(&texture));
            }

            glib::ControlFlow::Continue
        });
    }

    window.present();
}

/// Create a `GestureDrag` for a specific mouse button and attach it to the
/// viewport widget. Computes frame-to-frame deltas from the cumulative offset
/// and applies orbit or pan to the camera.
fn attach_drag_gesture(
    widget: &gtk4::Picture,
    button: u32,
    force_pan: bool,
    state: Rc<RefCell<AppState>>,
) {
    let drag = gtk4::GestureDrag::new();
    drag.set_button(button);

    // Track previous offset so we can compute frame-to-frame deltas.
    // GestureDrag's `drag-update` signal gives the total offset from
    // the drag start, not the per-frame delta.
    let prev_offset = Rc::new(Cell::new((0.0f64, 0.0f64)));

    {
        let prev_offset = prev_offset.clone();
        drag.connect_drag_begin(move |_gesture, _x, _y| {
            prev_offset.set((0.0, 0.0));
        });
    }

    {
        let state = state.clone();
        let prev_offset = prev_offset.clone();
        let widget = widget.clone();
        drag.connect_drag_update(move |gesture, offset_x, offset_y| {
            let (px, py) = prev_offset.get();
            let dx = (offset_x - px) as f32;
            let dy = (offset_y - py) as f32;
            prev_offset.set((offset_x, offset_y));

            if dx.abs() < 0.001 && dy.abs() < 0.001 {
                return;
            }

            // Middle button: check shift modifier to decide orbit vs pan.
            let is_pan = if force_pan {
                true
            } else if button == 2 {
                gesture
                    .current_event_state()
                    .contains(gdk::ModifierType::SHIFT_MASK)
            } else {
                false
            };

            let mut s = state.borrow_mut();
            if is_pan {
                let viewport_h = s.tex_h.max(1) as f32;
                let cam = s.scene_renderer.camera_mut();
                let pan_scale = 2.0 * cam.distance * (cam.fov_y / 2.0).tan() / viewport_h;
                let right = cam.right();
                let up = cam.up();
                cam.center -= right * dx * pan_scale;
                cam.center += up * dy * pan_scale;
            } else {
                let cam = s.scene_renderer.camera_mut();
                let q_yaw = glam::Quat::from_rotation_y(-dx * ORBIT_SENSITIVITY);
                let q_pitch = glam::Quat::from_rotation_x(-dy * ORBIT_SENSITIVITY);
                cam.orientation = (q_yaw * cam.orientation * q_pitch).normalize();
            }
            s.dirty = true;
            drop(s);
            widget.queue_draw();
        });
    }

    widget.add_controller(drag);
}
