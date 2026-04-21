//! GTK4 integration for `viewport-lib`.
//!
//! This example renders offscreen and displays the result through
//! `gdk::MemoryTexture`, which keeps it usable even though GTK4 does not expose
//! a shareable `wgpu` device here.

mod viewport_bridge;

use std::cell::RefCell;
use std::rc::Rc;

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;
use viewport_bridge::SceneRenderer;
use viewport_lib::{ButtonState, Modifiers, MouseButton, ScrollUnits, ViewportEvent};

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

    // --- Viewport input handling —- forward raw events to OrbitCameraController ---
    //
    // EventControllerMotion  → PointerMoved / PointerLeft
    // GestureClick (all btns) → MouseButton Pressed / Released
    // EventControllerKey     → ModifiersChanged (shift tracking)
    // EventControllerScroll  → Wheel

    // Motion: update cursor position for the controller.
    {
        let state = state.clone();
        let motion = gtk4::EventControllerMotion::new();
        let state_enter = state.clone();
        motion.connect_enter(move |_ctrl, x, y| {
            let mut s = state_enter.borrow_mut();
            s.scene_renderer.push_event(ViewportEvent::PointerMoved {
                position: glam::vec2(x as f32, y as f32),
            });
            s.dirty = true;
        });
        let state_motion = state.clone();
        let picture_ref = picture.clone();
        motion.connect_motion(move |_ctrl, x, y| {
            let mut s = state_motion.borrow_mut();
            s.scene_renderer.push_event(ViewportEvent::PointerMoved {
                position: glam::vec2(x as f32, y as f32),
            });
            s.dirty = true;
            drop(s);
            picture_ref.queue_draw();
        });
        let state_leave = state.clone();
        motion.connect_leave(move |_ctrl| {
            let mut s = state_leave.borrow_mut();
            s.scene_renderer.push_event(ViewportEvent::PointerLeft);
            s.dirty = true;
        });
        picture.add_controller(motion);
    }

    // Button press/release: GestureClick with button=0 captures all buttons.
    {
        let click = gtk4::GestureClick::new();
        click.set_button(0);
        let state_press = state.clone();
        let picture_press = picture.clone();
        click.connect_pressed(move |gesture, _n_press, x, y| {
            let gtk_btn = gesture.current_button();
            let vp_btn = match gtk_btn {
                1 => MouseButton::Left,
                3 => MouseButton::Right,
                2 => MouseButton::Middle,
                _ => return,
            };
            let shift = gesture
                .current_event_state()
                .contains(gdk::ModifierType::SHIFT_MASK);
            let mut s = state_press.borrow_mut();
            s.scene_renderer.push_event(ViewportEvent::ModifiersChanged(
                if shift { Modifiers::SHIFT } else { Modifiers::NONE },
            ));
            s.scene_renderer.push_event(ViewportEvent::PointerMoved {
                position: glam::vec2(x as f32, y as f32),
            });
            s.scene_renderer.push_event(ViewportEvent::MouseButton {
                button: vp_btn,
                state: ButtonState::Pressed,
            });
            s.dirty = true;
            drop(s);
            picture_press.queue_draw();
        });
        let state_release = state.clone();
        click.connect_released(move |gesture, _n_press, _x, _y| {
            let gtk_btn = gesture.current_button();
            let vp_btn = match gtk_btn {
                1 => MouseButton::Left,
                3 => MouseButton::Right,
                2 => MouseButton::Middle,
                _ => return,
            };
            state_release
                .borrow_mut()
                .scene_renderer
                .push_event(ViewportEvent::MouseButton {
                    button: vp_btn,
                    state: ButtonState::Released,
                });
        });
        picture.add_controller(click);
    }

    // Modifier tracking: catch shift press/release even without a button held.
    {
        let state_key = state.clone();
        let key_ctrl = gtk4::EventControllerKey::new();
        key_ctrl.connect_modifiers(move |_ctrl, mods| {
            let shift = mods.contains(gdk::ModifierType::SHIFT_MASK);
            state_key
                .borrow_mut()
                .scene_renderer
                .push_event(ViewportEvent::ModifiersChanged(
                    if shift { Modifiers::SHIFT } else { Modifiers::NONE },
                ));
            false
        });
        window.add_controller(key_ctrl);
    }

    // Scroll: zoom.
    {
        let state = state.clone();
        let picture_ref = picture.clone();
        let scroll_ctrl =
            gtk4::EventControllerScroll::new(gtk4::EventControllerScrollFlags::VERTICAL);
        scroll_ctrl.connect_scroll(move |_ctrl, _dx, dy| {
            let scroll_y = dy as f32 * -28.0; // GTK dy is inverted relative to convention
            let mut s = state.borrow_mut();
            s.scene_renderer.push_event(ViewportEvent::Wheel {
                delta: glam::vec2(0.0, scroll_y),
                units: ScrollUnits::Pixels,
            });
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

