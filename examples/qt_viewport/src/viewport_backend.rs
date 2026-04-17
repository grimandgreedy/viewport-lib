//! CXX-Qt bridge: defines a QObject (`ViewportBackend`) exposed to QML.
//!
//! The QObject manages the scene state and viewport rendering. QML calls
//! invokable methods for add/remove objects and camera input. After each
//! render, the pixel data is pushed to a C++ `ViewportImageProvider` so
//! QML can display it via `Image { source: "image://viewport/..." }`.

#[cxx_qt::bridge]
pub mod qobject {
    unsafe extern "C++" {
        include!("cxx-qt-lib/qstring.h");
        type QString = cxx_qt_lib::QString;
    }

    extern "RustQt" {
        #[qobject]
        #[qml_element]
        #[qproperty(i32, frame_counter)]
        #[qproperty(i32, object_count)]
        type ViewportBackend = super::ViewportBackendRust;

        /// Add a box to the scene. Returns the display name.
        #[qinvokable]
        #[cxx_name = "addObject"]
        fn add_object(self: Pin<&mut ViewportBackend>) -> QString;

        /// Remove an object by index.
        #[qinvokable]
        #[cxx_name = "removeObject"]
        fn remove_object(self: Pin<&mut ViewportBackend>, index: i32);

        /// Render the viewport at the given size and push pixels to the
        /// image provider. Call this from QML after layout is known.
        #[qinvokable]
        #[cxx_name = "renderFrame"]
        fn render_frame(self: Pin<&mut ViewportBackend>, width: i32, height: i32);

        /// Orbit the camera by pixel deltas.
        #[qinvokable]
        fn orbit(self: Pin<&mut ViewportBackend>, dx: f32, dy: f32);

        /// Pan the camera by pixel deltas.
        #[qinvokable]
        fn pan(self: Pin<&mut ViewportBackend>, dx: f32, dy: f32, viewport_height: f32);

        /// Zoom the camera by scroll delta.
        #[qinvokable]
        fn zoom(self: Pin<&mut ViewportBackend>, delta: f32);
    }
}

use core::pin::Pin;
use cxx_qt::CxxQtType;
use cxx_qt_lib::QString;

// C++ image provider update function (defined in cpp/image_provider.cpp).
// Declared outside the cxx_qt bridge to avoid pulling QQuickImageProvider
// headers into all generated C++ files.
unsafe extern "C" {
    fn viewport_provider_update(data: *const u8, width: i32, height: i32);
}

const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.001;
const MIN_DISTANCE: f32 = 0.1;
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

pub struct ViewportBackendRust {
    frame_counter: i32,
    object_count: i32,
    scene_renderer: Option<crate::viewport_bridge::SceneRenderer>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    objects: Vec<(u64, [f32; 3])>,
    next_id: u64,
}

impl Default for ViewportBackendRust {
    fn default() -> Self {
        // Create wgpu device.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }));

        let (device, queue, renderer) = if let Ok(adapter) = adapter {
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor { label: Some("qt_viewport"), ..Default::default() },
            ))
            .expect("Failed to create wgpu device");
            let renderer = crate::viewport_bridge::SceneRenderer::new(&device);
            (Some(device), Some(queue), Some(renderer))
        } else {
            eprintln!("Warning: no GPU adapter found, viewport will not render");
            (None, None, None)
        };

        Self {
            frame_counter: 0,
            object_count: 0,
            scene_renderer: renderer,
            device,
            queue,
            objects: Vec::new(),
            next_id: 1,
        }
    }
}

impl qobject::ViewportBackend {
    fn add_object(mut self: Pin<&mut Self>) -> QString {
        let id = self.rust().next_id;
        let n = self.rust().objects.len() as f32;
        let x = (n % 4.0) * 2.0 - 3.0;
        let z = (n / 4.0).floor() * 2.0 - 3.0;
        let name = format!("Box {id}");

        let mut rust = self.as_mut().rust_mut();
        rust.next_id += 1;
        rust.objects.push((id, [x, 0.0, z]));
        drop(rust);

        let count = self.rust().objects.len() as i32;
        self.set_object_count(count);
        QString::from(&name)
    }

    fn remove_object(mut self: Pin<&mut Self>, index: i32) {
        let idx = index as usize;
        if idx < self.rust().objects.len() {
            self.as_mut().rust_mut().objects.remove(idx);
            let count = self.rust().objects.len() as i32;
            self.set_object_count(count);
        }
    }

    fn render_frame(mut self: Pin<&mut Self>, width: i32, height: i32) {
        eprintln!("[qt_viewport] render_frame called: {}x{}", width, height);
        if width <= 0 || height <= 0 {
            eprintln!("[qt_viewport] skipping: invalid size");
            return;
        }
        let (device, queue, has_renderer) = match (
            &self.rust().device,
            &self.rust().queue,
            &self.rust().scene_renderer,
        ) {
            (Some(d), Some(q), Some(_)) => (d as *const _, q as *const _, true),
            _ => {
                eprintln!("[qt_viewport] skipping: no device/queue/renderer");
                return;
            }
        };
        if !has_renderer {
            return;
        }

        // Safety: we hold a Pin<&mut Self> so no one else can access these.
        let objects = self.rust().objects.clone();
        eprintln!("[qt_viewport] rendering {} objects", objects.len());
        let mut rust = self.as_mut().rust_mut();
        let device = unsafe { &*device };
        let queue = unsafe { &*(queue as *const wgpu::Queue) };
        let sr = rust.scene_renderer.as_mut().unwrap();
        let pixels = sr.render(device, queue, width as u32, height as u32, &objects);
        eprintln!("[qt_viewport] got {} bytes of pixels", pixels.len());
        drop(rust);

        // Push pixels to the C++ image provider.
        unsafe {
            viewport_provider_update(
                pixels.as_ptr(),
                width,
                height,
            );
        }
        eprintln!("[qt_viewport] pushed pixels to provider, incrementing counter");

        let counter = self.rust().frame_counter + 1;
        self.set_frame_counter(counter);
    }

    fn orbit(self: Pin<&mut Self>, dx: f32, dy: f32) {
        if let Some(sr) = self.rust_mut().scene_renderer.as_mut() {
            sr.camera.yaw -= dx * ORBIT_SENSITIVITY;
            sr.camera.pitch =
                (sr.camera.pitch + dy * ORBIT_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);
        }
    }

    fn pan(self: Pin<&mut Self>, dx: f32, dy: f32, viewport_height: f32) {
        if let Some(sr) = self.rust_mut().scene_renderer.as_mut() {
            let h = viewport_height.max(1.0);
            let pan_scale = 2.0 * sr.camera.distance * (sr.camera.fov_y / 2.0).tan() / h;
            let right = sr.camera.right();
            let up = sr.camera.up();
            sr.camera.center -= right * dx * pan_scale;
            sr.camera.center += up * dy * pan_scale;
        }
    }

    fn zoom(self: Pin<&mut Self>, delta: f32) {
        if let Some(sr) = self.rust_mut().scene_renderer.as_mut() {
            sr.camera.distance =
                (sr.camera.distance * (1.0 - delta * ZOOM_SENSITIVITY)).max(MIN_DISTANCE);
        }
    }
}
