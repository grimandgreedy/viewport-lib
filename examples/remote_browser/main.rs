//! Browser-based remote viewport viewer.
//!
//! Runs two servers:
//!   http://localhost:3000  --  serves the HTML client page
//!   ws://localhost:3001    --  WebSocket: receives input, streams JPEG frames
//!
//! Usage:
//!   cargo run --release --example remote-browser
//!   Open http://localhost:3000 in a browser.
//!
//! Navigation:
//!   Left drag / Middle drag  : orbit
//!   Right drag               : pan
//!   Scroll                   : zoom

mod renderer;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tungstenite::{Message, WebSocket, accept};
use viewport_lib::{
    ButtonState, Camera, MouseButton, OrbitCameraController, ScrollUnits, ViewportContext,
    ViewportEvent,
};

use renderer::{RenderFrame, RenderRequest};

const HTML_ADDR: &str = "127.0.0.1:3000";
const WS_ADDR:   &str = "127.0.0.1:3001";

fn main() {
    let (req_tx, req_rx) = mpsc::channel::<RenderRequest>();
    let (frame_tx, frame_rx) = mpsc::channel::<RenderFrame>();

    std::thread::spawn(move || renderer::run(req_rx, frame_tx));

    let html_listener = TcpListener::bind(HTML_ADDR)
        .unwrap_or_else(|e| panic!("could not bind HTML port {HTML_ADDR}: {e}"));
    println!("[html]      listening on http://{HTML_ADDR}");

    let ws_listener = TcpListener::bind(WS_ADDR)
        .unwrap_or_else(|e| panic!("could not bind WebSocket port {WS_ADDR}: {e}"));
    println!("[websocket] listening on ws://{WS_ADDR}");

    println!();
    println!("Open http://{HTML_ADDR} in your browser");
    println!("Press Ctrl+C to stop");
    println!();

    std::thread::spawn(move || serve_html(html_listener));
    serve_websocket(ws_listener, req_tx, frame_rx);
}

// ---------------------------------------------------------------------------
// HTML server
// ---------------------------------------------------------------------------

fn serve_html(listener: TcpListener) {
    let html = include_str!("client.html");
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        html.len(),
        html
    );
    for stream in listener.incoming().flatten() {
        let mut stream = stream;
        let mut buf = [0u8; 1024];
        let _ = stream.read(&mut buf);
        let _ = stream.write_all(response.as_bytes());
    }
}

// ---------------------------------------------------------------------------
// WebSocket server
// ---------------------------------------------------------------------------

fn serve_websocket(
    listener: TcpListener,
    req_tx: Sender<RenderRequest>,
    frame_rx: Receiver<RenderFrame>,
) {
    for stream in listener.incoming().flatten() {
        let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
        match accept(stream) {
            Ok(ws) => {
                println!("[websocket] client connected: {peer}");
                handle_client(ws, &req_tx, &frame_rx);
                println!("[websocket] client disconnected: {peer}");
            }
            Err(e) => eprintln!("[websocket] handshake failed from {peer}: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-connection render loop
// ---------------------------------------------------------------------------

fn handle_client(
    mut ws: WebSocket<TcpStream>,
    req_tx: &Sender<RenderRequest>,
    frame_rx: &Receiver<RenderFrame>,
) {
    ws.get_ref()
        .set_read_timeout(Some(Duration::from_millis(1)))
        .ok();

    let mut camera = Camera { distance: 10.0, ..Camera::default() };
    let mut controller = OrbitCameraController::viewport_primitives();
    let mut width = 1280u32;
    let mut height = 720u32;

    let mut frames: u32 = 0;
    let mut last_log = Instant::now();

    loop {
        controller.begin_frame(ViewportContext {
            hovered: true,
            focused: true,
            viewport_size: [width as f32, height as f32],
        });

        // Drain all pending input messages.
        loop {
            match ws.read() {
                Ok(Message::Text(text)) => {
                    match serde_json::from_str::<InputEvent>(&text) {
                        Ok(InputEvent::Resize { width: w, height: h }) => {
                            width  = w.max(1);
                            height = h.max(1);
                        }
                        Ok(event) => {
                            if let Some(vp) = viewport_event(event) {
                                controller.push_event(vp);
                            }
                        }
                        Err(_) => {}
                    }
                }
                Ok(Message::Close(_)) => return,
                Ok(_) => {}
                Err(tungstenite::Error::Io(ref e))
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    break;
                }
                Err(_) => return,
            }
        }

        controller.apply_to_camera(&mut camera);
        camera.set_aspect_ratio(width as f32, height as f32);

        if req_tx
            .send(RenderRequest { camera: camera.clone(), width, height })
            .is_err()
        {
            return;
        }

        let frame = match frame_rx.recv() {
            Ok(f) => f,
            Err(_) => return,
        };

        let jpeg = encode_jpeg(&frame.pixels, frame.width, frame.height);
        if ws.send(Message::Binary(jpeg.into())).is_err() {
            return;
        }

        frames += 1;
        if last_log.elapsed() >= Duration::from_secs(3) {
            let fps = frames as f32 / last_log.elapsed().as_secs_f32();
            println!("[websocket] {fps:.0} fps  ({width}x{height})");
            frames = 0;
            last_log = Instant::now();
        }
    }
}

// ---------------------------------------------------------------------------
// Input event types (JSON from browser)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputEvent {
    PointerMoved { x: f32, y: f32 },
    MouseButton { button: String, state: String },
    Wheel { dx: f32, dy: f32 },
    Resize { width: u32, height: u32 },
}

fn viewport_event(event: InputEvent) -> Option<ViewportEvent> {
    match event {
        InputEvent::PointerMoved { x, y } => {
            Some(ViewportEvent::PointerMoved { position: glam::Vec2::new(x, y) })
        }
        InputEvent::MouseButton { button, state } => {
            let btn = match button.as_str() {
                "left"   => MouseButton::Left,
                "right"  => MouseButton::Right,
                "middle" => MouseButton::Middle,
                _ => return None,
            };
            let state = match state.as_str() {
                "pressed"  => ButtonState::Pressed,
                "released" => ButtonState::Released,
                _ => return None,
            };
            Some(ViewportEvent::MouseButton { button: btn, state })
        }
        InputEvent::Wheel { dx, dy } => Some(ViewportEvent::Wheel {
            delta: glam::Vec2::new(dx, dy),
            units: ScrollUnits::Pixels,
        }),
        InputEvent::Resize { .. } => None,
    }
}

// ---------------------------------------------------------------------------
// JPEG encoding
// ---------------------------------------------------------------------------

fn encode_jpeg(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    // RGBA -> RGB: JPEG has no alpha channel.
    let rgb: Vec<u8> = pixels
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85)
        .encode(&rgb, width, height, image::ExtendedColorType::Rgb8)
        .expect("JPEG encode failed");
    buf
}
