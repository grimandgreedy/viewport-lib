//! Spike: is it safe to create GPU resources, call `queue.write_buffer`, and
//! encode command buffers on a **worker thread**, as long as every `queue.submit`
//! happens on the **main (device-driving) thread**?
//!
//! This is the load-bearing assumption behind the editor's render worker
//! (`drake-gui`, plan `docs/plans/drake-gui/viewport-render-thread-worker.md`).
//! The NVIDIA Linux Vulkan driver is documented to corrupt the command pushbuffer
//! (NVRM Xid 32, surfacing as a lost device) when GPU work is *submitted* from a
//! non-driving thread. The worker design keeps all submits on the main thread and
//! only moves encoding + `write_buffer` + resource creation to the worker. Whether
//! *those* off-thread operations are also unsafe is not documented, so this soak
//! settles it empirically.
//!
//! Run on the target GPU (especially NVIDIA/Linux):
//! ```sh
//! cargo run --release --example offthread-submit-spike
//! ```
//! Optional first arg: soak seconds (default 30).
//!
//! Verdict:
//! - Prints `SPIKE PASSED` and exits 0 → off-thread write_buffer + create_buffer
//!   with main-thread submit is safe on this GPU; the worker design stands.
//! - Panics / Xid 32 in dmesg / "device lost" → off-thread `write_buffer` is
//!   unsafe here; take the fallback (worker emits write-lists, main replays them
//!   before submit).

use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Buffers written and copies encoded per worker iteration, to load the pushbuffer
/// more like a real frame (many uniform writes + draws) than a single copy would.
const WRITES_PER_FRAME: usize = 64;
const BUFFER_SIZE: u64 = 1024;

fn main() {
    let soak_secs: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(30);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no GPU adapter");
    let info = adapter.get_info();
    println!(
        "adapter: {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    );

    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("no device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // A lost device fires this callback rather than silently wedging the soak.
    device.set_device_lost_callback(Box::new(|reason, msg| {
        eprintln!("DEVICE LOST ({reason:?}): {msg}");
        eprintln!("SPIKE FAILED: off-thread write_buffer / encode is unsafe on this GPU.");
        std::process::exit(2);
    }));

    // worker -> main: one encoded batch per frame. main -> worker: submitted ack,
    // so only one frame is ever in flight (matching the real handshake, which
    // stops the worker overwriting uniforms the main thread has not yet submitted).
    let (buf_tx, buf_rx) = mpsc::channel::<Vec<wgpu::CommandBuffer>>();
    let (ack_tx, ack_rx) = mpsc::channel::<()>();

    let w_device = device.clone();
    let w_queue = queue.clone();
    let worker = std::thread::spawn(move || {
        // A persistent buffer rewritten every frame, like a uniform buffer.
        let uniform = w_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spike_uniform"),
            size: BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut frame: u64 = 0;
        loop {
            let mut batch = Vec::with_capacity(WRITES_PER_FRAME);
            for i in 0..WRITES_PER_FRAME {
                // Off-thread queue write : the operation under test.
                let byte = ((frame as usize + i) & 0xff) as u8;
                w_queue.write_buffer(&uniform, 0, &[byte; BUFFER_SIZE as usize]);
                // Off-thread resource creation + command encoding.
                let dst = w_device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("spike_dst"),
                    size: BUFFER_SIZE,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let mut enc = w_device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.copy_buffer_to_buffer(&uniform, 0, &dst, 0, BUFFER_SIZE);
                batch.push(enc.finish());
            }
            if buf_tx.send(batch).is_err() {
                break; // main hung up : soak over.
            }
            if ack_rx.recv().is_err() {
                break;
            }
            frame += 1;
        }
    });

    let soak = Duration::from_secs(soak_secs);
    let start = Instant::now();
    let mut frames: u64 = 0;
    let mut last_report = Instant::now();
    println!("soaking {soak_secs}s : worker encodes + write_buffers, main submits...");

    while start.elapsed() < soak {
        match buf_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(batch) => {
                // Submit on the driving (main) thread : the one operation the NVIDIA
                // driver requires stay here.
                queue.submit(batch);
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(5)),
                });
                frames += 1;
                if ack_tx.send(()).is_err() {
                    break;
                }
                if last_report.elapsed() >= Duration::from_secs(1) {
                    println!("  {frames} frames, {:.0}s", start.elapsed().as_secs_f32());
                    last_report = Instant::now();
                }
            }
            Err(_) => {
                eprintln!("SPIKE FAILED: worker produced no batch for 5s (stall/deadlock).");
                std::process::exit(3);
            }
        }
    }

    drop(buf_rx); // release the worker's next send.
    let _ = ack_tx.send(()); // unblock a worker parked on the ack.
    let _ = worker.join();

    let total = frames * WRITES_PER_FRAME as u64;
    println!(
        "SPIKE PASSED: {frames} frames ({total} off-thread write_buffer+encode ops) over {:.1}s, no device loss.",
        start.elapsed().as_secs_f32()
    );
    println!(
        "Off-thread write_buffer + create_buffer with main-thread submit is safe on this GPU."
    );
}
