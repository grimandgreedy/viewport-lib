//! Device-loss detection.
//!
//! When the GPU device dies mid-frame (driver reset, OS command-buffer
//! watchdog killing an over-long submission, out-of-memory), wgpu reports it
//! through a per-device callback and a log line. An application that
//! installs neither sees no error: submissions become no-ops, buffer
//! readbacks return stale or zero data, and the app appears to freeze or
//! render garbage while running "normally".
//!
//! [`DeviceLostWatcher::install`] wires that callback to a flag you can poll
//! once per frame.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// What the device-lost callback reported.
#[derive(Clone, Debug)]
pub struct DeviceLostInfo {
    /// Why the device was lost. `Unknown` covers driver errors and OS
    /// watchdog kills; `Destroyed` means [`wgpu::Device::destroy`] was
    /// called.
    pub reason: crate::gpu::DeviceLostReason,
    /// The backend's message, if any.
    pub message: String,
}

/// Polls whether the GPU device has been lost.
///
/// Check [`is_lost`](Self::is_lost) once per frame (after your
/// `device.poll` or at the end of the frame) and stop rendering when it
/// returns true: recreate the device and re-upload resources, or report the
/// failure. Continuing to render on a lost device produces no error, just
/// silent no-ops and stale readbacks.
///
/// Cloning is cheap; clones observe the same device.
#[derive(Clone)]
pub struct DeviceLostWatcher {
    inner: Arc<WatcherInner>,
}

struct WatcherInner {
    lost: AtomicBool,
    info: Mutex<Option<DeviceLostInfo>>,
}

impl DeviceLostWatcher {
    /// Install the watcher on `device`.
    ///
    /// wgpu holds a single device-lost callback per device, so this replaces
    /// any callback the application set earlier; if you need your own
    /// handling, install your own callback instead and skip the watcher.
    ///
    /// wgpu delivers the callback when the device is next maintained (a
    /// `device.poll`, a submit, or drop), so a loss becomes visible on the
    /// frame after it happens, not at the exact failing call.
    pub fn install(device: &crate::gpu::Device) -> Self {
        let inner = Arc::new(WatcherInner {
            lost: AtomicBool::new(false),
            info: Mutex::new(None),
        });
        let cb_inner = inner.clone();
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut slot) = cb_inner.info.lock() {
                slot.get_or_insert(DeviceLostInfo { reason, message });
            }
            if reason != crate::gpu::DeviceLostReason::Destroyed {
                cb_inner.lost.store(true, Ordering::Release);
            }
        });
        Self { inner }
    }

    /// True once the device has been lost for any reason other than an
    /// explicit `device.destroy()` call.
    pub fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Acquire)
    }

    /// The callback's report, including losses from `device.destroy()`.
    /// `None` until the callback has fired.
    pub fn info(&self) -> Option<DeviceLostInfo> {
        self.inner.info.lock().ok().and_then(|slot| slot.clone())
    }
}
