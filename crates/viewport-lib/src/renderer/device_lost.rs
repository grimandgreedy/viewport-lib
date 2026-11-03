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
//! once per frame. `ViewportRenderer::try_with_config` installs one itself
//! unless the config passes a shared one.

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

impl DeviceLostInfo {
    /// The error calls on a lost renderer return.
    pub(crate) fn to_error(&self) -> crate::ViewportError {
        crate::ViewportError::DeviceLost {
            reason: format!("{:?}", self.reason),
            message: self.message.clone(),
        }
    }
}

/// Polls whether the GPU device has been lost.
///
/// Check [`is_lost`](Self::is_lost) once per frame (after your
/// `device.poll` or at the end of the frame) and stop rendering when it
/// returns true: recreate the device and re-upload resources, or report the
/// failure. Continuing to render on a lost device produces no error, just
/// silent no-ops and stale readbacks.
///
/// wgpu keeps one lost callback per device, so a host that runs several
/// renderers on one device, or wants the flag itself, installs one watcher
/// and passes it to each renderer with
/// [`RendererConfig::with_device_lost_watcher`](crate::RendererConfig::with_device_lost_watcher).
///
/// Cloning is cheap; clones observe the same device.
#[derive(Clone)]
pub struct DeviceLostWatcher {
    inner: Arc<WatcherInner>,
}

impl std::fmt::Debug for DeviceLostWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceLostWatcher")
            .field("lost", &self.is_lost())
            .field("info", &self.info())
            .finish()
    }
}

struct WatcherInner {
    /// Lost for a reason other than `Destroyed`.
    lost: AtomicBool,
    /// Lost for any reason, `Destroyed` included.
    ended: AtomicBool,
    info: Mutex<Option<DeviceLostInfo>>,
}

impl DeviceLostWatcher {
    fn empty() -> Self {
        Self {
            inner: Arc::new(WatcherInner {
                lost: AtomicBool::new(false),
                ended: AtomicBool::new(false),
                info: Mutex::new(None),
            }),
        }
    }

    /// Install the watcher on `device`.
    ///
    /// wgpu holds a single device-lost callback per device, so this replaces
    /// any callback set earlier, including the one a renderer installs for
    /// itself. Install one watcher per device and share it.
    ///
    /// wgpu delivers the callback when the device is next maintained (a
    /// `device.poll`, a submit, or drop), so a loss becomes visible on the
    /// frame after it happens, not at the exact failing call.
    pub fn install(device: &crate::gpu::Device) -> Self {
        let watcher = Self::empty();
        let cb = watcher.clone();
        device.set_device_lost_callback(move |reason, message| {
            cb.record(reason, message);
        });
        watcher
    }

    /// A watcher on no device, for resources built outside a renderer. It
    /// reports lost only once something marks it.
    pub(crate) fn detached() -> Self {
        Self::empty()
    }

    fn record(&self, reason: crate::gpu::DeviceLostReason, message: String) {
        if let Ok(mut slot) = self.inner.info.lock() {
            slot.get_or_insert(DeviceLostInfo { reason, message });
        }
        if reason != crate::gpu::DeviceLostReason::Destroyed {
            self.inner.lost.store(true, Ordering::Release);
        }
        self.inner.ended.store(true, Ordering::Release);
    }

    /// True once the device has been lost for any reason other than an
    /// explicit `device.destroy()` call.
    pub fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Acquire)
    }

    /// True once the device has been lost for any reason, an explicit
    /// `device.destroy()` included. The renderer goes by this: a destroyed
    /// device draws nothing either.
    pub fn is_lost_or_destroyed(&self) -> bool {
        self.inner.ended.load(Ordering::Acquire)
    }

    /// The callback's report, including losses from `device.destroy()`.
    /// `None` until the callback has fired.
    pub fn info(&self) -> Option<DeviceLostInfo> {
        self.inner.info.lock().ok().and_then(|slot| slot.clone())
    }
}

/// Whether an error captured by an error scope says the device is lost,
/// rather than reporting a validation failure.
pub(crate) fn is_device_lost_error(error: &crate::gpu::Error) -> bool {
    error
        .to_string()
        .to_ascii_lowercase()
        .contains("device is lost")
}

/// The error for a lost device an error scope reported.
pub(crate) fn scope_lost_error(error: &crate::gpu::Error) -> crate::ViewportError {
    crate::ViewportError::DeviceLost {
        reason: format!("{:?}", crate::gpu::DeviceLostReason::Unknown),
        message: error.to_string(),
    }
}
