//! DeviceLostWatcher: the callback path we can trigger deterministically is
//! an explicit `device.destroy()`. A real loss (driver reset, watchdog kill)
//! takes the same callback with `DeviceLostReason::Unknown`, which is what
//! flips `is_lost()`.

use viewport_lib::DeviceLostWatcher;
use viewport_lib::wgpu;
use viewport_lib_testkit::headless_device;

#[test]
fn destroy_reports_through_watcher() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let watcher = DeviceLostWatcher::install(&device);
    assert!(!watcher.is_lost());
    assert!(watcher.info().is_none());

    device.destroy();
    // The callback is delivered on maintain, not at the destroy call.
    let _ = device.poll(wgpu::PollType::Poll);

    let info = watcher
        .info()
        .expect("device-lost callback did not fire after destroy + poll");
    assert_eq!(info.reason, wgpu::DeviceLostReason::Destroyed);
    // An explicit destroy is not a loss.
    assert!(!watcher.is_lost());
}
