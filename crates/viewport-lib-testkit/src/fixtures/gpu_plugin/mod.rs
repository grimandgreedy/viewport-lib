//! Fixtures for [`GpuPlugin`](viewport_lib::GpuPlugin): GPU work the host
//! submits around the renderer's frame, registered with
//! `ViewportRuntime::add_gpu_plugin` / `with_gpu_plugin` and driven by the
//! host's `pre_prepare` / `post_paint` calls.
//!
//! - [`LoggingGpuPlugin`]: records its lifecycle, returns one command buffer
//!   from `pre_prepare`, and reads the post-paint targets without drawing.

mod logging;

pub use logging::LoggingGpuPlugin;
