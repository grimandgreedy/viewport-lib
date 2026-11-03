//! Fixtures for [`RuntimePlugin`](viewport_lib::RuntimePlugin): CPU work in
//! the runtime's phase loop, registered with `ViewportRuntime::add_plugin` /
//! `with_plugin` and dispatched in ascending priority.
//!
//! - [`LoggingRuntimePlugin`]: records every hook it is given, emits one
//!   typed event and one camera command per step.

mod logging;

pub use logging::{LoggingRuntimePlugin, ProbeEvent};
