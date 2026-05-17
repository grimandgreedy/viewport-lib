//! Async job handoff types for runtime plugins.
//!
//! [`JobSlot`] and [`JobSender`] provide a typed result slot shared between a
//! background thread and the runtime frame loop. A plugin creates a linked pair
//! with [`JobSlot::new`], moves the [`JobSender`] to a background thread, and
//! polls the [`JobSlot`] in `collect` calls until a result arrives.
//!
//! # Usage pattern
//!
//! ```rust,ignore
//! use viewport_lib::runtime::{RuntimePlugin, RuntimeStepContext};
//! use viewport_lib::runtime::jobs::{JobSlot, JobPoll};
//! use viewport_lib::runtime::plugin::phase;
//!
//! struct LoaderPlugin {
//!     slot: JobSlot<Vec<u8>>,
//! }
//!
//! impl RuntimePlugin for LoaderPlugin {
//!     fn priority(&self) -> i32 { phase::PREPARE }
//!
//!     fn submit(&mut self, _ctx: &RuntimeStepContext) {
//!         // Start a new job only when the slot is idle.
//!         if self.slot.is_empty() {
//!             let (new_slot, sender) = JobSlot::new();
//!             self.slot = new_slot;
//!             std::thread::spawn(move || {
//!                 // ... load data ...
//!                 sender.complete(vec![1, 2, 3]);
//!             });
//!         }
//!     }
//!
//!     fn collect(&mut self, ctx: &mut RuntimeStepContext) {
//!         match self.slot.take() {
//!             JobPoll::Ready(data) => {
//!                 ctx.resources.insert(LoadedData { data });
//!             }
//!             JobPoll::Failed(msg) => {
//!                 tracing::warn!("load failed: {msg}");
//!             }
//!             _ => {}
//!         }
//!     }
//!
//!     fn step(&mut self, _ctx: &mut RuntimeStepContext) {}
//! }
//! ```
//!
//! # When to use submit/collect vs resources
//!
//! - Use `submit` + `collect` when a plugin does out-of-frame work (disk IO, decompression,
//!   heavy CPU) and needs the result one or more frames later.
//! - Use `collect` alone when the work is light enough to do in-frame after the step loop.
//! - Store the integrated result in [`super::resources::RuntimeResources`] so other plugins
//!   can read it. The `JobSlot` stays in the plugin that owns the job; downstream plugins
//!   read from resources, not from the slot directly.
//! - To replace stale data, remove the old resource and insert the new one in the same
//!   `collect` call. Other plugins that run `collect` later in priority order will see
//!   the updated value immediately.
//!
//! # Cross-plugin jobs
//!
//! If more than one plugin needs to check or fulfill the same job, store the
//! [`JobSlot`] in [`super::resources::RuntimeResources`]. The owning plugin
//! inserts it; other plugins read it via `resources.get::<JobSlot<T>>()`.

use std::sync::{Arc, Mutex};

/// Result of polling a [`JobSlot`].
pub enum JobPoll<T> {
    /// No job has been started. The slot is idle.
    Empty,
    /// A job is running and has not finished yet.
    Pending,
    /// The job completed successfully. The slot is reset to empty.
    Ready(T),
    /// The job failed. The slot is reset to empty.
    Failed(String),
    /// The job was cancelled (or the sender was dropped). The slot is reset to empty.
    Cancelled,
}

#[derive(Debug, PartialEq)]
enum SlotState<T> {
    Empty,
    Pending,
    Ready(T),
    Failed(String),
    Cancelled,
}

/// Typed result slot for handing off async work to the runtime frame loop.
///
/// Create a linked pair with [`JobSlot::new`]. The plugin holds the slot; the
/// [`JobSender`] is sent to a background thread. Poll the slot each frame in
/// `collect` until a result arrives.
///
/// `JobSlot<T>` is `Send + 'static` when `T: Send + 'static`, so it can be
/// stored in [`super::resources::RuntimeResources`] for cross-plugin access.
pub struct JobSlot<T: Send + 'static> {
    state: Arc<Mutex<SlotState<T>>>,
}

// Manual Send/Sync impls: Arc<Mutex<SlotState<T>>> is Send + Sync when T: Send.
// This is already guaranteed by the trait bounds, but spelled out for clarity.
unsafe impl<T: Send + 'static> Send for JobSlot<T> {}
unsafe impl<T: Send + 'static> Sync for JobSlot<T> {}

impl<T: Send + 'static> Default for JobSlot<T> {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(SlotState::Empty)),
        }
    }
}

impl<T: Send + 'static> JobSlot<T> {
    /// Create an idle slot with no job running.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a linked (slot, sender) pair.
    ///
    /// The slot starts in the `Pending` state. The sender signals completion,
    /// failure, or cancellation from a background thread.
    pub fn new() -> (Self, JobSender<T>) {
        let state = Arc::new(Mutex::new(SlotState::Pending));
        let slot = Self { state: state.clone() };
        let sender = JobSender { state, consumed: false };
        (slot, sender)
    }

    /// Poll the slot for a result.
    ///
    /// If the job has finished (`Ready`, `Failed`, or `Cancelled`), the result
    /// is returned and the slot resets to empty, making it ready for a new job.
    /// Returns `Pending` while the background thread is still running.
    pub fn take(&mut self) -> JobPoll<T> {
        let mut guard = self.state.lock().unwrap();
        match &*guard {
            SlotState::Empty => JobPoll::Empty,
            SlotState::Pending => JobPoll::Pending,
            SlotState::Ready(_) => {
                let SlotState::Ready(v) = std::mem::replace(&mut *guard, SlotState::Empty)
                    else { unreachable!() };
                JobPoll::Ready(v)
            }
            SlotState::Failed(_) => {
                let SlotState::Failed(msg) = std::mem::replace(&mut *guard, SlotState::Empty)
                    else { unreachable!() };
                JobPoll::Failed(msg)
            }
            SlotState::Cancelled => {
                *guard = SlotState::Empty;
                JobPoll::Cancelled
            }
        }
    }

    /// Returns `true` if no job is running and no result is waiting.
    pub fn is_empty(&self) -> bool {
        matches!(*self.state.lock().unwrap(), SlotState::Empty)
    }

    /// Returns `true` if a job is running and has not yet finished.
    pub fn is_pending(&self) -> bool {
        matches!(*self.state.lock().unwrap(), SlotState::Pending)
    }

    /// Returns `true` if a result is ready to take.
    pub fn is_ready(&self) -> bool {
        matches!(*self.state.lock().unwrap(), SlotState::Ready(_))
    }
}

/// Send end of a [`JobSlot`].
///
/// Created by [`JobSlot::new`]. Move this to a background thread and call
/// [`complete`](Self::complete), [`fail`](Self::fail), or [`cancel`](Self::cancel)
/// when work finishes. Dropping the sender without signaling transitions the
/// slot to `Cancelled`.
pub struct JobSender<T: Send + 'static> {
    state: Arc<Mutex<SlotState<T>>>,
    consumed: bool,
}

unsafe impl<T: Send + 'static> Send for JobSender<T> {}

impl<T: Send + 'static> JobSender<T> {
    /// Signal successful completion with `result`.
    pub fn complete(mut self, result: T) {
        *self.state.lock().unwrap() = SlotState::Ready(result);
        self.consumed = true;
    }

    /// Signal failure with an error message.
    pub fn fail(mut self, msg: impl Into<String>) {
        *self.state.lock().unwrap() = SlotState::Failed(msg.into());
        self.consumed = true;
    }

    /// Signal that the job was cancelled.
    pub fn cancel(mut self) {
        *self.state.lock().unwrap() = SlotState::Cancelled;
        self.consumed = true;
    }
}

impl<T: Send + 'static> Drop for JobSender<T> {
    fn drop(&mut self) {
        if !self.consumed {
            *self.state.lock().unwrap() = SlotState::Cancelled;
        }
    }
}
