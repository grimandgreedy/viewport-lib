//! Where a prepare/render pass sends its finished command buffers.
//!
//! Historically every pass called `queue.submit` the moment it finished encoding.
//! That couples encoding (pure CPU, safe on any thread) to submission, which is
//! not: the NVIDIA Linux Vulkan driver corrupts the command pushbuffer (NVRM Xid
//! 32, surfacing as a lost device) when GPU work is submitted from a thread other
//! than the one driving the device. To let the CPU-side prepare run on a worker
//! while submission stays on the driving thread, passes push their command buffers
//! into a [`SubmitSink`] instead of submitting directly.
//!
//! [`SubmitSink::Inline`] preserves the original behaviour exactly (submit on
//! push), so existing single-threaded callers are unaffected. [`SubmitSink::Deferred`]
//! collects the buffers so the caller can submit them, in order, on the driving
//! thread.

/// A destination for finished command buffers: submit now, or collect for later.
pub enum SubmitSink<'q> {
    /// Submit each buffer immediately on the given queue: the historical
    /// behaviour, unchanged, for callers already on the device-driving thread.
    Inline(&'q crate::gpu::Queue),
    /// Collect buffers in submission order so the caller can submit them all on
    /// the driving thread once the (worker-side) encoding is done.
    Deferred(Vec<crate::gpu::CommandBuffer>),
}

impl<'q> SubmitSink<'q> {
    /// A sink that submits on `queue` the moment a buffer is pushed.
    pub fn inline(queue: &'q crate::gpu::Queue) -> Self {
        Self::Inline(queue)
    }

    /// A sink that collects buffers for the caller to submit later.
    pub fn deferred() -> Self {
        Self::Deferred(Vec::new())
    }

    /// Hand one finished command buffer to the sink. In [`Inline`](Self::Inline)
    /// mode it is submitted at once; in [`Deferred`](Self::Deferred) mode it is
    /// appended, preserving submission order.
    pub fn push(&mut self, buffer: crate::gpu::CommandBuffer) {
        match self {
            Self::Inline(queue) => {
                queue.submit(std::iter::once(buffer));
            }
            Self::Deferred(buffers) => buffers.push(buffer),
        }
    }

    /// Hand several finished command buffers to the sink in one go. In inline mode
    /// they are submitted as a single ordered submission, matching a direct
    /// `queue.submit(bufs)`.
    pub fn extend(&mut self, buffers: impl IntoIterator<Item = crate::gpu::CommandBuffer>) {
        match self {
            Self::Inline(queue) => {
                queue.submit(buffers);
            }
            Self::Deferred(collected) => collected.extend(buffers),
        }
    }

    /// Take the collected buffers. Empty for an [`Inline`](Self::Inline) sink,
    /// whose buffers were already submitted.
    pub fn into_buffers(self) -> Vec<crate::gpu::CommandBuffer> {
        match self {
            Self::Inline(_) => Vec::new(),
            Self::Deferred(buffers) => buffers,
        }
    }
}
