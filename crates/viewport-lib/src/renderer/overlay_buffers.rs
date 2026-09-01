//! Persistent, grow-on-demand GPU buffers for per-frame overlay geometry.
//!
//! The overlay prepare passes rebuild their vertex data every frame, but the GPU
//! buffer backing it does not need to be reallocated every frame. A `GrowBuffer`
//! keeps one buffer alive across frames, growing it only when a frame needs more
//! room than the current capacity and overwriting it in place with
//! `queue.write_buffer` otherwise. In steady state (a UI whose vertex count has
//! settled) there is no per-frame buffer allocation at all.
//!
//! `write` hands back a clone of the wgpu buffer handle, which is Arc-backed and
//! cheap, so callers store it exactly as they stored the freshly created buffer
//! before. Overwriting a buffer a prior frame may still be reading is safe:
//! `queue.write_buffer` is ordered on the queue timeline, so the write lands
//! after earlier submissions that read the old contents.

/// A vertex buffer that persists across frames and grows only when a frame needs
/// more capacity than it currently has.
pub(crate) struct GrowBuffer {
    buf: Option<crate::gpu::Buffer>,
    capacity: u64,
    label: &'static str,
}

impl GrowBuffer {
    /// A `VERTEX | COPY_DST` grow buffer with no allocation yet.
    pub(crate) fn vertex(label: &'static str) -> Self {
        Self {
            buf: None,
            capacity: 0,
            label,
        }
    }

    /// Ensure capacity for `verts`, upload them at offset 0, and return a handle
    /// to the buffer. Reallocates only when the current buffer is too small;
    /// otherwise the existing allocation is reused and overwritten. The tail past
    /// the written range is left stale and is never read, since draws bound the
    /// range by vertex count.
    pub(crate) fn write<T: bytemuck::Pod>(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        verts: &[T],
    ) -> crate::gpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(verts);
        let needed = bytes.len() as u64;
        if self.buf.is_none() || needed > self.capacity {
            let cap = grow_capacity(needed, self.capacity);
            self.buf = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(self.label),
                size: cap,
                usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = cap;
        }
        let buf = self.buf.as_ref().unwrap();
        queue.write_buffer(buf, 0, bytes);
        buf.clone()
    }
}

/// Grow the capacity to hold at least `needed` bytes, in 1.5x steps from a small
/// floor so tiny overlays do not thrash and large ones settle in a few frames.
/// Kept a multiple of 4 to satisfy the copy alignment.
fn grow_capacity(needed: u64, current: u64) -> u64 {
    let mut cap = current.max(4096);
    while cap < needed {
        cap += cap / 2;
    }
    (cap + 3) & !3
}
