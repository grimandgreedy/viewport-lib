//! A growable GPU buffer addressed in elements, with a ranged write.
//!
//! The storage half of a partial update. An item type that holds content on a
//! consumer's behalf keeps one of these per channel (separate parallel buffers)
//! or one for the whole record (an interleaved instance buffer), and writes the
//! part that changed instead of rebuilding.

use crate::error::{ViewportError, ViewportResult};

/// One growable GPU buffer, addressed in fixed-size elements.
///
/// Three things it keeps that a bare `wgpu::Buffer` does not: how many elements
/// the allocation holds, how many of them are live, and whether the allocation
/// has been replaced since anything was built against it.
///
/// # Writing
///
/// [`write_range`](Self::write_range) writes at an element offset and touches
/// nothing else. It does not grow the buffer: a write past
/// [`capacity`](Self::capacity) is an error, because growing behind a call the
/// caller believes is cheap is the cost this type exists to remove. Call
/// [`reserve`](Self::reserve) first, which is the step that is allowed to be
/// expensive. A write that runs past the live count raises it, so an appending
/// feed reserves once and then only writes.
///
/// # Growth
///
/// [`reserve`](Self::reserve) doubles or fits, whichever is larger, and never
/// shrinks. It preserves what is already there by copying on the device, which
/// is why it needs a queue. A caller about to rewrite everything anyway wants
/// [`resize_discarding`](Self::resize_discarding) instead: the copy would be
/// pure waste, and that is the shape a per-frame crowd rebuild actually has.
///
/// # Rebinding
///
/// Growing allocates a new buffer, so any bind group holding the old one is
/// stale. Rather than a callback, the buffer carries a
/// [`generation`](Self::generation) that changes whenever the allocation is
/// replaced. Record it beside a bind group and compare before use:
///
/// ```no_run
/// # use viewport_lib::resources::ContentBuffer;
/// # struct Entry { bind_group: viewport_lib::gpu::BindGroup, built_at: u64 }
/// # fn rebuild(b: &ContentBuffer) -> viewport_lib::gpu::BindGroup { unimplemented!() }
/// # fn f(buffer: &ContentBuffer, entry: &mut Entry) {
/// if entry.built_at != buffer.generation() {
///     entry.bind_group = rebuild(buffer);
///     entry.built_at = buffer.generation();
/// }
/// # }
/// ```
pub struct ContentBuffer {
    buffer: crate::gpu::Buffer,
    label: &'static str,
    usage: crate::gpu::BufferUsages,
    stride_bytes: u32,
    capacity: u32,
    len: u32,
    generation: u64,
}

impl ContentBuffer {
    /// Allocate a buffer for `capacity` elements of `stride_bytes` each.
    ///
    /// `COPY_DST` is added to `usage` whatever is passed, because a buffer that
    /// cannot be written into is not one of these. Starts empty: capacity is
    /// what is allocated, [`len`](Self::len) is what draws.
    ///
    /// A zero capacity still allocates one element, because wgpu rejects a
    /// zero-sized buffer and a store that holds an empty channel still has to
    /// put something in its bind group.
    pub fn new(
        device: &crate::gpu::Device,
        label: &'static str,
        usage: crate::gpu::BufferUsages,
        stride_bytes: u32,
        capacity: u32,
    ) -> Self {
        assert!(stride_bytes > 0, "content buffer stride must be non-zero");
        let usage = usage | crate::gpu::BufferUsages::COPY_DST;
        let capacity = capacity.max(1);
        let buffer = Self::alloc(device, label, usage, stride_bytes, capacity);
        Self {
            buffer,
            label,
            usage,
            stride_bytes,
            capacity,
            len: 0,
            generation: 0,
        }
    }

    fn alloc(
        device: &crate::gpu::Device,
        label: &'static str,
        usage: crate::gpu::BufferUsages,
        stride_bytes: u32,
        capacity: u32,
    ) -> crate::gpu::Buffer {
        device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some(label),
            size: capacity as u64 * stride_bytes as u64,
            // COPY_SRC so a preserving grow can copy the old contents forward.
            usage: usage | crate::gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// The buffer to bind. Changes identity when the allocation grows; see
    /// [`generation`](Self::generation).
    pub fn buffer(&self) -> &crate::gpu::Buffer {
        &self.buffer
    }

    /// Bytes per element, fixed at construction.
    pub fn stride_bytes(&self) -> u32 {
        self.stride_bytes
    }

    /// Elements the current allocation holds.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Live elements: what a draw should read, and what
    /// [`live_bytes`](Self::live_bytes) counts.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether any element is live. The allocation may still be large.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Changes whenever the allocation is replaced, which is the only time a
    /// bind group holding this buffer goes stale. Unchanged by a write.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Bytes allocated, live or not.
    ///
    /// This is the figure to report from `ItemTypePlugin::resident_bytes`:
    /// reserved capacity occupies VRAM whether or not anything is drawn from
    /// it, and a consumer budgeting against a ceiling is misled by the live
    /// figure.
    pub fn allocated_bytes(&self) -> u64 {
        self.capacity as u64 * self.stride_bytes as u64
    }

    /// Bytes the live elements occupy. Always at most
    /// [`allocated_bytes`](Self::allocated_bytes).
    pub fn live_bytes(&self) -> u64 {
        self.len as u64 * self.stride_bytes as u64
    }

    /// Set the live element count. Errors past the capacity rather than
    /// clamping, so a miscounted feed is a failure rather than silently short
    /// geometry.
    pub fn set_len(&mut self, len: u32) -> ViewportResult<()> {
        if len > self.capacity {
            return Err(ViewportError::ContentBufferWriteOutOfRange {
                first_element: 0,
                element_count: len,
                capacity: self.capacity,
                stride_bytes: self.stride_bytes,
            });
        }
        self.len = len;
        Ok(())
    }

    /// Overwrite `data.len() / stride` elements starting at `first_element`.
    ///
    /// The partial update. Writes only the named window, allocates nothing, and
    /// leaves the buffer's identity alone, so no bind group is invalidated.
    /// Raises the live count when the window ends past it.
    ///
    /// Errors when the window runs past [`capacity`](Self::capacity), or when
    /// `data` is not a whole number of elements. An empty write is a no-op.
    pub fn write_range(
        &mut self,
        queue: &crate::gpu::Queue,
        first_element: u32,
        data: &[u8],
    ) -> ViewportResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        let stride = self.stride_bytes as usize;
        if data.len() % stride != 0 {
            return Err(ViewportError::ContentBufferWriteOutOfRange {
                first_element,
                element_count: 0,
                capacity: self.capacity,
                stride_bytes: self.stride_bytes,
            });
        }
        let element_count = (data.len() / stride) as u32;
        let end = first_element.saturating_add(element_count);
        if end > self.capacity {
            return Err(ViewportError::ContentBufferWriteOutOfRange {
                first_element,
                element_count,
                capacity: self.capacity,
                stride_bytes: self.stride_bytes,
            });
        }
        queue.write_buffer(
            &self.buffer,
            first_element as u64 * self.stride_bytes as u64,
            data,
        );
        self.len = self.len.max(end);
        Ok(())
    }

    /// Grow to hold at least `capacity` elements, preserving what is there.
    ///
    /// Doubles or fits, whichever is larger, and never shrinks, so an appending
    /// feed pays a reallocation a logarithmic number of times rather than per
    /// element. Returns `true` when the allocation was replaced, which is also
    /// when [`generation`](Self::generation) changed and any bind group holding
    /// the buffer needs rebuilding.
    ///
    /// The contents are copied on the device, so this is not free. A caller
    /// that is about to overwrite everything should call
    /// [`resize_discarding`](Self::resize_discarding) instead.
    pub fn reserve(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        capacity: u32,
    ) -> bool {
        if capacity <= self.capacity {
            return false;
        }
        let capacity = capacity.max(self.capacity.saturating_mul(2));
        let new_buffer = Self::alloc(device, self.label, self.usage, self.stride_bytes, capacity);
        let live = self.live_bytes();
        if live > 0 {
            let mut encoder =
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("content_buffer_grow"),
                });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, live);
            queue.submit(std::iter::once(encoder.finish()));
        }
        self.buffer = new_buffer;
        self.capacity = capacity;
        self.generation += 1;
        true
    }

    /// Reallocate to hold at least `capacity` elements without preserving
    /// anything, and reset the live count to zero.
    ///
    /// For the caller who grows and then rewrites the whole buffer in the same
    /// frame, where a preserving copy moves bytes that are about to be
    /// overwritten. Same growth policy as [`reserve`](Self::reserve), and the
    /// same rebinding consequence. Returns `true` when the allocation was
    /// replaced.
    pub fn resize_discarding(&mut self, device: &crate::gpu::Device, capacity: u32) -> bool {
        self.len = 0;
        if capacity <= self.capacity {
            return false;
        }
        let capacity = capacity.max(self.capacity.saturating_mul(2));
        self.buffer = Self::alloc(device, self.label, self.usage, self.stride_bytes, capacity);
        self.capacity = capacity;
        self.generation += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu;

    fn device() -> Option<(gpu::Device, gpu::Queue)> {
        crate::resources::test_support::try_make_device()
    }

    fn buffer(device: &gpu::Device, stride: u32, capacity: u32) -> ContentBuffer {
        ContentBuffer::new(
            device,
            "test_content_buffer",
            gpu::BufferUsages::STORAGE,
            stride,
            capacity,
        )
    }

    /// Read `count` elements back, so the tests assert on what the GPU holds
    /// rather than on what the calls returned.
    fn read_back(device: &gpu::Device, queue: &gpu::Queue, cb: &ContentBuffer) -> Vec<u8> {
        let size = cb.allocated_bytes();
        let staging = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("content_buffer_readback"),
            size,
            usage: gpu::BufferUsages::COPY_DST | gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&gpu::CommandEncoderDescriptor {
            label: Some("content_buffer_readback"),
        });
        encoder.copy_buffer_to_buffer(cb.buffer(), 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(gpu::MapMode::Read, |_| {});
        let _ = device.poll(gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        });
        let view = gpu::mapped_range(slice).to_vec();
        staging.unmap();
        view
    }

    /// A write lands at the element offset it names and nowhere else.
    #[test]
    fn a_write_lands_at_its_element_offset() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut cb = buffer(&device, 4, 8);
        cb.write_range(&queue, 0, &[0u8; 32]).unwrap();
        cb.write_range(&queue, 2, &[7u8, 7, 7, 7]).unwrap();

        let bytes = read_back(&device, &queue, &cb);
        assert_eq!(&bytes[8..12], &[7, 7, 7, 7], "element 2 holds the write");
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0], "element 1 is untouched");
        assert_eq!(&bytes[12..16], &[0, 0, 0, 0], "element 3 is untouched");
    }

    /// A write ending past the live count raises it, so an appending feed
    /// reserves once and then only writes.
    #[test]
    fn a_write_past_the_live_count_raises_it() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut cb = buffer(&device, 4, 8);
        assert_eq!(cb.len(), 0, "a fresh buffer draws nothing");
        cb.write_range(&queue, 0, &[1u8; 8]).unwrap();
        assert_eq!(cb.len(), 2);
        cb.write_range(&queue, 5, &[1u8; 4]).unwrap();
        assert_eq!(cb.len(), 6, "the live count covers the end of the window");
        // A write inside the live region does not lower it.
        cb.write_range(&queue, 0, &[2u8; 4]).unwrap();
        assert_eq!(cb.len(), 6);
    }

    /// Growing preserves the live bytes, which is what separates `reserve` from
    /// `resize_discarding`.
    #[test]
    fn reserve_preserves_what_is_there() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut cb = buffer(&device, 4, 4);
        cb.write_range(&queue, 0, &[9u8; 16]).unwrap();
        assert_eq!(cb.len(), 4);

        assert!(cb.reserve(&device, &queue, 5), "growing past 4 reallocates");
        assert!(cb.capacity() >= 8, "double-or-fit, so 4 becomes at least 8");
        assert_eq!(cb.len(), 4, "growing does not change what is live");

        let bytes = read_back(&device, &queue, &cb);
        assert_eq!(&bytes[0..16], &[9u8; 16], "the old contents came forward");
    }

    /// The discarding form does not copy and resets the live count, for the
    /// caller who grows and rewrites in the same breath.
    #[test]
    fn resize_discarding_drops_the_contents() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut cb = buffer(&device, 4, 4);
        cb.write_range(&queue, 0, &[9u8; 16]).unwrap();
        assert!(cb.resize_discarding(&device, 9));
        assert_eq!(cb.len(), 0, "nothing is live after a discarding resize");
        assert!(cb.capacity() >= 9);
    }

    /// Reserving up front means the writes that follow reallocate nothing,
    /// which is the whole ergonomic point of `reserve` existing.
    #[test]
    fn reserving_up_front_then_filling_reallocates_once() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut cb = buffer(&device, 4, 1);
        assert!(cb.reserve(&device, &queue, 64));
        let after_reserve = cb.generation();
        for i in 0..64u32 {
            cb.write_range(&queue, i, &[1u8; 4]).unwrap();
        }
        assert_eq!(
            cb.generation(),
            after_reserve,
            "filling a reserved buffer must not reallocate"
        );
        assert_eq!(cb.len(), 64);
    }

    /// The generation is the rebinding signal: writes leave it alone, growth
    /// changes it, and a no-op reserve does neither.
    #[test]
    fn the_generation_tracks_reallocation_only() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut cb = buffer(&device, 4, 8);
        let start = cb.generation();
        cb.write_range(&queue, 0, &[1u8; 32]).unwrap();
        assert_eq!(cb.generation(), start, "a write invalidates no bind group");

        assert!(!cb.reserve(&device, &queue, 8), "already big enough");
        assert_eq!(cb.generation(), start, "a no-op reserve changes nothing");

        assert!(cb.reserve(&device, &queue, 9));
        assert_ne!(cb.generation(), start, "growth needs a rebind");
    }

    /// Residency is the allocation, not the live part: reserved capacity
    /// occupies VRAM whether or not anything draws from it.
    #[test]
    fn residency_reports_the_allocation() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut cb = buffer(&device, 16, 100);
        assert_eq!(cb.allocated_bytes(), 1600);
        assert_eq!(cb.live_bytes(), 0);
        cb.write_range(&queue, 0, &[0u8; 160]).unwrap();
        assert_eq!(cb.live_bytes(), 160);
        assert_eq!(
            cb.allocated_bytes(),
            1600,
            "the reserved tail is still resident"
        );
    }

    /// The buffer does not grow behind a write. Growing inside a call the
    /// caller believes is cheap is the cost this type exists to remove.
    #[test]
    fn a_write_past_capacity_is_refused_rather_than_grown() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut cb = buffer(&device, 4, 4);
        let generation = cb.generation();
        assert!(
            cb.write_range(&queue, 2, &[0u8; 16]).is_err(),
            "a window running past the end is refused"
        );
        assert!(
            cb.write_range(&queue, 0, &[0u8; 6]).is_err(),
            "six bytes is not a whole number of four-byte elements"
        );
        assert!(
            cb.set_len(5).is_err(),
            "a live count past capacity is wrong"
        );
        assert_eq!(cb.generation(), generation, "and none of that reallocated");
        assert!(
            cb.write_range(&queue, 0, &[]).is_ok(),
            "an empty write is a no-op, not an error"
        );
    }

    /// The shape `viewport-lib-mesh-assembly` hand-rolled: grow to fit a bigger
    /// crowd, then refill wholesale in the same frame.
    ///
    /// That crate arrived at double-or-fit-never-shrink independently under
    /// real load, and its grow does not preserve, because it rewrites
    /// everything immediately afterwards. If expressing it here is not simpler
    /// than what it has, this primitive is the wrong shape.
    #[test]
    fn the_grow_and_refill_shape_expresses_cleanly() {
        let Some((device, queue)) = device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let stride = 64; // one mat4 per instance, as a palette or model array
        let mut models = buffer(&device, stride, 4);

        for instances in [3u32, 7, 40] {
            // Grow if the crowd outgrew the allocation, then write every
            // instance. No copy, because all of it is about to be overwritten.
            models.resize_discarding(&device, instances);
            let payload = vec![1u8; (instances * stride) as usize];
            models.write_range(&queue, 0, &payload).unwrap();
            assert_eq!(models.len(), instances);
            assert!(models.capacity() >= instances);
        }
        // Never shrinks: the 40-instance allocation survives a smaller crowd.
        let big = models.capacity();
        models.resize_discarding(&device, 2);
        assert_eq!(models.capacity(), big, "capacity is not given back");
    }
}
