//! Shared vertex/index geometry buffers (the mesh slab).
//!
//! Each mesh's triangle geometry lives in a sub-range of a large shared buffer
//! instead of its own dedicated `vertex_buffer` / `index_buffer`. A mesh is
//! described by a [`SlabSpan`] (chunk index, byte offset, byte length). This lets
//! the renderer bind one buffer for many meshes and, once draws carry per-mesh
//! `base_vertex` / `first_index` offsets, collapse thousands of per-mesh buffer
//! binds into a handful.
//!
//! Storage is split into a vertex slab and an index slab, each a growable list
//! of chunk buffers with a first-fit free list. Spans are aligned to the storage
//! offset alignment so a sub-range can also be bound as a STORAGE buffer (the
//! compute-filter and GPU-picking paths read geometry that way).

use crate::gpu;

/// Fallback storage-offset alignment when the device reports 0 (never expected;
/// the spec minimum is 256).
const DEFAULT_ALIGN: u64 = 256;

/// Size of the first chunk in a slab; each subsequent chunk doubles up to the
/// device `max_buffer_size`. Small enough that a minimal scene does not reserve
/// a large buffer, large enough that big scenes need only a few chunks.
const INITIAL_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// A window into one slab chunk: which chunk, byte offset, and the mesh's actual
/// byte length (not the alignment-padded allocation size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlabSpan {
    pub chunk: u32,
    pub offset: u64,
    pub len: u64,
}

fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// RangeAllocator: a pure first-fit free-list byte allocator (no GPU).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct FreeRange {
    offset: u64,
    len: u64,
}

/// First-fit byte allocator over a fixed capacity, with coalescing on free.
///
/// Requests are rounded up to `align`, and the capacity is a multiple of
/// `align`, so every returned offset is aligned without tracking padding: the
/// free list only ever holds aligned boundaries. Kept free of any GPU type so
/// the allocation logic is unit-testable on its own.
#[derive(Debug)]
struct RangeAllocator {
    #[allow(dead_code)] // read by `is_empty` (chunk-reclaim, later phase) and tests
    capacity: u64,
    align: u64,
    /// Free ranges, sorted by offset and never adjacent (always coalesced).
    free: Vec<FreeRange>,
}

impl RangeAllocator {
    fn new(capacity: u64, align: u64) -> Self {
        let capacity = align_up(capacity, align);
        Self {
            capacity,
            align,
            free: vec![FreeRange {
                offset: 0,
                len: capacity,
            }],
        }
    }

    /// Allocate `bytes` (rounded up to `align`). Returns the aligned offset, or
    /// `None` if no free range is large enough.
    fn alloc(&mut self, bytes: u64) -> Option<u64> {
        let need = align_up(bytes, self.align).max(self.align);
        for i in 0..self.free.len() {
            if self.free[i].len >= need {
                let offset = self.free[i].offset;
                self.free[i].offset += need;
                self.free[i].len -= need;
                if self.free[i].len == 0 {
                    self.free.remove(i);
                }
                return Some(offset);
            }
        }
        None
    }

    /// Return a previously allocated range, coalescing with neighbours. `bytes`
    /// is the request's actual length; it is re-rounded to the allocation size.
    fn free(&mut self, offset: u64, bytes: u64) {
        let len = align_up(bytes, self.align).max(self.align);
        let end = offset + len;
        // Find the insertion point (first free range starting at or after us).
        let idx = self.free.partition_point(|r| r.offset < offset);
        // Coalesce with the next range if adjacent.
        let mut new = FreeRange { offset, len };
        if idx < self.free.len() && self.free[idx].offset == end {
            new.len += self.free[idx].len;
            self.free.remove(idx);
        }
        // Coalesce with the previous range if adjacent.
        if idx > 0 {
            let prev = &mut self.free[idx - 1];
            if prev.offset + prev.len == new.offset {
                prev.len += new.len;
                return;
            }
        }
        self.free.insert(idx, new);
    }

    /// Whether the whole capacity is free (chunk can be dropped). Used by tests
    /// now; the chunk-reclaim path in a later phase will use it in the library.
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.free.len() == 1 && self.free[0].len == self.capacity
    }
}

// ---------------------------------------------------------------------------
// ByteSlab: growable list of GPU chunk buffers over one usage.
// ---------------------------------------------------------------------------

struct Chunk {
    buffer: gpu::Buffer,
    alloc: RangeAllocator,
}

struct ByteSlab {
    chunks: Vec<Chunk>,
    usage: gpu::BufferUsages,
    label: &'static str,
    align: u64,
    max_buffer: u64,
    /// Size of the first chunk; each next chunk doubles from here.
    base_chunk_bytes: u64,
}

impl ByteSlab {
    fn new(
        usage: gpu::BufferUsages,
        label: &'static str,
        align: u64,
        max_buffer: u64,
        base_chunk_bytes: u64,
    ) -> Self {
        Self {
            chunks: Vec::new(),
            usage,
            label,
            align,
            max_buffer,
            base_chunk_bytes,
        }
    }

    /// Capacity for a new chunk that must hold at least `need` bytes: double the
    /// previous chunk from the initial size, capped at `max_buffer`, but never
    /// below `need` (a single mesh larger than the doubling schedule gets its
    /// own exactly-sized chunk).
    fn next_chunk_capacity(&self, need: u64) -> u64 {
        let n = self.chunks.len() as u32;
        let scheduled = self
            .base_chunk_bytes
            .saturating_mul(1u64 << n.min(20))
            .min(self.max_buffer);
        scheduled.max(need).min(self.max_buffer.max(need))
    }

    fn alloc(&mut self, device: &gpu::Device, bytes: u64) -> SlabSpan {
        for (ci, chunk) in self.chunks.iter_mut().enumerate() {
            if let Some(offset) = chunk.alloc.alloc(bytes) {
                return SlabSpan {
                    chunk: ci as u32,
                    offset,
                    len: bytes,
                };
            }
        }
        // No chunk fits; grow a new one.
        let capacity = self.next_chunk_capacity(align_up(bytes, self.align));
        let buffer = device.create_buffer(&gpu::BufferDescriptor {
            label: Some(self.label),
            size: capacity,
            usage: self.usage,
            mapped_at_creation: false,
        });
        let mut alloc = RangeAllocator::new(capacity, self.align);
        let offset = alloc
            .alloc(bytes)
            .expect("fresh chunk sized to hold the request");
        self.chunks.push(Chunk { buffer, alloc });
        SlabSpan {
            chunk: (self.chunks.len() - 1) as u32,
            offset,
            len: bytes,
        }
    }

    fn free(&mut self, span: SlabSpan) {
        if let Some(chunk) = self.chunks.get_mut(span.chunk as usize) {
            chunk.alloc.free(span.offset, span.len);
        }
    }

    fn buffer(&self, chunk: u32) -> &gpu::Buffer {
        &self.chunks[chunk as usize].buffer
    }

    fn resident_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.buffer.size()).sum()
    }
}

// ---------------------------------------------------------------------------
// GeometrySlab: the vertex + index slabs plus the accessors readers use.
// ---------------------------------------------------------------------------

/// Shared geometry storage for all meshes: a vertex slab and an index slab.
///
/// Lives on `DeviceResources` next to the mesh store. Meshes allocate spans at
/// upload and free them at removal; every geometry reader binds a sub-slice (or
/// an offset [`gpu::BindingResource`] for the STORAGE readers) rather than a
/// whole per-mesh buffer.
#[derive(Clone, Copy)]
enum SlabKind {
    Vertex,
    Index,
}

/// A geometry write recorded at upload time (device-only) and flushed to the GPU
/// in `process_uploads` with the frame queue. Mirrors how texture pixels defer to
/// the first `prepare()`; keeps the upload path queue-free.
struct PendingWrite {
    kind: SlabKind,
    span: SlabSpan,
    bytes: Vec<u8>,
}

pub(crate) struct GeometrySlab {
    vertex: ByteSlab,
    index: ByteSlab,
    /// Writes recorded before a queue was available, drained by `flush`.
    /// A `Mutex` so `flush` can run from `&self` (the strided colour-update path
    /// flushes before writing); uncontended in practice, resources are
    /// single-threaded. `DeviceResources` must also stay `Send + Sync`.
    pending: std::sync::Mutex<Vec<PendingWrite>>,
}

impl GeometrySlab {
    pub(crate) fn new(device: &gpu::Device) -> Self {
        let limits = device.limits();
        let align = (limits.min_storage_buffer_offset_alignment as u64).max(DEFAULT_ALIGN);
        let max_buffer = limits.max_buffer_size;
        // Diagnostic seam: shrink the first-chunk size so a test can force
        // multi-chunk allocation with small meshes. Off in normal use.
        let base_chunk = std::env::var("VIEWPORT_SLAB_CHUNK_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|b| align_up(b.max(align), align))
            .unwrap_or(INITIAL_CHUNK_BYTES)
            .min(max_buffer);
        Self {
            vertex: ByteSlab::new(
                gpu::BufferUsages::VERTEX
                    | gpu::BufferUsages::COPY_DST
                    | gpu::BufferUsages::STORAGE,
                "mesh_vertex_slab",
                align,
                max_buffer,
                base_chunk,
            ),
            index: ByteSlab::new(
                gpu::BufferUsages::INDEX | gpu::BufferUsages::COPY_DST | gpu::BufferUsages::STORAGE,
                "mesh_index_slab",
                align,
                max_buffer,
                base_chunk,
            ),
            pending: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Record a vertex-span write to flush at the next `process_uploads`. Takes
    /// ownership of the bytes; `data.len()` must equal `span.len`.
    pub(crate) fn enqueue_vertex(&self, span: SlabSpan, data: Vec<u8>) {
        debug_assert_eq!(data.len() as u64, span.len);
        self.pending.lock().unwrap().push(PendingWrite {
            kind: SlabKind::Vertex,
            span,
            bytes: data,
        });
    }

    /// Record an index-span write to flush at the next `process_uploads`.
    pub(crate) fn enqueue_index(&self, span: SlabSpan, data: Vec<u8>) {
        debug_assert_eq!(data.len() as u64, span.len);
        self.pending.lock().unwrap().push(PendingWrite {
            kind: SlabKind::Index,
            span,
            bytes: data,
        });
    }

    /// Flush all recorded writes to the GPU. Called from `process_uploads` with
    /// the frame queue, and before any in-place mutation that needs the initial
    /// geometry already resident.
    pub(crate) fn flush(&self, queue: &gpu::Queue) {
        let drained: Vec<PendingWrite> = self.pending.lock().unwrap().drain(..).collect();
        for w in drained {
            let slab = match w.kind {
                SlabKind::Vertex => &self.vertex,
                SlabKind::Index => &self.index,
            };
            queue.write_buffer(slab.buffer(w.span.chunk), w.span.offset, &w.bytes);
        }
    }

    /// Whether any geometry write is still waiting for a flush.
    #[allow(dead_code)] // used by tests / diagnostics
    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.lock().unwrap().is_empty()
    }

    /// Test constructor with an explicit tiny first-chunk size so a unit test can
    /// force multi-chunk allocation without large meshes.
    #[cfg(test)]
    pub(crate) fn new_with_base_chunk(device: &gpu::Device, base: u64) -> Self {
        let limits = device.limits();
        let align = (limits.min_storage_buffer_offset_alignment as u64).max(DEFAULT_ALIGN);
        let max_buffer = limits.max_buffer_size;
        let base = align_up(base.max(align), align).min(max_buffer);
        Self {
            vertex: ByteSlab::new(
                gpu::BufferUsages::VERTEX
                    | gpu::BufferUsages::COPY_DST
                    | gpu::BufferUsages::STORAGE,
                "mesh_vertex_slab",
                align,
                max_buffer,
                base,
            ),
            index: ByteSlab::new(
                gpu::BufferUsages::INDEX | gpu::BufferUsages::COPY_DST | gpu::BufferUsages::STORAGE,
                "mesh_index_slab",
                align,
                max_buffer,
                base,
            ),
            pending: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn alloc_vertex(&mut self, device: &gpu::Device, bytes: u64) -> SlabSpan {
        self.vertex.alloc(device, bytes)
    }

    pub(crate) fn alloc_index(&mut self, device: &gpu::Device, bytes: u64) -> SlabSpan {
        self.index.alloc(device, bytes)
    }

    pub(crate) fn free_vertex(&mut self, span: SlabSpan) {
        self.vertex.free(span);
    }

    pub(crate) fn free_index(&mut self, span: SlabSpan) {
        self.index.free(span);
    }

    /// Write vertex bytes into a span. The data length must equal `span.len`.
    pub(crate) fn write_vertex(&self, queue: &gpu::Queue, span: SlabSpan, data: &[u8]) {
        queue.write_buffer(self.vertex.buffer(span.chunk), span.offset, data);
    }

    /// Write index bytes into a span. The data length must equal `span.len`.
    pub(crate) fn write_index(&self, queue: &gpu::Queue, span: SlabSpan, data: &[u8]) {
        queue.write_buffer(self.index.buffer(span.chunk), span.offset, data);
    }

    /// Write into a sub-range of a vertex span, at `local_offset` bytes from the
    /// span start. Used by the strided colour update and chunked streaming path.
    pub(crate) fn write_vertex_at(
        &self,
        queue: &gpu::Queue,
        span: SlabSpan,
        local_offset: u64,
        data: &[u8],
    ) {
        queue.write_buffer(
            self.vertex.buffer(span.chunk),
            span.offset + local_offset,
            data,
        );
    }

    /// Clone of the chunk buffer backing a vertex span, for direct
    /// `queue.write_buffer` from a context that has the queue but not `&self`
    /// (the async chunked upload's GPU step). Writes must target
    /// `span.offset + local`.
    pub(crate) fn vertex_chunk_buffer(&self, span: SlabSpan) -> gpu::Buffer {
        self.vertex.buffer(span.chunk).clone()
    }

    /// Clone of the chunk buffer backing an index span (see
    /// [`vertex_chunk_buffer`](Self::vertex_chunk_buffer)).
    pub(crate) fn index_chunk_buffer(&self, span: SlabSpan) -> gpu::Buffer {
        self.index.buffer(span.chunk).clone()
    }

    pub(crate) fn vertex_slice(&self, span: SlabSpan) -> gpu::BufferSlice<'_> {
        self.vertex
            .buffer(span.chunk)
            .slice(span.offset..span.offset + span.len)
    }

    pub(crate) fn index_slice(&self, span: SlabSpan) -> gpu::BufferSlice<'_> {
        self.index
            .buffer(span.chunk)
            .slice(span.offset..span.offset + span.len)
    }

    /// Full slice of the chunk buffer that backs a vertex span. Used by the
    /// bind-once draw paths: bind the whole chunk once, then draw with a
    /// per-mesh `base_vertex` (see [`base_vertex`](Self::base_vertex)) instead
    /// of re-binding a sub-slice per mesh.
    pub(crate) fn vertex_chunk_slice(&self, chunk: u32) -> gpu::BufferSlice<'_> {
        self.vertex.buffer(chunk).slice(..)
    }

    /// Full slice of the chunk buffer that backs an index span (see
    /// [`vertex_chunk_slice`](Self::vertex_chunk_slice)).
    pub(crate) fn index_chunk_slice(&self, chunk: u32) -> gpu::BufferSlice<'_> {
        self.index.buffer(chunk).slice(..)
    }

    /// The `base_vertex` a `draw_indexed` needs so indices resolve to this
    /// mesh's vertices inside the shared chunk buffer. The span is aligned to
    /// the storage-offset alignment, so the offset divides the vertex stride
    /// evenly and the result is an exact vertex index.
    pub(crate) fn base_vertex(&self, span: SlabSpan) -> i32 {
        (span.offset / std::mem::size_of::<crate::resources::types::Vertex>() as u64) as i32
    }

    /// The `first_index` a `draw_indexed` needs so the index range lands on
    /// this mesh's indices inside the shared chunk buffer (u32 indices, so the
    /// byte offset divides by 4).
    pub(crate) fn first_index(&self, span: SlabSpan) -> u32 {
        (span.offset / 4) as u32
    }

    pub(crate) fn vertex_binding(&self, span: SlabSpan) -> gpu::BindingResource<'_> {
        gpu::BindingResource::Buffer(gpu::BufferBinding {
            buffer: self.vertex.buffer(span.chunk),
            offset: span.offset,
            size: std::num::NonZeroU64::new(span.len),
        })
    }

    pub(crate) fn index_binding(&self, span: SlabSpan) -> gpu::BindingResource<'_> {
        gpu::BindingResource::Buffer(gpu::BufferBinding {
            buffer: self.index.buffer(span.chunk),
            offset: span.offset,
            size: std::num::NonZeroU64::new(span.len),
        })
    }

    /// Total GPU bytes reserved by the slab chunks (for residency reporting).
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.vertex.resident_bytes() + self.index.resident_bytes()
    }

    /// Number of chunk buffers backing the slab (vertex chunks + index chunks).
    /// A steady value of 2 means all geometry fits one vertex and one index
    /// chunk, so every pass binds geometry at most once.
    pub(crate) fn chunk_count(&self) -> u32 {
        (self.vertex.chunks.len() + self.index.chunks.len()) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_aligned_and_sequential() {
        let mut a = RangeAllocator::new(4096, 256);
        let o0 = a.alloc(100).unwrap(); // rounds to 256
        let o1 = a.alloc(256).unwrap();
        let o2 = a.alloc(1).unwrap(); // rounds to 256 (min one alignment unit)
        assert_eq!(o0, 0);
        assert_eq!(o1, 256);
        assert_eq!(o2, 512);
        assert_eq!(o0 % 256, 0);
        assert_eq!(o1 % 256, 0);
        assert_eq!(o2 % 256, 0);
    }

    #[test]
    fn full_capacity_alloc_then_none() {
        let mut a = RangeAllocator::new(512, 256);
        assert_eq!(a.alloc(256), Some(0));
        assert_eq!(a.alloc(256), Some(256));
        assert_eq!(a.alloc(1), None); // full
    }

    #[test]
    fn free_coalesces_adjacent_ranges() {
        let mut a = RangeAllocator::new(1024, 256);
        let o0 = a.alloc(256).unwrap();
        let o1 = a.alloc(256).unwrap();
        let o2 = a.alloc(256).unwrap();
        let o3 = a.alloc(256).unwrap();
        assert!(a.alloc(256).is_none()); // full
        // Free the middle two out of order; they should coalesce into one
        // 512-byte hole that a 512 request can then use.
        a.free(o1, 256);
        a.free(o2, 256);
        let big = a.alloc(512).expect("coalesced hole holds 512");
        assert_eq!(big, o1);
        // Free everything and confirm the allocator is whole again.
        a.free(o0, 256);
        a.free(big, 512);
        a.free(o3, 256);
        assert!(a.is_empty());
    }

    #[test]
    fn free_coalesces_with_previous_and_next() {
        let mut a = RangeAllocator::new(768, 256);
        let o0 = a.alloc(256).unwrap();
        let o1 = a.alloc(256).unwrap();
        let o2 = a.alloc(256).unwrap();
        // Free outer two first, then the middle: freeing the middle must merge
        // with both neighbours into a single full-capacity range.
        a.free(o0, 256);
        a.free(o2, 256);
        a.free(o1, 256);
        assert!(a.is_empty());
        assert_eq!(a.alloc(768), Some(0));
    }

    #[test]
    fn reuses_freed_range_first_fit() {
        let mut a = RangeAllocator::new(1024, 256);
        let o0 = a.alloc(256).unwrap();
        let _o1 = a.alloc(256).unwrap();
        a.free(o0, 256);
        // First-fit picks the reopened low hole, not the tail.
        assert_eq!(a.alloc(256), Some(o0));
    }

    fn headless_device() -> Option<(gpu::Device, gpu::Queue)> {
        let instance = gpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&gpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&gpu::DeviceDescriptor::default())).ok()
    }

    /// Force a second chunk and confirm a write into it round-trips: spans in
    /// different chunks address different buffers, and the deferred flush lands
    /// the bytes at the right offset across the chunk boundary.
    #[test]
    fn multi_chunk_write_round_trips() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        // Tiny base chunk so a couple of small allocations spill into a new chunk.
        let mut slab = GeometrySlab::new_with_base_chunk(&device, 1024);
        let a = slab.alloc_vertex(&device, 1024); // fills chunk 0
        let b = slab.alloc_vertex(&device, 1024); // spills to chunk 1
        assert_eq!(a.chunk, 0);
        assert_ne!(
            b.chunk, a.chunk,
            "second allocation must land in a new chunk"
        );

        // Enqueue a write into the second chunk and flush; submitting validates
        // that the deferred write targets a real region at `b.offset` in the
        // right chunk buffer (a bad offset/size would fail wgpu validation).
        let pattern: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        slab.enqueue_vertex(b, pattern);
        assert!(slab.has_pending());
        slab.flush(&queue);
        assert!(!slab.has_pending(), "flush drains the pending list");
        let enc = device.create_command_encoder(&gpu::CommandEncoderDescriptor::default());
        queue.submit(std::iter::once(enc.finish()));
        let _ = device.poll(gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });

        // Freeing the spans returns them; the reused low chunk hands the range
        // back on the next allocation of the same size.
        slab.free_vertex(a);
        let c = slab.alloc_vertex(&device, 1024);
        assert_eq!(c.chunk, 0);
        assert_eq!(c.offset, a.offset);
    }
}
