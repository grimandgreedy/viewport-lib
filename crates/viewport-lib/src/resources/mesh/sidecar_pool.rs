//! Per-mesh sidecar streams packed into shared storage chunks.
//!
//! A mesh's per-vertex sidecars (scalar and face attributes, face colours,
//! vector attributes, the extension attribute and the lightmap UV1) and its
//! per-mesh draw words live in one record inside one chunk. A draw binds that
//! chunk at group 1 binding 6 and reaches every stream through the record, so
//! the sidecars cost one storage binding however many a mesh carries.
//!
//! Record layout, in `u32` words from the record base (mirrored in
//! `shaders/helpers/sidecar.wgsl`):
//!
//! | word | contents |
//! | --- | --- |
//! | 0 | first vertex of the mesh in its slab chunk, for the uv1 stream (0 without uv1) |
//! | 1, 2 | position override window: base element, element count |
//! | 3, 4 | normal override window |
//! | 5 | extension attribute stream index, [`NO_STREAM`] when absent |
//! | 6 | lightmap UV1 stream index, [`NO_STREAM`] when absent |
//! | 7 | stream count |
//! | 8.. | the stream table, `(offset, len)` per stream (offset in words from the record base, len in elements), then the streams |
//!
//! The table sits at a fixed offset so a shader reaches a stream in one
//! dependent load. It never grows in place: adding a stream moves the record.
//!
//! Chunks are added and never reallocated, so a bind group keyed on a chunk
//! stays valid. Writes, record moves and frees are queued and run in order at
//! [`SidecarPool::flush`]. A move copies the kept streams on the GPU, so it
//! runs after the writes queued before it and before the ones queued after.

use crate::gpu;
use crate::resources::mesh::geometry_slab::{ByteSlab, SlabSpan};

/// Stream index meaning "no stream", in the header and the selection word.
pub(crate) const NO_STREAM: u32 = u32::MAX;

/// Words before the stream table.
pub(crate) const HEADER_WORDS: u32 = 8;

const RECORD_ALIGN: u64 = 16;
const INITIAL_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// What a stream holds, which fixes its element size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StreamKind {
    /// One `f32` per vertex (`Vertex`, `Cell` and `Edge` attributes).
    Scalar,
    /// One `f32` per triangle corner (`Face`, `Halfedge`, `Corner`).
    FaceScalar,
    /// One `vec4<f32>` per triangle corner.
    FaceColour,
    /// Three `f32` per vertex (`VertexVector`).
    Vector,
    /// One `vec4<f32>` per vertex (`MeshData::extension_attributes`).
    Extension,
    /// One `vec4<f32>` per vertex: lightmap UV in `.xy`, atlas page in `.z`.
    LightmapUv,
}

impl StreamKind {
    /// Words per element.
    pub(crate) fn words(self) -> u32 {
        match self {
            StreamKind::Scalar | StreamKind::FaceScalar => 1,
            StreamKind::Vector => 3,
            StreamKind::FaceColour | StreamKind::Extension | StreamKind::LightmapUv => 4,
        }
    }
}

/// One stream of a record.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SidecarStream {
    pub kind: StreamKind,
    /// Attribute name; empty for the extension attribute and lightmap UV1.
    pub name: String,
    /// Position in the record's stream table.
    pub index: u32,
    /// Offset in words from the record base.
    pub offset: u32,
    /// Element count, after padding to the mesh's vertex or corner count.
    pub len: u32,
    /// Element count the caller supplied (`len` or fewer), which an in-place
    /// replacement must match.
    pub given: u32,
}

/// Stream contents handed to [`SidecarPool::build`] or [`SidecarPool::rebuild`].
pub(crate) struct StreamData {
    pub kind: StreamKind,
    pub name: String,
    pub words: Vec<u32>,
    /// Elements the caller supplied, before [`StreamData::padded`].
    pub given: u32,
}

impl StreamData {
    pub(crate) fn from_f32(kind: StreamKind, name: impl Into<String>, values: &[f32]) -> Self {
        Self {
            kind,
            name: name.into(),
            words: bytemuck::cast_slice(values).to_vec(),
            given: values.len() as u32 / kind.words(),
        }
    }

    pub(crate) fn from_vec4(
        kind: StreamKind,
        name: impl Into<String>,
        values: &[[f32; 4]],
    ) -> Self {
        Self {
            kind,
            name: name.into(),
            words: bytemuck::cast_slice(values).to_vec(),
            given: values.len() as u32,
        }
    }

    /// Extend a short stream to `elements` by repeating its last element, or
    /// with zeros when `zero` is set. The vertex stage reads a stream without
    /// a length, so every element a draw can index must exist; repeating the
    /// last element gives the image a clamped read used to.
    pub(crate) fn padded(mut self, elements: usize, zero: bool) -> Self {
        let w = self.kind.words() as usize;
        let want = elements * w;
        if self.words.len() < want {
            let fill: Vec<u32> = match self.words.len().checked_sub(w) {
                Some(last) if !zero => self.words[last..].to_vec(),
                _ => vec![0; w],
            };
            while self.words.len() < want {
                self.words.extend_from_slice(&fill);
            }
        }
        self
    }
}

/// The per-mesh words at the head of a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SidecarHeader {
    pub uv1_base: u32,
    pub position_override: (u32, u32),
    pub normal_override: (u32, u32),
}

impl Default for SidecarHeader {
    fn default() -> Self {
        Self {
            uv1_base: 0,
            position_override: (0, 0),
            normal_override: (0, 0),
        }
    }
}

/// A mesh's record: where it sits and what it holds.
#[derive(Clone, Debug)]
pub(crate) struct SidecarRecord {
    pub span: SlabSpan,
    pub header: SidecarHeader,
    pub streams: Vec<SidecarStream>,
}

impl SidecarRecord {
    pub(crate) fn chunk(&self) -> u32 {
        self.span.chunk
    }

    /// The record base, in words from the start of its chunk.
    pub(crate) fn base_word(&self) -> u32 {
        (self.span.offset / 4) as u32
    }

    pub(crate) fn find(&self, kind: StreamKind, name: &str) -> Option<&SidecarStream> {
        self.streams
            .iter()
            .find(|s| s.kind == kind && s.name == name)
    }

    pub(crate) fn first_of(&self, kind: StreamKind) -> Option<&SidecarStream> {
        self.streams.iter().find(|s| s.kind == kind)
    }

    /// Names of the streams of `kind`, in table order.
    pub(crate) fn names(&self, kind: StreamKind) -> impl Iterator<Item = &str> {
        self.streams
            .iter()
            .filter(move |s| s.kind == kind)
            .map(|s| s.name.as_str())
    }

    /// First word of a stream in its chunk. Never 0: word 0 of a chunk is a
    /// record header, so 0 can stand for "no stream".
    pub(crate) fn stream_word(&self, s: &SidecarStream) -> u32 {
        self.base_word() + s.offset
    }

    /// Byte range of a stream within its chunk.
    pub(crate) fn byte_range(&self, s: &SidecarStream) -> std::ops::Range<u64> {
        let start = self.span.offset + s.offset as u64 * 4;
        start..start + (s.len * s.kind.words()) as u64 * 4
    }

    /// Remove the first stream of `kind`, leaving its words in place. Later
    /// streams move up one table slot. Returns whether one was removed.
    pub(crate) fn drop_stream(&mut self, kind: StreamKind) -> bool {
        let Some(pos) = self.streams.iter().position(|s| s.kind == kind) else {
            return false;
        };
        self.streams.remove(pos);
        for (i, s) in self.streams.iter_mut().enumerate() {
            s.index = i as u32;
        }
        true
    }

    fn header_words(&self) -> [u32; HEADER_WORDS as usize] {
        let index_of = |kind| self.first_of(kind).map_or(NO_STREAM, |s| s.index);
        [
            self.header.uv1_base,
            self.header.position_override.0,
            self.header.position_override.1,
            self.header.normal_override.0,
            self.header.normal_override.1,
            index_of(StreamKind::Extension),
            index_of(StreamKind::LightmapUv),
            self.streams.len() as u32,
        ]
    }

    fn table_words(&self) -> Vec<u32> {
        self.streams
            .iter()
            .flat_map(|s| [s.offset, s.len])
            .collect()
    }
}

enum Op {
    Write {
        chunk: u32,
        offset: u64,
        words: Vec<u32>,
    },
    Copy {
        src_chunk: u32,
        src_offset: u64,
        dst_chunk: u32,
        dst_offset: u64,
        bytes: u64,
    },
    Free(SlabSpan),
}

/// The shared sidecar chunks and their queued work.
pub(crate) struct SidecarPool {
    slab: ByteSlab,
    ops: Vec<Op>,
    /// A header with no streams and empty windows, bound for meshes without a
    /// record (their instances carry record base 0).
    fallback: gpu::Buffer,
}

impl SidecarPool {
    pub(crate) fn new(device: &gpu::Device) -> Self {
        let limits = device.limits();
        let max_buffer = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size as u64);
        let mut words = [0u32; HEADER_WORDS as usize];
        words[5] = NO_STREAM;
        words[6] = NO_STREAM;
        let fallback = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("sidecar_fallback"),
            size: (HEADER_WORDS * 4) as u64,
            usage: gpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(fallback.slice(..), bytemuck::cast_slice(&words));
        fallback.unmap();
        Self {
            slab: ByteSlab::new(
                gpu::BufferUsages::STORAGE
                    | gpu::BufferUsages::VERTEX
                    | gpu::BufferUsages::COPY_DST
                    | gpu::BufferUsages::COPY_SRC,
                "sidecar_pool",
                RECORD_ALIGN,
                max_buffer,
                INITIAL_CHUNK_BYTES.min(max_buffer),
            ),
            ops: Vec::new(),
            fallback,
        }
    }

    /// Lay out `streams` behind `header` in a fresh record and queue its
    /// upload.
    pub(crate) fn build(
        &mut self,
        device: &gpu::Device,
        header: SidecarHeader,
        streams: Vec<StreamData>,
    ) -> SidecarRecord {
        let (layout, total) = Self::layout(
            streams
                .iter()
                .map(|s| (s.kind, s.name.clone(), s.words.len() as u32, s.given)),
        );
        let span = self.slab.alloc(device, total as u64 * 4);
        let record = SidecarRecord {
            span,
            header,
            streams: layout,
        };
        let mut words = Vec::with_capacity(total as usize);
        words.extend_from_slice(&record.header_words());
        words.extend(record.table_words());
        for s in &streams {
            words.extend_from_slice(&s.words);
        }
        self.ops.push(Op::Write {
            chunk: span.chunk,
            offset: span.offset,
            words,
        });
        record
    }

    /// Move `old` to a fresh record holding the streams `keep` accepts plus
    /// `add`, copying the kept streams on the GPU. Frees the old record.
    pub(crate) fn rebuild(
        &mut self,
        device: &gpu::Device,
        old: SidecarRecord,
        keep: impl Fn(&SidecarStream) -> bool,
        add: Vec<StreamData>,
    ) -> SidecarRecord {
        let kept: Vec<&SidecarStream> = old.streams.iter().filter(|s| keep(s)).collect();
        let (layout, total) = Self::layout(
            kept.iter()
                .map(|s| (s.kind, s.name.clone(), s.len * s.kind.words(), s.given))
                .chain(
                    add.iter()
                        .map(|s| (s.kind, s.name.clone(), s.words.len() as u32, s.given)),
                ),
        );
        let span = self.slab.alloc(device, total as u64 * 4);
        let record = SidecarRecord {
            span,
            header: old.header,
            streams: layout,
        };
        for (src, dst) in kept.iter().zip(record.streams.iter()) {
            let from = old.byte_range(src);
            let to = record.byte_range(dst);
            if from.end > from.start {
                self.ops.push(Op::Copy {
                    src_chunk: old.span.chunk,
                    src_offset: from.start,
                    dst_chunk: span.chunk,
                    dst_offset: to.start,
                    bytes: from.end - from.start,
                });
            }
        }
        for (s, dst) in add.iter().zip(record.streams.iter().skip(kept.len())) {
            if !s.words.is_empty() {
                self.ops.push(Op::Write {
                    chunk: span.chunk,
                    offset: record.byte_range(dst).start,
                    words: s.words.clone(),
                });
            }
        }
        self.write_header(&record);
        self.ops.push(Op::Free(old.span));
        record
    }

    /// Queue a rewrite of the record's header and stream table (after a
    /// header change, such as an override window).
    pub(crate) fn write_header(&mut self, record: &SidecarRecord) {
        let mut words = record.header_words().to_vec();
        words.extend(record.table_words());
        self.ops.push(Op::Write {
            chunk: record.span.chunk,
            offset: record.span.offset,
            words,
        });
    }

    /// Queue a rewrite of one stream's contents in place. `words` must be the
    /// stream's full length.
    pub(crate) fn write_stream(
        &mut self,
        record: &SidecarRecord,
        stream: &SidecarStream,
        words: Vec<u32>,
    ) {
        debug_assert_eq!(words.len() as u32, stream.len * stream.kind.words());
        self.ops.push(Op::Write {
            chunk: record.span.chunk,
            offset: record.byte_range(stream).start,
            words,
        });
    }

    /// Write one stream's contents in place now when nothing is queued ahead
    /// of it, otherwise queue it behind the earlier work.
    pub(crate) fn write_stream_with(
        &mut self,
        queue: &gpu::Queue,
        record: &SidecarRecord,
        stream: &SidecarStream,
        words: Vec<u32>,
    ) {
        if self.ops.is_empty() {
            debug_assert_eq!(words.len() as u32, stream.len * stream.kind.words());
            queue.write_buffer(
                self.slab.buffer(record.span.chunk),
                record.byte_range(stream).start,
                bytemuck::cast_slice(&words),
            );
        } else {
            self.write_stream(record, stream, words);
        }
    }

    /// Queue the record's release.
    pub(crate) fn free(&mut self, record: &SidecarRecord) {
        self.ops.push(Op::Free(record.span));
    }

    /// Run the queued work in order.
    pub(crate) fn flush(&mut self, device: &gpu::Device, queue: &gpu::Queue) {
        if self.ops.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.ops);
        let mut encoder: Option<gpu::CommandEncoder> = None;
        for op in ops {
            match op {
                Op::Write {
                    chunk,
                    offset,
                    words,
                } => {
                    // A queued write lands before the next submit's commands,
                    // so copies recorded so far are submitted first.
                    if let Some(enc) = encoder.take() {
                        queue.submit(std::iter::once(enc.finish()));
                    }
                    queue.write_buffer(
                        self.slab.buffer(chunk),
                        offset,
                        bytemuck::cast_slice(&words),
                    );
                }
                Op::Copy {
                    src_chunk,
                    src_offset,
                    dst_chunk,
                    dst_offset,
                    bytes,
                } => {
                    let enc = encoder.get_or_insert_with(|| {
                        device.create_command_encoder(&gpu::CommandEncoderDescriptor {
                            label: Some("sidecar_pool_move"),
                        })
                    });
                    if src_chunk == dst_chunk {
                        // A buffer cannot be both ends of one copy, so a move
                        // within a chunk goes through a scratch buffer.
                        let scratch = device.create_buffer(&gpu::BufferDescriptor {
                            label: Some("sidecar_pool_move_scratch"),
                            size: bytes,
                            usage: gpu::BufferUsages::COPY_SRC | gpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let chunk = self.slab.buffer(src_chunk);
                        enc.copy_buffer_to_buffer(chunk, src_offset, &scratch, 0, bytes);
                        enc.copy_buffer_to_buffer(&scratch, 0, chunk, dst_offset, bytes);
                    } else {
                        enc.copy_buffer_to_buffer(
                            self.slab.buffer(src_chunk),
                            src_offset,
                            self.slab.buffer(dst_chunk),
                            dst_offset,
                            bytes,
                        );
                    }
                }
                Op::Free(span) => self.slab.free(span),
            }
        }
        if let Some(enc) = encoder {
            queue.submit(std::iter::once(enc.finish()));
        }
    }

    pub(crate) fn chunk_buffer(&self, chunk: u32) -> &gpu::Buffer {
        self.slab.buffer(chunk)
    }

    /// The buffer a draw binds at group 1 binding 6 for `record`.
    pub(crate) fn binding_buffer(&self, record: Option<&SidecarRecord>) -> &gpu::Buffer {
        record.map_or(&self.fallback, |r| self.slab.buffer(r.chunk()))
    }

    pub(crate) fn fallback(&self) -> &gpu::Buffer {
        &self.fallback
    }

    #[allow(dead_code)]
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.slab.resident_bytes()
    }

    #[allow(dead_code)]
    pub(crate) fn has_pending(&self) -> bool {
        !self.ops.is_empty()
    }

    /// Stream layout for `(kind, name, words, given)` entries: offsets from
    /// the record base, and the record's total word count including the table.
    fn layout(
        entries: impl Iterator<Item = (StreamKind, String, u32, u32)>,
    ) -> (Vec<SidecarStream>, u32) {
        let entries: Vec<_> = entries.collect();
        let mut offset = HEADER_WORDS + 2 * entries.len() as u32;
        let mut out = Vec::with_capacity(entries.len());
        for (index, (kind, name, words, given)) in entries.into_iter().enumerate() {
            out.push(SidecarStream {
                kind,
                name,
                index: index as u32,
                offset,
                len: words / kind.words(),
                given,
            });
            offset += words;
        }
        (out, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_repeats_the_last_element() {
        let s = StreamData::from_vec4(StreamKind::FaceColour, "c", &[[1.0, 2.0, 3.0, 4.0]])
            .padded(3, false);
        let v: &[f32] = bytemuck::cast_slice(&s.words);
        assert_eq!(
            v,
            &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(s.given, 1);
    }

    #[test]
    fn padding_with_zeros_and_from_empty() {
        let s = StreamData::from_f32(StreamKind::Vector, "w", &[1.0, 2.0, 3.0]).padded(2, true);
        let v: &[f32] = bytemuck::cast_slice(&s.words);
        assert_eq!(v, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let e = StreamData::from_f32(StreamKind::Scalar, "e", &[]).padded(2, false);
        assert_eq!(e.words, vec![0, 0]);
    }

    #[test]
    fn a_long_stream_is_left_alone() {
        let s = StreamData::from_f32(StreamKind::Scalar, "s", &[1.0, 2.0, 3.0]).padded(2, false);
        assert_eq!(s.words.len(), 3);
        assert_eq!(s.given, 3);
    }
}
