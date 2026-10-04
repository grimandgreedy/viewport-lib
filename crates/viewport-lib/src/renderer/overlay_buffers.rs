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
    usage: crate::gpu::BufferUsages,
    /// The bytes last written, kept by `write_changed` to skip identical writes.
    written: Vec<u8>,
}

impl GrowBuffer {
    /// A `VERTEX | COPY_DST` grow buffer with no allocation yet.
    pub(crate) fn vertex(label: &'static str) -> Self {
        Self::with_usage(label, crate::gpu::BufferUsages::VERTEX)
    }

    /// An `INDEX | COPY_DST` grow buffer with no allocation yet.
    pub(crate) fn index(label: &'static str) -> Self {
        Self::with_usage(label, crate::gpu::BufferUsages::INDEX)
    }

    /// A `STORAGE | COPY_DST` grow buffer with no allocation yet. Shaders read
    /// it as a runtime-sized array and index only what this frame wrote, so the
    /// stale tail is never read.
    pub(crate) fn storage(label: &'static str) -> Self {
        Self::with_usage(label, crate::gpu::BufferUsages::STORAGE)
    }

    fn with_usage(label: &'static str, usage: crate::gpu::BufferUsages) -> Self {
        Self {
            buf: None,
            capacity: 0,
            label,
            usage: usage | crate::gpu::BufferUsages::COPY_DST,
            written: Vec::new(),
        }
    }

    /// The current allocation, if any write has happened.
    pub(crate) fn buffer(&self) -> Option<&crate::gpu::Buffer> {
        self.buf.as_ref()
    }

    /// Like `write`, but skips the upload when the bytes match the last
    /// `write_changed`. For buffers written several times a frame with the
    /// same contents, or re-written unchanged frame to frame.
    pub(crate) fn write_changed<T: bytemuck::Pod>(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &[T],
    ) -> crate::gpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        if let Some(buf) = &self.buf {
            if self.written == bytes {
                return buf.clone();
            }
        }
        let buf = self.write(device, queue, data);
        self.written.clear();
        self.written.extend_from_slice(bytes);
        buf
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
                usage: self.usage,
                mapped_at_creation: false,
            }));
            self.capacity = cap;
        }
        let buf = self.buf.as_ref().unwrap();
        queue.write_buffer(buf, 0, bytes);
        self.written.clear();
        buf.clone()
    }
}

/// Buffers for one flat-colour line or triangle overlay (a constraint guide or
/// a section cap), reused across frames at the same position in its list.
pub(crate) struct OverlayGeometrySlot {
    vertices: GrowBuffer,
    indices: GrowBuffer,
    uniform: Option<(crate::gpu::Buffer, crate::gpu::BindGroup, [f32; 4])>,
}

/// What the overlay draw paths read: vertex buffer, index buffer, index count,
/// uniform buffer and its bind group.
pub(crate) type OverlayGeometryDraw = (
    crate::gpu::Buffer,
    crate::gpu::Buffer,
    u32,
    crate::gpu::Buffer,
    crate::gpu::BindGroup,
);

impl OverlayGeometrySlot {
    pub(crate) fn new() -> Self {
        Self {
            vertices: GrowBuffer::vertex("overlay_geometry_vbuf"),
            indices: GrowBuffer::index("overlay_geometry_ibuf"),
            uniform: None,
        }
    }

    /// Write this frame's geometry and colour, allocating only when the
    /// geometry outgrew the buffers or on first use.
    pub(crate) fn write(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        layout: &crate::gpu::BindGroupLayout,
        vertices: &[crate::resources::OverlayVertex],
        indices: &[u32],
        colour: [f32; 4],
    ) -> OverlayGeometryDraw {
        let vbuf = self.vertices.write_changed(device, queue, vertices);
        let ibuf = self.indices.write_changed(device, queue, indices);
        let data = crate::resources::overlay::overlays::OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            colour,
        };
        let (ubuf, bg, written) = self.uniform.get_or_insert_with(|| {
            let ubuf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("overlay_geometry_ubuf"),
                size: std::mem::size_of::<crate::resources::overlay::overlays::OverlayUniform>()
                    as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&data));
            let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("overlay_geometry_bg"),
                layout,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                }],
            });
            (ubuf, bg, colour)
        });
        if *written != colour {
            queue.write_buffer(ubuf, 0, bytemuck::bytes_of(&data));
            *written = colour;
        }
        (vbuf, ibuf, indices.len() as u32, ubuf.clone(), bg.clone())
    }
}

/// The overlay pass's per-frame storage buffers and the bind groups over
/// them, kept across frames. Buffers grow on demand and are rewritten only when
/// their contents change; bind groups are rebuilt only when a buffer, the glyph
/// atlas or a texture they bind was replaced.
pub(crate) struct OverlayBindings {
    /// Clip-mask shapes, shared by the text, shape and textured-shape pipelines.
    pub clip: GrowBuffer,
    /// Slot 0 identity plus one instance per retained group.
    pub instances: GrowBuffer,
    /// Shape shadow layers, read by all three shape pipelines.
    pub shape_shadow: GrowBuffer,
    pub label_bg: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::TextureView,
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        crate::gpu::Buffer,
    )>,
    pub shape_shadow_bg:
        crate::resources::cached_bind_group::CachedBindGroup<[crate::gpu::Buffer; 4]>,
    pub tex_clip_bg: crate::resources::cached_bind_group::CachedBindGroup<[crate::gpu::Buffer; 3]>,
    /// Per retained group, keyed by the group's own shadow buffer.
    pub retained_shape_bgs: std::collections::HashMap<
        crate::gpu::Buffer,
        crate::resources::cached_bind_group::CachedBindGroup<[crate::gpu::Buffer; 3]>,
    >,
    /// Per overlay texture.
    pub tex_bgs: std::collections::HashMap<crate::gpu::TextureView, crate::gpu::BindGroup>,
}

impl OverlayBindings {
    pub(crate) fn new() -> Self {
        Self {
            clip: GrowBuffer::storage("overlay_clip_buf"),
            instances: GrowBuffer::storage("overlay_instances_buf"),
            shape_shadow: GrowBuffer::storage("overlay_shape_shadow_buf"),
            label_bg: crate::resources::cached_bind_group::CachedBindGroup::new(),
            shape_shadow_bg: crate::resources::cached_bind_group::CachedBindGroup::new(),
            tex_clip_bg: crate::resources::cached_bind_group::CachedBindGroup::new(),
            retained_shape_bgs: std::collections::HashMap::new(),
            tex_bgs: std::collections::HashMap::new(),
        }
    }
}

/// One retained overlay group's per-frame draw: its cached vertex buffer (a
/// clone of the `CompiledOverlay` buffer, resolved from the store this frame),
/// vertex count, and the index of its per-draw instance in the frame's instance
/// buffer. Rebuilt each frame in the overlay label prepare and consumed in the
/// ordered overlay emit.
pub(crate) struct RetainedDraw {
    pub vertex_buf: crate::gpu::Buffer,
    pub vertex_count: u32,
    pub instance_index: u32,
}

/// One retained group's SDF shape-stream draw: the group's cached shape vertex
/// buffer, its count, a bind group (the group's shadow buffer plus the shared clip
/// / viewport / instances), and the per-draw instance index. Drawn through the
/// overlay shape pipeline.
pub(crate) struct RetainedShapeDraw {
    pub vertex_buf: crate::gpu::Buffer,
    pub vertex_count: u32,
    pub bind_group: crate::gpu::BindGroup,
    pub instance_index: u32,
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
