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
    /// The smallest allocation, for a uniform buffer whose binding has a fixed
    /// size.
    min_capacity: u64,
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

    /// A `UNIFORM | COPY_DST` buffer of exactly `size` bytes, for a fixed-size
    /// uniform array. Shaders index only what this frame wrote, so the stale
    /// tail is never read.
    pub(crate) fn uniform(label: &'static str, size: u64) -> Self {
        Self {
            min_capacity: size,
            ..Self::with_usage(label, crate::gpu::BufferUsages::UNIFORM)
        }
    }

    fn with_usage(label: &'static str, usage: crate::gpu::BufferUsages) -> Self {
        Self {
            buf: None,
            capacity: 0,
            label,
            usage: usage | crate::gpu::BufferUsages::COPY_DST,
            min_capacity: 0,
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
            let cap = grow_capacity(needed, self.capacity.max(self.min_capacity));
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

/// An `Rgba32Float` data texture (the shadow layers, the fill records) that
/// persists across frames, grows by rows when a frame has more texels than it
/// holds, and is rewritten only when its contents change.
pub(crate) struct DataTexture {
    label: &'static str,
    row_texels: u32,
    texture: Option<(crate::gpu::Texture, crate::gpu::TextureView)>,
    /// The texels last written, so an unchanged frame skips the upload.
    written: Vec<u8>,
}

impl DataTexture {
    pub(crate) fn new(label: &'static str, row_texels: u32) -> Self {
        Self {
            label,
            row_texels,
            texture: None,
            written: Vec::new(),
        }
    }

    /// The shadow-layer texture, laid out as `shadow_layer_texels`.
    pub(crate) fn shadow_layers() -> Self {
        Self::new(
            "overlay_shadow_layers",
            crate::resources::overlay::overlay_shape::SHADOW_ROW_TEXELS,
        )
    }

    /// A fill-record texture, as the fill module reads it.
    pub(crate) fn fill_records() -> Self {
        Self::new(
            "overlay_fill_records",
            crate::resources::overlay::overlay_shape::FILL_TEXELS_PER_ROW,
        )
    }

    /// The view last written, if any.
    pub(crate) fn view(&self) -> Option<crate::gpu::TextureView> {
        self.texture.as_ref().map(|(_, v)| v.clone())
    }

    /// Upload `data` (whole texels, at least one) and return the view to bind.
    pub(crate) fn write_changed<T: bytemuck::Pod>(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &[T],
    ) -> crate::gpu::TextureView {
        use crate::resources::overlay::overlay_shape as os;
        let (texels, rows) = os::data_texels(bytemuck::cast_slice(data), self.row_texels);
        let fits = self
            .texture
            .as_ref()
            .is_some_and(|(t, _)| t.height() >= rows);
        if !fits {
            // Grow in steps so a frame-to-frame wobble in the count does not
            // reallocate each time.
            let rows = rows.next_power_of_two();
            let texture = os::create_data_texture(device, self.label, self.row_texels, rows);
            let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
            self.texture = Some((texture, view));
            self.written.clear();
        }
        let (texture, view) = self.texture.as_ref().unwrap();
        if self.written != texels {
            os::write_data_texels(queue, texture, &texels, self.row_texels, rows);
            self.written = texels;
        }
        view.clone()
    }
}

/// The overlay pass's per-frame buffers and the bind groups over them, kept
/// across frames. Buffers grow on demand and are rewritten only when their
/// contents change; bind groups are rebuilt only when a buffer, the glyph atlas
/// or a texture they bind was replaced.
pub(crate) struct OverlayBindings {
    /// Clip-mask shapes, shared by the text, shape and textured-shape pipelines:
    /// a uniform array of `MAX_OVERLAY_CLIP_SHAPES`.
    pub clip: GrowBuffer,
    /// Slot 0 identity plus one instance per retained group, bound as the
    /// per-instance vertex stream of the text pipeline.
    pub instances: GrowBuffer,
    /// The same instances, one per `INSTANCE_UNIFORM_STRIDE` bytes, bound by
    /// the shape pipeline as a uniform at a dynamic offset.
    pub instances_uniform: GrowBuffer,
    /// Shape shadow layers, read by the shape and backdrop pipelines.
    pub shape_shadow: DataTexture,
    /// The frame's fill records for the shape pass.
    pub shape_fills: DataTexture,
    /// The frame's fill records for the text stream.
    pub text_fills: DataTexture,
    /// Fill bind groups (group 1) the shape pass drew with last frame.
    pub shape_fill_bgs: crate::resources::overlay::overlay_shape::OverlayFillBindGroups,
    /// Fill bind groups the text stream and the retained groups drew with
    /// last frame.
    pub text_fill_bgs: crate::resources::overlay::overlay_shape::OverlayFillBindGroups,
    pub label_bg: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::TextureView,
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        crate::gpu::TextureView,
        crate::gpu::TextureView,
    )>,
    pub shape_shadow_bg: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::TextureView,
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        crate::gpu::Buffer,
    )>,
    pub tex_clip_bg: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        crate::gpu::TextureView,
    )>,
    /// Per retained group, keyed by the group's own shadow-layer texture.
    pub retained_shape_bgs: std::collections::HashMap<
        crate::gpu::TextureView,
        crate::resources::cached_bind_group::CachedBindGroup<[crate::gpu::Buffer; 3]>,
    >,
}

impl OverlayBindings {
    pub(crate) fn new() -> Self {
        Self {
            clip: GrowBuffer::uniform(
                "overlay_clip_buf",
                crate::resources::overlay::overlay_shape::CLIP_UNIFORM_BYTES,
            ),
            instances: GrowBuffer::vertex("overlay_instances_buf"),
            instances_uniform: GrowBuffer::uniform(
                "overlay_instances_uniform",
                crate::resources::overlay::geometry::INSTANCE_UNIFORM_STRIDE,
            ),
            shape_shadow: DataTexture::shadow_layers(),
            shape_fills: DataTexture::fill_records(),
            text_fills: DataTexture::fill_records(),
            shape_fill_bgs: Default::default(),
            text_fill_bgs: Default::default(),
            label_bg: crate::resources::cached_bind_group::CachedBindGroup::new(),
            shape_shadow_bg: crate::resources::cached_bind_group::CachedBindGroup::new(),
            tex_clip_bg: crate::resources::cached_bind_group::CachedBindGroup::new(),
            retained_shape_bgs: std::collections::HashMap::new(),
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
    /// Group 1: the group's own fill records (or the frame's, when it has
    /// none) with the blank image.
    pub fill_bind_group: crate::gpu::BindGroup,
}

/// One retained group's SDF shape-stream draw: the group's cached shape
/// instance buffer, its count, a bind group (the group's shadow-layer texture
/// plus the shared clip, viewport and group-instance uniform), and the index of
/// its group instance, which the draw binds at a dynamic offset. Drawn through
/// the overlay shape pipeline.
pub(crate) struct RetainedShapeDraw {
    pub instance_buf: crate::gpu::Buffer,
    pub instance_count: u32,
    pub bind_group: crate::gpu::BindGroup,
    pub instance_index: u32,
    /// Group 1: the group's own shape fill records with the blank image.
    pub fill_bind_group: crate::gpu::BindGroup,
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
