//! Ranged writes into the content an item type holds, keyed on the channel.
//!
//! The calling half of a partial update. [`Uploads`](super::Uploads) replaces
//! content whole; this writes part of it. An item type that stores element
//! addressed content publishes one [`Channel`] marker per writable array and
//! implements [`Writes`] for each, so every type is driven through the same
//! calls rather than a named verb per type per array.
//!
//! The channel is passed by value, which is what makes the call site say which
//! array it means:
//!
//! ```ignore
//! use viewport_lib::plugin_api::Writes;
//! use viewport_lib_item_types::channels::point_cloud as pc;
//!
//! renderer.reserve(pc::Positions, &device, &queue, id, 1_000_000)?;
//! renderer.write_range(pc::Positions, &queue, id, 250_000, &sector)?;
//! renderer.write_range(pc::Scalars, &queue, id, 250_000, &intensities)?;
//! ```

/// One writable per-element channel of one item type.
///
/// A unit struct, passed to the [`Writes`] calls to name which array they mean.
/// It carries the handle the channel belongs to and the element type the
/// *caller* supplies, which is not necessarily what the GPU buffer stores:
/// padding, packing, interleaving and stride belong to the item type, which is
/// the only thing that knows its own layout. An item type whose storage is one
/// interleaved record has a single channel whose input is the whole sample.
pub trait Channel: Copy {
    /// Handle to the stored object the channel belongs to.
    type Id: Copy;

    /// One element as the caller supplies it.
    type Input: Copy;

    /// Name used in errors, for example `positions`. Not an identifier the
    /// library resolves anything by.
    const NAME: &'static str;
}

/// One contiguous run of elements to write, for [`Writes::write_spans`].
///
/// Several disjoint runs in one call is the shape a consumer with more than one
/// dirty region actually has: a streaming cloud refreshing two sectors, or an
/// assembly whose soft regions are not adjacent. Collapsing them into one span
/// covering both would rewrite everything in between.
#[derive(Copy, Clone)]
pub struct Span<'a, T> {
    /// Index of the first element this run overwrites.
    pub first_element: u32,
    /// The run's elements, in order.
    pub data: &'a [T],
}

impl<'a, T> Span<'a, T> {
    /// A run starting at `first_element`.
    pub fn new(first_element: u32, data: &'a [T]) -> Self {
        Self {
            first_element,
            data,
        }
    }
}

/// What one stored channel can hold and how much of it draws.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Extent {
    /// Elements the allocation holds.
    pub capacity: u32,
    /// Elements a draw reads.
    pub len: u32,
}

/// Writing part of what a store already holds, keyed on the channel.
///
/// Implemented on [`ViewportRenderer`](crate::renderer::ViewportRenderer) once
/// per channel, so passing a channel marker picks the item type that owns it.
/// Keyed on the channel rather than the handle or the payload because neither
/// identifies a write: the handle names the object but not which of its arrays
/// to write, and two arrays of one object can share an element type. A vector
/// field's positions and its vectors are both `[f32; 3]`, so a slice argument
/// alone would be ambiguous.
///
/// # What a write does not do
///
/// It does not grow the object. A window past the reserved capacity is an error
/// rather than a reallocation, because the reallocation drags a bind group
/// rebuild behind it and hiding that inside a per-frame call is the cost this
/// trait exists to remove. [`reserve`](Self::reserve) is the explicit step that
/// is allowed to be expensive.
///
/// It does not create a channel. Which channels an object holds is fixed when it
/// is uploaded, so writing one that was absent is
/// [`ChannelNotPresent`](crate::error::ViewportError::ChannelNotPresent) rather
/// than a new binding appearing under a streaming loop.
///
/// It does not re-derive anything that spans the whole array. A colourmap domain
/// or size domain left for the library to derive cannot survive a write that
/// sees part of the values, which is
/// [`ChannelDomainNotFixed`](crate::error::ViewportError::ChannelDomainNotFixed).
pub trait Writes<C: Channel> {
    /// Overwrite `data.len()` elements starting at `first_element`.
    ///
    /// Writing past the live count raises it, so an appending feed reserves
    /// once and then only writes. An empty `data` is a no-op.
    fn write_range(
        &mut self,
        channel: C,
        queue: &crate::gpu::Queue,
        id: C::Id,
        first_element: u32,
        data: &[C::Input],
    ) -> crate::error::ViewportResult<()>;

    /// Several disjoint runs in one call.
    ///
    /// The default walks them in order through
    /// [`write_range`](Self::write_range), which is the right implementation
    /// when each run is a queue write. A type that can batch them into one
    /// staging copy overrides it. Overlapping runs resolve in the order given.
    fn write_spans(
        &mut self,
        channel: C,
        queue: &crate::gpu::Queue,
        id: C::Id,
        spans: &[Span<'_, C::Input>],
    ) -> crate::error::ViewportResult<()> {
        for span in spans {
            self.write_range(channel, queue, id, span.first_element, span.data)?;
        }
        Ok(())
    }

    /// Grow the object to hold at least `capacity` elements, keeping what is
    /// already there.
    ///
    /// The call that is allowed to reallocate and rebind. Doubling means an
    /// appending feed pays it a logarithmic number of times rather than per
    /// write. Reserving no more than is already held does nothing.
    ///
    /// An item type whose channels share an element index reserves all of them
    /// together, so which channel is named does not matter; one that can size
    /// them independently reserves only the one named. Each type says which it
    /// is.
    fn reserve(
        &mut self,
        channel: C,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: C::Id,
        capacity: u32,
    ) -> crate::error::ViewportResult<()>;

    /// Set the live element count: what a draw reads, as distinct from what is
    /// allocated.
    ///
    /// Lowering it hides elements without freeing anything, so a feed whose
    /// extent shrinks does not give the capacity back. Raising it past what has
    /// been written exposes whatever the buffer held there. Above the reserved
    /// capacity it is an error.
    fn set_len(&mut self, channel: C, id: C::Id, len: u32) -> crate::error::ViewportResult<()>;

    /// Elements the object can hold before a [`reserve`](Self::reserve), and how
    /// many of them are live.
    ///
    /// `None` when the handle does not resolve. Reading it back is what lets a
    /// feed decide between a write and a replace.
    fn extent(&self, channel: C, id: C::Id) -> Option<Extent>;
}

/// Replacing a channel's storage with a buffer the caller owns and keeps filled.
///
/// The other end of [`Writes`]. A ranged write moves bytes from the host into
/// storage the item type allocated; this points the item type at storage the
/// consumer allocated, and then no bytes move at all. For a producer whose data
/// is already on the device (a compute simulation, a GPU decoder, a solver) that
/// removes the readback and the upload rather than shrinking them, which is the
/// larger win where it applies.
///
/// Keyed on the channel for the same reason [`Writes`] is: the handle names the
/// object but not which of its arrays to re-point.
///
/// # What the item type keeps
///
/// The item type still owns the channel's original allocation and its length. A
/// source stands in for the *contents*: the draw reads the caller's buffer for as
/// many elements as the channel is long, and
/// [`set_source(.., None)`](Self::set_source) puts the original back with
/// whatever it last held. So a consumer can hand a buffer over for a few frames
/// and take it back without re-uploading.
///
/// # Synchronisation is submission order
///
/// The renderer neither waits on nor fences the caller's writes: submit the work
/// that fills the buffer before the frame that draws from it. This is the same
/// contract the mesh position-override buffers carry.
///
/// # Reallocation
///
/// The renderer holds a clone of the buffer handle, which keeps the allocation
/// alive but does not track it. A consumer who grows and reallocates must call
/// this again with the new buffer; holding the old one draws whatever was last
/// written to the old allocation.
pub trait Sourced<C: Channel> {
    /// Draw this channel from `source` instead of from the item type's own
    /// buffer, or from its own buffer again when `source` is `None`.
    ///
    /// The buffer must carry the usage the channel is bound with and hold at
    /// least as many elements as the channel is long. Both are checked here
    /// rather than at draw time, because a wgpu validation failure on a binding
    /// takes the device down and a returned error does not.
    ///
    /// `device` is passed rather than held: swapping a binding means building a
    /// bind group, and this library keeps no device or queue of its own.
    fn set_source(
        &mut self,
        channel: C,
        device: &crate::gpu::Device,
        id: C::Id,
        source: Option<crate::gpu::Buffer>,
    ) -> crate::error::ViewportResult<()>;

    /// Whether this channel is currently drawn from a caller-owned buffer.
    ///
    /// `None` when the handle does not resolve. Worth reading back before a
    /// consumer decides whether a [`Writes::write_range`] would be ignored: a
    /// write still lands in the item type's own buffer, which is not what draws
    /// while a source is set.
    fn has_source(&self, channel: C, id: C::Id) -> Option<bool>;
}
