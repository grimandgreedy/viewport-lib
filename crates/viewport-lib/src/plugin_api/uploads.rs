//! The standard upload surface for an item type that holds its own content.
//!
//! Two traits rather than one, split by what each call is given. [`Uploads`] is
//! keyed on the content, so `renderer.upload(&device, &queue, &item)` picks the
//! item type from the item itself. [`Handles`] is keyed on the handle, so
//! `renderer.release(id)` picks it from the id. A single trait would have to
//! infer the item type from an associated handle type, which Rust cannot do,
//! and every release would need naming the item type by hand.

/// Uploading and replacing the content an item type holds on the consumer's
/// behalf.
///
/// An item type with a store gives its consumer the same calls, differing only
/// in the content they take and the handle they mint. Implementing this on
/// [`ViewportRenderer`](crate::renderer::ViewportRenderer) for one content type
/// is what makes `renderer.upload(..)` resolve to that item type, the same way
/// `frame.scene.items_mut::<T>()` picks a collection.
///
/// `T` is what a consumer hands over. That is usually the item struct, but an
/// item type whose per-frame item is a reference to an upload keys this on the
/// uploaded data instead.
///
/// Not every item type fits. One that wraps a buffer the consumer already owns,
/// or that runs a simulation rather than holding content, has different verbs
/// and should publish them itself. An item type with two stores behind one
/// content type implements this for the store a consumer reaches for first and
/// names the other explicitly, because `Id` is one type per implementation.
pub trait Uploads<T> {
    /// Handle to one uploaded item.
    type Id: Copy;

    /// Upload `item`, returning a handle valid until
    /// [`release`](Handles::release).
    fn upload(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &T,
    ) -> crate::error::ViewportResult<Self::Id>;

    /// Start an off-thread upload. Poll the returned job with
    /// [`upload_status`](crate::renderer::ViewportRenderer::upload_status) and
    /// take the handle from [`upload_result`](Handles::upload_result).
    fn begin_upload(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: T,
    ) -> crate::error::ViewportResult<crate::resources::JobId>;

    /// Replace what is behind a live handle, keeping the handle.
    fn replace(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: Self::Id,
        item: &T,
    ) -> crate::error::ViewportResult<()>;
}

/// The calls that take a handle and nothing else.
///
/// Keyed on the handle type, so `renderer.release(id)` needs no annotation.
/// [`upload_result`](Self::upload_result) has only a job id to go on, so it is
/// the one call where a consumer says which handle it expects back, either by
/// annotating the binding or with a turbofish.
pub trait Handles<Id> {
    /// Take the handle from a finished
    /// [`begin_upload`](Uploads::begin_upload) job. The content enters the
    /// store here, so the handle is minted on this call rather than on the
    /// worker thread.
    fn upload_result(&mut self, job: crate::resources::JobId) -> crate::error::ViewportResult<Id>;

    /// Release the content behind a handle. `false` when the handle did not
    /// resolve, which is not an error: releasing twice is harmless.
    fn release(&mut self, id: Id) -> bool;
}
