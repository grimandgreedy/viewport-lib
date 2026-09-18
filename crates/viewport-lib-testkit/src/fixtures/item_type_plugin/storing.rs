//! An item-type plugin that holds the content it draws, uploaded into from
//! outside the crate.

use std::sync::Arc;

use viewport_lib::plugin_api::{ItemTypePlugin, PluginItemCollection};
use viewport_lib::resources::{DeviceResources, JobId, Jobs, TextureId, UploadStatus};
use viewport_lib::wgpu;

/// Handle to a buffer this plugin holds. The index into its own store, which
/// is all a toy store needs; a real one carries a generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredId(pub usize);

/// An item type with a store of its own, a synchronous upload, and an async
/// upload pair, all reached through
/// [`ViewportRenderer::item_type_plugin_host`](viewport_lib::ViewportRenderer::item_type_plugin_host).
///
/// The built-in types that hold their own content work exactly this way, and
/// the property under test is that they use no route a plugin outside the
/// crate is missing: the upload call needs the plugin, the job runner, and a
/// read of the shared content arenas at the same time, and all three come
/// from one accessor. If that accessor stops being public or stops handing
/// out all three, this fixture fails to compile.
#[derive(Default)]
pub struct StoringItemTypePlugin {
    buffers: Vec<Option<wgpu::Buffer>>,
}

impl StoringItemTypePlugin {
    /// The name this fixture registers under.
    pub const TYPE_NAME: &'static str = "testkit.storing";

    /// Upload `data` and return its handle.
    ///
    /// `texture` is validated against the shared texture store, the way an
    /// item type whose content names a texture has to: an upload path needs
    /// to read content it does not own.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &DeviceResources,
        data: &[u8],
        texture: Option<TextureId>,
    ) -> Option<StoredId> {
        if texture.is_some_and(|id| !resources.has_texture(id)) {
            return None;
        }
        Some(self.insert(build_buffer(device, queue, data)))
    }

    /// Submit the buffer build to a worker thread. The handle is minted by
    /// [`take_upload_result`](Self::take_upload_result), because that is the
    /// call that has `&mut self` to insert with.
    pub fn begin_upload(
        &self,
        jobs: &Jobs<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: Vec<u8>,
    ) -> JobId {
        let device = device.clone();
        let queue = queue.clone();
        jobs.try_submit_cpu(move |progress| {
            progress.set(0.5);
            if data.is_empty() {
                return Err(viewport_lib::error::ViewportError::SlotEmpty { index: 0 });
            }
            Ok(Arc::new(build_buffer(&device, &queue, &data)))
        })
    }

    /// Take a finished async upload into the store.
    pub fn take_upload_result(&mut self, jobs: &Jobs<'_>, id: JobId) -> Option<StoredId> {
        match jobs.status(id) {
            UploadStatus::Ready => {
                let buffer = jobs.take::<Arc<wgpu::Buffer>>(id)?;
                Some(self.insert(Arc::unwrap_or_clone(buffer)))
            }
            _ => None,
        }
    }

    /// Whether a handle still resolves.
    pub fn contains(&self, id: StoredId) -> bool {
        self.buffers.get(id.0).is_some_and(|slot| slot.is_some())
    }

    /// Drop a stored buffer, freeing its slot for reuse.
    pub fn free(&mut self, id: StoredId) {
        if let Some(slot) = self.buffers.get_mut(id.0) {
            *slot = None;
        }
    }

    fn insert(&mut self, buffer: wgpu::Buffer) -> StoredId {
        match self.buffers.iter().position(|slot| slot.is_none()) {
            Some(index) => {
                self.buffers[index] = Some(buffer);
                StoredId(index)
            }
            None => {
                self.buffers.push(Some(buffer));
                StoredId(self.buffers.len() - 1)
            }
        }
    }
}

fn build_buffer(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len() as u64).max(4).next_multiple_of(4);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("testkit_storing_buf"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, data);
    buffer
}

impl ItemTypePlugin for StoringItemTypePlugin {
    fn type_name(&self) -> &'static str {
        Self::TYPE_NAME
    }

    /// What the store holds, so a host sizing a working set can see it.
    fn resident_bytes(&self) -> u64 {
        self.buffers
            .iter()
            .flatten()
            .map(|buffer| buffer.size())
            .sum()
    }

    fn prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _ctx: &viewport_lib::plugin_api::ItemFrameContext<'_>,
        _items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        Vec::new()
    }
}
