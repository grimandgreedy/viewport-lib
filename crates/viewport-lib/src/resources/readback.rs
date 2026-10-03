//! A GPU readback the caller can poll instead of waiting on.
//!
//! The copy into a staging buffer is submitted and the map requested straight
//! away; [`PendingReadback::poll`] checks whether the map has completed without
//! blocking, and [`PendingReadback::wait`] blocks until it has.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const PENDING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// A buffer or texture copied into a staging buffer that is being mapped.
pub(crate) struct PendingReadback {
    staging: crate::gpu::Buffer,
    state: Arc<AtomicU8>,
    /// Bytes per row as copied, and as wanted; equal for a buffer copy, padded
    /// to the copy alignment for a texture.
    padded_row: usize,
    unpadded_row: usize,
}

impl PendingReadback {
    /// Copy the first `bytes` of `src` back.
    pub(crate) fn buffer(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        src: &crate::gpu::Buffer,
        bytes: u64,
    ) -> Self {
        let staging = staging_buffer(device, bytes);
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        Self::map(staging, bytes as usize, bytes as usize)
    }

    /// Copy a `width` x `height` texture with `bytes_per_texel` back, without
    /// the row padding the copy needs.
    pub(crate) fn texture(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        texture: &crate::gpu::Texture,
        width: u32,
        height: u32,
        bytes_per_texel: u32,
    ) -> Self {
        let unpadded_row = width * bytes_per_texel;
        let align = crate::gpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;
        let staging = staging_buffer(device, u64::from(padded_row) * u64::from(height));
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_texture_to_buffer(
            crate::gpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            crate::gpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        Self::map(staging, padded_row as usize, unpadded_row as usize)
    }

    fn map(staging: crate::gpu::Buffer, padded_row: usize, unpadded_row: usize) -> Self {
        let state = Arc::new(AtomicU8::new(PENDING));
        let signal = Arc::clone(&state);
        staging
            .slice(..)
            .map_async(crate::gpu::MapMode::Read, move |result| {
                signal.store(
                    if result.is_ok() { MAPPED } else { FAILED },
                    Ordering::Release,
                );
            });
        Self {
            staging,
            state,
            padded_row,
            unpadded_row,
        }
    }

    /// Whether the data has arrived, without waiting. A failed map counts as
    /// arrived; [`take`](Self::take) then returns `None`.
    pub(crate) fn poll(&self, device: &crate::gpu::Device) -> bool {
        if self.state.load(Ordering::Acquire) == PENDING {
            let _ = device.poll(crate::gpu::PollType::Poll);
        }
        self.state.load(Ordering::Acquire) != PENDING
    }

    /// Block until the data has arrived.
    pub(crate) fn wait(&self, device: &crate::gpu::Device) {
        while self.state.load(Ordering::Acquire) == PENDING {
            let _ = device.poll(crate::gpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(60)),
            });
        }
    }

    /// The bytes, row padding removed, once [`poll`](Self::poll) or
    /// [`wait`](Self::wait) has seen them arrive. `None` if the map failed.
    pub(crate) fn take(self) -> Option<Vec<u8>> {
        if self.state.load(Ordering::Acquire) != MAPPED {
            return None;
        }
        let out = {
            let data = crate::gpu::mapped_range(self.staging.slice(..));
            if self.padded_row == self.unpadded_row {
                data.to_vec()
            } else {
                data.chunks_exact(self.padded_row)
                    .flat_map(|row| &row[..self.unpadded_row])
                    .copied()
                    .collect()
            }
        };
        self.staging.unmap();
        Some(out)
    }
}

fn staging_buffer(device: &crate::gpu::Device, bytes: u64) -> crate::gpu::Buffer {
    device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("readback_staging"),
        size: bytes,
        usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}
