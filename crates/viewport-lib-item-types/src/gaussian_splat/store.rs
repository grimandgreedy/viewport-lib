//! The splat sets this item type holds on the consumer's behalf.
//!
//! A `GaussianSplatItem` names a set by [`GaussianSplatId`] rather than
//! carrying its splats, so the data is uploaded once and drawn every frame
//! from here. The upload entry points are the `*_gaussian_splat` methods on
//! [`ViewportRenderer`](viewport_lib::renderer::ViewportRenderer), which find this
//! plugin by name and call through to the methods below.

use super::types::GaussianSplatId;

pub use super::types::{GaussianSplatData, ShDegree};
use viewport_lib::error::{ViewportError, ViewportResult};
use viewport_lib::gpu;
use viewport_lib::plugin_api::Extent;
use viewport_lib::resources::ContentBuffer;

/// Check that a splat set is non-empty and its per-attribute vectors agree in
/// length. Shared by the sync, async, and replace upload paths.
pub(super) fn validate_gaussian_splat_data(
    data: &GaussianSplatData,
) -> viewport_lib::error::ViewportResult<()> {
    if data.positions.is_empty() {
        return Err(
            viewport_lib::error::ViewportError::InvalidGaussianSplatData {
                reason: "empty splat list",
            },
        );
    }
    let n = data.positions.len();
    if data.scales.len() != n || data.rotations.len() != n || data.opacities.len() != n {
        return Err(
            viewport_lib::error::ViewportError::InvalidGaussianSplatData {
                reason: "mismatched buffer lengths",
            },
        );
    }
    Ok(())
}

/// Bytes per element of each splat channel. Positions, scales and rotations are
/// padded to `vec4` on the GPU whatever the caller hands over; the SH stride is
/// a runtime property of the set's degree and so is not here.
const VEC4_STRIDE: u32 = 16;
const OPACITY_STRIDE: u32 = 4;

/// Which channel of a splat set a call means.
///
/// Every channel is indexed by splat, the SH one included: its stride is one
/// splat's worth of coefficients, so a write covers whole splats even though the
/// caller supplies loose `f32`s.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum SplatChannel {
    Positions,
    Scales,
    Rotations,
    Opacities,
    ShCoefficients,
}

/// Build the persistent GPU buffers for a splat set and assemble the
/// `GaussianSplatGpuSet`. Assumes `data` already passed
/// [`validate_gaussian_splat_data`]. Shared by the sync upload, the async
/// worker, and the replace path so all three produce identical resources.
///
/// `capacity` is the number of splats to allocate room for, which the plain
/// upload leaves at the splat count.
pub(super) fn build_gaussian_splat_set(
    device: &gpu::Device,
    queue: &gpu::Queue,
    data: &GaussianSplatData,
    capacity: u32,
) -> GaussianSplatGpuSet {
    let count = data.positions.len() as u32;
    let capacity = capacity.max(count);
    let sh_stride = (data.sh_degree.coeff_count() * 4) as u32;

    // Pad positions/scales/rotations to vec4 (w=1 / w=0 / raw).
    let pos_data: Vec<[f32; 4]> = data
        .positions
        .iter()
        .map(|p| [p[0], p[1], p[2], 1.0])
        .collect();
    let scale_data: Vec<[f32; 4]> = data
        .scales
        .iter()
        .map(|s| [s[0], s[1], s[2], 0.0])
        .collect();

    let usage = gpu::BufferUsages::STORAGE;
    let mut positions =
        ContentBuffer::new(device, "splat_position_buf", usage, VEC4_STRIDE, capacity);
    let mut scales = ContentBuffer::new(device, "splat_scale_buf", usage, VEC4_STRIDE, capacity);
    let mut rotations =
        ContentBuffer::new(device, "splat_rotation_buf", usage, VEC4_STRIDE, capacity);
    let mut opacities =
        ContentBuffer::new(device, "splat_opacity_buf", usage, OPACITY_STRIDE, capacity);
    // A set with no coefficients still needs something bindable, so the buffer
    // exists at one splat's worth and nothing is live in it.
    let mut sh = ContentBuffer::new(
        device,
        "splat_sh_buf",
        usage,
        sh_stride.max(4),
        if data.sh_coefficients.is_empty() {
            0
        } else {
            capacity
        },
    );

    let _ = positions.write_range(queue, 0, bytemuck::cast_slice(&pos_data));
    let _ = scales.write_range(queue, 0, bytemuck::cast_slice(&scale_data));
    let _ = rotations.write_range(queue, 0, bytemuck::cast_slice(&data.rotations));
    let _ = opacities.write_range(queue, 0, bytemuck::cast_slice(&data.opacities));
    if !data.sh_coefficients.is_empty() {
        let _ = sh.write_range(queue, 0, bytemuck::cast_slice(&data.sh_coefficients));
    }

    GaussianSplatGpuSet {
        positions,
        scales,
        rotations,
        opacities,
        sh,
        sh_degree: data.sh_degree,
        cpu_positions: data.positions.clone(),
        cpu_scales: data.scales.clone(),
    }
}

/// Persistent GPU state for one uploaded Gaussian splat set.
///
/// Every channel is a [`ContentBuffer`], so a consumer can rewrite part of a set
/// rather than replacing it. The buffers are addressed in splats, the SH channel
/// included: its stride is one splat's worth of coefficients.
pub(crate) struct GaussianSplatGpuSet {
    /// Centres as vec4<f32> (w=1), one per splat. Its live count is the draw
    /// count for the whole set.
    pub positions: ContentBuffer,
    /// Scales as vec4<f32> (w=0), one per splat.
    pub scales: ContentBuffer,
    /// Rotations as vec4<f32> [x,y,z,w], one per splat.
    pub rotations: ContentBuffer,
    /// Opacity per splat.
    pub opacities: ContentBuffer,
    /// SH coefficients, one splat's worth per element.
    pub sh: ContentBuffer,
    /// SH degree for this set, which fixes the SH stride.
    pub sh_degree: ShDegree,
    /// CPU centres kept for the proximity pick and the wireframe rings. A plain
    /// `Vec` rather than an `Arc`: a ranged write updates part of it, and behind
    /// an `Arc` shared with a frame snapshot that would mean cloning the whole
    /// thing on every write.
    pub cpu_positions: Vec<[f32; 3]>,
    /// CPU scales kept for picking and the wireframe overlay.
    pub cpu_scales: Vec<[f32; 3]>,
}

impl viewport_lib::resources::handle::GpuByteSize for GaussianSplatGpuSet {
    /// Resident GPU bytes for the persistent source buffers, reserved capacity
    /// included. Per-viewport sort scratch is derived and grows lazily, so it is
    /// not counted here.
    fn gpu_bytes(&self) -> u64 {
        self.positions.allocated_bytes()
            + self.scales.allocated_bytes()
            + self.rotations.allocated_bytes()
            + self.opacities.allocated_bytes()
            + self.sh.allocated_bytes()
    }
}

impl GaussianSplatGpuSet {
    /// Splats a draw reads.
    pub(crate) fn count(&self) -> u32 {
        self.positions.len()
    }

    /// Coefficients one splat occupies in the SH channel, which is what makes a
    /// ranged SH write addressable in splats.
    pub(crate) fn sh_coefficients_per_splat(&self) -> u32 {
        self.sh_degree.coeff_count() as u32
    }

    fn channel(&self, which: SplatChannel) -> &ContentBuffer {
        match which {
            SplatChannel::Positions => &self.positions,
            SplatChannel::Scales => &self.scales,
            SplatChannel::Rotations => &self.rotations,
            SplatChannel::Opacities => &self.opacities,
            SplatChannel::ShCoefficients => &self.sh,
        }
    }

    fn channel_mut(&mut self, which: SplatChannel) -> &mut ContentBuffer {
        match which {
            SplatChannel::Positions => &mut self.positions,
            SplatChannel::Scales => &mut self.scales,
            SplatChannel::Rotations => &mut self.rotations,
            SplatChannel::Opacities => &mut self.opacities,
            SplatChannel::ShCoefficients => &mut self.sh,
        }
    }

    /// Only the SH channel can be absent: a set uploaded with no coefficients
    /// holds a one-splat placeholder that the shader never indexes.
    fn has_channel(&self, which: SplatChannel) -> bool {
        match which {
            SplatChannel::ShCoefficients => !self.sh.is_empty(),
            _ => true,
        }
    }

    /// Write bytes into one channel at a splat offset.
    ///
    /// The CPU mirror follows for the two channels that have one, so picking and
    /// the wireframe rings keep agreeing with what is drawn. A write that leaves
    /// the mirror behind is a picture that picks in the wrong place, and it would
    /// not show up until someone clicked.
    pub(crate) fn write_channel(
        &mut self,
        queue: &gpu::Queue,
        which: SplatChannel,
        name: &'static str,
        first_element: u32,
        data: &[u8],
        mirror: Option<&[[f32; 3]]>,
    ) -> ViewportResult<()> {
        if !self.has_channel(which) {
            return Err(ViewportError::ChannelNotPresent {
                type_name: super::TYPE_NAME,
                channel: name,
            });
        }
        self.channel_mut(which)
            .write_range(queue, first_element, data)?;
        if let Some(values) = mirror {
            let target = match which {
                SplatChannel::Positions => &mut self.cpu_positions,
                SplatChannel::Scales => &mut self.cpu_scales,
                _ => return Ok(()),
            };
            let first = first_element as usize;
            if target.len() < first + values.len() {
                target.resize(first + values.len(), [0.0; 3]);
            }
            target[first..first + values.len()].copy_from_slice(values);
        }
        Ok(())
    }

    /// Grow every channel the set holds to at least `capacity` splats.
    ///
    /// `true` when an allocation was replaced, which is when the per-viewport
    /// sort scratch built over these buffers has to be dropped.
    pub(crate) fn reserve(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        capacity: u32,
    ) -> bool {
        let mut moved = false;
        for which in [
            SplatChannel::Positions,
            SplatChannel::Scales,
            SplatChannel::Rotations,
            SplatChannel::Opacities,
            SplatChannel::ShCoefficients,
        ] {
            if self.has_channel(which) {
                moved |= self.channel_mut(which).reserve(device, queue, capacity);
            }
        }
        if moved {
            self.cpu_positions.resize(capacity as usize, [0.0; 3]);
            self.cpu_scales.resize(capacity as usize, [0.0; 3]);
        }
        moved
    }

    /// Set how many splats draw, across every channel the set holds.
    pub(crate) fn set_live_len(&mut self, len: u32) -> ViewportResult<()> {
        for which in [
            SplatChannel::Positions,
            SplatChannel::Scales,
            SplatChannel::Rotations,
            SplatChannel::Opacities,
            SplatChannel::ShCoefficients,
        ] {
            if self.has_channel(which) {
                self.channel_mut(which).set_len(len)?;
            }
        }
        Ok(())
    }

    /// What one channel holds and how much of it draws.
    pub(crate) fn extent(&self, which: SplatChannel) -> Extent {
        let cb = self.channel(which);
        Extent {
            capacity: cb.capacity(),
            len: cb.len(),
        }
    }
}

/// Slotted store for Gaussian splat sets.
///
/// A removed set leaves an empty slot that a later insert reuses. Each slot
/// carries a generation bumped on removal, and a [`GaussianSplatId`] captures
/// the generation it was issued against, so a stale handle resolves to `None`
/// rather than aliasing the set now in its slot. An entry's byte charge is its
/// [`GpuByteSize::gpu_bytes`](viewport_lib::resources::handle::GpuByteSize::gpu_bytes),
/// and its revision is what the item type keys its per-viewport sort scratch on.
pub(crate) type GaussianSplatStore =
    viewport_lib::resources::handle::SlotStore<GaussianSplatGpuSet, GaussianSplatId>;
