//! The point cloud's writable channels.
//!
//! One marker per array a consumer can write part of. Pass one to the
//! [`Writes`](viewport_lib::plugin_api::Writes) calls to say which array is
//! meant:
//!
//! ```no_run
//! # use viewport_lib::plugin_api::Writes;
//! # use viewport_lib_plugins::item_types::point_cloud::channels as pc;
//! # fn f(
//! #     renderer: &mut viewport_lib::renderer::ViewportRenderer,
//! #     device: &viewport_lib::gpu::Device,
//! #     queue: &viewport_lib::gpu::Queue,
//! #     id: viewport_lib_plugins::item_types::point_cloud::PointCloudId,
//! #     sector: &[[f32; 3]],
//! # ) -> viewport_lib::error::ViewportResult<()> {
//! renderer.reserve(pc::Positions, device, queue, id, 1_000_000)?;
//! renderer.write_range(pc::Positions, queue, id, 250_000, sector)?;
//! # Ok(()) }
//! ```
//!
//! Every channel is indexed by point, so a cloud reserves and sizes all of them
//! together: which marker a [`reserve`](viewport_lib::plugin_api::Writes::reserve)
//! or [`set_len`](viewport_lib::plugin_api::Writes::set_len) names makes no
//! difference. Only the writes are per channel.
//!
//! Which channels a cloud has is settled when it is uploaded. A cloud uploaded
//! with `ColourSource::Solid` has no per-point colour buffer, and writing one
//! would mean allocating and rebuilding the bind group under what the caller
//! believes is a cheap call, so it is refused. Upload the channel populated, with
//! placeholder values if that is all there is, and then write it.

use super::types::PointCloudId;
use viewport_lib::Colour;
use viewport_lib::plugin_api::Channel;

/// World-space point positions. Always present.
#[derive(Copy, Clone, Debug, Default)]
pub struct Positions;

/// Per-point scalars the colourmap maps.
///
/// Present when the cloud was uploaded with `ColourSource::Scalar`. The domain
/// the colourmap spans is fixed at upload, so the source must supply it
/// (`range: Some(..)`): a derived domain covers the whole array and a ranged
/// write cannot know the new extremes without rescanning everything, which is
/// the cost being avoided.
#[derive(Copy, Clone, Debug, Default)]
pub struct Scalars;

/// Per-point colours. Present when the cloud was uploaded with
/// `ColourSource::PerSample`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Colours;

/// Per-point sizes, in the unit the cloud draws in.
///
/// Present when the cloud was uploaded with any `SizeSource` other than
/// `Uniform`. These are final sizes, not the values a `SizeSource::Scalar` maps:
/// the mapping happened at upload, and a ranged write supplies the mapped
/// result. A consumer streaming sizes through a scalar mapping applies it
/// itself.
#[derive(Copy, Clone, Debug, Default)]
pub struct Sizes;

/// Per-point transparency. Present when the uploaded cloud carried any.
#[derive(Copy, Clone, Debug, Default)]
pub struct Transparencies;

impl Channel for Positions {
    type Id = PointCloudId;
    type Input = [f32; 3];
    const NAME: &'static str = "positions";
}

impl Channel for Scalars {
    type Id = PointCloudId;
    type Input = f32;
    const NAME: &'static str = "scalars";
}

impl Channel for Colours {
    type Id = PointCloudId;
    type Input = Colour;
    const NAME: &'static str = "colours";
}

impl Channel for Sizes {
    type Id = PointCloudId;
    type Input = f32;
    const NAME: &'static str = "sizes";
}

impl Channel for Transparencies {
    type Id = PointCloudId;
    type Input = f32;
    const NAME: &'static str = "transparencies";
}
