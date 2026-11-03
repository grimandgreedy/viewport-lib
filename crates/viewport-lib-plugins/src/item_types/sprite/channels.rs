//! The sprite batch's writable channels.
//!
//! Two channels: the positions, which are a vertex stream of their own, and the
//! per-sprite records, which are one interleaved 64-byte block each. Splitting
//! them matters because a particle feed usually moves every sprite and recolours
//! none of them, so the position write is a quarter of the traffic of a full
//! update.
//!
//! The record is not what a caller holds, so [`Sprite`] is the input and the
//! write builds the record: a [`Colour`] becomes linear RGBA, and the defaults an
//! item applies to short per-sprite lists are already resolved by the time a
//! ranged write supplies a value.
//!
//! These cover the batch a `SpriteSetId` names. The instance-set store keeps its
//! own handle type and is not writable this way.

use super::types::SpriteSetId;
use viewport_lib::Colour;
use viewport_lib::plugin_api::Channel;

/// One sprite of a batch, as the caller supplies it for a ranged write.
///
/// Total rather than partial: the store keeps no CPU copy of a record, so
/// "leave the colour alone" would mean reading it back off the GPU. Every write
/// supplies every field, and the values are final rather than defaults to be
/// filled in.
#[derive(Copy, Clone, Debug)]
pub struct Sprite {
    /// Final colour, multiplied into the texture sample.
    pub colour: Colour,
    /// Size in the unit the batch's size mode measures in.
    pub size: f32,
    /// Rotation about the view axis, in radians.
    pub rotation: f32,
    /// Soft-particle fade distance. `0.0` disables the fade for this sprite.
    pub soft_distance: f32,
    /// Sub-rectangle of the batch's texture as `[u0, v0, u1, v1]`. Use
    /// `[0.0, 0.0, 1.0, 1.0]` for the whole texture.
    pub uv_rect: [f32; 4],
    /// Velocity, read only when the batch orients itself by it.
    pub velocity: [f32; 3],
}

impl Default for Sprite {
    /// A white sprite of unit size, unrotated, covering its whole texture.
    fn default() -> Self {
        Self {
            colour: Colour::WHITE,
            size: 1.0,
            rotation: 0.0,
            soft_distance: 0.0,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            velocity: [0.0; 3],
        }
    }
}

/// World-space sprite positions, one per sprite. Their live count is the batch's
/// draw count.
#[derive(Copy, Clone, Debug, Default)]
pub struct Positions;

/// The interleaved per-sprite records: colour, size, rotation, soft-particle
/// distance, uv rect and velocity.
#[derive(Copy, Clone, Debug, Default)]
pub struct Sprites;

impl Channel for Positions {
    type Id = SpriteSetId;
    type Input = [f32; 3];
    const NAME: &'static str = "positions";
}

impl Channel for Sprites {
    type Id = SpriteSetId;
    type Input = Sprite;
    const NAME: &'static str = "sprites";
}
