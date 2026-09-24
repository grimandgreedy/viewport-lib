//! The blend mode a batch of transparent geometry draws with.
//!
//! Declared here rather than beside any one item type because three
//! unrelated submissions select a pipeline with it: instanced mesh batches,
//! ribbons, and billboard sprites, including the ones a GPU particle system
//! draws. The instancing arena keys its additive and premultiplied buffers on
//! it, so it is part of the shared substrate rather than of a single type.

/// GPU blend state used when drawing a batch of transparent geometry.
///
/// `AlphaBlend` is the default and matches normal transparent sprites.
/// `Additive` accumulates colour into the framebuffer without subtracting
/// background, which is the usual choice for sparks, fire, and other
/// emissive particles. `Premultiplied` is for sources whose RGB has already
/// been multiplied by alpha, typically when sampling a premultiplied texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpriteBlend {
    /// Standard transparency: `src.rgb * src.a + dst.rgb * (1 - src.a)`.
    #[default]
    AlphaBlend,
    /// Additive: `src.rgb + dst.rgb`. Alpha is unused for the colour result.
    Additive,
    /// Premultiplied alpha: `src.rgb + dst.rgb * (1 - src.a)`.
    Premultiplied,
}
