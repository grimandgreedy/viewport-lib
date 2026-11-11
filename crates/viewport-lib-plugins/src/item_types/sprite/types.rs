use viewport_lib_types::colour::Colour;
use viewport_lib_types::ids::TextureId;
use viewport_lib_types::material::{ItemSettings, TextureFilter};
use viewport_lib_types::plugin::material::ParamValue;
use viewport_lib_types::render_item::{MaskTest, SpriteBlend};

viewport_lib_types::slot_handle! {
    /// Handle to a sprite batch uploaded once through
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload).
    /// Backs static billboards such as foliage, signage and light flares.
    pub struct SpriteSetId;
}

viewport_lib_types::slot_handle! {
    /// Handle to a sprite instance set uploaded once through
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload).
    /// Backs entity sprites such as NPCs, item drops and damage numbers.
    pub struct SpriteInstanceSetId;
}

/// Controls whether sprite sizes are measured in screen-space points or world-space units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpriteSizeMode {
    /// Sizes in points (device-independent pixels): a sprite of size `s`
    /// covers `s` points, `s * pixels_per_point` device pixels, at every
    /// distance and zoom. The default, for markers and particles in a 3D view.
    /// A sprite in painter's order with this size raises
    /// `NoticeDetail::PainterSpriteScreenSized`, since a 2D scene wants
    /// world sizes.
    #[default]
    ScreenSpace,
    /// Sizes in world-space units. Sprites shrink with distance like regular geometry.
    WorldSpace,
}

/// How a sprite's image fills its quad.
///
/// Evaluated per fragment against the sprite's UV rect and size, so resizing
/// a sliced or tiled sprite costs nothing on the CPU. Flips, pivots, the
/// texture filter, effects and masks all apply after it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SpriteDrawMode {
    /// The UV rect stretched across the quad. The default.
    #[default]
    Simple,
    /// Nine-slice: the borders keep their size and the centre stretches.
    Sliced {
        /// Border widths `[left, bottom, right, top]` in texels of the
        /// texture (in size units for an untextured sprite). A border draws
        /// one texel per unit of the sprite's size, so a sprite drawn at its
        /// texel size shows the image unchanged; a quad too small for two
        /// opposite borders shrinks them in proportion.
        border: [f32; 4],
    },
    /// The UV rect repeated across the quad from its top-left corner.
    Tiled {
        /// One tile's size in the sprite's size units.
        tile_size: [f32; 2],
        /// What happens where the quad is not a whole number of tiles.
        mode: TileMode,
    },
}

/// How a [`SpriteDrawMode::Tiled`] sprite fills a quad that is not a whole
/// number of tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TileMode {
    /// Tiles keep their size and the last one along each axis (the right
    /// and bottom ones, unflipped) is cut off. The default.
    #[default]
    Continuous,
    /// The nearest whole number of tiles (at least one) is drawn, each
    /// stretched to fill the quad.
    Adaptive,
}

/// A batch of instanced billboard sprites rendered as camera-facing textured quads.
///
/// Each instance is one billboard at a world-space position. All instances in the batch share
/// one texture (or render as solid-colour quads when `texture_id` is `None`). Per-instance
/// colour, size, rotation, and atlas UV rect are specified via parallel `Vec` fields; empty
/// vecs fall back to the batch defaults.
///
/// # Particle effects
///
/// Submit a new `SpriteItem` each frame with updated `positions` and `colours` to animate
/// CPU-simulated particle effects. The host application owns simulation state (velocity,
/// lifetime, emission); the renderer only handles drawing.
///
/// # Texture atlases
///
/// Set `uv_rects` to select sub-regions of the texture per sprite, enabling flip-book
/// animation or mixed icon sets from a single atlas texture.
#[non_exhaustive]
#[derive(Clone)]
pub struct SpriteItem {
    /// Texture ID from `DeviceResources::upload_texture`.
    /// `None` renders solid-colour quads using `colours` / `default_colour` only.
    ///
    /// Colour, so upload it sRGB
    /// ([`TextureData::srgb`](viewport_lib_types::data::texture::TextureData::srgb)).
    pub texture_id: Option<viewport_lib_types::ids::TextureId>,
    /// World-space positions, one per sprite instance.
    pub positions: Vec<[f32; 3]>,
    /// Per-instance RGBA colour tints. Empty = use `default_colour` for all.
    /// Multiplied with the texture sample (or used directly when `texture_id` is `None`).
    pub colours: Vec<viewport_lib_types::colour::Colour>,
    /// Per-instance sizes as `[width, height]` along the quad's right and up
    /// axes. Empty = use `default_size` for all. Interpretation depends on
    /// `size_mode`. [`SpriteItem::square_sizes`] builds the list from one edge
    /// length per sprite.
    pub sizes: Vec<[f32; 2]>,
    /// Per-instance rotation angles in radians, CCW around the camera-forward axis.
    /// Empty = no rotation applied. The quad rotates about its pivot.
    pub rotations: Vec<f32>,
    /// Per-instance UV rects `[u0, v0, u1, v1]` selecting atlas sub-regions,
    /// with `v = 0` at the image's top row: `(u0, v0)` draws at the quad's
    /// top-left corner, so an image uploaded top row first, as image loaders
    /// return it, draws upright. Empty = full texture `[0.0, 0.0, 1.0, 1.0]`
    /// for all.
    pub uv_rects: Vec<[f32; 4]>,
    /// Per-instance pivots: the point of the quad that sits at the sprite's
    /// position, as `[x, y]` in `0..=1` of the quad (`[0, 0]` the bottom-left
    /// corner, `[1, 1]` the top-right, along the quad's right and up axes).
    /// Rotation and size act about it. Empty = `default_pivot` for all.
    pub pivots: Vec<[f32; 2]>,
    /// Pivot used when `pivots` is empty or shorter than `positions`.
    /// Default `[0.5, 0.5]`, the quad's centre.
    pub default_pivot: [f32; 2],
    /// Per-instance flips, mirroring the sprite's UV rect horizontally and/or
    /// vertically. Empty = no flip for all.
    pub flips: Vec<SpriteFlip>,
    /// Fallback RGBA colour tint used when `colours` is empty. Default: opaque white.
    pub default_colour: viewport_lib_types::colour::Colour,
    /// Default edge length when `sizes` is empty or shorter than `positions`:
    /// a square sprite. Pixels (ScreenSpace) or world units (WorldSpace).
    pub default_size: f32,
    /// Whether sizes are in screen-space pixels or world-space units.
    pub size_mode: SpriteSizeMode,
    /// World-space model transform applied to all positions. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Whether this batch writes to the depth buffer. Default: `false`.
    ///
    /// Set `false` for transparent or additive particle effects so sprites do not occlude
    /// each other based on submission order. Set `true` for opaque world-space markers
    /// that should participate in depth testing normally.
    pub depth_write: bool,
    /// GPU blend state for this batch. Default: [`SpriteBlend::AlphaBlend`].
    pub blend: SpriteBlend,
    /// If `Some(d)`, fades the sprite's alpha as it approaches opaque scene geometry
    /// behind it, over a distance of `d` world-space units. `None` disables soft fade.
    ///
    /// Useful for transparent particles like smoke or fog to avoid hard intersection
    /// lines against walls and ground. Requires a sampleable scene depth resolve;
    /// see crate docs for current support status.
    ///
    /// Used as the fallback when [`Self::soft_particle_distances`] is empty or shorter
    /// than `positions`, or when a per-instance entry is zero.
    pub soft_particle_distance: Option<f32>,
    /// Per-instance soft-fade distances, in world-space units. Lets a single batch
    /// mix large smoke puffs (long fade) and small embers (short fade) without
    /// splitting the draw. Empty falls back to [`Self::soft_particle_distance`]
    /// for every instance; a zero entry falls back to the batch default as well.
    pub soft_particle_distances: Vec<f32>,
    /// If `Some(d)`, the sprite is drawn as a refractive distortion rather
    /// than a normal alpha-blended billboard: the renderer samples the
    /// already-resolved scene colour at an offset driven by the sprite's
    /// texture (red/green channels as signed displacement, alpha as mask)
    /// and writes the distorted result back to the scene. `d` scales the
    /// displacement in NDC pixels. Use for heat haze, shockwaves,
    /// force-field hits, water-splash droplets.
    ///
    /// Available on the HDR path only; the direct LDR `paint_to` cannot
    /// resolve scene colour for sampling (same constraint as soft particles).
    /// `None` disables refraction and the sprite draws normally.
    pub refraction_strength: Option<f32>,
    /// How each billboard is oriented in world space. Default
    /// [`SpriteOrientation::CameraFacing`] matches the historical behaviour:
    /// every quad turns to face the camera.
    pub orientation: SpriteOrientation,
    /// Per-instance velocity vectors. Used by
    /// [`SpriteOrientation::VelocityStretched`] to align each quad's long axis
    /// with the projected motion direction. Empty disables stretching for that
    /// instance (it falls back to a square camera-facing quad).
    pub velocities: Vec<[f32; 3]>,
    /// World-space axis used by [`SpriteOrientation::AxisLocked`] for the
    /// quad's long axis. Ignored in other orientation modes. Default
    /// `[0, 0, 1]` (world up).
    pub axis: [f32; 3],
    /// When `true`, the batch runs through the lit sprite pipeline and
    /// participates in the scene's hemisphere ambient and direct lighting.
    /// Default `false` preserves the emissive billboard look.
    pub lit: bool,
    /// Lighting parameters used when `lit` is `true`. Ignored otherwise.
    pub lit_params: SpriteLitParams,
    /// Texture sampled as a tangent-space normal map when `lit_params.normal_mode`
    /// is [`SpriteNormalMode::NormalMap`]. `None` falls back to the spherical
    /// normal even when the mode requests a map.
    ///
    /// Directions, not colour, so upload it linear
    /// ([`TextureData::normal_map`](viewport_lib_types::data::texture::TextureData::normal_map)).
    pub normal_texture_id: Option<viewport_lib_types::ids::TextureId>,
    /// How the batch's texture is filtered. Default [`TextureFilter::Linear`];
    /// [`TextureFilter::Nearest`] keeps pixel art crisp under magnification.
    pub filter: TextureFilter,
    /// When `true`, each sprite's position is snapped to a whole device pixel
    /// after projection, so pixel art drawn at an integer scale keeps every
    /// texel on the pixel grid. Sprites whose size covers an odd number of
    /// pixels need a pivot on a pixel boundary (a corner, say) to stay aligned.
    /// Default `false`.
    pub pixel_snap: bool,
    /// How blended sprites in this batch are ordered against other blended
    /// content. Default [`SpriteOrder::Depth`].
    pub order: SpriteOrder,
    /// Not lit by 2D lighting: the sprites do not sample the light layer of
    /// their painter range (layer 0 for `SpriteOrder::Depth`) and keep their
    /// colour (an emissive sprite, a glow). A [`SpriteOrder::Painter`] batch
    /// also draws after its range's lit content, so it sits over what it
    /// lights. Unrelated to [`lit`](Self::lit), which shades with the scene's
    /// 3D lights. Default `false`.
    pub unlit: bool,
    /// Per-instance sort keys for [`SpriteOrder::Painter`]: lower keys draw
    /// first and higher keys draw on top, within the batch and across the
    /// painter batches on the same layer, of this type or another (a blended
    /// tile layer sorts at key `0.0`). A top-down game passes `-y` so sprites
    /// further down the screen cover the ones behind them. Empty keeps
    /// submission order and sorts at `0.0`. Ignored by `SpriteOrder::Depth`.
    ///
    /// Every sprite takes its place in the painter sequence by its own key,
    /// against this type's batches and every other type's painter content
    /// (compose keys with `viewport_lib::painter_key` to keep an entity's
    /// parts together). Consecutive sprites of one batch with nothing of
    /// another batch between them still draw as one call, so batches whose
    /// key ranges do not overlap draw whole, and two large batches with evenly
    /// mixed keys cost up to a draw per sprite.
    ///
    /// The batch is sorted when it is built. An uploaded painter set is
    /// re-sorted only by a ranged write to its sort keys
    /// ([`channels::SortKeys`](crate::item_types::sprite::channels::SortKeys)),
    /// at the next frame; a write that does not include keys never re-sorts,
    /// so moved sprites keep their place in the order. The renderer does not
    /// derive keys from positions: a set sorted by `-y` writes its keys
    /// together with its positions, and a positions-only write suits a set
    /// whose keys do not depend on where its sprites are.
    pub sort_keys: Vec<f32>,
    /// When set, a GPU pick discards the sprite's fragments whose alpha (the
    /// texel's alpha times the sprite's colour alpha) is below the threshold,
    /// so a pick on a transparent part of a cut-out sprite misses it and finds
    /// what is underneath. `None` (the default) picks the whole quad.
    pub pick_alpha_threshold: Option<f32>,
    /// How the image fills each quad: stretched, nine-sliced or tiled.
    /// Default [`SpriteDrawMode::Simple`].
    pub draw_mode: SpriteDrawMode,
    /// Whether a painter mask cuts this batch. Read only in painter's order
    /// ([`SpriteOrder::Painter`]): the batch is cut when its layer is in the
    /// range of the mask in the slot it names. Default [`MaskTest::None`].
    pub mask: MaskTest,
    /// Flash, dissolve, palette swap, outline and a custom sprite material.
    /// `None` (the default) draws through the plain sprite pipelines and costs
    /// nothing; `Some` draws through the effect pipelines. See
    /// [`SpriteEffects`].
    pub effects: Option<SpriteEffects>,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl SpriteItem {
    /// `sizes` for square sprites, from one edge length per sprite.
    pub fn square_sizes(sizes: impl IntoIterator<Item = f32>) -> Vec<[f32; 2]> {
        sizes.into_iter().map(|s| [s, s]).collect()
    }
}

/// Which of a sprite's axes are mirrored. Flipping swaps the matching edges
/// of the sprite's UV rect, so the quad, its pivot and its rotation are
/// unchanged and only the image is mirrored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct SpriteFlip {
    /// Mirror left to right.
    pub horizontal: bool,
    /// Mirror top to bottom.
    pub vertical: bool,
}

impl SpriteFlip {
    /// No flip.
    pub const NONE: Self = Self {
        horizontal: false,
        vertical: false,
    };
    /// Mirrored left to right.
    pub const HORIZONTAL: Self = Self {
        horizontal: true,
        vertical: false,
    };
    /// Mirrored top to bottom.
    pub const VERTICAL: Self = Self {
        horizontal: false,
        vertical: true,
    };
    /// Mirrored on both axes.
    pub const BOTH: Self = Self {
        horizontal: true,
        vertical: true,
    };

    /// `uv_rect` with the flipped edges swapped.
    pub fn apply(self, uv_rect: [f32; 4]) -> [f32; 4] {
        let [mut u0, mut v0, mut u1, mut v1] = uv_rect;
        if self.horizontal {
            std::mem::swap(&mut u0, &mut u1);
        }
        if self.vertical {
            std::mem::swap(&mut v0, &mut v1);
        }
        [u0, v0, u1, v1]
    }
}

/// How a batch's blended sprites are ordered against other blended content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpriteOrder {
    /// The scene's usual rules: a depth-writing batch is ordered by the depth
    /// buffer, and a blended one composites through order-independent
    /// transparency (or, with a soft fade or a blend mode transparency cannot
    /// carry, draws after the opaque scene in submission order).
    #[default]
    Depth,
    /// Drawn in an exact order after the opaque scene and the resolved
    /// order-independent transparency, never through it: by `layer` (lower
    /// first), then by [`SpriteItem::sort_keys`], then submission order, in
    /// one sequence with the other painter's-order content (blended tile
    /// layers, painter-placed meshes and particles) and the world overlay
    /// layers. A 3D transparent surface in front of a painter sprite therefore
    /// draws behind it. Depth writes are off whatever `depth_write` says. Use
    /// for 2D scenes that stack cutouts.
    Painter {
        /// Layer order, for example the `order` of the scene layer the batch
        /// belongs to. Lower layers draw first.
        layer: u32,
        /// When `true` the sprites still test against scene depth, so nearer
        /// 3D geometry covers them; when `false` they draw over everything
        /// already in the scene.
        depth_test: bool,
    },
}

/// How a lit sprite recovers a per-fragment normal for shading.
///
/// Picked per batch; affects sprites whose [`SpriteItem::lit`] is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpriteNormalMode {
    /// Treat the quad as the cross-section of a sphere. The normal points from
    /// the quad centre outward through the fragment, giving a soft round
    /// falloff. The standard pick for smoke, dust, and fog.
    #[default]
    Spherical,
    /// Use the quad's screen-facing normal. Cheapest; suits grass cards and
    /// other already-oriented art.
    Flat,
    /// Sample a tangent-space normal map. The map's RGB encodes the perturbed
    /// normal; the quad's local right/up axes serve as the tangent basis.
    /// Falls back to `Spherical` when no normal map is bound.
    NormalMap,
}

/// Lighting parameters for a lit sprite batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteLitParams {
    /// Surface roughness in `[0, 1]`. Used to attenuate the specular response;
    /// `1.0` is fully diffuse. Smoke and dust look right near `0.9`.
    pub roughness: f32,
    /// How per-fragment normals are derived.
    pub normal_mode: SpriteNormalMode,
    /// Whether the batch samples the directional shadow map. Currently a
    /// reserved flag: the lit sprite shader leaves shadows unsampled and the
    /// hemisphere ambient + direct light contributions handle the look.
    pub receive_shadows: bool,
    /// Multiplier on the hemisphere ambient term, `1.0` matches the mesh
    /// shading path. Lower values darken the unlit side of smoke.
    pub ambient_scale: f32,
}

impl Default for SpriteLitParams {
    fn default() -> Self {
        Self {
            roughness: 0.9,
            normal_mode: SpriteNormalMode::default(),
            receive_shadows: false,
            ambient_scale: 1.0,
        }
    }
}

/// How a sprite batch's billboards are oriented in world space.
///
/// Picked per batch; all instances in one [`SpriteItem`] share the same mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpriteOrientation {
    /// Every quad turns to face the camera. This is the standard particle and
    /// marker behaviour and the historical default.
    #[default]
    CameraFacing,
    /// Each quad's long axis follows the projected per-instance velocity
    /// vector, with length scaled by the velocity magnitude. Use for sparks,
    /// muzzle flashes, bullets, and rain streaks. Requires the matching entry
    /// in [`SpriteItem::velocities`] for each instance.
    VelocityStretched,
    /// Each quad's long axis is locked to a single world-space direction
    /// shared by every instance. Use for grass cards, plume columns, and
    /// anything tethered to world up. The axis comes from [`SpriteItem::axis`].
    AxisLocked,
}

impl Default for SpriteItem {
    fn default() -> Self {
        Self {
            texture_id: None,
            positions: Vec::new(),
            colours: Vec::new(),
            sizes: Vec::new(),
            rotations: Vec::new(),
            uv_rects: Vec::new(),
            pivots: Vec::new(),
            default_pivot: [0.5, 0.5],
            flips: Vec::new(),
            default_colour: Colour::linear(1.0, 1.0, 1.0, 1.0),
            default_size: 32.0,
            size_mode: SpriteSizeMode::ScreenSpace,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            depth_write: false,
            blend: SpriteBlend::AlphaBlend,
            soft_particle_distance: None,
            soft_particle_distances: Vec::new(),
            refraction_strength: None,
            orientation: SpriteOrientation::default(),
            velocities: Vec::new(),
            axis: [0.0, 0.0, 1.0],
            lit: false,
            lit_params: SpriteLitParams::default(),
            normal_texture_id: None,
            filter: TextureFilter::Linear,
            pixel_snap: false,
            order: SpriteOrder::Depth,
            unlit: false,
            sort_keys: Vec::new(),
            effects: None,
            pick_alpha_threshold: None,
            draw_mode: SpriteDrawMode::Simple,
            mask: MaskTest::None,
            settings: ItemSettings::default(),
        }
    }
}

#[cfg(test)]
mod lit_sprite_tests {
    use super::*;

    #[test]
    fn sprite_lit_defaults_match_emissive_behaviour() {
        let s = SpriteItem::default();
        assert!(!s.lit);
        assert!(s.normal_texture_id.is_none());
        // Default lit_params should be the safe set: spherical normals, no
        // shadow sampling, neutral roughness, full ambient.
        assert_eq!(s.lit_params.normal_mode, SpriteNormalMode::Spherical);
        assert!(!s.lit_params.receive_shadows);
        assert!((s.lit_params.ambient_scale - 1.0).abs() < 1e-6);
        assert!(s.lit_params.roughness >= 0.0 && s.lit_params.roughness <= 1.0);
    }

    #[test]
    fn sprite_lit_fields_compose_with_other_options() {
        // A reasonably exotic configuration: lit + velocity-stretched +
        // soft-particle fade + per-instance distances + a normal map.
        let item = SpriteItem {
            lit: true,
            lit_params: SpriteLitParams {
                roughness: 0.4,
                normal_mode: SpriteNormalMode::NormalMap,
                receive_shadows: true,
                ambient_scale: 0.2,
            },
            normal_texture_id: Some(viewport_lib_types::ids::TextureId::from_raw(7)),
            orientation: SpriteOrientation::VelocityStretched,
            velocities: vec![[1.0, 0.0, 0.0]],
            positions: vec![[0.0, 0.0, 0.0]],
            soft_particle_distance: Some(0.5),
            soft_particle_distances: vec![1.2],
            ..SpriteItem::default()
        };
        assert!(item.lit);
        assert_eq!(item.lit_params.normal_mode, SpriteNormalMode::NormalMap);
        assert!(item.lit_params.receive_shadows);
        assert_eq!(
            item.normal_texture_id,
            Some(viewport_lib_types::ids::TextureId::from_raw(7))
        );
        assert_eq!(item.orientation, SpriteOrientation::VelocityStretched);
    }

    #[test]
    fn sprite_normal_mode_default_is_spherical() {
        assert_eq!(SpriteNormalMode::default(), SpriteNormalMode::Spherical);
    }
}

/// Per-frame reference to a pre-uploaded sprite set (static billboards).
///
/// The sprite set's positions, texture, sizes, and colours are baked at
/// upload time. The ref item lets the renderer redraw the same set every
/// frame with no per-frame upload cost.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpriteSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload)
    /// or `Uploads::begin_upload`.
    pub source: SpriteSetId,
    /// Per-item render settings (visibility, wireframe, selection).
    pub settings: ItemSettings,
}

impl SpriteSetRefItem {
    /// Visible reference at the upload-time transform.
    pub fn new(id: SpriteSetId) -> Self {
        Self {
            source: id,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded sprite instance set (entity sprites).
///
/// The sprites (texture, blend, size mode and every sprite's record) are
/// baked at upload time and drawn as uploaded, unless
/// [`overrides`](Self::overrides) changes some of them for this frame.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpriteInstanceSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload)
    /// or `Uploads::begin_upload`.
    pub source: SpriteInstanceSetId,
    /// Positions, colours and sort keys for some of the set's sprites, for
    /// this frame only. `None` (the default) draws the set as uploaded.
    pub overrides: Option<SpriteOverrides>,
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl SpriteInstanceSetRefItem {
    /// Visible reference drawing the set as uploaded.
    pub fn new(id: SpriteInstanceSetId) -> Self {
        Self {
            source: id,
            overrides: None,
            settings: ItemSettings::default(),
        }
    }

    /// This reference with `overrides` applied for the frame.
    pub fn with_overrides(mut self, overrides: SpriteOverrides) -> Self {
        self.overrides = Some(overrides);
        self
    }
}

/// Per-frame values for a subset of an uploaded instance set's sprites,
/// applied on top of the set for one frame without changing it.
///
/// `indices` names the sprites by their index in the uploaded set. The other
/// lists run parallel to it: entry `j` of `positions` replaces the position of
/// sprite `indices[j]`, and likewise for `colours` and `sort_keys`. A list that
/// is empty, or shorter than `indices`, keeps the uploaded value for the
/// sprites past its end. Indices past the end of the set are ignored; a
/// repeated index takes its last entry.
///
/// On a painter set ([`SpriteOrder::Painter`]) an overridden sort key moves
/// the sprite to its place in the key order for the frame, so a sprite that
/// walks down a y-sorted set can carry its new key here.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct SpriteOverrides {
    /// Indices into the uploaded set of the sprites to change.
    pub indices: Vec<u32>,
    /// World-space positions, parallel to `indices`.
    pub positions: Vec<[f32; 3]>,
    /// Colour tints, parallel to `indices`. They replace the uploaded tint.
    pub colours: Vec<Colour>,
    /// Sort keys, parallel to `indices`. Read by painter sets only.
    pub sort_keys: Vec<f32>,
}

impl SpriteOverrides {
    /// Overrides for the sprites at `indices`, with no values yet.
    pub fn new(indices: Vec<u32>) -> Self {
        Self {
            indices,
            ..Self::default()
        }
    }

    /// Whether applying these changes nothing.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
            || (self.positions.is_empty() && self.colours.is_empty() && self.sort_keys.is_empty())
    }
}

/// Effects drawn over a sprite batch: a flash towards a colour, a dissolve
/// against a noise texture with a burning rim, a palette swap, an outline, and
/// a custom [`SpriteMaterial`].
///
/// The flash amount, the dissolve threshold and the palette row are per
/// sprite, each from a list that falls back to a batch default where it is
/// empty or short, so one enemy in a batch can flash while the rest do not.
/// The colours, the textures and the outline are per batch.
///
/// The order is fixed: the palette swap replaces the sampled texel, the tint
/// multiplies it, then (after lighting, for a lit batch) the flash mixes the
/// colour towards `flash_colour`, the dissolve cuts it away, the outline draws
/// behind it and the material's `shade_sprite` runs last.
///
/// A batch with effects draws after the opaque scene in submission order (or
/// in painter's order, or in the opaque pass when it writes depth), never
/// through order-independent transparency.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SpriteEffects {
    /// Per-sprite flash amounts in `0..=1`: `0` leaves the colour alone, `1`
    /// draws the sprite's shape in `flash_colour`. Empty = `default_flash`.
    pub flash: Vec<f32>,
    /// Flash amount for sprites past the end of `flash`. Default `0`.
    pub default_flash: f32,
    /// What a flash mixes towards. Default white.
    pub flash_colour: Colour,
    /// Per-sprite dissolve thresholds in `0..=1`: texels whose noise value is
    /// below the threshold are cut away, so `0` shows the whole sprite and `1`
    /// none of it. Empty = `default_dissolve`.
    pub dissolve: Vec<f32>,
    /// Dissolve threshold for sprites past the end of `dissolve`. Default `0`.
    pub default_dissolve: f32,
    /// Noise the dissolve thresholds against, read from its red channel across
    /// the sprite's quad (not its atlas cell). Upload it linear
    /// ([`TextureData::linear`](viewport_lib_types::data::texture::TextureData::linear)).
    /// `None` uses a built-in value noise.
    pub dissolve_noise: Option<TextureId>,
    /// How many times the noise repeats across the quad. Default `1`.
    pub noise_scale: f32,
    /// Colour of the rim a dissolve leaves, drawn where the noise is within
    /// `dissolve_edge_width` above the threshold. Default orange.
    pub dissolve_edge_colour: Colour,
    /// Width of the rim in noise units (`0..1`). `0` draws no rim. Default
    /// `0.05`.
    pub dissolve_edge_width: f32,
    /// Palette swap: when set, each texel's index picks a column of the
    /// palette texture and the sprite's palette row picks the row. `None`
    /// samples the texture as colour.
    pub palette: Option<SpritePalette>,
    /// Per-sprite palette rows. Empty = `default_palette_row`.
    pub palette_rows: Vec<u16>,
    /// Palette row for sprites past the end of `palette_rows`. Default `0`.
    pub default_palette_row: u16,
    /// An outline around the sprite's opaque texels. `None` draws none.
    pub outline: Option<SpriteOutline>,
    /// A custom fragment stage registered with
    /// [`SpriteMaterials::register_sprite_material`](crate::item_types::sprite::SpriteMaterials::register_sprite_material).
    pub material: Option<SpriteMaterial>,
}

impl Default for SpriteEffects {
    fn default() -> Self {
        Self {
            flash: Vec::new(),
            default_flash: 0.0,
            flash_colour: Colour::WHITE,
            dissolve: Vec::new(),
            default_dissolve: 0.0,
            dissolve_noise: None,
            noise_scale: 1.0,
            dissolve_edge_colour: Colour::linear(1.0, 0.45, 0.05, 1.0),
            dissolve_edge_width: 0.05,
            palette: None,
            palette_rows: Vec::new(),
            default_palette_row: 0,
            outline: None,
            material: None,
        }
    }
}

impl SpriteEffects {
    /// The packed per-sprite word the effect shaders read for sprite `i`:
    /// flash in bits 0-7 and dissolve in bits 8-15 (both unorm8), palette row
    /// in bits 16-31.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn instance_word(&self, i: usize) -> u32 {
        pack_effect_word(
            self.flash.get(i).copied().unwrap_or(self.default_flash),
            self.dissolve
                .get(i)
                .copied()
                .unwrap_or(self.default_dissolve),
            self.palette_rows
                .get(i)
                .copied()
                .unwrap_or(self.default_palette_row),
        )
    }
}

/// Pack one sprite's flash, dissolve and palette row the way the effect
/// shaders unpack them.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn pack_effect_word(flash: f32, dissolve: f32, palette_row: u16) -> u32 {
    let unorm8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    unorm8(flash) | (unorm8(dissolve) << 8) | ((palette_row as u32) << 16)
}

/// A palette texture for [`SpriteEffects::palette`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpritePalette {
    /// The palette: one palette per row, one colour per column. Colour, so
    /// upload it sRGB.
    pub texture: TextureId,
    /// What in the sprite's texel picks the column.
    pub index: PaletteIndex,
}

/// How a palette-swapped sprite's texel picks its palette column. A value `v`
/// in `0..=1` picks column `round(v * (width - 1))`, so an index map for a
/// 16-colour palette stores `i / 15`. Upload the index map linear so the value
/// reads back as stored, and draw it with `TextureFilter::Nearest` so
/// neighbouring indices are not blended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PaletteIndex {
    /// The red channel.
    #[default]
    Red,
    /// Rec. 709 luminance of the RGB channels.
    Luminance,
}

/// An outline around a sprite's opaque texels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteOutline {
    /// Outline colour; its alpha scales the outline's coverage.
    pub colour: Colour,
    /// Width in device pixels, whatever the sprite's size, the camera's zoom
    /// or `pixel_snap`. The quad grows by this much on each side so the
    /// outline is not cut off at its edge.
    pub width_px: f32,
}

impl SpriteOutline {
    /// An outline of `colour`, `width_px` device pixels wide.
    pub fn new(colour: Colour, width_px: f32) -> Self {
        Self { colour, width_px }
    }
}

/// Handle to a sprite material registered with
/// [`SpriteMaterials::register_sprite_material`](crate::item_types::sprite::SpriteMaterials::register_sprite_material).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpriteMaterialId(pub(crate) u32);

impl SpriteMaterialId {
    /// Registry index.
    pub fn index(self) -> u32 {
        self.0
    }
}

/// A batch's custom sprite material: the registered plugin and the values of
/// the parameters and textures it declares.
#[derive(Clone, Debug, PartialEq)]
pub struct SpriteMaterial {
    /// The registered material.
    pub id: SpriteMaterialId,
    /// Parameter values by name. Parameters not named take the defaults the
    /// plugin declares.
    pub params: Vec<(String, ParamValue)>,
    /// Texture per declared slot, `material_texture_0..N`. A missing or `None`
    /// slot binds 1x1 white.
    pub textures: Vec<Option<TextureId>>,
}

impl SpriteMaterial {
    /// The material `id` with every parameter at its default and no textures.
    pub fn new(id: SpriteMaterialId) -> Self {
        Self {
            id,
            params: Vec::new(),
            textures: Vec::new(),
        }
    }

    /// Set parameter `name`.
    pub fn with_param(mut self, name: impl Into<String>, value: impl Into<ParamValue>) -> Self {
        let name = name.into();
        let value = value.into();
        match self.params.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => self.params.push((name, value)),
        }
        self
    }

    /// Bind `texture` to slot `slot`.
    pub fn with_texture(mut self, slot: usize, texture: TextureId) -> Self {
        if self.textures.len() <= slot {
            self.textures.resize(slot + 1, None);
        }
        self.textures[slot] = Some(texture);
        self
    }
}

#[cfg(test)]
mod effect_tests {
    use super::*;

    #[test]
    fn effect_word_packs_flash_dissolve_and_row() {
        assert_eq!(pack_effect_word(0.0, 0.0, 0), 0);
        assert_eq!(pack_effect_word(1.0, 0.0, 0), 0xff);
        assert_eq!(pack_effect_word(0.0, 1.0, 3), 0xff00 | (3 << 16));
        let fx = SpriteEffects {
            flash: vec![0.5],
            default_dissolve: 1.0,
            ..SpriteEffects::default()
        };
        assert_eq!(fx.instance_word(0), 128 | (255 << 8));
        assert_eq!(fx.instance_word(1), 255 << 8);
    }
}

/// What a [`SpriteMaskItem`] covers.
#[derive(Clone)]
#[non_exhaustive]
pub enum MaskSource {
    /// Every quad of `sprite` where its texel alpha times its tint alpha is
    /// at least `threshold`. The quads are placed as the sprite batch would
    /// draw them (position, size, pivot, rotation, size mode, orientation and
    /// model); its order, blend, effects and draw mode are ignored.
    Alpha {
        /// The sprites whose alpha is the mask.
        sprite: SpriteItem,
        /// Alpha at and above which a texel is inside the mask.
        threshold: f32,
    },
    /// `shape` inscribed in every quad of `sprite`, as an overlay shape is
    /// inscribed in its box, with radii in the sprite's size units. Rect,
    /// rounded rect, circle, ellipse, capsule and ring are evaluated exactly;
    /// any other shape covers the whole quad. The sprite's texture is ignored.
    Shape {
        /// The shape.
        shape: viewport_lib_types::overlay::OverlayShape,
        /// The quads it fills.
        sprite: SpriteItem,
    },
}

/// A painter mask drawn from sprites or shapes: its coverage goes into
/// `slot` of the frame's mask target, and painter content on a layer in
/// `layers` that tests that slot ([`SpriteItem::mask`], `TilemapItem::mask`,
/// `WorldOverlayLayer::mask`) shows only inside or only outside it.
///
/// The mask itself is not drawn. Masks in one slot combine into their union,
/// and the slot's layers run from the lowest to the highest any of them names.
#[derive(Clone)]
#[non_exhaustive]
pub struct SpriteMaskItem {
    /// What the mask covers.
    pub source: MaskSource,
    /// The painter layers content testing the slot is cut on.
    pub layers: std::ops::RangeInclusive<u32>,
    /// The slot, `0` to `PAINTER_MASK_SLOTS - 1`; higher values wrap.
    pub slot: u8,
    /// Visibility; only `hidden` is read.
    pub settings: ItemSettings,
}

impl SpriteMaskItem {
    /// A mask in `slot` over `layers` covering `source`.
    pub fn new(source: MaskSource, slot: u8, layers: std::ops::RangeInclusive<u32>) -> Self {
        Self {
            source,
            layers,
            slot,
            settings: ItemSettings::default(),
        }
    }
}
