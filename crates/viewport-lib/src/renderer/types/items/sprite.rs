use crate::scene::material::ItemSettings;

/// Controls whether sprite sizes are measured in screen-space pixels or world-space units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpriteSizeMode {
    /// Sizes in screen-space pixels. Sprites maintain constant apparent size at all distances.
    #[default]
    ScreenSpace,
    /// Sizes in world-space units. Sprites shrink with distance like regular geometry.
    WorldSpace,
}

/// GPU blend state used when drawing a sprite batch.
///
/// `AlphaBlend` is the default and matches normal transparent sprites.
/// `Additive` accumulates colour into the framebuffer without subtracting
/// background, which is the usual choice for sparks, fire, and other
/// emissive particles. `Premultiplied` is for sources whose RGB has already
/// been multiplied by alpha, typically when sampling a premultiplied texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpriteBlend {
    /// Standard transparency: `src.rgb * src.a + dst.rgb * (1 - src.a)`.
    #[default]
    AlphaBlend,
    /// Additive: `src.rgb + dst.rgb`. Alpha is unused for the colour result.
    Additive,
    /// Premultiplied alpha: `src.rgb + dst.rgb * (1 - src.a)`.
    Premultiplied,
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
    /// Texture ID from [`DeviceResources::upload_texture`].
    /// `None` renders solid-colour quads using `colours` / `default_colour` only.
    pub texture_id: Option<crate::resources::TextureId>,
    /// World-space positions, one per sprite instance.
    pub positions: Vec<[f32; 3]>,
    /// Per-instance RGBA colour tints. Empty = use `default_colour` for all.
    /// Multiplied with the texture sample (or used directly when `texture_id` is `None`).
    pub colours: Vec<[f32; 4]>,
    /// Per-instance sizes. Empty = use `default_size` for all.
    /// Interpretation depends on `size_mode`.
    pub sizes: Vec<f32>,
    /// Per-instance rotation angles in radians, CCW around the camera-forward axis.
    /// Empty = no rotation applied.
    pub rotations: Vec<f32>,
    /// Per-instance UV rects `[u0, v0, u1, v1]` selecting atlas sub-regions.
    /// Empty = full texture `[0.0, 0.0, 1.0, 1.0]` for all.
    pub uv_rects: Vec<[f32; 4]>,
    /// Fallback RGBA colour tint used when `colours` is empty. Default: opaque white.
    pub default_colour: [f32; 4],
    /// Default size when `sizes` is empty. Pixels (ScreenSpace) or world units (WorldSpace).
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
    pub normal_texture_id: Option<crate::resources::TextureId>,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
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
            default_colour: [1.0, 1.0, 1.0, 1.0],
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
            normal_texture_id: Some(crate::resources::TextureId(7)),
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
        assert_eq!(item.normal_texture_id, Some(crate::resources::TextureId(7)));
        assert_eq!(item.orientation, SpriteOrientation::VelocityStretched);
    }

    #[test]
    fn sprite_normal_mode_default_is_spherical() {
        assert_eq!(SpriteNormalMode::default(), SpriteNormalMode::Spherical);
    }
}
