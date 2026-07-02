use crate::scene::material::ItemSettings;

/// How a decal's colour is composited onto the receiver surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecalBlendMode {
    /// Alpha-blend the decal over the receiver colour (standard transparency).
    #[default]
    Replace,
    /// Multiply the receiver colour by the decal colour (grime, dirt, weathering).
    Multiply,
    /// Add the decal colour to the receiver colour (sparks, fire, glows, emissive overlays).
    ///
    /// Stacking multiple additive decals accumulates correctly: each one brightens the
    /// receiver independently. Combined with `emissive > 0`, the contribution is doubly
    /// additive (emissive adds on top of the already-additive blend result).
    Additive,
}

/// Which side of a cylinder surface receives a cylindrical decal.
///
/// Use `Outward` for projecting onto the outside of a pipe or column (convex surface,
/// normals point away from the axis). Use `Inward` for projecting onto the inside of a
/// tube or barrel (concave surface, normals point toward the axis).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CylindricalFacing {
    /// Project onto the outside of the cylinder (normals point away from the axis). Default.
    #[default]
    Outward,
    /// Project onto the inside of the cylinder (normals point toward the axis).
    Inward,
}

/// How a decal texture is projected onto the receiver surface.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum DecalProjection {
    /// Standard projection: the decal texture is mapped using the XY axes of the decal's local
    /// space. Works well on flat or gently curved surfaces aligned with the projection direction.
    #[default]
    Planar,
    /// Tri-planar projection: the texture is sampled three times (from the X, Y, and Z axes of
    /// decal local space) and blended by the surface normal. Avoids UV stretching on corners and
    /// non-planar surfaces. `blend_sharpness` controls how quickly the blend transitions at edges
    /// between faces: higher values produce a sharper transition, lower values a wider blend zone.
    /// Three texture samples per pixel; the bandwidth cost is documented in the field.
    TriPlanar {
        /// Exponent applied to the absolute normal components before normalising blend weights.
        /// Range 1.0-16.0; 4.0 is a good starting value.
        blend_sharpness: f32,
    },
    /// Cylindrical projection: wraps the texture around a cylinder whose axis is the decal's
    /// local Z axis. UV.x is the angle around the axis (0 at local +X, wrapping at 360
    /// degrees back to the start); UV.y is position along the axis (0 at local -Z, 1 at
    /// local +Z). The decal volume is a cylinder of radius 0.5 and height 1.0 in local space.
    ///
    /// Use this for labels on pipes, warning bands on columns, or any decal that wraps
    /// continuously around a curved surface.
    Cylindrical {
        /// Which side of the cylinder surface to project onto.
        facing: CylindricalFacing,
    },
}

/// UV animation applied to a [`LiveDecal`](crate::scene::LiveDecal).
///
/// The renderer computes the effective `uv_offset` and `uv_scale` each frame from
/// the decal's current age and the animation parameters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DecalAnimation {
    /// Continuously scrolls the UV coordinates at the given world-units-per-second velocity.
    /// The texture wraps (repeat address mode).
    UvScroll {
        /// Horizontal scroll velocity in UV units per second.
        vx: f32,
        /// Vertical scroll velocity in UV units per second.
        vy: f32,
    },
    /// Cycles through frames of a sprite sheet packed into the texture.
    /// Columns advance first (left to right), then rows (top to bottom).
    SpriteSheet {
        /// Number of columns in the sprite sheet.
        cols: u32,
        /// Number of rows in the sprite sheet.
        rows: u32,
        /// Playback rate in frames per second.
        fps: f32,
    },
}

/// A screen-space decal projected onto whatever surface lies inside its volume.
///
/// The decal's projection volume is the unit box [-0.5, 0.5]^3 in local space.
/// `transform` maps local space to world space: scale the local box to set the
/// decal's world-space size, rotate to orient the projection direction, and
/// translate to position it.
///
/// The decal texture is sampled using local XY as UVs. The Z axis of the local
/// box is the projection direction (front face = +Z in local space).
///
/// `texture_id` must be a value returned by
/// [`ViewportGpuResources::upload_texture`].
///
/// # Normal map (D2)
///
/// Set `normal_texture_id` to a tangent-space normal map (same UV space as
/// `texture_id`). The renderer approximates the receiver surface normal from
/// depth derivatives, applies the decal normal map in that tangent frame, and
/// uses the blended normal to modulate the shading. Set `normal_blend_strength`
/// between 0.0 (no effect) and 1.0 (full decal normal).
///
/// # Sorting (D3)
///
/// When multiple decals overlap, `sort_key` controls the composite order.
/// Lower keys render first (underneath); higher keys render on top.
/// Decals with equal keys render in insertion order.
///
/// # Roughness / metallic (D3)
///
/// `roughness` and `metallic` add a view-angle specular approximation on top
/// of the decal colour. Because decals run post-opaque without access to scene
/// light data, the highlight uses the view direction as a retroreflection proxy:
/// low roughness produces a tight glossy highlight at near-normal incidence;
/// high metallic tints the highlight by the albedo colour. This approximation
/// is sufficient for wet-surface and scuff effects; it is not physically
/// accurate PBR.
///
/// # UV animation (D4)
///
/// `uv_offset` and `uv_scale` shift and scale the final UV before texture
/// sampling. For sprite-sheet and scroll animations, use
/// [`LiveDecal`](crate::scene::LiveDecal) with a
/// [`DecalAnimation`]: the scene updates `uv_offset` / `uv_scale` from the
/// animation each frame.
///
/// # Emissive (D6)
///
/// `emissive` adds a self-illumination contribution on top of whatever the
/// blend mode produces. The emissive colour is sampled from `emissive_texture_id`
/// when set, or taken from the albedo colour otherwise. The contribution is
/// always additive regardless of blend mode, so `emissive = 0.0` (default) has
/// no visible effect. Values above 1.0 are meaningful in HDR: they bloom under
/// tone-mapping when `post_process.enabled = true`.
///
/// # Soft edges (D7)
///
/// `edge_fade` fades the decal alpha to zero near the boundary of its projection
/// box, hiding the rectangular cutoff. Range [0.0, 0.5]; 0.0 (default) = hard
/// edge, 0.5 = fade covers the full half-extent. A value around 0.15-0.25 suits
/// most decals.
#[derive(Clone)]
#[non_exhaustive]
pub struct DecalItem {
    /// Model matrix: local [-0.5, 0.5]^3 -> world space.
    pub transform: [[f32; 4]; 4],
    /// Texture ID from `resources.upload_texture()`.
    pub texture_id: u64,
    /// How the decal colour blends with the receiver. Default: `Replace`.
    pub blend_mode: DecalBlendMode,
    /// Overall opacity multiplier applied on top of the texture alpha. Default: 1.0.
    pub alpha: f32,
    /// Optional tangent-space normal map texture ID (D2). Default: `None`.
    pub normal_texture_id: Option<u64>,
    /// How strongly the decal normal map overrides the receiver normal. Range [0, 1]. Default: 1.0.
    pub normal_blend_strength: f32,
    // -- D3 fields --
    /// Draw order key. Lower = rendered first (underneath). Default: 0.
    pub sort_key: i32,
    /// Surface roughness in [0, 1]. 0 = mirror-smooth, 1 = fully matte. Default: 1.0.
    pub roughness: f32,
    /// Optional per-texel roughness map (single-channel, R component used). Default: `None`.
    pub roughness_texture_id: Option<u64>,
    /// Metallic factor in [0, 1]. 0 = dielectric, 1 = metal. Default: 0.0.
    pub metallic: f32,
    /// Optional per-texel metallic map (single-channel, R component used). Default: `None`.
    pub metallic_texture_id: Option<u64>,
    // -- D4 fields --
    /// UV offset applied before texture sampling. Modified by [`DecalAnimation`]. Default: [0, 0].
    pub uv_offset: [f32; 2],
    /// UV scale applied before texture sampling. Modified by [`DecalAnimation`]. Default: [1, 1].
    pub uv_scale: [f32; 2],
    // -- D6 fields --
    /// Emissive intensity multiplier. 0.0 = no emission. Default: 0.0.
    pub emissive: f32,
    /// Optional emissive texture. When `None`, the albedo colour is used as the emissive colour.
    pub emissive_texture_id: Option<u64>,
    // -- D7 fields --
    /// Fraction of each local half-extent over which the alpha fades to zero at the box boundary.
    /// Range [0.0, 0.5]. Default: 0.0 (hard edge).
    pub edge_fade: f32,
    // -- D8 fields --
    /// How the decal texture is projected onto the receiver surface. Default: `Planar`.
    ///
    /// `TriPlanar` samples the texture from three axes and blends by the surface normal,
    /// avoiding UV stretching on corners. It costs three texture samples per pixel instead of one;
    /// use it only when the decal spans a non-planar or multi-face surface.
    pub projection: DecalProjection,
    /// Visibility and opacity overrides. `hidden` skips the decal entirely; `opacity`
    /// multiplies the final alpha. `unlit` and `wireframe` are accepted but have no
    /// effect on decals. `pick_id` and `selected` are also no-op for decals. Default: all no-op.
    pub settings: ItemSettings,
}

impl Default for DecalItem {
    fn default() -> Self {
        Self {
            transform: glam::Mat4::IDENTITY.to_cols_array_2d(),
            texture_id: 0,
            blend_mode: DecalBlendMode::Replace,
            alpha: 1.0,
            normal_texture_id: None,
            normal_blend_strength: 1.0,
            sort_key: 0,
            roughness: 1.0,
            roughness_texture_id: None,
            metallic: 0.0,
            metallic_texture_id: None,
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
            emissive: 0.0,
            emissive_texture_id: None,
            edge_fade: 0.0,
            projection: DecalProjection::Planar,
            settings: ItemSettings::default(),
        }
    }
}
