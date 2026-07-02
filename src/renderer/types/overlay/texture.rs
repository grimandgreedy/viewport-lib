/// Handle to a texture uploaded via `ViewportGpuResources::upload_overlay_texture`.
///
/// Pass this to `OverlayShapeItem::texture` to use the image as fill. The
/// handle remains valid for the lifetime of the `ViewportGpuResources` it
/// came from; using it after the resources are dropped is a logic error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OverlayTextureId(pub(crate) u64);

/// Nine-patch / 9-slice texture sampling parameters.
///
/// Treats the bound texture as a panel with four corner regions that stay at
/// their authored pixel size, four edge regions that tile or stretch along one
/// axis, and a centre region that follows both axes. The standard way to ship
/// resizable button, dialog, and scrollbar art without the corners stretching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NineSlice {
    /// Inset in *texture pixels* from each edge to the centre region:
    /// `[top, right, bottom, left]`. Defines where the four corner regions end.
    pub insets_px: [f32; 4],
    /// Sampling for the centre region (between all four insets).
    pub centre_mode: TileMode,
    /// Sampling for the four edge regions (between two opposing insets).
    pub edge_mode: TileMode,
}

impl Default for NineSlice {
    fn default() -> Self {
        Self {
            insets_px: [0.0; 4],
            centre_mode: TileMode::Stretch,
            edge_mode: TileMode::Stretch,
        }
    }
}

/// How a texture region remaps its UV coordinate before sampling.
///
/// Shared by [`NineSlice`] and [`TextureTransform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TileMode {
    /// Linear stretch across the region. Default; matches non-9-slice
    /// behaviour for backwards compatibility.
    #[default]
    Stretch,
    /// Repeat the source region at its native pixel size.
    Tile,
    /// Repeat the source region with every other tile flipped along the
    /// matching axis, so the seam between tiles is mirrored. Smoothes the
    /// repeat for symmetric textures.
    Mirror,
}

/// Affine transform applied to the texture sample before lookup.
///
/// Lets a single texture pan, scale, rotate, tile, and flip independently
/// of the shape it fills. Use for scrolling marquees inside a shape, zooming
/// detail, rotating an icon without rotating the shape, repeating tile art,
/// and mirroring asymmetric assets.
///
/// The transform applies to the bounding-box UV in [0, 1] coordinates,
/// centred at (0.5, 0.5) for scale and rotation. When [`NineSlice`] is also
/// set on the same shape, the 9-slice remap wins and the texture transform
/// is ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureTransform {
    /// UV offset added after scale and rotation. `[0.0, 0.0]` keeps the
    /// sample centred. Drives panning / scrolling.
    pub offset: [f32; 2],
    /// UV scale multiplier applied around the centre. `[1.0, 1.0]` is
    /// 1:1. Larger values widen the sampled UV range (more tiles with
    /// [`TileMode::Tile`], or a wider view of the source with `Stretch`
    /// or `Mirror`). Smaller values zoom in.
    pub scale: [f32; 2],
    /// Rotation around the UV centre `(0.5, 0.5)` in radians, CCW in math
    /// coordinates.
    pub rotation: f32,
    /// Sampling mode for UV values outside `[0, 1]`. `Stretch` clamps to
    /// the edge (default), `Tile` wraps, `Mirror` ping-pongs.
    pub tile_mode: TileMode,
    /// Flip the sample horizontally before lookup.
    pub flip_x: bool,
    /// Flip the sample vertically before lookup.
    pub flip_y: bool,
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
            tile_mode: TileMode::Stretch,
            flip_x: false,
            flip_y: false,
        }
    }
}

impl TextureTransform {
    /// Returns `true` when the transform is the identity (no offset, unit
    /// scale, no rotation, no flips, `Stretch` mode).
    pub fn is_identity(&self) -> bool {
        self.offset == [0.0, 0.0]
            && self.scale == [1.0, 1.0]
            && self.rotation == 0.0
            && matches!(self.tile_mode, TileMode::Stretch)
            && !self.flip_x
            && !self.flip_y
    }
}
