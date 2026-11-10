use viewport_lib_types::colour::Colour;
use viewport_lib_types::ids::TextureId;
use viewport_lib_types::material::{ItemSettings, TextureFilter};

viewport_lib_types::slot_handle! {
    /// Handle to a tilemap uploaded once through
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload). A
    /// [`TilemapItem`] names it each frame.
    pub struct TilemapId;
}

/// Most tile animations one tileset drives. Tiles whose animation is past
/// this many draw their first frame without animating.
pub const MAX_TILE_ANIMATIONS: usize = 256;

/// Most tiles along either side of a map: a tile's column and row each go to
/// the GPU as 16 bits.
pub const MAX_TILEMAP_SIDE: u32 = 1 << 16;

/// The world plane a tilemap lies in. The same two planes as
/// `viewport_lib::Plane2d` and `viewport_lib::WorldPlane`, with the same axes
/// and elevation direction, so a map drawn on a plane lines up with a
/// `Camera2d` looking at it and with a world overlay layer on it. Converts to
/// `WorldPlane` with `From`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TilemapPlane {
    /// The world XY plane seen from above, looking along -Z. Plane x is world
    /// +X and plane y is world +Y; elevation is world +Z.
    #[default]
    Xy,
    /// The world XZ plane seen from the front, looking along +Y. Plane x is
    /// world +X and plane y is world +Z; elevation is world -Y, towards the
    /// camera.
    Xz,
}

impl From<TilemapPlane> for viewport_lib_types::overlay::WorldPlane {
    fn from(plane: TilemapPlane) -> Self {
        match plane {
            TilemapPlane::Xy => Self::Xy,
            TilemapPlane::Xz => Self::Xz,
        }
    }
}

impl TilemapPlane {
    /// World direction of plane +x.
    pub fn x_axis(self) -> glam::Vec3 {
        glam::Vec3::X
    }

    /// World direction of plane +y.
    pub fn y_axis(self) -> glam::Vec3 {
        match self {
            Self::Xy => glam::Vec3::Y,
            Self::Xz => glam::Vec3::Z,
        }
    }

    /// World direction of increasing elevation: out of the plane, towards a
    /// camera looking at it.
    pub fn normal(self) -> glam::Vec3 {
        match self {
            Self::Xy => glam::Vec3::Z,
            Self::Xz => glam::Vec3::NEG_Y,
        }
    }

    /// The world point at plane position `p` and `elevation`.
    pub fn to_world(self, p: glam::DVec2, elevation: f64) -> glam::DVec3 {
        self.x_axis().as_dvec3() * p.x
            + self.y_axis().as_dvec3() * p.y
            + self.normal().as_dvec3() * elevation
    }

    /// The plane position of a world point, dropping its elevation.
    pub fn to_plane(self, p: glam::DVec3) -> glam::DVec2 {
        glam::DVec2::new(
            p.dot(self.x_axis().as_dvec3()),
            p.dot(self.y_axis().as_dvec3()),
        )
    }

    /// The elevation of a world point above the plane.
    pub fn elevation_of(self, p: glam::DVec3) -> f64 {
        p.dot(self.normal().as_dvec3())
    }
}

/// How a tile's image is mirrored and turned, with Tiled's three flags.
///
/// The diagonal flip mirrors the image across its top-left to bottom-right
/// diagonal and applies first, then the horizontal and vertical flips. The
/// eight combinations are the eight orientations of a square, so a quarter
/// turn is the diagonal flip plus one of the others; the constants name them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct TileFlip {
    /// Mirror left to right.
    pub horizontal: bool,
    /// Mirror top to bottom.
    pub vertical: bool,
    /// Mirror across the top-left to bottom-right diagonal, applied first.
    pub diagonal: bool,
}

impl TileFlip {
    /// As drawn in the tileset.
    pub const NONE: Self = Self::new(false, false, false);
    /// Mirrored left to right.
    pub const HORIZONTAL: Self = Self::new(true, false, false);
    /// Mirrored top to bottom.
    pub const VERTICAL: Self = Self::new(false, true, false);
    /// Mirrored across the top-left to bottom-right diagonal.
    pub const DIAGONAL: Self = Self::new(false, false, true);
    /// A quarter turn clockwise.
    pub const ROTATE_90: Self = Self::new(true, false, true);
    /// A half turn.
    pub const ROTATE_180: Self = Self::new(true, true, false);
    /// A quarter turn anticlockwise.
    pub const ROTATE_270: Self = Self::new(false, true, true);
    /// Mirrored across the top-right to bottom-left diagonal.
    pub const ANTI_DIAGONAL: Self = Self::new(true, true, true);

    /// The flags as Tiled names them.
    pub const fn new(horizontal: bool, vertical: bool, diagonal: bool) -> Self {
        Self {
            horizontal,
            vertical,
            diagonal,
        }
    }

    /// Split a Tiled global tile id into the id with its flag bits cleared
    /// and the flip they encode. Bit 31 is the horizontal flip, bit 30 the
    /// vertical and bit 29 the diagonal; bit 28, the hexagonal turn, is
    /// cleared and ignored.
    pub const fn from_tiled_gid(gid: u32) -> (u32, Self) {
        let flip = Self::new(
            gid & (1 << 31) != 0,
            gid & (1 << 30) != 0,
            gid & (1 << 29) != 0,
        );
        (gid & 0x0fff_ffff, flip)
    }

    /// The point of the tileset image a point of the drawn tile shows.
    ///
    /// Both points are in image coordinates of one tile: `x` right and `y`
    /// down, `0..=1` across the tile. Each flag is its own inverse, so the
    /// flips are undone in reverse order: vertical, horizontal, then diagonal.
    pub fn source_point(self, p: [f32; 2]) -> [f32; 2] {
        let [mut x, mut y] = p;
        if self.vertical {
            y = 1.0 - y;
        }
        if self.horizontal {
            x = 1.0 - x;
        }
        if self.diagonal {
            std::mem::swap(&mut x, &mut y);
        }
        [x, y]
    }

    /// The three flags as bits: horizontal 1, vertical 2, diagonal 4.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn bits(self) -> u32 {
        self.horizontal as u32 | (self.vertical as u32) << 1 | (self.diagonal as u32) << 2
    }
}

/// One cell of a tilemap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tile {
    /// Index into the tileset, counting along each atlas row from the top
    /// left. [`Tile::EMPTY`] draws nothing.
    pub index: u32,
    /// How the tile's image is mirrored and turned.
    pub flip: TileFlip,
    /// Multiplied into the tile's colour, after the item's own tint.
    pub tint: Colour,
}

impl Tile {
    /// The index of a cell with no tile.
    pub const EMPTY_INDEX: u32 = u32::MAX;
    /// A cell with no tile.
    pub const EMPTY: Self = Self {
        index: Self::EMPTY_INDEX,
        flip: TileFlip::NONE,
        tint: Colour::WHITE,
    };

    /// Tile `index` of the tileset, unflipped and untinted.
    pub const fn new(index: u32) -> Self {
        Self {
            index,
            flip: TileFlip::NONE,
            tint: Colour::WHITE,
        }
    }

    /// The same tile with `flip`.
    pub const fn with_flip(mut self, flip: TileFlip) -> Self {
        self.flip = flip;
        self
    }

    /// The same tile with `tint`.
    pub const fn with_tint(mut self, tint: Colour) -> Self {
        self.tint = tint;
        self
    }

    /// Whether the cell holds no tile.
    pub const fn is_empty(&self) -> bool {
        self.index == Self::EMPTY_INDEX
    }
}

impl Default for Tile {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// One frame of a tile animation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileFrame {
    /// The tileset tile this frame shows.
    pub tile: u32,
    /// How long the frame shows, in seconds.
    pub duration: f32,
}

/// A tile that cycles through other tiles of its tileset: every cell holding
/// `tile` shows the frame the item's clock selects.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TileAnimation {
    /// The tile index the map holds for the animated tile.
    pub tile: u32,
    /// The frames, shown in order and repeated.
    pub frames: Vec<TileFrame>,
}

impl TileAnimation {
    /// The tile shown at `time` seconds: frames run in order from time zero
    /// and repeat. Frames with no positive duration are skipped; with none
    /// left, the animation shows `tile`.
    pub fn frame_at(&self, time: f32) -> u32 {
        let total: f32 = self
            .frames
            .iter()
            .map(|f| f.duration)
            .filter(|d| *d > 0.0)
            .sum();
        if !(total > 0.0) || !time.is_finite() {
            return self
                .frames
                .iter()
                .find(|f| f.duration > 0.0)
                .map_or(self.tile, |f| f.tile);
        }
        let mut t = time.rem_euclid(total);
        let mut last = self.tile;
        for frame in self.frames.iter().filter(|f| f.duration > 0.0) {
            if t < frame.duration {
                return frame.tile;
            }
            t -= frame.duration;
            last = frame.tile;
        }
        // Rounding left `t` a hair past the last frame's end.
        last
    }
}

/// The atlas a tilemap's tiles come from: one texture holding equal-sized
/// tiles in rows, as Tiled describes a tileset image.
#[derive(Debug, Clone, PartialEq)]
pub struct Tileset {
    /// The atlas texture, from `DeviceResources::upload_texture`, uploaded
    /// sRGB. Row 0 of the image is its top. `None` draws each tile as a solid
    /// quad in its tint.
    pub texture_id: Option<TextureId>,
    /// Width and height of one tile, in texels.
    pub tile_size: [u32; 2],
    /// Tiles per atlas row.
    pub columns: u32,
    /// Texels between neighbouring tiles.
    pub spacing: u32,
    /// Texels between the image edge and the first tile.
    pub margin: u32,
    /// How the atlas is filtered. [`TextureFilter::Nearest`] keeps pixel art
    /// crisp. The atlas is always read at its top mip level, so tiles never
    /// blend with their neighbours under minification.
    pub filter: TextureFilter,
    /// Animated tiles. At most [`MAX_TILE_ANIMATIONS`] animate; the first
    /// entry naming a tile wins.
    pub animations: Vec<TileAnimation>,
}

impl Tileset {
    /// A tileset of `columns` tiles per row, each `tile_size` texels, with no
    /// spacing or margin, nearest-filtered and unanimated.
    pub fn new(texture_id: Option<TextureId>, tile_size: [u32; 2], columns: u32) -> Self {
        Self {
            texture_id,
            tile_size,
            columns: columns.max(1),
            spacing: 0,
            margin: 0,
            filter: TextureFilter::Nearest,
            animations: Vec::new(),
        }
    }

    /// The texel rectangle of tile `index` as `[x, y, width, height]`, `y`
    /// measured down from the top of the image.
    pub fn tile_rect(&self, index: u32) -> [u32; 4] {
        let columns = self.columns.max(1);
        let (col, row) = (index % columns, index / columns);
        let [w, h] = self.tile_size;
        [
            self.margin + col * (w + self.spacing),
            self.margin + row * (h + self.spacing),
            w,
            h,
        ]
    }

    /// The animation slot of `tile`, when it animates.
    pub fn animation_of(&self, tile: u32) -> Option<usize> {
        self.animations
            .iter()
            .take(MAX_TILE_ANIMATIONS)
            .position(|a| a.tile == tile)
    }
}

/// A fixed-size block of a tilemap's cells, drawn with one draw call.
#[derive(Debug, Clone, PartialEq)]
pub struct TileChunk {
    /// The chunk's cells, row by row from the bottom row, `chunk_size` to a
    /// row. Cells past the map's edge stay empty.
    pub tiles: Vec<Tile>,
}

impl TileChunk {
    /// How many cells hold a tile.
    pub fn occupied(&self) -> usize {
        self.tiles.iter().filter(|t| !t.is_empty()).count()
    }
}

/// A grid of tiles from one tileset, held in square chunks.
///
/// Column 0 is the left edge and row 0 the bottom: row numbers count up the
/// plane's +y. Upload it once, draw it with a [`TilemapItem`] each frame, and
/// edit it in place with `set_tile`, which re-uploads only the chunk holding
/// the cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Tilemap {
    /// The atlas and animations every tile reads.
    pub tileset: Tileset,
    size: [u32; 2],
    chunk_size: u32,
    chunks: Vec<TileChunk>,
}

impl Tilemap {
    /// Chunk side used by [`Tilemap::new`].
    pub const DEFAULT_CHUNK_SIZE: u32 = 32;

    /// An empty map `size` tiles across (columns, rows), in chunks of
    /// [`DEFAULT_CHUNK_SIZE`](Self::DEFAULT_CHUNK_SIZE) tiles a side.
    pub fn new(tileset: Tileset, size: [u32; 2]) -> Self {
        Self::with_chunk_size(tileset, size, Self::DEFAULT_CHUNK_SIZE)
    }

    /// An empty map in chunks of `chunk_size` tiles a side. Each side of the
    /// map is clamped to [`MAX_TILEMAP_SIDE`] and the chunk side to `1..=256`.
    pub fn with_chunk_size(tileset: Tileset, size: [u32; 2], chunk_size: u32) -> Self {
        let size = size.map(|s| s.min(MAX_TILEMAP_SIDE));
        let chunk_size = chunk_size.clamp(1, 256);
        let chunks_x = size[0].div_ceil(chunk_size);
        let chunks_y = size[1].div_ceil(chunk_size);
        let chunk = TileChunk {
            tiles: vec![Tile::EMPTY; (chunk_size * chunk_size) as usize],
        };
        Self {
            tileset,
            size,
            chunk_size,
            chunks: vec![chunk; (chunks_x * chunks_y) as usize],
        }
    }

    /// A map from rows of tile indices, top row first as a level is usually
    /// written down, with [`Tile::EMPTY_INDEX`] for an empty cell. Each row
    /// is as long as the first; shorter rows leave the rest empty.
    pub fn from_rows(tileset: Tileset, rows: &[&[u32]], chunk_size: u32) -> Self {
        let width = rows.first().map_or(0, |r| r.len()) as u32;
        let height = rows.len() as u32;
        let mut map = Self::with_chunk_size(tileset, [width, height], chunk_size);
        for (i, row) in rows.iter().enumerate() {
            let r = height - 1 - i as u32;
            for (c, &index) in row.iter().enumerate().take(width as usize) {
                map.set(c as u32, r, Tile::new(index));
            }
        }
        map
    }

    /// Columns and rows of tiles.
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// Tiles along each side of a chunk.
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// Chunks across and up the map.
    pub fn chunk_grid(&self) -> [u32; 2] {
        [
            self.size[0].div_ceil(self.chunk_size),
            self.size[1].div_ceil(self.chunk_size),
        ]
    }

    /// Every chunk, row by row from the bottom-left one.
    pub fn chunks(&self) -> &[TileChunk] {
        &self.chunks
    }

    /// The chunk and the cell within it that hold cell (`col`, `row`), as
    /// flat indices: chunk `cy * chunks_across + cx`, cell `ty * chunk_size +
    /// tx`. `None` off the map.
    pub fn locate(&self, col: u32, row: u32) -> Option<(u32, u32)> {
        if col >= self.size[0] || row >= self.size[1] {
            return None;
        }
        let cs = self.chunk_size;
        let across = self.chunk_grid()[0];
        let chunk = (row / cs) * across + col / cs;
        let tile = (row % cs) * cs + col % cs;
        Some((chunk, tile))
    }

    /// The map cell (`col`, `row`) of `tile` in `chunk`, the inverse of
    /// [`locate`](Self::locate). `None` when the pair names no cell on the map.
    pub fn cell_of(&self, chunk: u32, tile: u32) -> Option<[u32; 2]> {
        let cs = self.chunk_size;
        let across = self.chunk_grid()[0].max(1);
        if chunk as usize >= self.chunks.len() || tile >= cs * cs {
            return None;
        }
        let col = (chunk % across) * cs + tile % cs;
        let row = (chunk / across) * cs + tile / cs;
        (col < self.size[0] && row < self.size[1]).then_some([col, row])
    }

    /// The tile at (`col`, `row`); `None` off the map.
    pub fn get(&self, col: u32, row: u32) -> Option<Tile> {
        let (chunk, tile) = self.locate(col, row)?;
        Some(self.chunks[chunk as usize].tiles[tile as usize])
    }

    /// Put `tile` at (`col`, `row`). Returns the chunk index it landed in, or
    /// `None` off the map.
    pub fn set(&mut self, col: u32, row: u32, tile: Tile) -> Option<u32> {
        let (chunk, index) = self.locate(col, row)?;
        self.chunks[chunk as usize].tiles[index as usize] = tile;
        Some(chunk)
    }

    /// The cell range chunk `chunk` covers, as `[col0, row0, col1, row1]`
    /// with the end exclusive and clamped to the map.
    pub fn chunk_cells(&self, chunk: u32) -> [u32; 4] {
        let cs = self.chunk_size;
        let across = self.chunk_grid()[0].max(1);
        let (cx, cy) = (chunk % across, chunk / across);
        [
            cx * cs,
            cy * cs,
            ((cx + 1) * cs).min(self.size[0]),
            ((cy + 1) * cs).min(self.size[1]),
        ]
    }
}

/// How a tilemap layer draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TilemapDraw {
    /// Opaque with alpha-tested holes: texels at or above `alpha_threshold`
    /// draw solid and write depth, the rest are discarded. The default, for
    /// ground and wall layers.
    Cutout {
        /// Alpha below which a texel is discarded. Default `0.5`.
        alpha_threshold: f32,
    },
    /// Alpha-blended, drawn after the opaque scene in painter's order: by
    /// `layer` (lower first, then submission order), each layer's chunks in
    /// map order. Never writes depth. For decoration layers with soft edges.
    Blended {
        /// Layer order. Lower layers draw first.
        layer: u32,
        /// When `true` the tiles test against scene depth, so nearer
        /// geometry covers them; when `false` they draw over everything
        /// already in the scene.
        depth_test: bool,
    },
}

impl Default for TilemapDraw {
    fn default() -> Self {
        Self::Cutout {
            alpha_threshold: 0.5,
        }
    }
}

/// One drawn layer of an uploaded [`Tilemap`], submitted each frame with
/// `frame.scene.items_mut::<TilemapItem>()`.
///
/// The map's data stays on the GPU; the item places it. Cell (`col`, `row`)
/// covers plane `origin + [col, row] * cell_size` to one cell further, on
/// `plane` at `elevation`. Several items may draw one map.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TilemapItem {
    /// The uploaded map.
    pub source: TilemapId,
    /// The world plane the map lies in. Default [`TilemapPlane::Xy`].
    pub plane: TilemapPlane,
    /// Plane position of the bottom-left corner of cell (0, 0).
    pub origin: [f32; 2],
    /// Position along the plane's normal, towards the camera. Under an
    /// orthographic camera looking at the plane this is the layer order: a
    /// higher layer covers a lower one. Default `0.0`.
    pub elevation: f32,
    /// World size of one cell along plane x and y. Default `[1.0, 1.0]`.
    pub cell_size: [f32; 2],
    /// How far the layer follows the camera across the plane, per axis. The
    /// layer is drawn shifted by `camera * (1 - parallax)`, where `camera` is
    /// the camera's position on the plane: `1.0` is the world layer, `0.5`
    /// scrolls at half the camera's speed, `0.0` stays fixed to the camera.
    /// Default `[1.0, 1.0]`.
    ///
    /// Zooming about a cursor moves the camera, so a layer with a factor
    /// below `1.0` shifts against the world layer under zoom as well as under
    /// a pan. A side view usually keeps the vertical factor at `1.0` so a far
    /// layer stays level with the ground.
    pub parallax: [f32; 2],
    /// Multiplied into every tile's colour. Default opaque white.
    pub tint: Colour,
    /// The animation clock, in seconds: each animated tile shows the frame
    /// [`TileAnimation::frame_at`] picks for it. Pass the frame time to
    /// animate, or a fixed value to hold a frame.
    pub time: f32,
    /// Cutout or blended. Default [`TilemapDraw::Cutout`].
    pub draw: TilemapDraw,
    /// When `true`, the layer is moved by less than a device pixel so its
    /// origin lands on the pixel grid, which keeps every texel of pixel art
    /// drawn at an integer scale on whole pixels. Default `false`.
    pub pixel_snap: bool,
    /// Visibility, pick identity and selection.
    pub settings: ItemSettings,
}

impl TilemapItem {
    /// A world layer drawing `source` on the XY plane at the origin, one
    /// world unit per cell.
    pub fn new(source: TilemapId) -> Self {
        Self {
            source,
            plane: TilemapPlane::Xy,
            origin: [0.0, 0.0],
            elevation: 0.0,
            cell_size: [1.0, 1.0],
            parallax: [1.0, 1.0],
            tint: Colour::WHITE,
            time: 0.0,
            draw: TilemapDraw::default(),
            pixel_snap: false,
            settings: ItemSettings::default(),
        }
    }

    /// Where the layer's cell (0, 0) corner is drawn on the plane for a
    /// camera at plane position `camera`: the origin plus the parallax shift.
    pub fn drawn_origin(&self, camera: glam::DVec2) -> glam::DVec2 {
        let origin = glam::DVec2::new(self.origin[0] as f64, self.origin[1] as f64);
        let follow = glam::DVec2::new(1.0 - self.parallax[0] as f64, 1.0 - self.parallax[1] as f64);
        origin + camera * follow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The drawn corners of a tile in image coordinates, in the order
    /// top-left, top-right, bottom-right, bottom-left, mapped to the source
    /// corners they show.
    fn corners(flip: TileFlip) -> [[f32; 2]; 4] {
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]].map(|p| flip.source_point(p))
    }

    const TL: [f32; 2] = [0.0, 0.0];
    const TR: [f32; 2] = [1.0, 0.0];
    const BR: [f32; 2] = [1.0, 1.0];
    const BL: [f32; 2] = [0.0, 1.0];

    #[test]
    fn the_eight_tiled_orientations_map_corners_as_tiled_draws_them() {
        // For each orientation: which source corner shows at the drawn
        // top-left, top-right, bottom-right and bottom-left.
        let cases = [
            (TileFlip::NONE, [TL, TR, BR, BL]),
            (TileFlip::HORIZONTAL, [TR, TL, BL, BR]),
            (TileFlip::VERTICAL, [BL, BR, TR, TL]),
            (TileFlip::ROTATE_180, [BR, BL, TL, TR]),
            // Transposed: the source's top-right shows at the bottom-left.
            (TileFlip::DIAGONAL, [TL, BL, BR, TR]),
            // A clockwise quarter turn brings the source's bottom-left up to
            // the top-left.
            (TileFlip::ROTATE_90, [BL, TL, TR, BR]),
            (TileFlip::ROTATE_270, [TR, BR, BL, TL]),
            (TileFlip::ANTI_DIAGONAL, [BR, TR, TL, BL]),
        ];
        for (flip, expected) in cases {
            assert_eq!(corners(flip), expected, "{flip:?}");
        }
    }

    #[test]
    fn the_eight_orientations_are_distinct() {
        let mut seen = Vec::new();
        for bits in 0..8u32 {
            let flip = TileFlip::new(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
            assert_eq!(flip.bits(), bits);
            let c = corners(flip);
            assert!(!seen.contains(&c), "{flip:?} repeats another orientation");
            seen.push(c);
        }
    }

    #[test]
    fn quarter_turns_compose_like_rotations() {
        // Turning a quarter twice is a half turn: apply the inverse map of
        // ROTATE_90 to itself and compare with ROTATE_180.
        let p = [0.25, 0.1];
        let twice = TileFlip::ROTATE_90.source_point(TileFlip::ROTATE_90.source_point(p));
        let half = TileFlip::ROTATE_180.source_point(p);
        assert!((twice[0] - half[0]).abs() < 1e-6 && (twice[1] - half[1]).abs() < 1e-6);
    }

    #[test]
    fn tiled_gid_flags_decode() {
        let (id, flip) = TileFlip::from_tiled_gid(0xA000_0007);
        assert_eq!(id, 7);
        assert_eq!(flip, TileFlip::ROTATE_90);
        let (id, flip) = TileFlip::from_tiled_gid(0x1000_0003);
        assert_eq!(id, 3, "the hexagonal bit is cleared");
        assert_eq!(flip, TileFlip::NONE);
    }

    #[test]
    fn animation_frames_follow_their_durations_and_repeat() {
        let anim = TileAnimation {
            tile: 10,
            frames: vec![
                TileFrame {
                    tile: 10,
                    duration: 0.5,
                },
                TileFrame {
                    tile: 11,
                    duration: 0.25,
                },
                TileFrame {
                    tile: 12,
                    duration: 0.25,
                },
            ],
        };
        assert_eq!(anim.frame_at(0.0), 10);
        assert_eq!(anim.frame_at(0.49), 10);
        assert_eq!(anim.frame_at(0.5), 11);
        assert_eq!(anim.frame_at(0.8), 12);
        // One second per cycle.
        assert_eq!(anim.frame_at(1.6), 11);
        assert_eq!(anim.frame_at(-0.1), 12, "negative times wrap");
    }

    #[test]
    fn an_animation_without_time_shows_its_tile() {
        let still = TileAnimation {
            tile: 4,
            frames: vec![TileFrame {
                tile: 9,
                duration: 0.0,
            }],
        };
        assert_eq!(still.frame_at(3.0), 4);
        assert_eq!(TileAnimation::default().frame_at(1.0), 0);
    }

    #[test]
    fn cells_round_trip_through_chunks() {
        let map = Tilemap::with_chunk_size(Tileset::new(None, [8, 8], 4), [10, 7], 4);
        assert_eq!(map.chunk_grid(), [3, 2]);
        assert_eq!(map.chunks().len(), 6);
        for row in 0..7 {
            for col in 0..10 {
                let (chunk, tile) = map.locate(col, row).unwrap();
                assert_eq!(map.cell_of(chunk, tile), Some([col, row]));
            }
        }
        assert_eq!(map.locate(10, 0), None);
        // Chunk 2 is the bottom-right one, clipped to the map.
        assert_eq!(map.chunk_cells(2), [8, 0, 10, 4]);
        // Cell 3 of chunk 2 is column 11, off the map.
        assert_eq!(map.cell_of(2, 3), None);
    }

    #[test]
    fn rows_are_written_top_first() {
        let map = Tilemap::from_rows(Tileset::new(None, [8, 8], 4), &[&[1, 2], &[3, 4]], 8);
        assert_eq!(map.get(0, 1).unwrap().index, 1);
        assert_eq!(map.get(1, 0).unwrap().index, 4);
    }

    #[test]
    fn tile_rects_step_over_margin_and_spacing() {
        let mut set = Tileset::new(None, [16, 8], 3);
        set.margin = 1;
        set.spacing = 2;
        assert_eq!(set.tile_rect(0), [1, 1, 16, 8]);
        assert_eq!(set.tile_rect(4), [1 + 18, 1 + 10, 16, 8]);
    }

    #[test]
    fn parallax_shifts_the_drawn_origin_by_the_unfollowed_share() {
        let mut item = TilemapItem::new(TilemapId::from_index(0));
        item.origin = [2.0, 0.0];
        item.parallax = [0.5, 0.0];
        let o = item.drawn_origin(glam::DVec2::new(10.0, 4.0));
        assert_eq!(o, glam::DVec2::new(7.0, 4.0));
    }
}
