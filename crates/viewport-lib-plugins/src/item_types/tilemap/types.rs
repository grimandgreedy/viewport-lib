use viewport_lib_types::colour::Colour;
use viewport_lib_types::ids::TextureId;
use viewport_lib_types::material::{ItemSettings, TextureFilter};
use viewport_lib_types::render_item::MaskTest;

use super::grid::TileOrientation;

viewport_lib_types::slot_handle! {
    /// Handle to a tilemap uploaded once through
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload). A
    /// [`TilemapItem`] names it each frame.
    pub struct TilemapId;
}

/// Most tile animations one tileset drives. Tiles whose animation is past
/// this many draw their first frame without animating.
pub const MAX_TILE_ANIMATIONS: usize = 256;

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
///
/// On a [`TileOrientation::Hexagonal`] grid the diagonal flag turns the image
/// 60 degrees clockwise about its centre instead, after the horizontal and
/// vertical flips, as Tiled draws it.
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
    /// cleared here and read by [`Tile::from_tiled_gid`].
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
    /// On a [`TileOrientation::Hexagonal`] grid, turn the image 120 degrees
    /// clockwise about its centre, after its flips; with the diagonal flag
    /// as well the turn is 180 degrees. Ignored on other grids.
    pub rotate_120: bool,
}

impl Tile {
    /// The index of a cell with no tile.
    pub const EMPTY_INDEX: u32 = u32::MAX;
    /// A cell with no tile.
    pub const EMPTY: Self = Self {
        index: Self::EMPTY_INDEX,
        flip: TileFlip::NONE,
        tint: Colour::WHITE,
        rotate_120: false,
    };

    /// Tile `index` of the tileset, unflipped and untinted.
    pub const fn new(index: u32) -> Self {
        Self {
            index,
            flip: TileFlip::NONE,
            tint: Colour::WHITE,
            rotate_120: false,
        }
    }

    /// The tile a Tiled global tile id names, with its flips and its
    /// hexagonal turn (bit 28) decoded and every flag bit cleared from the
    /// index. The index is still the global id: take the tileset's first id
    /// off it.
    pub const fn from_tiled_gid(gid: u32) -> Self {
        let (index, flip) = TileFlip::from_tiled_gid(gid);
        Self {
            index,
            flip,
            tint: Colour::WHITE,
            rotate_120: gid & (1 << 28) != 0,
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

    /// The same tile turned 120 degrees on a hexagonal grid.
    pub const fn with_rotate_120(mut self, rotate_120: bool) -> Self {
        self.rotate_120 = rotate_120;
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

/// A tile drawn at a size other than the tileset's: a tree two cells tall, a
/// wide roof. The quad is `size` texels of the atlas, scaled to world units
/// as one tileset tile is to one cell, with its bottom-left corner at
/// `anchor` of the cell, in cells from the cell's bottom-left corner.
///
/// The atlas rectangle starts at the tile's slot (its top-left texel stays
/// where [`Tileset::tile_rect`] would put it) and runs `size` texels right
/// and down, so a tall tile laid out over two atlas rows is named by the
/// index of its top slot. Flips turn the image within the quad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileSize {
    /// The tileset index this size applies to.
    pub tile: u32,
    /// Width and height of the tile's image, in texels.
    pub size: [u32; 2],
    /// Where the quad's bottom-left corner sits, in cells from the cell's
    /// bottom-left corner. Default `[0.0, 0.0]`, so the quad stands on the
    /// cell and grows up and to the right of it, as Tiled draws an oversized
    /// tile.
    pub anchor: [f32; 2],
}

impl TileSize {
    /// Tile `tile` drawn `size` texels across, standing on its cell's
    /// bottom-left corner.
    pub const fn new(tile: u32, size: [u32; 2]) -> Self {
        Self {
            tile,
            size,
            anchor: [0.0, 0.0],
        }
    }

    /// The same size anchored at `anchor` of the cell.
    pub const fn with_anchor(mut self, anchor: [f32; 2]) -> Self {
        self.anchor = anchor;
        self
    }
}

/// The atlas a tilemap's tiles come from: one texture holding equal-sized
/// tiles in rows, as Tiled describes a tileset image, and the grid its tiles
/// are laid out on.
#[derive(Debug, Clone, PartialEq)]
pub struct Tileset {
    /// The atlas texture, from `DeviceResources::upload_texture`, uploaded
    /// sRGB. Row 0 of the image is its top. `None` draws each tile as a solid
    /// quad in its tint.
    pub texture_id: Option<TextureId>,
    /// Width and height of one tile, in texels. One tile is one cell: the
    /// bounding box of a cell of any [`orientation`](Self::orientation).
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
    /// Tiles drawn at their own size instead of one cell. At most
    /// [`MAX_TILE_SIZES`] apply; the first entry naming a tile wins. An
    /// animated tile takes the size of the tile the map holds for it, in
    /// every frame.
    pub tile_sizes: Vec<TileSize>,
    /// How the cells are laid out: square-on, isometric, staggered or
    /// hexagonal, as Tiled lays them out. Default
    /// [`TileOrientation::Orthogonal`].
    pub orientation: TileOrientation,
    /// A tangent-space normal map laid out as the atlas, tile for tile,
    /// uploaded linear (`TextureData::normal_map`). Green points up the
    /// image. Flipped and turned tiles turn their normals with them. Read by
    /// 2D lighting with `Lighting2dItem::normals` on; `None` lights the tiles
    /// flat. Default `None`.
    pub normal_atlas: Option<TextureId>,
    /// One user value per tile index, opaque to the renderer: a tile
    /// material's vertex hook reads the value of the tile a cell shows as
    /// `SpriteVertex::user`, so "this tile sways" is set once here and holds
    /// wherever the tile is painted. Tiles past the end read `0`. Default
    /// empty.
    pub user_values: Vec<u32>,
}

/// Most [`TileSize`] overrides one tileset applies.
pub const MAX_TILE_SIZES: usize = 255;

impl Tileset {
    /// A tileset of `columns` tiles per row, each `tile_size` texels, with no
    /// spacing or margin, nearest-filtered, unanimated and every tile one
    /// cell.
    pub fn new(texture_id: Option<TextureId>, tile_size: [u32; 2], columns: u32) -> Self {
        Self {
            texture_id,
            tile_size,
            columns: columns.max(1),
            spacing: 0,
            margin: 0,
            filter: TextureFilter::Nearest,
            animations: Vec::new(),
            tile_sizes: Vec::new(),
            orientation: TileOrientation::Orthogonal,
            normal_atlas: None,
            user_values: Vec::new(),
        }
    }

    /// This tileset with `normal_atlas` as its normal map.
    pub fn with_normal_atlas(mut self, normal_atlas: TextureId) -> Self {
        self.normal_atlas = Some(normal_atlas);
        self
    }

    /// The texel rectangle of tile `index` as `[x, y, width, height]`, `y`
    /// measured down from the top of the image. A tile with a
    /// [`TileSize`] keeps its slot's corner and takes that size.
    pub fn tile_rect(&self, index: u32) -> [u32; 4] {
        let columns = self.columns.max(1);
        let (col, row) = (index % columns, index / columns);
        let [w, h] = self.tile_size;
        let [w, h] = self
            .size_of(index)
            .map_or([w, h], |s| self.tile_sizes[s].size);
        [
            self.margin + col * (self.tile_size[0] + self.spacing),
            self.margin + row * (self.tile_size[1] + self.spacing),
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

    /// The entry of [`tile_sizes`](Self::tile_sizes) that applies to `tile`.
    pub fn size_of(&self, tile: u32) -> Option<usize> {
        self.tile_sizes
            .iter()
            .take(MAX_TILE_SIZES)
            .position(|s| s.tile == tile)
    }

    /// How far the tiles of this set reach past their cells' bounding
    /// boxes, in cells: `[left, bottom, right, top]`, each zero or more. Zero
    /// everywhere when every tile is one cell, outside a hexagonal grid,
    /// where a turned tile reaches past its box.
    pub fn overhang(&self) -> [f32; 4] {
        let [tw, th] = self.tile_size.map(|s| s.max(1) as f32);
        let turns = matches!(self.orientation, TileOrientation::Hexagonal { .. });
        let mut out = [0.0f32; 4];
        let mut reach = |lo: glam::Vec2, size: glam::Vec2| {
            let (mut lo, mut hi) = (lo, lo + size);
            if turns {
                // Any turn about the quad's centre stays within the circle
                // through its corners, measured in texels.
                let centre = (lo + hi) * 0.5;
                let radius = (size * glam::Vec2::new(tw, th) * 0.5).length();
                let r = glam::Vec2::new(radius / tw, radius / th);
                lo = lo.min(centre - r);
                hi = hi.max(centre + r);
            }
            out[0] = out[0].max(-lo.x);
            out[1] = out[1].max(-lo.y);
            out[2] = out[2].max(hi.x - 1.0);
            out[3] = out[3].max(hi.y - 1.0);
        };
        reach(glam::Vec2::ZERO, glam::Vec2::ONE);
        for (i, s) in self.tile_sizes.iter().take(MAX_TILE_SIZES).enumerate() {
            if self.size_of(s.tile) != Some(i) {
                continue;
            }
            let size = glam::Vec2::new(s.size[0] as f32 / tw, s.size[1] as f32 / th);
            reach(glam::Vec2::from(s.anchor), size);
        }
        out
    }
}

/// Lowest column or row a tilemap cell can have. Cells go to the GPU as
/// signed 16-bit columns and rows.
pub const MIN_TILE_CELL: i32 = i16::MIN as i32;

/// Highest column or row a tilemap cell can have.
pub const MAX_TILE_CELL: i32 = i16::MAX as i32;

/// A square block of a tilemap's cells, drawn with one draw call.
#[derive(Debug, Clone, PartialEq)]
pub struct TileChunk {
    /// The chunk's cells, row by row from the bottom row, `chunk_size` to a
    /// row.
    pub tiles: Vec<Tile>,
}

impl TileChunk {
    /// A chunk of `chunk_size` by `chunk_size` empty cells.
    pub fn new(chunk_size: u32) -> Self {
        Self {
            tiles: vec![Tile::EMPTY; (chunk_size * chunk_size) as usize],
        }
    }

    /// How many cells hold a tile.
    pub fn occupied(&self) -> usize {
        self.tiles.iter().filter(|t| !t.is_empty()).count()
    }
}

/// A grid of tiles from one tileset, held in square chunks keyed by chunk
/// coordinate.
///
/// Column 0 is the left edge of chunk 0 and row 0 its bottom: rows count up
/// the plane's +y, and both may be negative, down to [`MIN_TILE_CELL`]. Chunk
/// `[cx, cy]` holds the cells from `[cx, cy] * chunk_size` to one chunk
/// further. The map is sparse: [`set`](Self::set) creates the chunk holding a
/// cell, [`remove_chunk`](Self::remove_chunk) drops one, and a cell no chunk
/// holds is empty.
///
/// Upload it once, draw it with a [`TilemapItem`] each frame, and edit it in
/// place with `set_tile`, `set_chunk` and `remove_chunk`, which re-upload
/// only the chunks they touch. Each uploaded chunk holds an instance per
/// occupied cell and no more.
#[derive(Debug, Clone, PartialEq)]
pub struct Tilemap {
    /// The atlas and animations every tile reads.
    pub tileset: Tileset,
    /// When `true` (the default), the renderer keeps its copy of the cells
    /// after upload, for `set_tile`, CPU picks and
    /// [`TilemapEdits::tilemap`](crate::item_types::tilemap::TilemapEdits::tilemap).
    /// When `false` it drops them once the chunks are on the GPU, keeping
    /// only which cells are occupied: edits then replace whole chunks with
    /// `set_chunk`, and `set_tile` writes nothing.
    pub keep_cpu_copy: bool,
    chunk_size: u32,
    /// Keyed by `(cy, cx)`, so iteration runs row by row from the bottom.
    chunks: std::collections::BTreeMap<(i32, i32), TileChunk>,
}

impl Tilemap {
    /// Chunk side used by [`Tilemap::new`].
    pub const DEFAULT_CHUNK_SIZE: u32 = 32;

    /// A map `size` tiles across (columns, rows) from cell (0, 0), in chunks
    /// of [`DEFAULT_CHUNK_SIZE`](Self::DEFAULT_CHUNK_SIZE) tiles a side, with
    /// the chunks over that range created empty.
    pub fn new(tileset: Tileset, size: [u32; 2]) -> Self {
        Self::with_chunk_size(tileset, size, Self::DEFAULT_CHUNK_SIZE)
    }

    /// A map `size` tiles across in chunks of `chunk_size` tiles a side, with
    /// the chunks over cells `0..size` created empty. Each side is clamped to
    /// the cells up to [`MAX_TILE_CELL`] and the chunk side to `1..=256`.
    pub fn with_chunk_size(tileset: Tileset, size: [u32; 2], chunk_size: u32) -> Self {
        let mut map = Self::sparse(tileset, chunk_size);
        let cs = map.chunk_size as i64;
        let side = |s: u32| (s as i64).min(MAX_TILE_CELL as i64 + 1);
        let chunks = [side(size[0]), side(size[1])].map(|s| ((s + cs - 1) / cs) as i32);
        for cy in 0..chunks[1] {
            for cx in 0..chunks[0] {
                map.chunks.insert((cy, cx), TileChunk::new(map.chunk_size));
            }
        }
        map
    }

    /// A map with no chunks, in chunks of `chunk_size` tiles a side (clamped
    /// to `1..=256`), that grows as cells are set.
    pub fn sparse(tileset: Tileset, chunk_size: u32) -> Self {
        Self {
            tileset,
            keep_cpu_copy: true,
            chunk_size: chunk_size.clamp(1, 256),
            chunks: Default::default(),
        }
    }

    /// A map from rows of tile indices, top row first as a level is usually
    /// written down, with [`Tile::EMPTY_INDEX`] for an empty cell. The bottom
    /// row is row 0. Each row is as long as the first; shorter rows leave the
    /// rest empty.
    pub fn from_rows(tileset: Tileset, rows: &[&[u32]], chunk_size: u32) -> Self {
        let width = rows.first().map_or(0, |r| r.len()) as u32;
        let height = rows.len() as u32;
        let mut map = Self::with_chunk_size(tileset, [width, height], chunk_size);
        for (i, row) in rows.iter().enumerate() {
            let r = (height - 1 - i as u32) as i32;
            for (c, &index) in row.iter().enumerate().take(width as usize) {
                map.set(c as i32, r, Tile::new(index));
            }
        }
        map
    }

    /// Tiles along each side of a chunk.
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// How many chunks the map holds.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Every chunk with its coordinate `[cx, cy]`, row by row from the
    /// bottom-left one.
    pub fn chunks(&self) -> impl Iterator<Item = ([i32; 2], &TileChunk)> + '_ {
        self.chunks.iter().map(|(&(cy, cx), c)| ([cx, cy], c))
    }

    /// The chunk at `[cx, cy]`, when the map holds one.
    pub fn chunk(&self, coord: [i32; 2]) -> Option<&TileChunk> {
        self.chunks.get(&(coord[1], coord[0]))
    }

    /// The chunk coordinate holding cell (`col`, `row`) and the cell's flat
    /// index within it, `ty * chunk_size + tx`. `None` outside the cell
    /// range, [`MIN_TILE_CELL`] to [`MAX_TILE_CELL`].
    pub fn locate(&self, col: i32, row: i32) -> Option<([i32; 2], u32)> {
        let range = MIN_TILE_CELL..=MAX_TILE_CELL;
        if !range.contains(&col) || !range.contains(&row) {
            return None;
        }
        let cs = self.chunk_size as i32;
        let chunk = [col.div_euclid(cs), row.div_euclid(cs)];
        let tile = (row.rem_euclid(cs) * cs + col.rem_euclid(cs)) as u32;
        Some((chunk, tile))
    }

    /// The cell (`col`, `row`) of flat index `tile` in chunk `chunk`, the
    /// inverse of [`locate`](Self::locate). `None` when `tile` is past the
    /// chunk or the cell is outside the cell range.
    pub fn cell_of(&self, chunk: [i32; 2], tile: u32) -> Option<[i32; 2]> {
        let cs = self.chunk_size;
        if tile >= cs * cs {
            return None;
        }
        let cs = cs as i64;
        let col = chunk[0] as i64 * cs + (tile as i64 % cs);
        let row = chunk[1] as i64 * cs + (tile as i64 / cs);
        let range = MIN_TILE_CELL as i64..=MAX_TILE_CELL as i64;
        (range.contains(&col) && range.contains(&row)).then_some([col as i32, row as i32])
    }

    /// The tile at (`col`, `row`); `None` where no chunk holds the cell.
    pub fn get(&self, col: i32, row: i32) -> Option<Tile> {
        let (chunk, tile) = self.locate(col, row)?;
        Some(self.chunk(chunk)?.tiles[tile as usize])
    }

    /// Put `tile` at (`col`, `row`), creating its chunk when the map has
    /// none there, and return the chunk's coordinate. Emptying a cell no
    /// chunk holds creates nothing. `None` outside the cell range.
    pub fn set(&mut self, col: i32, row: i32, tile: Tile) -> Option<[i32; 2]> {
        let (chunk, index) = self.locate(col, row)?;
        let key = (chunk[1], chunk[0]);
        if tile.is_empty() && !self.chunks.contains_key(&key) {
            return Some(chunk);
        }
        let cs = self.chunk_size;
        self.chunks
            .entry(key)
            .or_insert_with(|| TileChunk::new(cs))
            .tiles[index as usize] = tile;
        Some(chunk)
    }

    /// Replace the chunk at `[cx, cy]` with `chunk`, creating it when the map
    /// has none there. `chunk.tiles` is cut or padded with empty cells to the
    /// chunk's size. `false`, changing nothing, when the chunk lies outside
    /// the cell range.
    pub fn set_chunk(&mut self, coord: [i32; 2], mut chunk: TileChunk) -> bool {
        let [c0, r0, c1, r1] = self.chunk_cells(coord);
        if c0 >= c1 || r0 >= r1 {
            return false;
        }
        let cells = (self.chunk_size * self.chunk_size) as usize;
        chunk.tiles.resize(cells, Tile::EMPTY);
        self.chunks.insert((coord[1], coord[0]), chunk);
        true
    }

    /// Drop the chunk at `[cx, cy]` and return it.
    pub fn remove_chunk(&mut self, coord: [i32; 2]) -> Option<TileChunk> {
        self.chunks.remove(&(coord[1], coord[0]))
    }

    /// The cell range chunk `[cx, cy]` covers, as `[col0, row0, col1, row1]`
    /// with the end exclusive and clamped to the cell range.
    pub fn chunk_cells(&self, coord: [i32; 2]) -> [i32; 4] {
        let cs = self.chunk_size as i64;
        let lo = MIN_TILE_CELL as i64;
        let hi = MAX_TILE_CELL as i64 + 1;
        let at = |c: i32, k: i64| ((c as i64 + k) * cs).clamp(lo, hi) as i32;
        [
            at(coord[0], 0),
            at(coord[1], 0),
            at(coord[0], 1),
            at(coord[1], 1),
        ]
    }

    /// The cells every chunk of the map covers together, as `[col0, row0,
    /// col1, row1]` with the end exclusive. `None` for a map with no chunks.
    pub fn cell_bounds(&self) -> Option<[i32; 4]> {
        self.chunks()
            .map(|(c, _)| self.chunk_cells(c))
            .reduce(|a, b| {
                [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ]
            })
    }

    /// Drop every chunk's cells, keeping the tileset and the chunk size.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn clear_chunks(&mut self) {
        self.chunks.clear();
    }
}

/// How a blended tilemap layer takes its place in the painter's order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TilemapPainter {
    /// The whole layer is one batch at sort key `key`, its chunks in map
    /// order.
    Layer {
        /// Sort key against other painter content on the same layer.
        key: f32,
    },
    /// Each drawn row of each chunk is its own batch, keyed by where the row
    /// stands so it sorts against sprites walking past it: the key of a row
    /// whose bottom edge is at plane `y` is `offset - y`, so rows further up
    /// the plane draw first and anything standing lower covers them. A
    /// sprite whose feet are at plane `y` sorts with the rows when its key is
    /// `offset - y` in the same layer. A tall tile draws in the batch of the
    /// row it stands on.
    ///
    /// On the other grids a row is every cell whose bounding box has the
    /// same bottom edge: a line of diamonds across the plane on an isometric
    /// grid, and on a grid of staggered columns the shifted and unshifted
    /// halves of a row separately.
    PerRow {
        /// Added to every row's key, to place the rows in the host's key
        /// space.
        offset: f32,
    },
}

impl Default for TilemapPainter {
    fn default() -> Self {
        Self::Layer { key: 0.0 }
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
    /// `layer` (lower first), then by the sort key `painter` gives it against
    /// other painter content on the same layer, then submission order.
    /// Never writes depth. For decoration layers with soft edges, and for
    /// walls and props that sort against sprites by row.
    Blended {
        /// Layer order. Lower layers draw first.
        layer: u32,
        /// When `true` the tiles test against scene depth, so nearer
        /// geometry covers them; when `false` they draw over everything
        /// already in the scene.
        depth_test: bool,
        /// One batch for the layer, or one per row.
        painter: TilemapPainter,
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
/// The map's data stays on the GPU; the item places it. On an orthogonal
/// grid cell (`col`, `row`) covers plane `origin + [col, row] * cell_size` to
/// one cell further, on `plane` at `elevation`; the other orientations place
/// the cell's bounding box at `origin + cell_origin * cell_size`, from
/// [`Tileset::cell_origin`]. Several items may draw one map.
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
    /// World size of one cell's bounding box along plane x and y. Default
    /// `[1.0, 1.0]`.
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
    /// animate, or a fixed value to hold a frame. Ignored while
    /// [`use_frame_clock`](Self::use_frame_clock) is set.
    pub time: f32,
    /// When `true`, tiles animate on the frame's own clock (the time the
    /// shaders see, `FrameData::time` or the renderer's clock) and `time` is
    /// ignored. Default `false`.
    pub use_frame_clock: bool,
    /// Cutout or blended. Default [`TilemapDraw::Cutout`].
    pub draw: TilemapDraw,
    /// When `true`, the layer is moved by less than a device pixel so its
    /// origin lands on the pixel grid, which keeps every texel of pixel art
    /// drawn at an integer scale on whole pixels. Default `false`.
    pub pixel_snap: bool,
    /// Whether a painter mask cuts the layer. Read by a blended layer only,
    /// which is cut when its layer is in the range of the mask in the slot
    /// it names. Default [`MaskTest::None`].
    pub mask: MaskTest,
    /// A tile material whose vertex hook moves each corner of every tile,
    /// registered with
    /// [`TileMaterials::register_tile_material`](crate::item_types::tilemap::TileMaterials::register_tile_material).
    /// Picks follow the moved tiles. `None` (the default) draws the grid as
    /// it lies.
    pub material: Option<TileMaterial>,
    /// The registered sprite deformers that move this layer's tiles, before
    /// its material. Empty (the default) runs none.
    pub deformers: crate::item_types::sprite::SpriteDeformerSet,
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
            use_frame_clock: false,
            draw: TilemapDraw::default(),
            pixel_snap: false,
            mask: MaskTest::None,
            material: None,
            deformers: crate::item_types::sprite::SpriteDeformerSet::NONE,
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

    /// The world position of the centre of cell (`col`, `row`) of a map
    /// laid out by `tileset`, drawn by this item for a camera at plane
    /// position `camera` (which only matters under parallax).
    pub fn cell_to_world(
        &self,
        tileset: &Tileset,
        col: i32,
        row: i32,
        camera: glam::DVec2,
    ) -> glam::DVec3 {
        let cell = glam::DVec2::new(self.cell_size[0] as f64, self.cell_size[1] as f64);
        let p = self.drawn_origin(camera) + tileset.cell_centre(col, row) * cell;
        self.plane.to_world(p, self.elevation as f64)
    }

    /// The cell of a map laid out by `tileset` whose outline holds the world
    /// point `world`, projected along the plane's normal, as this item
    /// draws it for a camera at plane position `camera`. `None` past the
    /// cell range or for a cell size of zero.
    pub fn world_to_cell(
        &self,
        tileset: &Tileset,
        world: glam::DVec3,
        camera: glam::DVec2,
    ) -> Option<[i32; 2]> {
        let cell = glam::DVec2::new(self.cell_size[0] as f64, self.cell_size[1] as f64);
        if cell.x == 0.0 || cell.y == 0.0 {
            return None;
        }
        let local = (self.plane.to_plane(world) - self.drawn_origin(camera)) / cell;
        if !local.is_finite() {
            return None;
        }
        tileset.cell_at(local)
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
        let tile = Tile::from_tiled_gid(0x3000_0003);
        assert_eq!(tile.index, 3);
        assert!(tile.rotate_120);
        assert_eq!(tile.flip, TileFlip::DIAGONAL);
        assert!(!Tile::from_tiled_gid(0xA000_0007).rotate_120);
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
        assert_eq!(map.chunk_count(), 6);
        assert_eq!(map.cell_bounds(), Some([0, 0, 12, 8]));
        for row in -9..9 {
            for col in -9..13 {
                let (chunk, tile) = map.locate(col, row).unwrap();
                assert_eq!(map.cell_of(chunk, tile), Some([col, row]));
            }
        }
        assert_eq!(map.locate(-3, 5), Some(([-1, 1], 4 + 1)));
        assert_eq!(map.locate(MAX_TILE_CELL + 1, 0), None);
        assert_eq!(map.chunk_cells([2, 0]), [8, 0, 12, 4]);
        assert_eq!(map.cell_of([0, 0], 16), None);
    }

    #[test]
    fn a_sparse_map_grows_and_shrinks_by_chunk() {
        let mut map = Tilemap::sparse(Tileset::new(None, [8, 8], 4), 4);
        assert_eq!(map.cell_bounds(), None);
        assert_eq!(map.set(-5, -1, Tile::EMPTY), Some([-2, -1]));
        assert_eq!(map.chunk_count(), 0, "emptying a cell creates no chunk");
        assert_eq!(map.set(-5, -1, Tile::new(3)), Some([-2, -1]));
        assert_eq!(map.chunk_count(), 1);
        assert_eq!(map.get(-5, -1).unwrap().index, 3);
        assert!(map.get(-6, -1).unwrap().is_empty());
        assert_eq!(map.get(0, 0), None);
        assert_eq!(map.cell_bounds(), Some([-8, -4, -4, 0]));
        assert!(map.remove_chunk([-2, -1]).is_some());
        assert_eq!(map.get(-5, -1), None);
        assert!(map.set_chunk(
            [1, 1],
            TileChunk {
                tiles: vec![Tile::new(1)]
            }
        ));
        assert_eq!(map.chunk([1, 1]).unwrap().tiles.len(), 16);
        assert_eq!(map.get(4, 4).unwrap().index, 1);
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
        set.tile_sizes.push(TileSize::new(4, [16, 16]));
        assert_eq!(set.tile_rect(4), [1 + 18, 1 + 10, 16, 16]);
    }

    #[test]
    fn tall_tiles_reach_past_their_cells() {
        let mut set = Tileset::new(None, [16, 16], 4);
        assert_eq!(set.overhang(), [0.0; 4]);
        set.tile_sizes.push(TileSize::new(2, [16, 32]));
        set.tile_sizes
            .push(TileSize::new(5, [48, 16]).with_anchor([-1.0, 0.0]));
        // A later entry for the same tile is ignored.
        set.tile_sizes.push(TileSize::new(2, [16, 160]));
        assert_eq!(set.overhang(), [1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn items_map_corner_cells_to_the_world_and_back() {
        use super::super::grid::{StaggerAxis, StaggerIndex};
        let mut item = TilemapItem::new(TilemapId::from_index(0));
        item.plane = TilemapPlane::Xz;
        item.origin = [-30.0, 12.0];
        item.cell_size = [24.0, 12.0];
        item.elevation = 2.0;
        item.parallax = [0.5, 1.0];
        let camera = glam::DVec2::new(40.0, -8.0);
        for orientation in [
            TileOrientation::Orthogonal,
            TileOrientation::Isometric,
            TileOrientation::Hexagonal {
                side_length: 6,
                axis: StaggerAxis::X,
                index: StaggerIndex::Even,
            },
        ] {
            let mut set = Tileset::new(None, [24, 12], 4);
            set.orientation = orientation;
            let map = Tilemap::with_chunk_size(set, [40, 30], 8);
            let [c0, r0, c1, r1] = map.cell_bounds().unwrap();
            for [col, row] in [[c0, r0], [c1 - 1, r0], [c0, r1 - 1], [c1 - 1, r1 - 1]] {
                let w = item.cell_to_world(&map.tileset, col, row, camera);
                assert!((w.y + 2.0).abs() < 1e-9, "on the plane at its elevation");
                assert_eq!(
                    item.world_to_cell(&map.tileset, w, camera),
                    Some([col, row]),
                    "{orientation:?}"
                );
            }
        }
    }

    #[test]
    fn turned_hexagons_reach_past_their_boxes() {
        use super::super::grid::{StaggerAxis, StaggerIndex};
        let mut set = Tileset::new(None, [28, 32], 4);
        assert_eq!(set.overhang(), [0.0; 4]);
        set.orientation = TileOrientation::Hexagonal {
            side_length: 16,
            axis: StaggerAxis::Y,
            index: StaggerIndex::Odd,
        };
        let o = set.overhang();
        // The circle through the corners of a 28 x 32 quad: radius 21.26.
        let r = (14.0f32.hypot(16.0) - 14.0) / 28.0;
        assert!((o[0] - r).abs() < 1e-5 && (o[2] - r).abs() < 1e-5, "{o:?}");
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

/// Handle to a tile material registered with
/// [`TileMaterials::register_tile_material`](crate::item_types::tilemap::TileMaterials::register_tile_material).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileMaterialId(pub(crate) u32);

impl TileMaterialId {
    /// Registry index.
    pub fn index(self) -> u32 {
        self.0
    }
}

/// A layer's tile material: the registered material and the values of the
/// parameters it declares.
#[derive(Clone, Debug, PartialEq)]
pub struct TileMaterial {
    /// The registered material.
    pub id: TileMaterialId,
    /// Parameter values by name. Parameters not named take the defaults the
    /// material declares.
    pub params: Vec<(String, viewport_lib_types::plugin::material::ParamValue)>,
}

impl TileMaterial {
    /// The material `id` with every parameter at its default.
    pub fn new(id: TileMaterialId) -> Self {
        Self {
            id,
            params: Vec::new(),
        }
    }

    /// Set parameter `name`.
    pub fn with_param(
        mut self,
        name: impl Into<String>,
        value: impl Into<viewport_lib_types::plugin::material::ParamValue>,
    ) -> Self {
        let name = name.into();
        let value = value.into();
        match self.params.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => self.params.push((name, value)),
        }
        self
    }
}
