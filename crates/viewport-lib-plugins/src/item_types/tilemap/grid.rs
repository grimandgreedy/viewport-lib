//! Where a tileset's cells sit on the plane, for each grid orientation Tiled
//! has, and which cell holds a plane point.
//!
//! Positions here are in cells: one cell is one tileset tile, `tile_size`
//! texels, which a [`TilemapItem`](super::TilemapItem) draws `cell_size`
//! world units across. Every cell has a bounding box one cell across; the
//! tile's image fills it from its bottom-left corner, as Tiled draws a tile
//! from the bottom-left of its cell.
//!
//! The geometry follows Tiled's renderers, turned so rows count up the plane:
//! Tiled's cell `(x, y)` is cell `(x, -y)` here, its picture mirrored top to
//! bottom. Stagger parity is a row's (or column's) own number, so the same
//! cells stagger as in Tiled for any conversion `row = k - y` with `k` even.

use super::types::Tileset;
use glam::{DVec2, I64Vec2};

/// Which axis of a staggered or hexagonal map is staggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StaggerAxis {
    /// Every other column is shifted down by half a row.
    X,
    /// Every other row is shifted right by half a column.
    #[default]
    Y,
}

/// Which rows (or columns) of a staggered or hexagonal map are shifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StaggerIndex {
    /// The odd ones, negative ones included.
    #[default]
    Odd,
    /// The even ones.
    Even,
}

/// How a tileset's cells are laid out on the plane, with Tiled's four map
/// orientations and the same parameters.
///
/// Rows count up the plane while Tiled's count down the screen, so Tiled's
/// cell `(x, y)` is cell `(x, -y)` here (or `(x, k - y)` for any even `k`,
/// which keeps every row's stagger). Every orientation puts the bounding box
/// of cell `(0, 0)` with its bottom-left corner at the layer's origin, moved
/// by the stagger when column or row 0 is a shifted one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TileOrientation {
    /// Square-on rows and columns: cell `(col, row)` covers `[col, row]` to
    /// one cell further.
    #[default]
    Orthogonal,
    /// Diamond cells: each column steps half a cell right and half a cell
    /// down, each row half a cell right and half a cell up.
    Isometric,
    /// Diamond cells in staggered rows or columns, filling a rectangle.
    Staggered {
        /// Which axis is staggered.
        axis: StaggerAxis,
        /// Whether the odd or the even rows (or columns) are shifted.
        index: StaggerIndex,
    },
    /// Hexagonal cells in staggered rows (pointy top) or columns (flat top).
    /// A tile's diagonal flip turns it 60 degrees clockwise instead, and
    /// [`Tile::rotate_120`](super::Tile::rotate_120) turns it 120 degrees,
    /// as Tiled draws them.
    Hexagonal {
        /// Length of the flat side, in texels: Tiled's `hexsidelength` for a
        /// tileset whose tiles are the map's cell size.
        side_length: u32,
        /// Which axis is staggered.
        axis: StaggerAxis,
        /// Whether the odd or the even rows (or columns) are shifted.
        index: StaggerIndex,
    },
}

/// How a grid places its cells, in half-texels so every corner Tiled's
/// renderers produce is a whole number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GridLayout {
    pub(crate) kind: GridKind,
    /// Which parity is shifted, for the staggered kinds.
    pub(crate) even: bool,
    /// Whether diagonal flips and `rotate_120` turn tiles.
    pub(crate) hexagonal: bool,
    /// One cell, `tile_size`, in half-texels.
    pub(crate) cell: I64Vec2,
    /// Column and row step of the staggered kinds, in half-texels.
    pub(crate) step: I64Vec2,
    /// How far a staggered column moves down or a staggered row right.
    pub(crate) shift: i64,
    /// The cell's outline from its bounding box's bottom-left corner,
    /// counter-clockwise, in half-texels.
    outline: [I64Vec2; 6],
    outline_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GridKind {
    Orthogonal,
    Isometric,
    /// Columns staggered.
    StaggerX,
    /// Rows staggered.
    StaggerY,
}

impl GridLayout {
    pub(crate) fn new(tileset: &Tileset) -> Self {
        let [tw, th] = tileset.tile_size.map(|s| s.max(1) as i64);
        let cell = I64Vec2::new(2 * tw, 2 * th);
        let square = |kind| {
            let outline = match kind {
                GridKind::Isometric => [
                    I64Vec2::new(tw, 0),
                    I64Vec2::new(2 * tw, th),
                    I64Vec2::new(tw, 2 * th),
                    I64Vec2::new(0, th),
                ],
                _ => [
                    I64Vec2::ZERO,
                    I64Vec2::new(2 * tw, 0),
                    I64Vec2::new(2 * tw, 2 * th),
                    I64Vec2::new(0, 2 * th),
                ],
            };
            let mut six = [I64Vec2::ZERO; 6];
            six[..4].copy_from_slice(&outline);
            Self {
                kind,
                even: false,
                hexagonal: false,
                cell,
                step: cell,
                shift: 0,
                outline: six,
                outline_len: 4,
            }
        };
        let (axis, index, side, hexagonal) = match tileset.orientation {
            TileOrientation::Orthogonal => return square(GridKind::Orthogonal),
            TileOrientation::Isometric => return square(GridKind::Isometric),
            TileOrientation::Staggered { axis, index } => (axis, index, 0, false),
            TileOrientation::Hexagonal {
                side_length,
                axis,
                index,
            } => (axis, index, side_length as i64, true),
        };
        // Tiled's hexagonal renderer, which also draws staggered maps with a
        // side of zero: even tile sizes, whole-texel halves.
        let (w, h) = (tw & !1, th & !1);
        let stagger_x = axis == StaggerAxis::X;
        let (side_x, side_y) = if stagger_x { (side, 0) } else { (0, side) };
        let side_x = side_x.min(w);
        let side_y = side_y.min(h);
        let offset_x = (w - side_x) / 2;
        let offset_y = (h - side_y) / 2;
        let column_width = offset_x + side_x;
        let row_height = offset_y + side_y;
        let (kind, step, shift) = if stagger_x {
            (
                GridKind::StaggerX,
                I64Vec2::new(column_width, h),
                row_height,
            )
        } else {
            (
                GridKind::StaggerY,
                I64Vec2::new(w, row_height),
                column_width,
            )
        };
        // Tiled's eight corners (y down), mirrored to y up and reversed into
        // counter-clockwise order, with repeated corners dropped.
        let tiled = [
            (0, h - offset_y),
            (0, offset_y),
            (offset_x, 0),
            (w - offset_x, 0),
            (w, offset_y),
            (w, h - offset_y),
            (w - offset_x, h),
            (offset_x, h),
        ];
        let mut outline = [I64Vec2::ZERO; 6];
        let mut len = 0;
        for &(x, y) in tiled.iter().rev() {
            let p = I64Vec2::new(2 * x, 2 * (h - y));
            if len > 0 && outline[len - 1] == p || len == 6 {
                continue;
            }
            if len > 0 && outline[0] == p {
                continue;
            }
            outline[len] = p;
            len += 1;
        }
        Self {
            kind,
            even: index == StaggerIndex::Even,
            hexagonal,
            cell,
            step: step * 2,
            shift: shift * 2,
            outline,
            outline_len: len,
        }
    }

    /// Whether row or column `i` is shifted.
    fn staggered(&self, i: i64) -> bool {
        (i & 1 != 0) != self.even
    }

    /// The bottom-left corner of cell (`col`, `row`)'s bounding box, in
    /// half-texels.
    pub(crate) fn origin_half_texels(&self, col: i64, row: i64) -> I64Vec2 {
        let c = self.cell;
        match self.kind {
            GridKind::Orthogonal => I64Vec2::new(col * c.x, row * c.y),
            GridKind::Isometric => I64Vec2::new((col + row) * c.x / 2, (row - col) * c.y / 2),
            GridKind::StaggerX => I64Vec2::new(
                col * self.step.x,
                row * self.step.y - if self.staggered(col) { self.shift } else { 0 },
            ),
            GridKind::StaggerY => I64Vec2::new(
                col * self.step.x + if self.staggered(row) { self.shift } else { 0 },
                row * self.step.y,
            ),
        }
    }

    /// The cell's outline, counter-clockwise from its bounding box's
    /// bottom-left corner, in half-texels.
    pub(crate) fn outline_half_texels(&self) -> &[I64Vec2] {
        &self.outline[..self.outline_len]
    }

    /// Half-texels to cells.
    pub(crate) fn to_cells(&self, p: I64Vec2) -> DVec2 {
        p.as_dvec2() / self.cell.as_dvec2()
    }

    /// The bottom-left corner of cell (`col`, `row`)'s bounding box, in
    /// cells.
    pub(crate) fn origin(&self, col: i32, row: i32) -> DVec2 {
        match self.kind {
            // Exactly the cell, as the shader computes it.
            GridKind::Orthogonal => DVec2::new(col as f64, row as f64),
            _ => self.to_cells(self.origin_half_texels(col as i64, row as i64)),
        }
    }

    /// The centre of the cell's outline, from its bounding box's bottom-left
    /// corner, in half-texels (doubled, so it stays whole).
    fn centre_twice(&self) -> I64Vec2 {
        let pts = self.outline_half_texels();
        let (lo, hi) = pts
            .iter()
            .fold((pts[0], pts[0]), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        lo + hi
    }

    /// The centre of cell (`col`, `row`), in cells.
    pub(crate) fn centre(&self, col: i32, row: i32) -> DVec2 {
        let o = self.origin_half_texels(col as i64, row as i64);
        (o.as_dvec2() + self.centre_twice().as_dvec2() * 0.5) / self.cell.as_dvec2()
    }

    /// The cell's outline in cells, counter-clockwise.
    pub(crate) fn outline(&self, col: i32, row: i32) -> Vec<DVec2> {
        let o = self.origin_half_texels(col as i64, row as i64);
        self.outline_half_texels()
            .iter()
            .map(|&p| self.to_cells(o + p))
            .collect()
    }

    /// Whether `p` (half-texels) lies in cell (`col`, `row`)'s outline,
    /// edges included.
    fn contains(&self, col: i64, row: i64, p: DVec2) -> bool {
        let o = self.origin_half_texels(col, row).as_dvec2();
        let pts = self.outline_half_texels();
        (0..pts.len()).all(|i| {
            let a = o + pts[i].as_dvec2();
            let b = o + pts[(i + 1) % pts.len()].as_dvec2();
            (b - a).perp_dot(p - a) >= -1e-9
        })
    }

    /// The cell holding plane point `p`, in cells.
    pub(crate) fn cell_at(&self, p: DVec2) -> [i64; 2] {
        match self.kind {
            GridKind::Orthogonal => [p.x.floor() as i64, p.y.floor() as i64],
            GridKind::Isometric => [
                (p.x - p.y + 0.5).floor() as i64,
                (p.x + p.y - 0.5).floor() as i64,
            ],
            GridKind::StaggerX | GridKind::StaggerY => {
                let h = p * self.cell.as_dvec2();
                let step = self.step.as_dvec2().max(DVec2::ONE);
                let shift = self.shift as f64;
                let mut candidates = Vec::with_capacity(9);
                if self.kind == GridKind::StaggerX {
                    let c0 = (h.x / step.x).floor() as i64;
                    for col in c0 - 1..=c0 + 1 {
                        let down = if self.staggered(col) { shift } else { 0.0 };
                        let r0 = ((h.y + down) / step.y).floor() as i64;
                        candidates.extend((r0 - 1..=r0 + 1).map(|row| [col, row]));
                    }
                } else {
                    let r0 = (h.y / step.y).floor() as i64;
                    for row in r0 - 1..=r0 + 1 {
                        let right = if self.staggered(row) { shift } else { 0.0 };
                        let c0 = ((h.x - right) / step.x).floor() as i64;
                        candidates.extend((c0 - 1..=c0 + 1).map(|col| [col, row]));
                    }
                }
                // The cell whose outline holds the point; on an edge two
                // share, or in a gap rounding left, the nearest centre.
                let centre = self.centre_twice().as_dvec2() * 0.5;
                let distance = |&[col, row]: &[i64; 2]| {
                    let c = self.origin_half_texels(col, row).as_dvec2() + centre;
                    (c - h).length_squared()
                };
                let inside: Vec<[i64; 2]> = candidates
                    .iter()
                    .copied()
                    .filter(|&[col, row]| self.contains(col, row, h))
                    .collect();
                let pool = if inside.is_empty() {
                    &candidates
                } else {
                    &inside
                };
                *pool
                    .iter()
                    .min_by(|a, b| distance(a).total_cmp(&distance(b)))
                    .expect("nine candidates")
            }
        }
    }

    /// The plane box, in cells, that the bounding boxes of cells
    /// `[col0, row0, col1, row1]` (end exclusive) cover, as `(min, max)`.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn extent(&self, cells: [i32; 4]) -> (DVec2, DVec2) {
        let [c0, r0, c1, r1] = cells;
        if self.kind == GridKind::Orthogonal {
            return (
                DVec2::new(c0 as f64, r0 as f64),
                DVec2::new(c1 as f64, r1 as f64),
            );
        }
        let (c1, r1) = (c1 - 1, r1 - 1);
        let mut lo = DVec2::MAX;
        let mut hi = DVec2::MIN;
        for (col, row) in [(c0, r0), (c1, r0), (c0, r1), (c1, r1)] {
            let o = self.origin(col, row);
            lo = lo.min(o);
            hi = hi.max(o + DVec2::ONE);
        }
        // A staggered column or row at a corner may sit either way.
        let shift = self.shift as f64;
        match self.kind {
            GridKind::StaggerX => lo.y -= shift / self.cell.y as f64,
            GridKind::StaggerY => hi.x += shift / self.cell.x as f64,
            _ => {}
        }
        (lo, hi)
    }

    /// The outline, in cells and counter-clockwise, of the region cells
    /// `[col0, row0, col1, row1]` (end exclusive) cover: exact for
    /// orthogonal and isometric grids, the bounding box for staggered ones.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn region_outline(&self, cells: [i32; 4]) -> [DVec2; 4] {
        let [c0, r0, c1, r1] = cells;
        match self.kind {
            GridKind::Isometric => {
                // The diamond grid is the square grid under one linear map:
                // corner (u, v) of the square grid sits at this point.
                let at = |u: i32, v: i32| {
                    let (u, v) = (u as f64, v as f64);
                    DVec2::new((u + v) * 0.5, (v - u) * 0.5 + 0.5)
                };
                [at(c0, r0), at(c1, r0), at(c1, r1), at(c0, r1)]
            }
            _ => {
                let (lo, hi) = self.extent(cells);
                [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)]
            }
        }
    }

    /// The neighbour across each side of cell (`col`, `row`), with the
    /// side's ends counter-clockwise, in half-texels.
    pub(crate) fn sides(&self, col: i64, row: i64) -> Vec<([i64; 2], I64Vec2, I64Vec2)> {
        let o = self.origin_half_texels(col, row);
        let pts = self.outline_half_texels();
        let centre = o.as_dvec2() + self.centre_twice().as_dvec2() * 0.5;
        (0..pts.len())
            .map(|i| {
                let a = o + pts[i];
                let b = o + pts[(i + 1) % pts.len()];
                // The neighbour's centre is this one's mirrored through the
                // side's midpoint.
                let mid = (a + b).as_dvec2() * 0.5;
                let across = mid * 2.0 - centre;
                (self.cell_at(across / self.cell.as_dvec2()), a, b)
            })
            .collect()
    }

    /// The tileset uniform's orientation word: the kind in bits 0 and 1
    /// (orthogonal, isometric, columns staggered, rows staggered), the even
    /// parity in bit 2 and hexagonal turns in bit 3.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn word(&self) -> u32 {
        let kind = match self.kind {
            GridKind::Orthogonal => 0,
            GridKind::Isometric => 1,
            GridKind::StaggerX => 2,
            GridKind::StaggerY => 3,
        };
        kind | (self.even as u32) << 2 | (self.hexagonal as u32) << 3
    }

    /// The tileset uniform's grid vector: the staggered kinds' column and
    /// row steps and their shift, in cells.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn grid(&self) -> [f32; 4] {
        let c = self.cell.as_dvec2();
        [
            (self.step.x as f64 / c.x) as f32,
            (self.step.y as f64 / c.y) as f32,
            (self.shift as f64
                / if self.kind == GridKind::StaggerX {
                    c.y
                } else {
                    c.x
                }) as f32,
            0.0,
        ]
    }
}

impl Tileset {
    /// The plane offset, in cells, of the bottom-left corner of cell
    /// (`col`, `row`)'s bounding box: where the tile's image starts.
    pub fn cell_origin(&self, col: i32, row: i32) -> DVec2 {
        GridLayout::new(self).origin(col, row)
    }

    /// The plane offset, in cells, of the centre of cell (`col`, `row`).
    pub fn cell_centre(&self, col: i32, row: i32) -> DVec2 {
        GridLayout::new(self).centre(col, row)
    }

    /// The outline of cell (`col`, `row`) in cells, counter-clockwise: four
    /// corners, or six on a hexagonal grid.
    pub fn cell_outline(&self, col: i32, row: i32) -> Vec<DVec2> {
        GridLayout::new(self).outline(col, row)
    }

    /// The cell whose outline holds plane offset `p`, in cells. Points on
    /// an edge two cells share go to one of them; `None` past the cell
    /// range.
    pub fn cell_at(&self, p: DVec2) -> Option<[i32; 2]> {
        let [col, row] = GridLayout::new(self).cell_at(p);
        let range = super::MIN_TILE_CELL as i64..=super::MAX_TILE_CELL as i64;
        (range.contains(&col) && range.contains(&row)).then_some([col as i32, row as i32])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(size: [u32; 2], orientation: TileOrientation) -> Tileset {
        let mut set = Tileset::new(None, size, 4);
        set.orientation = orientation;
        set
    }

    /// Tiled's own placement, ported from its renderers: the top-left corner
    /// of the bounding box of Tiled's cell (`x`, `y`) in pixels, y down.
    fn tiled_top_left(size: [u32; 2], orientation: TileOrientation, x: i64, y: i64) -> (f64, f64) {
        let (tw, th) = (size[0] as i64, size[1] as i64);
        match orientation {
            TileOrientation::Orthogonal => ((x * tw) as f64, (y * th) as f64),
            TileOrientation::Isometric => {
                // `tileToScreenCoords` gives the top corner, with the
                // map-height origin left out.
                let top_x = (x - y) as f64 * tw as f64 / 2.0;
                let top_y = (x + y) as f64 * th as f64 / 2.0;
                (top_x - tw as f64 / 2.0, top_y)
            }
            TileOrientation::Staggered { axis, index }
            | TileOrientation::Hexagonal { axis, index, .. } => {
                let side = match orientation {
                    TileOrientation::Hexagonal { side_length, .. } => side_length as i64,
                    _ => 0,
                };
                let (w, h) = (tw & !1, th & !1);
                let stagger_x = axis == StaggerAxis::X;
                let even = index == StaggerIndex::Even;
                let side_x = if stagger_x { side } else { 0 };
                let side_y = if stagger_x { 0 } else { side };
                let offset_x = (w - side_x) / 2;
                let offset_y = (h - side_y) / 2;
                let column_width = offset_x + side_x;
                let row_height = offset_y + side_y;
                if stagger_x {
                    let mut py = y * (h + side_y);
                    if ((x & 1) != 0) ^ even {
                        py += row_height;
                    }
                    ((x * column_width) as f64, py as f64)
                } else {
                    let mut px = x * (w + side_x);
                    if ((y & 1) != 0) ^ even {
                        px += column_width;
                    }
                    (px as f64, (y * row_height) as f64)
                }
            }
        }
    }

    fn orientations() -> Vec<([u32; 2], TileOrientation)> {
        use StaggerAxis::{X, Y};
        use StaggerIndex::{Even, Odd};
        let mut out = vec![
            ([16, 16], TileOrientation::Orthogonal),
            ([32, 16], TileOrientation::Isometric),
            ([64, 32], TileOrientation::Isometric),
        ];
        for axis in [X, Y] {
            for index in [Odd, Even] {
                out.push(([32, 16], TileOrientation::Staggered { axis, index }));
                out.push((
                    [28, 32],
                    TileOrientation::Hexagonal {
                        side_length: 16,
                        axis,
                        index,
                    },
                ));
                out.push((
                    [32, 28],
                    TileOrientation::Hexagonal {
                        side_length: 14,
                        axis,
                        index,
                    },
                ));
                // Odd sizes take Tiled's whole-texel halves.
                out.push((
                    [31, 27],
                    TileOrientation::Hexagonal {
                        side_length: 10,
                        axis,
                        index,
                    },
                ));
            }
        }
        out
    }

    #[test]
    fn cells_sit_where_tiled_puts_them_mirrored() {
        for (size, orientation) in orientations() {
            let set = set(size, orientation);
            let layout = GridLayout::new(&set);
            let (tw, th) = (size[0] as f64, size[1] as f64);
            // Tiled's top-left corner, mirrored to y up, differs from the
            // bottom-left corner here by one fixed offset over every cell.
            let mirrored = |col: i64, row: i64| {
                let (x, y) = tiled_top_left(size, orientation, col, -row);
                DVec2::new(x / tw, -y / th)
            };
            let fixed = layout.origin(0, 0) - mirrored(0, 0);
            for row in -7..=7 {
                for col in -7..=7 {
                    let ours = layout.origin(col as i32, row as i32);
                    let theirs = mirrored(col, row) + fixed;
                    assert!(
                        (ours - theirs).length() < 1e-9,
                        "{orientation:?} {size:?} cell ({col}, {row}): {ours} vs {theirs}"
                    );
                }
            }
            if layout.even && layout.kind == GridKind::StaggerX {
                assert!(layout.origin(0, 0).y < 0.0, "{orientation:?}");
            } else if layout.even && layout.kind == GridKind::StaggerY {
                assert!(layout.origin(0, 0).x > 0.0, "{orientation:?}");
            } else {
                assert_eq!(layout.origin(0, 0), DVec2::ZERO, "{orientation:?}");
            }
        }
    }

    #[test]
    fn outlines_tile_the_plane_and_hold_their_centres() {
        for (size, orientation) in orientations() {
            let set = set(size, orientation);
            let layout = GridLayout::new(&set);
            let n = layout.outline_half_texels().len();
            let expected = match orientation {
                TileOrientation::Hexagonal { .. } => 6,
                _ => 4,
            };
            assert_eq!(n, expected, "{orientation:?} {size:?}");
            // Counter-clockwise: positive area.
            let pts = layout.outline(0, 0);
            let area: f64 = (0..n).map(|i| pts[i].perp_dot(pts[(i + 1) % n])).sum();
            assert!(area > 0.0, "{orientation:?} winds clockwise");
            for row in -6..=6 {
                for col in -6..=6 {
                    let centre = layout.centre(col, row);
                    assert_eq!(
                        set.cell_at(centre),
                        Some([col, row]),
                        "{orientation:?} {size:?} centre of ({col}, {row})"
                    );
                    // Each side's neighbour is another cell, and its side
                    // runs back the other way.
                    for (n, a, b) in layout.sides(col as i64, row as i64) {
                        assert_ne!(n, [col as i64, row as i64]);
                        let back = layout.sides(n[0], n[1]);
                        assert!(
                            back.iter().any(|&(m, c, d)| m == [col as i64, row as i64]
                                && c == b
                                && d == a),
                            "{orientation:?} {size:?} ({col}, {row}) side to {n:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn corner_cells_round_trip() {
        for (size, orientation) in orientations() {
            let set = set(size, orientation);
            let m = super::super::MAX_TILE_CELL;
            let n = super::super::MIN_TILE_CELL;
            for [col, row] in [[n, n], [m, n], [n, m], [m, m], [0, 0], [-1, -1]] {
                let p = set.cell_centre(col, row);
                assert_eq!(
                    set.cell_at(p),
                    Some([col, row]),
                    "{orientation:?} ({col}, {row})"
                );
            }
        }
    }

    #[test]
    fn isometric_neighbours_step_half_a_cell() {
        let set = set([32, 16], TileOrientation::Isometric);
        assert_eq!(set.cell_origin(1, 0), DVec2::new(0.5, -0.5));
        assert_eq!(set.cell_origin(0, 1), DVec2::new(0.5, 0.5));
        assert_eq!(
            set.cell_outline(0, 0),
            vec![
                DVec2::new(0.5, 0.0),
                DVec2::new(1.0, 0.5),
                DVec2::new(0.5, 1.0),
                DVec2::new(0.0, 0.5)
            ]
        );
    }

    #[test]
    fn pointy_hexagons_stagger_their_odd_rows() {
        let set = set(
            [28, 32],
            TileOrientation::Hexagonal {
                side_length: 16,
                axis: StaggerAxis::Y,
                index: StaggerIndex::Odd,
            },
        );
        // Rows step 24 of 32 texels; odd rows move half a cell right.
        assert_eq!(set.cell_origin(0, 1), DVec2::new(0.5, 0.75));
        assert_eq!(set.cell_origin(0, -1), DVec2::new(0.5, -0.75));
        assert_eq!(set.cell_origin(1, 2), DVec2::new(1.0, 1.5));
    }
}
