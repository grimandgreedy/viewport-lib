//! 2D lighting as an [`ItemTypePlugin`]: lights on a world plane, hard
//! shadows cast by occluders, and light layers the lit content samples.
//!
//! Submit a [`Lighting2dItem`] a frame. It names the plane (the same two
//! planes as the tilemap and `Camera2d`), the ambient light, the lights and
//! the occluders: rects, closed outlines, one-sided edges and free segments in
//! plane coordinates, built by hand or with [`occluders_from_tilemap`] and
//! [`occluders_from_sprites`]. The lights are the item's own
//! [`Light2d`]s (points, cones, directional and global lights, textured
//! and freeform shapes) and, unless [`Lighting2dItem::use_frame_lights`] is
//! off, the frame's point and spot lights (`EffectsFrame::lighting.lights`)
//! projected onto the plane.
//!
//! Each item lights a range of painter layers ([`Lighting2dItem::layers`],
//! every layer by default) and accumulates its lights into a light layer of
//! its own: a target the size of the scene, cleared to the ambient colour.
//! Each light draws its occluders' shadows into the stencil and then adds its
//! light where its shadow is not, so a light costs two draws; this runs before
//! the frame's passes. Lit content multiplies its colour by the light under
//! it in the layer of the range its own painter layer lies in, so a range is
//! lit by its own layer only: painter sprites, blended tile layers, GPU
//! particles, painter-placed meshes and mesh instances, and world overlay
//! layers. Sprites, cutout tile layers and particles drawn outside the
//! painter sequence are lit as painter layer 0; other 3D content is not lit.
//! Content marked unlit (a [`PainterBatch`] with `unlit`, a sprite with
//! [`unlit`](super::sprite::SpriteItem::unlit), a placement or world overlay
//! layer with `unlit`) does not sample, draws after the lit content of its
//! range and keeps its colour. Bloom and the rest of the post chain run
//! after the scene, so a lit highlight still blooms; screen overlays draw
//! later and are not lit.
//!
//! A light whose [`layers`](Light2d::layers) cover only part of the item's
//! range splits the range: each part where a different set of lights applies
//! gets a light layer of its own, holding the ambient and those lights. The
//! renderer binds at most
//! [`MAX_PAINTER_LIGHT_RANGES`](viewport_lib::resources::MAX_PAINTER_LIGHT_RANGES)
//! parts a frame; content in the rest is not lit.
//!
//! With [`Lighting2dItem::normals`] on, each light layer gets a second array
//! layer holding the direction its light arrives from, weighted by how much
//! each light adds, and a sprite with a
//! [`normal_texture_id`](super::sprite::SpriteItem::normal_texture_id) or a
//! tile layer whose tileset has a
//! [`normal_atlas`](super::tilemap::Tileset::normal_atlas) shades the light
//! by its normal against that direction, using each light's
//! [`height`](Light2d::height). Ambient, global and polygon light stays flat.
//! Where several lights overlap they shade as one light from their weighted
//! mean direction, which is exact while the normal faces all of them.
//!
//! A light with a [`shadow_softness`](Light2d::shadow_softness) draws its
//! shadows into a mask, blurs it over that many device pixels and adds its
//! light through the blurred mask instead of the stencil, so its shadow edges
//! fade; that costs three extra passes for the light.
//!
//! The layers run whenever an item is submitted, lights or not, so the
//! ambient light darkens lit content with every light off. The background is
//! never lit.
//!
//! Shadows are geometry: each occluder edge is drawn as the quad it casts,
//! its ends pushed away from the light past the light's reach. That needs one
//! instanced draw per light over a vertex stream of edges, with no storage
//! buffers and no render target per light; a polar shadow map would need a 1D
//! target per light and a pass to fill each. An outline's edges cast only when
//! they face away from the light, so a wall's lit face is lit; a free segment
//! casts from both sides.
//!
//! The data types here build without the `gpu` feature; the plugin needs it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin
//! [`PainterBatch`]: viewport_lib::plugin_api::PainterBatch

use std::ops::RangeInclusive;

use glam::Vec2;
use viewport_lib_types::colour::Colour;
use viewport_lib_types::ids::TextureId;
use viewport_lib_types::material::ItemSettings;

use super::tilemap::{Tile, Tilemap, TilemapPlane};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.lighting_2d";

/// Most lights one light layer draws: each takes a stencil value.
pub const MAX_LIGHTS_2D: u32 = 255;

/// Most outline points a [`Light2dKind::Polygon`] light keeps; the rest are
/// dropped.
pub const MAX_POLYGON_LIGHT_POINTS: usize = 64;

#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub use plugin::{Lighting2dPlugin, Lighting2dStats};

/// One frame's 2D lighting over a range of painter layers: the plane, the
/// ambient light, the lights and the occluders.
///
/// Submit one item per range to light background, world and foreground
/// differently. Items whose [`layers`](Self::layers) overlap raise
/// `NoticeDetail::OverlappingLightRanges` and the later one is ignored.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Lighting2dItem {
    /// The plane the lights are projected onto and the occluders lie in.
    /// Default [`TilemapPlane::Xy`].
    pub plane: TilemapPlane,
    /// The plane's position along its normal. Default `0.0`.
    pub elevation: f32,
    /// Light that reaches everything, shadowed or not. The layer is cleared
    /// to it, so `0.2` grey keeps unlit areas at a fifth of their colour.
    /// Default 0.2 grey.
    pub ambient: Colour,
    /// Multiplier on every light's brightness. Default `1.0`.
    pub intensity: f32,
    /// Most lights drawn: the item's [`lights`](Self::lights) first, then the
    /// frame's, in order; the rest are left out. Clamped to
    /// [`MAX_LIGHTS_2D`]. Default 32.
    pub max_lights: u32,
    /// Whether occluders cast shadows. Default `true`.
    pub shadows: bool,
    /// What casts shadows, in plane coordinates.
    pub occluders: Vec<Occluder2d>,
    /// The item's own lights, in plane coordinates. Default empty.
    pub lights: Vec<Light2d>,
    /// Whether the frame's point and spot lights (`EffectsFrame::lighting`)
    /// join [`lights`](Self::lights), projected onto the plane as
    /// [`Light2dKind::Point`] and [`Light2dKind::Cone`] lights over every
    /// layer, with their `range` as the radius and their colour times their
    /// intensity as the brightness at the centre. Default `true`.
    pub use_frame_lights: bool,
    /// The painter layers this item lights: lit content on these layers
    /// samples this item's light layer. Default every layer.
    pub layers: RangeInclusive<u32>,
    /// Whether the layers also carry the direction each light arrives from,
    /// so sprites and tiles with a normal map are shaded by it. Content
    /// without one is lit flat either way. Needs float light layers: where
    /// the device has no HDR path the layers stay flat and a notice says so.
    /// Default `false`.
    pub normals: bool,
    /// Visibility. A hidden item draws no layer.
    pub settings: ItemSettings,
}

impl Default for Lighting2dItem {
    fn default() -> Self {
        Self {
            plane: TilemapPlane::Xy,
            elevation: 0.0,
            ambient: Colour::linear_rgb(0.2, 0.2, 0.2),
            intensity: 1.0,
            max_lights: 32,
            shadows: true,
            occluders: Vec::new(),
            lights: Vec::new(),
            use_frame_lights: true,
            layers: 0..=u32::MAX,
            normals: false,
            settings: ItemSettings::default(),
        }
    }
}

/// A light on the 2D plane, as [`Lighting2dItem::lights`] holds it.
///
/// `position` and `radius` place and size the kinds that need them; each kind
/// says which it reads. Shadows are cast from `position`, except by a
/// directional light, whose shadows are parallel, and a global light, which
/// casts none.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Light2d {
    /// What the light lights.
    pub kind: Light2dKind,
    /// Plane position of the light.
    pub position: [f32; 2],
    /// Reach of a point or cone light, where its falloff reaches zero; the
    /// width of a polygon light's fade outside its outline.
    pub radius: f32,
    /// Light colour. Default white.
    pub colour: Colour,
    /// Multiplier on the colour. Default `1.0`.
    pub intensity: f32,
    /// How a point, cone or polygon light fades. Default [`Falloff::Smooth`].
    pub falloff: Falloff,
    /// Height of the light above the plane, in plane units. Content with a
    /// normal map under [`Lighting2dItem::normals`] is shaded by the
    /// direction from it to the light raised by this height, so at `0.0` a
    /// flat-facing normal meets the light edge-on and stays dark; content lit
    /// flat ignores it. For a directional light it is the rise per plane unit
    /// of travel. Default `0.0`.
    pub height: f32,
    /// Width of the soft edge of the light's shadows, in device pixels: its
    /// shadows are blurred over this radius. `0.0` keeps hard shadows and
    /// costs nothing extra. Default `0.0`.
    pub shadow_softness: f32,
    /// The painter layers the light lights, within its item's range. Default
    /// every layer.
    pub layers: RangeInclusive<u32>,
}

impl Light2d {
    fn with_kind(kind: Light2dKind, position: [f32; 2], radius: f32) -> Self {
        Self {
            kind,
            position,
            radius,
            colour: Colour::WHITE,
            intensity: 1.0,
            falloff: Falloff::Smooth,
            height: 0.0,
            shadow_softness: 0.0,
            layers: 0..=u32::MAX,
        }
    }

    /// A point light at `position` reaching `radius`.
    pub fn point(position: [f32; 2], radius: f32) -> Self {
        Self::with_kind(Light2dKind::Point, position, radius)
    }

    /// A cone light at `position` reaching `radius`, pointing along
    /// `direction`, full within `inner` and fading to nothing at `outer`
    /// (half-angles in radians).
    pub fn cone(
        position: [f32; 2],
        radius: f32,
        direction: [f32; 2],
        inner: f32,
        outer: f32,
    ) -> Self {
        Self::with_kind(
            Light2dKind::Cone {
                direction,
                inner,
                outer,
            },
            position,
            radius,
        )
    }

    /// A directional light travelling along `direction` over the whole plane.
    pub fn directional(direction: [f32; 2]) -> Self {
        Self::with_kind(Light2dKind::Directional { direction }, [0.0; 2], 0.0)
    }

    /// A global light over the whole plane, casting no shadows.
    pub fn global() -> Self {
        Self::with_kind(Light2dKind::Global, [0.0; 2], 0.0)
    }

    /// `texture` as the light's shape, centred on `position`, `size` plane
    /// units across and turned `rotation` radians counter-clockwise.
    pub fn textured(texture: TextureId, position: [f32; 2], size: [f32; 2], rotation: f32) -> Self {
        Self::with_kind(
            Light2dKind::Textured {
                texture,
                size,
                rotation,
            },
            position,
            0.0,
        )
    }

    /// A freeform light filling the outline `points` (relative to
    /// `position`), fading out over `radius` outside it.
    pub fn polygon(position: [f32; 2], points: Vec<[f32; 2]>, radius: f32) -> Self {
        Self::with_kind(Light2dKind::Polygon { points }, position, radius)
    }

    /// This light with `colour`.
    pub fn with_colour(mut self, colour: Colour) -> Self {
        self.colour = colour;
        self
    }

    /// This light with `intensity`.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// This light with `falloff`.
    pub fn with_falloff(mut self, falloff: Falloff) -> Self {
        self.falloff = falloff;
        self
    }

    /// This light at `height` above the plane.
    pub fn with_height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// This light with shadows blurred over `softness` device pixels.
    pub fn with_shadow_softness(mut self, softness: f32) -> Self {
        self.shadow_softness = softness;
        self
    }

    /// This light lighting only the painter layers `layers`.
    pub fn with_layers(mut self, layers: RangeInclusive<u32>) -> Self {
        self.layers = layers;
        self
    }
}

/// What a [`Light2d`] lights.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Light2dKind {
    /// A disc around `position`, fading to nothing at `radius`.
    Point,
    /// The part of a point light's disc around `direction`: full within
    /// `inner` of it, fading to nothing at `outer`. Shadows fall only where
    /// the cone lights.
    Cone {
        /// The way the cone points, in plane coordinates. Need not be unit
        /// length.
        direction: [f32; 2],
        /// Half-angle of the full-brightness cone, in radians.
        inner: f32,
        /// Half-angle where the light reaches zero, in radians.
        outer: f32,
    },
    /// Light over the whole plane from far away, travelling along
    /// `direction`: occluders cast parallel shadows away from it, and there
    /// is no falloff. Reads neither `position` nor `radius`.
    Directional {
        /// The way the light travels, in plane coordinates.
        direction: [f32; 2],
    },
    /// Light over the whole plane, with no shadows and no falloff. Reads
    /// neither `position` nor `radius`.
    Global,
    /// A texture as the light's shape: centred on `position`, its colour
    /// times its alpha scales the light, with texture row 0 at the top. No
    /// falloff; `radius` is unused.
    Textured {
        /// The shape.
        texture: TextureId,
        /// Width and height on the plane.
        size: [f32; 2],
        /// Counter-clockwise turn, in radians.
        rotation: f32,
    },
    /// A freeform shape: full brightness inside the outline, fading by the
    /// falloff over `radius` outside it (a hard edge at `0.0`). At most
    /// [`MAX_POLYGON_LIGHT_POINTS`] points are kept.
    Polygon {
        /// The outline, relative to `position`, in either winding.
        points: Vec<[f32; 2]>,
    },
}

/// How a light fades from its centre (or a polygon light from its outline)
/// to nothing at its radius. `d` below is the distance over the radius.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Falloff {
    /// `1 - d`.
    Linear,
    /// `(1 - d^2)^2`: flat near the centre, easing out at the edge.
    #[default]
    Smooth,
    /// An inverse-square curve, `1 / (1 + 16 d^2)`, shifted and scaled to
    /// reach zero at the radius: bright at the centre with a long tail.
    InverseSquare,
    /// `(1 - d)^e` for the given `e`.
    Exponent(f32),
}

/// The parts of `range` where a different set of lights applies: each part
/// with the indices of the lights, of those whose layers are in
/// `light_layers`, that cover it. Adjacent parts lit by the same lights are
/// one part. A light whose layers miss `range` lights no part.
pub fn light_parts(
    range: &RangeInclusive<u32>,
    light_layers: &[RangeInclusive<u32>],
) -> Vec<(RangeInclusive<u32>, Vec<usize>)> {
    let (lo, hi) = (*range.start(), *range.end());
    if lo > hi {
        return Vec::new();
    }
    // Every layer where the set of lights can change, as a u64 so the end of
    // the last layer fits.
    let mut cuts: Vec<u64> = vec![lo as u64, hi as u64 + 1];
    for l in light_layers {
        let (a, b) = (*l.start().max(&lo), *l.end().min(&hi));
        if a <= b {
            cuts.push(a as u64);
            cuts.push(b as u64 + 1);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    let mut parts: Vec<(RangeInclusive<u32>, Vec<usize>)> = Vec::new();
    for w in cuts.windows(2) {
        let (first, last) = (w[0] as u32, (w[1] - 1) as u32);
        let lit: Vec<usize> = light_layers
            .iter()
            .enumerate()
            .filter(|(_, l)| *l.start() <= first && last <= *l.end())
            .map(|(i, _)| i)
            .collect();
        match parts.last_mut() {
            Some((r, set)) if *set == lit => *r = *r.start()..=last,
            _ => parts.push((first..=last, lit)),
        }
    }
    parts
}

/// Something that casts a 2D shadow, in plane coordinates.
#[derive(Clone, Debug, PartialEq)]
pub enum Occluder2d {
    /// An axis-aligned rectangle.
    Rect {
        /// Lower-left corner.
        min: [f32; 2],
        /// Upper-right corner.
        max: [f32; 2],
    },
    /// A closed outline, in either winding. Its lit side stays lit.
    Polygon(Vec<[f32; 2]>),
    /// A segment that casts from both sides, such as a fence.
    Segment {
        /// One end.
        a: [f32; 2],
        /// The other end.
        b: [f32; 2],
    },
    /// Boundary edges of solid regions, each casting from one side only.
    /// They need not join into closed outlines, so a region with holes, or
    /// many regions, can be one occluder. [`occluders_from_tilemap`] builds
    /// one.
    Edges(Vec<OccluderEdge>),
}

/// One edge of an [`Occluder2d::Edges`] occluder. The solid lies to the left
/// of `a -> b` and its outside to the right, so the winding gives the outward
/// normal: an outer boundary runs counter-clockwise and the edges around a
/// hole run clockwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OccluderEdge {
    /// First end.
    pub a: [f32; 2],
    /// Second end.
    pub b: [f32; 2],
}

/// One occluder edge as the shadow pass draws it: the two ends, and whether
/// it casts only when facing away from the light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowEdge {
    /// First end.
    pub a: Vec2,
    /// Second end. For a one-sided edge the occluder lies to the left of
    /// `a -> b`.
    pub b: Vec2,
    /// Whether the edge belongs to a closed outline.
    pub one_sided: bool,
}

impl Occluder2d {
    /// Append this occluder's edges to `out`, outlines wound counter-clockwise.
    pub fn edges(&self, out: &mut Vec<ShadowEdge>) {
        match self {
            Occluder2d::Rect { min, max } => {
                let (x0, y0, x1, y1) = (min[0], min[1], max[0], max[1]);
                outline_edges(
                    &[[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(Vec2::from),
                    out,
                );
            }
            Occluder2d::Polygon(points) => {
                let points: Vec<Vec2> = points.iter().map(|p| Vec2::from(*p)).collect();
                outline_edges(&points, out);
            }
            Occluder2d::Segment { a, b } => out.push(ShadowEdge {
                a: Vec2::from(*a),
                b: Vec2::from(*b),
                one_sided: false,
            }),
            Occluder2d::Edges(edges) => out.extend(edges.iter().map(|e| ShadowEdge {
                a: Vec2::from(e.a),
                b: Vec2::from(e.b),
                one_sided: true,
            })),
        }
    }
}

/// The edges of a closed outline, turned counter-clockwise when given
/// clockwise so every edge's outside is on its right.
fn outline_edges(points: &[Vec2], out: &mut Vec<ShadowEdge>) {
    let n = points.len();
    if n < 3 {
        return;
    }
    let area: f32 = (0..n)
        .map(|i| points[i].perp_dot(points[(i + 1) % n]))
        .sum();
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        let (a, b) = if area < 0.0 { (b, a) } else { (a, b) };
        out.push(ShadowEdge {
            a,
            b,
            one_sided: true,
        });
    }
}

/// The quad edge `a`-`b` casts from a light at `light` reaching `radius`, as
/// `[a, b, b', a']`: `a'` and `b'` are the ends pushed along the rays from the
/// light far enough that the side between them is at least `radius` from the
/// light, so the quad holds the whole shadow inside the light's reach. `None`
/// when the light sits on an end.
///
/// The shadow shader builds the same quad.
pub fn shadow_quad(light: Vec2, radius: f32, a: Vec2, b: Vec2) -> Option<[Vec2; 4]> {
    let (da, db) = (a - light, b - light);
    let (la, lb) = (da.length(), db.length());
    if la < 1e-6 || lb < 1e-6 {
        return None;
    }
    let (ua, ub) = (da / la, db / lb);
    let cos_half = ((1.0 + ua.dot(ub)) * 0.5).max(0.0).sqrt();
    let far = (radius / cos_half.max(1e-3)).max(la.max(lb));
    Some([a, b, light + ub * far, light + ua * far])
}

/// Whether `edge` casts a shadow from `light`: a one-sided edge only when it
/// faces away from it.
pub fn edge_casts(edge: &ShadowEdge, light: Vec2) -> bool {
    let outward = Vec2::new(edge.b.y - edge.a.y, edge.a.x - edge.b.x);
    !edge.one_sided || outward.dot(light - edge.a) <= 0.0
}

/// Whether `p` lies in the shadow any of `edges` casts from a light at
/// `light` reaching `radius`: the test the stencil makes, on the CPU.
pub fn in_shadow(light: Vec2, radius: f32, edges: &[ShadowEdge], p: Vec2) -> bool {
    edges.iter().any(|e| {
        edge_casts(e, light)
            && shadow_quad(light, radius, e.a, e.b)
                .is_some_and(|[a, b, b2, a2]| in_triangle(p, a, b, b2) || in_triangle(p, a, b2, a2))
    })
}

fn in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = (b - a).perp_dot(p - a);
    let d2 = (c - b).perp_dot(p - b);
    let d3 = (a - c).perp_dot(p - c);
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

/// The boundary of the cells of `map` for which `solid` holds, for a layer
/// drawn with its cell (0, 0) corner at plane position `origin` and cells
/// `cell_size` across, as one [`Occluder2d::Edges`] (none when no cell is
/// solid).
///
/// A cell side is an edge only where a solid cell meets a cell that is not
/// solid or that no chunk holds, facing out of the solid, and consecutive
/// sides along one line merge into one edge. A wall several cells thick
/// therefore casts from its outline alone, with no edges inside it to shadow
/// its own face, and the inside of a hole casts outward into the hole. The
/// sides are those of the cell outlines the map's
/// [`TileOrientation`](super::tilemap::TileOrientation) gives: squares,
/// diamonds or hexagons.
pub fn occluders_from_tilemap(
    map: &Tilemap,
    origin: [f32; 2],
    cell_size: [f32; 2],
    solid: impl Fn(Tile) -> bool,
) -> Vec<Occluder2d> {
    if map.tileset.orientation != super::tilemap::TileOrientation::Orthogonal {
        return grid_occluders(map, origin, cell_size, solid);
    }
    let Some([col0, row0, col1, row1]) = map.cell_bounds() else {
        return Vec::new();
    };
    let (cols, rows) = ((col1 - col0) as u32, (row1 - row0) as u32);
    // Lines are numbered from the bounds' bottom-left corner.
    let is_solid = |col: i64, row: i64| {
        map.get((col0 as i64 + col) as i32, (row0 as i64 + row) as i32)
            .is_some_and(|t| solid(t))
    };
    let x = |col: u32| origin[0] + (col0 as i64 + col as i64) as f32 * cell_size[0];
    let y = |row: u32| origin[1] + (row0 as i64 + row as i64) as f32 * cell_size[1];
    let mut edges = Vec::new();
    // Runs of cell sides along one line: `side(i)` is `Some(true)` where the
    // solid is on the line's positive side, `Some(false)` where it is on the
    // negative side, `None` where there is no edge. `emit` gets the run's
    // start, end and which side the solid is on.
    fn runs(len: u32, side: impl Fn(u32) -> Option<bool>, mut emit: impl FnMut(u32, u32, bool)) {
        let mut i = 0;
        while i < len {
            let Some(s) = side(i) else {
                i += 1;
                continue;
            };
            let start = i;
            while i < len && side(i) == Some(s) {
                i += 1;
            }
            emit(start, i, s);
        }
    }
    let face = |below: bool, above: bool| (below != above).then_some(above);
    // Horizontal lines between rows. With the solid above, the edge runs
    // towards +x; with it below, towards -x.
    for row in 0..=rows {
        let side = |col: u32| {
            face(
                is_solid(col as i64, row as i64 - 1),
                is_solid(col as i64, row as i64),
            )
        };
        runs(cols, side, |c0, c1, above| {
            let (l, r) = ([x(c0), y(row)], [x(c1), y(row)]);
            let (a, b) = if above { (l, r) } else { (r, l) };
            edges.push(OccluderEdge { a, b });
        });
    }
    // Vertical lines between columns. With the solid to the right, the edge
    // runs towards -y; with it to the left, towards +y.
    for col in 0..=cols {
        let side = |row: u32| {
            face(
                is_solid(col as i64 - 1, row as i64),
                is_solid(col as i64, row as i64),
            )
        };
        runs(rows, side, |r0, r1, right| {
            let (lo, hi) = ([x(col), y(r0)], [x(col), y(r1)]);
            let (a, b) = if right { (hi, lo) } else { (lo, hi) };
            edges.push(OccluderEdge { a, b });
        });
    }
    if edges.is_empty() {
        Vec::new()
    } else {
        vec![Occluder2d::Edges(edges)]
    }
}

/// [`occluders_from_tilemap`] for a grid of diamonds or hexagons: every
/// outline side of a solid cell whose neighbour across it is not solid,
/// joined where sides continue along one line.
fn grid_occluders(
    map: &Tilemap,
    origin: [f32; 2],
    cell_size: [f32; 2],
    solid: impl Fn(Tile) -> bool,
) -> Vec<Occluder2d> {
    use glam::I64Vec2;
    let layout = super::tilemap::GridLayout::new(&map.tileset);
    let is_solid = |[col, row]: [i64; 2]| {
        let range = super::tilemap::MIN_TILE_CELL as i64..=super::tilemap::MAX_TILE_CELL as i64;
        range.contains(&col)
            && range.contains(&row)
            && map.get(col as i32, row as i32).is_some_and(&solid)
    };
    // Sides in half-texels, so the ends two sides share compare exactly.
    let mut sides: Vec<(I64Vec2, I64Vec2)> = Vec::new();
    for (coord, chunk) in map.chunks() {
        for (i, tile) in chunk.tiles.iter().enumerate() {
            if tile.is_empty() || !solid(*tile) {
                continue;
            }
            let Some([col, row]) = map.cell_of(coord, i as u32) else {
                continue;
            };
            for (neighbour, a, b) in layout.sides(col as i64, row as i64) {
                if !is_solid(neighbour) {
                    sides.push((a, b));
                }
            }
        }
    }
    // Join each side to the one that carries on from its end in the same
    // direction, walking from the sides nothing leads into.
    let direction = |(a, b): (I64Vec2, I64Vec2)| {
        let d = b - a;
        let g = gcd(d.x.unsigned_abs(), d.y.unsigned_abs()).max(1) as i64;
        d / g
    };
    let mut from: std::collections::HashMap<(I64Vec2, I64Vec2), usize> = Default::default();
    for (i, &s) in sides.iter().enumerate() {
        from.insert((s.0, direction(s)), i);
    }
    let ends: std::collections::HashSet<(I64Vec2, I64Vec2)> =
        sides.iter().map(|&s| (s.1, direction(s))).collect();
    let leads_in = |s: (I64Vec2, I64Vec2)| ends.contains(&(s.0, direction(s)));
    let to_plane = |p: I64Vec2| {
        let q = layout.to_cells(p);
        [
            origin[0] + (q.x * cell_size[0] as f64) as f32,
            origin[1] + (q.y * cell_size[1] as f64) as f32,
        ]
    };
    let mut used = vec![false; sides.len()];
    let mut edges = Vec::new();
    let mut walk = |start: usize, used: &mut Vec<bool>| {
        let d = direction(sides[start]);
        let a = sides[start].0;
        let mut end = start;
        used[start] = true;
        while let Some(&next) = from.get(&(sides[end].1, d)) {
            if used[next] {
                break;
            }
            used[next] = true;
            end = next;
        }
        edges.push(OccluderEdge {
            a: to_plane(a),
            b: to_plane(sides[end].1),
        });
    };
    for i in 0..sides.len() {
        if !used[i] && !leads_in(sides[i]) {
            walk(i, &mut used);
        }
    }
    // Lines that close on themselves have no first side.
    for i in 0..sides.len() {
        if !used[i] {
            walk(i, &mut used);
        }
    }
    if edges.is_empty() {
        Vec::new()
    } else {
        vec![Occluder2d::Edges(edges)]
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// One outline per sprite of a world-space sprite batch drawn on `plane`
/// under a camera looking along the plane's normal: each sprite's quad with
/// its size, pivot and rotation. Screen-space batches give none, as their size
/// on the plane depends on the camera.
pub fn occluders_from_sprites(
    item: &super::sprite::SpriteItem,
    plane: TilemapPlane,
) -> Vec<Occluder2d> {
    if item.size_mode != super::sprite::SpriteSizeMode::WorldSpace {
        return Vec::new();
    }
    let model = glam::Mat4::from_cols_array_2d(&item.model);
    let (x_axis, y_axis) = (plane.x_axis(), plane.y_axis());
    item.positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let w = model.transform_point3(glam::Vec3::from(*p));
            let centre = Vec2::new(w.dot(x_axis), w.dot(y_axis));
            let size = Vec2::from(item.sizes.get(i).copied().unwrap_or([item.default_size; 2]));
            let pivot = Vec2::from(item.pivots.get(i).copied().unwrap_or(item.default_pivot));
            let rotation = item.rotations.get(i).copied().unwrap_or(0.0);
            let offset = pivot * 2.0 - Vec2::ONE;
            let (s, c) = rotation.sin_cos();
            let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]].map(|k| {
                let q = (Vec2::from(k) - offset) * size * 0.5;
                let r = Vec2::new(c * q.x - s * q.y, s * q.x + c * q.y);
                (centre + r).to_array()
            });
            Occluder2d::Polygon(corners.to_vec())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lights_over_every_layer_make_one_part() {
        let parts = light_parts(&(0..=u32::MAX), &[0..=u32::MAX, 0..=u32::MAX]);
        assert_eq!(parts, vec![(0..=u32::MAX, vec![0, 1])]);
        assert_eq!(light_parts(&(2..=5), &[]), vec![(2..=5, vec![])]);
    }

    #[test]
    fn a_restricted_light_splits_the_range_where_it_starts_and_ends() {
        let parts = light_parts(&(0..=u32::MAX), &[0..=u32::MAX, 1..=2]);
        assert_eq!(
            parts,
            vec![
                (0..=0, vec![0]),
                (1..=2, vec![0, 1]),
                (3..=u32::MAX, vec![0]),
            ]
        );
        // Clipped to the item's range; a light outside it lights nothing.
        let parts = light_parts(&(2..=6), &[0..=3, 9..=12]);
        assert_eq!(parts, vec![(2..=3, vec![0]), (4..=6, vec![])]);
        // Two lights over the same layers keep one part.
        let parts = light_parts(&(0..=4), &[1..=2, 1..=2]);
        assert_eq!(
            parts,
            vec![(0..=0, vec![]), (1..=2, vec![0, 1]), (3..=4, vec![])]
        );
    }

    fn edges(occluders: &[Occluder2d]) -> Vec<ShadowEdge> {
        let mut out = Vec::new();
        for o in occluders {
            o.edges(&mut out);
        }
        out
    }

    #[test]
    fn a_point_occluder_shadows_what_is_behind_it_only() {
        let light = Vec2::ZERO;
        // A small square between the light and the sample point.
        let e = edges(&[Occluder2d::Rect {
            min: [4.9, -0.1],
            max: [5.1, 0.1],
        }]);
        assert!(in_shadow(light, 20.0, &e, Vec2::new(10.0, 0.0)));
        // Beside the line from the light, and in front of the occluder.
        assert!(!in_shadow(light, 20.0, &e, Vec2::new(10.0, 3.0)));
        assert!(!in_shadow(light, 20.0, &e, Vec2::new(3.0, 0.0)));
        // Behind the light.
        assert!(!in_shadow(light, 20.0, &e, Vec2::new(-10.0, 0.0)));
    }

    #[test]
    fn a_segment_casts_from_both_sides() {
        let e = edges(&[Occluder2d::Segment {
            a: [5.0, -1.0],
            b: [5.0, 1.0],
        }]);
        assert!(in_shadow(Vec2::ZERO, 20.0, &e, Vec2::new(8.0, 0.0)));
        assert!(in_shadow(
            Vec2::new(10.0, 0.0),
            20.0,
            &e,
            Vec2::new(2.0, 0.0)
        ));
        assert!(!in_shadow(Vec2::ZERO, 20.0, &e, Vec2::new(8.0, 5.0)));
    }

    #[test]
    fn an_outline_keeps_its_lit_face_lit() {
        let wall = [Occluder2d::Rect {
            min: [4.0, -2.0],
            max: [6.0, 2.0],
        }];
        let e = edges(&wall);
        // Inside the wall, near the face towards the light, and behind it.
        assert!(!in_shadow(Vec2::ZERO, 20.0, &e, Vec2::new(4.2, 0.0)));
        assert!(in_shadow(Vec2::ZERO, 20.0, &e, Vec2::new(9.0, 1.0)));
        // The same outline wound the other way casts the same shadow.
        let cw = [Occluder2d::Polygon(vec![
            [4.0, -2.0],
            [4.0, 2.0],
            [6.0, 2.0],
            [6.0, -2.0],
        ])];
        let e2 = edges(&cw);
        assert!(!in_shadow(Vec2::ZERO, 20.0, &e2, Vec2::new(4.2, 0.0)));
        assert!(in_shadow(Vec2::ZERO, 20.0, &e2, Vec2::new(9.0, 1.0)));
    }

    #[test]
    fn the_cast_quad_covers_the_reach_even_for_a_wide_edge() {
        // The light is close to a long wall: the shadow's angle is nearly a
        // half turn, and the far side must still clear the light's reach.
        let e = edges(&[Occluder2d::Segment {
            a: [-50.0, 1.0],
            b: [50.0, 1.0],
        }]);
        for x in [-15.0, -5.0, 0.0, 5.0, 15.0] {
            assert!(in_shadow(Vec2::ZERO, 20.0, &e, Vec2::new(x, 19.0)), "x {x}");
        }
    }

    /// A map of `cols` x `rows` unit cells with `solid` cells set to 1.
    fn map(cols: u32, rows: u32, solid: &[(u32, u32)]) -> Tilemap {
        let mut map = Tilemap::new(
            super::super::tilemap::Tileset::new(None, [16, 16], 4),
            [cols, rows],
        );
        for &(col, row) in solid {
            map.set(col as i32, row as i32, Tile::new(1));
        }
        map
    }

    fn tile_edges(map: &Tilemap) -> Vec<OccluderEdge> {
        match occluders_from_tilemap(map, [0.0, 0.0], [1.0, 1.0], |t| t.index == 1).as_slice() {
            [Occluder2d::Edges(e)] => e.clone(),
            [] => Vec::new(),
            other => panic!("unexpected occluders {other:?}"),
        }
    }

    /// The outward normal of a one-sided edge: to the right of `a -> b`.
    fn outward(e: &OccluderEdge) -> Vec2 {
        let d = Vec2::from(e.b) - Vec2::from(e.a);
        Vec2::new(d.y, -d.x).normalize()
    }

    fn block(c0: u32, r0: u32, c1: u32, r1: u32) -> Vec<(u32, u32)> {
        (c0..c1)
            .flat_map(|c| (r0..r1).map(move |r| (c, r)))
            .collect()
    }

    #[test]
    fn a_solid_block_gives_its_four_sides() {
        let e = tile_edges(&map(5, 5, &block(1, 1, 4, 4)));
        assert_eq!(e.len(), 4);
        for edge in &e {
            // Every edge lies on the block's outline, facing out of it.
            let mid = (Vec2::from(edge.a) + Vec2::from(edge.b)) * 0.5;
            let on_x = mid.x == 1.0 || mid.x == 4.0;
            let on_y = mid.y == 1.0 || mid.y == 4.0;
            assert!(on_x || on_y, "{edge:?} inside the block");
            assert!((Vec2::from(edge.a) - Vec2::from(edge.b)).length() == 3.0);
            let out = mid + outward(edge) * 0.5;
            assert!(
                !(1.0..4.0).contains(&out.x) || !(1.0..4.0).contains(&out.y),
                "{edge:?} faces into the block"
            );
        }
        // The same block against the map's edge is closed by the map's extent.
        assert_eq!(tile_edges(&map(3, 3, &block(0, 0, 3, 3))).len(), 4);
        assert!(tile_edges(&map(3, 3, &[])).is_empty());
    }

    #[test]
    fn an_l_shape_gives_six_edges() {
        let mut cells = block(0, 0, 3, 1);
        cells.extend(block(0, 1, 1, 3));
        assert_eq!(tile_edges(&map(4, 4, &cells)).len(), 6);
    }

    #[test]
    fn a_hole_casts_into_itself() {
        let cells: Vec<_> = block(0, 0, 3, 3)
            .into_iter()
            .filter(|&c| c != (1, 1))
            .collect();
        let e = tile_edges(&map(3, 3, &cells));
        assert_eq!(e.len(), 8);
        let hole_centre = Vec2::splat(1.5);
        let inner: Vec<_> = e
            .iter()
            .filter(|edge| {
                let mid = (Vec2::from(edge.a) + Vec2::from(edge.b)) * 0.5;
                (mid - hole_centre).length() <= 0.5
            })
            .collect();
        assert_eq!(inner.len(), 4);
        for edge in inner {
            let mid = (Vec2::from(edge.a) + Vec2::from(edge.b)) * 0.5;
            assert!(
                outward(edge).dot(hole_centre - mid) > 0.0,
                "{edge:?} faces out of the hole"
            );
        }
    }

    fn oriented(orientation: super::super::tilemap::TileOrientation, size: [u32; 2]) -> Tilemap {
        let mut set = super::super::tilemap::Tileset::new(None, size, 4);
        set.orientation = orientation;
        Tilemap::sparse(set, 4)
    }

    /// Every edge has the solid on its left: a point just to its left is in
    /// a solid cell and one just to its right is not.
    fn check_facing(map: &Tilemap, edges: &[OccluderEdge]) {
        let solid = |p: Vec2| {
            map.tileset
                .cell_at(p.as_dvec2())
                .and_then(|[c, r]| map.get(c, r))
                .is_some_and(|t| t.index == 1)
        };
        for e in edges {
            let (a, b) = (Vec2::from(e.a), Vec2::from(e.b));
            let mid = (a + b) * 0.5;
            let left = (b - a).perp().normalize() * 0.01;
            assert!(solid(mid + left), "{e:?} has no solid on its left");
            assert!(!solid(mid - left), "{e:?} has solid on its right");
        }
    }

    #[test]
    fn diamond_and_hexagon_grids_cast_from_their_cell_sides() {
        use super::super::tilemap::{StaggerAxis, StaggerIndex, TileOrientation};
        // A 3 x 3 block of diamonds is one big diamond: four edges.
        let mut iso = oriented(TileOrientation::Isometric, [32, 16]);
        for (c, r) in block(0, 0, 3, 3) {
            iso.set(c as i32, r as i32, Tile::new(1));
        }
        let e = tile_edges(&iso);
        assert_eq!(e.len(), 4, "{e:?}");
        check_facing(&iso, &e);
        for edge in &e {
            assert!((Vec2::from(edge.a) - Vec2::from(edge.b)).length() > 2.0);
        }

        // One hexagon has six sides; two side by side share one, leaving ten.
        let hex = TileOrientation::Hexagonal {
            side_length: 16,
            axis: StaggerAxis::Y,
            index: StaggerIndex::Odd,
        };
        let mut one = oriented(hex, [28, 32]);
        one.set(2, 2, Tile::new(1));
        let e = tile_edges(&one);
        assert_eq!(e.len(), 6);
        check_facing(&one, &e);
        one.set(3, 2, Tile::new(1));
        let e = tile_edges(&one);
        assert_eq!(e.len(), 10);
        check_facing(&one, &e);

        // A ring of six hexagons round an empty one casts into the hole too.
        for axis in [StaggerAxis::X, StaggerAxis::Y] {
            let hex = TileOrientation::Hexagonal {
                side_length: 14,
                axis,
                index: StaggerIndex::Even,
            };
            let mut ring = oriented(hex, [32, 28]);
            let layout = super::super::tilemap::GridLayout::new(&ring.tileset);
            for (n, _, _) in layout.sides(4, 4) {
                ring.set(n[0] as i32, n[1] as i32, Tile::new(1));
            }
            let e = tile_edges(&ring);
            assert_eq!(e.len(), 6 * 6 - 6 * 2, "{axis:?}");
            check_facing(&ring, &e);
        }

        // Staggered rows of diamonds: a filled 4 x 4 patch has no inner
        // edges, and its outline joins along each diagonal.
        let mut staggered = oriented(
            TileOrientation::Staggered {
                axis: StaggerAxis::Y,
                index: StaggerIndex::Odd,
            },
            [32, 16],
        );
        for (c, r) in block(0, 0, 4, 4) {
            staggered.set(c as i32, r as i32, Tile::new(1));
        }
        let e = tile_edges(&staggered);
        check_facing(&staggered, &e);
        let perimeter: f32 = e
            .iter()
            .map(|e| (Vec2::from(e.a) - Vec2::from(e.b)).length())
            .sum();
        // 16 diamonds of side sqrt(0.5^2 + 0.5^2): 4 x 4 staggered rows show
        // 4 + 4 sides along the top and bottom and 3 + 4 down each end.
        let side = 0.5f32.hypot(0.5);
        assert!((perimeter - 22.0 * side).abs() < 1e-4, "{perimeter}");
    }

    #[test]
    fn a_tall_wall_does_not_shadow_its_own_face() {
        // A wall one cell wide and six tall, lit from above and to its left.
        let cells = block(5, 0, 6, 6);
        let wall = occluders_from_tilemap(&map(10, 8, &cells), [0.0, 0.0], [1.0, 1.0], |t| {
            t.index == 1
        });
        let e = edges(&wall);
        assert_eq!(e.len(), 4);
        let light = Vec2::new(2.0, 10.0);
        // On the wall's face, below its top row.
        let face = Vec2::new(5.5, 4.5);
        assert!(!in_shadow(light, 30.0, &e, face));
        // Beside the wall on the lit side, and behind it.
        assert!(!in_shadow(light, 30.0, &e, Vec2::new(4.0, 3.0)));
        assert!(in_shadow(light, 30.0, &e, Vec2::new(8.0, 3.0)));
        // One rect per row puts edges inside the wall, and the row above's
        // lower edge shadows the face.
        let rows: Vec<_> = (0..6)
            .map(|r| Occluder2d::Rect {
                min: [5.0, r as f32],
                max: [6.0, r as f32 + 1.0],
            })
            .collect();
        assert!(in_shadow(light, 30.0, &edges(&rows), face));
    }
}
