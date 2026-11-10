//! 2D lighting as an [`ItemTypePlugin`]: the scene's point lights projected
//! onto a world plane, hard shadows cast by occluders, and a light layer
//! multiplied over the scene.
//!
//! Submit one [`Lighting2dItem`] a frame. It names the plane (the same two
//! planes as the tilemap and `Camera2d`), the ambient light, how many lights
//! to draw and the occluders: rects, closed outlines and free segments in
//! plane coordinates, built by hand or with [`occluders_from_tilemap`] and
//! [`occluders_from_sprites`]. The lights are the frame's point lights
//! (`EffectsFrame::lighting.lights`), each projected onto the plane with its
//! `range` as the radius and its colour times its intensity as the brightness
//! at its centre.
//!
//! The light layer is a target the size of the scene, cleared to the ambient
//! colour. Each light draws its occluders' shadows into the stencil and then
//! adds its falloff where its shadow is not, so a light costs two draws and
//! no extra pass. A full-screen pass then multiplies the layer over the scene
//! colour. Both passes run after the scene and its transparency, before bloom,
//! tone mapping and the rest of the post chain, so a lit highlight still
//! blooms; overlays draw later and are not lit. With no point lights the
//! layer is skipped and the scene draws as it would without it.
//!
//! Shadows are geometry: each occluder edge is drawn as the quad it casts,
//! its ends pushed away from the light past the light's reach. That needs one
//! instanced draw per light over a vertex stream of edges, with no storage
//! buffers and no render target per light; a polar shadow map would need a 1D
//! target per light and a pass to fill each. An outline's edges cast only when
//! they face away from the light, so a wall's lit face is lit; a free segment
//! casts from both sides.
//!
//! The layer runs on the HDR path only, so it registers on the full tier: the
//! direct (LDR) path, the only one below it, has no point after the scene to
//! multiply it in.
//!
//! The data types here build without the `gpu` feature; the plugin needs it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

use glam::Vec2;
use viewport_lib_types::colour::Colour;
use viewport_lib_types::material::ItemSettings;

use super::tilemap::{Tile, Tilemap, TilemapPlane};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.lighting_2d";

/// Most lights one frame draws: each takes a stencil value.
pub const MAX_LIGHTS_2D: u32 = 255;

#[cfg(feature = "gpu")]
mod plugin;

#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub use plugin::{Lighting2dPlugin, Lighting2dStats};

/// One frame's 2D lighting: the plane, the ambient light, the light cap and
/// the occluders.
///
/// The frame draws one light layer. When several items are submitted their
/// occluders are combined and the first visible item's other fields are used.
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
    /// Most lights drawn, in the order the frame lists them; the rest are
    /// left out. Clamped to [`MAX_LIGHTS_2D`]. Default 32.
    pub max_lights: u32,
    /// Whether occluders cast shadows. Default `true`.
    pub shadows: bool,
    /// What casts shadows, in plane coordinates.
    pub occluders: Vec<Occluder2d>,
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
            settings: ItemSettings::default(),
        }
    }
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

/// Rects covering the cells of `map` for which `solid` holds, for a layer drawn
/// with its cell (0, 0) corner at plane position `origin` and cells
/// `cell_size` across. Runs of solid cells along a row merge into one rect, so
/// a wall casts from its outline rather than from every tile edge.
pub fn occluders_from_tilemap(
    map: &Tilemap,
    origin: [f32; 2],
    cell_size: [f32; 2],
    solid: impl Fn(Tile) -> bool,
) -> Vec<Occluder2d> {
    let [cols, rows] = map.size();
    let mut out = Vec::new();
    for row in 0..rows {
        let mut col = 0;
        while col < cols {
            if !map.get(col, row).is_some_and(&solid) {
                col += 1;
                continue;
            }
            let start = col;
            while col < cols && map.get(col, row).is_some_and(&solid) {
                col += 1;
            }
            let y0 = origin[1] + row as f32 * cell_size[1];
            out.push(Occluder2d::Rect {
                min: [origin[0] + start as f32 * cell_size[0], y0],
                max: [origin[0] + col as f32 * cell_size[0], y0 + cell_size[1]],
            });
        }
    }
    out
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

    #[test]
    fn tilemap_rows_merge_into_runs() {
        let mut map = Tilemap::new(
            super::super::tilemap::Tileset::new(None, [16, 16], 4),
            [4, 2],
        );
        for col in [0, 1, 3] {
            map.set(col, 0, Tile::new(1));
        }
        let rects = occluders_from_tilemap(&map, [0.0, 0.0], [16.0, 16.0], |t| t.index == 1);
        assert_eq!(
            rects,
            vec![
                Occluder2d::Rect {
                    min: [0.0, 0.0],
                    max: [32.0, 16.0]
                },
                Occluder2d::Rect {
                    min: [48.0, 0.0],
                    max: [64.0, 16.0]
                },
            ]
        );
    }
}
