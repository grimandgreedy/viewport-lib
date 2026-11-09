//! Triangle meshes drawn as screen-space overlays.

use crate::colour::Colour;
use crate::overlay::*;

/// The colours an [`OverlayMeshItem`] gives its vertices.
#[derive(Debug, Clone, PartialEq)]
pub enum OverlayMeshColours {
    /// One colour for every vertex.
    Uniform(Colour),
    /// One colour per vertex, interpolated across each triangle. Must have one
    /// entry per position; a list of another length draws the mesh in white.
    PerVertex(Vec<Colour>),
}

impl Default for OverlayMeshColours {
    fn default() -> Self {
        OverlayMeshColours::Uniform(Colour::WHITE)
    }
}

impl OverlayMeshColours {
    /// The colour of vertex `i`, white when a per-vertex list does not match
    /// the position count `len`.
    pub fn at(&self, i: usize, len: usize) -> Colour {
        match self {
            OverlayMeshColours::Uniform(c) => *c,
            OverlayMeshColours::PerVertex(list) if list.len() == len => list[i],
            OverlayMeshColours::PerVertex(_) => Colour::WHITE,
        }
    }
}

/// A coloured triangle list drawn as a screen-space overlay: heatmap cells,
/// contour bands, or anything a consumer tessellates itself.
///
/// The mesh draws through the same pipeline as polylines, labels and vector
/// shapes, so it batches with them. Its edges are drawn exactly as given: there
/// is no anti-aliasing fringe, because the consumer owns the geometry. Cells
/// that share edges therefore meet without a seam, and an outer edge is as
/// sharp as the rasteriser makes it.
///
/// `style.fill` multiplies the vertex colours: a solid fill tints them, a
/// gradient or a texture is evaluated per pixel and multiplied in. The fill
/// spans the mesh's bounding box, or, with `uvs` set, each vertex reads the
/// fill at the point of that box its UV names, so a texture fill samples the
/// image at the vertex UVs. Shadow layers and a backdrop are not drawn (ask
/// [`OverlayStyleSupport::for_mesh`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OverlayMeshItem {
    /// Vertex positions in logical pixels, relative to the resolved anchor
    /// origin plus `transform.translate`. With the default anchor and a zero
    /// translate they are absolute screen coordinates.
    pub positions: Vec<[f32; 2]>,
    /// Triangle list indices into `positions`, three per triangle. A trailing
    /// partial triangle and any triangle naming a missing vertex are skipped.
    pub indices: Vec<u32>,
    /// The vertex colours.
    pub colours: OverlayMeshColours,
    /// Optional texture coordinates, one per position, in `[0, 1]` across the
    /// fill's box. A list of another length is ignored.
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Where the item hangs from and which point of its bounding box lands
    /// there. `transform.translate` is a nudge from the resolved origin.
    pub anchoring: OverlayAnchoring,
    /// Translate, rotate and scale. The linear part turns the mesh about the
    /// centre of its bounding box plus `pivot`.
    pub transform: OverlayTransform,
    /// Fill (a multiplier over the vertex colours), tint, opacity, blend and
    /// colour matrix.
    pub style: OverlayStyle,
    /// Animation tracks resolved each frame against `OverlayFrame::time`.
    pub animations: Option<Box<OverlayAnimations>>,
    /// What this item is clipped to. The default clips nothing.
    pub clip: OverlayClip,
    /// Draw order relative to the other overlay items. Lower values render
    /// first.
    pub z_order: i32,
}

impl Default for OverlayMeshItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            indices: Vec::new(),
            colours: OverlayMeshColours::default(),
            uvs: None,
            anchoring: OverlayAnchoring::default(),
            transform: OverlayTransform::IDENTITY,
            style: OverlayStyle::solid(Colour::WHITE),
            animations: None,
            clip: OverlayClip::default(),
            z_order: 0,
        }
    }
}

impl OverlayMeshItem {
    /// A mesh of `positions` and triangle-list `indices`, white, with a white
    /// solid fill so the vertex colours draw as they are.
    pub fn new(positions: Vec<[f32; 2]>, indices: Vec<u32>) -> Self {
        Self {
            positions,
            indices,
            ..Default::default()
        }
    }

    /// Give every vertex `colour`.
    pub fn with_colour(mut self, colour: impl Into<Colour>) -> Self {
        self.colours = OverlayMeshColours::Uniform(colour.into());
        self
    }

    /// Give each vertex its own colour, one per position.
    pub fn with_colours(mut self, colours: Vec<Colour>) -> Self {
        self.colours = OverlayMeshColours::PerVertex(colours);
        self
    }

    /// Set texture coordinates, one per position.
    pub fn with_uvs(mut self, uvs: Vec<[f32; 2]>) -> Self {
        self.uvs = Some(uvs);
        self
    }

    /// Set the fill multiplied over the vertex colours.
    pub fn with_fill(mut self, fill: OverlayFill) -> Self {
        self.style.fill = fill;
        self
    }

    /// Set the whole style at once.
    pub fn with_style(mut self, style: OverlayStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the placement in logical pixels from the resolved anchor origin.
    pub fn with_position(mut self, position: [f32; 2]) -> Self {
        self.transform.translate = position;
        self
    }

    /// Set the transform.
    pub fn with_transform(mut self, transform: OverlayTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Set the origin the mesh hangs from.
    pub fn with_anchor(mut self, anchor: OverlayOrigin) -> Self {
        self.anchoring.origin = anchor;
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.style.opacity = opacity;
        self
    }

    /// Set how the item combines with what is drawn under it.
    pub fn with_blend(mut self, blend: OverlayBlend) -> Self {
        self.style.blend = blend;
        self
    }

    /// Set the draw order.
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Set the clip.
    pub fn with_clip(mut self, clip: OverlayClip) -> Self {
        self.clip = clip;
        self
    }

    /// Set the animation tracks.
    pub fn with_animations(mut self, animations: OverlayAnimations) -> Self {
        self.animations = Some(Box::new(animations));
        self
    }

    /// The bounding box of `positions` as `(min, max)`, before the transform.
    /// `None` for a mesh with no positions.
    pub fn bounds(&self) -> Option<([f32; 2], [f32; 2])> {
        let (first, rest) = self.positions.split_first()?;
        let (mut min, mut max) = (*first, *first);
        for p in rest {
            min = [min[0].min(p[0]), min[1].min(p[1])];
            max = [max[0].max(p[0]), max[1].max(p[1])];
        }
        Some((min, max))
    }

    /// Resolve the screen-pixel offset added to every position for a frame:
    /// the anchor origin, plus `translate`, shifted by `anchoring.align` for
    /// the mesh's bounding box. `None` when a `World` anchor is culled.
    pub fn resolve_offset_at(
        &self,
        viewport_size: [f32; 2],
        view: &glam::Mat4,
        proj: &glam::Mat4,
        world_origin: glam::DVec3,
    ) -> Option<[f32; 2]> {
        let origin = crate::overlay::resolve_anchor_origin_at(
            &self.anchoring.origin,
            viewport_size,
            view,
            proj,
            world_origin,
        )?;
        let (min, max) = self.bounds().unwrap_or(([0.0; 2], [0.0; 2]));
        Some([
            origin[0]
                + self.transform.translate[0]
                + self.anchoring.align.x.align_shift(max[0] - min[0]),
            origin[1]
                + self.transform.translate[1]
                + self.anchoring.align.y.align_shift(max[1] - min[1], None),
        ])
    }

    /// Whether the screen-space point falls inside one of the mesh's
    /// triangles, under its transform.
    ///
    /// The point is in logical pixels from the top-left of the viewport, and
    /// `transform.translate` is taken as the absolute offset: for a mesh with
    /// a non-default anchor, hit-test a copy whose translate is the offset
    /// [`resolve_offset_at`](Self::resolve_offset_at) returns. A point on a
    /// shared edge is inside.
    pub fn contains(&self, point: [f32; 2]) -> bool {
        let Some((min, max)) = self.bounds() else {
            return false;
        };
        let Some(inv) = self.transform.inverse_linear() else {
            return false;
        };
        let t = self.transform.translate;
        // Undo the linear map about the bounding-box centre plus the pivot,
        // then the translate.
        let piv = [
            t[0] + (min[0] + max[0]) * 0.5 + self.transform.pivot[0],
            t[1] + (min[1] + max[1]) * 0.5 + self.transform.pivot[1],
        ];
        let d = [point[0] - piv[0], point[1] - piv[1]];
        let q = [
            piv[0] + inv[0][0] * d[0] + inv[0][1] * d[1] - t[0],
            piv[1] + inv[1][0] * d[0] + inv[1][1] * d[1] - t[1],
        ];
        let cross = |a: [f32; 2], b: [f32; 2], p: [f32; 2]| {
            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        };
        self.indices.chunks_exact(3).any(|tri| {
            let get = |i: u32| self.positions.get(i as usize).copied();
            let (Some(a), Some(b), Some(c)) = (get(tri[0]), get(tri[1]), get(tri[2])) else {
                return false;
            };
            let d0 = cross(a, b, q);
            let d1 = cross(b, c, q);
            let d2 = cross(c, a, q);
            let neg = d0 < 0.0 || d1 < 0.0 || d2 < 0.0;
            let pos = d0 > 0.0 || d1 > 0.0 || d2 > 0.0;
            !(neg && pos)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> OverlayMeshItem {
        OverlayMeshItem::new(
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            vec![0, 1, 2, 0, 2, 3],
        )
    }

    #[test]
    fn contains_tests_each_triangle() {
        let m = quad().with_position([100.0, 50.0]);
        assert!(m.contains([105.0, 55.0]));
        assert!(m.contains([110.0, 60.0]));
        assert!(!m.contains([95.0, 55.0]));
        assert!(!m.contains([105.0, 61.0]));
    }

    #[test]
    fn contains_follows_the_transform() {
        // A quarter turn about the box centre maps the square onto itself,
        // but a thin strip turns from horizontal to vertical.
        let strip = OverlayMeshItem::new(
            vec![[0.0, 4.0], [10.0, 4.0], [10.0, 6.0], [0.0, 6.0]],
            vec![0, 1, 2, 0, 2, 3],
        )
        .with_transform(OverlayTransform::IDENTITY.with_rotation(std::f32::consts::FRAC_PI_2));
        assert!(strip.contains([5.0, 1.0]));
        assert!(!strip.contains([1.0, 5.0]));
    }

    #[test]
    fn a_partial_or_out_of_range_triangle_is_skipped() {
        let m = OverlayMeshItem::new(vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], vec![0, 1, 7, 0]);
        assert!(!m.contains([1.0, 1.0]));
    }
}
