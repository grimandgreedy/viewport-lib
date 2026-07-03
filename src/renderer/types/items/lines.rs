use super::SpriteBlend;
use super::common::IDENTITY_MAT4;
use crate::resources::ColourmapId;
use crate::scene::material::ItemSettings;

/// A polyline (stream tracer) item to render in the viewport.
///
/// All streamlines for one source are concatenated into a single vertex buffer.
/// `strip_lengths` records how many vertices belong to each individual streamline.
///
/// # Curve network quantities
///
/// In addition to the existing per-node scalar path (`scalars`/`colourmap_id`), this
/// item supports several curve-network quantities:
///
/// - **Per-edge scalars** (`edge_scalars`): one value per segment; rendered as a flat
///   constant colour per edge (both endpoints share the same LUT value).
/// - **Per-node colours** (`node_colours`): direct RGBA per node; takes priority over
///   scalar-driven colouring.
/// - **Per-edge colours** (`edge_colours`): direct RGBA per segment; takes priority over
///   edge scalars.
/// - **Per-node radius** (`node_radii`): per-node line width in pixels; overrides the
///   global `line_width`.
/// - **Node vectors** (`node_vectors`): world-space 3-D arrows at each node, rendered
///   automatically as `GlyphItem` arrows.
/// - **Edge vectors** (`edge_vectors`): world-space 3-D arrows at each segment midpoint,
///   also rendered as `GlyphItem` arrows.
///
/// Colour priority per segment: `node_colours`/`edge_colours` (direct) > `edge_scalars` >
/// `scalars` (per-node) > `default_colour`.
#[derive(Clone)]
#[non_exhaustive]
pub struct PolylineItem {
    /// World-space positions for all streamlines, concatenated.
    pub positions: Vec<[f32; 3]>,
    /// Per-node scalar values (same length as `positions`). Empty = no scalar colouring.
    pub scalars: Vec<f32>,
    /// Number of vertices per individual streamline strip.
    pub strip_lengths: Vec<u32>,
    /// Scalar range for LUT mapping. None = auto from min/max of `scalars` or `edge_scalars`.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. None = viridis.
    pub colourmap_id: Option<ColourmapId>,
    /// Fallback colour when no scalar or direct-colour data is provided.
    pub default_colour: [f32; 4],
    /// Global line width in pixels. Used when `node_radii` is empty.
    pub line_width: f32,
    /// Per-node direct RGBA colours. Length must match `positions`. Empty = not used.
    /// Takes priority over scalar-driven colouring when non-empty.
    pub node_colours: Vec<[f32; 4]>,
    /// Per-edge scalar values. Length = total segment count across all strips (sum of
    /// `strip_lengths[i] - 1`). Used when `scalars` is empty; both endpoints of each
    /// segment share the same LUT value (flat constant colour per edge).
    pub edge_scalars: Vec<f32>,
    /// Per-edge direct RGBA colours. Length = total segment count. Takes priority over
    /// `edge_scalars` when non-empty.
    pub edge_colours: Vec<[f32; 4]>,
    /// Per-node line width in pixels. Length must match `positions`. When non-empty,
    /// overrides the global `line_width`; adjacent endpoints are linearly interpolated
    /// along each segment.
    pub node_radii: Vec<f32>,
    /// Per-node world-space vectors. Length must match `positions`. When non-empty the
    /// renderer automatically generates a [`GlyphItem`](super::GlyphItem) (arrows at node positions).
    pub node_vectors: Vec<[f32; 3]>,
    /// Per-edge world-space vectors. Length = total segment count. When non-empty the
    /// renderer automatically generates a [`GlyphItem`](super::GlyphItem) (arrows at segment midpoints).
    pub edge_vectors: Vec<[f32; 3]>,
    /// Scale applied to generated arrow glyphs from `node_vectors`/`edge_vectors`.
    pub vector_scale: f32,
    /// Per-frame model matrix applied to `positions` in the vertex shader.
    /// Identity (the default) renders `positions` as world-space coordinates,
    /// preserving the historical behaviour. Set this to a translation, rotation,
    /// or scale to move a pre-uploaded polyline without rebuilding its vertex data.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for PolylineItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            scalars: Vec::new(),
            strip_lengths: Vec::new(),
            scalar_range: None,
            colourmap_id: None,
            default_colour: [0.9, 0.92, 0.96, 1.0],
            line_width: 2.0,
            node_colours: Vec::new(),
            edge_scalars: Vec::new(),
            edge_colours: Vec::new(),
            node_radii: Vec::new(),
            node_vectors: Vec::new(),
            edge_vectors: Vec::new(),
            vector_scale: 1.0,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Build a `PolylineItem` that draws the 12 edges of an axis-aligned bounding box.
///
/// Produces 6 strips: bottom face loop (5 pts), top face loop (5 pts), and
/// 4 vertical edges (2 pts each). Pass `colour` as RGBA in linear space.
pub fn aabb_wireframe_polyline(aabb: &crate::scene::aabb::Aabb, colour: [f32; 4]) -> PolylineItem {
    let mn = aabb.min;
    let mx = aabb.max;
    PolylineItem {
        positions: vec![
            // Bottom face loop
            [mn.x, mn.y, mn.z],
            [mx.x, mn.y, mn.z],
            [mx.x, mx.y, mn.z],
            [mn.x, mx.y, mn.z],
            [mn.x, mn.y, mn.z],
            // Top face loop
            [mn.x, mn.y, mx.z],
            [mx.x, mn.y, mx.z],
            [mx.x, mx.y, mx.z],
            [mn.x, mx.y, mx.z],
            [mn.x, mn.y, mx.z],
            // Vertical edges
            [mn.x, mn.y, mn.z],
            [mn.x, mn.y, mx.z],
            [mx.x, mn.y, mn.z],
            [mx.x, mn.y, mx.z],
            [mx.x, mx.y, mn.z],
            [mx.x, mx.y, mx.z],
            [mn.x, mx.y, mn.z],
            [mn.x, mx.y, mx.z],
        ],
        strip_lengths: vec![5, 5, 2, 2, 2, 2],
        default_colour: colour,
        ..Default::default()
    }
}

/// Build a `PolylineItem` that draws three great-circle outlines for a sphere.
///
/// Produces three closed loops in the XY, XZ, and YZ planes through the given
/// centre, each sampled at `segments` points. Pass `colour` as RGBA in linear
/// space. Used as the selection outline for `ScatterShape::Sphere`.
pub fn sphere_wireframe_polyline(
    center: [f32; 3],
    radius: f32,
    segments: u32,
    colour: [f32; 4],
) -> PolylineItem {
    let n = segments.max(8) as usize;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(3 * (n + 1));
    let two_pi = std::f32::consts::TAU;
    let cx = center[0];
    let cy = center[1];
    let cz = center[2];
    // XY plane circle
    for i in 0..=n {
        let t = i as f32 / n as f32 * two_pi;
        positions.push([cx + radius * t.cos(), cy + radius * t.sin(), cz]);
    }
    // XZ plane circle
    for i in 0..=n {
        let t = i as f32 / n as f32 * two_pi;
        positions.push([cx + radius * t.cos(), cy, cz + radius * t.sin()]);
    }
    // YZ plane circle
    for i in 0..=n {
        let t = i as f32 / n as f32 * two_pi;
        positions.push([cx, cy + radius * t.cos(), cz + radius * t.sin()]);
    }
    let strip = (n + 1) as u32;
    PolylineItem {
        positions,
        strip_lengths: vec![strip, strip, strip],
        default_colour: colour,
        ..Default::default()
    }
}

/// A streamtube item: polyline strips rendered as instanced 3D cylinder segments.
///
/// Each consecutive pair of positions within a strip becomes one cylinder instance,
/// oriented along the segment direction, scaled to the configured radius.  The
/// cylinder mesh is an 8-sided built-in uploaded once at pipeline creation time.
///
/// `StreamtubeItem` is `#[non_exhaustive]` so future fields (e.g. per-point radius
/// from a scalar attribute) can be added without breaking existing callers.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StreamtubeItem {
    /// World-space positions for all strips, concatenated.
    pub positions: Vec<[f32; 3]>,
    /// Number of vertices per individual strip.
    pub strip_lengths: Vec<u32>,
    /// Tube radius in world-space units.  Default: `0.05`.
    pub radius: f32,
    /// RGBA colour for all tube segments in this item.  Default: opaque white.
    pub colour: [f32; 4],
    /// Per-frame model matrix applied to `positions` in the vertex shader.
    /// Identity (the default) renders the tube at the world-space coordinates
    /// passed in `positions`. Set this to move a pre-uploaded streamtube without
    /// rebuilding its mesh.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for StreamtubeItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            strip_lengths: Vec::new(),
            radius: 0.05,
            colour: [1.0, 1.0, 1.0, 1.0],
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// A general tube item: polyline strips swept into a tube mesh with per-point radius
/// and scalar colourmap support.
///
/// Similar to `StreamtubeItem` but with configurable cross-section resolution,
/// optional per-point radius from a separate attribute, and per-vertex scalar colouring.
/// The CPU sweep generates a full connected mesh submitted to the streamtube pipeline.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TubeItem {
    /// World-space positions for all strips, concatenated.
    pub positions: Vec<[f32; 3]>,
    /// Number of vertices per individual strip.
    pub strip_lengths: Vec<u32>,
    /// Uniform tube radius in world-space units. Default: `0.05`.
    pub radius: f32,
    /// Optional per-point radii in world-space units. If non-empty (and same length as positions),
    /// overrides `radius` per-vertex.
    pub radius_attribute: Option<Vec<f32>>,
    /// Number of sides in the tube cross-section. Default: 8.
    pub sides: u32,
    /// Optional per-point scalar values for LUT colouring. If empty, uses `colour`.
    pub scalars: Vec<f32>,
    /// Scalar range for LUT mapping. `None` = auto from data min/max.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. `None` = default builtin (viridis).
    pub colourmap_id: Option<crate::resources::ColourmapId>,
    /// Flat RGBA colour used when `scalars` is empty.  Default: opaque white.
    pub colour: [f32; 4],
    /// Per-frame model matrix applied to `positions` in the vertex shader.
    /// Identity (the default) renders the tube at the world-space coordinates
    /// passed in `positions`. Set this to move a pre-uploaded tube without
    /// rebuilding its mesh.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for TubeItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            strip_lengths: Vec::new(),
            radius: 0.05,
            radius_attribute: None,
            sides: 8,
            scalars: Vec::new(),
            scalar_range: None,
            colourmap_id: None,
            colour: [1.0, 1.0, 1.0, 1.0],
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// A ribbon strip rendered as a flat quad surface swept along a path.
///
/// Each strip in `strip_lengths` is swept from `positions`. The ribbon lies in
/// the plane defined by the parallel-transport frame or the optional
/// `twist_attribute` vectors. Width can be uniform or per-point.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RibbonItem {
    /// World-space positions for all strips, concatenated.
    pub positions: Vec<[f32; 3]>,
    /// Number of vertices per individual strip.
    pub strip_lengths: Vec<u32>,
    /// Uniform ribbon half-width in world-space units. Default: `0.1`.
    pub width: f32,
    /// Optional per-point widths. When set, overrides `width` at each point.
    pub width_attribute: Option<Vec<f32>>,
    /// Optional per-point direction vectors that orient the ribbon face normal.
    /// When set, the ribbon is aligned with the projection of this vector onto
    /// the plane perpendicular to the local tangent.
    pub twist_attribute: Option<Vec<[f32; 3]>>,
    /// Optional per-point scalar values for LUT colouring. Empty = use `colour`.
    pub scalars: Vec<f32>,
    /// Scalar range for LUT mapping. `None` = auto from data min/max.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. `None` = default builtin (viridis).
    pub colourmap_id: Option<crate::resources::ColourmapId>,
    /// Flat RGBA colour used when `scalars` and `colour_attribute` are empty.
    /// Default: opaque white.
    pub colour: [f32; 4],
    /// Optional per-point RGBA colour. When non-empty this overrides `colour`
    /// and the `scalars`/`colourmap_id` path, and is the natural way to express
    /// a trail that fades along its length (set each entry's alpha directly).
    pub colour_attribute: Vec<[f32; 4]>,
    /// GPU blend state for this ribbon. Default: [`SpriteBlend::AlphaBlend`].
    /// Use [`SpriteBlend::Additive`] for energy or spark trails.
    pub blend: SpriteBlend,
    /// Optional streak texture sampled along the ribbon. `None` renders the
    /// ribbon without a texture (the resolved colour is used directly). Use
    /// for lightning, slash arcs, dragon breath, laser beams.
    pub texture_id: Option<crate::resources::TextureId>,
    /// Optional per-vertex `u` coordinate along the strip. When empty, `u` is
    /// derived from cumulative arc length: 0.0 at the first vertex of each
    /// strip, 1.0 at the last. The cross-strip `v` is fixed at 0.0 on one
    /// edge and 1.0 on the other.
    pub u_attribute: Vec<f32>,
    /// Per-frame model matrix applied to `positions` in the vertex shader.
    /// Identity (the default) renders the ribbon at the world-space coordinates
    /// passed in `positions`. Set this to move a pre-uploaded ribbon without
    /// rebuilding its mesh.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for RibbonItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            strip_lengths: Vec::new(),
            width: 0.1,
            width_attribute: None,
            twist_attribute: None,
            scalars: Vec::new(),
            scalar_range: None,
            colourmap_id: None,
            colour: [1.0, 1.0, 1.0, 1.0],
            colour_attribute: Vec::new(),
            blend: SpriteBlend::AlphaBlend,
            texture_id: None,
            u_attribute: Vec::new(),
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}
