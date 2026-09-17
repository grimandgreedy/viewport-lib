//! The three swept-curve item types: streamtube, tube and ribbon.
//!
//! They share a shape (a control polyline plus per-point width) and a pipeline
//! family, so they share a module the way the plugins that draw them do.

use crate::renderer::SpriteBlend;
use crate::renderer::types::items::IDENTITY_MAT4;
use crate::resources::ColourmapId;
use crate::scene::material::ItemSettings;

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
    pub colour: crate::Colour,
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
            colour: [1.0, 1.0, 1.0, 1.0].into(),
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
    pub colour: crate::Colour,
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
            colour: [1.0, 1.0, 1.0, 1.0].into(),
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
    pub colour: crate::Colour,
    /// Optional per-point RGBA colour. When non-empty this overrides `colour`
    /// and the `scalars`/`colourmap_id` path, and is the natural way to express
    /// a trail that fades along its length (set each entry's alpha directly).
    pub colour_attribute: Vec<crate::Colour>,
    /// GPU blend state for this ribbon. Default: [`SpriteBlend::AlphaBlend`].
    /// Use [`SpriteBlend::Additive`] for energy or spark trails.
    pub blend: SpriteBlend,
    /// Whether the ribbon writes depth. Default: `true`, matching tubes and
    /// streamtubes, which are the other swept surfaces in this family.
    ///
    /// A ribbon that writes depth draws with the opaque scene and is visible to
    /// everything that reads the depth buffer: it occludes, it receives
    /// projected decals, and soft-particle sprites fade against it. Clear this
    /// for a genuinely translucent ribbon and an `AlphaBlend` or
    /// `Premultiplied` ribbon routes through order-independent transparency
    /// instead, which resolves overlapping segments without sorting them but
    /// contributes no depth. `Additive` ribbons never write depth, since
    /// accumulating is the point of that blend.
    pub depth_write: bool,
    /// Optional streak texture sampled along the ribbon. `None` renders the
    /// ribbon without a texture (the resolved colour is used directly). Use
    /// for lightning, slash arcs, dragon breath, laser beams.
    ///
    /// Colour, so upload it sRGB
    /// ([`TextureData::srgb`](crate::resources::TextureData::srgb)).
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
            colour: [1.0, 1.0, 1.0, 1.0].into(),
            colour_attribute: Vec::new(),
            blend: SpriteBlend::AlphaBlend,
            depth_write: true,
            texture_id: None,
            u_attribute: Vec::new(),
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded streamtube. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StreamtubeRefItem {
    /// Handle to GPU buffers produced by
    /// [`DeviceResources::upload_streamtube`](crate::resources::DeviceResources::upload_streamtube)
    /// or `begin_upload_streamtube`.
    pub source: crate::resources::StreamtubeId,
    /// Per-frame model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl StreamtubeRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::StreamtubeId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded tube. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TubeRefItem {
    /// Handle to GPU buffers produced by
    /// [`DeviceResources::upload_tube`](crate::resources::DeviceResources::upload_tube)
    /// or `begin_upload_tube`.
    pub source: crate::resources::TubeId,
    /// Per-frame model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl TubeRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::TubeId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded ribbon. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RibbonRefItem {
    /// Handle to GPU buffers produced by
    /// [`DeviceResources::upload_ribbon`](crate::resources::DeviceResources::upload_ribbon)
    /// or `begin_upload_ribbon`.
    pub source: crate::resources::RibbonId,
    /// Per-frame model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl RibbonRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::RibbonId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}
