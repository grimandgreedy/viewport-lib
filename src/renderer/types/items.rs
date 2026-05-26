use crate::resources::ColourmapId;
use crate::scene::material::{ItemSettings, Material};

// ---------------------------------------------------------------------------
// Per-frame data types
// ---------------------------------------------------------------------------

/// LIC overlay data attached to a surface item.
///
/// Set `SceneRenderItem::lic` to `Some(LicOverlay { ... })` to render a
/// Line Integral Convolution flow visualisation on that surface mesh.
/// The mesh must have a `VertexVector` attribute matching `vector_attribute`
/// uploaded via `ViewportGpuResources::upload_mesh_data`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LicOverlay {
    /// Name of the `AttributeData::VertexVector` attribute on the mesh.
    pub vector_attribute: String,
    /// LIC rendering configuration (step count, step size, strength).
    pub config: crate::renderer::types::frame::SurfaceLICConfig,
}

impl LicOverlay {
    /// Create a new `LicOverlay` with the given vector attribute name and LIC configuration.
    pub fn new(
        vector_attribute: impl Into<String>,
        config: crate::renderer::types::frame::SurfaceLICConfig,
    ) -> Self {
        Self {
            vector_attribute: vector_attribute.into(),
            config,
        }
    }
}

/// Typed GPU pick identifier for a renderable surface.
///
/// The renderer includes a surface in the GPU pick pass only when its `pick_id`
/// is not [`PickId::NONE`]. Helper geometry, transient previews, and any surface
/// that should be invisible to clicks should use `PickId::NONE` (the default).
///
/// Application code owns pick identity: IDs do not need to come from the scene
/// graph. Any nonzero `u64` that the application can map back to a domain object
/// is a valid `PickId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PickId(pub u64);

impl PickId {
    /// Sentinel value meaning "not pickable".
    ///
    /// Surfaces with `pick_id == PickId::NONE` are excluded from the GPU pick
    /// pass. This is the default value for [`SceneRenderItem::pick_id`].
    pub const NONE: Self = Self(0);
}

/// Per-object render data for one frame.
#[derive(Clone)]
#[non_exhaustive]
pub struct SceneRenderItem {
    /// `MeshId` of the uploaded GPU mesh for this object.
    pub mesh_id: crate::resources::mesh_store::MeshId,
    /// World-space model matrix (Translation * Rotation * Scale).
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
    /// Whether to render per-vertex normal visualization lines for this object.
    pub show_normals: bool,
    /// Per-object material (colour, shading coefficients, texture).
    pub material: Material,
    /// Named scalar attribute to colour by. `None` = use material base colour.
    pub active_attribute: Option<crate::resources::AttributeRef>,
    /// Explicit scalar range `(min, max)`. `None` = use auto-range computed at upload time.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap to use for scalar colouring. Ignored when `active_attribute` is `None`.
    pub colourmap_id: Option<crate::resources::ColourmapId>,
    /// RGBA colour for NaN scalar values. `None` = discard (fully transparent).
    pub nan_colour: Option<[f32; 4]>,
    /// Named vector attribute (from `AttributeData::VertexVector`) used to displace
    /// vertex positions in the vertex shader. `None` = no warp. See also `warp_scale`.
    ///
    /// The attribute must be uploaded as part of the mesh's `MeshData::attributes` with
    /// kind `AttributeData::VertexVector`. The vertex shader applies:
    /// `local_pos += warp_scale * warp_buffer[vertex_index]` before the model transform.
    pub warp_attribute: Option<String>,
    /// Scale factor applied to the warp vector. Default: 1.0.
    pub warp_scale: f32,
    /// Which skinning instance palette to use when `mesh_id` has been marked
    /// skinnable via [`crate::ViewportGpuResources::set_skin_weights`].
    ///
    /// - `None`: this item is treated as static even if the mesh has skinning
    ///   data attached. The standard (non-skinned) pipeline is used.
    /// - `Some(instance_id)`: the renderer routes this item through the
    ///   skinned pipeline variant and binds the joint palette uploaded via
    ///   `set_skin_palette(mesh_id, instance_id, ...)`. If no palette has
    ///   been uploaded for this `(mesh_id, instance_id)`, the item falls
    ///   back to the static pipeline for this frame.
    ///
    /// Allows multiple `SceneRenderItem`s that share a bind-pose mesh to each
    /// pose independently (crowd / instanced characters).
    pub skin_instance: Option<u32>,
    /// Whether this surface receives projected decals. Default: `true`.
    pub receives_decals: bool,
    /// LIC flow overlay for this surface. `None` disables LIC for this item.
    ///
    /// The mesh must have a `VertexVector` attribute matching
    /// `LicOverlay::vector_attribute`.
    pub lic: Option<LicOverlay>,
}

impl Default for SceneRenderItem {
    fn default() -> Self {
        Self {
            mesh_id: crate::resources::mesh_store::MeshId(0),
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
            show_normals: false,
            material: Material::default(),
            active_attribute: None,
            scalar_range: None,
            colourmap_id: None,
            nan_colour: None,
            warp_attribute: None,
            warp_scale: 1.0,
            skin_instance: None,
            receives_decals: true,
            lic: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque volume mesh item
// ---------------------------------------------------------------------------

/// Render item for an opaque volume mesh uploaded via
/// [`upload_volume_mesh_data`](crate::resources::ViewportGpuResources::upload_volume_mesh_data).
///
/// Wraps the `MeshId` produced by boundary extraction together with the
/// face-to-cell mapping so consumers can recover cell-level identity from
/// face-level pick hits. Call [`to_render_item`](Self::to_render_item) each
/// frame to produce the [`SceneRenderItem`] submitted to the renderer.
///
/// ```rust,ignore
/// let (mesh_id, face_to_cell) = resources.upload_volume_mesh_data(&device, &data)?;
/// let item = VolumeMeshItem::new(mesh_id, face_to_cell);
///
/// // Each frame:
/// scene_frame.surfaces = SurfaceSubmission::Flat(vec![item.to_render_item()].into());
/// ```
#[non_exhaustive]
#[derive(Clone)]
pub struct VolumeMeshItem {
    /// GPU mesh slot for the extracted boundary surface.
    pub mesh_id: crate::resources::mesh_store::MeshId,
    /// Maps each boundary triangle to its originating cell index.
    ///
    /// `face_to_cell[face_index]` is the cell index in the original
    /// [`VolumeMeshData::cells`](crate::VolumeMeshData::cells) array.
    /// Use this to convert a [`SubObjectRef::Face`](crate::interaction::sub_object::SubObjectRef::Face)
    /// pick hit into a cell index.
    pub face_to_cell: Vec<u32>,
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
    /// Per-object material.
    pub material: crate::scene::material::Material,
    /// Named scalar or colour attribute to colour by.
    pub active_attribute: Option<crate::resources::AttributeRef>,
    /// Explicit scalar range `(min, max)`. `None` = auto-range from upload.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring.
    pub colourmap_id: Option<ColourmapId>,
}

impl VolumeMeshItem {
    /// Create a new item from the mesh ID and face-to-cell map returned by
    /// [`upload_volume_mesh_data`](crate::resources::ViewportGpuResources::upload_volume_mesh_data).
    pub fn new(mesh_id: crate::resources::mesh_store::MeshId, face_to_cell: Vec<u32>) -> Self {
        Self {
            mesh_id,
            face_to_cell,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
            material: crate::scene::material::Material::default(),
            active_attribute: None,
            scalar_range: None,
            colourmap_id: None,
        }
    }

    /// Build the [`SceneRenderItem`] that the renderer consumes.
    ///
    /// The volume mesh renders through the standard surface pipeline; this
    /// method copies the per-frame fields into a `SceneRenderItem`.
    pub fn to_render_item(&self) -> SceneRenderItem {
        SceneRenderItem {
            mesh_id: self.mesh_id,
            model: self.model,
            settings: self.settings,
            material: self.material.clone(),
            active_attribute: self.active_attribute.clone(),
            scalar_range: self.scalar_range,
            colourmap_id: self.colourmap_id,
            ..SceneRenderItem::default()
        }
    }

    /// Look up the cell index for a boundary face hit.
    ///
    /// Returns `None` if `face_index` is out of range.
    pub fn cell_for_face(&self, face_index: u32) -> Option<u32> {
        self.face_to_cell.get(face_index as usize).copied()
    }

    /// Replace the mesh ID and face-to-cell map, for example after a clipped
    /// re-upload via
    /// [`replace_clipped_volume_mesh_data`](crate::resources::ViewportGpuResources::replace_clipped_volume_mesh_data).
    pub fn update_mesh(
        &mut self,
        mesh_id: crate::resources::mesh_store::MeshId,
        face_to_cell: Vec<u32>,
    ) {
        self.mesh_id = mesh_id;
        self.face_to_cell = face_to_cell;
    }
}

// ---------------------------------------------------------------------------
// SciVis Phase B : point cloud and glyph renderers
// ---------------------------------------------------------------------------

/// Render mode for point cloud items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointRenderMode {
    /// Flat disc: billboard quad clipped to a circle. Fast, no shading.
    #[default]
    ScreenSpaceCircle,
    /// Shaded sphere: billboard quad with hemisphere normal shading (ambient + diffuse + specular).
    /// Points look like small lit spheres without actual geometry cost.
    Sphere,
}

/// A point cloud item to render in the viewport.
#[derive(Clone)]
#[non_exhaustive]
pub struct PointCloudItem {
    /// World-space positions (one vec3 per point).
    pub positions: Vec<[f32; 3]>,
    /// Optional per-point RGBA colours in linear `[0,1]`. If empty, uses `default_colour`.
    pub colours: Vec<[f32; 4]>,
    /// Optional per-point scalar values for LUT colouring. If non-empty, overrides `colours`.
    pub scalars: Vec<f32>,
    /// Scalar range for LUT mapping. None = auto from min/max of `scalars`.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. None = use default builtin (viridis).
    pub colourmap_id: Option<ColourmapId>,
    /// Screen-space point size in pixels. Default: 4.0.
    pub point_size: f32,
    /// Fallback colour when neither `colours` nor `scalars` are provided.
    pub default_colour: [f32; 4],
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Render mode. Default: ScreenSpaceCircle.
    pub render_mode: PointRenderMode,
    /// Optional per-point radii in pixels. If non-empty, overrides `point_size` for each point.
    pub radii: Vec<f32>,
    /// Optional per-point opacity values in `[0, 1]`. If non-empty, scales each point's alpha.
    pub transparencies: Vec<f32>,
    /// When true, each point is rendered as a soft Gaussian splat instead of a flat circle.
    /// The alpha falls off as `exp(-3 * d²)` where `d` is the normalised distance from the
    /// point centre. Default: false.
    pub gaussian: bool,
    /// Optional per-point scalars that drive the splat radius.  If non-empty, these values
    /// are mapped from `radius_scalar_range` (or data min/max when `None`) to `radius_range`
    /// (pixels) and used as per-point radii, overriding `radii` and `point_size`.
    pub radius_scalars: Vec<f32>,
    /// Normalization range for `radius_scalars`.  `None` = auto from data min/max.
    pub radius_scalar_range: Option<(f32, f32)>,
    /// Output pixel-radius range `[min_px, max_px]` for the radius scalar mapping.
    /// Default: `(2.0, 12.0)`.
    pub radius_range: (f32, f32),
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for PointCloudItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            colours: Vec::new(),
            scalars: Vec::new(),
            scalar_range: None,
            colourmap_id: None,
            point_size: 4.0,
            default_colour: [1.0, 1.0, 1.0, 1.0],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            render_mode: PointRenderMode::ScreenSpaceCircle,
            radii: Vec::new(),
            transparencies: Vec::new(),
            gaussian: false,
            radius_scalars: Vec::new(),
            radius_scalar_range: None,
            radius_range: (2.0, 12.0),
            settings: ItemSettings::default(),
        }
    }
}

/// Glyph shape type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlyphType {
    /// Cone tip + cylinder shaft.
    #[default]
    Arrow,
    /// Icosphere.
    Sphere,
    /// Unit cube.
    Cube,
}

/// A set of instanced glyphs to render (e.g. velocity arrows).
#[derive(Clone)]
#[non_exhaustive]
pub struct GlyphItem {
    /// World-space base positions (one per glyph instance).
    pub positions: Vec<[f32; 3]>,
    /// Per-instance direction vectors. Length = magnitude (used for orientation + optional scale).
    pub vectors: Vec<[f32; 3]>,
    /// Global scale factor applied to all glyph instances. Default: 1.0.
    pub scale: f32,
    /// Whether glyph size scales with vector magnitude. Default: true.
    pub scale_by_magnitude: bool,
    /// Clamp magnitude range for scaling. None = no clamping.
    pub magnitude_clamp: Option<(f32, f32)>,
    /// Optional per-instance scalar values for LUT colouring. Empty = colour by magnitude.
    pub scalars: Vec<f32>,
    /// Scalar range for LUT mapping. None = auto from data.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. None = use default builtin (viridis).
    pub colourmap_id: Option<ColourmapId>,
    /// Fallback RGBA colour used when `use_default_colour` is true. Default: transparent (unused).
    pub default_colour: [f32; 4],
    /// When true, glyphs are coloured by `default_colour` (with per-instance scalar as brightness)
    /// instead of the LUT. Default: false.
    pub use_default_colour: bool,
    /// Glyph shape. Default: Arrow.
    pub glyph_type: GlyphType,
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for GlyphItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            vectors: Vec::new(),
            scale: 1.0,
            scale_by_magnitude: true,
            magnitude_clamp: None,
            scalars: Vec::new(),
            scalar_range: None,
            colourmap_id: None,
            default_colour: [0.0; 4],
            use_default_colour: false,
            glyph_type: GlyphType::Arrow,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
        }
    }
}

/// A set of instanced tensor glyphs for stress/strain visualization.
///
/// Each instance is an ellipsoid at `positions[i]`, scaled anisotropically by the
/// absolute eigenvalues along the eigenvector axes. Colour comes from `colour_attribute`
/// if provided, otherwise from the sign of the first (dominant) eigenvalue.
#[derive(Clone)]
#[non_exhaustive]
pub struct TensorGlyphItem {
    /// World-space positions, one per instance.
    pub positions: Vec<[f32; 3]>,
    /// Per-instance eigenvalues `[lambda0, lambda1, lambda2]`.
    /// The ellipsoid is scaled by `|lambda_i| * scale` along each eigenvector axis.
    pub eigenvalues: Vec<[f32; 3]>,
    /// Per-instance eigenvectors as column vectors `[[e0x,e0y,e0z], [e1x,...], [e2x,...]]`.
    /// Must form an orthonormal basis. Length must match `positions`.
    pub eigenvectors: Vec<[[f32; 3]; 3]>,
    /// Global scale factor applied to all instances. Default: 1.0.
    pub scale: f32,
    /// Optional per-instance scalar values for LUT colouring.
    /// When `None`, colours by sign of `eigenvalues[i][0]`: positive -> upper LUT, negative -> lower LUT.
    pub colour_attribute: Option<Vec<f32>>,
    /// Scalar range for LUT mapping. `None` = auto from data.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. `None` = viridis. For sign colouring, a diverging map works best.
    pub colourmap_id: Option<ColourmapId>,
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for TensorGlyphItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            scale: 1.0,
            colour_attribute: None,
            scalar_range: None,
            colourmap_id: None,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
        }
    }
}

/// A volume item to render via GPU ray-marching.
///
/// The caller uploads a 3D scalar field via [`ViewportGpuResources::upload_volume`](crate::resources::ViewportGpuResources::upload_volume) and
/// receives a [`VolumeId`](crate::resources::VolumeId). Each frame, submit a `VolumeItem` referencing that id plus
/// transfer function and display parameters.
///
/// # Picking
///
/// Set `pick_id` to a non-zero value and provide `volume_data` to enable voxel-level
/// picking via `renderer.pick()` and `renderer.pick_rect()`. `pick_id` is the ID
/// returned in the `PickHit` when a voxel is hit. A `pick_id` of zero means the volume
/// is not pickable. `volume_data` must match the data that was passed to
/// `upload_volume` for this `volume_id`.
#[derive(Clone)]
#[non_exhaustive]
pub struct VolumeItem {
    /// Reference to a previously uploaded 3D texture.
    pub volume_id: crate::resources::VolumeId,
    /// CPU scalar data for voxel picking.
    ///
    /// Must match the data passed to `upload_volume` for `volume_id`.
    /// `None` disables voxel-level picking regardless of `settings.pick_id`.
    pub volume_data: Option<std::sync::Arc<crate::geometry::marching_cubes::VolumeData>>,
    /// Colour transfer function LUT. `None` = use default builtin (viridis).
    pub colour_lut: Option<ColourmapId>,
    /// Opacity transfer function LUT. `None` = linear ramp (0 at min, 1 at max).
    pub opacity_lut: Option<ColourmapId>,
    /// Scalar range for normalization [min, max].
    pub scalar_range: (f32, f32),
    /// World-space bounding box minimum corner.
    pub bbox_min: [f32; 3],
    /// World-space bounding box maximum corner.
    pub bbox_max: [f32; 3],
    /// Ray step multiplier. Lower = higher quality, slower. Default: 1.0.
    pub step_scale: f32,
    /// World-space transform. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Whether to apply gradient-based Phong shading. Default: false.
    pub enable_shading: bool,
    /// Global opacity multiplier. Default: 1.0.
    pub opacity_scale: f32,
    /// Scalar threshold range [min, max]. Samples outside this range are discarded (opacity = 0).
    /// Default: same as scalar_range (no clipping).
    pub threshold_min: f32,
    /// Upper scalar threshold. Samples above this value are discarded.
    /// Default: same as scalar_range.1 (no clipping).
    pub threshold_max: f32,
    /// Colour and opacity to use for NaN scalar samples. `None` = skip NaN samples entirely
    /// (same as current behaviour: discard). `Some([r, g, b, a])` = render NaN voxels with
    /// this fixed RGBA colour instead of sampling the transfer function.
    pub nan_colour: Option<[f32; 4]>,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for VolumeItem {
    fn default() -> Self {
        Self {
            volume_id: crate::resources::VolumeId(0),
            volume_data: None,
            colour_lut: None,
            opacity_lut: None,
            scalar_range: (0.0, 1.0),
            bbox_min: [0.0, 0.0, 0.0],
            bbox_max: [1.0, 1.0, 1.0],
            step_scale: 1.0,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            enable_shading: false,
            opacity_scale: 1.0,
            threshold_min: 0.0,
            threshold_max: 1.0,
            nan_colour: None,
            settings: ItemSettings::default(),
        }
    }
}

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
    /// renderer automatically generates a [`GlyphItem`] (arrows at node positions).
    pub node_vectors: Vec<[f32; 3]>,
    /// Per-edge world-space vectors. Length = total segment count. When non-empty the
    /// renderer automatically generates a [`GlyphItem`] (arrows at segment midpoints).
    pub edge_vectors: Vec<[f32; 3]>,
    /// Scale applied to generated arrow glyphs from `node_vectors`/`edge_vectors`.
    pub vector_scale: f32,
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

// ---------------------------------------------------------------------------
// SciVis Phase M : streamtube renderer
// ---------------------------------------------------------------------------

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
            settings: ItemSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3 : General Tube representation
// ---------------------------------------------------------------------------

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
            settings: ItemSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 8.1 : Ribbon representation
// ---------------------------------------------------------------------------

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
    /// Flat RGBA colour used when `scalars` is empty. Default: opaque white.
    pub colour: [f32; 4],
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
            settings: ItemSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3 : 2D Image Slice representation
// ---------------------------------------------------------------------------

/// Axis for an axis-aligned image slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceAxis {
    /// Slice perpendicular to the X axis (YZ plane).
    X,
    /// Slice perpendicular to the Y axis (XZ plane).
    #[default]
    Y,
    /// Slice perpendicular to the Z axis (XY plane).
    Z,
}

/// A 2D image slice item: renders one axis-aligned cross-section of an uploaded volume
/// as a flat coloured quad.
///
/// Faster and simpler than full volume ray-marching. Use it to inspect individual
/// slices of a structured grid without the depth ambiguity of ray-marching.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ImageSliceItem {
    /// Reference to a previously uploaded 3D volume texture.
    pub volume_id: crate::resources::VolumeId,
    /// Axis perpendicular to the slice plane. Default: `SliceAxis::Z`.
    pub axis: SliceAxis,
    /// Normalized position along the axis in `[0, 1]`. Default: `0.5`.
    pub offset: f32,
    /// World-space bounding box minimum corner of the volume.
    pub bbox_min: [f32; 3],
    /// World-space bounding box maximum corner of the volume.
    pub bbox_max: [f32; 3],
    /// Scalar range for colourmap mapping `[min, max]`. Default: `(0.0, 1.0)`.
    pub scalar_range: (f32, f32),
    /// Colour LUT. `None` = default builtin (viridis).
    pub colour_lut: Option<crate::resources::ColourmapId>,
    /// Overall opacity of the slice quad. Default: `1.0`.
    pub opacity: f32,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for ImageSliceItem {
    fn default() -> Self {
        Self {
            volume_id: crate::resources::VolumeId(0),
            axis: SliceAxis::Z,
            offset: 0.5,
            bbox_min: [0.0, 0.0, 0.0],
            bbox_max: [1.0, 1.0, 1.0],
            scalar_range: (0.0, 1.0),
            colour_lut: None,
            opacity: 1.0,
            settings: ItemSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 10 : Volume surface slice representation
// ---------------------------------------------------------------------------

/// A volume slice sampled on an arbitrary surface mesh.
///
/// Unlike [`ImageSliceItem`] which is restricted to axis-aligned flat quads,
/// this item renders any uploaded mesh and colours each fragment by the volume
/// scalar at that world-space position. The slice surface can be a flat plane,
/// a disk, a saddle, a paraboloid -- any shape that can be expressed as a mesh.
///
/// Upload the surface mesh once with [`ViewportGpuResources::upload_mesh_data`]
/// to get a [`MeshId`](crate::resources::mesh_store::MeshId), then submit a
/// `VolumeSurfaceSliceItem` referencing that mesh each frame.
///
/// Fragments whose world position falls outside the volume bounding box are
/// discarded, so the mesh can extend beyond the volume without clipping artifacts.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct VolumeSurfaceSliceItem {
    /// Reference to a previously uploaded 3D volume texture.
    pub volume_id: crate::resources::VolumeId,
    /// Mesh defining the slice surface shape. Any mesh works: flat quad, disk, saddle, etc.
    pub mesh_id: crate::resources::mesh_store::MeshId,
    /// World-space bounding box minimum corner of the volume.
    pub bbox_min: [f32; 3],
    /// World-space bounding box maximum corner of the volume.
    pub bbox_max: [f32; 3],
    /// Scalar range for colourmap mapping `[min, max]`. Default: `(0.0, 1.0)`.
    pub scalar_range: (f32, f32),
    /// Colour LUT. `None` = default builtin (viridis).
    pub colour_lut: Option<crate::resources::ColourmapId>,
    /// Overall opacity of the slice. Default: `1.0`.
    pub opacity: f32,
    /// World-space model matrix for the slice mesh. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for VolumeSurfaceSliceItem {
    fn default() -> Self {
        Self {
            volume_id: crate::resources::VolumeId(0),
            mesh_id: crate::resources::mesh_store::MeshId::from_index(0),
            bbox_min: [0.0, 0.0, 0.0],
            bbox_max: [1.0, 1.0, 1.0],
            scalar_range: (0.0, 1.0),
            colour_lut: None,
            opacity: 1.0,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 10B : Screen-space image overlays
// ---------------------------------------------------------------------------

/// Anchor corner for a [`ScreenImageItem`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ImageAnchor {
    /// Top-left corner of the viewport (default).
    #[default]
    TopLeft,
    /// Top-right corner of the viewport.
    TopRight,
    /// Bottom-left corner of the viewport.
    BottomLeft,
    /// Bottom-right corner of the viewport.
    BottomRight,
    /// Centered in the viewport.
    Center,
}

/// A floating screen-space RGBA image rendered as a viewport overlay.
///
/// The image is drawn after all 3D geometry and anchored to one of the viewport
/// corners or the center.
///
/// ## Migration note
///
/// If you do not need depth compositing against scene geometry, prefer
/// [`OverlayImageItem`] in [`OverlayFrame`] instead. `OverlayImageItem` has no
/// `depth` field and renders after post-processing alongside other semantic
/// overlays (labels, scalar bars, rulers).
///
/// ## Depth compositing (Phase 12)
///
/// When `depth` is `Some`, the image composites against 3D scene geometry:
/// pixels whose depth value exceeds the scene depth at that screen position are
/// discarded, so near geometry occludes the image correctly.
///
/// `depth` must contain exactly `width * height` `f32` values in row-major,
/// top-to-bottom order. Each value is an NDC depth in `[0.0, 1.0]` where
/// `0.0` = near plane and `1.0` = far plane, matching wgpu's depth convention.
///
/// Depth compositing is only active in the full `render()` path. When using
/// `paint()` / `paint_to()` (external render passes), the image is drawn
/// without a depth test regardless of this field.
#[non_exhaustive]
#[derive(Clone)]
pub struct ScreenImageItem {
    /// RGBA8 pixel data, row-major, top-to-bottom.
    pub pixels: Vec<[u8; 4]>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Which corner (or center) of the viewport to anchor the image to.
    pub anchor: ImageAnchor,
    /// Scale factor relative to natural pixel size (`1.0` = one pixel per screen pixel).
    pub scale: f32,
    /// Overall opacity multiplier applied on top of per-pixel alpha. Default: `1.0`.
    pub alpha: f32,
    /// Per-pixel NDC depth values `[0.0, 1.0]` for depth compositing against scene
    /// geometry. Must contain exactly `width * height` values if `Some`.
    /// `None` (default) renders the image on top of all geometry (no depth test).
    pub depth: Option<Vec<f32>>,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for ScreenImageItem {
    fn default() -> Self {
        Self {
            pixels: Vec::new(),
            width: 0,
            height: 0,
            anchor: ImageAnchor::TopLeft,
            scale: 1.0,
            alpha: 1.0,
            depth: None,
            settings: ItemSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase G : GPU compute filter types
// ---------------------------------------------------------------------------

/// Whether a filter runs on CPU or GPU compute shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// CPU-side index compaction (existing path, always works).
    #[default]
    Cpu,
    /// GPU compute shader index compaction (faster for large meshes).
    Gpu,
}

/// Kind of GPU compute filter operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeFilterKind {
    /// Clip: discard triangles where all 3 vertices are on the negative side of a plane.
    /// Plane defined as (normal.xyz, distance) where dot(pos, normal) < distance => clipped.
    Clip {
        /// Unit normal of the clip plane.
        plane_normal: [f32; 3],
        /// Signed distance from origin along the plane normal.
        plane_dist: f32,
    },
    /// Box clip: discard triangles where all 3 vertices are outside an oriented box region.
    ClipBox {
        /// Box center in world space.
        center: [f32; 3],
        /// Half-extents (per-axis radius) of the box.
        half_extents: [f32; 3],
        /// Rotation matrix columns: local X, Y, Z axes in world space.
        orientation: [[f32; 3]; 3],
    },
    /// Sphere clip: discard triangles where all 3 vertices are outside a sphere region.
    ClipSphere {
        /// Sphere center in world space.
        center: [f32; 3],
        /// Sphere radius in world units.
        radius: f32,
    },
    /// Threshold: discard triangles where all 3 vertex scalars are outside [min, max].
    Threshold {
        /// Minimum scalar value (inclusive).
        min: f32,
        /// Maximum scalar value (inclusive).
        max: f32,
    },
}

/// A GPU compute filter item : references an existing uploaded mesh.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ComputeFilterItem {
    /// `MeshId` of the uploaded GPU mesh for this compute filter item.
    pub mesh_id: crate::resources::mesh_store::MeshId,
    /// Which filter to apply.
    pub kind: ComputeFilterKind,
    /// Name of the scalar attribute buffer (for Threshold). Ignored for Clip.
    pub attribute_name: Option<String>,
}

impl Default for ComputeFilterItem {
    fn default() -> Self {
        Self {
            mesh_id: crate::resources::mesh_store::MeshId(0),
            kind: ComputeFilterKind::Clip {
                plane_normal: [0.0, 0.0, 1.0],
                plane_dist: 0.0,
            },
            attribute_name: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Sprite and particle billboard rendering
// ---------------------------------------------------------------------------

/// Controls whether sprite sizes are measured in screen-space pixels or world-space units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpriteSizeMode {
    /// Sizes in screen-space pixels. Sprites maintain constant apparent size at all distances.
    #[default]
    ScreenSpace,
    /// Sizes in world-space units. Sprites shrink with distance like regular geometry.
    WorldSpace,
}

/// A batch of instanced billboard sprites rendered as camera-facing textured quads.
///
/// Each instance is one billboard at a world-space position. All instances in the batch share
/// one texture (or render as solid-colour quads when `texture_id` is `None`). Per-instance
/// colour, size, rotation, and atlas UV rect are specified via parallel `Vec` fields; empty
/// vecs fall back to the batch defaults.
///
/// # Particle effects
///
/// Submit a new `SpriteItem` each frame with updated `positions` and `colours` to animate
/// CPU-simulated particle effects. The host application owns simulation state (velocity,
/// lifetime, emission); the renderer only handles drawing.
///
/// # Texture atlases
///
/// Set `uv_rects` to select sub-regions of the texture per sprite, enabling flip-book
/// animation or mixed icon sets from a single atlas texture.
#[non_exhaustive]
#[derive(Clone)]
pub struct SpriteItem {
    /// Texture ID from [`ViewportGpuResources::upload_texture`].
    /// `None` renders solid-colour quads using `colours` / `default_colour` only.
    pub texture_id: Option<u64>,
    /// World-space positions, one per sprite instance.
    pub positions: Vec<[f32; 3]>,
    /// Per-instance RGBA colour tints. Empty = use `default_colour` for all.
    /// Multiplied with the texture sample (or used directly when `texture_id` is `None`).
    pub colours: Vec<[f32; 4]>,
    /// Per-instance sizes. Empty = use `default_size` for all.
    /// Interpretation depends on `size_mode`.
    pub sizes: Vec<f32>,
    /// Per-instance rotation angles in radians, CCW around the camera-forward axis.
    /// Empty = no rotation applied.
    pub rotations: Vec<f32>,
    /// Per-instance UV rects `[u0, v0, u1, v1]` selecting atlas sub-regions.
    /// Empty = full texture `[0.0, 0.0, 1.0, 1.0]` for all.
    pub uv_rects: Vec<[f32; 4]>,
    /// Fallback RGBA colour tint used when `colours` is empty. Default: opaque white.
    pub default_colour: [f32; 4],
    /// Default size when `sizes` is empty. Pixels (ScreenSpace) or world units (WorldSpace).
    pub default_size: f32,
    /// Whether sizes are in screen-space pixels or world-space units.
    pub size_mode: SpriteSizeMode,
    /// World-space model transform applied to all positions. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Whether this batch writes to the depth buffer. Default: `false`.
    ///
    /// Set `false` for transparent or additive particle effects so sprites do not occlude
    /// each other based on submission order. Set `true` for opaque world-space markers
    /// that should participate in depth testing normally.
    pub depth_write: bool,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for SpriteItem {
    fn default() -> Self {
        Self {
            texture_id: None,
            positions: Vec::new(),
            colours: Vec::new(),
            sizes: Vec::new(),
            rotations: Vec::new(),
            uv_rects: Vec::new(),
            default_colour: [1.0, 1.0, 1.0, 1.0],
            default_size: 32.0,
            size_mode: SpriteSizeMode::ScreenSpace,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            depth_write: false,
            settings: ItemSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Gaussian splat renderer types
// ---------------------------------------------------------------------------

/// SH degree stored with a Gaussian splat set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShDegree {
    /// 3 floats per splat (base RGB only).
    #[default]
    Zero,
    /// 12 floats per splat.
    One,
    /// 48 floats per splat.
    Three,
}

impl ShDegree {
    /// Number of SH coefficients per splat for this degree.
    pub fn coeff_count(self) -> usize {
        match self {
            ShDegree::Zero => 3,
            ShDegree::One => 12,
            ShDegree::Three => 48,
        }
    }
}

/// Opaque handle to an uploaded Gaussian splat set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GaussianSplatId(pub(crate) usize);

/// Upload data for a Gaussian splat set. Submitted once via
/// `resources_mut().upload_gaussian_splats(data)`.
pub struct GaussianSplatData {
    /// Object-space center positions, one [f32;3] per splat.
    pub positions: Vec<[f32; 3]>,
    /// Scale (positive floats, world-space metres) per splat, one [f32;3].
    pub scales: Vec<[f32; 3]>,
    /// Unit quaternion rotation per splat [x, y, z, w].
    pub rotations: Vec<[f32; 4]>,
    /// Opacity per splat in [0, 1].
    pub opacities: Vec<f32>,
    /// SH coefficients. Length must equal `positions.len() * sh_degree.coeff_count()`.
    /// For ShDegree::Zero these are [r, g, b] base colours per splat.
    pub sh_coefficients: Vec<f32>,
    /// SH degree for this splat set.
    pub sh_degree: ShDegree,
}

impl Default for GaussianSplatData {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            scales: Vec::new(),
            rotations: Vec::new(),
            opacities: Vec::new(),
            sh_coefficients: Vec::new(),
            sh_degree: ShDegree::Zero,
        }
    }
}

/// Per-frame reference to an uploaded Gaussian splat set.
#[derive(Clone)]
#[non_exhaustive]
pub struct GaussianSplatItem {
    /// Handle to the uploaded splat set.
    pub id: GaussianSplatId,
    /// World-space model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for GaussianSplatItem {
    fn default() -> Self {
        Self {
            id: GaussianSplatId(usize::MAX),
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Decal item (D1)
// ---------------------------------------------------------------------------

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
