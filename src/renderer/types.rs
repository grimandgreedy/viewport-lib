//! `ViewportRenderer` : the main entry point for the viewport library.
//!
//! Wraps [`ViewportGpuResources`] and provides `prepare()` / `paint()` methods
//! that take raw `wgpu` types. GUI framework adapters (e.g. the egui
//! `CallbackTrait` impl in the application crate) delegate to these methods.

use crate::interaction::gizmo::{GizmoAxis, GizmoMode};
use crate::interaction::snap::ConstraintOverlay;
use crate::resources::{CameraUniform, ColormapId};
use crate::scene::material::Material;

/// Minimum scene item count to activate the instanced draw path.
/// Use instancing for any scene with more than 1 object. The per-object path
/// writes uniforms into a per-mesh buffer, so two scene nodes sharing the same
/// mesh would clobber each other. Instancing avoids this by keeping per-item
/// data in a separate instance buffer indexed by draw-call range.
pub(super) const INSTANCING_THRESHOLD: usize = 1;

/// A batch of instances sharing the same mesh and material textures, drawn in one call.
#[derive(Debug, Clone)]
pub(crate) struct InstancedBatch {
    pub mesh_id: crate::resources::mesh_store::MeshId,
    pub texture_id: Option<u64>,
    pub normal_map_id: Option<u64>,
    pub ao_map_id: Option<u64>,
    pub instance_offset: u32,
    pub instance_count: u32,
    pub is_transparent: bool,
}

// ---------------------------------------------------------------------------
// Section view / clip plane / clip volume
// ---------------------------------------------------------------------------

/// A world-space half-space clipping plane for section views.
///
/// A fragment at world position `p` is discarded if `dot(p, normal) + distance < 0`.
///
/// This type is kept `pub(crate)` for internal use by `ClipPlaneController` session
/// storage and `volumes.rs` cap fill. Public API uses [`ClipObject`] instead.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClipPlane {
    /// Unit normal of the clip plane (pointing into the preserved half-space).
    pub normal: [f32; 3],
    /// Signed distance from the origin along `normal`.
    pub distance: f32,
    /// Whether this plane is active. Inactive planes are ignored.
    pub enabled: bool,
    /// Cap fill color override. `None` = use the clipped object's material base_color.
    pub cap_color: Option<[f32; 4]>,
}

/// The shape of a [`ClipObject`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClipShape {
    /// Half-space plane : fragments where `dot(p, normal) + distance >= 0` are kept.
    Plane {
        /// Unit normal pointing into the preserved half-space.
        normal: [f32; 3],
        /// Signed distance from origin along `normal`.
        distance: f32,
        /// Cap fill color override. `None` = use the clipped mesh's base_color.
        cap_color: Option<[f32; 4]>,
    },
    /// Oriented box : fragments inside the box are kept.
    Box {
        center: [f32; 3],
        half_extents: [f32; 3],
        /// 3×3 rotation matrix columns.
        orientation: [[f32; 3]; 3],
    },
    /// Sphere : fragments inside the sphere are kept.
    Sphere { center: [f32; 3], radius: f32 },
}

/// A clip object : defines a clipping region and optional visual boundary rendering.
///
/// Push into `EffectsFrame::clip_objects` each frame. Up to 6 `Plane` variants are
/// supported simultaneously; only the first `Box` or `Sphere` variant takes effect
/// (subsequent ones are silently ignored).
///
/// Set `color` to `Some(rgba)` to have the renderer draw the clip boundary automatically.
/// For planes this produces a semi-transparent fill quad + border; for box/sphere, a
/// wireframe outline. Leave `color` as `None` for silent clipping with no visual.
///
/// The `hovered` and `active` flags are written by `ClipPlaneController` and read by
/// the renderer to vary the plane overlay appearance (brighter when hovered, tinted when active).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipObject {
    pub shape: ClipShape,
    /// RGBA boundary color. `None` = no visual drawn.
    pub color: Option<[f32; 4]>,
    /// Whether this object is active. Disabled objects are ignored entirely.
    pub enabled: bool,
    /// Visual and hit-test half-extent for `Plane` shapes (world units). Default `4.5`.
    pub extent: f32,
    /// Hover state : set by `ClipPlaneController`, read by renderer.
    pub hovered: bool,
    /// Active drag state : set by `ClipPlaneController`, read by renderer.
    pub active: bool,
}

impl Default for ClipObject {
    fn default() -> Self {
        Self {
            shape: ClipShape::Plane {
                normal: [0.0, 0.0, 1.0],
                distance: 0.0,
                cap_color: None,
            },
            color: None,
            enabled: true,
            extent: 4.5,
            hovered: false,
            active: false,
        }
    }
}

impl ClipObject {
    /// Create a half-space plane clip object.
    pub fn plane(normal: [f32; 3], distance: f32) -> Self {
        Self {
            shape: ClipShape::Plane {
                normal,
                distance,
                cap_color: None,
            },
            ..Default::default()
        }
    }
    /// Create an oriented box clip object.
    pub fn box_shape(center: [f32; 3], half_extents: [f32; 3], orientation: [[f32; 3]; 3]) -> Self {
        Self {
            shape: ClipShape::Box {
                center,
                half_extents,
                orientation,
            },
            ..Default::default()
        }
    }
    /// Create a sphere clip object.
    pub fn sphere(center: [f32; 3], radius: f32) -> Self {
        Self {
            shape: ClipShape::Sphere { center, radius },
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Post-processing settings
// ---------------------------------------------------------------------------

/// Tone mapping operator applied when HDR post-processing is enabled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToneMapping {
    /// Reinhard tone mapping (simple, good for scenes without extreme HDR).
    Reinhard,
    /// ACES filmic tone mapping (cinematic look, recommended).
    #[default]
    Aces,
    /// Khronos Neutral tone mapping (perceptually uniform).
    KhronosNeutral,
}

/// Optional post-processing effects applied after the main render pass.
///
/// All fields default to disabled/off for backward compatibility.
#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    /// Enable the HDR render target and tone mapping pipeline.
    /// When `false`, the viewport renders directly to the output surface (LDR).
    pub enabled: bool,
    /// Tone mapping operator. Default: `Aces`.
    pub tone_mapping: ToneMapping,
    /// Pre-tone-mapping exposure multiplier. Default: `1.0`.
    pub exposure: f32,
    /// Enable screen-space ambient occlusion. Requires `enabled = true`.
    pub ssao: bool,
    /// Enable bloom. Requires `enabled = true`.
    pub bloom: bool,
    /// HDR luminance threshold for bloom extraction. Default: `1.0`.
    pub bloom_threshold: f32,
    /// Bloom contribution multiplier. Default: `0.1`.
    pub bloom_intensity: f32,
    /// Enable FXAA (Fast Approximate Anti-Aliasing) fullscreen pass. Requires `enabled = true`.
    pub fxaa: bool,
    /// Supersampling anti-aliasing factor. 1 = off, 2 = 2×, 4 = 4×.
    ///
    /// When `> 1`, scene geometry is rendered at `ssaa_factor × resolution` and downsampled
    /// before post-processing. Produces sharper edges than FXAA at the cost of rendering
    /// `ssaa_factor²` times more pixels. Intended for offline/screenshot use, not interactive
    /// rendering. Requires `enabled = true`.
    pub ssaa_factor: u32,
    /// Enable screen-space contact shadows (thin shadows at object-ground contact). Requires `enabled = true`.
    pub contact_shadows: bool,
    /// Maximum ray-march distance in view space. Default: 0.5.
    pub contact_shadow_max_distance: f32,
    /// Number of ray-march steps. Default: 16.
    pub contact_shadow_steps: u32,
    /// Depth thickness threshold for occlusion test. Default: 0.1.
    pub contact_shadow_thickness: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            tone_mapping: ToneMapping::Aces,
            exposure: 1.0,
            ssao: false,
            bloom: false,
            bloom_threshold: 1.0,
            bloom_intensity: 0.1,
            fxaa: false,
            ssaa_factor: 1,
            contact_shadows: false,
            contact_shadow_max_distance: 0.5,
            contact_shadow_steps: 16,
            contact_shadow_thickness: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Lighting configuration types
// ---------------------------------------------------------------------------

/// Light source type.
///
/// `Directional` emits parallel rays from a fixed direction (infinite distance).
/// `Point` emits rays from a position with distance-based falloff.
/// `Spot` emits a cone of light with inner (full-intensity) and outer (cutoff) angles.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum LightKind {
    /// Infinitely distant light with parallel rays (e.g. the sun).
    Directional {
        /// World-space direction the light travels toward (not the source direction).
        direction: [f32; 3],
    },
    /// Omnidirectional point light with distance falloff.
    Point {
        /// World-space position of the light source.
        position: [f32; 3],
        /// Maximum range (world units) beyond which the light contributes nothing.
        range: f32,
    },
    /// Cone-shaped spotlight.
    Spot {
        /// World-space position of the light source.
        position: [f32; 3],
        /// World-space direction the cone points toward.
        direction: [f32; 3],
        /// Maximum range (world units).
        range: f32,
        /// Inner cone half-angle (radians) : full intensity within this cone.
        inner_angle: f32,
        /// Outer cone half-angle (radians) : light fades to zero at this angle.
        outer_angle: f32,
    },
}

/// A single light source with color and intensity.
#[derive(Clone, Debug)]
pub struct LightSource {
    /// The type and geometric parameters of this light.
    pub kind: LightKind,
    /// RGB light color in linear 0..1. Default [1.0, 1.0, 1.0].
    pub color: [f32; 3],
    /// Intensity multiplier. Default 1.0.
    pub intensity: f32,
}

impl Default for LightSource {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional {
                // Surface-to-light direction. Z is up in the default coordinate system.
                // ~65° elevation: mostly overhead, slight front-right bias.
                direction: [0.4, 0.3, 1.5],
            },
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        }
    }
}

/// Shadow filtering mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowFilter {
    /// Standard 3×3 PCF (fast).
    #[default]
    Pcf,
    /// Percentage-Closer Soft Shadows (variable penumbra width, higher cost).
    Pcss,
}

/// Per-frame lighting configuration for the viewport.
///
/// Supports up to 8 light sources. Only `lights[0]` casts shadows.
/// Blinn-Phong shading coefficients (ambient, diffuse, specular, shininess) have
/// moved to per-object [`Material`] structs.
#[derive(Clone, Debug)]
pub struct LightingSettings {
    /// Active light sources (max 8). Default: one directional light.
    pub lights: Vec<LightSource>,
    /// Shadow map depth bias to reduce shadow acne. Default: 0.0001.
    pub shadow_bias: f32,
    /// Whether shadow maps are computed and sampled. Default: true.
    pub shadows_enabled: bool,
    /// Sky color for hemisphere ambient. Default [0.8, 0.9, 1.0].
    pub sky_color: [f32; 3],
    /// Ground color for hemisphere ambient. Default [0.3, 0.2, 0.1].
    pub ground_color: [f32; 3],
    /// Hemisphere ambient intensity. 0.0 = disabled. Default 0.0.
    pub hemisphere_intensity: f32,
    /// Override the shadow frustum half-extent (world units). None = auto (20.0).
    /// Tighter values improve shadow map texel density and reduce contact-shadow penumbra.
    pub shadow_extent_override: Option<f32>,

    /// Number of cascaded shadow map splits (1–4). Default: 4.
    pub shadow_cascade_count: u32,
    /// Blend factor between logarithmic and linear cascade splits (0.0 = linear, 1.0 = log).
    /// Default: 0.75. Higher values allocate more resolution near the camera.
    pub cascade_split_lambda: f32,
    /// Shadow atlas resolution (width = height). Default: 4096.
    /// Each cascade tile is `atlas_resolution / 2`.
    pub shadow_atlas_resolution: u32,
    /// Shadow filtering mode. Default: PCF.
    pub shadow_filter: ShadowFilter,
    /// PCSS light source radius in shadow-map UV space. Controls penumbra width. Default: 0.02.
    pub pcss_light_radius: f32,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            lights: vec![LightSource::default()],
            shadow_bias: 0.0001,
            shadows_enabled: true,
            sky_color: [0.8, 0.9, 1.0],
            ground_color: [0.3, 0.2, 0.1],
            hemisphere_intensity: 0.5,
            shadow_extent_override: None,
            shadow_cascade_count: 4,
            cascade_split_lambda: 0.75,
            shadow_atlas_resolution: 4096,
            shadow_filter: ShadowFilter::Pcf,
            pcss_light_radius: 0.02,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame data types
// ---------------------------------------------------------------------------

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
    /// Whether this object is selected (drives orange tint in WGSL).
    pub selected: bool,
    /// Whether this object is visible. Hidden objects are not drawn.
    pub visible: bool,
    /// Whether to render per-vertex normal visualization lines for this object.
    pub show_normals: bool,
    /// Per-object material (color, shading coefficients, opacity, texture).
    pub material: Material,
    /// Named scalar attribute to colour by. `None` = use material base colour.
    pub active_attribute: Option<crate::resources::AttributeRef>,
    /// Explicit scalar range `(min, max)`. `None` = use auto-range computed at upload time.
    pub scalar_range: Option<(f32, f32)>,
    /// Colormap to use for scalar colouring. Ignored when `active_attribute` is `None`.
    pub colormap_id: Option<crate::resources::ColormapId>,
    /// RGBA color for NaN scalar values. `None` = discard (fully transparent).
    pub nan_color: Option<[f32; 4]>,
    /// GPU pick identifier for this surface. [`PickId::NONE`] = not pickable.
    ///
    /// The renderer only includes surfaces with a nonzero pick ID in the GPU
    /// pick pass. Set a nonzero value for any surface the user should be able to
    /// click to select. Helper geometry and transient previews that should not
    /// participate in picking should leave this at the default [`PickId::NONE`].
    pub pick_id: PickId,
}

impl Default for SceneRenderItem {
    fn default() -> Self {
        Self {
            mesh_id: crate::resources::mesh_store::MeshId(0),
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            selected: false,
            visible: true,
            show_normals: false,
            material: Material::default(),
            active_attribute: None,
            scalar_range: None,
            colormap_id: None,
            nan_color: None,
            pick_id: PickId::NONE,
        }
    }
}

/// Scalar bar (colour legend) overlay descriptor.
///
/// Not part of `FrameData` : the application draws scalar bars after `show_viewport()`
/// returns, using `egui::Painter` and the colormap data from
/// [`ViewportGpuResources::get_colormap_rgba`](crate::resources::ViewportGpuResources::get_colormap_rgba).
#[derive(Debug, Clone)]
pub struct ScalarBar {
    /// Colormap to display.
    pub colormap_id: crate::resources::ColormapId,
    /// Scalar value at the low end of the gradient.
    pub scalar_min: f32,
    /// Scalar value at the high end of the gradient.
    pub scalar_max: f32,
    /// Title label shown above the bar.
    pub title: String,
    /// Corner of the viewport rect to anchor the bar to.
    pub anchor: ScalarBarAnchor,
    /// Whether to draw the bar vertically or horizontally.
    pub orientation: ScalarBarOrientation,
}

/// Anchor corner for a [`ScalarBar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBarAnchor {
    /// Top-left corner of the viewport.
    TopLeft,
    /// Top-right corner of the viewport.
    TopRight,
    /// Bottom-left corner of the viewport.
    BottomLeft,
    /// Bottom-right corner of the viewport.
    BottomRight,
}

/// Orientation of a [`ScalarBar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBarOrientation {
    /// Gradient runs from bottom (min) to top (max).
    Vertical,
    /// Gradient runs from left (min) to right (max).
    Horizontal,
}

// ---------------------------------------------------------------------------
// SciVis Phase B : point cloud and glyph renderers
// ---------------------------------------------------------------------------

/// Render mode for point cloud items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointRenderMode {
    /// GPU point primitives with `point_size` uniform (fastest, no shading).
    #[default]
    ScreenSpaceCircle,
    // Future: BillboardQuad, FixedSphere
}

/// A point cloud item to render in the viewport.
#[non_exhaustive]
pub struct PointCloudItem {
    /// World-space positions (one vec3 per point).
    pub positions: Vec<[f32; 3]>,
    /// Optional per-point RGBA colors in linear `[0,1]`. If empty, uses `default_color`.
    pub colors: Vec<[f32; 4]>,
    /// Optional per-point scalar values for LUT coloring. If non-empty, overrides `colors`.
    pub scalars: Vec<f32>,
    /// Scalar range for LUT mapping. None = auto from min/max of `scalars`.
    pub scalar_range: Option<(f32, f32)>,
    /// Colormap for scalar coloring. None = use default builtin (viridis).
    pub colormap_id: Option<ColormapId>,
    /// Screen-space point size in pixels. Default: 4.0.
    pub point_size: f32,
    /// Fallback color when neither `colors` nor `scalars` are provided.
    pub default_color: [f32; 4],
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Render mode. Default: ScreenSpaceCircle.
    pub render_mode: PointRenderMode,
    /// Unique ID for picking. 0 = not pickable.
    pub id: u64,
}

impl Default for PointCloudItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            colors: Vec::new(),
            scalars: Vec::new(),
            scalar_range: None,
            colormap_id: None,
            point_size: 4.0,
            default_color: [1.0, 1.0, 1.0, 1.0],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            render_mode: PointRenderMode::ScreenSpaceCircle,
            id: 0,
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
    /// Optional per-instance scalar values for LUT coloring. Empty = color by magnitude.
    pub scalars: Vec<f32>,
    /// Scalar range for LUT mapping. None = auto from data.
    pub scalar_range: Option<(f32, f32)>,
    /// Colormap for scalar coloring. None = use default builtin (viridis).
    pub colormap_id: Option<ColormapId>,
    /// Glyph shape. Default: Arrow.
    pub glyph_type: GlyphType,
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Unique ID for picking. 0 = not pickable.
    pub id: u64,
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
            colormap_id: None,
            glyph_type: GlyphType::Arrow,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            id: 0,
        }
    }
}

/// A volume item to render via GPU ray-marching.
///
/// The caller uploads a 3D scalar field via [`ViewportGpuResources::upload_volume`](crate::resources::ViewportGpuResources::upload_volume) and
/// receives a [`VolumeId`](crate::resources::VolumeId). Each frame, submit a `VolumeItem` referencing that id plus
/// transfer function and display parameters.
#[non_exhaustive]
pub struct VolumeItem {
    /// Reference to a previously uploaded 3D texture.
    pub volume_id: crate::resources::VolumeId,
    /// Color transfer function LUT. `None` = use default builtin (viridis).
    pub color_lut: Option<ColormapId>,
    /// Opacity transfer function LUT. `None` = linear ramp (0 at min, 1 at max).
    pub opacity_lut: Option<ColormapId>,
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
    /// Color and opacity to use for NaN scalar samples. `None` = skip NaN samples entirely
    /// (same as current behaviour: discard). `Some([r, g, b, a])` = render NaN voxels with
    /// this fixed RGBA color instead of sampling the transfer function.
    pub nan_color: Option<[f32; 4]>,
}

impl Default for VolumeItem {
    fn default() -> Self {
        Self {
            volume_id: crate::resources::VolumeId(0),
            color_lut: None,
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
            nan_color: None,
        }
    }
}

/// A polyline (stream tracer) item to render in the viewport.
///
/// All streamlines for one source are concatenated into a single vertex buffer.
/// `strip_lengths` records how many vertices belong to each individual streamline.
#[non_exhaustive]
pub struct PolylineItem {
    /// World-space positions for all streamlines, concatenated.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex scalar values (same length as `positions`). Empty = no scalar coloring.
    pub scalars: Vec<f32>,
    /// Number of vertices per individual streamline strip.
    pub strip_lengths: Vec<u32>,
    /// Scalar range for LUT mapping. None = auto from min/max of `scalars`.
    pub scalar_range: Option<(f32, f32)>,
    /// Colormap for scalar coloring. None = viridis.
    pub colormap_id: Option<ColormapId>,
    /// Fallback color when `scalars` is empty.
    pub default_color: [f32; 4],
    /// Hardware line width in pixels (may be clamped to 1 by some GPU drivers).
    pub line_width: f32,
    /// Unique ID for identification. 0 = not pickable.
    pub id: u64,
}

impl Default for PolylineItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            scalars: Vec::new(),
            strip_lengths: Vec::new(),
            scalar_range: None,
            colormap_id: None,
            default_color: [0.9, 0.92, 0.96, 1.0],
            line_width: 2.0,
            id: 0,
        }
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
    pub color: [f32; 4],
    /// Unique ID (reserved for future picking support).  Default: `0`.
    pub id: u64,
}

impl Default for StreamtubeItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            strip_lengths: Vec::new(),
            radius: 0.05,
            color: [1.0, 1.0, 1.0, 1.0],
            id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 10A : Camera frustum wireframe
// ---------------------------------------------------------------------------

/// A renderable camera frustum wireframe item.
///
/// Converted to [`PolylineItem`] geometry in `prepare.rs` (no new GPU pipeline).
/// The frustum is drawn as near quad + far quad + 4 lateral edges, with an
/// optional image-plane quad at a configurable depth.
///
/// Use [`CameraFrustumItem::camera_target`] to get a fly-to target that frames
/// the frustum from a comfortable standoff distance.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CameraFrustumItem {
    /// View-to-world transform (the camera's world-space pose).
    ///
    /// Pass `camera.view_matrix().inverse().to_cols_array_2d()` for the current
    /// viewport camera, or any other camera-world transform.
    pub pose: [[f32; 4]; 4],
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// Near clip distance (world units).
    pub near: f32,
    /// Far clip distance (world units).
    pub far: f32,
    /// RGBA line color. Default: `[0.8, 0.8, 0.9, 1.0]` (light blue-grey).
    pub color: [f32; 4],
    /// Screen-space line width in pixels. Default: `2.0`.
    pub line_width: f32,
    /// If `Some(d)`, draw a closed quad at depth `d` (world units) inside the frustum.
    /// Useful to visualise the image plane of a camera.
    pub image_plane_depth: Option<f32>,
}

impl Default for CameraFrustumItem {
    fn default() -> Self {
        Self {
            pose: glam::Mat4::IDENTITY.to_cols_array_2d(),
            fov_y: std::f32::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 10.0,
            color: [0.8, 0.8, 0.9, 1.0],
            line_width: 2.0,
            image_plane_depth: None,
        }
    }
}

impl CameraFrustumItem {
    /// Compute the world-space corners of a frustum plane at depth `d`.
    ///
    /// Returns `[top_left, top_right, bottom_right, bottom_left]` in world space.
    fn plane_corners(&self, d: f32) -> [[f32; 3]; 4] {
        let half_h = (self.fov_y * 0.5).tan() * d;
        let half_w = half_h * self.aspect;
        let pose = glam::Mat4::from_cols_array_2d(&self.pose);
        let corners_cam = [
            glam::vec3(-half_w, half_h, -d),
            glam::vec3(half_w, half_h, -d),
            glam::vec3(half_w, -half_h, -d),
            glam::vec3(-half_w, -half_h, -d),
        ];
        corners_cam.map(|c| {
            let w = pose.transform_point3(c);
            [w.x, w.y, w.z]
        })
    }

    /// Convert this frustum into a [`PolylineItem`] for the polyline pipeline.
    ///
    /// Produces: near quad strip (5 verts), far quad strip (5 verts),
    /// 4 lateral edge strips (2 verts each), and optionally an image-plane
    /// quad strip (5 verts).
    pub(crate) fn to_polyline(&self) -> PolylineItem {
        let near = self.plane_corners(self.near);
        let far = self.plane_corners(self.far);

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut strip_lengths: Vec<u32> = Vec::new();

        // Near quad (closed loop: TL->TR->BR->BL->TL)
        positions.extend_from_slice(&[near[0], near[1], near[2], near[3], near[0]]);
        strip_lengths.push(5);

        // Far quad
        positions.extend_from_slice(&[far[0], far[1], far[2], far[3], far[0]]);
        strip_lengths.push(5);

        // Lateral edges (near corner -> far corner, for each of 4 corners)
        for i in 0..4 {
            positions.extend_from_slice(&[near[i], far[i]]);
            strip_lengths.push(2);
        }

        // Optional image-plane quad
        if let Some(d) = self.image_plane_depth {
            let ip = self.plane_corners(d);
            positions.extend_from_slice(&[ip[0], ip[1], ip[2], ip[3], ip[0]]);
            strip_lengths.push(5);
        }

        PolylineItem {
            positions,
            strip_lengths,
            default_color: self.color,
            line_width: self.line_width,
            ..PolylineItem::default()
        }
    }

    /// Compute a [`crate::camera::CameraTarget`] that frames this frustum.
    ///
    /// `standoff_factor` controls how far back the viewing camera sits relative
    /// to the frustum diagonal (2.5 is a comfortable default). The returned
    /// orientation faces the frustum from its front (along the frustum's +Z axis).
    ///
    /// Feed the result directly into [`crate::camera::CameraAnimator::fly_to`]:
    ///
    /// ```rust,ignore
    /// let t = frustum.camera_target(2.5);
    /// animator.fly_to(&camera, t.center, t.distance, t.orientation, 1.0);
    /// ```
    pub fn camera_target(&self, standoff_factor: f32) -> crate::camera::CameraTarget {
        let near = self.plane_corners(self.near);
        let far = self.plane_corners(self.far);

        // World-space center: midpoint of all 8 corners.
        let mut sum = glam::Vec3::ZERO;
        for c in near.iter().chain(far.iter()) {
            sum += glam::Vec3::from(*c);
        }
        let center = sum / 8.0;

        // Distance: half-diagonal of the frustum bounding box, scaled by standoff.
        let mut max_dist_sq: f32 = 0.0;
        for c in near.iter().chain(far.iter()) {
            let d = (glam::Vec3::from(*c) - center).length_squared();
            if d > max_dist_sq {
                max_dist_sq = d;
            }
        }
        let distance = max_dist_sq.sqrt() * standoff_factor;

        // Orientation: look from camera +Z axis (frustum's back) toward center.
        let pose = glam::Mat4::from_cols_array_2d(&self.pose);
        // Frustum's world-space forward (into scene) is -Z of the camera frame.
        // We want to view the frustum from the +Z side (behind the camera).
        let cam_z_world = pose.transform_vector3(glam::Vec3::Z); // frustum's +Z (back)
        let eye = center + cam_z_world.normalize() * distance;
        let forward = (center - eye).normalize();
        // Build orientation quaternion: look along `forward` with world-up hint.
        let up_hint = if forward.dot(glam::Vec3::Z).abs() > 0.99 {
            glam::Vec3::Y
        } else {
            glam::Vec3::Z
        };
        let right = forward.cross(up_hint).normalize();
        let up = right.cross(forward).normalize();
        let rot = glam::Mat3::from_cols(right, up, -forward);
        let orientation = glam::Quat::from_mat3(&rot);

        crate::camera::CameraTarget {
            center,
            distance,
            orientation,
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
/// The image is drawn after all 3D geometry (no depth test) and anchored to
/// one of the viewport corners or the center.
///
/// `depth_composite: true` is reserved for a future phase; set it to `false`.
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
    /// Reserved : must be `false` in Phase 10.
    pub depth_composite: bool,
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
            depth_composite: false,
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
// 0.2.0 grouped frame API types
// ---------------------------------------------------------------------------

/// Canonical renderer-facing camera state.
///
/// Replaces the flat camera fields that were previously scattered across
/// `FrameData`. Application-side orbit cameras resolve into this type
/// before frame submission.
#[derive(Debug, Clone)]
pub struct RenderCamera {
    /// World-to-view transform matrix.
    pub view: glam::Mat4,
    /// View-to-clip (projection) matrix.
    pub projection: glam::Mat4,
    /// Camera eye position in world space.
    pub eye_position: [f32; 3],
    /// Camera forward direction in world space.
    pub forward: [f32; 3],
    /// Camera orientation quaternion.
    pub orientation: glam::Quat,
    /// Near clip plane distance. Default: 0.1.
    pub near: f32,
    /// Far clip plane distance. Default: 1000.0.
    pub far: f32,
    /// Vertical field of view in radians. Default: PI/4.
    pub fov: f32,
    /// Aspect ratio (width / height). Default: 1.333.
    pub aspect: f32,
}

impl RenderCamera {
    /// Build the GPU-facing camera uniform from this camera's state.
    pub fn camera_uniform(&self) -> CameraUniform {
        let vp = self.view_proj();
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            eye_pos: self.eye_position,
            _pad: 0.0,
            forward: self.forward,
            _pad1: 0.0,
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            view: self.view.to_cols_array_2d(),
        }
    }

    /// Combined view-projection matrix (projection * view).
    pub fn view_proj(&self) -> glam::Mat4 {
        self.projection * self.view
    }

    /// Build a `RenderCamera` from an app-side [`Camera`](crate::camera::Camera).
    ///
    /// This is the intended conversion path: resolve the orbit camera to a
    /// `RenderCamera` once per frame and pass it through `CameraFrame`.
    pub fn from_camera(cam: &crate::camera::Camera) -> Self {
        let eye = cam.eye_position();
        let forward = (cam.center - eye).normalize_or_zero();
        Self {
            view: cam.view_matrix(),
            projection: cam.proj_matrix(),
            eye_position: eye.to_array(),
            forward: forward.to_array(),
            orientation: cam.orientation,
            near: cam.znear,
            far: cam.zfar,
            fov: cam.fov_y,
            aspect: cam.aspect,
        }
    }
}

impl Default for RenderCamera {
    fn default() -> Self {
        Self {
            view: glam::Mat4::IDENTITY,
            projection: glam::Mat4::IDENTITY,
            eye_position: [0.0, 0.0, 5.0],
            forward: [0.0, 0.0, -1.0],
            orientation: glam::Quat::IDENTITY,
            near: 0.1,
            far: 1000.0,
            fov: std::f32::consts::FRAC_PI_4,
            aspect: 1.333,
        }
    }
}

/// Camera submission state for one frame.
///
/// Groups the canonical render camera with viewport sizing and multi-viewport
/// slot index. This is the single owner of all camera-derived state submitted
/// to the renderer each frame.
#[non_exhaustive]
pub struct CameraFrame {
    /// Canonical renderer-facing camera state.
    pub render_camera: RenderCamera,
    /// Viewport size in physical pixels (width, height). Default: [800.0, 600.0].
    pub viewport_size: [f32; 2],
    /// Multi-viewport slot index. Default: 0 (single-viewport mode).
    pub viewport_index: usize,
}

impl Default for CameraFrame {
    fn default() -> Self {
        Self {
            render_camera: RenderCamera::default(),
            viewport_size: [800.0, 600.0],
            viewport_index: 0,
        }
    }
}

impl CameraFrame {
    /// Build a camera frame from a render camera and viewport size.
    pub fn new(render_camera: RenderCamera, viewport_size: [f32; 2]) -> Self {
        Self {
            render_camera,
            viewport_size,
            viewport_index: 0,
        }
    }

    /// Build a camera frame from an app-side camera and viewport size.
    pub fn from_camera(cam: &crate::camera::Camera, viewport_size: [f32; 2]) -> Self {
        Self::new(RenderCamera::from_camera(cam), viewport_size)
    }

    /// Set the multi-viewport slot index for this camera frame.
    pub fn with_viewport_index(mut self, viewport_index: usize) -> Self {
        self.viewport_index = viewport_index;
        self
    }
}

/// Surface submission seam for world-space geometry.
///
/// For 0.2.0, only `Flat` submission is supported. This enum exists to
/// provide an explicit seam for future large-scene or chunked submission
/// without changing the `SceneFrame` public type.
#[non_exhaustive]
pub enum SurfaceSubmission {
    /// A flat list of scene render items (current behavior).
    Flat(Vec<SceneRenderItem>),
}

impl Default for SurfaceSubmission {
    fn default() -> Self {
        SurfaceSubmission::Flat(Vec::new())
    }
}

/// World-space scene content for one frame.
///
/// Groups all renderable world-space content submitted to the renderer.
/// Surfaces are submitted through the [`SurfaceSubmission`] seam; scientific
/// visualization primitives are first-class members alongside surfaces.
#[non_exhaustive]
pub struct SceneFrame {
    /// Scene version counter from `Scene::version()`. Default: 0 (triggers rebuild on first frame).
    ///
    /// The renderer uses this to skip batch rebuild and GPU upload when scene content
    /// has not changed since the previous frame.
    pub generation: u64,
    /// Surface geometry submission (opaque and transparent meshes).
    pub surfaces: SurfaceSubmission,
    /// Point cloud items to render this frame.
    pub point_clouds: Vec<PointCloudItem>,
    /// Instanced glyph items to render this frame.
    pub glyphs: Vec<GlyphItem>,
    /// Polyline (streamline) items to render this frame.
    pub polylines: Vec<PolylineItem>,
    /// Volume items to render this frame via GPU ray-marching.
    pub volumes: Vec<VolumeItem>,
    /// Isoline (contour line) items to render on mesh surfaces.
    pub isolines: Vec<crate::geometry::isoline::IsolineItem>,
    /// Streamtube items to render this frame.
    pub streamtube_items: Vec<StreamtubeItem>,
    /// Camera frustum wireframe items to render this frame (Phase 10).
    pub camera_frustums: Vec<CameraFrustumItem>,
    /// Screen-space image overlay items to render this frame (Phase 10).
    pub screen_images: Vec<ScreenImageItem>,
}

impl Default for SceneFrame {
    fn default() -> Self {
        Self {
            generation: 0,
            surfaces: SurfaceSubmission::default(),
            point_clouds: Vec::new(),
            glyphs: Vec::new(),
            polylines: Vec::new(),
            volumes: Vec::new(),
            isolines: Vec::new(),
            streamtube_items: Vec::new(),
            camera_frustums: Vec::new(),
            screen_images: Vec::new(),
        }
    }
}

impl SceneFrame {
    /// Build a scene frame from a surface submission.
    pub fn new(surfaces: SurfaceSubmission) -> Self {
        Self {
            surfaces,
            ..Self::default()
        }
    }

    /// Build a scene frame from a flat list of surface render items.
    pub fn from_surface_items(items: Vec<SceneRenderItem>) -> Self {
        Self::new(SurfaceSubmission::Flat(items))
    }

    /// Build a scene frame by collecting render items from a [`Scene`](crate::scene::scene::Scene).
    ///
    /// Calls [`Scene::collect_render_items`](crate::scene::scene::Scene::collect_render_items)
    /// and stamps `generation` with the current scene version so the renderer can
    /// skip batch rebuilds on unchanged frames.
    ///
    /// This is the preferred constructor for the common single-viewport path.
    /// Use [`SceneFrame::from_surface_items`] when you need to assemble render items manually.
    pub fn from_scene(
        scene: &mut crate::scene::scene::Scene,
        selection: &crate::interaction::selection::Selection,
    ) -> Self {
        let items = scene.collect_render_items(selection);
        Self {
            generation: scene.version(),
            surfaces: SurfaceSubmission::Flat(items),
            ..Self::default()
        }
    }
}

/// Viewport presentation settings for one frame.
///
/// Groups background, grid, and axes indicator : the viewport chrome that is
/// independent of world-space content.
#[non_exhaustive]
pub struct ViewportFrame {
    /// Optional background/clear color [r, g, b, a]. None = adapter default.
    pub background_color: Option<[f32; 4]>,
    /// Whether to render the scene in wireframe mode. Default: false.
    pub wireframe_mode: bool,
    /// Whether to render the ground-plane grid. Default: false.
    pub show_grid: bool,
    /// Grid cell size in world units. Zero = camera-distance-based adaptive spacing.
    pub grid_cell_size: f32,
    /// Half-extent of the grid in world units. Zero = 1000 (effectively infinite).
    pub grid_half_extent: f32,
    /// World-space Z coordinate of the grid plane (3D mode only, Z-up). Default: 0.0.
    pub grid_z: f32,
    /// Whether to draw the axes orientation indicator overlay. Default: true.
    pub show_axes_indicator: bool,
}

impl Default for ViewportFrame {
    fn default() -> Self {
        Self {
            background_color: None,
            wireframe_mode: false,
            show_grid: false,
            grid_cell_size: 0.0,
            grid_half_extent: 0.0,
            grid_z: 0.0,
            show_axes_indicator: true,
        }
    }
}

/// Interaction and selection visualization state for one frame.
///
/// Groups the gizmo, selection overlays, constraint guides, outline, and
/// x-ray state : everything that communicates selection and interaction
/// feedback to the user.
#[non_exhaustive]
pub struct InteractionFrame {
    /// Selection version counter from `Selection::version()`. Default: 0.
    ///
    /// The renderer uses this to skip batch rebuild and GPU upload when selection
    /// state has not changed since the previous frame.
    pub selection_generation: u64,
    /// Gizmo model matrix. Some = selected object exists and gizmo should render.
    pub gizmo_model: Option<glam::Mat4>,
    /// Current gizmo interaction mode.
    pub gizmo_mode: GizmoMode,
    /// Current hovered gizmo axis.
    pub gizmo_hovered: GizmoAxis,
    /// Orientation for gizmo space (identity for world, object orientation for local).
    pub gizmo_space_orientation: glam::Quat,
    /// Constraint guide lines to render this frame.
    pub constraint_overlays: Vec<ConstraintOverlay>,
    /// Draw a stencil-outline ring around selected objects. Default: false.
    pub outline_selected: bool,
    /// RGBA color of the selection outline ring. Default: white [1.0, 1.0, 1.0, 1.0].
    pub outline_color: [f32; 4],
    /// Width of the outline ring in pixels. Default: 2.0.
    pub outline_width_px: f32,
    /// Render selected objects as a semi-transparent x-ray overlay. Default: false.
    pub xray_selected: bool,
    /// RGBA color of the x-ray tint (should have alpha < 1). Default: [0.3, 0.7, 1.0, 0.25].
    pub xray_color: [f32; 4],
}

impl Default for InteractionFrame {
    fn default() -> Self {
        Self {
            selection_generation: 0,
            gizmo_model: None,
            gizmo_mode: GizmoMode::Translate,
            gizmo_hovered: GizmoAxis::None,
            gizmo_space_orientation: glam::Quat::IDENTITY,
            constraint_overlays: Vec::new(),
            outline_selected: false,
            outline_color: [1.0, 1.0, 1.0, 1.0],
            outline_width_px: 2.0,
            xray_selected: false,
            xray_color: [0.3, 0.7, 1.0, 0.25],
        }
    }
}

impl InteractionFrame {
    /// Build an interaction frame stamped with the current selection version.
    ///
    /// Sets `selection_generation` from [`Selection::version`](crate::interaction::selection::Selection::version)
    /// so the renderer can skip overlay rebuilds on unchanged frames.
    /// All other fields remain at their defaults.
    pub fn from_selection(selection: &crate::interaction::selection::Selection) -> Self {
        Self {
            selection_generation: selection.version(),
            ..Self::default()
        }
    }
}

/// Environment map configuration for IBL (image-based lighting) and skybox.
///
/// Ground plane rendering mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GroundPlaneMode {
    /// No ground plane rendered (default, zero overhead).
    #[default]
    None,
    /// Invisible plane that receives and displays shadows only.
    ShadowOnly,
    /// Procedural checkerboard tile pattern.
    Tile,
    /// Flat solid color.
    SolidColor,
}

/// Ground plane configuration for the viewport.
///
/// Renders a large horizontal plane at a configurable world-space Z height.
/// Provides spatial grounding without explicit scene geometry.
#[derive(Clone, Debug)]
pub struct GroundPlane {
    /// Rendering mode. Default: `None` (plane not drawn).
    pub mode: GroundPlaneMode,
    /// World-space Z coordinate of the ground plane. Default: `0.0`.
    pub height: f32,
    /// Ground color for `Tile` and `SolidColor` modes. Default: `[0.3, 0.3, 0.3, 1.0]`.
    pub color: [f32; 4],
    /// Checker tile size in world units (`Tile` mode). Default: `1.0`.
    pub tile_size: f32,
    /// Shadow tint color (`ShadowOnly` mode). Default: `[0.0, 0.0, 0.0, 1.0]`.
    pub shadow_color: [f32; 4],
    /// Maximum shadow opacity (`ShadowOnly` mode). `0.0` = transparent, `1.0` = fully opaque. Default: `0.5`.
    pub shadow_opacity: f32,
}

impl Default for GroundPlane {
    fn default() -> Self {
        Self {
            mode: GroundPlaneMode::None,
            height: 0.0,
            color: [0.3, 0.3, 0.3, 1.0],
            tile_size: 1.0,
            shadow_color: [0.0, 0.0, 0.0, 1.0],
            shadow_opacity: 0.5,
        }
    }
}

/// When set on `EffectsFrame::environment`, the renderer uses the environment
/// map for PBR ambient lighting (irradiance + specular) and optionally renders
/// it as the scene background (skybox).
#[derive(Clone, Debug)]
pub struct EnvironmentMap {
    /// Intensity multiplier for IBL contribution. Default: 1.0.
    pub intensity: f32,
    /// Y-axis rotation in radians. Default: 0.0.
    pub rotation: f32,
    /// Whether to render the environment as a visible skybox background.
    /// When false, IBL still contributes lighting but the background uses
    /// `ViewportFrame::background_color`. Default: true.
    pub show_skybox: bool,
}

impl Default for EnvironmentMap {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            rotation: 0.0,
            show_skybox: true,
        }
    }
}

/// Global rendering effects and modifiers for one frame.
///
/// Groups lighting, clipping, post-processing, compute filtering, and clip
/// volumes : effects that apply globally across the scene rather than to
/// individual objects.
#[non_exhaustive]
pub struct EffectsFrame {
    /// Per-frame lighting configuration.
    pub lighting: LightingSettings,
    /// Active clip objects (planes, boxes, spheres). Max 6 planes + 1 box/sphere.
    /// Default: empty (no clipping).
    pub clip_objects: Vec<ClipObject>,
    /// Whether to render filled caps at clip plane cross-sections. Default: true.
    pub cap_fill_enabled: bool,
    /// Optional post-processing settings. Default: disabled.
    pub post_process: PostProcessSettings,
    /// GPU compute filter items dispatched before the render pass.
    pub compute_filter_items: Vec<ComputeFilterItem>,
    /// Optional environment map for IBL and skybox. Default: None.
    pub environment: Option<EnvironmentMap>,
    /// Ground plane configuration. Default: mode = None (not drawn, zero overhead).
    pub ground_plane: GroundPlane,
}

impl Default for EffectsFrame {
    fn default() -> Self {
        Self {
            lighting: LightingSettings::default(),
            clip_objects: Vec::new(),
            cap_fill_enabled: true,
            post_process: PostProcessSettings::default(),
            compute_filter_items: Vec::new(),
            environment: None,
            ground_plane: GroundPlane::default(),
        }
    }
}

/// Scene-global effects for one frame, consumed by [`ViewportRenderer::prepare_scene`].
///
/// Groups the lighting, environment, and compute-filter configuration that applies
/// to the whole scene (not per-viewport). Construct directly or obtain via
/// [`EffectsFrame::split`].
///
/// # Multi-viewport usage
/// Call [`ViewportRenderer::prepare_scene`] once per frame with this struct.
/// Each viewport's per-viewport effects are passed separately via
/// [`ViewportEffects`] in [`ViewportRenderer::prepare_viewport`].
pub struct SceneEffects<'a> {
    /// Per-frame lighting configuration (drives the shadow pass and light uniform).
    pub lighting: &'a LightingSettings,
    /// Optional environment map for IBL and skybox.
    pub environment: &'a Option<EnvironmentMap>,
    /// GPU compute filter items dispatched before the render pass.
    pub compute_filter_items: &'a [ComputeFilterItem],
}

/// Per-viewport effects for one frame, consumed by [`ViewportRenderer::prepare_viewport`].
///
/// Groups the clip objects and post-processing settings that differ
/// per viewport. Construct directly or obtain via [`EffectsFrame::split`].
///
/// # Multi-viewport usage
/// Pass one `ViewportEffects` per viewport to [`ViewportRenderer::prepare_viewport`].
/// Scene-global effects are passed once via [`SceneEffects`] in
/// [`ViewportRenderer::prepare_scene`].
pub struct ViewportEffects<'a> {
    /// Active clip objects (planes, boxes, spheres).
    pub clip_objects: &'a [ClipObject],
    /// Whether to render filled caps at clip plane cross-sections.
    pub cap_fill_enabled: bool,
    /// Optional post-processing settings (tone mapping, bloom, SSAO).
    pub post_process: &'a PostProcessSettings,
    /// Ground plane configuration for this viewport.
    pub ground_plane: &'a GroundPlane,
}

impl EffectsFrame {
    /// Decompose into scene-global and per-viewport effect references.
    ///
    /// Both halves borrow from `self` and cannot outlive the `EffectsFrame`.
    /// The scene half is passed to [`ViewportRenderer::prepare_scene`]; the
    /// viewport half is passed to [`ViewportRenderer::prepare_viewport`].
    ///
    /// Single-viewport callers can continue using [`ViewportRenderer::prepare`]
    /// directly without calling `split()`.
    pub fn split(&self) -> (SceneEffects<'_>, ViewportEffects<'_>) {
        (
            SceneEffects {
                lighting: &self.lighting,
                environment: &self.environment,
                compute_filter_items: &self.compute_filter_items,
            },
            ViewportEffects {
                clip_objects: &self.clip_objects,
                cap_fill_enabled: self.cap_fill_enabled,
                post_process: &self.post_process,
                ground_plane: &self.ground_plane,
            },
        )
    }
}

/// All data needed to render one frame of the viewport.
///
/// Fields are grouped by responsibility. Build the sub-objects you need,
/// leave others at their `Default`, then call `prepare()` followed by
/// `paint()` or `paint_to()`.
#[non_exhaustive]
pub struct FrameData {
    /// Camera state, viewport size, and viewport slot.
    pub camera: CameraFrame,
    /// World-space scene content (surfaces, point clouds, glyphs, etc.).
    pub scene: SceneFrame,
    /// Viewport presentation settings (background, grid, axes indicator).
    pub viewport: ViewportFrame,
    /// Interaction and selection visualization (gizmo, outline, x-ray).
    pub interaction: InteractionFrame,
    /// Global rendering effects (lighting, clipping, post-process).
    pub effects: EffectsFrame,
}

impl Default for FrameData {
    fn default() -> Self {
        Self {
            camera: CameraFrame::default(),
            scene: SceneFrame::default(),
            viewport: ViewportFrame::default(),
            interaction: InteractionFrame::default(),
            effects: EffectsFrame::default(),
        }
    }
}

impl FrameData {
    /// Build frame data from the required camera and scene groups.
    pub fn new(camera: CameraFrame, scene: SceneFrame) -> Self {
        Self {
            camera,
            scene,
            ..Self::default()
        }
    }

    /// Build frame data from a camera, scene, and selection in one call.
    ///
    /// This is the preferred constructor for the common single-viewport path.
    /// It collects render items, stamps the scene and selection generation counters,
    /// and leaves viewport chrome and effects at their defaults.
    ///
    /// Override individual settings with the builder methods:
    ///
    /// ```rust,ignore
    /// let frame = FrameData::from_scene(
    ///     CameraFrame::from_camera(&camera, [w, h]),
    ///     &mut scene,
    ///     &selection,
    /// )
    /// .with_background([0.1, 0.1, 0.12, 1.0])
    /// .with_lighting(lighting);
    /// ```
    pub fn from_scene(
        camera: CameraFrame,
        scene: &mut crate::scene::scene::Scene,
        selection: &crate::interaction::selection::Selection,
    ) -> Self {
        Self {
            camera,
            scene: SceneFrame::from_scene(scene, selection),
            interaction: InteractionFrame::from_selection(selection),
            ..Self::default()
        }
    }

    /// Set the viewport background clear color.
    pub fn with_background(mut self, color: [f32; 4]) -> Self {
        self.viewport.background_color = Some(color);
        self
    }

    /// Override the per-frame lighting configuration.
    pub fn with_lighting(mut self, lighting: LightingSettings) -> Self {
        self.effects.lighting = lighting;
        self
    }

    /// Override the post-processing settings.
    pub fn with_post_process(mut self, post: PostProcessSettings) -> Self {
        self.effects.post_process = post;
        self
    }

    /// Override the ground plane configuration.
    pub fn with_ground_plane(mut self, ground: GroundPlane) -> Self {
        self.effects.ground_plane = ground;
        self
    }
}

// ---------------------------------------------------------------------------
// Draw-call macro (must be defined before use in impl block)
// ---------------------------------------------------------------------------

/// Internal macro that emits all draw calls. Used by both `paint` (egui /
/// `'static`) and `paint_to` (iced / any lifetime) to avoid duplicating
/// ~90 lines of rendering code while satisfying Rust's lifetime invariance
/// on `&mut RenderPass<'a>`.
macro_rules! emit_draw_calls {
    ($resources:expr, $render_pass:expr, $frame:expr, $use_instancing:expr, $batches:expr, $camera_bg:expr, $grid_bg:expr, $compute_filter_results:expr, $slot:expr) => {{
        let resources = $resources;
        let render_pass = $render_pass;
        let frame = $frame;
        let use_instancing: bool = $use_instancing;
        let _vp_slot: Option<&ViewportSlot> = $slot;
        // Phase G compute filter results: used by per-object path to override index buffers.
        let compute_filter_results: &[crate::resources::ComputeFilterResult] = $compute_filter_results;
        let batches: &[InstancedBatch] = $batches;
        let camera_bg: &wgpu::BindGroup = $camera_bg;
        let grid_bg: &wgpu::BindGroup = $grid_bg;

        // Resolve scene items from the SurfaceSubmission seam.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items,
        };

        render_pass.set_bind_group(0, camera_bg, &[]);

        // Grid pass : full-screen analytical shader drawn first so scene geometry
        // occludes it. No vertex buffer; depth is written via @builtin(frag_depth).
        // Camera bind group is restored immediately after for subsequent passes.
        if frame.viewport.show_grid {
            render_pass.set_pipeline(&resources.grid_pipeline);
            render_pass.set_bind_group(0, grid_bg, &[]);
            render_pass.draw(0..3, 0..1);
            render_pass.set_bind_group(0, camera_bg, &[]);
        }

        // Ground plane pass : drawn after grid, before scene geometry.
        // Uses its own bind group (group 0: uniform + shadow atlas + sampler).
        if !matches!(
            frame.effects.ground_plane.mode,
            crate::renderer::types::GroundPlaneMode::None
        ) {
            render_pass.set_pipeline(&resources.ground_plane_pipeline);
            render_pass.set_bind_group(0, &resources.ground_plane_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
            render_pass.set_bind_group(0, camera_bg, &[]);
        }

            if !scene_items.is_empty() {
                if use_instancing && !batches.is_empty() {
                    let excluded_items: Vec<&SceneRenderItem> = scene_items
                        .iter()
                        .filter(|item| {
                            item.visible
                                && (item.active_attribute.is_some()
                                    || item.material.is_two_sided()
                                    || item.material.param_vis.is_some())
                                && resources
                                    .mesh_store
                                    .get(item.mesh_id)
                                    .is_some()
                        })
                        .collect();

                // --- Instanced draw path ---
                // Separate opaque and transparent batches.
                let mut opaque_batches: Vec<&InstancedBatch> = Vec::new();
                let mut transparent_batches: Vec<&InstancedBatch> = Vec::new();
                for batch in batches {
                    if batch.is_transparent {
                        transparent_batches.push(batch);
                    } else {
                        opaque_batches.push(batch);
                    }
                }

                    // Draw opaque instanced batches.
                    if !opaque_batches.is_empty() && !frame.viewport.wireframe_mode {
                        if let Some(ref pipeline) = resources.solid_instanced_pipeline {
                            render_pass.set_pipeline(pipeline);
                            for batch in &opaque_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else { continue };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                // Combined (instance storage + texture) bind group, primed in prepare().
                                let Some(inst_tex_bg) = resources.instance_bind_groups.get(&mat_key) else { continue };
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset..batch.instance_offset + batch.instance_count,
                                );
                            }
                        }
                    }

                    // Draw transparent instanced batches.
                    if !transparent_batches.is_empty() && !frame.viewport.wireframe_mode {
                        if let Some(ref pipeline) = resources.transparent_instanced_pipeline {
                            render_pass.set_pipeline(pipeline);
                            for batch in &transparent_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else { continue };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) = resources.instance_bind_groups.get(&mat_key) else { continue };
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset..batch.instance_offset + batch.instance_count,
                                );
                            }
                        }
                    }

                    // Wireframe mode fallback: draw per-object.
                    // mesh.object_bind_group (group 1) already contains the object uniform
                    // and fallback textures : no separate group 2 needed.
                    if frame.viewport.wireframe_mode {
                        for item in scene_items {
                            if !item.visible { continue; }
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else { continue };
                            render_pass.set_pipeline(&resources.wireframe_pipeline);
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(mesh.edge_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                        }
                    } else {
                        for item in &excluded_items {
                            let Some(mesh) = resources
                                .mesh_store
                                .get(item.mesh_id)
                            else {
                                continue;
                            };
                            let pipeline = if item.material.opacity < 1.0 {
                                &resources.transparent_pipeline
                            } else if item.material.is_two_sided() {
                                &resources.solid_two_sided_pipeline
                            } else {
                                &resources.solid_pipeline
                            };
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
            } else {
                // --- Per-object draw path (original) ---
                let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);

                let dist_from_eye = |item: &&SceneRenderItem| -> f32 {
                    let pos = glam::Vec3::new(
                        item.model[3][0],
                        item.model[3][1],
                        item.model[3][2],
                    );
                    (pos - eye).length()
                };

                let mut opaque: Vec<&SceneRenderItem> = Vec::new();
                let mut transparent: Vec<&SceneRenderItem> = Vec::new();
                for item in scene_items {
                    if !item.visible || resources.mesh_store.get(item.mesh_id).is_none() {
                        continue;
                    }
                    if item.material.opacity < 1.0 {
                        transparent.push(item);
                    } else {
                        opaque.push(item);
                    }
                }
                opaque.sort_by(|a, b| dist_from_eye(a).partial_cmp(&dist_from_eye(b)).unwrap_or(std::cmp::Ordering::Equal));
                transparent.sort_by(|a, b| dist_from_eye(b).partial_cmp(&dist_from_eye(a)).unwrap_or(std::cmp::Ordering::Equal));

                macro_rules! draw_item {
                    ($item:expr, $pipeline:expr) => {{
                        let item = $item;
                        let mesh = resources.mesh_store.get(item.mesh_id).unwrap();
                        render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);

                        // mesh.object_bind_group (group 1) already carries the object uniform
                        // and the correct texture views : updated in prepare() if material changed.
                        let is_face_attr = item.active_attribute.as_ref().map_or(false, |a| {
                            matches!(
                                a.kind,
                                crate::resources::AttributeKind::Face
                                    | crate::resources::AttributeKind::FaceColor
                            )
                        });

                        if frame.viewport.wireframe_mode {
                            render_pass.set_pipeline(&resources.wireframe_pipeline);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(
                                mesh.edge_index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                        } else if is_face_attr {
                            if let Some(ref fvb) = mesh.face_vertex_buffer {
                                render_pass.set_pipeline($pipeline);
                                render_pass.set_vertex_buffer(0, fvb.slice(..));
                                render_pass.draw(0..mesh.index_count, 0..1);
                            }
                        } else {
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            // Phase G: check for a compute-filtered index buffer override.
                            let filter_result = compute_filter_results
                                .iter()
                                .find(|r| r.mesh_id == item.mesh_id);
                            render_pass.set_pipeline($pipeline);
                            if let Some(fr) = filter_result {
                                render_pass.set_index_buffer(
                                    fr.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..fr.index_count, 0, 0..1);
                            } else {
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            }
                        }

                        if item.show_normals {
                            if let Some(ref nl_buf) = mesh.normal_line_buffer {
                                if mesh.normal_line_count > 0 {
                                    render_pass.set_pipeline(&resources.wireframe_pipeline);
                                    render_pass.set_bind_group(1, &mesh.normal_bind_group, &[]);
                                    render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                                    render_pass.draw(0..mesh.normal_line_count, 0..1);
                                }
                            }
                        }
                    }};
                }

                for item in &opaque {
                    let pl = if item.material.is_two_sided() {
                        &resources.solid_two_sided_pipeline
                    } else {
                        &resources.solid_pipeline
                    };
                    draw_item!(item, pl);
                }
                for item in &transparent {
                    draw_item!(item, &resources.transparent_pipeline);
                }
            }
        }

        // Gizmo pass.
        if let Some(slot) = _vp_slot {
            if frame.interaction.gizmo_model.is_some() && slot.gizmo_index_count > 0 {
                render_pass.set_pipeline(&resources.gizmo_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                render_pass.set_bind_group(1, &slot.gizmo_bind_group, &[]);
                render_pass.set_vertex_buffer(0, slot.gizmo_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    slot.gizmo_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..slot.gizmo_index_count, 0, 0..1);
            }
        }

        // Constraint guide line pass.
        if let Some(slot) = _vp_slot {
            if !slot.constraint_line_buffers.is_empty() {
                render_pass.set_pipeline(&resources.overlay_line_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, index_count, _ubuf, bg) in &slot.constraint_line_buffers {
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*index_count, 0, 0..1);
                }
            }
        }

        // Cap fill pass (section view cross-section fill).
        if let Some(slot) = _vp_slot {
            if !slot.cap_buffers.is_empty() {
                render_pass.set_pipeline(&resources.overlay_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.cap_buffers {
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                }
            }
        }

        // Clip plane handle fill pass (semi-transparent quad fills, alpha blended).
        if let Some(slot) = _vp_slot {
            if !slot.clip_plane_fill_buffers.is_empty() {
                render_pass.set_pipeline(&resources.overlay_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_fill_buffers {
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                }
            }
        }

        // Clip plane handle border and normal indicator pass (line list).
        if let Some(slot) = _vp_slot {
            if !slot.clip_plane_line_buffers.is_empty() {
                render_pass.set_pipeline(&resources.overlay_line_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_line_buffers {
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                }
            }
        }

        // X-ray pass: render selected objects as semi-transparent overlay through geometry.
        if let Some(slot) = _vp_slot {
            if !slot.xray_object_buffers.is_empty() {
                render_pass.set_pipeline(&resources.xray_pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (mesh_id, _buf, bg) in &slot.xray_object_buffers {
                    let Some(mesh) = resources.mesh_store.get(*mesh_id) else { continue };
                    render_pass.set_bind_group(1, bg, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        // Outline composite: blit the offscreen outline texture (rendered in prepare()).
        if let Some(slot) = _vp_slot {
            if !slot.outline_object_buffers.is_empty() {
                let composite_bg = slot.hdr.as_ref().map(|h| &h.outline_composite_bind_group);
                let pipeline = resources.outline_composite_pipeline_msaa.as_ref()
                    .or(resources.outline_composite_pipeline_single.as_ref());
                if let (Some(pipeline), Some(bg)) = (pipeline, composite_bg) {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, bg, &[]);
                    render_pass.draw(0..3, 0..1);
                }
            }
        }

        // Axes indicator pass (screen-space, last so it draws on top).
        if let Some(slot) = _vp_slot {
            if frame.viewport.show_axes_indicator && slot.axes_vertex_count > 0 {
                render_pass.set_pipeline(&resources.axes_pipeline);
                render_pass.set_vertex_buffer(0, slot.axes_vertex_buffer.slice(..));
                render_pass.draw(0..slot.axes_vertex_count, 0..1);
            }
        }
    }};
}

/// Draw point cloud and glyph items from per-frame GPU data prepared in `prepare()`.
///
/// Called by both `paint` and `paint_to` after `emit_draw_calls!` to render scivis layers.
macro_rules! emit_scivis_draw_calls {
    ($resources:expr, $render_pass:expr, $pc_gpu_data:expr, $glyph_gpu_data:expr, $polyline_gpu_data:expr, $volume_gpu_data:expr, $streamtube_gpu_data:expr, $camera_bg:expr) => {{
        let resources = $resources;
        let render_pass = $render_pass;
        let camera_bg: &wgpu::BindGroup = $camera_bg;

        // Point cloud pass.
        if !$pc_gpu_data.is_empty() {
            if let Some(ref pipeline) = resources.point_cloud_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for pc in $pc_gpu_data.iter() {
                    render_pass.set_bind_group(1, &pc.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, pc.vertex_buffer.slice(..));
                    // 6 vertices per point (billboard quad = 2 triangles), point_count instances.
                    render_pass.draw(0..6, 0..pc.point_count);
                }
            }
        }

        // Glyph pass.
        if !$glyph_gpu_data.is_empty() {
            if let Some(ref pipeline) = resources.glyph_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for glyph in $glyph_gpu_data.iter() {
                    render_pass.set_bind_group(1, &glyph.uniform_bind_group, &[]);
                    render_pass.set_bind_group(2, &glyph.instance_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, glyph.mesh_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        glyph.mesh_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..glyph.mesh_index_count, 0, 0..glyph.instance_count);
                }
            }
        }

        // Polyline pass : screen-space thick lines via instanced quad expansion.
        // Each segment instance is drawn as 6 vertices (2 triangles).
        if !$polyline_gpu_data.is_empty() {
            if let Some(ref pipeline) = resources.polyline_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for pl in $polyline_gpu_data.iter() {
                    if pl.segment_count == 0 {
                        continue;
                    }
                    render_pass.set_bind_group(1, &pl.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, pl.vertex_buffer.slice(..));
                    render_pass.draw(0..6, 0..pl.segment_count);
                }
            }
        }

        // Volume pass (after glyphs : volumes are translucent, rendered last).
        if !$volume_gpu_data.is_empty() {
            if let Some(ref pipeline) = resources.volume_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for vol in $volume_gpu_data.iter() {
                    render_pass.set_bind_group(1, &vol.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, vol.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(vol.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..36, 0, 0..1);
                }
            }
        }

        // Streamtube pass (SciVis Phase M : connected tube mesh per strip set).
        if !$streamtube_gpu_data.is_empty() {
            if let Some(ref pipeline) = resources.streamtube_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, camera_bg, &[]);
                for tube in $streamtube_gpu_data.iter() {
                    if tube.index_count == 0 {
                        continue;
                    }
                    render_pass.set_bind_group(1, &tube.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, tube.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(tube.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..tube.index_count, 0, 0..1);
                }
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_camera_from_camera_roundtrip() {
        let cam = crate::camera::Camera::default();
        let rc = RenderCamera::from_camera(&cam);
        assert_eq!(rc.eye_position, cam.eye_position().to_array());
        assert_eq!(rc.orientation, cam.orientation);
        assert_eq!(rc.near, cam.znear);
        assert_eq!(rc.far, cam.zfar);
        assert_eq!(rc.fov, cam.fov_y);
        assert_eq!(rc.aspect, cam.aspect);
        // view_proj should match Camera's own method
        let expected_vp = cam.view_proj_matrix();
        let actual_vp = rc.view_proj();
        assert!(
            (expected_vp - actual_vp).abs_diff_eq(glam::Mat4::ZERO, 1e-5),
            "view_proj mismatch"
        );
    }

    #[test]
    fn render_camera_uniform_contains_eye_and_forward() {
        let rc = RenderCamera {
            eye_position: [1.0, 2.0, 3.0],
            forward: [0.0, 0.0, -1.0],
            ..RenderCamera::default()
        };
        let u = rc.camera_uniform();
        assert_eq!(u.eye_pos, [1.0, 2.0, 3.0]);
        assert_eq!(u.forward, [0.0, 0.0, -1.0]);
        assert_eq!(u.view_proj, rc.view_proj().to_cols_array_2d());
    }
}
