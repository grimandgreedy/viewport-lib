use crate::scene::material::ItemSettings;

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

/// A volume slice sampled on an arbitrary surface mesh.
///
/// Unlike [`ImageSliceItem`] which is restricted to axis-aligned flat quads,
/// this item renders any uploaded mesh and colours each fragment by the volume
/// scalar at that world-space position. The slice surface can be a flat plane,
/// a disk, a saddle, a paraboloid -- any shape that can be expressed as a mesh.
///
/// Upload the surface mesh once with [`ViewportGpuResources::upload_mesh_data`]
/// to get a [`MeshId`](crate::resources::mesh::mesh_store::MeshId), then submit a
/// `VolumeSurfaceSliceItem` referencing that mesh each frame.
///
/// Fragments whose world position falls outside the volume bounding box are
/// discarded, so the mesh can extend beyond the volume without clipping artifacts.
///
/// # `ItemSettings.unlit` and `ItemSettings.wireframe`
///
/// Accepted but no-op. The slice colours each fragment by sampling the volume
/// LUT directly; there is no lighting calculation to skip and no edge-pass
/// variant of the pipeline. Setting either flag compiles and renders identically
/// to the default.
///
/// # `ItemSettings.opacity`
///
/// Multiplied into the type's own [`opacity`](Self::opacity) field at upload
/// time so `settings.opacity` controls transparency consistently across item
/// types. The two fields compose multiplicatively; the type field is retained
/// for back-compat.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct VolumeSurfaceSliceItem {
    /// Reference to a previously uploaded 3D volume texture.
    pub volume_id: crate::resources::VolumeId,
    /// Mesh defining the slice surface shape. Any mesh works: flat quad, disk, saddle, etc.
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// World-space bounding box minimum corner of the volume.
    pub bbox_min: [f32; 3],
    /// World-space bounding box maximum corner of the volume.
    pub bbox_max: [f32; 3],
    /// Scalar range for colourmap mapping `[min, max]`. Default: `(0.0, 1.0)`.
    pub scalar_range: (f32, f32),
    /// Colour LUT. `None` = default builtin (viridis).
    pub colour_lut: Option<crate::resources::ColourmapId>,
    /// Overall opacity of the slice. Default: `1.0`.
    ///
    /// Prefer [`ItemSettings::opacity`] (`settings.opacity`) for new code; the two
    /// fields compose multiplicatively so existing consumers of this field keep
    /// working.
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
            mesh_id: crate::resources::mesh::mesh_store::MeshId::INVALID,
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
