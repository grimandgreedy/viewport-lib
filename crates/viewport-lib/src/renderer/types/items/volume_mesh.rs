use super::{SceneRenderItem, VolumeTransparency};
use crate::resources::ColourmapId;
use crate::scene::material::ItemSettings;
#[allow(unused_imports)]
use crate::scene::material::Material;

/// An unstructured volume mesh submitted for one frame.
///
/// A volume mesh has two render modes that share the same source data
/// ([`VolumeMeshData`](crate::resources::VolumeMeshData)):
///
/// 1. **Boundary surface (default).** `transparency: None`. The renderer draws
///    the extracted boundary as a standard surface mesh: full [`Material`]
///    support (shading, textures, shadows), per-frame colourmap lookup,
///    selection outline, face- and cell-level picking. This is the cheap and
///    most common mode.
/// 2. **Volumetric (transparent).** `transparency: Some(VolumeTransparency { .. })`.
///    The renderer rasterises every interior cell through the projected-tetrahedra
///    pipeline and integrates Beer-Lambert opacity along view rays. Use this when
///    you need to see *through* the volume. Requires the item to have been uploaded
///    via [`upload_volume_mesh_with_transparency`](crate::resources::DeviceResources::upload_volume_mesh_with_transparency)
///    so [`projected_tet_id`](Self::projected_tet_id) is populated; otherwise the
///    transparency request is silently ignored.
///
/// Typical usage:
///
/// ```rust,ignore
/// // Cheap path: boundary only.
/// let item = resources.upload_volume_mesh(&device, &data)?;
/// scene.volume_meshes.push(item);
///
/// // Volumetric path: opt in to transparency support at upload time.
/// let mut item = resources.upload_volume_mesh_with_transparency(
///     &device, &queue, data, "pressure",
/// )?;
/// item.transparency = Some(VolumeTransparency { density: 2.0, ..Default::default() });
/// scene.volume_meshes.push(item);
/// ```
#[non_exhaustive]
#[derive(Clone)]
pub struct VolumeMeshItem {
    /// GPU mesh slot for the extracted boundary surface.
    ///
    /// Always populated. Used for the opaque draw, the selection-outline mask,
    /// and face- or cell-level picking via [`face_to_cell`](Self::face_to_cell).
    pub boundary_mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Maps each boundary triangle to its originating cell index.
    ///
    /// `face_to_cell[face_index]` is the cell index in the original
    /// [`VolumeMeshData::cells`](crate::VolumeMeshData::cells) array.
    /// Used to convert a [`SubObjectRef::Face`](crate::renderer::SubObjectRef::Face)
    /// pick hit into a cell index.
    pub face_to_cell: Vec<u32>,
    /// Projected-tet GPU handle. `None` for items uploaded via
    /// [`upload_volume_mesh`](crate::resources::DeviceResources::upload_volume_mesh)
    /// (boundary-only path); `Some(_)` for items uploaded via
    /// [`upload_volume_mesh_with_transparency`](crate::resources::DeviceResources::upload_volume_mesh_with_transparency).
    ///
    /// Transparency requires this to be `Some`. The handle is crate-internal,
    /// produced by the upload helper.
    pub projected_tet_id: Option<crate::resources::ProjectedTetId>,
    /// CPU-side volume mesh data used for interior-inclusive cell picking when
    /// transparency is active.
    ///
    /// Populated by [`upload_volume_mesh_with_transparency`](crate::resources::DeviceResources::upload_volume_mesh_with_transparency).
    /// Without it, transparent items still render but cell-level picking via
    /// `renderer.pick()` falls back to face-on-boundary hits only.
    pub volume_mesh_data:
        Option<std::sync::Arc<crate::resources::volume::volume_mesh::VolumeMeshData>>,
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, opacity, pick identity, selection state).
    pub settings: ItemSettings,
    /// Per-object material. Consumed only by the boundary-surface render path.
    /// The projected-tet path uses the colourmap LUT and ignores [`Material`].
    pub material: crate::scene::material::Material,
    /// Named scalar or colour attribute to colour the boundary surface by.
    pub active_attribute: Option<crate::resources::AttributeRef>,
    /// Explicit scalar range `(min, max)`. `None` = use the range computed at upload.
    /// Applies to both render paths.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap. Used by the boundary-surface path when `active_attribute` is
    /// set, and by the projected-tet path for the volumetric LUT lookup.
    pub colourmap_id: Option<ColourmapId>,
    /// Volumetric render mode. `None` = render the boundary as a surface mesh
    /// (default); `Some(_)` = render through the interior via projected
    /// tetrahedra (requires [`projected_tet_id`](Self::projected_tet_id) to be set).
    pub transparency: Option<VolumeTransparency>,
}

impl VolumeMeshItem {
    /// Construct a boundary-only volume mesh item.
    ///
    /// Prefer using
    /// [`upload_volume_mesh`](crate::resources::DeviceResources::upload_volume_mesh)
    /// which returns a fully populated item. This constructor is provided for
    /// host code that already has a `MeshId` and `face_to_cell` map (for
    /// example after a clipped re-upload).
    pub fn new(
        boundary_mesh_id: crate::resources::mesh::mesh_store::MeshId,
        face_to_cell: Vec<u32>,
    ) -> Self {
        Self {
            boundary_mesh_id,
            face_to_cell,
            projected_tet_id: None,
            volume_mesh_data: None,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
            material: crate::scene::material::Material::default(),
            active_attribute: None,
            scalar_range: None,
            colourmap_id: None,
            transparency: None,
        }
    }

    /// Build the [`SceneRenderItem`] that the boundary-surface pipeline consumes.
    ///
    /// Used internally by the renderer when [`transparency`](Self::transparency)
    /// is `None`, and by selection-outline / wireframe passes regardless of
    /// render mode. Host code does not usually need to call this.
    pub fn to_render_item(&self) -> SceneRenderItem {
        SceneRenderItem {
            mesh_id: self.boundary_mesh_id,
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

    /// Replace the boundary mesh ID and face-to-cell map, for example after a
    /// clipped re-upload via
    /// [`replace_clipped_volume_mesh`](crate::resources::DeviceResources::replace_clipped_volume_mesh).
    pub fn update_mesh(
        &mut self,
        boundary_mesh_id: crate::resources::mesh::mesh_store::MeshId,
        face_to_cell: Vec<u32>,
    ) {
        self.boundary_mesh_id = boundary_mesh_id;
        self.face_to_cell = face_to_cell;
    }
}
