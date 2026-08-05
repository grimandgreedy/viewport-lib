use crate::scene::material::{ItemSettings, Material};

/// LIC overlay data attached to a surface item.
///
/// Set `SceneRenderItem::lic` to `Some(LicOverlay { ... })` to render a
/// Line Integral Convolution flow visualisation on that surface mesh.
/// The mesh must have a `VertexVector` attribute matching `vector_attribute`
/// uploaded via `DeviceResources::upload_mesh_data`.
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

/// Per-object render data for one frame.
#[derive(Clone)]
#[non_exhaustive]
pub struct SceneRenderItem {
    /// `MeshId` of the uploaded GPU mesh for this object.
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
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
    /// Which per-instance deformer data to bind for this item.
    ///
    /// - `None`: the item uses only per-mesh deformer data.
    /// - `Some(instance_id)`: the renderer binds the per-(mesh, instance)
    ///   deformer slot data attached via
    ///   [`crate::resources::DeviceResources::attach_deform_slot_instance`]
    ///   (or a plugin handle like
    ///   [`SkinningPlugin::attach_palette`](crate::plugins::skinning::SkinningPlugin::attach_palette)).
    ///
    /// Allows multiple `SceneRenderItem`s that share a bind-pose mesh to
    /// each pose independently (crowd / instanced characters with GPU
    /// skinning).
    ///
    /// # Trap: silently undeformed
    ///
    /// A deformer whose data lives in the **per-instance** slot (GPU skinning's
    /// joint palette, a morph-target weight palette) does nothing unless this
    /// field selects the instance the data was attached to. Attaching a palette
    /// at instance `0` and leaving `deform_instance` at `None` renders the mesh
    /// undeformed with no error. The single-instance convention is
    /// `deform_instance = Some(0)` alongside `attach_deform_slot_instance(.., 0, ..)`.
    pub deform_instance: Option<u32>,
    /// Whether this surface receives projected decals. Default: `true`.
    pub receives_decals: bool,
    /// LIC flow overlay for this surface. `None` disables LIC for this item.
    ///
    /// The mesh must have a `VertexVector` attribute matching
    /// `LicOverlay::vector_attribute`.
    pub lic: Option<LicOverlay>,
    /// LOD group to draw this object from. `None` means draw `mesh_id` directly.
    ///
    /// When set, the renderer measures how large the object appears each frame
    /// and overwrites `mesh_id` with the matching level before drawing. Set
    /// `pick_id` for the switch to use hysteresis across frames; without one the
    /// level is picked fresh each frame. Build a group with
    /// [`DeviceResources::register_lod_group`](crate::DeviceResources::register_lod_group).
    pub lod_group: Option<crate::resources::LodGroupId>,
    /// Where this object's indirect diffuse light comes from. Default:
    /// [`IndirectLightSource::GlobalIbl`] (the existing hemisphere / global IBL
    /// behaviour). Set to [`IndirectLightSource::LightProbe`] to sample the
    /// uploaded SH light-probe field at the object's position instead.
    pub indirect_light: IndirectLightSource,
}

/// Source of an object's indirect diffuse light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndirectLightSource {
    /// The scene-wide hemisphere ambient or global IBL environment. Default.
    #[default]
    GlobalIbl,
    /// The SH light-probe field, sampled at the object's world position. Has no
    /// effect until light probes are uploaded and only applies to opaque,
    /// non-instanced meshes.
    LightProbe,
}

impl Default for SceneRenderItem {
    fn default() -> Self {
        Self {
            mesh_id: crate::resources::mesh::mesh_store::MeshId::INVALID,
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
            deform_instance: None,
            receives_decals: true,
            lic: None,
            lod_group: None,
            indirect_light: IndirectLightSource::default(),
        }
    }
}

/// Optional volumetric render mode for a [`VolumeMeshItem`](super::VolumeMeshItem).
///
/// When set, the renderer draws the interior cells via projected tetrahedra
/// (Beer-Lambert through the volume) instead of the boundary surface. Requires
/// that the item was uploaded with
/// [`upload_volume_mesh_with_transparency`](crate::resources::DeviceResources::upload_volume_mesh_with_transparency)
/// so the per-tet GPU buffer exists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeTransparency {
    /// Beer-Lambert extinction coefficient (1/world unit). Typical range 0.1..5.0.
    pub density: f32,
    /// Discard cells whose scalar value is below this threshold.
    /// Defaults to [`f32::NEG_INFINITY`] (no lower bound).
    pub threshold_min: f32,
    /// Discard cells whose scalar value is above this threshold.
    /// Defaults to [`f32::INFINITY`] (no upper bound).
    pub threshold_max: f32,
}

impl Default for VolumeTransparency {
    fn default() -> Self {
        Self {
            density: 1.0,
            threshold_min: f32::NEG_INFINITY,
            threshold_max: f32::INFINITY,
        }
    }
}
