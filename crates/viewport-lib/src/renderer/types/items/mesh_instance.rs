use super::blend::SpriteBlend;
use crate::scene::material::ItemSettings;

/// A host-built instanced draw of a single mesh at many per-instance transforms.
///
/// Use this for mesh-based particle effects (falling leaves, debris, alembic
/// snippets, projectile clouds) where the simulation lives in the host and the
/// renderer only needs to draw N copies of one mesh per frame in a single
/// draw call. The list of instances is rebuilt every frame from the supplied
/// `transforms` and `colours` vectors; nothing is retained between frames.
///
/// All instances in a batch share the same `mesh_id`, `texture_id`, and
/// `blend` mode. To mix meshes or blend modes, submit multiple
/// [`MeshInstanceItem`] entries on [`SceneFrame::mesh_instances`].
///
/// The fragment shader is the unlit instanced mesh shader (the same shader
/// backing the scene-graph instanced path). Mesh particles do not receive
/// shadows or lighting; they are tinted by `colours` and optionally sampled
/// from `texture_id`.
#[non_exhaustive]
#[derive(Clone)]
pub struct MeshInstanceItem {
    /// Mesh handle returned by `DeviceResources::upload_mesh_data`.
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Optional albedo texture handle. `None` renders flat-shaded with the
    /// per-instance `colours` alone.
    ///
    /// Colour, so upload it sRGB
    /// ([`TextureData::srgb`](crate::resources::TextureData::srgb)).
    pub texture_id: Option<crate::resources::TextureId>,
    /// Per-instance world-space TRS matrices. Length defines the instance count.
    pub transforms: Vec<[[f32; 4]; 4]>,
    /// Per-instance RGBA tints. If shorter than `transforms`, missing entries
    /// fall back to opaque white.
    pub colours: Vec<crate::Colour>,
    /// GPU blend state for this batch. Shares [`SpriteBlend`] with the sprite
    /// path so both particle systems share one enum.
    pub blend: SpriteBlend,
    /// Per-item render settings (visibility, pick identity, selection state).
    pub settings: ItemSettings,
    /// LOD group to draw these instances from. `None` means draw `mesh_id`
    /// directly for every instance.
    ///
    /// When set, the renderer measures each instance's on-screen size and groups
    /// the instances by level, drawing each level's subset with its own mesh. So
    /// near instances in the batch draw the full mesh while far ones drop to a
    /// cheaper one, all from a single submitted item. `mesh_id` is ignored.
    /// Set `pick_id` for the per-instance level to use hysteresis across frames.
    pub lod_group: Option<crate::resources::LodGroupId>,
}

impl Default for MeshInstanceItem {
    fn default() -> Self {
        Self {
            mesh_id: crate::resources::mesh::mesh_store::MeshId::INVALID,
            texture_id: None,
            transforms: Vec::new(),
            colours: Vec::new(),
            blend: SpriteBlend::AlphaBlend,
            settings: ItemSettings::default(),
            lod_group: None,
        }
    }
}
