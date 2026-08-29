//! Per-frame submission for external instance sets.

use crate::scene::material::ItemSettings;

/// Draw a window of an external instance set this frame.
///
/// Submit on `SceneFrame::external_instances`. The set behind `set_id` was
/// created with
/// [`DeviceResources::create_external_instance_set`](crate::resources::DeviceResources::create_external_instance_set)
/// and holds a consumer-owned GPU buffer of packed `[x, y, z]` `f32`
/// positions; one mesh renders per element in
/// `first_instance..first_instance + instance_count`. The range indexes the
/// buffer in elements, so disjoint items can render different regions of one
/// pooled buffer.
///
/// Draw support matches the GPU particle mesh route: HDR render path only,
/// opaque, no shadow casting, no picking, no culling.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExternalInstancesItem {
    /// Target set.
    pub set_id: crate::resources::ExternalInstanceSetId,
    /// Whole-set transform applied on top of the per-instance translation.
    pub model: [[f32; 4]; 4],
    /// First element (vec3 index) of the positions buffer to draw.
    pub first_instance: u32,
    /// Number of instances to draw. Clamped to the buffer's element count.
    pub instance_count: u32,
    /// Uniform scale applied to the mesh per instance.
    pub scale: f32,
    /// Instance colour (RGBA albedo).
    pub colour: [f32; 4],
    /// Per-item settings; only `hidden` is honoured.
    pub settings: ItemSettings,
}

impl ExternalInstancesItem {
    /// Visible white item drawing `instance_count` instances from the start
    /// of the buffer at scale 1 with an identity transform.
    pub fn new(set_id: crate::resources::ExternalInstanceSetId, instance_count: u32) -> Self {
        Self {
            set_id,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            first_instance: 0,
            instance_count,
            scale: 1.0,
            colour: [1.0, 1.0, 1.0, 1.0],
            settings: ItemSettings::default(),
        }
    }
}
