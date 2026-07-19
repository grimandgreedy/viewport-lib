//! Result types shared by the CPU and GPU pick backends: [`PickHit`],
//! [`GpuPickHit`], and [`PickRectResult`].

use super::sub_object::SubObjectRef;

// ---------------------------------------------------------------------------
// PickHit : rich hit result
// ---------------------------------------------------------------------------

/// Result of a successful ray-cast pick against a scene object.
///
/// Contains the picked object's ID plus geometric metadata about the hit point.
/// Use this for snapping, measurement, surface painting, and other hit-dependent features.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct PickHit {
    /// The object/node ID of the hit.
    pub id: u64,
    /// Typed sub-object reference : the authoritative source for sub-object identity.
    ///
    /// `Some(SubObjectRef::Face(i))` for mesh picks; `Some(SubObjectRef::Point(i))` for
    /// point cloud picks; `None` when no specific sub-object could be identified.
    pub sub_object: Option<SubObjectRef>,
    /// World-space position of the hit point (`ray_origin + ray_dir * toi`).
    pub world_pos: glam::Vec3,
    /// Surface normal at the hit point, in world space.
    pub normal: glam::Vec3,
    /// Which triangle was hit (from parry3d `FeatureId::Face`).
    /// `u32::MAX` if the feature was not a face (edge/vertex hit : rare for TriMesh).
    ///
    /// **Deprecated** : use [`sub_object`](Self::sub_object) instead.
    #[deprecated(since = "0.5.0", note = "use `sub_object` instead")]
    pub triangle_index: u32,
    /// Index of the hit point within a [`crate::renderer::PointCloudItem`].
    /// `None` for mesh picks; set when a point cloud item is hit.
    ///
    /// **Deprecated** : use [`sub_object`](Self::sub_object) instead.
    #[deprecated(since = "0.5.0", note = "use `sub_object` instead")]
    pub point_index: Option<u32>,
    /// Interpolated scalar attribute value at the hit point.
    ///
    /// Populated by the `_with_probe` picking variants when an active attribute
    /// is provided. For vertex attributes, the value is barycentric-interpolated
    /// from the three triangle corner values. For cell attributes, the value is
    /// read directly from the hit triangle index.
    pub scalar_value: Option<f32>,
    /// World-space position of the resolved sub-object feature, for snapping a
    /// gizmo handle to it.
    ///
    /// `Some` for point-like features whose exact coordinate is known: a surface
    /// vertex (the chosen corner), a curve node (`SubObjectRef::Point` on a
    /// tube / ribbon / streamtube), and a point-cloud point. For a face or plain
    /// object hit it is the hit point on the surface ([`world_pos`](Self::world_pos)).
    /// `None` for features with no single snap point (cell, voxel) and for the
    /// async poll path. Distinct from `world_pos`, which is always the raw hit
    /// point under the cursor: for a vertex pick, `world_pos` is where the ray met
    /// the triangle, while `sub_object_world_pos` is the corner itself.
    pub sub_object_world_pos: Option<glam::Vec3>,
}

impl PickHit {
    /// Construct a minimal `PickHit` for cases where no sub-object is identified
    /// (e.g. volume AABB hits). `normal` is an approximate inward normal.
    ///
    /// This is also the constructor a plugin's
    /// [`ItemTypePlugin::pick`](crate::plugin_api::ItemTypePlugin::pick) uses to
    /// return a hit: `PickHit` is `#[non_exhaustive]`, so out-of-crate code
    /// cannot build one with a struct literal and goes through this instead.
    ///
    /// ```
    /// use viewport_lib::PickHit;
    /// use glam::Vec3;
    ///
    /// let hit = PickHit::object_hit(7, Vec3::new(1.0, 2.0, 3.0), Vec3::Z);
    /// assert_eq!(hit.id, 7);
    /// assert!(hit.sub_object.is_none());
    /// ```
    #[allow(deprecated)]
    pub fn object_hit(id: u64, world_pos: glam::Vec3, normal: glam::Vec3) -> Self {
        Self {
            id,
            sub_object: None,
            world_pos,
            normal,
            triangle_index: u32::MAX,
            point_index: None,
            scalar_value: None,
            sub_object_world_pos: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SnapHit : screen-tolerance snap result
// ---------------------------------------------------------------------------

/// Best snap candidate within a screen-pixel tolerance of the cursor, from
/// [`ViewportRenderer::snap_query`](crate::renderer::ViewportRenderer::snap_query).
///
/// Unlike [`PickHit`], the cursor need not be exactly on the feature: the query
/// searches a window around the cursor and returns the nearest high-priority
/// feature (a vertex or node beats an edge beats a surface), for snapping a
/// gizmo to a feature the cursor is close to.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct SnapHit {
    /// World-space position to snap to: the exact feature coordinate (vertex
    /// corner, edge closest-point, curve / cloud node) when it is known,
    /// otherwise the reconstructed world position of the covered pixel.
    pub world_pos: glam::Vec3,
    /// The `pick_id` of the object the snap feature belongs to.
    pub object_id: u64,
    /// The resolved sub-object, or `None` for an object-level snap.
    pub sub_object: Option<SubObjectRef>,
}

// ---------------------------------------------------------------------------
// GpuPickHit : GPU object-ID pick result
// ---------------------------------------------------------------------------

/// Result of a GPU object-ID pick pass.
///
/// Lighter than [`PickHit`] : carries only the object identifier and the
/// clip-space depth value at the picked pixel. World position can be
/// reconstructed from `depth` + the inverse view-projection matrix if needed.
///
/// Obtained from [`crate::renderer::ViewportRenderer::pick_scene_gpu`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct GpuPickHit {
    /// The `pick_id` of the surface that was hit.
    ///
    /// Matches the `SceneRenderItem::pick_id` set by the application.
    /// Map to a domain object using whatever id-to-object registry the app
    /// maintains. [`crate::renderer::PickId::NONE`] is never returned
    /// (non-pickable surfaces are excluded from the pick pass).
    pub object_id: crate::renderer::PickId,
    /// Clip-space depth value in `[0, 1]` at the picked pixel.
    /// `0.0` = near plane, `1.0` = far plane.
    ///
    /// Reconstruct world position:
    /// ```ignore
    /// let ndc = Vec3::new(ndc_x, ndc_y, hit.depth);
    /// let world = view_proj_inv.project_point3(ndc);
    /// ```
    pub depth: f32,
    /// Raw sub-primitive index read back from the pick pass's second channel.
    ///
    /// What it means depends on the hit item type: the triangle index for
    /// triangle-meshed types (surfaces, tubes), the instance index for glyphs /
    /// sprites, or the segment index for polylines. It is `0` when the pick pass
    /// wrote no sub-primitive (object-level proxies, or a device without
    /// `SHADER_PRIMITIVE_INDEX` for triangle-meshed types).
    ///
    /// Prefer the resolved [`PickHit::sub_object`] from
    /// [`ViewportRenderer::pick_object`](crate::renderer::ViewportRenderer::pick_object):
    /// this raw index is only meaningful alongside the item's type and geometry.
    pub sub_primitive: u32,
}

impl GpuPickHit {
    /// Convert to an object-level [`PickHit`] so a caller can treat a GPU hit
    /// the same as a CPU one.
    ///
    /// The world position is reconstructed from the read-back depth and the
    /// cursor position. The object-id buffer carries no sub-object identity and
    /// no scalar, so `sub_object` and `scalar_value` are `None`. The normal
    /// points back toward the camera: it is a stand-in, not the surface normal,
    /// which the GPU pick does not know.
    ///
    /// `cursor` is in viewport-local pixels (top-left origin) and `view_proj` is
    /// the same combined matrix used to render the frame.
    pub fn to_pick_hit(
        &self,
        cursor: glam::Vec2,
        viewport_size: glam::Vec2,
        view_proj: glam::Mat4,
    ) -> PickHit {
        use crate::interaction::query::picking::screen_to_ray;

        let view_proj_inv = view_proj.inverse();
        let ndc_x = (cursor.x / viewport_size.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor.y / viewport_size.y) * 2.0;
        let world_pos = view_proj_inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, self.depth));
        let (_, ray_dir) = screen_to_ray(cursor, viewport_size, view_proj_inv);
        PickHit::object_hit(self.object_id.0, world_pos, -ray_dir)
    }
}

// ---------------------------------------------------------------------------
// PickRectResult : rect pick result
// ---------------------------------------------------------------------------

/// Result of a [`crate::renderer::ViewportRenderer::pick_rect`] call.
#[derive(Clone, Debug, Default)]
pub struct PickRectResult {
    /// IDs of whole items that have geometry inside the pick rect.
    ///
    /// Populated when [`super::PickMask::OBJECT`] is set.
    pub objects: Vec<u64>,
    /// Sub-elements inside the pick rect as `(item_id, sub_object)` pairs.
    ///
    /// Populated when any sub-element bit is set in the mask. All entries
    /// belong to the same geometric dimension when the mask is
    /// dimension-homogeneous (the common case).
    pub elements: Vec<(u64, SubObjectRef)>,
}

impl PickRectResult {
    /// Returns `true` when no objects or elements were found.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.elements.is_empty()
    }
}
