//! Retained non-mesh scene items.
//!
//! Meshes live in the retained [`Scene`](crate::scene::scene::Scene) graph and
//! are added once. Point clouds, glyphs, volumes, and gaussian splats live only
//! on the per-frame [`SceneFrame`](crate::SceneFrame), so without help a static
//! one has to be re-submitted every frame. The session keeps a list of retained
//! extras and re-injects them during assembly, so they are added once like a
//! mesh node.
//!
//! Each retained item is cloned into the frame each assembly (the same cost as
//! re-pushing it by hand). For data that changes every frame, prefer the
//! per-frame injection closure ([`update_orbit_with`](ViewportSession::update_orbit_with))
//! instead; for large static data, upload it through
//! [`resources_mut`](ViewportSession::resources_mut) and retain a lightweight
//! reference item via the per-frame path.

use super::ViewportSession;
use crate::{GaussianSplatItem, GlyphItem, PointCloudItem, VolumeItem};

/// Handle to a retained scene extra, returned by the `add_*` methods and passed
/// to [`remove_extra`](ViewportSession::remove_extra).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtraId(u64);

/// A retained non-mesh item. Kept private: callers add through the typed
/// `add_*` methods and refer to items by [`ExtraId`].
pub(super) enum SceneExtra {
    PointCloud(PointCloudItem),
    Glyphs(GlyphItem),
    Volume(VolumeItem),
    GaussianSplat(GaussianSplatItem),
}

impl ViewportSession {
    /// Retain a point cloud, re-injected into the scene each frame. Returns a
    /// handle for [`remove_extra`](Self::remove_extra).
    pub fn add_point_cloud(&mut self, item: PointCloudItem) -> ExtraId {
        self.push_extra(SceneExtra::PointCloud(item))
    }

    /// Retain a glyph set, re-injected into the scene each frame.
    pub fn add_glyphs(&mut self, item: GlyphItem) -> ExtraId {
        self.push_extra(SceneExtra::Glyphs(item))
    }

    /// Retain a volume, re-injected into the scene each frame.
    pub fn add_volume(&mut self, item: VolumeItem) -> ExtraId {
        self.push_extra(SceneExtra::Volume(item))
    }

    /// Retain a gaussian splat set, re-injected into the scene each frame.
    pub fn add_gaussian_splat(&mut self, item: GaussianSplatItem) -> ExtraId {
        self.push_extra(SceneExtra::GaussianSplat(item))
    }

    /// Remove a retained extra by handle. Returns `true` if it was present.
    pub fn remove_extra(&mut self, id: ExtraId) -> bool {
        let before = self.extras.len();
        self.extras.retain(|(eid, _)| *eid != id);
        self.extras.len() != before
    }

    /// Remove all retained extras.
    pub fn clear_extras(&mut self) {
        self.extras.clear();
    }

    fn push_extra(&mut self, extra: SceneExtra) -> ExtraId {
        let id = ExtraId(self.next_extra_id);
        self.next_extra_id += 1;
        self.extras.push((id, extra));
        id
    }

    /// Append retained extras onto the freshly assembled scene sub-frame.
    /// Called from assembly, after the scene is rebuilt from the graph.
    pub(super) fn inject_extras(&mut self) {
        for (_, extra) in &self.extras {
            match extra {
                SceneExtra::PointCloud(item) => self.frame.scene.point_clouds.push(item.clone()),
                SceneExtra::Glyphs(item) => self.frame.scene.glyphs.push(item.clone()),
                SceneExtra::Volume(item) => self.frame.scene.volumes.push(item.clone()),
                SceneExtra::GaussianSplat(item) => {
                    self.frame.scene.gaussian_splats.push(item.clone())
                }
            }
        }
    }
}
