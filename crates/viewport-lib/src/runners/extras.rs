//! Retained non-mesh scene items.
//!
//! Meshes live in the retained [`Scene`](crate::scene::scene::Scene) graph and
//! are added once. Item-type content lives only on the per-frame
//! [`SceneFrame`](crate::SceneFrame), so without help a static one has to be
//! re-submitted every frame. The session keeps a list of retained extras and
//! re-injects them during assembly, so they are added once like a mesh node.
//!
//! Each retained item is cloned into the frame each assembly (the same cost as
//! re-pushing it by hand). For data that changes every frame, prefer the
//! per-frame injection closure ([`update_orbit_with`](ViewportInstance::update_orbit_with))
//! instead; for large static data, upload it through
//! [`resources_mut`](ViewportInstance::resources_mut) and retain a lightweight
//! reference item via the per-frame path.

use super::ViewportInstance;

/// Handle to a retained scene extra, returned by [`add_item`](ViewportInstance::add_item)
/// and passed to [`remove_extra`](ViewportInstance::remove_extra).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtraId(u64);

/// A retained item, held as the push that re-injects it. Boxing the push
/// rather than the item keeps this free of any item type's name, so an item
/// type the session has never heard of retains the same way a built-in one
/// does.
pub(super) struct SceneExtra(Box<dyn Fn(&mut crate::renderer::SceneFrame)>);

impl ViewportInstance {
    /// Retain an item of any item type, re-injected into the scene each frame.
    /// Returns a handle for [`remove_extra`](Self::remove_extra).
    pub fn add_item<T: crate::plugin_api::PluginItem + Clone>(&mut self, item: T) -> ExtraId {
        self.push_extra(SceneExtra(Box::new(move |scene| {
            scene.items_mut::<T>().push(item.clone())
        })))
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
            (extra.0)(&mut self.frame.scene);
        }
    }
}
