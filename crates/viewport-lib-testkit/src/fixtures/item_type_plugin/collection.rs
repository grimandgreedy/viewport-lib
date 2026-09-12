//! The item collection the item-type fixtures are submitted with.

use std::any::Any;
use viewport_lib::plugin_api::PluginItemCollection;
use viewport_lib::{ItemSettings, PickId};

/// A collection of `n` identical items, each with its own pick id.
///
/// Carries no geometry: the item-type fixtures assert on dispatch, so the
/// collection only has to report a length and per-item settings the renderer
/// can read.
pub struct CountedItemCollection {
    settings: Vec<ItemSettings>,
}

impl CountedItemCollection {
    /// `n` visible items, with pick ids `1..=n`.
    pub fn new(n: usize) -> Self {
        let settings = (0..n)
            .map(|i| {
                let mut s = ItemSettings::default();
                s.pick_id = PickId(i as u64 + 1);
                s
            })
            .collect();
        Self { settings }
    }

    /// Mark the item at `index` hidden, so a test can check the plugin still
    /// receives the collection and is left to honour the flag itself.
    pub fn hide(&mut self, index: usize) {
        self.settings[index].hidden = true;
    }
}

impl PluginItemCollection for CountedItemCollection {
    fn len(&self) -> usize {
        self.settings.len()
    }

    fn item_settings(&self, index: usize) -> &ItemSettings {
        &self.settings[index]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
