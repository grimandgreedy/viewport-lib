//! Storage for the renderer's registered [`ItemTypePlugin`]s.
//!
//! A plain `HashMap` would be enough to look plugins up by name, but the
//! renderer also walks the whole set to issue draw calls, and a hash map's
//! iteration order is not stable from one process to the next. Two plugins
//! whose items overlap on screen would then blend in a different order run to
//! run. This keeps the plugins in the order they were registered so the draw
//! order is fixed, and carries a name index so lookups stay O(1).

use crate::plugin_api::ItemTypePlugin;

/// Registered item-type plugins, in registration order.
#[derive(Default)]
pub(crate) struct ItemPluginRegistry {
    entries: Vec<(&'static str, Box<dyn ItemTypePlugin>)>,
    by_name: std::collections::HashMap<&'static str, usize>,
    /// The id each entry was registered with, parallel to `entries`.
    ids: Vec<u64>,
}

impl ItemPluginRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Register `plugin` under `name`. A repeat registration of the same name
    /// replaces the plugin but keeps its original position, so adding a plugin
    /// twice does not move it in the draw order.
    #[cfg(test)]
    pub(crate) fn insert(&mut self, name: &'static str, plugin: Box<dyn ItemTypePlugin>) {
        self.insert_with_id(name, u64::MAX, plugin);
    }

    /// [`insert`](Self::insert), recording `id` against the entry. Returns the
    /// id the name held before, if it was registered.
    pub(crate) fn insert_with_id(
        &mut self,
        name: &'static str,
        id: u64,
        plugin: Box<dyn ItemTypePlugin>,
    ) -> Option<u64> {
        match self.by_name.get(name) {
            Some(&index) => {
                self.entries[index] = (name, plugin);
                Some(std::mem::replace(&mut self.ids[index], id))
            }
            None => {
                self.by_name.insert(name, self.entries.len());
                self.entries.push((name, plugin));
                self.ids.push(id);
                None
            }
        }
    }

    /// The id `name` was registered with.
    pub(crate) fn id_of(&self, name: &str) -> Option<u64> {
        Some(self.ids[*self.by_name.get(name)?])
    }

    /// The name registered with `id`.
    pub(crate) fn name_of(&self, id: u64) -> Option<&'static str> {
        let index = self.ids.iter().position(|&i| i == id)?;
        Some(self.entries[index].0)
    }

    /// Remove the plugin under `name`. The plugins that remain keep their
    /// relative order.
    pub(crate) fn remove(&mut self, name: &str) -> Option<Box<dyn ItemTypePlugin>> {
        let index = self.by_name.remove(name)?;
        let (_, plugin) = self.entries.remove(index);
        self.ids.remove(index);
        for slot in self.by_name.values_mut() {
            if *slot > index {
                *slot -= 1;
            }
        }
        Some(plugin)
    }

    pub(crate) fn contains_key(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    pub(crate) fn get(&self, name: &str) -> Option<&dyn ItemTypePlugin> {
        let index = *self.by_name.get(name)?;
        Some(self.entries[index].1.as_ref())
    }

    pub(crate) fn get_mut(&mut self, name: &str) -> Option<&mut dyn ItemTypePlugin> {
        let index = *self.by_name.get(name)?;
        Some(self.entries[index].1.as_mut())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.entries.iter().map(|(name, _)| *name)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&'static str, &dyn ItemTypePlugin)> + '_ {
        self.entries
            .iter()
            .map(|(name, plugin)| (*name, plugin.as_ref()))
    }

    pub(crate) fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (&'static str, &mut dyn ItemTypePlugin)> + '_ {
        self.entries
            .iter_mut()
            .map(|(name, plugin)| (*name, plugin.as_mut()))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &dyn ItemTypePlugin> + '_ {
        self.entries.iter().map(|(_, plugin)| plugin.as_ref())
    }

    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut dyn ItemTypePlugin> + '_ {
        self.entries.iter_mut().map(|(_, plugin)| plugin.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Stub(&'static str);

    impl ItemTypePlugin for Stub {
        fn type_name(&self) -> &'static str {
            self.0
        }
    }

    fn registry(names: &[&'static str]) -> ItemPluginRegistry {
        let mut reg = ItemPluginRegistry::new();
        for name in names {
            reg.insert(name, Box::new(Stub(name)));
        }
        reg
    }

    #[test]
    fn iteration_follows_registration_order() {
        let reg = registry(&["c", "a", "b"]);
        assert_eq!(reg.keys().collect::<Vec<_>>(), vec!["c", "a", "b"]);
        assert_eq!(
            reg.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            vec!["c", "a", "b"]
        );
    }

    #[test]
    fn re_registering_replaces_in_place() {
        let mut reg = registry(&["c", "a", "b"]);
        reg.insert("a", Box::new(Stub("a")));
        assert_eq!(reg.keys().collect::<Vec<_>>(), vec!["c", "a", "b"]);
    }

    #[test]
    fn removal_keeps_the_order_of_the_rest() {
        let mut reg = ItemPluginRegistry::new();
        for (id, name) in ["c", "a", "b", "d"].into_iter().enumerate() {
            reg.insert_with_id(name, id as u64, Box::new(Stub(name)));
        }
        assert!(reg.remove("a").is_some());
        assert!(reg.remove("a").is_none());
        assert_eq!(reg.keys().collect::<Vec<_>>(), vec!["c", "b", "d"]);
        assert_eq!(reg.get("d").map(|p| p.type_name()), Some("d"));
        assert_eq!(reg.name_of(2), Some("b"));
        assert_eq!(reg.name_of(1), None);
    }

    #[test]
    fn lookup_finds_every_registered_name() {
        let reg = registry(&["c", "a", "b"]);
        for name in ["a", "b", "c"] {
            assert!(reg.contains_key(name));
            assert_eq!(reg.get(name).map(|p| p.type_name()), Some(name));
        }
        assert!(!reg.contains_key("d"));
        assert!(reg.get("d").is_none());
        assert!(!reg.is_empty());
        assert!(ItemPluginRegistry::new().is_empty());
    }
}
