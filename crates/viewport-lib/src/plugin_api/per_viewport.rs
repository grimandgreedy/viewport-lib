//! Per-viewport state for item-type plugins.

/// One `T` per viewport, by [`ItemFrameContext::viewport_index`].
///
/// With several viewports over one scene, an item-type plugin's `prepare`
/// and `cull` run once per viewport before any viewport paints. Whatever they
/// build from the viewport's camera, frame or interaction state goes in one
/// of these: `prepare` and `cull` write the entry for their context's
/// `viewport_index` and the draw hooks read the entry for theirs.
///
/// [`ItemFrameContext::viewport_index`]: crate::plugin_api::ItemFrameContext::viewport_index
#[derive(Debug, Clone)]
pub struct PerViewport<T> {
    slots: Vec<T>,
}

impl<T> Default for PerViewport<T> {
    fn default() -> Self {
        Self { slots: Vec::new() }
    }
}

impl<T: Default> PerViewport<T> {
    /// The entry for `viewport_index`, made with `T::default()` the first
    /// time that viewport asks.
    pub fn get_mut(&mut self, viewport_index: usize) -> &mut T {
        if self.slots.len() <= viewport_index {
            self.slots.resize_with(viewport_index + 1, T::default);
        }
        &mut self.slots[viewport_index]
    }
}

impl<T> PerViewport<T> {
    /// The entry for `viewport_index`, or `None` when that viewport has not
    /// prepared.
    pub fn get(&self, viewport_index: usize) -> Option<&T> {
        self.slots.get(viewport_index)
    }

    /// Every viewport's entry, by index.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots.iter()
    }

    /// Every viewport's entry, by index, mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_made_on_first_write_and_kept_apart() {
        let mut v: PerViewport<Vec<u32>> = PerViewport::default();
        assert!(v.get(0).is_none());
        v.get_mut(2).push(7);
        v.get_mut(0).push(1);
        assert_eq!(v.get(0), Some(&vec![1]));
        assert_eq!(v.get(1), Some(&vec![]));
        assert_eq!(v.get(2), Some(&vec![7]));
        assert!(v.get(3).is_none());
    }
}
