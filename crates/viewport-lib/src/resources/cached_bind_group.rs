//! A bind group cache keyed on the handles it binds.

/// A bind group rebuilt only when the resources it binds change. The key is
/// the bound handles themselves (wgpu handles compare by identity), so a
/// pooled buffer that has not been reallocated reuses the bind group.
pub(crate) struct CachedBindGroup<K> {
    entry: Option<(K, crate::gpu::BindGroup)>,
}

impl<K: PartialEq> CachedBindGroup<K> {
    pub(crate) const fn new() -> Self {
        Self { entry: None }
    }

    /// The bind group for `key`, built with `build` when the key changed.
    pub(crate) fn get(
        &mut self,
        key: K,
        build: impl FnOnce() -> crate::gpu::BindGroup,
    ) -> crate::gpu::BindGroup {
        match &self.entry {
            Some((k, bg)) if *k == key => bg.clone(),
            _ => {
                let bg = build();
                self.entry = Some((key, bg.clone()));
                bg
            }
        }
    }
}
