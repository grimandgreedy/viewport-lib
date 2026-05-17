//! Typed resource registry for sharing engine-owned state across runtime plugins.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Typed resource registry stored on [`super::ViewportRuntime`].
///
/// Accessible from [`super::context::RuntimeStepContext`] each frame. Resources
/// persist across frames until explicitly removed or the runtime is dropped.
///
/// Each type `T` maps to at most one value. Inserting again overwrites the previous
/// value of the same type.
///
/// # Ownership rules
///
/// Resources are engine-owned: they live for the lifetime of the `ViewportRuntime`
/// they belong to. Use resources for state that two or more plugins need to share
/// within the same runtime instance. Use plugin-local `struct` fields for state
/// that belongs to only one plugin and does not need to be shared.
///
/// # Example
///
/// ```rust,ignore
/// // Plugin A (ANIMATE phase) writes a shared value.
/// fn step(&mut self, ctx: &mut RuntimeStepContext) {
///     ctx.resources.insert(MySharedState { value: 42 });
/// }
///
/// // Plugin B (POST_SIM phase) reads it.
/// fn step(&mut self, ctx: &mut RuntimeStepContext) {
///     if let Some(state) = ctx.resources.get::<MySharedState>() {
///         println!("shared value: {}", state.value);
///     }
/// }
/// ```
#[derive(Default)]
pub struct RuntimeResources {
    map: HashMap<TypeId, Box<dyn Any + Send + 'static>>,
}

impl RuntimeResources {
    /// Create an empty resource registry.
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Insert a resource, replacing any existing value of the same type.
    pub fn insert<T: Send + 'static>(&mut self, value: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Get a shared reference to a resource. Returns `None` if not present.
    pub fn get<T: Send + 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    /// Get a mutable reference to a resource. Returns `None` if not present.
    pub fn get_mut<T: Send + 'static>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }

    /// Remove a resource and return it. Returns `None` if not present.
    pub fn remove<T: Send + 'static>(&mut self) -> Option<T> {
        self.map
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok())
            .map(|b| *b)
    }

    /// Returns `true` if a resource of this type is present.
    pub fn contains<T: Send + 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }
}
