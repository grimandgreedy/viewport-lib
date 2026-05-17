//! Generic typed event bus for runtime plugin communication.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Typed per-frame event accumulator for runtime plugin communication.
///
/// Stored on [`super::output::RuntimeOutput`] and accessible via `ctx.output.events`
/// from any plugin step. Events accumulate during the frame and are returned to the
/// app in [`super::output::RuntimeOutput`] at the end of each [`super::ViewportRuntime::step`] call.
///
/// Events are cleared each frame because `RuntimeOutput` is constructed fresh on
/// every `step` call. Use [`super::resources::RuntimeResources`] for state that
/// must persist across frames.
///
/// The event type IS the category. Define a distinct struct for each kind of event
/// your plugin emits. Multiple plugins can emit events of the same type; all
/// accumulate and are visible together.
///
/// # Example
///
/// ```rust,ignore
/// // Emit from a plugin step.
/// ctx.output.events.emit(TriggerEntered { trigger_id, actor_id });
///
/// // Read from a later plugin in the same frame.
/// for ev in ctx.output.events.read::<TriggerEntered>() { ... }
///
/// // Read from the app after step() returns.
/// for ev in output.events.read::<TriggerEntered>() { ... }
///
/// // Drain (take ownership) after step() returns.
/// for ev in output.events.drain::<TriggerEntered>() { ... }
/// ```
#[derive(Default)]
pub struct RuntimeEventBus {
    map: HashMap<TypeId, Vec<Box<dyn Any + Send + 'static>>>,
}

impl RuntimeEventBus {
    /// Create an empty event bus.
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Emit an event. Multiple calls accumulate; all are visible to later readers.
    pub fn emit<T: Send + 'static>(&mut self, event: T) {
        self.map
            .entry(TypeId::of::<T>())
            .or_default()
            .push(Box::new(event));
    }

    /// Iterate over all events of type `T` emitted this frame.
    pub fn read<T: Send + 'static>(&self) -> impl Iterator<Item = &T> + '_ {
        self.map
            .get(&TypeId::of::<T>())
            .into_iter()
            .flat_map(|v| v.iter().filter_map(|b| b.downcast_ref::<T>()))
    }

    /// Remove and return all events of type `T`, leaving the slot empty.
    ///
    /// Calling this twice for the same type returns an empty vec on the second call.
    pub fn drain<T: Send + 'static>(&mut self) -> Vec<T> {
        self.map
            .remove(&TypeId::of::<T>())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|b| b.downcast::<T>().ok().map(|b| *b))
            .collect()
    }

    /// Returns `true` if any events of type `T` have been emitted this frame.
    pub fn has<T: Send + 'static>(&self) -> bool {
        self.map
            .get(&TypeId::of::<T>())
            .map_or(false, |v| !v.is_empty())
    }

    /// Number of events of type `T` emitted this frame.
    pub fn count<T: Send + 'static>(&self) -> usize {
        self.map
            .get(&TypeId::of::<T>())
            .map_or(0, |v| v.len())
    }

    /// Returns `true` if no events of any type have been emitted this frame.
    pub fn is_empty(&self) -> bool {
        self.map.values().all(|v| v.is_empty())
    }
}
