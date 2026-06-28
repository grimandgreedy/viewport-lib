//! Cubemap shadow slot pool for point lights.
//!
//! A bounded pool maps light indices (from the per-frame `combined_lights`
//! list) to cubemap shadow slots. Each slot owns six faces in the shared
//! `point_shadow_cube` texture array. When more lights request shadows than
//! the pool can hold, the slot with the oldest `last_frame_used` is evicted.
//!
//! The pool is queried each frame in two phases:
//!
//! 1. `begin_frame(frame_id)` records the current frame for LRU bookkeeping.
//! 2. `acquire(light_id)` returns a slot for the given light, allocating or
//!    reusing as needed. The returned slot index multiplied by 6 gives the
//!    base face layer in the cubemap array.

use crate::renderer::types::MAX_POINT_SHADOW_LIGHTS;

/// Stable identity of a point light across frames.
///
/// Today this is the light's index in `LightingSettings::lights`. If light
/// identity ever needs to be more robust (e.g. through reordering), this
/// can become a fully opaque newtype without changing the pool API.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LightKey(pub u32);

#[derive(Copy, Clone, Debug)]
struct Slot {
    light: LightKey,
    last_frame_used: u64,
}

/// Fixed-capacity allocator for point-light shadow slots.
pub struct PointShadowPool {
    slots: [Option<Slot>; MAX_POINT_SHADOW_LIGHTS as usize],
    current_frame: u64,
}

impl PointShadowPool {
    pub fn new() -> Self {
        Self {
            slots: [None; MAX_POINT_SHADOW_LIGHTS as usize],
            current_frame: 0,
        }
    }

    /// Start a new frame. Subsequent `acquire` calls stamp their slot's
    /// `last_frame_used` with `frame_id`.
    pub fn begin_frame(&mut self, frame_id: u64) {
        self.current_frame = frame_id;
    }

    /// Return the slot index for `light`, allocating a new one or evicting
    /// the least-recently-used slot if every slot is occupied this frame.
    ///
    /// Returns `None` only when the pool is sized at zero capacity.
    pub fn acquire(&mut self, light: LightKey) -> Option<u32> {
        if self.slots.is_empty() {
            return None;
        }
        // 1. Exact match: reuse this light's existing slot.
        for (i, s) in self.slots.iter_mut().enumerate() {
            if let Some(slot) = s {
                if slot.light == light {
                    slot.last_frame_used = self.current_frame;
                    return Some(i as u32);
                }
            }
        }
        // 2. Empty slot: allocate.
        for (i, s) in self.slots.iter_mut().enumerate() {
            if s.is_none() {
                *s = Some(Slot {
                    light,
                    last_frame_used: self.current_frame,
                });
                return Some(i as u32);
            }
        }
        // 3. Evict the slot with the smallest `last_frame_used`.
        let (victim, _) = self
            .slots
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.unwrap().last_frame_used))
            .min_by_key(|(_, f)| *f)
            .unwrap();
        self.slots[victim] = Some(Slot {
            light,
            last_frame_used: self.current_frame,
        });
        Some(victim as u32)
    }

    /// Iterate active slots: (slot_index, light_key) pairs touched this frame.
    #[cfg(test)]
    pub fn active_slots(&self) -> impl Iterator<Item = (u32, LightKey)> + '_ {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.and_then(|slot| {
                if slot.last_frame_used == self.current_frame {
                    Some((i as u32, slot.light))
                } else {
                    None
                }
            })
        })
    }
}

impl Default for PointShadowPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_then_reuse() {
        let mut p = PointShadowPool::new();
        p.begin_frame(1);
        let a = p.acquire(LightKey(0)).unwrap();
        let b = p.acquire(LightKey(0)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn lru_eviction() {
        let cap = MAX_POINT_SHADOW_LIGHTS;
        let mut p = PointShadowPool::new();
        // Fill the pool across distinct frames so each slot gets a unique LRU age.
        for i in 0..cap {
            p.begin_frame(i as u64 + 1);
            p.acquire(LightKey(i)).unwrap();
        }
        // Light 0 should be the oldest. Acquiring a new light evicts it.
        p.begin_frame(100);
        let _ = p.acquire(LightKey(999));
        // Now light 0 should not be present; acquiring it must allocate a fresh slot
        // (evicting the next-oldest, which is light 1).
        p.begin_frame(101);
        let _ = p.acquire(LightKey(0));
        let live: Vec<u32> = p.active_slots().map(|(_, k)| k.0).collect();
        assert!(live.contains(&0));
        assert!(!live.contains(&1));
    }
}
