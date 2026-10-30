//! Layer allocator for the punctual shadow array.
//!
//! Point and spot lights share one `Depth32Float` 2D array. The array is cut
//! into slots of six layers: a point light takes a whole slot (its six cube
//! faces, in cube-face order), and up to six spot lights share one, a layer
//! each. Each light keeps its place across frames while it stays selected, so
//! its layers keep their contents and the shadow pass can skip re-rendering
//! them; places nobody selected this frame are reused oldest first.

/// Identity of a light across frames: which list it came from and its index
/// there. Stays put when the ranking reorders the lights.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShadowKey {
    /// 0 for `LightingSettings::lights`, 1 for the scene's lights, 2 for
    /// lights contributed by item-type plugins.
    pub(crate) source: u8,
    /// Index of the light in its source list.
    pub(crate) index: u32,
}

/// Layers per slot: one point light's cube faces.
pub(crate) const SLOT_LAYERS: u32 = 6;

#[derive(Copy, Clone, Debug)]
enum SlotUse {
    Cube {
        key: ShadowKey,
        last_used: u64,
    },
    Spots {
        keys: [Option<(ShadowKey, u64)>; SLOT_LAYERS as usize],
    },
}

impl SlotUse {
    /// The most recent frame any occupant of the slot was used in.
    fn last_used(&self) -> u64 {
        match self {
            Self::Cube { last_used, .. } => *last_used,
            Self::Spots { keys } => keys.iter().flatten().map(|(_, f)| *f).max().unwrap_or(0),
        }
    }
}

/// A light the pool places: its key and whether it needs a whole slot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShadowRequest {
    pub(crate) key: ShadowKey,
    pub(crate) cube: bool,
}

/// Slots of six layers, each holding one point light or up to six spot lights.
pub(crate) struct PunctualShadowPool {
    slots: Vec<Option<SlotUse>>,
    /// Content hash of what each layer last rendered, indexed by layer. A
    /// point light keeps its hash at its slot's base layer. `None` means never
    /// rendered or invalidated.
    layer_hashes: Vec<Option<u64>>,
    frame: u64,
}

impl PunctualShadowPool {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            layer_hashes: Vec::new(),
            frame: 0,
        }
    }

    /// Size the pool at `slots` slots. Occupants of slots past the new end are
    /// dropped.
    pub(crate) fn set_capacity(&mut self, slots: usize) {
        if slots == self.slots.len() {
            return;
        }
        self.slots.resize(slots, None);
        self.layer_hashes.resize(slots * SLOT_LAYERS as usize, None);
    }

    /// Whether `key` holds a place in the pool, used or not this frame.
    pub(crate) fn holds(&self, key: ShadowKey) -> bool {
        self.slots.iter().flatten().any(|s| match s {
            SlotUse::Cube { key: k, .. } => *k == key,
            SlotUse::Spots { keys } => keys.iter().flatten().any(|(k, _)| *k == key),
        })
    }

    /// Number of slots from the first to the last occupied one: the slots the
    /// texture has to hold.
    pub(crate) fn used_slots(&self) -> usize {
        self.slots
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |i| i + 1)
    }

    /// Place this frame's casters and return each one's base layer, in the
    /// order given. The requests must fit: six layers per cube and one per
    /// spot, within the pool's capacity. A request that does not fit gets
    /// `None`.
    pub(crate) fn place(&mut self, requests: &[ShadowRequest]) -> Vec<Option<u32>> {
        self.frame += 1;
        let frame = self.frame;
        let mut out = vec![None; requests.len()];
        let selected = |key: ShadowKey| requests.iter().any(|r| r.key == key);

        // Cubes first, so a slot is never split by spots placed ahead of a
        // cube that needed it whole.
        for (i, r) in requests.iter().enumerate().filter(|(_, r)| r.cube) {
            if let Some(s) = self
                .slots
                .iter()
                .position(|s| matches!(s, Some(SlotUse::Cube { key, .. }) if *key == r.key))
            {
                self.slots[s] = Some(SlotUse::Cube {
                    key: r.key,
                    last_used: frame,
                });
                out[i] = Some(s as u32 * SLOT_LAYERS);
            }
        }
        for (i, r) in requests.iter().enumerate().filter(|(_, r)| r.cube) {
            if out[i].is_some() {
                continue;
            }
            let Some(s) = self.slot_for_cube(frame, &selected) else {
                continue;
            };
            self.slots[s] = Some(SlotUse::Cube {
                key: r.key,
                last_used: frame,
            });
            self.clear_hashes(s * SLOT_LAYERS as usize..(s + 1) * SLOT_LAYERS as usize);
            out[i] = Some(s as u32 * SLOT_LAYERS);
        }

        // Spots that still hold their layer keep it.
        for (i, r) in requests.iter().enumerate().filter(|(_, r)| !r.cube) {
            for (s, slot) in self.slots.iter_mut().enumerate() {
                let Some(SlotUse::Spots { keys }) = slot else {
                    continue;
                };
                if let Some(j) = keys.iter().position(|k| k.is_some_and(|(k, _)| k == r.key)) {
                    keys[j] = Some((r.key, frame));
                    out[i] = Some(s as u32 * SLOT_LAYERS + j as u32);
                    break;
                }
            }
        }
        for (i, r) in requests.iter().enumerate().filter(|(_, r)| !r.cube) {
            if out[i].is_some() {
                continue;
            }
            let Some(layer) = self.layer_for_spot(frame) else {
                continue;
            };
            let s = (layer / SLOT_LAYERS) as usize;
            let j = (layer % SLOT_LAYERS) as usize;
            if let Some(SlotUse::Spots { keys }) = &mut self.slots[s] {
                keys[j] = Some((r.key, frame));
            }
            self.clear_hashes(layer as usize..layer as usize + 1);
            out[i] = Some(layer);
        }
        out
    }

    /// A slot for a cube that has none: an empty slot, else the slot used
    /// longest ago whose occupants are not in use this frame, else the spot
    /// slot holding the fewest selected spots (they move elsewhere).
    fn slot_for_cube(
        &mut self,
        frame: u64,
        selected: &impl Fn(ShadowKey) -> bool,
    ) -> Option<usize> {
        if let Some(s) = self.slots.iter().position(Option::is_none) {
            return Some(s);
        }
        let free = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(s, u)| u.as_ref().map(|u| (s, u)))
            .filter(|(_, u)| match u {
                SlotUse::Cube { last_used, .. } => *last_used < frame,
                SlotUse::Spots { keys } => keys
                    .iter()
                    .flatten()
                    .all(|(k, f)| *f < frame && !selected(*k)),
            })
            .min_by_key(|(_, u)| u.last_used())
            .map(|(s, _)| s);
        if free.is_some() {
            return free;
        }
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(s, u)| match u {
                Some(SlotUse::Spots { keys }) => {
                    let held = keys.iter().flatten().filter(|(k, _)| selected(*k)).count();
                    Some((s, held, u.as_ref().map_or(0, SlotUse::last_used)))
                }
                _ => None,
            })
            .min_by_key(|&(_, held, last)| (held, last))
            .map(|(s, _, _)| s)
    }

    /// A layer for a spot that has none: an empty layer in a spot slot, else
    /// a new spot slot, else the layer in a spot slot used longest ago and not
    /// this frame, else a stale cube slot turned over to spots.
    fn layer_for_spot(&mut self, frame: u64) -> Option<u32> {
        for (s, slot) in self.slots.iter().enumerate() {
            if let Some(SlotUse::Spots { keys }) = slot {
                if let Some(j) = keys.iter().position(Option::is_none) {
                    return Some(s as u32 * SLOT_LAYERS + j as u32);
                }
            }
        }
        if let Some(s) = self.slots.iter().position(Option::is_none) {
            self.slots[s] = Some(SlotUse::Spots {
                keys: [None; SLOT_LAYERS as usize],
            });
            return Some(s as u32 * SLOT_LAYERS);
        }
        let stale_layer = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(s, slot)| match slot {
                Some(SlotUse::Spots { keys }) => Some((s, keys)),
                _ => None,
            })
            .flat_map(|(s, keys)| {
                keys.iter().enumerate().filter_map(move |(j, k)| {
                    k.map(|(_, f)| (s as u32 * SLOT_LAYERS + j as u32, f))
                })
            })
            .filter(|(_, f)| *f < frame)
            .min_by_key(|(_, f)| *f)
            .map(|(layer, _)| layer);
        if stale_layer.is_some() {
            return stale_layer;
        }
        let stale_cube = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(s, slot)| match slot {
                Some(SlotUse::Cube { last_used, .. }) if *last_used < frame => {
                    Some((s, *last_used))
                }
                _ => None,
            })
            .min_by_key(|(_, f)| *f)
            .map(|(s, _)| s)?;
        self.slots[stale_cube] = Some(SlotUse::Spots {
            keys: [None; SLOT_LAYERS as usize],
        });
        self.clear_hashes(
            stale_cube * SLOT_LAYERS as usize..(stale_cube + 1) * SLOT_LAYERS as usize,
        );
        Some(stale_cube as u32 * SLOT_LAYERS)
    }

    fn clear_hashes(&mut self, layers: std::ops::Range<usize>) {
        for h in self.layer_hashes.get_mut(layers).into_iter().flatten() {
            *h = None;
        }
    }

    /// The content hash recorded for `layer`.
    pub(crate) fn layer_hash(&self, layer: u32) -> Option<u64> {
        self.layer_hashes.get(layer as usize).copied().flatten()
    }

    /// Record what `layer` holds after rendering, or `None` when its content
    /// cannot be cached.
    pub(crate) fn set_layer_hash(&mut self, layer: u32, hash: Option<u64>) {
        if let Some(h) = self.layer_hashes.get_mut(layer as usize) {
            *h = hash;
        }
    }

    /// Forget every recorded layer content, so every placed light re-renders.
    pub(crate) fn invalidate(&mut self) {
        self.layer_hashes.fill(None);
    }
}

impl Default for PunctualShadowPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(index: u32) -> ShadowKey {
        ShadowKey { source: 0, index }
    }

    fn cube(index: u32) -> ShadowRequest {
        ShadowRequest {
            key: key(index),
            cube: true,
        }
    }

    fn spot(index: u32) -> ShadowRequest {
        ShadowRequest {
            key: key(index),
            cube: false,
        }
    }

    #[test]
    fn a_cube_keeps_its_slot() {
        let mut p = PunctualShadowPool::new();
        p.set_capacity(2);
        assert_eq!(p.place(&[cube(0), cube(1)]), vec![Some(0), Some(6)]);
        assert_eq!(p.place(&[cube(1), cube(0)]), vec![Some(6), Some(0)]);
    }

    #[test]
    fn spots_share_a_slot() {
        let mut p = PunctualShadowPool::new();
        p.set_capacity(2);
        let placed = p.place(&[spot(0), spot(1), spot(2)]);
        assert_eq!(placed, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(p.used_slots(), 1);
        // A cube joins without moving the spots.
        let placed = p.place(&[spot(0), cube(3), spot(1), spot(2)]);
        assert_eq!(placed, vec![Some(0), Some(6), Some(1), Some(2)]);
    }

    #[test]
    fn a_new_cube_displaces_the_oldest_unused_light() {
        let mut p = PunctualShadowPool::new();
        p.set_capacity(2);
        p.place(&[cube(0), cube(1)]);
        p.place(&[cube(1)]);
        // Light 0 went unused longest; light 2 takes its slot.
        assert_eq!(p.place(&[cube(1), cube(2)]), vec![Some(6), Some(0)]);
        assert!(!p.holds(key(0)));
    }

    #[test]
    fn a_cube_moves_spots_out_of_the_slot_it_needs() {
        let mut p = PunctualShadowPool::new();
        p.set_capacity(2);
        // Seven spots fill the first slot and spill into the second.
        let seven: Vec<ShadowRequest> = (0..7).map(spot).collect();
        p.place(&seven);
        assert_eq!(p.used_slots(), 2);
        // Spots 0 and 6, one in each slot, and a cube fit 12 layers: the cube
        // takes a whole slot and the spots end up together in the other.
        let placed = p.place(&[spot(0), spot(6), cube(2)]);
        let cube_base = placed[2].unwrap();
        assert_eq!(cube_base % 6, 0);
        for s in &placed[..2] {
            let layer = s.unwrap();
            assert!(layer < cube_base || layer >= cube_base + 6);
        }
    }

    #[test]
    fn a_new_occupant_clears_the_layer_hash() {
        let mut p = PunctualShadowPool::new();
        p.set_capacity(1);
        p.place(&[spot(0)]);
        p.set_layer_hash(0, Some(7));
        p.place(&[spot(0)]);
        assert_eq!(p.layer_hash(0), Some(7));
        p.place(&[spot(1)]);
        assert_eq!(p.place(&[spot(2)]), vec![Some(2)]);
        // Spot 2 took a fresh layer; spot 0's layer keeps its hash until it
        // is reused.
        assert_eq!(p.layer_hash(0), Some(7));
        p.place(&[cube(3)]);
        assert_eq!(p.layer_hash(0), None);
    }
}
