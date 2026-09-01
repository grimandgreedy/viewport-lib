//! `FrameStats` counter snapshots: the deterministic, hardware-independent subset.
//!
//! These counters are CPU-side integers (object counts, batch counts, triangle
//! counts, draw calls, per-object bind groups), so they carry no timing noise and
//! are stable across machines. Snapshotting them catches a whole class of
//! regressions the instant they happen: batching degrading to the per-object
//! path, a static scene re-uploading every frame, culling that stops culling.
//!
//! [`CounterSnapshot`] captures the subset from a [`FrameStats`]; equality is
//! exact. Format one with `Display` (a single CSV line) to record or diff it.

use std::fmt;
use viewport_lib::FrameStats;

/// The deterministic counter subset of a settled frame's [`FrameStats`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterSnapshot {
    /// Objects submitted this frame.
    pub total_objects: u32,
    /// Objects that passed culling.
    pub visible_objects: u32,
    /// Objects culled.
    pub culled_objects: u32,
    /// Draw calls issued.
    pub draw_calls: u32,
    /// Instanced batches drawn.
    pub instanced_batches: u32,
    /// Items routed through the per-object (non-instanced) path.
    pub per_object_items: u32,
    /// Per-object bind groups built this frame (0 on a settled cache).
    pub per_object_bind_groups_built: u32,
    /// Triangles submitted to the rasteriser.
    pub triangles_submitted: u64,
    /// Instanced batches whose vertex/instance buffers were re-uploaded.
    pub batches_reuploaded: u32,
    /// Instanced batches skipped as unchanged.
    pub batches_skipped: u32,
}

impl CounterSnapshot {
    /// Capture the counter subset from a frame's stats.
    pub fn capture(s: &FrameStats) -> Self {
        Self {
            total_objects: s.total_objects,
            visible_objects: s.visible_objects,
            culled_objects: s.culled_objects,
            draw_calls: s.draw_calls,
            instanced_batches: s.instanced_batches,
            per_object_items: s.per_object_items,
            per_object_bind_groups_built: s.per_object_bind_groups_built,
            triangles_submitted: s.triangles_submitted,
            batches_reuploaded: s.batches_reuploaded,
            batches_skipped: s.batches_skipped,
        }
    }

    /// The `Display` column order, for a header row when recording several.
    pub const HEADER: &'static str = "total_objects,visible_objects,culled_objects,draw_calls,instanced_batches,per_object_items,per_object_bgs,triangles,batches_reup,batches_skip";
}

impl fmt::Display for CounterSnapshot {
    /// One CSV line matching [`CounterSnapshot::HEADER`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{},{},{},{},{},{},{},{},{},{}",
            self.total_objects,
            self.visible_objects,
            self.culled_objects,
            self.draw_calls,
            self.instanced_batches,
            self.per_object_items,
            self.per_object_bind_groups_built,
            self.triangles_submitted,
            self.batches_reuploaded,
            self.batches_skipped,
        )
    }
}
