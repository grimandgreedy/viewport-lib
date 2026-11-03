//! Point clouds from a debug-draw buffer. Needs the `gpu` feature, since the
//! buffer is the renderer's.

use super::PointCloudItem;
use viewport_lib_types::encoding::{ColourSource, SizeSource};

impl PointCloudItem {
    /// Collect the point primitives a [`DebugDraw`](viewport_lib::runtime::DebugDraw)
    /// has accumulated into one cloud.
    ///
    /// `None` when there are no point primitives, or when the buffer is
    /// disabled. Dev-layer points are skipped unless `dev_enabled` is set, the
    /// same filtering the polyline and label conversions apply.
    pub fn from_debug_draw(dd: &viewport_lib::runtime::DebugDraw) -> Option<Self> {
        use viewport_lib::runtime::{DebugLayer, DebugPrim};

        if !dd.enabled {
            return None;
        }
        let mut positions = Vec::new();
        let mut colours = Vec::new();
        let mut radii = Vec::new();
        for prim in dd.prims() {
            if prim.layer() == DebugLayer::Dev && !dd.dev_enabled {
                continue;
            }
            if let DebugPrim::Point {
                position,
                radius,
                colour,
                ..
            } = prim
            {
                positions.push((*position).into());
                colours.push(*colour);
                radii.push(*radius);
            }
        }
        if positions.is_empty() {
            return None;
        }
        Some(Self {
            positions,
            colour: ColourSource::PerSample(colours),
            size: SizeSource::PerSample(radii),
            ..Self::default()
        })
    }
}
