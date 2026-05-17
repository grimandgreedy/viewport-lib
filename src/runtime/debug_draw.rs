//! Debug draw accumulator for runtime plugins.
//!
//! [`DebugDraw`] is a shared resource that plugins use to submit transient and
//! persistent debug visuals each frame. Store one instance in the runtime
//! resource registry and call [`DebugDraw::begin_frame`] before each call to
//! [`super::ViewportRuntime::step`] to clear one-frame primitives.
//!
//! # Usage
//!
//! ```rust,ignore
//! use viewport_lib::runtime::debug_draw::{DebugDraw, DebugLayer};
//!
//! // At startup: insert a DebugDraw resource (dev_enabled controls dev-layer draws).
//! runtime.resources_mut().insert(DebugDraw::new());
//!
//! // In each frame, before runtime.step():
//! if let Some(dd) = runtime.resources_mut().get_mut::<DebugDraw>() {
//!     dd.begin_frame();
//! }
//! let output = runtime.step(&mut scene, &mut selection, &frame_ctx);
//!
//! // After step: convert accumulated primitives to render items.
//! if let Some(dd) = runtime.resources().get::<DebugDraw>() {
//!     frame_data.scene.polylines.extend(dd.to_polylines());
//!     if let Some(pc) = dd.to_point_cloud() {
//!         frame_data.scene.point_clouds.push(pc);
//!     }
//!     frame_data.overlays.labels.extend(dd.to_labels());
//! }
//! ```
//!
//! # Layers
//!
//! Each primitive carries a [`DebugLayer`] tag:
//!
//! - [`DebugLayer::Dev`]: dev-mode visuals (contacts, AABBs, normals). Suppressed
//!   when [`DebugDraw::dev_enabled`] is `false`. Set this to `false` in ship builds.
//! - [`DebugLayer::Overlay`]: always shown regardless of `dev_enabled`. Use for
//!   persistent status overlays that should appear in release builds.
//!
//! # Lifetime
//!
//! Primitives submitted via the `line`, `point`, `aabb`, `sphere`, and `label`
//! methods are *transient*: cleared by the next [`begin_frame`](DebugDraw::begin_frame)
//! call. Persistent primitives are added with [`add_persistent`](DebugDraw::add_persistent)
//! (keyed by a caller-assigned `u64`) and stay until explicitly removed with
//! [`remove_persistent`](DebugDraw::remove_persistent).

use std::collections::HashMap;

use crate::renderer::{LabelItem, PointCloudItem, PolylineItem};

// ---------------------------------------------------------------------------
// DebugLayer
// ---------------------------------------------------------------------------

/// Category of a debug primitive, controlling visibility in ship vs dev builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebugLayer {
    /// Shown only in dev/debug mode. Contacts, AABBs, velocity arrows, and
    /// similar development visuals should use this layer. Suppressed when
    /// [`DebugDraw::dev_enabled`] is `false`.
    #[default]
    Dev,
    /// Always shown regardless of [`DebugDraw::dev_enabled`]. Use for
    /// ship-mode status overlays that need to be visible in release builds.
    Overlay,
}

// ---------------------------------------------------------------------------
// DebugPrim
// ---------------------------------------------------------------------------

/// A single debug draw primitive.
#[derive(Debug, Clone)]
pub enum DebugPrim {
    /// A world-space line segment.
    Line {
        /// Start position in world space.
        start: glam::Vec3,
        /// End position in world space.
        end: glam::Vec3,
        /// RGBA colour in linear space.
        colour: [f32; 4],
        /// Layer controlling visibility in ship vs dev modes.
        layer: DebugLayer,
    },
    /// A world-space point marker.
    Point {
        /// Position in world space.
        position: glam::Vec3,
        /// Screen-space radius in pixels.
        radius: f32,
        /// RGBA colour in linear space.
        colour: [f32; 4],
        /// Layer controlling visibility in ship vs dev modes.
        layer: DebugLayer,
    },
    /// An axis-aligned bounding box wireframe.
    Aabb {
        /// Minimum corner in world space.
        min: glam::Vec3,
        /// Maximum corner in world space.
        max: glam::Vec3,
        /// RGBA colour in linear space.
        colour: [f32; 4],
        /// Layer controlling visibility in ship vs dev modes.
        layer: DebugLayer,
    },
    /// A sphere wireframe drawn as three great circles.
    Sphere {
        /// Center position in world space.
        center: glam::Vec3,
        /// Sphere radius in world-space units.
        radius: f32,
        /// RGBA colour in linear space.
        colour: [f32; 4],
        /// Layer controlling visibility in ship vs dev modes.
        layer: DebugLayer,
    },
    /// A text label anchored to a world-space position.
    Label {
        /// World-space position of the label anchor.
        position: glam::Vec3,
        /// Text content.
        text: String,
        /// RGBA colour in linear space.
        colour: [f32; 4],
        /// Layer controlling visibility in ship vs dev modes.
        layer: DebugLayer,
    },
}

impl DebugPrim {
    /// The layer this primitive belongs to.
    pub fn layer(&self) -> DebugLayer {
        match self {
            DebugPrim::Line { layer, .. }
            | DebugPrim::Point { layer, .. }
            | DebugPrim::Aabb { layer, .. }
            | DebugPrim::Sphere { layer, .. }
            | DebugPrim::Label { layer, .. } => *layer,
        }
    }
}

// ---------------------------------------------------------------------------
// DebugDraw
// ---------------------------------------------------------------------------

/// Accumulated debug visuals for the current and future frames.
///
/// Store one instance in [`super::RuntimeResources`]. Plugins access it via
/// `ctx.resources.get_mut::<DebugDraw>()`. Call [`begin_frame`](Self::begin_frame)
/// before each [`super::ViewportRuntime::step`] call to clear transient primitives.
///
/// When both `enabled` and the relevant layer flag are true, submission methods
/// record primitives into the internal buffers. When disabled they are no-ops,
/// so no branches are needed in plugin code.
pub struct DebugDraw {
    /// Master switch. When `false`, all submission methods are no-ops.
    pub enabled: bool,
    /// Controls whether [`DebugLayer::Dev`] primitives are recorded.
    ///
    /// Set to `false` in ship builds to suppress development-only visuals
    /// without changing plugin code. [`DebugLayer::Overlay`] primitives are
    /// not affected by this flag.
    pub dev_enabled: bool,

    transient: Vec<DebugPrim>,
    persistent: HashMap<u64, DebugPrim>,
}

impl Default for DebugDraw {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugDraw {
    /// Create a new `DebugDraw` with both `enabled` and `dev_enabled` set to `true`.
    pub fn new() -> Self {
        Self {
            enabled: true,
            dev_enabled: true,
            transient: Vec::new(),
            persistent: HashMap::new(),
        }
    }

    /// Create a `DebugDraw` with `dev_enabled = false` (overlay-only mode).
    ///
    /// Use this in ship builds to suppress dev-layer visuals while keeping
    /// any overlay-layer primitives active.
    pub fn overlay_only() -> Self {
        Self {
            enabled: true,
            dev_enabled: false,
            transient: Vec::new(),
            persistent: HashMap::new(),
        }
    }

    // ---- Lifecycle --------------------------------------------------------

    /// Clear all transient primitives.
    ///
    /// Call this before each frame (before [`super::ViewportRuntime::step`]) so
    /// that one-frame debug draws do not accumulate across frames. Persistent
    /// primitives (added with [`add_persistent`](Self::add_persistent)) are not
    /// affected.
    pub fn begin_frame(&mut self) {
        self.transient.clear();
    }

    // ---- Transient submission ---------------------------------------------

    fn accept(&self, layer: DebugLayer) -> bool {
        self.enabled && (layer == DebugLayer::Overlay || self.dev_enabled)
    }

    /// Submit a line segment for this frame.
    ///
    /// The default layer is [`DebugLayer::Dev`]. Use [`line_overlay`](Self::line_overlay)
    /// for a line that appears in ship builds.
    pub fn line(&mut self, start: glam::Vec3, end: glam::Vec3, colour: [f32; 4]) -> &mut Self {
        self.line_layer(start, end, colour, DebugLayer::Dev)
    }

    /// Submit a line segment on the [`DebugLayer::Overlay`] layer.
    pub fn line_overlay(
        &mut self,
        start: glam::Vec3,
        end: glam::Vec3,
        colour: [f32; 4],
    ) -> &mut Self {
        self.line_layer(start, end, colour, DebugLayer::Overlay)
    }

    /// Submit a line segment on a specific layer.
    pub fn line_layer(
        &mut self,
        start: glam::Vec3,
        end: glam::Vec3,
        colour: [f32; 4],
        layer: DebugLayer,
    ) -> &mut Self {
        if self.accept(layer) {
            self.transient.push(DebugPrim::Line { start, end, colour, layer });
        }
        self
    }

    /// Submit a point marker for this frame.
    ///
    /// `radius` is in screen-space pixels.
    pub fn point(
        &mut self,
        position: glam::Vec3,
        radius: f32,
        colour: [f32; 4],
    ) -> &mut Self {
        self.point_layer(position, radius, colour, DebugLayer::Dev)
    }

    /// Submit a point marker on the [`DebugLayer::Overlay`] layer.
    pub fn point_overlay(
        &mut self,
        position: glam::Vec3,
        radius: f32,
        colour: [f32; 4],
    ) -> &mut Self {
        self.point_layer(position, radius, colour, DebugLayer::Overlay)
    }

    /// Submit a point marker on a specific layer.
    pub fn point_layer(
        &mut self,
        position: glam::Vec3,
        radius: f32,
        colour: [f32; 4],
        layer: DebugLayer,
    ) -> &mut Self {
        if self.accept(layer) {
            self.transient.push(DebugPrim::Point { position, radius, colour, layer });
        }
        self
    }

    /// Submit an axis-aligned box wireframe for this frame.
    pub fn aabb(
        &mut self,
        min: glam::Vec3,
        max: glam::Vec3,
        colour: [f32; 4],
    ) -> &mut Self {
        self.aabb_layer(min, max, colour, DebugLayer::Dev)
    }

    /// Submit an AABB wireframe on the [`DebugLayer::Overlay`] layer.
    pub fn aabb_overlay(
        &mut self,
        min: glam::Vec3,
        max: glam::Vec3,
        colour: [f32; 4],
    ) -> &mut Self {
        self.aabb_layer(min, max, colour, DebugLayer::Overlay)
    }

    /// Submit an AABB wireframe on a specific layer.
    pub fn aabb_layer(
        &mut self,
        min: glam::Vec3,
        max: glam::Vec3,
        colour: [f32; 4],
        layer: DebugLayer,
    ) -> &mut Self {
        if self.accept(layer) {
            self.transient.push(DebugPrim::Aabb { min, max, colour, layer });
        }
        self
    }

    /// Submit a sphere wireframe for this frame.
    ///
    /// The sphere is drawn as three great circles (XY, XZ, and YZ planes).
    pub fn sphere(
        &mut self,
        center: glam::Vec3,
        radius: f32,
        colour: [f32; 4],
    ) -> &mut Self {
        self.sphere_layer(center, radius, colour, DebugLayer::Dev)
    }

    /// Submit a sphere wireframe on the [`DebugLayer::Overlay`] layer.
    pub fn sphere_overlay(
        &mut self,
        center: glam::Vec3,
        radius: f32,
        colour: [f32; 4],
    ) -> &mut Self {
        self.sphere_layer(center, radius, colour, DebugLayer::Overlay)
    }

    /// Submit a sphere wireframe on a specific layer.
    pub fn sphere_layer(
        &mut self,
        center: glam::Vec3,
        radius: f32,
        colour: [f32; 4],
        layer: DebugLayer,
    ) -> &mut Self {
        if self.accept(layer) {
            self.transient.push(DebugPrim::Sphere { center, radius, colour, layer });
        }
        self
    }

    /// Submit a text label anchored to a world-space position for this frame.
    pub fn label(
        &mut self,
        position: glam::Vec3,
        text: impl Into<String>,
        colour: [f32; 4],
    ) -> &mut Self {
        self.label_layer(position, text, colour, DebugLayer::Dev)
    }

    /// Submit a label on the [`DebugLayer::Overlay`] layer.
    pub fn label_overlay(
        &mut self,
        position: glam::Vec3,
        text: impl Into<String>,
        colour: [f32; 4],
    ) -> &mut Self {
        self.label_layer(position, text, colour, DebugLayer::Overlay)
    }

    /// Submit a label on a specific layer.
    pub fn label_layer(
        &mut self,
        position: glam::Vec3,
        text: impl Into<String>,
        colour: [f32; 4],
        layer: DebugLayer,
    ) -> &mut Self {
        if self.accept(layer) {
            self.transient.push(DebugPrim::Label {
                position,
                text: text.into(),
                colour,
                layer,
            });
        }
        self
    }

    // ---- Persistent primitives --------------------------------------------

    /// Insert or replace a persistent primitive keyed by `id`.
    ///
    /// Persistent primitives survive [`begin_frame`](Self::begin_frame) and stay
    /// until removed with [`remove_persistent`](Self::remove_persistent). The `id`
    /// is caller-assigned; any `u64` value works as a stable identifier.
    ///
    /// Insertion respects the layer flags: if the layer would be suppressed by
    /// the current `enabled`/`dev_enabled` state, the primitive is still stored
    /// (so it becomes visible when the flag is re-enabled) but it will be skipped
    /// during rendering conversion.
    pub fn add_persistent(&mut self, id: u64, prim: DebugPrim) {
        self.persistent.insert(id, prim);
    }

    /// Remove a persistent primitive by `id`.
    ///
    /// Returns the removed primitive, or `None` if `id` was not found.
    pub fn remove_persistent(&mut self, id: u64) -> Option<DebugPrim> {
        self.persistent.remove(&id)
    }

    /// True if a persistent primitive with the given `id` exists.
    pub fn has_persistent(&self, id: u64) -> bool {
        self.persistent.contains_key(&id)
    }

    /// Remove all persistent primitives.
    pub fn clear_persistent(&mut self) {
        self.persistent.clear();
    }

    // ---- Iteration --------------------------------------------------------

    /// Iterate over all active primitives (both transient and persistent).
    ///
    /// Primitives on the [`DebugLayer::Dev`] layer are included regardless of
    /// `dev_enabled`; the caller decides whether to filter. Use
    /// [`to_polylines`](Self::to_polylines) and friends to get renderer-ready
    /// items with layer filtering already applied.
    pub fn prims(&self) -> impl Iterator<Item = &DebugPrim> {
        self.transient.iter().chain(self.persistent.values())
    }

    /// Number of transient primitives currently accumulated.
    pub fn transient_count(&self) -> usize {
        self.transient.len()
    }

    /// Number of persistent primitives currently stored.
    pub fn persistent_count(&self) -> usize {
        self.persistent.len()
    }

    /// True when no primitives (transient or persistent) are present.
    pub fn is_empty(&self) -> bool {
        self.transient.is_empty() && self.persistent.is_empty()
    }

    // ---- Rendering conversion ---------------------------------------------

    /// Convert accumulated line, AABB, and sphere primitives to polyline render items.
    ///
    /// Primitives are grouped by colour to minimize the number of render items.
    /// Dev-layer primitives are skipped when [`dev_enabled`](Self::dev_enabled) is `false`.
    /// Returns an empty `Vec` when no line-type primitives are present or when `enabled` is `false`.
    pub fn to_polylines(&self) -> Vec<PolylineItem> {
        if !self.enabled {
            return Vec::new();
        }

        // Bucket positions and strip_lengths by colour key.
        let mut groups: HashMap<u32, (Vec<[f32; 3]>, Vec<u32>, Vec<[f32; 4]>)> = HashMap::new();

        for prim in self.prims() {
            if prim.layer() == DebugLayer::Dev && !self.dev_enabled {
                continue;
            }
            match prim {
                DebugPrim::Line { start, end, colour, .. } => {
                    let key = colour_key(*colour);
                    let entry = groups.entry(key).or_default();
                    entry.0.push((*start).into());
                    entry.0.push((*end).into());
                    entry.1.push(2);
                    entry.2.push(*colour);
                    entry.2.push(*colour);
                }
                DebugPrim::Aabb { min, max, colour, .. } => {
                    let key = colour_key(*colour);
                    let entry = groups.entry(key).or_default();
                    let (positions, strips, colours) = (&mut entry.0, &mut entry.1, &mut entry.2);
                    // 6 strips: bottom loop (5), top loop (5), 4 vertical edges (2 each)
                    let mn = *min;
                    let mx = *max;
                    let aabb_positions: &[[f32; 3]] = &[
                        // Bottom face loop
                        [mn.x, mn.y, mn.z],
                        [mx.x, mn.y, mn.z],
                        [mx.x, mx.y, mn.z],
                        [mn.x, mx.y, mn.z],
                        [mn.x, mn.y, mn.z],
                        // Top face loop
                        [mn.x, mn.y, mx.z],
                        [mx.x, mn.y, mx.z],
                        [mx.x, mx.y, mx.z],
                        [mn.x, mx.y, mx.z],
                        [mn.x, mn.y, mx.z],
                        // Vertical edges
                        [mn.x, mn.y, mn.z],
                        [mn.x, mn.y, mx.z],
                        [mx.x, mn.y, mn.z],
                        [mx.x, mn.y, mx.z],
                        [mx.x, mx.y, mn.z],
                        [mx.x, mx.y, mx.z],
                        [mn.x, mx.y, mn.z],
                        [mn.x, mx.y, mx.z],
                    ];
                    let strip_lens: &[u32] = &[5, 5, 2, 2, 2, 2];
                    for &p in aabb_positions {
                        positions.push(p);
                        colours.push(*colour);
                    }
                    strips.extend_from_slice(strip_lens);
                }
                DebugPrim::Sphere { center, radius, colour, .. } => {
                    let key = colour_key(*colour);
                    let entry = groups.entry(key).or_default();
                    let (positions, strips, colours) = (&mut entry.0, &mut entry.1, &mut entry.2);
                    const SEGS: usize = 32;
                    // Three great circles: XY, XZ, YZ planes.
                    for circle_axis in 0..3_usize {
                        for i in 0..=SEGS {
                            let t = i as f32 / SEGS as f32 * std::f32::consts::TAU;
                            let (s, c) = (t.sin(), t.cos());
                            let p = match circle_axis {
                                0 => glam::Vec3::new(c * *radius + center.x, s * *radius + center.y, center.z),
                                1 => glam::Vec3::new(c * *radius + center.x, center.y, s * *radius + center.z),
                                _ => glam::Vec3::new(center.x, c * *radius + center.y, s * *radius + center.z),
                            };
                            positions.push(p.into());
                            colours.push(*colour);
                        }
                        strips.push((SEGS + 1) as u32);
                    }
                }
                _ => {} // Points and Labels handled separately.
            }
        }

        groups
            .into_values()
            .filter(|(positions, _, _)| !positions.is_empty())
            .map(|(positions, strip_lengths, node_colours)| PolylineItem {
                positions,
                strip_lengths,
                node_colours,
                line_width: 1.5,
                ..PolylineItem::default()
            })
            .collect()
    }

    /// Convert accumulated point primitives to a single point cloud render item.
    ///
    /// Returns `None` when no point primitives are present or when `enabled` is `false`.
    /// Dev-layer points are skipped when [`dev_enabled`](Self::dev_enabled) is `false`.
    pub fn to_point_cloud(&self) -> Option<PointCloudItem> {
        if !self.enabled {
            return None;
        }

        let mut positions = Vec::new();
        let mut colours = Vec::new();
        let mut radii = Vec::new();

        for prim in self.prims() {
            if prim.layer() == DebugLayer::Dev && !self.dev_enabled {
                continue;
            }
            if let DebugPrim::Point { position, radius, colour, .. } = prim {
                positions.push((*position).into());
                colours.push(*colour);
                radii.push(*radius);
            }
        }

        if positions.is_empty() {
            return None;
        }

        Some(PointCloudItem {
            positions,
            colours,
            radii,
            ..PointCloudItem::default()
        })
    }

    /// Convert accumulated label primitives to label render items.
    ///
    /// Returns an empty `Vec` when no label primitives are present or when `enabled` is `false`.
    /// Dev-layer labels are skipped when [`dev_enabled`](Self::dev_enabled) is `false`.
    pub fn to_labels(&self) -> Vec<LabelItem> {
        if !self.enabled {
            return Vec::new();
        }

        let mut out = Vec::new();
        for prim in self.prims() {
            if prim.layer() == DebugLayer::Dev && !self.dev_enabled {
                continue;
            }
            if let DebugPrim::Label { position, text, colour, .. } = prim {
                out.push(LabelItem {
                    world_anchor: Some((*position).into()),
                    text: text.clone(),
                    colour: *colour,
                    leader_line: true,
                    font_size: 12.0,
                    ..LabelItem::default()
                });
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a float RGBA colour to a u32 key for grouping.
fn colour_key(c: [f32; 4]) -> u32 {
    let r = (c[0].clamp(0.0, 1.0) * 255.0).round() as u32;
    let g = (c[1].clamp(0.0, 1.0) * 255.0).round() as u32;
    let b = (c[2].clamp(0.0, 1.0) * 255.0).round() as u32;
    let a = (c[3].clamp(0.0, 1.0) * 255.0).round() as u32;
    (r << 24) | (g << 16) | (b << 8) | a
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn red() -> [f32; 4] { [1.0, 0.0, 0.0, 1.0] }
    fn green() -> [f32; 4] { [0.0, 1.0, 0.0, 1.0] }

    // ---- accumulation tests -----------------------------------------------

    #[test]
    fn test_line_accumulates() {
        let mut dd = DebugDraw::new();
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red());
        assert_eq!(dd.transient_count(), 1);
        assert!(matches!(dd.prims().next(), Some(DebugPrim::Line { .. })));
    }

    #[test]
    fn test_point_accumulates() {
        let mut dd = DebugDraw::new();
        dd.point(glam::Vec3::Y, 4.0, green());
        assert_eq!(dd.transient_count(), 1);
        assert!(matches!(dd.prims().next(), Some(DebugPrim::Point { .. })));
    }

    #[test]
    fn test_aabb_accumulates() {
        let mut dd = DebugDraw::new();
        dd.aabb(glam::Vec3::NEG_ONE, glam::Vec3::ONE, red());
        assert_eq!(dd.transient_count(), 1);
    }

    #[test]
    fn test_sphere_accumulates() {
        let mut dd = DebugDraw::new();
        dd.sphere(glam::Vec3::ZERO, 1.0, green());
        assert_eq!(dd.transient_count(), 1);
    }

    #[test]
    fn test_label_accumulates() {
        let mut dd = DebugDraw::new();
        dd.label(glam::Vec3::ZERO, "hello", red());
        assert_eq!(dd.transient_count(), 1);
        assert!(matches!(dd.prims().next(), Some(DebugPrim::Label { .. })));
    }

    #[test]
    fn test_multiple_prims_accumulate() {
        let mut dd = DebugDraw::new();
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red());
        dd.line(glam::Vec3::ZERO, glam::Vec3::Y, green());
        dd.point(glam::Vec3::Z, 5.0, red());
        assert_eq!(dd.transient_count(), 3);
    }

    // ---- lifetime tests ---------------------------------------------------

    #[test]
    fn test_begin_frame_clears_transient() {
        let mut dd = DebugDraw::new();
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red());
        dd.sphere(glam::Vec3::ZERO, 1.0, green());
        assert_eq!(dd.transient_count(), 2);

        dd.begin_frame();
        assert_eq!(dd.transient_count(), 0);
        assert!(dd.is_empty());
    }

    #[test]
    fn test_persistent_survives_begin_frame() {
        let mut dd = DebugDraw::new();
        dd.add_persistent(1, DebugPrim::Line {
            start: glam::Vec3::ZERO,
            end: glam::Vec3::X,
            colour: red(),
            layer: DebugLayer::Dev,
        });
        assert_eq!(dd.persistent_count(), 1);

        dd.begin_frame();
        assert_eq!(dd.transient_count(), 0);
        assert_eq!(dd.persistent_count(), 1);
        assert!(!dd.is_empty());
    }

    #[test]
    fn test_persistent_removed_explicitly() {
        let mut dd = DebugDraw::new();
        dd.add_persistent(42, DebugPrim::Sphere {
            center: glam::Vec3::ZERO,
            radius: 1.0,
            colour: green(),
            layer: DebugLayer::Dev,
        });
        assert!(dd.has_persistent(42));

        let removed = dd.remove_persistent(42);
        assert!(removed.is_some());
        assert!(!dd.has_persistent(42));
        assert_eq!(dd.persistent_count(), 0);
    }

    #[test]
    fn test_persistent_remove_missing_returns_none() {
        let mut dd = DebugDraw::new();
        assert!(dd.remove_persistent(999).is_none());
    }

    #[test]
    fn test_persistent_and_transient_both_in_prims() {
        let mut dd = DebugDraw::new();
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red());
        dd.add_persistent(1, DebugPrim::Sphere {
            center: glam::Vec3::ZERO,
            radius: 1.0,
            colour: green(),
            layer: DebugLayer::Dev,
        });
        assert_eq!(dd.prims().count(), 2);
    }

    #[test]
    fn test_clear_persistent() {
        let mut dd = DebugDraw::new();
        dd.add_persistent(1, DebugPrim::Point { position: glam::Vec3::ZERO, radius: 4.0, colour: red(), layer: DebugLayer::Dev });
        dd.add_persistent(2, DebugPrim::Point { position: glam::Vec3::X, radius: 4.0, colour: green(), layer: DebugLayer::Dev });
        dd.clear_persistent();
        assert_eq!(dd.persistent_count(), 0);
    }

    // ---- disabled / no-op fallback tests ----------------------------------

    #[test]
    fn test_enabled_false_is_noop() {
        let mut dd = DebugDraw::new();
        dd.enabled = false;
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red());
        dd.sphere(glam::Vec3::ZERO, 1.0, green());
        dd.label(glam::Vec3::ZERO, "text", red());
        assert_eq!(dd.transient_count(), 0);
        assert!(dd.to_polylines().is_empty());
        assert!(dd.to_point_cloud().is_none());
        assert!(dd.to_labels().is_empty());
    }

    #[test]
    fn test_dev_disabled_suppresses_dev_prims_in_output() {
        let mut dd = DebugDraw::new();
        dd.dev_enabled = false;

        // Dev-layer line: should not appear in submission output.
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red()); // uses Dev layer by default
        // The method is a no-op when dev_enabled is false.
        assert_eq!(dd.transient_count(), 0);
        assert!(dd.to_polylines().is_empty());
    }

    #[test]
    fn test_overlay_layer_shown_when_dev_disabled() {
        let mut dd = DebugDraw::new();
        dd.dev_enabled = false;

        // Overlay-layer line should still be submitted.
        dd.line_overlay(glam::Vec3::ZERO, glam::Vec3::X, red());
        assert_eq!(dd.transient_count(), 1);
        assert!(!dd.to_polylines().is_empty());
    }

    #[test]
    fn test_persistent_dev_skipped_in_rendering_when_dev_disabled() {
        let mut dd = DebugDraw::new();
        // Add a persistent dev-layer line when dev_enabled is true.
        dd.add_persistent(1, DebugPrim::Line {
            start: glam::Vec3::ZERO,
            end: glam::Vec3::X,
            colour: red(),
            layer: DebugLayer::Dev,
        });
        // Now disable dev.
        dd.dev_enabled = false;
        // The persistent primitive is still stored but should not appear in to_polylines.
        assert_eq!(dd.persistent_count(), 1);
        assert!(dd.to_polylines().is_empty());
    }

    // ---- rendering conversion tests ---------------------------------------

    #[test]
    fn test_to_polylines_produces_output_for_lines() {
        let mut dd = DebugDraw::new();
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red());
        let polylines = dd.to_polylines();
        assert!(!polylines.is_empty());
        let total_positions: usize = polylines.iter().map(|p| p.positions.len()).sum();
        assert_eq!(total_positions, 2);
    }

    #[test]
    fn test_to_polylines_aabb_18_positions() {
        let mut dd = DebugDraw::new();
        dd.aabb(glam::Vec3::NEG_ONE, glam::Vec3::ONE, red());
        let polylines = dd.to_polylines();
        assert!(!polylines.is_empty());
        // The AABB wireframe has 18 positions (5+5+2+2+2+2).
        let total: usize = polylines.iter().map(|p| p.positions.len()).sum();
        assert_eq!(total, 18);
    }

    #[test]
    fn test_to_polylines_sphere_three_circles() {
        let mut dd = DebugDraw::new();
        dd.sphere(glam::Vec3::ZERO, 1.0, green());
        let polylines = dd.to_polylines();
        assert!(!polylines.is_empty());
        // 3 circles * 33 points each (32 segs + close).
        let total: usize = polylines.iter().map(|p| p.positions.len()).sum();
        assert_eq!(total, 3 * 33);
    }

    #[test]
    fn test_to_point_cloud_produces_output() {
        let mut dd = DebugDraw::new();
        dd.point(glam::Vec3::ZERO, 5.0, red());
        dd.point(glam::Vec3::X, 3.0, green());
        let pc = dd.to_point_cloud();
        assert!(pc.is_some());
        let pc = pc.unwrap();
        assert_eq!(pc.positions.len(), 2);
        assert_eq!(pc.colours.len(), 2);
        assert_eq!(pc.radii.len(), 2);
    }

    #[test]
    fn test_to_labels_produces_output() {
        let mut dd = DebugDraw::new();
        dd.label(glam::Vec3::ZERO, "A", red());
        dd.label(glam::Vec3::X, "B", green());
        let labels = dd.to_labels();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].text, "A");
        assert_eq!(labels[1].text, "B");
    }

    #[test]
    fn test_same_colour_grouped_into_one_polyline() {
        let mut dd = DebugDraw::new();
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red());
        dd.line(glam::Vec3::Y, glam::Vec3::Z, red());
        // Both lines share the same colour; they should be in a single PolylineItem.
        let polylines = dd.to_polylines();
        assert_eq!(polylines.len(), 1);
        assert_eq!(polylines[0].positions.len(), 4);
        assert_eq!(polylines[0].strip_lengths, vec![2, 2]);
    }

    #[test]
    fn test_different_colours_produce_separate_polylines() {
        let mut dd = DebugDraw::new();
        dd.line(glam::Vec3::ZERO, glam::Vec3::X, red());
        dd.line(glam::Vec3::Y, glam::Vec3::Z, green());
        let polylines = dd.to_polylines();
        assert_eq!(polylines.len(), 2);
    }
}
