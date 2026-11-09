/// Coverage cells for vector overlay shapes.
pub(crate) mod coverage_atlas;
/// Fill functions registered for `OverlayFill::Custom`.
pub(crate) mod custom_fill;
/// Font atlas and single-line text layout for overlay rendering.
pub(crate) mod font;
/// Retained overlay geometry: compiled buffers keyed by OverlayGeometryId.
pub(crate) mod geometry;
/// Shadow and outline styling applied to glyph coverage.
pub(crate) mod glyph_style;
/// Scene-overlay scaffolding: floor grid, axes indicator, base overlay, constraint lines.
pub(crate) mod guides;
pub(crate) mod highlight;
pub(crate) mod overlay_layer;
pub(crate) mod overlay_shape;
pub(crate) mod overlay_text;
pub(crate) mod overlays;
/// Section cap fill pipelines.
pub(crate) mod section_cap;
