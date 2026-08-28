//! Showcase 59: vector art (SVG) as filled overlay shapes.
//!
//! Loads an SVG through viewport-lib-io's `vector_from_path`, maps the neutral
//! paths onto `OverlayShape::Vector`, and draws them scaled and centred in the
//! viewport. This is the whole authoring path end to end: an SVG file becomes
//! neutral vector data, which becomes a tessellated overlay fill.
//!
//! The neutral io types (`viewport_lib_io::SubPath` and friends) mirror the
//! renderer types field for field, so the bridge is the small `map_subpaths` /
//! `map_rule` match below, not a dependency between the two crates.

use eframe::egui;
use viewport_lib::{FillRule, OverlayFill, OverlayShapeItem, PathSegment, SubPath};

use crate::App;

/// Bundled sample art. Paths are relative to the repo root (the example is run
/// from there). See `assets/ATTRIBUTION.md` for licences.
pub(crate) const ASSETS: &[(&str, &str)] = &[
    ("Tiger", "examples/eframe_showcase/assets/tiger.svg"),
    ("Rocket", "examples/eframe_showcase/assets/rocket.svg"),
    ("Yin Yang", "examples/eframe_showcase/assets/yin_yang.svg"),
];

pub(crate) struct VectorArtState {
    /// Index into [`ASSETS`].
    pub selected: usize,
    /// User scale, percent of the fit-to-viewport size.
    pub scale_pct: f32,
    /// Draw a thin outline on every contour (on top of the fill).
    pub show_outline: bool,
    /// Parsed source art for `selected`, kept so scaling does not re-parse.
    art: Option<(usize, viewport_lib_io::VectorArt)>,
    /// Built overlay items plus the signature they were built for.
    cache: Option<(u64, Vec<OverlayShapeItem>)>,
    /// One-line summary of the loaded art, for the controls panel.
    pub info: String,
    /// Load or parse error, if any.
    pub error: Option<String>,
}

impl Default for VectorArtState {
    fn default() -> Self {
        Self {
            selected: 0,
            scale_pct: 100.0,
            show_outline: false,
            art: None,
            cache: None,
            info: String::new(),
            error: None,
        }
    }
}

fn map_rule(rule: viewport_lib_io::FillRule) -> FillRule {
    match rule {
        viewport_lib_io::FillRule::EvenOdd => FillRule::EvenOdd,
        viewport_lib_io::FillRule::NonZero => FillRule::NonZero,
    }
}

/// Map io subpaths onto renderer subpaths, scaling every coordinate by `s`.
fn map_subpaths(src: &[viewport_lib_io::SubPath], s: f32) -> Vec<SubPath> {
    let sp = |p: [f32; 2]| [p[0] * s, p[1] * s];
    src.iter()
        .map(|c| SubPath {
            start: sp(c.start),
            segments: c
                .segments
                .iter()
                .map(|seg| match *seg {
                    viewport_lib_io::PathSegment::Line { to } => PathSegment::Line { to: sp(to) },
                    viewport_lib_io::PathSegment::Quad { ctrl, to } => PathSegment::Quad {
                        ctrl: sp(ctrl),
                        to: sp(to),
                    },
                    viewport_lib_io::PathSegment::Cubic { ctrl1, ctrl2, to } => {
                        PathSegment::Cubic {
                            ctrl1: sp(ctrl1),
                            ctrl2: sp(ctrl2),
                            to: sp(to),
                        }
                    }
                })
                .collect(),
            closed: c.closed,
        })
        .collect()
}

/// Parse the selected SVG into `state.art` if not already loaded.
fn ensure_loaded(state: &mut VectorArtState) {
    if state.art.as_ref().map(|(i, _)| *i) == Some(state.selected) {
        return;
    }
    let (_, path) = ASSETS[state.selected];
    match viewport_lib_io::loaders::svg::vector_from_path(std::path::Path::new(path)) {
        Ok(art) => {
            let filled = art.shapes.iter().filter(|s| s.fill.is_some()).count();
            state.info = format!(
                "{} shapes ({filled} filled), {:.0} x {:.0}",
                art.shapes.len(),
                art.size[0],
                art.size[1]
            );
            state.error = None;
            state.art = Some((state.selected, art));
            state.cache = None;
        }
        Err(e) => {
            state.error = Some(e.to_string());
            state.art = None;
            state.cache = None;
        }
    }
}

/// Build the overlay shapes for the current asset, scaled to the viewport and
/// centred. Results are cached; only a change of asset, scale, outline, or
/// viewport size rebuilds them.
pub(crate) fn build_overlay_shapes(app: &mut App, vp_w: f32, vp_h: f32) -> Vec<OverlayShapeItem> {
    let state = &mut app.va_state;
    ensure_loaded(state);
    let Some((_, art)) = &state.art else {
        return Vec::new();
    };
    if art.shapes.is_empty() || vp_w <= 0.0 || vp_h <= 0.0 {
        return Vec::new();
    }

    // Cache signature: rebuild only when something affecting geometry changes.
    let sig = {
        let mut h: u64 = state.selected as u64;
        h = h
            .wrapping_mul(31)
            .wrapping_add((state.scale_pct as u64).wrapping_add(1));
        h = h.wrapping_mul(31).wrapping_add(state.show_outline as u64);
        h = h.wrapping_mul(31).wrapping_add(vp_w as u64);
        h = h.wrapping_mul(31).wrapping_add(vp_h as u64);
        h
    };
    if let Some((cached_sig, items)) = &state.cache {
        if *cached_sig == sig {
            return items.clone();
        }
    }

    let (aw, ah) = (art.size[0].max(1.0), art.size[1].max(1.0));
    // Fit into 70% of the smaller viewport dimension, then apply the user scale.
    let base = 0.7 * vp_w.min(vp_h) / aw.max(ah);
    let s = base * (state.scale_pct / 100.0).max(0.01);
    let (dw, dh) = (aw * s, ah * s);
    let origin = [(vp_w - dw) * 0.5, (vp_h - dh) * 0.5];
    let size = [dw, dh];

    let mut items = Vec::with_capacity(art.shapes.len());
    for shape in &art.shapes {
        let subpaths = map_subpaths(&shape.subpaths, s);
        let mut item = OverlayShapeItem::vector(subpaths, map_rule(shape.fill_rule), origin, size)
            .with_z_order(10);
        item = match shape.fill {
            Some(rgba) => item.with_fill(OverlayFill::Solid(rgba)),
            // Stroke-only paths (fill "none", or gradients we do not resolve)
            // carry no fill; the outline below makes them visible.
            None => item.with_fill(OverlayFill::Solid([0.0, 0.0, 0.0, 0.0])),
        };
        if state.show_outline || shape.fill.is_none() {
            item = item.with_border([0.08, 0.08, 0.08, 0.9], 1.0);
        }
        items.push(item);
    }

    state.cache = Some((sig, items.clone()));
    items
}

pub(crate) fn controls_vector_art(app: &mut App, ui: &mut egui::Ui) {
    ui.heading("Vector Art (SVG)");
    ui.label("SVG loaded as neutral paths, drawn as OverlayShape::Vector fills.");
    ui.separator();

    let mut changed = false;
    ui.label("Artwork:");
    for (i, (name, _)) in ASSETS.iter().enumerate() {
        if ui
            .selectable_label(app.va_state.selected == i, *name)
            .clicked()
            && app.va_state.selected != i
        {
            app.va_state.selected = i;
            changed = true;
        }
    }

    ui.separator();
    if ui
        .add(egui::Slider::new(&mut app.va_state.scale_pct, 20.0..=200.0).text("scale %"))
        .changed()
    {
        changed = true;
    }
    if ui
        .checkbox(&mut app.va_state.show_outline, "outline every contour")
        .changed()
    {
        changed = true;
    }
    if changed {
        app.va_state.cache = None;
    }

    ui.separator();
    if let Some(err) = &app.va_state.error {
        ui.colored_label(egui::Color32::LIGHT_RED, format!("load error: {err}"));
    } else if !app.va_state.info.is_empty() {
        ui.label(&app.va_state.info);
    }
    ui.label(
        "Loader: viewport_lib_io::loaders::svg::vector_from_path. Solid fills \
         resolve to colour; gradients and patterns are left unfilled and shown \
         as outlines.",
    );
}
