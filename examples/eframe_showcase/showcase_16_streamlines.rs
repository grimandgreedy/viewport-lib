//! Showcase 16: Streamlines & Tubes
//!
//! Demonstrates `PolylineItem` (thin streamlines) and `StreamtubeItem` (3-D tube
//! rendering) for flow visualization. Synthetic streamlines are seeded from a grid
//! of starting points and integrated through a simple analytic vortex/dipole
//! velocity field. The same path data is submitted either as `PolylineItem` or
//! `StreamtubeItem` depending on a toggle.

use crate::App;
use eframe::egui;
use viewport_lib::{
    BuiltinColourmap, ColourmapId, PolylineItem, RibbonItem, StreamtubeItem, TubeItem,
};

// ---------------------------------------------------------------------------
// Enum
// ---------------------------------------------------------------------------

/// Render mode for Showcase 16 (streamlines).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamRenderMode {
    Polylines,
    Streamtube,
    GeneralTube,
    Ribbon,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct StreamlinesState {
    pub built: bool,
    pub use_tubes: bool,
    pub render_mode: StreamRenderMode,
    pub tube_radius: f32,
    pub line_width: f32,
    pub colour_by_speed: bool,
    pub colourmap: BuiltinColourmap,
    /// Flat RGBA colour used when `colour_by_speed` is false, and as tube colour.
    pub flat_colour: [f32; 4],
    pub seed_count: usize,
    pub step_size: f32,
    /// Cross-section resolution for GeneralTube mode (number of sides).
    pub tube_sides: u32,
    /// Half-width of the ribbon in Ribbon mode.
    pub ribbon_width: f32,
    /// CPU-side streamline paths (one `Vec<[f32;3]>` per seed).
    pub paths: Vec<Vec<[f32; 3]>>,
    /// Per-vertex speed values parallel to `paths`.
    pub scalars: Vec<Vec<f32>>,
}

impl Default for StreamlinesState {
    fn default() -> Self {
        Self {
            built: false,
            use_tubes: false,
            render_mode: StreamRenderMode::Polylines,
            tube_radius: 0.06,
            line_width: 4.0,
            colour_by_speed: true,
            colourmap: BuiltinColourmap::Viridis,
            flat_colour: [0.4, 0.7, 1.0, 1.0],
            seed_count: 32,
            step_size: 0.08,
            tube_sides: 8,
            ribbon_width: 0.15,
            paths: Vec::new(),
            scalars: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// One-time CPU setup for Showcase 16.
    ///
    /// Generates stream paths from the current seed count and integration step.
    /// The result is stored in `stream_state.paths` and `stream_state.scalars` (per-vertex speed).
    pub(crate) fn build_stream_scene(&mut self) {
        let (paths, scalars) =
            integrate_streamlines(self.stream_state.seed_count, self.stream_state.step_size);
        self.stream_state.paths = paths;
        self.stream_state.scalars = scalars;
        self.stream_state.built = true;
    }

    // -------------------------------------------------------------------------
    // Frame-data helpers (called from build_frame_data)
    // -------------------------------------------------------------------------

    /// Build a `PolylineItem` from cached stream paths and current control state.
    pub(crate) fn make_stream_polyline_item(&self) -> PolylineItem {
        let s = &self.stream_state;
        let (positions, strip_lengths, scalars) = flatten_paths(&s.paths, &s.scalars);
        let mut item = PolylineItem::default();
        item.positions = positions;
        item.strip_lengths = strip_lengths;
        item.line_width = s.line_width;
        if s.colour_by_speed {
            item.scalars = scalars;
            item.colourmap_id = Some(ColourmapId(s.colourmap as usize));
        } else {
            item.default_colour = s.flat_colour;
        }
        item
    }

    /// Build a `TubeItem` from cached stream paths for the GeneralTube mode.
    pub(crate) fn make_stream_general_tube_item(&self) -> TubeItem {
        let s = &self.stream_state;
        let (positions, strip_lengths, scalars) = flatten_paths(&s.paths, &s.scalars);
        let mut item = TubeItem::default();
        item.positions = positions;
        item.strip_lengths = strip_lengths;
        item.radius = s.tube_radius;
        item.sides = s.tube_sides;
        if s.colour_by_speed {
            item.scalars = scalars;
            item.colourmap_id = Some(ColourmapId(s.colourmap as usize));
        } else {
            item.colour = s.flat_colour;
        }
        item
    }

    /// Build a `StreamtubeItem` from cached stream paths and current control state.
    pub(crate) fn make_stream_tube_item(&self) -> StreamtubeItem {
        let s = &self.stream_state;
        let (positions, strip_lengths, _scalars) = flatten_paths(&s.paths, &s.scalars);
        let mut item = StreamtubeItem::default();
        item.positions = positions;
        item.strip_lengths = strip_lengths;
        item.radius = s.tube_radius;
        item.colour = s.flat_colour;
        item
    }

    /// Build a `RibbonItem` from cached stream paths and current control state.
    pub(crate) fn make_stream_ribbon_item(&self) -> RibbonItem {
        let s = &self.stream_state;
        let (positions, strip_lengths, scalars) = flatten_paths(&s.paths, &s.scalars);
        let mut item = RibbonItem::default();
        item.positions = positions;
        item.strip_lengths = strip_lengths;
        item.width = s.ribbon_width;
        if s.colour_by_speed {
            item.scalars = scalars;
            item.colourmap_id = Some(ColourmapId(s.colourmap as usize));
        } else {
            item.colour = s.flat_colour;
        }
        item
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_streamlines(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.stream_state;

    ui.label("Render mode:");
    ui.horizontal(|ui| {
        if ui
            .radio(s.render_mode == StreamRenderMode::Polylines, "Polylines")
            .clicked()
        {
            s.render_mode = StreamRenderMode::Polylines;
            s.use_tubes = false;
        }
        if ui
            .radio(s.render_mode == StreamRenderMode::Streamtube, "Streamtube")
            .clicked()
        {
            s.render_mode = StreamRenderMode::Streamtube;
            s.use_tubes = true;
        }
        if ui
            .radio(
                s.render_mode == StreamRenderMode::GeneralTube,
                "General Tube",
            )
            .clicked()
        {
            s.render_mode = StreamRenderMode::GeneralTube;
            s.use_tubes = false;
        }
        if ui
            .radio(s.render_mode == StreamRenderMode::Ribbon, "Ribbon")
            .clicked()
        {
            s.render_mode = StreamRenderMode::Ribbon;
            s.use_tubes = false;
        }
    });

    ui.separator();

    match s.render_mode {
        StreamRenderMode::Polylines => {
            ui.label("Line width (px):");
            ui.add(egui::Slider::new(&mut s.line_width, 0.5..=8.0).step_by(0.5));
        }
        StreamRenderMode::Streamtube => {
            ui.label("Tube radius:");
            ui.add(egui::Slider::new(&mut s.tube_radius, 0.01..=0.3).step_by(0.01));
        }
        StreamRenderMode::GeneralTube => {
            ui.label("Tube radius:");
            ui.add(egui::Slider::new(&mut s.tube_radius, 0.01..=0.3).step_by(0.01));
            ui.label("Cross-section sides:");
            ui.add(egui::Slider::new(&mut s.tube_sides, 3..=24).step_by(1.0));
        }
        StreamRenderMode::Ribbon => {
            ui.label("Ribbon half-width:");
            ui.add(egui::Slider::new(&mut s.ribbon_width, 0.02..=0.5).step_by(0.01));
        }
    }

    ui.separator();
    ui.label("Colouring:");

    // StreamtubeItem only supports flat colour; scalar colouring requires Polylines or
    // GeneralTube (which bakes per-vertex colours CPU-side via TubeItem).
    let scalar_colouring_supported = s.render_mode != StreamRenderMode::Streamtube;

    if scalar_colouring_supported {
        ui.horizontal(|ui| {
            if ui.radio(!s.colour_by_speed, "Flat colour").clicked() {
                s.colour_by_speed = false;
            }
            if ui.radio(s.colour_by_speed, "Speed").clicked() {
                s.colour_by_speed = true;
            }
        });
    } else {
        // Force flat colour when switching to Streamtube mode.
        s.colour_by_speed = false;
    }

    if s.colour_by_speed {
        ui.label("Colourmap:");
        for (preset, label) in [
            (BuiltinColourmap::Viridis, "Viridis"),
            (BuiltinColourmap::Plasma, "Plasma"),
            (BuiltinColourmap::Magma, "Magma"),
            (BuiltinColourmap::Inferno, "Inferno"),
            (BuiltinColourmap::Turbo, "Turbo"),
            (BuiltinColourmap::Greyscale, "Greyscale"),
            (BuiltinColourmap::Coolwarm, "Coolwarm"),
            (BuiltinColourmap::RdBu, "RdBu"),
            (BuiltinColourmap::Rainbow, "Rainbow"),
            (BuiltinColourmap::Jet, "Jet"),
        ] {
            if ui.radio(s.colourmap == preset, label).clicked() {
                s.colourmap = preset;
            }
        }
    } else {
        ui.label("Tube/line colour:");
        let mut rgb = [s.flat_colour[0], s.flat_colour[1], s.flat_colour[2]];
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            s.flat_colour[0] = rgb[0];
            s.flat_colour[1] = rgb[1];
            s.flat_colour[2] = rgb[2];
        }
    }

    ui.separator();
    ui.label("Seed count:");
    let seed_response = ui.add(egui::Slider::new(&mut s.seed_count, 8..=64).step_by(4.0));
    ui.label("Integration step:");
    let step_response = ui.add(egui::Slider::new(&mut s.step_size, 0.02..=0.2).step_by(0.01));
    let need_rebuild = seed_response.drag_stopped() || step_response.drag_stopped();

    // Regenerate paths when seed or step sliders are released.
    if need_rebuild {
        app.build_stream_scene();
    }
}

// ---------------------------------------------------------------------------
// Path generation
// ---------------------------------------------------------------------------

/// Integrate `seed_count` streamlines through an analytic vortex field.
///
/// Seeds are placed on a ring in the XY plane. Integration uses a simple
/// fixed-step Euler method for clarity; each streamline runs for up to 200
/// steps or until it exits a bounding sphere.
///
/// Returns `(paths, scalars)` where each `paths[i]` is the list of 3-D
/// positions for streamline `i`, and `scalars[i]` holds the per-vertex speed
/// (magnitude of the velocity vector at that point).
fn integrate_streamlines(seed_count: usize, step_size: f32) -> (Vec<Vec<[f32; 3]>>, Vec<Vec<f32>>) {
    use std::f32::consts::TAU;

    let mut paths: Vec<Vec<[f32; 3]>> = Vec::with_capacity(seed_count);
    let mut scalars: Vec<Vec<f32>> = Vec::with_capacity(seed_count);

    let seed_radius = 2.5_f32;
    let max_steps = 200_usize;
    let bound = 6.0_f32;

    for i in 0..seed_count {
        let angle = (i as f32 / seed_count as f32) * TAU;
        let sx = seed_radius * angle.cos();
        let sy = seed_radius * angle.sin();
        let sz = (i as f32 / seed_count as f32 - 0.5) * 2.0; // spread in Z

        let mut p = [sx, sy, sz];
        let mut path: Vec<[f32; 3]> = Vec::with_capacity(max_steps);
        let mut spd: Vec<f32> = Vec::with_capacity(max_steps);

        path.push(p);
        spd.push(velocity_magnitude(p));

        for _ in 0..max_steps {
            let v = velocity(p);
            let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if mag < 1e-4 {
                break;
            }
            p[0] += v[0] * step_size;
            p[1] += v[1] * step_size;
            p[2] += v[2] * step_size;

            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if r > bound {
                break;
            }
            path.push(p);
            spd.push(velocity_magnitude(p));
        }

        if path.len() >= 2 {
            paths.push(path);
            scalars.push(spd);
        }
    }

    (paths, scalars)
}

/// Analytic velocity field: a spiraling vortex with a gentle upward drift.
///
/// The field is:
///   vx = -y / (r² + ε)  (tangential)
///   vy =  x / (r² + ε)
///   vz =  0.3 * exp(-r² / 4)  (upwelling near origin)
fn velocity(p: [f32; 3]) -> [f32; 3] {
    let x = p[0];
    let y = p[1];
    let z = p[2];
    let r2 = x * x + y * y + z * z;
    let eps = 0.5_f32;
    let denom = r2 + eps;
    let vx = -y / denom;
    let vy = x / denom;
    let vz = 0.3 * (-r2 / 4.0).exp();
    [vx, vy, vz]
}

fn velocity_magnitude(p: [f32; 3]) -> f32 {
    let v = velocity(p);
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Flatten a list of per-streamline paths into concatenated buffers.
///
/// Returns `(positions, strip_lengths, scalars)` suitable for direct use in
/// `PolylineItem` and `StreamtubeItem`.
fn flatten_paths(
    paths: &[Vec<[f32; 3]>],
    scalars: &[Vec<f32>],
) -> (Vec<[f32; 3]>, Vec<u32>, Vec<f32>) {
    let total: usize = paths.iter().map(|p| p.len()).sum();
    let mut positions = Vec::with_capacity(total);
    let mut strip_lengths = Vec::with_capacity(paths.len());
    let mut flat_scalars = Vec::with_capacity(total);

    for (path, spd) in paths.iter().zip(scalars.iter()) {
        positions.extend_from_slice(path);
        strip_lengths.push(path.len() as u32);
        flat_scalars.extend_from_slice(spd);
    }

    (positions, strip_lengths, flat_scalars)
}
