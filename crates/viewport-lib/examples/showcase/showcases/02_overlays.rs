//! Overlays and annotations: the 2D layer viewport-lib draws over the 3D scene
//! after post-processing. World-anchored labels track scene points; a
//! `GlyphRunItem` draws pre-positioned, per-glyph-coloured glyphs (the low-level
//! text path); and a gallery of `OverlayShapeItem`s and `OverlayPolylineItem`s
//! exercises SDF shapes, fills, gradients, shadows, border modes, animations,
//! texture masking, 9-slice, clip masks, backdrop blur, and stroke patterns.
//!
//! Everything overlay-side is rebuilt each frame and pushed into
//! `session.frame_data_mut().overlays` after `ctx.drive_camera()`, because
//! assembly resets the overlay frame (including its animation clock) each frame.

use std::f32::consts::{PI, TAU};
use viewport_lib as vpl;

use eframe::egui;
use glam::{Mat4, Vec3};
use vpl::{
    AnimTrack, BorderMode, FontHandle, GlyphRunItem, GradientStop, LabelAnchor, LabelItem, LineCap,
    Material, NineSlice, OverlayAnimation, OverlayAnimations, OverlayEasing, OverlayFill,
    OverlayPolylineItem, OverlayShape, OverlayShapeItem, OverlayTextureId, PolylineCap,
    PositionedGlyph, RepeatMode, StrokePattern, TextureTransform, TileMode, TriangleDirection,
    primitives,
};

use crate::showcase::{SetupCtx, Showcase, ShowcaseCtx};

/// The three world anchors the labels attach to (Z-up).
const PEAK: [f32; 3] = [2.5, 0.5, 2.0];
const TROUGH: [f32; 3] = [-2.5, -0.5, 0.4];
const ORIGIN: [f32; 3] = [0.0, 0.0, 0.6];

/// Screen-space centre of the figure-eight path/dot (polyline row).
const FIG8_CX: f32 = 900.0;
const FIG8_CY: f32 = 790.0;

/// Carl Gauss portrait, pre-converted raw RGBA, shared with the old showcase.
const CARLGAUSS_W: u32 = 1500;
const CARLGAUSS_H: u32 = 1000;
const CARLGAUSS_RGBA: &[u8] = include_bytes!("../../eframe_showcase/carlgauss.rgba");

pub struct OverlaysShowcase {
    show_labels: bool,
    show_glyph_run: bool,
    show_emoji: bool,
    /// Handle to a system color-emoji font, if one was found at setup.
    emoji_font: Option<FontHandle>,
    /// Glyph ids of the emoji to draw, resolved from the emoji font's cmap.
    emoji_glyphs: Vec<u16>,
    show_shapes: bool,
    show_tex_shapes: bool,
    corner_radius: f32,
    border_width: f32,
    backdrop_blur: f32,
    demo_tex: Option<OverlayTextureId>,
    nine_slice_tex: Option<OverlayTextureId>,
    carlgauss_tex: Option<OverlayTextureId>,
    /// Animation clock, accumulated from per-frame dt.
    time: f32,
}

impl OverlaysShowcase {
    pub fn new() -> Self {
        Self {
            show_labels: true,
            show_glyph_run: true,
            show_emoji: true,
            emoji_font: None,
            emoji_glyphs: Vec::new(),
            show_shapes: true,
            show_tex_shapes: true,
            corner_radius: 8.0,
            border_width: 1.5,
            backdrop_blur: 14.0,
            demo_tex: None,
            nine_slice_tex: None,
            carlgauss_tex: None,
            time: 0.0,
        }
    }
}

impl Showcase for OverlaysShowcase {
    fn name(&self) -> &str {
        "Overlays & annotations"
    }

    fn setup(&mut self, ctx: &mut SetupCtx) {
        // A small 3D scene so the labels have something to anchor to.
        let cube = ctx
            .session
            .resources_mut()
            .upload_mesh_data(ctx.device, &primitives::cube(1.0))
            .unwrap();
        for (pos, colour) in [
            (PEAK, [0.30, 0.80, 0.45]),
            (TROUGH, [0.90, 0.40, 0.30]),
            (ORIGIN, [0.85, 0.85, 0.88]),
        ] {
            ctx.session.scene_mut().add(
                Some(cube),
                Mat4::from_translation(Vec3::from(pos)),
                Material::from_colour(colour),
            );
        }

        ctx.session.camera_mut().distance = 14.0;

        // Overlay textures for the texture-masked and 9-slice shapes.
        let (w, h, rgba) = build_demo_texture();
        self.demo_tex = Some(
            ctx.session
                .resources_mut()
                .upload_overlay_texture(ctx.device, ctx.queue, w, h, &rgba),
        );
        let (w, h, rgba) = build_nine_slice_texture();
        self.nine_slice_tex = Some(
            ctx.session
                .resources_mut()
                .upload_overlay_texture(ctx.device, ctx.queue, w, h, &rgba),
        );
        self.carlgauss_tex = Some(ctx.session.resources_mut().upload_overlay_texture(
            ctx.device,
            ctx.queue,
            CARLGAUSS_W,
            CARLGAUSS_H,
            CARLGAUSS_RGBA,
        ));

        self.load_emoji_font(ctx);
    }

    fn update(&mut self, ctx: &mut ShowcaseCtx) {
        self.time += ctx.dt;

        // Orbit/fly camera assembles the frame (and clears overlays).
        ctx.drive_camera();

        // Rebuild and inject the overlay layer for this frame.
        let session = &mut *ctx.session;
        let fd = session.frame_data_mut();
        fd.overlays.time = self.time as f64;

        if self.show_labels {
            self.build_labels(&mut fd.overlays.labels);
        }
        if self.show_glyph_run {
            self.build_glyph_run(&mut fd.overlays.glyph_runs);
        }
        if self.show_emoji {
            self.build_emoji_run(&mut fd.overlays.glyph_runs);
        }
        if self.show_shapes {
            self.build_shapes(&mut fd.overlays.shapes);
            self.build_polylines(&mut fd.overlays.polylines);
        }
    }

    fn description(&self) -> &str {
        "The 2D overlay layer: labels anchored to the 3D scene, a pre-positioned \
         glyph run, a colour-emoji row (when a system emoji font is present), and \
         a gallery of overlay shapes and polylines."
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Overlays draw over the 3D scene and ignore tone-mapping.");
        ui.label("Adjust the shape gallery in the right panel.");
        ui.label("Labels are anchored to the three cubes.");
    }

    fn has_controls(&self) -> bool {
        true
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Annotations");
        ui.checkbox(&mut self.show_labels, "World labels");
        ui.checkbox(&mut self.show_glyph_run, "Glyph run");
        ui.add_enabled(
            self.emoji_font.is_some(),
            egui::Checkbox::new(&mut self.show_emoji, "Colour emoji"),
        );
        if self.emoji_font.is_none() {
            ui.label(egui::RichText::new("No system colour-emoji font found.").weak());
        }

        ui.separator();
        ui.heading("Shape gallery");
        ui.checkbox(&mut self.show_shapes, "Show shapes");
        if self.show_shapes {
            ui.checkbox(&mut self.show_tex_shapes, "Texture-masked shapes");
            ui.add(egui::Slider::new(&mut self.corner_radius, 0.0..=40.0).text("Corner radius"));
            ui.add(egui::Slider::new(&mut self.border_width, 0.0..=6.0).text("Border width"));
            ui.add(egui::Slider::new(&mut self.backdrop_blur, 0.0..=40.0).text("Backdrop blur"));
        }
    }
}

// ---------------------------------------------------------------------------
// Overlay builders
// ---------------------------------------------------------------------------

impl OverlaysShowcase {
    fn build_labels(&self, out: &mut Vec<LabelItem>) {
        for (text, pos, colour) in [
            ("Peak +2.0 m", PEAK, [0.3, 1.0, 0.5, 1.0]),
            ("Trough +0.4 m", TROUGH, [1.0, 0.4, 0.3, 1.0]),
            ("Origin", ORIGIN, [0.9, 0.9, 0.9, 1.0]),
        ] {
            out.push(
                LabelItem::new(text)
                    .with_world_anchor(pos)
                    .with_colour(colour)
                    .with_leader_line(true)
                    .with_leader_colour([colour[0], colour[1], colour[2], 0.7])
                    .with_background(true)
                    .with_background_colour([0.0, 0.0, 0.0, 0.5])
                    .with_border_radius(4.0)
                    .with_padding(4.0)
                    .with_align_x(LabelAnchor::Left),
            );
        }
    }

    /// A `GlyphRunItem`: glyphs placed and coloured one by one, which is what a
    /// shaping engine would feed. `LabelItem` takes a string and lays it out;
    /// this takes the glyphs already positioned, so here they ride an animated
    /// wave with a per-glyph rainbow tint. The ids are raw font glyph indices
    /// (no character lookup): the point is the caller-owned layout, not the text.
    fn build_glyph_run(&self, out: &mut Vec<GlyphRunItem>) {
        let count = 24usize;
        let mut glyphs = Vec::with_capacity(count);
        let mut colours = Vec::with_capacity(count);
        for i in 0..count {
            let x = i as f32 * 24.0;
            let y = (i as f32 * 0.5 + self.time * 2.0).sin() * 12.0;
            // Bias into the letter/digit range of the built-in font; a few ids
            // may be blank and are simply skipped.
            glyphs.push(PositionedGlyph::new(20 + i as u16, x, y));
            let [r, g, b] = hsv_to_rgb(i as f32 / count as f32, 0.85, 1.0);
            colours.push([r, g, b, 1.0]);
        }
        out.push(
            GlyphRunItem::new(glyphs)
                // Just above the emoji row at the bottom.
                .with_position([40.0, 870.0])
                .with_font_size(30.0)
                .with_colours(colours),
        );
    }

    /// Load a system color-emoji font, if one is present, and resolve a handful of
    /// emoji to glyph ids. The overlay atlas draws these from the font's bitmap
    /// strikes (sbix / CBDT), so they come out in full colour rather than as
    /// monochrome coverage. Absent a font, the emoji row simply does not appear.
    fn load_emoji_font(&mut self, ctx: &mut SetupCtx) {
        const CANDIDATES: &[&str] = &[
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
        ];
        let Some(bytes) = CANDIDATES.iter().find_map(|p| std::fs::read(p).ok()) else {
            return;
        };
        let Ok(handle) = ctx.session.resources_mut().upload_font(&bytes) else {
            return;
        };
        // Resolve characters to glyph ids from the same bytes the atlas rasterizes.
        let Ok(face) = ttf_parser::Face::parse(&bytes, 0) else {
            return;
        };
        let wanted = ['😀', '😍', '👍', '🎉', '🚀', '🌍', '🔥', '⭐', '🍕', '🎨'];
        let glyphs: Vec<u16> = wanted
            .iter()
            .filter_map(|&c| face.glyph_index(c).map(|g| g.0))
            .collect();
        if glyphs.is_empty() {
            return;
        }
        self.emoji_font = Some(handle);
        self.emoji_glyphs = glyphs;
    }

    /// A `GlyphRunItem` of color emoji, laid out in a row with a gentle bob. The
    /// run carries no colour: color glyphs draw their own bitmap and take only the
    /// run opacity, so the tint is ignored.
    fn build_emoji_run(&self, out: &mut Vec<GlyphRunItem>) {
        let Some(font) = self.emoji_font else {
            return;
        };
        let size = 44.0;
        let spacing = size * 1.15;
        let glyphs: Vec<PositionedGlyph> = self
            .emoji_glyphs
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let x = i as f32 * spacing;
                let y = (i as f32 * 0.6 + self.time * 2.0).sin() * 6.0;
                PositionedGlyph::new(id, x, y)
            })
            .collect();
        out.push(
            GlyphRunItem::new(glyphs)
                .with_font(font)
                // A new bottom row, below the shape and polyline rows.
                .with_position([40.0, 930.0])
                .with_font_size(size),
        );
    }

    fn build_shapes(&self, out: &mut Vec<OverlayShapeItem>) {
        self.row_shape_kinds(out);
        self.row_gradients(out);
        self.row_shadows(out);
        self.row_border_anim(out);
        self.row_extra_shapes(out);
        self.row_textures(out);
        self.push_fig8_dot(out);
    }

    fn row_shape_kinds(&self, out: &mut Vec<OverlayShapeItem>) {
        let cr = self.corner_radius;
        let bw = self.border_width;
        let gap = 16.0;
        // Row 1: SDF shape kinds.
        let mut x = 20.0;
        let y = 150.0;
        let row1: &[(f32, f32, OverlayShape, [f32; 4], [f32; 4])] = &[
            (
                120.0,
                70.0,
                OverlayShape::Rect { corner_radius: cr },
                [0.12, 0.12, 0.18, 0.85],
                [0.8, 0.8, 0.8, 0.9],
            ),
            (
                120.0,
                70.0,
                OverlayShape::RoundedRect {
                    radii: [cr, 0.0, cr, 0.0],
                },
                [0.05, 0.15, 0.25, 0.85],
                [0.3, 0.7, 1.0, 0.9],
            ),
            (
                70.0,
                70.0,
                OverlayShape::Circle,
                [0.2, 0.5, 0.15, 0.85],
                [0.4, 1.0, 0.3, 0.9],
            ),
            (
                120.0,
                60.0,
                OverlayShape::Ellipse,
                [0.3, 0.1, 0.3, 0.85],
                [0.8, 0.4, 1.0, 0.9],
            ),
            (
                120.0,
                40.0,
                OverlayShape::Capsule,
                [0.3, 0.2, 0.05, 0.85],
                [1.0, 0.8, 0.3, 0.9],
            ),
            (
                70.0,
                70.0,
                OverlayShape::Ring {
                    inner_radius_frac: 0.65,
                },
                [0.15, 0.35, 0.5, 0.85],
                [0.3, 0.8, 1.0, 0.9],
            ),
            (
                70.0,
                70.0,
                OverlayShape::Arc {
                    inner_radius_frac: 0.6,
                    start_angle: 0.0,
                    end_angle: PI * 1.5,
                },
                [0.5, 0.2, 0.1, 0.85],
                [1.0, 0.5, 0.2, 0.9],
            ),
            (
                60.0,
                60.0,
                OverlayShape::Triangle {
                    direction: TriangleDirection::Up,
                },
                [0.4, 0.4, 0.1, 0.85],
                [1.0, 1.0, 0.3, 0.9],
            ),
        ];
        for (w, h, shape, fill, border) in row1.iter().cloned() {
            out.push(
                OverlayShapeItem::new(shape, [x, y + (70.0 - h) * 0.5], [w, h])
                    .with_fill(OverlayFill::Solid(fill))
                    .with_border(border, bw),
            );
            x += w + gap;
        }
        // Backdrop-blur circle sits at the end of row 1.
        if self.backdrop_blur > 0.0 {
            out.push(
                OverlayShapeItem::new(OverlayShape::Circle, [x, y - 35.0], [140.0, 140.0])
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 0.12]))
                    .with_border([1.0, 1.0, 1.0, 0.3], 1.0)
                    .with_backdrop_blur(self.backdrop_blur),
            );
        }
    }

    fn row_gradients(&self, out: &mut Vec<OverlayShapeItem>) {
        let cr = self.corner_radius;
        let bw = self.border_width;
        // Row 2: fills and gradients.
        let mut x = 20.0;
        let y = 260.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: cr },
                [x, y],
                [120.0, 70.0],
            )
            .with_fill(OverlayFill::LinearGradient {
                start_colour: [0.05, 0.15, 0.55, 0.9],
                end_colour: [0.05, 0.65, 0.65, 0.9],
                angle: 0.0,
            })
            .with_border([0.3, 0.7, 1.0, 0.8], bw),
        );
        x += 136.0;
        out.push(
            OverlayShapeItem::new(OverlayShape::Circle, [x, y], [70.0, 70.0])
                .with_fill(OverlayFill::RadialGradient {
                    centre_colour: [1.0, 0.95, 0.7, 1.0],
                    edge_colour: [0.2, 0.05, 0.0, 0.9],
                })
                .with_border([1.0, 0.8, 0.4, 0.8], bw),
        );
        x += 86.0;
        out.push(
            OverlayShapeItem::new(OverlayShape::Circle, [x, y], [70.0, 70.0])
                .with_fill(OverlayFill::ConicalGradient {
                    start_colour: [0.95, 0.2, 0.4, 1.0],
                    end_colour: [0.2, 0.6, 1.0, 1.0],
                    offset_angle: 0.0,
                })
                .with_border([0.9, 0.9, 0.9, 0.8], bw),
        );
        x += 86.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: cr },
                [x, y],
                [120.0, 70.0],
            )
            .with_fill(OverlayFill::LinearGradientMulti {
                stops: vec![
                    GradientStop::new(0.0, [0.05, 0.05, 0.20, 1.0]),
                    GradientStop::new(0.4, [0.55, 0.10, 0.45, 1.0]),
                    GradientStop::new(0.75, [0.95, 0.45, 0.20, 1.0]),
                    GradientStop::new(1.0, [1.0, 0.95, 0.55, 1.0]),
                ],
                angle: 0.0,
            })
            .with_border([1.0, 0.85, 0.4, 0.8], bw),
        );
        x += 136.0;
        out.push(
            OverlayShapeItem::new(OverlayShape::Circle, [x, y], [70.0, 70.0])
                .with_fill(OverlayFill::ConicalGradientMulti {
                    stops: vec![
                        GradientStop::new(0.0, [0.95, 0.2, 0.2, 1.0]),
                        GradientStop::new(0.25, [1.0, 0.85, 0.2, 1.0]),
                        GradientStop::new(0.6, [0.2, 0.85, 0.45, 1.0]),
                        GradientStop::new(1.0, [0.3, 0.5, 1.0, 1.0]),
                    ],
                    offset_angle: 0.0,
                })
                .with_border([0.9, 0.9, 0.9, 0.8], bw),
        );
    }

    fn row_shadows(&self, out: &mut Vec<OverlayShapeItem>) {
        let cr = self.corner_radius;
        let bw = self.border_width;
        // Row 3: shadows, glow, inset.
        let mut x = 20.0;
        let y = 360.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: cr },
                [x, y],
                [120.0, 70.0],
            )
            .with_fill(OverlayFill::Solid([0.15, 0.15, 0.2, 0.95]))
            .with_border([0.5, 0.5, 0.6, 0.8], bw)
            .with_shadow([0.0, 0.0, 0.0, 0.5], 12.0, [4.0, 4.0]),
        );
        x += 152.0;
        out.push(
            OverlayShapeItem::new(OverlayShape::Circle, [x, y], [70.0, 70.0])
                .with_fill(OverlayFill::Solid([0.1, 0.15, 0.35, 0.95]))
                .with_border([0.3, 0.5, 1.0, 0.9], bw)
                .with_shadow([0.2, 0.4, 1.0, 0.6], 16.0, [0.0, 0.0]),
        );
        x += 102.0;
        out.push(
            OverlayShapeItem::new(OverlayShape::Capsule, [x, y + 15.0], [120.0, 40.0])
                .with_fill(OverlayFill::Solid([0.3, 0.15, 0.05, 0.95]))
                .with_border([1.0, 0.6, 0.2, 0.9], bw)
                .with_shadow([1.0, 0.5, 0.1, 0.45], 14.0, [0.0, 2.0]),
        );
        x += 152.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: cr },
                [x, y],
                [120.0, 70.0],
            )
            .with_fill(OverlayFill::Solid([0.22, 0.24, 0.30, 1.0]))
            .with_border([0.05, 0.07, 0.12, 0.9], 1.0)
            .with_shadow([0.0, 0.0, 0.0, 0.7], 14.0, [0.0, 4.0])
            .with_shadow_inset(true),
        );
    }

    fn row_border_anim(&self, out: &mut Vec<OverlayShapeItem>) {
        let cr = self.corner_radius;
        let bw = self.border_width;
        // Row 4: border modes and animations.
        let mut x = 20.0;
        let y = 460.0;
        for (mode, colour) in [
            (BorderMode::Inset, [0.9, 0.9, 0.3, 1.0]),
            (BorderMode::Outer, [0.3, 0.9, 0.5, 1.0]),
            (BorderMode::Center, [0.5, 0.5, 1.0, 1.0]),
        ] {
            out.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: cr },
                    [x, y],
                    [90.0, 70.0],
                )
                .with_fill(OverlayFill::Solid([0.15, 0.15, 0.2, 0.9]))
                .with_border(colour, 3.0)
                .with_border_mode(mode),
            );
            x += 106.0;
        }
        // Pulsing circle (built-in animation, resolved against overlays.time).
        out.push(
            OverlayShapeItem::new(OverlayShape::Circle, [x, y], [70.0, 70.0])
                .with_fill(OverlayFill::Solid([0.2, 0.5, 1.0, 0.9]))
                .with_border([0.4, 0.7, 1.0, 0.9], bw)
                .with_animation(OverlayAnimation::Pulse {
                    start_time: 0.0,
                    period: 2.0,
                }),
        );
        x += 86.0;
        // Fade-in capsule that restarts every 4 seconds.
        let cycle = 4.0_f64;
        let fade_start = (self.time as f64 / cycle).floor() * cycle;
        out.push(
            OverlayShapeItem::new(OverlayShape::Capsule, [x, y + 15.0], [120.0, 40.0])
                .with_fill(OverlayFill::Solid([0.6, 0.2, 0.1, 0.9]))
                .with_border([1.0, 0.5, 0.3, 0.9], bw)
                .with_animation(OverlayAnimation::FadeIn {
                    start_time: fade_start,
                    duration: 3.0,
                }),
        );
        x += 136.0;
        // Multi-channel: a sliding rect via the position track.
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 4.0 },
                [x, y + 20.0],
                [44.0, 28.0],
            )
            .with_fill(OverlayFill::Solid([0.95, 0.65, 0.25, 0.95]))
            .with_border([1.0, 0.85, 0.4, 0.9], bw)
            .with_animations(OverlayAnimations::default().with_position(AnimTrack {
                start_time: 0.0,
                duration: 1.8,
                from: [x, y + 20.0],
                to: [x + 50.0, y + 20.0],
                easing: OverlayEasing::EaseInOut,
                repeat: RepeatMode::PingPong,
            })),
        );
    }

    fn row_extra_shapes(&self, out: &mut Vec<OverlayShapeItem>) {
        let bw = self.border_width;
        // Row 5: extra shape kinds, some rotating (manual clock).
        let t = self.time;
        let mut x = 20.0;
        let y = 560.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Line {
                    thickness: 6.0,
                    cap: LineCap::Round,
                },
                [x, y],
                [100.0, 70.0],
            )
            .with_fill(OverlayFill::Solid([0.2, 0.7, 1.0, 0.9])),
        );
        x += 116.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Star {
                    points: 5,
                    inner_radius_frac: 0.45,
                },
                [x, y],
                [70.0, 70.0],
            )
            .with_fill(OverlayFill::Solid([1.0, 0.85, 0.1, 0.9]))
            .with_border([1.0, 1.0, 0.5, 0.9], bw)
            .with_rotation(t * 0.8),
        );
        x += 86.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::RegularPolygon { sides: 6 },
                [x, y],
                [70.0, 70.0],
            )
            .with_fill(OverlayFill::Solid([0.1, 0.5, 0.9, 0.9]))
            .with_border([0.3, 0.7, 1.0, 0.9], bw),
        );
        x += 86.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Cross {
                    arm_width_frac: 0.35,
                },
                [x, y],
                [70.0, 70.0],
            )
            .with_fill(OverlayFill::Solid([0.3, 0.8, 0.5, 0.9]))
            .with_border([0.5, 1.0, 0.7, 0.9], bw)
            .with_rotation(-t * 1.2),
        );
        x += 86.0;
        // Clip-mask group: a rotating decagon clipped to its left half.
        let clip = [x, y];
        let clip_size = 70.0;
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 0.0 },
                clip,
                [clip_size * 0.5, clip_size],
            )
            .with_clip_mask(7),
        );
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 0.0 },
                clip,
                [clip_size * 0.5, clip_size],
            )
            .with_fill(OverlayFill::Solid([0.0, 0.0, 0.0, 0.0]))
            .with_border([1.0, 1.0, 1.0, 0.5], 1.0)
            .with_border_mode(BorderMode::Outer),
        );
        out.push(
            OverlayShapeItem::new(
                OverlayShape::RegularPolygon { sides: 10 },
                clip,
                [clip_size, clip_size],
            )
            .with_rotation(t * 0.6)
            .with_fill(OverlayFill::Solid([0.9, 0.55, 0.2, 0.95]))
            .with_border([1.0, 0.8, 0.3, 0.9], bw)
            .with_clip(7),
        );
    }

    fn row_textures(&self, out: &mut Vec<OverlayShapeItem>) {
        let cr = self.corner_radius;
        let bw = self.border_width;
        let t = self.time;
        // Row 6: texture-masked shapes and 9-slice A/B.
        if self.show_tex_shapes {
            let mut x = 20.0;
            let y = 660.0;
            if let Some(tid) = self.demo_tex {
                out.push(
                    OverlayShapeItem::new(OverlayShape::Circle, [x, y], [90.0, 90.0])
                        .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                        .with_border([1.0, 1.0, 1.0, 0.9], bw)
                        .with_texture(tid),
                );
                x += 106.0;
                // Same texture spinning inside a static circle.
                out.push(
                    OverlayShapeItem::new(OverlayShape::Circle, [x, y], [90.0, 90.0])
                        .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                        .with_border([1.0, 1.0, 1.0, 0.9], bw)
                        .with_texture(tid)
                        .with_texture_transform(TextureTransform {
                            rotation: t * 0.5,
                            ..Default::default()
                        }),
                );
                x += 106.0;
                // Tiled 3x3.
                out.push(
                    OverlayShapeItem::new(
                        OverlayShape::Rect { corner_radius: cr },
                        [x, y],
                        [150.0, 90.0],
                    )
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                    .with_border([1.0, 1.0, 1.0, 0.9], bw)
                    .with_texture(tid)
                    .with_texture_transform(TextureTransform {
                        scale: [3.0, 3.0],
                        tile_mode: TileMode::Tile,
                        ..Default::default()
                    }),
                );
                x += 166.0;
            }
            // Carl Gauss portrait masked into a rounded rect.
            if let Some(tid) = self.carlgauss_tex {
                out.push(
                    OverlayShapeItem::new(
                        OverlayShape::Rect { corner_radius: cr },
                        [x, y],
                        [140.0, 90.0],
                    )
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                    .with_border([0.8, 0.8, 0.8, 0.9], bw)
                    .with_texture(tid),
                );
                x += 156.0;
            }
            // 9-slice A/B: stretched vs corner-preserving.
            if let Some(tid) = self.nine_slice_tex {
                out.push(
                    OverlayShapeItem::new(
                        OverlayShape::Rect { corner_radius: 0.0 },
                        [x, y],
                        [180.0, 90.0],
                    )
                    .with_texture(tid)
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0])),
                );
                x += 196.0;
                out.push(
                    OverlayShapeItem::new(
                        OverlayShape::Rect { corner_radius: 0.0 },
                        [x, y],
                        [180.0, 90.0],
                    )
                    .with_texture(tid)
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
                    .with_nine_slice(NineSlice {
                        insets_px: [10.0, 10.0, 10.0, 10.0],
                        centre_mode: TileMode::Stretch,
                        edge_mode: TileMode::Stretch,
                    }),
                );
            }
        }
    }

    fn push_fig8_dot(&self, out: &mut Vec<OverlayShapeItem>) {
        let bw = self.border_width;
        // The dot tracing the figure-eight drawn in build_polylines.
        let phase = (self.time * 0.22).fract();
        let p = infinity_bezier_point(phase, FIG8_CX, FIG8_CY);
        out.push(
            OverlayShapeItem::new(
                OverlayShape::Circle,
                [p[0] - 11.0, p[1] - 11.0],
                [22.0, 22.0],
            )
            .with_fill(OverlayFill::Solid([0.95, 0.45, 0.85, 1.0]))
            .with_border([1.0, 0.7, 0.95, 0.9], bw),
        );
    }

    fn build_polylines(&self, out: &mut Vec<OverlayPolylineItem>) {
        let t = self.time;
        let y = 790.0;
        let mut x = 40.0;

        // Closed polygon with a gradient fill.
        out.push(
            OverlayPolylineItem::new(vec![
                [x, y - 30.0],
                [x + 48.0, y - 12.0],
                [x + 42.0, y + 28.0],
                [x - 18.0, y + 22.0],
                [x - 36.0, y - 10.0],
            ])
            .with_thickness(2.0)
            .with_colour([1.0, 1.0, 1.0, 0.8])
            .with_closed(true)
            .with_fill(OverlayFill::LinearGradient {
                start_colour: [0.12, 0.65, 0.95, 0.78],
                end_colour: [0.95, 0.35, 0.7, 0.82],
                angle: PI * 0.25,
            })
            .with_z_order(1),
        );
        x += 130.0;

        // Marching-ants dashed marquee.
        out.push(
            OverlayPolylineItem::new(vec![
                [x, y - 28.0],
                [x + 64.0, y - 28.0],
                [x + 64.0, y + 28.0],
                [x, y + 28.0],
            ])
            .with_thickness(2.0)
            .with_colour([1.0, 1.0, 1.0, 0.85])
            .with_closed(true)
            .with_stroke_pattern(StrokePattern::Dashed {
                dash_length: 8.0,
                gap_length: 6.0,
                offset: t * 20.0,
            })
            .with_z_order(1),
        );
        x += 120.0;

        // Round-capped dashed sine wave.
        out.push(
            OverlayPolylineItem::new(
                (0..=40)
                    .map(|i| {
                        let f = i as f32 / 40.0;
                        [x + f * 90.0, y + (f * TAU).sin() * 22.0]
                    })
                    .collect(),
            )
            .with_thickness(5.0)
            .with_colour([0.3, 0.9, 0.6, 0.9])
            .with_cap(PolylineCap::Round)
            .with_stroke_pattern(StrokePattern::Dashed {
                dash_length: 18.0,
                gap_length: 9.0,
                offset: 0.0,
            })
            .with_z_order(1),
        );
        x += 150.0;

        // Dotted circle.
        out.push(
            OverlayPolylineItem::new(
                (0..48)
                    .map(|i| {
                        let a = i as f32 / 48.0 * TAU;
                        [x + a.cos() * 26.0, y + a.sin() * 26.0]
                    })
                    .collect(),
            )
            .with_thickness(4.0)
            .with_colour([1.0, 0.75, 0.3, 0.9])
            .with_closed(true)
            .with_stroke_pattern(StrokePattern::Dotted {
                spacing: 10.0,
                offset: 0.0,
            })
            .with_z_order(1),
        );
        x += 90.0;

        // Closure-generated wobbling blob, resampled each frame.
        let x_blob = x + 60.0;
        let blob_path = move |u: f32| {
            let a = u * TAU;
            let r = 28.0 + 6.0 * (a * 3.0 + t * 2.0).sin();
            [x_blob + a.cos() * r, y + a.sin() * r]
        };
        out.push(
            OverlayPolylineItem::closed_from_path(
                blob_path,
                64,
                Some(OverlayFill::RadialGradient {
                    centre_colour: [0.9, 0.7, 0.2, 0.85],
                    edge_colour: [0.7, 0.2, 0.5, 0.85],
                }),
                [1.0, 1.0, 1.0, 0.85],
                2.0,
            )
            .with_z_order(1),
        );

        // The figure-eight path outline; the moving dot (a shape) is added in
        // build_shapes, sharing the same centre so the two stay in sync.
        let _ = x_blob;
        let trace = OverlayPolylineItem::from_path(
            |u| infinity_bezier_point(u, FIG8_CX, FIG8_CY),
            160,
            2.0,
            [1.0, 1.0, 1.0, 0.45],
        );
        out.push(trace.with_closed(true).with_z_order(0));

        // Carl Gauss portrait masked into closed textured polylines: a heart
        // (explicit UVs) and a rounded pentagon (bounds UVs).
        if let Some(tid) = self.carlgauss_tex {
            let hx = 1040.0;
            let hy = y;
            let heart: Vec<[f32; 2]> = (0..40)
                .map(|i| {
                    let a = i as f32 / 40.0 * TAU;
                    let s = a.sin();
                    let c = a.cos();
                    [
                        hx + 2.1 * 16.0 * s.powi(3),
                        hy - 2.1
                            * (13.0 * c
                                - 5.0 * (2.0 * a).cos()
                                - 2.0 * (3.0 * a).cos()
                                - (4.0 * a).cos()),
                    ]
                })
                .collect();
            let uvs: Vec<[f32; 2]> = heart
                .iter()
                .map(|p| [(p[0] - (hx - 36.0)) / 72.0, (p[1] - (hy - 36.0)) / 72.0])
                .collect();
            out.push(
                OverlayPolylineItem::new(heart)
                    .with_thickness(2.0)
                    .with_colour([1.0, 0.78, 0.9, 0.9])
                    .with_closed(true)
                    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 0.95]))
                    .with_texture(tid)
                    .with_uvs(uvs)
                    .with_z_order(1),
            );

            let px = 1160.0;
            let mut pentagon = OverlayPolylineItem::closed_from_path(
                |u| {
                    let a = u * TAU;
                    let r = 30.0 + (a * 5.0).cos() * 3.0;
                    [px + a.cos() * r, y + a.sin() * r]
                },
                60,
                Some(OverlayFill::Solid([1.0, 1.0, 1.0, 0.95])),
                [1.0, 0.9, 0.7, 0.9],
                2.0,
            );
            pentagon.texture = Some(tid);
            pentagon.z_order = 1;
            out.push(pentagon);
        }
    }
}

// ---------------------------------------------------------------------------
// Procedural textures and path helpers
// ---------------------------------------------------------------------------

/// A 128x128 colour-wheel: hue from angle, saturation from radius. Shows up
/// well under any SDF mask.
fn build_demo_texture() -> (u32, u32, Vec<u8>) {
    let size: u32 = 128;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let nx = (x as f32 / (size - 1) as f32) * 2.0 - 1.0;
            let ny = (y as f32 / (size - 1) as f32) * 2.0 - 1.0;
            let hue = (ny.atan2(nx) / TAU + 0.5).fract();
            let sat = (nx * nx + ny * ny).sqrt().min(1.0);
            let [r, g, b] = hsv_to_rgb(hue, sat, 1.0);
            px.push((r * 255.0) as u8);
            px.push((g * 255.0) as u8);
            px.push((b * 255.0) as u8);
            px.push(255);
        }
    }
    (size, size, px)
}

/// A 64x32 rounded-rect button texture for the 9-slice demo: transparent
/// outside the rounded shape, gradient body, rim, and a top bevel highlight.
fn build_nine_slice_texture() -> (u32, u32, Vec<u8>) {
    let w: u32 = 64;
    let h: u32 = 32;
    let mut px = vec![0u8; (w * h * 4) as usize];
    let radius = 10.0_f32;
    let rim = 2.0_f32;
    let body_top = [0.30_f32, 0.62, 0.90];
    let body_bottom = [0.10_f32, 0.25, 0.55];
    let rim_col = [0.04_f32, 0.08, 0.18];
    let bevel = [0.80_f32, 0.92, 1.0];
    let sd = |fx: f32, fy: f32| {
        let half_w = w as f32 * 0.5;
        let half_h = h as f32 * 0.5;
        let qx = (fx - half_w).abs() - half_w + radius;
        let qy = (fy - half_h).abs() - half_h + radius;
        let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
        qx.max(qy).min(0.0) + outside - radius
    };
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;
            let d = sd(fx, fy);
            let i = ((y * w + x) * 4) as usize;
            if d > 0.5 {
                continue;
            }
            let v = (y as f32 / (h - 1) as f32).clamp(0.0, 1.0);
            let body = [
                body_top[0] * (1.0 - v) + body_bottom[0] * v,
                body_top[1] * (1.0 - v) + body_bottom[1] * v,
                body_top[2] * (1.0 - v) + body_bottom[2] * v,
            ];
            let c = if d > -rim {
                rim_col
            } else if d > -(rim + 1.0) && fy < h as f32 * 0.5 {
                bevel
            } else {
                body
            };
            let a = (1.0 - (d + 0.5).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            px[i] = (c[0] * 255.0) as u8;
            px[i + 1] = (c[1] * 255.0) as u8;
            px[i + 2] = (c[2] * 255.0) as u8;
            px[i + 3] = (a * 255.0) as u8;
        }
    }
    (w, h, px)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor() as u32 % 6;
    let f = h * 6.0 - (h * 6.0).floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// A stylised figure-eight from four cubic Bezier segments, centred on (cx, cy).
fn infinity_bezier_point(t: f32, cx: f32, cy: f32) -> [f32; 2] {
    let raw = (t.clamp(0.0, 1.0) * 4.0).min(4.0 - 1e-6);
    let seg = raw.floor() as usize;
    let u = raw - seg as f32;
    let lw = 70.0_f32;
    let lh = 44.0_f32;
    let (p0, p1, p2, p3) = match seg {
        0 => (
            [cx, cy],
            [cx + lw * 0.4, cy - lh * 1.4],
            [cx + lw * 1.2, cy - lh * 0.9],
            [cx + lw * 1.2, cy],
        ),
        1 => (
            [cx + lw * 1.2, cy],
            [cx + lw * 1.2, cy + lh * 0.9],
            [cx + lw * 0.4, cy + lh * 1.4],
            [cx, cy],
        ),
        2 => (
            [cx, cy],
            [cx - lw * 0.4, cy + lh * 1.4],
            [cx - lw * 1.2, cy + lh * 0.9],
            [cx - lw * 1.2, cy],
        ),
        _ => (
            [cx - lw * 1.2, cy],
            [cx - lw * 1.2, cy - lh * 0.9],
            [cx - lw * 0.4, cy - lh * 1.4],
            [cx, cy],
        ),
    };
    let o = 1.0 - u;
    let w0 = o * o * o;
    let w1 = 3.0 * o * o * u;
    let w2 = 3.0 * o * u * u;
    let w3 = u * u * u;
    [
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
    ]
}
