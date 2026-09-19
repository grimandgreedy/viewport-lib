//! Font atlas and single-line text layout for overlay rendering.
//!
//! This module is the text back-end for [`LabelItem`](crate::LabelItem) and
//! [`GlyphRunItem`](crate::GlyphRunItem).  It uses [`fontdue`] for glyph
//! rasterization and packs glyphs into a single GPU texture atlas on demand.
//!
//! Public surface: [`FontHandle`] (opaque font identifier) and
//! [`super::DeviceResources::upload_font`].  Everything else is `pub(crate)`.

use std::collections::HashMap;

/// Default font embedded in the library binary (Inter Regular, SIL OFL 1.1).
const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("../../fonts/Inter-Regular.ttf");

// ---------------------------------------------------------------------------
// FontHandle : public opaque identifier
// ---------------------------------------------------------------------------

// The handle a consumer names on an overlay item lives in `viewport-lib-types`;
// the font store and rasterization below stay here. Re-exported so the existing
// `crate::resources::overlay::font::FontHandle` path keeps resolving.
pub use viewport_lib_types::overlay::font::FontHandle;

// ---------------------------------------------------------------------------
// GlyphKey / GlyphEntry : atlas bookkeeping
// ---------------------------------------------------------------------------

/// How a glyph cell is post-processed after rasterization.
///
/// The plain style is the glyph itself. A shadow style grows the coverage by
/// `spread`, fades it over `blur`, and shapes that fade by `falloff`, producing
/// a cell that is drawn behind the glyph in the shadow colour. Baking it here
/// rather than at draw time means the work happens once per distinct style and
/// size, not per frame, and the result is exact rather than an approximation
/// built from offset copies.
///
/// All three are quantised for the same reason the font size is: to keep the
/// number of distinct atlas cells bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(crate) struct GlyphStyle {
    /// Dilation in tenths of a physical pixel.
    spread_tenths: u32,
    /// Fade distance in tenths of a physical pixel.
    blur_tenths: u32,
    /// Falloff exponent in tenths. `0` marks the plain (unstyled) glyph.
    falloff_tenths: u32,
}

/// Largest dilation or blur honoured on a glyph cell, in physical pixels.
/// Past this the cell dwarfs the glyph and the atlas cost stops being worth it.
const MAX_GLYPH_STYLE_PX: f32 = 32.0;

impl GlyphStyle {
    /// The glyph as rasterized, with no shadow processing.
    pub(crate) const PLAIN: Self = Self {
        spread_tenths: 0,
        blur_tenths: 0,
        falloff_tenths: 0,
    };

    /// Build a style from a shadow layer's physical-pixel spread and blur.
    /// Returns [`GlyphStyle::PLAIN`] when the layer would not change the cell.
    pub(crate) fn from_shadow(spread_px: f32, blur_px: f32, falloff: f32) -> Self {
        let spread = spread_px.clamp(0.0, MAX_GLYPH_STYLE_PX);
        let blur = blur_px.clamp(0.0, MAX_GLYPH_STYLE_PX);
        if spread <= 0.0 && blur <= 0.0 {
            return Self::PLAIN;
        }
        Self {
            spread_tenths: (spread * 10.0).round() as u32,
            blur_tenths: (blur * 10.0).round() as u32,
            falloff_tenths: ((falloff.clamp(0.05, 16.0)) * 10.0).round().max(1.0) as u32,
        }
    }

    fn is_plain(&self) -> bool {
        self.falloff_tenths == 0
    }

    fn spread(&self) -> f32 {
        self.spread_tenths as f32 * 0.1
    }

    fn blur(&self) -> f32 {
        self.blur_tenths as f32 * 0.1
    }

    fn falloff(&self) -> f32 {
        self.falloff_tenths as f32 * 0.1
    }

    /// Physical pixels the styled cell grows on every side.
    fn pad(&self) -> u32 {
        if self.is_plain() {
            return 0;
        }
        (self.spread() + self.blur()).ceil() as u32 + 1
    }
}

/// Unique key for a rasterized glyph in the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey {
    font_index: usize,
    glyph_index: u16,
    /// Font size in tenths of a pixel (e.g. 140 = 14.0 px).
    /// Quantised to avoid unbounded atlas growth from fractional sizes.
    size_tenths: u32,
    /// Post-processing applied to the cell (plain glyph, or a shadow bake).
    style: GlyphStyle,
}

/// Location and metrics of a single rasterized glyph in the atlas texture.
#[derive(Debug, Clone, Copy)]
struct GlyphEntry {
    /// Top-left pixel coordinate in the atlas.
    x: u32,
    y: u32,
    /// Rasterized bitmap dimensions.
    width: u32,
    height: u32,
    /// Offset from the pen position to the top-left of the bitmap.
    offset_x: f32,
    offset_y: f32,
    /// `true` for a color glyph (the atlas cell holds real RGBA); `false` for a
    /// coverage glyph (the cell holds `[255, 255, 255, coverage]`).
    color: bool,
}

// ---------------------------------------------------------------------------
// GlyphQuad / TextLayout : internal layout output
// ---------------------------------------------------------------------------

/// A positioned, textured quad for one glyph, ready for vertex generation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GlyphQuad {
    /// Screen-space top-left corner (pixels from top-left of viewport).
    pub pos: [f32; 2],
    /// Screen-space size [w, h] in pixels.
    pub size: [f32; 2],
    /// UV top-left in the atlas (0..1).
    pub uv_min: [f32; 2],
    /// UV bottom-right in the atlas (0..1).
    pub uv_max: [f32; 2],
    /// `true` when the atlas cell holds a color bitmap (drawn as-is), `false` for
    /// a coverage cell (tinted by the run colour).
    pub color: bool,
}

/// Metrics for an overlay text run, matching how a [`LabelItem`] with the same
/// text, size, and font would be laid out.
///
/// All values are in logical pixels. Use these to size and align overlay-based
/// UI (panel widths, right-aligned columns, per-row hit rectangles, vertical
/// centring) before drawing the text with a `LabelItem`.
///
/// [`LabelItem`]: crate::renderer::types::LabelItem
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextMetrics {
    /// Total advance width of the run in logical pixels. For multi-line text
    /// (embedded `\n`) this is the width of the widest line.
    pub width: f32,
    /// Total height in logical pixels: one line height per line.
    pub height: f32,
    /// Distance from the top of the run to the first baseline, in logical
    /// pixels. Use for baseline-accurate vertical placement.
    pub ascent: f32,
}

/// Layout result for a single-line text string.
#[derive(Debug, Clone)]
pub(crate) struct TextLayout {
    /// One quad per visible glyph (whitespace characters are skipped).
    pub quads: Vec<GlyphQuad>,
    /// Total advance width of the laid-out string in pixels.
    pub total_width: f32,
    /// Line height in pixels (ascent - descent + line gap at the requested size).
    pub height: f32,
}

// ---------------------------------------------------------------------------
// GlyphAtlas
// ---------------------------------------------------------------------------

/// A dynamically-growing glyph atlas backed by a single `Rgba8Unorm` texture.
///
/// Owned by [`DeviceResources`]; never exposed in the public API.
pub(crate) struct GlyphAtlas {
    /// Parsed fontdue fonts.  Index 0 is always the built-in default.
    fonts: Vec<fontdue::Font>,

    /// Raw font bytes, parallel to `fonts`, kept so a `ttf_parser::Face` can be
    /// built on demand to read color bitmap strikes that fontdue does not.
    font_bytes: Vec<Vec<u8>>,

    /// Whether each font has a readable color bitmap table, computed once at
    /// upload. When `false`, glyphs skip the color path entirely.
    font_has_color: Vec<bool>,

    /// Cached rasterized glyphs.
    entries: HashMap<GlyphKey, GlyphEntry>,

    /// CPU-side atlas pixel data (single-channel alpha, packed row-major).
    /// Stored as RGBA for direct GPU upload: R=G=B=255, A=coverage.
    pixels: Vec<[u8; 4]>,

    /// Current atlas dimensions (always square, power of two).
    size: u32,

    /// Simple row-based packer state.
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,

    /// GPU texture (recreated when the atlas grows).
    pub texture: crate::gpu::Texture,
    /// View into the atlas texture.
    pub view: crate::gpu::TextureView,

    /// Set to `true` whenever new glyphs have been rasterized since the last
    /// GPU upload.  Cleared by [`GlyphAtlas::upload_if_dirty`].
    dirty: bool,

    /// Monotonic counter bumped every time the atlas grows. Growing doubles
    /// `size`, so every existing glyph's UV (`pixel / size`) changes even though
    /// its pixel position is unchanged. Cached glyph geometry (retained overlay
    /// groups) records the version it baked UVs against and re-emits when this
    /// moves. Unlike `dirty` (a per-frame upload gate reset every frame), this is
    /// a persistent invalidation signal.
    atlas_version: u64,
}

impl GlyphAtlas {
    /// Initial atlas size in pixels (width = height).
    const INITIAL_SIZE: u32 = 512;

    /// Create a new atlas with the built-in default font pre-loaded.
    pub fn new(device: &crate::gpu::Device) -> Self {
        let default_font =
            fontdue::Font::from_bytes(DEFAULT_FONT_BYTES, fontdue::FontSettings::default())
                .expect("built-in default font must parse");

        let size = Self::INITIAL_SIZE;
        let pixel_count = (size * size) as usize;
        let pixels = vec![[255, 255, 255, 0]; pixel_count];

        let (texture, view) = Self::create_texture(device, size);

        Self {
            fonts: vec![default_font],
            font_bytes: vec![DEFAULT_FONT_BYTES.to_vec()],
            font_has_color: vec![super::color_glyph::has_color_bitmaps(DEFAULT_FONT_BYTES)],
            entries: HashMap::new(),
            pixels,
            size,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            texture,
            view,
            dirty: false,
            atlas_version: 0,
        }
    }

    /// The atlas growth generation. Bumped each time the atlas grows (which
    /// changes every glyph's UV). Retained glyph geometry re-emits when this moves.
    pub fn version(&self) -> u64 {
        self.atlas_version
    }

    /// Register a user-supplied TTF font.  Returns a [`FontHandle`] that can be
    /// passed to overlay items.
    pub fn upload_font(&mut self, ttf_bytes: &[u8]) -> Result<FontHandle, FontError> {
        let font = fontdue::Font::from_bytes(ttf_bytes, fontdue::FontSettings::default())
            .map_err(|e| FontError::ParseFailed(e.to_string()))?;
        let index = self.fonts.len();
        self.fonts.push(font);
        self.font_has_color
            .push(super::color_glyph::has_color_bitmaps(ttf_bytes));
        self.font_bytes.push(ttf_bytes.to_vec());
        Ok(FontHandle(index))
    }

    /// The raw bytes of the font at `index` (a [`FontHandle`]'s value), if it has
    /// been uploaded. Index 0 is the built-in default font.
    pub(crate) fn font_bytes(&self, index: usize) -> Option<&[u8]> {
        self.font_bytes.get(index).map(Vec::as_slice)
    }

    /// Lay out a single-line string and return positioned glyph quads.
    ///
    /// Glyphs that are not yet in the atlas are rasterized and packed on the
    /// fly.  Call [`upload_if_dirty`] after all layout calls for the frame to
    /// push new glyphs to the GPU.
    /// Lay out a single run of text.
    ///
    /// `ppp` is the display's pixels-per-point. Glyphs are rasterised into the
    /// atlas at the physical size (`font_size * ppp`) so the bitmap matches the
    /// display resolution, and the returned quad positions, sizes, and metrics
    /// are converted back to logical points. At `ppp == 1` this is identical to
    /// laying out at `font_size` directly.
    pub fn layout_text(
        &mut self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
        ppp: f32,
        device: &crate::gpu::Device,
        style: GlyphStyle,
    ) -> TextLayout {
        let font_index = font.map_or(0, |h| h.0);
        let px = font_size * ppp;
        let size_tenths = (px * 10.0).round() as u32;

        let metrics = self.fonts[font_index].horizontal_line_metrics(px);
        let line_height = metrics
            .map(|m| m.ascent - m.descent + m.line_gap)
            .unwrap_or(px * 1.2);

        let mut quads = Vec::new();
        let mut pen_x: f32 = 0.0;
        let mut pen_y: f32 = 0.0;
        let mut max_width: f32 = 0.0;

        let mut prev_glyph: Option<u16> = None;
        for ch in text.chars() {
            if ch == '\n' {
                max_width = max_width.max(pen_x);
                pen_x = 0.0;
                pen_y += line_height;
                prev_glyph = None;
                continue;
            }

            let glyph_index = self.fonts[font_index].lookup_glyph_index(ch);

            // Kerning.
            if let Some(prev) = prev_glyph {
                if let Some(kern) =
                    self.fonts[font_index].horizontal_kern_indexed(prev, glyph_index, px)
                {
                    pen_x += kern;
                }
            }
            prev_glyph = Some(glyph_index);

            // Get metrics for advance, even for whitespace.
            let m = self.fonts[font_index].metrics_indexed(glyph_index, px);

            // Emit a quad for glyphs with a visible outline, or any glyph in a
            // color font (emoji have no outline, so fontdue reports zero area).
            if (m.width > 0 && m.height > 0) || self.font_has_color[font_index] {
                let entry =
                    self.ensure_glyph(device, font_index, glyph_index, size_tenths, px, style);
                if entry.width > 0 {
                    let atlas_size = self.size as f32;
                    quads.push(GlyphQuad {
                        pos: [pen_x + entry.offset_x, pen_y + entry.offset_y],
                        size: [entry.width as f32, entry.height as f32],
                        uv_min: [entry.x as f32 / atlas_size, entry.y as f32 / atlas_size],
                        uv_max: [
                            (entry.x + entry.width) as f32 / atlas_size,
                            (entry.y + entry.height) as f32 / atlas_size,
                        ],
                        color: entry.color,
                    });
                }
            }

            pen_x += m.advance_width;
        }
        max_width = max_width.max(pen_x);

        // Physical -> logical. UVs are untouched: they index the physical atlas
        // cell, which is what keeps the text crisp when the logical quad is
        // stretched across the physical target at NDC time.
        let inv = 1.0 / ppp;
        for q in &mut quads {
            q.pos = [q.pos[0] * inv, q.pos[1] * inv];
            q.size = [q.size[0] * inv, q.size[1] * inv];
        }

        TextLayout {
            quads,
            total_width: max_width * inv,
            height: (pen_y + line_height) * inv,
        }
    }

    /// Lay out text with word wrapping at a maximum width.
    ///
    /// Words that exceed `max_width` on their own are not broken: they extend
    /// past the boundary.  The returned `total_width` is the maximum line width
    /// actually used.
    pub fn layout_text_wrapped(
        &mut self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
        max_width: f32,
        ppp: f32,
        device: &crate::gpu::Device,
        style: GlyphStyle,
    ) -> TextLayout {
        let font_index = font.map_or(0, |h| h.0);
        // Lay out at physical size (see `layout_text`). `max_width` arrives in
        // logical points, so scale it up to match the physical pen units the
        // wrap test below works in.
        let px = font_size * ppp;
        let max_width = max_width * ppp;

        let metrics = self.fonts[font_index].horizontal_line_metrics(px);
        let line_height = metrics
            .map(|m| m.ascent - m.descent + m.line_gap)
            .unwrap_or(px * 1.2);

        let size_tenths = (px * 10.0).round() as u32;
        let space_advance = {
            let gi = self.fonts[font_index].lookup_glyph_index(' ');
            self.fonts[font_index].metrics_indexed(gi, px).advance_width
        };

        let mut quads = Vec::new();
        let mut line_x: f32 = 0.0;
        let mut line_y: f32 = 0.0;
        let mut max_line_width: f32 = 0.0;

        // Process each hard line (\n-delimited) independently, then word-wrap within it.
        for (logical_line_idx, logical_line) in text.split('\n').enumerate() {
            if logical_line_idx > 0 {
                max_line_width = max_line_width.max(line_x);
                line_x = 0.0;
                line_y += line_height;
            }

            let words: Vec<&str> = logical_line.split_whitespace().collect();
            if words.is_empty() {
                continue;
            }

            let mut first_on_line = true;

            for word in &words {
                let mut word_quads: Vec<GlyphQuad> = Vec::new();
                let mut pen_x: f32 = 0.0;
                let mut prev_glyph: Option<u16> = None;

                for ch in word.chars() {
                    let glyph_index = self.fonts[font_index].lookup_glyph_index(ch);
                    if let Some(prev) = prev_glyph {
                        if let Some(kern) =
                            self.fonts[font_index].horizontal_kern_indexed(prev, glyph_index, px)
                        {
                            pen_x += kern;
                        }
                    }
                    prev_glyph = Some(glyph_index);
                    let m = self.fonts[font_index].metrics_indexed(glyph_index, px);
                    if (m.width > 0 && m.height > 0) || self.font_has_color[font_index] {
                        let entry = self.ensure_glyph(
                            device,
                            font_index,
                            glyph_index,
                            size_tenths,
                            px,
                            style,
                        );
                        if entry.width > 0 {
                            let atlas_size = self.size as f32;
                            word_quads.push(GlyphQuad {
                                pos: [pen_x + entry.offset_x, entry.offset_y],
                                size: [entry.width as f32, entry.height as f32],
                                uv_min: [entry.x as f32 / atlas_size, entry.y as f32 / atlas_size],
                                uv_max: [
                                    (entry.x + entry.width) as f32 / atlas_size,
                                    (entry.y + entry.height) as f32 / atlas_size,
                                ],
                                color: entry.color,
                            });
                        }
                    }
                    pen_x += m.advance_width;
                }
                let word_width = pen_x;

                // Soft-wrap if the word doesn't fit on the current line.
                let test_x = if first_on_line {
                    line_x
                } else {
                    line_x + space_advance
                };
                if !first_on_line && test_x + word_width > max_width {
                    max_line_width = max_line_width.max(line_x);
                    line_x = 0.0;
                    line_y += line_height;
                    first_on_line = true;
                }

                let start_x = if first_on_line {
                    line_x
                } else {
                    line_x + space_advance
                };
                for mut gq in word_quads {
                    gq.pos[0] += start_x;
                    gq.pos[1] += line_y;
                    quads.push(gq);
                }
                line_x = start_x + word_width;
                first_on_line = false;
            }
        }

        max_line_width = max_line_width.max(line_x);
        let total_height = if quads.is_empty() && text.is_empty() {
            line_height
        } else {
            line_y + line_height
        };

        // Physical -> logical, matching `layout_text`.
        let inv = 1.0 / ppp;
        for q in &mut quads {
            q.pos = [q.pos[0] * inv, q.pos[1] * inv];
            q.size = [q.size[0] * inv, q.size[1] * inv];
        }

        TextLayout {
            quads,
            total_width: max_line_width * inv,
            height: total_height * inv,
        }
    }

    /// Lay out a run of pre-positioned glyphs and return positioned quads.
    ///
    /// Unlike [`layout_text`], the caller supplies each glyph's id and pen
    /// position, so no shaping, kerning, or pen advance happens here: this only
    /// rasterizes each glyph and places its bitmap quad. It is the back-end for
    /// [`GlyphRunItem`], where a shaping engine upstream has already produced the
    /// glyph ids and positions.
    ///
    /// `glyphs` yields `(glyph_id, x, y, payload)` where `x` and `y` are the pen
    /// position in logical pixels relative to the run origin, and `payload` is any
    /// per-glyph value the caller wants paired with the emitted quad (a tint
    /// colour, for instance). Glyphs are rasterized at the physical size
    /// (`font_size * ppp`) like [`layout_text`], and the returned quad positions
    /// and sizes are converted back to logical pixels, so the run stays crisp on
    /// HiDPI. Zero-area glyphs (whitespace and the like) are skipped, matching
    /// [`layout_text`]; the payload is threaded through the skip so it stays
    /// aligned with the quad it belongs to.
    ///
    /// [`layout_text`]: Self::layout_text
    /// [`GlyphRunItem`]: crate::renderer::types::GlyphRunItem
    pub fn layout_glyph_run<I, P>(
        &mut self,
        glyphs: I,
        font_size: f32,
        font: Option<FontHandle>,
        ppp: f32,
        device: &crate::gpu::Device,
        style: GlyphStyle,
    ) -> Vec<(GlyphQuad, P)>
    where
        I: IntoIterator<Item = (u16, f32, f32, P)>,
    {
        let font_index = font.map_or(0, |h| h.0);
        let px = font_size * ppp;
        let size_tenths = (px * 10.0).round() as u32;
        let inv = 1.0 / ppp;

        let mut quads = Vec::new();
        for (glyph_id, x, y, payload) in glyphs {
            // Skip glyphs with no visible bitmap (whitespace), as `layout_text`
            // does, so zero-area entries never reach the packer. Color-font glyphs
            // go through even when fontdue reports zero area (emoji have no
            // outline).
            let m = self.fonts[font_index].metrics_indexed(glyph_id, px);
            if (m.width == 0 || m.height == 0) && !self.font_has_color[font_index] {
                continue;
            }

            let entry = self.ensure_glyph(device, font_index, glyph_id, size_tenths, px, style);
            if entry.width == 0 {
                continue;
            }
            let atlas_size = self.size as f32;

            // Pen position arrives in logical pixels; the glyph's bitmap bearing
            // and size come from the physical rasterization, so scale those by
            // `inv` to land in the same logical space.
            let quad = GlyphQuad {
                pos: [x + entry.offset_x * inv, y + entry.offset_y * inv],
                size: [entry.width as f32 * inv, entry.height as f32 * inv],
                uv_min: [entry.x as f32 / atlas_size, entry.y as f32 / atlas_size],
                uv_max: [
                    (entry.x + entry.width) as f32 / atlas_size,
                    (entry.y + entry.height) as f32 / atlas_size,
                ],
                color: entry.color,
            };
            quads.push((quad, payload));
        }
        quads
    }

    /// Return the font ascent in pixels for the given font index and size.
    ///
    /// The ascent is the distance from the baseline to the top of the tallest
    /// glyph.  Used to position glyph quads relative to a text origin at the
    /// top-left corner of the bounding box.
    pub fn font_ascent(&self, font_index: usize, font_size: f32) -> f32 {
        self.fonts[font_index]
            .horizontal_line_metrics(font_size)
            .map(|m| m.ascent)
            .unwrap_or(font_size * 0.8)
    }

    /// Measure a text run without rasterizing or uploading any glyphs.
    ///
    /// Returns the same `width` and `height` that [`layout_text`] would produce
    /// for the same `text`, `font_size`, and `font`, plus the ascent. This is a
    /// pure read of the font metrics: no atlas mutation and no `device`, so it
    /// can be called with only `&self`.
    ///
    /// `ppp` does not appear here because it cancels out: `layout_text` lays out
    /// at `font_size * ppp` and scales the result back by `1 / ppp`, and font
    /// advances and line metrics scale linearly with size, so the logical width
    /// and height are independent of `ppp`. This measures at `font_size`
    /// directly, matching the drawn result.
    ///
    /// [`layout_text`]: Self::layout_text
    pub fn measure_text(
        &self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
    ) -> TextMetrics {
        let font_index = font.map_or(0, |h| h.0);
        let fd = &self.fonts[font_index];

        let line_height = fd
            .horizontal_line_metrics(font_size)
            .map(|m| m.ascent - m.descent + m.line_gap)
            .unwrap_or(font_size * 1.2);

        // Mirror the advance/kern accumulation in `layout_text`, skipping only
        // the glyph rasterization (which is all that path needs a device for).
        let mut pen_x: f32 = 0.0;
        let mut pen_y: f32 = 0.0;
        let mut max_width: f32 = 0.0;
        let mut prev_glyph: Option<u16> = None;
        for ch in text.chars() {
            if ch == '\n' {
                max_width = max_width.max(pen_x);
                pen_x = 0.0;
                pen_y += line_height;
                prev_glyph = None;
                continue;
            }

            let glyph_index = fd.lookup_glyph_index(ch);
            if let Some(prev) = prev_glyph {
                if let Some(kern) = fd.horizontal_kern_indexed(prev, glyph_index, font_size) {
                    pen_x += kern;
                }
            }
            prev_glyph = Some(glyph_index);
            pen_x += fd.metrics_indexed(glyph_index, font_size).advance_width;
        }
        max_width = max_width.max(pen_x);

        TextMetrics {
            width: max_width,
            height: pen_y + line_height,
            ascent: self.font_ascent(font_index, font_size),
        }
    }

    /// Upload new glyph data to the GPU if any glyphs were rasterized since
    /// the last upload.
    pub fn upload_if_dirty(&mut self, queue: &crate::gpu::Queue) {
        if !self.dirty {
            return;
        }
        let flat: Vec<u8> = self.pixels.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            &flat,
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size * 4),
                rows_per_image: Some(self.size),
            },
            crate::gpu::Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: 1,
            },
        );
        self.dirty = false;
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Ensure a glyph is in the atlas, rasterizing and packing it if needed.
    /// Returns the atlas entry.
    fn ensure_glyph(
        &mut self,
        device: &crate::gpu::Device,
        font_index: usize,
        glyph_index: u16,
        size_tenths: u32,
        px: f32,
        style: GlyphStyle,
    ) -> GlyphEntry {
        let key = GlyphKey {
            font_index,
            glyph_index,
            size_tenths,
            style,
        };

        if let Some(&entry) = self.entries.get(&key) {
            return entry;
        }

        // Color bitmap glyphs (emoji) first, for fonts that carry a strike table.
        // fontdue would rasterize these to nothing, so this is the only path that
        // draws them.
        if self.font_has_color[font_index] {
            if let Some(color) =
                super::color_glyph::rasterize(&self.font_bytes[font_index], glyph_index, px)
            {
                if !style.is_plain() {
                    // A shadow cast by an emoji is its silhouette, not a second
                    // copy of the emoji. The decoded RGBA is straight (not
                    // premultiplied), so its alpha is exactly that silhouette.
                    let coverage: Vec<u8> = color.rgba.iter().map(|p| p[3]).collect();
                    let (cell, w, h, pad) =
                        style_coverage(&coverage, color.width, color.height, style);
                    return self.pack_rgba(
                        device,
                        key,
                        &cell,
                        w,
                        h,
                        color.offset_x - pad as f32,
                        color.offset_y - pad as f32,
                        false,
                    );
                }
                return self.pack_rgba(
                    device,
                    key,
                    &color.rgba,
                    color.width,
                    color.height,
                    color.offset_x,
                    color.offset_y,
                    true,
                );
            }
        }

        // Coverage rasterization (fontdue): store `[255, 255, 255, coverage]`.
        let (metrics, bitmap) = self.fonts[font_index].rasterize_indexed(glyph_index, px);
        let w = metrics.width as u32;
        let h = metrics.height as u32;
        let offset_x = metrics.xmin as f32;
        let offset_y = -(metrics.ymin as f32 + h as f32);

        if !style.is_plain() && w > 0 && h > 0 {
            let (cell, sw, sh, pad) = style_coverage(&bitmap, w, h, style);
            return self.pack_rgba(
                device,
                key,
                &cell,
                sw,
                sh,
                offset_x - pad as f32,
                offset_y - pad as f32,
                false,
            );
        }

        if w == 0 || h == 0 {
            // Whitespace glyph: insert a zero-area entry.
            let entry = GlyphEntry {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                offset_x,
                offset_y,
                color: false,
            };
            self.entries.insert(key, entry);
            return entry;
        }

        let cell: Vec<[u8; 4]> = bitmap.iter().map(|&a| [255, 255, 255, a]).collect();
        self.pack_rgba(device, key, &cell, w, h, offset_x, offset_y, false)
    }

    /// Pack a `w * h` RGBA cell into the atlas, growing if needed, and record the
    /// entry. Shared by the coverage and color glyph paths.
    #[allow(clippy::too_many_arguments)]
    fn pack_rgba(
        &mut self,
        device: &crate::gpu::Device,
        key: GlyphKey,
        cell: &[[u8; 4]],
        w: u32,
        h: u32,
        offset_x: f32,
        offset_y: f32,
        color: bool,
    ) -> GlyphEntry {
        // Simple row packer with 1px padding.
        let pad = 1;
        if self.cursor_x + w + pad > self.size {
            self.cursor_y += self.row_height + pad;
            self.cursor_x = 0;
            self.row_height = 0;
        }
        if self.cursor_y + h + pad > self.size {
            self.grow(device);
        }

        let x = self.cursor_x;
        let y = self.cursor_y;

        for row in 0..h {
            for col in 0..w {
                let src = (row * w + col) as usize;
                let dst = ((y + row) * self.size + (x + col)) as usize;
                self.pixels[dst] = cell[src];
            }
        }
        self.dirty = true;

        self.cursor_x = x + w + pad;
        self.row_height = self.row_height.max(h);

        let entry = GlyphEntry {
            x,
            y,
            width: w,
            height: h,
            offset_x,
            offset_y,
            color,
        };
        self.entries.insert(key, entry);
        entry
    }

    /// Double the atlas size, copying existing pixel data into the new buffer
    /// and recreating the GPU texture.
    fn grow(&mut self, device: &crate::gpu::Device) {
        let old_size = self.size;
        let new_size = old_size * 2;
        tracing::info!(
            "Growing glyph atlas from {}x{} to {}x{}",
            old_size,
            old_size,
            new_size,
            new_size
        );

        let mut new_pixels = vec![[255, 255, 255, 0u8]; (new_size * new_size) as usize];
        for row in 0..old_size {
            let src_start = (row * old_size) as usize;
            let dst_start = (row * new_size) as usize;
            new_pixels[dst_start..dst_start + old_size as usize]
                .copy_from_slice(&self.pixels[src_start..src_start + old_size as usize]);
        }

        self.pixels = new_pixels;
        self.size = new_size;

        let (texture, view) = Self::create_texture(device, new_size);
        self.texture = texture;
        self.view = view;
        self.dirty = true; // Full re-upload needed.
        // Every existing glyph's UV divisor (the atlas size) just changed, so any
        // cached glyph geometry baked against the old size is now stale.
        self.atlas_version += 1;
    }

    fn create_texture(
        device: &crate::gpu::Device,
        size: u32,
    ) -> (crate::gpu::Texture, crate::gpu::TextureView) {
        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("glyph_atlas"),
            size: crate::gpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            // sRGB so colour glyph bytes (sRGB-encoded PNG) sample as linear for
            // the shader, matching the linear-float tint contract. sRGB decodes RGB
            // only, not alpha, so the coverage path (which reads `.a` and supplies
            // its own tint) is unaffected.
            format: crate::gpu::TextureFormat::Rgba8UnormSrgb,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

// ---------------------------------------------------------------------------
// FontError
// ---------------------------------------------------------------------------

/// Error returned by [`super::DeviceResources::upload_font`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum FontError {
    /// The TTF data could not be parsed.
    #[error("font parsing failed: {0}")]
    ParseFailed(String),
}

// ---------------------------------------------------------------------------
// DeviceResources integration
// ---------------------------------------------------------------------------

impl crate::resources::DeviceResources {
    /// Upload a user-supplied TTF font for use with overlay items.
    ///
    /// Returns an opaque [`FontHandle`] that can be passed to
    /// [`LabelItem`](crate::LabelItem) or [`GlyphRunItem`](crate::GlyphRunItem)
    /// via their `font` field.  Pass `None` on those items to use the built-in
    /// default font instead.
    ///
    /// The font bytes must be a valid TrueType (`.ttf`) file.
    pub fn upload_font(&mut self, ttf_bytes: &[u8]) -> Result<FontHandle, FontError> {
        self.content.glyph_atlas.upload_font(ttf_bytes)
    }

    /// Measure a text run as it would be laid out for a [`LabelItem`], returning
    /// its [`TextMetrics`] in logical pixels.
    ///
    /// Pass the same `text`, `font_size`, and `font` you would give the
    /// `LabelItem`; `font` is `None` for the built-in default font. The result
    /// matches the drawn width and height, so it can size and align overlay UI
    /// (panel widths, right-aligned columns, per-row hit rectangles) without
    /// re-parsing the font. Embedded `\n` is measured as multiple lines.
    ///
    /// This only reads font metrics: no glyphs are rasterized or uploaded, so it
    /// takes `&self` and needs no `device`.
    ///
    /// [`LabelItem`]: crate::renderer::types::LabelItem
    pub fn measure_overlay_text(
        &self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
    ) -> TextMetrics {
        self.content.glyph_atlas.measure_text(text, font_size, font)
    }

    /// The raw bytes of the font `font` refers to (`None` = the built-in default),
    /// if uploaded. A downstream text shaper can register these exact bytes so its
    /// glyph ids match what the overlay atlas rasterizes them to.
    pub fn font_bytes(&self, font: Option<FontHandle>) -> Option<&[u8]> {
        self.content.glyph_atlas.font_bytes(font.map_or(0, |h| h.0))
    }
}

// ---------------------------------------------------------------------------
// Shadow cell baking
// ---------------------------------------------------------------------------

/// Turn a glyph coverage bitmap into a shadow cell: dilate by the style's
/// spread, fade over its blur, then shape that fade by its falloff.
///
/// Returns the new cell, its dimensions, and the padding added on each side
/// (the caller shifts the glyph's bearing by this so the cell stays registered
/// with the glyph it backs).
///
/// Dilation is a max over a disc, which keeps round letterforms round; a square
/// window would square off the contour on curves. The fade is two box passes,
/// whose triangle kernel is a closer match to the smoothstep the SDF shape path
/// uses than a single box would be.
fn style_coverage(
    coverage: &[u8],
    w: u32,
    h: u32,
    style: GlyphStyle,
) -> (Vec<[u8; 4]>, u32, u32, u32) {
    let pad = style.pad();
    let ow = w + pad * 2;
    let oh = h + pad * 2;

    // Place the source in the padded cell.
    let mut a = vec![0u8; (ow * oh) as usize];
    for y in 0..h {
        for x in 0..w {
            a[((y + pad) * ow + (x + pad)) as usize] = coverage[(y * w + x) as usize];
        }
    }

    let spread = style.spread();
    if spread > 0.0 {
        a = dilate_disc(&a, ow, oh, spread);
    }

    let blur = style.blur();
    if blur > 0.0 {
        // Two box passes of radius blur/4 give a ramp about `blur` wide.
        let r = (blur * 0.25).round().max(1.0) as u32;
        a = box_blur(&a, ow, oh, r);
        a = box_blur(&a, ow, oh, r);
    }

    let falloff = style.falloff();
    if (falloff - 1.0).abs() > f32::EPSILON {
        for v in a.iter_mut() {
            let t = (*v as f32 / 255.0).powf(falloff);
            *v = (t * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    let cell: Vec<[u8; 4]> = a.iter().map(|&v| [255, 255, 255, v]).collect();
    (cell, ow, oh, pad)
}

/// Grow coverage by taking the maximum over a disc of `radius` pixels.
fn dilate_disc(src: &[u8], w: u32, h: u32, radius: f32) -> Vec<u8> {
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
    // Offsets inside the disc, computed once rather than per pixel.
    let mut disc: Vec<(i32, i32)> = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as f32 <= r2 {
                disc.push((dx, dy));
            }
        }
    }

    let mut out = vec![0u8; src.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut m = 0u8;
            for &(dx, dy) in &disc {
                let (sx, sy) = (x + dx, y + dy);
                if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                    continue;
                }
                let v = src[(sy * w as i32 + sx) as usize];
                if v > m {
                    m = v;
                    if m == 255 {
                        break;
                    }
                }
            }
            out[(y * w as i32 + x) as usize] = m;
        }
    }
    out
}

/// Separable box blur of the given radius, run horizontally then vertically.
fn box_blur(src: &[u8], w: u32, h: u32, radius: u32) -> Vec<u8> {
    let r = radius as i32;
    let n = (r * 2 + 1) as u32;
    let mut tmp = vec![0u8; src.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = 0u32;
            for d in -r..=r {
                let sx = (x + d).clamp(0, w as i32 - 1);
                sum += src[(y * w as i32 + sx) as usize] as u32;
            }
            tmp[(y * w as i32 + x) as usize] = (sum / n) as u8;
        }
    }
    let mut out = vec![0u8; src.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = 0u32;
            for d in -r..=r {
                let sy = (y + d).clamp(0, h as i32 - 1);
                sum += tmp[(sy * w as i32 + x) as usize] as u32;
            }
            out[(y * w as i32 + x) as usize] = (sum / n) as u8;
        }
    }
    out
}
