//! Font atlas and single-line text layout for overlay rendering.
//!
//! This module is the text back-end for [`LabelItem`](crate::LabelItem) and
//! [`GlyphRunItem`](crate::GlyphRunItem).  It shapes, measures and rasterizes
//! with [`swash`], packing glyphs into a single GPU texture atlas on demand.
//!
//! [`LabelItem`](crate::LabelItem) shapes one run per line (or per word when
//! wrapping), which gives it kerning, ligatures and mark attachment for a single
//! font and direction. Bidi, script itemisation and font fallback are not done
//! here: a caller that needs them runs its own shaper and submits the positioned
//! glyphs through [`GlyphRunItem`](crate::GlyphRunItem).
//!
//! Public surface: [`FontHandle`] (opaque font identifier) and
//! [`super::DeviceResources::upload_font`].  Everything else is `pub(crate)`.

use std::collections::HashMap;

use swash::scale::image::Content;
use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::shape::ShapeContext;
use swash::text::Script;
use swash::{CacheKey, FontRef};

/// Default font embedded in the library binary (Inter Regular, SIL OFL 1.1).
const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("../../fonts/Inter-Regular.ttf");

/// Whether glyph outlines are hinted before filling. Off: hinting snaps stems to
/// the pixel grid, which is sharper at small sizes but distorts the shapes a font
/// designer drew, and the call belongs with a side-by-side rather than a default.
const HINT_GLYPHS: bool = false;

/// The table-directory offset and a cache key for `font_bytes`, or `None` if swash
/// cannot read it. The key is minted once per font and reused for every scaler
/// built from it, which is what lets `ScaleContext` cache per font.
fn swash_key(font_bytes: &[u8]) -> Option<(u32, CacheKey)> {
    let font = FontRef::from_index(font_bytes, 0)?;
    Some((font.offset, font.key))
}

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
    /// Erode inward from the glyph edge instead of dilating outward, for an
    /// inset layer. The cell is the glyph's own coverage with its interior
    /// eaten away, so it draws over the glyph rather than behind it.
    inner: bool,
    /// Where the glyph sits relative to this cell once the layer offset has
    /// moved it, in whole physical pixels. An outer cell is cut back to outside
    /// the letterform, and the cut has to land where the letterform actually
    /// is, so the offset is part of what makes a cell distinct.
    clip_dx: i32,
    clip_dy: i32,
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
        inner: false,
        clip_dx: 0,
        clip_dy: 0,
    };

    /// Build a style from a shadow layer's physical-pixel spread, blur and
    /// offset. Returns [`GlyphStyle::PLAIN`] when the layer would not change
    /// the cell.
    pub(crate) fn from_shadow(
        spread_px: f32,
        blur_px: f32,
        falloff: f32,
        offset_px: [f32; 2],
    ) -> Self {
        let mut s = Self::from_layer(spread_px, blur_px, falloff, false);
        if !s.is_plain() {
            let clamp = |v: f32| v.clamp(-MAX_GLYPH_STYLE_PX, MAX_GLYPH_STYLE_PX).round() as i32;
            s.clip_dx = clamp(offset_px[0]);
            s.clip_dy = clamp(offset_px[1]);
        }
        s
    }

    /// The inset counterpart: the cell is the glyph with a band eaten inward
    /// from its edge, drawn over the glyph in the layer colour.
    pub(crate) fn from_inner_shadow(spread_px: f32, blur_px: f32, falloff: f32) -> Self {
        Self::from_layer(spread_px, blur_px, falloff, true)
    }

    fn from_layer(spread_px: f32, blur_px: f32, falloff: f32, inner: bool) -> Self {
        let spread = spread_px.clamp(0.0, MAX_GLYPH_STYLE_PX);
        let blur = blur_px.clamp(0.0, MAX_GLYPH_STYLE_PX);
        if spread <= 0.0 && blur <= 0.0 {
            return Self::PLAIN;
        }
        Self {
            spread_tenths: (spread * 10.0).round() as u32,
            blur_tenths: (blur * 10.0).round() as u32,
            falloff_tenths: ((falloff.clamp(0.05, 16.0)) * 10.0).round().max(1.0) as u32,
            inner,
            clip_dx: 0,
            clip_dy: 0,
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
        if self.inner {
            // An inset cell never grows past the glyph; one pixel of margin
            // keeps the blur off the cell border.
            return 1;
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

/// One glyph as the shaper placed it: the id it chose, its advance, and the
/// offsets positioning it against the pen.
#[derive(Clone, Copy)]
struct ShapedGlyph {
    id: u16,
    advance: f32,
    x: f32,
    y: f32,
}

// ---------------------------------------------------------------------------
// GlyphAtlas
// ---------------------------------------------------------------------------

/// A dynamically-growing glyph atlas backed by a single `Rgba8Unorm` texture.
///
/// Owned by [`DeviceResources`]; never exposed in the public API.
pub(crate) struct GlyphAtlas {
    /// Raw font bytes, parallel to `fonts`, kept so a swash `FontRef` can be
    /// rebuilt on demand: it borrows the bytes rather than owning them.
    font_bytes: Vec<Vec<u8>>,

    /// Table-directory offset and cache key per font, parallel to `fonts`. The key
    /// is minted once at upload and reused, because `ScaleContext` caches per font
    /// by it: minting a fresh one per glyph would defeat that cache.
    font_keys: Vec<(u32, CacheKey)>,

    /// swash scaler caches and scratch buffers. Holds the hinting and outline
    /// caches, so it is kept for the atlas's lifetime rather than per glyph.
    scale: ScaleContext,

    /// swash shaper caches. Behind a lock because `measure_text` takes `&self`:
    /// measuring is a read as far as a caller is concerned, and the public
    /// `DeviceResources::measure_overlay_text` must stay on `&self`. A `RefCell`
    /// would cost `DeviceResources` its `Sync`.
    shape: std::sync::Mutex<ShapeContext>,

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
        let size = Self::INITIAL_SIZE;
        let pixel_count = (size * size) as usize;
        let pixels = vec![[255, 255, 255, 0]; pixel_count];

        let (texture, view) = Self::create_texture(device, size);

        let default_key = swash_key(DEFAULT_FONT_BYTES).expect("built-in default font must parse");

        Self {
            font_bytes: vec![DEFAULT_FONT_BYTES.to_vec()],
            font_keys: vec![default_key],
            scale: ScaleContext::new(),
            shape: std::sync::Mutex::new(ShapeContext::new()),
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
        let key = swash_key(ttf_bytes)
            .ok_or_else(|| FontError::ParseFailed("not a readable font".into()))?;
        let index = self.font_bytes.len();
        self.font_keys.push(key);
        self.font_bytes.push(ttf_bytes.to_vec());
        Ok(FontHandle(index))
    }

    /// The raw bytes of the font at `index` (a [`FontHandle`]'s value), if it has
    /// been uploaded. Index 0 is the built-in default font.
    pub(crate) fn font_bytes(&self, index: usize) -> Option<&[u8]> {
        self.font_bytes.get(index).map(Vec::as_slice)
    }

    /// A swash font handle for `font_index`. Cheap: it borrows the stored bytes
    /// and reuses the cache key minted at upload.
    fn font_ref(&self, font_index: usize) -> FontRef<'_> {
        let (offset, key) = self.font_keys[font_index];
        FontRef {
            data: &self.font_bytes[font_index],
            offset,
            key,
        }
    }

    /// Ascent and line height at `px`, both in pixels.
    fn line_metrics(&self, font_index: usize, px: f32) -> (f32, f32) {
        let m = self.font_ref(font_index).metrics(&[]).scale(px);
        // swash reports descent as a positive distance below the baseline, so the
        // three sum rather than subtracting the middle one.
        (m.ascent, m.ascent + m.descent + m.leading)
    }

    /// The total advance of `text` shaped as one run at `px`.
    fn advance_of(&self, font_index: usize, text: &str, px: f32) -> f32 {
        let mut shaped = Vec::new();
        self.shape_run(font_index, text, px, &mut shaped);
        shaped.iter().map(|g| g.advance).sum()
    }

    /// Shape one run of text into positioned glyphs at `px`, appending to `out`.
    ///
    /// One font, one direction, no line breaking: the caller splits lines and
    /// words first, so this is the single-run case a label needs. A caller that
    /// wants bidi, script itemisation or font fallback runs its own shaper and
    /// submits a [`GlyphRunItem`](crate::GlyphRunItem), which does not come
    /// through here.
    fn shape_run(&self, font_index: usize, text: &str, px: f32, out: &mut Vec<ShapedGlyph>) {
        out.clear();
        if text.is_empty() {
            return;
        }
        let font = self.font_ref(font_index);
        let mut ctx = self.shape.lock().expect("shaper lock");
        let mut shaper = ctx.builder(font).script(Script::Latin).size(px).build();
        shaper.add_str(text);
        shaper.shape_with(|cluster| {
            for g in cluster.glyphs {
                out.push(ShapedGlyph {
                    id: g.id,
                    advance: g.advance,
                    x: g.x,
                    y: g.y,
                });
            }
        });
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

        let (_, line_height) = self.line_metrics(font_index, px);

        let mut quads = Vec::new();
        let mut pen_y: f32 = 0.0;
        let mut max_width: f32 = 0.0;

        // Each hard line shapes as its own run: a newline is a break, not a
        // character the shaper should see.
        let mut shaped = Vec::new();
        for (line_idx, line) in text.split('\n').enumerate() {
            if line_idx > 0 {
                pen_y += line_height;
            }
            self.shape_run(font_index, line, px, &mut shaped);

            let mut pen_x: f32 = 0.0;
            for g in std::mem::take(&mut shaped) {
                // Whether a glyph draws is `ensure_glyph`'s call: it may have an
                // outline, a colour strike, or nothing at all.
                let entry = self.ensure_glyph(device, font_index, g.id, size_tenths, px, style);
                if entry.width > 0 {
                    let atlas_size = self.size as f32;
                    quads.push(GlyphQuad {
                        pos: [pen_x + g.x + entry.offset_x, pen_y - g.y + entry.offset_y],
                        size: [entry.width as f32, entry.height as f32],
                        uv_min: [entry.x as f32 / atlas_size, entry.y as f32 / atlas_size],
                        uv_max: [
                            (entry.x + entry.width) as f32 / atlas_size,
                            (entry.y + entry.height) as f32 / atlas_size,
                        ],
                        color: entry.color,
                    });
                }
                pen_x += g.advance;
            }
            max_width = max_width.max(pen_x);
        }

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

        let (_, line_height) = self.line_metrics(font_index, px);

        let size_tenths = (px * 10.0).round() as u32;
        let space_advance = self.advance_of(font_index, " ", px);

        let mut quads = Vec::new();
        let mut shaped = Vec::new();
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

                // One word is one run. Shaping stops at the word boundary the
                // wrapper already chose, which is where a line may break anyway.
                self.shape_run(font_index, word, px, &mut shaped);
                for g in std::mem::take(&mut shaped) {
                    let entry = self.ensure_glyph(device, font_index, g.id, size_tenths, px, style);
                    if entry.width > 0 {
                        let atlas_size = self.size as f32;
                        word_quads.push(GlyphQuad {
                            pos: [pen_x + g.x + entry.offset_x, -g.y + entry.offset_y],
                            size: [entry.width as f32, entry.height as f32],
                            uv_min: [entry.x as f32 / atlas_size, entry.y as f32 / atlas_size],
                            uv_max: [
                                (entry.x + entry.width) as f32 / atlas_size,
                                (entry.y + entry.height) as f32 / atlas_size,
                            ],
                            color: entry.color,
                        });
                    }
                    pen_x += g.advance;
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
            // Glyphs with no visible bitmap (whitespace) are skipped, as in
            // `layout_text`. That is decided by `ensure_glyph` rather than by
            // fontdue's metrics here, because a glyph fontdue reports as empty may
            // still rasterize from its outline, and an emoji has no outline at all.
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
        self.line_metrics(font_index, font_size).0
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
        let (_, line_height) = self.line_metrics(font_index, font_size);

        // Mirror the line splitting and shaping in `layout_text`, skipping only
        // the glyph rasterization (which is all that path needs a device for).
        let mut pen_y: f32 = 0.0;
        let mut max_width: f32 = 0.0;
        for (line_idx, line) in text.split('\n').enumerate() {
            if line_idx > 0 {
                pen_y += line_height;
            }
            max_width = max_width.max(self.advance_of(font_index, line, font_size));
        }

        TextMetrics {
            width: max_width,
            height: pen_y + line_height,
            ascent: self.font_ascent(font_index, font_size),
        }
    }

    /// Measure a word-wrapped text run without rasterizing or uploading any
    /// glyphs.
    ///
    /// Returns the same `width` and `height` that [`layout_text_wrapped`] would
    /// produce for the same `text`, `font_size`, `font`, and `max_width`, plus
    /// the ascent. Like [`measure_text`] this is a pure read of the font
    /// metrics, so it takes `&self` and needs no `device`, and the result is
    /// independent of `pixels_per_point` for the same reason.
    ///
    /// The break rule is the one the drawn text uses: hard `\n` splits first,
    /// then words are packed onto a line while they fit, and a word wider than
    /// `max_width` on its own is not broken and overhangs instead. `width` is
    /// the widest line actually used, which can be less than `max_width`.
    ///
    /// [`layout_text_wrapped`]: Self::layout_text_wrapped
    /// [`measure_text`]: Self::measure_text
    pub fn measure_text_wrapped(
        &self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
        max_width: f32,
    ) -> TextMetrics {
        let font_index = font.map_or(0, |h| h.0);
        let (_, line_height) = self.line_metrics(font_index, font_size);
        let space_advance = self.advance_of(font_index, " ", font_size);

        // Mirror the word packing in `layout_text_wrapped`, skipping only the
        // glyph rasterization. Each word shapes on its own, matching the per-word
        // pen that path uses.
        let mut line_x: f32 = 0.0;
        let mut line_y: f32 = 0.0;
        let mut max_line_width: f32 = 0.0;

        for (logical_line_idx, logical_line) in text.split('\n').enumerate() {
            if logical_line_idx > 0 {
                max_line_width = max_line_width.max(line_x);
                line_x = 0.0;
                line_y += line_height;
            }

            let mut first_on_line = true;
            for word in logical_line.split_whitespace() {
                let word_width = self.advance_of(font_index, word, font_size);

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
                line_x = start_x + word_width;
                first_on_line = false;
            }
        }
        max_line_width = max_line_width.max(line_x);

        TextMetrics {
            width: max_line_width,
            height: line_y + line_height,
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

        // One cascade for every glyph, in priority order: layered colour outlines,
        // then a colour bitmap strike, then the outline. Everything is addressed by
        // glyph id, so an id that reached us from a table this crate does not read
        // (the OpenType `MATH` size variants, say) rasterizes like any other.
        //
        // COLR outranks a bitmap strike because it is vector: a font carrying both
        // scales cleanly from the outlines rather than resampling a fixed strike.
        // The palette is the font's default (index 0). Alternate CPAL palettes are
        // mostly light/dark emoji variants, and exposing a choice would mean a new
        // field on every text item plus a palette in `GlyphKey`, which is not
        // earned. Adding one later means widening that key: two palettes of one
        // glyph at one size are different pixels, and the cache cannot tell them
        // apart as it stands.
        let image = {
            let Self {
                scale,
                font_bytes,
                font_keys,
                ..
            } = self;
            let (offset, key) = font_keys[font_index];
            let font = FontRef {
                data: &font_bytes[font_index],
                offset,
                key,
            };
            let mut scaler = scale.builder(font).size(px).hint(HINT_GLYPHS).build();
            Render::new(&[
                Source::ColorOutline(0),
                Source::ColorBitmap(StrikeWith::BestFit),
                Source::Outline,
            ])
            .render(&mut scaler, glyph_index)
        };

        // No outline and no strike: whitespace, or an id past the end of the font.
        // Cache a zero-area entry so the miss is paid once.
        let Some(image) = image.filter(|i| i.placement.width > 0 && i.placement.height > 0) else {
            let entry = GlyphEntry {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                offset_x: 0.0,
                offset_y: 0.0,
                color: false,
            };
            self.entries.insert(key, entry);
            return entry;
        };

        let w = image.placement.width;
        let h = image.placement.height;
        let offset_x = image.placement.left as f32;
        // `placement.top` is the baseline-to-top distance measured upwards; the
        // atlas wants the pen-to-top offset measured downwards.
        let offset_y = -(image.placement.top as f32);
        let is_colour = image.content == Content::Color;

        if !style.is_plain() {
            // A shadow cast by an emoji is its silhouette, not a second copy of the
            // emoji, so a colour glyph contributes its alpha channel here.
            let coverage: Vec<u8> = if is_colour {
                image.data.chunks_exact(4).map(|p| p[3]).collect()
            } else {
                image.data
            };
            let (cell, sw, sh, pad) = style_coverage(&coverage, w, h, style);
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

        // Coverage is stored as `[255, 255, 255, alpha]`; a colour glyph's RGBA goes
        // in as it comes out, straight (non-premultiplied).
        let cell: Vec<[u8; 4]> = if is_colour {
            image
                .data
                .chunks_exact(4)
                .map(|p| [p[0], p[1], p[2], p[3]])
                .collect()
        } else {
            image.data.iter().map(|&a| [255, 255, 255, a]).collect()
        };
        self.pack_rgba(device, key, &cell, w, h, offset_x, offset_y, is_colour)
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

    /// Measure a text run as it would be laid out for a [`LabelItem`] with
    /// `max_width` set, returning its [`TextMetrics`] in logical pixels.
    ///
    /// Same as [`measure_overlay_text`], but with the word wrapping the label
    /// applies when it has a `max_width`: hard `\n` splits first, then words
    /// are packed onto a line while they fit, and a word too wide to fit on its
    /// own is not broken and overhangs. The returned `width` is the widest line
    /// actually used, so it can be narrower than `max_width`.
    ///
    /// This only reads font metrics: no glyphs are rasterized or uploaded, so it
    /// takes `&self` and needs no `device`.
    ///
    /// [`LabelItem`]: crate::renderer::types::LabelItem
    /// [`measure_overlay_text`]: Self::measure_overlay_text
    pub fn measure_overlay_text_wrapped(
        &self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
        max_width: f32,
    ) -> TextMetrics {
        self.content
            .glyph_atlas
            .measure_text_wrapped(text, font_size, font, max_width)
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

    // Kept for the cases that need the glyph's own coverage back: the inset
    // band is what an erosion ate, and an outer layer is cut back to outside
    // the letterform.
    let src = a.clone();

    let spread = style.spread();
    if spread > 0.0 {
        a = if style.inner {
            erode_disc(&a, ow, oh, spread)
        } else {
            dilate_disc(&a, ow, oh, spread)
        };
    }

    let blur = style.blur();
    if blur > 0.0 {
        // Two box passes of radius blur/4 give a ramp about `blur` wide.
        let r = (blur * 0.25).round().max(1.0) as u32;
        a = box_blur(&a, ow, oh, r);
        a = box_blur(&a, ow, oh, r);
    }

    if style.inner {
        // The band is what the erosion ate: the glyph's own coverage minus
        // what survived. Multiplying by the source keeps the outer edge as
        // clean as the glyph's, which is what an inset layer needs, since it
        // draws over the letterform rather than behind it.
        for (v, &s) in a.iter_mut().zip(src.iter()) {
            *v = 255 - *v;
            *v = ((*v as u32 * s as u32) / 255) as u8;
        }
    } else {
        // Cut the cell back to outside the letterform, the way an outer
        // box-shadow is clipped to outside the border box. The layer offset
        // moves the cell, so the letterform is sampled at that offset: the
        // hole then lands on the glyph once the cell is drawn. Each glyph is
        // cut against its own coverage, so a neighbour's shadow can still show
        // through a translucent letterform where the two overlap.
        for y in 0..oh as i32 {
            for x in 0..ow as i32 {
                let (sx, sy) = (x + style.clip_dx, y + style.clip_dy);
                if sx < 0 || sy < 0 || sx >= ow as i32 || sy >= oh as i32 {
                    continue;
                }
                let g = src[(sy * ow as i32 + sx) as usize] as u32;
                let i = (y * ow as i32 + x) as usize;
                a[i] = ((a[i] as u32 * (255 - g)) / 255) as u8;
            }
        }
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

/// Shrink coverage by taking the minimum over a disc of `radius` pixels.
/// Outside the cell counts as empty, so the glyph erodes in from its edge.
fn erode_disc(src: &[u8], w: u32, h: u32, radius: f32) -> Vec<u8> {
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
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
            let mut m = 255u8;
            for &(dx, dy) in &disc {
                let (sx, sy) = (x + dx, y + dy);
                let v = if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                    0
                } else {
                    src[(sy * w as i32 + sx) as usize]
                };
                if v < m {
                    m = v;
                    if m == 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use viewport_lib_testkit::headless_device;

    /// Build a COLR v0 / CPAL v0 font by appending those two tables to the
    /// bundled one, so the colour-outline path can be tested without shipping a
    /// COLR font or depending on one being installed. The base glyph is `A`,
    /// drawn as two layers: `A` in palette colour 0 and `O` in palette colour 1.
    fn synth_colr_font(base: u16, layer: u16) -> Vec<u8> {
        fn u16b(v: u16) -> [u8; 2] {
            v.to_be_bytes()
        }
        fn u32b(v: u32) -> [u8; 4] {
            v.to_be_bytes()
        }

        // COLR v0: two layers on one base glyph.
        let mut colr = Vec::new();
        colr.extend(u16b(0)); // version
        colr.extend(u16b(1)); // numBaseGlyphRecords
        colr.extend(u32b(14)); // baseGlyphRecordsOffset (header is 14 bytes)
        colr.extend(u32b(14 + 6)); // layerRecordsOffset
        colr.extend(u16b(2)); // numLayerRecords
        colr.extend(u16b(base)); // base glyph id
        colr.extend(u16b(0)); // firstLayerIndex
        colr.extend(u16b(2)); // numLayers
        colr.extend(u16b(base)); // layer 0 glyph
        colr.extend(u16b(0)); // layer 0 palette entry
        colr.extend(u16b(layer)); // layer 1 glyph
        colr.extend(u16b(1)); // layer 1 palette entry

        // CPAL v0: one palette, two entries. Colour records are BGRA.
        let mut cpal = Vec::new();
        cpal.extend(u16b(0)); // version
        cpal.extend(u16b(2)); // numPaletteEntries
        cpal.extend(u16b(1)); // numPalettes
        cpal.extend(u16b(2)); // numColorRecords
        cpal.extend(u32b(14)); // colorRecordsArrayOffset
        cpal.extend(u16b(0)); // colorRecordIndices[0]
        cpal.extend([0, 0, 255, 255]); // entry 0: red
        cpal.extend([255, 0, 0, 255]); // entry 1: blue

        // Rebuild the table directory with the two new tables spliced in. Tag
        // order matters: readers binary-search the directory.
        let src = DEFAULT_FONT_BYTES;
        let num = u16::from_be_bytes([src[4], src[5]]) as usize;
        let mut tables: Vec<([u8; 4], Vec<u8>)> = Vec::with_capacity(num + 2);
        for i in 0..num {
            let rec = 12 + i * 16;
            let tag = [src[rec], src[rec + 1], src[rec + 2], src[rec + 3]];
            let off = u32::from_be_bytes(src[rec + 8..rec + 12].try_into().unwrap()) as usize;
            let len = u32::from_be_bytes(src[rec + 12..rec + 16].try_into().unwrap()) as usize;
            tables.push((tag, src[off..off + len].to_vec()));
        }
        tables.push((*b"COLR", colr));
        tables.push((*b"CPAL", cpal));
        tables.sort_by_key(|(tag, _)| *tag);

        let count = tables.len();
        let mut out = Vec::new();
        out.extend(&src[0..4]); // sfnt version
        out.extend(u16b(count as u16));
        // searchRange / entrySelector / rangeShift are not read by the parsers
        // here, and a wrong value is not a parse failure.
        out.extend(u16b(0));
        out.extend(u16b(0));
        out.extend(u16b(0));

        let mut offset = 12 + count * 16;
        let mut directory = Vec::new();
        let mut body = Vec::new();
        for (tag, data) in &tables {
            directory.extend(tag);
            directory.extend(u32b(0)); // checksum: not verified by these readers
            directory.extend(u32b(offset as u32));
            directory.extend(u32b(data.len() as u32));
            body.extend(data);
            let pad = (4 - data.len() % 4) % 4;
            body.extend(std::iter::repeat_n(0u8, pad));
            offset += data.len() + pad;
        }
        out.extend(directory);
        out.extend(body);
        out
    }

    /// A COLR base glyph renders as a colour image, and its palette colours come
    /// through. The base glyph is also an ordinary outline, so this doubles as a
    /// check that `Source::ColorOutline` is reached before `Source::Outline`.
    #[test]
    fn colr_glyph_renders_with_its_palette() {
        let plain = FontRef::from_index(DEFAULT_FONT_BYTES, 0).unwrap();
        let base = plain.charmap().map('A');
        let layer = plain.charmap().map('O');
        assert!(base != 0 && layer != 0);

        let bytes = synth_colr_font(base, layer);
        let font = FontRef::from_index(&bytes, 0).expect("synthesised font parses");

        let mut ctx = ScaleContext::new();
        let mut scaler = ctx.builder(font).size(64.0).hint(HINT_GLYPHS).build();
        let image = Render::new(&[
            Source::ColorOutline(0),
            Source::ColorBitmap(StrikeWith::BestFit),
            Source::Outline,
        ])
        .render(&mut scaler, base)
        .expect("the base glyph renders");

        assert_eq!(
            image.content,
            Content::Color,
            "a COLR base glyph must take the colour-outline path, not the outline one"
        );
        assert_eq!(
            image.data.len(),
            (image.placement.width * image.placement.height * 4) as usize
        );

        // Both palette entries are present: red from layer 0, blue from layer 1.
        let mut red = 0usize;
        let mut blue = 0usize;
        for p in image.data.chunks_exact(4) {
            if p[3] < 128 {
                continue;
            }
            if p[0] > 180 && p[1] < 80 && p[2] < 80 {
                red += 1;
            }
            if p[2] > 180 && p[0] < 80 && p[1] < 80 {
                blue += 1;
            }
        }
        assert!(red > 0, "expected the first palette colour in the output");
        assert!(blue > 0, "expected the second palette colour in the output");
    }

    /// Glyph 646 of the bundled font has no codepoint: nothing in the font's
    /// `cmap` reaches it. That is the shape of an OpenType MATH size variant, and
    /// the reason the atlas rasterizes by glyph id rather than through a
    /// `cmap`-driven cache, since a shaper hands over ids of exactly this kind.
    #[test]
    fn rasterises_a_glyph_with_no_codepoint() {
        let (offset, key) = swash_key(DEFAULT_FONT_BYTES).unwrap();
        let font = FontRef {
            data: DEFAULT_FONT_BYTES,
            offset,
            key,
        };

        let mut mapped = false;
        font.charmap().enumerate(|_, id| {
            if id == 646 {
                mapped = true;
            }
        });
        assert!(
            !mapped,
            "glyph 646 is the test case because it has no codepoint"
        );

        let mut ctx = ScaleContext::new();
        let mut scaler = ctx.builder(font).size(48.0).hint(HINT_GLYPHS).build();
        let image = Render::new(&[
            Source::ColorOutline(0),
            Source::ColorBitmap(StrikeWith::BestFit),
            Source::Outline,
        ])
        .render(&mut scaler, 646)
        .expect("the font outlines glyph 646");

        assert!(image.placement.width > 0 && image.placement.height > 0);
        assert_eq!(image.content, Content::Mask);
        assert_eq!(
            image.data.len(),
            (image.placement.width * image.placement.height) as usize
        );
        assert!(
            image.data.iter().any(|&a| a > 128),
            "expected solid coverage, not a sliver"
        );
        // Sits above the baseline, so the atlas offset is negative.
        assert!(-(image.placement.top as f32) < 0.0);
    }

    /// The strings and widths the wrap tests share: a run that breaks in several
    /// places, one that fits on a line untouched, one with a word wider than the
    /// limit, and one with a hard newline on top of the wrapping.
    const CASES: &[(&str, f32)] = &[
        ("the quick brown fox jumps over the lazy dog", 120.0),
        ("the quick brown fox jumps over the lazy dog", 400.0),
        ("short", 200.0),
        ("antidisestablishmentarianism is long", 60.0),
        ("first line here\nsecond line wraps around", 90.0),
        ("", 100.0),
    ];

    /// `measure_text_wrapped` reports what `layout_text_wrapped` draws, for the
    /// same text and limit. This is the property the measurement exists for: a
    /// consumer sizing a backing gets the renderer's box, not an approximation.
    #[test]
    fn wrapped_measure_matches_wrapped_layout() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);

        for &(text, max_width) in CASES {
            for font_size in [12.0_f32, 28.0] {
                let laid = atlas.layout_text_wrapped(
                    text,
                    font_size,
                    None,
                    max_width,
                    1.0,
                    &device,
                    GlyphStyle::PLAIN,
                );
                let measured = atlas.measure_text_wrapped(text, font_size, None, max_width);
                assert!(
                    (measured.width - laid.total_width).abs() < 0.01,
                    "width {} vs {} for {text:?} at {font_size} wrapped to {max_width}",
                    measured.width,
                    laid.total_width
                );
                assert!(
                    (measured.height - laid.height).abs() < 0.01,
                    "height {} vs {} for {text:?} at {font_size} wrapped to {max_width}",
                    measured.height,
                    laid.height
                );
            }
        }
    }

    /// Measurement is in logical pixels, so it does not move with
    /// `pixels_per_point`: the layout scales up to physical and back down, and
    /// the measure never leaves logical units. Same note as `measure_text`.
    #[test]
    fn wrapped_measure_is_independent_of_pixels_per_point() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);

        for &(text, max_width) in CASES {
            let measured = atlas.measure_text_wrapped(text, 16.0, None, max_width);
            for ppp in [1.0_f32, 1.5, 2.0] {
                let laid = atlas.layout_text_wrapped(
                    text,
                    16.0,
                    None,
                    max_width,
                    ppp,
                    &device,
                    GlyphStyle::PLAIN,
                );
                assert!(
                    (measured.width - laid.total_width).abs() < 0.05
                        && (measured.height - laid.height).abs() < 0.05,
                    "ppp {ppp} moved the box for {text:?}: measured {measured:?}, laid out {} x {}",
                    laid.total_width,
                    laid.height
                );
            }
        }
    }

    /// An unwrapped measure and a wrapped one with a limit nothing reaches agree
    /// on a single-line run, so a consumer can hold one code path for both.
    #[test]
    fn wrapped_measure_matches_plain_measure_when_nothing_breaks() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let atlas = GlyphAtlas::new(&device);

        let plain = atlas.measure_text("short run", 18.0, None);
        let wrapped = atlas.measure_text_wrapped("short run", 18.0, None, 10_000.0);
        assert!((plain.width - wrapped.width).abs() < 0.01);
        assert!((plain.height - wrapped.height).abs() < 0.01);
        assert_eq!(plain.ascent, wrapped.ascent);
    }
}

// The atlas shapes behind a `Mutex` rather than a `RefCell` so that
// `DeviceResources` stays `Sync` for a consumer holding it across threads.
// Swapping in a cell would compile here and break them, so it is asserted.
const _: () = {
    fn assert_sync<T: Sync>() {}
    fn probe() {
        assert_sync::<GlyphAtlas>();
        assert_sync::<crate::resources::DeviceResources>();
    }
    let _ = probe;
};
