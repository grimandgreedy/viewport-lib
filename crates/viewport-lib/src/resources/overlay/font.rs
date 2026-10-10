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
//! Public surface: [`FontHandle`] (opaque font identifier),
//! [`super::DeviceResources::upload_font`] and
//! [`super::DeviceResources::upload_font_face`].  Everything else is `pub(crate)`.

use crate::resources::builders::LoggedAlloc;
use std::collections::HashMap;
use std::sync::Arc;

use swash::scale::image::Content;
use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::shape::ShapeContext;
use swash::text::Script;
use swash::{CacheKey, FontRef};

/// Default font embedded in the library binary: Roboto Regular (Apache 2.0),
/// cut down to the characters listed in `fonts/README.md`. Text outside that
/// set needs a font passed to `upload_font`.
const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("../../fonts/Roboto-Regular.ttf");

/// Whether glyph outlines are hinted before filling. Off: hinting snaps stems to
/// the pixel grid, which is sharper at small sizes but distorts the shapes a font
/// designer drew, and the call belongs with a side-by-side rather than a default.
const HINT_GLYPHS: bool = false;

/// The table-directory offset and a cache key for face `face` of `font_bytes`.
/// The key is minted once per font and reused for every scaler built from it,
/// which is what lets `ScaleContext` cache per font.
fn swash_key(font_bytes: &[u8], face: u32) -> Result<(u32, CacheKey), FontError> {
    let data = swash::FontDataRef::new(font_bytes)
        .ok_or_else(|| FontError::ParseFailed("not a readable font".into()))?;
    let count = data.len() as u32;
    if face >= count {
        return Err(FontError::FaceOutOfRange { face, count });
    }
    let font = data
        .get(face as usize)
        .ok_or_else(|| FontError::ParseFailed(format!("face {face} is not readable")))?;
    Ok((font.offset, font.key))
}

/// The bytes of a font file. The built-in font stays borrowed from the binary,
/// so building the atlas copies nothing; uploaded fonts are shared, so two faces
/// of one collection, or a font a caller already holds, are not copied again.
enum FontData {
    Static(&'static [u8]),
    Shared(Arc<[u8]>),
}

impl FontData {
    fn bytes(&self) -> &[u8] {
        match self {
            FontData::Static(b) => b,
            FontData::Shared(b) => b,
        }
    }

    /// The bytes as a shared buffer. Copies the built-in font, which is only
    /// asked for when a caller reads it back.
    fn shared(&self) -> Arc<[u8]> {
        match self {
            FontData::Static(b) => Arc::from(*b),
            FontData::Shared(b) => b.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// FontHandle : public opaque identifier
// ---------------------------------------------------------------------------

// The handle a consumer names on an overlay item lives in `viewport-lib-types`;
// the font store and rasterization below stay here. Re-exported so the existing
// `crate::resources::overlay::font::FontHandle` path keeps resolving.
pub use viewport_lib_types::overlay::font::FontHandle;

use super::glyph_style::{GlyphStyle, style_coverage};

// ---------------------------------------------------------------------------
// GlyphKey / GlyphEntry : atlas bookkeeping
// ---------------------------------------------------------------------------

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
    /// Font file bytes, one per `FontHandle`, kept so a swash `FontRef` can be
    /// rebuilt on demand: it borrows the bytes rather than owning them.
    font_bytes: Vec<FontData>,

    /// Table-directory offset, cache key and face index per font, parallel to
    /// `font_bytes`. The offset picks the face within a collection. The key is
    /// minted once at upload and reused, because `ScaleContext` caches per font
    /// by it: minting a fresh one per glyph would defeat that cache.
    font_keys: Vec<(u32, CacheKey, u32)>,

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

    /// Current atlas dimensions (always square, power of two). Zero until the
    /// first glyph is packed: an application that draws no text never pays
    /// for the pixel buffer or the texture.
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

    /// Create a new atlas with the built-in default font pre-loaded. The
    /// texture is a 1x1 placeholder until the first glyph is packed.
    pub fn new(device: &crate::gpu::Device) -> Self {
        let size = 0;
        let pixels = Vec::new();

        let (texture, view) = Self::create_texture(device, 1);

        let (offset, key) =
            swash_key(DEFAULT_FONT_BYTES, 0).expect("built-in default font must parse");

        Self {
            font_bytes: vec![FontData::Static(DEFAULT_FONT_BYTES)],
            font_keys: vec![(offset, key, 0)],
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

    /// Register face `face` of a font file. Returns a [`FontHandle`] that can be
    /// passed to overlay items.
    pub fn upload_font_face(
        &mut self,
        data: Arc<[u8]>,
        face: u32,
    ) -> Result<FontHandle, FontError> {
        let (offset, key) = swash_key(&data, face)?;
        let index = self.font_bytes.len();
        self.font_keys.push((offset, key, face));
        self.font_bytes.push(FontData::Shared(data));
        Ok(FontHandle(index))
    }

    /// The font file and face index at `index` (a [`FontHandle`]'s value), if it
    /// has been uploaded. Index 0 is the built-in default font.
    pub(crate) fn font_face(&self, index: usize) -> Option<(Arc<[u8]>, u32)> {
        let data = self.font_bytes.get(index)?;
        Some((data.shared(), self.font_keys[index].2))
    }

    /// The first character of `text` that `font` has no glyph for, skipping
    /// whitespace and control characters. A handle that was never uploaded
    /// covers nothing.
    pub fn missing_glyph(&self, text: &str, font: Option<FontHandle>) -> Option<char> {
        let font_index = font.map_or(0, |h| h.0);
        let mut drawn = text
            .chars()
            .filter(|c| !c.is_whitespace() && !c.is_control());
        if font_index >= self.font_keys.len() {
            return drawn.next();
        }
        let charmap = self.font_ref(font_index).charmap();
        drawn.find(|&c| charmap.map(c) == 0)
    }

    /// A swash font handle for `font_index`. Cheap: it borrows the stored bytes
    /// and reuses the cache key minted at upload.
    fn font_ref(&self, font_index: usize) -> FontRef<'_> {
        let (offset, key, _) = self.font_keys[font_index];
        FontRef {
            data: self.font_bytes[font_index].bytes(),
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

    /// Shape one word and rasterize its glyphs, returning the quads positioned
    /// against a pen at the origin plus the word's total advance. The caller
    /// translates them once the wrap has decided where the word sits.
    fn word_quads(
        &mut self,
        device: &crate::gpu::Device,
        font_index: usize,
        word: &str,
        px: f32,
        size_tenths: u32,
        style: GlyphStyle,
    ) -> (Vec<GlyphQuad>, f32) {
        // One word is one run. Shaping stops at the word boundary the wrapper
        // already chose, which is where a line may break anyway.
        let mut shaped = Vec::new();
        self.shape_run(font_index, word, px, &mut shaped);

        let mut quads = Vec::new();
        let mut pen_x: f32 = 0.0;
        for g in shaped {
            let entry = self.ensure_glyph(device, font_index, g.id, size_tenths, px, style);
            if entry.width > 0 {
                let atlas_size = self.size as f32;
                quads.push(GlyphQuad {
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
        (quads, pen_x)
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

        // Shape and rasterize each word at its own origin, then let the shared
        // rule decide where the words land. Quads are built first because a word's
        // width is not known until it is shaped, and the wrap depends on it.
        let mut word_quads: Vec<Vec<GlyphQuad>> = Vec::new();
        let mut lines: Vec<Vec<f32>> = Vec::new();
        for logical_line in text.split('\n') {
            let mut widths = Vec::new();
            for word in logical_line.split_whitespace() {
                let (quads, width) =
                    self.word_quads(device, font_index, word, px, size_tenths, style);
                word_quads.push(quads);
                widths.push(width);
            }
            lines.push(widths);
        }

        let (origins, max_line_width, total_height) =
            wrap_words(&lines, space_advance, max_width, line_height);

        let mut quads = Vec::new();
        for (word, origin) in word_quads.into_iter().zip(origins) {
            for mut q in word {
                q.pos[0] += origin[0];
                q.pos[1] += origin[1];
                quads.push(q);
            }
        }

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
            // `layout_text`. `ensure_glyph` decides that: a glyph may draw from an
            // outline, from a colour strike, or not at all.
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

        // The same rule `layout_text_wrapped` packs with, over the same per-word
        // widths, so the two cannot disagree about where a line breaks.
        let lines: Vec<Vec<f32>> = text
            .split('\n')
            .map(|line| {
                line.split_whitespace()
                    .map(|word| self.advance_of(font_index, word, font_size))
                    .collect()
            })
            .collect();

        let (_, max_line_width, total_height) =
            wrap_words(&lines, space_advance, max_width, line_height);

        TextMetrics {
            width: max_line_width,
            height: total_height,
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
            let (offset, key, _) = font_keys[font_index];
            let font = FontRef {
                data: font_bytes[font_index].bytes(),
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
        if self.size == 0 {
            self.grow(device);
        }
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

    /// Double the atlas size (or allocate it at its initial size), copying
    /// existing pixel data into the new buffer and recreating the GPU texture.
    fn grow(&mut self, device: &crate::gpu::Device) {
        let old_size = self.size;
        let new_size = if old_size == 0 {
            Self::INITIAL_SIZE
        } else {
            old_size * 2
        };
        if old_size > 0 {
            tracing::info!(
                "Growing glyph atlas from {}x{} to {}x{}",
                old_size,
                old_size,
                new_size,
                new_size
            );
        }

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
        let texture = device.logged_texture(&crate::gpu::TextureDescriptor {
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

/// Error returned by [`super::DeviceResources::upload_font`] and
/// [`super::DeviceResources::upload_font_face`].
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum FontError {
    /// The font data could not be parsed.
    #[error("font parsing failed: {0}")]
    ParseFailed(String),
    /// The requested face is past the end of the file: `count` is how many
    /// faces it holds (1 for a single font, more for a collection).
    #[error("font face {face} requested, but the file has {count}")]
    FaceOutOfRange {
        /// The face index asked for.
        face: u32,
        /// How many faces the file holds.
        count: u32,
    },
}

// ---------------------------------------------------------------------------
// DeviceResources integration
// ---------------------------------------------------------------------------

impl crate::resources::DeviceResources {
    /// Upload a user-supplied font for use with overlay items.
    ///
    /// Returns an opaque [`FontHandle`] that can be passed to
    /// [`LabelItem`](crate::LabelItem) or [`GlyphRunItem`](crate::GlyphRunItem)
    /// via their `font` field.  Pass `None` on those items to use the built-in
    /// default font instead.
    ///
    /// `ttf_bytes` is a TrueType or OpenType file, copied in. For a collection
    /// (`.ttc` / `.otc`) this takes the first face; use
    /// [`upload_font_face`](Self::upload_font_face) to pick another.
    pub fn upload_font(&mut self, ttf_bytes: &[u8]) -> Result<FontHandle, FontError> {
        self.content
            .glyph_atlas
            .upload_font_face(Arc::from(ttf_bytes), 0)
    }

    /// Upload face `face` of a font file for use with overlay items.
    ///
    /// A font collection (`.ttc` / `.otc`) holds several faces in one file, and
    /// a system font lookup answers with a file and a face index; pass both
    /// here. A single font has one face, index 0. An index past the last face
    /// returns [`FontError::FaceOutOfRange`].
    ///
    /// `data` is kept shared rather than copied when it is already an
    /// `Arc<[u8]>`, so uploading several faces of one collection, or a file a
    /// font database already holds, stores the bytes once. A `Vec<u8>` or a
    /// slice is copied in once; from a `&Vec<u8>`, pass `bytes.as_slice()`.
    ///
    /// Taking a font from a system lookup, then checking it can draw a label
    /// before using it. `lookup` stands for whichever font database the
    /// application uses; each answers with a file and a face index:
    ///
    /// ```ignore
    /// let (path, face) = lookup.find_family("Noto Sans CJK SC")?;
    /// let bytes: Arc<[u8]> = std::fs::read(path)?.into();
    /// let font = resources.upload_font_face(bytes, face)?;
    /// if let Some(c) = resources.missing_glyph(label, Some(font)) {
    ///     // This face has no glyph for `c`: try the next candidate.
    /// }
    /// ```
    pub fn upload_font_face(
        &mut self,
        data: impl Into<Arc<[u8]>>,
        face: u32,
    ) -> Result<FontHandle, FontError> {
        self.content.glyph_atlas.upload_font_face(data.into(), face)
    }

    /// The first character of `text` that `font` cannot draw, or `None` when
    /// it covers all of it. `font` is `None` for the built-in default font.
    ///
    /// A label in a font that lacks its characters draws empty boxes without
    /// any error, so check a font before choosing it for text it was not
    /// picked for: a label in another script, or text the user typed.
    /// Whitespace and control characters are skipped. The built-in font only
    /// covers a cut-down Latin set, so most other scripts need a font passed to
    /// [`upload_font_face`](Self::upload_font_face).
    ///
    /// Reads the font's character map only, so it takes `&self` and needs no
    /// `device`.
    pub fn missing_glyph(&self, text: &str, font: Option<FontHandle>) -> Option<char> {
        self.content.glyph_atlas.missing_glyph(text, font)
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

    /// The font file and face index `font` refers to (`None` = the built-in
    /// default), if uploaded. A downstream text shaper can register these exact
    /// bytes at this face so its glyph ids match what the overlay atlas
    /// rasterizes; the bytes alone name the wrong face for a collection.
    ///
    /// The bytes are the shared buffer the atlas holds, so this does not copy an
    /// uploaded font. The built-in font is copied out of the binary on each call.
    pub fn font_face(&self, font: Option<FontHandle>) -> Option<(Arc<[u8]>, u32)> {
        self.content.glyph_atlas.font_face(font.map_or(0, |h| h.0))
    }
}

/// Where each word lands when the words of each hard line are packed into lines
/// no wider than `max_width`. `lines` holds one width per word, grouped by
/// `\n`-delimited line, and the returned origins are flat in the same order.
/// Also returns the widest line and the total height.
///
/// This is the whole of the wrapping rule, in one place because laying text out
/// and measuring it must agree: they differ only in whether glyphs are
/// rasterized, and a second copy of this loop is a second chance to drift. A
/// word wider than `max_width` on its own is not broken, and words sharing a
/// line are separated by one space advance.
fn wrap_words(
    lines: &[Vec<f32>],
    space_advance: f32,
    max_width: f32,
    line_height: f32,
) -> (Vec<[f32; 2]>, f32, f32) {
    let mut origins = Vec::new();
    let mut line_x: f32 = 0.0;
    let mut line_y: f32 = 0.0;
    let mut max_line_width: f32 = 0.0;

    for (line_idx, widths) in lines.iter().enumerate() {
        // A hard line break lands even when the line it opens has no words, so an
        // empty line still takes up its height.
        if line_idx > 0 {
            max_line_width = max_line_width.max(line_x);
            line_x = 0.0;
            line_y += line_height;
        }

        let mut first_on_line = true;
        for &width in widths {
            // Soft-wrap when the word does not fit after the space that would
            // precede it. The first word on a line always stays, however wide.
            if !first_on_line && line_x + space_advance + width > max_width {
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
            origins.push([start_x, line_y]);
            line_x = start_x + width;
            first_on_line = false;
        }
    }

    max_line_width = max_line_width.max(line_x);
    (origins, max_line_width, line_y + line_height)
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

    /// Pack single fonts into one collection (`ttcf`), each face keeping its own
    /// table directory, so face selection can be tested without shipping a
    /// `.ttc`.
    fn collection(fonts: &[&[u8]]) -> Vec<u8> {
        let mut faces = Vec::new();
        for src in fonts {
            let num = u16::from_be_bytes([src[4], src[5]]) as usize;
            let mut tables = Vec::with_capacity(num);
            for i in 0..num {
                let rec = 12 + i * 16;
                let tag: [u8; 4] = src[rec..rec + 4].try_into().unwrap();
                let off = u32::from_be_bytes(src[rec + 8..rec + 12].try_into().unwrap()) as usize;
                let len = u32::from_be_bytes(src[rec + 12..rec + 16].try_into().unwrap()) as usize;
                tables.push((tag, &src[off..off + len]));
            }
            faces.push((&src[0..4], tables));
        }

        let header_len = 12 + 4 * faces.len();
        let dirs_len: usize = faces.iter().map(|(_, t)| 12 + 16 * t.len()).sum();
        let mut out = Vec::new();
        out.extend(b"ttcf");
        out.extend(1u16.to_be_bytes());
        out.extend(0u16.to_be_bytes());
        out.extend((faces.len() as u32).to_be_bytes());
        let mut dir_offset = header_len;
        for (_, tables) in &faces {
            out.extend((dir_offset as u32).to_be_bytes());
            dir_offset += 12 + 16 * tables.len();
        }

        let mut body_offset = header_len + dirs_len;
        let mut body = Vec::new();
        for (version, tables) in &faces {
            out.extend(*version);
            out.extend((tables.len() as u16).to_be_bytes());
            out.extend([0u8; 6]);
            for (tag, data) in tables {
                out.extend(tag);
                out.extend(0u32.to_be_bytes());
                out.extend((body_offset as u32).to_be_bytes());
                out.extend((data.len() as u32).to_be_bytes());
                body.extend(*data);
                let pad = (4 - data.len() % 4) % 4;
                body.extend(std::iter::repeat_n(0u8, pad));
                body_offset += data.len() + pad;
            }
        }
        out.extend(body);
        out
    }

    /// A two-face collection: the bundled font as face 0, and the bundled font
    /// with a COLR table as face 1, so the faces can be told apart.
    fn two_face_collection() -> Vec<u8> {
        let plain = FontRef::from_index(DEFAULT_FONT_BYTES, 0).unwrap();
        let colr = synth_colr_font(plain.charmap().map('A'), plain.charmap().map('O'));
        collection(&[DEFAULT_FONT_BYTES, &colr])
    }

    fn has_colr(bytes: &[u8], offset: u32, key: CacheKey) -> bool {
        let font = FontRef {
            data: bytes,
            offset,
            key,
        };
        font.table(swash::tag_from_bytes(b"COLR")).is_some()
    }

    /// Each face of a collection keys to its own table directory, and an index
    /// past the last face is an error rather than a panic.
    #[test]
    fn swash_key_picks_the_face_of_a_collection() {
        let ttc = two_face_collection();
        let (off0, key0) = swash_key(&ttc, 0).unwrap();
        let (off1, key1) = swash_key(&ttc, 1).unwrap();
        assert_ne!(off0, off1);
        assert!(!has_colr(&ttc, off0, key0));
        assert!(has_colr(&ttc, off1, key1));

        assert!(matches!(
            swash_key(&ttc, 2),
            Err(FontError::FaceOutOfRange { face: 2, count: 2 })
        ));
        assert!(matches!(
            swash_key(DEFAULT_FONT_BYTES, 1),
            Err(FontError::FaceOutOfRange { face: 1, count: 1 })
        ));
        assert!(matches!(
            swash_key(b"not a font", 0),
            Err(FontError::ParseFailed(_))
        ));
    }

    /// Uploading two faces of one collection shares one buffer, and reading a
    /// handle back gives the face it was uploaded as, so a downstream shaper
    /// reads the same face the atlas draws.
    #[test]
    fn uploaded_faces_share_bytes_and_read_back_their_index() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        let ttc: Arc<[u8]> = two_face_collection().into();

        let a = atlas.upload_font_face(ttc.clone(), 0).unwrap();
        let b = atlas.upload_font_face(ttc.clone(), 1).unwrap();
        assert!(atlas.upload_font_face(ttc.clone(), 2).is_err());

        let (bytes_a, face_a) = atlas.font_face(a.0).unwrap();
        let (bytes_b, face_b) = atlas.font_face(b.0).unwrap();
        assert_eq!((face_a, face_b), (0, 1));
        assert!(Arc::ptr_eq(&bytes_a, &ttc) && Arc::ptr_eq(&bytes_b, &ttc));

        let (offset, key, _) = atlas.font_keys[b.0];
        assert!(has_colr(&bytes_b, offset, key));

        let (default, face) = atlas.font_face(0).unwrap();
        assert_eq!((&*default, face), (DEFAULT_FONT_BYTES, 0));
    }

    #[test]
    fn missing_glyph_reports_the_first_uncovered_character() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        assert_eq!(atlas.missing_glyph("Front, Back 0-9", None), None);
        assert_eq!(
            atlas.missing_glyph("X\u{4e1c}\u{5317}", None),
            Some('\u{4e1c}')
        );
        assert_eq!(
            atlas.missing_glyph(" \n\t", None),
            None,
            "whitespace is skipped"
        );
        assert_eq!(atlas.missing_glyph("", None), None);

        // An uploaded face answers from its own character map.
        let ttc: Arc<[u8]> = two_face_collection().into();
        let face1 = atlas.upload_font_face(ttc, 1).unwrap();
        assert_eq!(atlas.missing_glyph("Top", Some(face1)), None);
        assert_eq!(
            atlas.missing_glyph("\u{4e0a}", Some(face1)),
            Some('\u{4e0a}')
        );

        // A handle that was never uploaded covers nothing.
        assert_eq!(atlas.missing_glyph(" ab", Some(FontHandle(99))), Some('a'));
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

    /// The bundled font carries glyphs that no codepoint reaches, only shaping
    /// (ligatures, alternates). That is also the shape of an OpenType MATH size
    /// variant, and the reason the atlas rasterizes by glyph id rather than
    /// through a `cmap`-driven cache, since a shaper hands over ids of exactly
    /// this kind.
    #[test]
    fn rasterises_a_glyph_with_no_codepoint() {
        let (offset, key) = swash_key(DEFAULT_FONT_BYTES, 0).unwrap();
        let font = FontRef {
            data: DEFAULT_FONT_BYTES,
            offset,
            key,
        };

        let mut mapped = std::collections::HashSet::new();
        font.charmap().enumerate(|_, id| {
            mapped.insert(id);
        });

        let mut ctx = ScaleContext::new();
        let mut scaler = ctx.builder(font).size(48.0).hint(HINT_GLYPHS).build();
        let image = (1..font.metrics(&[]).glyph_count)
            .filter(|id| !mapped.contains(id))
            .find_map(|id| {
                Render::new(&[
                    Source::ColorOutline(0),
                    Source::ColorBitmap(StrikeWith::BestFit),
                    Source::Outline,
                ])
                .render(&mut scaler, id)
                .filter(|img| img.data.iter().any(|&a| a > 128))
            })
            .expect("the font outlines a glyph that has no codepoint");

        assert!(image.placement.width > 0 && image.placement.height > 0);
        assert_eq!(image.content, Content::Mask);
        assert_eq!(
            image.data.len(),
            (image.placement.width * image.placement.height) as usize
        );
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
    ///
    /// Both pack through `wrap_words`, so this no longer guards two copies of the
    /// wrapping rule. What it still guards is that they feed it the same widths:
    /// one measures a word with `advance_of`, the other sums the advances of the
    /// glyphs it shaped, and those must agree.
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
