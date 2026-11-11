//! Font atlas and single-line text layout for overlay rendering.
//!
//! This module is the text back-end for [`LabelItem`](crate::LabelItem) and
//! [`GlyphRunItem`](crate::GlyphRunItem).  It shapes, measures and rasterizes
//! with [`swash`], packing glyphs on demand into two atlas pages: one byte per
//! pixel for coverage glyphs and RGBA for colour glyphs.
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
use swash::zeno::Vector;
use swash::{CacheKey, FontRef};

/// Default font embedded in the library binary: Roboto Regular (Apache 2.0),
/// cut down to the characters listed in `fonts/README.md`. Text outside that
/// set needs a font passed to `upload_font`.
const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("../../fonts/Roboto-Regular.ttf");

/// Whether glyph outlines are hinted before filling. Off: hinting snaps stems to
/// the pixel grid, which is sharper at small sizes but distorts the shapes a font
/// designer drew, and the call belongs with a side-by-side rather than a default.
const HINT_GLYPHS: bool = false;

/// Text at or below this physical size rasterises at its own size, to a tenth of
/// a pixel. Above it, sizes snap to steps of `SIZE_STEP` and the quad is scaled
/// to the requested size, so a smooth zoom reuses a bounded set of cells instead
/// of caching new ones every frame.
const EXACT_SIZE_PX: f32 = 48.0;

/// Ratio between neighbouring raster sizes above `EXACT_SIZE_PX`.
const SIZE_STEP: f32 = 1.04;

/// Largest physical size a glyph is rasterised at. Larger text draws this raster
/// scaled up, so no single glyph needs a cell anywhere near the atlas limit.
const MAX_RASTER_PX: f32 = 256.0;

/// The size a glyph drawn at some physical size is rasterised at, and how that
/// raster scales back up to the drawn size.
#[derive(Debug, Clone, Copy)]
struct RasterSize {
    px: f32,
    /// `px` in tenths of a pixel, the size part of the cache key.
    tenths: u32,
    /// Drawn size over raster size. Exactly 1 up to `EXACT_SIZE_PX`.
    scale: f32,
}

impl RasterSize {
    fn for_px(px: f32) -> Self {
        let raster = if px <= EXACT_SIZE_PX {
            px
        } else {
            let capped = px.min(MAX_RASTER_PX);
            let steps = ((capped / EXACT_SIZE_PX).ln() / SIZE_STEP.ln()).round();
            (EXACT_SIZE_PX * SIZE_STEP.powf(steps)).min(MAX_RASTER_PX)
        };
        Self {
            px: raster,
            tenths: (raster * 10.0).round() as u32,
            scale: if raster > 0.0 && raster != px {
                px / raster
            } else {
                1.0
            },
        }
    }

    /// The raster for text drawn at `px` whose size sweeps as a camera
    /// zooms: every size snaps to the ladder `for_px` uses above
    /// `EXACT_SIZE_PX`, extended below it, so a zoom reuses a bounded set of
    /// cells at every size. A size already on the ladder (within float
    /// error) rasterises at itself, so it still snaps to the pixel grid.
    fn for_px_stepped(px: f32) -> Self {
        if !(px > 0.0) {
            return Self::for_px(px);
        }
        let capped = px.min(MAX_RASTER_PX);
        let steps = ((capped / EXACT_SIZE_PX).ln() / SIZE_STEP.ln()).round();
        let raster = (EXACT_SIZE_PX * SIZE_STEP.powf(steps)).min(MAX_RASTER_PX);
        let scale = px / raster;
        Self {
            px: raster,
            tenths: (raster * 10.0).round() as u32,
            scale: if (scale - 1.0).abs() < 1.0e-5 {
                1.0
            } else {
                scale
            },
        }
    }
}

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
    /// A bitmap face, which has no font file: its glyphs are cut out of the
    /// pages it was uploaded with.
    Bitmap(Box<BitmapFace>),
}

impl FontData {
    fn bytes(&self) -> &[u8] {
        match self {
            FontData::Static(b) => b,
            FontData::Shared(b) => b,
            FontData::Bitmap(_) => &[],
        }
    }

    /// The bytes as a shared buffer, or `None` for a bitmap face. Copies the
    /// built-in font, which is only asked for when a caller reads it back.
    fn shared(&self) -> Option<Arc<[u8]>> {
        match self {
            FontData::Static(b) => Some(Arc::from(*b)),
            FontData::Shared(b) => Some(b.clone()),
            FontData::Bitmap(_) => None,
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

use super::bitmap_font::BitmapFace;
use super::glyph_style::{GlyphStyle, style_coverage};

/// Horizontal sub-pixel positions a glyph is rasterised at: a pen lands on the
/// nearest quarter pixel and the outline is shifted by that fraction before it
/// is filled, so glyph edges keep their true position without the quad being
/// resampled. Vertical positions snap to whole pixels.
const SUBPIXEL_BINS: f32 = 4.0;

/// Edge of one plot, the unit the atlas packs into and evicts. Large enough for
/// the largest cell (a glyph at `MAX_RASTER_PX` with a shadow pad).
const PLOT_SIZE: u32 = 512;

/// Glyphs with no bitmap (whitespace, ids past the end of a font) are cached so
/// a miss is paid once; past this many the set is dropped and rebuilt.
const MAX_EMPTY_ENTRIES: usize = 4096;

// ---------------------------------------------------------------------------
// GlyphKey / GlyphEntry : atlas bookkeeping
// ---------------------------------------------------------------------------

/// Unique key for a rasterized glyph in the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey {
    font_index: usize,
    glyph_index: u16,
    /// Raster size in tenths of a pixel (e.g. 140 = 14.0 px). See `RasterSize`.
    size_tenths: u32,
    /// Post-processing applied to the cell (plain glyph, or a shadow bake).
    style: GlyphStyle,
    /// Horizontal sub-pixel offset the outline was rasterised at, in quarters
    /// of a pixel (`0..SUBPIXEL_BINS`).
    bin: u8,
}

/// Location and metrics of a single rasterized glyph in the atlas texture.
#[derive(Debug, Clone, Copy)]
struct GlyphEntry {
    /// Top-left pixel coordinate in its page.
    x: u32,
    y: u32,
    /// Rasterized bitmap dimensions.
    width: u32,
    height: u32,
    /// Offset from the pen position to the top-left of the bitmap.
    offset_x: f32,
    offset_y: f32,
    /// `true` for a colour glyph, held as RGBA in the colour page; `false` for
    /// a coverage glyph, held as one byte per pixel in the coverage page.
    color: bool,
    /// The plot of that page the cell sits in.
    plot: u16,
    /// A bitmap face's cell, magnified with no filtering.
    nearest: bool,
}

impl GlyphEntry {
    /// A glyph that draws nothing.
    const EMPTY: Self = Self {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        offset_x: 0.0,
        offset_y: 0.0,
        color: false,
        plot: 0,
        nearest: false,
    };
}

// ---------------------------------------------------------------------------
// GlyphQuad / ShapedText / GlyphGrid : layout input and output
// ---------------------------------------------------------------------------

/// A positioned, textured quad for one glyph, ready for vertex generation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GlyphQuad {
    /// Top-left corner in logical pixels, relative to the origin the glyphs
    /// were placed against.
    pub pos: [f32; 2],
    /// Size [w, h] in logical pixels.
    pub size: [f32; 2],
    /// UV top-left in the glyph's page (0..1).
    pub uv_min: [f32; 2],
    /// UV bottom-right in the glyph's page (0..1).
    pub uv_max: [f32; 2],
    /// `true` when the cell is in the colour page (drawn as-is), `false` for
    /// the coverage page (tinted by the run colour).
    pub color: bool,
    /// `true` for a bitmap face's cell, which is read texel by texel rather
    /// than filtered.
    pub nearest: bool,
}

/// Metrics for an overlay text run, matching how a [`LabelItem`] with the same
/// text, size, and font would be laid out.
///
/// All values are in logical pixels. Use these to size and align overlay-based
/// UI (panel widths, right-aligned columns, per-row hit rectangles, vertical
/// centring) before drawing the text with a `LabelItem`.
///
/// [`LabelItem`]: crate::LabelItem
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

/// Layout result for a text string, with its glyphs rasterised.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct TextLayout {
    /// One quad per visible glyph (whitespace characters are skipped).
    pub quads: Vec<GlyphQuad>,
    /// Total advance width of the laid-out string in pixels.
    pub total_width: f32,
    /// Height in pixels: one line height per line.
    pub height: f32,
}

/// A string shaped into positioned glyphs, before any glyph is rasterised.
///
/// Shaping does not depend on where the text lands, so a label shapes once and
/// places the result for each of its layers (shadows, the glyphs themselves,
/// inset layers) once its origin and transform are known.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShapedText {
    /// Glyph id and pen position per glyph, in logical pixels from the left
    /// edge of the box at the first baseline.
    pub glyphs: Vec<(u16, f32, f32)>,
    /// Advance width of the widest line, in logical pixels.
    pub total_width: f32,
    /// One line height per line, in logical pixels.
    pub height: f32,
}

/// How the logical pixels glyphs are placed in map to device pixels, which
/// decides the size glyphs are rasterised at and whether they snap.
///
/// `device = offset + scale * logical`. The scale is the frame's pixels per
/// point times the scale of the item's transform, so a label scaled 2x is
/// rasterised at twice its font size rather than magnified after rasterising.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GlyphGrid {
    /// Device pixels per logical pixel, before the item transform is applied.
    pub scale: f32,
    /// Device position of logical `[0, 0]` once the transform is applied.
    pub offset: [f32; 2],
    /// Whether a cell's texels land one-to-one on device pixels, so a pen
    /// snaps to the pixel grid and rasterises at its sub-pixel offset.
    pub snap: bool,
    /// Whether the raster size snaps to the stepped ladder at every size,
    /// for text whose size follows a zooming camera.
    pub stepped: bool,
}

impl GlyphGrid {
    /// Text with no item transform at `ppp` pixels per point.
    #[cfg(test)]
    pub(crate) fn screen(ppp: f32) -> Self {
        Self {
            scale: ppp,
            offset: [0.0, 0.0],
            snap: true,
            stepped: false,
        }
    }

    /// Text drawn under the linear map `linear` (row-major) applied about
    /// `centre`, at `ppp` pixels per point.
    ///
    /// A positive uniform scale with no rotation, stretch or skew keeps the
    /// glyphs axis-aligned: they rasterise at the scaled size and snap to the
    /// device pixels the transform lands them on. Any other map rasterises at
    /// the largest factor it stretches by, with no sub-pixel offset and no
    /// snapping, since its glyphs are resampled by the transform anyway.
    pub(crate) fn for_transform(linear: [[f32; 2]; 2], centre: [f32; 2], ppp: f32) -> Self {
        let k = linear[0][0];
        if linear[0][1] == 0.0 && linear[1][0] == 0.0 && linear[1][1] == k && k > 0.0 {
            return Self {
                scale: ppp * k,
                offset: [ppp * centre[0] * (1.0 - k), ppp * centre[1] * (1.0 - k)],
                snap: true,
                stepped: false,
            };
        }
        Self {
            scale: ppp * spectral_norm(linear).max(1.0 / 64.0),
            offset: [0.0, 0.0],
            snap: false,
            stepped: false,
        }
    }

    /// The same grid with its raster size on the stepped ladder when
    /// `stepped` is set.
    pub(crate) fn with_stepped(mut self, stepped: bool) -> Self {
        self.stepped = stepped;
        self
    }
}

/// The largest factor by which the 2x2 map `m` stretches any direction.
fn spectral_norm(m: [[f32; 2]; 2]) -> f32 {
    let a = m[0][0] * m[0][0] + m[1][0] * m[1][0];
    let b = m[0][1] * m[0][1] + m[1][1] * m[1][1];
    let c = m[0][0] * m[0][1] + m[1][0] * m[1][1];
    ((a + b) * 0.5 + (((a - b) * 0.5).powi(2) + c * c).sqrt()).sqrt()
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
// AtlasPage : one texture, packed in plots
// ---------------------------------------------------------------------------

/// A texel type an atlas page stores.
trait Texel: bytemuck::Pod {
    /// The value of an empty texel, which the gutters between cells hold.
    const BLANK: Self;
    const FORMAT: crate::gpu::TextureFormat;
}

impl Texel for u8 {
    const BLANK: Self = 0;
    const FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::R8Unorm;
}

impl Texel for [u8; 4] {
    const BLANK: Self = [255, 255, 255, 0];
    // sRGB so colour glyph bytes (sRGB-encoded) sample as linear for the
    // shader, matching the linear-float tint contract.
    const FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rgba8UnormSrgb;
}

/// A square region of a page with its own row packer, the unit of eviction.
#[derive(Debug)]
struct Plot {
    /// Top-left pixel of the plot in the page.
    x: u32,
    y: u32,
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
    /// The atlas frame a glyph in this plot was last drawn in.
    last_used: u64,
    /// Bumped each time the plot is evicted, so geometry that baked UVs into
    /// it can tell its cells are gone.
    generation: u32,
    /// The glyphs packed here, removed from the cache when the plot is evicted.
    keys: Vec<GlyphKey>,
}

impl Plot {
    fn new(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            last_used: 0,
            generation: 0,
            keys: Vec::new(),
        }
    }

    /// Reserve a `w * h` cell plus a one-pixel gutter in a plot `edge` pixels
    /// on a side, returning its page position. Leaves the plot untouched when
    /// the cell does not fit.
    fn alloc(&mut self, w: u32, h: u32, edge: u32) -> Option<(u32, u32)> {
        let pad = 1;
        let (mut cx, mut cy, mut rh) = (self.cursor_x, self.cursor_y, self.row_height);
        if cx > 0 && cx + w + pad > edge {
            cy += rh + pad;
            cx = 0;
            rh = 0;
        }
        if cx + w + pad > edge || cy + h + pad > edge {
            return None;
        }
        self.cursor_x = cx + w + pad;
        self.cursor_y = cy;
        self.row_height = rh.max(h);
        Some((self.x + cx, self.y + cy))
    }
}

/// One atlas texture and its CPU copy, packed in plots.
///
/// The page starts empty and grows by doubling up to `max_size`, each growth
/// adding plots beside the existing ones. Once it is at its largest, a cell
/// that fits nowhere evicts the least recently used plot not drawn from this
/// frame; the rest of the page, and every glyph in it, stays.
struct AtlasPage<T: Texel> {
    label: &'static str,
    /// CPU copy of the page, row-major.
    pixels: Vec<T>,
    /// Page edge in pixels; zero until the first cell is packed, so an
    /// application that never draws a colour glyph never pays for that page.
    size: u32,
    max_size: u32,
    /// Plot edge: `PLOT_SIZE`, or the page's largest size if that is smaller.
    plot_size: u32,
    plots: Vec<Plot>,
    texture: crate::gpu::Texture,
    view: crate::gpu::TextureView,
    /// Rows written since the last upload, as `start..end`.
    dirty_rows: Option<(u32, u32)>,
}

impl<T: Texel> AtlasPage<T> {
    /// Initial page size in pixels (width = height).
    const INITIAL_SIZE: u32 = 512;

    fn new(device: &crate::gpu::Device, label: &'static str, max_size: u32) -> Self {
        let (texture, view) = Self::create_texture(device, label, 1);
        Self {
            label,
            pixels: Vec::new(),
            size: 0,
            max_size,
            plot_size: PLOT_SIZE.min(max_size),
            plots: Vec::new(),
            texture,
            view,
            dirty_rows: None,
        }
    }

    /// Find room for a `w * h` cell: in an existing plot, in a new plot, in a
    /// grown page, or by evicting the least recently used plot not used in
    /// `frame`. Returns the plot index and page position, appending the keys
    /// of an evicted plot to `evicted`. Sets `grew` when the page grew.
    /// `None` when every plot was used this frame and none has room.
    fn alloc(
        &mut self,
        device: &crate::gpu::Device,
        w: u32,
        h: u32,
        frame: u64,
        evicted: &mut Vec<GlyphKey>,
        grew: &mut bool,
    ) -> Option<(u16, u32, u32)> {
        let edge = self.plot_size;
        for (i, plot) in self.plots.iter_mut().enumerate() {
            if let Some((x, y)) = plot.alloc(w, h, edge) {
                plot.last_used = frame;
                return Some((i as u16, x, y));
            }
        }
        loop {
            if self.size == 0 {
                self.grow(device);
                *grew = true;
            }
            if let Some((x, y)) = self.free_slot() {
                let mut plot = Plot::new(x, y);
                let at = plot.alloc(w, h, edge)?;
                plot.last_used = frame;
                self.plots.push(plot);
                return Some(((self.plots.len() - 1) as u16, at.0, at.1));
            }
            if self.size >= self.max_size {
                break;
            }
            self.grow(device);
            *grew = true;
        }

        let (index, _) = self
            .plots
            .iter()
            .enumerate()
            .filter(|(_, p)| p.last_used < frame)
            .min_by_key(|(_, p)| p.last_used)?;
        let plot = &mut self.plots[index];
        evicted.append(&mut plot.keys);
        plot.cursor_x = 0;
        plot.cursor_y = 0;
        plot.row_height = 0;
        plot.generation = plot.generation.wrapping_add(1);
        plot.last_used = frame;
        let (px, py) = (plot.x, plot.y);
        let at = plot.alloc(w, h, edge)?;
        // Cleared rather than left stale: the gutters between the new cells
        // must be empty, or bilinear filtering bleeds old glyphs into new ones.
        for row in py..py + edge {
            let start = (row * self.size + px) as usize;
            self.pixels[start..start + edge as usize].fill(T::BLANK);
        }
        self.mark_dirty(py, py + edge);
        Some((index as u16, at.0, at.1))
    }

    /// The top-left of a plot-sized square of the page no plot covers yet.
    fn free_slot(&self) -> Option<(u32, u32)> {
        let n = self.size / self.plot_size;
        for gy in 0..n {
            for gx in 0..n {
                let (x, y) = (gx * self.plot_size, gy * self.plot_size);
                if !self.plots.iter().any(|p| p.x == x && p.y == y) {
                    return Some((x, y));
                }
            }
        }
        None
    }

    /// Copy a `w * h` cell into the page at `(x, y)`.
    fn write(&mut self, x: u32, y: u32, w: u32, h: u32, cell: &[T]) {
        for row in 0..h {
            let src = (row * w) as usize;
            let dst = ((y + row) * self.size + x) as usize;
            self.pixels[dst..dst + w as usize].copy_from_slice(&cell[src..src + w as usize]);
        }
        self.mark_dirty(y, y + h);
    }

    fn mark_dirty(&mut self, start: u32, end: u32) {
        self.dirty_rows = Some(match self.dirty_rows {
            Some((s, e)) => (s.min(start), e.max(end)),
            None => (start, end),
        });
    }

    fn upload_if_dirty(&mut self, queue: &crate::gpu::Queue) {
        let Some((start, end)) = self.dirty_rows.take() else {
            return;
        };
        let row = self.size as usize;
        let rows = &self.pixels[start as usize * row..end as usize * row];
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d {
                    x: 0,
                    y: start,
                    z: 0,
                },
                aspect: crate::gpu::TextureAspect::All,
            },
            bytemuck::cast_slice(rows),
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size * std::mem::size_of::<T>() as u32),
                rows_per_image: Some(end - start),
            },
            crate::gpu::Extent3d {
                width: self.size,
                height: end - start,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Double the page (or allocate it at its initial size), up to
    /// `max_size`, keeping every cell at its pixel position.
    fn grow(&mut self, device: &crate::gpu::Device) {
        let old_size = self.size;
        let new_size = if old_size == 0 {
            Self::INITIAL_SIZE
        } else {
            old_size * 2
        }
        .min(self.max_size);
        if old_size > 0 {
            tracing::info!(
                "Growing {} from {}x{} to {}x{}",
                self.label,
                old_size,
                old_size,
                new_size,
                new_size
            );
        }
        let mut new_pixels = vec![T::BLANK; (new_size * new_size) as usize];
        for row in 0..old_size {
            let src = (row * old_size) as usize;
            let dst = (row * new_size) as usize;
            new_pixels[dst..dst + old_size as usize]
                .copy_from_slice(&self.pixels[src..src + old_size as usize]);
        }
        self.pixels = new_pixels;
        self.size = new_size;
        let (texture, view) = Self::create_texture(device, self.label, new_size);
        self.texture = texture;
        self.view = view;
        self.dirty_rows = Some((0, new_size));
    }

    fn create_texture(
        device: &crate::gpu::Device,
        label: &'static str,
        size: u32,
    ) -> (crate::gpu::Texture, crate::gpu::TextureView) {
        let texture = device.logged_texture(&crate::gpu::TextureDescriptor {
            label: Some(label),
            size: crate::gpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: T::FORMAT,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

/// The atlas cells a piece of retained geometry baked UVs against: the page
/// growth version and each plot it drew from with that plot's generation.
/// The geometry is current while neither has moved.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct GlyphDeps {
    version: u64,
    /// `(colour page, plot index, generation)`.
    plots: Vec<(bool, u16, u32)>,
}

// ---------------------------------------------------------------------------
// GlyphAtlas
// ---------------------------------------------------------------------------

/// The glyph cache: rasterised glyphs packed into two pages, an `R8Unorm` page
/// for coverage glyphs (text, and the shadow cells baked from it) and an
/// `Rgba8UnormSrgb` page for colour glyphs (emoji, colour bitmap fonts).
///
/// A glyph is keyed by font, id, raster size, style and horizontal sub-pixel
/// bin. A full page evicts its least recently used plot rather than clearing,
/// so the glyphs a frame keeps drawing stay resident.
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

    /// Glyphs resident in a page.
    entries: HashMap<GlyphKey, GlyphEntry>,
    /// Glyphs that rasterise to nothing.
    empty: std::collections::HashSet<GlyphKey>,

    /// Coverage glyphs, one byte per pixel.
    coverage: AtlasPage<u8>,
    /// Colour glyphs, RGBA.
    colour: AtlasPage<[u8; 4]>,

    /// The frame glyphs are being drawn for, advanced by `begin_frame`. A
    /// plot used in this frame is never evicted: quads laid out against it
    /// earlier in the frame still point at its cells.
    frame: u64,

    /// Set when a glyph found no room because every plot was used this
    /// frame. Further misses this frame skip rasterising and draw nothing;
    /// `begin_frame` clears it.
    full: bool,

    /// Bumped every time a page grows, which changes the UV of every cell in
    /// it (`pixel / size`) though no pixel moves. Retained geometry records it
    /// in its `GlyphDeps` and re-emits when it moves.
    atlas_version: u64,

    /// Glyph cells rasterised since the atlas was created: the cache misses.
    rasters: u64,

    /// Plots evicted since the atlas was created.
    evictions: u64,

    /// The plots drawn from since `begin_recording`, for retained geometry.
    recording: Option<Vec<(bool, u16)>>,
}

impl GlyphAtlas {
    /// Largest page edge in pixels, whatever the device allows, which keeps
    /// text memory bounded on mobile and web.
    const MAX_SIZE: u32 = 4096;

    /// Create a new atlas with the built-in default font pre-loaded. Each page
    /// is a 1x1 placeholder until its first glyph is packed.
    pub fn new(device: &crate::gpu::Device) -> Self {
        let (offset, key) =
            swash_key(DEFAULT_FONT_BYTES, 0).expect("built-in default font must parse");
        let max_size = device.limits().max_texture_dimension_2d.min(Self::MAX_SIZE);

        Self {
            font_bytes: vec![FontData::Static(DEFAULT_FONT_BYTES)],
            font_keys: vec![(offset, key, 0)],
            scale: ScaleContext::new(),
            shape: std::sync::Mutex::new(ShapeContext::new()),
            entries: HashMap::new(),
            empty: std::collections::HashSet::new(),
            coverage: AtlasPage::new(device, "glyph_atlas_coverage", max_size),
            colour: AtlasPage::new(device, "glyph_atlas_colour", max_size),
            frame: 1,
            full: false,
            atlas_version: 0,
            rasters: 0,
            evictions: 0,
            recording: None,
        }
    }

    /// The page growth version. Retained glyph geometry re-emits when this
    /// moves; see [`GlyphDeps`].
    pub fn version(&self) -> u64 {
        self.atlas_version
    }

    /// Glyph cells rasterised since the atlas was created: the cache misses.
    pub(crate) fn rasters(&self) -> u64 {
        self.rasters
    }

    /// Fonts uploaded, the built-in one included. A font is never replaced,
    /// so text laid out against the same count used the same fonts.
    pub(crate) fn font_count(&self) -> usize {
        self.font_bytes.len()
    }

    /// The coverage page, sampled for coverage glyphs and shadow cells.
    pub(crate) fn coverage_view(&self) -> &crate::gpu::TextureView {
        &self.coverage.view
    }

    /// The colour page, sampled for colour glyphs.
    pub(crate) fn colour_view(&self) -> &crate::gpu::TextureView {
        &self.colour.view
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

    /// Register a bitmap face. Returns a [`FontHandle`] that can be passed to
    /// overlay items.
    pub fn upload_bitmap_font(
        &mut self,
        font: &viewport_lib_types::overlay::BitmapFont,
    ) -> Result<FontHandle, FontError> {
        let face = BitmapFace::new(font)?;
        let index = self.font_bytes.len();
        // A bitmap face shapes and rasterises without swash; the key is
        // never read.
        self.font_keys.push((0, CacheKey::new(), 0));
        self.font_bytes.push(FontData::Bitmap(Box::new(face)));
        Ok(FontHandle(index))
    }

    /// The font index `font` names, or the built-in font's for a handle this
    /// atlas never issued (one from another renderer), so text laid out with
    /// it draws in the built-in font rather than panicking.
    fn font_index(&self, font: Option<FontHandle>) -> usize {
        match font {
            Some(h) if h.0 < self.font_bytes.len() => h.0,
            _ => 0,
        }
    }

    /// The bitmap face at `font_index`, if that font is one.
    fn bitmap(&self, font_index: usize) -> Option<&BitmapFace> {
        match self.font_bytes.get(font_index)? {
            FontData::Bitmap(face) => Some(face),
            _ => None,
        }
    }

    /// The font file and face index at `index` (a [`FontHandle`]'s value), if it
    /// has been uploaded and is not a bitmap face. Index 0 is the built-in
    /// default font.
    pub(crate) fn font_face(&self, index: usize) -> Option<(Arc<[u8]>, u32)> {
        let data = self.font_bytes.get(index)?;
        Some((data.shared()?, self.font_keys[index].2))
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
        if let Some(face) = self.bitmap(font_index) {
            return drawn.find(|&c| face.id(c).is_none());
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

    /// Shape a string, one run per `\n`-delimited line, without rasterising.
    ///
    /// Shaping runs at the physical size (`font_size * ppp`) and the result
    /// comes back in logical pixels, so the advances are the ones the display
    /// resolution gives.
    pub(crate) fn shape_text(
        &self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
        ppp: f32,
    ) -> ShapedText {
        let font_index = self.font_index(font);
        if let Some(face) = self.bitmap(font_index) {
            return bitmap_shape(face, text, font_size, ppp, None);
        }
        let px = font_size * ppp;
        let (_, line_height) = self.line_metrics(font_index, px);

        let mut glyphs = Vec::new();
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
            for g in &shaped {
                glyphs.push((g.id, pen_x + g.x, pen_y - g.y));
                pen_x += g.advance;
            }
            max_width = max_width.max(pen_x);
        }

        let inv = 1.0 / ppp;
        for g in &mut glyphs {
            g.1 *= inv;
            g.2 *= inv;
        }
        ShapedText {
            glyphs,
            total_width: max_width * inv,
            height: (pen_y + line_height) * inv,
        }
    }

    /// Shape a string with word wrapping at `max_width` logical pixels.
    ///
    /// Words that exceed `max_width` on their own are not broken: they extend
    /// past the boundary. `total_width` is the widest line actually used.
    pub(crate) fn shape_text_wrapped(
        &self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
        max_width: f32,
        ppp: f32,
    ) -> ShapedText {
        let font_index = self.font_index(font);
        if let Some(face) = self.bitmap(font_index) {
            return bitmap_shape(face, text, font_size, ppp, Some(max_width));
        }
        // Shape at physical size (see `shape_text`). `max_width` arrives in
        // logical points, so scale it up to match the physical pen units the
        // wrap test below works in.
        let px = font_size * ppp;
        let max_width = max_width * ppp;
        let (_, line_height) = self.line_metrics(font_index, px);
        let space_advance = self.advance_of(font_index, " ", px);

        // Shape each word at its own origin, then let the shared rule decide
        // where the words land. One word is one run: shaping stops at the
        // word boundary the wrapper chose, which is where a line may break
        // anyway.
        let mut words: Vec<Vec<(u16, f32, f32)>> = Vec::new();
        let mut lines: Vec<Vec<f32>> = Vec::new();
        let mut shaped = Vec::new();
        for logical_line in text.split('\n') {
            let mut widths = Vec::new();
            for word in logical_line.split_whitespace() {
                self.shape_run(font_index, word, px, &mut shaped);
                let mut pen_x: f32 = 0.0;
                let mut glyphs = Vec::with_capacity(shaped.len());
                for g in &shaped {
                    glyphs.push((g.id, pen_x + g.x, -g.y));
                    pen_x += g.advance;
                }
                words.push(glyphs);
                widths.push(pen_x);
            }
            lines.push(widths);
        }

        let (origins, max_line_width, total_height) =
            wrap_words(&lines, space_advance, max_width, line_height);

        let inv = 1.0 / ppp;
        let glyphs = words
            .into_iter()
            .zip(origins)
            .flat_map(|(word, o)| {
                word.into_iter()
                    .map(move |(id, x, y)| (id, (o[0] + x) * inv, (o[1] + y) * inv))
            })
            .collect();
        ShapedText {
            glyphs,
            total_width: max_line_width * inv,
            height: total_height * inv,
        }
    }

    /// Rasterise shaped text placed with its first baseline's left end at
    /// `origin`; see [`Self::place_glyphs`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn place_text(
        &mut self,
        shaped: &ShapedText,
        font_size: f32,
        font: Option<FontHandle>,
        origin: [f32; 2],
        grid: GlyphGrid,
        device: &crate::gpu::Device,
        style: GlyphStyle,
    ) -> Vec<GlyphQuad> {
        self.place_glyphs(
            shaped.glyphs.iter().map(|&(id, x, y)| (id, x, y, ())),
            font_size,
            font,
            origin,
            grid,
            device,
            style,
        )
        .into_iter()
        .map(|(q, ())| q)
        .collect()
    }

    /// Rasterise glyphs at their pen positions and return a quad for each one
    /// that draws, paired with its payload.
    ///
    /// `glyphs` yields `(glyph_id, x, y, payload)`, with the pen position in
    /// logical pixels relative to `origin`, and the returned quads are relative
    /// to `origin` too. No shaping happens here, which makes this the back-end
    /// for [`GlyphRunItem`] as well as for shaped labels.
    ///
    /// Glyphs rasterise at `font_size * grid.scale` device pixels. On a grid
    /// that snaps, each pen is moved to the nearest quarter pixel across and
    /// the nearest whole pixel down, the outline is rasterised at that quarter
    /// offset, and the quad lands on whole device pixels, so its texels map
    /// one-to-one onto the target. Above `EXACT_SIZE_PX` the raster is a
    /// quantised size scaled to the drawn one, and nothing snaps.
    ///
    /// Zero-area glyphs (whitespace and the like) are skipped; the payload is
    /// threaded through the skip so it stays aligned with the quad it belongs
    /// to.
    ///
    /// [`GlyphRunItem`]: crate::renderer::types::GlyphRunItem
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn place_glyphs<I, P>(
        &mut self,
        glyphs: I,
        font_size: f32,
        font: Option<FontHandle>,
        origin: [f32; 2],
        grid: GlyphGrid,
        device: &crate::gpu::Device,
        style: GlyphStyle,
    ) -> Vec<(GlyphQuad, P)>
    where
        I: IntoIterator<Item = (u16, f32, f32, P)>,
    {
        let font_index = self.font_index(font);
        // A bitmap face draws its cells magnified by a whole multiple, so its
        // texels cover whole device pixels at any size and its pens snap to
        // whole pixels. The stepped ladder does not apply to it.
        let bitmap = self
            .bitmap(font_index)
            .map(|face| face.scale_for(font_size * grid.scale));
        let raster = if let Some(k) = bitmap {
            RasterSize {
                px: 0.0,
                tenths: 0,
                scale: k as f32,
            }
        } else if grid.stepped {
            RasterSize::for_px_stepped(font_size * grid.scale)
        } else {
            RasterSize::for_px(font_size * grid.scale)
        };
        // Snapping only lands texels on pixels while a texel is a pixel.
        let snap = grid.snap && (raster.scale == 1.0 || bitmap.is_some());
        let inv = 1.0 / grid.scale;

        let mut quads = Vec::new();
        for (glyph_id, x, y, payload) in glyphs {
            let dx = grid.offset[0] + (origin[0] + x) * grid.scale;
            let dy = grid.offset[1] + (origin[1] + y) * grid.scale;
            let (pen, bin) = if snap && bitmap.is_some() {
                ([dx.round(), dy.round()], 0)
            } else if snap {
                let quarters = (dx * SUBPIXEL_BINS).round();
                let whole = (quarters / SUBPIXEL_BINS).floor();
                (
                    [whole, dy.round()],
                    (quarters - whole * SUBPIXEL_BINS) as u8,
                )
            } else {
                ([dx, dy], 0)
            };
            // Whether a glyph draws is `ensure_glyph`'s call: it may have an
            // outline, a colour strike, or nothing at all.
            let entry = self.ensure_glyph(device, font_index, glyph_id, raster, bin, style);
            if entry.width == 0 {
                continue;
            }
            // The cell in device pixels about the pen, then back to logical
            // pixels relative to `origin`.
            let mut quad = self.quad(&entry, raster.scale);
            quad.pos = [
                (pen[0] + quad.pos[0] - grid.offset[0]) * inv - origin[0],
                (pen[1] + quad.pos[1] - grid.offset[1]) * inv - origin[1],
            ];
            quad.size = [quad.size[0] * inv, quad.size[1] * inv];
            quads.push((quad, payload));
        }
        quads
    }

    /// Lay out and rasterise a single-line string with no item transform.
    #[cfg(test)]
    pub fn layout_text(
        &mut self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
        ppp: f32,
        device: &crate::gpu::Device,
        style: GlyphStyle,
    ) -> TextLayout {
        let shaped = self.shape_text(text, font_size, font, ppp);
        let quads = self.place_text(
            &shaped,
            font_size,
            font,
            [0.0, 0.0],
            GlyphGrid::screen(ppp),
            device,
            style,
        );
        TextLayout {
            quads,
            total_width: shaped.total_width,
            height: shaped.height,
        }
    }

    /// Lay out and rasterise a wrapped string with no item transform.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
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
        let shaped = self.shape_text_wrapped(text, font_size, font, max_width, ppp);
        let quads = self.place_text(
            &shaped,
            font_size,
            font,
            [0.0, 0.0],
            GlyphGrid::screen(ppp),
            device,
            style,
        );
        TextLayout {
            quads,
            total_width: shaped.total_width,
            height: shaped.height,
        }
    }

    /// Return the font ascent in pixels for the given font index and size.
    ///
    /// The ascent is the distance from the baseline to the top of the tallest
    /// glyph.  Used to position glyph quads relative to a text origin at the
    /// top-left corner of the bounding box.
    ///
    /// `ppp` matters only for a bitmap face, whose size rounds to a whole
    /// multiple of its own in device pixels.
    pub fn font_ascent(&self, font_index: usize, font_size: f32, ppp: f32) -> f32 {
        let font_index = self.font_index(Some(FontHandle(font_index)));
        if let Some(face) = self.bitmap(font_index) {
            let k = face.scale_for(font_size * ppp);
            return (face.base * k) as f32 / ppp;
        }
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
        let font_index = self.font_index(font);
        if let Some(face) = self.bitmap(font_index) {
            let shaped = bitmap_shape(face, text, font_size, 1.0, None);
            return TextMetrics {
                width: shaped.total_width,
                height: shaped.height,
                ascent: self.font_ascent(font_index, font_size, 1.0),
            };
        }
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
            ascent: self.font_ascent(font_index, font_size, 1.0),
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
        let font_index = self.font_index(font);
        if let Some(face) = self.bitmap(font_index) {
            let shaped = bitmap_shape(face, text, font_size, 1.0, Some(max_width));
            return TextMetrics {
                width: shaped.total_width,
                height: shaped.height,
                ascent: self.font_ascent(font_index, font_size, 1.0),
            };
        }
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
            ascent: self.font_ascent(font_index, font_size, 1.0),
        }
    }

    /// Upload the glyphs rasterized since the last upload, page by page.
    pub fn upload_if_dirty(&mut self, queue: &crate::gpu::Queue) {
        self.coverage.upload_if_dirty(queue);
        self.colour.upload_if_dirty(queue);
    }

    /// Start a new frame of glyph use. Call once per presented frame, before
    /// any text is laid out: plots used only in earlier frames become
    /// evictable, and a page that was full last frame is tried again.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.full = false;
    }

    /// Start collecting the plots glyph placement draws from, for retained
    /// geometry to record in its [`GlyphDeps`].
    pub(crate) fn begin_recording(&mut self) {
        self.recording = Some(Vec::new());
    }

    /// The plots drawn from since [`Self::begin_recording`], with the current
    /// page version.
    pub(crate) fn finish_recording(&mut self) -> GlyphDeps {
        let mut used = self.recording.take().unwrap_or_default();
        used.sort_unstable();
        used.dedup();
        let plots = used
            .into_iter()
            .map(|(colour, plot)| (colour, plot, self.plot(colour, plot).generation))
            .collect();
        GlyphDeps {
            version: self.atlas_version,
            plots,
        }
    }

    /// Whether every cell `deps` baked UVs against is still where it was: no
    /// page has grown and none of its plots has been evicted since.
    pub(crate) fn is_current(&self, deps: &GlyphDeps) -> bool {
        deps.version == self.atlas_version
            && deps
                .plots
                .iter()
                .all(|&(colour, plot, generation)| self.plot(colour, plot).generation == generation)
    }

    /// Mark the plots `deps` draws from as used this frame, so nothing laid
    /// out later in the frame evicts them from under it.
    pub(crate) fn touch(&mut self, deps: &GlyphDeps) {
        for &(colour, plot, _) in &deps.plots {
            self.touch_plot(colour, plot);
        }
    }

    fn plot(&self, colour: bool, plot: u16) -> &Plot {
        if colour {
            &self.colour.plots[plot as usize]
        } else {
            &self.coverage.plots[plot as usize]
        }
    }

    fn touch_plot(&mut self, colour: bool, plot: u16) {
        let frame = self.frame;
        let p = if colour {
            &mut self.colour.plots[plot as usize]
        } else {
            &mut self.coverage.plots[plot as usize]
        };
        p.last_used = frame;
        if let Some(rec) = &mut self.recording {
            rec.push((colour, plot));
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// The quad for `entry` with its pen at the origin, in drawn physical
    /// pixels. `scale` takes the cell from its raster size to the drawn size.
    fn quad(&self, entry: &GlyphEntry, scale: f32) -> GlyphQuad {
        let page_size = if entry.color {
            self.colour.size
        } else {
            self.coverage.size
        } as f32;
        GlyphQuad {
            pos: [entry.offset_x * scale, entry.offset_y * scale],
            size: [entry.width as f32 * scale, entry.height as f32 * scale],
            uv_min: [entry.x as f32 / page_size, entry.y as f32 / page_size],
            uv_max: [
                (entry.x + entry.width) as f32 / page_size,
                (entry.y + entry.height) as f32 / page_size,
            ],
            color: entry.color,
            nearest: entry.nearest,
        }
    }

    /// Ensure a glyph is in the atlas, rasterizing and packing it if needed.
    /// `bin` is the horizontal sub-pixel offset in quarters of a pixel.
    /// Returns the atlas entry.
    fn ensure_glyph(
        &mut self,
        device: &crate::gpu::Device,
        font_index: usize,
        glyph_index: u16,
        raster: RasterSize,
        bin: u8,
        style: GlyphStyle,
    ) -> GlyphEntry {
        // A style is in drawn pixels; the cell is baked at the raster size.
        let style = style.scaled(1.0 / raster.scale);
        let key = GlyphKey {
            font_index,
            glyph_index,
            size_tenths: raster.tenths,
            style,
            bin,
        };

        if let Some(&entry) = self.entries.get(&key) {
            self.touch_plot(entry.color, entry.plot);
            return entry;
        }
        if self.empty.contains(&key) {
            return GlyphEntry::EMPTY;
        }
        // Nothing more fits this frame, so skip the rasterisation as well.
        if self.full {
            return GlyphEntry::EMPTY;
        }

        if self.bitmap(font_index).is_some() {
            return self.ensure_bitmap_glyph(device, key, font_index, glyph_index, style);
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
            let mut scaler = scale
                .builder(font)
                .size(raster.px)
                .hint(HINT_GLYPHS)
                .build();
            Render::new(&[
                Source::ColorOutline(0),
                Source::ColorBitmap(StrikeWith::BestFit),
                Source::Outline,
            ])
            .offset(Vector::new(bin as f32 / SUBPIXEL_BINS, 0.0))
            .render(&mut scaler, glyph_index)
        };
        self.rasters += 1;

        // No outline and no strike: whitespace, or an id past the end of the font.
        // Remember the miss so it is paid once.
        let Some(image) = image.filter(|i| i.placement.width > 0 && i.placement.height > 0) else {
            if self.empty.len() >= MAX_EMPTY_ENTRIES {
                self.empty.clear();
            }
            self.empty.insert(key);
            return GlyphEntry::EMPTY;
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
            return self.pack(
                device,
                key,
                Cell::Coverage(&cell),
                sw,
                sh,
                offset_x - pad as f32,
                offset_y - pad as f32,
            );
        }

        if is_colour {
            // Straight (non-premultiplied) RGBA, as it comes out.
            let cell: &[[u8; 4]] = bytemuck::cast_slice(&image.data);
            self.pack(device, key, Cell::Colour(cell), w, h, offset_x, offset_y)
        } else {
            self.pack(
                device,
                key,
                Cell::Coverage(&image.data),
                w,
                h,
                offset_x,
                offset_y,
            )
        }
    }

    /// Pack a bitmap face's glyph, or the shadow cell baked from it, at the
    /// face's own size. `style` is already in the face's pixels.
    fn ensure_bitmap_glyph(
        &mut self,
        device: &crate::gpu::Device,
        key: GlyphKey,
        font_index: usize,
        glyph_index: u16,
        style: GlyphStyle,
    ) -> GlyphEntry {
        let Some(face) = self.bitmap(font_index) else {
            return GlyphEntry::EMPTY;
        };
        let Some(cell) = face.cells.get(glyph_index as usize) else {
            self.empty.insert(key);
            return GlyphEntry::EMPTY;
        };
        let (w, h) = (cell.width, cell.height);
        if w == 0 || h == 0 {
            self.empty.insert(key);
            return GlyphEntry::EMPTY;
        }
        let offset_x = cell.offset[0] as f32;
        let offset_y = (cell.offset[1] - face.base as i32) as f32;
        let colour = face.colour;
        // The cells are copied out so the face's borrow ends before packing.
        if !style.is_plain() {
            let coverage = cell.coverage.clone();
            let (styled, sw, sh, pad) = style_coverage(&coverage, w, h, style);
            return self.pack(
                device,
                key,
                Cell::Coverage(&styled),
                sw,
                sh,
                offset_x - pad as f32,
                offset_y - pad as f32,
            );
        }
        if colour {
            let rgba = cell.rgba.clone();
            self.pack(device, key, Cell::Colour(&rgba), w, h, offset_x, offset_y)
        } else {
            let coverage = cell.coverage.clone();
            self.pack(
                device,
                key,
                Cell::Coverage(&coverage),
                w,
                h,
                offset_x,
                offset_y,
            )
        }
    }

    /// Pack a `w * h` cell into its page, growing the page or evicting a plot
    /// if needed, and record the entry.
    ///
    /// A cell bigger than a plot never fits and is remembered as empty. A cell
    /// that finds no room because every plot was drawn from this frame marks
    /// the atlas full and draws nothing; it is not cached, so it packs again
    /// next frame.
    #[allow(clippy::too_many_arguments)]
    fn pack(
        &mut self,
        device: &crate::gpu::Device,
        key: GlyphKey,
        cell: Cell<'_>,
        w: u32,
        h: u32,
        offset_x: f32,
        offset_y: f32,
    ) -> GlyphEntry {
        let color = matches!(cell, Cell::Colour(_));
        let plot_size = if color {
            self.colour.plot_size
        } else {
            self.coverage.plot_size
        };
        if w + 1 > plot_size || h + 1 > plot_size {
            self.empty.insert(key);
            return GlyphEntry::EMPTY;
        }

        let mut evicted = Vec::new();
        let mut grew = false;
        let frame = self.frame;
        let placed = match cell {
            Cell::Coverage(px) => {
                let at = self
                    .coverage
                    .alloc(device, w, h, frame, &mut evicted, &mut grew);
                if let Some((_, x, y)) = at {
                    self.coverage.write(x, y, w, h, px);
                }
                at
            }
            Cell::Colour(px) => {
                let at = self
                    .colour
                    .alloc(device, w, h, frame, &mut evicted, &mut grew);
                if let Some((_, x, y)) = at {
                    self.colour.write(x, y, w, h, px);
                }
                at
            }
        };
        if grew {
            // Every cell's UV divisor (the page size) just changed, so cached
            // glyph geometry baked against the old size is stale.
            self.atlas_version += 1;
        }
        if !evicted.is_empty() {
            self.evictions += 1;
        }
        for k in evicted.drain(..) {
            self.entries.remove(&k);
        }
        let Some((plot, x, y)) = placed else {
            self.full = true;
            return GlyphEntry::EMPTY;
        };

        let entry = GlyphEntry {
            x,
            y,
            width: w,
            height: h,
            offset_x,
            offset_y,
            color,
            plot,
            nearest: self.bitmap(key.font_index).is_some(),
        };
        self.entries.insert(key, entry);
        let keys = if color {
            &mut self.colour.plots[plot as usize].keys
        } else {
            &mut self.coverage.plots[plot as usize].keys
        };
        keys.push(key);
        self.touch_plot(color, plot);
        entry
    }
}

/// The pixels of a cell to pack, which also say which page it goes in.
enum Cell<'a> {
    Coverage(&'a [u8]),
    Colour(&'a [[u8; 4]]),
}

// ---------------------------------------------------------------------------
// FontError
// ---------------------------------------------------------------------------

/// Error returned by [`DeviceResources::upload_font`](crate::DeviceResources::upload_font)
/// and [`DeviceResources::upload_font_face`](crate::DeviceResources::upload_font_face).
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
    /// A bitmap font could not be cut into glyphs: a glyph names a page the
    /// font does not have or reaches past a page's edge, or a page's pixels
    /// cannot be read.
    #[error("bitmap font is not usable: {0}")]
    InvalidBitmapFont(String),
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

    /// Upload a bitmap font for use with overlay items.
    ///
    /// Returns a [`FontHandle`] that labels and glyph runs name like any other
    /// font, and that [`measure_overlay_text`](Self::measure_overlay_text)
    /// measures. The face's glyphs are cut out of its pages here, and a font
    /// that packs glyphs into separate channels is unpacked; the pages are not
    /// kept.
    ///
    /// The face draws at whole multiples of its size only: the text size in
    /// device pixels rounds to the nearest multiple of
    /// [`BitmapFont::size`](crate::BitmapFont::size) (at least one), glyphs
    /// are magnified by that factor with no filtering, and pens land on whole
    /// device pixels, so a label measures the same at every pixels per point
    /// when its size is a whole multiple of the face's. Measuring takes one
    /// pixel per point. Fills, shadows and alignment apply. A face whose
    /// glyphs are white, or grey at their own alpha, takes the fill's colour
    /// like vector text; a face with coloured glyphs draws its colours, with
    /// the fill's alpha, the opacity and the colour matrix applied.
    ///
    /// A glyph run names a bitmap face's glyphs by their index in code point
    /// order. [`font_face`](Self::font_face) answers `None` for a bitmap
    /// face, since it has no font file.
    pub fn upload_bitmap_font(&mut self, font: crate::BitmapFont) -> Result<FontHandle, FontError> {
        self.content.glyph_atlas.upload_bitmap_font(&font)
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
    /// [`LabelItem`]: crate::LabelItem
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
    /// [`LabelItem`]: crate::LabelItem
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
    /// default), if uploaded and not a bitmap face. A downstream text shaper can register these exact
    /// bytes at this face so its glyph ids match what the overlay atlas
    /// rasterizes; the bytes alone name the wrong face for a collection.
    ///
    /// The bytes are the shared buffer the atlas holds, so this does not copy an
    /// uploaded font. The built-in font is copied out of the binary on each call.
    pub fn font_face(&self, font: Option<FontHandle>) -> Option<(Arc<[u8]>, u32)> {
        self.content.glyph_atlas.font_face(font.map_or(0, |h| h.0))
    }
}

/// Lay out `text` in a bitmap face at `font_size` logical pixels, one line
/// per `\n`, and with word wrapping at `max_width` logical pixels when set.
/// The pens are whole device pixels at `ppp`; the result is in logical pixels
/// at the first baseline, as [`GlyphAtlas::shape_text`] returns it.
fn bitmap_shape(
    face: &BitmapFace,
    text: &str,
    font_size: f32,
    ppp: f32,
    max_width: Option<f32>,
) -> ShapedText {
    let k = face.scale_for(font_size * ppp);
    let line_height = (face.line_height * k) as f32;
    let mut glyphs = Vec::new();
    let mut run = Vec::new();
    let (width, height) = if let Some(max_width) = max_width {
        let mut space = Vec::new();
        let space_advance = face.run(" ", k, &mut space);
        let mut words: Vec<Vec<(u16, f32)>> = Vec::new();
        let lines: Vec<Vec<f32>> = text
            .split('\n')
            .map(|line| {
                line.split_whitespace()
                    .map(|word| {
                        run.clear();
                        let w = face.run(word, k, &mut run);
                        words.push(run.clone());
                        w
                    })
                    .collect()
            })
            .collect();
        let (origins, widest, total) =
            wrap_words(&lines, space_advance, max_width * ppp, line_height);
        for (word, o) in words.iter().zip(origins) {
            glyphs.extend(word.iter().map(|&(id, x)| (id, o[0] + x, o[1])));
        }
        (widest, total)
    } else {
        let mut widest: f32 = 0.0;
        let mut pen_y = 0.0;
        for (i, line) in text.split('\n').enumerate() {
            if i > 0 {
                pen_y += line_height;
            }
            run.clear();
            widest = widest.max(face.run(line, k, &mut run));
            glyphs.extend(run.iter().map(|&(id, x)| (id, x, pen_y)));
        }
        (widest, pen_y + line_height)
    };
    let inv = 1.0 / ppp;
    ShapedText {
        glyphs: glyphs
            .into_iter()
            .map(|(id, x, y)| (id, x * inv, y * inv))
            .collect(),
        total_width: width * inv,
        height: height * inv,
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

    /// Up to `EXACT_SIZE_PX` a glyph rasterises at its own size; above it the
    /// raster sizes form a short ladder capped at `MAX_RASTER_PX`.
    #[test]
    fn raster_size_is_exact_when_small_and_bounded_when_large() {
        for px in [0.0_f32, 13.0, 13.37, 44.0, EXACT_SIZE_PX] {
            let r = RasterSize::for_px(px);
            assert_eq!((r.px, r.scale), (px, 1.0), "{px}");
        }
        let mut seen = std::collections::HashSet::new();
        let mut px = EXACT_SIZE_PX + 0.1;
        while px < 5000.0 {
            let r = RasterSize::for_px(px);
            assert!(r.px <= MAX_RASTER_PX && (r.px * r.scale - px).abs() < px * 1e-4);
            if px <= MAX_RASTER_PX {
                assert!(r.scale > 0.97 && r.scale < 1.03, "{px} -> {}", r.px);
            }
            seen.insert(r.tenths);
            px += 0.1;
        }
        assert!(seen.len() < 50, "{} raster sizes", seen.len());
    }

    /// The stepped raster ladder is the one world-sized screen text rounds
    /// its drawn size to, so that text rasterises at its drawn size.
    #[test]
    fn the_stepped_ladder_matches_the_text_size_steps() {
        assert_eq!(SIZE_STEP, viewport_lib_types::overlay::TextSizing::STEP);
        let mut px = 3.0_f32;
        while px < MAX_RASTER_PX {
            let step = viewport_lib_types::overlay::TextSizing::step_px(px);
            // Above the cap the raster is magnified.
            for ppp in [1.0_f32, 1.5, 2.0]
                .into_iter()
                .filter(|_| step <= MAX_RASTER_PX)
            {
                let drawn = (step / ppp) * ppp;
                let r = RasterSize::for_px_stepped(drawn);
                assert_eq!(r.scale, 1.0, "{px} at {ppp}: raster {} for {drawn}", r.px);
            }
            let r = RasterSize::for_px_stepped(px);
            assert!((r.px * r.scale - px).abs() < px * 1e-4 && r.scale > 0.97 && r.scale < 1.03);
            px += 0.13;
        }
        let r = RasterSize::for_px_stepped(1000.0);
        assert_eq!(r.px, RasterSize::for_px(1000.0).px);
    }

    /// Text on a stepped grid zoomed smoothly through 2x rasterises each glyph
    /// at one size per ladder step, however many frames the zoom takes.
    #[test]
    fn a_stepped_zoom_rasterises_a_bounded_set() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let count = |stepped: bool| {
            let mut atlas = GlyphAtlas::new(&device);
            let before = atlas.rasters();
            for i in 0..=400 {
                atlas.begin_frame();
                let px = 14.0 * 2.0_f32.powf(i as f32 / 400.0);
                let shaped = atlas.shape_text("0123456789", px, None, 1.0);
                let grid = GlyphGrid::screen(1.0).with_stepped(stepped);
                atlas.place_text(
                    &shaped,
                    px,
                    None,
                    [0.0, 0.0],
                    grid,
                    &device,
                    GlyphStyle::PLAIN,
                );
            }
            atlas.rasters() - before
        };
        let stepped = count(true);
        let exact = count(false);
        // About 18 steps through 2x, ten glyphs each; the sub-pixel bins
        // only come in at sizes exactly on a step.
        assert!(stepped <= 19 * 10 * 4, "{stepped} rasters stepped");
        assert!(stepped * 4 < exact, "{stepped} stepped, {exact} exact");
    }

    /// One glyph at 5,000px draws, at about its drawn size, without the atlas
    /// leaving its cap.
    #[test]
    fn one_huge_glyph_draws_within_the_atlas_cap() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        let laid = atlas.layout_text("W", 5000.0, None, 1.0, &device, GlyphStyle::PLAIN);
        atlas.upload_if_dirty(&queue);
        assert_eq!(laid.quads.len(), 1);
        assert!(laid.quads[0].size[1] > 2500.0, "{:?}", laid.quads[0].size);
        assert!(atlas.coverage.size <= atlas.coverage.max_size);
        assert!(atlas.coverage.size <= GlyphAtlas::MAX_SIZE);

        // A shadow cell scales with the raster too, so it stays registered.
        let style = GlyphStyle::from_shadow(0.0, 20.0, 1.0, [0.0, 0.0]);
        let shadow = atlas.layout_text("W", 5000.0, None, 1.0, &device, style);
        assert!(shadow.quads[0].size[0] > laid.quads[0].size[0]);
    }

    /// Text zoomed smoothly from 10px to 200px in tenth-of-a-pixel steps never
    /// grows a page past its cap: a full page evicts plots instead.
    #[test]
    fn smooth_zoom_stays_within_the_atlas_cap() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        // A small cap, so eviction is reached quickly.
        atlas.coverage.max_size = 1024;
        let mut tenths = 100;
        while tenths <= 2000 {
            atlas.begin_frame();
            let laid = atlas.layout_text(
                "Sphinx of black quartz",
                tenths as f32 * 0.1,
                None,
                1.0,
                &device,
                GlyphStyle::PLAIN,
            );
            atlas.upload_if_dirty(&queue);
            assert!(atlas.coverage.size <= 1024);
            assert_eq!(laid.quads.len(), 19, "every glyph drew at {tenths}");
            tenths += 1;
        }
        assert!(atlas.evictions > 0, "the zoom should fill a 1024 page");
    }

    /// Filling a page evicts the least recently used plot rather than
    /// clearing the page: glyphs drawn every frame stay resident, and so does
    /// their atlas version.
    #[test]
    fn a_full_page_evicts_old_plots_and_keeps_the_ones_in_use() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        atlas.coverage.max_size = 1024;
        let kept = "Kept every frame";
        let mut size = 30.0;
        let mut version = None;
        while atlas.evictions == 0 {
            atlas.begin_frame();
            let before = atlas.rasters();
            let laid = atlas.layout_text(kept, 14.0, None, 1.0, &device, GlyphStyle::PLAIN);
            assert_eq!(laid.quads.len(), 14);
            if version.is_some() {
                assert_eq!(atlas.rasters(), before, "the kept text stays cached");
            }
            if atlas.coverage.size == 1024 && version.is_none() {
                version = Some(atlas.version());
            }
            atlas.layout_text("ABCDEFGHIJ", size, None, 1.0, &device, GlyphStyle::PLAIN);
            atlas.upload_if_dirty(&queue);
            size += 0.1;
            assert!(size < 200.0, "the page never filled");
        }
        assert!(!atlas.full);
        atlas.begin_frame();
        let before = atlas.rasters();
        let laid = atlas.layout_text(kept, 14.0, None, 1.0, &device, GlyphStyle::PLAIN);
        assert_eq!(laid.quads.len(), 14);
        assert_eq!(
            atlas.rasters(),
            before,
            "the kept text survived the eviction"
        );
        assert_eq!(
            Some(atlas.version()),
            version,
            "an eviction does not move UVs"
        );
    }

    /// Retained geometry records the plots it drew from, and goes stale when
    /// one of them is evicted but not when an unrelated one is.
    #[test]
    fn glyph_deps_follow_their_plots() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        atlas.coverage.max_size = 1024;
        // Grown to its cap first: growing moves every UV, which is stale too.
        let mut size = 30.0;
        while atlas.coverage.size < 1024 {
            atlas.begin_frame();
            atlas.layout_text("ABCDEFGHIJ", size, None, 1.0, &device, GlyphStyle::PLAIN);
            size += 0.1;
        }
        atlas.begin_frame();
        atlas.begin_recording();
        atlas.layout_text("Retained", 14.0, None, 1.0, &device, GlyphStyle::PLAIN);
        let deps = atlas.finish_recording();
        assert!(!deps.plots.is_empty());
        assert!(atlas.is_current(&deps));

        // Touched every frame, the group's plot is never the one evicted.
        while atlas.evictions == 0 {
            atlas.begin_frame();
            atlas.touch(&deps);
            atlas.layout_text("ABCDEFGHIJ", size, None, 1.0, &device, GlyphStyle::PLAIN);
            size += 0.1;
        }
        assert!(atlas.is_current(&deps), "a touched plot survives");

        // Left untouched, it ages out.
        let mut small = 10.0;
        while atlas.is_current(&deps) {
            atlas.begin_frame();
            atlas.layout_text("ABCDEFGHIJ", small, None, 1.0, &device, GlyphStyle::PLAIN);
            small += 0.1;
            assert!(small < EXACT_SIZE_PX, "the plot was never evicted");
        }
    }

    /// Pens land on whole device pixels across and down, and the quarter of a
    /// pixel each pen was moved from selects the cell it rasterises into.
    #[test]
    fn pens_snap_to_pixels_and_rasterise_at_quarter_offsets() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        let shaped = atlas.shape_text("l", 13.0, None, 1.5);
        let place = |atlas: &mut GlyphAtlas, x: f32| {
            let origin = [x, 20.3];
            let quads = atlas.place_text(
                &shaped,
                13.0,
                None,
                origin,
                GlyphGrid::screen(1.5),
                &device,
                GlyphStyle::PLAIN,
            );
            let q = quads[0];
            // Back to device pixels, as the shader maps it.
            [(origin[0] + q.pos[0]) * 1.5, (origin[1] + q.pos[1]) * 1.5]
        };
        let mut rasters = Vec::new();
        for i in 0..8 {
            let device_px = place(&mut atlas, 10.0 + i as f32 * 0.25 / 1.5);
            for v in device_px {
                assert!(
                    (v - v.round()).abs() < 1e-3,
                    "{device_px:?} is off the grid"
                );
            }
            rasters.push(atlas.rasters());
        }
        // Four new cells for the first four quarters, then the same four again.
        assert_eq!(rasters[3] - rasters[0], 3);
        assert_eq!(rasters[7], rasters[3]);
    }

    /// A uniform item scale rasterises at the scaled size and still snaps; a
    /// rotation rasterises at its scale with no sub-pixel offset.
    #[test]
    fn a_scaled_grid_rasterises_at_the_scaled_size() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        let shaped = atlas.shape_text("H", 14.0, None, 1.0);
        let at = |atlas: &mut GlyphAtlas, grid: GlyphGrid| {
            atlas.place_text(
                &shaped,
                14.0,
                None,
                [3.3, 7.7],
                grid,
                &device,
                GlyphStyle::PLAIN,
            )[0]
        };
        let plain = at(&mut atlas, GlyphGrid::screen(1.0));
        let grid = GlyphGrid::for_transform([[2.0, 0.0], [0.0, 2.0]], [10.0, 10.0], 1.0);
        assert!(grid.snap && grid.scale == 2.0);
        let scaled = at(&mut atlas, grid);
        assert!(atlas.entries.keys().any(|k| k.size_tenths == 280));
        // The same logical box, give or take the pixel the snap moves it.
        assert!((scaled.size[1] - plain.size[1]).abs() <= 1.0);
        // Once the transform is applied, the scaled quad sits on the grid.
        let device_x = 10.0 + 2.0 * (3.3 + scaled.pos[0] - 10.0);
        assert!((device_x - device_x.round()).abs() < 1e-3, "{device_x}");

        let (s, c) = 0.3f32.sin_cos();
        let grid =
            GlyphGrid::for_transform([[2.0 * c, -2.0 * s], [2.0 * s, 2.0 * c]], [0.0; 2], 1.0);
        assert!(!grid.snap && (grid.scale - 2.0).abs() < 1e-4);
        let before: std::collections::HashSet<GlyphKey> = atlas.entries.keys().copied().collect();
        at(&mut atlas, grid);
        let added: Vec<&GlyphKey> = atlas
            .entries
            .keys()
            .filter(|k| !before.contains(k))
            .collect();
        assert!(!added.is_empty());
        assert!(added.iter().all(|k| k.bin == 0 && k.size_tenths == 280));
    }

    /// A cell larger than a plot is skipped, and a cell that fits one packs.
    #[test]
    fn a_cell_larger_than_a_plot_is_skipped() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        let key = |glyph_index| GlyphKey {
            font_index: 0,
            glyph_index,
            size_tenths: 0,
            style: GlyphStyle::PLAIN,
            bin: 0,
        };
        let cell = vec![255u8; 500 * 10];
        let entry = atlas.pack(&device, key(1), Cell::Coverage(&cell), 500, 10, 0.0, 0.0);
        assert_eq!(entry.width, 500);
        assert!(
            entry.x + entry.width <= atlas.coverage.size,
            "packed past its row"
        );

        let wide = vec![255u8; 600 * 10];
        let entry = atlas.pack(&device, key(2), Cell::Coverage(&wide), 600, 10, 0.0, 0.0);
        assert_eq!(entry.width, 0);
        assert!(atlas.empty.contains(&key(2)));
    }

    /// Coverage glyphs pack into the one-byte page and colour glyphs into the
    /// RGBA page, which stays unallocated until a colour glyph needs it.
    #[test]
    fn coverage_and_colour_glyphs_use_their_own_pages() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut atlas = GlyphAtlas::new(&device);
        let laid = atlas.layout_text("Text", 16.0, None, 1.0, &device, GlyphStyle::PLAIN);
        assert!(laid.quads.iter().all(|q| !q.color));
        assert!(atlas.coverage.size > 0);
        assert_eq!(atlas.colour.size, 0);

        let plain = FontRef::from_index(DEFAULT_FONT_BYTES, 0).unwrap();
        let colr = synth_colr_font(plain.charmap().map('A'), plain.charmap().map('O'));
        let font = atlas.upload_font_face(colr.into(), 0).unwrap();
        let laid = atlas.layout_text("A", 32.0, Some(font), 1.0, &device, GlyphStyle::PLAIN);
        assert!(laid.quads[0].color);
        assert!(atlas.colour.size > 0);
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
