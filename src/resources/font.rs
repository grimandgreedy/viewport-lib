//! Font atlas and single-line text layout for overlay rendering.
//!
//! This module is the text back-end for [`LabelItem`], [`ScalarBarItem`], and
//! [`RulerItem`].  It uses [`fontdue`] for glyph rasterization and packs glyphs
//! into a single GPU texture atlas on demand.
//!
//! Public surface: [`FontHandle`] (opaque font identifier) and
//! [`super::ViewportGpuResources::upload_font`].  Everything else is `pub(crate)`.

use std::collections::HashMap;

/// Default font embedded in the library binary (Inter Regular, SIL OFL 1.1).
const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");

// ---------------------------------------------------------------------------
// FontHandle : public opaque identifier
// ---------------------------------------------------------------------------

/// Opaque handle to a font uploaded via
/// [`ViewportGpuResources::upload_font`](super::ViewportGpuResources::upload_font).
///
/// Pass `None` (or omit the field) on overlay items to use the built-in default
/// font.  Pass `Some(handle)` to use a user-supplied TTF font.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontHandle(pub usize);

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
/// Owned by [`ViewportGpuResources`]; never exposed in the public API.
pub(crate) struct GlyphAtlas {
    /// Parsed fontdue fonts.  Index 0 is always the built-in default.
    fonts: Vec<fontdue::Font>,

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
    pub texture: wgpu::Texture,
    /// View into the atlas texture.
    pub view: wgpu::TextureView,

    /// Set to `true` whenever new glyphs have been rasterized since the last
    /// GPU upload.  Cleared by [`GlyphAtlas::upload_if_dirty`].
    dirty: bool,
}

impl GlyphAtlas {
    /// Initial atlas size in pixels (width = height).
    const INITIAL_SIZE: u32 = 512;

    /// Create a new atlas with the built-in default font pre-loaded.
    pub fn new(device: &wgpu::Device) -> Self {
        let default_font =
            fontdue::Font::from_bytes(DEFAULT_FONT_BYTES, fontdue::FontSettings::default())
                .expect("built-in default font must parse");

        let size = Self::INITIAL_SIZE;
        let pixel_count = (size * size) as usize;
        let pixels = vec![[255, 255, 255, 0]; pixel_count];

        let (texture, view) = Self::create_texture(device, size);

        Self {
            fonts: vec![default_font],
            entries: HashMap::new(),
            pixels,
            size,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            texture,
            view,
            dirty: false,
        }
    }

    /// Register a user-supplied TTF font.  Returns a [`FontHandle`] that can be
    /// passed to overlay items.
    pub fn upload_font(&mut self, ttf_bytes: &[u8]) -> Result<FontHandle, FontError> {
        let font = fontdue::Font::from_bytes(ttf_bytes, fontdue::FontSettings::default())
            .map_err(|e| FontError::ParseFailed(e.to_string()))?;
        let index = self.fonts.len();
        self.fonts.push(font);
        Ok(FontHandle(index))
    }

    /// Lay out a single-line string and return positioned glyph quads.
    ///
    /// Glyphs that are not yet in the atlas are rasterized and packed on the
    /// fly.  Call [`upload_if_dirty`] after all layout calls for the frame to
    /// push new glyphs to the GPU.
    pub fn layout_text(
        &mut self,
        text: &str,
        font_size: f32,
        font: Option<FontHandle>,
        device: &wgpu::Device,
    ) -> TextLayout {
        let font_index = font.map_or(0, |h| h.0);
        let size_tenths = (font_size * 10.0).round() as u32;
        let px = font_size;

        let metrics = self.fonts[font_index].horizontal_line_metrics(px);
        let height = metrics
            .map(|m| m.ascent - m.descent + m.line_gap)
            .unwrap_or(px * 1.2);

        let mut quads = Vec::new();
        let mut pen_x: f32 = 0.0;

        let mut prev_glyph: Option<u16> = None;
        for ch in text.chars() {
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

            // Only emit a quad for glyphs with visible bitmap area.
            if m.width > 0 && m.height > 0 {
                let entry = self.ensure_glyph(device, font_index, glyph_index, size_tenths, px);
                let atlas_size = self.size as f32;

                quads.push(GlyphQuad {
                    pos: [pen_x + entry.offset_x, entry.offset_y],
                    size: [entry.width as f32, entry.height as f32],
                    uv_min: [entry.x as f32 / atlas_size, entry.y as f32 / atlas_size],
                    uv_max: [
                        (entry.x + entry.width) as f32 / atlas_size,
                        (entry.y + entry.height) as f32 / atlas_size,
                    ],
                });
            }

            pen_x += m.advance_width;
        }

        TextLayout {
            quads,
            total_width: pen_x,
            height,
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
        device: &wgpu::Device,
    ) -> TextLayout {
        let font_index = font.map_or(0, |h| h.0);
        let px = font_size;

        let metrics = self.fonts[font_index].horizontal_line_metrics(px);
        let line_height = metrics
            .map(|m| m.ascent - m.descent + m.line_gap)
            .unwrap_or(px * 1.2);

        // Split into words and lay out greedily.
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return TextLayout {
                quads: Vec::new(),
                total_width: 0.0,
                height: line_height,
            };
        }

        let size_tenths = (px * 10.0).round() as u32;
        let space_advance = {
            let gi = self.fonts[font_index].lookup_glyph_index(' ');
            self.fonts[font_index].metrics_indexed(gi, px).advance_width
        };

        let mut quads = Vec::new();
        let mut line_x: f32 = 0.0;
        let mut line_y: f32 = 0.0;
        let mut max_line_width: f32 = 0.0;
        let mut first_on_line = true;

        for word in &words {
            // Measure the word width.
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
                if m.width > 0 && m.height > 0 {
                    let entry = self.ensure_glyph(device, font_index, glyph_index, size_tenths, px);
                    let atlas_size = self.size as f32;
                    word_quads.push(GlyphQuad {
                        pos: [pen_x + entry.offset_x, entry.offset_y],
                        size: [entry.width as f32, entry.height as f32],
                        uv_min: [entry.x as f32 / atlas_size, entry.y as f32 / atlas_size],
                        uv_max: [
                            (entry.x + entry.width) as f32 / atlas_size,
                            (entry.y + entry.height) as f32 / atlas_size,
                        ],
                    });
                }
                pen_x += m.advance_width;
            }
            let word_width = pen_x;

            // Wrap if this word doesn't fit on the current line.
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

        max_line_width = max_line_width.max(line_x);
        let total_height = line_y + line_height;

        TextLayout {
            quads,
            total_width: max_line_width,
            height: total_height,
        }
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

    /// Upload new glyph data to the GPU if any glyphs were rasterized since
    /// the last upload.
    pub fn upload_if_dirty(&mut self, queue: &wgpu::Queue) {
        if !self.dirty {
            return;
        }
        let flat: Vec<u8> = self.pixels.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &flat,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size * 4),
                rows_per_image: Some(self.size),
            },
            wgpu::Extent3d {
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
        device: &wgpu::Device,
        font_index: usize,
        glyph_index: u16,
        size_tenths: u32,
        px: f32,
    ) -> GlyphEntry {
        let key = GlyphKey {
            font_index,
            glyph_index,
            size_tenths,
        };

        if let Some(&entry) = self.entries.get(&key) {
            return entry;
        }

        // Rasterize.
        let (metrics, bitmap) = self.fonts[font_index].rasterize_indexed(glyph_index, px);
        let w = metrics.width as u32;
        let h = metrics.height as u32;

        if w == 0 || h == 0 {
            // Whitespace glyph: insert a zero-area entry.
            let entry = GlyphEntry {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                offset_x: metrics.xmin as f32,
                offset_y: -(metrics.ymin as f32 + h as f32),
            };
            self.entries.insert(key, entry);
            return entry;
        }

        // Pack into the atlas (simple row packer with 1px padding).
        let pad = 1;
        if self.cursor_x + w + pad > self.size {
            // Move to next row.
            self.cursor_y += self.row_height + pad;
            self.cursor_x = 0;
            self.row_height = 0;
        }
        if self.cursor_y + h + pad > self.size {
            // Atlas is full : grow.
            self.grow(device);
        }

        let x = self.cursor_x;
        let y = self.cursor_y;

        // Blit coverage into the RGBA pixel buffer.
        for row in 0..h {
            for col in 0..w {
                let src = (row * w + col) as usize;
                let dst = ((y + row) * self.size + (x + col)) as usize;
                self.pixels[dst] = [255, 255, 255, bitmap[src]];
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
            offset_x: metrics.xmin as f32,
            offset_y: -(metrics.ymin as f32 + h as f32),
        };
        self.entries.insert(key, entry);
        entry
    }

    /// Double the atlas size, copying existing pixel data into the new buffer
    /// and recreating the GPU texture.
    fn grow(&mut self, device: &wgpu::Device) {
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
    }

    fn create_texture(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glyph_atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

// ---------------------------------------------------------------------------
// FontError
// ---------------------------------------------------------------------------

/// Error returned by [`super::ViewportGpuResources::upload_font`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum FontError {
    /// The TTF data could not be parsed.
    #[error("font parsing failed: {0}")]
    ParseFailed(String),
}

// ---------------------------------------------------------------------------
// ViewportGpuResources integration
// ---------------------------------------------------------------------------

impl super::ViewportGpuResources {
    /// Upload a user-supplied TTF font for use with overlay items.
    ///
    /// Returns an opaque [`FontHandle`] that can be passed to [`LabelItem`],
    /// [`ScalarBarItem`], or [`RulerItem`] via their `font` field.  Pass
    /// `None` on those items to use the built-in default font instead.
    ///
    /// The font bytes must be a valid TrueType (`.ttf`) file.
    pub fn upload_font(&mut self, ttf_bytes: &[u8]) -> Result<FontHandle, FontError> {
        self.glyph_atlas.upload_font(ttf_bytes)
    }
}
