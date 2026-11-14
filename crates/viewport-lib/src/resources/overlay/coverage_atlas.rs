//! The coverage atlas: single-channel cells holding the exact coverage of
//! vector paths, which small and retained vector overlay shapes draw their
//! fills from as one textured quad instead of tessellated triangles.
//!
//! It packs with a row packer that grows the page by doubling, and clears the
//! page at the start of the next frame once it is full at its largest. It is a
//! page of its own, so vector shapes never evict text.

use crate::resources::builders::LoggedAlloc;
use std::collections::HashMap;

/// Cache key for a vector path's coverage cell: the path's content (its
/// subpaths and fill rule, hashed), the linear map it was rasterised under in
/// device pixels (scale and rotation, to 1/1024), and its sub-pixel phase (to a
/// quarter pixel). Two draws with the same key are the same pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct VectorKey {
    pub hash: u64,
    pub linear: [i32; 4],
    pub phase: [u8; 2],
}

/// A cell's top-left pixel and size in the page.
#[derive(Debug, Clone, Copy)]
struct Cell {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

pub(crate) struct CoverageAtlas {
    entries: HashMap<VectorKey, Cell>,
    /// CPU copy of the page, one coverage byte per pixel, row-major.
    pixels: Vec<u8>,
    /// Page edge in pixels; zero until the first cell is packed.
    size: u32,
    max_size: u32,
    /// Set when a cell did not fit at `max_size`; cleared by `evict_if_full`.
    full: bool,
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
    pub texture: crate::gpu::Texture,
    pub view: crate::gpu::TextureView,
    /// Rows written since the last upload.
    dirty_rows: Option<(u32, u32)>,
    /// Bumped when the page grows (every UV changes) or is cleared.
    version: u64,
    /// Cells rasterised since the atlas was created.
    rasters: u64,
}

impl CoverageAtlas {
    const INITIAL_SIZE: u32 = 512;
    /// Largest page edge, whatever the device allows.
    const MAX_SIZE: u32 = 2048;

    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let (texture, view) = Self::create_texture(device, 1);
        Self {
            entries: HashMap::new(),
            pixels: Vec::new(),
            size: 0,
            max_size: device.limits().max_texture_dimension_2d.min(Self::MAX_SIZE),
            full: false,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            texture,
            view,
            dirty_rows: None,
            version: 0,
            rasters: 0,
        }
    }

    /// The page generation. Geometry that baked UVs against the page re-emits
    /// when this moves, as glyph geometry does on the glyph atlas's version.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// Cells rasterised since the atlas was created: the cache misses.
    pub(crate) fn rasters(&self) -> u64 {
        self.rasters
    }

    /// The UVs `(min, max)` of the cell for `key`, rasterising it with `raster`
    /// into a zeroed `w * h` buffer (one coverage byte per pixel, row-major) on
    /// a miss. `None` when the cell does not fit: it is bigger than the page
    /// can be, or the page is full until the next `evict_if_full`.
    pub(crate) fn ensure(
        &mut self,
        device: &crate::gpu::Device,
        key: VectorKey,
        w: u32,
        h: u32,
        raster: impl FnOnce(&mut [u8]),
    ) -> Option<([f32; 2], [f32; 2])> {
        let cell = match self.entries.get(&key) {
            Some(&c) => c,
            None => {
                if self.full || w == 0 || h == 0 || w + 1 > self.max_size || h + 1 > self.max_size {
                    return None;
                }
                let (x, y) = self.place(device, w, h)?;
                let mut coverage = vec![0u8; (w * h) as usize];
                raster(&mut coverage);
                self.rasters += 1;
                for row in 0..h {
                    let src = (row * w) as usize;
                    let dst = ((y + row) * self.size + x) as usize;
                    self.pixels[dst..dst + w as usize]
                        .copy_from_slice(&coverage[src..src + w as usize]);
                }
                self.dirty_rows = Some(match self.dirty_rows {
                    Some((start, end)) => (start.min(y), end.max(y + h)),
                    None => (y, y + h),
                });
                let c = Cell { x, y, w, h };
                self.entries.insert(key, c);
                c
            }
        };
        let s = self.size as f32;
        Some((
            [cell.x as f32 / s, cell.y as f32 / s],
            [(cell.x + cell.w) as f32 / s, (cell.y + cell.h) as f32 / s],
        ))
    }

    /// Reserve a `w * h` cell (plus a one-pixel gutter), growing the page if
    /// needed. Sets `full` and returns `None` when it does not fit at the
    /// largest size.
    fn place(&mut self, device: &crate::gpu::Device, w: u32, h: u32) -> Option<(u32, u32)> {
        let pad = 1;
        if self.size == 0 {
            self.grow(device);
        }
        loop {
            if self.cursor_x > 0 && self.cursor_x + w + pad > self.size {
                self.cursor_y += self.row_height + pad;
                self.cursor_x = 0;
                self.row_height = 0;
            }
            if self.cursor_x + w + pad <= self.size && self.cursor_y + h + pad <= self.size {
                break;
            }
            if self.size >= self.max_size {
                self.full = true;
                return None;
            }
            self.grow(device);
        }
        let at = (self.cursor_x, self.cursor_y);
        self.cursor_x += w + pad;
        self.row_height = self.row_height.max(h);
        Some(at)
    }

    /// Upload the rows written since the last upload.
    pub(crate) fn upload_if_dirty(&mut self, queue: &crate::gpu::Queue) {
        let Some((start, end)) = self.dirty_rows.take() else {
            return;
        };
        let row = self.size as usize;
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
            &self.pixels[start as usize * row..end as usize * row],
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size),
                rows_per_image: Some(end - start),
            },
            crate::gpu::Extent3d {
                width: self.size,
                height: end - start,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Clear the page if a cell failed to fit at its largest size. Call once
    /// per frame before any vector shape is emitted; bumps the version.
    pub(crate) fn evict_if_full(&mut self) {
        if !self.full {
            return;
        }
        tracing::info!(
            "Coverage atlas full at {}x{}, clearing {} cached cells",
            self.size,
            self.size,
            self.entries.len()
        );
        self.entries.clear();
        // Zeroed, so the gutters between new cells stay empty under bilinear
        // filtering.
        self.pixels.fill(0);
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.row_height = 0;
        self.full = false;
        self.dirty_rows = Some((0, self.size));
        self.version += 1;
    }

    fn grow(&mut self, device: &crate::gpu::Device) {
        let old = self.size;
        let new = if old == 0 {
            Self::INITIAL_SIZE
        } else {
            old * 2
        }
        .min(self.max_size);
        let mut pixels = vec![0u8; (new * new) as usize];
        for row in 0..old {
            let s = (row * old) as usize;
            let d = (row * new) as usize;
            pixels[d..d + old as usize].copy_from_slice(&self.pixels[s..s + old as usize]);
        }
        self.pixels = pixels;
        self.size = new;
        let (texture, view) = Self::create_texture(device, new);
        self.texture = texture;
        self.view = view;
        self.dirty_rows = Some((0, new));
        self.version += 1;
    }

    fn create_texture(
        device: &crate::gpu::Device,
        size: u32,
    ) -> (crate::gpu::Texture, crate::gpu::TextureView) {
        let texture = device.logged_texture(&crate::gpu::TextureDescriptor {
            label: Some("overlay_coverage_atlas"),
            size: crate::gpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::R8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        (texture, view)
    }
}
