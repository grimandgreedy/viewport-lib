//! The tilemaps this item type holds on the consumer's behalf: the CPU copy of
//! each map, one instance buffer per occupied chunk, and the tileset bind
//! group, plus the per-layer uniform every frame writes.

use super::types::{Tile, TileChunk, Tilemap, TilemapDraw, TilemapId, TilemapItem, TilemapPlane};
use viewport_lib::resources::DeviceResources;

/// Bytes per tile instance: cell, tile, flags and tint, four bytes each.
pub(crate) const INSTANCE_STRIDE: u64 = std::mem::size_of::<GpuTile>() as u64;

/// One occupied cell as the vertex stream carries it. Matches `TileIn` in
/// `tilemap_common.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuTile {
    /// Column in the low 16 bits, row in the high 16.
    cell: u32,
    /// Tileset index, or the animation slot when [`FLAG_ANIMATED`] is set.
    tile: u32,
    /// Flip bits (horizontal 1, vertical 2, diagonal 4) and [`FLAG_ANIMATED`].
    flags: u32,
    /// Linear RGBA tint, one byte a channel.
    tint: [u8; 4],
}

/// [`GpuTile::flags`] bit: `tile` names an animation slot.
const FLAG_ANIMATED: u32 = 1 << 3;

/// The tileset uniform, matching `TilesetUniform` in `tilemap_common.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TilesetUniformData {
    tile_size: [f32; 2],
    columns: u32,
    has_texture: u32,
    spacing: f32,
    margin: f32,
    _pad: [u32; 2],
}

/// Animation slots per layer uniform, four to a `vec4<u32>`.
const ANIM_SLOTS: usize = super::types::MAX_TILE_ANIMATIONS;

/// The per-layer uniform, matching `LayerUniform` in `tilemap_common.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LayerUniformData {
    translation: [f32; 2],
    cell_size: [f32; 2],
    follow: [f32; 2],
    elevation: f32,
    plane: u32,
    tint: [f32; 4],
    alpha_threshold: f32,
    flags: u32,
    pick_id: u32,
    /// The painter mask test word, read by a blended layer.
    mask: u32,
    /// The blended layer's painter layer, then padding.
    painter_layer: [u32; 4],
    anim: [u32; ANIM_SLOTS],
    /// A tile material's parameter lanes, lane 0 reserved.
    pub(crate) lanes: [[f32; 4]; 8],
}

/// Bytes between layers in the frame's uniform buffer: the uniform rounded up
/// to the dynamic offset alignment every device accepts.
pub(crate) const LAYER_STRIDE: u64 =
    (std::mem::size_of::<LayerUniformData>() as u64).div_ceil(256) * 256;

/// The two bind group layouts tilemap draws use.
pub(crate) struct TilemapLayouts {
    /// Group 1: tileset uniform, atlas, sampler.
    pub(crate) map: viewport_lib::gpu::BindGroupLayout,
    /// Group 2: the layer uniform at a dynamic offset.
    pub(crate) layer: viewport_lib::gpu::BindGroupLayout,
}

impl TilemapLayouts {
    pub(crate) fn new(device: &viewport_lib::gpu::Device) -> Self {
        use viewport_lib::gpu;
        let map = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("tilemap_bgl"),
            entries: &[
                gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Texture {
                        sample_type: gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Sampler(gpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layer = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("tilemap_layer_bgl"),
            entries: &[gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT,
                ty: gpu::BindingType::Buffer {
                    ty: gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<LayerUniformData>() as u64,
                    ),
                },
                count: None,
            }],
        });
        Self { map, layer }
    }
}

/// The renderer-owned handles a tileset binds, resolved before the build so
/// the buffer work can run on a worker thread.
#[derive(Clone)]
pub(crate) struct TilesetBindings {
    view: viewport_lib::gpu::TextureView,
    has_texture: u32,
    sampler: viewport_lib::gpu::Sampler,
    layout: viewport_lib::gpu::BindGroupLayout,
}

/// Resolve the atlas view and sampler `map`'s tileset names. A missing or
/// stale texture binds the neutral fallback and draws untextured.
pub(crate) fn resolve_bindings(
    device: &viewport_lib::gpu::Device,
    resources: &DeviceResources,
    layouts: &TilemapLayouts,
    texture_id: Option<viewport_lib::resources::TextureId>,
    filter: viewport_lib::TextureFilter,
    report: bool,
) -> TilesetBindings {
    use viewport_lib::resources::TextureRequirement;
    if report {
        resources.check_texture_slot(
            texture_id,
            TextureRequirement::colour("Tileset::texture_id"),
        );
    }
    let (view, has_texture) = match texture_id.and_then(|id| resources.texture_view(id)) {
        Some(view) => (view.clone(), 1),
        None => (resources.fallback_colourmap_view().clone(), 0),
    };
    let sampler = match filter {
        viewport_lib::TextureFilter::Linear => resources.material_sampler().clone(),
        // Clamped, so a tile at the atlas edge never samples the far side.
        viewport_lib::TextureFilter::Nearest => resources.sampler_for(
            device,
            viewport_lib::SamplerKey {
                wrap_u: viewport_lib::WrapMode::ClampToEdge,
                wrap_v: viewport_lib::WrapMode::ClampToEdge,
                filter: viewport_lib::TextureFilter::Nearest,
                ..Default::default()
            },
        ),
    };
    TilesetBindings {
        view,
        has_texture,
        sampler,
        layout: layouts.map.clone(),
    }
}

/// One chunk's instance buffer. Allocated for the whole chunk the first time
/// it holds a tile, so later edits only write.
pub(crate) struct StoredChunk {
    pub(crate) buffer: Option<viewport_lib::gpu::Buffer>,
    /// Instances that draw: the chunk's occupied cells.
    pub(crate) count: u32,
}

/// One uploaded map: its CPU copy, its chunk buffers and its tileset binding.
pub(crate) struct StoredTilemap {
    pub(crate) map: Tilemap,
    pub(crate) chunks: Vec<StoredChunk>,
    pub(crate) bind_group: viewport_lib::gpu::BindGroup,
    uniform: TilesetUniformData,
    uniform_buf: viewport_lib::gpu::Buffer,
    /// For each tileset tile index that animates, its slot.
    anim_slots: std::collections::HashMap<u32, u32>,
    /// Chunk instance bytes written since the plugin last read it.
    pub(crate) uploaded_bytes: u64,
    /// Chunk uploads over the map's lifetime.
    pub(crate) chunk_uploads: u64,
}

impl viewport_lib::resources::handle::GpuByteSize for StoredTilemap {
    fn gpu_bytes(&self) -> u64 {
        self.chunks
            .iter()
            .filter_map(|c| c.buffer.as_ref())
            .map(|b| b.size())
            .sum::<u64>()
            + self.uniform_buf.size()
    }
}

/// Slotted store of uploaded tilemaps.
pub(crate) type TilemapStore = viewport_lib::resources::handle::SlotStore<StoredTilemap, TilemapId>;

fn tint_bytes(c: viewport_lib::Colour) -> [u8; 4] {
    c.to_linear_rgba()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// The instances of `chunk`, one per occupied cell in row order.
pub(crate) fn chunk_instances(
    map: &Tilemap,
    anim_slots: &std::collections::HashMap<u32, u32>,
    chunk_index: u32,
    chunk: &TileChunk,
) -> Vec<GpuTile> {
    let [col0, row0, _, _] = map.chunk_cells(chunk_index);
    let cs = map.chunk_size();
    chunk
        .tiles
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.is_empty())
        .filter_map(|(i, t)| {
            let (col, row) = (col0 + i as u32 % cs, row0 + i as u32 / cs);
            let [w, h] = map.size();
            (col < w && row < h).then(|| gpu_tile(anim_slots, col, row, t))
        })
        .collect()
}

fn gpu_tile(
    anim_slots: &std::collections::HashMap<u32, u32>,
    col: u32,
    row: u32,
    t: &Tile,
) -> GpuTile {
    let (tile, animated) = match anim_slots.get(&t.index) {
        Some(&slot) => (slot, FLAG_ANIMATED),
        None => (t.index, 0),
    };
    GpuTile {
        cell: (col & 0xffff) | (row << 16),
        tile,
        flags: t.flip.bits() | animated,
        tint: tint_bytes(t.tint),
    }
}

impl StoredTilemap {
    /// Build every occupied chunk's buffer and the tileset binding for `map`.
    pub(crate) fn build(
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        binds: &TilesetBindings,
        map: Tilemap,
    ) -> Self {
        let set = &map.tileset;
        let anim_slots = (0..set.animations.len().min(ANIM_SLOTS))
            .filter_map(|slot| {
                let tile = set.animations[slot].tile;
                // The first entry naming a tile wins.
                (set.animation_of(tile) == Some(slot)).then_some((tile, slot as u32))
            })
            .collect();
        let uniform = TilesetUniformData {
            tile_size: set.tile_size.map(|s| s as f32),
            columns: set.columns.max(1),
            has_texture: binds.has_texture,
            spacing: set.spacing as f32,
            margin: set.margin as f32,
            _pad: [0; 2],
        };
        let uniform_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("tileset_uniform_buf"),
            size: std::mem::size_of::<TilesetUniformData>() as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform));
        let bind_group = build_bind_group(device, binds, &uniform_buf);
        let chunks = map
            .chunks()
            .iter()
            .map(|_| StoredChunk {
                buffer: None,
                count: 0,
            })
            .collect();
        let mut stored = Self {
            map,
            chunks,
            bind_group,
            uniform,
            uniform_buf,
            anim_slots,
            uploaded_bytes: 0,
            chunk_uploads: 0,
        };
        for chunk in 0..stored.chunks.len() as u32 {
            stored.upload_chunk(device, queue, chunk);
        }
        stored
    }

    /// Rewrite chunk `chunk`'s instances from the CPU copy, allocating its
    /// buffer the first time it holds a tile.
    pub(crate) fn upload_chunk(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        chunk: u32,
    ) {
        let instances = chunk_instances(
            &self.map,
            &self.anim_slots,
            chunk,
            &self.map.chunks()[chunk as usize],
        );
        let stored = &mut self.chunks[chunk as usize];
        stored.count = instances.len() as u32;
        if instances.is_empty() {
            return;
        }
        let cs = self.map.chunk_size() as u64;
        let buffer = stored.buffer.get_or_insert_with(|| {
            device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                label: Some("tilemap_chunk_buf"),
                size: cs * cs * INSTANCE_STRIDE,
                usage: viewport_lib::gpu::BufferUsages::VERTEX
                    | viewport_lib::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        queue.write_buffer(buffer, 0, bytes);
        self.uploaded_bytes += bytes.len() as u64;
        self.chunk_uploads += 1;
    }

    /// The tile each animation slot shows at `time`.
    pub(crate) fn animation_frames(&self, time: f32) -> [u32; ANIM_SLOTS] {
        let mut out = [0u32; ANIM_SLOTS];
        for (&tile, &slot) in &self.anim_slots {
            if let Some(anim) = self.map.tileset.animations.get(slot as usize) {
                out[slot as usize] = anim.frame_at(time);
            } else {
                out[slot as usize] = tile;
            }
        }
        out
    }

    /// Rebind against freshly resolved views, keeping the chunk buffers.
    pub(crate) fn rebind(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        binds: &TilesetBindings,
    ) {
        self.uniform.has_texture = binds.has_texture;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniform));
        self.bind_group = build_bind_group(device, binds, &self.uniform_buf);
    }
}

fn build_bind_group(
    device: &viewport_lib::gpu::Device,
    binds: &TilesetBindings,
    uniform_buf: &viewport_lib::gpu::Buffer,
) -> viewport_lib::gpu::BindGroup {
    use viewport_lib::gpu;
    device.create_bind_group(&gpu::BindGroupDescriptor {
        label: Some("tilemap_bind_group"),
        layout: &binds.layout,
        entries: &[
            gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            gpu::BindGroupEntry {
                binding: 1,
                resource: gpu::BindingResource::TextureView(&binds.view),
            },
            gpu::BindGroupEntry {
                binding: 2,
                resource: gpu::BindingResource::Sampler(&binds.sampler),
            },
        ],
    })
}

/// The layer uniform for one drawn item against `stored`, relative to
/// `origin`, the frame's world origin. `frame_seconds` is the frame clock,
/// read when the item animates on it.
pub(crate) fn layer_uniform(
    item: &TilemapItem,
    stored: &StoredTilemap,
    origin: glam::DVec3,
    frame_seconds: f64,
) -> LayerUniformData {
    let plane = item.plane;
    // The layer at world `p` draws at `p - origin` relative to the origin;
    // the shader adds `eye_rel * (1 - parallax)`, and the absolute eye is
    // `eye_rel + origin`, so the origin's share of the parallax shift,
    // `origin * (1 - parallax)`, is folded in here in f64.
    let o = plane.to_plane(origin);
    let parallax = glam::DVec2::new(item.parallax[0] as f64, item.parallax[1] as f64);
    let translation = glam::DVec2::new(item.origin[0] as f64, item.origin[1] as f64) - o * parallax;
    let elevation = item.elevation as f64 - plane.elevation_of(origin);
    let (alpha_threshold, mask, painter_layer) = match item.draw {
        TilemapDraw::Cutout { alpha_threshold } => (alpha_threshold, 0, 0),
        TilemapDraw::Blended { layer, .. } => (0.0, item.mask.word(), layer),
    };
    LayerUniformData {
        translation: translation.as_vec2().to_array(),
        cell_size: item.cell_size,
        follow: [1.0 - item.parallax[0], 1.0 - item.parallax[1]],
        elevation: elevation as f32,
        plane: match plane {
            TilemapPlane::Xy => 0,
            TilemapPlane::Xz => 1,
        },
        tint: item.tint.to_linear_rgba(),
        alpha_threshold,
        flags: item.pixel_snap as u32,
        pick_id: item.settings.pick_id.0 as u32,
        mask,
        painter_layer: [painter_layer, 0, 0, 0],
        anim: stored.animation_frames(if item.use_frame_clock {
            frame_seconds as f32
        } else {
            item.time
        }),
        lanes: [[0.0; 4]; 8],
    }
}

/// The world-space box chunk `chunk` of `stored` covers when `item` draws it
/// for a camera at `eye`, as `(min, max)`.
pub(crate) fn chunk_bounds(
    item: &TilemapItem,
    map: &Tilemap,
    chunk: u32,
    eye: glam::DVec3,
) -> (glam::Vec3, glam::Vec3) {
    let plane = item.plane;
    let shift = item.drawn_origin(plane.to_plane(eye));
    let [c0, r0, c1, r1] = map.chunk_cells(chunk);
    let cell = glam::DVec2::new(item.cell_size[0] as f64, item.cell_size[1] as f64);
    let a = shift + glam::DVec2::new(c0 as f64, r0 as f64) * cell;
    let b = shift + glam::DVec2::new(c1 as f64, r1 as f64) * cell;
    let e = item.elevation as f64;
    let p = plane.to_world(a, e).as_vec3();
    let q = plane.to_world(b, e).as_vec3();
    (p.min(q), p.max(q))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sizes_match_the_shader() {
        assert_eq!(INSTANCE_STRIDE, 16);
        assert_eq!(std::mem::size_of::<TilesetUniformData>(), 32);
        assert_eq!(
            std::mem::size_of::<LayerUniformData>(),
            80 + 4 * ANIM_SLOTS + 128
        );
        assert_eq!(LAYER_STRIDE % 256, 0);
    }

    #[test]
    fn animated_tiles_carry_their_slot() {
        use super::super::types::{TileAnimation, TileFlip, TileFrame, Tileset};
        let mut set = Tileset::new(None, [8, 8], 4);
        set.animations.push(TileAnimation {
            tile: 5,
            frames: vec![TileFrame {
                tile: 6,
                duration: 1.0,
            }],
        });
        let mut map = Tilemap::with_chunk_size(set, [4, 4], 4);
        map.set(1, 2, Tile::new(5).with_flip(TileFlip::ROTATE_90));
        map.set(3, 3, Tile::new(2));
        let slots = [(5u32, 0u32)].into_iter().collect();
        let tiles = chunk_instances(&map, &slots, 0, &map.chunks()[0]);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].cell, 1 | (2 << 16));
        assert_eq!(tiles[0].tile, 0);
        assert_eq!(tiles[0].flags, TileFlip::ROTATE_90.bits() | FLAG_ANIMATED);
        assert_eq!(tiles[1].tile, 2);
        assert_eq!(tiles[1].tint, [255; 4]);
    }
}
