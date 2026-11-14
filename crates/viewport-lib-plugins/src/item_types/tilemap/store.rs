//! The tilemaps this item type holds on the consumer's behalf: the CPU copy of
//! each map, one instance buffer per occupied chunk, and the tileset bind
//! group, plus the per-layer uniform every frame writes.

use super::grid::{GridKind, GridLayout};
use super::types::{
    MAX_TILE_SIZES, Tile, TileChunk, Tilemap, TilemapDraw, TilemapId, TilemapItem, TilemapPlane,
};
use viewport_lib::resources::DeviceResources;

/// Bytes per tile instance: cell, tile, flags and tint, four bytes each.
pub(crate) const INSTANCE_STRIDE: u64 = std::mem::size_of::<GpuTile>() as u64;

/// One occupied cell as the vertex stream carries it. Matches `TileIn` in
/// `tilemap_common.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuTile {
    /// Column in the low 16 bits, row in the high 16, each two's complement.
    cell: u32,
    /// Tileset index, or the animation slot when [`FLAG_ANIMATED`] is set.
    tile: u32,
    /// Flip bits (horizontal 1, vertical 2, diagonal 4), [`FLAG_ANIMATED`],
    /// [`FLAG_ROTATE_120`], and in bits 8 to 15 the tile's size slot, `0`
    /// for one cell.
    flags: u32,
    /// Linear RGBA tint, one byte a channel.
    tint: [u8; 4],
}

/// [`GpuTile::flags`] bit: `tile` names an animation slot.
const FLAG_ANIMATED: u32 = 1 << 3;

/// [`GpuTile::flags`] bit: the tile turns 120 degrees on a hexagonal grid.
const FLAG_ROTATE_120: u32 = 1 << 4;

/// [`GpuTile::flags`] shift of the size slot.
const SIZE_SLOT_SHIFT: u32 = 8;

/// A cell as the vertex stream and the pick channel carry it.
pub(crate) fn pack_cell(col: i32, row: i32) -> u32 {
    (col as u16 as u32) | ((row as u16 as u32) << 16)
}

/// The cell `pack_cell` packed.
pub(crate) fn unpack_cell(cell: u32) -> [i32; 2] {
    [
        (cell & 0xffff) as u16 as i16 as i32,
        (cell >> 16) as u16 as i16 as i32,
    ]
}

/// The tileset uniform, matching `TilesetUniform` in `tilemap_common.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TilesetUniformData {
    tile_size: [f32; 2],
    columns: u32,
    has_texture: u32,
    spacing: f32,
    margin: f32,
    /// [`GridLayout::word`]: the grid kind, its parity and its turns.
    orientation: u32,
    has_normal_atlas: u32,
    /// [`GridLayout::grid`]: a staggered grid's steps and shift, in cells.
    grid: [f32; 4],
    /// Per size slot: the image size in texels, then the anchor in cells.
    /// Slot 0 is unused, so a zero slot draws one cell.
    sizes: [[f32; 4]; SIZE_SLOTS],
}

/// Size slots in the tileset uniform: one per override and the unused zero.
const SIZE_SLOTS: usize = MAX_TILE_SIZES + 1;

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
    /// The blended layer's painter layer, the sprite deformers the layer
    /// runs, then padding.
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
    /// Group 1: tileset uniform, atlas, sampler, normal atlas, user values.
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
                gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Texture {
                        sample_type: gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: gpu::ShaderStages::VERTEX,
                    ty: gpu::BindingType::Texture {
                        sample_type: gpu::TextureSampleType::Uint,
                        view_dimension: gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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
    normal_view: viewport_lib::gpu::TextureView,
    has_normal_atlas: u32,
    sampler: viewport_lib::gpu::Sampler,
    layout: viewport_lib::gpu::BindGroupLayout,
}

/// Resolve the atlas and normal atlas views and the sampler `map`'s tileset
/// names. A missing or stale atlas binds the neutral fallback and draws
/// untextured; a missing normal atlas lights the tiles flat.
pub(crate) fn resolve_bindings(
    device: &viewport_lib::gpu::Device,
    resources: &DeviceResources,
    layouts: &TilemapLayouts,
    texture_id: Option<viewport_lib::resources::TextureId>,
    normal_id: Option<viewport_lib::resources::TextureId>,
    filter: viewport_lib::TextureFilter,
    report: bool,
) -> TilesetBindings {
    use viewport_lib::resources::TextureRequirement;
    if report {
        resources.check_texture_slot(
            texture_id,
            TextureRequirement::colour("Tileset::texture_id"),
        );
        resources.check_texture_slot(
            normal_id,
            TextureRequirement::normal_map("Tileset::normal_atlas"),
        );
    }
    let (normal_view, has_normal_atlas) = match normal_id.and_then(|id| resources.texture_view(id))
    {
        Some(view) => (view.clone(), 1),
        // Never read with the flag clear; the colourmap fallback exists on
        // every tier.
        None => (resources.fallback_colourmap_view().clone(), 0),
    };
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
        normal_view,
        has_normal_atlas,
        sampler,
        layout: layouts.map.clone(),
    }
}

/// The key chunks are stored under: `(cy, cx)`, so iteration runs row by row
/// from the bottom, as the map's own chunks do.
pub(crate) type ChunkKey = (i32, i32);

/// A run of a chunk's instances whose cells' bounding boxes share a bottom
/// edge: the edge's plane y in cells, the first instance and how many.
pub(crate) type RowRun = (f64, u32, u32);

/// One chunk's instance buffer, sized to its occupied cells.
pub(crate) struct StoredChunk {
    pub(crate) buffer: Option<viewport_lib::gpu::Buffer>,
    /// Instances that draw: the chunk's occupied cells.
    pub(crate) count: u32,
    /// Runs of instances standing on one line, in draw order.
    pub(crate) rows: Vec<RowRun>,
    /// One bit per cell of the chunk that holds a tile, in cell order, for
    /// CPU picks that run without the CPU copy.
    pub(crate) occupied: Vec<u64>,
}

impl StoredChunk {
    /// Whether cell `tile` of the chunk holds a tile.
    pub(crate) fn holds(&self, tile: u32) -> bool {
        self.occupied
            .get(tile as usize / 64)
            .is_some_and(|w| w & (1 << (tile % 64)) != 0)
    }
}

/// One uploaded map: its CPU copy, its chunk buffers and its tileset binding.
pub(crate) struct StoredTilemap {
    pub(crate) map: Tilemap,
    pub(crate) chunks: std::collections::BTreeMap<ChunkKey, StoredChunk>,
    pub(crate) bind_group: viewport_lib::gpu::BindGroup,
    uniform: TilesetUniformData,
    uniform_buf: viewport_lib::gpu::Buffer,
    /// [`Tileset::user_values`](super::types::Tileset::user_values) as the
    /// vertex hook reads them.
    user_table: viewport_lib::gpu::TextureView,
    /// For each tileset tile index that animates, its slot.
    anim_slots: std::collections::HashMap<u32, u32>,
    /// For each tileset tile index with its own size, its size slot.
    size_slots: std::collections::HashMap<u32, u32>,
    /// How far tiles reach past their cells, from [`Tileset::overhang`].
    ///
    /// [`Tileset::overhang`]: super::types::Tileset::overhang
    pub(crate) overhang: [f32; 4],
    /// Where the tileset's grid puts each cell.
    pub(crate) layout: GridLayout,
    /// Chunk instance bytes written since the plugin last read it.
    pub(crate) uploaded_bytes: u64,
    /// Chunk uploads over the map's lifetime.
    pub(crate) chunk_uploads: u64,
}

impl viewport_lib::resources::handle::GpuByteSize for StoredTilemap {
    fn gpu_bytes(&self) -> u64 {
        self.chunk_buffer_bytes() + self.uniform_buf.size()
    }
}

/// Slotted store of uploaded tilemaps.
pub(crate) type TilemapStore = viewport_lib::resources::handle::SlotStore<StoredTilemap, TilemapId>;

fn tint_bytes(c: viewport_lib::Colour) -> [u8; 4] {
    c.to_linear_rgba()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// The instances of the chunk at `coord`, one per occupied cell, and the
/// runs of them whose cells stand on one line.
///
/// An orthogonal chunk runs row by row from the bottom. The other grids run
/// from the top of the plane down, left to right along each line, as Tiled
/// draws them, so a cell lower on the plane covers the one behind it.
pub(crate) fn chunk_instances(
    map: &Tilemap,
    anim_slots: &std::collections::HashMap<u32, u32>,
    size_slots: &std::collections::HashMap<u32, u32>,
    coord: [i32; 2],
    chunk: &TileChunk,
) -> (Vec<GpuTile>, Vec<RowRun>) {
    let layout = GridLayout::new(&map.tileset);
    let mut cells: Vec<(glam::DVec2, i32, i32, &Tile)> = Vec::new();
    for (i, t) in chunk.tiles.iter().enumerate() {
        if t.is_empty() {
            continue;
        }
        let Some([col, row]) = map.cell_of(coord, i as u32) else {
            continue;
        };
        cells.push((layout.origin(col, row), col, row, t));
    }
    if layout.kind != GridKind::Orthogonal {
        cells.sort_by(|a, b| b.0.y.total_cmp(&a.0.y).then(a.0.x.total_cmp(&b.0.x)));
    }
    let mut tiles = Vec::with_capacity(cells.len());
    let mut rows: Vec<RowRun> = Vec::new();
    for (origin, col, row, t) in cells {
        match rows.last_mut() {
            Some(run) if run.0 == origin.y => run.2 += 1,
            _ => rows.push((origin.y, tiles.len() as u32, 1)),
        }
        tiles.push(gpu_tile(anim_slots, size_slots, col, row, t));
    }
    (tiles, rows)
}

fn gpu_tile(
    anim_slots: &std::collections::HashMap<u32, u32>,
    size_slots: &std::collections::HashMap<u32, u32>,
    col: i32,
    row: i32,
    t: &Tile,
) -> GpuTile {
    let (tile, animated) = match anim_slots.get(&t.index) {
        Some(&slot) => (slot, FLAG_ANIMATED),
        None => (t.index, 0),
    };
    let size = size_slots.get(&t.index).copied().unwrap_or(0);
    GpuTile {
        cell: pack_cell(col, row),
        tile,
        flags: t.flip.bits()
            | animated
            | if t.rotate_120 { FLAG_ROTATE_120 } else { 0 }
            | size << SIZE_SLOT_SHIFT,
        tint: tint_bytes(t.tint),
    }
}

impl StoredTilemap {
    /// Build every occupied chunk's buffer and the tileset binding for `map`.
    /// A map that keeps no CPU copy drops its cells once they are uploaded.
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
        let mut sizes = [[0.0f32; 4]; SIZE_SLOTS];
        let mut size_slots = std::collections::HashMap::new();
        for (i, s) in set.tile_sizes.iter().take(MAX_TILE_SIZES).enumerate() {
            if set.size_of(s.tile) == Some(i) {
                let slot = size_slots.len() + 1;
                sizes[slot] = [s.size[0] as f32, s.size[1] as f32, s.anchor[0], s.anchor[1]];
                size_slots.insert(s.tile, slot as u32);
            }
        }
        let layout = GridLayout::new(set);
        let uniform = TilesetUniformData {
            tile_size: set.tile_size.map(|s| s as f32),
            columns: set.columns.max(1),
            has_texture: binds.has_texture,
            spacing: set.spacing as f32,
            margin: set.margin as f32,
            orientation: layout.word(),
            has_normal_atlas: binds.has_normal_atlas,
            grid: layout.grid(),
            sizes,
        };
        let overhang = set.overhang();
        let uniform_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("tileset_uniform_buf"),
            size: std::mem::size_of::<TilesetUniformData>() as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform));
        let user_table = user_table(device, queue, &set.user_values);
        let bind_group = build_bind_group(device, binds, &uniform_buf, &user_table);
        let keys: Vec<[i32; 2]> = map.chunks().map(|(c, _)| c).collect();
        let mut stored = Self {
            map,
            chunks: Default::default(),
            bind_group,
            uniform,
            uniform_buf,
            user_table,
            anim_slots,
            size_slots,
            overhang,
            layout,
            uploaded_bytes: 0,
            chunk_uploads: 0,
        };
        for coord in keys {
            stored.upload_chunk(device, queue, coord);
        }
        if !stored.map.keep_cpu_copy {
            stored.map.clear_chunks();
        }
        stored
    }

    /// Rewrite the chunk at `coord`'s instances from the CPU copy, growing
    /// its buffer when it holds more tiles than it has room for and freeing
    /// it when the copy holds no tile there.
    pub(crate) fn upload_chunk(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        coord: [i32; 2],
    ) {
        let Some(chunk) = self.map.chunk(coord) else {
            self.chunks.remove(&(coord[1], coord[0]));
            return;
        };
        let (instances, rows) =
            chunk_instances(&self.map, &self.anim_slots, &self.size_slots, coord, chunk);
        let occupied = occupancy(chunk);
        self.write_chunk(device, queue, coord, instances, rows, occupied);
    }

    /// Replace the chunk at `coord` with `chunk` on the GPU, and in the CPU
    /// copy when the map keeps one.
    pub(crate) fn replace_chunk(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        coord: [i32; 2],
        chunk: TileChunk,
    ) -> bool {
        if self.map.keep_cpu_copy {
            if !self.map.set_chunk(coord, chunk) {
                return false;
            }
            self.upload_chunk(device, queue, coord);
            return true;
        }
        // Without the copy, the chunk is checked and sized through a
        // throwaway map holding only it.
        let mut scratch = Tilemap::sparse(self.map.tileset.clone(), self.map.chunk_size());
        if !scratch.set_chunk(coord, chunk) {
            return false;
        }
        let chunk = scratch.chunk(coord).expect("just set");
        let (instances, rows) =
            chunk_instances(&scratch, &self.anim_slots, &self.size_slots, coord, chunk);
        let occupied = occupancy(chunk);
        self.write_chunk(device, queue, coord, instances, rows, occupied);
        true
    }

    /// Drop the chunk at `coord` and free its buffer. `false` when the map
    /// had no chunk there.
    pub(crate) fn remove_chunk(&mut self, coord: [i32; 2]) -> bool {
        let in_copy = self.map.remove_chunk(coord).is_some();
        let on_gpu = self.chunks.remove(&(coord[1], coord[0])).is_some();
        in_copy || on_gpu
    }

    fn write_chunk(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        coord: [i32; 2],
        instances: Vec<GpuTile>,
        rows: Vec<RowRun>,
        occupied: Vec<u64>,
    ) {
        let key = (coord[1], coord[0]);
        if instances.is_empty() {
            self.chunks.remove(&key);
            return;
        }
        let stored = self.chunks.entry(key).or_insert_with(|| StoredChunk {
            buffer: None,
            count: 0,
            rows: Vec::new(),
            occupied: Vec::new(),
        });
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        if stored
            .buffer
            .as_ref()
            .is_none_or(|b| b.size() < bytes.len() as u64)
        {
            stored.buffer = Some(device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                label: Some("tilemap_chunk_buf"),
                size: bytes.len() as u64,
                usage: viewport_lib::gpu::BufferUsages::VERTEX
                    | viewport_lib::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        queue.write_buffer(stored.buffer.as_ref().expect("allocated"), 0, bytes);
        stored.count = instances.len() as u32;
        stored.rows = rows;
        stored.occupied = occupied;
        self.uploaded_bytes += bytes.len() as u64;
        self.chunk_uploads += 1;
    }

    /// Bytes held by the chunk instance buffers.
    pub(crate) fn chunk_buffer_bytes(&self) -> u64 {
        self.chunks
            .values()
            .filter_map(|c| c.buffer.as_ref())
            .map(|b| b.size())
            .sum()
    }

    /// Whether cell (`col`, `row`) holds a tile, from the stored occupancy.
    pub(crate) fn holds(&self, col: i32, row: i32) -> bool {
        self.map
            .locate(col, row)
            .and_then(|(c, tile)| Some(self.chunks.get(&(c[1], c[0]))?.holds(tile)))
            .unwrap_or(false)
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
        self.uniform.has_normal_atlas = binds.has_normal_atlas;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniform));
        self.bind_group = build_bind_group(device, binds, &self.uniform_buf, &self.user_table);
    }
}

/// One bit per cell of `chunk` that holds a tile.
fn occupancy(chunk: &TileChunk) -> Vec<u64> {
    let mut bits = vec![0u64; chunk.tiles.len().div_ceil(64)];
    for (i, t) in chunk.tiles.iter().enumerate() {
        if !t.is_empty() {
            bits[i / 64] |= 1 << (i % 64);
        }
    }
    bits
}

/// Width of the user-value table, in values per row.
const USER_TABLE_WIDTH: u32 = 256;

/// `values` as a single-channel integer texture, [`USER_TABLE_WIDTH`] to a
/// row and zero past the end; one zero texel for an empty list.
fn user_table(
    device: &viewport_lib::gpu::Device,
    queue: &viewport_lib::gpu::Queue,
    values: &[u32],
) -> viewport_lib::gpu::TextureView {
    use viewport_lib::gpu;
    let width = (values.len() as u32).clamp(1, USER_TABLE_WIDTH);
    let height = (values.len() as u32).div_ceil(width).max(1);
    let mut texels = vec![0u32; (width * height) as usize];
    texels[..values.len()].copy_from_slice(values);
    let size = gpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&gpu::TextureDescriptor {
        label: Some("tileset_user_table"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: gpu::TextureDimension::D2,
        format: gpu::TextureFormat::R32Uint,
        usage: gpu::TextureUsages::TEXTURE_BINDING | gpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&texels),
        gpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        size,
    );
    texture.create_view(&Default::default())
}

fn build_bind_group(
    device: &viewport_lib::gpu::Device,
    binds: &TilesetBindings,
    uniform_buf: &viewport_lib::gpu::Buffer,
    user_table: &viewport_lib::gpu::TextureView,
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
            gpu::BindGroupEntry {
                binding: 3,
                resource: gpu::BindingResource::TextureView(&binds.normal_view),
            },
            gpu::BindGroupEntry {
                binding: 4,
                resource: gpu::BindingResource::TextureView(user_table),
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
        painter_layer: [painter_layer, item.deformers.bits(), 0, 0],
        anim: stored.animation_frames(if item.use_frame_clock {
            frame_seconds as f32
        } else {
            item.time
        }),
        lanes: [[0.0; 4]; 8],
    }
}

/// The world-space box the cells `[col0, row0, col1, row1]` (end exclusive)
/// of `stored` cover when `item` draws them for a camera at `eye`, tiles that
/// reach past their cells included, as `(min, max)`.
pub(crate) fn cell_box_bounds(
    item: &TilemapItem,
    stored: &StoredTilemap,
    cells: [i32; 4],
    eye: glam::DVec3,
) -> (glam::Vec3, glam::Vec3) {
    let plane = item.plane;
    let shift = item.drawn_origin(plane.to_plane(eye));
    let (lo, hi) = stored.layout.extent(cells);
    let [l, b, r, t] = stored.overhang.map(|v| v as f64);
    let cell = glam::DVec2::new(item.cell_size[0] as f64, item.cell_size[1] as f64);
    let a = shift + glam::DVec2::new(lo.x - l, lo.y - b) * cell;
    let b = shift + glam::DVec2::new(hi.x + r, hi.y + t) * cell;
    let e = item.elevation as f64;
    let p = plane.to_world(a, e).as_vec3();
    let q = plane.to_world(b, e).as_vec3();
    (p.min(q), p.max(q))
}

/// The world-space box the chunk at `key` of `stored` covers when `item`
/// draws it for a camera at `eye`, as `(min, max)`.
pub(crate) fn chunk_bounds(
    item: &TilemapItem,
    stored: &StoredTilemap,
    key: ChunkKey,
    eye: glam::DVec3,
) -> (glam::Vec3, glam::Vec3) {
    let cells = stored.map.chunk_cells([key.1, key.0]);
    cell_box_bounds(item, stored, cells, eye)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sizes_match_the_shader() {
        assert_eq!(INSTANCE_STRIDE, 16);
        assert_eq!(
            std::mem::size_of::<TilesetUniformData>(),
            48 + 16 * SIZE_SLOTS
        );
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
        let sizes = [(2u32, 3u32)].into_iter().collect();
        let (tiles, rows) =
            chunk_instances(&map, &slots, &sizes, [0, 0], map.chunk([0, 0]).unwrap());
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].cell, 1 | (2 << 16));
        assert_eq!(tiles[0].tile, 0);
        assert_eq!(tiles[0].flags, TileFlip::ROTATE_90.bits() | FLAG_ANIMATED);
        assert_eq!(tiles[1].tile, 2);
        assert_eq!(tiles[1].flags, 3 << SIZE_SLOT_SHIFT);
        assert_eq!(tiles[1].tint, [255; 4]);
        assert_eq!(rows, vec![(2.0, 0, 1), (3.0, 1, 1)]);
    }

    #[test]
    fn diamond_cells_draw_from_the_top_of_the_plane_down() {
        use super::super::grid::TileOrientation;
        use super::super::types::Tileset;
        let mut set = Tileset::new(None, [32, 16], 4);
        set.orientation = TileOrientation::Isometric;
        let mut map = Tilemap::with_chunk_size(set, [2, 2], 2);
        for (col, row) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            map.set(col, row, Tile::new(1).with_rotate_120(col == 1));
        }
        let none = std::collections::HashMap::new();
        let (tiles, rows) = chunk_instances(&map, &none, &none, [0, 0], map.chunk([0, 0]).unwrap());
        // Cell (0, 1) stands highest, (1, 0) lowest; (0, 0) and (1, 1) share
        // a line, left to right.
        let cells: Vec<_> = tiles.iter().map(|t| unpack_cell(t.cell)).collect();
        assert_eq!(cells, vec![[0, 1], [0, 0], [1, 1], [1, 0]]);
        assert_eq!(rows, vec![(0.5, 0, 1), (0.0, 1, 2), (-0.5, 3, 1)]);
        assert_eq!(tiles[2].flags, FLAG_ROTATE_120);
    }

    #[test]
    fn cells_pack_signed() {
        for cell in [[0, 0], [-1, 5], [32767, -32768], [-300, -2]] {
            assert_eq!(unpack_cell(pack_cell(cell[0], cell[1])), cell);
        }
    }
}
