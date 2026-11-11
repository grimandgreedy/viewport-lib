//! The sprite batches this item type holds on the consumer's behalf, and the
//! per-frame GPU data every sprite draw is built from.
//!
//! A `SpriteItem` carries its billboards and is rebuilt each frame; a
//! `SpriteSetRefItem` or `SpriteInstanceSetRefItem` names a batch uploaded once
//! through the `*_sprite_set` / `*_sprite_instance_set` methods on
//! [`ViewportRenderer`](viewport_lib::renderer::ViewportRenderer). All three end up as
//! the same [`SpriteGpuData`], which is why the builder is shared.
//!
//! The two bind group layouts live here rather than with the pipelines, because
//! an upload builds its bind groups against them and an upload can arrive long
//! before the first frame that draws one.

use bytemuck::Zeroable;
use viewport_lib::error::ViewportResult;
use viewport_lib::plugin_api::Extent;
use viewport_lib::resources::{ContentBuffer, DeviceResources};

pub(crate) use super::types::{SpriteInstanceSetId, SpriteSetId};

/// The bind group layouts sprite uploads build against.
///
/// Group 1 carries a batch's uniform, texture, sampler and instance buffer;
/// group 3 carries the optional tangent-space normal map for a lit batch.
/// Both are built once at startup rather than lazily, because an upload can
/// arrive before the first frame that draws one.
pub(super) struct SpriteLayouts {
    /// Bind group layout for sprite uniforms + texture + instance buffer (group 1).
    pub(super) bgl: viewport_lib::gpu::BindGroupLayout,
    /// Group 3 layout for the optional lit normal map (texture + sampler).
    pub(super) lit_bgl: viewport_lib::gpu::BindGroupLayout,
    /// Group 3 layout of every effect pipeline: the lit normal map, the effect
    /// block, the noise and the palette, and a material's sampler and textures.
    pub(super) fx_bgl: viewport_lib::gpu::BindGroupLayout,
}

impl SpriteLayouts {
    pub(super) fn new(device: &viewport_lib::gpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
            label: Some("sprite_bgl"),
            entries: &[
                // binding 0: SpriteUniform (model, world_space, has_texture)
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: viewport_lib::gpu::ShaderStages::VERTEX
                        | viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Buffer {
                        ty: viewport_lib::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: sprite texture (or fallback 1x1 when has_texture == 0)
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Texture {
                        sample_type: viewport_lib::gpu::TextureSampleType::Float {
                            filterable: true,
                        },
                        view_dimension: viewport_lib::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: sampler
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Sampler(
                        viewport_lib::gpu::SamplerBindingType::Filtering,
                    ),
                    count: None,
                },
                // binding 3: per-sprite instance storage buffer
                viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: viewport_lib::gpu::ShaderStages::VERTEX,
                    ty: viewport_lib::gpu::BindingType::Buffer {
                        ty: viewport_lib::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let lit_bgl = viewport_lib::plugin_api::builders::texture_sampler_bgl(
            device,
            "sprite_lit_bgl",
            viewport_lib::gpu::ShaderStages::FRAGMENT,
        );

        let fx_bgl = super::effects::fx_layout(device);

        Self {
            bgl,
            lit_bgl,
            fx_bgl,
        }
    }
}

/// The renderer-owned handles one sprite upload binds, resolved from
/// `DeviceResources` before the buffers are built.
///
/// They are separated out because they are the only thing the build needs that
/// the plugin does not own, and because wgpu views and samplers are cheap
/// clonable handles: resolving them up front is what lets the buffer work run
/// on a worker thread, where no `DeviceResources` borrow exists.
#[derive(Clone)]
pub(super) struct SpriteBindings {
    texture_view: viewport_lib::gpu::TextureView,
    has_texture: u32,
    normal_view: viewport_lib::gpu::TextureView,
    has_normal_map: u32,
    sampler: viewport_lib::gpu::Sampler,
    bgl: viewport_lib::gpu::BindGroupLayout,
    lit_bgl: viewport_lib::gpu::BindGroupLayout,
    /// The group-3 effect bindings, when the batch has effects.
    fx: Option<super::effects::FxBindings>,
}

/// Resolve the albedo and normal-map views an item names, reporting either
/// through the lib's texture-slot check when it was uploaded in the wrong
/// colour space. A missing or stale id binds the neutral fallback for the slot
/// and clears the shader's `has_*` flag.
pub(super) fn resolve_bindings(
    device: &viewport_lib::gpu::Device,
    resources: &DeviceResources,
    layouts: &SpriteLayouts,
    materials: &[super::effects::SpriteMaterialEntry],
    item: &super::types::SpriteItem,
) -> SpriteBindings {
    resolve_bindings_for(
        device,
        resources,
        layouts,
        materials,
        item.texture_id,
        item.normal_texture_id,
        item.filter,
        item.effects.as_ref(),
        true,
    )
}

/// [`resolve_bindings`] against ids rather than an item, so a stored batch can
/// be resolved again after a texture it names is freed or swapped, where the
/// item that built it is long gone.
///
/// `report` is false on that path: the colour-space mismatch was already
/// reported when the batch was uploaded, and repeating it on every
/// revalidation would spam the log.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_bindings_for(
    device: &viewport_lib::gpu::Device,
    resources: &DeviceResources,
    layouts: &SpriteLayouts,
    materials: &[super::effects::SpriteMaterialEntry],
    texture_id: Option<viewport_lib::resources::TextureId>,
    normal_texture_id: Option<viewport_lib::resources::TextureId>,
    filter: viewport_lib::TextureFilter,
    effects: Option<&super::types::SpriteEffects>,
    report: bool,
) -> SpriteBindings {
    use viewport_lib::resources::TextureRequirement;
    if report {
        // A palette-swapped sprite's texture holds indices, not colour.
        let requirement = if effects.is_some_and(|fx| fx.palette.is_some()) {
            TextureRequirement::linear("SpriteItem::texture_id")
        } else {
            TextureRequirement::colour("SpriteItem::texture_id")
        };
        resources.check_texture_slot(texture_id, requirement);
        resources.check_texture_slot(
            normal_texture_id,
            TextureRequirement::normal_map("SpriteItem::normal_texture_id"),
        );
    }

    let (texture_view, has_texture) = match texture_id.and_then(|id| resources.texture_view(id)) {
        Some(view) => (view.clone(), 1),
        None => (resources.fallback_colourmap_view().clone(), 0),
    };
    let (normal_view, has_normal_map) =
        match normal_texture_id.and_then(|id| resources.texture_view(id)) {
            Some(view) => (view.clone(), 1),
            None => (
                resources
                    .fallback_texture_view(viewport_lib::TextureSlot::Normal)
                    .clone(),
                0,
            ),
        };
    SpriteBindings {
        texture_view,
        has_texture,
        normal_view,
        has_normal_map,
        sampler: match filter {
            viewport_lib::TextureFilter::Linear => resources.material_sampler().clone(),
            viewport_lib::TextureFilter::Nearest => resources.sampler_for(
                device,
                viewport_lib::SamplerKey {
                    filter: viewport_lib::TextureFilter::Nearest,
                    ..Default::default()
                },
            ),
        },
        bgl: layouts.bgl.clone(),
        lit_bgl: layouts.lit_bgl.clone(),
        fx: effects
            .map(|fx| super::effects::resolve(resources, &layouts.fx_bgl, materials, fx, report)),
    }
}

/// Uniform block for one sprite batch: model matrix + flags + soft-particle
/// distance + orientation + refraction strength + lit parameters. Layout
/// mirrors `SpriteUniform` in `sprite_lit.wgsl`; the emissive `sprite.wgsl`
/// reads only the first half and ignores the trailing lit fields.
///
/// Kept on the batch rather than written and forgotten, because the two
/// `has_*` flags depend on textures that can be freed after the upload: a
/// revalidation rewrites them without needing the item back.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SpriteUniformData {
    model: [[f32; 4]; 4],
    world_space: u32,
    has_texture: u32,
    soft_particle_distance: f32,
    orientation: u32,
    axis: [f32; 3],
    refraction_strength: f32,
    lit: u32,
    normal_mode: u32,
    has_normal_map: u32,
    ambient_scale: f32,
    roughness: f32,
    receive_shadows: u32,
    /// Bit 0: pixel snap. Bits 1-2: how the fragment shapes its output for the
    /// fixed-function blend, [`BLEND_OUT_MULTIPLY`] or [`BLEND_OUT_SCREEN`].
    flags: u32,
    _pad_lit_c: u32,
    /// The painter mask test word and the batch's painter layer, then
    /// padding; zero for a batch in depth order.
    mask: [u32; 4],
}

/// [`SpriteUniformData::flags`] bit: snap each sprite to whole device pixels.
const FLAG_PIXEL_SNAP: u32 = 1;
/// [`SpriteUniformData::flags`] blend output for `SpriteBlend::Multiply`.
const BLEND_OUT_MULTIPLY: u32 = 1 << 1;
/// [`SpriteUniformData::flags`] blend output for `SpriteBlend::Screen`.
const BLEND_OUT_SCREEN: u32 = 2 << 1;
/// [`SpriteUniformData::flags`] bit: the blend does not weight the colour by
/// its alpha (premultiplied and additive), so a mask scales every channel
/// rather than the alpha alone.
const FLAG_PREMULTIPLIED: u32 = 1 << 3;

/// The per-sprite record the shader reads, matching `SpriteInstance` in
/// `sprite_common.wgsl`. 80 bytes each.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuSpriteInstance {
    colour: [f32; 4],
    size: [f32; 2],
    rotation: f32,
    soft_distance: f32,
    /// Already flipped: a flip swaps the rect's edges.
    uv_rect: [f32; 4],
    velocity: [f32; 3],
    /// The pivot as an offset from the quad centre in `-1..=1` of the half
    /// size, two snorm16 values (x low, y high). Zero is the centre.
    pivot: u32,
    /// The batch's draw mode, as [`draw_word`] packs it.
    draw: [u32; 4],
}

/// A sprite draw mode as the shaders read it: the mode in bits 0-1 of the
/// first word (adaptive tiling in bit 2), then the border or tile size as two
/// pairs of half floats.
pub(crate) fn draw_word(mode: super::types::SpriteDrawMode) -> [u32; 4] {
    use super::types::{SpriteDrawMode, TileMode};
    let pair = |a: f32, b: f32| f16_bits(a) as u32 | ((f16_bits(b) as u32) << 16);
    match mode {
        SpriteDrawMode::Simple => [0; 4],
        SpriteDrawMode::Sliced {
            border: [l, b, r, t],
        } => [
            1,
            pair(l.max(0.0), b.max(0.0)),
            pair(r.max(0.0), t.max(0.0)),
            0,
        ],
        SpriteDrawMode::Tiled { tile_size, mode } => [
            2 | if mode == TileMode::Adaptive { 4 } else { 0 },
            pair(tile_size[0].max(0.0), tile_size[1].max(0.0)),
            0,
            0,
        ],
    }
}

/// `v` as an IEEE half float, rounded to nearest even, saturating to
/// infinity.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let a = v.abs();
    if a.is_nan() {
        return sign | 0x7e00;
    }
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    if a < 6.103_515_6e-5 {
        // Subnormal: a whole number of the smallest half step.
        return sign | (a / 5.960_464_5e-8).round() as u16;
    }
    let abits = a.to_bits();
    let exp = ((abits >> 23) & 0xff) - 127 + 15;
    let mant = abits & 0x7f_ffff;
    let mut h = (exp << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

impl GpuSpriteInstance {
    #[allow(clippy::too_many_arguments)]
    fn new(
        colour: [f32; 4],
        size: [f32; 2],
        rotation: f32,
        soft_distance: f32,
        uv_rect: [f32; 4],
        flip: super::types::SpriteFlip,
        velocity: [f32; 3],
        pivot: [f32; 2],
        draw: [u32; 4],
    ) -> Self {
        Self {
            colour,
            size,
            rotation,
            soft_distance: soft_distance.max(0.0),
            uv_rect: flip.apply(uv_rect),
            velocity,
            pivot: pack_pivot(pivot),
            draw,
        }
    }
}

/// Pack a `0..=1` pivot as two snorm16 offsets from the quad centre, so the
/// default centre pivot packs to zero and the corners are exact.
fn pack_pivot(pivot: [f32; 2]) -> u32 {
    let snorm = |p: f32| {
        let offset = (p * 2.0 - 1.0).clamp(-1.0, 1.0);
        ((offset * 32767.0).round() as i16) as u16 as u32
    };
    snorm(pivot[0]) | (snorm(pivot[1]) << 16)
}

/// Bytes per element of the two writable sprite channels.
const POSITION_STRIDE: u32 = 12;
const SPRITE_STRIDE: u32 = std::mem::size_of::<GpuSpriteInstance>() as u32;

/// Encode a run of caller-supplied sprites into the bytes a ranged write sends.
pub(crate) fn encode_sprites(sprites: &[super::channels::Sprite]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(sprites.len() * SPRITE_STRIDE as usize);
    for s in sprites {
        let record = GpuSpriteInstance::new(
            s.colour.to_linear_rgba(),
            s.size,
            s.rotation,
            s.soft_distance,
            s.uv_rect,
            s.flip,
            s.velocity,
            s.pivot,
            // The batch's own mode goes in when the write lands.
            [0; 4],
        );
        bytes.extend_from_slice(bytemuck::bytes_of(&record));
    }
    bytes
}

/// Whether a built batch keeps a caller-ordered copy of its sprites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Retain {
    /// A per-frame item, rebuilt every frame: no copy.
    Frame,
    /// An uploaded set: a copy when it draws in painter's order, so ranged
    /// writes can address sprites by index and a key write can re-sort it.
    Set,
    /// An uploaded instance set: always a copy, which per-frame overrides are
    /// applied to.
    InstanceSet,
}

/// Build the GPU data for one sprite batch: its buffers and its bind groups.
///
/// Shared by the per-frame item path and the two stores, so a reference draw
/// and an inline draw are bit-for-bit the same work.
pub(super) fn build_sprite(
    device: &viewport_lib::gpu::Device,
    queue: &viewport_lib::gpu::Queue,
    binds: &SpriteBindings,
    item: &super::types::SpriteItem,
    capacity: u32,
    retain: Retain,
) -> StoredSprite {
    {
        let count = item.positions.len() as u32;
        let capacity = capacity.max(count);
        let painter = match item.order {
            super::types::SpriteOrder::Painter { layer, depth_test } => {
                Some(PainterOrder { layer, depth_test })
            }
            super::types::SpriteOrder::Depth => None,
        };
        // A painter batch is drawn in sort-key order, so its records are
        // written in that order; `order` maps a drawn index back to the
        // caller's.
        let order = if painter.is_some() {
            painter_instance_order(&item.sort_keys, item.positions.len())
        } else {
            None
        };
        let source = |drawn: usize| order.as_ref().map_or(drawn, |o| o[drawn] as usize);
        let keep_copy = match retain {
            Retain::Frame => false,
            Retain::Set => painter.is_some(),
            Retain::InstanceSet => true,
        };

        // Position vertex buffer (one vec3 per sprite, instance-stepped).
        let mut positions = ContentBuffer::new(
            device,
            "sprite_vertex_buf",
            viewport_lib::gpu::BufferUsages::VERTEX,
            POSITION_STRIDE,
            capacity,
        );
        if order.is_some() {
            let sorted: Vec<[f32; 3]> = (0..item.positions.len())
                .map(|i| item.positions[source(i)])
                .collect();
            let _ = positions.write_range(queue, 0, bytemuck::cast_slice(&sorted));
        } else {
            let _ = positions.write_range(queue, 0, bytemuck::cast_slice(&item.positions));
        }

        // The records in the caller's order, then in drawn order.
        let draw_mode = draw_word(item.draw_mode);
        let records: Vec<GpuSpriteInstance> = (0..item.positions.len())
            .map(|i| {
                GpuSpriteInstance::new(
                    item.colours
                        .get(i)
                        .unwrap_or(&item.default_colour)
                        .to_linear_rgba(),
                    item.sizes.get(i).copied().unwrap_or([item.default_size; 2]),
                    item.rotations.get(i).copied().unwrap_or(0.0),
                    item.soft_particle_distances.get(i).copied().unwrap_or(0.0),
                    item.uv_rects
                        .get(i)
                        .copied()
                        .unwrap_or([0.0, 0.0, 1.0, 1.0]),
                    item.flips.get(i).copied().unwrap_or_default(),
                    item.velocities.get(i).copied().unwrap_or([0.0; 3]),
                    item.pivots.get(i).copied().unwrap_or(item.default_pivot),
                    draw_mode,
                )
            })
            .collect();
        let drawn_records: std::borrow::Cow<'_, [GpuSpriteInstance]> = match &order {
            Some(o) => o.iter().map(|&i| records[i as usize]).collect(),
            None => std::borrow::Cow::Borrowed(&records),
        };

        let mut sprites = ContentBuffer::new(
            device,
            "sprite_instance_buf",
            viewport_lib::gpu::BufferUsages::STORAGE,
            SPRITE_STRIDE,
            capacity,
        );
        let _ = sprites.write_range(queue, 0, bytemuck::cast_slice(&drawn_records));
        let mut uploaded_bytes = positions.live_bytes() + sprites.live_bytes();

        let (texture_view, has_texture) = (&binds.texture_view, binds.has_texture);

        let orientation = match item.orientation {
            super::types::SpriteOrientation::CameraFacing => 0u32,
            super::types::SpriteOrientation::VelocityStretched => 1u32,
            super::types::SpriteOrientation::AxisLocked => 2u32,
        };

        let normal_mode = match item.lit_params.normal_mode {
            super::types::SpriteNormalMode::Spherical => 0u32,
            super::types::SpriteNormalMode::Flat => 1u32,
            super::types::SpriteNormalMode::NormalMap => 2u32,
        };

        let (normal_view, has_normal_map) = (&binds.normal_view, binds.has_normal_map);

        let uniform_data = SpriteUniformData {
            model: item.model,
            world_space: if item.size_mode == super::types::SpriteSizeMode::WorldSpace {
                1
            } else {
                0
            },
            has_texture,
            soft_particle_distance: item
                .soft_particle_distance
                .filter(|d| *d > 0.0)
                .unwrap_or(0.0),
            orientation,
            axis: item.axis,
            refraction_strength: item.refraction_strength.filter(|s| *s > 0.0).unwrap_or(0.0),
            lit: item.lit as u32,
            normal_mode,
            has_normal_map,
            ambient_scale: item.lit_params.ambient_scale,
            roughness: item.lit_params.roughness,
            receive_shadows: item.lit_params.receive_shadows as u32,
            flags: if item.pixel_snap { FLAG_PIXEL_SNAP } else { 0 }
                | match item.blend {
                    viewport_lib::renderer::SpriteBlend::Multiply => BLEND_OUT_MULTIPLY,
                    viewport_lib::renderer::SpriteBlend::Screen => BLEND_OUT_SCREEN,
                    viewport_lib::renderer::SpriteBlend::Premultiplied
                    | viewport_lib::renderer::SpriteBlend::Additive => FLAG_PREMULTIPLIED,
                    _ => 0,
                },
            _pad_lit_c: 0,
            mask: match painter {
                Some(p) => [item.mask.word(), p.layer, 0, 0],
                None => [0; 4],
            },
        };
        let uniform_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("sprite_uniform_buf"),
            size: std::mem::size_of::<SpriteUniformData>() as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = &binds.bgl;

        let bind_group = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("sprite_bind_group"),
            layout: bgl,
            entries: &[
                viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_lib::gpu::BindingResource::TextureView(texture_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 2,
                    resource: viewport_lib::gpu::BindingResource::Sampler(&binds.sampler),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 3,
                    resource: sprites.buffer().as_entire_binding(),
                },
            ],
        });

        let lit_normal_bg = if item.lit {
            Some({
                let lit_bgl = &binds.lit_bgl;
                device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
                    label: Some("sprite_lit_normal_bg"),
                    layout: lit_bgl,
                    entries: &[
                        viewport_lib::gpu::BindGroupEntry {
                            binding: 0,
                            resource: viewport_lib::gpu::BindingResource::TextureView(normal_view),
                        },
                        viewport_lib::gpu::BindGroupEntry {
                            binding: 1,
                            resource: viewport_lib::gpu::BindingResource::Sampler(&binds.sampler),
                        },
                    ],
                })
            })
        } else {
            None
        };

        // The effect stream, block and group 3, for a batch with effects.
        let mut fx_stream = None;
        let fx = match (&binds.fx, &item.effects) {
            (Some(fxb), Some(effects)) => {
                let words = super::effects::instance_words(effects, item.positions.len(), source);
                uploaded_bytes += words.len() as u64 * super::effects::FX_STRIDE as u64;
                let mut stream = ContentBuffer::new(
                    device,
                    "sprite_fx_stream",
                    viewport_lib::gpu::BufferUsages::VERTEX,
                    super::effects::FX_STRIDE,
                    capacity,
                );
                let _ = stream.write_range(queue, 0, bytemuck::cast_slice(&words));
                let uniform_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
                    label: Some("sprite_fx_uniform_buf"),
                    size: std::mem::size_of::<super::effects::SpriteFxUniform>() as u64,
                    usage: viewport_lib::gpu::BufferUsages::UNIFORM
                        | viewport_lib::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&fxb.uniform));
                let bind_group = super::effects::bind_group(
                    device,
                    fxb,
                    normal_view,
                    &binds.sampler,
                    &uniform_buf,
                );
                let draw = SpriteFxDraw {
                    family: fxb.family,
                    bind_group,
                    stream: stream.buffer().clone(),
                    uniform_buf,
                    effects: std::sync::Arc::new(effects.clone()),
                };
                fx_stream = Some(stream);
                Some(draw)
            }
            _ => None,
        };

        let oit_eligible = matches!(
            item.blend,
            viewport_lib::renderer::SpriteBlend::AlphaBlend
                | viewport_lib::renderer::SpriteBlend::Premultiplied
        ) && !item.depth_write
            && painter.is_none()
            && item.soft_particle_distance.is_none_or(|d| d <= 0.0)
            && item.soft_particle_distances.is_empty()
            && item.refraction_strength.is_none_or(|s| s <= 0.0)
            && fx.is_none();

        // The keys in drawn order, so the frame can split the batch where
        // another batch's keys fall between its own.
        let painter_keys = (painter.is_some() && !item.sort_keys.is_empty()).then(|| {
            (0..count as usize)
                .map(|d| item.sort_keys.get(source(d)).copied().unwrap_or(0.0))
                .collect::<std::sync::Arc<[f32]>>()
        });
        let draw = SpriteGpuData {
            vertex_buffer: positions.buffer().clone(),
            sprite_count: count,
            pick_id: item.settings.pick_id,
            pick_alpha_threshold: item.pick_alpha_threshold,
            bind_group,
            texture_id: item.texture_id,
            normal_texture_id: item.normal_texture_id,
            uniform: uniform_data,
            // Painter's order never writes depth.
            depth_write: item.depth_write && painter.is_none(),
            painter,
            painter_keys,
            instance_order: order.as_deref().map(std::sync::Arc::from),
            filter: item.filter,
            blend: item.blend,
            wireframe: false,
            refraction_strength: item.refraction_strength.filter(|s| *s > 0.0).unwrap_or(0.0),
            lit: item.lit,
            lit_normal_bg,
            oit_eligible,
            fx,
            _uniform_buf: uniform_buf,
        };
        let copy = keep_copy.then(|| {
            let n = item.positions.len();
            SpriteCopy {
                len: count,
                positions: item.positions.clone(),
                records,
                fx_words: item
                    .effects
                    .as_ref()
                    .filter(|_| fx_stream.is_some())
                    .map(|fx| super::effects::instance_words(fx, n, |i| i)),
                keys: (0..n)
                    .map(|i| item.sort_keys.get(i).copied().unwrap_or(0.0))
                    .collect(),
                order: (0..n as u32).map(|d| source(d as usize) as u32).collect(),
                dirty: Dirty::default(),
            }
        });
        StoredSprite {
            positions,
            sprites,
            fx_stream,
            binds: binds.clone(),
            draw,
            model: item.model,
            origin: glam::DVec3::ZERO,
            copy,
            draw_word: draw_mode,
            uploaded_bytes,
        }
    }
}

/// One sprite batch as a store holds it: the growable channels, the bindings a
/// grow needs to rebuild the bind group, and the draw data the frame path clones.
///
/// The two are separate because the draw data is cloned per drawn item per frame
/// and a growable buffer must have one owner: two owners would each have their
/// own idea of how much is allocated. The draw's buffer handles are wgpu's own
/// reference-counted ones, so a clone keeps the allocation alive without owning
/// the capacity.
pub(crate) struct StoredSprite {
    positions: ContentBuffer,
    sprites: ContentBuffer,
    /// The packed effect words, on a batch with effects.
    fx_stream: Option<ContentBuffer>,
    binds: SpriteBindings,
    pub(crate) draw: SpriteGpuData,
    /// The batch's model matrix in world coordinates.
    model: [[f32; 4]; 4],
    /// The world origin the uniform's model matrix is relative to.
    origin: glam::DVec3,
    /// The caller-ordered copy, on a painter set and an instance set.
    copy: Option<SpriteCopy>,
    /// The batch's draw mode word, written into every record a ranged write
    /// supplies.
    draw_word: [u32; 4],
    /// Instance bytes written to the GPU since the frame last took them.
    uploaded_bytes: u64,
}

/// A stored batch's sprites in the caller's order, and the order they are
/// laid out in on the GPU.
///
/// A painter batch draws its sprites sorted by key, so the caller's sprite `i`
/// sits at some other slot of the buffers. The copy is what lets a ranged
/// write address sprites by the caller's index, a key write re-sort the batch
/// without reading the buffers back, and an override build a frame's variant
/// of the batch.
pub(crate) struct SpriteCopy {
    /// Live sprites, shared by every channel.
    len: u32,
    positions: Vec<[f32; 3]>,
    records: Vec<GpuSpriteInstance>,
    /// The packed effect words, on a batch with effects.
    fx_words: Option<Vec<u32>>,
    /// Sort keys; `0.0` where none was given.
    keys: Vec<f32>,
    /// The caller's index of each slot as the buffers hold them now.
    order: Vec<u32>,
    /// What changed since the buffers were last written.
    dirty: Dirty,
}

/// Sprites written since the last flush, by caller index, per channel.
#[derive(Default)]
struct Dirty {
    positions: Vec<bool>,
    records: Vec<bool>,
    fx: Vec<bool>,
    /// The keys changed: the batch re-sorts at the flush.
    resort: bool,
    /// The live count changed: the slots are laid out again at the flush.
    relayout: bool,
}

impl Dirty {
    fn any(&self) -> bool {
        self.resort
            || self.relayout
            || self.positions.contains(&true)
            || self.records.contains(&true)
            || self.fx.contains(&true)
    }
}

/// Mark `range` of `flags`, growing it as needed.
fn mark(flags: &mut Vec<bool>, range: std::ops::Range<usize>) {
    if flags.len() < range.end {
        flags.resize(range.end, false);
    }
    flags[range].fill(true);
}

/// Copy `values` into `into` at `first`, padding with `pad` up to it.
fn put<T: Copy>(into: &mut Vec<T>, first: usize, values: &[T], pad: T) {
    let end = first + values.len();
    if into.len() < end {
        into.resize(end, pad);
    }
    into[first..end].copy_from_slice(values);
}

/// Runs of written slots past which a write covers the span from the first
/// to the last instead: each run is a queue write of its own, and a re-sort
/// that shuffles every sprite a little leaves thousands of short runs.
const MAX_RUNS: usize = 32;

/// Write `data[order[d]]` for every slot `d` that `wanted` names, one queue
/// write per run of consecutive slots, or one over their whole span when
/// there are more than [`MAX_RUNS`] runs. Returns the bytes written.
fn write_slots<T: bytemuck::Pod>(
    buffer: &mut ContentBuffer,
    queue: &viewport_lib::gpu::Queue,
    order: &[u32],
    data: &[T],
    pad: T,
    wanted: impl Fn(usize) -> bool,
) -> u64 {
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    let mut d = 0;
    while d < order.len() {
        if !wanted(d) {
            d += 1;
            continue;
        }
        let start = d;
        while d < order.len() && wanted(d) {
            d += 1;
        }
        runs.push(start..d);
    }
    if runs.len() > MAX_RUNS {
        let span = runs[0].start..runs[runs.len() - 1].end;
        runs = vec![span];
    }
    let mut written = 0;
    let mut values: Vec<T> = Vec::new();
    for run in runs {
        values.clear();
        values.extend(
            order[run.clone()]
                .iter()
                .map(|&i| data.get(i as usize).copied().unwrap_or(pad)),
        );
        let bytes: &[u8] = bytemuck::cast_slice(&values);
        if buffer.write_range(queue, run.start as u32, bytes).is_ok() {
            written += bytes.len() as u64;
        }
    }
    written
}

/// The keys of a painter batch in drawn order, or `None` when every key is
/// `0.0`, which is how the frame reads a batch without keys.
fn drawn_keys(keys: &[f32], order: &[u32]) -> Option<std::sync::Arc<[f32]>> {
    keys.iter().any(|&k| k != 0.0).then(|| {
        order
            .iter()
            .map(|&i| keys.get(i as usize).copied().unwrap_or(0.0))
            .collect()
    })
}

/// `order` as the draw data carries it: `None` when it is the identity.
fn instance_order(order: &[u32]) -> Option<std::sync::Arc<[u32]>> {
    order
        .iter()
        .enumerate()
        .any(|(d, &i)| d as u32 != i)
        .then(|| std::sync::Arc::from(order))
}

impl StoredSprite {
    /// Rewrite the uniform's model matrix relative to `origin` when it is not
    /// already.
    pub(crate) fn rebase(&mut self, queue: &viewport_lib::gpu::Queue, origin: glam::DVec3) {
        if origin == self.origin {
            return;
        }
        self.origin = origin;
        self.draw.uniform.model = crate::item_types::origin::rel_model(self.model, origin);
        queue.write_buffer(
            &self.draw._uniform_buf,
            0,
            bytemuck::bytes_of(&self.draw.uniform),
        );
    }

    /// Write positions, records, effect words or sort keys at a sprite
    /// offset, keeping the draw count in step with the positions.
    ///
    /// A batch with a copy takes the write into the copy and writes the
    /// buffers at the next [`flush`](Self::flush), at the slots the written
    /// sprites hold; a batch without one writes straight through.
    pub(crate) fn write_channel(
        &mut self,
        queue: &viewport_lib::gpu::Queue,
        channel: SpriteChannel,
        first_element: u32,
        data: &[u8],
    ) -> ViewportResult<()> {
        let not_present =
            |channel: &'static str| viewport_lib::error::ViewportError::ChannelNotPresent {
                type_name: super::TYPE_NAME,
                channel,
            };
        if self.copy.is_some() {
            return self.write_copy(channel, first_element, data);
        }
        match channel {
            SpriteChannel::Positions => {
                self.positions.write_range(queue, first_element, data)?;
                self.draw.sprite_count = self.positions.len();
            }
            SpriteChannel::Sprites => {
                let records = self.with_draw_word(data);
                self.sprites.write_range(queue, first_element, &records)?
            }
            SpriteChannel::Effects => self
                .fx_stream
                .as_mut()
                .ok_or(not_present("effects"))?
                .write_range(queue, first_element, data)?,
            // Keys order a painter batch only, and every painter set keeps a
            // copy.
            SpriteChannel::SortKeys => return Err(not_present("sort_keys")),
        }
        self.uploaded_bytes += data.len() as u64;
        Ok(())
    }

    /// `data`, whole sprite records, with the batch's draw mode in each.
    fn with_draw_word(&self, data: &[u8]) -> Vec<u8> {
        let mut out = data.to_vec();
        let at = std::mem::offset_of!(GpuSpriteInstance, draw);
        let word: &[u8] = bytemuck::cast_slice(&self.draw_word);
        for record in out.chunks_exact_mut(SPRITE_STRIDE as usize) {
            record[at..at + word.len()].copy_from_slice(word);
        }
        out
    }

    /// [`write_channel`](Self::write_channel) into the copy.
    fn write_copy(
        &mut self,
        channel: SpriteChannel,
        first_element: u32,
        data: &[u8],
    ) -> ViewportResult<()> {
        let (stride, buffer_capacity) = match channel {
            SpriteChannel::Positions => (POSITION_STRIDE, self.positions.capacity()),
            SpriteChannel::Sprites => (SPRITE_STRIDE, self.sprites.capacity()),
            SpriteChannel::Effects => match &self.fx_stream {
                Some(stream) => (super::effects::FX_STRIDE, stream.capacity()),
                None => {
                    return Err(viewport_lib::error::ViewportError::ChannelNotPresent {
                        type_name: super::TYPE_NAME,
                        channel: "effects",
                    });
                }
            },
            SpriteChannel::SortKeys => (4, self.positions.capacity()),
        };
        if data.is_empty() {
            return Ok(());
        }
        let count = (data.len() / stride as usize) as u32;
        let end = first_element.saturating_add(count);
        if data.len() % stride as usize != 0 || end > buffer_capacity {
            return Err(
                viewport_lib::error::ViewportError::ContentBufferWriteOutOfRange {
                    first_element,
                    element_count: if data.len() % stride as usize == 0 {
                        count
                    } else {
                        0
                    },
                    capacity: buffer_capacity,
                    stride_bytes: stride,
                },
            );
        }
        let copy = self.copy.as_mut().expect("checked by the caller");
        let first = first_element as usize;
        let range = first..end as usize;
        match channel {
            SpriteChannel::Positions => {
                let values: Vec<[f32; 3]> = bytemuck::pod_collect_to_vec(data);
                put(&mut copy.positions, first, &values, [0.0; 3]);
                mark(&mut copy.dirty.positions, range);
            }
            SpriteChannel::Sprites => {
                let mut values: Vec<GpuSpriteInstance> = bytemuck::pod_collect_to_vec(data);
                for v in &mut values {
                    v.draw = self.draw_word;
                }
                put(
                    &mut copy.records,
                    first,
                    &values,
                    GpuSpriteInstance::zeroed(),
                );
                mark(&mut copy.dirty.records, range);
            }
            SpriteChannel::Effects => {
                let values: Vec<u32> = bytemuck::pod_collect_to_vec(data);
                let words = copy.fx_words.get_or_insert_with(Vec::new);
                put(words, first, &values, 0);
                mark(&mut copy.dirty.fx, range);
            }
            SpriteChannel::SortKeys => {
                let values: Vec<f32> = bytemuck::pod_collect_to_vec(data);
                put(&mut copy.keys, first, &values, 0.0);
                copy.dirty.resort = true;
            }
        }
        if end > copy.len {
            copy.len = end;
            copy.dirty.relayout = true;
        }
        Ok(())
    }

    /// Write what the copy changed since the last flush into the buffers:
    /// re-sort after a key write, lay the slots out again after the live
    /// count changed, and write each written sprite at its slot. A sprite
    /// whose slot changed is written on every channel. Nothing for a batch
    /// without a copy or with nothing changed.
    pub(crate) fn flush(&mut self, queue: &viewport_lib::gpu::Queue) {
        let Some(copy) = self.copy.as_mut() else {
            return;
        };
        if !copy.dirty.any() {
            return;
        }
        let n = copy.len;
        let target: Vec<u32> = if copy.dirty.resort && self.draw.painter.is_some() {
            painter_instance_order(&copy.keys, n as usize).unwrap_or_else(|| (0..n).collect())
        } else {
            // The sprites that stay keep their slots' order; new ones follow.
            let old = copy.order.len() as u32;
            copy.order
                .iter()
                .copied()
                .filter(|&i| i < n)
                .chain(old..n.max(old))
                .collect()
        };
        let moved = |d: usize| copy.order.get(d) != Some(&target[d]);
        let dirty =
            |flags: &[bool], d: usize| flags.get(target[d] as usize).copied().unwrap_or(false);
        let mut written = write_slots(
            &mut self.positions,
            queue,
            &target,
            &copy.positions,
            [0.0; 3],
            |d| moved(d) || dirty(&copy.dirty.positions, d),
        );
        written += write_slots(
            &mut self.sprites,
            queue,
            &target,
            &copy.records,
            GpuSpriteInstance::zeroed(),
            |d| moved(d) || dirty(&copy.dirty.records, d),
        );
        if let (Some(stream), Some(words)) = (&mut self.fx_stream, &copy.fx_words) {
            written += write_slots(stream, queue, &target, words, 0, |d| {
                moved(d) || dirty(&copy.dirty.fx, d)
            });
            let _ = stream.set_len(n);
        }
        let _ = self.positions.set_len(n);
        let _ = self.sprites.set_len(n);
        self.draw.sprite_count = n;
        if self.draw.painter.is_some() {
            self.draw.painter_keys =
                drawn_keys(&copy.keys[..(n as usize).min(copy.keys.len())], &target);
        }
        self.draw.instance_order = instance_order(&target);
        copy.order = target;
        copy.dirty = Dirty::default();
        self.uploaded_bytes += written;
    }

    /// Instance bytes written since the last call.
    pub(crate) fn take_uploaded_bytes(&mut self) -> u64 {
        std::mem::take(&mut self.uploaded_bytes)
    }

    /// Grow both channels to at least `capacity` sprites, rebuilding the bind
    /// group and refreshing the draw data if either allocation moved. `true` when
    /// one did.
    pub(crate) fn reserve(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        capacity: u32,
    ) -> bool {
        let mut moved = self.positions.reserve(device, queue, capacity);
        moved |= self.sprites.reserve(device, queue, capacity);
        if let Some(stream) = &mut self.fx_stream {
            moved |= stream.reserve(device, queue, capacity);
        }
        if moved {
            self.draw.vertex_buffer = self.positions.buffer().clone();
            if let (Some(fx), Some(stream)) = (&mut self.draw.fx, &self.fx_stream) {
                fx.stream = stream.buffer().clone();
            }
            self.draw.bind_group = build_bind_group(
                device,
                &self.binds,
                &self.draw._uniform_buf,
                self.sprites.buffer(),
            );
        }
        moved
    }

    /// Set how many sprites draw. On a batch with a copy the sprites past
    /// `len` leave the draw at the next flush, whichever slots they hold.
    pub(crate) fn set_live_len(&mut self, len: u32) -> ViewportResult<()> {
        if let Some(copy) = &mut self.copy {
            let capacity = self.positions.capacity();
            if len > capacity {
                return Err(
                    viewport_lib::error::ViewportError::ContentBufferWriteOutOfRange {
                        first_element: 0,
                        element_count: len,
                        capacity,
                        stride_bytes: POSITION_STRIDE,
                    },
                );
            }
            if len != copy.len {
                copy.len = len;
                copy.dirty.relayout = true;
            }
            return Ok(());
        }
        self.positions.set_len(len)?;
        self.sprites.set_len(len)?;
        if let Some(stream) = &mut self.fx_stream {
            stream.set_len(len)?;
        }
        self.draw.sprite_count = len;
        Ok(())
    }

    /// What one channel holds and how much of it draws. `None` for the
    /// effects channel of a batch without effects.
    pub(crate) fn extent(&self, channel: SpriteChannel) -> Option<Extent> {
        let cb = match channel {
            SpriteChannel::Positions => &self.positions,
            SpriteChannel::Sprites => &self.sprites,
            SpriteChannel::Effects => self.fx_stream.as_ref()?,
            SpriteChannel::SortKeys => {
                self.copy.as_ref().filter(|_| self.draw.painter.is_some())?;
                &self.positions
            }
        };
        Some(Extent {
            capacity: cb.capacity(),
            // A batch with a copy has one live count, ahead of the buffers
            // until the next flush.
            len: self.copy.as_ref().map_or(cb.len(), |c| c.len),
        })
    }

    /// This batch with `overrides` applied, for one frame: the stored buffers
    /// are untouched, and only the streams the overrides change are copied
    /// into frame buffers. Returns the draw data and the bytes written.
    /// `None` when the batch keeps no copy or the overrides change nothing.
    pub(crate) fn overridden(
        &self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        overrides: &super::types::SpriteOverrides,
    ) -> Option<(SpriteGpuData, u64)> {
        let copy = self.copy.as_ref()?;
        if overrides.is_empty() {
            return None;
        }
        let n = copy.len as usize;
        let live = |v: usize| v.min(n);
        let mut positions = None::<Vec<[f32; 3]>>;
        let mut records = None::<Vec<GpuSpriteInstance>>;
        let mut keys = None::<Vec<f32>>;
        for (j, &index) in overrides.indices.iter().enumerate() {
            let i = index as usize;
            if i >= n {
                continue;
            }
            if let Some(p) = overrides.positions.get(j) {
                let v = positions.get_or_insert_with(|| {
                    let mut v = copy.positions[..live(copy.positions.len())].to_vec();
                    v.resize(n, [0.0; 3]);
                    v
                });
                v[i] = *p;
            }
            if let Some(c) = overrides.colours.get(j) {
                let v = records.get_or_insert_with(|| {
                    let mut v = copy.records[..live(copy.records.len())].to_vec();
                    v.resize(n, GpuSpriteInstance::zeroed());
                    v
                });
                v[i].colour = c.to_linear_rgba();
            }
            if let Some(k) = overrides.sort_keys.get(j)
                && self.draw.painter.is_some()
            {
                let v = keys.get_or_insert_with(|| {
                    let mut v = copy.keys[..live(copy.keys.len())].to_vec();
                    v.resize(n, 0.0);
                    v
                });
                v[i] = *k;
            }
        }
        if positions.is_none() && records.is_none() && keys.is_none() {
            return None;
        }
        let order: std::borrow::Cow<'_, [u32]> = match &keys {
            Some(k) => painter_instance_order(k, n)
                .unwrap_or_else(|| (0..n as u32).collect())
                .into(),
            None => std::borrow::Cow::Borrowed(&copy.order),
        };
        let reordered = *order != copy.order[..];
        let mut draw = self.draw.clone();
        let mut written = 0;
        let frame_buffer = |label, usage, stride, data: &[u8]| {
            let mut buffer = ContentBuffer::new(device, label, usage, stride, n as u32);
            let _ = buffer.write_range(queue, 0, data);
            buffer.buffer().clone()
        };
        if positions.is_some() || reordered {
            let source = positions.as_deref().unwrap_or(&copy.positions);
            let drawn: Vec<[f32; 3]> = order
                .iter()
                .map(|&i| source.get(i as usize).copied().unwrap_or([0.0; 3]))
                .collect();
            let bytes: &[u8] = bytemuck::cast_slice(&drawn);
            written += bytes.len() as u64;
            draw.vertex_buffer = frame_buffer(
                "sprite_vertex_buf",
                viewport_lib::gpu::BufferUsages::VERTEX,
                POSITION_STRIDE,
                bytes,
            );
        }
        if records.is_some() || reordered {
            let source = records.as_deref().unwrap_or(&copy.records);
            let drawn: Vec<GpuSpriteInstance> = order
                .iter()
                .map(|&i| {
                    source
                        .get(i as usize)
                        .copied()
                        .unwrap_or(GpuSpriteInstance::zeroed())
                })
                .collect();
            let bytes: &[u8] = bytemuck::cast_slice(&drawn);
            written += bytes.len() as u64;
            let buffer = frame_buffer(
                "sprite_instance_buf",
                viewport_lib::gpu::BufferUsages::STORAGE,
                SPRITE_STRIDE,
                bytes,
            );
            draw.bind_group =
                build_bind_group(device, &self.binds, &self.draw._uniform_buf, &buffer);
        }
        if reordered && let (Some(fx), Some(words)) = (&mut draw.fx, &copy.fx_words) {
            let drawn: Vec<u32> = order
                .iter()
                .map(|&i| words.get(i as usize).copied().unwrap_or(0))
                .collect();
            let bytes: &[u8] = bytemuck::cast_slice(&drawn);
            written += bytes.len() as u64;
            fx.stream = frame_buffer(
                "sprite_fx_stream",
                viewport_lib::gpu::BufferUsages::VERTEX,
                super::effects::FX_STRIDE,
                bytes,
            );
        }
        if let Some(k) = &keys {
            draw.painter_keys = drawn_keys(k, &order);
        }
        draw.instance_order = instance_order(&order);
        Some((draw, written))
    }
}

/// Build a batch's group-1 bind group over its uniform, texture and records.
///
/// Shared by the upload, the texture rebind and the grow, so the three cannot
/// disagree about the layout.
fn build_bind_group(
    device: &viewport_lib::gpu::Device,
    binds: &SpriteBindings,
    uniform_buf: &viewport_lib::gpu::Buffer,
    sprites: &viewport_lib::gpu::Buffer,
) -> viewport_lib::gpu::BindGroup {
    device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
        label: Some("sprite_bind_group"),
        layout: &binds.bgl,
        entries: &[
            viewport_lib::gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            viewport_lib::gpu::BindGroupEntry {
                binding: 1,
                resource: viewport_lib::gpu::BindingResource::TextureView(&binds.texture_view),
            },
            viewport_lib::gpu::BindGroupEntry {
                binding: 2,
                resource: viewport_lib::gpu::BindingResource::Sampler(&binds.sampler),
            },
            viewport_lib::gpu::BindGroupEntry {
                binding: 3,
                resource: sprites.as_entire_binding(),
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// The two stores, and the uploads that fill them
// ---------------------------------------------------------------------------

/// Slotted store of pre-uploaded sprite batches: static billboards such as
/// foliage, signage and light flares.
pub(super) type SpriteSetStore =
    viewport_lib::resources::handle::SlotStore<StoredSprite, SpriteSetId>;

/// Slotted store of pre-uploaded sprite instance sets: entity sprites such as
/// NPCs, item drops and damage numbers. Same payload, separate handle space.
pub(super) type SpriteInstanceSetStore =
    viewport_lib::resources::handle::SlotStore<StoredSprite, SpriteInstanceSetId>;

impl viewport_lib::resources::handle::GpuByteSize for StoredSprite {
    /// Reserved capacity included: headroom holds VRAM whether or not it draws.
    fn gpu_bytes(&self) -> u64 {
        self.positions.allocated_bytes()
            + self.sprites.allocated_bytes()
            + self.fx_stream.as_ref().map_or(0, |s| s.allocated_bytes())
            + self.draw.fx.as_ref().map_or(0, |fx| fx.uniform_buf.size())
            + self.draw._uniform_buf.size()
    }
}

/// GPU data for one sprite batch: the inline items build it each frame, the
/// two stores hold it across frames.
#[derive(Clone)]
pub(crate) struct SpriteGpuData {
    /// Object-level pick id shared by every instance in the batch (from the
    /// item's `settings.pick_id`); `PickId::NONE` when not pickable.
    pub(crate) pick_id: viewport_lib::PickId,
    /// The item's `pick_alpha_threshold`.
    pub(crate) pick_alpha_threshold: Option<f32>,
    /// Position vertex buffer: one `vec3` per sprite, instance-stepped.
    pub(crate) vertex_buffer: viewport_lib::gpu::Buffer,
    /// Number of sprites (= draw instance count).
    pub(crate) sprite_count: u32,
    /// Bind group (group 1): uniform + texture + sampler + instance storage buffer.
    pub(crate) bind_group: viewport_lib::gpu::BindGroup,
    /// Albedo texture this batch's bind group holds, when it names one. Kept so
    /// a stored batch can tell whether the texture is still resident and be
    /// rebound if it is not.
    pub(crate) texture_id: Option<viewport_lib::resources::TextureId>,
    /// Normal map this batch's group-3 bind group holds, on the same terms.
    pub(crate) normal_texture_id: Option<viewport_lib::resources::TextureId>,
    /// The uniform block as written, so a rebind can flip the `has_*` flags
    /// without rebuilding the batch from an item it no longer has.
    pub(crate) uniform: SpriteUniformData,
    /// Whether this batch writes depth: `depth_write: true` and not in
    /// painter's order.
    pub(crate) depth_write: bool,
    /// Set when the batch draws in painter's order.
    pub(crate) painter: Option<PainterOrder>,
    /// For a painter batch with sort keys: the key of each drawn sprite,
    /// ascending. `None` when every key is `0.0`.
    pub(crate) painter_keys: Option<std::sync::Arc<[f32]>>,
    /// For a painter batch whose sort keys reordered it: the caller's index
    /// of each drawn sprite. `None` when drawn and caller order agree.
    pub(crate) instance_order: Option<std::sync::Arc<[u32]>>,
    /// The filter the batch's sampler was resolved for, kept for rebinds.
    pub(crate) filter: viewport_lib::TextureFilter,
    /// Blend mode requested by the host for this batch.
    pub(crate) blend: viewport_lib::renderer::SpriteBlend,
    /// When true, skip the billboard draw; the wireframe overlay polyline is rendered instead.
    pub(crate) wireframe: bool,
    /// Refractive distortion strength in NDC pixels; `0.0` means a regular
    /// sprite. Routes the draw through the sprite refraction post-pass
    /// instead of the normal sprite pass.
    pub(crate) refraction_strength: f32,
    /// When true, this batch was submitted with `SpriteItem::lit = true` and
    /// is drawn through the lit sprite pipeline.
    pub(crate) lit: bool,
    /// Group 3 bind group for the lit normal-map binding. Always populated for
    /// lit batches: a fallback texture is bound when no normal map is supplied
    /// so the same pipeline layout is honoured.
    pub(crate) lit_normal_bg: Option<viewport_lib::gpu::BindGroup>,
    /// True when this batch qualifies for true (weighted-blended) OIT
    /// instead of ordinary alpha blending: `blend` is `AlphaBlend` or
    /// `Premultiplied` (not `Additive`, which is already order-independent
    /// at the GPU blend-state level), `depth_write` is `false` (OIT
    /// pipelines never write depth), and neither soft-particle fade nor
    /// refractive distortion is active -- both need to sample a resolved
    /// scene texture mid-fragment (depth for soft-particle, colour for
    /// refraction), which the OIT pass exposes for neither. Read by the
    /// sprite item type to route the batch through the OIT pass instead of the
    /// ordinary sprite draws.
    pub(crate) oit_eligible: bool,
    /// The effect draw state, on a batch with effects.
    pub(crate) fx: Option<SpriteFxDraw>,
    // Keep buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: viewport_lib::gpu::Buffer,
}

/// One effect batch's draw state.
#[derive(Clone)]
pub(crate) struct SpriteFxDraw {
    /// The pipeline family: 0 for the built-in effects, `1 + i` for material `i`.
    pub(crate) family: u32,
    /// Group 3: the normal map, the effect block and the effect textures.
    pub(crate) bind_group: viewport_lib::gpu::BindGroup,
    /// The packed effect words, one per sprite, instance-stepped at slot 1.
    pub(crate) stream: viewport_lib::gpu::Buffer,
    pub(crate) uniform_buf: viewport_lib::gpu::Buffer,
    /// The effects the batch was built from, for a rebind.
    pub(crate) effects: std::sync::Arc<super::types::SpriteEffects>,
}

/// One of a stored batch's writable channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpriteChannel {
    Positions,
    Sprites,
    Effects,
    SortKeys,
}

/// Where a painter batch sits in the painter sequence and how it tests depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PainterOrder {
    pub(crate) layer: u32,
    pub(crate) depth_test: bool,
}

impl SpriteGpuData {
    /// Whether the batch's sizes are screen-space points.
    pub(crate) fn screen_sized(&self) -> bool {
        self.uniform.world_space == 0
    }

    /// The drawn index of the caller's sprite `index`.
    pub(crate) fn drawn_index(&self, index: u32) -> Option<u32> {
        match &self.instance_order {
            None => (index < self.sprite_count).then_some(index),
            Some(order) => order.iter().position(|&i| i == index).map(|d| d as u32),
        }
    }

    /// The caller's index of drawn sprite `drawn`.
    pub(crate) fn source_index(&self, drawn: u32) -> u32 {
        match &self.instance_order {
            None => drawn,
            Some(order) => order.get(drawn as usize).copied().unwrap_or(drawn),
        }
    }
}

/// The order a painter batch draws its `count` sprites in: ascending sort key,
/// ties and sprites past the end of `keys` (which sort as `0.0`) in caller
/// order. `None` when that is the caller's order already.
pub(crate) fn painter_instance_order(keys: &[f32], count: usize) -> Option<Vec<u32>> {
    if keys.is_empty() {
        return None;
    }
    let key = |i: u32| keys.get(i as usize).copied().unwrap_or(0.0);
    let mut order: Vec<u32> = (0..count as u32).collect();
    order.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
    order
        .iter()
        .enumerate()
        .any(|(d, &i)| d as u32 != i)
        .then_some(order)
}

/// One painter batch's input to [`painter_pieces`].
pub(crate) struct PainterInput<'a> {
    pub(crate) layer: u32,
    pub(crate) count: u32,
    /// The drawn sprites' keys, ascending; `None` when all are `0.0`.
    pub(crate) keys: Option<&'a [f32]>,
}

/// A contiguous range of one painter batch's drawn sprites that share one
/// key. Consecutive pieces of one batch that the sequence keeps together draw
/// as one call.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PainterPiece {
    /// Index of the batch in the input.
    pub(crate) batch: usize,
    pub(crate) instances: std::ops::Range<u32>,
    pub(crate) layer: u32,
    /// The key every sprite of the piece shares.
    pub(crate) key: f32,
}

/// The order painter batches draw in, split into pieces of one key each so
/// that every sprite takes its place in the painter sequence by its own key:
/// ascending layer, then a merge of that layer's batches by key, ties to the
/// earlier submission. A batch also splits where another batch's key falls
/// strictly between two of its own. Empty batches draw nothing and get no
/// piece.
pub(crate) fn painter_pieces(batches: &[PainterInput<'_>]) -> Vec<PainterPiece> {
    let key = |b: usize, i: u32| {
        batches[b]
            .keys
            .and_then(|k| k.get(i as usize).copied())
            .unwrap_or(0.0)
    };
    // (key, submission) ascending; total so NaN keys still order.
    let before = |a: (f32, usize), b: (f32, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).is_lt();
    let mut layers: Vec<u32> = batches.iter().map(|b| b.layer).collect();
    layers.sort_unstable();
    layers.dedup();
    let mut pieces = Vec::new();
    let mut heads = vec![0u32; batches.len()];
    for layer in layers {
        let members: Vec<usize> = (0..batches.len())
            .filter(|&b| batches[b].layer == layer && batches[b].count > 0)
            .collect();
        loop {
            let live = members
                .iter()
                .copied()
                .filter(|&b| heads[b] < batches[b].count);
            let Some(first) = live.clone().min_by(|&a, &b| {
                key(a, heads[a])
                    .total_cmp(&key(b, heads[b]))
                    .then(a.cmp(&b))
            }) else {
                break;
            };
            // The next head of any other batch bounds how far this one runs.
            let bound = live
                .filter(|&b| b != first)
                .map(|b| (key(b, heads[b]), b))
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let start = heads[first];
            let mut end = start + 1;
            // A piece holds one key: the sequence places it by that key
            // against every other painter draw.
            while end < batches[first].count
                && key(first, end).total_cmp(&key(first, start)).is_eq()
                && bound.is_none_or(|o| before((key(first, end), first), o))
            {
                end += 1;
            }
            pieces.push(PainterPiece {
                batch: first,
                instances: start..end,
                layer,
                key: key(first, start),
            });
            heads[first] = end;
        }
    }
    pieces
}

/// Whether every texture one stored batch's bind groups name is still
/// uploaded. The renderer's resource epoch says when to ask; this says what a
/// batch bound.
pub(super) fn textures_resident(
    stored: &StoredSprite,
    resources: &viewport_lib::resources::DeviceResources,
) -> bool {
    let data = &stored.draw;
    let fx = data
        .fx
        .iter()
        .flat_map(|fx| super::effects::texture_ids(&fx.effects));
    [data.texture_id, data.normal_texture_id]
        .into_iter()
        .flatten()
        .chain(fx)
        .all(|id| resources.has_texture(id))
}

/// Point a stored batch's bind groups at freshly resolved views, keeping its
/// buffers and its handle.
///
/// A batch is uploaded once and drawn for as long as the host holds the
/// handle, so the views it bound at upload can be freed or swapped underneath
/// it. Rebuilding is the only correct answer: dropping the batch would lose
/// content the host still owns, and leaving it alone would keep a freed
/// texture alive and keep sampling it.
pub(super) fn rebind_sprite(
    device: &viewport_lib::gpu::Device,
    queue: &viewport_lib::gpu::Queue,
    binds: &SpriteBindings,
    stored: &mut StoredSprite,
) {
    let data = &mut stored.draw;
    data.uniform.has_texture = binds.has_texture;
    data.uniform.has_normal_map = binds.has_normal_map;
    queue.write_buffer(&data._uniform_buf, 0, bytemuck::bytes_of(&data.uniform));

    data.bind_group = build_bind_group(device, binds, &data._uniform_buf, stored.sprites.buffer());
    // The batch keeps the views it just rebound, so a later grow rebuilds the
    // group against the same ones rather than the ones it was uploaded with.
    stored.binds = binds.clone();

    if let (Some(fx), Some(fxb)) = (&mut data.fx, &binds.fx) {
        queue.write_buffer(&fx.uniform_buf, 0, bytemuck::bytes_of(&fxb.uniform));
        fx.bind_group = super::effects::bind_group(
            device,
            fxb,
            &binds.normal_view,
            &binds.sampler,
            &fx.uniform_buf,
        );
    }

    // Only a lit batch has the group-3 binding; an unlit one never built it
    // and the normal map it named, if any, is bound nowhere.
    if data.lit_normal_bg.is_some() {
        data.lit_normal_bg = Some(device.create_bind_group(
            &viewport_lib::gpu::BindGroupDescriptor {
                label: Some("sprite_lit_normal_bg"),
                layout: &binds.lit_bgl,
                entries: &[
                    viewport_lib::gpu::BindGroupEntry {
                        binding: 0,
                        resource: viewport_lib::gpu::BindingResource::TextureView(
                            &binds.normal_view,
                        ),
                    },
                    viewport_lib::gpu::BindGroupEntry {
                        binding: 1,
                        resource: viewport_lib::gpu::BindingResource::Sampler(&binds.sampler),
                    },
                ],
            },
        ));
    }
}

#[cfg(test)]
mod painter_tests {
    use super::*;

    #[test]
    fn instances_sort_by_key_and_keep_ties_in_caller_order() {
        let keys = [3.0, -1.0, 3.0, 0.5];
        assert_eq!(painter_instance_order(&keys, 4), Some(vec![1, 3, 0, 2]));
        // Missing keys sort as zero.
        assert_eq!(painter_instance_order(&[1.0], 3), Some(vec![1, 2, 0]));
    }

    #[test]
    fn instances_already_in_order_need_no_permutation() {
        assert_eq!(painter_instance_order(&[], 5), None);
        assert_eq!(painter_instance_order(&[0.0, 1.0, 2.0], 3), None);
    }

    fn input(layer: u32, keys: Option<&[f32]>, count: u32) -> PainterInput<'_> {
        PainterInput { layer, count, keys }
    }

    #[test]
    fn pieces_interleave_batches_on_one_layer_by_key() {
        let crowd = [-3.0, -1.0, 1.0, 3.0];
        let one = [0.0];
        let pieces = painter_pieces(&[input(0, Some(&crowd), 4), input(0, Some(&one), 1)]);
        let got: Vec<(usize, std::ops::Range<u32>)> = pieces
            .iter()
            .map(|p| (p.batch, p.instances.clone()))
            .collect();
        assert_eq!(
            got,
            vec![(0, 0..1), (0, 1..2), (1, 0..1), (0, 2..3), (0, 3..4)]
        );
        assert_eq!(pieces[3].key, 1.0);
    }

    #[test]
    fn pieces_split_at_every_key_change_and_keep_equal_keys_together() {
        let keys = [1.0, 1.0, 2.0, 2.0, 2.0, 5.0];
        let pieces = painter_pieces(&[input(0, Some(&keys), 6)]);
        let got: Vec<(std::ops::Range<u32>, f32)> = pieces
            .iter()
            .map(|p| (p.instances.clone(), p.key))
            .collect();
        assert_eq!(got, vec![(0..2, 1.0), (2..5, 2.0), (5..6, 5.0)]);
    }

    #[test]
    fn pieces_keep_layers_and_whole_batches_without_overlap() {
        let low = [0.0, 1.0];
        let high = [5.0, 6.0];
        let pieces = painter_pieces(&[
            input(1, Some(&high), 2),
            input(1, Some(&low), 2),
            input(0, None, 3),
            input(0, None, 0),
        ]);
        let got: Vec<(usize, std::ops::Range<u32>)> = pieces
            .iter()
            .map(|p| (p.batch, p.instances.clone()))
            .collect();
        assert_eq!(
            got,
            vec![(2, 0..3), (1, 0..1), (1, 1..2), (0, 0..1), (0, 1..2)]
        );
        // Equal keys keep submission order and do not split.
        let ties = painter_pieces(&[input(0, None, 2), input(0, None, 2)]);
        assert_eq!(ties.len(), 2);
        assert_eq!(ties[0].batch, 0);
    }

    #[test]
    fn layer_wins_over_key() {
        // A batch on a lower layer draws first whatever its sprites' keys:
        // the key orders sprites within a batch only.
        let back = painter_instance_order(&[9.0, 8.0], 2).unwrap();
        let front = painter_instance_order(&[-9.0, -8.0], 2);
        assert_eq!(back, vec![1, 0]);
        assert_eq!(front, None);
    }

    #[test]
    fn pivot_packs_centre_to_zero_and_corners_exactly() {
        assert_eq!(pack_pivot([0.5, 0.5]), 0);
        assert_eq!(pack_pivot([0.0, 1.0]), (0x8001u32) | (0x7fff << 16));
    }
}
