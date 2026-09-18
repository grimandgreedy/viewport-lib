//! The sprite batches this item type holds on the consumer's behalf, and the
//! per-frame GPU data every sprite draw is built from.
//!
//! A `SpriteItem` carries its billboards and is rebuilt each frame; a
//! `SpriteSetRefItem` or `SpriteInstanceSetRefItem` names a batch uploaded once
//! through the `*_sprite_set` / `*_sprite_instance_set` methods on
//! [`ViewportRenderer`](crate::renderer::ViewportRenderer). All three end up as
//! the same [`SpriteGpuData`], which is why the builder is shared.
//!
//! The two bind group layouts live here rather than with the pipelines, because
//! an upload builds its bind groups against them and an upload can arrive long
//! before the first frame that draws one.

use crate::resources::DeviceResources;

pub(crate) use super::types::{SpriteInstanceSetId, SpriteSetId};

/// The bind group layouts sprite uploads build against.
///
/// Group 1 carries a batch's uniform, texture, sampler and instance buffer;
/// group 3 carries the optional tangent-space normal map for a lit batch.
/// Both are built once at startup rather than lazily, because an upload can
/// arrive before the first frame that draws one.
pub(super) struct SpriteLayouts {
    /// Bind group layout for sprite uniforms + texture + instance buffer (group 1).
    pub(super) bgl: crate::gpu::BindGroupLayout,
    /// Group 3 layout for the optional lit normal map (texture + sampler).
    pub(super) lit_bgl: crate::gpu::BindGroupLayout,
}

impl SpriteLayouts {
    pub(super) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("sprite_bgl"),
            entries: &[
                // binding 0: SpriteUniform (model, world_space, has_texture)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: sprite texture (or fallback 1x1 when has_texture == 0)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: sampler
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: per-sprite instance storage buffer
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let lit_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "sprite_lit_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        Self { bgl, lit_bgl }
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
    texture_view: crate::gpu::TextureView,
    has_texture: u32,
    normal_view: crate::gpu::TextureView,
    has_normal_map: u32,
    sampler: crate::gpu::Sampler,
    bgl: crate::gpu::BindGroupLayout,
    lit_bgl: crate::gpu::BindGroupLayout,
}

/// Resolve the albedo and normal-map views an item names, reporting either
/// through the lib's texture-slot check when it was uploaded in the wrong
/// colour space. A missing or stale id binds the neutral fallback for the slot
/// and clears the shader's `has_*` flag.
pub(super) fn resolve_bindings(
    resources: &DeviceResources,
    layouts: &SpriteLayouts,
    item: &crate::renderer::SpriteItem,
) -> SpriteBindings {
    use crate::resources::TextureSlot;
    resources.check_texture_slot(item.texture_id, TextureSlot::SpriteAlbedo);
    resources.check_texture_slot(item.normal_texture_id, TextureSlot::SpriteNormalMap);

    let (texture_view, has_texture) =
        match item.texture_id.and_then(|id| resources.texture_view(id)) {
            Some(view) => (view.clone(), 1),
            None => (resources.fallback_colourmap_view().clone(), 0),
        };
    let (normal_view, has_normal_map) = match item
        .normal_texture_id
        .and_then(|id| resources.texture_view(id))
    {
        Some(view) => (view.clone(), 1),
        None => (
            resources
                .fallback_texture_view(crate::scene::material::TextureSlot::Normal)
                .clone(),
            0,
        ),
    };
    SpriteBindings {
        texture_view,
        has_texture,
        normal_view,
        has_normal_map,
        sampler: resources.material_sampler().clone(),
        bgl: layouts.bgl.clone(),
        lit_bgl: layouts.lit_bgl.clone(),
    }
}

/// Build the GPU data for one sprite batch: its buffers and its bind groups.
///
/// Shared by the per-frame item path and the two stores, so a reference draw
/// and an inline draw are bit-for-bit the same work.
pub(super) fn build_sprite(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    binds: &SpriteBindings,
    item: &crate::renderer::SpriteItem,
) -> SpriteGpuData {
    {
        let count = item.positions.len() as u32;

        // Position vertex buffer (one vec3 per sprite, instance-stepped).
        let pos_bytes: Vec<u8> = item
            .positions
            .iter()
            .flat_map(|p| bytemuck::bytes_of(p).iter().copied())
            .collect();
        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("sprite_vertex_buf"),
            size: pos_bytes.len().max(12) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &pos_bytes);

        // Per-instance storage buffer: build by zipping item vecs with defaults.
        // Layout matches `SpriteInstance` in `sprite.wgsl`. 64 bytes per instance.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuSpriteInstance {
            colour: [f32; 4],
            size: f32,
            rotation: f32,
            soft_distance: f32,
            _pad1: f32,
            uv_rect: [f32; 4],
            velocity: [f32; 3],
            _pad2: f32,
        }

        let instances: Vec<GpuSpriteInstance> = (0..item.positions.len())
            .map(|i| GpuSpriteInstance {
                colour: if i < item.colours.len() {
                    item.colours[i].to_linear_rgba()
                } else {
                    item.default_colour.to_linear_rgba()
                },
                size: if i < item.sizes.len() {
                    item.sizes[i]
                } else {
                    item.default_size
                },
                rotation: if i < item.rotations.len() {
                    item.rotations[i]
                } else {
                    0.0
                },
                soft_distance: if i < item.soft_particle_distances.len() {
                    item.soft_particle_distances[i].max(0.0)
                } else {
                    0.0
                },
                _pad1: 0.0,
                uv_rect: if i < item.uv_rects.len() {
                    item.uv_rects[i]
                } else {
                    [0.0, 0.0, 1.0, 1.0]
                },
                velocity: if i < item.velocities.len() {
                    item.velocities[i]
                } else {
                    [0.0, 0.0, 0.0]
                },
                _pad2: 0.0,
            })
            .collect();

        let instance_bytes = bytemuck::cast_slice(&instances);
        let instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("sprite_instance_buf"),
            size: instance_bytes.len().max(48) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, instance_bytes);

        // Uniform buffer: model matrix + flags + soft-particle distance + orientation
        // + refraction strength + lit parameters. Layout mirrors `SpriteUniform`
        // in `sprite_lit.wgsl`; the emissive `sprite.wgsl` reads only the first
        // half and ignores the trailing lit fields.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SpriteUniformData {
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
            _pad_lit_b: u32,
            _pad_lit_c: u32,
        }

        let (texture_view, has_texture) = (&binds.texture_view, binds.has_texture);

        let orientation = match item.orientation {
            crate::renderer::SpriteOrientation::CameraFacing => 0u32,
            crate::renderer::SpriteOrientation::VelocityStretched => 1u32,
            crate::renderer::SpriteOrientation::AxisLocked => 2u32,
        };

        let normal_mode = match item.lit_params.normal_mode {
            crate::renderer::SpriteNormalMode::Spherical => 0u32,
            crate::renderer::SpriteNormalMode::Flat => 1u32,
            crate::renderer::SpriteNormalMode::NormalMap => 2u32,
        };

        let (normal_view, has_normal_map) = (&binds.normal_view, binds.has_normal_map);

        let uniform_data = SpriteUniformData {
            model: item.model,
            world_space: if item.size_mode == crate::renderer::SpriteSizeMode::WorldSpace {
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
            _pad_lit_b: 0,
            _pad_lit_c: 0,
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("sprite_uniform_buf"),
            size: std::mem::size_of::<SpriteUniformData>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = &binds.bgl;

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("sprite_bind_group"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(texture_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&binds.sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_buf.as_entire_binding(),
                },
            ],
        });

        let lit_normal_bg = if item.lit {
            Some({
                let lit_bgl = &binds.lit_bgl;
                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("sprite_lit_normal_bg"),
                    layout: lit_bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: crate::gpu::BindingResource::TextureView(normal_view),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::Sampler(&binds.sampler),
                        },
                    ],
                })
            })
        } else {
            None
        };

        let oit_eligible = matches!(
            item.blend,
            crate::renderer::SpriteBlend::AlphaBlend | crate::renderer::SpriteBlend::Premultiplied
        ) && !item.depth_write
            && item.soft_particle_distance.is_none_or(|d| d <= 0.0)
            && item.soft_particle_distances.is_empty()
            && item.refraction_strength.is_none_or(|s| s <= 0.0);

        SpriteGpuData {
            vertex_buffer,
            sprite_count: count,
            pick_id: item.settings.pick_id,
            bind_group,
            depth_write: item.depth_write,
            blend: item.blend,
            wireframe: false,
            refraction_strength: item.refraction_strength.filter(|s| *s > 0.0).unwrap_or(0.0),
            lit: item.lit,
            lit_normal_bg,
            oit_eligible,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }
}

// ---------------------------------------------------------------------------
// The two stores, and the uploads that fill them
// ---------------------------------------------------------------------------

/// Slotted store of pre-uploaded sprite batches: static billboards such as
/// foliage, signage and light flares.
pub(super) type SpriteSetStore = crate::resources::handle::SlotStore<SpriteGpuData, SpriteSetId>;

/// Slotted store of pre-uploaded sprite instance sets: entity sprites such as
/// NPCs, item drops and damage numbers. Same payload, separate handle space.
pub(super) type SpriteInstanceSetStore =
    crate::resources::handle::SlotStore<SpriteGpuData, SpriteInstanceSetId>;

impl crate::resources::handle::GpuByteSize for SpriteGpuData {
    fn gpu_bytes(&self) -> u64 {
        self.vertex_buffer.size() + self._uniform_buf.size() + self._instance_buf.size()
    }
}

/// GPU data for one sprite batch: the inline items build it each frame, the
/// two stores hold it across frames.
#[derive(Clone)]
pub(crate) struct SpriteGpuData {
    /// Object-level pick id shared by every instance in the batch (from the
    /// item's `settings.pick_id`); `PickId::NONE` when not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Position vertex buffer: one `vec3` per sprite, instance-stepped.
    pub(crate) vertex_buffer: crate::gpu::Buffer,
    /// Number of sprites (= draw instance count).
    pub(crate) sprite_count: u32,
    /// Bind group (group 1): uniform + texture + sampler + instance storage buffer.
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Whether this batch was submitted with `depth_write: true`.
    pub(crate) depth_write: bool,
    /// Blend mode requested by the host for this batch.
    pub(crate) blend: crate::renderer::SpriteBlend,
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
    pub(crate) lit_normal_bg: Option<crate::gpu::BindGroup>,
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
    // Keep buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) _instance_buf: crate::gpu::Buffer,
}
