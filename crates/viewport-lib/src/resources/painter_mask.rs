//! Painter masks: a viewport-sized `Rgba8Unorm` target with one coverage
//! channel per slot, drawn before the painter sequence by the item types that
//! report masks, and read by painter content that tests one.
//!
//! The layouts and fallbacks live here, built on every tier at startup; each
//! viewport's target, uniform and bind groups live in its slot as a
//! [`PainterMaskState`].

use crate::resources::builders::LoggedAlloc;

/// The mask target's format: one coverage channel per slot.
pub const PAINTER_MASK_FORMAT: crate::gpu::TextureFormat = crate::gpu::TextureFormat::Rgba8Unorm;

/// The blend a mask draw uses: each channel keeps the larger of what is there
/// and what is drawn, so a draw that writes coverage into its slot's channel
/// and zero elsewhere leaves the other slots alone and overlapping masks in
/// one slot form their union.
pub const PAINTER_MASK_BLEND: crate::gpu::BlendState = crate::gpu::BlendState {
    color: crate::gpu::BlendComponent {
        src_factor: crate::gpu::BlendFactor::One,
        dst_factor: crate::gpu::BlendFactor::One,
        operation: crate::gpu::BlendOperation::Max,
    },
    alpha: crate::gpu::BlendComponent {
        src_factor: crate::gpu::BlendFactor::One,
        dst_factor: crate::gpu::BlendFactor::One,
        operation: crate::gpu::BlendOperation::Max,
    },
};

/// Bytes of the `PainterMasks` uniform: four `vec4<u32>`.
pub(crate) const PAINTER_MASK_UNIFORM_BYTES: u64 = 64;

/// The painter mask layout and the fallbacks a frame without masks binds.
pub(crate) struct PainterMaskResources {
    /// Binding 0 the mask texture, binding 1 the `PainterMasks` uniform.
    pub(crate) bgl: crate::gpu::BindGroupLayout,
    /// A 1x1 mask with every slot empty.
    pub(crate) fallback_view: crate::gpu::TextureView,
    /// A 1x1 depth for the depth-read group of a pass whose depth attachment
    /// is writable.
    pub(crate) depth_fallback_view: crate::gpu::TextureView,
    /// A group over the fallback and a uniform naming no slot, which cuts
    /// nothing.
    pub(crate) empty_group: crate::gpu::BindGroup,
}

/// The entries of [`PainterMaskResources::bgl`], starting at `first`: also the
/// last two entries of the depth-read layout.
pub(crate) fn painter_mask_entries(first: u32) -> [crate::gpu::BindGroupLayoutEntry; 2] {
    [
        crate::gpu::BindGroupLayoutEntry {
            binding: first,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Texture {
                sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
                view_dimension: crate::gpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        crate::gpu::BindGroupLayoutEntry {
            binding: first + 1,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(PAINTER_MASK_UNIFORM_BYTES),
            },
            count: None,
        },
    ]
}

impl PainterMaskResources {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("painter_mask_bgl"),
            entries: &painter_mask_entries(0),
        });
        let one = crate::gpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        // wgpu zero-fills a new texture, so the fallback mask is empty in
        // every slot without a write.
        let fallback = device.logged_texture(&crate::gpu::TextureDescriptor {
            label: Some("painter_mask_fallback"),
            size: one,
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: PAINTER_MASK_FORMAT,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = device.logged_texture(&crate::gpu::TextureDescriptor {
            label: Some("painter_mask_depth_fallback"),
            size: one,
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth32Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                | crate::gpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let fallback_view = fallback.create_view(&Default::default());
        // wgpu zero-fills a new buffer: no slot has a mask.
        let none = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("painter_mask_none"),
            size: PAINTER_MASK_UNIFORM_BYTES,
            usage: crate::gpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let empty_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("painter_mask_empty_bg"),
            layout: &bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&fallback_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: none.as_entire_binding(),
                },
            ],
        });
        Self {
            bgl,
            empty_group,
            fallback_view,
            depth_fallback_view: depth.create_view(&crate::gpu::TextureViewDescriptor {
                aspect: crate::gpu::TextureAspect::DepthOnly,
                ..Default::default()
            }),
        }
    }
}

/// The slots' layer ranges as the `PainterMasks` uniform holds them: per slot
/// the first layer, the last, and 1 when the slot has a mask.
pub(crate) fn painter_mask_uniform(slots: &[Option<(u32, u32)>; 4]) -> [u32; 16] {
    let mut out = [0u32; 16];
    for (i, slot) in slots.iter().enumerate() {
        if let Some((lo, hi)) = slot {
            out[i * 4] = *lo;
            out[i * 4 + 1] = *hi;
            out[i * 4 + 2] = 1;
        }
    }
    out
}

/// One viewport's painter masks: the target (made the first frame a mask
/// draws), the uniform, and the bind groups over them.
pub(crate) struct PainterMaskState {
    /// The mask target and its size, once a mask has drawn.
    target: Option<(crate::gpu::Texture, crate::gpu::TextureView, [u32; 2])>,
    pub(crate) uniform: crate::gpu::Buffer,
    /// Whether this frame's mask target holds masks; when not, the groups
    /// bind the fallback.
    pub(crate) active: bool,
    /// Laid out as the painter mask layout.
    group: crate::resources::cached_bind_group::CachedBindGroup<crate::gpu::TextureView>,
    /// Laid out as the depth-read layout over the depth fallback, for the LDR
    /// scene pass. Full tier only.
    ldr_depth_group: crate::resources::cached_bind_group::CachedBindGroup<crate::gpu::TextureView>,
}

impl PainterMaskState {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let uniform = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("painter_mask_uniform"),
            size: PAINTER_MASK_UNIFORM_BYTES,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            target: None,
            uniform,
            active: false,
            group: crate::resources::cached_bind_group::CachedBindGroup::new(),
            ldr_depth_group: crate::resources::cached_bind_group::CachedBindGroup::new(),
        }
    }

    /// The mask target, made or remade at `size` device pixels.
    pub(crate) fn target(
        &mut self,
        device: &crate::gpu::Device,
        size: [u32; 2],
    ) -> &crate::gpu::TextureView {
        let size = [size[0].max(1), size[1].max(1)];
        if self.target.as_ref().is_none_or(|(_, _, s)| *s != size) {
            let texture = device.logged_texture(&crate::gpu::TextureDescriptor {
                label: Some("painter_mask_target"),
                size: crate::gpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: PAINTER_MASK_FORMAT,
                usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                    | crate::gpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.target = Some((texture, view, size));
        }
        &self.target.as_ref().expect("made above").1
    }

    /// The view the groups bind this frame: the target when masks drew, the
    /// fallback otherwise.
    pub(crate) fn bound_view<'a>(
        &'a self,
        resources: &'a PainterMaskResources,
    ) -> &'a crate::gpu::TextureView {
        match (&self.target, self.active) {
            (Some((_, view, _)), true) => view,
            _ => &resources.fallback_view,
        }
    }

    /// The group laid out as the painter mask layout.
    pub(crate) fn bind_group(
        &mut self,
        device: &crate::gpu::Device,
        resources: &PainterMaskResources,
    ) -> crate::gpu::BindGroup {
        let view = self.bound_view(resources).clone();
        let uniform = self.uniform.clone();
        self.group.get(view.clone(), || {
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("painter_mask_bg"),
                layout: &resources.bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(&view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            })
        })
    }

    /// The group laid out as the depth-read layout, over `depth` and this
    /// viewport's masks.
    pub(crate) fn depth_read_group(
        &self,
        device: &crate::gpu::Device,
        resources: &PainterMaskResources,
        layout: &crate::gpu::BindGroupLayout,
        sampler: &crate::gpu::Sampler,
        depth: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        depth_read_group(
            device,
            layout,
            sampler,
            depth,
            self.bound_view(resources),
            &self.uniform,
        )
    }

    /// The depth-read group over the depth fallback, for the LDR scene pass.
    pub(crate) fn ldr_depth_read_group(
        &mut self,
        device: &crate::gpu::Device,
        resources: &PainterMaskResources,
        layout: &crate::gpu::BindGroupLayout,
        sampler: &crate::gpu::Sampler,
    ) -> crate::gpu::BindGroup {
        let view = self.bound_view(resources).clone();
        let uniform = self.uniform.clone();
        self.ldr_depth_group.get(view.clone(), || {
            depth_read_group(
                device,
                layout,
                sampler,
                &resources.depth_fallback_view,
                &view,
                &uniform,
            )
        })
    }

    /// The groups as the last prepare left them, for passes that cannot
    /// build one.
    pub(crate) fn current(
        &self,
    ) -> (
        Option<&crate::gpu::BindGroup>,
        Option<&crate::gpu::BindGroup>,
    ) {
        (self.group.current(), self.ldr_depth_group.current())
    }
}

/// A group laid out as the depth-read layout: `depth` and `sampler`, then the
/// painter masks `mask` and `uniform`.
pub(crate) fn depth_read_group(
    device: &crate::gpu::Device,
    layout: &crate::gpu::BindGroupLayout,
    sampler: &crate::gpu::Sampler,
    depth: &crate::gpu::TextureView,
    mask: &crate::gpu::TextureView,
    uniform: &crate::gpu::Buffer,
) -> crate::gpu::BindGroup {
    device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some("plugin_depth_read_bg"),
        layout,
        entries: &[
            crate::gpu::BindGroupEntry {
                binding: 0,
                resource: crate::gpu::BindingResource::TextureView(depth),
            },
            crate::gpu::BindGroupEntry {
                binding: 1,
                resource: crate::gpu::BindingResource::Sampler(sampler),
            },
            crate::gpu::BindGroupEntry {
                binding: 2,
                resource: crate::gpu::BindingResource::TextureView(mask),
            },
            crate::gpu::BindGroupEntry {
                binding: 3,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}
