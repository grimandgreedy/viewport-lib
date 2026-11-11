//! Painter masks and light layers.
//!
//! The masks are a viewport-sized `Rgba8Unorm` target with one coverage
//! channel per slot, drawn before the painter sequence by the item types that
//! report masks, and read by painter content that tests one. The light layers
//! are a plugin's 2D array texture, one array layer per run of painter
//! layers, which lit content samples under its fragment. Both ride on the
//! same groups.
//!
//! The layouts and fallbacks live here, built on every tier at startup; each
//! viewport's target, uniforms and bind groups live in its slot as a
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

/// The most light layer ranges a frame binds; ranges past it are not lit.
pub const MAX_PAINTER_LIGHT_RANGES: usize = 16;

/// Bytes of the `PainterLights` uniform: a header `vec4` and one `vec4<u32>`
/// per range.
pub const PAINTER_LIGHTS_UNIFORM_BYTES: u64 = 16 + 16 * MAX_PAINTER_LIGHT_RANGES as u64;

/// Distance between the entries of the light pick uniform a painter mesh
/// selects with a dynamic offset: the largest uniform offset alignment any
/// backend asks for.
pub(crate) const LIGHT_PICK_STRIDE: u64 = 256;

/// The painter mask layout and the fallbacks a frame without masks binds.
pub(crate) struct PainterMaskResources {
    /// Binding 0 the mask texture, binding 1 the `PainterMasks` uniform,
    /// binding 2 the light layers, binding 3 the `PainterLights` uniform.
    pub(crate) bgl: crate::gpu::BindGroupLayout,
    /// The light layers at binding 2 and a light pick uniform with a dynamic
    /// offset at binding 3, for the renderer's own painter meshes, which
    /// select their range on the CPU.
    pub(crate) light_pick_bgl: crate::gpu::BindGroupLayout,
    /// An empty layout and its group, for the slot before the light pick
    /// group in the painter mesh layout.
    pub(crate) empty_bgl: crate::gpu::BindGroupLayout,
    pub(crate) empty_bg: crate::gpu::BindGroup,
    /// A 1x1 mask with every slot empty.
    pub(crate) fallback_view: crate::gpu::TextureView,
    /// A 1x1 light layer of two array layers (one-layer arrays do not view as
    /// arrays on GL), bound while no plugin reports light layers.
    pub(crate) light_fallback_view: crate::gpu::TextureView,
    /// A 1x1 depth for the depth-read group of a pass whose depth attachment
    /// is writable.
    pub(crate) depth_fallback_view: crate::gpu::TextureView,
    /// A group over the fallbacks and uniforms naming no slot and no range,
    /// which cuts nothing and lights nothing.
    pub(crate) empty_group: crate::gpu::BindGroup,
}

/// The light layers' layout entry at `binding`.
fn light_layers_entry(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::FRAGMENT,
        ty: crate::gpu::BindingType::Texture {
            sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
            view_dimension: crate::gpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

/// The entries of [`PainterMaskResources::bgl`], starting at `first`: also the
/// last four entries of the depth-read layout.
pub(crate) fn painter_mask_entries(first: u32) -> [crate::gpu::BindGroupLayoutEntry; 4] {
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
        light_layers_entry(first + 2),
        crate::gpu::BindGroupLayoutEntry {
            binding: first + 3,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(PAINTER_LIGHTS_UNIFORM_BYTES),
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
        let light_pick_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("painter_light_pick_bgl"),
                entries: &[
                    light_layers_entry(2),
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: std::num::NonZeroU64::new(16),
                        },
                        count: None,
                    },
                ],
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
        let light_fallback = device.logged_texture(&crate::gpu::TextureDescriptor {
            label: Some("painter_light_fallback"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba8Unorm,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let light_fallback_view = light_fallback.create_view(&crate::gpu::TextureViewDescriptor {
            dimension: Some(crate::gpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        // wgpu zero-fills a new buffer: no slot has a mask.
        let none = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("painter_mask_none"),
            size: PAINTER_MASK_UNIFORM_BYTES,
            usage: crate::gpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        // And no light range.
        let no_lights = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("painter_lights_none"),
            size: PAINTER_LIGHTS_UNIFORM_BYTES,
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
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::TextureView(&light_fallback_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: no_lights.as_entire_binding(),
                },
            ],
        });
        let empty_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("painter_empty_bgl"),
            entries: &[],
        });
        let empty_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("painter_empty_bg"),
            layout: &empty_bgl,
            entries: &[],
        });
        Self {
            bgl,
            light_pick_bgl,
            empty_bgl,
            empty_bg,
            empty_group,
            fallback_view,
            light_fallback_view,
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

/// The light layer ranges as the `PainterLights` uniform holds them: the
/// count and the scale, then per range the first layer, the last and the
/// array layer.
pub(crate) fn painter_lights_uniform(ranges: &[[u32; 2]], scale: f32) -> Vec<u32> {
    let count = ranges.len().min(MAX_PAINTER_LIGHT_RANGES);
    let mut out = vec![0u32; PAINTER_LIGHTS_UNIFORM_BYTES as usize / 4];
    out[0] = count as u32;
    out[1] = scale.to_bits();
    for (i, r) in ranges.iter().take(count).enumerate() {
        out[4 + i * 4] = r[0];
        out[4 + i * 4 + 1] = r[1];
        out[4 + i * 4 + 2] = i as u32;
    }
    out
}

/// The light pick uniform: entry 0 picks no array layer, entry `k + 1`
/// array layer `k`, each with the scale.
fn light_pick_uniform(scale: f32) -> Vec<u8> {
    let stride = LIGHT_PICK_STRIDE as usize;
    let mut out = vec![0u8; stride * (MAX_PAINTER_LIGHT_RANGES + 1)];
    for k in 0..=MAX_PAINTER_LIGHT_RANGES {
        let slice = k as i32 - 1;
        out[k * stride..k * stride + 4].copy_from_slice(&slice.to_le_bytes());
        out[k * stride + 4..k * stride + 8].copy_from_slice(&scale.to_le_bytes());
    }
    out
}

/// The array layer of `ranges` that lights painter `layer`, or `None`.
pub(crate) fn light_slice(ranges: &[[u32; 2]], layer: u32) -> Option<usize> {
    ranges
        .iter()
        .take(MAX_PAINTER_LIGHT_RANGES)
        .position(|r| r[0] <= layer && layer <= r[1])
}

/// The light layers a plugin reported for this frame.
pub(crate) struct FrameLights {
    pub(crate) view: crate::gpu::TextureView,
    pub(crate) ranges: Vec<[u32; 2]>,
    pub(crate) scale: f32,
}

/// One viewport's painter masks and light layers: the mask target (made the
/// first frame a mask draws), the uniforms, and the bind groups over them.
pub(crate) struct PainterMaskState {
    /// The mask target and its size, once a mask has drawn.
    target: Option<(crate::gpu::Texture, crate::gpu::TextureView, [u32; 2])>,
    pub(crate) uniform: crate::gpu::Buffer,
    /// Whether this frame's mask target holds masks; when not, the groups
    /// bind the fallback.
    pub(crate) active: bool,
    /// The `PainterLights` uniform.
    pub(crate) light_uniform: crate::gpu::Buffer,
    /// The light pick entries painter meshes select with a dynamic offset.
    light_pick_uniform: crate::gpu::Buffer,
    /// This frame's light layers, when a plugin reported some.
    pub(crate) lights: Option<FrameLights>,
    /// Laid out as the painter mask layout. Keyed on the mask and light views.
    group: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::TextureView,
        crate::gpu::TextureView,
    )>,
    /// Laid out as the depth-read layout over the depth fallback, for passes
    /// whose depth attachment is writable. Full tier only.
    ldr_depth_group: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::TextureView,
        crate::gpu::TextureView,
    )>,
    /// Laid out as the light pick layout.
    light_pick_group: crate::resources::cached_bind_group::CachedBindGroup<crate::gpu::TextureView>,
}

impl PainterMaskState {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let uniform = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("painter_mask_uniform"),
            size: PAINTER_MASK_UNIFORM_BYTES,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_uniform = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("painter_lights_uniform"),
            size: PAINTER_LIGHTS_UNIFORM_BYTES,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_pick_uniform = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("painter_light_pick_uniform"),
            size: LIGHT_PICK_STRIDE * (MAX_PAINTER_LIGHT_RANGES as u64 + 1),
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            target: None,
            uniform,
            active: false,
            light_uniform,
            light_pick_uniform,
            lights: None,
            group: crate::resources::cached_bind_group::CachedBindGroup::new(),
            ldr_depth_group: crate::resources::cached_bind_group::CachedBindGroup::new(),
            light_pick_group: crate::resources::cached_bind_group::CachedBindGroup::new(),
        }
    }

    /// Set this frame's light layers and write their uniforms.
    pub(crate) fn set_lights(&mut self, queue: &crate::gpu::Queue, lights: Option<FrameLights>) {
        let (ranges, scale) = lights
            .as_ref()
            .map_or((&[][..], 1.0), |l| (&l.ranges[..], l.scale));
        queue.write_buffer(
            &self.light_uniform,
            0,
            bytemuck::cast_slice(&painter_lights_uniform(ranges, scale)),
        );
        queue.write_buffer(&self.light_pick_uniform, 0, &light_pick_uniform(scale));
        self.lights = lights;
    }

    /// The light layer ranges this frame, lowest array layer first.
    pub(crate) fn light_ranges(&self) -> &[[u32; 2]] {
        self.lights.as_ref().map_or(&[], |l| &l.ranges[..])
    }

    /// The dynamic offset into the light pick group for content on painter
    /// `layer`, lit unless `unlit`.
    pub(crate) fn light_pick_offset(&self, layer: u32, unlit: bool) -> u32 {
        match light_slice(self.light_ranges(), layer) {
            Some(k) if !unlit => (k as u64 + 1) as u32 * LIGHT_PICK_STRIDE as u32,
            _ => 0,
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

    /// The mask view the groups bind this frame: the target when masks drew,
    /// the fallback otherwise.
    pub(crate) fn bound_view<'a>(
        &'a self,
        resources: &'a PainterMaskResources,
    ) -> &'a crate::gpu::TextureView {
        match (&self.target, self.active) {
            (Some((_, view, _)), true) => view,
            _ => &resources.fallback_view,
        }
    }

    /// The light layers view the groups bind this frame: the plugin's when it
    /// reported some, the fallback otherwise.
    pub(crate) fn bound_light_view<'a>(
        &'a self,
        resources: &'a PainterMaskResources,
    ) -> &'a crate::gpu::TextureView {
        self.lights
            .as_ref()
            .map_or(&resources.light_fallback_view, |l| &l.view)
    }

    /// The group laid out as the painter mask layout.
    pub(crate) fn bind_group(
        &mut self,
        device: &crate::gpu::Device,
        resources: &PainterMaskResources,
    ) -> crate::gpu::BindGroup {
        let view = self.bound_view(resources).clone();
        let light = self.bound_light_view(resources).clone();
        let uniform = self.uniform.clone();
        let light_uniform = self.light_uniform.clone();
        self.group.get((view.clone(), light.clone()), || {
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
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::TextureView(&light),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: light_uniform.as_entire_binding(),
                    },
                ],
            })
        })
    }

    /// The group laid out as the light pick layout.
    pub(crate) fn light_pick_group(
        &mut self,
        device: &crate::gpu::Device,
        resources: &PainterMaskResources,
    ) -> crate::gpu::BindGroup {
        let light = self.bound_light_view(resources).clone();
        let uniform = self.light_pick_uniform.clone();
        self.light_pick_group.get(light.clone(), || {
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("painter_light_pick_bg"),
                layout: &resources.light_pick_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::TextureView(&light),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: crate::gpu::BindingResource::Buffer(crate::gpu::BufferBinding {
                            buffer: &uniform,
                            offset: 0,
                            size: crate::gpu::BufferSize::new(16),
                        }),
                    },
                ],
            })
        })
    }

    /// The group laid out as the depth-read layout, over `depth` and this
    /// viewport's masks and light layers.
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
            [self.bound_view(resources), self.bound_light_view(resources)],
            [&self.uniform, &self.light_uniform],
        )
    }

    /// The depth-read group over the depth fallback, for passes whose depth
    /// attachment is writable.
    pub(crate) fn ldr_depth_read_group(
        &mut self,
        device: &crate::gpu::Device,
        resources: &PainterMaskResources,
        layout: &crate::gpu::BindGroupLayout,
        sampler: &crate::gpu::Sampler,
    ) -> crate::gpu::BindGroup {
        let view = self.bound_view(resources).clone();
        let light = self.bound_light_view(resources).clone();
        let uniform = self.uniform.clone();
        let light_uniform = self.light_uniform.clone();
        self.ldr_depth_group.get((view.clone(), light.clone()), || {
            depth_read_group(
                device,
                layout,
                sampler,
                &resources.depth_fallback_view,
                [&view, &light],
                [&uniform, &light_uniform],
            )
        })
    }

    /// The groups as the last prepare left them, for passes that cannot
    /// build one: the painter mask group, the depth-read group over the depth
    /// fallback and the light pick group.
    pub(crate) fn current(
        &self,
    ) -> (
        Option<&crate::gpu::BindGroup>,
        Option<&crate::gpu::BindGroup>,
        Option<&crate::gpu::BindGroup>,
    ) {
        (
            self.group.current(),
            self.ldr_depth_group.current(),
            self.light_pick_group.current(),
        )
    }
}

/// A group laid out as the depth-read layout: `depth` and `sampler`, then the
/// painter masks and the light layers, `views` and `uniforms` in that order.
pub(crate) fn depth_read_group(
    device: &crate::gpu::Device,
    layout: &crate::gpu::BindGroupLayout,
    sampler: &crate::gpu::Sampler,
    depth: &crate::gpu::TextureView,
    views: [&crate::gpu::TextureView; 2],
    uniforms: [&crate::gpu::Buffer; 2],
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
                resource: crate::gpu::BindingResource::TextureView(views[0]),
            },
            crate::gpu::BindGroupEntry {
                binding: 3,
                resource: uniforms[0].as_entire_binding(),
            },
            crate::gpu::BindGroupEntry {
                binding: 4,
                resource: crate::gpu::BindingResource::TextureView(views[1]),
            },
            crate::gpu::BindGroupEntry {
                binding: 5,
                resource: uniforms[1].as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_ranges_pick_their_array_layer() {
        let ranges = [[0, 2], [3, 3], [5, 9]];
        assert_eq!(light_slice(&ranges, 1), Some(0));
        assert_eq!(light_slice(&ranges, 3), Some(1));
        assert_eq!(light_slice(&ranges, 4), None);
        assert_eq!(light_slice(&ranges, 9), Some(2));
        let u = painter_lights_uniform(&ranges, 2.0);
        assert_eq!(u.len() * 4, PAINTER_LIGHTS_UNIFORM_BYTES as usize);
        assert_eq!(&u[..2], &[3, 2.0f32.to_bits()]);
        assert_eq!(&u[8..11], &[3, 3, 1]);
    }
}
