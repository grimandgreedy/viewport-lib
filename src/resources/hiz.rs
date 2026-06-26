//! Hierarchical-Z max-depth pyramid used by the GPU occlusion cull.
//!
//! After the opaque scene pass writes depth, [`HizState::encode`] copies that
//! depth into mip 0 of an `R32Float` texture and runs a max-reduction down the
//! mip chain. The next frame's cull samples the pyramid to drop instances whose
//! screen-space box is entirely behind nearer geometry. Max-reduction keeps the
//! test conservative: a texel reports the farthest depth in its region, so an
//! instance is only culled when it is behind the farthest occluder it covers.

/// Number of mip levels for a `w x h` pyramid (full chain down to 1x1).
fn mip_count(w: u32, h: u32) -> u32 {
    1 + (w.max(h) as f32).log2().floor() as u32
}

/// Mip dimensions at `level`, clamped to a minimum of 1.
fn level_dims(w: u32, h: u32, level: u32) -> (u32, u32) {
    ((w >> level).max(1), (h >> level).max(1))
}

/// GPU resources for one pyramid size. Recreated when the depth target the
/// pyramid samples changes dimensions.
pub(crate) struct HizState {
    /// mip-0 dimensions in pixels (matches the depth target).
    pub(crate) dims: [u32; 2],
    /// Full-mip-chain view sampled by the cull shader.
    all_view: wgpu::TextureView,
    /// One single-mip storage view per level (reduction write targets).
    storage_views: Vec<wgpu::TextureView>,
    /// Copy pipeline: scene depth -> mip 0.
    copy_pipeline: wgpu::ComputePipeline,
    copy_bgl: wgpu::BindGroupLayout,
    /// Reduce pipeline: mip N -> mip N+1 (max of 2x2).
    reduce_pipeline: wgpu::ComputePipeline,
    /// Cached reduction bind groups, one per destination level 1..mip_count.
    reduce_bind_groups: Vec<wgpu::BindGroup>,
}

impl HizState {
    pub(crate) fn new(device: &wgpu::Device, w: u32, h: u32) -> Self {
        let w = w.max(1);
        let h = h.max(1);
        let mips = mip_count(w, h);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz_pyramid"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        // TextureView keeps the underlying texture alive, so the `texture`
        // handle is not stored on the state.
        let all_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let single_mip_view = |level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz_mip_view"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let storage_views: Vec<_> = (0..mips).map(single_mip_view).collect();
        let sampled_views: Vec<_> = (0..mips).map(single_mip_view).collect();

        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_copy_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/hiz_copy.wgsl")).into(),
            ),
        });
        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_reduce_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/hiz_reduce.wgsl")).into(),
            ),
        });

        let compute = wgpu::ShaderStages::COMPUTE;
        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_copy_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(1),
            ],
        });
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_reduce_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(1),
            ],
        });

        let copy_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz_copy_layout"),
            bind_group_layouts: &[&copy_bgl],
            push_constant_ranges: &[],
        });
        let reduce_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz_reduce_layout"),
            bind_group_layouts: &[&reduce_bgl],
            push_constant_ranges: &[],
        });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hiz_copy_pipeline"),
            layout: Some(&copy_layout),
            module: &copy_shader,
            entry_point: Some("copy_depth"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hiz_reduce_pipeline"),
            layout: Some(&reduce_layout),
            module: &reduce_shader,
            entry_point: Some("reduce"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // One reduction bind group per destination level: src = previous mip
        // (sampled), dst = this mip (storage).
        let reduce_bind_groups: Vec<_> = (1..mips)
            .map(|level| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hiz_reduce_bg"),
                    layout: &reduce_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &sampled_views[(level - 1) as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &storage_views[level as usize],
                            ),
                        },
                    ],
                })
            })
            .collect();

        Self {
            dims: [w, h],
            all_view,
            storage_views,
            copy_pipeline,
            copy_bgl,
            reduce_pipeline,
            reduce_bind_groups,
        }
    }

    /// Full-mip view the cull shader binds at group 0 binding 6.
    pub(crate) fn cull_view(&self) -> &wgpu::TextureView {
        &self.all_view
    }

    /// Encode the copy + reduction passes into `encoder`. `depth_view` is the
    /// depth-aspect view of the scene depth target written this frame.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
    ) {
        let [w, h] = self.dims;
        let mips = self.storage_views.len() as u32;

        // mip 0: copy scene depth.
        let copy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_copy_bg"),
            layout: &self.copy_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.storage_views[0]),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_copy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, &copy_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }

        // mips 1..n: max-reduce from the level above. wgpu inserts a storage
        // barrier between compute passes, so each level sees the previous one.
        for level in 1..mips {
            let (lw, lh) = level_dims(w, h, level);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce_pipeline);
            pass.set_bind_group(0, &self.reduce_bind_groups[(level - 1) as usize], &[]);
            pass.dispatch_workgroups(lw.div_ceil(8), lh.div_ceil(8), 1);
        }
    }
}

impl crate::resources::ViewportGpuResources {
    /// Build the HiZ pyramid from this frame's scene depth. Called after the
    /// opaque pass writes depth; the pyramid is consumed by next frame's cull.
    /// `depth_view` must be the depth-aspect view of the scene depth target,
    /// `w`/`h` its dimensions.
    pub(crate) fn build_hiz(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        w: u32,
        h: u32,
    ) {
        if w == 0 || h == 0 {
            return;
        }
        let stale = self.hiz.as_ref().map_or(true, |s| s.dims != [w, h]);
        if stale {
            self.hiz = Some(HizState::new(device, w, h));
        }
        self.hiz
            .as_ref()
            .unwrap()
            .encode(device, encoder, depth_view);
    }

    /// HiZ view plus its mip-0 dimensions for the cull's occlusion test, or
    /// `None` when no pyramid has been built yet.
    pub(crate) fn hiz_cull_view(&self) -> Option<(&wgpu::TextureView, [f32; 2])> {
        self.hiz
            .as_ref()
            .map(|s| (s.cull_view(), [s.dims[0] as f32, s.dims[1] as f32]))
    }

    /// Enable or disable the HiZ occlusion test on the main-camera cull.
    pub(crate) fn set_occlusion_culling(&mut self, enabled: bool) {
        self.occlusion_culling_enabled = enabled;
    }

    /// Whether the HiZ occlusion test is currently enabled.
    pub(crate) fn occlusion_culling_enabled(&self) -> bool {
        self.occlusion_culling_enabled
    }
}
