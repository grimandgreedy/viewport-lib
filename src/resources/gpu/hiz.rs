//! Hierarchical-Z max-depth pyramid used by the GPU occlusion cull.
//!
//! The cull runs in `prepare`, before this frame's scene pass, so there is no
//! current depth to test against. Instead the renderer keeps last frame's scene
//! depth and reprojects it into this frame's camera at cull time, builds the
//! pyramid from the reprojected depth, and the cull samples it. This tracks
//! camera motion, unlike sampling last frame's depth directly.
//!
//! Per frame:
//!   1. `store_prev_depth` (end of the scene pass) copies the depth just written
//!      into `prev_depth` and records the camera that drew it.
//!   2. `build_reprojected` (start of the cull, into the cull encoder) reprojects
//!      `prev_depth` into the current camera, scatters it into mip 0, and
//!      max-reduces down the chain.
//!
//! Max-reduction keeps the test conservative: each texel reports the farthest
//! depth in its region, so an instance is only culled when it is behind the
//! farthest occluder it covers. Disoccluded pixels (revealed this frame) carry
//! no reprojected sample and stay at the far value, so they never cause a cull.

/// Reprojection parameters uploaded to `hiz_reproject.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReprojUniform {
    inv_prev_vp: [[f32; 4]; 4],
    cur_vp: [[f32; 4]; 4],
    dims: [u32; 2],
    _pad: [u32; 2],
}

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

    /// Last frame's scene depth, copied as R32Float and reprojected each frame.
    prev_depth_storage_view: wgpu::TextureView,
    /// Set once the first depth has been stored. Until then there is nothing to
    /// reproject and the pyramid is not built.
    has_prev_depth: bool,
    /// View-projection of the camera that drew `prev_depth`.
    prev_view_proj: [[f32; 4]; 4],

    /// Reprojection uniform (inverse-prev and current view-projection). The
    /// scatter buffer is owned by the bind groups that reference it.
    reproj_uniform_buf: wgpu::Buffer,

    /// Full-mip-chain view sampled by the cull shader.
    all_view: wgpu::TextureView,
    /// One single-mip storage view per level (reduction write targets).
    storage_views: Vec<wgpu::TextureView>,

    /// depth -> prev_depth copy (reuses the mip-0 copy shader).
    copy_pipeline: wgpu::ComputePipeline,
    copy_bgl: wgpu::BindGroupLayout,
    /// Reprojection: clear, scatter, then buffer -> mip 0.
    init_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    to_texture_pipeline: wgpu::ComputePipeline,
    /// mip N -> mip N+1 (max of 2x2).
    reduce_pipeline: wgpu::ComputePipeline,

    /// Cached bind groups for the per-frame passes (the depth-copy group is
    /// rebuilt each call because the depth view changes).
    reproj_bg: wgpu::BindGroup,
    to_texture_bg: wgpu::BindGroup,
    reduce_bind_groups: Vec<wgpu::BindGroup>,
}

impl HizState {
    pub(crate) fn new(device: &wgpu::Device, w: u32, h: u32) -> Self {
        let w = w.max(1);
        let h = h.max(1);
        let mips = mip_count(w, h);

        let storage_texture = |label: &str, mip_levels: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        };

        // Previous-frame depth (single mip): written by the depth copy, read by
        // the scatter pass. TextureView keeps the texture alive, so the texture
        // handle is not stored.
        let prev_depth = storage_texture("hiz_prev_depth", 1);
        let prev_depth_storage_view = prev_depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hiz_prev_depth_storage"),
            ..Default::default()
        });
        let prev_depth_sampled_view = prev_depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hiz_prev_depth_sampled"),
            ..Default::default()
        });

        // Pyramid.
        let pyramid = storage_texture("hiz_pyramid", mips);
        let all_view = pyramid.create_view(&wgpu::TextureViewDescriptor::default());
        let single_mip_view = |level: u32| {
            pyramid.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz_mip_view"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let storage_views: Vec<_> = (0..mips).map(single_mip_view).collect();
        let sampled_views: Vec<_> = (0..mips).map(single_mip_view).collect();

        let scatter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hiz_scatter_buf"),
            size: (w as u64 * h as u64 * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let reproj_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hiz_reproj_uniform"),
            size: std::mem::size_of::<ReprojUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute = wgpu::ShaderStages::COMPUTE;
        let storage_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let sampled_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let buffer_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let wgsl = |label: &str, src: &'static str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let copy_shader = wgsl(
            "hiz_copy_shader",
            include_str!(concat!(env!("OUT_DIR"), "/hiz_copy.wgsl")),
        );
        let reproject_shader = wgsl(
            "hiz_reproject_shader",
            include_str!(concat!(env!("OUT_DIR"), "/hiz_reproject.wgsl")),
        );
        let to_texture_shader = wgsl(
            "hiz_to_texture_shader",
            include_str!(concat!(env!("OUT_DIR"), "/hiz_to_texture.wgsl")),
        );
        let reduce_shader = wgsl(
            "hiz_reduce_shader",
            include_str!(concat!(env!("OUT_DIR"), "/hiz_reduce.wgsl")),
        );

        // depth -> prev_depth (texture_depth_2d in, R32Float storage out).
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
                storage_tex_entry(1),
            ],
        });
        // Reprojection (init + scatter share this layout; init ignores the
        // texture binding, which is allowed).
        let reproj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_reproj_bgl"),
            entries: &[
                uniform_entry(0),
                sampled_tex_entry(1),
                buffer_entry(2, false),
            ],
        });
        let to_texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_to_texture_bgl"),
            entries: &[buffer_entry(0, true), storage_tex_entry(1)],
        });
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_reduce_bgl"),
            entries: &[sampled_tex_entry(0), storage_tex_entry(1)],
        });

        let pipeline =
            |label: &str, bgl: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, ep: &str| {
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[bgl],
                    push_constant_ranges: &[],
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module,
                    entry_point: Some(ep),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            };
        let copy_pipeline = pipeline("hiz_copy_pipeline", &copy_bgl, &copy_shader, "copy_depth");
        let init_pipeline = pipeline("hiz_init_pipeline", &reproj_bgl, &reproject_shader, "init");
        let scatter_pipeline = pipeline(
            "hiz_scatter_pipeline",
            &reproj_bgl,
            &reproject_shader,
            "scatter",
        );
        let to_texture_pipeline = pipeline(
            "hiz_to_texture_pipeline",
            &to_texture_bgl,
            &to_texture_shader,
            "to_texture",
        );
        let reduce_pipeline =
            pipeline("hiz_reduce_pipeline", &reduce_bgl, &reduce_shader, "reduce");

        let reproj_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_reproj_bg"),
            layout: &reproj_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reproj_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&prev_depth_sampled_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scatter_buf.as_entire_binding(),
                },
            ],
        });
        let to_texture_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_to_texture_bg"),
            layout: &to_texture_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scatter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&storage_views[0]),
                },
            ],
        });
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
            prev_depth_storage_view,
            has_prev_depth: false,
            prev_view_proj: [[0.0; 4]; 4],
            reproj_uniform_buf,
            all_view,
            storage_views,
            copy_pipeline,
            copy_bgl,
            init_pipeline,
            scatter_pipeline,
            to_texture_pipeline,
            reduce_pipeline,
            reproj_bg,
            to_texture_bg,
            reduce_bind_groups,
        }
    }

    /// Full-mip view the cull shader binds at group 0 binding 6.
    pub(crate) fn cull_view(&self) -> &wgpu::TextureView {
        &self.all_view
    }

    /// Copy the scene depth just written into `prev_depth` and record the camera
    /// that drew it, for next frame's reprojection. `depth_view` is the
    /// depth-aspect view of the scene depth target.
    pub(crate) fn store_prev_depth(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        view_proj: [[f32; 4]; 4],
    ) {
        let [w, h] = self.dims;
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
                    resource: wgpu::BindingResource::TextureView(&self.prev_depth_storage_view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hiz_store_prev_depth"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.copy_pipeline);
        pass.set_bind_group(0, &copy_bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        drop(pass);

        self.prev_view_proj = view_proj;
        self.has_prev_depth = true;
    }

    /// Reproject `prev_depth` into `cur_view_proj` and build the pyramid into
    /// `encoder`, before the cull dispatch that reads it. Returns false (and
    /// builds nothing) until a depth has been stored, so the first frame after
    /// a resize runs the cull frustum-only.
    pub(crate) fn build_reprojected(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cur_view_proj: [[f32; 4]; 4],
    ) -> bool {
        if !self.has_prev_depth {
            return false;
        }
        let [w, h] = self.dims;

        let inv_prev_vp = glam::Mat4::from_cols_array_2d(&self.prev_view_proj)
            .inverse()
            .to_cols_array_2d();
        let uniform = ReprojUniform {
            inv_prev_vp,
            cur_vp: cur_view_proj,
            dims: [w, h],
            _pad: [0, 0],
        };
        queue.write_buffer(
            &self.reproj_uniform_buf,
            0,
            bytemuck::cast_slice(std::slice::from_ref(&uniform)),
        );

        // Clear the scatter target to far, then scatter the reprojected depth.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_reproject_init"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.init_pipeline);
            pass.set_bind_group(0, &self.reproj_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_reproject_scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &self.reproj_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        // Scattered depth -> mip 0.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_reproject_to_texture"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.to_texture_pipeline);
            pass.set_bind_group(0, &self.to_texture_bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        // mips 1..n: max-reduce from the level above.
        let mips = self.storage_views.len() as u32;
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
        true
    }
}

impl crate::resources::ViewportCullState {
    /// Copy this frame's scene depth into this viewport's HiZ prev-depth target
    /// for next frame's reprojection. Called at the end of the scene pass.
    /// Allocates or resizes the HiZ state to match the depth target.
    pub(crate) fn store_hiz_prev_depth(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        w: u32,
        h: u32,
        view_proj: [[f32; 4]; 4],
    ) {
        if w == 0 || h == 0 {
            return;
        }
        let stale = self.hiz.as_ref().map_or(true, |s| s.dims != [w, h]);
        if stale {
            self.hiz = Some(HizState::new(device, w, h));
        }
        self.hiz
            .as_mut()
            .unwrap()
            .store_prev_depth(device, encoder, depth_view, view_proj);
    }

    /// Reproject last frame's depth and build the pyramid into the cull encoder.
    /// Returns true when a pyramid was built and is safe to sample this frame.
    pub(crate) fn build_hiz_reprojected(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cur_view_proj: [[f32; 4]; 4],
    ) -> bool {
        match self.hiz.as_ref() {
            Some(s) => s.build_reprojected(queue, encoder, cur_view_proj),
            None => false,
        }
    }

    /// HiZ view plus its mip-0 dimensions for the cull's occlusion test, or
    /// `None` when no pyramid exists. Only valid to sample on a frame where
    /// `build_hiz_reprojected` returned true.
    pub(crate) fn hiz_cull_view(&self) -> Option<(&wgpu::TextureView, [f32; 2])> {
        self.hiz
            .as_ref()
            .map(|s| (s.cull_view(), [s.dims[0] as f32, s.dims[1] as f32]))
    }
}

impl crate::resources::ViewportGpuResources {
    /// Enable or disable the HiZ occlusion test on the main-camera cull.
    pub(crate) fn set_occlusion_culling(&mut self, enabled: bool) {
        self.occlusion_culling_enabled = enabled;
    }

    /// Whether the HiZ occlusion test is currently enabled.
    pub(crate) fn occlusion_culling_enabled(&self) -> bool {
        self.occlusion_culling_enabled
    }
}
