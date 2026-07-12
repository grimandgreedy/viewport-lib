use super::*;
use crate::renderer::ShDegree;

/// Gaussian splat render pipeline, the radix-sort compute passes, their bind
/// group layouts, and the slotted store of uploaded splat sets. All lazily
/// built; `store` holds the uploaded sets keyed by id.
pub(crate) struct GaussianSplatResources {
    /// Gaussian splat render pipeline. None until first splat set is submitted.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Bind group layout for group 1 of the render pipeline.
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    /// Compute pipeline for per-splat view-space depth.
    pub(crate) depth_pipeline: Option<crate::gpu::ComputePipeline>,
    /// Compute pipeline that clears the sort histogram.
    pub(crate) sort_clear_pipeline: Option<crate::gpu::ComputePipeline>,
    /// Radix-sort histogram pass.
    pub(crate) sort_histogram_pipeline: Option<crate::gpu::ComputePipeline>,
    /// Radix-sort prefix-sum pass.
    pub(crate) sort_prefix_pipeline: Option<crate::gpu::ComputePipeline>,
    /// Radix-sort scatter pass.
    pub(crate) sort_scatter_pipeline: Option<crate::gpu::ComputePipeline>,
    /// Compute pipeline that initialises sort index values.
    pub(crate) sort_init_pipeline: Option<crate::gpu::ComputePipeline>,
    /// Bind group layout for the depth compute pass.
    pub(crate) depth_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Bind group layout for the sort compute passes.
    pub(crate) sort_bgl: Option<crate::gpu::BindGroupLayout>,
}

impl Default for GaussianSplatResources {
    fn default() -> Self {
        Self {
            pipeline: None,
            bgl: None,
            depth_pipeline: None,
            sort_clear_pipeline: None,
            sort_histogram_pipeline: None,
            sort_prefix_pipeline: None,
            sort_scatter_pipeline: None,
            sort_init_pipeline: None,
            depth_bgl: None,
            sort_bgl: None,
        }
    }
}

/// Check that a splat set is non-empty and its per-attribute vectors agree in
/// length. Shared by the sync, async, and replace upload paths.
fn validate_gaussian_splat_data(
    data: &crate::renderer::GaussianSplatData,
) -> crate::error::ViewportResult<()> {
    if data.positions.is_empty() {
        return Err(crate::error::ViewportError::InvalidGaussianSplatData {
            reason: "empty splat list",
        });
    }
    let n = data.positions.len();
    if data.scales.len() != n || data.rotations.len() != n || data.opacities.len() != n {
        return Err(crate::error::ViewportError::InvalidGaussianSplatData {
            reason: "mismatched buffer lengths",
        });
    }
    Ok(())
}

/// Build the persistent GPU buffers for a splat set and assemble the
/// `GaussianSplatGpuSet`. Assumes `data` already passed
/// [`validate_gaussian_splat_data`]. Shared by the sync `upload_gaussian_splat`,
/// the async worker, and `replace_gaussian_splat` so all three produce identical
/// resources.
fn build_gaussian_splat_set(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    data: &crate::renderer::GaussianSplatData,
) -> GaussianSplatGpuSet {
    let count = data.positions.len() as u32;

    // Pad positions/scales/rotations to vec4 (w=1 / w=0 / raw).
    let pos_data: Vec<[f32; 4]> = data
        .positions
        .iter()
        .map(|p| [p[0], p[1], p[2], 1.0])
        .collect();
    let scale_data: Vec<[f32; 4]> = data
        .scales
        .iter()
        .map(|s| [s[0], s[1], s[2], 0.0])
        .collect();
    let rotation_data: Vec<[f32; 4]> = data
        .rotations
        .iter()
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect();

    let buf_size_pos = (pos_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
    let buf_size_scale = (scale_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
    let buf_size_rot = (rotation_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
    let buf_size_opa = (data.opacities.len() * 4).max(4) as u64;
    let buf_size_sh = (data.sh_coefficients.len() * 4).max(4) as u64;

    let position_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_position_buf"),
        size: buf_size_pos,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&position_buf, 0, bytemuck::cast_slice(&pos_data));

    let scale_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_scale_buf"),
        size: buf_size_scale,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&scale_buf, 0, bytemuck::cast_slice(&scale_data));

    let rotation_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_rotation_buf"),
        size: buf_size_rot,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&rotation_buf, 0, bytemuck::cast_slice(&rotation_data));

    let opacity_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_opacity_buf"),
        size: buf_size_opa,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&opacity_buf, 0, bytemuck::cast_slice(&data.opacities));

    let sh_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_sh_buf"),
        size: buf_size_sh,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.sh_coefficients.is_empty() {
        queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&data.sh_coefficients));
    }

    GaussianSplatGpuSet {
        position_buf,
        scale_buf,
        rotation_buf,
        opacity_buf,
        sh_buf,
        sh_degree: data.sh_degree,
        count,
        viewport_sort: Vec::new(),
        cpu_positions: data.positions.clone(),
        cpu_scales: data.scales.clone(),
    }
}

// Per-viewport SplatUniform layout (must match gaussian_splat.wgsl).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SplatUniform {
    model: [[f32; 4]; 4],
    viewport_w: f32,
    viewport_h: f32,
    sh_degree: u32,
    count: u32,
}

// Depth compute uniform (must match gaussian_splat_sort.wgsl DepthUniform).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthUniform {
    model: [[f32; 4]; 4],
    eye: [f32; 3],
    count: u32,
}

// Sort pass uniform (must match gaussian_splat_sort.wgsl SortUniform).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SortUniform {
    shift: u32,
    count: u32,
    pass_num: u32,
    _pad: u32,
}

impl DeviceResources {
    /// Lazily create all Gaussian splat render and compute pipelines.
    ///
    /// No-op after first call. Called from prepare when gaussian_splats is non-empty.
    pub(crate) fn ensure_gaussian_splat_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.gaussian_splat.bgl.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // ---------------------------------------------------------------
        // Render pipeline
        // ---------------------------------------------------------------

        // Group 1 BGL: SplatUniform (b0), sorted_indices (b1), positions (b2),
        //              scales (b3), rotations (b4), opacities (b5), sh_coefficients (b6).
        let splat_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_bgl"),
            entries: &[
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
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 6,
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

        let render_shader = crate::resources::builders::wgsl_module(
            device,
            "gaussian_splat_shader",
            crate::resources::builders::wgsl_source!("gaussian_splat"),
        );

        let render_layout = crate::resources::builders::standard_scene_layout(
            device,
            "gaussian_splat_pipeline_layout",
            &self.camera_bind_group_layout,
            &splat_bgl,
        );

        // No MSAA for Gaussian splats (alpha blending requires single-sample).
        let gaussian_splat_pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "gaussian_splat_pipeline",
                layout: &render_layout,
                shader: &render_shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[],
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: 1,
                ldr_format: self.target_format,
            },
        );

        // ---------------------------------------------------------------
        // Sort compute pipelines
        // ---------------------------------------------------------------

        let sort_shader = crate::resources::builders::wgsl_module(
            device,
            "gaussian_splat_sort_shader",
            crate::resources::builders::wgsl_source!("gaussian_splat_sort"),
        );

        // Depth compute BGL: DepthUniform (b0), positions (b1), keys_ping_out (b2).
        let depth_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_depth_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let depth_layout = crate::resources::builders::pipeline_layout(
            device,
            "gaussian_splat_depth_layout",
            &[&depth_bgl],
        );

        let gaussian_splat_depth_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "gaussian_splat_depth_pipeline",
            &depth_layout,
            &sort_shader,
            "compute_depths",
        );

        // Sort BGL: SortUniform (b0), keys_ping (b1), keys_pong (b2),
        //           vals_ping (b3), vals_pong (b4), histogram (b5).
        let sort_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_sort_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sort_layout = crate::resources::builders::pipeline_layout(
            device,
            "gaussian_splat_sort_layout",
            &[&sort_bgl],
        );

        let gaussian_splat_sort_init_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "gaussian_splat_sort_init_pipeline",
            &sort_layout,
            &sort_shader,
            "init_indices",
        );

        let gaussian_splat_sort_clear_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "gaussian_splat_sort_clear_pipeline",
            &sort_layout,
            &sort_shader,
            "clear_histogram",
        );

        let gaussian_splat_sort_histogram_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "gaussian_splat_sort_histogram_pipeline",
            &sort_layout,
            &sort_shader,
            "histogram_pass",
        );

        let gaussian_splat_sort_prefix_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "gaussian_splat_sort_prefix_pipeline",
            &sort_layout,
            &sort_shader,
            "prefix_sum_pass",
        );

        let gaussian_splat_sort_scatter_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "gaussian_splat_sort_scatter_pipeline",
            &sort_layout,
            &sort_shader,
            "scatter_pass",
        );

        self.gaussian_splat.bgl = Some(splat_bgl);
        self.gaussian_splat.pipeline = Some(gaussian_splat_pipeline);
        self.gaussian_splat.depth_bgl = Some(depth_bgl);
        self.gaussian_splat.depth_pipeline = Some(gaussian_splat_depth_pipeline);
        self.gaussian_splat.sort_bgl = Some(sort_bgl);
        self.gaussian_splat.sort_init_pipeline = Some(gaussian_splat_sort_init_pipeline);
        self.gaussian_splat.sort_clear_pipeline = Some(gaussian_splat_sort_clear_pipeline);
        self.gaussian_splat.sort_histogram_pipeline = Some(gaussian_splat_sort_histogram_pipeline);
        self.gaussian_splat.sort_prefix_pipeline = Some(gaussian_splat_sort_prefix_pipeline);
        self.gaussian_splat.sort_scatter_pipeline = Some(gaussian_splat_sort_scatter_pipeline);
    }

    /// Upload one Gaussian splat set to the GPU and return its handle.
    ///
    /// Call once per splat set at startup (or when the set changes). The returned
    /// [`GaussianSplatId`] is stable until [`free_gaussian_splat`] is called.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// if `data.positions` is empty or if the lengths of `positions`, `scales`,
    /// `rotations`, and `opacities` do not all match.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use viewport_lib::error::ViewportError;
    /// # use viewport_lib::renderer::{GaussianSplatData, ViewportRenderer};
    /// # fn demo(renderer: &mut ViewportRenderer, device: &wgpu::Device, queue: &wgpu::Queue) {
    /// let result = renderer.upload_gaussian_splat(device, queue, &GaussianSplatData::default());
    /// assert!(matches!(result, Err(ViewportError::InvalidGaussianSplatData { .. })));
    /// # }
    /// ```
    pub fn upload_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &crate::renderer::GaussianSplatData,
    ) -> crate::error::ViewportResult<crate::renderer::GaussianSplatId> {
        validate_gaussian_splat_data(data)?;
        // Build the splat pipelines now so a load-time upload also pays the
        // pipeline compile, not the first frame that draws the splat.
        self.ensure_gaussian_splat_pipelines(device);
        let gpu_set = build_gaussian_splat_set(device, queue, data);
        Ok(self.content.gaussian_splat_store.insert(gpu_set))
    }

    /// Replace the contents of an uploaded Gaussian splat set in place, keeping
    /// the same [`GaussianSplatId`](crate::renderer::GaussianSplatId).
    ///
    /// Items holding the handle pick up the new splats on the next frame with no
    /// reassignment. The generation check is the in-flight guard: a stale handle
    /// (its slot freed and reused) returns
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) instead of
    /// overwriting whatever now occupies the slot. Use this for content that
    /// changes over time (a re-trained or streamed splat set).
    ///
    /// # Errors
    ///
    /// [`InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// when `data` is empty or its per-attribute vectors disagree in length, or
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` does not
    /// resolve to a live set.
    pub fn replace_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::renderer::GaussianSplatId,
        data: &crate::renderer::GaussianSplatData,
    ) -> crate::error::ViewportResult<()> {
        validate_gaussian_splat_data(data)?;
        let gpu_set = build_gaussian_splat_set(device, queue, data);
        if self.content.gaussian_splat_store.replace(id, gpu_set) {
            Ok(())
        } else {
            Err(crate::error::ViewportError::StaleHandle {
                index: id.index() as usize,
                count: self.content.gaussian_splat_store.slot_count(),
            })
        }
    }

    /// Remove an uploaded Gaussian splat set by handle.
    pub fn free_gaussian_splat(&mut self, id: crate::renderer::GaussianSplatId) {
        self.content.gaussian_splat_store.remove(id);
    }

    /// Start an asynchronous Gaussian splat upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Vec4
    /// padding for positions / scales / rotations and storage buffer
    /// creation + writes all run on a worker thread on a cloned `Device`
    /// and `Queue`. The apply step inserts the prepared `GaussianSplatGpuSet`
    /// into the store and surfaces the resulting [`GaussianSplatId`].
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidGaussianSplatData`] before any job
    /// is submitted when `data.positions` is empty or the per-attribute
    /// vectors disagree in length.
    pub fn begin_upload_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: crate::renderer::GaussianSplatData,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        validate_gaussian_splat_data(&data)?;

        let slot = crate::resources::ResultSlot::<crate::renderer::GaussianSplatId>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let queue_for_worker = queue.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let gpu_set =
                    build_gaussian_splat_set(&device_for_worker, &queue_for_worker, &data);
                progress.set(0.95);

                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let id = resources.content.gaussian_splat_store.insert(gpu_set);
                        slot_for_apply.set(id);
                    }),
                ))
            })
        };

        self.job_results
            .gaussian_splat
            .lock()
            .expect("gaussian splat result map poisoned")
            .insert(id, slot);
        Ok(id)
    }

    /// Take the [`GaussianSplatId`](crate::renderer::GaussianSplatId) produced by a
    /// completed [`begin_upload_gaussian_splat`](Self::begin_upload_gaussian_splat) job.
    pub fn upload_result_gaussian_splat(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::renderer::GaussianSplatId> {
        let mut map = self
            .job_results
            .gaussian_splat
            .lock()
            .expect("gaussian splat result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(splat_id) => {
                map.remove(&id);
                Ok(splat_id)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Ensure per-viewport sort buffers exist for (store_index, viewport_index).
    ///
    /// Also creates the render bind group for the render pipeline (group 1).
    /// Called from run_gaussian_splat_sort before dispatching.
    pub(crate) fn ensure_gaussian_splat_sort_buffers(
        &mut self,
        device: &crate::gpu::Device,
        store_index: usize,
        viewport_index: usize,
    ) {
        let set = match self.content.gaussian_splat_store.get_by_index(store_index) {
            Some(s) => s,
            None => return,
        };
        let count = set.count as usize;

        // Grow the per-viewport vec if needed.
        if viewport_index >= set.viewport_sort.len() || set.viewport_sort[viewport_index].is_none()
        {
            // We need mutable access - re-borrow.
            let set_mut = self
                .content
                .gaussian_splat_store
                .get_mut_by_index(store_index)
                .unwrap();
            while set_mut.viewport_sort.len() <= viewport_index {
                set_mut.viewport_sort.push(None);
            }

            if set_mut.viewport_sort[viewport_index].is_none() {
                let buf_size = (count * 4).max(4) as u64;

                let depth_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("splat_depth_buf"),
                    size: buf_size,
                    usage: crate::gpu::BufferUsages::STORAGE
                        | crate::gpu::BufferUsages::COPY_DST
                        | crate::gpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let sort_buf_usage = crate::gpu::BufferUsages::STORAGE
                    | crate::gpu::BufferUsages::COPY_SRC
                    | crate::gpu::BufferUsages::COPY_DST;
                let keys_ping = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("splat_keys_ping"),
                    size: buf_size,
                    usage: sort_buf_usage,
                    mapped_at_creation: false,
                });
                let keys_pong = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("splat_keys_pong"),
                    size: buf_size,
                    usage: sort_buf_usage,
                    mapped_at_creation: false,
                });
                let vals_ping = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("splat_vals_ping"),
                    size: buf_size,
                    usage: sort_buf_usage,
                    mapped_at_creation: false,
                });
                let vals_pong = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("splat_vals_pong"),
                    size: buf_size,
                    usage: sort_buf_usage,
                    mapped_at_creation: false,
                });
                // Histogram: 256 x u32 atomic.
                let histogram_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("splat_histogram"),
                    size: 256 * 4,
                    usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // Per-viewport SplatUniform buffer.
                let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("splat_uniform_buf"),
                    size: std::mem::size_of::<SplatUniform>() as u64,
                    usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // Build the render bind group (group 1). vals_ping holds sorted indices
                // after 4 sort passes (even number of passes means result ends in ping).
                let render_bg = {
                    let bgl = self.gaussian_splat.bgl.as_ref().unwrap();
                    let set_ref = self
                        .content
                        .gaussian_splat_store
                        .get_by_index(store_index)
                        .unwrap();
                    device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("splat_render_bg"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: vals_ping.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 2,
                                resource: set_ref.position_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 3,
                                resource: set_ref.scale_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 4,
                                resource: set_ref.rotation_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 5,
                                resource: set_ref.opacity_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 6,
                                resource: set_ref.sh_buf.as_entire_binding(),
                            },
                        ],
                    })
                };

                let vp_sort = GaussianSplatViewportSort {
                    depth_buf,
                    keys_ping,
                    keys_pong,
                    vals_ping,
                    vals_pong,
                    histogram_buf,
                    render_bg,
                    last_eye: [f32::NAN; 3],
                    uniform_buf,
                };

                let set_mut2 = self
                    .content
                    .gaussian_splat_store
                    .get_mut_by_index(store_index)
                    .unwrap();
                set_mut2.viewport_sort[viewport_index] = Some(vp_sort);
            }
        }
    }

    /// Run the GPU depth compute + 4-pass radix sort for one splat set / viewport.
    ///
    /// Uploads updated SplatUniform (viewport dims, model, sh_degree), dispatches
    /// the depth compute shader, then runs init_indices + 4 sort passes.
    pub(crate) fn run_gaussian_splat_sort(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        store_index: usize,
        viewport_index: usize,
        eye: [f32; 3],
        model: [[f32; 4]; 4],
        vp_w: f32,
        vp_h: f32,
        sh_degree: ShDegree,
    ) {
        // Ensure sort buffers and render BG exist.
        self.ensure_gaussian_splat_sort_buffers(device, store_index, viewport_index);

        let set = match self.content.gaussian_splat_store.get_by_index(store_index) {
            Some(s) => s,
            None => return,
        };
        let count = set.count;
        if count == 0 {
            return;
        }

        let vp_sort = match set.viewport_sort.get(viewport_index) {
            Some(Some(s)) => s,
            _ => return,
        };

        // Update the SplatUniform for this viewport.
        let splat_uni = SplatUniform {
            model,
            viewport_w: vp_w,
            viewport_h: vp_h,
            sh_degree: match sh_degree {
                ShDegree::Zero => 0,
                ShDegree::One => 1,
                ShDegree::Three => 3,
            },
            count,
        };
        queue.write_buffer(&vp_sort.uniform_buf, 0, bytemuck::bytes_of(&splat_uni));

        // Upload depth uniform.
        let depth_uni = DepthUniform { model, eye, count };
        let depth_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("splat_depth_uniform_tmp"),
            size: std::mem::size_of::<DepthUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&depth_uniform_buf, 0, bytemuck::bytes_of(&depth_uni));

        // Build depth BG.
        let depth_bg = {
            let bgl = self.gaussian_splat.depth_bgl.as_ref().unwrap();
            let set_ref = self
                .content
                .gaussian_splat_store
                .get_by_index(store_index)
                .unwrap();
            let vp_sort_ref = set_ref.viewport_sort[viewport_index].as_ref().unwrap();
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("splat_depth_bg"),
                layout: bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: depth_uniform_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: set_ref.position_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: vp_sort_ref.depth_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let workgroups = (count + 255) / 256;

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("splat_sort_encoder"),
        });

        // --- Depth compute pass ---
        {
            let depth_pipeline = self.gaussian_splat.depth_pipeline.as_ref().unwrap();
            let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("splat_depth_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(depth_pipeline);
            cpass.set_bind_group(0, &depth_bg, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy depth keys into keys_ping (depth_buf -> keys_ping).
        {
            let set_ref = self
                .content
                .gaussian_splat_store
                .get_by_index(store_index)
                .unwrap();
            let vp_sort_ref = set_ref.viewport_sort[viewport_index].as_ref().unwrap();
            encoder.copy_buffer_to_buffer(
                &vp_sort_ref.depth_buf,
                0,
                &vp_sort_ref.keys_ping,
                0,
                (count as u64) * 4,
            );
        }

        // --- 4-pass radix sort ---
        // Build per-pass sort uniforms (shift = 0, 8, 16, 24; pass_num = 0..3).
        // We need sort_bgl and references to sort buffers for each pass.
        let sort_bgl = self.gaussian_splat.sort_bgl.as_ref().unwrap();

        for pass in 0u32..4u32 {
            let sort_uni = SortUniform {
                shift: pass * 8,
                count,
                pass_num: pass,
                _pad: 0,
            };
            let sort_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("splat_sort_uniform_tmp"),
                size: std::mem::size_of::<SortUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&sort_uniform_buf, 0, bytemuck::bytes_of(&sort_uni));

            let set_ref = self
                .content
                .gaussian_splat_store
                .get_by_index(store_index)
                .unwrap();
            let vp_sort_ref = set_ref.viewport_sort[viewport_index].as_ref().unwrap();

            let sort_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("splat_sort_bg"),
                layout: sort_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: sort_uniform_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: vp_sort_ref.keys_ping.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: vp_sort_ref.keys_pong.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: vp_sort_ref.vals_ping.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 4,
                        resource: vp_sort_ref.vals_pong.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 5,
                        resource: vp_sort_ref.histogram_buf.as_entire_binding(),
                    },
                ],
            });

            // If pass 0: run init_indices first.
            if pass == 0 {
                let init_pipeline = self.gaussian_splat.sort_init_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                    label: Some("splat_init_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(init_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Clear histogram.
            {
                let clear_pipeline = self.gaussian_splat.sort_clear_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                    label: Some("splat_clear_hist"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(clear_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Histogram pass.
            {
                let hist_pipeline = self
                    .gaussian_splat
                    .sort_histogram_pipeline
                    .as_ref()
                    .unwrap();
                let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                    label: Some("splat_hist_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(hist_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Prefix sum.
            {
                let prefix_pipeline = self.gaussian_splat.sort_prefix_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                    label: Some("splat_prefix_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(prefix_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Scatter.
            {
                let scatter_pipeline = self.gaussian_splat.sort_scatter_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                    label: Some("splat_scatter_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(scatter_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Record eye so callers can skip re-sort if unchanged (optional optimisation).
        if let Some(set_mut) = self
            .content
            .gaussian_splat_store
            .get_mut_by_index(store_index)
        {
            if let Some(Some(vp_sort_mut)) = set_mut.viewport_sort.get_mut(viewport_index) {
                vp_sort_mut.last_eye = eye;
            }
        }
    }
}

#[cfg(test)]
mod async_tests {
    use crate::DeviceResources;
    use crate::renderer::GaussianSplatData;
    use crate::resources::UploadStatus;

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::Instance::new(&crate::gpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn sample_splats(n: usize) -> GaussianSplatData {
        let mut data = GaussianSplatData::default();
        data.positions = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
        data.scales = vec![[0.1, 0.1, 0.1]; n];
        data.rotations = vec![[0.0, 0.0, 0.0, 1.0]; n];
        data.opacities = vec![0.5; n];
        data
    }

    #[test]
    fn stale_splat_handle_does_not_alias_after_slot_reuse() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        // Upload a splat set, then remove it. The handle is now stale.
        let id1 = resources
            .upload_gaussian_splat(&device, &queue, &sample_splats(8))
            .unwrap();
        assert!(resources.content.gaussian_splat_store.get(id1).is_some());
        resources.free_gaussian_splat(id1);
        assert!(
            resources.content.gaussian_splat_store.get(id1).is_none(),
            "a removed handle must not resolve"
        );

        // The next upload reuses the freed slot at a new generation.
        let id2 = resources
            .upload_gaussian_splat(&device, &queue, &sample_splats(4))
            .unwrap();
        assert_eq!(id1.index(), id2.index(), "the freed slot should be reused");
        assert_ne!(id1, id2, "the reused slot must carry a new generation");
        assert!(resources.content.gaussian_splat_store.get(id2).is_some());
        assert!(
            resources.content.gaussian_splat_store.get(id1).is_none(),
            "the stale handle must not alias the set now occupying its slot"
        );
    }

    #[test]
    fn begin_upload_gaussian_splat_validates() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let err = resources
            .begin_upload_gaussian_splat(&device, &queue, GaussianSplatData::default())
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::InvalidGaussianSplatData { .. }
        ));
    }

    #[test]
    fn begin_upload_gaussian_splat_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources
            .begin_upload_gaussian_splat(&device, &queue, sample_splats(8))
            .expect("job submitted");
        for _ in 0..200 {
            resources.process_uploads(&device, &queue);
            match resources.upload_status(job) {
                UploadStatus::Ready => break,
                UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                UploadStatus::Unknown => panic!("job id disappeared"),
            }
        }
        let _id = resources.upload_result_gaussian_splat(job).expect("ready");
    }

    #[test]
    fn replace_gaussian_splat_keeps_handle_and_updates_bytes() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let id = resources
            .upload_gaussian_splat(&device, &queue, &sample_splats(8))
            .unwrap();
        let bytes_before = resources.resident_bytes().gaussian_splat_bytes;
        assert!(
            bytes_before > 0,
            "an uploaded set must count resident bytes"
        );

        // Replacing with a smaller set keeps the handle valid and shrinks bytes.
        resources
            .replace_gaussian_splat(&device, &queue, id, &sample_splats(2))
            .expect("replace on a live handle succeeds");
        assert!(resources.content.gaussian_splat_store.get(id).is_some());
        let bytes_after = resources.resident_bytes().gaussian_splat_bytes;
        assert!(
            bytes_after < bytes_before,
            "replacing with fewer splats must reduce resident bytes"
        );

        // A stale handle is rejected, not silently applied.
        resources.free_gaussian_splat(id);
        assert_eq!(resources.resident_bytes().gaussian_splat_bytes, 0);
        let err = resources
            .replace_gaussian_splat(&device, &queue, id, &sample_splats(2))
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::StaleHandle { .. }
        ));
    }
}

/// Per-viewport sort buffers for one Gaussian splat set.
pub(crate) struct GaussianSplatViewportSort {
    /// u32 view-space depth keys (flipped for back-to-front), written by depth compute each frame.
    pub depth_buf: crate::gpu::Buffer,
    /// Ping/pong key buffers for radix sort.
    pub keys_ping: crate::gpu::Buffer,
    pub keys_pong: crate::gpu::Buffer,
    /// Ping/pong value (index) buffers for radix sort.
    pub vals_ping: crate::gpu::Buffer,
    pub vals_pong: crate::gpu::Buffer,
    /// 256-entry atomic histogram / prefix-sum scratch.
    pub histogram_buf: crate::gpu::Buffer,
    /// Render bind group (group 1). Contains sorted_indices, positions, scales, rotations,
    /// opacities, sh_coefficients, and the per-viewport SplatUniform.
    pub render_bg: crate::gpu::BindGroup,
    /// Eye position at last sort; skip re-sort when unchanged.
    pub last_eye: [f32; 3],
    /// Per-viewport uniform buffer holding SplatUniform (model, viewport dims, sh_degree, count).
    pub uniform_buf: crate::gpu::Buffer,
}

/// Persistent GPU state for one uploaded Gaussian splat set.
pub(crate) struct GaussianSplatGpuSet {
    /// Positions as vec4<f32> (w=1), one per splat.
    pub position_buf: crate::gpu::Buffer,
    /// Scales as vec4<f32> (w=0), one per splat.
    pub scale_buf: crate::gpu::Buffer,
    /// Rotations as vec4<f32> [x,y,z,w], one per splat.
    pub rotation_buf: crate::gpu::Buffer,
    /// Opacities as f32, one per splat.
    pub opacity_buf: crate::gpu::Buffer,
    /// SH coefficients as f32, count = splat_count * sh_degree.coeff_count().
    pub sh_buf: crate::gpu::Buffer,
    /// SH degree for this set.
    pub sh_degree: crate::renderer::ShDegree,
    /// Number of splats.
    pub count: u32,
    /// Per-viewport sort buffers; index = viewport_index. Grown lazily.
    pub viewport_sort: Vec<Option<GaussianSplatViewportSort>>,
    /// CPU positions kept for potential picking (object-space).
    #[allow(dead_code)]
    pub cpu_positions: Vec<[f32; 3]>,
    /// CPU scales kept for potential picking.
    #[allow(dead_code)]
    pub cpu_scales: Vec<[f32; 3]>,
}

impl GaussianSplatGpuSet {
    /// Resident GPU bytes for the persistent source buffers (position, scale,
    /// rotation, opacity, SH). Per-viewport sort scratch is derived and grows
    /// lazily, so it is not counted here.
    pub fn gpu_bytes(&self) -> u64 {
        self.position_buf.size()
            + self.scale_buf.size()
            + self.rotation_buf.size()
            + self.opacity_buf.size()
            + self.sh_buf.size()
    }
}

/// Per-frame draw data produced in prepare_viewport_internal.
pub(crate) struct GaussianSplatDrawData {
    /// Index into gaussian_splat_store.
    pub store_index: usize,
    /// Viewport index that prepared this data.
    pub viewport_index: usize,
    /// Model matrix for this item.
    #[allow(dead_code)]
    pub model: [[f32; 4]; 4],
    /// Number of splats.
    pub count: u32,
    /// When true, skip the splat rasterization draw; a wireframe polyline overlay is rendered instead.
    pub wireframe: bool,
}

/// Slotted store for Gaussian splat sets with generational handles.
///
/// A removed set leaves an empty slot that a later insert reuses. Each slot
/// carries a generation bumped on removal, and a [`GaussianSplatId`] captures
/// the generation it was issued against, so a stale handle resolves to `None`
/// rather than aliasing the set now in its slot. An entry's byte charge is its
/// [`GaussianSplatGpuSet::gpu_bytes`].
pub(crate) struct GaussianSplatStore {
    store:
        crate::resources::handle::SlotStore<GaussianSplatGpuSet, crate::renderer::GaussianSplatId>,
}

impl GaussianSplatStore {
    pub fn new() -> Self {
        Self {
            store: crate::resources::handle::SlotStore::default(),
        }
    }

    pub fn insert(&mut self, set: GaussianSplatGpuSet) -> crate::renderer::GaussianSplatId {
        let bytes = set.gpu_bytes();
        self.store.insert(set, bytes)
    }

    /// Swap the set in `id`'s slot for `set`, keeping the slot generation so the
    /// handle stays valid. Returns `true` on success, `false` for a stale handle
    /// or an empty slot.
    pub fn replace(
        &mut self,
        id: crate::renderer::GaussianSplatId,
        set: GaussianSplatGpuSet,
    ) -> bool {
        let bytes = set.gpu_bytes();
        self.store.replace(id, set, bytes).is_some()
    }

    /// Total resident GPU bytes across every live splat set.
    pub fn allocated_bytes(&self) -> u64 {
        self.store.allocated_bytes()
    }

    /// Look up a set by handle, validating the generation. Returns `None` for a
    /// stale handle, an empty slot, or an out-of-range index.
    pub fn get(&self, id: crate::renderer::GaussianSplatId) -> Option<&GaussianSplatGpuSet> {
        self.store.get(id)
    }

    /// Look up a set by raw slot index, without a generation check. For the
    /// per-frame draw path, where the index was already validated through
    /// [`get`](Self::get) earlier in the same frame.
    pub fn get_by_index(&self, idx: usize) -> Option<&GaussianSplatGpuSet> {
        self.store.get_by_index(idx)
    }

    /// Mutable raw-index lookup, same contract as [`get_by_index`](Self::get_by_index).
    pub fn get_mut_by_index(&mut self, idx: usize) -> Option<&mut GaussianSplatGpuSet> {
        self.store.get_mut_by_index(idx)
    }

    /// Total number of slots (occupied plus free). Reported in stale-handle
    /// errors to show how many slots exist.
    pub fn slot_count(&self) -> usize {
        self.store.slot_count()
    }

    /// Remove a set by handle, bumping the slot generation and freeing the slot.
    /// Returns `true` if a set was removed, `false` for a stale handle or an
    /// already-empty slot.
    pub fn remove(&mut self, id: crate::renderer::GaussianSplatId) -> bool {
        self.store.remove(id).is_some()
    }
}
