use super::*;

/// The point cloud group-1 bind group layout. Uploads build their bind groups
/// against it, so it lives here with the store rather than with the item type's
/// pipelines, and is created up front: it is a layout, not a compiled pipeline.
pub(crate) struct PointCloudResources {
    /// Bind group layout for point cloud uniforms (group 1).
    pub(crate) bgl: crate::gpu::BindGroupLayout,
}

impl PointCloudResources {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_bgl"),
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
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
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

        Self { bgl }
    }
}

impl DeviceResources {
    /// Upload one [`PointCloudItem`] to the GPU and return draw data.
    ///
    /// Shared by the per-frame item upload and the pre-upload store.
    pub(crate) fn upload_point_cloud_per_frame(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::PointCloudItem,
    ) -> PointCloudGpuData {
        let point_count = item.positions.len() as u32;

        let pos_bytes: Vec<u8> = item
            .positions
            .iter()
            .flat_map(|p| bytemuck::bytes_of(p).iter().copied())
            .collect();
        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pc_vertex_buf"),
            size: pos_bytes.len().max(12) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &pos_bytes);

        let (scalar_buf, has_scalars, scalar_min, scalar_max) = if !item.scalars.is_empty() {
            let min = item
                .scalar_range
                .map(|r| r.0)
                .unwrap_or_else(|| item.scalars.iter().cloned().fold(f32::INFINITY, f32::min));
            let max = item.scalar_range.map(|r| r.1).unwrap_or_else(|| {
                item.scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            });
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_scalar_buf"),
                size: (std::mem::size_of::<f32>() * item.scalars.len()).max(4) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.scalars));
            (buf, 1u32, min, max)
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_scalar_buf_fallback"),
                size: 4,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32, 0.0f32, 1.0f32)
        };

        let (colour_buf, has_colours) = if !item.colours.is_empty() && has_scalars == 0 {
            let bytes: &[u8] = bytemuck::cast_slice(&item.colours);
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_colour_buf"),
                size: bytes.len().max(16) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytes);
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_colour_buf_fallback"),
                size: 16,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        // Radius buffer: radius_scalars (mapped to radius_range) take priority over
        // explicit per-point radii.
        let (radius_buf, has_radius) = if !item.radius_scalars.is_empty() {
            let r_min = item.radius_scalar_range.map(|r| r.0).unwrap_or_else(|| {
                item.radius_scalars
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min)
            });
            let r_max = item.radius_scalar_range.map(|r| r.1).unwrap_or_else(|| {
                item.radius_scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            });
            let range = (r_max - r_min).max(f32::EPSILON);
            let (out_min, out_max) = item.radius_range;
            let mapped: Vec<f32> = item
                .radius_scalars
                .iter()
                .map(|&s| {
                    let t = ((s - r_min) / range).clamp(0.0, 1.0);
                    out_min + t * (out_max - out_min)
                })
                .collect();
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_radius_buf"),
                size: (std::mem::size_of::<f32>() * mapped.len()).max(4) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&mapped));
            (buf, 1u32)
        } else if !item.radii.is_empty() {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_radius_buf"),
                size: (std::mem::size_of::<f32>() * item.radii.len()).max(4) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.radii));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_radius_buf_fallback"),
                size: 4,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        let (transparency_buf, has_transparency) = if !item.transparencies.is_empty() {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_transparency_buf"),
                size: (std::mem::size_of::<f32>() * item.transparencies.len()).max(4) as u64,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.transparencies));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("pc_transparency_buf_fallback"),
                size: 4,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PointCloudUniform {
            model: [[f32; 4]; 4],
            default_colour: [f32; 4],
            point_size: f32,
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            has_colours: u32,
            has_radius: u32,
            has_transparency: u32,
            gaussian: u32,
            // 0 = ScreenSpaceCircle, 1 = Sphere
            render_mode: u32,
            _pad: [u32; 3],
        }
        let uniform_data = PointCloudUniform {
            model: item.model,
            default_colour: item.default_colour.to_linear_rgba(),
            point_size: item.point_size,
            has_scalars,
            scalar_min,
            scalar_max,
            has_colours,
            has_radius,
            has_transparency,
            gaussian: if item.gaussian { 1 } else { 0 },
            render_mode: match item.render_mode {
                crate::renderer::PointRenderMode::ScreenSpaceCircle => 0,
                crate::renderer::PointRenderMode::Sphere => 1,
            },
            _pad: [0; 3],
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("pc_uniform_buf"),
            size: std::mem::size_of::<PointCloudUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = self
            .content
            .builtin_colourmap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colourmap_id
                    .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                self.content.colourmap_views.get(preset_id.0)
            })
            .unwrap_or(&self.content.fallback_lut_view);

        let lut_sampler = &self.material.sampler;

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("pc_bind_group"),
            layout: &self.point_cloud.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(lut_sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: scalar_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: colour_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: radius_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: transparency_buf.as_entire_binding(),
                },
            ],
        });

        PointCloudGpuData {
            vertex_buffer,
            point_count,
            pick_id: item.settings.pick_id,
            bind_group,
            _uniform_buf: uniform_buf,
            _scalar_buf: scalar_buf,
            _colour_buf: colour_buf,
            _radius_buf: radius_buf,
            _transparency_buf: transparency_buf,
        }
    }

    /// Pre-upload a point cloud and return a typed handle.
    ///
    /// Prefer [`ViewportRenderer::upload_point_cloud`](crate::renderer::ViewportRenderer::upload_point_cloud),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_point_cloud instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::PointCloudItem,
    ) -> crate::resources::PointCloudId {
        let gpu = self.upload_point_cloud_per_frame(device, queue, item);
        self.content.point_cloud_store.insert_sized(gpu)
    }

    /// Remove a pre-uploaded point cloud.
    ///
    /// Prefer [`ViewportRenderer::drop_point_cloud`](crate::renderer::ViewportRenderer::drop_point_cloud),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::drop_point_cloud instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn drop_point_cloud(&mut self, id: crate::resources::PointCloudId) -> bool {
        self.content.point_cloud_store.remove(id).is_some()
    }

    /// Replace the geometry of a pre-uploaded point cloud, keeping the same id.
    ///
    /// Prefer [`ViewportRenderer::replace_point_cloud`](crate::renderer::ViewportRenderer::replace_point_cloud),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::replace_point_cloud instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn replace_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::PointCloudId,
        item: &crate::renderer::PointCloudItem,
    ) -> bool {
        if !self.content.point_cloud_store.contains(id) {
            return false;
        }
        let gpu = self.upload_point_cloud_per_frame(device, queue, item);
        self.content
            .point_cloud_store
            .replace_sized(id, gpu)
            .is_some()
    }

    /// Start an asynchronous point cloud upload.
    ///
    /// Prefer [`ViewportRenderer::begin_upload_point_cloud`](crate::renderer::ViewportRenderer::begin_upload_point_cloud),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::begin_upload_point_cloud instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    // The apply step inserts through the synchronous upload, which goes with it.
    #[allow(deprecated)]
    pub fn begin_upload_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::PointCloudItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::PointCloudId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let pid = resources.upload_point_cloud(
                            &device_for_apply,
                            &queue_for_apply,
                            &item,
                        );
                        slot_for_apply.set(pid);
                    }),
                ))
            })
        };
        self.job_results
            .point_cloud
            .lock()
            .expect("point cloud result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`PointCloudId`](crate::resources::PointCloudId) produced by a
    /// completed [`begin_upload_point_cloud`](Self::begin_upload_point_cloud) job.
    ///
    /// Prefer [`ViewportRenderer::upload_result_point_cloud`](crate::renderer::ViewportRenderer::upload_result_point_cloud),
    /// which stays reachable when an item type holds its own storage.
    #[deprecated(
        since = "0.23.0",
        note = "call ViewportRenderer::upload_result_point_cloud instead: this content moves to the item type that draws it, which DeviceResources cannot reach"
    )]
    pub fn upload_result_point_cloud(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::PointCloudId> {
        let mut map = self
            .job_results
            .point_cloud
            .lock()
            .expect("point cloud result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(pid) => {
                map.remove(&id);
                Ok(pid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }
}

#[cfg(test)]
mod tests {
    // These drive the DeviceResources upload calls directly, which is the
    // point: they test the methods the renderer-level ones forward to.
    #![allow(deprecated)]
    use crate::DeviceResources;
    use crate::renderer::PointCloudItem;
    use crate::resources::UploadStatus;

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    /// The compute-filter pipeline pair is still lazily built. The point cloud
    /// layout is not: uploads bind against it, so it is created up front while
    /// the pipelines that use it belong to the item type.
    #[test]
    fn lazy_pipeline_pairs_start_empty() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let res = DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        assert!(res.compute_filter.pipeline.is_none());
        assert!(res.compute_filter.bgl.is_none());
    }

    fn sample_point_cloud() -> PointCloudItem {
        let mut item = PointCloudItem::default();
        item.positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        item.point_size = 6.0;
        item
    }

    #[test]
    fn upload_point_cloud_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_point_cloud(&device, &queue, &sample_point_cloud());
        assert!(resources.content.point_cloud_store.contains(id));
        assert!(resources.drop_point_cloud(id));
        assert!(!resources.content.point_cloud_store.contains(id));
    }

    #[test]
    fn begin_upload_point_cloud_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_point_cloud(&device, &queue, sample_point_cloud());
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
        let id = resources.upload_result_point_cloud(job).expect("ready");
        assert!(resources.content.point_cloud_store.contains(id));
        let err = resources.upload_result_point_cloud(job).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::JobResultMissing { .. }
        ));
    }
}

/// Per-frame GPU data for one point cloud item, created in `prepare()`.
#[derive(Clone)]
pub struct PointCloudGpuData {
    /// Vertex buffer: one entry per point, packed as `[position: vec3, _pad: f32]` (16 bytes).
    /// The shader reads colour/scalar from storage buffers indexed by `vertex_index`.
    pub(crate) vertex_buffer: crate::gpu::Buffer,
    /// Number of points (= draw count).
    pub(crate) point_count: u32,
    /// The item's pick id (from `settings.pick_id`); `PickId::NONE` when not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Bind group (group 1): uniform + LUT + sampler + scalar + colour + radius + transparency.
    pub(crate) bind_group: crate::gpu::BindGroup,
    // Keep the buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) _scalar_buf: crate::gpu::Buffer,
    pub(crate) _colour_buf: crate::gpu::Buffer,
    pub(crate) _radius_buf: crate::gpu::Buffer,
    pub(crate) _transparency_buf: crate::gpu::Buffer,
}
