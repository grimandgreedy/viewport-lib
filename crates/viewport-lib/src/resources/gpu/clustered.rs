//! Clustered-shading GPU resources.
//!
//! The cluster grid partitions screen space into `X_TILES * Y_TILES * Z_SLICES`
//! view-frustum cells. Each frame the build compute pass tags each cell with
//! the list of lights whose volume of influence intersects it; lit pipelines
//! then read just their cell's slice of the index list instead of scanning
//! every active light per fragment. Every cluster owns a fixed
//! `MAX_LIGHTS_PER_CLUSTER` slice of the index list, written in the host's
//! priority order (directionals, then punctuals ranked by importance and
//! proximity), so a crowded cluster degrades by dropping its lowest-priority
//! lights and never disturbs any other cluster.
//!
//! Bindings 14, 15, and 16 of the camera bind group expose the grid uniform,
//! the per-cell offsets, and the global index list to every lit pipeline. The
//! build pass uses a separate compute bind group with read-write access.

use crate::resources::builders::LoggedAlloc;

/// X (screen-tile) count of the cluster grid. Aligns with 16:9 aspect framing.
pub const CLUSTER_X_TILES: u32 = 16;
/// Y (screen-tile) count of the cluster grid.
pub const CLUSTER_Y_TILES: u32 = 9;
/// Z (depth-slice) count. The far bound is fitted per frame to the punctual
/// lights' reach (max view depth + range, clamped to the camera far) and the
/// cluster near plane sits at far/32, so the slices run log-uniform across
/// the depth range that actually holds lit geometry instead of burning most
/// of the budget on the first few metres. Fragments nearer than the cluster
/// near plane clamp to slice 0 (whose AABB extends to the camera); fragments
/// beyond the fitted far clamp to the last slice.
pub const CLUSTER_Z_SLICES: u32 = 24;
/// Total cluster cell count (`16 * 9 * 24 = 3456`).
pub const CLUSTER_COUNT: u32 = CLUSTER_X_TILES * CLUSTER_Y_TILES * CLUSTER_Z_SLICES;
/// Fixed light-index capacity per cluster cell. Each cluster owns its own
/// slice of the index list, so a crowded cluster can never starve another,
/// and overflow within a cluster drops the lowest-priority lights (the host
/// orders the light array by importance and proximity). Must match
/// `MAX_PER_CLUSTER` in `cluster_build.wgsl`.
pub const MAX_LIGHTS_PER_CLUSTER: u32 = 64;
/// Total light-index list capacity: one fixed slice per cluster
/// (`3456 * 64` entries, 864 KB at 4 bytes per index).
pub const MAX_LIGHT_INDICES: u32 = CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER;
/// Below this active-light count the build pass is skipped and the fragment
/// shader iterates the full light array directly. Straight iteration is
/// cheaper than cluster lookup overhead for small light counts.
pub const SMALL_N_THRESHOLD: u32 = 16;

/// Per-frame cluster grid metadata uniform.
///
/// Bound at group 0 binding 14. The fragment shader reads `dimensions` and
/// `depth` to map a view-space fragment to a cluster index. The same uniform
/// drives the build compute pass.
///
/// Layout is 64 bytes, 16-byte aligned: four `vec4` worth of state with the
/// fields documented on each `pub` member below.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterGridUniform {
    /// (x_tiles, y_tiles, z_slices, total_count).
    pub dimensions: [u32; 4],
    /// (cluster_near, far, log(far/cluster_near), active_light_count). The
    /// cluster near plane is far/32 (clamped to at least the camera near),
    /// not the camera near plane; see `CLUSTER_Z_SLICES`.
    pub depth: [f32; 4],
    /// (screen_w, screen_h, fallback_mode, _pad). `fallback_mode != 0` signals
    /// the small-N fallback path to the shader, which then iterates the full
    /// light array instead of the cluster list.
    pub screen: [f32; 4],
    /// (tan_half_fov_x, tan_half_fov_y, _pad, _pad). Used by the build pass
    /// to compute per-cluster view-space AABBs from screen-tile NDC bounds.
    pub proj_scale: [f32; 4],
    /// World-to-view matrix. Lets the fragment shader compute a view-space
    /// position without growing each consumer's per-shader `Camera` struct.
    pub view: [[f32; 4]; 4],
}

impl Default for ClusterGridUniform {
    fn default() -> Self {
        Self {
            dimensions: [
                CLUSTER_X_TILES,
                CLUSTER_Y_TILES,
                CLUSTER_Z_SLICES,
                CLUSTER_COUNT,
            ],
            depth: [0.1, 1000.0, (1000.0_f32 / 0.1_f32).ln(), 0.0],
            screen: [1.0, 1.0, 1.0, 0.0],
            proj_scale: [1.0, 1.0, 0.0, 0.0],
            view: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

/// Per-frame, per-light view-space data consumed by the cluster build pass.
///
/// Indices into this buffer match indices into `light_storage_buf` one-to-one,
/// so the `u32` written by the build pass into the global index list is also
/// a valid index into the per-fragment light array.
///
/// Layout is 48 bytes, 16-byte aligned. Field semantics are documented on
/// each member below.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ActiveLightView {
    /// (view_pos.xyz, range). Directional lights leave xyz = 0, range = inf.
    pub view_pos_range: [f32; 4],
    /// (light_type, _pad, _pad, _pad). 0=directional, 1=point, 2=spot.
    pub type_pad: [u32; 4],
    /// (view_spot_dir.xyz, cos_outer_angle). Unused for non-spot lights.
    pub spot_data: [f32; 4],
}

/// Uniform written by the host to drive the no-op clear compute pass.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ClearParams {
    cluster_count: u32,
    index_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU-side cluster-cell layout. 8 bytes per cluster; matches WGSL struct
/// `ClusterCell { offset: u32, count: u32 }`.
/// Per-frame diagnostics produced by reading back the cluster cell array.
///
/// Useful for verifying that the build pass is doing meaningful work and for
/// choosing grid dimensions in tuning. Pulled by the host on demand via
/// `ViewportRenderer::cluster_stats`; the readback is skipped when no
/// consumer asks for it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClusterStats {
    /// Total cluster cells in the grid (constant per build).
    pub total_cells: u32,
    /// Cells with at least one punctual (point or spot) light assigned.
    pub non_empty_cells: u32,
    /// Maximum punctual demand across all cells (lights intersecting the
    /// cell, before the per-cluster capacity cap).
    pub max_punctual: u32,
    /// Median punctual demand across cells with at least one punctual.
    pub median_punctual: u32,
    /// 99th-percentile punctual demand across cells with at least one
    /// punctual.
    pub p99_punctual: u32,
    /// Mean punctual demand across non-empty cells.
    pub mean_punctual: f32,
    /// Sum of `cell.count` across all cells : how many light-index slots the
    /// build pass actually wrote this frame.
    pub total_index_slots_used: u32,
    /// Capacity of the light index list (one fixed
    /// `MAX_LIGHTS_PER_CLUSTER` slice per cluster).
    pub max_index_slots: u32,
    /// Punctual lights dropped by per-cluster capacity this frame, summed
    /// over all cells (`punctual_demand - punctual_count`). Zero means no
    /// cluster overflowed its slice.
    pub dropped_punctual_slots: u32,
    /// Active light count after the CPU frustum cull.
    pub active_light_count: u32,
    /// True if the frame ran the small-N or force-fallback path. In that
    /// case the cell stats are stale (last build) but `active_light_count`
    /// is still meaningful.
    pub fallback_active: bool,
}

/// One cell in the clustered light grid. Points into the global light index
/// list and records how many lights of each kind affect the cluster.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterCell {
    /// Offset into the global light index list at which this cluster's light
    /// indices start.
    pub offset: u32,
    /// Number of light indices owned by this cluster. Includes directionals,
    /// since the fragment shader iterates this many slots out of the global
    /// index list.
    pub count: u32,
    /// Subset of `count` covering point and spot lights only. The debug
    /// overlay reads this so the ever-present directional fill doesn't drown
    /// out the per-cluster light density signal.
    pub punctual_count: u32,
    /// Punctual lights that intersect this cluster, before the per-cluster
    /// capacity cap. `punctual_demand - punctual_count` is how many lights
    /// this cluster dropped.
    pub punctual_demand: u32,
}

/// What the clear and build compute builds read: a label, layout and module
/// per pipeline.
pub struct ClusterRecipe {
    device: crate::gpu::Device,
    stages: [(
        &'static str,
        crate::gpu::PipelineLayout,
        crate::resources::pipeline_slot::LazyModule,
    ); 2],
}

const CLUSTER_CLEAR: usize = 0;
const CLUSTER_BUILD: usize = 1;

fn build_cluster(r: &ClusterRecipe, i: usize) -> crate::gpu::ComputePipeline {
    let (label, layout, shader) = &r.stages[i];
    crate::resources::builders::compute_pipeline(&r.device, label, layout, shader.get(), "main")
}

/// All clustered-shading state owned by `DeviceResources`.
pub struct ClusteredResources {
    /// `ClusterGridUniform` uniform buffer (group 0 binding 14).
    pub grid_uniform_buf: crate::gpu::Buffer,
    /// Cluster cell storage (group 0 binding 15, read-only fragment). A
    /// one-cell placeholder until `ensure_pipelines` allocates the full grid;
    /// the fragment shader reads it only once a frame uses the grid.
    pub cluster_grid_buf: crate::gpu::Buffer,
    /// Global light index list (group 0 binding 16, read-only fragment). A
    /// placeholder until `ensure_pipelines`, like `cluster_grid_buf`.
    pub light_index_buf: crate::gpu::Buffer,
    /// View-space data for the active (post-cull) light set, uploaded each
    /// frame and consumed by the build pass.
    pub active_lights_buf: crate::gpu::Buffer,
    /// Single u32 counter from the old shared-allocator scheme. The build
    /// pass no longer touches it (clusters own fixed slices); it stays
    /// allocated and zeroed so the clear bind group layout is unchanged.
    pub global_offset_buf: crate::gpu::Buffer,
    /// CPU-readable staging buffer that mirrors `cluster_grid_buf`. Allocated
    /// by the first `read_stats`.
    stats_staging_buf: std::sync::OnceLock<crate::gpu::Buffer>,
    /// Bind group for the cluster-clear compute pass, made with the full
    /// buffers by `ensure_pipelines`.
    clear_bind_group: Option<crate::gpu::BindGroup>,
    /// Layout the clear pipeline is built against.
    clear_bgl: crate::gpu::BindGroupLayout,
    /// The clear pipeline (zeroes both storage buffers) and the build
    /// pipeline (intersects each cluster with the active lights). `None`
    /// until a frame needs a dispatch; see `ensure_pipelines`.
    pipelines: Option<
        crate::resources::pipeline_slot::LazyFamily<ClusterRecipe, 2, crate::gpu::ComputePipeline>,
    >,
    /// Bind group for the cluster-build compute pass, made by `ensure_pipelines`.
    build_bind_group: Option<crate::gpu::BindGroup>,
    /// Layout the build pipeline is built against.
    build_bgl: crate::gpu::BindGroupLayout,
    /// Whether the cluster grid and index list are known to hold zero. wgpu
    /// zero-initialises both buffers, so this starts true and only a build
    /// dispatch makes it false: a viewport whose lights never reach the cluster
    /// threshold never runs the clear at all.
    grid_zeroed: bool,
    /// Uniform buffer for the clear pass parameters (constants for now).
    clear_params_buf: crate::gpu::Buffer,
}

impl ClusteredResources {
    /// Allocate the cluster grid uniform, the cluster-cell storage, the global
    /// light index list, and the clear / build compute pipelines.
    pub fn new(device: &crate::gpu::Device) -> Self {
        let grid_uniform_buf = device.logged_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("cluster_grid_uniform_buf"),
            contents: bytemuck::cast_slice(&[ClusterGridUniform::default()]),
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
        });

        let (cluster_grid_buf, light_index_buf) = Self::grid_buffers(device, 1, 4);

        let active_lights_bytes = (crate::resources::MAX_SCENE_LIGHTS as u64)
            * std::mem::size_of::<ActiveLightView>() as u64;
        let active_lights_buf = device.logged_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cluster_active_lights_buf"),
            size: active_lights_bytes,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let global_offset_buf = device.logged_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cluster_global_offset_buf"),
            size: 4,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clear_params_buf = device.logged_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("cluster_clear_params_buf"),
            contents: bytemuck::cast_slice(&[ClearParams {
                cluster_count: CLUSTER_COUNT,
                index_count: MAX_LIGHT_INDICES,
                _pad0: 0,
                _pad1: 0,
            }]),
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
        });

        let storage_entry = |binding: u32, read_only: bool| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: crate::gpu::ShaderStages::COMPUTE,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: crate::gpu::ShaderStages::COMPUTE,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let clear_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("cluster_clear_bgl"),
            entries: &[
                storage_entry(0, false), // cluster_grid
                storage_entry(1, false), // light_indices
                storage_entry(2, false), // global_offset_counter
                uniform_entry(3),        // ClearParams
            ],
        });

        // Build pass : intersects each cluster's view-space AABB with the
        // active-light set and writes the per-cluster light index ranges.
        let build_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("cluster_build_bgl"),
            entries: &[
                storage_entry(0, false), // cluster_grid
                storage_entry(1, false), // light_indices
                storage_entry(2, false), // global_offset_counter
                uniform_entry(3),        // GridUniform
                storage_entry(4, true),  // active_lights
            ],
        });
        Self {
            grid_uniform_buf,
            cluster_grid_buf,
            light_index_buf,
            active_lights_buf,
            global_offset_buf,
            stats_staging_buf: std::sync::OnceLock::new(),
            clear_bind_group: None,
            clear_bgl,
            pipelines: None,
            build_bind_group: None,
            build_bgl,
            grid_zeroed: true,
            clear_params_buf,
        }
    }

    /// The cluster grid and light index storage, sized for `cells` cells and
    /// `indices` indices.
    fn grid_buffers(
        device: &crate::gpu::Device,
        cells: u32,
        indices: u32,
    ) -> (crate::gpu::Buffer, crate::gpu::Buffer) {
        let cluster_grid_buf = device.logged_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cluster_grid_buf"),
            size: cells as u64 * std::mem::size_of::<ClusterCell>() as u64,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::COPY_DST
                | crate::gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let light_index_buf = device.logged_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cluster_light_index_buf"),
            size: indices as u64 * 4,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (cluster_grid_buf, light_index_buf)
    }

    /// Replace the placeholder grid buffers with full-size ones and make the
    /// compute bind groups over them. Returns false if already done.
    fn allocate_grid(&mut self, device: &crate::gpu::Device) -> bool {
        if self.clear_bind_group.is_some() {
            return false;
        }
        let (cluster_grid_buf, light_index_buf) =
            Self::grid_buffers(device, CLUSTER_COUNT, MAX_LIGHT_INDICES);
        self.cluster_grid_buf = cluster_grid_buf;
        self.light_index_buf = light_index_buf;
        let clear_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("cluster_clear_bind_group"),
            layout: &self.clear_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cluster_grid_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: self.light_index_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: self.global_offset_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: self.clear_params_buf.as_entire_binding(),
                },
            ],
        });

        let build_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("cluster_build_bind_group"),
            layout: &self.build_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cluster_grid_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: self.light_index_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: self.global_offset_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: self.grid_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: self.active_lights_buf.as_entire_binding(),
                },
            ],
        });
        self.clear_bind_group = Some(clear_bind_group);
        self.build_bind_group = Some(build_bind_group);
        true
    }

    /// Allocate the full cluster grid and compose the clear and build compute
    /// pipelines: their layouts and modules, with each pipeline built under
    /// the compilation policy the first time
    /// [`pipelines_ready`](Self::pipelines_ready) asks for it. Called by the
    /// lighting prepare on the first frame that has a dispatch to encode.
    ///
    /// Returns true when it replaced the grid buffers, which the camera bind
    /// groups name: the caller rebuilds those before anything draws.
    pub fn ensure_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        compiler: &std::sync::Arc<crate::resources::pipeline_slot::PipelineCompiler>,
    ) -> bool {
        let replaced = self.allocate_grid(device);
        if self.pipelines.is_some() {
            return replaced;
        }
        let module = |label: &str, source: &str| {
            crate::resources::pipeline_slot::LazyModule::new(
                device,
                label,
                source,
                Default::default(),
            )
        };
        let recipe = ClusterRecipe {
            device: device.clone(),
            stages: [
                (
                    "cluster_clear_pipeline",
                    crate::resources::builders::pipeline_layout(
                        device,
                        "cluster_clear_pipeline_layout",
                        &[&self.clear_bgl],
                    ),
                    module(
                        "cluster_clear_shader",
                        crate::resources::builders::wgsl_source!("cluster_clear"),
                    ),
                ),
                (
                    "cluster_build_pipeline",
                    crate::resources::builders::pipeline_layout(
                        device,
                        "cluster_build_pipeline_layout",
                        &[&self.build_bgl],
                    ),
                    module(
                        "cluster_build_shader",
                        crate::resources::builders::wgsl_source!("cluster_build"),
                    ),
                ),
            ],
        };
        self.pipelines = Some(crate::resources::pipeline_slot::LazyFamily::new(
            recipe,
            std::sync::Arc::clone(compiler),
            build_cluster,
        ));
        replaced
    }

    /// Ask for both compute pipelines, for a warm-up. Needs `ensure_pipelines`
    /// first.
    pub fn request_all(&self) {
        if let Some(p) = &self.pipelines {
            p.request_all();
        }
    }

    /// Whether both compute pipelines are built, asking for any that is not.
    /// While this is false the lighting prepare takes the per-light fallback,
    /// which shades the same without the grid.
    pub fn pipelines_ready(&self) -> bool {
        self.pipelines
            .as_ref()
            .is_some_and(|p| p.get(CLUSTER_CLEAR).is_some() & p.get(CLUSTER_BUILD).is_some())
    }

    /// Whether a clear dispatch is owed: something has written the cluster grid
    /// since it was last known to hold zero.
    pub fn grid_dirty(&self) -> bool {
        !self.grid_zeroed
    }

    /// Copy `cluster_grid_buf` to host-readable memory, map it, and compute
    /// `ClusterStats`. Blocks on a device poll while the GPU finishes the
    /// copy, so this should be called sparingly : it's a debug-path readback
    /// behind a host-controlled toggle, not a per-frame operation.
    pub fn read_stats(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        active_light_count: u32,
        fallback_active: bool,
    ) -> ClusterStats {
        if self.clear_bind_group.is_none() {
            // No frame has used the grid, so every cell is empty.
            return compute_stats(&[], active_light_count, fallback_active);
        }
        let bytes = (CLUSTER_COUNT as u64) * std::mem::size_of::<ClusterCell>() as u64;
        let staging = self.stats_staging_buf.get_or_init(|| {
            device.logged_buffer(&crate::gpu::BufferDescriptor {
                label: Some("cluster_stats_staging_buf"),
                size: bytes,
                usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("cluster_stats_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.cluster_grid_buf, 0, staging, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(crate::gpu::MapMode::Read, |_| {});
        let _ = device.poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });

        let stats = {
            let data = crate::gpu::mapped_range(slice);
            let cells: &[ClusterCell] = bytemuck::cast_slice(&data);
            compute_stats(cells, active_light_count, fallback_active)
        };
        staging.unmap();
        stats
    }

    /// Update the per-frame `ClusterGridUniform` (screen size, near/far, fallback mode).
    pub fn write_grid_uniform(&self, queue: &crate::gpu::Queue, uniform: &ClusterGridUniform) {
        queue.write_buffer(&self.grid_uniform_buf, 0, bytemuck::cast_slice(&[*uniform]));
    }

    /// Upload the active-lights view-space data for the build pass. Truncates
    /// silently if the slice is larger than `MAX_SCENE_LIGHTS`.
    pub fn write_active_lights(&self, queue: &crate::gpu::Queue, lights: &[ActiveLightView]) {
        if lights.is_empty() {
            return;
        }
        let n = lights.len().min(crate::resources::MAX_SCENE_LIGHTS);
        queue.write_buffer(
            &self.active_lights_buf,
            0,
            bytemuck::cast_slice(&lights[..n]),
        );
    }

    /// Encode the per-frame clear + build dispatches. The clear returns the
    /// cluster grid and the global reservation counter to zero, and runs only
    /// when a previous build left them non-zero; the build is skipped when no
    /// active lights survive the CPU cull. Both are skipped when
    /// `ensure_pipelines` has not run, which is the case until a frame needs one.
    pub fn dispatch_frame(
        &mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        active_light_count: u32,
        ts_query_set: Option<&crate::gpu::QuerySet>,
    ) {
        let built = |i| self.pipelines.as_ref().and_then(|p| p.get(i));
        let (Some(clear_bind_group), Some(build_bind_group)) =
            (&self.clear_bind_group, &self.build_bind_group)
        else {
            return;
        };
        if let (false, Some(clear_pipeline)) = (self.grid_zeroed, built(CLUSTER_CLEAR)) {
            let clear_workgroups = MAX_LIGHT_INDICES.max(CLUSTER_COUNT).div_ceil(64);
            let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("cluster_clear_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(clear_pipeline);
            pass.set_bind_group(0, clear_bind_group, &[]);
            pass.dispatch_workgroups(clear_workgroups, 1, 1);
            self.grid_zeroed = true;
        }
        if active_light_count == 0 {
            return;
        }
        let Some(build_pipeline) = built(CLUSTER_BUILD) else {
            return;
        };
        {
            let slot = crate::renderer::GPU_TS_CLUSTER;
            let ts_writes = ts_query_set.map(|qs| crate::gpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(slot * 2),
                end_of_pass_write_index: Some(slot * 2 + 1),
            });
            let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("cluster_build_pass"),
                timestamp_writes: ts_writes,
            });
            pass.set_pipeline(build_pipeline);
            pass.set_bind_group(0, build_bind_group, &[]);
            // One workgroup per cluster cell.
            pass.dispatch_workgroups(CLUSTER_COUNT, 1, 1);
        }
        self.grid_zeroed = false;
    }
}

/// Build a `ClusterStats` snapshot from a host-visible copy of the cluster
/// cell array. Empty cells (`punctual_count == 0`) are excluded from the
/// median, p99, and mean so they don't drag the signal toward zero on
/// sparsely-populated grids.
fn compute_stats(
    cells: &[ClusterCell],
    active_light_count: u32,
    fallback_active: bool,
) -> ClusterStats {
    let total_cells = cells.len() as u32;
    let mut total_index_slots_used: u32 = 0;
    let mut dropped: u32 = 0;
    let mut punctuals: Vec<u32> = Vec::with_capacity(cells.len());
    let mut max_punctual: u32 = 0;
    let mut non_empty: u32 = 0;
    for c in cells {
        total_index_slots_used = total_index_slots_used.saturating_add(c.count);
        dropped = dropped.saturating_add(c.punctual_demand.saturating_sub(c.punctual_count));
        if c.punctual_demand > 0 {
            non_empty += 1;
            punctuals.push(c.punctual_demand);
            if c.punctual_demand > max_punctual {
                max_punctual = c.punctual_demand;
            }
        }
    }
    punctuals.sort_unstable();

    let median = if punctuals.is_empty() {
        0
    } else {
        punctuals[punctuals.len() / 2]
    };
    let p99 = if punctuals.is_empty() {
        0
    } else {
        let idx = ((punctuals.len() as f32) * 0.99) as usize;
        punctuals[idx.min(punctuals.len() - 1)]
    };
    let mean = if punctuals.is_empty() {
        0.0
    } else {
        let sum: u64 = punctuals.iter().map(|&v| v as u64).sum();
        (sum as f32) / (punctuals.len() as f32)
    };

    ClusterStats {
        total_cells,
        non_empty_cells: non_empty,
        max_punctual,
        median_punctual: median,
        p99_punctual: p99,
        mean_punctual: mean,
        total_index_slots_used,
        max_index_slots: MAX_LIGHT_INDICES,
        dropped_punctual_slots: dropped,
        active_light_count,
        fallback_active,
    }
}
