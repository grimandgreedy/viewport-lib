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

use wgpu::util::DeviceExt;

/// X (screen-tile) count of the cluster grid. Aligns with 16:9 aspect framing.
pub const CLUSTER_X_TILES: u32 = 16;
/// Y (screen-tile) count of the cluster grid.
pub const CLUSTER_Y_TILES: u32 = 9;
/// Z (depth-slice) count. Log-uniform from near to far in the build pass.
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
    /// (near, far, log(far/near), active_light_count).
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

/// All clustered-shading state owned by `DeviceResources`.
pub struct ClusteredResources {
    /// `ClusterGridUniform` uniform buffer (group 0 binding 14).
    pub grid_uniform_buf: wgpu::Buffer,
    /// Cluster cell storage (group 0 binding 15, read-only fragment).
    pub cluster_grid_buf: wgpu::Buffer,
    /// Global light index list (group 0 binding 16, read-only fragment).
    pub light_index_buf: wgpu::Buffer,
    /// View-space data for the active (post-cull) light set, uploaded each
    /// frame and consumed by the build pass.
    pub active_lights_buf: wgpu::Buffer,
    /// Single u32 counter from the old shared-allocator scheme. The build
    /// pass no longer touches it (clusters own fixed slices); it stays
    /// allocated and zeroed so the clear bind group layout is unchanged.
    pub global_offset_buf: wgpu::Buffer,
    /// CPU-readable staging buffer that mirrors `cluster_grid_buf`. Populated
    /// only when the host calls `read_stats`.
    stats_staging_buf: wgpu::Buffer,
    /// Bind group for the cluster-clear compute pass.
    clear_bind_group: wgpu::BindGroup,
    /// Compute pipeline that zeroes both storage buffers each frame.
    clear_pipeline: wgpu::ComputePipeline,
    /// Bind group for the cluster-build compute pass.
    build_bind_group: wgpu::BindGroup,
    /// Compute pipeline that intersects each cluster with the active lights.
    build_pipeline: wgpu::ComputePipeline,
    /// Uniform buffer for the clear pass parameters (constants for now).
    #[allow(dead_code)]
    clear_params_buf: wgpu::Buffer,
}

impl ClusteredResources {
    /// Allocate the cluster grid uniform, the cluster-cell storage, the global
    /// light index list, and the clear / build compute pipelines.
    pub fn new(device: &wgpu::Device) -> Self {
        let grid_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cluster_grid_uniform_buf"),
            contents: bytemuck::cast_slice(&[ClusterGridUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cluster_grid_bytes = (CLUSTER_COUNT as u64) * std::mem::size_of::<ClusterCell>() as u64;
        let cluster_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_grid_buf"),
            size: cluster_grid_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let light_index_bytes = (MAX_LIGHT_INDICES as u64) * 4;
        let light_index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_light_index_buf"),
            size: light_index_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let active_lights_bytes = (crate::resources::MAX_SCENE_LIGHTS as u64)
            * std::mem::size_of::<ActiveLightView>() as u64;
        let active_lights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_active_lights_buf"),
            size: active_lights_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let global_offset_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_global_offset_buf"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stats_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_stats_staging_buf"),
            size: cluster_grid_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let clear_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cluster_clear_params_buf"),
            contents: bytemuck::cast_slice(&[ClearParams {
                cluster_count: CLUSTER_COUNT,
                index_count: MAX_LIGHT_INDICES,
                _pad0: 0,
                _pad1: 0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster_clear_bgl"),
            entries: &[
                storage_entry(0, false), // cluster_grid
                storage_entry(1, false), // light_indices
                storage_entry(2, false), // global_offset_counter
                uniform_entry(3),        // ClearParams
            ],
        });

        let clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster_clear_bind_group"),
            layout: &clear_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cluster_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: global_offset_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: clear_params_buf.as_entire_binding(),
                },
            ],
        });

        let clear_shader = crate::resources::builders::wgsl_module(
            device,
            "cluster_clear_shader",
            crate::resources::builders::wgsl_source!("cluster_clear"),
        );
        let clear_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster_clear_pipeline_layout"),
            bind_group_layouts: &[&clear_bgl],
            push_constant_ranges: &[],
        });
        let clear_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "cluster_clear_pipeline",
            &clear_layout,
            &clear_shader,
            "main",
        );

        // Build pass : intersects each cluster's view-space AABB with the
        // active-light set and writes the per-cluster light index ranges.
        let build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster_build_bgl"),
            entries: &[
                storage_entry(0, false), // cluster_grid
                storage_entry(1, false), // light_indices
                storage_entry(2, false), // global_offset_counter
                uniform_entry(3),        // GridUniform
                storage_entry(4, true),  // active_lights
            ],
        });
        let build_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster_build_bind_group"),
            layout: &build_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cluster_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: global_offset_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grid_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: active_lights_buf.as_entire_binding(),
                },
            ],
        });
        let build_shader = crate::resources::builders::wgsl_module(
            device,
            "cluster_build_shader",
            crate::resources::builders::wgsl_source!("cluster_build"),
        );
        let build_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster_build_pipeline_layout"),
            bind_group_layouts: &[&build_bgl],
            push_constant_ranges: &[],
        });
        let build_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "cluster_build_pipeline",
            &build_layout,
            &build_shader,
            "main",
        );

        Self {
            grid_uniform_buf,
            cluster_grid_buf,
            light_index_buf,
            active_lights_buf,
            global_offset_buf,
            stats_staging_buf,
            clear_bind_group,
            clear_pipeline,
            build_bind_group,
            build_pipeline,
            clear_params_buf,
        }
    }

    /// Copy `cluster_grid_buf` to host-readable memory, map it, and compute
    /// `ClusterStats`. Blocks on a device poll while the GPU finishes the
    /// copy, so this should be called sparingly : it's a debug-path readback
    /// behind a host-controlled toggle, not a per-frame operation.
    pub fn read_stats(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        active_light_count: u32,
        fallback_active: bool,
    ) -> ClusterStats {
        let bytes = (CLUSTER_COUNT as u64) * std::mem::size_of::<ClusterCell>() as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cluster_stats_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.cluster_grid_buf, 0, &self.stats_staging_buf, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.stats_staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });

        let stats = {
            let data = slice.get_mapped_range();
            let cells: &[ClusterCell] = bytemuck::cast_slice(&data);
            compute_stats(cells, active_light_count, fallback_active)
        };
        self.stats_staging_buf.unmap();
        stats
    }

    /// Update the per-frame `ClusterGridUniform` (screen size, near/far, fallback mode).
    pub fn write_grid_uniform(&self, queue: &wgpu::Queue, uniform: &ClusterGridUniform) {
        queue.write_buffer(&self.grid_uniform_buf, 0, bytemuck::cast_slice(&[*uniform]));
    }

    /// Upload the active-lights view-space data for the build pass. Truncates
    /// silently if the slice is larger than `MAX_SCENE_LIGHTS`.
    pub fn write_active_lights(&self, queue: &wgpu::Queue, lights: &[ActiveLightView]) {
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

    /// Encode the per-frame clear + build dispatches. Always runs the clear so
    /// the cluster grid and global reservation counter return to a known zero
    /// state; the build is skipped when no active lights survive the CPU cull.
    pub fn dispatch_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        active_light_count: u32,
        ts_query_set: Option<&wgpu::QuerySet>,
    ) {
        {
            let clear_workgroups = MAX_LIGHT_INDICES.max(CLUSTER_COUNT).div_ceil(64);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cluster_clear_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, &self.clear_bind_group, &[]);
            pass.dispatch_workgroups(clear_workgroups, 1, 1);
        }
        if active_light_count == 0 {
            return;
        }
        {
            let slot = crate::renderer::GPU_TS_CLUSTER;
            let ts_writes = ts_query_set.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(slot * 2),
                end_of_pass_write_index: Some(slot * 2 + 1),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cluster_build_pass"),
                timestamp_writes: ts_writes,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, &self.build_bind_group, &[]);
            // One workgroup per cluster cell.
            pass.dispatch_workgroups(CLUSTER_COUNT, 1, 1);
        }
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
