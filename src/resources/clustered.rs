//! Clustered-shading GPU resources.
//!
//! The cluster grid partitions screen space into `X_TILES * Y_TILES * Z_SLICES`
//! view-frustum cells. Each frame the build compute pass tags each cell with
//! the list of lights whose volume of influence intersects it; lit pipelines
//! then read just their cell's slice of the global index list instead of
//! scanning every active light per fragment.
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
/// Maximum total light-index references shared across all clusters. At 4 bytes
/// per index this caps the global list at 128 KB. The build pass drops the
/// low-importance tail for any clusters that would push past this cap.
pub const MAX_LIGHT_INDICES: u32 = 32 * 1024;
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
}

impl Default for ClusterGridUniform {
    fn default() -> Self {
        Self {
            dimensions: [CLUSTER_X_TILES, CLUSTER_Y_TILES, CLUSTER_Z_SLICES, CLUSTER_COUNT],
            depth: [0.1, 1000.0, (1000.0_f32 / 0.1_f32).ln(), 0.0],
            screen: [1.0, 1.0, 1.0, 0.0],
            proj_scale: [1.0, 1.0, 0.0, 0.0],
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
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterCell {
    /// Offset into the global light index list at which this cluster's light
    /// indices start.
    pub offset: u32,
    /// Number of light indices owned by this cluster.
    pub count: u32,
}

/// All clustered-shading state owned by `ViewportGpuResources`.
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
    /// Single u32 atomic counter used by the build pass to reserve contiguous
    /// regions of `light_index_buf`. Reset to zero each frame by the clear.
    pub global_offset_buf: wgpu::Buffer,
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
    /// light index list, and the no-op clear compute pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let grid_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cluster_grid_uniform_buf"),
            contents: bytemuck::cast_slice(&[ClusterGridUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cluster_grid_bytes =
            (CLUSTER_COUNT as u64) * std::mem::size_of::<ClusterCell>() as u64;
        let cluster_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_grid_buf"),
            size: cluster_grid_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_index_bytes = (MAX_LIGHT_INDICES as u64) * 4;
        let light_index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster_light_index_buf"),
            size: light_index_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let active_lights_bytes =
            (crate::resources::MAX_SCENE_LIGHTS as u64) * std::mem::size_of::<ActiveLightView>() as u64;
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

        let storage_entry =
            |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
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

        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster_clear_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/cluster_clear.wgsl")).into(),
            ),
        });
        let clear_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster_clear_pipeline_layout"),
            bind_group_layouts: &[&clear_bgl],
            push_constant_ranges: &[],
        });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster_clear_pipeline"),
            layout: Some(&clear_layout),
            module: &clear_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

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
                wgpu::BindGroupEntry { binding: 0, resource: cluster_grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: light_index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: global_offset_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grid_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: active_lights_buf.as_entire_binding() },
            ],
        });
        let build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster_build_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/cluster_build.wgsl")).into(),
            ),
        });
        let build_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster_build_pipeline_layout"),
            bind_group_layouts: &[&build_bgl],
            push_constant_ranges: &[],
        });
        let build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster_build_pipeline"),
            layout: Some(&build_layout),
            module: &build_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            grid_uniform_buf,
            cluster_grid_buf,
            light_index_buf,
            active_lights_buf,
            global_offset_buf,
            clear_bind_group,
            clear_pipeline,
            build_bind_group,
            build_pipeline,
            clear_params_buf,
        }
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
    pub fn dispatch_frame(&self, encoder: &mut wgpu::CommandEncoder, active_light_count: u32) {
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
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cluster_build_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, &self.build_bind_group, &[]);
            // One workgroup per cluster cell.
            pass.dispatch_workgroups(CLUSTER_COUNT, 1, 1);
        }
    }
}
