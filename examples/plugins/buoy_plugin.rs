//! GPU buoy plugin: reads the wave plugin's position buffer (GPU-side) and
//! drives a set of small spheres that ride on the wave surface.
//!
//! Chained-compute demo:
//!   - Wave plugin produces a GPU position buffer for the surface mesh.
//!   - This plugin takes that buffer at construction (shared `wgpu::Buffer`
//!     handle, no copy) and runs a second compute pass that samples the wave
//!     height at each buoy's (x, y) anchor.
//!   - The output is bound via `set_position_override_buffer` to a separate
//!     "buoys" mesh, so the standard mesh pipeline renders the spheres at
//!     positions that depend on the wave's GPU state, with zero CPU work
//!     between the two stages.
//!
//! Priority is `gpu_phase::PRE_PREPARE + 100`, which guarantees this runs in
//! the same `runtime.pre_prepare(...)` pass *after* the wave plugin (lower
//! priority) so the buoy compute sees fresh wave data.

use bytemuck::{Pod, Zeroable};
use viewport_lib::runtime::{GpuFrameContext, GpuPlugin, gpu_phase};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    grid_dim: u32,
    buoy_count: u32,
    verts_per_buoy: u32,
    _pad0: u32,
    world_half_extent: f32,
    waterline_offset: f32,
    _pad1: [f32; 2],
}

/// Reads the wave's GPU position buffer; writes per-vertex positions for a
/// baked-instances "buoys" mesh whose layout is `buoy_count` copies of one
/// small sphere.
pub struct BuoyPlugin {
    total_vertex_count: u32,
    uniforms_dirty: bool,
    uniforms: Uniforms,
    uniform_buf: wgpu::Buffer,
    /// Output position buffer consumed by the renderer via `set_position_override_buffer`.
    out_buf: wgpu::Buffer,
    /// Persistent handles kept alive for the bind group.
    _wave_buf: wgpu::Buffer,
    _anchors_buf: wgpu::Buffer,
    _sphere_local_buf: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl BuoyPlugin {
    /// Build the plugin.
    ///
    /// - `wave_output_buf`: the wave plugin's per-vertex position buffer (a
    ///   shared handle; both plugins read from it without copying).
    /// - `grid_dim`: the wave mesh's grid dimension (cols == rows).
    /// - `world_half_extent`: half of the wave plane's side length (e.g. 4.0
    ///   for an 8x8 plane).
    /// - `buoy_anchors`: flat `[x, y, x, y, ...]` world positions for each buoy.
    /// - `sphere_local_positions`: per-vertex local positions of one sphere
    ///   mesh, flat `[x, y, z, ...]`. Replicated implicitly for every buoy.
    /// - `waterline_offset`: vertical lift applied to every buoy so its sphere
    ///   sits above (not embedded in) the surface.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        wave_output_buf: wgpu::Buffer,
        grid_dim: u32,
        world_half_extent: f32,
        buoy_anchors: &[f32],
        sphere_local_positions: &[f32],
        waterline_offset: f32,
    ) -> Self {
        assert!(
            buoy_anchors.len() % 2 == 0,
            "buoy_anchors must hold 2 floats per buoy (x, y)"
        );
        assert!(
            sphere_local_positions.len() % 3 == 0,
            "sphere_local_positions must hold 3 floats per vertex"
        );
        let buoy_count = (buoy_anchors.len() / 2) as u32;
        let verts_per_buoy = (sphere_local_positions.len() / 3) as u32;
        let total_vertex_count = buoy_count * verts_per_buoy;

        let uniforms = Uniforms {
            grid_dim,
            buoy_count,
            verts_per_buoy,
            _pad0: 0,
            world_half_extent,
            waterline_offset,
            _pad1: [0.0; 2],
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buoy_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let anchors_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buoy_anchors"),
            contents: bytemuck::cast_slice(buoy_anchors),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let sphere_local_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buoy_sphere_local"),
            contents: bytemuck::cast_slice(sphere_local_positions),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buoy_out_positions"),
            size: (total_vertex_count as u64) * 3 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed with a guess so the first frame before `pre_prepare` runs shows
        // something rather than 0. We place each buoy at its anchor (x, y, 0)
        // with the local sphere offset added.
        let mut seed = vec![0.0_f32; (total_vertex_count as usize) * 3];
        for b in 0..buoy_count as usize {
            let ax = buoy_anchors[b * 2];
            let ay = buoy_anchors[b * 2 + 1];
            for v in 0..verts_per_buoy as usize {
                let lx = sphere_local_positions[v * 3];
                let ly = sphere_local_positions[v * 3 + 1];
                let lz = sphere_local_positions[v * 3 + 2];
                let gi = (b * verts_per_buoy as usize + v) * 3;
                seed[gi] = ax + lx;
                seed[gi + 1] = ay + ly;
                seed[gi + 2] = lz + waterline_offset;
            }
        }
        queue.write_buffer(&out_buf, 0, bytemuck::cast_slice(&seed));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("buoy_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("buoy_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("buoy.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("buoy_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("buoy_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("buoy_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wave_output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: anchors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sphere_local_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            total_vertex_count,
            uniforms_dirty: false,
            uniforms,
            uniform_buf,
            out_buf,
            _wave_buf: wave_output_buf,
            _anchors_buf: anchors_buf,
            _sphere_local_buf: sphere_local_buf,
            pipeline,
            bind_group,
        }
    }

    /// Clone-able handle to the buoy mesh's per-vertex position output. Pass
    /// to `ViewportGpuResources::set_position_override_buffer` once at setup.
    pub fn output_buffer(&self) -> wgpu::Buffer {
        self.out_buf.clone()
    }

    /// Adjust the lift above the wave surface at runtime.
    pub fn set_waterline_offset(&mut self, v: f32) {
        if (self.uniforms.waterline_offset - v).abs() > f32::EPSILON {
            self.uniforms.waterline_offset = v;
            self.uniforms_dirty = true;
        }
    }
}

impl GpuPlugin for BuoyPlugin {
    fn priority(&self) -> i32 {
        // Run AFTER the wave plugin so we sample fresh wave heights.
        gpu_phase::PRE_PREPARE + 100
    }

    fn pre_prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        if self.uniforms_dirty {
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
            self.uniforms_dirty = false;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("buoy_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("buoy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (self.total_vertex_count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        vec![encoder.finish()]
    }
}
