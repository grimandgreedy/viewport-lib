//! GPU wave-displacement plugin built on `viewport_lib::runtime::GpuPlugin`.
//!
//! Owns a compute pipeline and an output storage buffer of per-vertex
//! positions. Each frame, `pre_prepare` writes the latest time uniform and
//! dispatches one workgroup per 64 vertices to produce displaced positions.
//! The output buffer is meant to be passed to
//! `ViewportGpuResources::set_position_override_buffer` once at setup; the
//! standard mesh pipeline then reads it every frame with no rebind needed.
//!
//! This is example code shared across showcases. Treat it as a reference
//! `GpuPlugin` implementation rather than production cloth/water code.

use bytemuck::{Pod, Zeroable};
use viewport_lib::runtime::{GpuFrameContext, GpuPlugin, gpu_phase};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaveUniforms {
    time: f32,
    amplitude: f32,
    frequency: f32,
    vertex_count: u32,
}

/// GPU compute plugin that animates per-vertex Z displacement.
///
/// The plugin uploads the mesh's rest-pose positions to an immutable storage
/// buffer at construction and writes deformed positions into an output buffer
/// each frame. The output buffer handle is exposed via
/// [`WavePlugin::output_buffer`] for the consumer to clone and hand to
/// `set_position_override_buffer`.
pub struct WavePlugin {
    vertex_count: u32,
    amplitude: f32,
    frequency: f32,
    elapsed: f32,
    /// Per-frame uniform.
    uniform_buf: wgpu::Buffer,
    /// Rest-pose positions, written once at construction.
    rest_buf: wgpu::Buffer,
    /// Output positions consumed by the renderer via `set_position_override_buffer`.
    out_buf: wgpu::Buffer,
    /// Output analytic normals consumed by the renderer via `set_normal_override_buffer`.
    out_normals_buf: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl WavePlugin {
    /// Build the plugin and upload `rest_positions` to the rest-pose buffer.
    ///
    /// `rest_positions` is the flat `[x, y, z, x, y, z, ...]` layout that the
    /// override buffer protocol expects (3 `f32` per vertex). Length must be
    /// `3 * vertex_count`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rest_positions: &[f32],
        amplitude: f32,
        frequency: f32,
    ) -> Self {
        assert!(
            rest_positions.len() % 3 == 0,
            "rest_positions must be a flat array of vec3 components"
        );
        let vertex_count = (rest_positions.len() / 3) as u32;

        let rest_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wave_rest_positions"),
            contents: bytemuck::cast_slice(rest_positions),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave_out_positions"),
            size: (rest_positions.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed the output with the rest positions so the first frame (which
        // runs before `pre_prepare`) renders the undeformed mesh.
        queue.write_buffer(&out_buf, 0, bytemuck::cast_slice(rest_positions));

        let out_normals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave_out_normals"),
            size: (rest_positions.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed flat upward normals so the first frame is shaded correctly.
        let mut seed_normals = vec![0.0_f32; rest_positions.len()];
        for i in (0..seed_normals.len()).step_by(3) {
            seed_normals[i + 2] = 1.0;
        }
        queue.write_buffer(&out_normals_buf, 0, bytemuck::cast_slice(&seed_normals));

        let uniforms = WaveUniforms {
            time: 0.0,
            amplitude,
            frequency,
            vertex_count,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wave_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wave_bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("wave_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wave.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wave_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wave_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wave_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rest_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_normals_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            vertex_count,
            amplitude,
            frequency,
            elapsed: 0.0,
            uniform_buf,
            rest_buf,
            out_buf,
            out_normals_buf,
            pipeline,
            bind_group,
        }
    }

    /// Clone-able handle to the output position buffer. Pass to
    /// `ViewportGpuResources::set_position_override_buffer` once at setup; the
    /// renderer re-reads it every frame without further rebinds.
    pub fn output_buffer(&self) -> wgpu::Buffer {
        self.out_buf.clone()
    }

    /// Clone-able handle to the analytic-normal output buffer. Pass to
    /// `ViewportGpuResources::set_normal_override_buffer` to get correctly
    /// shaded illumination on the displaced surface.
    pub fn normal_buffer(&self) -> wgpu::Buffer {
        self.out_normals_buf.clone()
    }

    /// Live-tunable amplitude. Wired to a UI slider in the showcase.
    pub fn set_amplitude(&mut self, a: f32) {
        self.amplitude = a;
    }

    /// Live-tunable frequency.
    pub fn set_frequency(&mut self, f: f32) {
        self.frequency = f;
    }
}

impl GpuPlugin for WavePlugin {
    fn priority(&self) -> i32 {
        gpu_phase::PRE_PREPARE
    }

    fn pre_prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        self.elapsed += ctx.dt;

        let uniforms = WaveUniforms {
            time: self.elapsed,
            amplitude: self.amplitude,
            frequency: self.frequency,
            vertex_count: self.vertex_count,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wave_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wave_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (self.vertex_count + 63) / 64;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        // Silence dead-code on `rest_buf`: we keep the handle alive because the
        // bind group references it. The field is otherwise read-only state.
        let _ = &self.rest_buf;
        vec![encoder.finish()]
    }
}
