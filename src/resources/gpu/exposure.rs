//! Auto-exposure GPU resources.
//!
//! Mirrors the clustered-shading resource pattern: shared compute pipelines and
//! a bind group layout on `DeviceResources`, with the per-viewport buffers
//! (histogram, adaptation state, params) living on `ViewportHdrState`. The three
//! entry points of `exposure.wgsl` (clear -> build -> resolve) run between the
//! HDR pass and the tone map in the same submission, so a single dirty render is
//! correctly exposed on its own frame with no CPU readback.

use crate::gpu::util::DeviceExt;

/// Number of log-luminance histogram bins. Must match `HISTOGRAM_BINS` in
/// `exposure.wgsl`.
pub const HISTOGRAM_BINS: u32 = 256;

/// Lower bound of the metered log2-luminance range (bin 0). Chosen to cover deep
/// shadow through a bright sky across the 256 bins.
pub const LOG_LUM_MIN: f32 = -10.0;
/// Upper bound of the metered log2-luminance range (last bin).
pub const LOG_LUM_MAX: f32 = 12.0;

/// Per-frame metering + adaptation parameters. Matches `ExposureParams` in
/// `exposure.wgsl` (64 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExposureParams {
    /// Lower bound of the metered log2-luminance range (bin 0).
    pub min_log_lum: f32,
    /// Reciprocal of `log_lum_range`, for mapping luminance to a bin.
    pub inv_log_lum_range: f32,
    /// Width of the metered log2-luminance range (`LOG_LUM_MAX - LOG_LUM_MIN`).
    pub log_lum_range: f32,
    /// Reflected-light meter calibration constant `K` (`12.5`).
    pub k_factor: f32,
    /// Lower clamp on the adapted EV100.
    pub min_ev: f32,
    /// Upper clamp on the adapted EV100.
    pub max_ev: f32,
    /// Exposure compensation in stops (positive brightens).
    pub compensation: f32,
    /// Interim neutral boost folded into the EV->multiplier form (temporary).
    pub exposure_boost: f32,
    /// Adaptation rate (per second) when the scene brightens.
    pub speed_up: f32,
    /// Adaptation rate (per second) when the scene darkens.
    pub speed_down: f32,
    /// Frame time in seconds; `<= 0` snaps to the target this frame.
    pub dt: f32,
    /// Fraction of darkest pixels discarded before averaging, in `[0, 1)`.
    pub low_percent: f32,
    /// Cumulative fraction above which the brightest pixels are discarded.
    pub high_percent: f32,
    /// HDR target width in texels.
    pub tex_width: f32,
    /// HDR target height in texels.
    pub tex_height: f32,
    /// Center-weighting of the meter, in `[0, 1]` (0 = uniform full frame).
    pub center_weight: f32,
}

const _: () = assert!(std::mem::size_of::<ExposureParams>() == 64);

/// Persistent adaptation state. Matches `ExposureState` in `exposure.wgsl` and
/// the tone-map exposure buffer (16 bytes). `exposure` is the linear multiplier
/// the tone map reads; the other fields carry adaptation state and the
/// "still adapting" signal for the optional readback.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExposureState {
    /// Linear exposure multiplier applied before tone mapping.
    pub exposure: f32,
    /// Current adapted EV100 (persists across frames for smoothing).
    pub current_ev: f32,
    /// Last metered target EV100.
    pub target_ev: f32,
    /// `1.0` while `current_ev` is still easing toward `target_ev`, else `0.0`.
    pub adapting: f32,
}

const _: () = assert!(std::mem::size_of::<ExposureState>() == 16);

/// Shared auto-exposure pipelines and bind group layout owned by
/// `DeviceResources`.
pub struct ExposureResources {
    /// Bind group layout for all three compute passes (hdr texture, params
    /// uniform, histogram storage, exposure-state storage).
    pub bgl: crate::gpu::BindGroupLayout,
    clear_pipeline: crate::gpu::ComputePipeline,
    build_pipeline: crate::gpu::ComputePipeline,
    resolve_pipeline: crate::gpu::ComputePipeline,
}

impl ExposureResources {
    /// Build the shared bind group layout and the clear/build/resolve pipelines.
    pub fn new(device: &crate::gpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("exposure_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
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
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "exposure_shader",
            crate::resources::builders::wgsl_source!("exposure"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "exposure_pipeline_layout",
            &[&bgl],
        );
        let clear_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "exposure_clear_pipeline",
            &layout,
            &shader,
            "clear_main",
        );
        let build_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "exposure_build_pipeline",
            &layout,
            &shader,
            "build_main",
        );
        let resolve_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "exposure_resolve_pipeline",
            &layout,
            &shader,
            "resolve_main",
        );

        Self {
            bgl,
            clear_pipeline,
            build_pipeline,
            resolve_pipeline,
        }
    }

    /// Allocate the per-viewport histogram, exposure-state, and params buffers.
    /// Returns them in the order stored on `ViewportHdrState`.
    pub fn create_viewport_buffers(
        device: &crate::gpu::Device,
    ) -> (crate::gpu::Buffer, crate::gpu::Buffer, crate::gpu::Buffer) {
        let histogram_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("exposure_histogram_buf"),
            size: (HISTOGRAM_BINS as u64) * 4,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed `current_ev` non-finite so the first automatic frame snaps, and
        // `exposure` to 1.0 so a manual/physical frame before the first CPU
        // write is still sane.
        let state_seed = ExposureState {
            exposure: 1.0,
            current_ev: f32::NAN,
            target_ev: 0.0,
            adapting: 0.0,
        };
        let state_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("exposure_state_buf"),
            contents: bytemuck::cast_slice(&[state_seed]),
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::COPY_DST
                | crate::gpu::BufferUsages::COPY_SRC,
        });
        let params_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("exposure_params_buf"),
            size: std::mem::size_of::<ExposureParams>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (histogram_buf, state_buf, params_buf)
    }

    /// Build the compute bind group for one viewport. `hdr_view` is the scene
    /// HDR target (sharp radiance, sampled read-only).
    pub fn create_bind_group(
        &self,
        device: &crate::gpu::Device,
        hdr_view: &crate::gpu::TextureView,
        params_buf: &crate::gpu::Buffer,
        histogram_buf: &crate::gpu::Buffer,
        state_buf: &crate::gpu::Buffer,
    ) -> crate::gpu::BindGroup {
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("exposure_bind_group"),
            layout: &self.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(hdr_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: histogram_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: state_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Write the metering parameters for this frame.
    pub fn write_params(
        &self,
        queue: &crate::gpu::Queue,
        buf: &crate::gpu::Buffer,
        params: &ExposureParams,
    ) {
        queue.write_buffer(buf, 0, bytemuck::cast_slice(&[*params]));
    }

    /// Encode the clear -> build -> resolve dispatches. `width`/`height` are the
    /// HDR target dimensions.
    pub fn dispatch(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        bind_group: &crate::gpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("exposure_clear_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(HISTOGRAM_BINS.div_ceil(256), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("exposure_build_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("exposure_resolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}
