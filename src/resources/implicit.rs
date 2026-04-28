//! GPU implicit surface types and pipeline for Phase 16.
//!
//! Public API: [`GpuImplicitItem`], [`ImplicitPrimitive`], [`ImplicitBlendMode`],
//! [`GpuImplicitOptions`].

use wgpu::util::DeviceExt as _;
use crate::resources::ViewportGpuResources;

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

/// Primitive descriptor for the GPU implicit SDF.
///
/// The shader evaluates each primitive independently and combines them
/// according to the item's [`ImplicitBlendMode`].
///
/// # Primitive kinds and `params` layout
///
/// | `kind` | Primitive | `params[0..4]`   | `params[4..8]`    |
/// |--------|-----------|-----------------|-------------------|
/// | 1      | Sphere    | cx,cy,cz,radius | unused            |
/// | 2      | Box       | cx,cy,cz,_      | hx,hy,hz,_ (half-extents) |
/// | 3      | Plane     | nx,ny,nz,d      | unused (normal + offset)  |
/// | 4      | Capsule   | ax,ay,az,radius | bx,by,bz,_ (endpoints)   |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImplicitPrimitive {
    /// Primitive type discriminant (1=sphere, 2=box, 3=plane, 4=capsule).
    pub kind: u32,
    /// Smooth-min blend radius used when the item's blend mode is `SmoothUnion`.
    /// Zero produces a hard union.
    pub blend: f32,
    #[doc(hidden)]
    pub _pad: [f32; 2],
    /// Kind-specific parameters, first four floats.
    pub params: [f32; 8],
    /// Linear RGBA colour for this primitive.
    /// Colours are blended by proximity weight at the hit point.
    pub color: [f32; 4],
}

/// How multiple primitives are combined into a single SDF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImplicitBlendMode {
    /// Hard min() union — sharp junctions between primitives.
    #[default]
    Union,
    /// Smooth-min union — primitives fuse organically; uses per-primitive `blend` radius.
    SmoothUnion,
    /// Max() intersection — only the region inside all primitives is visible.
    Intersection,
}

/// March configuration for a [`GpuImplicitItem`].
#[derive(Clone, Copy, Debug)]
pub struct GpuImplicitOptions {
    /// Maximum ray-march steps before the ray is considered a miss. Default: 128.
    pub max_steps: u32,
    /// Step-scale applied to the SDF distance each iteration (< 1 improves thin-feature quality).
    /// Default: 0.85.
    pub step_scale: f32,
    /// Distance threshold for a ray-surface hit. Default: 5e-4.
    pub hit_threshold: f32,
    /// Maximum ray length before miss. Default: 40.0.
    pub max_distance: f32,
}

impl Default for GpuImplicitOptions {
    fn default() -> Self {
        Self {
            max_steps: 128,
            step_scale: 0.85,
            hit_threshold: 5e-4,
            max_distance: 40.0,
        }
    }
}

/// One GPU implicit surface draw item submitted via [`SceneFrame::gpu_implicit`].
///
/// Up to 16 [`ImplicitPrimitive`] entries are supported per item.
///
/// # Example
/// ```no_run
/// use viewport_lib::{GpuImplicitItem, GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive};
///
/// let mut prim = ImplicitPrimitive::zeroed();
/// prim.kind   = 1;  // sphere
/// prim.blend  = 0.9;
/// prim.params = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];  // center=origin, radius=1
/// prim.color  = [1.0, 0.5, 0.2, 1.0];
///
/// let item = GpuImplicitItem {
///     primitives:    vec![prim],
///     blend_mode:    ImplicitBlendMode::SmoothUnion,
///     march_options: GpuImplicitOptions::default(),
/// };
/// ```
pub struct GpuImplicitItem {
    /// Primitive descriptors (max 16 entries; excess entries are ignored).
    pub primitives: Vec<ImplicitPrimitive>,
    /// How the primitives are combined.
    pub blend_mode: ImplicitBlendMode,
    /// Ray-march quality settings.
    pub march_options: GpuImplicitOptions,
}

impl ImplicitPrimitive {
    /// Return a zeroed primitive with all fields set to zero.
    pub fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

// ---------------------------------------------------------------------------
// GPU-internal types (not exported from the crate root)
// ---------------------------------------------------------------------------

/// Flat uniform buffer layout matching the WGSL `ImplicitUniform` struct.
///
/// Total size: 32 header bytes + 16 * 64 primitive bytes = 1056 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ImplicitUniformRaw {
    pub num_primitives: u32,
    pub blend_mode: u32,
    pub max_steps: u32,
    pub _pad0: u32,
    pub step_scale: f32,
    pub hit_threshold: f32,
    pub max_distance: f32,
    pub _pad1: f32,
    pub primitives: [ImplicitPrimitive; 16],
}

/// Per-draw GPU data for one [`GpuImplicitItem`].
pub(crate) struct ImplicitGpuItem {
    pub uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Pipeline init and upload (impl ViewportGpuResources)
// ---------------------------------------------------------------------------

impl ViewportGpuResources {
    /// Lazily create the GPU implicit surface render pipeline.
    ///
    /// No-op if already created.  Called from `prepare()` when any
    /// `GpuImplicitItem` is submitted via `SceneFrame::gpu_implicit`.
    pub(crate) fn ensure_implicit_pipeline(&mut self, device: &wgpu::Device) {
        if self.implicit_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("implicit_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/implicit.wgsl").into()),
        });

        // Group 1: single uniform buffer containing ImplicitUniformRaw.
        let implicit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("implicit_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 0 reuses camera_bind_group_layout (provides CameraUniform + LightsUniform).
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("implicit_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &implicit_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("implicit_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Write depth so subsequent screen-image depth-composite items test against it.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        self.implicit_bgl = Some(implicit_bgl);
        self.implicit_pipeline = Some(pipeline);
    }

    /// Upload one [`GpuImplicitItem`] to GPU, returning the per-draw GPU data.
    ///
    /// Panics if called before `ensure_implicit_pipeline`.
    pub(crate) fn upload_implicit_item(
        &self,
        device: &wgpu::Device,
        item: &GpuImplicitItem,
    ) -> ImplicitGpuItem {
        // Build the flat uniform struct.
        let blend_mode_u32 = match item.blend_mode {
            ImplicitBlendMode::Union => 0u32,
            ImplicitBlendMode::SmoothUnion => 1,
            ImplicitBlendMode::Intersection => 2,
        };

        let mut raw = ImplicitUniformRaw {
            num_primitives: item.primitives.len().min(16) as u32,
            blend_mode: blend_mode_u32,
            max_steps: item.march_options.max_steps,
            _pad0: 0,
            step_scale: item.march_options.step_scale,
            hit_threshold: item.march_options.hit_threshold,
            max_distance: item.march_options.max_distance,
            _pad1: 0.0,
            primitives: [ImplicitPrimitive::zeroed(); 16],
        };

        for (i, prim) in item.primitives.iter().take(16).enumerate() {
            raw.primitives[i] = *prim;
        }

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("implicit_uniform_buf"),
            contents: bytemuck::bytes_of(&raw),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bgl = self
            .implicit_bgl
            .as_ref()
            .expect("ensure_implicit_pipeline not called");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("implicit_bind_group"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        ImplicitGpuItem { uniform_buf, bind_group }
    }
}
