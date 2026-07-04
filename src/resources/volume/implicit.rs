//! GPU implicit surface types and pipeline.
//!
//! Public API: [`GpuImplicitItem`], [`ImplicitPrimitive`], [`ImplicitBlendMode`],
//! [`GpuImplicitOptions`].

use crate::renderer::GpuImplicitItem;
use crate::resources::DeviceResources;
use wgpu::util::DeviceExt as _;

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
    pub colour: [f32; 4],
}

/// How multiple primitives are combined into a single SDF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImplicitBlendMode {
    /// Hard min() union (sharp junctions between primitives)
    #[default]
    Union,
    /// Smooth-min union (primitives fuse organically; uses per-primitive `blend` radius)
    SmoothUnion,
    /// Max() intersection (only the region inside all primitives is visible)
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
    pub unlit: u32,
    pub step_scale: f32,
    pub hit_threshold: f32,
    pub max_distance: f32,
    pub opacity: f32,
    pub primitives: [ImplicitPrimitive; 16],
}

/// Per-draw GPU data for one [`GpuImplicitItem`].
pub(crate) struct ImplicitGpuItem {
    pub _uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Pipeline init and upload (impl DeviceResources)
// ---------------------------------------------------------------------------

impl DeviceResources {
    /// Lazily create the GPU implicit surface render pipeline.
    ///
    /// No-op if already created.  Called from `prepare()` when any
    /// `GpuImplicitItem` is submitted via `SceneFrame::gpu_implicit`.
    pub(crate) fn ensure_implicit_pipeline(&mut self, device: &wgpu::Device) {
        if self.implicit.pipeline.is_some() {
            return;
        }

        let shader = crate::resources::builders::wgsl_module(
            device,
            "implicit_shader",
            crate::resources::builders::wgsl_source!("implicit"),
        );

        // Group 1: single uniform buffer containing ImplicitUniformRaw.
        let implicit_bgl = crate::resources::builders::uniform_bgl(
            device,
            "implicit_bgl",
            wgpu::ShaderStages::FRAGMENT,
        );

        // Group 0 reuses camera_bind_group_layout (provides CameraUniform + LightsUniform).
        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "implicit_pipeline_layout",
            &self.camera_bind_group_layout,
            &implicit_bgl,
        );

        self.implicit.bgl = Some(implicit_bgl);
        // depth_write is on so subsequent screen-image depth-composite items test against it.
        self.implicit.pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "implicit_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[],
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                sample_count: 1,
                ldr_format: self.target_format,
            },
        ));
    }

    /// Lazily create the implicit surface outline mask pipeline.
    ///
    /// Reuses the same bind group layouts as `ensure_implicit_pipeline`. Must be
    /// called after `ensure_implicit_pipeline` so that `implicit_bgl` is set.
    /// No-op if already created.
    pub(crate) fn ensure_implicit_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.implicit.outline_mask_pipeline.is_some() {
            return;
        }

        let implicit_bgl = self.implicit.bgl.as_ref().expect(
            "ensure_implicit_pipeline must be called before ensure_implicit_outline_mask_pipeline",
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "implicit_outline_mask_shader",
            crate::resources::builders::wgsl_source!("implicit_outline_mask"),
        );

        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "implicit_outline_mask_pipeline_layout",
            &self.camera_bind_group_layout,
            implicit_bgl,
        );

        self.implicit.outline_mask_pipeline =
            Some(crate::resources::builders::build_outline_mask_pipeline(
                device,
                "implicit_outline_mask_pipeline",
                &layout,
                &shader,
                wgpu::TextureFormat::R8Unorm,
                &[],
                None,
                true,
                wgpu::CompareFunction::Less,
            ));
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
            unlit: item.settings.unlit as u32,
            step_scale: item.march_options.step_scale,
            hit_threshold: item.march_options.hit_threshold,
            max_distance: item.march_options.max_distance,
            opacity: item.settings.opacity,
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
            .implicit
            .bgl
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

        ImplicitGpuItem {
            _uniform_buf: uniform_buf,
            bind_group,
        }
    }
}
