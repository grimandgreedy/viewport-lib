//! Offscreen path tracer for reference images.
//!
//! A compute-only path tracer that renders to an offscreen buffer. It builds a
//! world-space triangle BVH, traces it with a megakernel
//! (`src/shaders/raytrace.wgsl`), and accumulates samples into an HDR buffer
//! that is read back to the CPU. It uses the same direct BRDF as the rasteriser
//! (`helpers/brdf.wgsl`), so both produce the same shading.
//!
//! The portable compute traversal runs on every backend and does not touch the
//! main render path: build an [`RtScene`], call [`trace`], get an [`RtImage`]
//! back. A dielectric transmission lobe and an optional edge-aware a-trous
//! denoiser ([`RtSettings::denoise`]) are built in. On ray miss the tracer reads
//! either an equirect HDR environment ([`RtScene::set_environment`], for
//! image-based lighting that matches the rasteriser's IBL) or a hemisphere sky.
//!
//! The same integrator also bakes lightmaps: [`bake_lightmap`] / [`Tracer::bake`]
//! shoot the GI hemisphere from per-texel surface points (a [`TexelSurfaces`]
//! G-buffer) instead of from a camera, and store incident irradiance into an
//! atlas. The traversal, shading, and environment code are shared; only the
//! primary-ray source differs.
//!
//! [`pick_backend`] reports whether a hardware ray-query traversal is available
//! (Vulkan/DX12 with the `raytrace-hardware` feature). The ray-query traversal
//! kernel itself is not wired up yet, so the tracer always runs the compute
//! traversal; the selection plumbing is in place for that kernel to slot in.
//!
//! Not handled yet: a two-level BVH, importance sampling of the environment (it
//! is sampled on miss but not used for next-event estimation, so pure-IBL scenes
//! converge more slowly), and the hardware ray-query kernel.

mod bvh;

use crate::gpu::util::DeviceExt;
use glam::{Mat4, Vec3};

/// Which traversal backend the tracer would use for a device.
///
/// The compute traversal is portable and always available. The hardware backend
/// needs the `raytrace-hardware` feature and a device that advertises
/// [`RAY_QUERY_FEATURE`](crate::gpu::RAY_QUERY_FEATURE) (Vulkan/DX12); it is
/// unsupported on Metal and the web.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtBackend {
    /// Portable compute traversal over the crate's own BVH.
    Software,
    /// Hardware ray-query traversal (Vulkan/DX12 only).
    Hardware,
}

/// Report the traversal backend the tracer would select for `device`.
///
/// Returns [`RtBackend::Hardware`] only when the `raytrace-hardware` feature is
/// enabled and `device` advertises [`RAY_QUERY_FEATURE`](crate::gpu::RAY_QUERY_FEATURE),
/// otherwise [`RtBackend::Software`]. Note the hardware traversal kernel is not
/// implemented yet: [`trace`] runs the compute traversal regardless. This
/// reports capability so the selection is testable ahead of that kernel.
pub fn pick_backend(device: &crate::gpu::Device) -> RtBackend {
    #[cfg(feature = "raytrace-hardware")]
    if device.features().contains(crate::gpu::RAY_QUERY_FEATURE) {
        return RtBackend::Hardware;
    }
    let _ = device;
    RtBackend::Software
}

/// Surface material for the tracer. A subset of the rasteriser's `Material`:
/// metallic-roughness and emissive.
#[derive(Clone, Copy, Debug)]
pub struct RtMaterial {
    /// Linear base colour (albedo for dielectrics, specular tint for metals,
    /// transmission tint for glass).
    pub base_colour: [f32; 3],
    /// 0 = dielectric, 1 = metal.
    pub metallic: f32,
    /// Perceptual roughness in [0, 1].
    pub roughness: f32,
    /// Linear emitted radiance.
    pub emissive: [f32; 3],
    /// Dielectric transmission weight in [0, 1]. 0 is opaque; 1 is clear glass.
    /// Metals (`metallic > 0`) do not transmit; keep it 0 for them.
    pub transmission: f32,
    /// Index of refraction for the transmission lobe (1.0 bends nothing, ~1.5
    /// for glass). Only read when `transmission > 0`.
    pub ior: f32,
}

impl Default for RtMaterial {
    fn default() -> Self {
        Self {
            base_colour: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0, 0.0, 0.0],
            transmission: 0.0,
            ior: 1.5,
        }
    }
}

/// An analytic light. Directional lights are delta lights sampled via next-event
/// estimation; the tracer does not importance-sample them via the BSDF.
#[derive(Clone, Copy, Debug)]
pub enum RtLight {
    /// `direction` points from the surface toward the light (i.e. the negated
    /// travel direction of the light). `colour` is radiance.
    Directional {
        /// Unit direction from the surface toward the light.
        direction: [f32; 3],
        /// Linear radiance.
        colour: [f32; 3],
    },
    /// `range` <= 0 disables the windowed falloff (pure inverse-square).
    Point {
        /// World-space light position.
        position: [f32; 3],
        /// Linear radiance (before distance attenuation).
        colour: [f32; 3],
        /// Falloff range in world units; <= 0 disables the windowed falloff.
        range: f32,
    },
}

/// Camera for the trace. `inv_view_proj` is the inverse of the combined
/// view-projection used by the rasteriser, so the tracer frames the scene
/// identically.
#[derive(Clone, Copy, Debug)]
pub struct RtCamera {
    /// Inverse of the combined view-projection matrix.
    pub inv_view_proj: Mat4,
    /// World-space camera position (ray origin).
    pub position: Vec3,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

/// Per-texel surfaces for a lightmap bake, row-major `width * height` with the
/// atlas origin at the top-left : the world position and normal behind every
/// atlas texel, as produced by the texel G-buffer pass (`bake` feature). Feeds
/// [`Tracer::bake`] / [`bake_lightmap`], which shoot the GI hemisphere from
/// these points instead of from a camera.
pub struct TexelSurfaces<'a> {
    /// Atlas width in texels.
    pub width: u32,
    /// Atlas height in texels.
    pub height: u32,
    /// World position in `xyz`; `w > 0` marks a covered texel, `w <= 0` an empty
    /// one that is left black. Length must be `width * height`.
    pub world_pos: &'a [[f32; 4]],
    /// World normal in `xyz` (`w` unused). Length must be `width * height`.
    pub world_normal: &'a [[f32; 4]],
}

/// Trace settings.
#[derive(Clone, Copy, Debug)]
pub struct RtSettings {
    /// Total samples per pixel to accumulate.
    pub samples: u32,
    /// Maximum path length (bounces).
    pub max_bounces: u32,
    /// Run the edge-aware a-trous denoiser over the accumulated image before
    /// readback. Cheap relative to tracing and useful for low-sample previews;
    /// leave off for a raw converged reference.
    pub denoise: bool,
}

impl Default for RtSettings {
    fn default() -> Self {
        Self {
            samples: 256,
            max_bounces: 8,
            denoise: false,
        }
    }
}

/// The traced image: linear HDR, row-major RGBA f32, `width * height * 4` long.
pub struct RtImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Linear HDR pixels, row-major RGBA f32 (`width * height * 4` elements).
    pub rgba: Vec<f32>,
}

/// An equirectangular HDR environment used on ray miss for image-based lighting.
/// `pixels` is linear RGBA f32, row-major, `width * height * 4` long.
struct EnvMap {
    pixels: Vec<f32>,
    width: u32,
    height: u32,
}

/// A scene to trace: triangles with per-vertex normals and a material index,
/// analytic lights, and either an equirect HDR environment or a hemisphere sky
/// used on ray miss.
#[derive(Default)]
pub struct RtScene {
    tris: Vec<[Vec3; 3]>,
    normals: Vec<[Vec3; 3]>,
    tri_mat: Vec<u32>,
    materials: Vec<RtMaterial>,
    lights: Vec<RtLight>,
    sky_top: [f32; 3],
    sky_bottom: [f32; 3],
    env: Option<EnvMap>,
}

impl RtScene {
    /// Empty scene with a mild default sky (used on ray miss).
    pub fn new() -> Self {
        Self {
            sky_top: [0.5, 0.7, 1.0],
            sky_bottom: [0.2, 0.2, 0.2],
            ..Default::default()
        }
    }

    /// Add an indexed triangle mesh in world space. `normals`, if given, are
    /// per-vertex (one per position) and interpolated for smooth shading;
    /// otherwise flat geometric normals are used. Every triangle gets the same
    /// `material`. Returns the material index assigned.
    pub fn add_mesh(
        &mut self,
        positions: &[Vec3],
        indices: &[u32],
        normals: Option<&[Vec3]>,
        material: RtMaterial,
    ) -> u32 {
        let mat_id = self.materials.len() as u32;
        self.materials.push(material);
        for tri in indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let p = [positions[i0], positions[i1], positions[i2]];
            let n = match normals {
                Some(ns) => [ns[i0], ns[i1], ns[i2]],
                None => {
                    let ng = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                    [ng, ng, ng]
                }
            };
            self.tris.push(p);
            self.normals.push(n);
            self.tri_mat.push(mat_id);
        }
        mat_id
    }

    /// Add an analytic light.
    pub fn add_light(&mut self, light: RtLight) {
        self.lights.push(light);
    }

    /// Set an equirectangular HDR environment sampled on ray miss, giving
    /// image-based lighting that matches the rasteriser's IBL. `pixels` is linear
    /// RGBA f32, row-major, `width * height * 4` long, in the same Z-up equirect
    /// projection the renderer uses (longitude around +Z, latitude with +Z at the
    /// top). Replaces the hemisphere sky while set. A wrong-length slice is
    /// ignored.
    pub fn set_environment(&mut self, pixels: &[f32], width: u32, height: u32) {
        if (width * height * 4) as usize != pixels.len() || width == 0 || height == 0 {
            return;
        }
        self.env = Some(EnvMap {
            pixels: pixels.to_vec(),
            width,
            height,
        });
    }

    /// Set the hemisphere sky colours (linear). `top` is straight up (+Z),
    /// `bottom` straight down.
    pub fn set_sky(&mut self, top: [f32; 3], bottom: [f32; 3]) {
        self.sky_top = top;
        self.sky_bottom = bottom;
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }
}

// ----- GPU-side structs (must match raytrace.wgsl) -----

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
struct GpuTri {
    p0: [f32; 3],
    mat: u32,
    p1: [f32; 3],
    _p1: u32,
    p2: [f32; 3],
    _p2: u32,
    n0: [f32; 3],
    _n0: u32,
    n1: [f32; 3],
    _n1: u32,
    n2: [f32; 3],
    _n2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
struct GpuMaterial {
    base: [f32; 3],
    metallic: f32,
    emissive: [f32; 3],
    roughness: f32,
    transmission: f32,
    ior: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
struct GpuLight {
    data: [f32; 4],
    colour: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
struct FrameUniform {
    inv_view_proj: [f32; 16],
    cam_pos: [f32; 4],
    sky_top: [f32; 4],
    sky_bottom: [f32; 4],
    dims: [u32; 4],
    params: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
struct GpuDenoiseParams {
    dims: [u32; 2],
    step: u32,
    _pad: u32,
    sigma_n: f32,
    sigma_l: f32,
    _pad2: [f32; 2],
}

/// Trace `scene` from `camera` and return the accumulated HDR image.
///
/// Runs entirely on the compute path (portable to every backend). Blocks until
/// the accumulation is read back to the CPU. For an empty scene, returns a black
/// image of the requested size.
///
/// This builds the pipeline and uploads the scene on every call. To trace the
/// same scene repeatedly (e.g. an interactive viewer re-tracing as the camera
/// moves), build a [`Tracer`] once and call [`Tracer::trace`] instead : it keeps
/// the compiled pipeline and uploaded scene, so each frame only re-dispatches.
pub fn trace(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    scene: &RtScene,
    camera: &RtCamera,
    settings: &RtSettings,
) -> RtImage {
    Tracer::new(device, queue, scene).trace(device, queue, camera, settings)
}

/// A reusable path tracer that holds a compiled pipeline and an uploaded scene.
///
/// The reference [`trace`] function rebuilds the BVH, re-uploads the scene, and
/// recompiles the compute pipeline on every call, which dominates the cost when
/// the same scene is traced many times (an interactive preview re-traces on
/// every camera move). A `Tracer` does that setup once in [`Tracer::new`]; each
/// [`Tracer::trace`] only rewrites the per-frame uniform, re-dispatches, and
/// reads back. Size-dependent buffers are (re)allocated when the output
/// resolution changes.
pub struct Tracer {
    pipeline: crate::gpu::ComputePipeline,
    bgl: crate::gpu::BindGroupLayout,
    // Lightmap-bake variant of the kernel: the same integrator with a texel
    // ray-gen front-end (bindings 10/11) instead of the camera. Compiled from
    // the same module as `pipeline`.
    bake_pipeline: crate::gpu::ComputePipeline,
    bake_bgl: crate::gpu::BindGroupLayout,
    denoise_pipeline: crate::gpu::ComputePipeline,
    denoise_bgl: crate::gpu::BindGroupLayout,
    node_buf: crate::gpu::Buffer,
    tri_buf: crate::gpu::Buffer,
    mat_buf: crate::gpu::Buffer,
    light_buf: crate::gpu::Buffer,
    frame_buf: crate::gpu::Buffer,
    // Equirect environment (or a 1x1 fallback). `env_view` is kept valid by
    // holding `_env_texture`; `has_env` gates the shader between the environment
    // and the hemisphere sky.
    _env_texture: crate::gpu::Texture,
    env_view: crate::gpu::TextureView,
    env_sampler: crate::gpu::Sampler,
    has_env: bool,
    num_lights: u32,
    sky_top: [f32; 3],
    sky_bottom: [f32; 3],
    has_geometry: bool,
    sized: Option<TracerSized>,
    // Samples already blended into the accumulation buffer. `trace` resets it to
    // 0 each call; `accumulate` adds on top for progressive convergence.
    accumulated: u32,
}

/// Output-resolution-dependent buffers, rebuilt when the size changes.
struct TracerSized {
    width: u32,
    height: u32,
    bytes: u64,
    accum_buf: crate::gpu::Buffer,
    staging_buf: crate::gpu::Buffer,
    gbuf_albedo: crate::gpu::Buffer,
    gbuf_normal: crate::gpu::Buffer,
    bind_group: crate::gpu::BindGroup,
}

impl Tracer {
    /// Build the pipeline and upload `scene`. Cheap to keep around and re-trace.
    pub fn new(device: &crate::gpu::Device, queue: &crate::gpu::Queue, scene: &RtScene) -> Self {
        let has_geometry = !scene.tris.is_empty();

        // Build the BVH and the reordered triangle array it references. An empty
        // scene still needs valid (single-element) buffers so the bind group and
        // pipeline exist; trace() short-circuits to black before dispatching.
        let (nodes, order) = bvh::build(&scene.tris);
        let gpu_tris: Vec<GpuTri> = order
            .iter()
            .map(|&k| {
                let k = k as usize;
                let p = scene.tris[k];
                let n = scene.normals[k];
                GpuTri {
                    p0: p[0].to_array(),
                    mat: scene.tri_mat[k],
                    p1: p[1].to_array(),
                    _p1: 0,
                    p2: p[2].to_array(),
                    _p2: 0,
                    n0: n[0].to_array(),
                    _n0: 0,
                    n1: n[1].to_array(),
                    _n1: 0,
                    n2: n[2].to_array(),
                    _n2: 0,
                }
            })
            .collect();
        let gpu_tris = if gpu_tris.is_empty() {
            vec![<GpuTri as bytemuck::Zeroable>::zeroed()]
        } else {
            gpu_tris
        };

        let mut gpu_mats: Vec<GpuMaterial> = scene
            .materials
            .iter()
            .map(|m| GpuMaterial {
                base: m.base_colour,
                metallic: m.metallic,
                emissive: m.emissive,
                roughness: m.roughness,
                transmission: m.transmission.clamp(0.0, 1.0),
                ior: m.ior.max(1.0),
                _pad0: 0.0,
                _pad1: 0.0,
            })
            .collect();
        if gpu_mats.is_empty() {
            gpu_mats.push(GpuMaterial {
                base: [0.8, 0.8, 0.8],
                metallic: 0.0,
                emissive: [0.0; 3],
                roughness: 0.5,
                transmission: 0.0,
                ior: 1.5,
                _pad0: 0.0,
                _pad1: 0.0,
            });
        }

        let mut gpu_lights: Vec<GpuLight> = scene
            .lights
            .iter()
            .map(|l| match *l {
                RtLight::Directional { direction, colour } => GpuLight {
                    data: [direction[0], direction[1], direction[2], 0.0],
                    colour: [colour[0], colour[1], colour[2], 0.0],
                },
                RtLight::Point {
                    position,
                    colour,
                    range,
                } => GpuLight {
                    data: [position[0], position[1], position[2], 1.0],
                    colour: [colour[0], colour[1], colour[2], range.max(0.0)],
                },
            })
            .collect();
        let num_lights = gpu_lights.len() as u32;
        if gpu_lights.is_empty() {
            gpu_lights.push(GpuLight {
                data: [0.0; 4],
                colour: [0.0; 4],
            });
        }

        let node_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("rt_nodes"),
            contents: bytemuck::cast_slice(&nodes),
            usage: crate::gpu::BufferUsages::STORAGE,
        });
        let tri_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("rt_tris"),
            contents: bytemuck::cast_slice(&gpu_tris),
            usage: crate::gpu::BufferUsages::STORAGE,
        });
        let mat_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("rt_materials"),
            contents: bytemuck::cast_slice(&gpu_mats),
            usage: crate::gpu::BufferUsages::STORAGE,
        });
        let light_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("rt_lights"),
            contents: bytemuck::cast_slice(&gpu_lights),
            usage: crate::gpu::BufferUsages::STORAGE,
        });
        let frame_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("rt_frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Equirect environment as an Rgba16Float texture (filterable everywhere
        // without an extra device feature). Falls back to a 1x1 black texture the
        // shader ignores when `has_env` is false.
        let has_env = scene.env.is_some();
        let (env_w, env_h): (u32, u32) = match &scene.env {
            Some(e) => (e.width, e.height),
            None => (1, 1),
        };
        let env_texels: Vec<u16> = match &scene.env {
            Some(e) => e
                .pixels
                .iter()
                .map(|&c| half::f16::from_f32(c).to_bits())
                .collect(),
            None => vec![0u16; 4],
        };
        let env_texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("rt_env"),
            size: crate::gpu::Extent3d {
                width: env_w,
                height: env_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba16Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING | crate::gpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &env_texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&env_texels),
            crate::gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(env_w * 8),
                rows_per_image: Some(env_h),
            },
            crate::gpu::Extent3d {
                width: env_w,
                height: env_h,
                depth_or_array_layers: 1,
            },
        );
        let env_view = env_texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        // Wrap longitude, clamp latitude, bilinear.
        let env_sampler = device.create_sampler(&crate::gpu::SamplerDescriptor {
            label: Some("rt_env_sampler"),
            address_mode_u: crate::gpu::AddressMode::Repeat,
            address_mode_v: crate::gpu::AddressMode::ClampToEdge,
            address_mode_w: crate::gpu::AddressMode::ClampToEdge,
            mag_filter: crate::gpu::FilterMode::Linear,
            min_filter: crate::gpu::FilterMode::Linear,
            mipmap_filter: crate::gpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("rt_bgl"),
            entries: &[
                bgl_entry(0, buffer_ty_uniform()),
                bgl_entry(1, buffer_ty_storage(true)),
                bgl_entry(2, buffer_ty_storage(true)),
                bgl_entry(3, buffer_ty_storage(true)),
                bgl_entry(4, buffer_ty_storage(true)),
                bgl_entry(5, buffer_ty_storage(false)),
                bgl_entry(6, buffer_ty_storage(false)),
                bgl_entry(7, buffer_ty_storage(false)),
                bgl_entry(8, texture_ty_float()),
                bgl_entry(9, sampler_ty_filtering()),
            ],
        });
        let shader = crate::resources::builders::wgsl_module(
            device,
            "rt_kernel",
            crate::resources::builders::wgsl_source!("raytrace"),
        );
        let layout =
            crate::resources::builders::pipeline_layout(device, Some("rt_layout"), &[&bgl]);
        let pipeline = crate::resources::builders::compute_pipeline(
            device,
            "rt_pipeline",
            &layout,
            &shader,
            "main",
        );

        // Lightmap-bake pipeline: same module, `bake_main` entry. Its layout
        // drops the denoiser guide buffers (6/7) and adds the texel G-buffer
        // (10/11); the shared scene, frame, accum, and environment bindings are
        // the same.
        let bake_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("rt_bake_bgl"),
            entries: &[
                bgl_entry(0, buffer_ty_uniform()),
                bgl_entry(1, buffer_ty_storage(true)),
                bgl_entry(2, buffer_ty_storage(true)),
                bgl_entry(3, buffer_ty_storage(true)),
                bgl_entry(4, buffer_ty_storage(true)),
                bgl_entry(5, buffer_ty_storage(false)),
                bgl_entry(8, texture_ty_float()),
                bgl_entry(9, sampler_ty_filtering()),
                bgl_entry(10, buffer_ty_storage(true)),
                bgl_entry(11, buffer_ty_storage(true)),
            ],
        });
        let bake_layout = crate::resources::builders::pipeline_layout(
            device,
            Some("rt_bake_layout"),
            &[&bake_bgl],
        );
        let bake_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "rt_bake_pipeline",
            &bake_layout,
            &shader,
            "bake_main",
        );

        // Denoiser pipeline, compiled once and reused across settled frames.
        let denoise_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("rt_denoise_bgl"),
            entries: &[
                bgl_entry(0, buffer_ty_uniform()),
                bgl_entry(1, buffer_ty_storage(true)),
                bgl_entry(2, buffer_ty_storage(true)),
                bgl_entry(3, buffer_ty_storage(true)),
                bgl_entry(4, buffer_ty_storage(false)),
            ],
        });
        let denoise_shader = crate::resources::builders::wgsl_module(
            device,
            "rt_denoise",
            crate::resources::builders::wgsl_source!("denoise"),
        );
        let denoise_layout = crate::resources::builders::pipeline_layout(
            device,
            Some("rt_denoise_layout"),
            &[&denoise_bgl],
        );
        let denoise_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "rt_denoise_pipeline",
            &denoise_layout,
            &denoise_shader,
            "main",
        );

        Self {
            pipeline,
            bgl,
            bake_pipeline,
            bake_bgl,
            denoise_pipeline,
            denoise_bgl,
            node_buf,
            tri_buf,
            mat_buf,
            light_buf,
            frame_buf,
            _env_texture: env_texture,
            env_view,
            env_sampler,
            has_env,
            num_lights,
            sky_top: scene.sky_top,
            sky_bottom: scene.sky_bottom,
            has_geometry,
            sized: None,
            accumulated: 0,
        }
    }

    /// Ensure the size-dependent buffers and bind group match `width`/`height`,
    /// (re)allocating them if the resolution changed since the last trace.
    fn ensure_sized(&mut self, device: &crate::gpu::Device, width: u32, height: u32) {
        if let Some(s) = &self.sized {
            if s.width == width && s.height == height {
                return;
            }
        }
        // New buffers mean a fresh accumulation; the old running average does not
        // carry to a different resolution.
        self.accumulated = 0;
        let bytes = (width as u64) * (height as u64) * 16;
        let storage = |label: &str, extra: crate::gpu::BufferUsages| {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: crate::gpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let accum_buf = storage("rt_accum", crate::gpu::BufferUsages::COPY_SRC);
        let gbuf_albedo = storage("rt_gbuf_albedo", crate::gpu::BufferUsages::empty());
        let gbuf_normal = storage("rt_gbuf_normal", crate::gpu::BufferUsages::empty());
        let staging_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("rt_staging"),
            size: bytes,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("rt_bg"),
            layout: &self.bgl,
            entries: &[
                bg_entry(0, &self.frame_buf),
                bg_entry(1, &self.node_buf),
                bg_entry(2, &self.tri_buf),
                bg_entry(3, &self.mat_buf),
                bg_entry(4, &self.light_buf),
                bg_entry(5, &accum_buf),
                bg_entry(6, &gbuf_albedo),
                bg_entry(7, &gbuf_normal),
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: crate::gpu::BindingResource::TextureView(&self.env_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 9,
                    resource: crate::gpu::BindingResource::Sampler(&self.env_sampler),
                },
            ],
        });
        self.sized = Some(TracerSized {
            width,
            height,
            bytes,
            accum_buf,
            staging_buf,
            gbuf_albedo,
            gbuf_normal,
            bind_group,
        });
    }

    /// Trace `camera` at `settings` from scratch and read back the HDR image.
    /// Discards any running accumulation first, so each call is an independent
    /// `settings.samples`-sample render. Reuses the pipeline and uploaded scene.
    pub fn trace(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        camera: &RtCamera,
        settings: &RtSettings,
    ) -> RtImage {
        self.reset_accumulation();
        self.accumulate(device, queue, camera, settings)
    }

    /// Add `settings.samples` more samples on top of the running accumulation and
    /// read back the current mean. Successive calls with the same camera converge
    /// toward the reference image (progressive rendering); call
    /// [`reset_accumulation`](Self::reset_accumulation) when the camera or output
    /// size changes (a size change resets automatically). [`accumulated_samples`](Self::accumulated_samples)
    /// reports how many samples are blended in so far.
    pub fn accumulate(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        camera: &RtCamera,
        settings: &RtSettings,
    ) -> RtImage {
        let width = camera.width.max(1);
        let height = camera.height.max(1);

        if !self.has_geometry {
            return RtImage {
                width,
                height,
                rgba: vec![0.0; (width * height * 4) as usize],
            };
        }

        self.ensure_sized(device, width, height);
        let s = self.sized.as_ref().expect("sized set by ensure_sized");

        let base_frame = FrameUniform {
            inv_view_proj: camera.inv_view_proj.to_cols_array(),
            cam_pos: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            sky_top: [self.sky_top[0], self.sky_top[1], self.sky_top[2], 0.0],
            sky_bottom: [
                self.sky_bottom[0],
                self.sky_bottom[1],
                self.sky_bottom[2],
                0.0,
            ],
            dims: [width, height, self.num_lights, settings.max_bounces.max(1)],
            params: [0, 0, 0, 0],
        };

        // Dispatch `settings.samples` more samples in batches, so no single
        // dispatch runs long enough to risk a GPU watchdog reset. `sample_base`
        // is the running count: the shader blends each batch into the mean, so
        // the first batch of a fresh accumulation (base 0) overwrites and later
        // batches refine. The seed is keyed on the running count so every batch,
        // within a call and across calls, draws a distinct sample set.
        const BATCH: u32 = 32;
        let gx = width.div_ceil(8);
        let gy = height.div_ceil(8);

        let start = self.accumulated;
        let target = start + settings.samples.max(1);
        let mut done = start;
        while done < target {
            let this_batch = (target - done).min(BATCH);
            let mut fu = base_frame;
            fu.params = [
                this_batch,
                done,
                done.wrapping_mul(2_654_435_761).wrapping_add(1),
                self.has_env as u32,
            ];
            queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&fu));

            let mut encoder =
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("rt_encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                    label: Some("rt_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &s.bind_group, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));

            done += this_batch;
        }
        self.accumulated = target;

        // Optionally denoise, then read back whichever buffer holds the result.
        let denoised;
        let result_buf: &crate::gpu::Buffer = if settings.denoise {
            denoised = denoise(
                device,
                queue,
                &self.denoise_pipeline,
                &self.denoise_bgl,
                width,
                height,
                s.bytes,
                &s.accum_buf,
                &s.gbuf_albedo,
                &s.gbuf_normal,
            );
            &denoised
        } else {
            &s.accum_buf
        };

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("rt_readback"),
        });
        encoder.copy_buffer_to_buffer(result_buf, 0, &s.staging_buf, 0, s.bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = s.staging_buf.slice(..);
        slice.map_async(crate::gpu::MapMode::Read, |_| {});
        let _ = device.poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        });
        let rgba: Vec<f32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        s.staging_buf.unmap();

        RtImage {
            width,
            height,
            rgba,
        }
    }

    /// Samples blended into the accumulation so far (0 after a reset or a size
    /// change). A convergence signal: keep calling [`accumulate`](Self::accumulate)
    /// until this reaches the target sample count.
    pub fn accumulated_samples(&self) -> u32 {
        self.accumulated
    }

    /// Discard the running accumulation so the next [`accumulate`](Self::accumulate)
    /// starts a fresh image. Call when the camera or scene changes.
    pub fn reset_accumulation(&mut self) {
        self.accumulated = 0;
    }

    /// Bake incident irradiance into a lightmap atlas.
    ///
    /// Runs the path integrator with a texel ray-gen front-end: one invocation
    /// per atlas texel shoots the GI hemisphere from the surface point behind it
    /// (from `surfaces`), summing direct irradiance from the analytic lights and
    /// cosine-sampled indirect + environment irradiance. Returns an [`RtImage`]
    /// sized to the atlas: linear HDR incident irradiance in `rgb`, with `a`
    /// carrying texel coverage (1 on a baked texel, 0 on an empty one). Empty
    /// texels (`world_pos.w <= 0`) are left black.
    ///
    /// This is a one-shot solve, not progressive: it allocates the atlas-sized
    /// buffers, accumulates `settings.samples` samples per texel, and reads back.
    /// `settings.denoise` is ignored (baked GI wants a dedicated guided denoiser,
    /// a later stage, not the interactive a-trous pass). Returns a black atlas if
    /// the scene has no geometry or the surface slices are the wrong length.
    pub fn bake(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        surfaces: &TexelSurfaces,
        settings: &RtSettings,
    ) -> RtImage {
        let width = surfaces.width.max(1);
        let height = surfaces.height.max(1);
        let texels = (width * height) as usize;

        let black = || RtImage {
            width,
            height,
            rgba: vec![0.0; texels * 4],
        };
        if !self.has_geometry
            || surfaces.world_pos.len() != texels
            || surfaces.world_normal.len() != texels
        {
            return black();
        }

        let bytes = (texels as u64) * 16;
        let pos_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("rt_bake_texel_pos"),
            contents: bytemuck::cast_slice(surfaces.world_pos),
            usage: crate::gpu::BufferUsages::STORAGE,
        });
        let nrm_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("rt_bake_texel_nrm"),
            contents: bytemuck::cast_slice(surfaces.world_normal),
            usage: crate::gpu::BufferUsages::STORAGE,
        });
        let accum_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("rt_bake_accum"),
            size: bytes,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("rt_bake_staging"),
            size: bytes,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("rt_bake_bg"),
            layout: &self.bake_bgl,
            entries: &[
                bg_entry(0, &self.frame_buf),
                bg_entry(1, &self.node_buf),
                bg_entry(2, &self.tri_buf),
                bg_entry(3, &self.mat_buf),
                bg_entry(4, &self.light_buf),
                bg_entry(5, &accum_buf),
                crate::gpu::BindGroupEntry {
                    binding: 8,
                    resource: crate::gpu::BindingResource::TextureView(&self.env_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 9,
                    resource: crate::gpu::BindingResource::Sampler(&self.env_sampler),
                },
                bg_entry(10, &pos_buf),
                bg_entry(11, &nrm_buf),
            ],
        });

        let base_frame = FrameUniform {
            inv_view_proj: Mat4::IDENTITY.to_cols_array(),
            cam_pos: [0.0; 4],
            sky_top: [self.sky_top[0], self.sky_top[1], self.sky_top[2], 0.0],
            sky_bottom: [
                self.sky_bottom[0],
                self.sky_bottom[1],
                self.sky_bottom[2],
                0.0,
            ],
            dims: [width, height, self.num_lights, settings.max_bounces.max(1)],
            params: [0, 0, 0, 0],
        };

        // Batch the samples so no single dispatch runs long enough to risk a GPU
        // watchdog reset; `sample_base` blends each batch into the running mean.
        const BATCH: u32 = 16;
        let gx = width.div_ceil(8);
        let gy = height.div_ceil(8);
        let target = settings.samples.max(1);
        let mut done = 0u32;
        while done < target {
            let this_batch = (target - done).min(BATCH);
            let mut fu = base_frame;
            fu.params = [
                this_batch,
                done,
                done.wrapping_mul(2_654_435_761).wrapping_add(1),
                self.has_env as u32,
            ];
            queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&fu));

            let mut encoder =
                device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
                    label: Some("rt_bake_encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                    label: Some("rt_bake_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bake_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            done += this_batch;
        }

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("rt_bake_readback"),
        });
        encoder.copy_buffer_to_buffer(&accum_buf, 0, &staging_buf, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buf.slice(..);
        slice.map_async(crate::gpu::MapMode::Read, |_| {});
        let _ = device.poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        });
        let rgba: Vec<f32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        staging_buf.unmap();

        RtImage {
            width,
            height,
            rgba,
        }
    }
}

/// Bake incident irradiance into a lightmap atlas from per-texel surfaces.
///
/// Builds a tracer for `scene` and runs [`Tracer::bake`]. Like [`trace`], this
/// rebuilds the BVH and uploads the scene each call; a bake is a one-shot solve,
/// so that setup cost is not worth caching. See [`Tracer::bake`] for the output
/// layout.
pub fn bake_lightmap(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    scene: &RtScene,
    surfaces: &TexelSurfaces,
    settings: &RtSettings,
) -> RtImage {
    Tracer::new(device, queue, scene).bake(device, queue, surfaces, settings)
}

/// Run the edge-aware a-trous denoiser over `accum` and return the buffer that
/// holds the filtered result (with `COPY_SRC` for readback). Ping-pongs between
/// two scratch buffers across a fixed set of growing-step iterations. The
/// pipeline and layout are owned by the [`Tracer`] and passed in, so no shader
/// is compiled here.
#[allow(clippy::too_many_arguments)]
fn denoise(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pipeline: &crate::gpu::ComputePipeline,
    bgl: &crate::gpu::BindGroupLayout,
    width: u32,
    height: u32,
    bytes: u64,
    accum: &crate::gpu::Buffer,
    gbuf_albedo: &crate::gpu::Buffer,
    gbuf_normal: &crate::gpu::Buffer,
) -> crate::gpu::Buffer {
    let make_scratch = |label: &str| {
        device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let ping = make_scratch("rt_denoise_ping");
    let pong = make_scratch("rt_denoise_pong");

    let param_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("rt_denoise_params"),
        size: std::mem::size_of::<GpuDenoiseParams>() as u64,
        usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    const ITERS: u32 = 5;
    let gx = width.div_ceil(8);
    let gy = height.div_ceil(8);

    let mut src: &crate::gpu::Buffer = accum;
    for i in 0..ITERS {
        let dst: &crate::gpu::Buffer = if i % 2 == 0 { &ping } else { &pong };
        let params = GpuDenoiseParams {
            dims: [width, height],
            step: 1u32 << i,
            _pad: 0,
            sigma_n: 64.0,
            sigma_l: 4.0,
            _pad2: [0.0, 0.0],
        };
        queue.write_buffer(&param_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("rt_denoise_bg"),
            layout: bgl,
            entries: &[
                bg_entry(0, &param_buf),
                bg_entry(1, gbuf_albedo),
                bg_entry(2, gbuf_normal),
                bg_entry(3, src),
                bg_entry(4, dst),
            ],
        });

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("rt_denoise_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("rt_denoise_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        src = dst;
    }

    // The last write landed in `ping` when ITERS is odd, else `pong`.
    if (ITERS - 1) % 2 == 0 { ping } else { pong }
}

fn bgl_entry(binding: u32, ty: crate::gpu::BindingType) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn buffer_ty_uniform() -> crate::gpu::BindingType {
    crate::gpu::BindingType::Buffer {
        ty: crate::gpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn buffer_ty_storage(read_only: bool) -> crate::gpu::BindingType {
    crate::gpu::BindingType::Buffer {
        ty: crate::gpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn texture_ty_float() -> crate::gpu::BindingType {
    crate::gpu::BindingType::Texture {
        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
        view_dimension: crate::gpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

fn sampler_ty_filtering() -> crate::gpu::BindingType {
    crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering)
}

fn bg_entry(binding: u32, buffer: &crate::gpu::Buffer) -> crate::gpu::BindGroupEntry<'_> {
    crate::gpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
