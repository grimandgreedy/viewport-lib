//! Offscreen path tracer for reference images.
//!
//! A compute-only path tracer that renders to an offscreen buffer. It builds a
//! world-space triangle BVH, traces it with a megakernel
//! (`src/shaders/raytrace.wgsl`), and accumulates samples into an HDR buffer
//! that is read back to the CPU. It uses the same direct BRDF as the rasteriser
//! (`helpers/brdf.wgsl`), so both produce the same shading.
//!
//! It needs no hardware ray query and does not touch the main render path: build
//! an [`RtScene`], call [`trace`], get an [`RtImage`] back.
//!
//! Not handled yet: a two-level BVH, image-based lighting from an environment
//! map, transmission, and denoising. The environment is a hemisphere sky.

mod bvh;

use crate::gpu::util::DeviceExt;
use glam::{Mat4, Vec3};

/// Surface material for the tracer. A subset of the rasteriser's `Material`:
/// metallic-roughness and emissive.
#[derive(Clone, Copy, Debug)]
pub struct RtMaterial {
    /// Linear base colour (albedo for dielectrics, specular tint for metals).
    pub base_colour: [f32; 3],
    /// 0 = dielectric, 1 = metal.
    pub metallic: f32,
    /// Perceptual roughness in [0, 1].
    pub roughness: f32,
    /// Linear emitted radiance.
    pub emissive: [f32; 3],
}

impl Default for RtMaterial {
    fn default() -> Self {
        Self {
            base_colour: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0, 0.0, 0.0],
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

/// Trace settings.
#[derive(Clone, Copy, Debug)]
pub struct RtSettings {
    /// Total samples per pixel to accumulate.
    pub samples: u32,
    /// Maximum path length (bounces).
    pub max_bounces: u32,
}

impl Default for RtSettings {
    fn default() -> Self {
        Self {
            samples: 256,
            max_bounces: 8,
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

/// A scene to trace: triangles with per-vertex normals and a material index,
/// analytic lights, and a hemisphere sky used on ray miss.
#[derive(Default)]
pub struct RtScene {
    tris: Vec<[Vec3; 3]>,
    normals: Vec<[Vec3; 3]>,
    tri_mat: Vec<u32>,
    materials: Vec<RtMaterial>,
    lights: Vec<RtLight>,
    sky_top: [f32; 3],
    sky_bottom: [f32; 3],
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

/// Trace `scene` from `camera` and return the accumulated HDR image.
///
/// Runs entirely on the compute path (portable to every backend). Blocks until
/// the accumulation is read back to the CPU. For an empty scene, returns a black
/// image of the requested size.
pub fn trace(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    scene: &RtScene,
    camera: &RtCamera,
    settings: &RtSettings,
) -> RtImage {
    let width = camera.width.max(1);
    let height = camera.height.max(1);
    let pixels = (width * height) as usize;

    if scene.tris.is_empty() {
        return RtImage {
            width,
            height,
            rgba: vec![0.0; pixels * 4],
        };
    }

    // Build the BVH and the reordered triangle array it references.
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

    let mut gpu_mats: Vec<GpuMaterial> = scene
        .materials
        .iter()
        .map(|m| GpuMaterial {
            base: m.base_colour,
            metallic: m.metallic,
            emissive: m.emissive,
            roughness: m.roughness,
        })
        .collect();
    if gpu_mats.is_empty() {
        gpu_mats.push(GpuMaterial {
            base: [0.8, 0.8, 0.8],
            metallic: 0.0,
            emissive: [0.0; 3],
            roughness: 0.5,
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

    // Buffers.
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

    let accum_bytes = (pixels * 16) as u64;
    let accum_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("rt_accum"),
        size: accum_bytes,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("rt_staging"),
        size: accum_bytes,
        usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let frame_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("rt_frame"),
        size: std::mem::size_of::<FrameUniform>() as u64,
        usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Bind group layout: uniform + 4 read storage + 1 read_write storage.
    let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("rt_bgl"),
        entries: &[
            bgl_entry(0, buffer_ty_uniform()),
            bgl_entry(1, buffer_ty_storage(true)),
            bgl_entry(2, buffer_ty_storage(true)),
            bgl_entry(3, buffer_ty_storage(true)),
            bgl_entry(4, buffer_ty_storage(true)),
            bgl_entry(5, buffer_ty_storage(false)),
        ],
    });
    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some("rt_bg"),
        layout: &bgl,
        entries: &[
            bg_entry(0, &frame_buf),
            bg_entry(1, &node_buf),
            bg_entry(2, &tri_buf),
            bg_entry(3, &mat_buf),
            bg_entry(4, &light_buf),
            bg_entry(5, &accum_buf),
        ],
    });

    let shader = crate::resources::builders::wgsl_module(
        device,
        "rt_kernel",
        crate::resources::builders::wgsl_source!("raytrace"),
    );
    let layout = crate::resources::builders::pipeline_layout(device, Some("rt_layout"), &[&bgl]);
    let pipeline = crate::resources::builders::compute_pipeline(
        device,
        "rt_pipeline",
        &layout,
        &shader,
        "main",
    );

    // Progressive accumulation, dispatched in batches so no single dispatch runs
    // long enough to risk a GPU watchdog reset.
    let base_frame = FrameUniform {
        inv_view_proj: camera.inv_view_proj.to_cols_array(),
        cam_pos: [camera.position.x, camera.position.y, camera.position.z, 0.0],
        sky_top: [scene.sky_top[0], scene.sky_top[1], scene.sky_top[2], 0.0],
        sky_bottom: [
            scene.sky_bottom[0],
            scene.sky_bottom[1],
            scene.sky_bottom[2],
            0.0,
        ],
        dims: [width, height, num_lights, settings.max_bounces.max(1)],
        params: [0, 0, 0, 0],
    };

    let total = settings.samples.max(1);
    const BATCH: u32 = 32;
    let gx = width.div_ceil(8);
    let gy = height.div_ceil(8);

    let mut done = 0u32;
    let mut batch_index = 0u32;
    while done < total {
        let this_batch = (total - done).min(BATCH);
        let mut fu = base_frame;
        fu.params = [this_batch, done, batch_index.wrapping_mul(2_654_435_761), 0];
        queue.write_buffer(&frame_buf, 0, bytemuck::bytes_of(&fu));

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("rt_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("rt_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        done += this_batch;
        batch_index += 1;
    }

    // Read back the accumulated image.
    let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("rt_readback"),
    });
    encoder.copy_buffer_to_buffer(&accum_buf, 0, &staging_buf, 0, accum_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging_buf.slice(..);
    slice.map_async(crate::gpu::MapMode::Read, |_| {});
    let _ = device.poll(crate::gpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(30)),
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

fn bg_entry(binding: u32, buffer: &crate::gpu::Buffer) -> crate::gpu::BindGroupEntry<'_> {
    crate::gpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
