//! GPU particle systems.
//!
//! A particle system owns a persistent GPU buffer holding `capacity` particles.
//! Each particle stores its world-space position, velocity, lifetime remaining,
//! starting lifetime, colour, and size.
//!
//! The host calls [`DeviceResources::create_gpu_particle_system`] once at
//! startup to allocate the buffer, then submits a
//! [`GpuParticleSystemItem`](crate::renderer::GpuParticleSystemItem) per frame.
//! The renderer dispatches an emit compute pass (recycling dead particles back
//! into live ones based on `EmitterConfig`), then a sim compute pass
//! (integrating `ForceField`s and decrementing lifetime), then draws the live
//! particles via the route picked in [`GpuParticleSystemConfig::render`].
//!
//! Dead particles are not compacted. The emit shader scans for slots with
//! `lifetime <= 0` and reuses them; the draw shader emits a degenerate clip
//! position for dead slots so they cost nothing in the rasteriser. Compaction
//! would require a prefix sum each frame and is not worth the cost at the
//! particle counts the API targets (1k - 200k).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::renderer::{ParticleMeshAlign, SpriteBlend, SpriteLitParams, SpriteSizeMode};

/// GPU particle-system compute/draw pipelines, their layouts, and the live
/// systems. All pipelines are lazily built; `systems` holds the persistent
/// per-system GPU state, indexed by `GpuParticleSystemId`.
#[derive(Default)]
pub(crate) struct ParticleResources {
    /// Live particle systems. Slots can be reused after `drop_gpu_particle_system`.
    pub(crate) systems: Vec<Option<ParticleSystem>>,
    /// Layout for the emit + sim compute pipelines (group 1).
    pub(crate) sim_bgl: Option<wgpu::BindGroupLayout>,
    /// Layout for emit/sim params (group 0).
    pub(crate) params_bgl: Option<wgpu::BindGroupLayout>,
    /// Layout for the particle-sprite draw pipeline (group 1).
    pub(crate) draw_bgl: Option<wgpu::BindGroupLayout>,
    /// Compute pipeline that pops free-list slots and writes new particles.
    pub(crate) emit_pipeline: Option<wgpu::ComputePipeline>,
    /// Compute pipeline that integrates forces and decrements lifetime.
    pub(crate) sim_pipeline: Option<wgpu::ComputePipeline>,
    /// Draw pipeline variants for the particle-sprite shader, keyed by blend.
    pub(crate) sprite_pipeline_alpha: Option<crate::resources::DualPipeline>,
    pub(crate) sprite_pipeline_additive: Option<crate::resources::DualPipeline>,
    pub(crate) sprite_pipeline_premultiplied: Option<crate::resources::DualPipeline>,
    /// Lit variants of the GPU particle sprite pipelines.
    pub(crate) sprite_lit_pipeline_alpha: Option<crate::resources::DualPipeline>,
    pub(crate) sprite_lit_pipeline_additive: Option<crate::resources::DualPipeline>,
    pub(crate) sprite_lit_pipeline_premultiplied: Option<crate::resources::DualPipeline>,
    /// Group 2 BGL for the lit particle path: optional normal map + sampler.
    pub(crate) sprite_lit_bgl: Option<wgpu::BindGroupLayout>,
    /// Fallback bind group for the lit particle normal-map binding (group 2).
    pub(crate) sprite_lit_fallback_bg: Option<wgpu::BindGroup>,
    /// Layout for the mesh-route particle draw pipeline (group 1).
    pub(crate) mesh_draw_bgl: Option<wgpu::BindGroupLayout>,
    /// Draw pipeline variants for the particle-mesh shader, keyed by blend.
    pub(crate) mesh_pipeline_alpha: Option<crate::resources::DualPipeline>,
    pub(crate) mesh_pipeline_additive: Option<crate::resources::DualPipeline>,
    pub(crate) mesh_pipeline_premultiplied: Option<crate::resources::DualPipeline>,
}

crate::resources::handle::registry_handle! {
    /// Handle to a persistent GPU particle system.
    ///
    /// Returned by [`DeviceResources::create_gpu_particle_system`]. Stable
    /// until [`DeviceResources::drop_gpu_particle_system`] is called. An
    /// append-only registry handle.
    pub struct GpuParticleSystemId;
}

/// Persistent configuration for a particle system. Set at creation; the render
/// route and capacity are stable for the system's lifetime.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GpuParticleSystemConfig {
    /// Maximum number of simultaneously live particles. Memory cost scales
    /// linearly with this value (currently 80 bytes per particle plus a small
    /// fixed overhead).
    pub capacity: u32,
    /// How the live particles are drawn each frame.
    pub render: ParticleRender,
}

impl Default for GpuParticleSystemConfig {
    fn default() -> Self {
        Self {
            capacity: 10_000,
            render: ParticleRender::default(),
        }
    }
}

/// How a particle system draws its live particles.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ParticleRender {
    /// Draw each particle as a camera-facing billboard sprite.
    Sprite {
        /// Optional texture sampled per fragment. `None` renders solid quads
        /// tinted by the particle colour.
        texture_id: Option<crate::resources::TextureId>,
        /// GPU blend state.
        blend: SpriteBlend,
        /// Screen-space or world-space sizing for the per-particle `size`.
        size_mode: SpriteSizeMode,
        /// Whether the draw writes to depth.
        depth_write: bool,
        /// When `true`, the draw runs through the lit particle sprite pipeline
        /// and picks up the scene lighting environment. Default `false`
        /// preserves the emissive billboard look.
        lit: bool,
        /// Lighting parameters used when `lit` is `true`.
        lit_params: SpriteLitParams,
        /// Optional tangent-space normal map for the `NormalMap` mode.
        normal_texture_id: Option<crate::resources::TextureId>,
    },
    /// Draw each particle as an instance of an uploaded mesh. The vertex
    /// shader composes the per-instance transform from the live particle's
    /// position, velocity, `size`, and (for `Random` align) the spawn seed.
    /// Unlit; the particle colour multiplies an optional albedo sample.
    Mesh {
        /// Mesh handle returned by `DeviceResources::upload_mesh_data`.
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        /// Optional albedo texture handle. `None` renders flat-tinted.
        texture_id: Option<crate::resources::TextureId>,
        /// GPU blend state.
        blend: SpriteBlend,
        /// How per-particle rotation is derived.
        align: ParticleMeshAlign,
    },
}

impl Default for ParticleRender {
    fn default() -> Self {
        ParticleRender::Sprite {
            texture_id: None,
            blend: SpriteBlend::AlphaBlend,
            size_mode: SpriteSizeMode::ScreenSpace,
            depth_write: false,
            lit: false,
            lit_params: SpriteLitParams {
                roughness: 0.9,
                normal_mode: crate::renderer::SpriteNormalMode::Spherical,
                receive_shadows: false,
                ambient_scale: 1.0,
            },
            normal_texture_id: None,
        }
    }
}

/// One particle as it lives on the GPU. Layout matches `Particle` in
/// `particle_emit.wgsl` and `particle_sim.wgsl`. Eighty bytes, naturally
/// 16-byte aligned.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuParticle {
    pub position: [f32; 3],
    pub lifetime: f32, // seconds remaining; <= 0 means dead
    pub velocity: [f32; 3],
    pub max_lifetime: f32, // initial lifetime, used for fade ramps
    pub colour: [f32; 4],
    pub size: f32,
    /// Stable per-spawn seed used by the mesh draw route for `Random` align
    /// rotation. Written by `particle_emit.wgsl`; left untouched by the sim.
    pub spawn_seed: f32,
    pub _pad: [f32; 2],
}

/// Uniform buffer matching `EmitParams` in `particle_emit.wgsl`. 96 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct EmitParamsGpu {
    pub spawn_min: [f32; 3],
    pub spawn_kind: u32, // 0=Point, 1=Box, 2=Sphere
    pub spawn_max: [f32; 3],
    pub spawn_radius: f32, // sphere radius (Sphere only)
    pub vel_min: [f32; 3],
    pub vel_kind: u32, // 0=Fixed, 1=UniformBox, 2=UniformCone
    pub vel_max: [f32; 3],
    pub cone_half_angle: f32,
    pub vel_axis: [f32; 3],
    pub cone_min_speed: f32,
    pub colour: [f32; 4],
    pub spawn_count: u32,
    pub capacity: u32,
    pub rng_seed: u32,
    pub size: f32,
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    pub cone_max_speed: f32,
    pub _pad: f32,
}

/// Maximum number of forces in a single sim dispatch. Forces are inlined into
/// the uniform buffer; bump this if it becomes a real limit (so far no game
/// effect uses more than 3-4 forces simultaneously).
pub(crate) const MAX_FORCES: usize = 8;

/// One force on the GPU. Tagged union; 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuForce {
    pub kind: u32, // 0=Gravity, 1=Drag, 2=PointAttractor
    pub _pad: [u32; 3],
    pub v0: [f32; 4], // Gravity: xyz=acceleration / Drag: x=coefficient / Attractor: xyz=position, w=strength
    pub v1: [f32; 4], // Attractor: x=falloff
}

/// Uniform buffer matching `SimParams` in `particle_sim.wgsl`. Forces are
/// inlined so the sim pipeline reads from a single uniform binding.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SimParamsGpu {
    pub dt: f32,
    pub capacity: u32,
    pub force_count: u32,
    pub _pad: u32,
    pub forces: [GpuForce; MAX_FORCES],
}

/// Per-system persistent GPU state.
pub(crate) struct ParticleSystem {
    pub capacity: u32,
    pub render: ParticleRender,
    /// `capacity` particles in `GpuParticle` layout. STORAGE + VERTEX usage so
    /// the draw pipeline can bind it directly.
    pub particle_buf: wgpu::Buffer,
    /// Single atomic u32 counter rewritten by the host before each emit
    /// dispatch and decremented by emit threads as they claim slots. Reused
    /// across frames; nothing is preserved between dispatches.
    pub emit_counter_buf: wgpu::Buffer,
    /// Bind group for the sim/emit compute pipelines (group 1).
    pub sim_bg: wgpu::BindGroup,
    /// Bind group for the sprite draw pipeline (group 1). `None` when the
    /// system's render route is not `Sprite`.
    pub draw_bg: Option<wgpu::BindGroup>,
    /// Bind group for the mesh draw pipeline (group 1). `None` when the
    /// system's render route is not `Mesh`.
    pub draw_bg_mesh: Option<wgpu::BindGroup>,
    /// Group 2 bind group for the lit draw pipeline (normal map + sampler).
    /// `None` when the system's render route is not lit.
    pub draw_lit_normal_bg: Option<wgpu::BindGroup>,
    /// Uniform buffers backing whichever draw bind group is populated.
    pub _draw_uniform_buf: Option<wgpu::Buffer>,
    /// Whether the slot is in use. The slot is reused lazily by future creates.
    pub alive: bool,
    /// Frame count since creation; used to seed the emit RNG so a freshly
    /// created system gets a different sequence from one that has been
    /// running for a while.
    pub frame_counter: u32,
    /// Fractional spawn accumulator. `rate * dt` rarely lands on an integer
    /// per frame; the fractional remainder rolls over to the next frame so
    /// the long-term average emission matches the configured rate.
    pub spawn_accumulator: f32,
}

/// Which draw family the render loop should dispatch for this system.
#[derive(Copy, Clone, Debug)]
pub(crate) enum ParticleDrawRoute {
    Sprite {
        lit: bool,
    },
    Mesh {
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
    },
}

/// Per-frame data for one particle system, populated in prepare and consumed
/// in render.
pub(crate) struct ParticleFrameData {
    /// Index into `particle_systems`.
    pub system_idx: usize,
    /// Picked at submit time; consumed by the draw pipeline router.
    pub blend: SpriteBlend,
    /// Whether to skip the draw (hidden item).
    pub hidden: bool,
    /// Which draw family to dispatch.
    pub route: ParticleDrawRoute,
}

impl crate::resources::DeviceResources {
    /// Allocate a persistent GPU particle system.
    ///
    /// The returned [`GpuParticleSystemId`] stays valid until
    /// [`drop_gpu_particle_system`](Self::drop_gpu_particle_system) is called
    /// or the renderer is dropped.
    pub fn create_gpu_particle_system(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &GpuParticleSystemConfig,
    ) -> GpuParticleSystemId {
        self.ensure_particle_pipelines(device);

        let capacity = config.capacity.max(1);

        // Persistent particle buffer, initialised to all-dead.
        let particle_bytes_len = (capacity as usize) * std::mem::size_of::<GpuParticle>();
        let zero_particles = vec![0u8; particle_bytes_len];
        let particle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_particle_buf"),
            contents: &zero_particles,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
        });

        // Atomic counter rewritten per emit dispatch.
        let emit_counter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_particle_emit_counter"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Draw-side state. The sprite route uses the existing sprite-style
        // uniform; the mesh route uses a smaller uniform with the align mode
        // and a `has_texture` flag.
        enum DrawState {
            Sprite {
                texture_id: Option<crate::resources::TextureId>,
                size_mode: SpriteSizeMode,
                lit: bool,
                lit_params: SpriteLitParams,
                normal_texture_id: Option<crate::resources::TextureId>,
            },
            Mesh {
                texture_id: Option<crate::resources::TextureId>,
                align: ParticleMeshAlign,
            },
        }
        let draw_state = match &config.render {
            ParticleRender::Sprite {
                texture_id,
                blend: _,
                size_mode,
                depth_write: _,
                lit,
                lit_params,
                normal_texture_id,
            } => DrawState::Sprite {
                texture_id: *texture_id,
                size_mode: *size_mode,
                lit: *lit,
                lit_params: *lit_params,
                normal_texture_id: *normal_texture_id,
            },
            ParticleRender::Mesh {
                texture_id,
                blend: _,
                align,
                mesh_id: _,
            } => DrawState::Mesh {
                texture_id: *texture_id,
                align: *align,
            },
        };

        let _ = queue; // queue currently unused; reserved for textures upload paths

        let sim_bgl = self
            .particle
            .sim_bgl
            .as_ref()
            .expect("ensure_particle_pipelines failed to create sim BGL");
        let sim_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_particle_sim_bg"),
            layout: sim_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: emit_counter_buf.as_entire_binding(),
                },
            ],
        });

        // Per-route draw resources. Exactly one of `sprite_draw_bg` and
        // `mesh_draw_bg` is populated; the other arm stays `None`.
        let mut sprite_draw_bg: Option<wgpu::BindGroup> = None;
        let mut mesh_draw_bg: Option<wgpu::BindGroup> = None;
        let mut sprite_lit_normal_bg: Option<wgpu::BindGroup> = None;
        let draw_uniform_buf: Option<wgpu::Buffer>;

        match draw_state {
            DrawState::Sprite {
                texture_id,
                size_mode,
                lit,
                lit_params,
                normal_texture_id,
            } => {
                #[repr(C)]
                #[derive(Copy, Clone, Pod, Zeroable)]
                struct SpriteDrawUniform {
                    model: [[f32; 4]; 4],
                    world_space: u32,
                    has_texture: u32,
                    normal_mode: u32,
                    has_normal_map: u32,
                    ambient_scale: f32,
                    roughness: f32,
                    _pad0: u32,
                    _pad1: u32,
                }
                let normal_mode_u32 = match lit_params.normal_mode {
                    crate::renderer::SpriteNormalMode::Spherical => 0u32,
                    crate::renderer::SpriteNormalMode::Flat => 1u32,
                    crate::renderer::SpriteNormalMode::NormalMap => 2u32,
                };
                let uniform = SpriteDrawUniform {
                    model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    world_space: matches!(size_mode, SpriteSizeMode::WorldSpace) as u32,
                    has_texture: texture_id.is_some() as u32,
                    normal_mode: normal_mode_u32,
                    has_normal_map: normal_texture_id.is_some() as u32,
                    ambient_scale: lit_params.ambient_scale,
                    roughness: lit_params.roughness,
                    _pad0: 0,
                    _pad1: 0,
                };
                let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gpu_particle_sprite_draw_uniform"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let texture_view = match texture_id {
                    Some(id) if self.content.textures.get(id).is_some() => {
                        &self.content.textures.get(id).unwrap().view
                    }
                    _ => &self.content.fallback_lut_view,
                };
                let draw_bgl = self
                    .particle
                    .draw_bgl
                    .as_ref()
                    .expect("ensure_particle_pipelines failed to create draw BGL");
                sprite_draw_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu_particle_draw_bg"),
                    layout: draw_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: particle_buf.as_entire_binding(),
                        },
                    ],
                }));
                if lit {
                    let lit_bgl = self
                        .particle
                        .sprite_lit_bgl
                        .as_ref()
                        .expect("ensure_particle_pipelines failed to create lit BGL");
                    let normal_view = match normal_texture_id {
                        Some(id) if self.content.textures.get(id).is_some() => {
                            &self.content.textures.get(id).unwrap().view
                        }
                        _ => &self.fallback_normal_map_view,
                    };
                    sprite_lit_normal_bg =
                        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("gpu_particle_lit_normal_bg"),
                            layout: lit_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(normal_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.material_sampler,
                                    ),
                                },
                            ],
                        }));
                }
                draw_uniform_buf = Some(uniform_buf);
            }
            DrawState::Mesh { texture_id, align } => {
                #[repr(C)]
                #[derive(Copy, Clone, Pod, Zeroable)]
                struct MeshDrawUniform {
                    align: u32,
                    has_texture: u32,
                    _pad0: u32,
                    _pad1: u32,
                }
                let align_u32 = match align {
                    ParticleMeshAlign::Identity => 0u32,
                    ParticleMeshAlign::Velocity => 1u32,
                    ParticleMeshAlign::Random => 2u32,
                };
                let uniform = MeshDrawUniform {
                    align: align_u32,
                    has_texture: texture_id.is_some() as u32,
                    _pad0: 0,
                    _pad1: 0,
                };
                let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gpu_particle_mesh_draw_uniform"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let texture_view = match texture_id {
                    Some(id) if self.content.textures.get(id).is_some() => {
                        &self.content.textures.get(id).unwrap().view
                    }
                    _ => &self.fallback_texture.view,
                };
                let mesh_bgl = self
                    .particle
                    .mesh_draw_bgl
                    .as_ref()
                    .expect("ensure_particle_pipelines failed to create mesh draw BGL");
                mesh_draw_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu_particle_mesh_draw_bg"),
                    layout: mesh_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: particle_buf.as_entire_binding(),
                        },
                    ],
                }));
                draw_uniform_buf = Some(uniform_buf);
            }
        }

        let system = ParticleSystem {
            capacity,
            render: config.render.clone(),
            particle_buf,
            emit_counter_buf,
            sim_bg,
            draw_bg: sprite_draw_bg,
            draw_bg_mesh: mesh_draw_bg,
            draw_lit_normal_bg: sprite_lit_normal_bg,
            _draw_uniform_buf: draw_uniform_buf,
            alive: true,
            frame_counter: 0,
            spawn_accumulator: 0.0,
        };

        if let Some(idx) = self
            .particle
            .systems
            .iter()
            .position(|slot: &Option<ParticleSystem>| slot.as_ref().is_none_or(|s| !s.alive))
        {
            self.particle.systems[idx] = Some(system);
            GpuParticleSystemId(idx)
        } else {
            self.particle.systems.push(Some(system));
            GpuParticleSystemId(self.particle.systems.len() - 1)
        }
    }

    /// Release a particle system. The handle becomes invalid; the slot is
    /// reused on the next `create_gpu_particle_system` call.
    pub fn drop_gpu_particle_system(&mut self, id: GpuParticleSystemId) {
        if let Some(Some(s)) = self.particle.systems.get_mut(id.0) {
            s.alive = false;
        }
    }

    #[allow(dead_code)]
    pub(crate) fn particle_system(&self, id: GpuParticleSystemId) -> Option<&ParticleSystem> {
        self.particle
            .systems
            .get(id.0)?
            .as_ref()
            .filter(|s| s.alive)
    }

    #[allow(dead_code)]
    pub(crate) fn particle_system_mut(
        &mut self,
        id: GpuParticleSystemId,
    ) -> Option<&mut ParticleSystem> {
        self.particle
            .systems
            .get_mut(id.0)?
            .as_mut()
            .filter(|s| s.alive)
    }

    /// Lazily create the compute + draw pipelines used by every particle
    /// system. No-op once the bind group layouts are present.
    pub(crate) fn ensure_particle_pipelines(&mut self, device: &wgpu::Device) {
        if self.particle.sim_bgl.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // Group 0: emit/sim params (uniform).
        let params_bgl = crate::resources::builders::uniform_bgl(
            device,
            "gpu_particle_params_bgl",
            wgpu::ShaderStages::COMPUTE,
        );

        // Group 1 (sim/emit): particle buffer + atomic emit counter.
        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_particle_sim_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

        // Group 1 (draw): sprite uniform + texture + sampler + particle buffer.
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_particle_draw_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Compute pipelines.
        let emit_shader = crate::resources::builders::wgsl_module(
            device,
            "particle_emit_shader",
            crate::resources::builders::wgsl_source!("particle_emit"),
        );
        let sim_shader = crate::resources::builders::wgsl_module(
            device,
            "particle_sim_shader",
            crate::resources::builders::wgsl_source!("particle_sim"),
        );

        let compute_layout = crate::resources::builders::pipeline_layout(
            device,
            "particle_compute_layout",
            &[&params_bgl, &sim_bgl],
        );

        let emit_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "particle_emit_pipeline",
            &compute_layout,
            &emit_shader,
            "emit_main",
        );

        let sim_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "particle_sim_pipeline",
            &compute_layout,
            &sim_shader,
            "sim_main",
        );

        // Draw pipelines: three blend variants of the same shader.
        let sprite_shader = crate::resources::builders::wgsl_module(
            device,
            "particle_sprite_shader",
            crate::resources::builders::wgsl_source!("particle_sprite"),
        );

        let draw_layout = crate::resources::builders::standard_scene_layout(
            device,
            "particle_draw_layout",
            &self.camera_bind_group_layout,
            &draw_bgl,
        );

        let sample_count = self.sample_count;
        let ldr_format = self.target_format;
        let alpha = wgpu::BlendState::ALPHA_BLENDING;
        let additive = crate::resources::builders::ADDITIVE_BLEND;
        let premul = crate::resources::builders::PREMULTIPLIED_BLEND;

        // Particle sprites are billboards: `Less` depth test, no depth write, no
        // culling. Only the blend mode varies across the three variants.
        let make_draw = |blend: wgpu::BlendState, label: &str| {
            crate::resources::builders::build_dual_pipeline(
                device,
                &crate::resources::builders::DualPipelineDesc {
                    label,
                    layout: &draw_layout,
                    shader: &sprite_shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &[],
                    blend: Some(blend),
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        self.particle.sprite_pipeline_alpha = Some(make_draw(alpha, "particle_sprite_alpha"));
        self.particle.sprite_pipeline_additive =
            Some(make_draw(additive, "particle_sprite_additive"));
        self.particle.sprite_pipeline_premultiplied =
            Some(make_draw(premul, "particle_sprite_premultiplied"));

        // Lit GPU particle sprite pipelines. Same vertex inputs and draw BGL
        // as the emissive path; group 2 adds the optional normal-map binding
        // and the shader pulls scene lighting via the camera bind group.
        let lit_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "gpu_particle_lit_bgl",
            wgpu::ShaderStages::FRAGMENT,
        );

        let lit_shader = crate::resources::builders::wgsl_module(
            device,
            "particle_sprite_lit_shader",
            crate::resources::builders::wgsl_source!("particle_sprite_lit"),
        );

        let lit_layout = crate::resources::builders::pipeline_layout(
            device,
            "particle_draw_lit_layout",
            &[&self.camera_bind_group_layout, &draw_bgl, &lit_bgl],
        );

        let make_lit_draw = |blend: wgpu::BlendState, label: &str| {
            crate::resources::builders::build_dual_pipeline(
                device,
                &crate::resources::builders::DualPipelineDesc {
                    label,
                    layout: &lit_layout,
                    shader: &lit_shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &[],
                    blend: Some(blend),
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        self.particle.sprite_lit_pipeline_alpha =
            Some(make_lit_draw(alpha, "particle_sprite_lit_alpha"));
        self.particle.sprite_lit_pipeline_additive =
            Some(make_lit_draw(additive, "particle_sprite_lit_additive"));
        self.particle.sprite_lit_pipeline_premultiplied =
            Some(make_lit_draw(premul, "particle_sprite_lit_premultiplied"));

        let lit_fallback_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_particle_lit_fallback_bg"),
            layout: &lit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_normal_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
            ],
        });

        self.particle.sprite_lit_bgl = Some(lit_bgl);
        self.particle.sprite_lit_fallback_bg = Some(lit_fallback_bg);

        // Mesh-route draw pipelines. Same blend variants as the sprite route,
        // but the vertex stage consumes the mesh's standard `Vertex` layout
        // on slot 0 and composes the per-instance transform inline from the
        // bound particle storage buffer.
        let mesh_draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_particle_mesh_draw_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let mesh_shader = crate::resources::builders::wgsl_module(
            device,
            "particle_mesh_shader",
            crate::resources::builders::wgsl_source!("particle_mesh"),
        );

        let mesh_layout = crate::resources::builders::standard_scene_layout(
            device,
            "particle_mesh_draw_layout",
            &self.camera_bind_group_layout,
            &mesh_draw_bgl,
        );

        // Particle meshes are closed solids, so back-face culled; still no depth
        // write since particles draw transparently after the opaque pass.
        let make_mesh_draw = |blend: wgpu::BlendState, label: &str| {
            crate::resources::builders::build_dual_pipeline(
                device,
                &crate::resources::builders::DualPipelineDesc {
                    label,
                    layout: &mesh_layout,
                    shader: &mesh_shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &[crate::resources::types::Vertex::buffer_layout()],
                    blend: Some(blend),
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    depth_write: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        self.particle.mesh_pipeline_alpha = Some(make_mesh_draw(alpha, "particle_mesh_alpha"));
        self.particle.mesh_pipeline_additive =
            Some(make_mesh_draw(additive, "particle_mesh_additive"));
        self.particle.mesh_pipeline_premultiplied =
            Some(make_mesh_draw(premul, "particle_mesh_premultiplied"));
        self.particle.mesh_draw_bgl = Some(mesh_draw_bgl);

        self.particle.params_bgl = Some(params_bgl);
        self.particle.sim_bgl = Some(sim_bgl);
        self.particle.draw_bgl = Some(draw_bgl);
        self.particle.emit_pipeline = Some(emit_pipeline);
        self.particle.sim_pipeline = Some(sim_pipeline);
    }

    /// Run emit + sim compute passes for every particle system referenced this
    /// frame. Returns per-job draw metadata for the render phase.
    pub(crate) fn run_particle_jobs(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        items: &[crate::renderer::GpuParticleSystemItem],
        sink: &mut crate::renderer::SubmitSink,
    ) -> Vec<ParticleFrameData> {
        if items.is_empty() {
            return Vec::new();
        }
        self.ensure_particle_pipelines(device);

        let emit_pipeline = self
            .particle
            .emit_pipeline
            .as_ref()
            .expect("particle pipelines should exist after ensure")
            .clone();
        let sim_pipeline = self
            .particle
            .sim_pipeline
            .as_ref()
            .expect("particle pipelines should exist after ensure")
            .clone();
        let params_bgl = self
            .particle
            .params_bgl
            .as_ref()
            .expect("particle params BGL")
            .clone();

        let mut frame_data: Vec<ParticleFrameData> = Vec::with_capacity(items.len());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("particle_compute_encoder"),
        });

        for item in items {
            let idx = item.system_id.0;
            let (blend, route) = match self.particle.systems.get(idx).and_then(|s| s.as_ref()) {
                Some(s) if s.alive => match &s.render {
                    ParticleRender::Sprite { blend, lit, .. } => {
                        (*blend, ParticleDrawRoute::Sprite { lit: *lit })
                    }
                    ParticleRender::Mesh { blend, mesh_id, .. } => {
                        (*blend, ParticleDrawRoute::Mesh { mesh_id: *mesh_id })
                    }
                },
                _ => continue,
            };

            let hidden = item.settings.hidden;

            // Pull mutable state out, build dispatch state outside the borrow.
            let (
                capacity,
                particle_buf_binding,
                emit_counter_binding,
                sim_bg,
                spawn_count,
                frame_counter,
            ) = {
                let system = self.particle.systems[idx].as_mut().unwrap();
                let dt = item.time_step.max(0.0);
                system.spawn_accumulator += item.emitter.rate * dt;
                let spawn_count = system.spawn_accumulator.floor() as u32;
                system.spawn_accumulator -= spawn_count as f32;
                system.frame_counter = system.frame_counter.wrapping_add(1);
                let frame_counter = system.frame_counter;
                (
                    system.capacity,
                    system.particle_buf.as_entire_binding(),
                    system.emit_counter_buf.as_entire_binding(),
                    // Need to clone the bind group; wgpu allows this cheaply (Arc inside).
                    system.sim_bg.clone(),
                    spawn_count,
                    frame_counter,
                )
            };
            let _ = (particle_buf_binding, emit_counter_binding); // bound through sim_bg

            let workgroups = capacity.div_ceil(64);

            // ----- Emit -----
            if spawn_count > 0 {
                queue.write_buffer(
                    &self.particle.systems[idx]
                        .as_ref()
                        .unwrap()
                        .emit_counter_buf,
                    0,
                    bytemuck::bytes_of(&spawn_count),
                );

                let emit_params =
                    build_emit_params(&item.emitter, capacity, spawn_count, frame_counter);
                let emit_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("particle_emit_params"),
                    contents: bytemuck::bytes_of(&emit_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let emit_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("particle_emit_params_bg"),
                    layout: &params_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: emit_uniform.as_entire_binding(),
                    }],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("particle_emit_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&emit_pipeline);
                pass.set_bind_group(0, &emit_params_bg, &[]);
                pass.set_bind_group(1, &sim_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // ----- Sim -----
            let sim_params = build_sim_params(item.time_step, capacity, &item.forces);
            let sim_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particle_sim_params"),
                contents: bytemuck::bytes_of(&sim_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let sim_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_sim_params_bg"),
                layout: &params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_uniform.as_entire_binding(),
                }],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_sim_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&sim_pipeline);
            pass.set_bind_group(0, &sim_params_bg, &[]);
            pass.set_bind_group(1, &sim_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(pass);

            frame_data.push(ParticleFrameData {
                system_idx: idx,
                blend,
                hidden,
                route,
            });
        }

        sink.push(encoder.finish());
        frame_data
    }
}

fn build_emit_params(
    e: &crate::renderer::EmitterConfig,
    capacity: u32,
    spawn_count: u32,
    frame_counter: u32,
) -> EmitParamsGpu {
    use crate::renderer::{SpawnShape, VelocityDist};

    let mut out = EmitParamsGpu {
        spawn_min: [0.0; 3],
        spawn_kind: 0,
        spawn_max: [0.0; 3],
        spawn_radius: 0.0,
        vel_min: [0.0; 3],
        vel_kind: 0,
        vel_max: [0.0; 3],
        cone_half_angle: 0.0,
        vel_axis: [0.0; 3],
        cone_min_speed: 0.0,
        colour: e.colour,
        spawn_count,
        capacity,
        rng_seed: frame_counter.wrapping_mul(0x9E3779B1),
        size: e.size,
        lifetime_min: e.lifetime.0,
        lifetime_max: e.lifetime.1,
        cone_max_speed: 0.0,
        _pad: 0.0,
    };

    match e.spawn_shape {
        SpawnShape::Point(p) => {
            out.spawn_kind = 0;
            out.spawn_min = p;
        }
        SpawnShape::Box { min, max } => {
            out.spawn_kind = 1;
            out.spawn_min = min;
            out.spawn_max = max;
        }
        SpawnShape::Sphere { center, radius } => {
            out.spawn_kind = 2;
            out.spawn_min = center;
            out.spawn_radius = radius;
        }
    }

    match e.initial_velocity {
        VelocityDist::Fixed(v) => {
            out.vel_kind = 0;
            out.vel_min = v;
        }
        VelocityDist::UniformBox { min, max } => {
            out.vel_kind = 1;
            out.vel_min = min;
            out.vel_max = max;
        }
        VelocityDist::UniformCone {
            axis,
            half_angle,
            min_speed,
            max_speed,
        } => {
            out.vel_kind = 2;
            out.vel_axis = axis;
            out.cone_half_angle = half_angle;
            out.cone_min_speed = min_speed;
            out.cone_max_speed = max_speed;
        }
    }

    out
}

fn build_sim_params(
    dt: f32,
    capacity: u32,
    forces: &[crate::renderer::ForceField],
) -> SimParamsGpu {
    use crate::renderer::ForceField;

    let mut gpu_forces = [GpuForce {
        kind: 0,
        _pad: [0; 3],
        v0: [0.0; 4],
        v1: [0.0; 4],
    }; MAX_FORCES];

    let n = forces.len().min(MAX_FORCES);
    for (i, f) in forces.iter().take(n).enumerate() {
        match *f {
            ForceField::Gravity(a) => {
                gpu_forces[i].kind = 0;
                gpu_forces[i].v0 = [a[0], a[1], a[2], 0.0];
            }
            ForceField::Drag(k) => {
                gpu_forces[i].kind = 1;
                gpu_forces[i].v0 = [k, 0.0, 0.0, 0.0];
            }
            ForceField::PointAttractor {
                position,
                strength,
                falloff,
            } => {
                gpu_forces[i].kind = 2;
                gpu_forces[i].v0 = [position[0], position[1], position[2], strength];
                gpu_forces[i].v1 = [falloff, 0.0, 0.0, 0.0];
            }
        }
    }

    SimParamsGpu {
        dt,
        capacity,
        force_count: n as u32,
        _pad: 0,
        forces: gpu_forces,
    }
}
