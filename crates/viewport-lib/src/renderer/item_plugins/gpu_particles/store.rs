//! The particle systems this item type holds on the consumer's behalf, and the
//! per-system GPU state every particle draw is built from.
//!
//! A system owns a persistent GPU buffer holding `capacity` particles. Each
//! particle stores its world-space position, velocity, lifetime remaining,
//! starting lifetime, colour, and size. A `GpuParticleSystemItem` names a
//! system by [`GpuParticleSystemId`] rather than carrying particles, so the
//! buffers are allocated once and simulated and drawn every frame from here.
//! The entry points are the `*_gpu_particle_system` methods on
//! [`ViewportRenderer`](crate::renderer::ViewportRenderer), which find this
//! plugin by name and call through to the methods below.
//!
//! Dead particles are not compacted. The emit shader scans for slots with
//! `lifetime <= 0` and reuses them; the draw shader emits a degenerate clip
//! position for dead slots so they cost nothing in the rasteriser. Compaction
//! would require a prefix sum each frame and is not worth the cost at the
//! particle counts the API targets (1k - 200k).
//!
//! The bind group layouts live here rather than with the pipelines, because a
//! system builds its persistent bind groups when it is created and a system can
//! be created long before the first frame that draws one.

use crate::gpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use crate::renderer::{ParticleMeshAlign, SpriteSizeMode};
use crate::resources::DeviceResources;

pub(crate) use super::types::{GpuParticleSystemId, ParticleRender};

/// The bind group layouts a particle system's own bind groups are built over.
///
/// Built once at startup rather than lazily: `create_gpu_particle_system` is
/// public and builds a system's persistent bind groups immediately, which can
/// happen long before the first frame that draws one. The pipelines that
/// consume these layouts live with the item type under
/// `renderer/item_plugins/gpu_particles/`.
pub(crate) struct ParticleLayouts {
    /// Layout for emit/sim params (group 0).
    pub(crate) params_bgl: crate::gpu::BindGroupLayout,
    /// Layout for the emit + sim compute pipelines (group 1).
    pub(crate) sim_bgl: crate::gpu::BindGroupLayout,
    /// Layout for the particle-sprite draw pipeline (group 1).
    pub(crate) draw_bgl: crate::gpu::BindGroupLayout,
    /// Group 2 layout for the lit particle path: optional normal map + sampler.
    pub(crate) sprite_lit_bgl: crate::gpu::BindGroupLayout,
    /// Layout for the mesh-route particle draw pipeline (group 1).
    pub(crate) mesh_draw_bgl: crate::gpu::BindGroupLayout,
}

impl ParticleLayouts {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        // Group 0: emit/sim params (uniform).
        let params_bgl = crate::resources::builders::uniform_bgl(
            device,
            "gpu_particle_params_bgl",
            crate::gpu::ShaderStages::COMPUTE,
        );

        // Group 1 (sim/emit): the particle buffer.
        let sim_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gpu_particle_sim_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::COMPUTE,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 1 (draw): sprite uniform + texture + sampler + particle buffer.
        let draw_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gpu_particle_draw_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let lit_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "gpu_particle_lit_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let mesh_draw_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("gpu_particle_mesh_draw_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::VERTEX
                            | crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Sampler(
                            crate::gpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: crate::gpu::ShaderStages::VERTEX,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        Self {
            params_bgl,
            sim_bgl,
            draw_bgl,
            sprite_lit_bgl: lit_bgl,
            mesh_draw_bgl,
        }
    }
}

/// Per-frame emission bookkeeping for one system.
#[derive(Default)]
pub(crate) struct EmitState {
    /// Frame count since creation; seeds the emit RNG so a freshly created
    /// system gets a different sequence from one that has been running.
    pub frame_counter: u32,
    /// First slot the next frame's spawn window covers. Advanced by that
    /// frame's spawn count and wrapped, so successive frames recycle the
    /// buffer in order.
    pub emit_cursor: u32,
    /// Fractional spawn accumulator. `rate * dt` rarely lands on an integer
    /// per frame; the remainder rolls over so the long-term average emission
    /// matches the configured rate.
    pub spawn_accumulator: f32,
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
    /// First slot this frame's spawn window covers. The window wraps, so a
    /// thread spawns when `(tid + capacity - emit_cursor) % capacity` is below
    /// `spawn_count`.
    pub emit_cursor: u32,
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

/// The draw-side bind groups for one particle system, plus the uniform
/// buffer behind them and the texture deps they bake in. Exactly one of
/// `draw_bg` and `draw_bg_mesh` is populated, per the system's render route.
#[derive(Default)]
pub(super) struct ParticleDrawBindings {
    pub(super) draw_bg: Option<crate::gpu::BindGroup>,
    pub(super) draw_bg_mesh: Option<crate::gpu::BindGroup>,
    pub(super) draw_lit_normal_bg: Option<crate::gpu::BindGroup>,
    pub(super) draw_uniform_buf: Option<crate::gpu::Buffer>,
    pub(super) draw_deps: crate::resources::resource_deps::ResourceDeps,
}

/// Per-system persistent GPU state.
pub(crate) struct ParticleSystem {
    pub capacity: u32,
    pub render: ParticleRender,
    /// `capacity` particles in `GpuParticle` layout. STORAGE + VERTEX usage,
    /// plus COPY_SRC so tests can read the live set back;
    /// bound through `sim_bg` and the draw bind groups, which keep it alive.
    pub particle_buf: crate::gpu::Buffer,
    /// Single atomic u32 counter rewritten by the host before each emit
    /// dispatch and decremented by emit threads as they claim slots. Reused
    /// across frames; nothing is preserved between dispatches.
    /// Bind group for the sim/emit compute pipelines (group 1).
    pub sim_bg: crate::gpu::BindGroup,
    /// Persistent uniform rewritten via `write_buffer` before each emit
    /// dispatch. Creating fresh buffers and bind groups per frame costs
    /// tens of microseconds of CPU per system; these are allocated once.
    pub emit_params_buf: crate::gpu::Buffer,
    /// Persistent uniform rewritten via `write_buffer` before each sim
    /// dispatch.
    pub sim_params_buf: crate::gpu::Buffer,
    /// Group 0 bind group over `emit_params_buf`.
    pub emit_params_bg: crate::gpu::BindGroup,
    /// Group 0 bind group over `sim_params_buf`.
    pub sim_params_bg: crate::gpu::BindGroup,
    /// Bind group for the sprite draw pipeline (group 1). `None` when the
    /// system's render route is not `Sprite`.
    pub draw_bg: Option<crate::gpu::BindGroup>,
    /// Bind group for the mesh draw pipeline (group 1). `None` when the
    /// system's render route is not `Mesh`.
    pub draw_bg_mesh: Option<crate::gpu::BindGroup>,
    /// Group 2 bind group for the lit draw pipeline (normal map + sampler).
    /// `None` when the system's render route is not lit.
    pub draw_lit_normal_bg: Option<crate::gpu::BindGroup>,
    /// Uniform buffers backing whichever draw bind group is populated.
    pub draw_uniform_buf: Option<crate::gpu::Buffer>,
    /// The texture ids baked into the draw bind groups, revalidated when the
    /// resource epochs move so a freed texture is neither pinned nor sampled.
    pub draw_deps: crate::resources::resource_deps::ResourceDeps,
    /// Per-frame emission bookkeeping, behind a lock because the item type
    /// advances it from a shared borrow of the resources.
    pub emit: std::sync::Mutex<EmitState>,
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

pub(crate) fn build_emit_params(
    e: &crate::renderer::EmitterConfig,
    capacity: u32,
    spawn_count: u32,
    frame_counter: u32,
    emit_cursor: u32,
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
        colour: e.colour.to_linear_rgba(),
        spawn_count,
        capacity,
        rng_seed: frame_counter.wrapping_mul(0x9E3779B1),
        emit_cursor,
        size: e.size,
        lifetime_min: e.lifetime.0,
        lifetime_max: e.lifetime.1,
        cone_max_speed: 0.0,
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

pub(crate) fn build_sim_params(
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

/// Build the per-route draw bind groups for a particle system: the sprite
/// or mesh group-1 bind group, the lit normal-map group when the route is
/// lit, the uniform buffer behind them, and the `ResourceDeps` naming the
/// textures they bake in. Called when a system is created, and again by the
/// plugin's revalidation whenever a named texture is freed or replaced.
///
/// A texture id that does not resolve binds the neutral fallback view and
/// clears the shader's `has_texture` flag, so a stale handle behaves as an
/// empty slot rather than sampling whatever occupies the storage now.
pub(super) fn build_particle_draw_bindings(
    device: &crate::gpu::Device,
    resources: &DeviceResources,
    layouts: &ParticleLayouts,
    render: &ParticleRender,
    particle_buf: &crate::gpu::Buffer,
) -> ParticleDrawBindings {
    let mut out = ParticleDrawBindings::default();
    match render {
        ParticleRender::Sprite {
            texture_id,
            blend: _,
            size_mode,
            depth_write: _,
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
            let texture_live = texture_id.is_some_and(|id| resources.has_texture(id));
            let normal_live = normal_texture_id.is_some_and(|id| resources.has_texture(id));
            let uniform = SpriteDrawUniform {
                model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                world_space: matches!(size_mode, SpriteSizeMode::WorldSpace) as u32,
                has_texture: texture_live as u32,
                normal_mode: normal_mode_u32,
                has_normal_map: normal_live as u32,
                ambient_scale: lit_params.ambient_scale,
                roughness: lit_params.roughness,
                _pad0: 0,
                _pad1: 0,
            };
            let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("gpu_particle_sprite_draw_uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            });
            let texture_view = if texture_live {
                resources
                    .texture_view(texture_id.unwrap())
                    .expect("checked live just above")
            } else {
                resources.fallback_colourmap_view()
            };
            let draw_bgl = &layouts.draw_bgl;
            out.draw_bg = Some(device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("gpu_particle_draw_bg"),
                layout: draw_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureView(texture_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(
                            resources.material_sampler(),
                        ),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: particle_buf.as_entire_binding(),
                    },
                ],
            }));
            if *lit {
                let lit_bgl = &layouts.sprite_lit_bgl;
                let normal_view = if normal_live {
                    resources
                        .texture_view(normal_texture_id.unwrap())
                        .expect("checked live just above")
                } else {
                    resources.fallback_texture_view(crate::scene::material::TextureSlot::Normal)
                };
                out.draw_lit_normal_bg =
                    Some(device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("gpu_particle_lit_normal_bg"),
                        layout: lit_bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::gpu::BindingResource::TextureView(normal_view),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: crate::gpu::BindingResource::Sampler(
                                    resources.material_sampler(),
                                ),
                            },
                        ],
                    }));
            }
            out.draw_uniform_buf = Some(uniform_buf);
            out.draw_deps = crate::resources::resource_deps::ResourceDeps::textures([
                *texture_id,
                *normal_texture_id,
                None,
                None,
                None,
            ]);
        }
        ParticleRender::Mesh {
            texture_id,
            blend: _,
            align,
            mesh_id: _,
        } => {
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
            let texture_live = texture_id.is_some_and(|id| resources.has_texture(id));
            let uniform = MeshDrawUniform {
                align: align_u32,
                has_texture: texture_live as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("gpu_particle_mesh_draw_uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            });
            let texture_view = if texture_live {
                resources
                    .texture_view(texture_id.unwrap())
                    .expect("checked live just above")
            } else {
                resources.fallback_texture_view(crate::scene::material::TextureSlot::Albedo)
            };
            let mesh_bgl = &layouts.mesh_draw_bgl;
            out.draw_bg_mesh = Some(device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("gpu_particle_mesh_draw_bg"),
                layout: mesh_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::TextureView(texture_view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::Sampler(
                            resources.material_sampler(),
                        ),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: particle_buf.as_entire_binding(),
                    },
                ],
            }));
            out.draw_uniform_buf = Some(uniform_buf);
            out.draw_deps = crate::resources::resource_deps::ResourceDeps::textures([
                *texture_id,
                None,
                None,
                None,
                None,
            ]);
        }
    }
    out
}

/// Slotted store of live particle systems.
///
/// A system removed by `drop_gpu_particle_system` leaves an empty slot that a
/// later create reuses. Each slot carries a generation bumped on removal, and a
/// [`GpuParticleSystemId`] captures the generation it was issued against, so a
/// stale handle resolves to `None` rather than aliasing the system now in its
/// slot. That matters more here than elsewhere: the plugin keeps per-system
/// emission state, and without the generation a new system would inherit the
/// previous occupant's spawn cursor.
pub(super) type ParticleStore =
    crate::resources::handle::SlotStore<ParticleSystem, GpuParticleSystemId>;

impl crate::resources::handle::GpuByteSize for ParticleSystem {
    /// The particle buffer plus the two params uniforms and whichever draw
    /// uniform the route built. The bind groups themselves hold no storage of
    /// their own.
    fn gpu_bytes(&self) -> u64 {
        self.particle_buf.size()
            + self.emit_params_buf.size()
            + self.sim_params_buf.size()
            + self.draw_uniform_buf.as_ref().map_or(0, |buf| buf.size())
    }
}
