//! The GPU particle system item type as an [`ItemTypePlugin`]: a persistent
//! particle buffer that lives on the GPU, advanced by an emit and a sim compute
//! pass each frame and drawn either as camera-facing billboards or as instances
//! of an uploaded mesh. Consumers create a system once with
//! [`GpuParticleSystems::create_gpu_particle_system`]
//! and submit a [`GpuParticleSystemItem`] each frame to advance and draw it.
//!
//! The compute work runs in `prepare` and is handed back as a command buffer,
//! so it is submitted before the frame's draws; the draw itself is an ordinary
//! `paint`, because particles depth-test against the opaque scene but never
//! write depth.

mod pipeline;
mod store;
mod types;
mod uploads;
pub use uploads::GpuParticleSystems;

use store::{
    EmitParamsGpu, EmitState, GpuParticle, ParticleDrawRoute, ParticleLayouts, ParticleStore,
    ParticleSystem, SimParamsGpu, build_emit_params, build_sim_params,
};
pub use types::GpuParticleSystemId;
use viewport_lib::gpu::util::DeviceExt;
use viewport_lib::plugin_api::{ItemCollections, ItemFrameContext, ItemTypePlugin, PaintContext};
use viewport_lib::renderer::SpriteBlend;

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.gpu_particles";

pub use types::{
    EmitterConfig, ForceField, GpuParticleSystemConfig, GpuParticleSystemItem, ParticleMeshAlign,
    ParticleRender, SpawnShape, VelocityDist,
};

/// This type's shaders as the pipelines compile them, shared sections already
/// spliced in front of each body.
pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
    use crate::item_types::shader::{lit_shader, scene_shader, wgsl_source};

    vec![
        (
            "particle_emit.wgsl",
            wgsl_source!("particle_emit").to_string(),
        ),
        (
            "particle_sim.wgsl",
            wgsl_source!("particle_sim").to_string(),
        ),
        (
            "particle_sprite.wgsl",
            scene_shader(&[], wgsl_source!("particle_sprite")),
        ),
        (
            "particle_sprite_lit.wgsl",
            lit_shader(&[], wgsl_source!("particle_sprite_lit")),
        ),
        (
            "particle_mesh.wgsl",
            scene_shader(&[], wgsl_source!("particle_mesh")),
        ),
    ]
}

impl viewport_lib::plugin_api::PluginItem for GpuParticleSystemItem {
    const TYPE_NAME: &'static str = TYPE_NAME;

    fn settings(&self) -> &viewport_lib::ItemSettings {
        &self.settings
    }
}

/// One system's draw state for this frame.
///
/// The bind groups and the capacity are cloned out of the store during
/// prepare rather than looked up at draw time: the paint hook gets no handle
/// to the resources, and the clones are cheap because wgpu's handles are
/// reference-counted.
struct ParticleFrame {
    blend: SpriteBlend,
    route: ParticleDrawRoute,
    capacity: u32,
    draw_bg: Option<viewport_lib::gpu::BindGroup>,
    draw_bg_mesh: Option<viewport_lib::gpu::BindGroup>,
    draw_lit_normal_bg: Option<viewport_lib::gpu::BindGroup>,
}

#[derive(Default)]
pub struct GpuParticlesPlugin {
    /// The live systems, owned by the type that simulates and draws them.
    systems: ParticleStore,
    /// The layouts every system's own bind groups are built over. Created on
    /// registration, because a system can be created before the first frame.
    layouts: Option<ParticleLayouts>,
    /// Resource epochs the systems' draw bind groups were last validated
    /// against. A system's draw bind group bakes a texture view in when the
    /// system is created, so a free or a replace since the last frame means
    /// some of them have to be rebuilt.
    deps_gate: viewport_lib::resources::ResourceGate,
    /// Draw bind groups rebuilt by revalidation since startup, for tests and
    /// diagnostics.
    draw_bg_rebuilds: u64,
    gpu: Option<pipeline::ParticleGpu>,
    frame: Vec<ParticleFrame>,
}

impl GpuParticlesPlugin {
    /// Allocate a persistent particle system and return its handle.
    ///
    /// The buffers and the per-system bind groups are built here rather than on
    /// the first frame that draws the system, because a host creates its
    /// systems at startup and the handle has to be usable straight away.
    pub(crate) fn create_system(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        resources: &viewport_lib::resources::DeviceResources,
        config: &GpuParticleSystemConfig,
    ) -> GpuParticleSystemId {
        // A create can beat `init_gpu` when a host registers and creates in the
        // same breath, so build the layouts here if registration has not.
        let layouts = self
            .layouts
            .get_or_insert_with(|| ParticleLayouts::new(device));
        {
            use viewport_lib::resources::TextureSlot;
            match &config.render {
                ParticleRender::Sprite {
                    texture_id,
                    normal_texture_id,
                    ..
                } => {
                    resources.check_texture_slot(*texture_id, TextureSlot::SpriteAlbedo);
                    resources.check_texture_slot(*normal_texture_id, TextureSlot::SpriteNormalMap);
                }
                ParticleRender::Mesh { texture_id, .. } => {
                    resources.check_texture_slot(*texture_id, TextureSlot::MeshInstanceAlbedo);
                }
            }
        }

        let capacity = config.capacity.max(1);

        // Persistent particle buffer, initialised to all-dead.
        let particle_bytes_len = (capacity as usize) * std::mem::size_of::<GpuParticle>();
        let zero_particles = vec![0u8; particle_bytes_len];
        let particle_buf =
            device.create_buffer_init(&viewport_lib::gpu::util::BufferInitDescriptor {
                label: Some("gpu_particle_buf"),
                contents: &zero_particles,
                usage: viewport_lib::gpu::BufferUsages::STORAGE
                    | viewport_lib::gpu::BufferUsages::VERTEX
                    | viewport_lib::gpu::BufferUsages::COPY_DST
                    | viewport_lib::gpu::BufferUsages::COPY_SRC,
            });

        let _ = queue; // queue currently unused; reserved for textures upload paths

        let sim_bgl = &layouts.sim_bgl;
        let sim_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("gpu_particle_sim_bg"),
            layout: sim_bgl,
            entries: &[viewport_lib::gpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            }],
        });

        // Persistent params uniforms + bind groups, rewritten per frame.
        let params_bgl = &layouts.params_bgl;
        let emit_params_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("gpu_particle_emit_params"),
            size: std::mem::size_of::<EmitParamsGpu>() as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sim_params_buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("gpu_particle_sim_params"),
            size: std::mem::size_of::<SimParamsGpu>() as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let emit_params_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("gpu_particle_emit_params_bg"),
            layout: params_bgl,
            entries: &[viewport_lib::gpu::BindGroupEntry {
                binding: 0,
                resource: emit_params_buf.as_entire_binding(),
            }],
        });
        let sim_params_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("gpu_particle_sim_params_bg"),
            layout: params_bgl,
            entries: &[viewport_lib::gpu::BindGroupEntry {
                binding: 0,
                resource: sim_params_buf.as_entire_binding(),
            }],
        });

        let bindings = store::build_particle_draw_bindings(
            device,
            resources,
            layouts,
            &config.render,
            &particle_buf,
        );

        let system = ParticleSystem {
            capacity,
            render: config.render.clone(),
            particle_buf,
            sim_bg,
            emit_params_buf,
            sim_params_buf,
            emit_params_bg,
            sim_params_bg,
            draw_bg: bindings.draw_bg,
            draw_bg_mesh: bindings.draw_bg_mesh,
            draw_lit_normal_bg: bindings.draw_lit_normal_bg,
            draw_uniform_buf: bindings.draw_uniform_buf,
            draw_textures: bindings.draw_textures,
            emit: std::sync::Mutex::new(EmitState::default()),
        };

        self.systems.insert_sized(system)
    }

    /// Rebuild the draw bind groups of every live system whose baked textures
    /// were freed or replaced since the last frame.
    ///
    /// A system is the host's content, held until the host drops the handle, so
    /// a freed texture rebinds the system against the neutral view rather than
    /// discarding it: the particles keep simulating and drawing, untextured.
    /// Leaving it alone instead would pin the freed texture's memory for the
    /// system's lifetime and go on sampling its contents.
    fn revalidate_draw_bindings(
        &mut self,
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::resources::DeviceResources,
    ) {
        use viewport_lib::resources::Revalidate;
        let verdict = self.deps_gate.poll(resources);
        if verdict == Revalidate::Valid {
            return;
        }
        // No layouts means nothing was ever created, so there is nothing to
        // rebuild: a system builds them on its way into the store.
        let Some(layouts) = self.layouts.as_ref() else {
            return;
        };
        let stale: Vec<GpuParticleSystemId> = self
            .systems
            .iter()
            .filter(|(_, s)| {
                verdict == Revalidate::RebuildAll
                    || !store::textures_resident(&s.draw_textures, resources)
            })
            .map(|(id, _)| id)
            .collect();
        for id in stale {
            let render = self
                .systems
                .get(id)
                .expect("stale handle came from a live slot")
                .render
                .clone();
            // The buffer handle is cloned rather than borrowed: wgpu handles
            // are reference-counted, and holding a borrow into the store would
            // block the `get_mut` that writes the rebuilt bindings back.
            let buf = self
                .systems
                .get(id)
                .expect("stale handle came from a live slot")
                .particle_buf
                .clone();
            let bindings =
                store::build_particle_draw_bindings(device, resources, layouts, &render, &buf);
            let system = self
                .systems
                .get_mut(id)
                .expect("stale handle came from a live slot");
            system.draw_bg = bindings.draw_bg;
            system.draw_bg_mesh = bindings.draw_bg_mesh;
            system.draw_lit_normal_bg = bindings.draw_lit_normal_bg;
            system.draw_uniform_buf = bindings.draw_uniform_buf;
            system.draw_textures = bindings.draw_textures;
            self.draw_bg_rebuilds += 1;
        }
    }

    /// Release a system. The handle stops resolving and the slot is reused by
    /// the next create.
    pub(crate) fn drop_system(&mut self, id: GpuParticleSystemId) {
        self.systems.remove(id);
    }

    /// Borrow a live system, or `None` when the handle does not resolve.
    #[cfg(test)]
    pub(crate) fn system(&self, id: GpuParticleSystemId) -> Option<&ParticleSystem> {
        self.systems.get(id)
    }
}

impl ItemTypePlugin for GpuParticlesPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    /// Build the layouts at registration rather than on the first frame that
    /// draws a system: `create_gpu_particle_system` builds a system's
    /// persistent bind groups immediately, and a host creates its systems at
    /// startup, long before any frame.
    fn init_gpu(
        &mut self,
        device: &viewport_lib::gpu::Device,
        _shared: &viewport_lib::plugin_api::SharedBindings<'_>,
    ) {
        self.layouts = Some(ParticleLayouts::new(device));
    }

    /// What the live systems hold: their particle buffers and uniforms. This is
    /// most of a particle-heavy scene's working set, and it was invisible to
    /// the renderer's byte accounting while the store sat in core.
    fn resident_bytes(&self) -> u64 {
        self.systems.allocated_bytes()
    }

    /// Builds the compute pipelines and asks for every HDR draw pipeline; the
    /// type does not draw into the LDR pass.
    fn warm(
        &mut self,
        device: &viewport_lib::gpu::Device,
        resources: &viewport_lib::DeviceResources,
    ) {
        let layouts = self
            .layouts
            .get_or_insert_with(|| ParticleLayouts::new(device));
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::ParticleGpu::new(device, resources, layouts));
        for route in [pipeline::SPRITE, pipeline::SPRITE_LIT, pipeline::MESH] {
            for blend in [
                viewport_lib::renderer::SpriteBlend::AlphaBlend,
                viewport_lib::renderer::SpriteBlend::Additive,
                viewport_lib::renderer::SpriteBlend::Premultiplied,
            ] {
                gpu.pipelines.get(pipeline::draw_index(route, blend, true));
            }
        }
    }

    fn on_device_recreated(
        &mut self,
        device: &viewport_lib::gpu::Device,
        _queue: &viewport_lib::gpu::Queue,
    ) {
        self.layouts = Some(ParticleLayouts::new(device));
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ItemCollections<'_>,
    ) -> Vec<viewport_lib::gpu::CommandBuffer> {
        self.frame.clear();
        self.revalidate_draw_bindings(device, ctx.resources);
        let items = items.of::<GpuParticleSystemItem>();
        if items.is_empty() {
            return Vec::new();
        }
        let layouts = self
            .layouts
            .get_or_insert_with(|| ParticleLayouts::new(device));
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::ParticleGpu::new(device, ctx.resources, layouts));

        // Stage every system's uniform writes first, then encode all the
        // dispatches into one compute pass. Params buffers and bind groups are
        // persistent per system, so the per-frame device work is two
        // `write_buffer` calls and the dispatch encoding.
        struct Staged {
            workgroups: u32,
            spawn: bool,
            sim_bg: viewport_lib::gpu::BindGroup,
            emit_params_bg: viewport_lib::gpu::BindGroup,
            sim_params_bg: viewport_lib::gpu::BindGroup,
        }
        let mut staged: Vec<Staged> = Vec::with_capacity(items.len());

        for item in items {
            let Some(system) = self.systems.get(item.system_id) else {
                continue;
            };
            let (blend, route) = match &system.render {
                ParticleRender::Sprite { blend, lit, .. } => {
                    (*blend, ParticleDrawRoute::Sprite { lit: *lit })
                }
                ParticleRender::Mesh { blend, mesh_id, .. } => {
                    (*blend, ParticleDrawRoute::Mesh { mesh_id: *mesh_id })
                }
            };

            let capacity = system.capacity;
            let dt = item.time_step.max(0.0);
            let spawn_count = {
                let mut emit = system.emit.lock().unwrap();
                emit.spawn_accumulator += item.emitter.rate * dt;
                let count = emit.spawn_accumulator.floor() as u32;
                emit.spawn_accumulator -= count as f32;
                emit.frame_counter = emit.frame_counter.wrapping_add(1);
                if count > 0 {
                    let params = build_emit_params(
                        &item.emitter,
                        capacity,
                        count,
                        emit.frame_counter,
                        emit.emit_cursor,
                    );
                    queue.write_buffer(&system.emit_params_buf, 0, bytemuck::bytes_of(&params));
                    // Next frame starts where this one stopped, so the buffer
                    // is recycled in order.
                    if capacity > 0 {
                        emit.emit_cursor = (emit.emit_cursor + count) % capacity;
                    }
                }
                count
            };
            let sim_params = build_sim_params(item.time_step, capacity, &item.forces);
            queue.write_buffer(&system.sim_params_buf, 0, bytemuck::bytes_of(&sim_params));

            staged.push(Staged {
                workgroups: capacity.div_ceil(64),
                spawn: spawn_count > 0,
                // Bind group clones are cheap (Arc inside).
                sim_bg: system.sim_bg.clone(),
                emit_params_bg: system.emit_params_bg.clone(),
                sim_params_bg: system.sim_params_bg.clone(),
            });
            // A hidden system still advances: hiding a particle effect should
            // not freeze it mid-flight and resume it later.
            if !item.settings.hidden {
                self.frame.push(ParticleFrame {
                    blend,
                    route,
                    capacity,
                    draw_bg: system.draw_bg.clone(),
                    draw_bg_mesh: system.draw_bg_mesh.clone(),
                    draw_lit_normal_bg: system.draw_lit_normal_bg.clone(),
                });
            }
        }

        if staged.is_empty() {
            return Vec::new();
        }

        let mut encoder =
            device.create_command_encoder(&viewport_lib::gpu::CommandEncoderDescriptor {
                label: Some("particle_compute_encoder"),
            });
        {
            // One pass for every system. All emits run first, then all sims;
            // dispatches within a pass are ordered, so each system's emit still
            // precedes its sim.
            let mut pass = encoder.begin_compute_pass(&viewport_lib::gpu::ComputePassDescriptor {
                label: Some("particle_compute_pass"),
                timestamp_writes: None,
            });
            if staged.iter().any(|s| s.spawn) {
                pass.set_pipeline(&gpu.emit_pipeline);
                for s in staged.iter().filter(|s| s.spawn) {
                    pass.set_bind_group(0, &s.emit_params_bg, &[]);
                    pass.set_bind_group(1, &s.sim_bg, &[]);
                    pass.dispatch_workgroups(s.workgroups, 1, 1);
                }
            }
            pass.set_pipeline(&gpu.sim_pipeline);
            for s in &staged {
                pass.set_bind_group(0, &s.sim_params_bg, &[]);
                pass.set_bind_group(1, &s.sim_bg, &[]);
                pass.dispatch_workgroups(s.workgroups, 1, 1);
            }
        }
        vec![encoder.finish()]
    }

    fn paint(
        &self,
        pass: &mut viewport_lib::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &ItemCollections<'_>,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if self.frame.is_empty() {
            return;
        }
        let hdr = ctx.target_format == viewport_lib::resources::HDR_COLOR_FORMAT;
        // Each system draws its full capacity; a dead particle emits a
        // degenerate vertex and contributes no fragments.
        for pd in &self.frame {
            match pd.route {
                ParticleDrawRoute::Sprite { lit } => {
                    let Some(draw_bg) = pd.draw_bg.as_ref() else {
                        continue;
                    };
                    let route = if lit {
                        pipeline::SPRITE_LIT
                    } else {
                        pipeline::SPRITE
                    };
                    // Still compiling: this system draws next frame.
                    let Some(pl) = gpu
                        .pipelines
                        .get(pipeline::draw_index(route, pd.blend, hdr))
                    else {
                        continue;
                    };
                    pass.set_pipeline(pl);
                    pass.set_bind_group(1, draw_bg, &[]);
                    if lit {
                        let normal_bg = pd
                            .draw_lit_normal_bg
                            .as_ref()
                            .unwrap_or(&gpu.sprite_lit_fallback_bg);
                        pass.set_bind_group(2, normal_bg, &[]);
                    }
                    pass.draw(0..6, 0..pd.capacity);
                }
                ParticleDrawRoute::Mesh { mesh_id } => {
                    let Some(draw_bg) = pd.draw_bg_mesh.as_ref() else {
                        continue;
                    };
                    let Some(pl) =
                        gpu.pipelines
                            .get(pipeline::draw_index(pipeline::MESH, pd.blend, hdr))
                    else {
                        continue;
                    };
                    pass.set_pipeline(pl);
                    pass.set_bind_group(1, draw_bg, &[]);
                    ctx.meshes
                        .draw_indexed_instanced(pass, mesh_id, pd.capacity);
                }
            }
        }
    }
}
#[cfg(test)]
mod emission_tests {
    use super::*;
    use crate::item_types::gpu_particles::GpuParticleSystems;
    use viewport_lib::renderer::ViewportRenderer;

    /// A renderer with this crate's item types registered, which is the only
    /// way in: creating a system and running a frame are both consumer calls.
    fn renderer() -> Option<(
        viewport_lib::gpu::Device,
        viewport_lib::gpu::Queue,
        ViewportRenderer,
    )> {
        let (device, queue) = viewport_lib_testkit::headless_device_with(
            &viewport_lib_testkit::DeviceProfile::low_power("gpu_particles"),
        )?;
        let mut renderer =
            ViewportRenderer::new(&device, viewport_lib::gpu::TextureFormat::Rgba8UnormSrgb);
        crate::item_types::install(&mut renderer, &device);
        Some((device, queue, renderer))
    }

    /// The plugin behind a renderer, for the assertions that read its store.
    fn plugin_of(renderer: &ViewportRenderer) -> &GpuParticlesPlugin {
        renderer
            .item_type_plugin::<GpuParticlesPlugin>(TYPE_NAME)
            .expect("registered by install()")
    }

    /// Read every slot's lifetime back off the GPU.
    fn lifetimes(
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        plugin: &GpuParticlesPlugin,
        id: GpuParticleSystemId,
    ) -> Vec<f32> {
        let system = plugin.system(id).expect("live system");
        let size = (system.capacity as u64) * std::mem::size_of::<GpuParticle>() as u64;
        let staging = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("particle_readback"),
            size,
            usage: viewport_lib::gpu::BufferUsages::MAP_READ
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&viewport_lib::gpu::CommandEncoderDescriptor {
                label: Some("particle_readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(&system.particle_buf, 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(viewport_lib::gpu::MapMode::Read, |_| {});
        let _ = device.poll(viewport_lib::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let out = {
            let data = viewport_lib::gpu::mapped_range(slice);
            bytemuck::cast_slice::<u8, GpuParticle>(&data)
                .iter()
                .map(|p| p.lifetime)
                .collect()
        };
        staging.unmap();
        out
    }

    fn steady_emitter(rate: f32) -> super::types::EmitterConfig {
        let mut e = super::types::EmitterConfig::default();
        e.rate = rate;
        // A single lifetime, so "how many are alive" is a function of how many
        // were emitted rather than of the lifetime draw.
        e.lifetime = (100.0, 100.0);
        e
    }

    /// Drive the item type's own prepare, which is where emission lives, and
    /// submit the compute work it hands back.
    fn run_frames(
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        renderer: &mut ViewportRenderer,
        id: GpuParticleSystemId,
        rate: f32,
        dt: f32,
        frames: usize,
    ) {
        for _ in 0..frames {
            let mut item = GpuParticleSystemItem::new(id, dt);
            item.emitter = steady_emitter(rate);
            let mut frame = viewport_lib::FrameData::default();
            frame.camera.viewport_size = [64.0, 64.0];
            frame.viewport.show_grid = false;
            frame.viewport.show_axes_indicator = false;
            frame.scene.items_mut::<GpuParticleSystemItem>().push(item);
            let _ = renderer.pass().prepare(device, queue, &frame);
        }
    }

    /// Every frame must emit exactly the budget the configured rate asks for
    /// while the system has slots to spare. The emit kernel picks slots by a
    /// wrapping window rather than by racing, so this is the check that the
    /// window does not quietly skip spawns.
    #[test]
    fn emission_matches_the_configured_rate() {
        let Some((device, queue, mut renderer)) = renderer() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 256;
        let id = renderer.create_gpu_particle_system(&device, &queue, &config);

        // 10 per frame for 5 frames, well inside a 256-slot buffer.
        run_frames(&device, &queue, &mut renderer, id, 10.0, 1.0, 5);
        let live = lifetimes(&device, &queue, plugin_of(&renderer), id)
            .iter()
            .filter(|l| **l > 0.0)
            .count();
        assert_eq!(live, 50, "5 frames at 10 per frame should leave 50 alive");
    }

    /// The window wraps past the end of the buffer without losing a frame's
    /// spawns: 30 frames of 10 against 256 slots crosses the end once.
    #[test]
    fn emission_survives_the_window_wrapping() {
        let Some((device, queue, mut renderer)) = renderer() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 256;
        let id = renderer.create_gpu_particle_system(&device, &queue, &config);

        run_frames(&device, &queue, &mut renderer, id, 10.0, 1.0, 30);
        let live = lifetimes(&device, &queue, plugin_of(&renderer), id)
            .iter()
            .filter(|l| **l > 0.0)
            .count();
        // 300 spawns into 256 slots with nothing dying: the buffer fills and
        // the rest land on slots that are still alive, which are skipped.
        assert_eq!(live, 256, "the buffer should fill and stay full");
    }

    /// Two systems given identical configuration and identical frames must end
    /// up with identical particles. Selecting slots by racing them through an
    /// atomic made this fail: which slots won was down to GPU scheduling, and
    /// every particle attribute is seeded from its slot index.
    #[test]
    fn emission_is_reproducible() {
        let Some((device, queue, mut ra)) = renderer() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let Some((device_b, queue_b, mut rb)) = renderer() else {
            return;
        };
        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 512;
        let a = ra.create_gpu_particle_system(&device, &queue, &config);
        let b = rb.create_gpu_particle_system(&device_b, &queue_b, &config);

        run_frames(&device, &queue, &mut ra, a, 40.0, 0.5, 4);
        run_frames(&device_b, &queue_b, &mut rb, b, 40.0, 0.5, 4);

        let la = lifetimes(&device, &queue, plugin_of(&ra), a);
        let lb = lifetimes(&device_b, &queue_b, plugin_of(&rb), b);
        assert_eq!(la, lb, "identical input must produce identical particles");
        assert!(
            la.iter().any(|l| *l > 0.0),
            "the test is vacuous if nothing was emitted"
        );
    }

    /// Dropping a system frees its slot for the next create. The dropped
    /// handle must not resolve to whatever lands in that slot afterwards:
    /// before the handle carried a generation, it did, and a stale id silently
    /// drove a different system's simulation.
    #[test]
    fn a_dropped_systems_handle_does_not_alias_the_slots_next_occupant() {
        let Some((device, queue, mut renderer)) = renderer() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let config = GpuParticleSystemConfig::default();

        let first = renderer.create_gpu_particle_system(&device, &queue, &config);
        renderer.drop_gpu_particle_system(first);
        let second = renderer.create_gpu_particle_system(&device, &queue, &config);

        assert_ne!(first, second, "the reused slot must carry a new generation");
        let plugin = plugin_of(&renderer);
        assert!(
            plugin.system(first).is_none(),
            "the dropped handle must not resolve"
        );
        assert!(
            plugin.system(second).is_some(),
            "the live handle must resolve"
        );
    }
}

#[cfg(test)]
mod revalidation_tests {
    use super::*;
    use crate::item_types::gpu_particles::GpuParticleSystems;
    use viewport_lib::renderer::ViewportRenderer;

    fn renderer() -> Option<(
        viewport_lib::gpu::Device,
        viewport_lib::gpu::Queue,
        ViewportRenderer,
    )> {
        let (device, queue) = viewport_lib_testkit::headless_device_with(
            &viewport_lib_testkit::DeviceProfile::low_power("gpu_particles"),
        )?;
        let mut renderer =
            ViewportRenderer::new(&device, viewport_lib::gpu::TextureFormat::Rgba8UnormSrgb);
        crate::item_types::install(&mut renderer, &device);
        Some((device, queue, renderer))
    }

    /// Run one revalidation pass and report the rebuild counter afterwards.
    /// The plugin and the resources it checks against come from one host
    /// borrow, which is how a plugin reaches both at once.
    fn revalidate(renderer: &mut ViewportRenderer, device: &viewport_lib::gpu::Device) -> u64 {
        let host = renderer
            .item_type_plugin_host::<GpuParticlesPlugin>(TYPE_NAME)
            .expect("registered by install()");
        host.plugin.revalidate_draw_bindings(device, host.resources);
        host.plugin.draw_bg_rebuilds
    }

    fn srgb_texture(px: u8) -> viewport_lib::resources::TextureData {
        viewport_lib::resources::TextureData::srgb(4, 4, vec![px; 4 * 4 * 4])
    }

    fn sprite_config(
        texture_id: Option<viewport_lib::resources::TextureId>,
    ) -> GpuParticleSystemConfig {
        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 16;
        if let ParticleRender::Sprite {
            texture_id: slot, ..
        } = &mut config.render
        {
            *slot = texture_id;
        }
        config
    }

    /// A freed texture must not stay baked into a system's draw bind group:
    /// the revalidation rebuilds it against the fallback view.
    #[test]
    fn freed_texture_rebuilds_draw_bind_group() {
        let Some((device, queue, mut renderer)) = renderer() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let tex = renderer
            .resources_mut()
            .upload_texture(&device, &queue, srgb_texture(200))
            .expect("texture upload");
        let _system =
            renderer.create_gpu_particle_system(&device, &queue, &sprite_config(Some(tex)));

        // Sync the gate so the assertion below isolates the free.
        let baseline = revalidate(&mut renderer, &device);

        renderer.resources_mut().free_texture(tex);
        assert_eq!(
            revalidate(&mut renderer, &device),
            baseline + 1,
            "free of a baked texture must rebuild the system's draw bind group"
        );

        // Nothing further changed: the next poll is a no-op.
        assert_eq!(revalidate(&mut renderer, &device), baseline + 1);
    }

    /// A replace swaps the view behind a live id, which no per-entry check can
    /// see, so it must rebuild unconditionally.
    #[test]
    fn replaced_texture_rebuilds_draw_bind_group() {
        let Some((device, queue, mut renderer)) = renderer() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let tex = renderer
            .resources_mut()
            .upload_texture(&device, &queue, srgb_texture(40))
            .expect("texture upload");
        let _system =
            renderer.create_gpu_particle_system(&device, &queue, &sprite_config(Some(tex)));

        let baseline = revalidate(&mut renderer, &device);

        renderer
            .resources_mut()
            .replace_texture(&device, &queue, tex, srgb_texture(220))
            .expect("texture replace");
        assert_eq!(
            revalidate(&mut renderer, &device),
            baseline + 1,
            "replace must rebuild every live system's draw bind group"
        );
    }

    /// A system with no texture never rebuilds on someone else's free.
    #[test]
    fn untextured_system_survives_unrelated_free() {
        let Some((device, queue, mut renderer)) = renderer() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let unrelated = renderer
            .resources_mut()
            .upload_texture(&device, &queue, srgb_texture(10))
            .expect("texture upload");
        let _system = renderer.create_gpu_particle_system(&device, &queue, &sprite_config(None));

        let baseline = revalidate(&mut renderer, &device);

        renderer.resources_mut().free_texture(unrelated);
        assert_eq!(
            revalidate(&mut renderer, &device),
            baseline,
            "a free the system does not name must not rebuild its bind groups"
        );
    }

    /// The systems a plugin holds are part of the renderer's working-set
    /// figure, which they were not while the store sat in core.
    #[test]
    fn live_systems_count_toward_resident_bytes() {
        let Some((device, queue, mut renderer)) = renderer() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let held_now = |r: &ViewportRenderer| {
            r.item_type_plugin::<GpuParticlesPlugin>(TYPE_NAME)
                .expect("registered by install()")
                .resident_bytes()
        };
        assert_eq!(held_now(&renderer), 0);

        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 4096;
        let id = renderer.create_gpu_particle_system(&device, &queue, &config);
        assert!(
            held_now(&renderer) >= 4096 * std::mem::size_of::<GpuParticle>() as u64,
            "the particle buffer is the bulk of what a system holds"
        );

        renderer.drop_gpu_particle_system(id);
        assert_eq!(held_now(&renderer), 0, "dropping gives the bytes back");
    }
}
