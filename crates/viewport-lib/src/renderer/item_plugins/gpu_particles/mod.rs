//! The GPU particle system item type as an [`ItemTypePlugin`]: a persistent
//! particle buffer that lives on the GPU, advanced by an emit and a sim compute
//! pass each frame and drawn either as camera-facing billboards or as instances
//! of an uploaded mesh. Consumers create a system once with
//! `create_gpu_particle_system` and submit a [`GpuParticleSystemItem`] on
//! `SceneFrame::gpu_particle_systems` each frame to advance and draw it.
//!
//! The compute work runs in `prepare` and is handed back as a command buffer,
//! so it is submitted before the frame's draws; the draw itself is an ordinary
//! `paint`, because particles depth-test against the opaque scene but never
//! write depth.

mod pipeline;
pub(crate) mod store;
pub(crate) mod types;

use crate::gpu::util::DeviceExt;
use crate::plugin_api::{ItemFrameContext, ItemTypePlugin, PaintContext, PluginItemCollection};
use crate::renderer::{GpuParticleSystemItem, SpriteBlend};
use store::{
    EmitParamsGpu, EmitState, GpuParticle, ParticleDrawRoute, ParticleLayouts, ParticleStore,
    ParticleSystem, SimParamsGpu, build_emit_params, build_sim_params,
};
use types::{GpuParticleSystemConfig, GpuParticleSystemId, ParticleRender};

pub(crate) const TYPE_NAME: &str = "vpl.gpu_particles";

impl PluginItemCollection for Vec<GpuParticleSystemItem> {
    fn len(&self) -> usize {
        self.len()
    }
    fn item_settings(&self, index: usize) -> &crate::scene::material::ItemSettings {
        &self[index].settings
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
    draw_bg: Option<crate::gpu::BindGroup>,
    draw_bg_mesh: Option<crate::gpu::BindGroup>,
    draw_lit_normal_bg: Option<crate::gpu::BindGroup>,
}

#[derive(Default)]
pub(crate) struct GpuParticlesPlugin {
    /// The live systems, owned by the type that simulates and draws them.
    systems: ParticleStore,
    /// The layouts every system's own bind groups are built over. Created on
    /// registration, because a system can be created before the first frame.
    layouts: Option<ParticleLayouts>,
    /// Resource epochs the systems' draw bind groups were last validated
    /// against. A system's draw bind group bakes a texture view in when the
    /// system is created, so a free or a replace since the last frame means
    /// some of them have to be rebuilt.
    deps_gate: crate::resources::resource_deps::ResourceGate,
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
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &crate::resources::DeviceResources,
        config: &GpuParticleSystemConfig,
    ) -> GpuParticleSystemId {
        // A create can beat `init_gpu` when a host registers and creates in the
        // same breath, so build the layouts here if registration has not.
        let layouts = self
            .layouts
            .get_or_insert_with(|| ParticleLayouts::new(device));
        {
            use crate::resources::TextureSlot;
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
        let particle_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("gpu_particle_buf"),
            contents: &zero_particles,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::VERTEX
                | crate::gpu::BufferUsages::COPY_DST
                | crate::gpu::BufferUsages::COPY_SRC,
        });

        let _ = queue; // queue currently unused; reserved for textures upload paths

        let sim_bgl = &layouts.sim_bgl;
        let sim_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("gpu_particle_sim_bg"),
            layout: sim_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            }],
        });

        // Persistent params uniforms + bind groups, rewritten per frame.
        let params_bgl = &layouts.params_bgl;
        let emit_params_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("gpu_particle_emit_params"),
            size: std::mem::size_of::<EmitParamsGpu>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sim_params_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("gpu_particle_sim_params"),
            size: std::mem::size_of::<SimParamsGpu>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let emit_params_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("gpu_particle_emit_params_bg"),
            layout: params_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: emit_params_buf.as_entire_binding(),
            }],
        });
        let sim_params_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("gpu_particle_sim_params_bg"),
            layout: params_bgl,
            entries: &[crate::gpu::BindGroupEntry {
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
            draw_deps: bindings.draw_deps,
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
        device: &crate::gpu::Device,
        resources: &crate::resources::DeviceResources,
    ) {
        use crate::resources::resource_deps::Revalidate;
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
            .filter(|(_, s)| verdict == Revalidate::RebuildAll || !s.draw_deps.resolves(resources))
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
            system.draw_deps = bindings.draw_deps;
            self.draw_bg_rebuilds += 1;
        }
    }

    /// Release a system. The handle stops resolving and the slot is reused by
    /// the next create.
    pub(crate) fn drop_system(&mut self, id: GpuParticleSystemId) {
        self.systems.remove(id);
    }

    /// Borrow a live system, or `None` when the handle does not resolve.
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
        device: &crate::gpu::Device,
        _shared: &crate::plugin_api::SharedBindings<'_>,
    ) {
        self.layouts = Some(ParticleLayouts::new(device));
    }

    /// What the live systems hold: their particle buffers and uniforms. This is
    /// most of a particle-heavy scene's working set, and it was invisible to
    /// the renderer's byte accounting while the store sat in core.
    fn resident_bytes(&self) -> u64 {
        self.systems.allocated_bytes()
    }

    fn on_device_recreated(&mut self, device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
        self.layouts = Some(ParticleLayouts::new(device));
        self.gpu = None;
        self.frame.clear();
    }

    fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.frame.clear();
        self.revalidate_draw_bindings(device, ctx.resources);
        let items = items
            .as_any()
            .downcast_ref::<Vec<GpuParticleSystemItem>>()
            .expect("particle collection is the SceneFrame field");
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
            sim_bg: crate::gpu::BindGroup,
            emit_params_bg: crate::gpu::BindGroup,
            sim_params_bg: crate::gpu::BindGroup,
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

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("particle_compute_encoder"),
        });
        {
            // One pass for every system. All emits run first, then all sims;
            // dispatches within a pass are ordered, so each system's emit still
            // precedes its sim.
            let mut pass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
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
        pass: &mut crate::gpu::RenderPass<'_>,
        ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(gpu) = &self.gpu else { return };
        if self.frame.is_empty() {
            return;
        }
        let hdr = ctx.target_format == crate::resources::HDR_COLOR_FORMAT;
        // Each system draws its full capacity; a dead particle emits a
        // degenerate vertex and contributes no fragments.
        for pd in &self.frame {
            match pd.route {
                ParticleDrawRoute::Sprite { lit } => {
                    let dual = match (pd.blend, lit) {
                        (SpriteBlend::Additive, false) => &gpu.sprite_pipeline_additive,
                        (SpriteBlend::Premultiplied, false) => &gpu.sprite_pipeline_premultiplied,
                        (SpriteBlend::AlphaBlend, false) => &gpu.sprite_pipeline_alpha,
                        (SpriteBlend::Additive, true) => &gpu.sprite_lit_pipeline_additive,
                        (SpriteBlend::Premultiplied, true) => {
                            &gpu.sprite_lit_pipeline_premultiplied
                        }
                        (SpriteBlend::AlphaBlend, true) => &gpu.sprite_lit_pipeline_alpha,
                    };
                    let Some(draw_bg) = pd.draw_bg.as_ref() else {
                        continue;
                    };
                    pass.set_pipeline(dual.for_format(hdr));
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
                    let dual = match pd.blend {
                        SpriteBlend::Additive => &gpu.mesh_pipeline_additive,
                        SpriteBlend::Premultiplied => &gpu.mesh_pipeline_premultiplied,
                        SpriteBlend::AlphaBlend => &gpu.mesh_pipeline_alpha,
                    };
                    let Some(draw_bg) = pd.draw_bg_mesh.as_ref() else {
                        continue;
                    };
                    pass.set_pipeline(dual.for_format(hdr));
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
    use crate::resources::DeviceResources;

    /// Read every slot's lifetime back off the GPU.
    fn lifetimes(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        plugin: &GpuParticlesPlugin,
        id: GpuParticleSystemId,
    ) -> Vec<f32> {
        let system = plugin.system(id).expect("live system");
        let size = (system.capacity as u64) * std::mem::size_of::<GpuParticle>() as u64;
        let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("particle_readback"),
            size,
            usage: crate::gpu::BufferUsages::MAP_READ | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("particle_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&system.particle_buf, 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(crate::gpu::MapMode::Read, |_| {});
        let _ = device.poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let out = {
            let data = crate::gpu::mapped_range(slice);
            bytemuck::cast_slice::<u8, GpuParticle>(&data)
                .iter()
                .map(|p| p.lifetime)
                .collect()
        };
        staging.unmap();
        out
    }

    fn steady_emitter(rate: f32) -> crate::renderer::EmitterConfig {
        let mut e = crate::renderer::EmitterConfig::default();
        e.rate = rate;
        // A single lifetime, so "how many are alive" is a function of how many
        // were emitted rather than of the lifetime draw.
        e.lifetime = (100.0, 100.0);
        e
    }

    /// Drive the item type's own prepare, which is where emission lives, and
    /// submit the compute work it hands back.
    fn run_frames(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &DeviceResources,
        plugin: &mut GpuParticlesPlugin,
        id: GpuParticleSystemId,
        rate: f32,
        dt: f32,
        frames: usize,
    ) {
        for _ in 0..frames {
            let mut item = GpuParticleSystemItem::new(id, dt);
            item.emitter = steady_emitter(rate);
            let items: Vec<GpuParticleSystemItem> = vec![item];
            let camera = crate::RenderCamera::default();
            let ctx = ItemFrameContext {
                camera: &camera,
                viewport_size: glam::Vec2::new(64.0, 64.0),
                viewport_index: 0,
                frame_index: 0,
                jobs: crate::resources::Jobs::new(resources),
                resources,
                wireframe_mode: false,
                outline_selected: false,
                sub_selection: None,
                clip_objects: &[],
                quality_reduced: false,
                decal_excluded_surfaces: &[],
                ref_items: [None, None],
            };
            let bufs = plugin.prepare(device, queue, &ctx, &items);
            queue.submit(bufs);
        }
    }

    /// Every frame must emit exactly the budget the configured rate asks for
    /// while the system has slots to spare. The emit kernel picks slots by a
    /// wrapping window rather than by racing, so this is the check that the
    /// window does not quietly skip spawns.
    #[test]
    fn emission_matches_the_configured_rate() {
        let Some((device, queue, resources)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 256;
        let mut plugin = GpuParticlesPlugin::default();
        let id = plugin.create_system(&device, &queue, &resources, &config);

        // 10 per frame for 5 frames, well inside a 256-slot buffer.
        run_frames(&device, &queue, &resources, &mut plugin, id, 10.0, 1.0, 5);
        let live = lifetimes(&device, &queue, &plugin, id)
            .iter()
            .filter(|l| **l > 0.0)
            .count();
        assert_eq!(live, 50, "5 frames at 10 per frame should leave 50 alive");
    }

    /// The window wraps past the end of the buffer without losing a frame's
    /// spawns: 30 frames of 10 against 256 slots crosses the end once.
    #[test]
    fn emission_survives_the_window_wrapping() {
        let Some((device, queue, resources)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 256;
        let mut plugin = GpuParticlesPlugin::default();
        let id = plugin.create_system(&device, &queue, &resources, &config);

        run_frames(&device, &queue, &resources, &mut plugin, id, 10.0, 1.0, 30);
        let live = lifetimes(&device, &queue, &plugin, id)
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
        let Some((device, queue, resources)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 512;
        let mut plugin_a = GpuParticlesPlugin::default();
        let mut plugin_b = GpuParticlesPlugin::default();
        let a = plugin_a.create_system(&device, &queue, &resources, &config);
        let b = plugin_b.create_system(&device, &queue, &resources, &config);

        run_frames(&device, &queue, &resources, &mut plugin_a, a, 40.0, 0.5, 4);
        run_frames(&device, &queue, &resources, &mut plugin_b, b, 40.0, 0.5, 4);

        let la = lifetimes(&device, &queue, &plugin_a, a);
        let lb = lifetimes(&device, &queue, &plugin_b, b);
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
        let Some((device, queue, resources)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let config = GpuParticleSystemConfig::default();
        let mut plugin = GpuParticlesPlugin::default();

        let first = plugin.create_system(&device, &queue, &resources, &config);
        plugin.drop_system(first);
        let second = plugin.create_system(&device, &queue, &resources, &config);

        assert_ne!(first, second, "the reused slot must carry a new generation");
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

    fn srgb_texture(px: u8) -> crate::resources::TextureData {
        crate::resources::TextureData::srgb(4, 4, vec![px; 4 * 4 * 4])
    }

    fn sprite_config(texture_id: Option<crate::resources::TextureId>) -> GpuParticleSystemConfig {
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
        let Some((device, queue, mut resources)) =
            crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let tex = resources
            .upload_texture(&device, &queue, srgb_texture(200))
            .expect("texture upload");
        let mut plugin = GpuParticlesPlugin::default();
        let _system = plugin.create_system(&device, &queue, &resources, &sprite_config(Some(tex)));

        // Sync the gate so the assertion below isolates the free.
        plugin.revalidate_draw_bindings(&device, &resources);
        let baseline = plugin.draw_bg_rebuilds;

        resources.free_texture(tex);
        plugin.revalidate_draw_bindings(&device, &resources);
        assert_eq!(
            plugin.draw_bg_rebuilds,
            baseline + 1,
            "free of a baked texture must rebuild the system's draw bind group"
        );

        // Nothing further changed: the next poll is a no-op.
        plugin.revalidate_draw_bindings(&device, &resources);
        assert_eq!(plugin.draw_bg_rebuilds, baseline + 1);
    }

    /// A replace swaps the view behind a live id, which no per-entry check can
    /// see, so it must rebuild unconditionally.
    #[test]
    fn replaced_texture_rebuilds_draw_bind_group() {
        let Some((device, queue, mut resources)) =
            crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let tex = resources
            .upload_texture(&device, &queue, srgb_texture(40))
            .expect("texture upload");
        let mut plugin = GpuParticlesPlugin::default();
        let _system = plugin.create_system(&device, &queue, &resources, &sprite_config(Some(tex)));

        plugin.revalidate_draw_bindings(&device, &resources);
        let baseline = plugin.draw_bg_rebuilds;

        resources
            .replace_texture(&device, &queue, tex, srgb_texture(220))
            .expect("texture replace");
        plugin.revalidate_draw_bindings(&device, &resources);
        assert_eq!(
            plugin.draw_bg_rebuilds,
            baseline + 1,
            "replace must rebuild every live system's draw bind group"
        );
    }

    /// A system with no texture never rebuilds on someone else's free.
    #[test]
    fn untextured_system_survives_unrelated_free() {
        let Some((device, queue, mut resources)) =
            crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let unrelated = resources
            .upload_texture(&device, &queue, srgb_texture(10))
            .expect("texture upload");
        let mut plugin = GpuParticlesPlugin::default();
        let _system = plugin.create_system(&device, &queue, &resources, &sprite_config(None));

        plugin.revalidate_draw_bindings(&device, &resources);
        let baseline = plugin.draw_bg_rebuilds;

        resources.free_texture(unrelated);
        plugin.revalidate_draw_bindings(&device, &resources);
        assert_eq!(
            plugin.draw_bg_rebuilds, baseline,
            "a free the system does not name must not rebuild its bind groups"
        );
    }

    /// The systems a plugin holds are part of the renderer's working-set
    /// figure, which they were not while the store sat in core.
    #[test]
    fn live_systems_count_toward_resident_bytes() {
        let Some((device, queue, resources)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut plugin = GpuParticlesPlugin::default();
        assert_eq!(plugin.resident_bytes(), 0);

        let mut config = GpuParticleSystemConfig::default();
        config.capacity = 4096;
        let id = plugin.create_system(&device, &queue, &resources, &config);
        let held = plugin.resident_bytes();
        assert!(
            held >= 4096 * std::mem::size_of::<GpuParticle>() as u64,
            "the particle buffer is the bulk of what a system holds"
        );

        plugin.drop_system(id);
        assert_eq!(plugin.resident_bytes(), 0, "dropping gives the bytes back");
    }
}
