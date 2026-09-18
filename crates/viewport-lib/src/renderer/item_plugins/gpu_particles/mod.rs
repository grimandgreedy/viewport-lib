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

use crate::plugin_api::{ItemFrameContext, ItemTypePlugin, PaintContext, PluginItemCollection};
use crate::renderer::{GpuParticleSystemItem, SpriteBlend};
use crate::resources::gpu::gpu_particles::{
    ParticleDrawRoute, ParticleRender, build_emit_params, build_sim_params,
};

pub(crate) const TYPE_NAME: &str = "viewport.gpu_particles";

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
    gpu: Option<pipeline::ParticleGpu>,
    frame: Vec<ParticleFrame>,
}

impl ItemTypePlugin for GpuParticlesPlugin {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }

    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {
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
        let items = items
            .as_any()
            .downcast_ref::<Vec<GpuParticleSystemItem>>()
            .expect("particle collection is the SceneFrame field");
        if items.is_empty() {
            return Vec::new();
        }
        let gpu = self
            .gpu
            .get_or_insert_with(|| pipeline::ParticleGpu::new(device, ctx.resources));

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
            let Some(system) = ctx.resources.particle_system(item.system_id) else {
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
    use crate::resources::gpu::gpu_particles::{
        GpuParticle, GpuParticleSystemConfig, GpuParticleSystemId,
    };

    /// Read every slot's lifetime back off the GPU.
    fn lifetimes(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        resources: &DeviceResources,
        id: GpuParticleSystemId,
    ) -> Vec<f32> {
        let system = resources.particle_system(id).expect("live system");
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
        let mut resources = resources;
        let id = resources.create_gpu_particle_system(&device, &queue, &config);
        let mut plugin = GpuParticlesPlugin::default();

        // 10 per frame for 5 frames, well inside a 256-slot buffer.
        run_frames(&device, &queue, &resources, &mut plugin, id, 10.0, 1.0, 5);
        let live = lifetimes(&device, &queue, &resources, id)
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
        let mut resources = resources;
        let id = resources.create_gpu_particle_system(&device, &queue, &config);
        let mut plugin = GpuParticlesPlugin::default();

        run_frames(&device, &queue, &resources, &mut plugin, id, 10.0, 1.0, 30);
        let live = lifetimes(&device, &queue, &resources, id)
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
        let mut resources = resources;
        let a = resources.create_gpu_particle_system(&device, &queue, &config);
        let b = resources.create_gpu_particle_system(&device, &queue, &config);
        let mut plugin_a = GpuParticlesPlugin::default();
        let mut plugin_b = GpuParticlesPlugin::default();

        run_frames(&device, &queue, &resources, &mut plugin_a, a, 40.0, 0.5, 4);
        run_frames(&device, &queue, &resources, &mut plugin_b, b, 40.0, 0.5, 4);

        let la = lifetimes(&device, &queue, &resources, a);
        let lb = lifetimes(&device, &queue, &resources, b);
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
        let mut resources = resources;
        let config = GpuParticleSystemConfig::default();

        let first = resources.create_gpu_particle_system(&device, &queue, &config);
        resources.drop_gpu_particle_system(first);
        let second = resources.create_gpu_particle_system(&device, &queue, &config);

        assert_ne!(first, second, "the reused slot must carry a new generation");
        assert!(
            resources.particle_system(first).is_none(),
            "the dropped handle must not resolve"
        );
        assert!(
            resources.particle_system(second).is_some(),
            "the live handle must resolve"
        );
    }
}
