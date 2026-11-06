//! Every plugin type registered through a `ViewportInstance`, before its first
//! frame and after it: it runs, it stops once removed, and `rebuild` drops
//! the renderer-side registrations while the runtime's stay.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use viewport_lib::plugins::skinning::{SkinWeights, SkinningPlugin};
use viewport_lib::runtime::gpu_plugin::gpu_phase;
use viewport_lib::runtime::plugin::phase;
use viewport_lib::wgpu;
use viewport_lib::{
    Colour, InputOwnership, Material, MeshId, NodeId, PipelineMode, PostEffectSlot, RendererConfig,
    ViewportContext, ViewportInstance, ViewportRuntime,
};
use viewport_lib_testkit::fixtures::{
    CallLog, ConstantOffsetDeformer, CountedItemCollection, FlatColourMaterialPlugin,
    LoggingGpuPlugin, LoggingItemTypePlugin, LoggingPostEffectProducer, LoggingRuntimePlugin,
    PassthroughPostEffectStage, constant_offset_deformer, flat_colour_shading_hook, probe_quad,
};
use viewport_lib_testkit::{DeviceProfile, headless_device_with};

const SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const ITEMS: &str = "instance_probe_items";

/// A device with the renderer's recommended limits, which deformers,
/// material plugins and shading hooks need.
fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(&DeviceProfile::low_power("instance-plugins"))
}

/// An instance drawing the probe quad over a black background, rendered into
/// a target the test can read back.
struct Probe {
    instance: ViewportInstance,
    mesh: MeshId,
    node: NodeId,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    /// Submit a collection under [`ITEMS`] each frame.
    submit_items: bool,
}

impl Probe {
    fn new(device: &wgpu::Device) -> Self {
        let mut config = RendererConfig::new(FORMAT)
            .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
        if device
            .features()
            .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8)
        {
            config = config.with_scene_depth_format(wgpu::TextureFormat::Depth32FloatStencil8);
        }
        let mut instance = ViewportInstance::with_config(device, &config);
        let mesh = instance.upload_mesh(device, &probe_quad()).expect("quad");
        let material = instance
            .resources_mut()
            .upload_material(Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.6)))
            .expect("material");
        let node = instance
            .scene_mut()
            .add(Some(mesh), glam::Mat4::IDENTITY, material);
        let viewport = instance.viewport_frame_mut();
        viewport.show_grid = false;
        viewport.show_axes_indicator = false;
        viewport.background_colour = Some(Colour::linear_rgb(0.0, 0.0, 0.0));
        let (target, view) = readback_target(device);
        Self {
            instance,
            mesh,
            node,
            target,
            view,
            submit_items: false,
        }
    }

    /// Step the runtime, assemble, render and wait for the GPU.
    fn frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mode: PipelineMode) {
        let _ = self.instance.step_runtime(1.0 / 60.0);
        self.instance.effects_mut().display.mode = mode;
        self.instance.frame(ViewportContext::new(
            InputOwnership::default(),
            [SIZE as f32, SIZE as f32],
        ));
        if self.submit_items {
            self.instance
                .frame_data_mut()
                .scene
                .submit_plugin_items(ITEMS, CountedItemCollection::new(1));
        }
        let cmd = self.instance.render(device, queue, &self.view);
        queue.submit(std::iter::once(cmd));
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
    }

    /// The frame's centre pixel, RGBA8.
    fn centre(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> [u8; 4] {
        // 64 pixels of 4 bytes is one 256-byte row, so no padding.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_plugins_readback"),
            size: (SIZE * SIZE * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 4),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .expect("readback poll");
        let mapped = viewport_lib::gpu::mapped_range(staging.slice(..));
        let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
        [mapped[i], mapped[i + 1], mapped[i + 2], mapped[i + 3]]
    }
}

/// A colour target the test can copy back.
fn readback_target(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("instance_plugins_target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    (target, view)
}

/// The RGB channels, widened so comparisons can add margins.
fn rgb(p: [u8; 4]) -> [i32; 3] {
    [p[0] as i32, p[1] as i32, p[2] as i32]
}

fn luma(p: [u8; 4]) -> u32 {
    p[0] as u32 + p[1] as u32 + p[2] as u32
}

/// Registration happens before the first frame (`late` false, as in a setup
/// callback) or after one (`late` true, as from a frame callback).
const WHEN: [bool; 2] = [false, true];

#[test]
fn runtime_plugin_through_the_instance() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    for late in WHEN {
        let log = CallLog::new();
        let mut probe = Probe::new(&device);
        probe.instance.set_runtime(ViewportRuntime::new());
        if late {
            probe.frame(&device, &queue, PipelineMode::Hdr);
        }
        let id = probe
            .instance
            .runtime_mut()
            .unwrap()
            .add_plugin(LoggingRuntimePlugin::new(log.clone(), "rt", phase::ANIMATE));
        probe.frame(&device, &queue, PipelineMode::Hdr);
        assert_eq!(log.count("rt:step"), 1, "late={late}");
        assert!(
            probe
                .instance
                .runtime_mut()
                .unwrap()
                .remove_plugin(id)
                .is_some()
        );
        probe.frame(&device, &queue, PipelineMode::Hdr);
        assert_eq!(log.count("rt:step"), 1, "late={late}: removed plugin ran");
    }
}

#[test]
fn gpu_plugin_through_the_instance() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    for late in WHEN {
        let log = CallLog::new();
        let mut probe = Probe::new(&device);
        probe.instance.set_runtime(ViewportRuntime::new());
        if late {
            probe.frame(&device, &queue, PipelineMode::Hdr);
        }
        let id = probe
            .instance
            .runtime_mut()
            .unwrap()
            .add_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
        probe.frame(&device, &queue, PipelineMode::Hdr);
        assert_eq!(log.count("init_gpu"), 1, "late={late}");
        assert_eq!(log.count("pre_prepare"), 1, "late={late}");
        assert_eq!(log.count("post_paint"), 1, "late={late}");
        assert!(
            probe
                .instance
                .runtime_mut()
                .unwrap()
                .remove_gpu_plugin(id)
                .is_some()
        );
        probe.frame(&device, &queue, PipelineMode::Hdr);
        assert_eq!(
            log.count("pre_prepare"),
            1,
            "late={late}: removed plugin ran"
        );
    }
}

#[test]
fn item_type_plugin_through_the_instance() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    for late in WHEN {
        let log = CallLog::new();
        let mut probe = Probe::new(&device);
        probe.submit_items = true;
        if late {
            probe.frame(&device, &queue, PipelineMode::Hdr);
        }
        let id = probe
            .instance
            .with_item_type_plugin(
                &device,
                Box::new(LoggingItemTypePlugin::new(log.clone(), ITEMS)),
            )
            .expect("item type");
        probe.frame(&device, &queue, PipelineMode::Hdr);
        assert_eq!(log.count(&format!("prepare:{ITEMS}")), 1, "late={late}");
        assert!(probe.instance.remove_item_type_plugin(id).is_some());
        probe.frame(&device, &queue, PipelineMode::Hdr);
        assert_eq!(
            log.count(&format!("prepare:{ITEMS}")),
            1,
            "late={late}: removed plugin ran"
        );
    }
}

/// Post effects run on HDR frames and are skipped on LDR ones.
#[test]
fn post_effects_through_the_instance() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    for late in WHEN {
        let log = CallLog::new();
        let enabled = Arc::new(AtomicBool::new(true));
        let mut probe = Probe::new(&device);
        if late {
            probe.frame(&device, &queue, PipelineMode::Hdr);
        }
        let renderer = probe.instance.renderer_mut();
        let producer = renderer
            .add_post_effect_producer(Box::new(LoggingPostEffectProducer::new(
                log.clone(),
                PostEffectSlot::Bloom,
                enabled.clone(),
            )))
            .expect("producer");
        let stage = renderer
            .add_post_effect_stage(
                Box::new(PassthroughPostEffectStage::new(
                    &device,
                    log.clone(),
                    "stage",
                    0.5,
                    enabled.clone(),
                )),
                viewport_lib::plugin_api::post_effect::stage_order::EXTERNAL_DEFAULT,
            )
            .expect("stage");
        probe.frame(&device, &queue, PipelineMode::Hdr);
        assert_eq!(log.count("encode:0"), 1, "late={late}: producer");
        assert_eq!(log.count("stage:encode:0"), 1, "late={late}: stage");

        log.clear();
        probe.frame(&device, &queue, PipelineMode::Direct);
        assert_eq!(
            log.count("encode"),
            0,
            "late={late}: LDR runs no post effects"
        );
        assert_eq!(log.count("stage:encode"), 0, "late={late}");

        let renderer = probe.instance.renderer_mut();
        renderer.remove_post_effect_producer(producer);
        renderer.remove_post_effect_stage(stage);
        probe.frame(&device, &queue, PipelineMode::Hdr);
        log.assert_empty();
    }
}

#[test]
fn deformer_through_the_instance() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    for late in WHEN {
        let mut probe = Probe::new(&device);
        probe.frame(&device, &queue, PipelineMode::Hdr);
        let covered = luma(probe.centre(&device, &queue));
        assert!(covered > 60, "the quad covers the centre ({covered})");

        let id = match probe
            .instance
            .resources_mut()
            .register_deformer(&device, constant_offset_deformer())
        {
            Ok(id) => id,
            Err(err) => {
                eprintln!("skipping: this device cannot register a deformer: {err}");
                return;
            }
        };
        let vertices = probe_quad().positions.len();
        let mesh = probe.mesh;
        ConstantOffsetDeformer::attach(
            probe.instance.resources_mut(),
            &device,
            mesh,
            id.slot(),
            vertices,
        );
        ConstantOffsetDeformer::set_offset(
            probe.instance.resources_mut(),
            &queue,
            id.slot(),
            glam::Vec3::new(4.0, 0.0, 0.0),
        );
        if late {
            probe.frame(&device, &queue, PipelineMode::Hdr);
        }
        probe.frame(&device, &queue, PipelineMode::Hdr);
        let moved = luma(probe.centre(&device, &queue));
        assert!(
            moved + 40 < covered,
            "late={late}: the deformer moves the quad off the centre ({covered} -> {moved})"
        );

        probe
            .instance
            .resources_mut()
            .remove_deformer(&device, id)
            .expect("remove deformer");
        probe.frame(&device, &queue, PipelineMode::Hdr);
        let back = luma(probe.centre(&device, &queue));
        assert!(
            back + 20 > covered,
            "late={late}: the removed deformer still moves the quad ({covered} -> {back})"
        );
    }
}

/// A material plugin shades the materials that select it; unregistering it
/// sends them back to built-in shading, and a replacement registered after
/// shades the materials that select the new id.
#[test]
fn material_plugin_through_the_instance() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    for late in WHEN {
        let mut probe = Probe::new(&device);
        if late {
            probe.frame(&device, &queue, PipelineMode::Hdr);
        }
        let green = FlatColourMaterialPlugin::new("instance_green", [0.0, 0.9, 0.0]);
        let green_id = match probe
            .instance
            .resources_mut()
            .register_material_plugin(&device, &green)
        {
            Ok(id) => id,
            Err(err) => {
                eprintln!("skipping: this device cannot register a material plugin: {err}");
                return;
            }
        };
        let material = probe
            .instance
            .resources_mut()
            .upload_material(
                Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.6)).with_plugin(green_id),
            )
            .expect("material");
        let node = probe.node;
        probe.instance.scene_mut().set_material(node, material);
        probe.frame(&device, &queue, PipelineMode::Hdr);
        let [r, g, b] = rgb(probe.centre(&device, &queue));
        assert!(
            g > r + 60 && g > b + 60,
            "late={late}: green plugin {r},{g},{b}"
        );

        probe
            .instance
            .resources_mut()
            .unregister_material_plugin(green_id)
            .expect("unregister");
        probe.frame(&device, &queue, PipelineMode::Hdr);
        let [r, g, b] = rgb(probe.centre(&device, &queue));
        assert!(
            g < r + 30,
            "late={late}: built-in shading after unregister {r},{g},{b}"
        );

        let red = FlatColourMaterialPlugin::new("instance_red", [0.9, 0.0, 0.0]);
        let red_id = probe
            .instance
            .resources_mut()
            .register_material_plugin(&device, &red)
            .expect("replacement");
        let material = probe
            .instance
            .resources_mut()
            .upload_material(
                Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.6)).with_plugin(red_id),
            )
            .expect("material");
        probe.instance.scene_mut().set_material(node, material);
        probe.frame(&device, &queue, PipelineMode::Hdr);
        let [r, g, b] = rgb(probe.centre(&device, &queue));
        assert!(
            r > g + 60 && r > b + 60,
            "late={late}: replacement plugin {r},{g},{b}"
        );
    }
}

/// A shading hook registers and unregisters through the instance's
/// resources. On its own it selects no draw: only a material plugin, which
/// sits on a hook, is chosen by a material.
#[test]
fn shading_hook_through_the_instance() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    for late in WHEN {
        let mut probe = Probe::new(&device);
        if late {
            probe.frame(&device, &queue, PipelineMode::Hdr);
        }
        let id = match probe.instance.resources_mut().register_shading_hook(
            &device,
            flat_colour_shading_hook("instance_hook", [0.0, 0.9, 0.0]),
        ) {
            Ok(id) => id,
            Err(err) => {
                eprintln!("skipping: this device cannot register a shading hook: {err}");
                return;
            }
        };
        assert_eq!(
            probe.instance.resources().registered_shading_hook_count(),
            1
        );
        assert_eq!(
            probe
                .instance
                .resources()
                .shading_hook_id_by_name("instance_hook"),
            Some(id)
        );
        probe.frame(&device, &queue, PipelineMode::Hdr);
        probe
            .instance
            .resources_mut()
            .unregister_shading_hook(id)
            .expect("unregister");
        assert_eq!(
            probe.instance.resources().registered_shading_hook_count(),
            0
        );
        probe.frame(&device, &queue, PipelineMode::Hdr);
    }
}

/// The skinning handle attaches weights and palettes through the instance's
/// resources; `uninstall` detaches the meshes and leaves the deformer.
#[test]
fn skinning_handle_through_the_instance() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    for late in WHEN {
        let mut probe = Probe::new(&device);
        if late {
            probe.frame(&device, &queue, PipelineMode::Hdr);
        }
        let skinning = match SkinningPlugin::install(probe.instance.resources_mut(), &device) {
            Ok(handle) => handle,
            Err(err) => {
                eprintln!("skipping: this device cannot install skinning: {err}");
                return;
            }
        };
        let mesh = probe.mesh;
        let vertices = probe_quad().positions.len();
        let weights = SkinWeights {
            joint_indices: vec![[0, 0, 0, 0]; vertices],
            joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; vertices],
        };
        skinning.attach_weights(probe.instance.resources_mut(), &device, mesh, &weights);
        assert!(skinning.attach_palette(
            probe.instance.resources_mut(),
            &device,
            &queue,
            mesh,
            0,
            &[glam::Mat4::IDENTITY],
        ));
        let slot = skinning.deformer_id().slot();
        probe.frame(&device, &queue, PipelineMode::Hdr);
        assert!(skinning.is_skinned_mesh(mesh));
        assert!(
            probe
                .instance
                .resources()
                .has_deform_slot_instance(mesh, 0, slot)
        );

        // The marker is shared between clones, so the copy kept here sees
        // what `uninstall` (which consumes the handle) detached.
        let kept = skinning.clone();
        skinning.uninstall(probe.instance.resources_mut(), &device);
        let skinning = kept;
        assert!(!skinning.is_skinned_mesh(mesh), "late={late}");
        assert!(
            probe
                .instance
                .resources()
                .deformer_id_by_name("viewport_skin")
                .is_some(),
            "late={late}: the skinning deformer stays registered"
        );
        probe.frame(&device, &queue, PipelineMode::Hdr);
    }
}

/// `rebuild` drops every renderer-side registration (item types, post
/// effects, deformers, material plugins and shading hooks, the skinning
/// deformer) and keeps the runtime with its runtime and GPU plugins.
#[test]
fn rebuild_drops_renderer_registrations_and_keeps_the_runtime() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter with recommended limits available");
        return;
    };
    let Some((new_device, new_queue)) = self::device() else {
        panic!("no adapter for the rebuilt device");
    };
    let log = CallLog::new();
    let enabled = Arc::new(AtomicBool::new(true));
    let mut probe = Probe::new(&device);
    probe.submit_items = true;
    probe.instance.set_runtime(
        ViewportRuntime::new()
            .with_plugin(LoggingRuntimePlugin::new(log.clone(), "rt", phase::ANIMATE))
            .with_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE)),
    );
    probe
        .instance
        .with_item_type_plugin(
            &device,
            Box::new(LoggingItemTypePlugin::new(log.clone(), ITEMS)),
        )
        .expect("item type");
    probe
        .instance
        .renderer_mut()
        .add_post_effect_producer(Box::new(LoggingPostEffectProducer::new(
            log.clone(),
            PostEffectSlot::Bloom,
            enabled.clone(),
        )))
        .expect("producer");
    let resources = probe.instance.resources_mut();
    let Ok(_deformer) = resources.register_deformer(&device, constant_offset_deformer()) else {
        eprintln!("skipping: this device cannot register a deformer");
        return;
    };
    resources
        .register_material_plugin(
            &device,
            &FlatColourMaterialPlugin::new("rebuild_green", [0.0, 0.9, 0.0]),
        )
        .expect("material plugin");
    resources
        .register_shading_hook(
            &device,
            flat_colour_shading_hook("rebuild_hook", [0.9, 0.0, 0.0]),
        )
        .expect("shading hook");
    SkinningPlugin::install(resources, &device).expect("skinning");
    probe.frame(&device, &queue, PipelineMode::Hdr);
    assert_eq!(log.count(&format!("prepare:{ITEMS}")), 1);
    assert_eq!(log.count("encode:0"), 1);

    let stale = probe.instance.rebuild(&new_device).expect("rebuild");
    let resources = probe.instance.resources();
    assert!(
        probe
            .instance
            .item_type_plugin::<LoggingItemTypePlugin>(ITEMS)
            .is_none()
    );
    assert!(
        resources
            .deformer_id_by_name(ConstantOffsetDeformer::NAME)
            .is_none()
    );
    assert!(resources.deformer_id_by_name("viewport_skin").is_none());
    assert_eq!(resources.registered_shading_hook_count(), 0);
    // `StaleHandles` names meshes, materials and textures only.
    assert_eq!(stale.meshes(), &[probe.mesh]);

    log.clear();
    // The quad's mesh is gone with the old renderer, so bind a new one.
    let mesh = probe
        .instance
        .upload_mesh(&new_device, &probe_quad())
        .expect("quad");
    let node = probe.node;
    probe.instance.scene_mut().set_mesh(node, Some(mesh));
    // The old target belongs to the old device.
    (probe.target, probe.view) = readback_target(&new_device);
    probe.frame(&new_device, &new_queue, PipelineMode::Hdr);
    assert_eq!(log.count("rt:step"), 1, "the runtime plugin is kept");
    assert_eq!(log.count("init_gpu"), 1, "the GPU plugin re-initialises");
    assert_eq!(log.count("pre_prepare"), 1, "the GPU plugin is kept");
    assert_eq!(
        log.count(&format!("prepare:{ITEMS}")),
        0,
        "the item type is gone"
    );
    assert_eq!(log.count("encode:0"), 0, "the post effect is gone");
}
