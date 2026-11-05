//! The scene colour and depth a `ViewportInstance` frame leaves behind, as a
//! GPU plugin's `post_paint` would receive them.

use viewport_lib::plugin_api::depth::DEPTH_CLEAR;
use viewport_lib::wgpu;
use viewport_lib::{
    InputOwnership, Material, OffscreenViewportTarget, PipelineMode, RendererConfig,
    ViewportContext, ViewportInstance, primitives,
};
use viewport_lib_testkit::{headless_device, orbit_camera};

const SIZE: u32 = 64;
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// An instance drawing one cube that covers the centre of the frame.
fn instance(device: &wgpu::Device) -> ViewportInstance {
    let mut config = RendererConfig::new(OffscreenViewportTarget::render_format(SURFACE_FORMAT))
        .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    if device
        .features()
        .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8)
    {
        config = config.with_scene_depth_format(wgpu::TextureFormat::Depth32FloatStencil8);
    }
    let mut instance = ViewportInstance::with_config(device, &config);
    let material = instance
        .resources_mut()
        .upload_material(Material::default())
        .expect("material");
    instance
        .upload_and_add(
            device,
            &primitives::cube(2.0),
            glam::Mat4::IDENTITY,
            material,
        )
        .expect("mesh");
    *instance.camera_mut() = orbit_camera(glam::Vec3::ZERO, 5.0, 0.6, 1.0);
    instance
}

/// Assemble and render two frames into `view`.
fn render(
    instance: &mut ViewportInstance,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    mode: PipelineMode,
) {
    instance.effects_mut().display.mode = mode;
    for _ in 0..2 {
        instance.frame(ViewportContext::new(
            InputOwnership::default(),
            [SIZE as f32, SIZE as f32],
        ));
        let cmd = instance.render(device, queue, view);
        queue.submit(std::iter::once(cmd));
    }
}

/// The depth at the centre of `view` (a depth-aspect view) and the largest
/// depth anywhere in it, read with a compute pass the way a plugin would.
fn probe_depth(device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView) -> [f32; 2] {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("depth_probe"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> out: array<f32, 2>;
@compute @workgroup_size(1)
fn main() {
    let dims = textureDimensions(depth_tex);
    out[0] = textureLoad(depth_tex, vec2<i32>(dims / 2u), 0);
    var m = 0.0;
    for (var y = 0u; y < dims.y; y++) {
        for (var x = 0u; x < dims.x; x++) {
            m = max(m, textureLoad(depth_tex, vec2<i32>(i32(x), i32(y)), 0));
        }
    }
    out[1] = m;
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("depth_probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("depth_probe_out"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("depth_probe_staging"),
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("depth_probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, 8);
    queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .expect("probe poll");
    let mapped = viewport_lib::gpu::mapped_range(staging.slice(..));
    let read = |i: usize| f32::from_le_bytes(mapped[i * 4..i * 4 + 4].try_into().unwrap());
    [read(0), read(1)]
}

/// After an HDR frame, and after an LDR frame into a sampleable offscreen
/// target, the instance hands out a scene colour and depth, both sampleable.
#[test]
fn scene_targets_are_sampleable_on_hdr_and_ldr() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    for mode in [PipelineMode::Hdr, PipelineMode::Direct] {
        let mut instance = instance(&device);
        render(&mut instance, &device, &queue, target.render_view(), mode);
        let targets = instance
            .scene_target_views(Some(target.render_view()))
            .unwrap_or_else(|| panic!("{mode:?}: no scene targets"));
        for (name, view) in [
            ("colour", targets.color_view),
            ("depth", targets.depth_view),
        ] {
            assert!(
                view.texture()
                    .usage()
                    .contains(wgpu::TextureUsages::TEXTURE_BINDING),
                "{mode:?}: {name} is not sampleable"
            );
        }
        // The probe binds the depth view as a depth texture, so a view that
        // cannot be sampled that way fails validation here.
        let _ = probe_depth(&device, &queue, targets.depth_view);
    }
}

/// An LDR frame into a target without `TEXTURE_BINDING` (a window surface)
/// has no readable colour, so there are no scene targets.
#[test]
fn ldr_frame_into_an_unsampleable_target_has_no_scene_targets() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_like_target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OffscreenViewportTarget::render_format(SURFACE_FORMAT),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let mut instance = instance(&device);
    render(&mut instance, &device, &queue, &view, PipelineMode::Direct);
    assert!(instance.scene_target_views(Some(&view)).is_none());
}

/// The depth a frame leaves behind holds the frame's geometry, with default
/// settings, on HDR and on LDR.
///
/// Fails today: at render scale 1 the HDR overlay pass attaches the scene
/// depth (as the output depth) and discards it, and the axes indicator puts
/// that pass in every default frame; the LDR scene pass discards its depth
/// unless occlusion culling is on. With the axes indicator off (HDR) or
/// occlusion culling on (LDR) the depth survives.
#[test]
#[ignore = "the scene depth is discarded at the end of the frame"]
fn scene_depth_holds_the_frame_after_render() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let target = OffscreenViewportTarget::new(&device, SURFACE_FORMAT, [SIZE, SIZE]);
    for mode in [PipelineMode::Hdr, PipelineMode::Direct] {
        let mut instance = instance(&device);
        render(&mut instance, &device, &queue, target.render_view(), mode);
        let targets = instance
            .scene_target_views(Some(target.render_view()))
            .unwrap_or_else(|| panic!("{mode:?}: no scene targets"));
        let [centre, max] = probe_depth(&device, &queue, targets.depth_view);
        assert!(
            centre > DEPTH_CLEAR && max > DEPTH_CLEAR,
            "{mode:?}: the depth should hold the cube, got centre {centre}, max {max}"
        );
    }
}
