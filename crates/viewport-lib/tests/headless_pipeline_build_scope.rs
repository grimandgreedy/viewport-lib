//! What a first frame compiles.
//!
//! Pipelines are built by the first frame that binds them, so a frame should
//! compile what it draws with and nothing else. These read the build log, which
//! is process-wide, so everything runs from one test and in sequence.

use viewport_lib::resources::build_log;
use viewport_lib::wgpu;

mod common;
use common::*;

fn frame() -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame
}

fn cubes(mesh_id: MeshId, count: usize) -> SurfaceSubmission {
    let items: Vec<SceneRenderItem> = (0..count)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(i as f32 * 1.5, 0.0, 0.0))
                .to_cols_array_2d();
            item
        })
        .collect();
    SurfaceSubmission::Flat(items.into())
}

/// Labels built by one frame on a fresh renderer.
fn built_by(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    setup: impl FnOnce(&mut ViewportRenderer) -> FrameData,
) -> Vec<String> {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let frame = setup(&mut renderer);
    let _ = build_log::drain();
    let _ = renderer.render_offscreen(device, queue, &frame, 64, 64);
    build_log::drain().into_iter().map(|(l, _)| l).collect()
}

fn assert_none(labels: &[String], what: &str, unwanted: &[&str]) {
    let found: Vec<&String> = labels
        .iter()
        .filter(|l| unwanted.iter().any(|u| l.contains(u)))
        .collect();
    assert!(found.is_empty(), "{what} built {found:?}");
}

fn assert_has(labels: &[String], what: &str, wanted: &str) {
    assert!(
        labels.iter().any(|l| l == wanted),
        "{what} did not build {wanted}; it built {labels:?}"
    );
}

/// The LDR mesh family: the four pipelines and their module.
const LDR_MESH: &[&str] = &[
    "module mesh_shader",
    "solid_pipeline",
    "solid_two_sided_pipeline",
    "transparent_pipeline",
    "wireframe_pipeline",
];

#[test]
fn a_frame_builds_only_what_it_binds() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    build_log::enable();

    // An HDR frame with nothing in it: the tone map and the OIT resolve, and
    // none of the mesh sets, the OIT mesh set, the exposure metering or any effect.
    let labels = built_by(&device, &queue, |_| frame());
    assert_has(&labels, "an empty HDR frame", "tone_map_pipeline");
    assert_none(
        &labels,
        "an empty HDR frame",
        &[
            "mesh",
            "solid",
            "oit_pipeline",
            "bloom",
            "ssao",
            "dof",
            "fxaa",
            "contact",
            "exposure",
            "lic",
            "ssaa",
            "foreground",
            "depth_blit",
            "shadow",
        ],
    );

    // An HDR frame with one mesh: the HDR family and not the LDR one.
    let labels = built_by(&device, &queue, |renderer| {
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &box_mesh())
            .unwrap();
        let mut frame = frame();
        frame.scene.surfaces = cubes(mesh_id, 1);
        frame
    });
    assert_has(&labels, "an HDR mesh frame", "hdr_solid_pipeline");
    let ldr: Vec<&String> = labels
        .iter()
        .filter(|l| LDR_MESH.contains(&l.as_str()))
        .collect();
    assert!(ldr.is_empty(), "an HDR mesh frame built {ldr:?}");
    assert_none(
        &labels,
        "an opaque HDR mesh frame",
        &["mesh_oit", "oit_pipeline", "bloom", "ssao"],
    );

    // A Direct frame with instanced meshes: the LDR instanced set, and neither
    // the HDR nor the OIT instanced set, nor any of the post chain.
    let labels = built_by(&device, &queue, |renderer| {
        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&device, &box_mesh())
            .unwrap();
        let mut frame = frame();
        frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        frame.scene.surfaces = cubes(mesh_id, 3);
        frame
    });
    assert_none(
        &labels,
        "a Direct instanced frame",
        &[
            "hdr_solid_instanced",
            "hdr_transparent_instanced",
            "hdr_instanced",
            "oit_instanced",
            "tone_map",
            "hdr_solid_pipeline",
            "mesh_shader_hdr",
        ],
    );

    build_log::disable();

    a_direct_paint_finds_the_ldr_pipelines(&device, &queue);
}

/// A frame in the default HDR mode painted straight into the caller's render
/// pass draws with the LDR pipelines. Preparing through `pass()` has to build
/// them, including right after the same renderer drew a frame through
/// `owned()`, which builds the HDR family instead. Run from the test above so
/// its builds stay out of the log that test reads.
fn a_direct_paint_finds_the_ldr_pipelines(device: &wgpu::Device, queue: &wgpu::Queue) {
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let mut renderer = ViewportRenderer::new(device, format);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(device, &box_mesh())
        .unwrap();
    let vp = renderer.create_viewport(device);

    let target = |label, format, usage| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let colour = target(
        "direct_paint_colour",
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    );
    let depth = target(
        "direct_paint_depth",
        SCENE_DEPTH_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    );

    for count in [1, 3] {
        let mut frame = frame();
        frame.camera = frame.camera.with_viewport_id(vp);
        frame.scene.surfaces = cubes(mesh_id, count);
        frame.scene.generation = count as u64;

        // The owned path first, so the direct paint below follows an HDR frame.
        let cmd = renderer.owned().render(device, queue, &colour, &frame);
        queue.submit([cmd]);

        let (scene_fx, _) = frame.effects.split();
        let token = renderer
            .pass()
            .prepare_scene(device, queue, &frame, &scene_fx);
        renderer
            .pass()
            .prepare_viewport(device, queue, &token, vp, &frame);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("direct_paint"),
        });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("direct_paint_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &colour,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            renderer.pass().paint_viewport(&mut rp, vp, &frame);
        }
        queue.submit([encoder.finish()]);
    }
}
