//! Overlays-only frames: a frame on the LDR path whose scene submits nothing
//! draws only its overlay pass.
//!
//! The reference for every image here is the same frame drawn through the LDR
//! scene path, forced by one hidden mesh item: it draws nothing, so the two
//! paths must agree pixel for pixel.

use viewport_lib::{FrameData, FramePath, PipelineMode, SceneRenderItem, wgpu};
use viewport_lib_testkit::scenes::overlays;
use viewport_lib_testkit::{Harness, frame_for};

const W: u32 = 192;
const H: u32 = 128;

/// The overlay scene `name` with its backdrop mesh removed, on the LDR path.
fn empty_frame(h: &mut Harness, name: &str) -> FrameData {
    let scene = overlays::scenes()
        .into_iter()
        .find(|s| s.name == name)
        .unwrap();
    let mut built = h.build_scene(&scene);
    built.items.clear();
    let mut frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    frame.effects.display.mode = PipelineMode::Direct;
    frame
}

/// A mesh item that draws nothing.
fn hidden_item(h: &mut Harness) -> SceneRenderItem {
    let mesh = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &viewport_lib::primitives::cube(1.0))
        .expect("upload cube");
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.settings.hidden = true;
    item
}

/// `frame` with `item` added, which sends it through the scene path.
fn with_item(mut frame: FrameData, item: &SceneRenderItem) -> FrameData {
    let generation = frame.scene.generation;
    frame.scene = viewport_lib::SceneFrame::from_surface_items(vec![item.clone()]);
    frame.scene.generation = generation;
    frame
}

fn assert_same(name: &str, want: &[u8], got: &[u8]) {
    assert!(
        want.chunks_exact(4).any(|p| p != &want[..4]),
        "{name}: the reference draws nothing"
    );
    let differing = want
        .chunks_exact(4)
        .zip(got.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differing, 0, "{name}: {differing} pixels differ");
}

#[test]
fn frame_path_follows_what_the_scene_submits() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let item = hidden_item(&mut h);
    let mut visible = item.clone();
    visible.settings.hidden = false;

    // A mesh frame first, so the empty frame after it has scene counters to
    // clear.
    let frame = with_item(empty_frame(&mut h, "overlay_shapes"), &visible);
    h.render(&frame, W, H);
    assert_eq!(h.stats().frame_path, FramePath::Ldr);
    assert!(h.stats().draw_calls > 0);

    let frame = empty_frame(&mut h, "overlay_shapes");
    h.render(&frame, W, H);
    let stats = h.stats();
    assert_eq!(stats.frame_path, FramePath::OverlaysOnly);
    assert_eq!(stats.total_objects, 0);
    assert_eq!(stats.draw_calls, 0);
    assert_eq!(stats.instanced_batches, 0);
    assert!(stats.overlay_draw_calls > 0);

    // A hidden item still submits a scene.
    let frame = with_item(empty_frame(&mut h, "overlay_shapes"), &item);
    h.render(&frame, W, H);
    assert_eq!(h.stats().frame_path, FramePath::Ldr);

    // Scene chrome on the full tier keeps the scene pass.
    let mut grid = empty_frame(&mut h, "overlay_shapes");
    grid.viewport.show_grid = true;
    h.render(&grid, W, H);
    assert_eq!(h.stats().frame_path, FramePath::Ldr);

    // The HDR path keeps its post chain.
    let mut hdr = empty_frame(&mut h, "overlay_shapes");
    hdr.effects.display.mode = PipelineMode::Hdr;
    h.render(&hdr, W, H);
    assert_eq!(h.stats().frame_path, FramePath::Hdr);
}

/// Every overlay scene without its backdrop, again with backdrop blurs on the
/// shapes, and again over a translucent background.
#[test]
fn overlays_only_draws_what_the_scene_path_draws() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let item = hidden_item(&mut h);
    for scene in overlays::scenes() {
        for variant in 0..3 {
            let make = |h: &mut Harness| {
                let mut frame = empty_frame(h, scene.name);
                if variant == 1 {
                    for shape in &mut frame.overlays.shapes {
                        shape.style.backdrop = viewport_lib::BackdropEffects::default()
                            .with_blur(4.0)
                            .with_filters(1.0, 0.25, 0.0);
                    }
                }
                if variant == 2 {
                    frame.viewport.background_colour =
                        Some(viewport_lib::Colour::linear(0.2, 0.1, 0.3, 0.5));
                }
                frame
            };
            let frame = with_item(make(&mut h), &item);
            let want = h.render(&frame, W, H);
            let frame = make(&mut h);
            assert_eq!(h.stats().frame_path, FramePath::Ldr);
            let got = h.render(&frame, W, H);
            assert_eq!(h.stats().frame_path, FramePath::OverlaysOnly);
            assert_same(&format!("{} variant {variant}", scene.name), &want, &got);
        }
    }
}

fn target(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("overlays_only_target"),
        size: wgpu::Extent3d {
            width: w,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let w = texture.width();
    let row =
        (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("overlays_only_readback"),
        size: (row * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let mapped = viewport_lib::gpu::mapped_range(slice);
    let mut out = Vec::with_capacity((w * H * 4) as usize);
    for y in 0..H as usize {
        let start = y * row as usize;
        out.extend_from_slice(&mapped[start..start + (w * 4) as usize]);
    }
    out
}

/// Draw `frames` side by side into one host-owned pass through the split
/// API, each viewport under its own viewport and scissor rect, as an egui or
/// iced host does. Returns the image and the path the last render reported.
fn paint_side_by_side(h: &mut Harness, frames: &mut [FrameData]) -> Vec<u8> {
    let device = h.device.clone();
    let queue = h.queue.clone();
    let r = &mut h.renderer;
    let ids: Vec<_> = frames.iter().map(|_| r.create_viewport(&device)).collect();
    for (frame, id) in frames.iter_mut().zip(&ids) {
        frame.camera.viewport_index = id.index();
    }
    let total_w = W * frames.len() as u32;
    let colour = target(&device, Harness::DEFAULT_TARGET_FORMAT, total_w);
    let depth = target(&device, r.scene_depth_format(), total_w);
    let (colour_view, depth_view) = (
        colour.create_view(&Default::default()),
        depth.create_view(&Default::default()),
    );
    let (scene_fx, _) = frames[0].effects.split();
    let token = r
        .pass()
        .prepare_scene(&device, &queue, &frames[0], &scene_fx);
    for (frame, id) in frames.iter().zip(&ids) {
        r.pass()
            .prepare_viewport(&device, &queue, &token, *id, frame);
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlays_only_host_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &colour_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(viewport_lib::plugin_api::depth::DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            // The tail fields differ between the wgpu legs.
            ..Default::default()
        });
        for (k, (frame, id)) in frames.iter().zip(&ids).enumerate() {
            let x = k as u32 * W;
            rp.set_viewport(x as f32, 0.0, W as f32, H as f32, 0.0, 1.0);
            rp.set_scissor_rect(x, 0, W, H);
            r.pass_view().paint_viewport(&mut rp, *id, frame);
        }
    }
    queue.submit([encoder.finish()]);
    read_back(&device, &queue, &colour)
}

/// Draw `frame` through the single-viewport host-owned pass path.
fn paint_single(h: &mut Harness, frame: &FrameData) -> Vec<u8> {
    let device = h.device.clone();
    let queue = h.queue.clone();
    let r = &mut h.renderer;
    let colour = target(&device, Harness::DEFAULT_TARGET_FORMAT, W);
    let depth = target(&device, r.scene_depth_format(), W);
    let (colour_view, depth_view) = (
        colour.create_view(&Default::default()),
        depth.create_view(&Default::default()),
    );
    let pre = r.pass().prepare(&device, &queue, frame);
    queue.submit(pre);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlays_only_host_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &colour_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(viewport_lib::plugin_api::depth::DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            ..Default::default()
        });
        r.pass_view().paint(&mut rp, frame);
    }
    queue.submit([encoder.finish()]);
    read_back(&device, &queue, &colour)
}

/// Draw `frames` through the split owned API, one target per viewport.
fn render_split(h: &mut Harness, frames: &mut [FrameData]) -> (Vec<Vec<u8>>, Vec<FramePath>) {
    let device = h.device.clone();
    let queue = h.queue.clone();
    let r = &mut h.renderer;
    let ids: Vec<_> = frames.iter().map(|_| r.create_viewport(&device)).collect();
    for (frame, id) in frames.iter_mut().zip(&ids) {
        frame.camera.viewport_index = id.index();
    }
    let (scene_fx, _) = frames[0].effects.split();
    let token = r
        .owned()
        .prepare_scene(&device, &queue, &frames[0], &scene_fx);
    for (frame, id) in frames.iter().zip(&ids) {
        r.owned()
            .prepare_viewport(&device, &queue, &token, *id, frame);
    }
    let mut images = Vec::new();
    let mut paths = Vec::new();
    for (frame, id) in frames.iter().zip(&ids) {
        let tex = target(&device, Harness::DEFAULT_TARGET_FORMAT, W);
        let cmd = r.owned().render_viewport(
            &device,
            &queue,
            &tex.create_view(&Default::default()),
            *id,
            frame,
        );
        paths.push(r.last_frame_stats().frame_path);
        queue.submit([cmd]);
        images.push(read_back(&device, &queue, &tex));
    }
    (images, paths)
}

/// The host-owned pass path, single and split with a scissor per viewport,
/// and the owned split path agree with the scene path on an empty scene.
#[test]
fn host_pass_and_split_paths_draw_what_the_scene_path_draws() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let item = hidden_item(&mut h);
    for name in ["overlay_shapes", "overlay_layers"] {
        let pair = |h: &mut Harness, with: bool| {
            let a = empty_frame(h, name);
            let mut b = empty_frame(h, "overlay_labels");
            b.scene.generation = a.scene.generation;
            if with {
                [with_item(a, &item), with_item(b, &item)]
            } else {
                [a, b]
            }
        };

        let frame = with_item(empty_frame(&mut h, name), &item);
        let want = paint_single(&mut h, &frame);
        let frame = empty_frame(&mut h, name);
        let got = paint_single(&mut h, &frame);
        assert_eq!(h.stats().frame_path, FramePath::OverlaysOnly);
        assert_same(&format!("{name} host pass"), &want, &got);

        let mut frames = pair(&mut h, true);
        let want = paint_side_by_side(&mut h, &mut frames);
        let mut frames = pair(&mut h, false);
        let got = paint_side_by_side(&mut h, &mut frames);
        assert_same(&format!("{name} host pass, two viewports"), &want, &got);

        let mut frames = pair(&mut h, true);
        let (want, want_paths) = render_split(&mut h, &mut frames);
        let mut frames = pair(&mut h, false);
        let (got, got_paths) = render_split(&mut h, &mut frames);
        assert_eq!(want_paths, [FramePath::Ldr, FramePath::Ldr]);
        assert_eq!(
            got_paths,
            [FramePath::OverlaysOnly, FramePath::OverlaysOnly]
        );
        for k in 0..2 {
            assert_same(&format!("{name} split viewport {k}"), &want[k], &got[k]);
        }
    }
}
