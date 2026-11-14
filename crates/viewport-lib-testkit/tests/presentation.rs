//! Reference-resolution presentation: the scene drawn into a target of a fixed
//! size and scaled into the viewport with nearest sampling.
//!
//! - A 320 x 180 reference in a 1920 x 1080 viewport is pixel-identical to
//!   the same scene drawn at 320 x 180 and scaled six times on the CPU, on
//!   both pipeline paths, whatever the render scale.
//! - In a 1366 x 768 viewport it draws four times over, centred, with bars of
//!   the chosen colour, and the crop and stretch fits cover the viewport.
//! - Picks take viewport positions: the GPU and CPU picks hit the item drawn
//!   under a viewport pixel and nothing in the bars.
//! - Screen overlays draw at the viewport's resolution unless asked into the
//!   reference target, and one that blurs its backdrop blurs the upscale.
//! - The `pass()` path presents the same image as the owned path.
//! - The catalogue's `presentation_reference` scene on the compatibility
//!   tier, against a golden of its own:
//!
//!     BLESS=1 cargo test --release --test presentation

use std::path::Path;
use viewport_lib::wgpu;
use viewport_lib::{
    Colour, FrameData, LabelItem, OverlayResolution, PickBackend, PickId, PickMask, PipelineMode,
    Presentation, ReferenceFit, Tier,
};
use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::scenes::presentation::{BAR_COLOUR, camera};
use viewport_lib_testkit::{BuiltScene, DeviceProfile, Harness, catalogue, frame_for};

const REFERENCE: [u32; 2] = [320, 180];

/// The catalogue's presentation scene, built on `h`, with the yellow square
/// pickable.
fn built(h: &mut Harness) -> BuiltScene {
    let scenes = catalogue();
    let scene = scenes
        .iter()
        .find(|s| s.name == "presentation_reference")
        .expect("presentation_reference is in the catalogue");
    let mut built = h.build_scene(scene);
    built.items[2].settings.pick_id = PickId(7);
    built.overlays = Default::default();
    built
}

/// The scene at `REFERENCE` under `presentation`, in a viewport of `size`.
fn frame(
    built: &BuiltScene,
    size: [u32; 2],
    presentation: Presentation,
    mode: PipelineMode,
) -> FrameData {
    let mut camera = camera();
    camera.set_aspect_ratio(REFERENCE[0] as f32, REFERENCE[1] as f32);
    let mut frame = frame_for(built, &camera, [size[0] as f32, size[1] as f32]);
    frame.viewport.presentation = presentation;
    frame.effects.display.mode = mode;
    frame
}

/// Drawn twice, so the second frame has every pipeline built.
fn draw(h: &mut Harness, frame: &FrameData, size: [u32; 2]) -> Vec<u8> {
    let _ = h.render(frame, size[0], size[1]);
    h.render(frame, size[0], size[1])
}

fn pixel(img: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [img[i], img[i + 1], img[i + 2], img[i + 3]]
}

/// `img` (`REFERENCE` sized) scaled `scale` times with nearest sampling and
/// placed at `origin` in a `size` image, the rest `bars`.
fn upscale(img: &[u8], scale: u32, origin: [i32; 2], size: [u32; 2], bars: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity((size[0] * size[1] * 4) as usize);
    for y in 0..size[1] as i32 {
        for x in 0..size[0] as i32 {
            let (rx, ry) = (x - origin[0], y - origin[1]);
            let inside = rx >= 0
                && ry >= 0
                && rx < (REFERENCE[0] * scale) as i32
                && ry < (REFERENCE[1] * scale) as i32;
            out.extend_from_slice(&if inside {
                pixel(img, REFERENCE[0], rx as u32 / scale, ry as u32 / scale)
            } else {
                bars
            });
        }
    }
    out
}

fn first_difference(a: &[u8], b: &[u8], width: u32) -> Option<(u32, u32, [u8; 4], [u8; 4])> {
    (0..a.len() / 4).find_map(|i| {
        let (x, y) = (i as u32 % width, i as u32 / width);
        let (p, q) = (pixel(a, width, x, y), pixel(b, width, x, y));
        (p != q).then_some((x, y, p, q))
    })
}

const BARS: ReferenceFit = ReferenceFit::IntegerBars { colour: BAR_COLOUR };

#[test]
fn a_reference_target_is_a_nearest_upscale_of_the_native_render() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let built = built(&mut h);
    let bars = BAR_COLOUR.to_srgb_u8();
    for mode in [PipelineMode::Hdr, PipelineMode::Direct] {
        let native = draw(
            &mut h,
            &frame(&built, REFERENCE, Presentation::Native, mode),
            REFERENCE,
        );

        let full = [1920, 1080];
        let reference = Presentation::reference(REFERENCE, BARS);
        let got = draw(&mut h, &frame(&built, full, reference, mode), full);
        let want = upscale(&native, 6, [0, 0], full, bars);
        if let Some(d) = first_difference(&got, &want, full[0]) {
            panic!("{mode:?} 1920x1080: first difference {d:?}");
        }

        // The render scale does not apply to the reference target.
        h.renderer.set_render_scale(0.5);
        let scaled = draw(&mut h, &frame(&built, full, reference, mode), full);
        h.renderer.set_render_scale(1.0);
        assert!(
            first_difference(&scaled, &want, full[0]).is_none(),
            "{mode:?}: the render scale changed the reference target"
        );

        let small = [1366, 768];
        let got = draw(&mut h, &frame(&built, small, reference, mode), small);
        let want = upscale(&native, 4, [43, 24], small, bars);
        if let Some(d) = first_difference(&got, &want, small[0]) {
            panic!("{mode:?} 1366x768: first difference {d:?}");
        }
        assert_eq!(pixel(&got, small[0], 5, 5), bars, "{mode:?}: corner bar");

        let crop = Presentation::reference(REFERENCE, ReferenceFit::IntegerCrop);
        let got = draw(&mut h, &frame(&built, small, crop, mode), small);
        let want = upscale(&native, 5, [-117, -66], small, [0; 4]);
        if let Some(d) = first_difference(&got, &want, small[0]) {
            panic!("{mode:?} crop: first difference {d:?}");
        }

        // A stretch covers the viewport: its corners are the reference's.
        let stretch = Presentation::reference(REFERENCE, ReferenceFit::Stretch);
        let got = draw(&mut h, &frame(&built, small, stretch, mode), small);
        let corners = [(0, 0, 0, 0), (1365, 767, 319, 179), (1365, 0, 319, 0)];
        for (x, y, rx, ry) in corners {
            assert_eq!(
                pixel(&got, small[0], x, y),
                pixel(&native, REFERENCE[0], rx, ry),
                "{mode:?} stretch at ({x}, {y})"
            );
        }
    }
}

#[test]
fn picks_take_viewport_positions() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    h.renderer.set_cpu_pick_cache(true);
    let built = built(&mut h);
    let size = [1366, 768];
    let frame = frame(
        &built,
        size,
        Presentation::reference(REFERENCE, BARS),
        PipelineMode::Hdr,
    );
    let _ = draw(&mut h, &frame, size);

    // The camera shows 60 world units over the reference's 180 pixels, so the
    // yellow square, centred on world (0, 12), is centred on reference pixel
    // (160, 54).
    let square = h
        .renderer
        .reference_to_window(&frame, glam::Vec2::new(160.5, 54.5))
        .expect("the square is inside the viewport");
    assert_eq!(square, glam::Vec2::new(43.0 + 642.0, 24.0 + 218.0));
    let bar = glam::Vec2::new(10.0, 300.0);
    assert_eq!(h.renderer.window_to_reference(&frame, bar), None);
    let plate = glam::Vec2::new(683.0, 600.0);

    for backend in [PickBackend::Gpu, PickBackend::Cpu] {
        let mut pick = |at: glam::Vec2| {
            let (device, queue) = (&h.device, &h.queue);
            h.renderer
                .pick_object(backend, at, &frame, device, queue, PickMask::OBJECT)
                .map(|hit| hit.id)
        };
        assert_eq!(pick(square), Some(7), "{backend:?}: the square");
        assert_eq!(pick(bar), None, "{backend:?}: the bars");
        assert_ne!(pick(plate), Some(7), "{backend:?}: beside the square");
    }

    // The CPU pick that takes its own viewport size maps through the last
    // prepared frame's fit.
    let view_proj = frame.camera.render_camera.view_proj();
    let viewport = glam::Vec2::new(size[0] as f32, size[1] as f32);
    let hit = h
        .renderer
        .pick(square, viewport, view_proj, PickMask::OBJECT);
    assert_eq!(hit.map(|hit| hit.id), Some(7));
    assert!(
        h.renderer
            .pick(bar, viewport, view_proj, PickMask::OBJECT)
            .is_none()
    );
    // A rectangle in viewport positions selects what is drawn inside it.
    let rect = h.renderer.pick_rect_objects(
        PickBackend::Gpu,
        square - glam::Vec2::splat(20.0),
        square + glam::Vec2::splat(20.0),
        &frame,
        &h.device,
        &h.queue,
        PickMask::OBJECT,
    );
    assert!(rect.objects.contains(&7), "rect pick: {:?}", rect.objects);
    let rect = h.renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(40.0, 20.0),
        &frame,
        &h.device,
        &h.queue,
        PickMask::OBJECT,
    );
    assert!(
        rect.objects.is_empty(),
        "a rectangle in the bars: {:?}",
        rect.objects
    );
}

/// Whether every aligned `block` x `block` square of `img` is one colour.
fn blocky(img: &[u8], size: [u32; 2], block: u32) -> bool {
    (0..size[1]).step_by(block as usize).all(|by| {
        (0..size[0]).step_by(block as usize).all(|bx| {
            let first = pixel(img, size[0], bx, by);
            (by..by + block).all(|y| (bx..bx + block).all(|x| pixel(img, size[0], x, y) == first))
        })
    })
}

#[test]
fn screen_overlays_draw_at_the_viewport_resolution() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let built = built(&mut h);
    let full = [1920, 1080];
    let mut frame = frame(
        &built,
        full,
        Presentation::reference(REFERENCE, BARS),
        PipelineMode::Hdr,
    );
    frame.overlays.labels.push(
        LabelItem::new("Score 1280")
            .with_position([400.0, 300.0])
            .with_fill(viewport_lib::OverlayFill::Solid(Colour::WHITE)),
    );
    let native = draw(&mut h, &frame, full);
    assert!(
        !blocky(&native, full, 6),
        "a label over the upscale has edges finer than a reference pixel"
    );

    frame.viewport.presentation = Presentation::Reference {
        size: REFERENCE,
        fit: BARS,
        overlays: OverlayResolution::Reference,
    };
    let inside = draw(&mut h, &frame, full);
    assert!(
        blocky(&inside, full, 6),
        "a label drawn into the reference target scales with it"
    );
    assert_ne!(native, inside);
}

#[test]
fn screen_overlays_over_the_upscale_read_what_it_drew() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let built = built(&mut h);
    let full = [1920, 1080];
    let mut frame = frame(
        &built,
        full,
        Presentation::reference(REFERENCE, BARS),
        PipelineMode::Hdr,
    );
    let plain = draw(&mut h, &frame, full);
    // A clear panel blurring what is under it, across the red square's right
    // edge at x = 528.
    frame.overlays.shapes.push(
        viewport_lib::OverlayShapeItem::new(
            viewport_lib::OverlayShape::Rect { corner_radius: 0.0 },
            [480.0, 300.0],
            [100.0, 60.0],
        )
        .with_fill(viewport_lib::OverlayFill::Solid(Colour::TRANSPARENT))
        .with_backdrop_blur(8.0),
    );
    let blurred = draw(&mut h, &frame, full);
    assert_ne!(
        pixel(&blurred, full[0], 528, 330),
        pixel(&plain, full[0], 528, 330),
        "the panel blurs the upscaled edge under it"
    );
    assert_eq!(
        pixel(&blurred, full[0], 1000, 900),
        pixel(&plain, full[0], 1000, 900),
        "away from the panel nothing changes"
    );
}

/// Copy a `Bgra8UnormSrgb` texture back as RGBA8.
fn read_back(h: &Harness, texture: &wgpu::Texture, size: [u32; 2]) -> Vec<u8> {
    let padded = (size[0] * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = h.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("presentation_readback"),
        size: (padded * size[1]) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = h
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size[1]),
            },
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );
    h.queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    h.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .expect("readback poll");
    let mut out = Vec::with_capacity((size[0] * size[1] * 4) as usize);
    {
        let mapped = viewport_lib::gpu::mapped_range(staging.slice(..));
        for row in 0..size[1] as usize {
            let start = row * padded as usize;
            for px in mapped[start..start + size[0] as usize * 4].chunks_exact(4) {
                out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
            }
        }
    }
    staging.unmap();
    out
}

#[test]
fn the_pass_path_presents_what_the_owned_path_does() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let built = built(&mut h);
    let size = [683, 384];
    for mode in [PipelineMode::Hdr, PipelineMode::Direct] {
        let frame = frame(&built, size, Presentation::reference(REFERENCE, BARS), mode);
        let owned = draw(&mut h, &frame, size);

        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let target = |format, usage| {
            h.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let colour = target(
            Harness::DEFAULT_TARGET_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = target(
            h.renderer.scene_depth_format(),
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        for _ in 0..2 {
            let buffers = h.renderer.pass().prepare(&h.device, &h.queue, &frame);
            h.queue.submit(buffers);
            let colour_view = colour.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let mut encoder = h
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
                    multiview_mask: None,
                    label: Some("host_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &colour_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                h.renderer.pass_view().paint(&mut pass, &frame);
            }
            h.queue.submit(std::iter::once(encoder.finish()));
        }
        let host = read_back(&h, &colour, size);
        if let Some(d) = first_difference(&host, &owned, size[0]) {
            panic!("{mode:?}: first difference {d:?}");
        }
    }
}

#[test]
fn compatibility_tier_presents_at_a_reference_resolution() {
    let Some(mut harness) = Harness::with_profile(&DeviceProfile::downlevel_webgl2(
        "presentation-compatibility",
    )) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), Tier::Compatibility);
    let scenes = catalogue();
    let scene = scenes
        .iter()
        .find(|s| s.name == "presentation_reference")
        .expect("presentation_reference is in the catalogue");
    let built = harness.build_scene(scene);
    let size = [400, 300];
    let frame = frame_for(&built, &scene.cameras[0].camera, [400.0, 300.0]);
    let pixels = draw(&mut harness, &frame, size);
    assert_eq!(
        pixel(&pixels, size[0], 5, 5),
        BAR_COLOUR.to_srgb_u8(),
        "the bars draw"
    );

    let snapshots = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
    let dir = snapshots.join(format!("{}-compatibility", harness.backend_dir_name()));
    let mode = GoldenMode::from_env();
    if mode != GoldenMode::Bless {
        let stamp_dir = snapshots.join(harness.backend_dir_name());
        if let Err(e) = golden::check_driver_stamp(&stamp_dir, harness.driver_info().unwrap_or(""))
        {
            panic!("{e}");
        }
    }
    let name = "presentation_reference";
    let tolerances = golden::load_tolerances(&dir.join("tolerances.txt"));
    let tolerance = tolerances.get(name).copied().unwrap_or(Tolerance::EXACT);
    let actual = RgbaImage::from_raw(size[0], size[1], pixels).expect("frame -> image");
    match golden::check(&dir, name, &actual, tolerance, mode) {
        Outcome::Blessed => eprintln!("blessed {name} in {}", dir.display()),
        Outcome::Rendered { path } => eprintln!("render only: {} (not compared)", path.display()),
        Outcome::Match => {}
        Outcome::Missing { .. } => panic!("{}", golden::missing_message(name, &dir)),
        Outcome::Mismatch { comparison, .. } => panic!(
            "{name}: max_channel_delta={} pixels_over={}/{}",
            comparison.max_channel_delta, comparison.pixels_over, comparison.total_pixels
        ),
    }
}
