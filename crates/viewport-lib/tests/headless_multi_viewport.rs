//! The split multi-viewport API, driven the way a host drives it: one
//! `prepare_scene`, then `prepare_viewport` for every viewport, and only then
//! a render or paint for each. Every test gives each viewport different
//! per-viewport content and checks that each draws its own, not the content of
//! whichever viewport was prepared last.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::wgpu;
use viewport_lib::{
    ClipObject, Colour, GizmoAxis, GizmoMode, GroundPlaneMode, LabelItem, OverlayFill,
    OverlayShape, OverlayShapeItem, PipelineMode, RetainedOverlay, ViewportId,
};

mod common;
use common::*;

const W: u32 = 64;
const H: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn renderer(device: &wgpu::Device) -> ViewportRenderer {
    let mut renderer = ViewportRenderer::new(device, FORMAT);
    renderer.set_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    renderer
}

/// A frame for `vp` looking straight down -Z at the origin, black background,
/// grid and axes indicator off.
fn base_frame(vp: ViewportId, mode: PipelineMode) -> FrameData {
    let cam = Camera {
        center: glam::Vec3::ZERO,
        distance: 10.0,
        orientation: glam::Quat::IDENTITY,
        ..Camera::default()
    };
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = W as f32 / H as f32;
        rc
    };
    frame.camera.viewport_size = [W as f32, H as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.camera = frame.camera.with_viewport_id(vp);
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame.effects.display.mode = mode;
    frame
}

/// A white SDF rect covering the left or right half of the viewport.
fn half_rect(left: bool) -> OverlayShapeItem {
    let x = if left { 0.0 } else { W as f32 / 2.0 };
    OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [x, 0.0],
        [W as f32 / 2.0, H as f32],
    )
    .with_fill(OverlayFill::Solid(Colour::linear(1.0, 1.0, 1.0, 1.0)))
}

fn colour_target(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("multi_viewport_target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Read an RGBA8 texture back, rows unpadded.
fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (texture.width(), texture.height());
    let padded =
        (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("multi_viewport_staging"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        let start = row * padded as usize;
        out.extend_from_slice(&data[start..start + (w * 4) as usize]);
    }
    out
}

/// One RGBA8 image and the queries the tests make of it.
struct Image {
    px: Vec<u8>,
    w: u32,
}

impl Image {
    fn rgb(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.w + x) * 4) as usize;
        [self.px[i], self.px[i + 1], self.px[i + 2]]
    }

    /// Pixels in columns `x0..x1` matching `pred`.
    fn count(&self, x0: u32, x1: u32, pred: impl Fn([u8; 3]) -> bool) -> usize {
        let h = self.px.len() as u32 / (self.w * 4);
        (0..h)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .filter(|&(x, y)| pred(self.rgb(x, y)))
            .count()
    }

    fn lit_left(&self, pred: impl Fn([u8; 3]) -> bool + Copy) -> usize {
        self.count(0, self.w / 2, pred)
    }

    fn lit_right(&self, pred: impl Fn([u8; 3]) -> bool + Copy) -> usize {
        self.count(self.w / 2, self.w, pred)
    }
}

fn bright(c: [u8; 3]) -> bool {
    c.iter().all(|&v| v > 200)
}

fn any_lit(c: [u8; 3]) -> bool {
    c.iter().any(|&v| v > 60)
}

/// Prepare every viewport, then render each into its own target. Runs twice
/// so first-frame setup is behind the frame that is read.
fn render_all(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frames: &[(ViewportId, FrameData)],
) -> Vec<Image> {
    let targets: Vec<wgpu::Texture> = frames.iter().map(|_| colour_target(device, W, H)).collect();
    for _ in 0..2 {
        let (scene_fx, _) = frames[0].1.effects.split();
        let token = renderer
            .owned()
            .prepare_scene(device, queue, &frames[0].1, &scene_fx);
        for (id, f) in frames {
            renderer
                .owned()
                .prepare_viewport(device, queue, &token, *id, f);
        }
        for ((id, f), t) in frames.iter().zip(&targets) {
            let view = t.create_view(&wgpu::TextureViewDescriptor::default());
            let cmd = renderer
                .owned()
                .render_viewport(device, queue, &view, *id, f);
            queue.submit([cmd]);
        }
    }
    targets
        .iter()
        .map(|t| Image {
            px: read_back(device, queue, t),
            w: W,
        })
        .collect()
}

const MODES: [PipelineMode; 2] = [PipelineMode::Direct, PipelineMode::Hdr];

#[test]
fn overlay_shapes_draw_in_their_own_viewport() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for mode in MODES {
        let mut r = renderer(&device);
        let (a, b) = (r.create_viewport(&device), r.create_viewport(&device));
        let mut fa = base_frame(a, mode);
        fa.overlays.shapes = vec![half_rect(true)];
        let mut fb = base_frame(b, mode);
        fb.overlays.shapes = vec![half_rect(false)];
        let imgs = render_all(&mut r, &device, &queue, &[(a, fa), (b, fb)]);
        assert!(
            bright(imgs[0].rgb(W / 4, H / 2)) && !any_lit(imgs[0].rgb(W * 3 / 4, H / 2)),
            "{mode:?}: viewport A drew another viewport's shapes"
        );
        assert!(
            !any_lit(imgs[1].rgb(W / 4, H / 2)) && bright(imgs[1].rgb(W * 3 / 4, H / 2)),
            "{mode:?}: viewport B drew another viewport's shapes"
        );
    }
}

#[test]
fn z_ordered_overlays_draw_in_their_own_viewport() {
    // A non-zero z_order sends the overlays through the sorted draw list.
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut r = renderer(&device);
    let (a, b) = (r.create_viewport(&device), r.create_viewport(&device));
    let mut fa = base_frame(a, PipelineMode::Hdr);
    fa.overlays.shapes = vec![half_rect(true).with_z_order(3)];
    let mut fb = base_frame(b, PipelineMode::Hdr);
    fb.overlays.shapes = vec![half_rect(false).with_z_order(3)];
    let imgs = render_all(&mut r, &device, &queue, &[(a, fa), (b, fb)]);
    assert!(bright(imgs[0].rgb(W / 4, H / 2)) && !any_lit(imgs[0].rgb(W * 3 / 4, H / 2)));
    assert!(!any_lit(imgs[1].rgb(W / 4, H / 2)) && bright(imgs[1].rgb(W * 3 / 4, H / 2)));
}

#[test]
fn labels_draw_in_their_own_viewport() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let label = |x: f32| {
        LabelItem::new("M")
            .with_position([x, 20.0])
            .with_font_size(16.0)
            .with_colour(Colour::linear(1.0, 1.0, 1.0, 1.0))
    };
    let mut r = renderer(&device);
    let (a, b) = (r.create_viewport(&device), r.create_viewport(&device));
    let mut fa = base_frame(a, PipelineMode::Hdr);
    fa.overlays.labels = vec![label(4.0)];
    let mut fb = base_frame(b, PipelineMode::Hdr);
    fb.overlays.labels = vec![label(W as f32 / 2.0 + 4.0)];
    let imgs = render_all(&mut r, &device, &queue, &[(a, fa), (b, fb)]);
    let (al, ar) = (imgs[0].lit_left(any_lit), imgs[0].lit_right(any_lit));
    let (bl, br) = (imgs[1].lit_left(any_lit), imgs[1].lit_right(any_lit));
    assert!(al > 20 && ar == 0, "viewport A label left={al} right={ar}");
    assert!(bl == 0 && br > 20, "viewport B label left={bl} right={br}");
}

#[test]
fn retained_overlays_draw_in_their_own_viewport() {
    // One compiled group, placed differently in each viewport.
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut r = renderer(&device);
    let square = OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [0.0, 0.0],
        [16.0, 16.0],
    )
    .with_fill(OverlayFill::Solid(Colour::linear(1.0, 1.0, 1.0, 1.0)));
    let id = r.compile_overlay_geometry(&device, &queue, &[], &[square], &[], &[], 1.0);
    let (a, b) = (r.create_viewport(&device), r.create_viewport(&device));
    let mut fa = base_frame(a, PipelineMode::Hdr);
    fa.overlays.retained = vec![RetainedOverlay::new(id).with_position([8.0, 24.0])];
    let mut fb = base_frame(b, PipelineMode::Hdr);
    fb.overlays.retained = vec![RetainedOverlay::new(id).with_position([40.0, 24.0])];
    let imgs = render_all(&mut r, &device, &queue, &[(a, fa), (b, fb)]);
    assert!(bright(imgs[0].rgb(16, 32)) && !any_lit(imgs[0].rgb(48, 32)));
    assert!(!any_lit(imgs[1].rgb(16, 32)) && bright(imgs[1].rgb(48, 32)));
}

#[test]
fn gizmo_draws_in_its_own_viewport() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let gizmo = |frame: &mut FrameData, x: f32| {
        frame.interaction.gizmo_model = Some(glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(1.0),
            glam::Quat::IDENTITY,
            glam::Vec3::new(x, 0.0, 0.0),
        ));
        frame.interaction.gizmo_mode = GizmoMode::Translate;
        frame.interaction.gizmo_hovered = GizmoAxis::None;
    };
    for mode in MODES {
        let mut r = renderer(&device);
        let (a, b) = (r.create_viewport(&device), r.create_viewport(&device));
        let mut fa = base_frame(a, mode);
        gizmo(&mut fa, -2.5);
        let mut fb = base_frame(b, mode);
        gizmo(&mut fb, 2.5);
        let imgs = render_all(&mut r, &device, &queue, &[(a, fa), (b, fb)]);
        let (al, ar) = (imgs[0].lit_left(any_lit), imgs[0].lit_right(any_lit));
        let (bl, br) = (imgs[1].lit_left(any_lit), imgs[1].lit_right(any_lit));
        assert!(
            al > 20 && ar < al / 4,
            "{mode:?}: viewport A gizmo left={al} right={ar}"
        );
        assert!(
            br > 20 && bl < br / 4,
            "{mode:?}: viewport B gizmo left={bl} right={br}"
        );
    }
}

#[test]
fn ground_plane_follows_its_own_viewport() {
    // Same camera in both; the ground sits below A's camera and above B's, so
    // A sees it and B does not.
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let ground = |frame: &mut FrameData, height: f32| {
        let gp = &mut frame.effects.ground_plane;
        gp.mode = GroundPlaneMode::SolidColour;
        gp.colour = Colour::linear(1.0, 0.0, 0.0, 1.0);
        gp.height = height;
    };
    let red = |c: [u8; 3]| c[0] > 120 && c[1] < 60 && c[2] < 60;
    for mode in MODES {
        let mut r = renderer(&device);
        let (a, b) = (r.create_viewport(&device), r.create_viewport(&device));
        let mut fa = base_frame(a, mode);
        ground(&mut fa, 0.0);
        let mut fb = base_frame(b, mode);
        ground(&mut fb, 20.0);
        let imgs = render_all(&mut r, &device, &queue, &[(a, fa), (b, fb)]);
        assert!(
            red(imgs[0].rgb(W / 2, H / 2)),
            "{mode:?}: viewport A lost its ground plane: {:?}",
            imgs[0].rgb(W / 2, H / 2)
        );
        assert!(
            !red(imgs[1].rgb(W / 2, H / 2)),
            "{mode:?}: viewport B drew a ground plane above its camera"
        );
    }
}

#[test]
fn paint_viewport_draws_each_viewports_overlays_in_one_pass() {
    // Two viewports side by side in one host-owned pass, each with its own
    // viewport rect, as an LDR quad layout paints them.
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut r = renderer(&device);
    let (a, b) = (r.create_viewport(&device), r.create_viewport(&device));
    let mut fa = base_frame(a, PipelineMode::Direct);
    fa.overlays.shapes = vec![half_rect(true)];
    let mut fb = base_frame(b, PipelineMode::Direct);
    fb.overlays.shapes = vec![half_rect(false)];
    let frames = [(a, fa), (b, fb)];

    let target = colour_target(&device, W * 2, H);
    let colour = target.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("multi_viewport_depth"),
            size: wgpu::Extent3d {
                width: W * 2,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());

    for _ in 0..2 {
        let (scene_fx, _) = frames[0].1.effects.split();
        let token = r
            .pass()
            .prepare_scene(&device, &queue, &frames[0].1, &scene_fx);
        for (id, f) in &frames {
            r.pass().prepare_viewport(&device, &queue, &token, *id, f);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multi_viewport_paint"),
        });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("multi_viewport_paint_pass"),
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
            for (i, (id, f)) in frames.iter().enumerate() {
                let x = (i as u32 * W) as f32;
                rp.set_viewport(x, 0.0, W as f32, H as f32, 0.0, 1.0);
                rp.set_scissor_rect(i as u32 * W, 0, W, H);
                r.pass().paint_viewport(&mut rp, *id, f);
            }
        }
        queue.submit([encoder.finish()]);
    }
    let img = Image {
        px: read_back(&device, &queue, &target),
        w: W * 2,
    };
    // A's rect is its own left half, B's its own right half.
    assert!(bright(img.rgb(W / 4, H / 2)), "A's overlay missing");
    assert!(!any_lit(img.rgb(W * 3 / 4, H / 2)), "A drew B's overlay");
    assert!(!any_lit(img.rgb(W + W / 4, H / 2)), "B drew A's overlay");
    assert!(bright(img.rgb(W + W * 3 / 4, H / 2)), "B's overlay missing");
}

#[test]
fn gpu_pick_uses_the_picked_viewports_clip_volumes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut r = renderer(&device);
    let mesh = r
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let (a, b) = (r.create_viewport(&device), r.create_viewport(&device));
    let with_box = |vp: ViewportId| {
        let mut frame = base_frame(vp, PipelineMode::Hdr);
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.settings.pick_id = PickId(7);
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
        frame
    };
    // A clips with a sphere away from the box; B does not clip.
    let mut fa = with_box(a);
    fa.effects
        .clip
        .objects
        .push(ClipObject::sphere([50.0, 0.0, 0.0], 1.0));
    let fb = with_box(b);
    let centre = glam::Vec2::new(W as f32 / 2.0, H as f32 / 2.0);

    let prepare = |r: &mut ViewportRenderer, frames: &[(ViewportId, &FrameData)]| {
        let (scene_fx, _) = frames[0].1.effects.split();
        let token = r
            .owned()
            .prepare_scene(&device, &queue, frames[0].1, &scene_fx);
        for (id, f) in frames {
            r.owned().prepare_viewport(&device, &queue, &token, *id, f);
        }
    };
    let pick = |r: &mut ViewportRenderer, f: &FrameData| {
        r.pick_scene_gpu(&device, &queue, centre, f)
            .map(|h| h.object_id)
    };

    // Each viewport alone, to know what its own clip does to the pick.
    prepare(&mut r, &[(a, &fa)]);
    let a_alone = pick(&mut r, &fa);
    prepare(&mut r, &[(b, &fb)]);
    let b_alone = pick(&mut r, &fb);
    assert_ne!(
        a_alone, b_alone,
        "the clip volume must change the pick for this test to mean anything"
    );

    // Both prepared, B last: picking A must still honour A's clip.
    prepare(&mut r, &[(a, &fa), (b, &fb)]);
    assert_eq!(pick(&mut r, &fa), a_alone);
    // And A last: picking B must not inherit A's clip.
    prepare(&mut r, &[(b, &fb), (a, &fa)]);
    assert_eq!(pick(&mut r, &fb), b_alone);
}
