//! The overlay's shape and text streams are uploaded only when something they
//! are built from changes.
//!
//! Each case renders a frame until it settles, checks that one more unchanged
//! frame uploads nothing, then changes one input and checks that the next
//! frame uploads.

use viewport_lib::{
    AnimTrack, CameraFrame, Colour, FrameData, LabelItem, OverlayAnimations, OverlayClip,
    OverlayFill, OverlayLayerStyle, OverlayPolylineItem, OverlayShape, OverlayShapeItem,
    SceneFrame,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 200;
const H: u32 = 150;

fn frame() -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 5.0, 0.7, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]).with_pixels_per_point(1.0),
        SceneFrame::from_surface_items(Vec::new()),
    );
    fd.viewport.show_axes_indicator = false;
    let o = &mut fd.overlays;
    o.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 4.0 },
            [10.0, 10.0],
            [60.0, 40.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.4, 0.2, 1.0))),
    );
    o.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [40.0, 30.0],
            [60.0, 40.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.4, 0.9, 1.0))),
    );
    o.labels
        .push(LabelItem::new("label").with_screen_anchor([20.0, 100.0]));
    o.polylines.push(
        OverlayPolylineItem::new(vec![[10.0, 140.0], [90.0, 120.0], [170.0, 140.0]])
            .with_thickness(3.0),
    );
    fd
}

/// Render `fd` until it settles and check that an unchanged frame uploads
/// nothing.
fn settle(h: &mut Harness, fd: &FrameData, size: [u32; 2]) {
    for _ in 0..3 {
        h.render(fd, size[0], size[1]);
    }
    assert_eq!(
        h.stats().overlay_stream_uploads,
        0,
        "an unchanged frame uploads nothing"
    );
}

fn assert_uploads(h: &mut Harness, what: &str, fd: &FrameData, size: [u32; 2]) {
    h.render(fd, size[0], size[1]);
    assert!(
        h.stats().overlay_stream_uploads > 0,
        "{what}: the changed frame uploaded nothing"
    );
}

fn assert_skips(h: &mut Harness, what: &str, fd: &FrameData, size: [u32; 2]) {
    h.render(fd, size[0], size[1]);
    assert_eq!(
        h.stats().overlay_stream_uploads,
        0,
        "{what}: the frame uploaded though its streams did not change"
    );
}

/// `frame()` changed by `edit`.
fn edited(edit: impl FnOnce(&mut FrameData)) -> FrameData {
    let mut fd = frame();
    edit(&mut fd);
    fd
}

/// `fd` seen from a camera that has orbited a little.
fn orbited(fd: &mut FrameData) {
    let camera = orbit_camera(glam::Vec3::ZERO, 5.0, 0.9, 1.1);
    fd.camera = CameraFrame::from_camera(&camera, [W as f32, H as f32]).with_pixels_per_point(1.0);
}

/// Each change to an item or to the space it is laid out in uploads.
#[test]
fn a_changed_input_uploads() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let size = [W, H];
    let base = frame();
    let cases: Vec<(&str, FrameData, [u32; 2])> = vec![
        (
            "content",
            edited(|fd| {
                fd.overlays.shapes[0].style.fill =
                    OverlayFill::Solid(Colour::srgb(0.1, 0.8, 0.1, 1.0))
            }),
            size,
        ),
        (
            "transform",
            edited(|fd| fd.overlays.shapes[0].transform.translate = [12.0, 10.0]),
            size,
        ),
        (
            "z-order",
            edited(|fd| fd.overlays.shapes[0].z_order = 5),
            size,
        ),
        (
            "submission order",
            edited(|fd| fd.overlays.shapes.swap(0, 1)),
            size,
        ),
        (
            "clip",
            edited(|fd| fd.overlays.shapes[1].clip = OverlayClip::rect([0.0, 0.0, 70.0, 60.0])),
            size,
        ),
        (
            "text",
            edited(|fd| fd.overlays.labels[0].text = "lapel".into()),
            size,
        ),
        (
            "layer",
            edited(|fd| {
                let mark = fd.overlays.layer_mark();
                fd.overlays.shapes.push(OverlayShapeItem::new(
                    OverlayShape::Circle,
                    [120.0, 20.0],
                    [30.0, 30.0],
                ));
                fd.overlays
                    .push_layer(mark, OverlayLayerStyle::default().with_opacity(0.5));
            }),
            size,
        ),
        (
            "viewport size",
            edited(|fd| fd.camera.viewport_size = [(W + 20) as f32, H as f32]),
            [W + 20, H],
        ),
        (
            "pixels per point",
            edited(|fd| fd.camera.pixels_per_point = 2.0),
            [W * 2, H * 2],
        ),
    ];
    for (what, changed, at) in &cases {
        settle(&mut h, &base, size);
        assert_uploads(&mut h, what, changed, *at);
    }
}

/// The clock reaches the streams only through animations, and the camera
/// only through world anchors and world-sized text.
#[test]
fn time_and_camera_upload_only_when_read() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let size = [W, H];
    settle(&mut h, &frame(), size);
    let later = edited(|fd| fd.overlays.time = 1.5);
    assert_skips(&mut h, "time with nothing animated", &later, size);

    settle(&mut h, &frame(), size);
    assert_skips(&mut h, "camera under screen items", &edited(orbited), size);

    let animate = |fd: &mut FrameData| {
        let shape = fd.overlays.shapes[0].clone();
        fd.overlays.shapes[0] =
            shape.with_animations(OverlayAnimations::default().with_translate(AnimTrack::new(
                0.0,
                4.0,
                [10.0, 10.0],
                [50.0, 30.0],
            )));
    };
    settle(&mut h, &edited(animate), size);
    let later = edited(|fd| {
        animate(fd);
        fd.overlays.time = 1.0;
    });
    assert_uploads(&mut h, "time with an animation", &later, size);

    let anchor = |fd: &mut FrameData| {
        fd.overlays
            .labels
            .push(LabelItem::new("world").with_world_anchor(glam::DVec3::ZERO));
    };
    settle(&mut h, &edited(anchor), size);
    let moved = edited(|fd| {
        anchor(fd);
        orbited(fd);
    });
    assert_uploads(&mut h, "camera under a world anchor", &moved, size);
}

/// Changes to the renderer's state the streams are built from upload: a
/// fill texture's size, a new font, and glyphs packed for another viewport.
#[test]
fn changed_renderer_state_uploads() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let size = [W, H];
    let image = h
        .renderer
        .resources_mut()
        .upload_overlay_texture(
            &h.device,
            &h.queue,
            viewport_lib::TextureData::srgb(2, 2, vec![255u8; 16]),
        )
        .expect("a well-formed sRGB image");
    let fd = edited(|fd| fd.overlays.shapes[1].style.fill = OverlayFill::texture(image));
    settle(&mut h, &fd, size);

    // New pixels at the same size reach the shader, not the streams.
    let updated = h.renderer.resources_mut().update_overlay_texture(
        &h.device,
        &h.queue,
        image,
        2,
        2,
        &[128u8; 16],
    );
    assert!(updated);
    assert_skips(&mut h, "texture pixels", &fd, size);

    let updated = h.renderer.resources_mut().update_overlay_texture(
        &h.device,
        &h.queue,
        image,
        4,
        2,
        &[128u8; 32],
    );
    assert!(updated);
    assert_uploads(&mut h, "texture size", &fd, size);

    settle(&mut h, &fd, size);
    h.renderer
        .resources_mut()
        .upload_font(include_bytes!(
            "../../viewport-lib/src/fonts/Roboto-Regular.ttf"
        ))
        .expect("a font");
    assert_uploads(&mut h, "font", &fd, size);

    // Text in a second viewport packs new glyphs into the shared atlas.
    settle(&mut h, &fd, size);
    let other = edited(|fd| {
        fd.camera.viewport_index = 1;
        fd.overlays.labels = vec![
            LabelItem::new("ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789")
                .with_screen_anchor([5.0, 20.0])
                .with_font_size(31.0),
        ];
    });
    h.render(&other, W, H);
    assert!(h.stats().overlay_glyph_rasters > 0);
    assert_uploads(&mut h, "glyph atlas", &fd, size);
}
