//! What an overlays-only frame runs, read from the renderer's frame spans.
//!
//! An empty scene on the LDR path encodes the overlay pass and nothing of
//! scene prepare, the scene pass or the post chain; one mesh brings the scene
//! pass back.
//!
//! On its own in this binary because tracing caches callsite interest process
//! wide: a test on another thread reaching a span before this one installs its
//! scoped subscriber can leave the span filtered out.

use std::sync::{Arc, Mutex};

use tracing_subscriber::layer::SubscriberExt;
use viewport_lib::{FramePath, PipelineMode};
use viewport_lib_testkit::scenes::overlays;
use viewport_lib_testkit::{Harness, frame_for};

const W: u32 = 128;
const H: u32 = 96;

/// Records the name of every span opened on the frame target.
struct Spans(Arc<Mutex<Vec<&'static str>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Spans {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let meta = attrs.metadata();
        if meta.target() == "viewport_lib::frame" {
            self.0.lock().unwrap().push(meta.name());
        }
    }
}

/// The scene and post spans an overlays-only frame must not open.
const SCENE_SPANS: &[&str] = &[
    "lighting",
    "uniforms",
    "instancing",
    "geometry",
    "shadow",
    "scene_pass",
    "ldr_scene_pass",
    "oit_pass",
    "post_pass",
    "post_effects",
];

/// Render `frame` and return the path it took and the spans it opened.
fn record(h: &mut Harness, frame: &viewport_lib::FrameData) -> (FramePath, Vec<&'static str>) {
    let spans = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(Spans(spans.clone()));
    tracing::subscriber::with_default(subscriber, || {
        h.render(frame, W, H);
    });
    let seen = spans.lock().unwrap().clone();
    (h.stats().frame_path, seen)
}

#[test]
fn overlays_only_frame_runs_no_scene_or_post_passes() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = overlays::scenes()
        .into_iter()
        .find(|s| s.name == "overlay_shapes")
        .unwrap();
    let make = |h: &mut Harness, with_mesh: bool| {
        let mut built = h.build_scene(&scene);
        if !with_mesh {
            built.items.clear();
        }
        let mut frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        frame.effects.display.mode = PipelineMode::Direct;
        frame
    };
    let empty = make(&mut h, false);
    let mesh = make(&mut h, true);
    // Warm both frames first so pipeline builds do not show up as spans of
    // their own.
    h.render(&mesh, W, H);
    h.render(&empty, W, H);

    let (path, seen) = record(&mut h, &empty);
    assert_eq!(path, FramePath::OverlaysOnly);
    assert!(seen.contains(&"overlay_frame"), "{seen:?}");
    for name in SCENE_SPANS {
        assert!(
            !seen.contains(name),
            "an overlays-only frame ran {name}: {seen:?}"
        );
    }

    let (path, seen) = record(&mut h, &mesh);
    assert_eq!(path, FramePath::Ldr);
    assert!(seen.contains(&"ldr_scene_pass"), "{seen:?}");
    assert!(seen.contains(&"lighting"), "{seen:?}");
    assert!(!seen.contains(&"overlay_frame"), "{seen:?}");

    let mut hdr = make(&mut h, false);
    hdr.effects.display.mode = PipelineMode::Hdr;
    h.render(&hdr, W, H);
    let (path, seen) = record(&mut h, &hdr);
    assert_eq!(path, FramePath::Hdr);
    assert!(seen.contains(&"scene_pass"), "{seen:?}");
}
