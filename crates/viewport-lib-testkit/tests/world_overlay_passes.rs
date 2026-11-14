//! Where a world overlay layer draws, read from the renderer's frame spans: in
//! the scene, after the scene pass and before the post effects and the screen
//! overlay pass on the HDR path, and inside the scene pass on the LDR path.
//!
//! On its own in this binary because tracing caches callsite interest process
//! wide: a test on another thread reaching a span before this one installs its
//! scoped subscriber can leave the span filtered out.

use std::sync::{Arc, Mutex};

use tracing_subscriber::layer::SubscriberExt;
use viewport_lib::{FramePath, PipelineMode};
use viewport_lib_testkit::scenes::world_overlay;
use viewport_lib_testkit::{Harness, frame_for};

const W: u32 = 400;
const H: u32 = 300;

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

fn record(h: &mut Harness, frame: &viewport_lib::FrameData) -> (FramePath, Vec<&'static str>) {
    let spans = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(Spans(spans.clone()));
    tracing::subscriber::with_default(subscriber, || {
        h.render(frame, W, H);
    });
    let seen = spans.lock().unwrap().clone();
    (h.stats().frame_path, seen)
}

fn first(seen: &[&str], name: &str) -> usize {
    seen.iter()
        .position(|s| *s == name)
        .unwrap_or_else(|| panic!("no {name} span: {seen:?}"))
}

#[test]
fn a_world_layer_draws_in_the_scene_not_the_overlay_pass() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = world_overlay::scenes().remove(0);
    let built = h.build_scene(&scene);
    let mut frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    // A screen overlay too, so the overlay pass runs and its place shows.
    frame
        .overlays
        .labels
        .push(viewport_lib::LabelItem::new("screen"));
    // Warm first so pipeline builds do not show up as spans of their own.
    h.render(&frame, W, H);

    let (path, seen) = record(&mut h, &frame);
    assert_eq!(path, FramePath::Hdr);
    let world = first(&seen, "world_overlay");
    assert!(first(&seen, "scene_pass") < world, "{seen:?}");
    assert!(world < first(&seen, "post_effects"), "{seen:?}");
    assert!(world < first(&seen, "overlay_pass"), "{seen:?}");
    assert_eq!(
        seen.iter().filter(|s| **s == "world_overlay").count(),
        1,
        "{seen:?}"
    );
    // The screen overlay's draws are the overlay pass's; the layer's are not
    // among them.
    assert_eq!(h.stats().overlay_draw_calls, 1);

    frame.effects.display.mode = PipelineMode::Direct;
    h.render(&frame, W, H);
    let (path, seen) = record(&mut h, &frame);
    assert_eq!(path, FramePath::Ldr);
    assert!(
        first(&seen, "ldr_scene_pass") < first(&seen, "world_overlay"),
        "{seen:?}"
    );
    assert!(!seen.contains(&"post_effects"), "{seen:?}");
}
