//! The notice tracing bridge: a condition reported every frame is logged once.
//!
//! On its own in this binary because tracing caches callsite interest process
//! wide: a test on another thread hitting the callsite before this one installs
//! its scoped subscriber can leave the event filtered out.

use std::io::Write;
use std::sync::{Arc, Mutex};

use viewport_lib::{NoticeCategory, NoticeDetail, PipelineMode, Severity};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::fixtures::{
    CallLog, CountedItemCollection, LoggingItemTypePlugin, probe_frame,
};

const SIZE: u32 = 32;
const PLUGIN: &str = "ldr_skipped_items";

/// A `MakeWriter` that appends every log line to a shared buffer.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Captured {
    type Writer = Captured;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl Captured {
    fn lines_containing(&self, needle: &str) -> Vec<String> {
        String::from_utf8_lossy(&self.0.lock().unwrap())
            .lines()
            .filter(|l| l.contains(needle))
            .map(str::to_owned)
            .collect()
    }
}

/// A harness with a plugin that does not draw on the LDR pipeline, and an LDR
/// frame that submits items to it.
fn ldr_skip_setup() -> Option<(Harness, viewport_lib::renderer::FrameData)> {
    let mut harness = Harness::new()?;
    harness
        .renderer
        .with_item_type_plugin(
            &harness.device,
            Box::new(LoggingItemTypePlugin::new(CallLog::new(), PLUGIN)),
        )
        .unwrap();
    let mut frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    frame.effects.display.mode = PipelineMode::Direct;
    frame
        .scene
        .submit_plugin_items(PLUGIN, CountedItemCollection::new(2));
    Some((harness, frame))
}

/// A plugin skipped on the LDR pipeline for 100 frames is one notice with a
/// count of 100, and the tracing bridge logs it once.
#[test]
fn items_skipped_on_ldr_coalesce_over_frames() {
    let Some((mut harness, frame)) = ldr_skip_setup() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let captured = Captured::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(captured.clone())
        .with_ansi(false)
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "viewport_lib::notice=info",
        ))
        .finish();

    const FRAMES: u64 = 100;
    tracing::subscriber::with_default(subscriber, || {
        for _ in 0..FRAMES {
            harness.render(&frame, SIZE, SIZE);
        }
    });

    let notices = harness.renderer.take_notices();
    let skipped: Vec<_> = notices
        .iter()
        .filter(|n| matches!(n.detail, NoticeDetail::ItemsSkippedOnLdr { .. }))
        .collect();
    assert_eq!(skipped.len(), 1, "one coalesced notice; got {notices:?}");
    let n = skipped[0];
    assert_eq!(n.count, FRAMES);
    assert_eq!(n.severity, Severity::Warning);
    assert_eq!(n.category, NoticeCategory::Item);
    assert_eq!(n.last_frame - n.first_frame + 1, FRAMES);
    match &n.detail {
        NoticeDetail::ItemsSkippedOnLdr { plugin } => assert_eq!(plugin, PLUGIN),
        other => panic!("unexpected detail {other:?}"),
    }

    let lines = captured.lines_containing("items_skipped_on_ldr");
    assert_eq!(lines.len(), 1, "one log line; got {lines:#?}");
    assert!(lines[0].contains(PLUGIN), "{}", lines[0]);
}
