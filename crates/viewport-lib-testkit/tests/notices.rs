//! The notice channel on a real renderer: `take_notices` drains what
//! `FrameStats::notices` counts, and the notice level filters reports.

use viewport_lib::{PipelineMode, Severity};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::fixtures::{
    CallLog, CountedItemCollection, LoggingItemTypePlugin, probe_frame,
};

const SIZE: u32 = 32;
const PLUGIN: &str = "ldr_skipped_items";

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

/// `FrameStats::notices` counts what `take_notices` returns, and taking empties
/// the buffer.
#[test]
fn take_notices_drains_what_frame_stats_counts() {
    let Some((mut harness, frame)) = ldr_skip_setup() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness.render(&frame, SIZE, SIZE);
    harness.render(&frame, SIZE, SIZE);

    let counts = harness.renderer.last_frame_stats().notices;
    let notices = harness.renderer.take_notices();
    assert!(!notices.is_empty());
    assert_eq!(counts.total() as usize, notices.len());
    for (severity, expected) in [
        (Severity::Info, counts.info),
        (Severity::Warning, counts.warning),
        (Severity::Error, counts.error),
    ] {
        let got = notices.iter().filter(|n| n.severity == severity).count();
        assert_eq!(got, expected as usize, "{severity:?}");
    }
    assert_eq!(counts.dropped, 0);

    assert!(harness.renderer.take_notices().is_empty());
    assert_eq!(harness.renderer.last_frame_stats().notices.total(), 0);
}

/// Below the notice level nothing is recorded.
#[test]
fn notice_level_filters_reports() {
    let Some((mut harness, frame)) = ldr_skip_setup() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness.renderer.set_notice_level(Severity::Error);
    harness.render(&frame, SIZE, SIZE);
    assert!(
        harness
            .renderer
            .take_notices()
            .iter()
            .all(|n| n.severity >= Severity::Error)
    );
}
