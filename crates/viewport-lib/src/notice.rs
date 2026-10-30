//! Notices the renderer reports while it keeps drawing, and the one
//! rate-limited log helper behind them.
//!
//! The data types live in `viewport-lib-types` and are re-exported here. Read
//! the reported notices with
//! [`ViewportRenderer::take_notices`](crate::ViewportRenderer::take_notices),
//! filter them with
//! [`set_notice_level`](crate::ViewportRenderer::set_notice_level), and count
//! the waiting ones through
//! [`FrameStats::notices`](crate::renderer::stats::FrameStats::notices).
//!
//! Each notice is also logged on the `viewport_lib::notice` tracing target: the
//! first report of a condition, then at most one "repeated N times" line per
//! condition every five seconds.

pub use viewport_lib_types::notice::{
    Notice, NoticeCategory, NoticeCounts, NoticeDetail, Severity,
};

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use web_time::{Duration, Instant};

/// Notices kept between takes. Past this the oldest is pushed out and counted
/// in `NoticeCounts::dropped`.
pub(crate) const RING_CAPACITY: usize = 256;

/// The shortest gap between two log lines for one condition.
const LOG_INTERVAL: Duration = Duration::from_secs(5);

/// Conditions whose log timing is remembered. Past this, entries idle for a
/// whole interval are forgotten.
const LOG_STATE_CAPACITY: usize = 1024;

/// What makes two reports the same condition: the category, the kind of
/// notice, and up to two ids that tell instances apart (a texture and its
/// slot, a mesh index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct NoticeKey {
    category: NoticeCategory,
    kind: &'static str,
    ids: [u64; 2],
}

impl NoticeKey {
    pub(crate) const fn new(category: NoticeCategory, kind: &'static str, ids: [u64; 2]) -> Self {
        Self {
            category,
            kind,
            ids,
        }
    }

    /// A key with no ids: one notice for the whole renderer.
    pub(crate) const fn single(category: NoticeCategory, kind: &'static str) -> Self {
        Self::new(category, kind, [0, 0])
    }
}

/// Stable hash of a string id, for keys that identify an instance by name.
pub(crate) fn name_id(name: &str) -> u64 {
    // FNV-1a: stable across runs and cheap for short names.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[derive(Debug, Clone, Copy)]
struct LogState {
    last: Instant,
    since_frame: u64,
    unlogged: u64,
}

#[derive(Default)]
struct SinkInner {
    ring: VecDeque<(NoticeKey, Notice)>,
    dropped: u32,
    logged: HashMap<NoticeKey, LogState>,
}

/// The renderer's notice buffer: a ring of [`RING_CAPACITY`] coalesced notices.
///
/// Reporting takes `&self`, so the paint path can report too. The lock is
/// taken only when a report passes the level check, and the payload is built
/// only for a condition not already in the ring.
pub(crate) struct NoticeSink {
    level: AtomicU8,
    frame: AtomicU64,
    inner: Mutex<SinkInner>,
}

impl Default for NoticeSink {
    fn default() -> Self {
        Self::new()
    }
}

fn severity_to_u8(s: Severity) -> u8 {
    match s {
        Severity::Info => 0,
        Severity::Warning => 1,
        Severity::Error => 2,
    }
}

fn severity_from_u8(v: u8) -> Severity {
    match v {
        0 => Severity::Info,
        1 => Severity::Warning,
        _ => Severity::Error,
    }
}

impl NoticeSink {
    pub(crate) fn new() -> Self {
        Self {
            level: AtomicU8::new(severity_to_u8(Severity::Info)),
            frame: AtomicU64::new(0),
            inner: Mutex::new(SinkInner::default()),
        }
    }

    /// Reports below `level` are ignored before their payload is built.
    pub(crate) fn set_level(&self, level: Severity) {
        self.level.store(severity_to_u8(level), Ordering::Relaxed);
    }

    pub(crate) fn level(&self) -> Severity {
        severity_from_u8(self.level.load(Ordering::Relaxed))
    }

    /// Advance the frame number stamped on reports.
    pub(crate) fn begin_frame(&self) {
        self.frame.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn frame(&self) -> u64 {
        self.frame.load(Ordering::Relaxed)
    }

    /// Report a condition. Repeats of `key` coalesce into one notice; `detail`
    /// is called only for the first report of `key` since the last take.
    pub(crate) fn report(
        &self,
        key: NoticeKey,
        severity: Severity,
        detail: impl FnOnce() -> NoticeDetail,
    ) {
        if severity < self.level() {
            return;
        }
        let frame = self.frame();
        let now = Instant::now();
        let line = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            let inner = &mut *inner;
            let idx = match inner.ring.iter().position(|(k, _)| *k == key) {
                Some(i) => {
                    let n = &mut inner.ring[i].1;
                    n.count += 1;
                    n.last_frame = frame;
                    n.severity = n.severity.max(severity);
                    i
                }
                None => {
                    if inner.ring.len() >= RING_CAPACITY {
                        inner.ring.pop_front();
                        inner.dropped = inner.dropped.saturating_add(1);
                    }
                    let notice = Notice::new(severity, key.category, detail(), frame);
                    inner.ring.push_back((key, notice));
                    inner.ring.len() - 1
                }
            };
            match inner.logged.get_mut(&key) {
                None => {
                    if inner.logged.len() >= LOG_STATE_CAPACITY {
                        inner
                            .logged
                            .retain(|_, s| now.duration_since(s.last) < LOG_INTERVAL);
                        if inner.logged.len() >= LOG_STATE_CAPACITY {
                            inner.logged.clear();
                        }
                    }
                    inner.logged.insert(
                        key,
                        LogState {
                            last: now,
                            since_frame: frame,
                            unlogged: 0,
                        },
                    );
                    Some(inner.ring[idx].1.detail.to_string())
                }
                Some(state) => {
                    state.unlogged += 1;
                    if now.duration_since(state.last) >= LOG_INTERVAL {
                        let line = format!(
                            "{} repeated {} times since frame {}",
                            key.kind, state.unlogged, state.since_frame
                        );
                        *state = LogState {
                            last: now,
                            since_frame: frame,
                            unlogged: 0,
                        };
                        Some(line)
                    } else {
                        None
                    }
                }
            }
        };
        if let Some(line) = line {
            emit(severity, key.kind, &line);
        }
    }

    /// Drain the ring, oldest first, and reset the dropped count.
    pub(crate) fn take(&self) -> Vec<Notice> {
        let Ok(mut inner) = self.inner.lock() else {
            return Vec::new();
        };
        inner.dropped = 0;
        inner.ring.drain(..).map(|(_, n)| n).collect()
    }

    /// The waiting notices by severity, and the dropped count.
    pub(crate) fn counts(&self) -> NoticeCounts {
        let Ok(inner) = self.inner.lock() else {
            return NoticeCounts::default();
        };
        let mut c = NoticeCounts {
            dropped: inner.dropped,
            ..NoticeCounts::default()
        };
        for (_, n) in &inner.ring {
            match n.severity {
                Severity::Info => c.info += 1,
                Severity::Warning => c.warning += 1,
                Severity::Error => c.error += 1,
            }
        }
        c
    }
}

fn emit(severity: Severity, kind: &str, line: &str) {
    match severity {
        Severity::Info => tracing::info!(target: "viewport_lib::notice", kind, "{line}"),
        Severity::Warning => tracing::warn!(target: "viewport_lib::notice", kind, "{line}"),
        Severity::Error => tracing::error!(target: "viewport_lib::notice", kind, "{line}"),
    }
}

/// Log `args` the first time `key` is seen in this process, then at most one
/// "repeated N times" line for `key` every five seconds. For sites with no
/// renderer to report a notice to.
pub(crate) fn log_rate_limited(key: &'static str, level: Severity, args: std::fmt::Arguments<'_>) {
    static STATE: std::sync::OnceLock<Mutex<HashMap<&'static str, (Instant, u64)>>> =
        std::sync::OnceLock::new();
    let now = Instant::now();
    let line = {
        let Ok(mut map) = STATE.get_or_init(Default::default).lock() else {
            return;
        };
        match map.get_mut(key) {
            None => {
                map.insert(key, (now, 0));
                args.to_string()
            }
            Some((last, unlogged)) => {
                *unlogged += 1;
                if now.duration_since(*last) < LOG_INTERVAL {
                    return;
                }
                let line = format!("{args} (repeated {unlogged} times since the last report)");
                *last = now;
                *unlogged = 0;
                line
            }
        }
    };
    emit(level, key, &line);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u64) -> NoticeKey {
        NoticeKey::new(NoticeCategory::Item, "test", [i, 0])
    }

    fn detail() -> NoticeDetail {
        NoticeDetail::PickCacheDisabled
    }

    #[test]
    fn repeats_of_a_key_coalesce() {
        let sink = NoticeSink::new();
        for _ in 0..3 {
            sink.begin_frame();
            sink.report(key(1), Severity::Warning, detail);
        }
        sink.report(key(2), Severity::Warning, detail);
        let taken = sink.take();
        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].count, 3);
        assert_eq!(taken[0].first_frame, 1);
        assert_eq!(taken[0].last_frame, 3);
        assert_eq!(taken[1].count, 1);
    }

    #[test]
    fn full_ring_drops_the_oldest_and_counts_it() {
        let sink = NoticeSink::new();
        let extra = 10;
        for i in 0..(RING_CAPACITY + extra) as u64 {
            sink.report(key(i), Severity::Warning, detail);
        }
        let counts = sink.counts();
        assert_eq!(counts.warning as usize, RING_CAPACITY);
        assert_eq!(counts.dropped as usize, extra);
        let taken = sink.take();
        assert_eq!(taken.len(), RING_CAPACITY);
        assert_eq!(sink.counts(), NoticeCounts::default());
    }

    #[test]
    fn level_filter_skips_payload_construction() {
        let sink = NoticeSink::new();
        sink.set_level(Severity::Error);
        let mut called = false;
        sink.report(key(1), Severity::Warning, || {
            called = true;
            detail()
        });
        assert!(!called, "a filtered report must not build its payload");
        assert!(sink.take().is_empty());

        sink.report(key(1), Severity::Error, detail);
        assert_eq!(sink.take().len(), 1);
    }

    #[test]
    fn coalesced_repeat_does_not_rebuild_payload() {
        let sink = NoticeSink::new();
        sink.report(key(1), Severity::Warning, detail);
        let mut called = false;
        sink.report(key(1), Severity::Warning, || {
            called = true;
            detail()
        });
        assert!(!called);
    }

    #[test]
    fn take_empties_the_ring() {
        let sink = NoticeSink::new();
        sink.report(key(1), Severity::Info, detail);
        sink.report(key(2), Severity::Error, detail);
        assert_eq!(
            sink.counts(),
            NoticeCounts {
                info: 1,
                warning: 0,
                error: 1,
                dropped: 0
            }
        );
        assert_eq!(sink.take().len(), 2);
        assert!(sink.take().is_empty());
        assert_eq!(sink.counts().total(), 0);
    }
}
