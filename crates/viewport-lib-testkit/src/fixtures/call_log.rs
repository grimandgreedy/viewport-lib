//! The shared call log every fixture records into.

use std::sync::{Arc, Mutex};

/// A shared, cloneable list of callback records.
///
/// Fixtures push one line per callback: the callback name, plus whichever
/// context facts a test wants to assert (viewport index, target size, frame
/// index, plugin label). Clones share one buffer, so a test holds a clone
/// while the renderer or runtime owns the fixture.
#[derive(Clone, Default)]
pub struct CallLog {
    entries: Arc<Mutex<Vec<String>>>,
}

impl CallLog {
    /// An empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one callback.
    pub fn record(&self, entry: impl Into<String>) {
        self.entries.lock().unwrap().push(entry.into());
    }

    /// A snapshot of the entries recorded so far.
    pub fn entries(&self) -> Vec<String> {
        self.entries.lock().unwrap().clone()
    }

    /// Take the entries recorded so far, leaving the log empty. The usual way
    /// to assert one frame's worth of calls before rendering the next.
    pub fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.entries.lock().unwrap())
    }

    /// Drop everything recorded so far.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    /// How many entries start with `prefix`.
    pub fn count(&self, prefix: &str) -> usize {
        self.entries()
            .iter()
            .filter(|e| e.starts_with(prefix))
            .count()
    }

    /// The index of the first entry equal to `entry`, or `None`.
    pub fn position(&self, entry: &str) -> Option<usize> {
        self.entries().iter().position(|e| e == entry)
    }

    /// Assert the log holds exactly `expected`, in order, and clear it.
    ///
    /// # Panics
    ///
    /// When the entries differ, naming both sequences.
    pub fn assert_take(&self, expected: &[&str]) {
        let got = self.take();
        assert_eq!(
            got.as_slice(),
            expected,
            "call log mismatch: got {got:?}, expected {expected:?}"
        );
    }

    /// Assert nothing has been recorded.
    ///
    /// # Panics
    ///
    /// When the log is not empty, naming what it holds.
    pub fn assert_empty(&self) {
        let got = self.entries();
        assert!(got.is_empty(), "expected no calls, got {got:?}");
    }

    /// Assert `first` was recorded before `second` (both must be present).
    ///
    /// # Panics
    ///
    /// When either entry is missing, or they are in the wrong order.
    pub fn assert_before(&self, first: &str, second: &str) {
        let got = self.entries();
        let a = self
            .position(first)
            .unwrap_or_else(|| panic!("{first:?} was never recorded; log holds {got:?}"));
        let b = self
            .position(second)
            .unwrap_or_else(|| panic!("{second:?} was never recorded; log holds {got:?}"));
        assert!(
            a < b,
            "{first:?} must be recorded before {second:?}; log holds {got:?}"
        );
    }
}

impl std::fmt::Debug for CallLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CallLog").field(&self.entries()).finish()
    }
}
