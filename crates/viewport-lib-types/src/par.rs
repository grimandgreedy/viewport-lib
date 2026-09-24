//! CPU parallelism, and what it degrades to where there are no threads.
//!
//! Native targets get rayon. `wasm32-unknown-unknown` has no OS threads, so
//! rayon cannot build its pool there: the first parallel operation panics while
//! trying to spawn, which is a crash rather than a slowdown. This module is the
//! one place that difference is decided, so call sites read the same on both
//! targets and nothing else in the workspace has to know.
//!
//! On wasm every operation here runs serially on the calling thread. The work
//! still completes and the results are identical; it just takes as long as it
//! takes. Anything that would block a frame for too long belongs behind the
//! upload job system's frame budget, not here.
//!
//! Use it in place of rayon's prelude:
//!
//! ```ignore
//! use viewport_lib_types::par::*;
//!
//! let out: Vec<u16> = input.par_iter().map(|&f| convert(f)).collect();
//! ```
//!
//! Real wasm threads (`wasm-bindgen-rayon` plus the `atomics` target feature)
//! would replace the serial half of this module. They also need
//! cross-origin-isolation headers on the page, which a library cannot impose on
//! whoever embeds it, so the serial path stays the default either way.

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    pub use rayon::current_num_threads;
    pub use rayon::prelude::*;

    /// Run `work` on the thread pool, returning immediately.
    pub fn spawn<F: FnOnce() + Send + 'static>(work: F) {
        rayon::spawn(work);
    }

    /// Whether the calling target can actually run work off the calling thread.
    pub const fn is_threaded() -> bool {
        true
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    /// One "thread", so any call sizing its work by the pool asks for one unit.
    pub fn current_num_threads() -> usize {
        1
    }

    /// Run `work` now, on the calling thread, and return once it is done.
    ///
    /// The native version returns immediately and the work lands later. There is
    /// no off-thread anywhere to put it here, so a caller that treats this as
    /// fire-and-forget gets the work done sooner than it expected rather than
    /// not at all. Callers that care about blocking should check
    /// [`is_threaded`] and split the work themselves.
    pub fn spawn<F: FnOnce() + Send + 'static>(work: F) {
        work();
    }

    /// Whether the calling target can actually run work off the calling thread.
    pub const fn is_threaded() -> bool {
        false
    }

    /// The slice entry points rayon adds, each returning the ordinary iterator.
    ///
    /// Every adapter chained after one of these (`map`, `enumerate`,
    /// `for_each`, `collect`) is then the `Iterator` method of the same name,
    /// so a call site written against rayon compiles unchanged. Order is
    /// preserved, which rayon's indexed iterators also guarantee.
    pub trait ParallelSliceExt<T> {
        fn par_iter(&self) -> core::slice::Iter<'_, T>;
        fn par_iter_mut(&mut self) -> core::slice::IterMut<'_, T>;
        fn par_chunks_mut(&mut self, size: usize) -> core::slice::ChunksMut<'_, T>;
        fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
        where
            F: FnMut(&T) -> K,
            K: Ord;
    }

    impl<T> ParallelSliceExt<T> for [T] {
        fn par_iter(&self) -> core::slice::Iter<'_, T> {
            self.iter()
        }

        fn par_iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
            self.iter_mut()
        }

        fn par_chunks_mut(&mut self, size: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(size)
        }

        fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
        where
            F: FnMut(&T) -> K,
            K: Ord,
        {
            self.sort_unstable_by_key(f);
        }
    }

    /// `flat_map_iter` is rayon's name for the flattening adapter whose closure
    /// returns a sequential iterator. `Iterator` spells the same thing
    /// `flat_map`, so this is the one adapter that needs a name rather than a
    /// different implementation.
    pub trait FlatMapIterExt: Iterator + Sized {
        fn flat_map_iter<U, F>(self, f: F) -> core::iter::FlatMap<Self, U, F>
        where
            U: IntoIterator,
            F: FnMut(Self::Item) -> U,
        {
            self.flat_map(f)
        }
    }

    impl<I: Iterator> FlatMapIterExt for I {}
}

pub use imp::*;
