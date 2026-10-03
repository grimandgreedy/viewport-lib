//! A pipeline built the first time something needs it, either on the thread
//! that asked or on a worker.
//!
//! [`PipelineSlot`] holds one pipeline. Reading it hands back the pipeline
//! when it is ready. When it is not, the read starts the compile (if nobody
//! has) and, under [`PipelineCompilation::Background`], returns nothing so the
//! caller can skip the draw and come back next frame. Under
//! [`PipelineCompilation::Blocking`] the read compiles on the spot and never
//! returns nothing. [`PipelineCompiler`] carries the policy and counts the
//! compiles in flight, so an application can wait for them.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

/// How the renderer compiles a pipeline the first time a frame needs it.
///
/// Set with [`ViewportRenderer::set_pipeline_compilation`](crate::ViewportRenderer::set_pipeline_compilation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PipelineCompilation {
    /// Compile on a worker thread. The draw or pass that needs the pipeline
    /// is skipped until it is ready, so the frame that first uses something
    /// does not stall; the thing appears a few frames later. The count of
    /// compiles still running is
    /// [`pipelines_pending`](crate::ViewportRenderer::pipelines_pending).
    Background,
    /// Compile on the calling thread, inside `prepare` or the draw. Nothing is
    /// ever skipped, and a frame that first uses something takes as long as
    /// the compile does.
    Blocking,
}

impl PipelineCompilation {
    /// The default for this platform: `Blocking` on macOS, iOS and the web,
    /// `Background` everywhere else. A worker thread compiling under Metal
    /// stalls the thread that renders, and the web has no threads.
    pub fn platform_default() -> Self {
        if cfg!(any(
            target_os = "macos",
            target_os = "ios",
            target_family = "wasm"
        )) {
            Self::Blocking
        } else {
            Self::Background
        }
    }

    fn from_env(v: &str) -> Option<Self> {
        match v {
            "background" => Some(Self::Background),
            "blocking" => Some(Self::Blocking),
            _ => None,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Background,
            _ => Self::Blocking,
        }
    }
}

/// The policy a new renderer starts with: the platform default, unless
/// `VPL_PIPELINE_COMPILATION` is `background` or `blocking`. The variable
/// lets a test suite run blocking on a platform that defaults to workers
/// without each test saying so; `set_pipeline_compilation` still wins.
pub(crate) fn initial_policy() -> PipelineCompilation {
    std::env::var("VPL_PIPELINE_COMPILATION")
        .ok()
        .and_then(|v| PipelineCompilation::from_env(&v))
        .unwrap_or_else(PipelineCompilation::platform_default)
}

/// The compiles a renderer has in flight.
#[derive(Default)]
struct Pending {
    count: AtomicUsize,
}

impl Pending {
    fn add(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }

    fn finish(&self) {
        self.count.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Shared by every slot one renderer owns: the policy its slots follow and the
/// compiles they have running.
pub(crate) struct PipelineCompiler {
    policy: AtomicU8,
    pending: Arc<Pending>,
}

impl PipelineCompiler {
    pub(crate) fn new(policy: PipelineCompilation) -> Self {
        Self {
            policy: AtomicU8::new(policy as u8),
            pending: Arc::new(Pending::default()),
        }
    }

    /// The policy slots follow. Always `Blocking` where there are no threads.
    pub(crate) fn policy(&self) -> PipelineCompilation {
        if cfg!(target_family = "wasm") {
            return PipelineCompilation::Blocking;
        }
        PipelineCompilation::from_u8(self.policy.load(Ordering::Relaxed))
    }

    pub(crate) fn set_policy(&self, policy: PipelineCompilation) {
        self.policy.store(policy as u8, Ordering::Relaxed);
    }

    fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        let pending = Arc::clone(&self.pending);
        pending.add();
        pool().submit(Box::new(move || {
            job();
            pending.finish();
        }));
    }
}

/// One pipeline, built on first use.
///
/// `P` is a render or compute pipeline, or anything else that is costly to
/// create and cheap to clone a handle to.
pub(crate) struct PipelineSlot<P = crate::gpu::RenderPipeline> {
    ready: OnceLock<P>,
    inflight: Mutex<Option<mpsc::Receiver<P>>>,
}

impl<P> Default for PipelineSlot<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> PipelineSlot<P> {
    pub(crate) const fn new() -> Self {
        Self {
            ready: OnceLock::new(),
            inflight: Mutex::new(None),
        }
    }

    /// Whether the pipeline has been built, without starting anything.
    pub(crate) fn is_ready(&self) -> bool {
        self.ready.get().is_some()
    }
}

impl<P: Send + 'static> PipelineSlot<P> {
    /// The pipeline, or `None` while it compiles on a worker.
    ///
    /// Under `Blocking` this builds on the calling thread (or waits for a
    /// worker that already has it) and always returns the pipeline. Under
    /// `Background` the first call hands `build` to a worker and returns
    /// `None`; later calls return `None` until the result is in, then the
    /// pipeline. `build` is only run once however many threads ask.
    pub(crate) fn get(
        &self,
        compiler: &PipelineCompiler,
        build: impl FnOnce() -> P + Send + 'static,
    ) -> Option<&P> {
        if let Some(p) = self.ready.get() {
            return Some(p);
        }
        let mut inflight = self.inflight.lock().unwrap();
        if let Some(p) = self.ready.get() {
            return Some(p);
        }
        let policy = compiler.policy();
        if let Some(rx) = inflight.as_ref() {
            let result = match policy {
                PipelineCompilation::Blocking => rx.recv().ok(),
                PipelineCompilation::Background => match rx.try_recv() {
                    Ok(p) => Some(p),
                    Err(mpsc::TryRecvError::Empty) => return None,
                    Err(mpsc::TryRecvError::Disconnected) => None,
                },
            };
            *inflight = None;
            // A disconnected channel means the worker panicked; build again
            // below as if nothing had been asked for.
            if let Some(p) = result {
                let _ = self.ready.set(p);
                return self.ready.get();
            }
        }
        match policy {
            PipelineCompilation::Blocking => {
                let _ = self.ready.set(build());
                self.ready.get()
            }
            PipelineCompilation::Background => {
                let (tx, rx) = mpsc::channel();
                compiler.spawn(move || {
                    // The slot may have been dropped meanwhile; the result is
                    // then thrown away with the channel.
                    let _ = tx.send(build());
                });
                *inflight = Some(rx);
                None
            }
        }
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

/// The process-wide worker pool pipeline compiles run on. Started on first
/// use; the threads sleep when there is nothing to do.
struct Pool {
    tx: mpsc::Sender<Job>,
}

impl Pool {
    fn submit(&self, job: Job) {
        // The receivers live for the process, so the send cannot fail.
        let _ = self.tx.send(job);
    }
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() / 2)
            .unwrap_or(2)
            .clamp(1, 8);
        for i in 0..threads {
            let rx = Arc::clone(&rx);
            std::thread::Builder::new()
                .name(format!("vpl-pipeline-compile-{i}"))
                .spawn(move || {
                    loop {
                        let job = rx.lock().unwrap().recv();
                        match job {
                            Ok(job) => job(),
                            Err(_) => return,
                        }
                    }
                })
                .expect("spawn a pipeline compile worker");
        }
        Pool { tx }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    fn background() -> PipelineCompiler {
        PipelineCompiler::new(PipelineCompilation::Background)
    }

    fn blocking() -> PipelineCompiler {
        PipelineCompiler::new(PipelineCompilation::Blocking)
    }

    fn pending(compiler: &PipelineCompiler) -> usize {
        compiler.pending.count.load(Ordering::SeqCst)
    }

    /// Poll until no compile is running.
    fn wait(compiler: &PipelineCompiler) {
        let start = std::time::Instant::now();
        while pending(compiler) > 0 {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "workers never finished"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Poll `slot` until the worker's result is in.
    fn settle<'a>(slot: &'a PipelineSlot<u32>, compiler: &PipelineCompiler) -> &'a u32 {
        let start = std::time::Instant::now();
        loop {
            if let Some(v) = slot.get(compiler, || unreachable!("already started")) {
                return v;
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "worker never finished"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_blocking_read_builds_on_the_spot() {
        let compiler = blocking();
        let slot = PipelineSlot::<u32>::new();
        assert!(!slot.is_ready());
        assert_eq!(slot.get(&compiler, || 7), Some(&7));
        assert!(slot.is_ready());
        assert_eq!(pending(&compiler), 0);
    }

    #[test]
    fn a_background_read_returns_nothing_until_the_worker_is_done() {
        let compiler = background();
        let slot = PipelineSlot::<u32>::new();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let first = slot.get(&compiler, move || {
            release_rx.recv().unwrap();
            42
        });
        assert_eq!(first, None);
        assert_eq!(pending(&compiler), 1);
        // A second read while compiling neither blocks nor starts another.
        assert_eq!(slot.get(&compiler, || panic!("built twice")), None);
        release_tx.send(()).unwrap();
        assert_eq!(settle(&slot, &compiler), &42);
        wait(&compiler);
        assert_eq!(pending(&compiler), 0);
    }

    #[test]
    fn the_build_runs_once_however_many_threads_ask() {
        let compiler = Arc::new(blocking());
        let slot = Arc::new(PipelineSlot::<u32>::new());
        let builds = Arc::new(AtomicU32::new(0));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (compiler, slot, builds) = (compiler.clone(), slot.clone(), builds.clone());
                std::thread::spawn(move || {
                    *slot
                        .get(&compiler, move || {
                            builds.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(Duration::from_millis(5));
                            9
                        })
                        .unwrap()
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 9);
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_slot_dropped_mid_compile_discards_the_result_and_the_count_settles() {
        let compiler = background();
        let slot = PipelineSlot::<u32>::new();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        assert_eq!(
            slot.get(&compiler, move || {
                release_rx.recv().unwrap();
                1
            }),
            None
        );
        assert_eq!(pending(&compiler), 1);
        drop(slot);
        release_tx.send(()).unwrap();
        wait(&compiler);
        assert_eq!(pending(&compiler), 0);
    }

    #[test]
    fn a_blocking_read_waits_for_a_worker_already_building() {
        let compiler = background();
        let slot = PipelineSlot::<u32>::new();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        assert_eq!(
            slot.get(&compiler, move || {
                release_rx.recv().unwrap();
                3
            }),
            None
        );
        compiler.set_policy(PipelineCompilation::Blocking);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            release_tx.send(()).unwrap();
        });
        // Waits for the worker rather than building a second copy.
        assert_eq!(slot.get(&compiler, || panic!("built twice")), Some(&3));
    }

    #[test]
    fn wait_returns_when_every_compile_is_done() {
        let compiler = background();
        let slots: Vec<PipelineSlot<u32>> = (0..6).map(|_| PipelineSlot::new()).collect();
        for (i, slot) in slots.iter().enumerate() {
            slot.get(&compiler, move || {
                std::thread::sleep(Duration::from_millis(10));
                i as u32
            });
        }
        assert!(pending(&compiler) > 0);
        wait(&compiler);
        assert_eq!(pending(&compiler), 0);
        for (i, slot) in slots.iter().enumerate() {
            assert_eq!(settle(slot, &compiler), &(i as u32));
        }
    }

    #[test]
    fn the_platform_default_is_blocking_only_on_apple_and_the_web() {
        let expected = if cfg!(any(
            target_os = "macos",
            target_os = "ios",
            target_family = "wasm"
        )) {
            PipelineCompilation::Blocking
        } else {
            PipelineCompilation::Background
        };
        assert_eq!(PipelineCompilation::platform_default(), expected);
    }
}
