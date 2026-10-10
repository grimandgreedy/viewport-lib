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

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};

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
    /// The default for this platform: `Background`, except `Blocking` on the
    /// web, which has no threads, and on macOS and iOS when built against
    /// wgpu 27. wgpu 27's Metal backend holds one device-wide lock for the
    /// whole of a pipeline compile, and every buffer and texture allocation
    /// takes it too, so a worker compile stalls the thread that renders for
    /// as long as the compile lasts. wgpu 29 and 30 dropped that lock.
    pub fn platform_default() -> Self {
        let apple = cfg!(any(target_os = "macos", target_os = "ios"));
        if cfg!(target_family = "wasm") || (apple && cfg!(feature = "wgpu27")) {
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

/// The compiles a renderer has in flight: a count, and a signal when one
/// finishes.
#[derive(Default)]
struct Pending {
    count: AtomicUsize,
    lock: Mutex<()>,
    finished: Condvar,
    /// Set when the renderer goes away. A job a worker has not started yet
    /// is then dropped instead of compiled.
    cancelled: AtomicBool,
}

impl Pending {
    fn add(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }

    fn finish(&self) {
        let _guard = self.lock.lock().unwrap();
        self.count.fetch_sub(1, Ordering::SeqCst);
        self.finished.notify_all();
    }

    fn wait(&self) {
        let mut guard = self.lock.lock().unwrap();
        while self.count.load(Ordering::SeqCst) > 0 {
            guard = self.finished.wait(guard).unwrap();
        }
    }
}

/// Shared by every slot one renderer owns: the policy its slots follow and the
/// compiles they have running.
pub struct PipelineCompiler {
    policy: AtomicU8,
    /// Set while the renderer captures: every read blocks whatever the
    /// policy, so a captured frame has nothing missing.
    capturing: AtomicBool,
    /// How many read-back calls are running: each needs a complete frame, so
    /// reads block while it is non-zero. See [`Self::blocking_scope`].
    blocking_scopes: AtomicUsize,
    pending: Arc<Pending>,
}

impl PipelineCompiler {
    pub(crate) fn new(policy: PipelineCompilation) -> Self {
        Self {
            policy: AtomicU8::new(policy as u8),
            capturing: AtomicBool::new(false),
            blocking_scopes: AtomicUsize::new(0),
            pending: Arc::new(Pending::default()),
        }
    }

    /// The policy slots follow right now: the one set, or `Blocking` while
    /// capturing or where there are no threads.
    pub(crate) fn policy(&self) -> PipelineCompilation {
        if cfg!(target_family = "wasm")
            || self.capturing.load(Ordering::Relaxed)
            || self.blocking_scopes.load(Ordering::Relaxed) > 0
        {
            return PipelineCompilation::Blocking;
        }
        self.configured_policy()
    }

    /// The policy as set, before capture or the platform overrides it.
    pub(crate) fn configured_policy(&self) -> PipelineCompilation {
        PipelineCompilation::from_u8(self.policy.load(Ordering::Relaxed))
    }

    pub(crate) fn set_policy(&self, policy: PipelineCompilation) {
        self.policy.store(policy as u8, Ordering::Relaxed);
    }

    pub(crate) fn set_capturing(&self, capturing: bool) {
        self.capturing.store(capturing, Ordering::Relaxed);
    }

    /// Make every read block until the returned guard is dropped. A call that
    /// hands pixels back (`render_offscreen`, the captures and probe bakes)
    /// has no later frame to catch up on, so it must not skip a draw.
    pub(crate) fn blocking_scope(self: &Arc<Self>) -> BlockingScope {
        self.blocking_scopes.fetch_add(1, Ordering::Relaxed);
        BlockingScope(Arc::clone(self))
    }

    /// Compiles handed to the workers that have not finished.
    pub(crate) fn pending(&self) -> usize {
        self.pending.count.load(Ordering::SeqCst)
    }

    /// Block until no compile is running on the workers.
    pub(crate) fn wait(&self) {
        self.pending.wait();
    }

    /// Drop the compiles no worker has started and wait for the ones that
    /// have. A process that exits while a worker is inside the driver's
    /// pipeline compile can abort in the driver's own teardown (seen on
    /// NVIDIA under Vulkan), so a renderer does this when it is dropped.
    pub(crate) fn shut_down(&self) {
        self.pending.cancelled.store(true, Ordering::SeqCst);
        self.pending.wait();
    }

    fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        let pending = Arc::clone(&self.pending);
        pending.add();
        pool().submit(Box::new(move || {
            if !pending.cancelled.load(Ordering::SeqCst) {
                job();
            }
            pending.finish();
        }));
    }
}

/// Ends a [`PipelineCompiler::blocking_scope`] when dropped.
pub(crate) struct BlockingScope(Arc<PipelineCompiler>);

impl Drop for BlockingScope {
    fn drop(&mut self) {
        self.0.blocking_scopes.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Shuts the renderer's compiler down when dropped: see
/// [`PipelineCompiler::shut_down`]. Held by `DeviceResources`, so dropping the
/// renderer returns only once no worker is compiling for it. The compiler
/// itself can outlive the renderer inside a build context a worker holds, so
/// the shutdown cannot live in its own `Drop`.
pub(crate) struct CompilerShutdown(pub(crate) Arc<PipelineCompiler>);

impl Drop for CompilerShutdown {
    fn drop(&mut self) {
        self.0.shut_down();
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

    /// The pipeline if it has been built, without starting anything.
    pub(crate) fn ready(&self) -> Option<&P> {
        self.ready.get()
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

impl<P: Send + 'static> PipelineSlot<P> {
    /// The pipeline, built on the calling thread or taken from a worker that
    /// already has it, whatever the policy. For a caller with no way to skip
    /// its work this frame.
    pub(crate) fn get_blocking(&self, build: impl FnOnce() -> P) -> &P {
        if let Some(p) = self.ready.get() {
            return p;
        }
        let mut inflight = self.inflight.lock().unwrap();
        if let Some(rx) = inflight.take()
            && let Ok(p) = rx.recv()
        {
            let _ = self.ready.set(p);
        }
        if self.ready.get().is_none() {
            let _ = self.ready.set(build());
        }
        self.ready.get().unwrap()
    }
}

/// `N` pipelines built from one shared context, each the first time something
/// reads it.
///
/// `C` holds by value what every build needs (the device, layouts, shader
/// modules, formats), so a build can run on a worker thread. `build` makes
/// member `i` from it. Reading a member with [`get`](Self::get) starts its
/// compile under the renderer's [`PipelineCompilation`] policy and returns
/// `None` while a worker has it; the draw that wanted it skips that frame.
/// Compiles in flight count towards `ViewportRenderer::pipelines_pending`.
///
/// `P` is a render pipeline unless named: a set of compute pipelines is
/// `LazyFamily<C, N, ComputePipeline>`, and a dispatch whose pipeline is not
/// ready is skipped the same way a draw is.
///
/// Make one with [`DeviceResources::lazy_pipelines`](crate::resources::DeviceResources::lazy_pipelines)
/// or [`DeviceResources::lazy_compute_pipelines`](crate::resources::DeviceResources::lazy_compute_pipelines).
pub struct LazyFamily<C, const N: usize, P = crate::gpu::RenderPipeline> {
    ctx: Arc<C>,
    compiler: Arc<PipelineCompiler>,
    slots: [PipelineSlot<P>; N],
    build: fn(&C, usize) -> P,
}

impl<C: Send + Sync + 'static, const N: usize, P: Send + 'static> LazyFamily<C, N, P> {
    pub(crate) fn new(ctx: C, compiler: Arc<PipelineCompiler>, build: fn(&C, usize) -> P) -> Self {
        Self {
            ctx: Arc::new(ctx),
            compiler,
            slots: std::array::from_fn(|_| PipelineSlot::new()),
            build,
        }
    }

    /// What the builds read.
    pub fn context(&self) -> &C {
        &self.ctx
    }

    /// Member `i`, or `None` while a worker has it.
    pub fn get(&self, i: usize) -> Option<&P> {
        let ctx = Arc::clone(&self.ctx);
        let build = self.build;
        self.slots[i].get(&self.compiler, move || build(&ctx, i))
    }

    /// Whether member `i` is built, without starting anything.
    pub fn is_ready(&self, i: usize) -> bool {
        self.slots[i].is_ready()
    }

    /// Whether [`get`](Self::get) would return member `i` this frame: it is
    /// built, or the policy is `Blocking` and `get` would build it. Starts
    /// nothing. A pass that should wait for another member (an outline or a
    /// pick waiting for the colour pipeline) checks the other one with this.
    pub fn available(&self, i: usize) -> bool {
        self.slots[i].is_ready() || self.compiler.policy() == PipelineCompilation::Blocking
    }

    /// Ask for members `0..end`: built now under `Blocking`, handed to the
    /// workers under `Background`.
    pub fn request(&self, end: usize) {
        for i in 0..end.min(N) {
            self.get(i);
        }
    }

    /// Ask for every member.
    pub fn request_all(&self) {
        self.request(N);
    }

    /// How many members are built.
    pub fn ready_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_ready()).count()
    }
}

/// The compiled module for one shader source, shared by every
/// [`LazyModule`] made from that source.
pub(crate) type ModuleCell = Arc<OnceLock<crate::gpu::ShaderModule>>;

/// A shader module that compiles the first time a pipeline build asks for it.
/// Make one with [`DeviceResources::lazy_module`](crate::resources::DeviceResources::lazy_module).
///
/// A family's context holds these instead of compiled modules, so the module
/// compile (milliseconds for the mesh shaders) runs inside the first member's
/// build, on a worker under `Background`, and not in the `ensure_*` that
/// composed the family. Handles made from the same source share one compile.
#[derive(Clone)]
pub struct LazyModule(Arc<LazyModuleInner>);

struct LazyModuleInner {
    device: crate::gpu::Device,
    label: String,
    source: String,
    cell: ModuleCell,
}

impl LazyModule {
    pub(crate) fn new(
        device: &crate::gpu::Device,
        label: &str,
        source: &str,
        cell: ModuleCell,
    ) -> Self {
        Self(Arc::new(LazyModuleInner {
            device: device.clone(),
            label: label.to_owned(),
            source: source.to_owned(),
            cell,
        }))
    }

    /// The module, compiled on the calling thread if nothing has compiled it.
    pub fn get(&self) -> &crate::gpu::ShaderModule {
        let m = &self.0;
        m.cell
            .get_or_init(|| crate::resources::builders::wgsl_module(&m.device, &m.label, &m.source))
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
                .name(format!("pipeline-compile-{i}"))
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
        compiler.pending()
    }

    fn wait(compiler: &PipelineCompiler) {
        compiler.wait();
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
    fn shutting_down_drops_queued_compiles_and_waits_for_running_ones() {
        let compiler = background();
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let built = Arc::new(AtomicU32::new(0));
        // More slots than the pool has workers, so some are still queued.
        let slots: Vec<PipelineSlot<u32>> = (0..32).map(|_| PipelineSlot::new()).collect();
        let release_rx = Arc::new(Mutex::new(release_rx));
        for slot in &slots {
            let (started_tx, release_rx, built) = (
                started_tx.clone(),
                Arc::clone(&release_rx),
                Arc::clone(&built),
            );
            slot.get(&compiler, move || {
                let _ = started_tx.send(());
                let _ = release_rx.lock().unwrap().recv();
                built.fetch_add(1, Ordering::SeqCst);
                0
            });
        }
        // One job is inside its build; let it finish once shutdown is waiting.
        started_rx.recv().unwrap();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            for _ in 0..32 {
                let _ = release_tx.send(());
            }
        });
        compiler.shut_down();
        releaser.join().unwrap();
        assert_eq!(pending(&compiler), 0);
        assert!(
            built.load(Ordering::SeqCst) < 32,
            "queued compiles still ran"
        );
    }

    #[test]
    fn a_blocking_scope_makes_reads_block_until_it_ends() {
        let compiler = Arc::new(background());
        {
            let _outer = compiler.blocking_scope();
            let _inner = compiler.blocking_scope();
            let slot = PipelineSlot::<u32>::new();
            assert_eq!(slot.get(&compiler, || 4), Some(&4));
        }
        assert_eq!(compiler.policy(), PipelineCompilation::Background);
    }

    #[test]
    fn capturing_makes_every_read_block() {
        let compiler = background();
        compiler.set_capturing(true);
        let slot = PipelineSlot::<u32>::new();
        assert_eq!(slot.get(&compiler, || 5), Some(&5));
        assert_eq!(pending(&compiler), 0);
        assert_eq!(
            compiler.configured_policy(),
            PipelineCompilation::Background
        );
        compiler.set_capturing(false);
        assert_eq!(compiler.policy(), PipelineCompilation::Background);
    }

    #[test]
    fn the_platform_default_is_blocking_only_on_the_web_and_apple_on_wgpu27() {
        let apple = cfg!(any(target_os = "macos", target_os = "ios"));
        let expected = if cfg!(target_family = "wasm") || (apple && cfg!(feature = "wgpu27")) {
            PipelineCompilation::Blocking
        } else {
            PipelineCompilation::Background
        };
        assert_eq!(PipelineCompilation::platform_default(), expected);
    }
}
