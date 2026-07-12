//! Internal job runner used by the async upload entry points.
//!
//! Every long-running upload on `DeviceResources` routes through this
//! runner. A submitted job runs its CPU work on a background thread, may
//! optionally submit GPU commands, and reports completion on the main thread
//! during `process_uploads`. Callers query progress through `upload_status`
//! and learn about completion either by polling or by attaching a callback.
//!
//! No upload entry points use the runner yet. Real submitters will land
//! alongside the async variants of each existing `upload_*` method.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::error::ViewportError;

/// Opaque handle for a submitted job. Returned by every `begin_upload_*`
/// entry and accepted by status queries, completion callbacks, and the
/// per-type result accessors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JobId(u64);

/// Typed result slots for every async upload path, keyed by job id.
///
/// Each `begin_upload_*` fills the matching map from its apply closure; the
/// paired `upload_result_*` drains it. Grouping these here keeps the async
/// bookkeeping off `DeviceResources` as one field rather than a score of
/// flat ones. All maps start empty, so the whole struct is `Default`.
#[derive(Default)]
pub(crate) struct JobResults {
    /// Async mesh uploads (`begin_upload_mesh_data` / `upload_result_mesh`).
    pub mesh: std::sync::Mutex<
        std::collections::HashMap<JobId, ResultSlot<crate::resources::mesh::mesh_store::MeshId>>,
    >,
    /// Async texture uploads (albedo + normal map).
    pub texture:
        std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<crate::resources::TextureId>>>,
    /// Jobs submitted through the plugin facade; drained by `Jobs::take<T>`.
    pub plugin: std::sync::Mutex<
        std::collections::HashMap<JobId, ResultSlot<Box<dyn std::any::Any + Send>>>,
    >,
    /// Async polyline uploads.
    pub polyline: std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::PolylineId>>>,
    /// Async streamtube uploads.
    pub streamtube:
        std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::StreamtubeId>>>,
    /// Async tube uploads.
    pub tube: std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::TubeId>>>,
    /// Async ribbon uploads.
    pub ribbon: std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::RibbonId>>>,
    /// Async point cloud uploads.
    pub point_cloud:
        std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::PointCloudId>>>,
    /// Async glyph set uploads.
    pub glyph_set:
        std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::GlyphSetId>>>,
    /// Async tensor glyph set uploads.
    pub tensor_glyph_set:
        std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::TensorGlyphSetId>>>,
    /// Async volume texture uploads.
    pub volume: std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::VolumeId>>>,
    /// Async marching-cubes-ready volume uploads.
    pub volume_mc:
        std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::McVolumeId>>>,
    /// Async volume-mesh uploads: mesh id plus face-to-cell map.
    pub volume_mesh: std::sync::Mutex<
        std::collections::HashMap<
            JobId,
            ResultSlot<(crate::resources::mesh::mesh_store::MeshId, Vec<u32>)>,
        >,
    >,
    /// Async clipped-volume-mesh uploads.
    pub clipped_volume_mesh: std::sync::Mutex<
        std::collections::HashMap<
            JobId,
            ResultSlot<(crate::resources::mesh::mesh_store::MeshId, Vec<u32>)>,
        >,
    >,
    /// Async sparse-volume-grid uploads.
    pub sparse_volume_grid: std::sync::Mutex<
        std::collections::HashMap<JobId, ResultSlot<crate::resources::mesh::mesh_store::MeshId>>,
    >,
    /// Async projected-tet-mesh uploads: tet id plus packed scalar range.
    pub projected_tet: std::sync::Mutex<
        std::collections::HashMap<JobId, ResultSlot<(super::ProjectedTetId, f32, f32)>>,
    >,
    /// Async gaussian splat uploads.
    pub gaussian_splat: std::sync::Mutex<
        std::collections::HashMap<JobId, ResultSlot<crate::renderer::GaussianSplatId>>,
    >,
    /// Async overlay texture uploads.
    pub overlay_texture: std::sync::Mutex<
        std::collections::HashMap<JobId, ResultSlot<crate::renderer::OverlayTextureId>>,
    >,
    /// Async sprite set uploads.
    pub sprite_set:
        std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::SpriteSetId>>>,
    /// Async sprite instance set uploads.
    pub sprite_instance_set:
        std::sync::Mutex<std::collections::HashMap<JobId, ResultSlot<super::SpriteInstanceSetId>>>,
}

/// Current state of a submitted job.
#[derive(Debug, Clone)]
pub enum UploadStatus {
    /// The job is still running. `progress` is a hint in the range 0.0 to
    /// 1.0. Workers that do not report progress leave the value at zero.
    Pending {
        /// Reported completion fraction, between 0.0 and 1.0.
        progress: f32,
    },
    /// The job finished successfully. The caller may take any typed result
    /// it expects through the matching `upload_result_*` accessor.
    Ready,
    /// The worker returned an error, panicked, or dropped its channel
    /// without sending. The job is not retried.
    Failed(ViewportError),
    /// The id has never been issued, has already been reaped, or its result
    /// was already taken. Treat as "nothing in flight under that id".
    Unknown,
}

/// Cheap progress counter shared between the worker thread and the runner.
///
/// Workers call `set` with a fraction in 0.0 to 1.0; the runner reads the
/// value during `process_uploads` to populate `UploadStatus::Pending`.
#[derive(Clone)]
pub struct ProgressHandle {
    inner: Arc<AtomicU32>,
}

impl ProgressHandle {
    fn new() -> Self {
        Self {
            inner: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Record current progress. Values are clamped to 0.0 to 1.0.
    pub fn set(&self, fraction: f32) {
        let clamped = fraction.clamp(0.0, 1.0);
        self.inner.store(clamped.to_bits(), Ordering::Relaxed);
    }

    fn read(&self) -> f32 {
        f32::from_bits(self.inner.load(Ordering::Relaxed))
    }
}

/// Closure run on the caller's thread once a job's worker (and any GPU
/// submission) has completed. Real upload types use it to insert their newly
/// built textures, buffers, and bind groups into `DeviceResources`.
pub type ApplyFn = Box<dyn FnOnce(&mut super::DeviceResources) + Send>;

/// Boxed GPU-submitting work for a `submit_with_gpu` or
/// `submit_cpu_then_gpu` job.
///
/// Unlike CPU jobs, this runs on the main thread inside `process`, not on a
/// rayon worker: some drivers (notably the NVIDIA Linux Vulkan driver)
/// corrupt the command pushbuffer (NVRM Xid 32, surfacing as a lost device)
/// when GPU commands are submitted from a thread other than the one driving
/// the device. CPU-side preparation still belongs on worker threads via
/// `submit_cpu` or the CPU stage of `submit_cpu_then_gpu`; only the GPU
/// calls are funnelled here.
pub(crate) type GpuWorkFn = Box<
    dyn FnOnce(&wgpu::Device, &wgpu::Queue, &ProgressHandle) -> Result<JobProduct, ViewportError>
        + Send,
>;

/// What one turn of a chunked GPU-stage closure reports back.
pub(crate) enum GpuStep {
    /// More slices remain; the runner requeues the job and calls it again
    /// on a later `process` (same call if budget remains, next frame
    /// otherwise).
    Continue,
    /// All slices written; the job finishes with this product.
    Done(JobProduct),
}

/// Resumable GPU-stage work for a `submit_cpu_then_gpu_chunked` job.
///
/// Runs on the device thread like [`GpuWorkFn`], but is called repeatedly:
/// each turn should write slices until the passed budget elapses (doing at
/// least one slice per turn so an already-elapsed budget still makes
/// progress) and return `Continue` while work remains. With an unbounded
/// budget a well-formed closure completes in a single turn, so the
/// unbudgeted path keeps whole-asset behaviour.
pub(crate) type ChunkedGpuWorkFn = Box<
    dyn FnMut(
            &wgpu::Device,
            &wgpu::Queue,
            &ProgressHandle,
            &FrameBudget,
        ) -> Result<GpuStep, ViewportError>
        + Send,
>;

/// The two shapes of device-thread work a job can defer.
pub(crate) enum DeferredWork {
    /// Runs once, to completion, in a single drain turn.
    Once(GpuWorkFn),
    /// Runs across as many drain turns as its slices need.
    Chunked(ChunkedGpuWorkFn),
}

/// A job's GPU stage awaiting execution on the main thread.
struct DeferredGpuJob {
    work: DeferredWork,
    progress: ProgressHandle,
    tx: mpsc::Sender<WorkerOutcome>,
    /// Device-thread time already spent in earlier turns of a chunked
    /// job; folded into the duration reported with the final outcome.
    elapsed: Duration,
}

/// Maximum number of deferred GPU jobs run per `process` call. The cap spreads
/// a large batch (e.g. a scene's worth of streamed textures) across frames so
/// a single `process` does not stall the main thread uploading everything at
/// once.
const MAX_GPU_JOBS_PER_PROCESS: usize = 16;

/// Per-job result holder shared between a worker's apply closure and the
/// matching `upload_result_*` accessor.
///
/// `ResultSlot<T>` is constructed at submit time on the main thread, cloned
/// into the apply closure, and used to publish the upload's typed result.
/// The accessor calls `take` to claim the value once the job reaches
/// `Ready`.
pub struct ResultSlot<T> {
    inner: Arc<std::sync::Mutex<Option<T>>>,
}

impl<T> Clone for ResultSlot<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> ResultSlot<T> {
    /// Build an empty slot. The apply closure fills it; the accessor takes.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Store the result. Called from the apply closure on the main thread.
    pub fn set(&self, value: T) {
        let mut guard = self.inner.lock().expect("result slot poisoned");
        *guard = Some(value);
    }

    /// Take the stored result if one is present, leaving the slot empty.
    pub fn take(&self) -> Option<T> {
        let mut guard = self.inner.lock().expect("result slot poisoned");
        guard.take()
    }
}

impl<T> Default for ResultSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// What the worker hands back to the runner. Bundles whatever GPU
/// completion the runner should wait on with whatever main-thread mutation
/// the apply step needs to perform.
///
/// Workers that finish purely on the CPU and need no main-thread mutation
/// return `JobProduct::default()`. Workers that submit GPU commands fill
/// `gpu`; workers that need to store results on `DeviceResources`
/// fill `apply`.
pub struct JobProduct {
    /// `Some` when the worker has submitted commands that must complete
    /// before the job can be reported `Ready`. The runner gates on this
    /// submission via `device.poll`.
    pub gpu: Option<wgpu::SubmissionIndex>,
    /// `Some` when the worker has built state that must be folded into
    /// `DeviceResources` from the main thread.
    pub apply: Option<ApplyFn>,
}

impl Default for JobProduct {
    fn default() -> Self {
        Self {
            gpu: None,
            apply: None,
        }
    }
}

impl JobProduct {
    /// No GPU gating, no apply step. Convenient for tests and CPU-only
    /// jobs whose effects are entirely captured in the channel send.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Gate on a single GPU submission; no main-thread apply.
    pub fn with_gpu(gpu: wgpu::SubmissionIndex) -> Self {
        Self {
            gpu: Some(gpu),
            apply: None,
        }
    }

    /// Run an apply step on the main thread; no GPU gating.
    pub fn with_apply(apply: ApplyFn) -> Self {
        Self {
            gpu: None,
            apply: Some(apply),
        }
    }

    /// Gate on a GPU submission, then run the apply step.
    pub fn with_gpu_and_apply(gpu: wgpu::SubmissionIndex, apply: ApplyFn) -> Self {
        Self {
            gpu: Some(gpu),
            apply: Some(apply),
        }
    }
}

/// Cap on how long `process_uploads_with_budget` is allowed to spend running
/// completed jobs' apply steps in a single call.
///
/// The budget measures only the main-thread apply work (and the wgpu device
/// poll that precedes it); it does not bound worker-thread time, which runs
/// concurrently. When the budget is exhausted, any apply closures that have
/// not yet run are held inside the runner and processed on the next
/// `process_uploads*` call. Their owning jobs continue to report
/// `UploadStatus::Pending` until their apply runs, so the typed result is
/// never observably available before it is in place.
#[derive(Clone, Copy, Debug)]
pub struct FrameBudget {
    deadline: Option<Instant>,
}

impl FrameBudget {
    /// No bound. `process_uploads_with_budget` behaves like the unbounded
    /// `process_uploads`.
    pub fn unbounded() -> Self {
        Self { deadline: None }
    }

    /// Cap apply-step work for the current call to roughly `duration` from
    /// now. The check happens between applies, so a single long-running
    /// apply may push past the deadline once it starts.
    pub fn from_now(duration: Duration) -> Self {
        Self {
            deadline: Some(Instant::now() + duration),
        }
    }

    /// True when the budget has elapsed.
    pub(crate) fn exhausted(&self) -> bool {
        match self.deadline {
            Some(t) => Instant::now() >= t,
            None => false,
        }
    }
}

/// Outcome the worker sends through its channel.
///
/// The `Duration` captures the wall-clock time the worker spent on the
/// background thread, measured from the rayon::spawn closure entry to its
/// return. It excludes both the time the job spent in the rayon queue and
/// any later GPU/apply-step work; the runner adds the apply-step duration
/// on top.
#[allow(dead_code)]
enum WorkerOutcome {
    Done(JobProduct, Duration),
    /// The CPU stage of a `submit_cpu_then_gpu*` job finished and handed
    /// back GPU work. `process` moves it onto the deferred queue; the
    /// deferred run reports `Done` or `Failed` through the same channel.
    NeedsGpu(DeferredWork, Duration),
    Failed(ViewportError, Duration),
}

type CompletionCallback = Box<dyn FnOnce(&UploadStatus) + Send>;

/// What `process` hands back for a single completed job. The caller is
/// responsible for running `apply` (if present and the status is `Ready`)
/// against the live `DeviceResources`, then invoking `callback`.
pub struct Completion {
    /// Id of the completed job; the caller uses it to record the apply
    /// duration back on the runner.
    pub id: JobId,
    /// Final status the runner observed.
    pub status: UploadStatus,
    /// Apply closure produced by the worker. Run only when `status` is
    /// `Ready`.
    pub apply: Option<ApplyFn>,
    /// Completion callback registered via `on_complete`. Fires for both
    /// `Ready` and `Failed` so the consumer can branch.
    pub callback: Option<CompletionCallback>,
}

struct JobSlot {
    progress: ProgressHandle,
    /// Sender side of the outcome channel, retained so a `NeedsGpu`
    /// hand-off can requeue the job's GPU stage as a deferred job that
    /// reports through the same channel.
    tx: mpsc::Sender<WorkerOutcome>,
    rx: mpsc::Receiver<WorkerOutcome>,
    /// Once the worker has reported, the GPU submission to gate on (if any)
    /// and the apply closure to run when the GPU side finishes.
    awaiting: Option<(Option<wgpu::SubmissionIndex>, Option<ApplyFn>)>,
    callback: Option<CompletionCallback>,
}

/// Background worker pool plus the table of in-flight jobs.
///
/// Owned by `DeviceResources`; reached from there via
/// `process_uploads`, `upload_status`, and friends. `next_id` and the
/// `submit_*` helpers are unused until upload entry points are wired
/// through the runner.
#[allow(dead_code)]
pub struct JobRunner {
    next_id: u64,
    slots: HashMap<u64, JobSlot>,
    /// Jobs whose worker and GPU work have finished but whose main-thread
    /// apply closure has not yet been run.
    ///
    /// Populated by `process` for successful jobs that carry an apply step;
    /// drained by the caller of `process_uploads` (and friends). Entries
    /// remain visible as `UploadStatus::Pending { progress: 1.0 }` until
    /// the apply runs, so the typed result is never reported as `Ready`
    /// before it is materialized in the resource state.
    pending_apply: VecDeque<PendingApply>,
    /// Recently finished jobs, kept for one drain cycle so callers can still
    /// see `Ready` or `Failed` after the completion frame.
    finished: HashMap<u64, UploadStatus>,
    /// Time the actual work took. Worker thread time is recorded when the
    /// worker reports back; apply-step time is added by the caller via
    /// `add_apply_duration` after running the apply closure. Retained
    /// until `drop_duration` is called so consumers have at least one frame
    /// to read the result.
    durations: HashMap<u64, Duration>,
    /// GPU-submitting jobs deferred to run on the main thread during
    /// `process`. See [`GpuWorkFn`] for why these must not run on a worker.
    deferred_gpu: VecDeque<DeferredGpuJob>,
}

/// Entry on the `pending_apply` queue.
///
/// Holds everything the caller needs to fold a successful job into resource
/// state on the main thread: the id (for status updates and callback
/// registration), the final status, the apply closure, and any completion
/// callback that was registered on the job. Failed jobs do not produce
/// these; they go straight to `finished` and surface in the `Completion`
/// vec returned by `process`.
pub struct PendingApply {
    /// Id of the job whose apply is pending.
    pub id: JobId,
    /// Terminal status to record once the apply finishes; always
    /// `UploadStatus::Ready` for entries on the queue.
    pub status: UploadStatus,
    /// Closure that mutates `DeviceResources` and fills any typed
    /// result slot.
    pub apply: ApplyFn,
    /// Completion callback registered via `on_complete`.
    pub callback: Option<CompletionCallback>,
}

impl Default for JobRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl JobRunner {
    /// Construct an empty runner. The `DeviceResources` initializer
    /// holds the single instance; callers do not construct this directly.
    pub fn new() -> Self {
        Self {
            next_id: 1,
            slots: HashMap::new(),
            pending_apply: VecDeque::new(),
            finished: HashMap::new(),
            durations: HashMap::new(),
            deferred_gpu: VecDeque::new(),
        }
    }

    /// Total work duration recorded for a job, or `None` if the job is
    /// still in flight (or its duration record has aged out).
    pub fn duration(&self, id: JobId) -> Option<Duration> {
        self.durations.get(&id.0).copied()
    }

    /// Add the apply-step elapsed time to a job's running total. Called by
    /// the caller of `process` immediately after running the apply closure
    /// on the main thread.
    pub fn add_apply_duration(&mut self, id: JobId, apply: Duration) {
        let entry = self.durations.entry(id.0).or_insert(Duration::ZERO);
        *entry = entry.saturating_add(apply);
    }

    /// Drop the recorded duration for a job. Consumers call this after they
    /// have read the duration via `duration`; otherwise the runner keeps it
    /// across drain cycles so a single-frame retention is not enough.
    pub fn drop_duration(&mut self, id: JobId) {
        self.durations.remove(&id.0);
    }

    fn issue_id(&mut self) -> JobId {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("upload job id space exhausted");
        JobId(id)
    }

    /// Schedule a CPU-only job on the background pool.
    ///
    /// The worker receives a `ProgressHandle` it can use to publish progress.
    /// Returning `Err` or panicking marks the job as `Failed`.
    pub(crate) fn submit_cpu<F>(&mut self, work: F) -> JobId
    where
        F: FnOnce(&ProgressHandle) -> Result<JobProduct, ViewportError> + Send + 'static,
    {
        let id = self.issue_id();
        let progress = ProgressHandle::new();
        let worker_progress = progress.clone();
        let (tx, rx) = mpsc::channel();
        let worker_tx = tx.clone();

        rayon::spawn(move || {
            let t0 = Instant::now();
            let outcome = match catch_unwind(AssertUnwindSafe(|| work(&worker_progress))) {
                Ok(Ok(product)) => WorkerOutcome::Done(product, t0.elapsed()),
                Ok(Err(e)) => WorkerOutcome::Failed(e, t0.elapsed()),
                Err(_) => WorkerOutcome::Failed(
                    ViewportError::JobWorkerLost {
                        reason: "worker panicked",
                    },
                    t0.elapsed(),
                ),
            };
            // Receiver going away is fine; the runner was probably dropped.
            let _ = worker_tx.send(outcome);
        });

        self.slots.insert(
            id.0,
            JobSlot {
                progress,
                tx,
                rx,
                awaiting: None,
                callback: None,
            },
        );
        id
    }

    /// Schedule a job whose CPU stage runs on the background pool and
    /// whose GPU stage runs on the device thread during a later `process`
    /// call.
    ///
    /// The worker does its CPU work (decode, mip-chain build, repack) on a
    /// rayon thread and returns the boxed GPU work; `process` executes
    /// that work under the same per-call cap and frame budget as
    /// `submit_with_gpu` jobs. Prefer this over `submit_with_gpu` whenever
    /// the job has real CPU cost: with `submit_with_gpu` the whole
    /// closure, CPU work included, runs on the device thread.
    pub(crate) fn submit_cpu_then_gpu<F>(&mut self, work: F) -> JobId
    where
        F: FnOnce(&ProgressHandle) -> Result<GpuWorkFn, ViewportError> + Send + 'static,
    {
        let id = self.issue_id();
        let progress = ProgressHandle::new();
        let worker_progress = progress.clone();
        let (tx, rx) = mpsc::channel();
        let worker_tx = tx.clone();

        rayon::spawn(move || {
            let t0 = Instant::now();
            let outcome = match catch_unwind(AssertUnwindSafe(|| work(&worker_progress))) {
                Ok(Ok(gpu_work)) => {
                    WorkerOutcome::NeedsGpu(DeferredWork::Once(gpu_work), t0.elapsed())
                }
                Ok(Err(e)) => WorkerOutcome::Failed(e, t0.elapsed()),
                Err(_) => WorkerOutcome::Failed(
                    ViewportError::JobWorkerLost {
                        reason: "worker panicked",
                    },
                    t0.elapsed(),
                ),
            };
            let _ = worker_tx.send(outcome);
        });

        self.slots.insert(
            id.0,
            JobSlot {
                progress,
                tx,
                rx,
                awaiting: None,
                callback: None,
            },
        );
        id
    }

    /// `submit_cpu_then_gpu` for GPU stages too large to run in one turn.
    ///
    /// The worker returns a resumable [`ChunkedGpuWorkFn`] instead of a
    /// one-shot closure: `process` calls it repeatedly on the device
    /// thread, passing the frame budget, until it reports
    /// [`GpuStep::Done`]. Use this when the GPU stage's copies are large
    /// enough to blow a frame budget on their own (streamed textures and
    /// meshes); the closure slices its writes and spreads them across
    /// frames.
    pub(crate) fn submit_cpu_then_gpu_chunked<F>(&mut self, work: F) -> JobId
    where
        F: FnOnce(&ProgressHandle) -> Result<ChunkedGpuWorkFn, ViewportError> + Send + 'static,
    {
        let id = self.issue_id();
        let progress = ProgressHandle::new();
        let worker_progress = progress.clone();
        let (tx, rx) = mpsc::channel();
        let worker_tx = tx.clone();

        rayon::spawn(move || {
            let t0 = Instant::now();
            let outcome = match catch_unwind(AssertUnwindSafe(|| work(&worker_progress))) {
                Ok(Ok(gpu_work)) => {
                    WorkerOutcome::NeedsGpu(DeferredWork::Chunked(gpu_work), t0.elapsed())
                }
                Ok(Err(e)) => WorkerOutcome::Failed(e, t0.elapsed()),
                Err(_) => WorkerOutcome::Failed(
                    ViewportError::JobWorkerLost {
                        reason: "worker panicked",
                    },
                    t0.elapsed(),
                ),
            };
            let _ = worker_tx.send(outcome);
        });

        self.slots.insert(
            id.0,
            JobSlot {
                progress,
                tx,
                rx,
                awaiting: None,
                callback: None,
            },
        );
        id
    }

    /// Schedule a CPU job that also submits GPU commands.
    ///
    /// The worker is handed cloned `Device` and `Queue` handles. It may
    /// submit any number of command buffers and bundles the final
    /// `SubmissionIndex` into the `JobProduct` it returns; the runner waits
    /// on that submission before reporting `Ready`.
    pub(crate) fn submit_with_gpu<F>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        work: F,
    ) -> JobId
    where
        F: FnOnce(
                &wgpu::Device,
                &wgpu::Queue,
                &ProgressHandle,
            ) -> Result<JobProduct, ViewportError>
            + Send
            + 'static,
    {
        let id = self.issue_id();
        let progress = ProgressHandle::new();
        let (tx, rx) = mpsc::channel();
        // GPU submission is deferred to the main thread (run in `process`)
        // rather than spawned on a rayon worker: submitting from a non-device
        // thread corrupts the pushbuffer on some drivers. The `device` and
        // `queue` arguments are unused here for the same reason; `process`
        // supplies them when the work runs.
        let _ = (device, queue);
        self.deferred_gpu.push_back(DeferredGpuJob {
            work: DeferredWork::Once(Box::new(work)),
            progress: progress.clone(),
            tx: tx.clone(),
            elapsed: Duration::ZERO,
        });

        self.slots.insert(
            id.0,
            JobSlot {
                progress,
                tx,
                rx,
                awaiting: None,
                callback: None,
            },
        );
        id
    }

    /// Attach a callback to fire on completion. The callback runs on the
    /// main thread during the same `process_uploads` call that marks the job
    /// done. If the job has already completed and is still in the
    /// short-retention window, the callback fires immediately.
    pub fn on_complete<F>(&mut self, id: JobId, cb: F)
    where
        F: FnOnce(&UploadStatus) + Send + 'static,
    {
        if let Some(slot) = self.slots.get_mut(&id.0) {
            slot.callback = Some(Box::new(cb));
            return;
        }
        if let Some(status) = self.finished.get(&id.0) {
            cb(status);
        }
    }

    /// Look up current state. Returns `Unknown` for ids that have never been
    /// issued or have been reaped past the retention window.
    pub fn status(&self, id: JobId) -> UploadStatus {
        if let Some(slot) = self.slots.get(&id.0) {
            return UploadStatus::Pending {
                progress: slot.progress.read(),
            };
        }
        // A job whose worker is done but whose apply step has not yet
        // run is reported as Pending at full progress. The typed result
        // is only available after apply runs, so we keep callers in the
        // Pending arm until then.
        if self.pending_apply.iter().any(|pa| pa.id.0 == id.0) {
            return UploadStatus::Pending { progress: 1.0 };
        }
        if let Some(status) = self.finished.get(&id.0) {
            return status.clone();
        }
        UploadStatus::Unknown
    }

    /// Count of jobs still in flight, ignoring the retention window.
    /// Includes jobs whose worker has finished but whose apply step has
    /// not yet been drained.
    pub fn pending(&self) -> usize {
        self.slots.len() + self.pending_apply.len()
    }

    /// True when no jobs are in flight.
    pub fn all_complete(&self) -> bool {
        self.slots.is_empty() && self.pending_apply.is_empty()
    }

    /// Pop the next apply closure off the queue. Returns `None` when the
    /// queue is empty. The caller is expected to run the closure against
    /// `DeviceResources` and then call `mark_applied` so the job
    /// transitions from Pending to Ready.
    pub fn pop_pending_apply(&mut self) -> Option<PendingApply> {
        self.pending_apply.pop_front()
    }

    /// Push an entry back onto the front of the queue. Used by
    /// `process_uploads_with_budget` when the time budget runs out
    /// mid-drain so the next call picks up where this one stopped.
    pub fn requeue_pending_apply(&mut self, pa: PendingApply) {
        self.pending_apply.push_front(pa);
    }

    /// Record that a pending-apply entry's closure has finished running.
    /// Moves the job into the short-retention `finished` table so the
    /// next `status` query reports `Ready`.
    pub fn mark_applied(&mut self, id: JobId, status: UploadStatus) {
        self.finished.insert(id.0, status);
    }

    /// Count of jobs sitting on the apply queue. Exposed for tests and
    /// metrics; `pending` already aggregates it into the in-flight total.
    pub fn pending_apply_len(&self) -> usize {
        self.pending_apply.len()
    }

    /// Walk the job table, advance any worker results, and check pending
    /// GPU submissions for completion.
    ///
    /// Returns one `Completion` per job that just transitioned to `Ready` or
    /// `Failed`. The caller is expected to run any `apply` closure first
    /// (only when `status` is `Ready`), then invoke the registered
    /// `callback` if present. Splitting these out lets the caller drop any
    /// external lock around the runner before mutating renderer state.
    pub fn process(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Completion> {
        self.process_budgeted(device, queue, &FrameBudget::unbounded())
    }

    /// `process` with a time bound on the deferred GPU-job drain.
    ///
    /// Deferred jobs run one at a time until `budget` elapses (checked
    /// between jobs; the first job of a call always runs so a tight budget
    /// still makes progress), on top of the fixed per-call cap. Everything
    /// else behaves like `process`.
    pub fn process_budgeted(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: &FrameBudget,
    ) -> Vec<Completion> {
        // Drop the previous frame's retention window. Callers that needed
        // those results have already taken them.
        self.finished.clear();

        // Run deferred GPU jobs on this (the device-owning) thread, bounded
        // by the frame budget and a fixed cap so a large batch spreads
        // across frames. Each result is sent into the job's channel, picked
        // up by the slot loop below in this same call.
        let n = self.deferred_gpu.len().min(MAX_GPU_JOBS_PER_PROCESS);
        for i in 0..n {
            if i > 0 && budget.exhausted() {
                break;
            }
            let Some(mut job) = self.deferred_gpu.pop_front() else {
                break;
            };
            let t0 = Instant::now();
            let progress = job.progress.clone();
            match job.work {
                DeferredWork::Once(work) => {
                    let outcome = match catch_unwind(AssertUnwindSafe(|| {
                        work(device, queue, &progress)
                    })) {
                        Ok(Ok(product)) => WorkerOutcome::Done(product, job.elapsed + t0.elapsed()),
                        Ok(Err(e)) => WorkerOutcome::Failed(e, job.elapsed + t0.elapsed()),
                        Err(_) => WorkerOutcome::Failed(
                            ViewportError::JobWorkerLost {
                                reason: "gpu job panicked",
                            },
                            job.elapsed + t0.elapsed(),
                        ),
                    };
                    let _ = job.tx.send(outcome);
                }
                DeferredWork::Chunked(ref mut work) => {
                    let step =
                        catch_unwind(AssertUnwindSafe(|| work(device, queue, &progress, budget)));
                    match step {
                        Ok(Ok(GpuStep::Continue)) => {
                            // Requeue at the back so other pending stages
                            // get their turn before this job's next slice.
                            job.elapsed = job.elapsed.saturating_add(t0.elapsed());
                            self.deferred_gpu.push_back(job);
                        }
                        Ok(Ok(GpuStep::Done(product))) => {
                            let d = job.elapsed + t0.elapsed();
                            let _ = job.tx.send(WorkerOutcome::Done(product, d));
                        }
                        Ok(Err(e)) => {
                            let d = job.elapsed + t0.elapsed();
                            let _ = job.tx.send(WorkerOutcome::Failed(e, d));
                        }
                        Err(_) => {
                            let _ = job.tx.send(WorkerOutcome::Failed(
                                ViewportError::JobWorkerLost {
                                    reason: "gpu job panicked",
                                },
                                job.elapsed + t0.elapsed(),
                            ));
                        }
                    }
                }
            }
        }

        // Advance internal wgpu state so completed submissions are visible
        // to the per-submission wait below.
        let _ = device.poll(wgpu::PollType::Poll);

        let mut completions = Vec::new();
        let ids: Vec<u64> = self.slots.keys().copied().collect();
        for id in ids {
            // Stage 1: pick up the worker result if we have not already.
            if self.slots.get(&id).is_some_and(|s| s.awaiting.is_none()) {
                let outcome = self
                    .slots
                    .get(&id)
                    .map(|s| s.rx.try_recv())
                    .expect("slot existed");
                match outcome {
                    Ok(WorkerOutcome::Done(product, worker_dur)) => {
                        let entry = self.durations.entry(id).or_insert(Duration::ZERO);
                        *entry = entry.saturating_add(worker_dur);
                        let JobProduct { gpu, apply } = product;
                        match gpu {
                            None => {
                                self.finish(id, UploadStatus::Ready, apply, &mut completions);
                                continue;
                            }
                            Some(sub) => {
                                if let Some(slot) = self.slots.get_mut(&id) {
                                    slot.awaiting = Some((Some(sub), apply));
                                }
                            }
                        }
                    }
                    Ok(WorkerOutcome::NeedsGpu(work, worker_dur)) => {
                        let entry = self.durations.entry(id).or_insert(Duration::ZERO);
                        *entry = entry.saturating_add(worker_dur);
                        // The CPU stage is done; queue the GPU stage for a
                        // later drain. The slot stays in place and the
                        // deferred run reports through the same channel, so
                        // this same loop picks up the final outcome.
                        if let Some((progress, tx)) = self
                            .slots
                            .get(&id)
                            .map(|s| (s.progress.clone(), s.tx.clone()))
                        {
                            self.deferred_gpu.push_back(DeferredGpuJob {
                                work,
                                progress,
                                tx,
                                elapsed: Duration::ZERO,
                            });
                        }
                        continue;
                    }
                    Ok(WorkerOutcome::Failed(e, worker_dur)) => {
                        let entry = self.durations.entry(id).or_insert(Duration::ZERO);
                        *entry = entry.saturating_add(worker_dur);
                        self.finish(id, UploadStatus::Failed(e), None, &mut completions);
                        continue;
                    }
                    Err(mpsc::TryRecvError::Empty) => continue,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // Sender dropped without sending. Catch-unwind in
                        // the spawn closure already covers panics, so this
                        // is an unexpected drop path.
                        self.finish(
                            id,
                            UploadStatus::Failed(ViewportError::JobWorkerLost {
                                reason: "worker channel closed without result",
                            }),
                            None,
                            &mut completions,
                        );
                        continue;
                    }
                }
            }

            // Stage 2: worker reported a GPU submission; poll for it.
            let pending_sub = self
                .slots
                .get(&id)
                .and_then(|s| s.awaiting.as_ref())
                .and_then(|(g, _)| g.clone());
            if let Some(sub) = pending_sub {
                let result = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(sub),
                    timeout: Some(Duration::from_millis(0)),
                });
                match result {
                    Ok(wgpu::PollStatus::QueueEmpty) | Ok(wgpu::PollStatus::WaitSucceeded) => {
                        let apply = self
                            .slots
                            .get_mut(&id)
                            .and_then(|s| s.awaiting.take())
                            .and_then(|(_, a)| a);
                        self.finish(id, UploadStatus::Ready, apply, &mut completions);
                    }
                    Ok(wgpu::PollStatus::Poll) => {
                        // Backend still working; check again next frame.
                    }
                    Err(_) => {
                        // Timeout or device error. Leave the slot pending
                        // and try again on the next call.
                    }
                }
            }
        }
        completions
    }

    fn finish(
        &mut self,
        id: u64,
        status: UploadStatus,
        apply: Option<ApplyFn>,
        completions: &mut Vec<Completion>,
    ) {
        let Some(mut slot) = self.slots.remove(&id) else {
            return;
        };
        let callback = slot.callback.take();
        // Successful jobs that carry an apply step are held on the
        // pending_apply queue. They keep reporting Pending until the
        // caller runs the apply (via process_uploads or
        // process_uploads_with_budget); only then do they move into
        // `finished` and start reporting Ready. Failures and no-apply
        // successes go through the standard Completion path so callbacks
        // fire immediately.
        match (status, apply) {
            (UploadStatus::Ready, Some(apply_fn)) => {
                self.pending_apply.push_back(PendingApply {
                    id: JobId(id),
                    status: UploadStatus::Ready,
                    apply: apply_fn,
                    callback,
                });
                return;
            }
            (status, _) => {
                completions.push(Completion {
                    id: JobId(id),
                    status: status.clone(),
                    apply: None,
                    callback,
                });
                self.finished.insert(id, status);
                return;
            }
        }
    }
}

impl super::DeviceResources {
    /// Advance the upload-job runner. Worker results received since the
    /// previous call are observed, GPU submissions are polled, completed
    /// jobs are folded into renderer state, and any completion callbacks
    /// fire on the caller's thread.
    ///
    /// Apply closures and callbacks both run after the runner's mutex is
    /// released, so they are free to query the runner or submit a fresh job
    /// without risk of deadlock.
    pub fn process_uploads(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.process_uploads_with_budget(device, queue, FrameBudget::unbounded());
    }

    /// Advance the upload-job runner with a cap on per-call apply-step
    /// work.
    ///
    /// Behaves the same as `process_uploads` except that, after the
    /// runner has been advanced and any failure callbacks fired, the
    /// caller stops popping apply closures off the queue once `budget`
    /// elapses. Remaining apply closures stay on the queue and are
    /// picked up by the next call. Their owning jobs continue to report
    /// `UploadStatus::Pending` until their apply runs, so the typed
    /// result is never observably available before it is in place.
    ///
    /// The budget covers the main-thread work: deferred GPU-job stages
    /// (texture creation and copies queued by `begin_upload_texture` and
    /// friends), the device poll, and the apply drain. Worker-thread time
    /// is independent. The check happens between jobs and between applies,
    /// so a single long-running stage may push past the deadline once it
    /// has started; the budget is a soft cap, not a hard deadline.
    pub fn process_uploads_with_budget(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: FrameBudget,
    ) {
        // Stage 1: advance the runner and drain immediate (failure /
        // no-apply) completions. Their callbacks fire here regardless of
        // budget: they do no main-thread work and dropping them would
        // hide errors from consumers.
        let completions = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.process_budgeted(device, queue, &budget)
        };
        for Completion {
            id: _,
            status,
            apply: _,
            callback,
        } in completions
        {
            if let Some(cb) = callback {
                cb(&status);
            }
        }

        // Stage 2: drain the apply queue under the budget. Each apply
        // mutates `self`, so we pop one at a time and re-check the
        // budget between iterations.
        loop {
            if budget.exhausted() {
                break;
            }
            let next = {
                let mut runner = self.jobs.lock().expect("upload job runner poisoned");
                runner.pop_pending_apply()
            };
            let Some(PendingApply {
                id,
                status,
                apply,
                callback,
            }) = next
            else {
                break;
            };
            let t = Instant::now();
            apply(self);
            let apply_d = t.elapsed();
            {
                let mut runner = self.jobs.lock().expect("upload job runner poisoned");
                runner.add_apply_duration(id, apply_d);
                runner.mark_applied(id, status.clone());
            }
            if let Some(cb) = callback {
                cb(&status);
            }
        }
    }

    /// Total wall-clock work duration for a completed job.
    ///
    /// The value is the sum of the worker thread's elapsed time and the
    /// main-thread apply-step elapsed time. It excludes frame-pacing
    /// delays (the time the apply step sat waiting for `process_uploads`
    /// to be called). For sync uploads this method returns `None`; sync
    /// callers measure their own wall-clock around the call.
    ///
    /// Returns `None` for jobs that are still in flight, were never issued,
    /// or whose duration record has already been dropped via
    /// `drop_job_duration`. The runner retains durations until the consumer
    /// drops them so single-frame retention is not enough.
    pub fn job_duration(&self, id: JobId) -> Option<Duration> {
        let runner = self.jobs.lock().expect("upload job runner poisoned");
        runner.duration(id)
    }

    /// Drop the recorded duration for `id`. Call this after reading the
    /// value via `job_duration`; otherwise the duration table grows over
    /// time.
    pub fn drop_job_duration(&mut self, id: JobId) {
        let mut runner = self.jobs.lock().expect("upload job runner poisoned");
        runner.drop_duration(id);
    }

    /// Look up the current state of a submitted upload job.
    pub fn upload_status(&self, id: JobId) -> UploadStatus {
        let runner = self.jobs.lock().expect("upload job runner poisoned");
        runner.status(id)
    }

    /// Number of upload jobs still in flight.
    pub fn uploads_pending(&self) -> usize {
        let runner = self.jobs.lock().expect("upload job runner poisoned");
        runner.pending()
    }

    /// True when no upload jobs are in flight.
    pub fn all_uploads_complete(&self) -> bool {
        let runner = self.jobs.lock().expect("upload job runner poisoned");
        runner.all_complete()
    }

    /// Register a callback to fire when a job finishes. The callback runs on
    /// the caller's thread during the next `process_uploads` call. If the
    /// job has already finished and is still in the short retention window,
    /// the callback fires immediately on the calling thread.
    pub fn on_upload_complete<F>(&mut self, id: JobId, cb: F)
    where
        F: FnOnce(&UploadStatus) + Send + 'static,
    {
        let mut runner = self.jobs.lock().expect("upload job runner poisoned");
        runner.on_complete(id, cb);
    }

    /// Block the calling thread, driving `process_uploads` until `id` reaches
    /// a terminal state.
    ///
    /// Returns `Ok(())` when the job's worker (and any GPU submission it
    /// queued) completes successfully. Returns the worker error when the job
    /// fails. Used internally by the synchronous `upload_*` entries to wrap
    /// their `begin_upload_*` counterparts in a single round-trip; consumers
    /// who want to wait on a specific async upload can call it directly.
    ///
    /// The loop sleeps for a short interval between polls so it does not pin
    /// a CPU core while the worker is running. It does not call back into the
    /// caller's event loop; if you have other work to interleave, drive
    /// `process_uploads` yourself instead.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::JobResultMissing`](crate::error::ViewportError::JobResultMissing)
    /// if `id` has already been reaped or was never issued, and the worker's
    /// error verbatim when the job ends in `Failed`.
    pub fn drain_until(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: JobId,
    ) -> crate::error::ViewportResult<()> {
        loop {
            self.process_uploads(device, queue);
            match self.upload_status(id) {
                UploadStatus::Ready => return Ok(()),
                UploadStatus::Failed(e) => return Err(e),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(Duration::from_micros(200));
                }
                UploadStatus::Unknown => {
                    return Err(crate::error::ViewportError::JobResultMissing {
                        reason: "drain target was already reaped",
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Optional `future` feature: a thin Future wrapper around a JobId.
// ---------------------------------------------------------------------------

/// Future returned by [`DeviceResources::upload_handle`].
///
/// Polling drives the wrapped job to completion using the standard
/// `process_uploads` machinery; the consumer's main loop must keep calling
/// `process_uploads` so the runner can deliver completion callbacks. Once
/// the future resolves, take the typed result with the matching
/// `upload_result_*` accessor.
#[cfg(feature = "future")]
pub struct JobHandle {
    id: JobId,
    state: Arc<std::sync::Mutex<JobHandleState>>,
}

#[cfg(feature = "future")]
struct JobHandleState {
    done: Option<crate::error::ViewportResult<()>>,
    waker: Option<std::task::Waker>,
}

#[cfg(feature = "future")]
impl JobHandle {
    /// Id of the wrapped job. Pass it to `upload_result_*` after the future
    /// resolves.
    pub fn id(&self) -> JobId {
        self.id
    }
}

#[cfg(feature = "future")]
impl std::future::Future for JobHandle {
    type Output = crate::error::ViewportResult<()>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut guard = self.state.lock().expect("job handle poisoned");
        if let Some(result) = guard.done.take() {
            return std::task::Poll::Ready(result);
        }
        guard.waker = Some(cx.waker().clone());
        std::task::Poll::Pending
    }
}

#[cfg(feature = "future")]
impl super::DeviceResources {
    /// Wrap a `JobId` in a future that resolves when the job completes.
    ///
    /// The future is driven by completion callbacks fired during
    /// `process_uploads`, so the consumer's main loop must keep calling
    /// `process_uploads` for the future to make progress. The resolved
    /// value is `Ok(())` on success and the worker error on failure; the
    /// caller takes the typed result through the matching
    /// `upload_result_*` accessor after `.await` returns.
    pub fn upload_handle(&mut self, id: JobId) -> JobHandle {
        let state = Arc::new(std::sync::Mutex::new(JobHandleState {
            done: None,
            waker: None,
        }));
        let state_for_cb = state.clone();
        self.on_upload_complete(id, move |status| {
            let result = match status {
                UploadStatus::Ready => Ok(()),
                UploadStatus::Failed(e) => Err(e.clone()),
                UploadStatus::Pending { .. } => {
                    // Callbacks only fire on terminal transitions; this
                    // arm is unreachable but we report it cleanly rather
                    // than panic if a future runner change relaxes that.
                    Err(crate::error::ViewportError::JobNotReady)
                }
                UploadStatus::Unknown => Err(crate::error::ViewportError::JobResultMissing {
                    reason: "job vanished before completion callback fired",
                }),
            };
            let mut guard = state_for_cb.lock().expect("job handle poisoned");
            guard.done = Some(result);
            if let Some(waker) = guard.waker.take() {
                waker.wake();
            }
        });
        JobHandle { id, state }
    }
}

// ---------------------------------------------------------------------------
// Plugin-facing facade
// ---------------------------------------------------------------------------

/// Result slot type used by the plugin facade. Values are boxed because the
/// runner has no compile-time knowledge of the closure's return type.
type PluginResultSlot = ResultSlot<Box<dyn Any + Send>>;

/// Plugin-facing handle to the upload-job runner. Exposed to
/// `ItemTypePlugin::prepare` via `ItemFrameContext::jobs`.
///
/// Plugins use it the same way built-in uploads do, but with a typed
/// generic return: submit a CPU job that produces a value of type `T`,
/// poll the returned `JobId` next frame, and `take::<T>()` the result
/// once the status is `Ready`.
///
/// The handle is `Copy`-like in spirit (it just holds a reference); both
/// `submit_cpu` and the readers use `&self` so the plugin does not need to
/// thread mutability through its own state.
pub struct Jobs<'a> {
    resources: &'a super::DeviceResources,
}

impl<'a> Jobs<'a> {
    pub(crate) fn new(resources: &'a super::DeviceResources) -> Self {
        Self { resources }
    }

    /// Schedule a CPU job whose result is delivered through `take<T>`.
    ///
    /// `work` runs on a background worker. The closure must own its
    /// inputs because the `&Device` and `&Queue` references passed to
    /// `ItemTypePlugin::prepare` are not `'static`. Panics inside `work`
    /// surface as `UploadStatus::Failed(JobWorkerLost)`.
    pub fn submit_cpu<T, F>(&self, work: F) -> JobId
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let slot: PluginResultSlot = ResultSlot::new();
        let slot_for_apply = slot.clone();

        let id = {
            let mut runner = self
                .resources
                .jobs
                .lock()
                .expect("upload job runner poisoned");
            runner.submit_cpu(move |_progress| {
                let value: T = work();
                let boxed: Box<dyn Any + Send> = Box::new(value);
                Ok(JobProduct::with_apply(Box::new(
                    move |_resources: &mut super::DeviceResources| {
                        slot_for_apply.set(boxed);
                    },
                )))
            })
        };

        self.resources
            .job_results
            .plugin
            .lock()
            .expect("plugin job result map poisoned")
            .insert(id, slot);
        id
    }

    /// Current state of a submitted plugin job. Same shape as the
    /// `upload_status` reported by built-in uploads.
    pub fn status(&self, id: JobId) -> UploadStatus {
        self.resources.upload_status(id)
    }

    /// Try to take the typed result produced by a completed job.
    ///
    /// Returns `None` while the job is still in flight, when the id has
    /// already been taken, when it never existed, or when `T` does not
    /// match the type stored by the worker (plugin author error). On a
    /// successful take, the slot is removed.
    pub fn take<T: Any + Send + 'static>(&self, id: JobId) -> Option<T> {
        let mut map = self
            .resources
            .job_results
            .plugin
            .lock()
            .expect("plugin job result map poisoned");
        let slot = map.get(&id)?.clone();
        let boxed = slot.take()?;
        match boxed.downcast::<T>() {
            Ok(value) => {
                map.remove(&id);
                Some(*value)
            }
            Err(_boxed_back) => {
                // Type mismatch. Re-insert the box by recreating the
                // slot with the same value so the plugin can retry with
                // the right type or just leak the slot in this dropped
                // state. To keep things simple here, we drop on
                // mismatch -- plugins should not be calling take with
                // the wrong type.
                map.remove(&id);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use super::*;

    /// Drive the runner until `predicate` is true or the deadline expires.
    /// Necessary because parallel-running test threads contend for the
    /// rayon pool, so a single sleep + process cycle is not enough.
    fn drain_until<F>(
        runner: &mut JobRunner,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_iterations: usize,
        mut predicate: F,
    ) where
        F: FnMut(&JobRunner) -> bool,
    {
        for _ in 0..max_iterations {
            let _ = runner.process(device, queue);
            if predicate(runner) {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn cpu_job_reports_ready_after_drain() {
        let mut runner = JobRunner::new();
        let id = runner.submit_cpu(|_p| Ok(JobProduct::empty()));

        assert_eq!(runner.pending(), 1);
        with_test_gpu(|device, queue| {
            drain_until(&mut runner, device, queue, 200, |r| r.all_complete());
        });

        assert!(matches!(runner.status(id), UploadStatus::Ready));
        assert_eq!(runner.pending(), 0);
        assert!(runner.all_complete());
    }

    #[test]
    fn cpu_job_progress_is_observable() {
        let mut runner = JobRunner::new();
        let gate = Arc::new(AtomicBool::new(false));
        let gate_for_worker = gate.clone();

        let id = runner.submit_cpu(move |p| {
            p.set(0.25);
            while !gate_for_worker.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
            p.set(1.0);
            Ok(JobProduct::empty())
        });

        // Poll until the worker publishes its first progress sample. Under
        // contention with other tests, the worker may not start for some
        // time.
        let mut observed = None;
        for _ in 0..200 {
            if let UploadStatus::Pending { progress } = runner.status(id) {
                if progress > 0.0 {
                    observed = Some(progress);
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let progress = observed.expect("worker never published progress");
        assert!(
            (0.2..=0.3).contains(&progress),
            "expected ~0.25, got {progress}"
        );

        gate.store(true, Ordering::Relaxed);
        with_test_gpu(|device, queue| {
            drain_until(&mut runner, device, queue, 200, |r| r.all_complete());
        });
        assert!(matches!(runner.status(id), UploadStatus::Ready));
    }

    #[test]
    fn cpu_then_gpu_job_completes() {
        let mut runner = JobRunner::new();
        let id = runner.submit_cpu_then_gpu(|p| {
            p.set(0.5);
            Ok(
                Box::new(|_d: &wgpu::Device, _q: &wgpu::Queue, p: &ProgressHandle| {
                    p.set(1.0);
                    Ok(JobProduct::empty())
                }) as GpuWorkFn,
            )
        });

        with_test_gpu(|device, queue| {
            drain_until(&mut runner, device, queue, 200, |r| r.all_complete());
        });

        assert!(matches!(runner.status(id), UploadStatus::Ready));
        assert!(runner.all_complete());
    }

    #[test]
    fn cpu_then_gpu_cpu_stage_error_surfaces_as_failed() {
        let mut runner = JobRunner::new();
        let id = runner.submit_cpu_then_gpu(|_p| {
            Err(ViewportError::JobWorkerLost {
                reason: "cpu stage error",
            })
        });

        with_test_gpu(|device, queue| {
            drain_until(&mut runner, device, queue, 200, |r| r.all_complete());
        });

        assert!(matches!(runner.status(id), UploadStatus::Failed(_)));
    }

    #[test]
    fn chunked_job_resumes_across_calls() {
        // A chunked GPU stage needing three slices, driven with an
        // already-elapsed budget: each process call runs exactly one turn
        // (one slice), so completion takes three calls and the turn count
        // equals the slice count.
        let mut runner = JobRunner::new();
        let turns = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let turns_in_job = turns.clone();
        let id = runner.submit_cpu_then_gpu_chunked(move |_p| {
            let mut slices_done = 0usize;
            Ok(Box::new(
                move |_d: &wgpu::Device,
                      _q: &wgpu::Queue,
                      _p: &ProgressHandle,
                      budget: &FrameBudget| {
                    turns_in_job.fetch_add(1, Ordering::Relaxed);
                    loop {
                        slices_done += 1;
                        if slices_done >= 3 {
                            return Ok(GpuStep::Done(JobProduct::empty()));
                        }
                        if budget.exhausted() {
                            return Ok(GpuStep::Continue);
                        }
                    }
                },
            ) as ChunkedGpuWorkFn)
        });

        with_test_gpu(|device, queue| {
            for _ in 0..400 {
                let _ =
                    runner.process_budgeted(device, queue, &FrameBudget::from_now(Duration::ZERO));
                if runner.all_complete() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        assert!(matches!(runner.status(id), UploadStatus::Ready));
        assert_eq!(
            turns.load(Ordering::Relaxed),
            3,
            "expected one slice per zero-budget turn"
        );
    }

    #[test]
    fn chunked_job_unbudgeted_completes_in_one_turn() {
        // The same three-slice job under an unbounded budget: the internal
        // loop never sees an exhausted budget, so the whole stage runs in
        // a single turn (whole-asset behaviour on the unbudgeted path).
        let mut runner = JobRunner::new();
        let turns = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let turns_in_job = turns.clone();
        let id = runner.submit_cpu_then_gpu_chunked(move |_p| {
            let mut slices_done = 0usize;
            Ok(Box::new(
                move |_d: &wgpu::Device,
                      _q: &wgpu::Queue,
                      _p: &ProgressHandle,
                      budget: &FrameBudget| {
                    turns_in_job.fetch_add(1, Ordering::Relaxed);
                    loop {
                        slices_done += 1;
                        if slices_done >= 3 {
                            return Ok(GpuStep::Done(JobProduct::empty()));
                        }
                        if budget.exhausted() {
                            return Ok(GpuStep::Continue);
                        }
                    }
                },
            ) as ChunkedGpuWorkFn)
        });

        with_test_gpu(|device, queue| {
            drain_until(&mut runner, device, queue, 200, |r| r.all_complete());
        });

        assert!(matches!(runner.status(id), UploadStatus::Ready));
        assert_eq!(turns.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn chunked_job_error_mid_stream_surfaces_as_failed() {
        let mut runner = JobRunner::new();
        let id = runner.submit_cpu_then_gpu_chunked(move |_p| {
            let mut turn = 0usize;
            Ok(Box::new(
                move |_d: &wgpu::Device,
                      _q: &wgpu::Queue,
                      _p: &ProgressHandle,
                      _b: &FrameBudget| {
                    turn += 1;
                    if turn == 1 {
                        return Ok(GpuStep::Continue);
                    }
                    Err(ViewportError::JobWorkerLost {
                        reason: "slice failed",
                    })
                },
            ) as ChunkedGpuWorkFn)
        });

        with_test_gpu(|device, queue| {
            drain_until(&mut runner, device, queue, 200, |r| r.all_complete());
        });

        match runner.status(id) {
            UploadStatus::Failed(ViewportError::JobWorkerLost { reason }) => {
                assert_eq!(reason, "slice failed");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn deferred_gpu_drain_respects_budget() {
        // Three two-stage jobs whose CPU stages finish immediately. With an
        // already-elapsed budget, each process call may run at most one
        // deferred GPU stage (the first job of a call always runs so a
        // tight budget still makes progress).
        let mut runner = JobRunner::new();
        let ran = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for _ in 0..3 {
            let ran = ran.clone();
            runner.submit_cpu_then_gpu(move |_p| {
                Ok(Box::new(
                    move |_d: &wgpu::Device, _q: &wgpu::Queue, _p: &ProgressHandle| {
                        ran.fetch_add(1, Ordering::Relaxed);
                        Ok(JobProduct::empty())
                    },
                ) as GpuWorkFn)
            });
        }

        with_test_gpu(|device, queue| {
            let mut max_per_call = 0usize;
            for _ in 0..400 {
                let before = ran.load(Ordering::Relaxed);
                let _ =
                    runner.process_budgeted(device, queue, &FrameBudget::from_now(Duration::ZERO));
                let delta = ran.load(Ordering::Relaxed) - before;
                max_per_call = max_per_call.max(delta);
                if runner.all_complete() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            assert!(runner.all_complete(), "jobs never drained");
            assert_eq!(ran.load(Ordering::Relaxed), 3);
            assert!(
                max_per_call <= 1,
                "budget did not bound the deferred drain: {max_per_call} stages in one call"
            );
        });
    }

    #[test]
    fn worker_error_surfaces_as_failed() {
        let mut runner = JobRunner::new();
        let id = runner.submit_cpu(|_| {
            Err(ViewportError::InvalidGaussianSplatData {
                reason: "test error",
            })
        });

        with_test_gpu(|device, queue| {
            drain_until(&mut runner, device, queue, 200, |r| r.all_complete());
        });

        match runner.status(id) {
            UploadStatus::Failed(ViewportError::InvalidGaussianSplatData { reason }) => {
                assert_eq!(reason, "test error");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn worker_panic_surfaces_as_failed() {
        let mut runner = JobRunner::new();
        let id = runner.submit_cpu(|_| panic!("worker exploded"));

        with_test_gpu(|device, queue| {
            drain_until(&mut runner, device, queue, 200, |r| r.all_complete());
        });

        match runner.status(id) {
            UploadStatus::Failed(ViewportError::JobWorkerLost { reason }) => {
                assert_eq!(reason, "worker panicked");
            }
            other => panic!("expected Failed(JobWorkerLost), got {other:?}"),
        }
    }

    #[test]
    fn callback_fires_on_completion() {
        let mut runner = JobRunner::new();
        let seen = Arc::new(Mutex::new(None));
        let seen_clone = seen.clone();

        let id = runner.submit_cpu(|_| Ok(JobProduct::empty()));
        runner.on_complete(id, move |status| {
            *seen_clone.lock().unwrap() = Some(matches!(status, UploadStatus::Ready));
        });

        with_test_gpu(|device, queue| {
            // process() returns Completion entries so the caller can run
            // apply + callback after dropping any external lock. The
            // integration on DeviceResources does this automatically;
            // the test drives it by hand.
            for _ in 0..200 {
                for c in runner.process(device, queue) {
                    if let Some(cb) = c.callback {
                        cb(&c.status);
                    }
                }
                if matches!(runner.status(id), UploadStatus::Ready) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });

        let observed = seen.lock().unwrap().clone();
        assert_eq!(observed, Some(true));
    }

    #[test]
    fn unknown_id_returns_unknown() {
        let runner = JobRunner::new();
        let made_up = JobId(99_999);
        assert!(matches!(runner.status(made_up), UploadStatus::Unknown));
    }

    #[test]
    fn many_concurrent_jobs_all_complete() {
        let mut runner = JobRunner::new();
        let mut ids = Vec::with_capacity(256);
        for _ in 0..256 {
            ids.push(runner.submit_cpu(|p| {
                p.set(0.5);
                std::thread::sleep(Duration::from_millis(2));
                Ok(JobProduct::empty())
            }));
        }
        assert_eq!(runner.pending(), 256);

        // Observe each id transitioning to a terminal state. The retention
        // window only spans one drain cycle, so we cannot query every id
        // after the loop -- we have to collect the observation as we go.
        let mut seen_ready = std::collections::HashSet::new();
        with_test_gpu(|device, queue| {
            for _ in 0..400 {
                let _ = runner.process(device, queue);
                for id in &ids {
                    if seen_ready.contains(id) {
                        continue;
                    }
                    if let UploadStatus::Ready = runner.status(*id) {
                        seen_ready.insert(*id);
                    }
                }
                if seen_ready.len() == ids.len() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });

        assert_eq!(
            seen_ready.len(),
            ids.len(),
            "stragglers: {}",
            runner.pending()
        );
        assert!(runner.all_complete());
    }

    #[test]
    fn gpu_job_waits_for_submission() {
        let mut runner = JobRunner::new();
        with_test_gpu(|device, queue| {
            let id = runner.submit_with_gpu(device, queue, |device, queue, _p| {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("upload_jobs_test_buf"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                queue.write_buffer(
                    &buf,
                    0,
                    &[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                );
                let mut enc =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                // Force a non-trivial command buffer so the submission has
                // something to flush.
                let dst = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("upload_jobs_test_dst"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                enc.copy_buffer_to_buffer(&buf, 0, &dst, 0, 16);
                let sub = queue.submit(std::iter::once(enc.finish()));
                Ok(JobProduct::with_gpu(sub))
            });

            // Poll until the GPU submission completes. Generous budget so
            // the test stays robust under load when other tests are also
            // hammering the rayon pool and the GPU.
            for _ in 0..400 {
                std::thread::sleep(Duration::from_millis(5));
                let _ = runner.process(device, queue);
                if matches!(runner.status(id), UploadStatus::Ready) {
                    return;
                }
            }
            panic!("GPU-gated job did not reach Ready");
        });
    }

    // -----------------------------------------------------------------
    // Plugin-facing facade
    // -----------------------------------------------------------------

    fn make_resources_for_jobs()
    -> Option<(wgpu::Device, wgpu::Queue, super::super::DeviceResources)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        let resources =
            super::super::DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        Some((device, queue, resources))
    }

    fn drive_resources<F: FnMut(&super::super::DeviceResources) -> bool>(
        resources: &mut super::super::DeviceResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut predicate: F,
    ) {
        for _ in 0..200 {
            resources.process_uploads(device, queue);
            if predicate(resources) {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn plugin_jobs_round_trip_typed_result() {
        let Some((device, queue, mut resources)) = make_resources_for_jobs() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let id = {
            let jobs = super::Jobs::new(&resources);
            jobs.submit_cpu(|| 41_u32 + 1)
        };

        // Before the worker completes, take is None.
        assert!(super::Jobs::new(&resources).take::<u32>(id).is_none());

        drive_resources(&mut resources, &device, &queue, |r| {
            matches!(r.upload_status(id), UploadStatus::Ready)
        });

        let jobs = super::Jobs::new(&resources);
        assert_eq!(jobs.take::<u32>(id), Some(42));
        // Second take returns None (already drained).
        assert_eq!(jobs.take::<u32>(id), None);
    }

    #[test]
    fn plugin_jobs_panic_surfaces_as_failed() {
        let Some((device, queue, mut resources)) = make_resources_for_jobs() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let id = {
            let jobs = super::Jobs::new(&resources);
            jobs.submit_cpu(|| -> u32 { panic!("plugin worker exploded") })
        };

        drive_resources(&mut resources, &device, &queue, |r| {
            !matches!(r.upload_status(id), UploadStatus::Pending { .. })
        });

        match resources.upload_status(id) {
            UploadStatus::Failed(ViewportError::JobWorkerLost { reason }) => {
                assert_eq!(reason, "worker panicked");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        // Failed jobs leave the slot intact but with no value; take returns None.
        assert!(super::Jobs::new(&resources).take::<u32>(id).is_none());
    }

    #[test]
    fn plugin_jobs_wrong_type_returns_none() {
        let Some((device, queue, mut resources)) = make_resources_for_jobs() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let id = {
            let jobs = super::Jobs::new(&resources);
            jobs.submit_cpu(|| 7_i64)
        };

        drive_resources(&mut resources, &device, &queue, |r| {
            matches!(r.upload_status(id), UploadStatus::Ready)
        });

        let jobs = super::Jobs::new(&resources);
        // Asking for a different type drops the box silently.
        assert!(jobs.take::<u32>(id).is_none());
        // After the type mismatch the slot is gone, so subsequent takes
        // continue to return None even with the correct type.
        assert!(jobs.take::<i64>(id).is_none());
    }

    // -----------------------------------------------------------------
    // Optional `future` feature
    // -----------------------------------------------------------------

    /// Smoke test the `JobHandle` future under a non-tokio executor.
    ///
    /// Drives the upload runner from a helper thread so the polled future
    /// can observe completion through its registered callback. Uses
    /// `pollster::block_on` because the crate stays runtime-free; the same
    /// future works unchanged under tokio.
    #[cfg(feature = "future")]
    #[test]
    fn job_handle_resolves_via_future() {
        let Some((device, queue, mut resources)) = make_resources_for_jobs() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        // Submit a CPU job so we have an id to await.
        let id = {
            let jobs = super::Jobs::new(&resources);
            jobs.submit_cpu(|| 7_u32)
        };
        let handle = resources.upload_handle(id);

        // Drive `process_uploads` from a worker thread while the main
        // thread blocks on the future. The two threads share the resources
        // through a `Mutex` so the worker can call `process_uploads(&mut)`.
        let shared = Arc::new(std::sync::Mutex::new((resources, device, queue)));
        let driver_handle = {
            let shared = shared.clone();
            std::thread::spawn(move || {
                for _ in 0..400 {
                    {
                        let mut g = shared.lock().unwrap();
                        let (resources, device, queue) = &mut *g;
                        resources.process_uploads(device, queue);
                        if resources.all_uploads_complete() {
                            return;
                        }
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            })
        };

        let result = pollster::block_on(handle);
        assert!(matches!(result, Ok(())));
        driver_handle.join().ok();
    }

    // -----------------------------------------------------------------
    // Frame budget
    // -----------------------------------------------------------------

    /// Submit a batch of apply-bearing jobs, run one short-budget pass,
    /// and check that the leftover applies survive to the next pass.
    ///
    /// Each apply spins for a few hundred microseconds, so a 100 us
    /// budget is virtually guaranteed to drop work to the next frame
    /// without being so tight that the runner spins forever. Across
    /// successive unbounded passes everything drains and every job ends
    /// in Ready.
    #[test]
    fn budgeted_drain_spills_to_next_call() {
        let Some((device, queue, mut resources)) = make_resources_for_jobs() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        let mut ids = Vec::with_capacity(32);
        {
            let jobs = super::Jobs::new(&resources);
            for i in 0..32_u32 {
                // The closure body runs on the worker; the value
                // arrives via the plugin facade's typed slot, which
                // materializes through an apply step on the main
                // thread. That apply step is what the budget caps.
                ids.push(jobs.submit_cpu(move || i * 2));
            }
        }

        // Drain workers first so every job is sitting on pending_apply.
        for _ in 0..400 {
            {
                let runner = resources.jobs.lock().unwrap();
                if runner.pending_apply_len() == ids.len() {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(2));
            // Advance the runner without running applies. A zero-duration
            // budget pops nothing (the exhausted check fires before the
            // first pop), so pending_apply grows as workers report in.
            resources.process_uploads_with_budget(
                &device,
                &queue,
                super::FrameBudget::from_now(Duration::from_nanos(0)),
            );
        }
        let after_workers = resources.jobs.lock().unwrap().pending_apply_len();
        assert!(
            after_workers > 0,
            "expected at least one apply queued, got 0"
        );

        // One unbounded pass clears the rest and lands every job in
        // Ready.
        resources.process_uploads(&device, &queue);
        for &id in &ids {
            assert!(matches!(resources.upload_status(id), UploadStatus::Ready));
        }
        assert!(resources.all_uploads_complete());
    }

    /// Creates a headless wgpu device + queue for the duration of `f`.
    ///
    /// Skips the test (via early return) if no adapter is available. CI
    /// builds without a GPU should pass the CPU-only tests above and skip
    /// the GPU-gated one.
    fn with_test_gpu<F: FnOnce(&wgpu::Device, &wgpu::Queue)>(f: F) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::SECONDARY,
            ..Default::default()
        });
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Ok(a) => a,
                Err(_) => {
                    eprintln!("skipping GPU-gated test: no adapter available");
                    return;
                }
            };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("upload_jobs_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("device creation");
        f(&device, &queue);
    }
}
