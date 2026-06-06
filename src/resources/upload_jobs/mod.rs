//! Internal job runner used by the async upload entry points.
//!
//! Every long-running upload on `ViewportGpuResources` routes through this
//! runner. A submitted job runs its CPU work on a background thread, may
//! optionally submit GPU commands, and reports completion on the main thread
//! during `process_uploads`. Callers query progress through `upload_status`
//! and learn about completion either by polling or by attaching a callback.
//!
//! No upload entry points use the runner yet. Real submitters will land
//! alongside the async variants of each existing `upload_*` method.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use crate::error::ViewportError;

/// Opaque handle for a submitted job. Returned by every `begin_upload_*`
/// entry and accepted by status queries, completion callbacks, and the
/// per-type result accessors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JobId(u64);

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
/// built textures, buffers, and bind groups into `ViewportGpuResources`.
pub type ApplyFn = Box<dyn FnOnce(&mut super::ViewportGpuResources) + Send>;

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
/// `gpu`; workers that need to store results on `ViewportGpuResources`
/// fill `apply`.
pub struct JobProduct {
    /// `Some` when the worker has submitted commands that must complete
    /// before the job can be reported `Ready`. The runner gates on this
    /// submission via `device.poll`.
    pub gpu: Option<wgpu::SubmissionIndex>,
    /// `Some` when the worker has built state that must be folded into
    /// `ViewportGpuResources` from the main thread.
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

/// Outcome the worker sends through its channel.
#[allow(dead_code)]
enum WorkerOutcome {
    Done(JobProduct),
    Failed(ViewportError),
}

type CompletionCallback = Box<dyn FnOnce(&UploadStatus) + Send>;

/// What `process` hands back for a single completed job. The caller is
/// responsible for running `apply` (if present and the status is `Ready`)
/// against the live `ViewportGpuResources`, then invoking `callback`.
pub struct Completion {
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
    rx: mpsc::Receiver<WorkerOutcome>,
    /// Once the worker has reported, the GPU submission to gate on (if any)
    /// and the apply closure to run when the GPU side finishes.
    awaiting: Option<(Option<wgpu::SubmissionIndex>, Option<ApplyFn>)>,
    callback: Option<CompletionCallback>,
}

/// Background worker pool plus the table of in-flight jobs.
///
/// Owned by `ViewportGpuResources`; reached from there via
/// `process_uploads`, `upload_status`, and friends. `next_id` and the
/// `submit_*` helpers are unused until upload entry points are wired
/// through the runner.
#[allow(dead_code)]
pub struct JobRunner {
    next_id: u64,
    slots: HashMap<u64, JobSlot>,
    /// Recently finished jobs, kept for one drain cycle so callers can still
    /// see `Ready` or `Failed` after the completion frame.
    finished: HashMap<u64, UploadStatus>,
}

impl Default for JobRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl JobRunner {
    /// Construct an empty runner. The `ViewportGpuResources` initializer
    /// holds the single instance; callers do not construct this directly.
    pub fn new() -> Self {
        Self {
            next_id: 1,
            slots: HashMap::new(),
            finished: HashMap::new(),
        }
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

        rayon::spawn(move || {
            let outcome = match catch_unwind(AssertUnwindSafe(|| work(&worker_progress))) {
                Ok(Ok(product)) => WorkerOutcome::Done(product),
                Ok(Err(e)) => WorkerOutcome::Failed(e),
                Err(_) => WorkerOutcome::Failed(ViewportError::JobWorkerLost {
                    reason: "worker panicked",
                }),
            };
            // Receiver going away is fine; the runner was probably dropped.
            let _ = tx.send(outcome);
        });

        self.slots.insert(
            id.0,
            JobSlot {
                progress,
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
        let worker_progress = progress.clone();
        let (tx, rx) = mpsc::channel();
        let device = device.clone();
        let queue = queue.clone();

        rayon::spawn(move || {
            let outcome =
                match catch_unwind(AssertUnwindSafe(|| work(&device, &queue, &worker_progress))) {
                    Ok(Ok(product)) => WorkerOutcome::Done(product),
                    Ok(Err(e)) => WorkerOutcome::Failed(e),
                    Err(_) => WorkerOutcome::Failed(ViewportError::JobWorkerLost {
                        reason: "worker panicked",
                    }),
                };
            let _ = tx.send(outcome);
        });

        self.slots.insert(
            id.0,
            JobSlot {
                progress,
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
        if let Some(status) = self.finished.get(&id.0) {
            return status.clone();
        }
        UploadStatus::Unknown
    }

    /// Count of jobs still in flight, ignoring the retention window.
    pub fn pending(&self) -> usize {
        self.slots.len()
    }

    /// True when no jobs are in flight.
    pub fn all_complete(&self) -> bool {
        self.slots.is_empty()
    }

    /// Walk the job table, advance any worker results, and check pending
    /// GPU submissions for completion.
    ///
    /// Returns one `Completion` per job that just transitioned to `Ready` or
    /// `Failed`. The caller is expected to run any `apply` closure first
    /// (only when `status` is `Ready`), then invoke the registered
    /// `callback` if present. Splitting these out lets the caller drop any
    /// external lock around the runner before mutating renderer state.
    pub fn process(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Vec<Completion> {
        // Drop the previous frame's retention window. Callers that needed
        // those results have already taken them.
        self.finished.clear();

        // Advance internal wgpu state so completed submissions are visible
        // to the per-submission wait below.
        let _ = device.poll(wgpu::PollType::Poll);

        let mut completions = Vec::new();
        let ids: Vec<u64> = self.slots.keys().copied().collect();
        for id in ids {
            // Stage 1: pick up the worker result if we have not already.
            if self
                .slots
                .get(&id)
                .is_some_and(|s| s.awaiting.is_none())
            {
                let outcome = self
                    .slots
                    .get(&id)
                    .map(|s| s.rx.try_recv())
                    .expect("slot existed");
                match outcome {
                    Ok(WorkerOutcome::Done(product)) => {
                        let JobProduct { gpu, apply } = product;
                        match gpu {
                            None => {
                                self.finish(
                                    id,
                                    UploadStatus::Ready,
                                    apply,
                                    &mut completions,
                                );
                                continue;
                            }
                            Some(sub) => {
                                if let Some(slot) = self.slots.get_mut(&id) {
                                    slot.awaiting = Some((Some(sub), apply));
                                }
                            }
                        }
                    }
                    Ok(WorkerOutcome::Failed(e)) => {
                        self.finish(
                            id,
                            UploadStatus::Failed(e),
                            None,
                            &mut completions,
                        );
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
                    Ok(wgpu::PollStatus::QueueEmpty)
                    | Ok(wgpu::PollStatus::WaitSucceeded) => {
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
        completions.push(Completion {
            status: status.clone(),
            apply,
            callback: slot.callback.take(),
        });
        self.finished.insert(id, status);
    }
}

impl super::ViewportGpuResources {
    /// Advance the upload-job runner. Worker results received since the
    /// previous call are observed, GPU submissions are polled, completed
    /// jobs are folded into renderer state, and any completion callbacks
    /// fire on the caller's thread.
    ///
    /// Apply closures and callbacks both run after the runner's mutex is
    /// released, so they are free to query the runner or submit a fresh job
    /// without risk of deadlock.
    pub fn process_uploads(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let completions = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.process(device, queue)
        };
        for Completion {
            status,
            apply,
            callback,
        } in completions
        {
            if matches!(status, UploadStatus::Ready) {
                if let Some(apply) = apply {
                    apply(self);
                }
            }
            if let Some(cb) = callback {
                cb(&status);
            }
        }
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
            // integration on ViewportGpuResources does this automatically;
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

        assert_eq!(seen_ready.len(), ids.len(), "stragglers: {}", runner.pending());
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
                queue.write_buffer(&buf, 0, &[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
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
        let adapter = match pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        )) {
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
