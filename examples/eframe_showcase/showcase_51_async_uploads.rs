//! Showcase 51: Async asset streaming
//!
//! Demonstrates the upload-job system. Press one of the asset buttons to
//! kick off a load. In Async mode the call returns immediately and the
//! orbit camera keeps moving while the worker runs; in Sync mode the call
//! blocks the calling thread and the camera visibly stutters.
//!
//! Each asset shows up in the scene once its job lands:
//!
//! - Env-map: HDR irradiance + prefilter + BRDF LUT, lights the scene.
//! - Mesh: a small textured sphere appears to the right.
//! - Texture: a checkered cube appears further right.
//! - Skin weights: a sphere on the left is marked skinnable.
//!
//! Other asset buttons (volume, sprite set, gaussian splat, overlay
//! texture) are greyed out and land in J5.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use eframe::egui;
use viewport_lib::{
    ColourmapId, GaussianSplatData, GaussianSplatId, GlyphItem, GlyphSetId, GlyphSetRefItem, JobId,
    LightKind, LightSource, LightingSettings, Material, MeshData, MeshId, OverlayTextureId,
    PointCloudId, PointCloudItem, PointCloudRefItem, PolylineId, PolylineItem, PolylineRefItem,
    RibbonId, RibbonItem, RibbonRefItem, SceneRenderItem, SkinWeights, SpriteInstanceSetId,
    SpriteItem, SpriteSetId, StreamtubeId, StreamtubeItem, StreamtubeRefItem, TensorGlyphItem,
    TensorGlyphSetId, TensorGlyphSetRefItem, TubeId, TubeItem, TubeRefItem, UploadStatus,
    ViewportRenderer, VolumeId, VolumeItem,
};

use crate::App;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Per-asset lifecycle. Drives both the button enable state and the
/// in-flight progress panel.
///
/// `InFlight` carries the launch instant so the controls panel can show a
/// live elapsed clock. `Loaded` and `Failed` carry the final wall-clock
/// duration in milliseconds so the user can compare sync vs async runs at
/// a glance.
#[derive(Clone)]
pub(crate) enum AssetState {
    Idle,
    InFlight {
        job: JobId,
        progress: f32,
        started: Instant,
    },
    Loaded {
        duration_ms: u64,
    },
    Failed {
        reason: String,
        duration_ms: u64,
    },
}

/// Payload size class. Light is the original tiny-asset scene; Heavy
/// scales meshes, textures, volumes, point clouds, and splats up to
/// sizes that actually exercise the upload pipeline. The whole point of
/// the showcase is to compare sync vs async under load, so Heavy is the
/// default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PayloadSize {
    Light,
    Heavy,
}

pub(crate) struct AsyncUploadsState {
    /// Handle to the GPU skinning deformer. `Some` after `build_async_uploads_scene`
    /// installs the plugin.
    pub skinning: Option<viewport_lib::plugins::skinning::SkinningPlugin>,
    /// When true, button clicks call the synchronous upload path. Useful
    /// for showing the difference in frame pacing.
    pub use_sync: bool,
    /// Selects the per-asset payload size class. Heavy puts enough work
    /// on the upload path that the sync vs async comparison is visible
    /// in frame timings and stress totals.
    pub payload_size: PayloadSize,

    // -- Auto-orbit + frame-time telemetry -------------------------
    /// Wall-clock instant of the previous `async_uploads_update` call.
    /// Used to compute dt for the auto-orbit camera and to feed the
    /// frame-time rolling window.
    pub last_frame_at: Option<Instant>,
    /// Recent inter-frame durations. Capped to roughly one second's
    /// worth of frames at 60 fps; the controls panel reports min, max,
    /// average, and effective fps from this window.
    pub frame_times: VecDeque<Duration>,
    /// Cumulative wall-clock since the showcase opened. Drives both the
    /// auto-orbit angle and the "target rotation" readout that sync
    /// loads visibly fall behind on.
    pub auto_orbit_started: Option<Instant>,

    // -- Stress-button timing --------------------------------------
    /// Individual asset durations recorded during the most recent
    /// "Load a level" run. Drained on the next stress click.
    pub stress_individual_ms: Vec<u64>,
    /// Wall-clock time the main thread spent inside `launch_all` itself.
    /// In sync mode this is the full stress total (every upload blocks
    /// the call). In async mode it is the cost of queueing sixteen
    /// `begin_upload_*` calls plus the tiny accounting around them. The
    /// gap between this and `load_all_duration_ms` is the value the
    /// async path actually delivers: time the main thread got back for
    /// rendering, input, and animation.
    pub launch_all_main_thread_ms: Option<u64>,
    /// Peak inter-frame interval observed during the most recent stress
    /// run, captured between the click and the moment every asset
    /// reaches a terminal state. Frozen on completion so the controls
    /// panel can show "worst stall: N ms" alongside the total.
    pub stress_max_frame_ms: f32,
    /// True while a stress run is in flight. Used to gate the
    /// stress_max_frame_ms accumulator so it only reflects the load
    /// window, not the steady-state idle pacing.
    pub stress_in_progress: bool,
    /// Per-frame cap on apply-step work, in milliseconds. `None`
    /// matches the historical behaviour: every completed apply runs
    /// the same frame the job finished. The radio in the controls
    /// panel writes this on each repaint into the renderer's
    /// `set_upload_budget`. Useful for taming the end-of-batch fat
    /// frame when many heavy completions land together.
    pub upload_budget_ms: Option<u32>,

    /// Base mesh uploaded synchronously on first build so the viewport is
    /// never empty.
    pub base_mesh_id: Option<MeshId>,
    /// Mesh created on the fly to host async skin weights.
    pub skin_target_mesh_id: Option<MeshId>,
    /// Vertex count of the skin target mesh, captured at upload time so the
    /// skin button does not need to peek at private renderer state.
    pub skin_target_vertex_count: usize,

    pub env_state: AssetState,
    pub mesh_state: AssetState,
    pub texture_state: AssetState,
    pub skin_state: AssetState,
    pub polyline_state: AssetState,
    pub streamtube_state: AssetState,
    pub tube_state: AssetState,
    pub ribbon_state: AssetState,
    pub point_cloud_state: AssetState,
    pub glyph_set_state: AssetState,
    pub tensor_glyph_set_state: AssetState,
    pub volume_state: AssetState,
    pub gaussian_splat_state: AssetState,
    pub overlay_texture_state: AssetState,
    pub sprite_set_state: AssetState,
    pub sprite_instance_set_state: AssetState,

    pub loaded_mesh_id: Option<MeshId>,
    pub loaded_texture_id: Option<u64>,
    pub loaded_polyline_id: Option<PolylineId>,
    pub loaded_streamtube_id: Option<StreamtubeId>,
    pub loaded_tube_id: Option<TubeId>,
    pub loaded_ribbon_id: Option<RibbonId>,
    pub loaded_point_cloud_id: Option<PointCloudId>,
    pub loaded_glyph_set_id: Option<GlyphSetId>,
    pub loaded_tensor_glyph_set_id: Option<TensorGlyphSetId>,
    pub loaded_volume_id: Option<VolumeId>,
    pub loaded_gaussian_splat_id: Option<GaussianSplatId>,
    pub loaded_overlay_texture_id: Option<OverlayTextureId>,
    pub loaded_sprite_set_id: Option<SpriteSetId>,
    pub loaded_sprite_instance_set_id: Option<SpriteInstanceSetId>,
    pub skin_installed: bool,
    pub built: bool,

    /// Wall-clock instant when the "Load a level" button was last clicked.
    /// `None` between sessions. Becomes `Some` on click; the controls
    /// panel reads it to compute the total once every asset has landed.
    pub load_all_started: Option<Instant>,
    /// Final duration of the most recent "Load a level" run, in ms. Set
    /// once every asset transitions to `Loaded` or `Failed` after a click.
    pub load_all_duration_ms: Option<u64>,
}

impl Default for AsyncUploadsState {
    fn default() -> Self {
        Self {
            skinning: None,
            use_sync: false,
            payload_size: PayloadSize::Heavy,
            last_frame_at: None,
            frame_times: VecDeque::with_capacity(120),
            auto_orbit_started: None,
            stress_individual_ms: Vec::new(),
            launch_all_main_thread_ms: None,
            stress_max_frame_ms: 0.0,
            stress_in_progress: false,
            upload_budget_ms: None,
            base_mesh_id: None,
            skin_target_mesh_id: None,
            skin_target_vertex_count: 0,
            env_state: AssetState::Idle,
            mesh_state: AssetState::Idle,
            texture_state: AssetState::Idle,
            skin_state: AssetState::Idle,
            polyline_state: AssetState::Idle,
            streamtube_state: AssetState::Idle,
            tube_state: AssetState::Idle,
            ribbon_state: AssetState::Idle,
            point_cloud_state: AssetState::Idle,
            glyph_set_state: AssetState::Idle,
            tensor_glyph_set_state: AssetState::Idle,
            volume_state: AssetState::Idle,
            gaussian_splat_state: AssetState::Idle,
            overlay_texture_state: AssetState::Idle,
            sprite_set_state: AssetState::Idle,
            sprite_instance_set_state: AssetState::Idle,
            loaded_mesh_id: None,
            loaded_texture_id: None,
            loaded_polyline_id: None,
            loaded_streamtube_id: None,
            loaded_tube_id: None,
            loaded_ribbon_id: None,
            loaded_point_cloud_id: None,
            loaded_glyph_set_id: None,
            loaded_tensor_glyph_set_id: None,
            loaded_volume_id: None,
            loaded_gaussian_splat_id: None,
            loaded_overlay_texture_id: None,
            loaded_sprite_set_id: None,
            loaded_sprite_instance_set_id: None,
            skin_installed: false,
            built: false,
            load_all_started: None,
            load_all_duration_ms: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_async_uploads_scene(&mut self, renderer: &mut ViewportRenderer) {
        // GPU skinning is opt-in: install once before uploading any skin data.
        self.async_uploads_state.skinning = Some(
            viewport_lib::plugins::skinning::SkinningPlugin::install(
                renderer.resources_mut(),
                &self.device,
            )
            .expect("install skinning"),
        );

        let plane_mesh = viewport_lib::primitives::sphere(0.7, 24, 18);
        self.async_uploads_state.base_mesh_id = Some(
            renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &plane_mesh)
                .expect("base mesh"),
        );
        // A second sphere serves as the target for async skin weights.
        let skin_mesh = viewport_lib::primitives::sphere(0.5, 16, 12);
        self.async_uploads_state.skin_target_vertex_count = skin_mesh.positions.len();
        self.async_uploads_state.skin_target_mesh_id = Some(
            renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &skin_mesh)
                .expect("skin target mesh"),
        );

        self.async_uploads_state.env_state = AssetState::Idle;
        self.async_uploads_state.mesh_state = AssetState::Idle;
        self.async_uploads_state.texture_state = AssetState::Idle;
        self.async_uploads_state.skin_state = AssetState::Idle;
        self.async_uploads_state.polyline_state = AssetState::Idle;
        self.async_uploads_state.streamtube_state = AssetState::Idle;
        self.async_uploads_state.tube_state = AssetState::Idle;
        self.async_uploads_state.ribbon_state = AssetState::Idle;
        self.async_uploads_state.point_cloud_state = AssetState::Idle;
        self.async_uploads_state.glyph_set_state = AssetState::Idle;
        self.async_uploads_state.tensor_glyph_set_state = AssetState::Idle;
        self.async_uploads_state.volume_state = AssetState::Idle;
        self.async_uploads_state.gaussian_splat_state = AssetState::Idle;
        self.async_uploads_state.overlay_texture_state = AssetState::Idle;
        self.async_uploads_state.sprite_set_state = AssetState::Idle;
        self.async_uploads_state.sprite_instance_set_state = AssetState::Idle;
        self.async_uploads_state.loaded_mesh_id = None;
        self.async_uploads_state.loaded_texture_id = None;
        self.async_uploads_state.loaded_polyline_id = None;
        self.async_uploads_state.loaded_streamtube_id = None;
        self.async_uploads_state.loaded_tube_id = None;
        self.async_uploads_state.loaded_ribbon_id = None;
        self.async_uploads_state.loaded_point_cloud_id = None;
        self.async_uploads_state.loaded_glyph_set_id = None;
        self.async_uploads_state.loaded_tensor_glyph_set_id = None;
        self.async_uploads_state.loaded_volume_id = None;
        self.async_uploads_state.loaded_gaussian_splat_id = None;
        self.async_uploads_state.loaded_overlay_texture_id = None;
        self.async_uploads_state.loaded_sprite_set_id = None;
        self.async_uploads_state.loaded_sprite_instance_set_id = None;
        self.async_uploads_state.skin_installed = false;
        self.async_uploads_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Per-frame update: advance the per-asset state machines
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn async_uploads_update(&mut self, renderer: &mut ViewportRenderer) {
        // Frame-time bookkeeping and auto-orbit. The whole point of the
        // showcase is to compare frame pacing across sync vs async uploads,
        // so we time every async_uploads_update call (one per repaint)
        // and continuously rotate the camera by `dt * rate`. Under sync
        // load the next call comes far later than the target frame
        // interval, the dt jumps to the full upload duration, and the
        // camera lurches; under async load dt stays near 16 ms and the
        // orbit reads as smooth.
        let now = Instant::now();
        if self.async_uploads_state.auto_orbit_started.is_none() {
            self.async_uploads_state.auto_orbit_started = Some(now);
        }
        let dt = match self.async_uploads_state.last_frame_at {
            Some(prev) => now.saturating_duration_since(prev),
            None => Duration::from_millis(16),
        };
        self.async_uploads_state.last_frame_at = Some(now);
        // 30 degrees per second around the world up axis.
        let rate_rad_per_s: f32 = 30.0_f32.to_radians();
        let yaw = rate_rad_per_s * dt.as_secs_f32();
        self.camera.orbit(yaw, 0.0);
        // Rolling window of recent frame times, capped at ~2 seconds at
        // 60 fps so the stats reflect what the user can actually see.
        self.async_uploads_state.frame_times.push_back(dt);
        while self.async_uploads_state.frame_times.len() > 120 {
            self.async_uploads_state.frame_times.pop_front();
        }
        // While a stress run is in flight, track the worst inter-frame
        // interval. This is the headline number: in sync mode it ends
        // up equal to the upload total (the main thread was blocked);
        // in async mode it stays near the vsync interval.
        if self.async_uploads_state.stress_in_progress {
            let dt_ms = dt.as_secs_f32() * 1000.0;
            if dt_ms > self.async_uploads_state.stress_max_frame_ms {
                self.async_uploads_state.stress_max_frame_ms = dt_ms;
            }
        }

        // Env map: status only, no typed result to take.
        let env_just_loaded =
            advance_status_only(&mut self.async_uploads_state.env_state, renderer);

        // Mesh: when Ready, take the MeshId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.mesh_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.upload_result_mesh(job) {
                    Ok(mesh_id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_mesh_id = Some(mesh_id);
                        self.async_uploads_state.mesh_state = AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.mesh_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.mesh_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.mesh_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.mesh_state = AssetState::Idle;
                }
            }
        }

        // Texture: when Ready, take the texture id.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.texture_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.upload_result_texture(job) {
                    Ok(tex_id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_texture_id = Some(tex_id);
                        self.async_uploads_state.texture_state = AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.texture_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.texture_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.texture_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.texture_state = AssetState::Idle;
                }
            }
        }

        // Skin: status only, but also flip the installed flag when Ready.
        let skin_just_loaded =
            advance_status_only(&mut self.async_uploads_state.skin_state, renderer);
        if skin_just_loaded {
            self.async_uploads_state.skin_installed = true;
        }

        // The env map's IBL textures are stored on the renderer once the
        // apply step runs. Rebuild the camera bind groups on the exact
        // frame the job transitioned to Loaded so the shaders pick them
        // up.
        if env_just_loaded {
            renderer.rebuild_camera_bind_groups(&self.device);
        }

        // Polyline: when Ready, take the PolylineId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.polyline_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.resources_mut().upload_result_polyline(job) {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_polyline_id = Some(id);
                        self.async_uploads_state.polyline_state =
                            AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.polyline_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.polyline_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.polyline_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.polyline_state = AssetState::Idle;
                }
            }
        }

        // Streamtube: when Ready, take the StreamtubeId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.streamtube_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.resources_mut().upload_result_streamtube(job)
                {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_streamtube_id = Some(id);
                        self.async_uploads_state.streamtube_state =
                            AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.streamtube_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.streamtube_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.streamtube_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.streamtube_state = AssetState::Idle;
                }
            }
        }

        // Tube: when Ready, take the TubeId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.tube_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.resources_mut().upload_result_tube(job) {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_tube_id = Some(id);
                        self.async_uploads_state.tube_state = AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.tube_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.tube_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.tube_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.tube_state = AssetState::Idle;
                }
            }
        }

        // Ribbon: when Ready, take the RibbonId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.ribbon_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.resources_mut().upload_result_ribbon(job) {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_ribbon_id = Some(id);
                        self.async_uploads_state.ribbon_state = AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.ribbon_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.ribbon_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.ribbon_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.ribbon_state = AssetState::Idle;
                }
            }
        }

        // Point cloud: when Ready, take the PointCloudId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.point_cloud_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => {
                    match renderer.resources_mut().upload_result_point_cloud(job) {
                        Ok(id) => {
                            let duration_ms = take_job_duration_ms(renderer, job, &started);
                            self.async_uploads_state.loaded_point_cloud_id = Some(id);
                            self.async_uploads_state.point_cloud_state =
                                AssetState::Loaded { duration_ms };
                        }
                        Err(e) => {
                            let duration_ms = take_job_duration_ms(renderer, job, &started);
                            self.async_uploads_state.point_cloud_state = AssetState::Failed {
                                reason: format!("{e}"),
                                duration_ms,
                            };
                        }
                    }
                }
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.point_cloud_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.point_cloud_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.point_cloud_state = AssetState::Idle;
                }
            }
        }

        // Glyph set: when Ready, take the GlyphSetId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.glyph_set_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => {
                    match renderer.resources_mut().upload_result_glyph_set(job) {
                        Ok(id) => {
                            let duration_ms = take_job_duration_ms(renderer, job, &started);
                            self.async_uploads_state.loaded_glyph_set_id = Some(id);
                            self.async_uploads_state.glyph_set_state =
                                AssetState::Loaded { duration_ms };
                        }
                        Err(e) => {
                            let duration_ms = take_job_duration_ms(renderer, job, &started);
                            self.async_uploads_state.glyph_set_state = AssetState::Failed {
                                reason: format!("{e}"),
                                duration_ms,
                            };
                        }
                    }
                }
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.glyph_set_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.glyph_set_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.glyph_set_state = AssetState::Idle;
                }
            }
        }

        // Tensor glyph set: when Ready, take the TensorGlyphSetId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.tensor_glyph_set_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => {
                    match renderer.resources_mut().upload_result_tensor_glyph_set(job) {
                        Ok(id) => {
                            let duration_ms = take_job_duration_ms(renderer, job, &started);
                            self.async_uploads_state.loaded_tensor_glyph_set_id = Some(id);
                            self.async_uploads_state.tensor_glyph_set_state =
                                AssetState::Loaded { duration_ms };
                        }
                        Err(e) => {
                            let duration_ms = take_job_duration_ms(renderer, job, &started);
                            self.async_uploads_state.tensor_glyph_set_state = AssetState::Failed {
                                reason: format!("{e}"),
                                duration_ms,
                            };
                        }
                    }
                }
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.tensor_glyph_set_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.tensor_glyph_set_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.tensor_glyph_set_state = AssetState::Idle;
                }
            }
        }

        // Volume: when Ready, take the VolumeId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.volume_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.upload_result_volume(job) {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_volume_id = Some(id);
                        self.async_uploads_state.volume_state = AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.volume_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.volume_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.volume_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.volume_state = AssetState::Idle;
                }
            }
        }

        // Gaussian splats: when Ready, take the GaussianSplatId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.gaussian_splat_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.upload_result_gaussian_splats(job) {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_gaussian_splat_id = Some(id);
                        self.async_uploads_state.gaussian_splat_state =
                            AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.gaussian_splat_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.gaussian_splat_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.gaussian_splat_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.gaussian_splat_state = AssetState::Idle;
                }
            }
        }

        // Overlay texture: when Ready, take the OverlayTextureId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.overlay_texture_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.upload_result_overlay_texture(job) {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_overlay_texture_id = Some(id);
                        self.async_uploads_state.overlay_texture_state =
                            AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.overlay_texture_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.overlay_texture_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.overlay_texture_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.overlay_texture_state = AssetState::Idle;
                }
            }
        }

        // Sprite set: when Ready, take the SpriteSetId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.sprite_set_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.resources_mut().upload_result_sprite_set(job)
                {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_sprite_set_id = Some(id);
                        self.async_uploads_state.sprite_set_state =
                            AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.sprite_set_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.sprite_set_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.sprite_set_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.sprite_set_state = AssetState::Idle;
                }
            }
        }

        // Sprite instance set: when Ready, take the SpriteInstanceSetId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.sprite_instance_set_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer
                    .resources_mut()
                    .upload_result_sprite_instance_set(job)
                {
                    Ok(id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_sprite_instance_set_id = Some(id);
                        self.async_uploads_state.sprite_instance_set_state =
                            AssetState::Loaded { duration_ms };
                    }
                    Err(e) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.sprite_instance_set_state = AssetState::Failed {
                            reason: format!("{e}"),
                            duration_ms,
                        };
                    }
                },
                UploadStatus::Failed(e) => {
                    let duration_ms = take_job_duration_ms(renderer, job, &started);
                    self.async_uploads_state.sprite_instance_set_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms,
                    };
                }
                UploadStatus::Pending { progress } => {
                    self.async_uploads_state.sprite_instance_set_state = AssetState::InFlight {
                        job,
                        progress,
                        started,
                    };
                }
                UploadStatus::Unknown => {
                    renderer.drop_job_duration(job);
                    self.async_uploads_state.sprite_instance_set_state = AssetState::Idle;
                }
            }
        }

        // Once a "Load a level" run has every asset in a terminal state,
        // stamp the total wall-clock duration so the controls panel can
        // show it next frame.
        if let Some(started) = self.async_uploads_state.load_all_started {
            if self.async_uploads_state.load_all_duration_ms.is_none()
                && all_assets_terminal(&self.async_uploads_state)
            {
                self.async_uploads_state.load_all_duration_ms =
                    Some(started.elapsed().as_millis() as u64);
                self.async_uploads_state.stress_individual_ms =
                    collect_terminal_durations_ms(&self.async_uploads_state);
                self.async_uploads_state.stress_in_progress = false;
            }
        }
    }
}

/// Collect each asset's recorded duration_ms once it has reached a
/// terminal state. Used to compute the "serial baseline" (sum) and
/// "longest single upload" (max) shown after a stress run.
fn collect_terminal_durations_ms(state: &AsyncUploadsState) -> Vec<u64> {
    let read = |s: &AssetState| -> Option<u64> {
        match s {
            AssetState::Loaded { duration_ms } => Some(*duration_ms),
            AssetState::Failed { duration_ms, .. } => Some(*duration_ms),
            _ => None,
        }
    };
    [
        read(&state.env_state),
        read(&state.mesh_state),
        read(&state.texture_state),
        read(&state.skin_state),
        read(&state.polyline_state),
        read(&state.streamtube_state),
        read(&state.tube_state),
        read(&state.ribbon_state),
        read(&state.point_cloud_state),
        read(&state.glyph_set_state),
        read(&state.tensor_glyph_set_state),
        read(&state.volume_state),
        read(&state.gaussian_splat_state),
        read(&state.overlay_texture_state),
        read(&state.sprite_set_state),
        read(&state.sprite_instance_set_state),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn all_assets_terminal(state: &AsyncUploadsState) -> bool {
    let is_terminal =
        |s: &AssetState| matches!(s, AssetState::Loaded { .. } | AssetState::Failed { .. });
    is_terminal(&state.env_state)
        && is_terminal(&state.mesh_state)
        && is_terminal(&state.texture_state)
        && is_terminal(&state.skin_state)
        && is_terminal(&state.polyline_state)
        && is_terminal(&state.streamtube_state)
        && is_terminal(&state.tube_state)
        && is_terminal(&state.ribbon_state)
        && is_terminal(&state.point_cloud_state)
        && is_terminal(&state.glyph_set_state)
        && is_terminal(&state.tensor_glyph_set_state)
        && is_terminal(&state.volume_state)
        && is_terminal(&state.gaussian_splat_state)
        && is_terminal(&state.overlay_texture_state)
        && is_terminal(&state.sprite_set_state)
        && is_terminal(&state.sprite_instance_set_state)
}

/// Advance an asset that has no typed result to take. Returns `true` if
/// the state just transitioned from `InFlight` to `Loaded` so callers can
/// run a one-shot side effect (rebuilding bind groups, flipping a flag,
/// etc.) on the matching frame.
fn advance_status_only(state: &mut AssetState, renderer: &mut ViewportRenderer) -> bool {
    let AssetState::InFlight { job, started, .. } = state.clone() else {
        return false;
    };
    match renderer.upload_status(job) {
        UploadStatus::Ready => {
            let duration_ms = take_job_duration_ms(renderer, job, &started);
            *state = AssetState::Loaded { duration_ms };
            true
        }
        UploadStatus::Failed(e) => {
            let duration_ms = take_job_duration_ms(renderer, job, &started);
            *state = AssetState::Failed {
                reason: format!("{e}"),
                duration_ms,
            };
            false
        }
        UploadStatus::Pending { progress } => {
            *state = AssetState::InFlight {
                job,
                progress,
                started,
            };
            false
        }
        UploadStatus::Unknown => {
            renderer.drop_job_duration(job);
            *state = AssetState::Idle;
            false
        }
    }
}

/// Read the runner-recorded duration for a finished job and drop it from
/// the runner's table. Falls back to the wall-clock from `submitted_at` if
/// the runner has no record (e.g. the synchronous fallback path).
fn take_job_duration_ms(
    renderer: &mut ViewportRenderer,
    job: JobId,
    submitted_at: &Instant,
) -> u64 {
    let dur = renderer
        .job_duration(job)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_else(|| submitted_at.elapsed().as_millis() as u64);
    renderer.drop_job_duration(job);
    dur
}

// ---------------------------------------------------------------------------
// Scene items
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn async_uploads_scene_items(&self) -> Vec<SceneRenderItem> {
        let mut items = Vec::new();
        let state = &self.async_uploads_state;

        if let Some(id) = state.base_mesh_id {
            let mut item = SceneRenderItem::default();
            item.mesh_id = id;
            item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
            item.material = Material::flat([0.7, 0.7, 0.7]);
            items.push(item);
        }

        if let Some(id) = state.loaded_mesh_id {
            let mut item = SceneRenderItem::default();
            item.mesh_id = id;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.0, 0.0)).to_cols_array_2d();
            item.material = Material::flat([0.4, 0.8, 0.4]);
            items.push(item);
        }

        if let (Some(mesh_id), Some(tex_id)) = (state.base_mesh_id, state.loaded_texture_id) {
            let mut material = Material::flat([1.0, 1.0, 1.0]);
            material.texture_id = Some(tex_id);
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(4.0, 0.0, 0.0)).to_cols_array_2d();
            item.material = material;
            items.push(item);
        }

        if let Some(id) = state.skin_target_mesh_id {
            let mut item = SceneRenderItem::default();
            item.mesh_id = id;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(-2.0, 0.0, 0.0)).to_cols_array_2d();
            let colour = if state.skin_installed {
                [0.9, 0.4, 0.9]
            } else {
                [0.4, 0.4, 0.4]
            };
            item.material = Material::flat(colour);
            items.push(item);
        }

        items
    }

    pub(crate) fn async_uploads_lighting(&self) -> LightingSettings {
        let mut lighting = LightingSettings::default();
        let mut sun = LightSource::default();
        sun.kind = LightKind::Directional {
            direction: [-0.4, -0.6, -0.7],
        };
        lighting.lights = vec![sun];
        lighting.hemisphere_intensity = 0.35;
        lighting.sky_colour = [0.85, 0.9, 1.0];
        lighting.ground_colour = [0.3, 0.3, 0.32];
        // When the env-map has landed, drop hemisphere a bit so the IBL
        // contribution is visible.
        if matches!(
            self.async_uploads_state.env_state,
            AssetState::Loaded { .. }
        ) {
            lighting.hemisphere_intensity = 0.1;
        }
        lighting
    }
}

// ---------------------------------------------------------------------------
// Demo asset data
// ---------------------------------------------------------------------------

fn solid_env_pixels(width: u32, height: u32) -> Vec<f32> {
    // Smooth vertical gradient from warm to cool, plus a hot spot near the
    // top to show that IBL picked up the new sky.
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let v = y as f32 / height.max(1) as f32;
        let r = 1.6 * (1.0 - v) + 0.4 * v;
        let g = 1.2 * (1.0 - v) + 0.6 * v;
        let b = 0.4 * (1.0 - v) + 1.8 * v;
        for _ in 0..width {
            out.extend_from_slice(&[r, g, b, 1.0]);
        }
    }
    out
}

fn checker_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    let cell = (width / 8).max(1);
    for y in 0..height {
        for x in 0..width {
            let on = ((x / cell) + (y / cell)) % 2 == 0;
            if on {
                out.extend_from_slice(&[240, 200, 80, 255]);
            } else {
                out.extend_from_slice(&[40, 50, 70, 255]);
            }
        }
    }
    out
}

fn demo_mesh(size: PayloadSize) -> MeshData {
    // Subdivision N gives 20 * 4^N tris. Light stays small for smoke
    // testing; Heavy goes to subdiv 8 (~1.3M tris, ~655k verts) so the
    // worker-side tangent compute and vertex repack take real time and
    // the sync stall is plainly visible.
    let subdivisions = match size {
        PayloadSize::Light => 3,
        PayloadSize::Heavy => 8,
    };
    viewport_lib::primitives::icosphere(0.6, subdivisions)
}

fn unit_skin_weights(vertex_count: usize) -> SkinWeights {
    SkinWeights {
        joint_indices: vec![[0u8; 4]; vertex_count],
        joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; vertex_count],
    }
}

// A short helix in the XZ plane, used by all four curve types so each
// pre-uploaded asset has visibly the same source geometry. Heavy mode
// runs the helix at higher sample density so tube / streamtube /
// ribbon generation has nontrivial CPU prep on the worker.
fn demo_curve_positions(size: PayloadSize) -> (Vec<[f32; 3]>, Vec<u32>) {
    let n: usize = match size {
        PayloadSize::Light => 48,
        PayloadSize::Heavy => 50_000,
    };
    let mut positions = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        let theta = t * std::f32::consts::TAU * 1.5;
        let x = 0.8 * theta.cos();
        let y = 0.0;
        let z = 0.8 * theta.sin() + t * 1.2 - 0.6;
        positions.push([x, y, z]);
    }
    let strip = vec![n as u32];
    (positions, strip)
}

fn demo_polyline(size: PayloadSize) -> PolylineItem {
    let (positions, strip_lengths) = demo_curve_positions(size);
    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.line_width = 3.0;
    item.default_colour = [0.95, 0.55, 0.35, 1.0];
    item
}

fn demo_streamtube(size: PayloadSize) -> StreamtubeItem {
    let (positions, strip_lengths) = demo_curve_positions(size);
    let mut item = StreamtubeItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.radius = 0.07;
    item.colour = [0.45, 0.85, 0.95, 1.0];
    item
}

fn demo_tube(size: PayloadSize) -> TubeItem {
    let (positions, strip_lengths) = demo_curve_positions(size);
    let mut item = TubeItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.radius = 0.08;
    item.sides = 16;
    item.colour = [0.95, 0.85, 0.45, 1.0];
    item
}

fn demo_ribbon(size: PayloadSize) -> RibbonItem {
    let (positions, strip_lengths) = demo_curve_positions(size);
    let mut item = RibbonItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.width = 0.18;
    item.colour = [0.65, 0.55, 0.95, 1.0];
    item
}

fn demo_point_cloud(size: PayloadSize) -> PointCloudItem {
    // Heavy uses a 1000x1000 grid (1M points) so the position + colour
    // buffer writes take real time. The layout stays a flat plane so
    // the cloud is visible regardless of camera direction.
    let mut item = PointCloudItem::default();
    let n: i32 = match size {
        PayloadSize::Light => 16,
        PayloadSize::Heavy => 1000,
    };
    let mut positions = Vec::with_capacity((n * n) as usize);
    for i in 0..n {
        for j in 0..n {
            let x = (i as f32 - n as f32 * 0.5) * 0.15;
            let y = (j as f32 - n as f32 * 0.5) * 0.15;
            positions.push([x, y, 0.0]);
        }
    }
    item.positions = positions;
    item.point_size = 6.0;
    item.default_colour = [0.35, 0.85, 0.55, 1.0];
    item
}

fn demo_glyph_set() -> GlyphItem {
    // A small ring of arrows pointing outward, in object-local coordinates
    // around the origin. The ref item's `model` matrix places it in the
    // showcase grid.
    let mut item = GlyphItem::default();
    let count = 12;
    let r = 0.6;
    let mut positions = Vec::with_capacity(count);
    let mut vectors = Vec::with_capacity(count);
    for i in 0..count {
        let theta = (i as f32) / (count as f32) * std::f32::consts::TAU;
        let (s, c) = theta.sin_cos();
        positions.push([c * r, 0.0, s * r]);
        vectors.push([c, 0.0, s]);
    }
    item.positions = positions;
    item.vectors = vectors;
    item.scale = 0.3;
    item.scale_by_magnitude = false;
    item.default_colour = [0.9, 0.7, 0.3, 1.0];
    item.use_default_colour = true;
    item
}

fn demo_gaussian_splats(size: PayloadSize) -> GaussianSplatData {
    let n = match size {
        PayloadSize::Light => 256,
        PayloadSize::Heavy => 1_000_000,
    };
    let splat_scale = match size {
        PayloadSize::Light => 0.05,
        PayloadSize::Heavy => 0.005,
    };
    let mut positions = Vec::with_capacity(n);
    let scales = vec![[splat_scale; 3]; n];
    let rotations = vec![[0.0_f32, 0.0, 0.0, 1.0]; n];
    let opacities = vec![0.7_f32; n];
    for i in 0..n {
        let theta = (i as f32) / (n as f32) * std::f32::consts::TAU * 4.0;
        let r = 0.4 + (i as f32) / (n as f32) * 0.8;
        positions.push([
            r * theta.cos(),
            (i as f32) / (n as f32) * 1.5 - 0.75,
            r * theta.sin(),
        ]);
    }
    let mut data = GaussianSplatData::default();
    data.positions = positions;
    data.scales = scales;
    data.rotations = rotations;
    data.opacities = opacities;
    data
}

fn demo_overlay_texture() -> (Vec<u8>, u32, u32) {
    let w = 128u32;
    let h = 128u32;
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - w as f32 * 0.5;
            let dy = y as f32 - h as f32 * 0.5;
            let r = (dx * dx + dy * dy).sqrt();
            let v = ((1.0 - r / (w as f32 * 0.5)).clamp(0.0, 1.0) * 255.0) as u8;
            data.extend_from_slice(&[v, v / 2, 255 - v, 255]);
        }
    }
    (data, w, h)
}

fn demo_sprite_set() -> SpriteItem {
    let mut item = SpriteItem::default();
    let n = 24;
    let r = 1.0_f32;
    let mut positions = Vec::with_capacity(n);
    for i in 0..n {
        let theta = (i as f32) / (n as f32) * std::f32::consts::TAU;
        positions.push([r * theta.cos(), 0.0, r * theta.sin()]);
    }
    item.positions = positions;
    item.default_size = 16.0;
    item.default_colour = [0.95, 0.7, 0.4, 1.0];
    item
}

fn demo_sprite_instance_set() -> SpriteItem {
    let mut item = SpriteItem::default();
    let n = 16;
    let mut positions = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f32 - n as f32 * 0.5) * 0.3;
        positions.push([x, 0.5, 0.0]);
    }
    item.positions = positions;
    item.default_size = 12.0;
    item.default_colour = [0.5, 0.85, 0.95, 1.0];
    item
}

fn demo_volume(size: PayloadSize) -> (Vec<f32>, [u32; 3]) {
    // Radial-falloff field; the iso-volume reads as a soft sphere via
    // the colour LUT in `VolumeItem`. Heavy is 256^3 (~64 MB of f32
    // voxels) so the worker has real fill + queue.write work to do.
    let dim = match size {
        PayloadSize::Light => 32u32,
        PayloadSize::Heavy => 256u32,
    };
    let n = dim as usize;
    let centre = (n as f32 - 1.0) * 0.5;
    let mut data = Vec::with_capacity(n * n * n);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let dx = x as f32 - centre;
                let dy = y as f32 - centre;
                let dz = z as f32 - centre;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let v = (1.0 - (r / centre)).clamp(0.0, 1.0);
                data.push(v);
            }
        }
    }
    (data, [dim, dim, dim])
}

fn demo_tensor_glyph_set() -> TensorGlyphItem {
    // A 3x3 grid of ellipsoids around the local origin. The ref item's
    // `model` matrix places it in the showcase grid.
    let mut item = TensorGlyphItem::default();
    let n: i32 = 3;
    for i in 0..n {
        for j in 0..n {
            let x = (i as f32 - 1.0) * 0.5;
            let z = (j as f32 - 1.0) * 0.5;
            item.positions.push([x, 0.0, z]);
            item.eigenvalues.push([1.0, 0.6, 0.3]);
            item.eigenvectors
                .push([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        }
    }
    item.scale = 0.25;
    item
}

// ---------------------------------------------------------------------------
// Asset launch helpers (Sync vs Async)
// ---------------------------------------------------------------------------

impl App {
    fn launch_env_map(&mut self, renderer: &mut ViewportRenderer) {
        let pixels = solid_env_pixels(32, 16);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            match renderer.upload_environment_map(&self.device, &self.queue, &pixels, 32, 16) {
                Ok(()) => {
                    self.async_uploads_state.env_state = AssetState::Loaded {
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.env_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
            renderer.rebuild_camera_bind_groups(&self.device);
        } else {
            match renderer.begin_upload_environment_map(&self.device, &self.queue, pixels, 32, 16) {
                Ok(job) => {
                    self.async_uploads_state.env_state = AssetState::InFlight {
                        job,
                        progress: 0.0,
                        started,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.env_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        }
    }

    fn launch_mesh(&mut self, renderer: &mut ViewportRenderer) {
        let data = demo_mesh(self.async_uploads_state.payload_size);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            match renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &data)
            {
                Ok(mesh_id) => {
                    self.async_uploads_state.loaded_mesh_id = Some(mesh_id);
                    self.async_uploads_state.mesh_state = AssetState::Loaded {
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.mesh_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        } else {
            match renderer.begin_upload_mesh_data(&self.device, data) {
                Ok(job) => {
                    self.async_uploads_state.mesh_state = AssetState::InFlight {
                        job,
                        progress: 0.0,
                        started,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.mesh_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        }
    }

    fn launch_texture(&mut self, renderer: &mut ViewportRenderer) {
        // 256x256 (Light, 256 KB) vs 4096x4096 (Heavy, 64 MB). The Heavy
        // texture forces a 64 MB pixel-buffer generation up front plus a
        // matching queue.write_texture; the sync path stalls visibly
        // while async pushes the queue work onto its worker thread.
        let dim = match self.async_uploads_state.payload_size {
            PayloadSize::Light => 256u32,
            PayloadSize::Heavy => 4096u32,
        };
        let rgba = checker_rgba(dim, dim);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            match renderer.resources_mut().upload_texture(
                &self.device,
                &self.queue,
                dim,
                dim,
                &rgba,
            ) {
                Ok(tex_id) => {
                    self.async_uploads_state.loaded_texture_id = Some(tex_id);
                    self.async_uploads_state.texture_state = AssetState::Loaded {
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.texture_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        } else {
            match renderer.begin_upload_texture(&self.device, &self.queue, dim, dim, rgba) {
                Ok(job) => {
                    self.async_uploads_state.texture_state = AssetState::InFlight {
                        job,
                        progress: 0.0,
                        started,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.texture_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        }
    }

    fn launch_skin(&mut self, renderer: &mut ViewportRenderer) {
        let Some(mesh_id) = self.async_uploads_state.skin_target_mesh_id else {
            return;
        };
        let vertex_count = self.async_uploads_state.skin_target_vertex_count;
        if vertex_count == 0 {
            return;
        }
        let weights = unit_skin_weights(vertex_count);
        let started = Instant::now();
        let Some(skinning) = self.async_uploads_state.skinning.clone() else {
            return;
        };
        if self.async_uploads_state.use_sync {
            skinning.attach_weights(renderer.resources_mut(), &self.device, mesh_id, &weights);
            self.async_uploads_state.skin_installed = true;
            self.async_uploads_state.skin_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job = skinning.begin_upload_weights(
                renderer.resources_mut(),
                &self.device,
                mesh_id,
                weights,
            );
            self.async_uploads_state.skin_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_polyline(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_polyline(self.async_uploads_state.payload_size);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer
                .resources_mut()
                .upload_polyline(&self.device, &self.queue, &item);
            self.async_uploads_state.loaded_polyline_id = Some(id);
            self.async_uploads_state.polyline_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job =
                renderer
                    .resources_mut()
                    .begin_upload_polyline(&self.device, &self.queue, item);
            self.async_uploads_state.polyline_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_streamtube(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_streamtube(self.async_uploads_state.payload_size);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer
                .resources_mut()
                .upload_streamtube(&self.device, &self.queue, &item);
            self.async_uploads_state.loaded_streamtube_id = Some(id);
            self.async_uploads_state.streamtube_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job =
                renderer
                    .resources_mut()
                    .begin_upload_streamtube(&self.device, &self.queue, item);
            self.async_uploads_state.streamtube_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_tube(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_tube(self.async_uploads_state.payload_size);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer
                .resources_mut()
                .upload_tube(&self.device, &self.queue, &item);
            self.async_uploads_state.loaded_tube_id = Some(id);
            self.async_uploads_state.tube_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job = renderer
                .resources_mut()
                .begin_upload_tube(&self.device, &self.queue, item);
            self.async_uploads_state.tube_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_ribbon(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_ribbon(self.async_uploads_state.payload_size);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer
                .resources_mut()
                .upload_ribbon(&self.device, &self.queue, &item);
            self.async_uploads_state.loaded_ribbon_id = Some(id);
            self.async_uploads_state.ribbon_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job = renderer
                .resources_mut()
                .begin_upload_ribbon(&self.device, &self.queue, item);
            self.async_uploads_state.ribbon_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_point_cloud(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_point_cloud(self.async_uploads_state.payload_size);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer
                .resources_mut()
                .upload_point_cloud(&self.device, &self.queue, &item);
            self.async_uploads_state.loaded_point_cloud_id = Some(id);
            self.async_uploads_state.point_cloud_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job =
                renderer
                    .resources_mut()
                    .begin_upload_point_cloud(&self.device, &self.queue, item);
            self.async_uploads_state.point_cloud_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_glyph_set(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_glyph_set();
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer
                .resources_mut()
                .upload_glyph_set(&self.device, &self.queue, &item);
            self.async_uploads_state.loaded_glyph_set_id = Some(id);
            self.async_uploads_state.glyph_set_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job =
                renderer
                    .resources_mut()
                    .begin_upload_glyph_set(&self.device, &self.queue, item);
            self.async_uploads_state.glyph_set_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_tensor_glyph_set(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_tensor_glyph_set();
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id =
                renderer
                    .resources_mut()
                    .upload_tensor_glyph_set(&self.device, &self.queue, &item);
            self.async_uploads_state.loaded_tensor_glyph_set_id = Some(id);
            self.async_uploads_state.tensor_glyph_set_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job = renderer.resources_mut().begin_upload_tensor_glyph_set(
                &self.device,
                &self.queue,
                item,
            );
            self.async_uploads_state.tensor_glyph_set_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_volume(&mut self, renderer: &mut ViewportRenderer) {
        let (data, dims) = demo_volume(self.async_uploads_state.payload_size);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer
                .resources_mut()
                .upload_volume(&self.device, &self.queue, &data, dims);
            self.async_uploads_state.loaded_volume_id = Some(id);
            self.async_uploads_state.volume_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            match renderer.begin_upload_volume(&self.device, &self.queue, data, dims) {
                Ok(job) => {
                    self.async_uploads_state.volume_state = AssetState::InFlight {
                        job,
                        progress: 0.0,
                        started,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.volume_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        }
    }

    fn launch_gaussian_splats(&mut self, renderer: &mut ViewportRenderer) {
        let data = demo_gaussian_splats(self.async_uploads_state.payload_size);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            match renderer
                .resources_mut()
                .upload_gaussian_splats(&self.device, &self.queue, &data)
            {
                Ok(id) => {
                    self.async_uploads_state.loaded_gaussian_splat_id = Some(id);
                    self.async_uploads_state.gaussian_splat_state = AssetState::Loaded {
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.gaussian_splat_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        } else {
            match renderer.begin_upload_gaussian_splats(&self.device, &self.queue, data) {
                Ok(job) => {
                    self.async_uploads_state.gaussian_splat_state = AssetState::InFlight {
                        job,
                        progress: 0.0,
                        started,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.gaussian_splat_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        }
    }

    fn launch_overlay_texture(&mut self, renderer: &mut ViewportRenderer) {
        let (data, w, h) = demo_overlay_texture();
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer.resources_mut().upload_overlay_texture(
                &self.device,
                &self.queue,
                w,
                h,
                &data,
            );
            self.async_uploads_state.loaded_overlay_texture_id = Some(id);
            self.async_uploads_state.overlay_texture_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            match renderer.begin_upload_overlay_texture(&self.device, &self.queue, w, h, data) {
                Ok(job) => {
                    self.async_uploads_state.overlay_texture_state = AssetState::InFlight {
                        job,
                        progress: 0.0,
                        started,
                    };
                }
                Err(e) => {
                    self.async_uploads_state.overlay_texture_state = AssetState::Failed {
                        reason: format!("{e}"),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            }
        }
    }

    fn launch_sprite_set(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_sprite_set();
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer
                .resources_mut()
                .upload_sprite_set(&self.device, &self.queue, &item);
            self.async_uploads_state.loaded_sprite_set_id = Some(id);
            self.async_uploads_state.sprite_set_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job =
                renderer
                    .resources_mut()
                    .begin_upload_sprite_set(&self.device, &self.queue, item);
            self.async_uploads_state.sprite_set_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_sprite_instance_set(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_sprite_instance_set();
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            let id = renderer.resources_mut().upload_sprite_instance_set(
                &self.device,
                &self.queue,
                &item,
            );
            self.async_uploads_state.loaded_sprite_instance_set_id = Some(id);
            self.async_uploads_state.sprite_instance_set_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job = renderer.resources_mut().begin_upload_sprite_instance_set(
                &self.device,
                &self.queue,
                item,
            );
            self.async_uploads_state.sprite_instance_set_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_all(&mut self, renderer: &mut ViewportRenderer) {
        let main_t = Instant::now();
        self.async_uploads_state.load_all_started = Some(main_t);
        self.async_uploads_state.load_all_duration_ms = None;
        self.async_uploads_state.stress_individual_ms.clear();
        self.async_uploads_state.launch_all_main_thread_ms = None;
        self.async_uploads_state.stress_max_frame_ms = 0.0;
        self.async_uploads_state.stress_in_progress = true;
        self.launch_env_map(renderer);
        self.launch_mesh(renderer);
        self.launch_texture(renderer);
        self.launch_skin(renderer);
        self.launch_polyline(renderer);
        self.launch_streamtube(renderer);
        self.launch_tube(renderer);
        self.launch_ribbon(renderer);
        self.launch_point_cloud(renderer);
        self.launch_glyph_set(renderer);
        self.launch_tensor_glyph_set(renderer);
        self.launch_volume(renderer);
        self.launch_gaussian_splats(renderer);
        self.launch_overlay_texture(renderer);
        self.launch_sprite_set(renderer);
        self.launch_sprite_instance_set(renderer);
        // Stamp the main-thread cost of the stress kickoff. In sync
        // mode this includes every upload (the launches blocked); in
        // async mode it covers only the begin_upload calls plus a tiny
        // amount of accounting.
        self.async_uploads_state.launch_all_main_thread_ms =
            Some(main_t.elapsed().as_millis() as u64);
    }
}

// Pushes per-frame reference items for any pre-uploaded curves into the
// scene frame. Each curve is positioned in front of the camera, fanned out
// horizontally so all four can be loaded at once.
pub(crate) fn submit_async_uploads_items(app: &mut crate::App, fd: &mut viewport_lib::FrameData) {
    if !app.async_uploads_state.built {
        return;
    }
    let translate = |x: f32, z: f32| -> [[f32; 4]; 4] {
        glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, z)).to_cols_array_2d()
    };

    if let Some(id) = app.async_uploads_state.loaded_polyline_id {
        let mut ref_item = PolylineRefItem::new(id);
        ref_item.model = translate(0.0, 2.4);
        fd.scene.polyline_refs.push(ref_item);
    }
    if let Some(id) = app.async_uploads_state.loaded_streamtube_id {
        let mut ref_item = StreamtubeRefItem::new(id);
        ref_item.model = translate(-2.4, 2.4);
        fd.scene.streamtube_refs.push(ref_item);
    }
    if let Some(id) = app.async_uploads_state.loaded_tube_id {
        let mut ref_item = TubeRefItem::new(id);
        ref_item.model = translate(2.4, 2.4);
        fd.scene.tube_refs.push(ref_item);
    }
    if let Some(id) = app.async_uploads_state.loaded_ribbon_id {
        let mut ref_item = RibbonRefItem::new(id);
        ref_item.model = translate(0.0, 4.8);
        fd.scene.ribbon_refs.push(ref_item);
    }
    if let Some(id) = app.async_uploads_state.loaded_point_cloud_id {
        let mut ref_item = PointCloudRefItem::new(id);
        ref_item.model = translate(-4.8, 2.4);
        fd.scene.point_cloud_refs.push(ref_item);
    }
    if let Some(id) = app.async_uploads_state.loaded_glyph_set_id {
        let mut ref_item = GlyphSetRefItem::new(id);
        ref_item.model = translate(4.8, 2.4);
        fd.scene.glyph_set_refs.push(ref_item);
    }
    if let Some(id) = app.async_uploads_state.loaded_tensor_glyph_set_id {
        let mut ref_item = TensorGlyphSetRefItem::new(id);
        ref_item.model = translate(-2.4, 4.8);
        fd.scene.tensor_glyph_set_refs.push(ref_item);
    }
    if let Some(id) = app.async_uploads_state.loaded_volume_id {
        let mut item = VolumeItem::default();
        item.volume_id = id;
        item.colour_lut = Some(ColourmapId(0));
        item.scalar_range = (0.0, 1.0);
        item.opacity_scale = 0.6;
        item.step_scale = 1.0;
        // Sit the volume to the far right of the showcase grid.
        item.bbox_min = [4.0, -1.5, -1.5];
        item.bbox_max = [7.0, 1.5, 1.5];
        fd.scene.volumes.push(item);
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_async_uploads(app: &mut App, ui: &mut egui::Ui, frame: &eframe::Frame) {
    ui.heading("Upload mode");
    let mut new_mode = app.async_uploads_state.use_sync;
    ui.horizontal(|ui| {
        if ui
            .radio(!app.async_uploads_state.use_sync, "Async (begin_upload_*)")
            .clicked()
        {
            new_mode = false;
        }
        if ui
            .radio(app.async_uploads_state.use_sync, "Sync (blocking)")
            .clicked()
        {
            new_mode = true;
        }
    });
    // Switching modes clears the per-asset state so timings from the
    // previous run do not bleed into the new one. The uploaded textures
    // and meshes on the renderer side stay alive; the showcase just
    // forgets about them and ignores them going forward.
    if new_mode != app.async_uploads_state.use_sync {
        app.async_uploads_state.use_sync = new_mode;
        app.async_uploads_state.env_state = AssetState::Idle;
        app.async_uploads_state.mesh_state = AssetState::Idle;
        app.async_uploads_state.texture_state = AssetState::Idle;
        app.async_uploads_state.skin_state = AssetState::Idle;
        app.async_uploads_state.polyline_state = AssetState::Idle;
        app.async_uploads_state.streamtube_state = AssetState::Idle;
        app.async_uploads_state.tube_state = AssetState::Idle;
        app.async_uploads_state.ribbon_state = AssetState::Idle;
        app.async_uploads_state.point_cloud_state = AssetState::Idle;
        app.async_uploads_state.glyph_set_state = AssetState::Idle;
        app.async_uploads_state.tensor_glyph_set_state = AssetState::Idle;
        app.async_uploads_state.volume_state = AssetState::Idle;
        app.async_uploads_state.gaussian_splat_state = AssetState::Idle;
        app.async_uploads_state.overlay_texture_state = AssetState::Idle;
        app.async_uploads_state.sprite_set_state = AssetState::Idle;
        app.async_uploads_state.sprite_instance_set_state = AssetState::Idle;
        app.async_uploads_state.loaded_mesh_id = None;
        app.async_uploads_state.loaded_texture_id = None;
        app.async_uploads_state.loaded_polyline_id = None;
        app.async_uploads_state.loaded_streamtube_id = None;
        app.async_uploads_state.loaded_tube_id = None;
        app.async_uploads_state.loaded_ribbon_id = None;
        app.async_uploads_state.loaded_point_cloud_id = None;
        app.async_uploads_state.loaded_glyph_set_id = None;
        app.async_uploads_state.loaded_tensor_glyph_set_id = None;
        app.async_uploads_state.loaded_volume_id = None;
        app.async_uploads_state.loaded_gaussian_splat_id = None;
        app.async_uploads_state.loaded_overlay_texture_id = None;
        app.async_uploads_state.loaded_sprite_set_id = None;
        app.async_uploads_state.loaded_sprite_instance_set_id = None;
        app.async_uploads_state.skin_installed = false;
        app.async_uploads_state.load_all_started = None;
        app.async_uploads_state.load_all_duration_ms = None;
    }
    ui.label(if app.async_uploads_state.use_sync {
        "Sync calls block this frame. Watch the orbit camera stutter."
    } else {
        "Async calls return immediately. Orbit stays smooth while workers run."
    });
    ui.add_space(6.0);

    // -- Payload size toggle --------------------------------------
    ui.heading("Payload size");
    let mut new_size = app.async_uploads_state.payload_size;
    ui.horizontal(|ui| {
        if ui
            .radio(matches!(new_size, PayloadSize::Heavy), "Heavy")
            .clicked()
        {
            new_size = PayloadSize::Heavy;
        }
        if ui
            .radio(matches!(new_size, PayloadSize::Light), "Light")
            .clicked()
        {
            new_size = PayloadSize::Light;
        }
    });
    if new_size != app.async_uploads_state.payload_size {
        app.async_uploads_state.payload_size = new_size;
    }
    ui.label(match app.async_uploads_state.payload_size {
        PayloadSize::Heavy => {
            "Heavy: 1.3M-tri mesh, 4K texture (64 MB), 256^3 volume (64 MB), 1M splats, 1M-pt cloud, 50k-sample curves. Sync will block for seconds."
        }
        PayloadSize::Light => "Light: tiny assets, useful for smoke testing the controls.",
    });
    ui.add_space(6.0);

    // -- Frame budget for apply step ------------------------------
    ui.heading("Frame budget");
    let mut new_budget = app.async_uploads_state.upload_budget_ms;
    ui.horizontal(|ui| {
        if ui.radio(new_budget.is_none(), "Off").clicked() {
            new_budget = None;
        }
        if ui.radio(new_budget == Some(2), "2 ms").clicked() {
            new_budget = Some(2);
        }
        if ui.radio(new_budget == Some(5), "5 ms").clicked() {
            new_budget = Some(5);
        }
        if ui.radio(new_budget == Some(10), "10 ms").clicked() {
            new_budget = Some(10);
        }
    });
    if new_budget != app.async_uploads_state.upload_budget_ms {
        app.async_uploads_state.upload_budget_ms = new_budget;
    }
    ui.label(
        "Caps the main-thread cost of running apply closures (buffer + bind-group creation) per frame. When sixteen heavy uploads complete around the same moment, applying them all in one frame produces a fat stutter; spreading them spills the cost across the next few frames at the cost of one-frame-late visibility for some assets.",
    );
    ui.add_space(6.0);
    ui.separator();

    // -- Frame-time telemetry -------------------------------------
    let stats = frame_time_stats(&app.async_uploads_state.frame_times);
    ui.heading("Frame timing");
    ui.label(format!(
        "fps: {:.1}   |   frame ms: avg {:.1}   max {:.1}   min {:.1}",
        stats.fps, stats.avg_ms, stats.max_ms, stats.min_ms,
    ));
    ui.label("Auto-orbit at 30 deg/s. Under sync load, max climbs into the hundreds of ms while async stays near the vsync interval.");
    ui.add_space(6.0);
    ui.separator();

    // Pending count at a glance, plus push the upload budget into the
    // renderer each repaint so live changes take effect on the next
    // prepare. Cheap; just a couple of stores.
    let rs = frame.wgpu_render_state().expect("wgpu");
    {
        let mut guard = rs.renderer.write();
        if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
            renderer.set_upload_budget(
                app.async_uploads_state
                    .upload_budget_ms
                    .map(|ms| std::time::Duration::from_millis(ms as u64)),
            );
        }
    }
    let pending = {
        let guard = rs.renderer.read();
        let r = guard
            .callback_resources
            .get::<ViewportRenderer>()
            .expect("renderer");
        r.resources().uploads_pending()
    };
    ui.label(format!("In flight: {pending}"));
    ui.add_space(4.0);

    let mut clicked_env = false;
    let mut clicked_mesh = false;
    let mut clicked_texture = false;
    let mut clicked_skin = false;
    let mut clicked_polyline = false;
    let mut clicked_streamtube = false;
    let mut clicked_tube = false;
    let mut clicked_ribbon = false;
    let mut clicked_point_cloud = false;
    let mut clicked_glyph_set = false;
    let mut clicked_tensor_glyph_set = false;
    let mut clicked_volume = false;
    let mut clicked_gaussian_splats = false;
    let mut clicked_overlay_texture = false;
    let mut clicked_sprite_set = false;
    let mut clicked_sprite_instance_set = false;
    let mut clicked_all = false;

    ui.heading("Assets");
    egui::Grid::new("asset_grid")
        .num_columns(3)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            asset_row(
                ui,
                "Env map (HDR)",
                &app.async_uploads_state.env_state,
                &mut clicked_env,
            );
            asset_row(
                ui,
                "Mesh (icosphere)",
                &app.async_uploads_state.mesh_state,
                &mut clicked_mesh,
            );
            asset_row(
                ui,
                "Texture (256x256)",
                &app.async_uploads_state.texture_state,
                &mut clicked_texture,
            );
            asset_row(
                ui,
                "Skin weights",
                &app.async_uploads_state.skin_state,
                &mut clicked_skin,
            );
            asset_row(
                ui,
                "Polyline",
                &app.async_uploads_state.polyline_state,
                &mut clicked_polyline,
            );
            asset_row(
                ui,
                "Streamtube",
                &app.async_uploads_state.streamtube_state,
                &mut clicked_streamtube,
            );
            asset_row(
                ui,
                "Tube",
                &app.async_uploads_state.tube_state,
                &mut clicked_tube,
            );
            asset_row(
                ui,
                "Ribbon",
                &app.async_uploads_state.ribbon_state,
                &mut clicked_ribbon,
            );
            asset_row(
                ui,
                "Point cloud",
                &app.async_uploads_state.point_cloud_state,
                &mut clicked_point_cloud,
            );
            asset_row(
                ui,
                "Glyph set",
                &app.async_uploads_state.glyph_set_state,
                &mut clicked_glyph_set,
            );
            asset_row(
                ui,
                "Tensor glyph set",
                &app.async_uploads_state.tensor_glyph_set_state,
                &mut clicked_tensor_glyph_set,
            );
            asset_row(
                ui,
                "Volume (32^3)",
                &app.async_uploads_state.volume_state,
                &mut clicked_volume,
            );
            asset_row(
                ui,
                "Gaussian splats",
                &app.async_uploads_state.gaussian_splat_state,
                &mut clicked_gaussian_splats,
            );
            asset_row(
                ui,
                "Overlay texture",
                &app.async_uploads_state.overlay_texture_state,
                &mut clicked_overlay_texture,
            );
            asset_row(
                ui,
                "Sprite set",
                &app.async_uploads_state.sprite_set_state,
                &mut clicked_sprite_set,
            );
            asset_row(
                ui,
                "Sprite instances",
                &app.async_uploads_state.sprite_instance_set_state,
                &mut clicked_sprite_instance_set,
            );
        });

    ui.add_space(6.0);
    if ui.button("Load a level (fire all sixteen)").clicked() {
        clicked_all = true;
    }
    // Total wall-clock for the most recent "fire all" run. While the run
    // is still in flight, show a live elapsed clock; once every asset has
    // landed, freeze on the recorded duration.
    if let Some(total_ms) = app.async_uploads_state.load_all_duration_ms {
        // The real story isn't wall-clock parity, it's where the time
        // was spent. Sync uploads block the main thread for the full
        // duration; async pushes the CPU prep onto rayon workers and
        // returns immediately. Total wall-clock for the whole batch is
        // similar either way (the workers share the rayon pool with
        // what the sync path would have used internally), but the
        // main-thread cost and the worst-frame-time differ sharply.
        let main_ms = app
            .async_uploads_state
            .launch_all_main_thread_ms
            .unwrap_or(0);
        let max_frame = app.async_uploads_state.stress_max_frame_ms;
        ui.label(
            egui::RichText::new(format!("Wall-clock total: {total_ms} ms"))
                .color(egui::Color32::from_rgb(120, 220, 120)),
        );
        ui.label(
            egui::RichText::new(format!("Main thread blocked: {main_ms} ms")).color(
                if main_ms < total_ms / 2 {
                    egui::Color32::from_rgb(120, 220, 120)
                } else {
                    egui::Color32::from_rgb(220, 140, 120)
                },
            ),
        );
        ui.label(
            egui::RichText::new(format!("Worst frame during load: {max_frame:.1} ms")).color(
                if max_frame < 50.0 {
                    egui::Color32::from_rgb(120, 220, 120)
                } else {
                    egui::Color32::from_rgb(220, 140, 120)
                },
            ),
        );
        ui.label(
            egui::RichText::new(
                "Compare sync vs async on those two lines. Total wall-clock is similar (rayon-parallel work is rayon-parallel either way); the gap shows up in main-thread blocking and frame pacing.",
            )
            .weak(),
        );
    } else if let Some(started) = app.async_uploads_state.load_all_started {
        let elapsed = started.elapsed().as_millis() as u64;
        ui.label(format!("Total: {elapsed} ms (in flight)"));
    }

    ui.add_space(6.0);
    ui.separator();
    // J5 is done — all formerly greyed-out buttons are now live above.

    if !(clicked_env
        || clicked_mesh
        || clicked_texture
        || clicked_skin
        || clicked_polyline
        || clicked_streamtube
        || clicked_tube
        || clicked_ribbon
        || clicked_point_cloud
        || clicked_glyph_set
        || clicked_tensor_glyph_set
        || clicked_volume
        || clicked_gaussian_splats
        || clicked_overlay_texture
        || clicked_sprite_set
        || clicked_sprite_instance_set
        || clicked_all)
    {
        return;
    }

    let mut guard = rs.renderer.write();
    let renderer = guard
        .callback_resources
        .get_mut::<ViewportRenderer>()
        .expect("renderer");

    if clicked_env {
        app.launch_env_map(renderer);
    }
    if clicked_mesh {
        app.launch_mesh(renderer);
    }
    if clicked_texture {
        app.launch_texture(renderer);
    }
    if clicked_skin {
        app.launch_skin(renderer);
    }
    if clicked_polyline {
        app.launch_polyline(renderer);
    }
    if clicked_streamtube {
        app.launch_streamtube(renderer);
    }
    if clicked_tube {
        app.launch_tube(renderer);
    }
    if clicked_ribbon {
        app.launch_ribbon(renderer);
    }
    if clicked_point_cloud {
        app.launch_point_cloud(renderer);
    }
    if clicked_glyph_set {
        app.launch_glyph_set(renderer);
    }
    if clicked_tensor_glyph_set {
        app.launch_tensor_glyph_set(renderer);
    }
    if clicked_volume {
        app.launch_volume(renderer);
    }
    if clicked_gaussian_splats {
        app.launch_gaussian_splats(renderer);
    }
    if clicked_overlay_texture {
        app.launch_overlay_texture(renderer);
    }
    if clicked_sprite_set {
        app.launch_sprite_set(renderer);
    }
    if clicked_sprite_instance_set {
        app.launch_sprite_instance_set(renderer);
    }
    if clicked_all {
        app.launch_all(renderer);
    }
}

/// Aggregate frame-time stats over the rolling window. Used by the
/// controls panel to expose the sync vs async pacing difference at a
/// glance.
struct FrameTimeStats {
    fps: f32,
    avg_ms: f32,
    min_ms: f32,
    max_ms: f32,
}

fn frame_time_stats(window: &VecDeque<Duration>) -> FrameTimeStats {
    if window.is_empty() {
        return FrameTimeStats {
            fps: 0.0,
            avg_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
        };
    }
    let mut sum = 0.0_f64;
    let mut mn = f32::INFINITY;
    let mut mx = 0.0_f32;
    for d in window {
        let ms = d.as_secs_f64() * 1000.0;
        sum += ms;
        let ms_f = ms as f32;
        if ms_f < mn {
            mn = ms_f;
        }
        if ms_f > mx {
            mx = ms_f;
        }
    }
    let avg = (sum / window.len() as f64) as f32;
    let fps = if avg > 0.0 { 1000.0 / avg } else { 0.0 };
    FrameTimeStats {
        fps,
        avg_ms: avg,
        min_ms: mn,
        max_ms: mx,
    }
}

fn asset_row(ui: &mut egui::Ui, name: &str, state: &AssetState, clicked: &mut bool) {
    ui.label(name);
    if ui.button("Load").clicked() {
        *clicked = true;
    }
    match state {
        AssetState::Idle => {
            ui.label(egui::RichText::new("idle").weak());
        }
        AssetState::InFlight {
            progress, started, ..
        } => {
            ui.horizontal(|ui| {
                ui.add(
                    egui::ProgressBar::new(*progress)
                        .show_percentage()
                        .desired_width(120.0),
                );
                ui.label(format!("{} ms", started.elapsed().as_millis()));
            });
        }
        AssetState::Loaded { duration_ms } => {
            ui.label(
                egui::RichText::new(format!("ready in {duration_ms} ms"))
                    .color(egui::Color32::from_rgb(120, 220, 120)),
            );
        }
        AssetState::Failed {
            reason,
            duration_ms,
        } => {
            ui.label(
                egui::RichText::new(format!("failed after {duration_ms} ms: {reason}"))
                    .color(egui::Color32::from_rgb(220, 110, 110)),
            );
        }
    }
    ui.end_row();
}
