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

use std::time::Instant;

use eframe::egui;
use viewport_lib::{
    GlyphItem, GlyphSetId, GlyphSetRefItem, JobId, LightKind, LightSource, LightingSettings,
    Material, MeshData, MeshId, PointCloudId, PointCloudItem, PointCloudRefItem, PolylineId,
    PolylineItem, PolylineRefItem, RibbonId, RibbonItem, RibbonRefItem, SceneRenderItem,
    SkinWeights, StreamtubeId, StreamtubeItem, StreamtubeRefItem, TensorGlyphItem,
    TensorGlyphSetId, TensorGlyphSetRefItem, TubeId, TubeItem, TubeRefItem, UploadStatus,
    ViewportRenderer,
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

pub(crate) struct AsyncUploadsState {
    /// When true, button clicks call the synchronous upload path. Useful
    /// for showing the difference in frame pacing.
    pub use_sync: bool,
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

    pub loaded_mesh_id: Option<MeshId>,
    pub loaded_texture_id: Option<u64>,
    pub loaded_polyline_id: Option<PolylineId>,
    pub loaded_streamtube_id: Option<StreamtubeId>,
    pub loaded_tube_id: Option<TubeId>,
    pub loaded_ribbon_id: Option<RibbonId>,
    pub loaded_point_cloud_id: Option<PointCloudId>,
    pub loaded_glyph_set_id: Option<GlyphSetId>,
    pub loaded_tensor_glyph_set_id: Option<TensorGlyphSetId>,
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
            use_sync: false,
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
            loaded_mesh_id: None,
            loaded_texture_id: None,
            loaded_polyline_id: None,
            loaded_streamtube_id: None,
            loaded_tube_id: None,
            loaded_ribbon_id: None,
            loaded_point_cloud_id: None,
            loaded_glyph_set_id: None,
            loaded_tensor_glyph_set_id: None,
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
        self.async_uploads_state.loaded_mesh_id = None;
        self.async_uploads_state.loaded_texture_id = None;
        self.async_uploads_state.loaded_polyline_id = None;
        self.async_uploads_state.loaded_streamtube_id = None;
        self.async_uploads_state.loaded_tube_id = None;
        self.async_uploads_state.loaded_ribbon_id = None;
        self.async_uploads_state.loaded_point_cloud_id = None;
        self.async_uploads_state.loaded_glyph_set_id = None;
        self.async_uploads_state.loaded_tensor_glyph_set_id = None;
        self.async_uploads_state.skin_installed = false;
        self.async_uploads_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Per-frame update: advance the per-asset state machines
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn async_uploads_update(&mut self, renderer: &mut ViewportRenderer) {
        // Env map: status only, no typed result to take.
        let env_just_loaded = advance_status_only(
            &mut self.async_uploads_state.env_state,
            renderer,
        );

        // Mesh: when Ready, take the MeshId.
        if let AssetState::InFlight { job, started, .. } =
            self.async_uploads_state.mesh_state.clone()
        {
            match renderer.upload_status(job) {
                UploadStatus::Ready => match renderer.upload_result_mesh(job) {
                    Ok(mesh_id) => {
                        let duration_ms = take_job_duration_ms(renderer, job, &started);
                        self.async_uploads_state.loaded_mesh_id = Some(mesh_id);
                        self.async_uploads_state.mesh_state =
                            AssetState::Loaded { duration_ms };
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
                        self.async_uploads_state.texture_state =
                            AssetState::Loaded { duration_ms };
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
        let skin_just_loaded = advance_status_only(
            &mut self.async_uploads_state.skin_state,
            renderer,
        );
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
                        self.async_uploads_state.tube_state =
                            AssetState::Loaded { duration_ms };
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
                        self.async_uploads_state.ribbon_state =
                            AssetState::Loaded { duration_ms };
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
                UploadStatus::Ready => match renderer.resources_mut().upload_result_point_cloud(job)
                {
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
                },
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
                UploadStatus::Ready => match renderer.resources_mut().upload_result_glyph_set(job) {
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
                },
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
                UploadStatus::Ready => match renderer
                    .resources_mut()
                    .upload_result_tensor_glyph_set(job)
                {
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
                },
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

        // Once a "Load a level" run has every asset in a terminal state,
        // stamp the total wall-clock duration so the controls panel can
        // show it next frame.
        if let Some(started) = self.async_uploads_state.load_all_started {
            if self.async_uploads_state.load_all_duration_ms.is_none()
                && all_assets_terminal(&self.async_uploads_state)
            {
                self.async_uploads_state.load_all_duration_ms =
                    Some(started.elapsed().as_millis() as u64);
            }
        }
    }
}

fn all_assets_terminal(state: &AsyncUploadsState) -> bool {
    let is_terminal = |s: &AssetState| {
        matches!(s, AssetState::Loaded { .. } | AssetState::Failed { .. })
    };
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
            item.model = glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.0, 0.0))
                .to_cols_array_2d();
            item.material = Material::flat([0.4, 0.8, 0.4]);
            items.push(item);
        }

        if let (Some(mesh_id), Some(tex_id)) = (state.base_mesh_id, state.loaded_texture_id) {
            let mut material = Material::flat([1.0, 1.0, 1.0]);
            material.texture_id = Some(tex_id);
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(4.0, 0.0, 0.0))
                .to_cols_array_2d();
            item.material = material;
            items.push(item);
        }

        if let Some(id) = state.skin_target_mesh_id {
            let mut item = SceneRenderItem::default();
            item.mesh_id = id;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(-2.0, 0.0, 0.0))
                .to_cols_array_2d();
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
        if matches!(self.async_uploads_state.env_state, AssetState::Loaded { .. }) {
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

fn demo_mesh() -> MeshData {
    // A small icosphere keeps the upload work nontrivial without freezing
    // the showcase if the user picks Sync mode.
    viewport_lib::primitives::icosphere(0.6, 3)
}

fn unit_skin_weights(vertex_count: usize) -> SkinWeights {
    SkinWeights {
        joint_indices: vec![[0u8; 4]; vertex_count],
        joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; vertex_count],
    }
}

// A short helix in the XZ plane, used by all four curve types so each
// pre-uploaded asset has visibly the same source geometry.
fn demo_curve_positions() -> (Vec<[f32; 3]>, Vec<u32>) {
    let n: usize = 48;
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

fn demo_polyline() -> PolylineItem {
    let (positions, strip_lengths) = demo_curve_positions();
    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.line_width = 3.0;
    item.default_colour = [0.95, 0.55, 0.35, 1.0];
    item
}

fn demo_streamtube() -> StreamtubeItem {
    let (positions, strip_lengths) = demo_curve_positions();
    let mut item = StreamtubeItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.radius = 0.07;
    item.colour = [0.45, 0.85, 0.95, 1.0];
    item
}

fn demo_tube() -> TubeItem {
    let (positions, strip_lengths) = demo_curve_positions();
    let mut item = TubeItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.radius = 0.08;
    item.sides = 16;
    item.colour = [0.95, 0.85, 0.45, 1.0];
    item
}

fn demo_ribbon() -> RibbonItem {
    let (positions, strip_lengths) = demo_curve_positions();
    let mut item = RibbonItem::default();
    item.positions = positions;
    item.strip_lengths = strip_lengths;
    item.width = 0.18;
    item.colour = [0.65, 0.55, 0.95, 1.0];
    item
}

fn demo_point_cloud() -> PointCloudItem {
    // A flat 16x16 grid of points so the cloud is visible regardless of
    // camera direction.
    let mut item = PointCloudItem::default();
    let n: i32 = 16;
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
        let data = demo_mesh();
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            match renderer.resources_mut().upload_mesh_data(&self.device, &data) {
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
        let rgba = checker_rgba(256, 256);
        let started = Instant::now();
        if self.async_uploads_state.use_sync {
            match renderer
                .resources_mut()
                .upload_texture(&self.device, &self.queue, 256, 256, &rgba)
            {
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
            match renderer.begin_upload_texture(&self.device, &self.queue, 256, 256, rgba) {
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
        if self.async_uploads_state.use_sync {
            renderer
                .resources_mut()
                .set_skin_weights(&self.device, mesh_id, &weights);
            self.async_uploads_state.skin_installed = true;
            self.async_uploads_state.skin_state = AssetState::Loaded {
                duration_ms: started.elapsed().as_millis() as u64,
            };
        } else {
            let job = renderer
                .resources_mut()
                .begin_upload_skin_weights(&self.device, mesh_id, weights);
            self.async_uploads_state.skin_state = AssetState::InFlight {
                job,
                progress: 0.0,
                started,
            };
        }
    }

    fn launch_polyline(&mut self, renderer: &mut ViewportRenderer) {
        let item = demo_polyline();
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
        let item = demo_streamtube();
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
        let item = demo_tube();
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
        let item = demo_ribbon();
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
        let item = demo_point_cloud();
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
            let job = renderer.resources_mut().begin_upload_point_cloud(
                &self.device,
                &self.queue,
                item,
            );
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
            let job = renderer.resources_mut().begin_upload_glyph_set(
                &self.device,
                &self.queue,
                item,
            );
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
            let id = renderer
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

    fn launch_all(&mut self, renderer: &mut ViewportRenderer) {
        self.async_uploads_state.load_all_started = Some(Instant::now());
        self.async_uploads_state.load_all_duration_ms = None;
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
    }
}

// Pushes per-frame reference items for any pre-uploaded curves into the
// scene frame. Each curve is positioned in front of the camera, fanned out
// horizontally so all four can be loaded at once.
pub(crate) fn submit_async_uploads_items(
    app: &mut crate::App,
    fd: &mut viewport_lib::FrameData,
) {
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
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_async_uploads(
    app: &mut App,
    ui: &mut egui::Ui,
    frame: &eframe::Frame,
) {
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
        app.async_uploads_state.loaded_mesh_id = None;
        app.async_uploads_state.loaded_texture_id = None;
        app.async_uploads_state.loaded_polyline_id = None;
        app.async_uploads_state.loaded_streamtube_id = None;
        app.async_uploads_state.loaded_tube_id = None;
        app.async_uploads_state.loaded_ribbon_id = None;
        app.async_uploads_state.loaded_point_cloud_id = None;
        app.async_uploads_state.loaded_glyph_set_id = None;
        app.async_uploads_state.loaded_tensor_glyph_set_id = None;
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
    ui.separator();

    // Pending count at a glance.
    let rs = frame.wgpu_render_state().expect("wgpu");
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
        });

    ui.add_space(6.0);
    if ui.button("Load a level (fire all eleven)").clicked() {
        clicked_all = true;
    }
    // Total wall-clock for the most recent "fire all" run. While the run
    // is still in flight, show a live elapsed clock; once every asset has
    // landed, freeze on the recorded duration.
    if let Some(total_ms) = app.async_uploads_state.load_all_duration_ms {
        ui.label(
            egui::RichText::new(format!("Total: {total_ms} ms"))
                .color(egui::Color32::from_rgb(120, 220, 120)),
        );
    } else if let Some(started) = app.async_uploads_state.load_all_started {
        let elapsed = started.elapsed().as_millis() as u64;
        ui.label(format!("Total: {elapsed} ms (in flight)"));
    }

    ui.add_space(6.0);
    ui.separator();
    ui.heading("J5 (greyed out, lands later)");
    ui.add_enabled_ui(false, |ui| {
        let _ = ui.button("Volume");
        let _ = ui.button("Gaussian splats");
        let _ = ui.button("Sprite set");
        let _ = ui.button("Overlay texture");
    });

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
    if clicked_all {
        app.launch_all(renderer);
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
