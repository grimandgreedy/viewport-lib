//! SkeletonPlugin: drives CPU linear blend skinning from a runtime Pose.

use super::skeleton::{JointMatrices, Pose, Skeleton, apply_skin};
use crate::plugins::skinning::SkinWeights;
use crate::plugins::skinning::{SkinnedMeshUpdate, SkinnedPoseUpdate};
use crate::resources::mesh_store::MeshId;
use crate::runtime::context::RuntimeStepContext;
use crate::runtime::plugin::{RuntimePlugin, phase};

/// Which deformation path a skinning plugin should emit each frame.
///
/// `Cpu` runs LBS on the CPU and emits [`SkinnedMeshUpdate`]; the host must
/// upload deformed positions/normals via `write_mesh_positions_normals`.
/// `Gpu` emits [`SkinnedPoseUpdate`] carrying joint matrices; the host
/// uploads them via
/// [`SkinningPlugin::attach_palette`](crate::plugins::skinning::SkinningPlugin::attach_palette)
/// and the registered skinning deformer does LBS in the vertex stage.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SkinningPath {
    /// CPU LBS, re-upload deformed vertices each frame.
    Cpu,
    /// GPU LBS, upload only the joint palette each frame.
    Gpu,
}

impl Default for SkinningPath {
    fn default() -> Self {
        SkinningPath::Cpu
    }
}

/// Built-in plugin that applies CPU linear blend skinning each frame.
///
/// Store a [`Pose`] in [`super::super::resources::RuntimeResources`] (keyed by
/// the `Pose` type) during the `ANIMATE` phase or earlier. `SkeletonPlugin`
/// runs at `POST_SIM`, reads the pose, computes the skinning matrices, and
/// emits a [`SkinnedMeshUpdate`] event onto `ctx.output.events`.
///
/// After `step()`, drain `output.events.drain::<SkinnedMeshUpdate>()` and
/// call `renderer.resources_mut().write_mesh_positions_normals(queue, id,
/// pos, nrm)` to upload the deformed geometry.
///
/// # Example
///
/// ```rust,ignore
/// // Startup: register the plugin with bind-pose mesh data.
/// let plugin = SkeletonPlugin::new(skeleton, mesh_id, positions, normals, skin_weights);
/// let runtime = ViewportRuntime::new().with_plugin(plugin);
///
/// // Per-frame: write the animated pose before step().
/// runtime.resources_mut().insert(my_pose);
///
/// let output = runtime.step(&mut scene, &mut sel, &frame_ctx);
///
/// for u in output.events.drain::<SkinnedMeshUpdate>() {
///     renderer.resources_mut()
///         .write_mesh_positions_normals(queue, u.mesh_id, &u.positions, &u.normals)
///         .ok();
/// }
/// ```
pub struct SkeletonPlugin {
    /// The skeleton this plugin deforms against.
    pub skeleton: Skeleton,
    /// The GPU mesh to update each frame.
    pub mesh_id: MeshId,
    /// Which deformation path to emit each frame. Defaults to `Cpu` so existing
    /// consumers keep working unchanged. Set to `Gpu` when the host has
    /// uploaded skin weights for `mesh_id` and is draining
    /// `output.events.drain::<SkinnedPoseUpdate>()` into
    /// `SkinningPlugin::attach_palette`.
    pub path: SkinningPath,
    cpu_positions: Vec<[f32; 3]>,
    cpu_normals: Vec<[f32; 3]>,
    skin_weights: SkinWeights,
}

impl SkeletonPlugin {
    /// Create a new `SkeletonPlugin`.
    ///
    /// `positions` and `normals` are the bind-pose vertex arrays (same as
    /// those passed to `upload_mesh_data`). `skin_weights` must have the same
    /// vertex count as `positions`.
    pub fn new(
        skeleton: Skeleton,
        mesh_id: MeshId,
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        skin_weights: SkinWeights,
    ) -> Self {
        Self {
            skeleton,
            mesh_id,
            path: SkinningPath::default(),
            cpu_positions: positions,
            cpu_normals: normals,
            skin_weights,
        }
    }

    /// Override the deformation path. Builder-style for ergonomic init.
    pub fn with_path(mut self, path: SkinningPath) -> Self {
        self.path = path;
        self
    }
}

impl RuntimePlugin for SkeletonPlugin {
    fn priority(&self) -> i32 {
        phase::POST_SIM
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        let Some(pose) = ctx.resources.get::<Pose>() else {
            return;
        };
        let matrices = JointMatrices::compute(&self.skeleton, pose);
        match self.path {
            SkinningPath::Cpu => {
                let (positions, normals) = apply_skin(
                    &self.cpu_positions,
                    &self.cpu_normals,
                    &self.skin_weights,
                    &matrices,
                );
                ctx.output.events.emit(SkinnedMeshUpdate {
                    mesh_id: self.mesh_id,
                    positions,
                    normals,
                });
            }
            SkinningPath::Gpu => {
                let joint_matrices: Vec<glam::Mat4> = matrices
                    .as_slice()
                    .iter()
                    .map(|m| glam::Mat4::from(*m))
                    .collect();
                ctx.output.events.emit(SkinnedPoseUpdate {
                    mesh_id: self.mesh_id,
                    instance_id: 0,
                    joint_matrices,
                });
            }
        }
    }
}
