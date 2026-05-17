//! SkeletonPlugin: drives CPU linear blend skinning from a runtime Pose.

use crate::resources::mesh_store::MeshId;
use crate::resources::SkinWeights;
use crate::runtime::context::RuntimeStepContext;
use crate::runtime::output::SkinnedMeshUpdate;
use crate::runtime::plugin::{RuntimePlugin, phase};
use crate::runtime::skeleton::{JointMatrices, Pose, Skeleton, apply_skin};

/// Built-in plugin that applies CPU linear blend skinning each frame.
///
/// Store a [`Pose`] in [`super::super::resources::RuntimeResources`] (keyed by
/// the `Pose` type) during the `ANIMATE` phase or earlier. `SkeletonPlugin`
/// runs at `POST_SIM`, reads the pose, computes the skinning matrices, and
/// pushes a [`SkinnedMeshUpdate`] to `ctx.output.skinned_mesh_updates`.
///
/// After `step()`, iterate `output.skinned_mesh_updates` and call
/// `renderer.resources_mut().write_mesh_positions_normals(queue, id, pos, nrm)`
/// to upload the deformed geometry.
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
/// for u in &output.skinned_mesh_updates {
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
        Self { skeleton, mesh_id, cpu_positions: positions, cpu_normals: normals, skin_weights }
    }
}

impl RuntimePlugin for SkeletonPlugin {
    fn priority(&self) -> i32 {
        phase::POST_SIM
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        if let Some(pose) = ctx.resources.get::<Pose>() {
            let matrices = JointMatrices::compute(&self.skeleton, pose);
            let (positions, normals) = apply_skin(
                &self.cpu_positions,
                &self.cpu_normals,
                &self.skin_weights,
                &matrices,
            );
            ctx.output.skinned_mesh_updates.push(SkinnedMeshUpdate {
                mesh_id: self.mesh_id,
                positions,
                normals,
            });
        }
    }
}
