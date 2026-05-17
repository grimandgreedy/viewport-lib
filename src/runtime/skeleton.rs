//! Skeleton, pose, and CPU linear blend skinning.
//!
//! These types form the substrate for skeletal animation. A [`Skeleton`] defines
//! the bone hierarchy and bind-pose inverses. A [`Pose`] holds local-space
//! transforms for each joint. [`JointMatrices::compute`] runs forward kinematics
//! and returns the per-joint skinning matrices ready for [`apply_skin`].
//!
//! # Workflow
//!
//! ```rust,ignore
//! // Once at startup:
//! let skeleton = Skeleton::new(joints);
//! let base_pose = Pose::identity(skeleton.joint_count());
//!
//! // Each frame (in a plugin at phase::ANIMATE or later):
//! ctx.resources.insert(my_pose); // write current pose
//!
//! // SkeletonPlugin at phase::POST_SIM reads the pose and pushes a
//! // SkinnedMeshUpdate to ctx.output.skinned_mesh_updates.
//!
//! // After runtime.step(), in the app:
//! for u in &output.skinned_mesh_updates {
//!     renderer.resources_mut()
//!         .write_mesh_positions_normals(queue, u.mesh_id, &u.positions, &u.normals)
//!         .ok();
//! }
//! ```

use crate::resources::SkinWeights;

/// Maximum number of joints in a skeleton.
pub const MAX_JOINTS: usize = 128;

/// A single joint in a skeleton hierarchy.
pub struct Joint {
    /// Display name for the joint.
    pub name: String,
    /// Index of the parent joint. `None` for root joints.
    ///
    /// Parent indices must be less than the joint's own index (topological
    /// order), so forward kinematics can be computed in a single pass.
    pub parent: Option<u8>,
    /// Inverse of the joint's world-space transform in the bind pose.
    ///
    /// `inverse_bind = bind_world_transform.inverse()`. The skinning matrix
    /// for joint `i` is `world_transform[i] * inverse_bind[i]`.
    pub inverse_bind: glam::Affine3A,
}

/// A joint hierarchy with bind-pose inverse matrices.
///
/// Joints must be stored in topological order: each joint's parent index is
/// less than its own. This is the standard glTF/FBX convention and allows
/// forward kinematics in a single forward pass.
pub struct Skeleton {
    joints: Vec<Joint>,
}

impl Skeleton {
    /// Create a skeleton from a list of joints in topological order.
    ///
    /// Panics in debug builds if any parent index is >= the joint's own index
    /// or if `joints.len() > MAX_JOINTS`.
    pub fn new(joints: Vec<Joint>) -> Self {
        debug_assert!(joints.len() <= MAX_JOINTS, "skeleton exceeds MAX_JOINTS");
        for (i, j) in joints.iter().enumerate() {
            if let Some(p) = j.parent {
                debug_assert!((p as usize) < i, "joint {i} has parent {p} >= own index");
            }
        }
        Self { joints }
    }

    /// All joints in topological order.
    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// Number of joints.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Find a joint by name. Returns the first match or `None`.
    pub fn find_joint(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }
}

/// Per-frame local-space transforms for each joint.
///
/// One `Affine3A` per joint, indexed in the same order as the parent
/// [`Skeleton`]. Store this in [`super::resources::RuntimeResources`] so
/// animation plugins can write it and [`super::plugins::SkeletonPlugin`] can
/// read it in the same frame.
pub struct Pose {
    /// Local-space transform for each joint. Must have the same length as the
    /// skeleton it is paired with.
    pub local_transforms: Vec<glam::Affine3A>,
}

impl Pose {
    /// Create a pose with all joints at identity.
    pub fn identity(joint_count: usize) -> Self {
        Self {
            local_transforms: vec![glam::Affine3A::IDENTITY; joint_count],
        }
    }

    /// Number of joint transforms in the pose.
    pub fn joint_count(&self) -> usize {
        self.local_transforms.len()
    }
}

/// Per-joint skinning matrices computed from a [`Skeleton`] and [`Pose`].
///
/// Each matrix is `world_transform[i] * inverse_bind[i]`. Multiply a
/// bind-pose vertex position by this matrix (with LBS blending) to get the
/// deformed position.
pub struct JointMatrices {
    matrices: Vec<glam::Affine3A>,
}

impl JointMatrices {
    /// Run forward kinematics and compute the skinning matrix palette.
    ///
    /// Joints are processed in topological order so each parent world
    /// transform is available when the child is processed.
    pub fn compute(skeleton: &Skeleton, pose: &Pose) -> Self {
        let n = skeleton.joint_count();
        let mut world = vec![glam::Affine3A::IDENTITY; n];

        for (i, joint) in skeleton.joints().iter().enumerate() {
            let local = pose.local_transforms.get(i).copied().unwrap_or(glam::Affine3A::IDENTITY);
            world[i] = match joint.parent {
                Some(p) => world[p as usize] * local,
                None => local,
            };
        }

        let matrices = world.iter().zip(skeleton.joints().iter())
            .map(|(w, j)| *w * j.inverse_bind)
            .collect();

        Self { matrices }
    }

    /// The skinning matrix palette as a slice.
    pub fn as_slice(&self) -> &[glam::Affine3A] {
        &self.matrices
    }
}

/// Apply CPU linear blend skinning to a mesh.
///
/// Returns `(skinned_positions, skinned_normals)`. Each vertex is transformed
/// by the weighted sum of up to four joint matrices. Zero-weight influences
/// are skipped. Output normals are re-normalized.
///
/// `positions`, `normals`, and the per-vertex arrays in `weights` must all
/// have the same length.
pub fn apply_skin(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    weights: &SkinWeights,
    joint_matrices: &JointMatrices,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let n = positions.len();
    let mut out_pos = vec![[0.0f32; 3]; n];
    let mut out_nrm = vec![[0.0f32; 3]; n];

    for i in 0..n {
        let p = glam::Vec3::from(positions[i]);
        let nm = glam::Vec3::from(normals[i]);
        let indices = weights.joint_indices[i];
        let ws = weights.joint_weights[i];

        let mut blended_p = glam::Vec3::ZERO;
        let mut blended_n = glam::Vec3::ZERO;

        for k in 0..4 {
            let w = ws[k];
            if w < 1e-6 {
                continue;
            }
            let m = joint_matrices.matrices[indices[k] as usize];
            blended_p += w * m.transform_point3(p);
            blended_n += w * m.transform_vector3(nm);
        }

        out_pos[i] = blended_p.to_array();
        out_nrm[i] = blended_n.normalize_or_zero().to_array();
    }

    (out_pos, out_nrm)
}
