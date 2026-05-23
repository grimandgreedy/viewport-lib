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
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::SkinWeights;
    use glam::{Affine3A, Vec3};

    fn two_joint_skeleton(joint_z: f32) -> Skeleton {
        Skeleton::new(vec![
            Joint {
                name: "root".into(),
                parent: None,
                inverse_bind: Affine3A::IDENTITY,
            },
            Joint {
                name: "child".into(),
                parent: Some(0),
                inverse_bind: Affine3A::from_translation(-Vec3::new(0.0, 0.0, joint_z)),
            },
        ])
    }

    /// Returns the pose whose forward kinematics reproduces the bind pose for
    /// `two_joint_skeleton(joint_z)`. Joint 0 stays at the origin; joint 1
    /// sits at z=joint_z, which is the inverse of its `inverse_bind`.
    fn bind_pose(joint_z: f32) -> Pose {
        let mut p = Pose::identity(2);
        p.local_transforms[1] = Affine3A::from_translation(Vec3::new(0.0, 0.0, joint_z));
        p
    }

    fn approx_eq(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    #[test]
    fn bind_pose_produces_identity_skinning_matrices() {
        let joint_z = 2.0;
        let sk = two_joint_skeleton(joint_z);
        let jm = JointMatrices::compute(&sk, &bind_pose(joint_z));
        for m in jm.as_slice() {
            let p = m.transform_point3(Vec3::new(1.0, 2.0, 3.0));
            assert!(approx_eq(p.to_array(), [1.0, 2.0, 3.0], 1e-5), "got {:?}", p);
        }
    }

    #[test]
    fn apply_skin_at_bind_pose_returns_input() {
        let joint_z = 2.0;
        let sk = two_joint_skeleton(joint_z);
        let jm = JointMatrices::compute(&sk, &bind_pose(joint_z));
        let positions = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 1.0], [0.0, 0.0, 4.0]];
        let normals = vec![[1.0, 0.0, 0.0]; 3];
        let weights = SkinWeights {
            joint_indices: vec![[0, 1, 0, 0]; 3],
            joint_weights: vec![[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        };
        let (out_p, out_n) = apply_skin(&positions, &normals, &weights, &jm);
        for i in 0..3 {
            assert!(approx_eq(out_p[i], positions[i], 1e-5), "pos {i}: {:?}", out_p[i]);
            assert!(approx_eq(out_n[i], normals[i], 1e-5), "nrm {i}: {:?}", out_n[i]);
        }
    }

    #[test]
    fn child_rotation_bends_around_joint() {
        // Rotating joint 1 by 90 deg around X with the bind transform applied
        // should swing a child-weighted vertex at (0,0,3) down to (0,-1,2).
        let joint_z = 2.0;
        let sk = two_joint_skeleton(joint_z);
        let mut pose = bind_pose(joint_z);
        pose.local_transforms[1] = Affine3A::from_translation(Vec3::new(0.0, 0.0, joint_z))
            * Affine3A::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let jm = JointMatrices::compute(&sk, &pose);

        let positions = vec![[0.0, 0.0, joint_z + 1.0]];
        let normals = vec![[0.0, 0.0, 1.0]];
        let weights = SkinWeights {
            joint_indices: vec![[0, 1, 0, 0]],
            joint_weights: vec![[0.0, 1.0, 0.0, 0.0]],
        };
        let (out_p, out_n) = apply_skin(&positions, &normals, &weights, &jm);
        assert!(approx_eq(out_p[0], [0.0, -1.0, joint_z], 1e-4), "got {:?}", out_p[0]);
        assert!(approx_eq(out_n[0], [0.0, -1.0, 0.0], 1e-4), "got {:?}", out_n[0]);
    }

    #[test]
    fn zero_weight_slots_are_skipped() {
        let joint_z = 2.0;
        let sk = two_joint_skeleton(joint_z);
        let mut pose = bind_pose(joint_z);
        // Add a huge translation to joint 1, but weight 0 for our vertex.
        pose.local_transforms[1] = pose.local_transforms[1]
            * Affine3A::from_translation(Vec3::new(100.0, 0.0, 0.0));
        let jm = JointMatrices::compute(&sk, &pose);

        let positions = vec![[0.0, 0.0, 0.0]];
        let normals = vec![[1.0, 0.0, 0.0]];
        let weights = SkinWeights {
            joint_indices: vec![[0, 1, 1, 1]],
            joint_weights: vec![[1.0, 0.0, 0.0, 0.0]],
        };
        let (out_p, _) = apply_skin(&positions, &normals, &weights, &jm);
        assert!(approx_eq(out_p[0], [0.0, 0.0, 0.0], 1e-5));
    }
}
