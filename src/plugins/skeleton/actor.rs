//! SkinnedActorPlugin: many independently-animated skinned actors sharing one
//! [`Skeleton`].
//!
//! Each [`SkinnedActor`] has its own playhead, clip, and play state. Within
//! an actor, all [`SkinnedActorPart`]s deform from that actor's pose, so the
//! parts of a multi-mesh character animate in sync. Across actors, animation
//! is independent: a crowd of N actors with staggered playheads animates as N
//! distinct loops over the same source data.
//!
//! Compared to [`super::SkeletonPlugin`] + [`super::ClipPlayerPlugin`]:
//!
//! - That pair runs *one* animation: the player writes one `Pose` to
//!   resources and one or more SkeletonPlugins consume it. Good for a single
//!   character (potentially with many parts) animating in lockstep.
//! - `SkinnedActorPlugin` is the multi-actor case: a crowd, a stadium of
//!   NPCs, an army. One plugin holds the shared skeleton and N actors that
//!   each animate independently, in a single phase tick.
//!
//! This phase is the CPU baseline. The same actor/part decomposition is what
//! a later GPU-skinning path will turn into per-actor joint-palette uploads
//! and per-part skinned draw calls, so callers should target this shape now.

use crate::resources::SkinWeights;
use crate::resources::mesh_store::MeshId;
use crate::runtime::context::RuntimeStepContext;
use crate::runtime::output::{SkinnedMeshUpdate, SkinnedPoseUpdate};
use crate::runtime::plugin::{RuntimePlugin, phase};

use super::clip::AnimationClip;
use super::plugin::SkinningPath;
use super::skeleton::{JointMatrices, Pose, Skeleton, apply_skin};

/// One skinned mesh of an actor. All parts of an actor share that actor's
/// per-frame pose; each part owns its own bind-pose vertex data and GPU mesh
/// because the CPU LBS path writes deformed vertices back per-mesh.
pub struct SkinnedActorPart {
    /// GPU mesh this part deforms each frame.
    pub mesh_id: MeshId,
    /// Bind-pose positions; same shape as the data uploaded to `mesh_id`.
    pub bind_positions: Vec<[f32; 3]>,
    /// Bind-pose normals.
    pub bind_normals: Vec<[f32; 3]>,
    /// Per-vertex joint indices and weights, parallel to `bind_positions`.
    pub skin_weights: SkinWeights,
}

/// One independently-animated actor. Its parts deform from this actor's pose
/// each frame; other actors in the same plugin animate from their own.
pub struct SkinnedActor {
    /// Mesh parts that make up this actor.
    pub parts: Vec<SkinnedActorPart>,
    /// Index into [`SkinnedActorPlugin::clips`] selecting the active clip.
    pub clip_index: usize,
    /// Current play position in seconds.
    pub playhead: f32,
    /// Playback speed multiplier. `1.0` = real time. Negative values reverse.
    pub speed: f32,
    /// Whether playback wraps at the active clip's duration.
    pub looping: bool,
    /// When false, the playhead does not advance.
    pub playing: bool,
}

impl SkinnedActor {
    /// Create an actor playing clip 0 at real-time speed, looping.
    pub fn new(parts: Vec<SkinnedActorPart>) -> Self {
        Self {
            parts,
            clip_index: 0,
            playhead: 0.0,
            speed: 1.0,
            looping: true,
            playing: true,
        }
    }

    /// Set the active clip index.
    pub fn with_clip(mut self, clip_index: usize) -> Self {
        self.clip_index = clip_index;
        self
    }

    /// Set the initial playhead, useful for de-phasing actors in a crowd.
    pub fn with_playhead(mut self, playhead: f32) -> Self {
        self.playhead = playhead;
        self
    }

    /// Set the playback speed multiplier.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }
}

/// Runtime plugin that animates many actors sharing one skeleton.
///
/// Runs at `phase::POST_SIM`. Does not read or write `RuntimeResources::Pose`;
/// each actor's pose is built internally from its own playhead and applied
/// directly. Emits one `SkinnedMeshUpdate` per part per actor each frame.
pub struct SkinnedActorPlugin {
    /// Skeleton shared by every actor.
    pub skeleton: Skeleton,
    /// Bind pose, cloned each step before the actor's clip is sampled over
    /// it. Provides default channel values for joints the clip does not
    /// animate.
    pub bind_pose: Pose,
    /// Clips available to actors via `SkinnedActor::clip_index`.
    pub clips: Vec<AnimationClip>,
    /// All actors driven by this plugin.
    pub actors: Vec<SkinnedActor>,
    /// Which deformation path to emit each frame. On `Gpu`, one
    /// [`SkinnedPoseUpdate`] is pushed per actor per part. The instance id is
    /// the actor's index in `actors` so the host can drive the right joint
    /// palette via `set_skin_palette`.
    pub path: SkinningPath,
}

impl SkinnedActorPlugin {
    /// Create a plugin with no actors yet.
    pub fn new(skeleton: Skeleton, bind_pose: Pose, clips: Vec<AnimationClip>) -> Self {
        Self {
            skeleton,
            bind_pose,
            clips,
            actors: Vec::new(),
            path: SkinningPath::default(),
        }
    }

    /// Override the deformation path. Builder-style for ergonomic init.
    pub fn with_path(mut self, path: SkinningPath) -> Self {
        self.path = path;
        self
    }

    /// Append an actor.
    pub fn with_actor(mut self, actor: SkinnedActor) -> Self {
        self.actors.push(actor);
        self
    }

    /// Append many actors at once.
    pub fn with_actors(mut self, actors: impl IntoIterator<Item = SkinnedActor>) -> Self {
        self.actors.extend(actors);
        self
    }
}

impl RuntimePlugin for SkinnedActorPlugin {
    fn priority(&self) -> i32 {
        phase::POST_SIM
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        for (actor_idx, actor) in self.actors.iter_mut().enumerate() {
            let clip = match self.clips.get(actor.clip_index) {
                Some(c) => c,
                None => continue,
            };

            if actor.playing {
                actor.playhead += ctx.dt * actor.speed;
                if clip.duration > 0.0 {
                    if actor.looping {
                        actor.playhead = actor.playhead.rem_euclid(clip.duration);
                    } else {
                        actor.playhead = actor.playhead.clamp(0.0, clip.duration);
                    }
                }
            }

            // One pose + one FK pass per actor; parts share the result.
            let mut pose = self.bind_pose.clone();
            clip.sample_into(actor.playhead, &mut pose);
            let matrices = JointMatrices::compute(&self.skeleton, &pose);

            match self.path {
                SkinningPath::Cpu => {
                    for part in &actor.parts {
                        let (positions, normals) = apply_skin(
                            &part.bind_positions,
                            &part.bind_normals,
                            &part.skin_weights,
                            &matrices,
                        );
                        ctx.output.skinned_mesh_updates.push(SkinnedMeshUpdate {
                            mesh_id: part.mesh_id,
                            positions,
                            normals,
                        });
                    }
                }
                SkinningPath::Gpu => {
                    let joint_matrices: Vec<glam::Mat4> = matrices
                        .as_slice()
                        .iter()
                        .map(|m| glam::Mat4::from(*m))
                        .collect();
                    for part in &actor.parts {
                        ctx.output.skinned_pose_updates.push(SkinnedPoseUpdate {
                            mesh_id: part.mesh_id,
                            instance_id: actor_idx as u32,
                            joint_matrices: joint_matrices.clone(),
                        });
                    }
                }
            }
        }
    }
}
