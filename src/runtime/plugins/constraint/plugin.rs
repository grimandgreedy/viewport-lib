//! ConstraintPlugin: spring-to-target, damping, and bounds constraints.

use std::collections::HashMap;

use crate::interaction::selection::NodeId;
use crate::runtime::context::RuntimeStepContext;
use crate::runtime::plugin::{phase, RuntimePlugin};

/// A positional constraint applied to a scene node.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Pull a node toward a fixed world-space target using a spring with damping.
    ///
    /// Uses semi-implicit Euler integration: velocity updated by spring and
    /// damping forces, then position updated by velocity.
    SpringTarget {
        /// Target node.
        node_id: NodeId,
        /// World-space point the spring pulls toward.
        target: glam::Vec3,
        /// Spring stiffness in 1/s^2. Higher values pull more aggressively.
        stiffness: f32,
        /// Velocity damping coefficient in 1/s. Prevents oscillation.
        damping: f32,
    },
    /// Drag a node's velocity to zero over time.
    ///
    /// Velocity is multiplied by `(1 - damping * dt)` each step. Requires an
    /// initial velocity in the internal state (starts at zero).
    Dampen {
        /// Target node.
        node_id: NodeId,
        /// Damping coefficient in 1/s.
        damping: f32,
    },
    /// Clamp a node's world-space position to stay inside an axis-aligned box.
    ///
    /// Velocity is zeroed on contact with a boundary.
    ClampBounds {
        /// Target node.
        node_id: NodeId,
        /// Minimum corner of the allowed region.
        min: glam::Vec3,
        /// Maximum corner of the allowed region.
        max: glam::Vec3,
    },
}

impl Constraint {
    fn node_id(&self) -> NodeId {
        match self {
            Constraint::SpringTarget { node_id, .. }
            | Constraint::Dampen { node_id, .. }
            | Constraint::ClampBounds { node_id, .. } => *node_id,
        }
    }
}

/// A plugin that applies positional constraints to scene nodes.
///
/// Runs in the [`RuntimePhase::Animate`] phase. Spring and damping constraints
/// maintain per-node velocity state internally.
pub struct ConstraintPlugin {
    constraints: Vec<Constraint>,
    velocities: HashMap<NodeId, glam::Vec3>,
}

impl Default for ConstraintPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintPlugin {
    /// Create with no constraints.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            velocities: HashMap::new(),
        }
    }

    /// Add a constraint (builder style).
    pub fn add(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add a constraint.
    pub fn push(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }
}

impl RuntimePlugin for ConstraintPlugin {
    fn priority(&self) -> i32 {
        phase::ANIMATE + 50
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        for constraint in &self.constraints {
            let id = constraint.node_id();
            let Some(node) = ctx.scene.node(id) else {
                continue;
            };
            let world = node.world_transform();
            let (scale, rotation, pos) = world.to_scale_rotation_translation();
            let vel = self.velocities.entry(id).or_insert(glam::Vec3::ZERO);

            let new_pos = match constraint {
                Constraint::SpringTarget {
                    target,
                    stiffness,
                    damping,
                    ..
                } => {
                    // Semi-implicit Euler spring.
                    let spring = (*target - pos) * *stiffness;
                    let damp = *vel * *damping;
                    *vel += (spring - damp) * ctx.dt;
                    pos + *vel * ctx.dt
                }
                Constraint::Dampen { damping, .. } => {
                    *vel *= (1.0 - damping * ctx.dt).max(0.0);
                    pos + *vel * ctx.dt
                }
                Constraint::ClampBounds { min, max, .. } => {
                    let clamped = pos.clamp(*min, *max);
                    if clamped != pos {
                        *vel = glam::Vec3::ZERO;
                    }
                    clamped
                }
            };

            let new_t = glam::Affine3A::from_scale_rotation_translation(scale, rotation, new_pos);
            ctx.writeback.set(id, new_t);
        }
    }
}
