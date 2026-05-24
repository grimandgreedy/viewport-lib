//! PhysicsLitePlugin: simple velocity integration, gravity, and bounded collision.

use crate::interaction::selection::NodeId;
use crate::runtime::context::RuntimeStepContext;
use crate::runtime::output::ContactEvent;
use crate::runtime::plugin::{phase, RuntimePlugin};
use crate::scene::aabb::Aabb;

/// A single physics body managed by [`PhysicsLitePlugin`].
#[derive(Debug, Clone)]
pub struct PhysicsBody {
    /// Scene node this body drives.
    pub node_id: NodeId,
    /// Linear velocity in world space (m/s).
    pub velocity: glam::Vec3,
    /// Multiplier on the plugin's global gravity vector.
    pub gravity_scale: f32,
    /// Bounce restitution coefficient (0 = no bounce, 1 = perfectly elastic).
    pub restitution: f32,
    /// Optional world-space bounding box. The body reflects off the faces of
    /// this box and a [`ContactEvent`] is emitted on each bounce.
    pub bounds: Option<Aabb>,
}

impl PhysicsBody {
    /// Create a body at rest with default gravity scale and restitution.
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            velocity: glam::Vec3::ZERO,
            gravity_scale: 1.0,
            restitution: 0.7,
            bounds: None,
        }
    }

    /// Set the initial velocity.
    pub fn with_velocity(mut self, v: glam::Vec3) -> Self {
        self.velocity = v;
        self
    }

    /// Set the gravity scale.
    pub fn with_gravity_scale(mut self, s: f32) -> Self {
        self.gravity_scale = s;
        self
    }

    /// Set the restitution coefficient.
    pub fn with_restitution(mut self, r: f32) -> Self {
        self.restitution = r;
        self
    }

    /// Constrain the body inside a world-space bounding box.
    pub fn with_bounds(mut self, bounds: Aabb) -> Self {
        self.bounds = Some(bounds);
        self
    }
}

/// A plugin that integrates velocity, applies gravity, and reflects bodies off
/// bounding box walls.
///
/// Runs in the [`RuntimePhase::Simulate`] phase. Pairs well with
/// [`crate::FixedTimestep`] for stable integration.
///
/// World-bounds collisions produce [`ContactEvent`]s in [`crate::RuntimeOutput`]
/// with `node_b` set to `NodeId::MAX` (a sentinel for world geometry).
///
/// # Example
///
/// ```rust,ignore
/// use viewport_lib::{Aabb, FixedTimestep, PhysicsBody, PhysicsLitePlugin, ViewportRuntime};
///
/// let bounds = Aabb { min: glam::Vec3::splat(-5.0), max: glam::Vec3::splat(5.0) };
/// let mut physics = PhysicsLitePlugin::new()
///     .with_gravity(glam::Vec3::new(0.0, 0.0, -9.81));
/// physics.add_body(
///     PhysicsBody::new(node_id)
///         .with_velocity(glam::Vec3::new(2.0, 1.0, 4.0))
///         .with_bounds(bounds),
/// );
///
/// let runtime = ViewportRuntime::new()
///     .with_fixed_timestep(FixedTimestep::new(60.0))
///     .with_plugin(physics);
/// ```
pub struct PhysicsLitePlugin {
    /// All bodies managed by this plugin.
    pub bodies: Vec<PhysicsBody>,
    /// Global gravity acceleration vector (world space).
    pub gravity: glam::Vec3,
}

impl Default for PhysicsLitePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsLitePlugin {
    /// Create with default downward gravity (-Z) and no bodies.
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            gravity: glam::Vec3::new(0.0, 0.0, -9.81),
        }
    }

    /// Set the gravity vector.
    pub fn with_gravity(mut self, gravity: glam::Vec3) -> Self {
        self.gravity = gravity;
        self
    }

    /// Add a body.
    pub fn add_body(&mut self, body: PhysicsBody) {
        self.bodies.push(body);
    }

    /// Get a mutable reference to the body for `node_id`, if it exists.
    pub fn body_mut(&mut self, node_id: NodeId) -> Option<&mut PhysicsBody> {
        self.bodies.iter_mut().find(|b| b.node_id == node_id)
    }
}

impl RuntimePlugin for PhysicsLitePlugin {
    fn priority(&self) -> i32 {
        phase::SIMULATE
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        for body in &mut self.bodies {
            let Some(node) = ctx.scene.node(body.node_id) else {
                continue;
            };
            let world = node.world_transform();
            let (scale, rotation, mut pos) = world.to_scale_rotation_translation();

            // Apply gravity.
            body.velocity += ctx.dt * body.gravity_scale * self.gravity;

            // Integrate position.
            pos += body.velocity * ctx.dt;

            // Reflect off bounding box faces.
            if let Some(ref bounds) = body.bounds {
                let mut normal = glam::Vec3::ZERO;
                let mut bounced = false;

                if pos.x < bounds.min.x && body.velocity.x < 0.0 {
                    pos.x = bounds.min.x;
                    body.velocity.x = -body.velocity.x * body.restitution;
                    normal = glam::Vec3::X;
                    bounced = true;
                } else if pos.x > bounds.max.x && body.velocity.x > 0.0 {
                    pos.x = bounds.max.x;
                    body.velocity.x = -body.velocity.x * body.restitution;
                    normal = -glam::Vec3::X;
                    bounced = true;
                }

                if pos.y < bounds.min.y && body.velocity.y < 0.0 {
                    pos.y = bounds.min.y;
                    body.velocity.y = -body.velocity.y * body.restitution;
                    normal = glam::Vec3::Y;
                    bounced = true;
                } else if pos.y > bounds.max.y && body.velocity.y > 0.0 {
                    pos.y = bounds.max.y;
                    body.velocity.y = -body.velocity.y * body.restitution;
                    normal = -glam::Vec3::Y;
                    bounced = true;
                }

                if pos.z < bounds.min.z && body.velocity.z < 0.0 {
                    pos.z = bounds.min.z;
                    body.velocity.z = -body.velocity.z * body.restitution;
                    normal = glam::Vec3::Z;
                    bounced = true;
                } else if pos.z > bounds.max.z && body.velocity.z > 0.0 {
                    pos.z = bounds.max.z;
                    body.velocity.z = -body.velocity.z * body.restitution;
                    normal = -glam::Vec3::Z;
                    bounced = true;
                }

                if bounced {
                    ctx.output.contact_events.push(ContactEvent {
                        node_a: body.node_id,
                        node_b: NodeId::MAX,
                        world_normal: normal,
                        impulse: body.velocity.length() * (1.0 + body.restitution),
                        contact_point: pos,
                    });
                }
            }

            let new_t =
                glam::Affine3A::from_scale_rotation_translation(scale, rotation, pos);
            ctx.writeback.set(body.node_id, new_t);
        }
    }
}
