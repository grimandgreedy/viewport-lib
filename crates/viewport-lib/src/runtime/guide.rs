//! Plugin authoring guide.
//!
//! This module documents how to write a [`super::RuntimePlugin`] and register it
//! with [`super::ViewportRuntime`].
//!
//! # The RuntimePlugin trait
//!
//! A plugin is any type that implements [`super::RuntimePlugin`]:
//!
//! ```rust,ignore
//! pub trait RuntimePlugin: Send + 'static {
//!     fn priority(&self) -> i32;
//!     fn submit(&mut self, ctx: &RuntimeStepContext<'_>) {}
//!     fn collect(&mut self, ctx: &mut RuntimeStepContext<'_>) {}
//!     fn on_event(&mut self, event: &RuntimeEvent, ctx: &mut RuntimeStepContext<'_>) {}
//!     fn step(&mut self, ctx: &mut RuntimeStepContext<'_>);
//! }
//! ```
//!
//! `step` is the only required method. The others have default no-op implementations.
//!
//! Hook call order each frame:
//!
//! 1. `on_event` -- called once per lifecycle event (node added or removed) before
//!    the step loop begins. Only fired when the scene's node set changes.
//! 2. `submit` -- called once per frame before the step loop, in priority order.
//!    Use this to kick off background work (e.g. spawning a compute task) that will
//!    be collected on the same or next frame. The context is `&RuntimeStepContext`
//!    (shared reference), so only `get` and `contains` on resources are available.
//! 3. `step` -- called once per phase execution, in priority order. For simulate-range
//!    plugins with a fixed timestep this is called once per accumulated step
//!    (possibly multiple times per frame). For all other plugins it is called once
//!    per frame. This is where the main per-frame or per-step work happens.
//! 4. `collect` -- called once per frame after the step loop, in priority order.
//!    Use this to read results from background work started in `submit`.
//!
//! # RuntimePhase ordering
//!
//! Phase constants are plain `i32` values in the [`super::plugin::phase`] module.
//! Plugins execute in ascending numeric order each frame:
//!
//! | Phase        | Value | When it runs                                       |
//! |--------------|-------|----------------------------------------------------|
//! | Prepare      | 100   | First. Update time-dependent state before queries. |
//! | Pick         | 200   | Ray-cast and object picking.                       |
//! | Select       | 300   | Selection updates driven by pick results.          |
//! | Manipulate   | 400   | Gizmo drag and keyboard transform sessions.        |
//! | Animate      | 500   | Procedural or keyframe animation.                  |
//! | Simulate     | 600   | Physics or simulation. Fixed timestep: runs N times per frame. |
//! | PostSimulate | 700   | After all Simulate iterations, before Writeback.   |
//! | Writeback    | 800   | Flush accumulated transform ops to the scene.      |
//!
//! You can use values between bands to place a plugin at a specific sub-position.
//! For example `phase::ANIMATE + 50` runs after standard Animate plugins but before
//! any plugin at `phase::SIMULATE`. Two plugins at the same priority run in
//! registration order (stable sort).
//!
//! The Simulate phase is special: with a fixed timestep configured, plugins in the
//! `[SIMULATE, POST_SIM)` range may be called zero or more times per rendered frame
//! depending on how much wall time accumulated since the last frame. All other phase
//! ranges always run exactly once per frame.
//!
//! # Hook registration
//!
//! Build the runtime and add plugins with `with_plugin`:
//!
//! ```rust,ignore
//! use viewport_lib::runtime::{ViewportRuntime, FixedTimestep};
//!
//! let mut runtime = ViewportRuntime::new()
//!     .with_fixed_timestep(FixedTimestep::new(60.0))
//!     .with_plugin(MyPhysicsPlugin::new())
//!     .with_plugin(MyAudioPlugin::new());
//! ```
//!
//! Multiple plugins can share the same priority. They run in registration order
//! within the same band. Registration order is preserved by a stable sort on
//! `priority()` at the start of each frame.
//!
//! Call `runtime.step(&mut scene, &mut selection, &frame_ctx)` once per frame to
//! drive all plugins. It returns a [`super::output::RuntimeOutput`] with selection
//! ops, contact events, transform ops, and any application-defined events.
//!
//! # Accessing scene data
//!
//! Each hook receives a [`super::context::RuntimeStepContext`] with the following fields:
//!
//! - `ctx.scene` -- read-only reference to the scene. Use this to query node
//!   transforms, materials, and spatial structure.
//! - `ctx.writeback` -- write transforms here. Call `ctx.writeback.set(node_id, affine)`.
//!   The runtime flushes accumulated writes to the scene after the writeback phase.
//! - `ctx.output` -- accumulate selection operations, contact events, camera commands,
//!   and typed application events here.
//! - `ctx.resources` -- shared typed resource registry. Use this to pass data between
//!   plugins within the same frame or across frames.
//! - `ctx.pick_hit` -- forwarded from `RuntimeFrameContext`. The pick result under the
//!   cursor for this frame, if picking was done by the caller.
//! - `ctx.dt` -- delta time for this step. For simulate-range plugins with a fixed
//!   timestep this is the fixed step size, not wall dt.
//!
//! # Resource sharing between plugins
//!
//! [`super::RuntimeResources`] is a typed map keyed by `TypeId`. Any `Send + 'static`
//! type can be stored as a resource. One value per type: inserting again overwrites.
//!
//! Typical pattern: one plugin writes, another reads in the same frame.
//!
//! ```rust,ignore
//! // PhysicsPlugin at SIMULATE inserts contact data.
//! fn step(&mut self, ctx: &mut RuntimeStepContext) {
//!     let contacts = self.solver.solve(ctx.scene, ctx.dt);
//!     ctx.resources.insert(contacts);
//! }
//!
//! // AudioPlugin at POST_SIM reads it.
//! fn step(&mut self, ctx: &mut RuntimeStepContext) {
//!     if let Some(contacts) = ctx.resources.get::<ContactList>() {
//!         for c in contacts.iter() {
//!             self.audio.play_impact(c.impulse);
//!         }
//!     }
//! }
//! ```
//!
//! Resources persist across frames until explicitly removed or the runtime is dropped.
//! Use `runtime.resources_mut().insert(...)` before the first `step` to pre-populate
//! resources that plugins expect to find on frame one.
//!
//! Available methods on `RuntimeResources`:
//! - `insert(value)` -- store a value, replacing any existing value of that type
//! - `get::<T>()` -- shared reference, returns `None` if not present
//! - `get_mut::<T>()` -- mutable reference
//! - `remove::<T>()` -- take ownership and remove
//! - `contains::<T>()` -- presence check without borrowing the value
//!
//! In `submit` (called with `&RuntimeStepContext`), only `get` and `contains` are
//! accessible. In `step`, `collect`, and `on_event` (called with `&mut RuntimeStepContext`),
//! full insert/get_mut/remove access is available.
//!
//! # Debug visualization with DebugDraw
//!
//! [`super::DebugDraw`] is a resource, not a renderer type. Plugins draw to it by
//! calling `ctx.resources.get_mut::<DebugDraw>()` in their `step` implementations.
//! After the frame, the host converts the accumulated primitives into render items.
//!
//! Setup pattern:
//!
//! ```rust,ignore
//! use viewport_lib::runtime::debug_draw::DebugDraw;
//!
//! // At startup: insert a DebugDraw resource.
//! runtime.resources_mut().insert(DebugDraw::new());
//!
//! // Each frame, before runtime.step():
//! if let Some(dd) = runtime.resources_mut().get_mut::<DebugDraw>() {
//!     dd.begin_frame();  // clears transient (one-frame) primitives
//! }
//! let output = runtime.step(&mut scene, &mut selection, &frame_ctx);
//!
//! // After step: convert accumulated primitives to render items.
//! if let Some(dd) = runtime.resources().get::<DebugDraw>() {
//!     frame_data.scene.polylines.extend(dd.to_polylines());
//!     if let Some(pc) = dd.to_point_cloud() {
//!         frame_data.scene.point_clouds.push(pc);
//!     }
//! }
//! ```
//!
//! Inside a plugin's `step`:
//!
//! ```rust,ignore
//! fn step(&mut self, ctx: &mut RuntimeStepContext) {
//!     if let Some(dd) = ctx.resources.get_mut::<DebugDraw>() {
//!         dd.line(contact.point_a, contact.point_b, [1.0, 0.0, 0.0, 1.0]);
//!     }
//! }
//! ```
//!
//! Primitives submitted with `line`, `point`, `aabb`, `sphere`, and `label` are
//! transient: `begin_frame` clears them. Persistent primitives are added with
//! `add_persistent(key, prim)` keyed by a `u64` and stay until `remove_persistent(key)`
//! is called. See the [`super::debug_draw`] module for the full API.
//!
//! # Fixed timestep and the Simulate phase
//!
//! Configure a fixed timestep with `ViewportRuntime::with_fixed_timestep`:
//!
//! ```rust,ignore
//! let mut runtime = ViewportRuntime::new()
//!     .with_fixed_timestep(FixedTimestep::new(60.0));  // 60 Hz
//! ```
//!
//! The runtime accumulates wall time. Each call to `runtime.step(...)` advances the
//! accumulator by `frame_ctx.dt`. If the accumulated time exceeds the step size, the
//! runtime calls `step()` on simulate-range plugins once per step. If wall dt is less
//! than the step size, `step()` may not be called at all for that frame.
//!
//! Inside simulate-range plugins, `ctx.dt` is the fixed step size (e.g. `1.0/60.0`),
//! not wall dt. This makes physics integration deterministic regardless of frame rate.
//!
//! For rendering, use `runtime.alpha()` to interpolate between the previous and current
//! simulation state:
//!
//! ```rust,ignore
//! let alpha = runtime.alpha();
//! if let Some(t) = runtime.snapshots().interpolated(node_id, alpha) {
//!     // use t as the render transform instead of the scene node transform
//! }
//! ```
//!
//! `alpha` is in `[0.0, 1.0]`: the fractional position between the last completed step
//! and the next. This eliminates the jitter that appears when frame rate and step rate
//! are out of phase. When no fixed timestep is configured, `alpha()` returns `1.0`.
//!
//! # File layout for new plugins
//!
//! New plugins live in a subdirectory under `src/runtime/plugins/`, not as a single
//! file. The [`super::plugins::skeleton_plugin`] module is the reference template:
//!
//! ```text
//! plugins/
//!   my_plugin/
//!     mod.rs        -- re-exports the public surface
//!     plugin.rs     -- the RuntimePlugin impl
//!     <feature>.rs  -- substrate types, math, helpers, internal state
//! ```
//!
//! Using a subdirectory from the start avoids a rename-and-rewire when the plugin
//! grows. Re-export all public types from `mod.rs` and add the module to
//! `src/runtime/plugins/mod.rs`.
//!
//! # Minimal plugin example
//!
//! A plugin that counts frames and logs every 60 steps:
//!
//! ```rust,ignore
//! use viewport_lib::runtime::{RuntimePlugin, RuntimeStepContext, ViewportRuntime};
//! use viewport_lib::runtime::plugin::phase;
//!
//! struct FrameCounter {
//!     count: u64,
//! }
//!
//! impl FrameCounter {
//!     fn new() -> Self {
//!         Self { count: 0 }
//!     }
//! }
//!
//! impl RuntimePlugin for FrameCounter {
//!     fn priority(&self) -> i32 {
//!         phase::PREPARE
//!     }
//!
//!     fn step(&mut self, _ctx: &mut RuntimeStepContext<'_>) {
//!         self.count += 1;
//!         if self.count % 60 == 0 {
//!             println!("frame {}", self.count);
//!         }
//!     }
//! }
//!
//! // Registration:
//! let mut runtime = ViewportRuntime::new()
//!     .with_plugin(FrameCounter::new());
//! ```
//!
//! This plugin runs once per frame at the Prepare phase. It owns its state as a
//! struct field. To share state with another plugin, insert it into `ctx.resources`
//! instead of keeping it in the struct.
