//! Fixtures for [`PostEffectStage`](viewport_lib::PostEffectStage): a
//! display-space pass chained after the tone-map composite, registered with
//! `ViewportRenderer::add_post_effect_stage` at an explicit order key.
//!
//! - [`PassthroughPostEffectStage`]: copies its input to its target, scaled by
//!   a constant, so a test can see which stages ran and in what order.

mod passthrough;

pub use passthrough::PassthroughPostEffectStage;
