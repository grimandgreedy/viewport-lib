//! Fixtures for [`PostEffectProducer`](viewport_lib::PostEffectProducer): a
//! pre-tone-map pass that fills one composite slot, registered with
//! `ViewportRenderer::add_post_effect_producer`.
//!
//! - [`LoggingPostEffectProducer`]: records the whole lifecycle and can fill
//!   its slot with a flat value.

mod logging;

pub use logging::LoggingPostEffectProducer;
