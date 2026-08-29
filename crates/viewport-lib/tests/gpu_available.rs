//! CI canary for the headless suite.
//!
//! Every headless_*.rs test skips (and so reports green) when no wgpu adapter is
//! available. That is what you want on a developer machine without a GPU, but in
//! CI it means the whole suite can pass while testing nothing. Set
//! `VIEWPORT_REQUIRE_GPU=1` in the CI environment: this test then fails loudly if
//! no adapter can be created, instead of letting the silent skips hide it. With
//! the variable unset the test is a no-op.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

#[test]
fn adapter_present_when_required() {
    if std::env::var_os("VIEWPORT_REQUIRE_GPU").is_none() {
        return;
    }
    assert!(
        headless_device().is_some(),
        "VIEWPORT_REQUIRE_GPU is set but no wgpu adapter could be created; the \
         headless tests would all silently skip. Check the CI GPU/software \
         adapter setup."
    );
}
