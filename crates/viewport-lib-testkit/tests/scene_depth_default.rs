//! The scene depth format defaults to `Depth24PlusStencil8` on every device,
//! so a host painting into a framework's 24-bit pass matches without
//! configuration, and the harness (which owns its depth target) asks for the
//! 32-bit float depth when the device has it.

use viewport_lib::{RendererConfig, ViewportRenderer, wgpu};
use viewport_lib_testkit::Harness;

#[test]
fn scene_depth_defaults_to_24_bit_and_the_harness_opts_into_float() {
    let Some(harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let has = harness
        .device
        .features()
        .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8);

    let default = ViewportRenderer::try_with_config(
        &harness.device,
        &RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT),
    )
    .expect("renderer");
    assert_eq!(
        default.scene_depth_format(),
        wgpu::TextureFormat::Depth24PlusStencil8
    );

    let expected = if has {
        wgpu::TextureFormat::Depth32FloatStencil8
    } else {
        wgpu::TextureFormat::Depth24PlusStencil8
    };
    eprintln!("harness scene depth: {expected:?}");
    assert_eq!(harness.renderer.scene_depth_format(), expected);
}
