//! The scene depth is 32-bit float when the device has
//! `DEPTH32FLOAT_STENCIL8` and nothing pins it, and an explicit
//! `Depth24PlusStencil8` stays as asked.

use viewport_lib::{RendererConfig, ViewportRenderer, wgpu};
use viewport_lib_testkit::Harness;

#[test]
fn scene_depth_defaults_to_32_bit_float_when_the_device_has_it() {
    let Some(harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let has = harness
        .device
        .features()
        .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8);
    let expected = if has {
        wgpu::TextureFormat::Depth32FloatStencil8
    } else {
        wgpu::TextureFormat::Depth24PlusStencil8
    };
    eprintln!("scene depth: {expected:?}");
    assert_eq!(harness.renderer.scene_depth_format(), expected);

    let pinned = ViewportRenderer::try_with_config(
        &harness.device,
        &RendererConfig::new(Harness::DEFAULT_TARGET_FORMAT)
            .with_scene_depth_format(wgpu::TextureFormat::Depth24PlusStencil8),
    )
    .expect("renderer");
    assert_eq!(
        pinned.scene_depth_format(),
        wgpu::TextureFormat::Depth24PlusStencil8
    );
}
