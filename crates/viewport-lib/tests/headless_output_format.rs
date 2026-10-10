//! A frame drawn into a target whose format the renderer was not created for
//! fails at the call, naming both formats, instead of in wgpu validation
//! inside whichever pass binds a pipeline first.

use viewport_lib::wgpu;

mod common;
use common::*;

fn target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    view_formats: &[wgpu::TextureFormat],
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output_format_target"),
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats,
    })
}

fn frame() -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [32.0, 32.0];
    frame
}

#[test]
#[should_panic(expected = "this renderer was created for Bgra8Unorm")]
fn an_srgb_target_for_a_linear_renderer_panics_with_both_formats() {
    let Some((device, queue)) = headless_device() else {
        // No adapter: nothing to check, but the expected panic must still occur.
        panic!("this renderer was created for Bgra8Unorm (no adapter, skipped)");
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8Unorm);
    let texture = target(&device, wgpu::TextureFormat::Bgra8UnormSrgb, &[]);
    let view = texture.create_view(&Default::default());
    renderer.render_to_texture(&device, &queue, &view, &frame());
}

#[test]
fn a_linear_texture_viewed_as_the_renderers_srgb_format_renders() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    renderer.set_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
    let texture = target(
        &device,
        wgpu::TextureFormat::Bgra8Unorm,
        &[wgpu::TextureFormat::Bgra8UnormSrgb],
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        format: Some(wgpu::TextureFormat::Bgra8UnormSrgb),
        ..Default::default()
    });
    renderer.render_to_texture(&device, &queue, &view, &frame());
}
