//! External post-effect implementations for `viewport-lib`.
//!
//! This crate exercises the post-effect plugin surface
//! (`viewport_lib::plugin_api::post_effect`) from outside the library, the
//! way a consumer crate would:
//!
//! - [`ContactShadowEffect`] and [`BloomEffect`] are faithful copies of the
//!   built-in contact shadows and bloom, implemented as
//!   [`PostEffectProducer`]s. With the matching built-in switched off in
//!   `PostProcessSettings` they render pixel-identically to it (the parity
//!   tests in `tests/parity.rs` hold them to that), which makes them a
//!   worked example of replacing a built-in effect's implementation.
//! - [`vfx`] is a three-stage [`PostEffectStage`] stack (colour grade,
//!   depth fog, edge detect) ported from the `viewport-lib-vfx` kit, driven
//!   through one shared settings handle.
//!
//! Every effect here owns its GPU resources: pipelines are built in
//! `init_gpu`, per-viewport textures in `on_viewport_resized`, uniforms are
//! written in `prepare`, and passes are encoded in `encode`. The host only
//! registers the effect and keeps the settings handle.
//!
//! [`PostEffectProducer`]: viewport_lib::PostEffectProducer
//! [`PostEffectStage`]: viewport_lib::PostEffectStage

use viewport_lib::wgpu;

pub mod bloom;
pub mod contact_shadow;
pub mod vfx;

pub use bloom::{BloomEffect, BloomEffectSettings};
pub use contact_shadow::{ContactShadowEffect, ContactShadowEffectSettings};
pub use vfx::{ColourGrade, DepthFog, EdgeDetect, VfxSettings, vfx_stack};

/// Shared settings handle: the renderer owns the boxed effect, the host
/// keeps this and mutates it between frames.
pub type SettingsHandle<T> = std::sync::Arc<std::sync::Mutex<T>>;

/// Encode one fullscreen-triangle pass: clear the target, bind one group,
/// draw three vertices.
pub(crate) fn fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    clear: wgpu::Color,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
        multiview_mask: None,
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Build a fullscreen post pipeline against a raw device: the same shape as
/// `DeviceResources::build_post_effect_pipeline`, usable from `init_gpu`
/// where only the device is available.
pub(crate) fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    bgl: &wgpu::BindGroupLayout,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    // 27 takes `push_constant_ranges` and `&[&BindGroupLayout]`; 29 replaced
    // push constants with `immediate_size` and takes
    // `&[Option<&BindGroupLayout>]`, which 30 keeps.
    #[cfg(feature = "wgpu27")]
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        // 29 renamed `multiview` to the `multiview_mask` bitmask form.
        #[cfg(feature = "wgpu27")]
        multiview: None,
        #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
        multiview_mask: None,
        cache: None,
    })
}

/// Create a colour render target with a sampleable view.
pub(crate) fn colour_target(
    device: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Clamp-to-edge sampler with the given filter mode.
pub(crate) fn clamp_sampler(
    device: &wgpu::Device,
    label: &str,
    filter: wgpu::FilterMode,
) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: filter,
        min_filter: filter,
        // 27 reuses `FilterMode` for the mip filter; 29 and 30 take the
        // split `MipmapFilterMode`.
        #[cfg(feature = "wgpu27")]
        mipmap_filter: wgpu::FilterMode::Nearest,
        #[cfg(any(feature = "wgpu29", feature = "wgpu30"))]
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}
