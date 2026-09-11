//! Contact shadows as an external [`PostEffectProducer`]: a faithful copy
//! of the built-in screen-space contact-shadow march, filling the
//! [`PostEffectSlot::ContactShadow`] composite slot.
//!
//! The shader, uniform layout, target format, and march parameters match
//! the built-in exactly; with `PostProcessSettings.contact_shadows.enabled
//! = false` and this producer registered, output is pixel-identical to the
//! built-in. The one input the post-effect context does not carry is the
//! light: the built-in derives its march direction from the frame's first
//! light, so the host sets [`ContactShadowEffectSettings::light_direction`]
//! itself (it owns the light).
//!
//! [`PostEffectProducer`]: viewport_lib::PostEffectProducer
//! [`PostEffectSlot::ContactShadow`]: viewport_lib::PostEffectSlot::ContactShadow

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use viewport_lib::wgpu;
use viewport_lib::{
    PostEffectContext, PostEffectProducer, PostEffectResizeContext, PostEffectSlot,
};

use crate::{SettingsHandle, clamp_sampler, colour_target, fullscreen_pass, fullscreen_pipeline};

/// Host-driven settings, matching the built-in
/// `ContactShadowSettings` fields plus the light direction the context
/// cannot supply.
#[derive(Clone, Copy, Debug)]
pub struct ContactShadowEffectSettings {
    pub enabled: bool,
    /// Maximum march distance in world units.
    pub max_distance: f32,
    /// Ray-march step count.
    pub steps: u32,
    /// Occluder thickness assumption in world units.
    pub thickness: f32,
    /// Surface-to-light march direction in world space. For a directional
    /// light this matches the built-in's use of `LightKind::Directional`'s
    /// `direction` field as-is; for a spot light the built-in negates the
    /// shining direction, so pass the negated vector.
    pub light_direction: [f32; 3],
}

impl Default for ContactShadowEffectSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_distance: 0.5,
            steps: 16,
            thickness: 0.3,
            light_direction: [0.0, -1.0, 0.0],
        }
    }
}

/// Matches the built-in `ContactShadowUniform` layout (176 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ContactShadowUniform {
    inv_proj: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    light_dir_view: [f32; 4],
    world_up_view: [f32; 4],
    params: [f32; 4],
}

struct CsViewport {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    /// Rebuilt each `prepare`: it binds the frame's scene depth view, which
    /// the resize signal does not carry.
    bind_group: Option<wgpu::BindGroup>,
}

/// The external contact-shadow producer. Construct with
/// [`ContactShadowEffect::new`], register the effect, keep the handle.
pub struct ContactShadowEffect {
    settings: SettingsHandle<ContactShadowEffectSettings>,
    device: Option<wgpu::Device>,
    pipeline: Option<wgpu::RenderPipeline>,
    bgl: Option<wgpu::BindGroupLayout>,
    sampler: Option<wgpu::Sampler>,
    per_viewport: HashMap<usize, CsViewport>,
}

impl ContactShadowEffect {
    pub fn new(
        settings: ContactShadowEffectSettings,
    ) -> (Self, SettingsHandle<ContactShadowEffectSettings>) {
        let handle = Arc::new(Mutex::new(settings));
        (
            Self {
                settings: handle.clone(),
                device: None,
                pipeline: None,
                bgl: None,
                sampler: None,
                per_viewport: HashMap::new(),
            },
            handle,
        )
    }
}

impl PostEffectProducer for ContactShadowEffect {
    fn type_name(&self) -> &'static str {
        "external_contact_shadow"
    }

    fn slot(&self) -> PostEffectSlot {
        PostEffectSlot::ContactShadow
    }

    fn enabled(&self) -> bool {
        self.settings.lock().unwrap().enabled
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        // Depth texture + non-filtering sampler + uniform: the built-in
        // contact-shadow layout.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("external_cs_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        self.pipeline = Some(fullscreen_pipeline(
            device,
            "external_cs_pipeline",
            include_str!("shaders/contact_shadow.wgsl"),
            &bgl,
            wgpu::TextureFormat::R8Unorm,
        ));
        self.sampler = Some(clamp_sampler(
            device,
            "external_cs_sampler",
            wgpu::FilterMode::Nearest,
        ));
        self.bgl = Some(bgl);
        self.device = Some(device.clone());
        self.per_viewport.clear();
    }

    fn on_viewport_resized(&mut self, device: &wgpu::Device, ctx: &PostEffectResizeContext<'_>) {
        let (texture, view) = colour_target(
            device,
            "external_cs_texture",
            ctx.scene_size,
            wgpu::TextureFormat::R8Unorm,
        );
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("external_cs_uniform"),
            size: std::mem::size_of::<ContactShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.per_viewport.insert(
            ctx.viewport_index,
            CsViewport {
                _texture: texture,
                view,
                uniform_buf,
                bind_group: None,
            },
        );
    }

    fn prepare(&mut self, queue: &wgpu::Queue, ctx: &PostEffectContext<'_>) {
        let (Some(device), Some(bgl), Some(sampler)) = (&self.device, &self.bgl, &self.sampler)
        else {
            return;
        };
        let Some(vp) = self.per_viewport.get_mut(&ctx.viewport_index) else {
            return;
        };
        let settings = *self.settings.lock().unwrap();

        // Same derivation as the built-in producer's upload.
        let light_dir_world = glam::Vec3::from(settings.light_direction).normalize();
        let light_dir_view = ctx.view.transform_vector3(light_dir_world).normalize();
        let world_up_view = ctx.view.transform_vector3(glam::Vec3::Z).normalize();
        let uniform = ContactShadowUniform {
            inv_proj: ctx.proj.inverse().to_cols_array_2d(),
            proj: ctx.proj.to_cols_array_2d(),
            light_dir_view: [light_dir_view.x, light_dir_view.y, light_dir_view.z, 0.0],
            world_up_view: [world_up_view.x, world_up_view.y, world_up_view.z, 0.0],
            params: [
                settings.max_distance,
                settings.steps as f32,
                settings.thickness,
                0.0,
            ],
        };
        queue.write_buffer(&vp.uniform_buf, 0, bytemuck::cast_slice(&[uniform]));

        // The scene depth view can change when the viewport's targets are
        // recreated, so bind it fresh each frame.
        vp.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("external_cs_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ctx.scene_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vp.uniform_buf.as_entire_binding(),
                },
            ],
        }));
    }

    fn encode<'a>(
        &'a mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PostEffectContext<'_>,
    ) -> Option<&'a wgpu::TextureView> {
        let pipeline = self.pipeline.as_ref()?;
        let vp = self.per_viewport.get(&ctx.viewport_index)?;
        let bind_group = vp.bind_group.as_ref()?;
        fullscreen_pass(
            encoder,
            "external_cs_pass",
            &vp.view,
            wgpu::Color::WHITE,
            pipeline,
            bind_group,
        );
        Some(&vp.view)
    }
}
