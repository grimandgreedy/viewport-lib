//! Transmission: the copy of the opaque scene that transmissive surfaces read,
//! and the mesh pipelines that draw them.
//!
//! A transmissive mesh draws after the opaque scene, through `mesh.wgsl`'s
//! `fs_main_transmission` entries with the copy bound at group 3. The copy is
//! the opaque HDR colour with a mip chain, so a rough surface reads a blurred
//! level, and the scene depth in mip 0's alpha. Group 2 is left empty: a
//! material plugin's shading does not transmit.

use crate::renderer::pipeline_key::FeatureSet;
use crate::resources::pipeline_slot::{FeatureGet, LazyFamily};

/// What a transmission mesh pipeline build reads.
pub(crate) struct TransmissionMeshContext {
    device: crate::gpu::Device,
    layout: crate::gpu::PipelineLayout,
    shader: crate::resources::pipeline_slot::LazyModule,
    depth_format: crate::gpu::TextureFormat,
}

fn build_mesh(ctx: &TransmissionMeshContext, i: usize) -> crate::gpu::RenderPipeline {
    build_mesh_features(ctx, i, FeatureSet::UNIVERSAL)
}

fn build_mesh_features(
    ctx: &TransmissionMeshContext,
    i: usize,
    feats: FeatureSet,
) -> crate::gpu::RenderPipeline {
    let two_sided = i & 1 != 0;
    let base = if two_sided {
        "hdr_transmission_two_sided_pipeline"
    } else {
        "hdr_transmission_pipeline"
    };
    let label = if feats == FeatureSet::UNIVERSAL {
        base.to_owned()
    } else {
        feats.label(base)
    };
    crate::resources::mesh::mesh_pipelines::hdr_transmission_pipeline(
        &ctx.device,
        &ctx.layout,
        ctx.shader.get(),
        ctx.depth_format,
        &label,
        two_sided,
        &feats.constants(),
    )
}

/// Device-shared transmission pipelines and layouts, built by the first frame
/// that draws a transmissive mesh.
#[derive(Default)]
pub(crate) struct TransmissionResources {
    /// The mesh family, keyed by facedness, and its twin on the bindless
    /// member of group 1.
    pipeline: Option<LazyFamily<TransmissionMeshContext, 2>>,
    pipeline_bindless: Option<LazyFamily<TransmissionMeshContext, 2>>,
    /// Copies the scene colour and depth into mip 0.
    copy: Option<super::LazyFullscreen>,
    /// Builds each further mip from the one above.
    downsample: Option<super::LazyFullscreen>,
    copy_bgl: Option<crate::gpu::BindGroupLayout>,
    downsample_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Group 3 of the mesh pipelines: the copy and its sampler.
    scene_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Group 2 of the mesh pipelines, and the bind group set there.
    empty_bgl: Option<crate::gpu::BindGroupLayout>,
    empty_bind_group: Option<crate::gpu::BindGroup>,
    linear_sampler: Option<crate::gpu::Sampler>,
    nearest_sampler: Option<crate::gpu::Sampler>,
}

/// One viewport's transmission target: the scene copy with its mip chain and
/// the bind groups that fill and read it. Sized to the viewport's scene.
pub(crate) struct TransmissionTarget {
    size: [u32; 2],
    /// One single-level view per mip, for the copy and downsample passes.
    mip_views: Vec<crate::gpu::TextureView>,
    /// `downsample_groups[i]` reads mip `i` to build mip `i + 1`.
    downsample_groups: Vec<crate::gpu::BindGroup>,
    /// The group-3 bind group the mesh pipelines read.
    scene_group: crate::gpu::BindGroup,
    /// The copy pass's group over the scene colour and depth views.
    copy_group: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::TextureView,
        crate::gpu::TextureView,
    )>,
}

/// Mip levels of a transmission target of `w` x `h`: one per halving of the
/// shorter side, down to a single texel.
pub(crate) fn transmission_mip_count(w: u32, h: u32) -> u32 {
    w.min(h).max(1).ilog2() + 1
}

impl TransmissionTarget {
    /// The group-3 bind group the transmission mesh pipelines read.
    pub(crate) fn scene_group(&self) -> &crate::gpu::BindGroup {
        &self.scene_group
    }
}

impl TransmissionResources {
    /// The mesh pipeline for `two_sided` built with `feats`, with the
    /// universal one standing in while it compiles. `bindless` selects the
    /// family on the bindless member of group 1.
    pub(crate) fn mesh_features(
        &self,
        two_sided: bool,
        feats: FeatureSet,
        bindless: bool,
    ) -> FeatureGet<'_, crate::gpu::RenderPipeline> {
        let family = if bindless {
            self.pipeline_bindless.as_ref()
        } else {
            self.pipeline.as_ref()
        };
        match family {
            Some(f) => f.get_features(two_sided as usize, feats),
            None => FeatureGet {
                pipeline: None,
                stand_in: false,
                capped: false,
            },
        }
    }

    /// Whether the mesh pipeline for `two_sided` and `feats` is built,
    /// without starting it.
    pub(crate) fn mesh_ready(&self, two_sided: bool, feats: FeatureSet, bindless: bool) -> bool {
        let family = if bindless {
            self.pipeline_bindless.as_ref()
        } else {
            self.pipeline.as_ref()
        };
        family.is_some_and(|f| f.features_ready(two_sided as usize, feats))
    }

    /// Feature-set pipelines asked of the mesh families.
    pub(crate) fn feature_sets(&self) -> usize {
        [&self.pipeline, &self.pipeline_bindless]
            .into_iter()
            .flatten()
            .map(|f| f.feature_sets())
            .sum()
    }

    /// The bind group the mesh pipelines take at group 2.
    pub(crate) fn empty_group(&self) -> Option<&crate::gpu::BindGroup> {
        self.empty_bind_group.as_ref()
    }

    /// The copy and downsample pipelines, when both are built.
    pub(crate) fn mip_pipelines(
        &self,
    ) -> Option<(&crate::gpu::RenderPipeline, &crate::gpu::RenderPipeline)> {
        Some((
            self.copy.as_ref()?.get(0)?,
            self.downsample.as_ref()?.get(0)?,
        ))
    }

    /// Fill `target`'s mip 0 from the scene colour and depth, then each
    /// further mip from the one above. Returns whether it encoded.
    pub(crate) fn encode_mips(
        &self,
        device: &crate::gpu::Device,
        encoder: &mut crate::gpu::CommandEncoder,
        target: &mut TransmissionTarget,
        scene_colour: &crate::gpu::TextureView,
        scene_depth: &crate::gpu::TextureView,
    ) -> bool {
        let (Some((copy, downsample)), Some(bgl), Some(nearest)) = (
            self.mip_pipelines(),
            self.copy_bgl.as_ref(),
            self.nearest_sampler.as_ref(),
        ) else {
            return false;
        };
        let copy_group = target
            .copy_group
            .get((scene_colour.clone(), scene_depth.clone()), || {
                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("transmission_copy_bind_group"),
                    layout: bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: crate::gpu::BindingResource::TextureView(scene_colour),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::TextureView(scene_depth),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 2,
                            resource: crate::gpu::BindingResource::Sampler(nearest),
                        },
                    ],
                })
            });
        let pass = |encoder: &mut crate::gpu::CommandEncoder,
                    label: &str,
                    view: &crate::gpu::TextureView,
                    pipeline: &crate::gpu::RenderPipeline,
                    group: &crate::gpu::BindGroup| {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some(label),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        };
        pass(
            encoder,
            "transmission_copy_pass",
            &target.mip_views[0],
            copy,
            &copy_group,
        );
        for (i, group) in target.downsample_groups.iter().enumerate() {
            pass(
                encoder,
                "transmission_downsample_pass",
                &target.mip_views[i + 1],
                downsample,
                group,
            );
        }
        true
    }

    /// Build or resize a viewport's target for a scene of `w` x `h`. Needs
    /// [`DeviceResources::ensure_transmission`](crate::resources::DeviceResources::ensure_transmission).
    pub(crate) fn ensure_target(
        &self,
        device: &crate::gpu::Device,
        target: &mut Option<TransmissionTarget>,
        w: u32,
        h: u32,
    ) {
        let size = [w.max(1), h.max(1)];
        if target.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let (Some(downsample_bgl), Some(scene_bgl), Some(linear)) = (
            self.downsample_bgl.as_ref(),
            self.scene_bgl.as_ref(),
            self.linear_sampler.as_ref(),
        ) else {
            return;
        };
        let mips = transmission_mip_count(size[0], size[1]);
        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("transmission_texture"),
            size: crate::gpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Rgba16Float,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mip_views: Vec<crate::gpu::TextureView> = (0..mips)
            .map(|level| {
                texture.create_view(&crate::gpu::TextureViewDescriptor {
                    label: Some("transmission_mip_view"),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let texture_group =
            |label: &str, layout: &crate::gpu::BindGroupLayout, view: &crate::gpu::TextureView| {
                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: crate::gpu::BindingResource::TextureView(view),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::Sampler(linear),
                        },
                    ],
                })
            };
        let downsample_groups = mip_views[..mip_views.len() - 1]
            .iter()
            .map(|view| texture_group("transmission_downsample_bind_group", downsample_bgl, view))
            .collect();
        let full_view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        let scene_group = texture_group("transmission_scene_bind_group", scene_bgl, &full_view);
        *target = Some(TransmissionTarget {
            size,
            mip_views,
            downsample_groups,
            scene_group,
            copy_group: crate::resources::cached_bind_group::CachedBindGroup::new(),
        });
    }
}

impl crate::resources::DeviceResources {
    /// Whether the device can draw transmissive meshes in their own pass: the
    /// mesh pipelines bind four groups. Without it they draw with the opaque
    /// scene, lit by the environment behind them.
    pub(crate) fn transmission_supported(&self) -> bool {
        self.capabilities.bind_groups >= 4
    }

    /// Compose the transmission pipelines and layouts. No-op after the first
    /// call. The pipelines are built as draws select them.
    pub(crate) fn ensure_transmission(&mut self, device: &crate::gpu::Device) {
        if self.transmission.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        use crate::gpu::ShaderStages;
        use crate::resources::builders as b;
        let fragment = ShaderStages::FRAGMENT;
        let copy_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("transmission_copy_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: fragment,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: false },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                b::depth_texel_entry(1, fragment),
                b::nearest_sampler_entry(2, fragment),
            ],
        });
        let downsample_bgl =
            b::texture_sampler_bgl(device, "transmission_downsample_bgl", fragment);
        let scene_bgl = b::texture_sampler_bgl(device, "transmission_scene_bgl", fragment);
        let empty_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("transmission_empty_bgl"),
            entries: &[],
        });
        let empty_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("transmission_empty_bind_group"),
            layout: &empty_bgl,
            entries: &[],
        });

        let copy_shader = self.shared_module(
            device,
            "transmission_copy_shader",
            b::wgsl_source!("transmission_copy"),
        );
        let downsample_shader = self.shared_module(
            device,
            "transmission_downsample_shader",
            b::wgsl_source!("transmission_downsample"),
        );
        let copy = self.fullscreen_pass_pipeline(
            device,
            "transmission_copy",
            copy_shader,
            &copy_bgl,
            crate::gpu::TextureFormat::Rgba16Float,
        );
        let downsample = self.fullscreen_pass_pipeline(
            device,
            "transmission_downsample",
            downsample_shader,
            &downsample_bgl,
            crate::gpu::TextureFormat::Rgba16Float,
        );

        // The HDR mesh source, so the module is the one the scene family
        // already compiles.
        let source = self.hdr_mesh_source();
        let shader = self.shared_module(device, "mesh_shader_hdr", &source);
        let layout = b::pipeline_layout(
            device,
            "hdr_transmission_pipeline_layout",
            &[
                &self.scene_tier.binds.camera_bgl,
                &self.scene_tier.binds.object_bgl,
                &empty_bgl,
                &scene_bgl,
            ],
        );
        let bindless = if self.bindless_textures() {
            self.ensure_object_bindless_bgl(device);
            let source = crate::resources::mesh::instanced_bindless::bindlessify_mesh(&source);
            let shader = self.shared_module(device, "mesh_bindless_shader_hdr", &source);
            let layout = b::pipeline_layout(
                device,
                "hdr_transmission_bindless_pipeline_layout",
                &[
                    &self.scene_tier.binds.camera_bgl,
                    self.scene_tier.binds.object_bindless_bgl.as_ref().unwrap(),
                    &empty_bgl,
                    &scene_bgl,
                ],
            );
            Some((shader, layout))
        } else {
            None
        };
        let family = |layout, shader| {
            LazyFamily::new(
                TransmissionMeshContext {
                    device: device.clone(),
                    layout,
                    shader,
                    depth_format: self.scene_depth_format(),
                },
                std::sync::Arc::clone(&self.pipeline_compiler),
                build_mesh,
            )
            .with_features(build_mesh_features)
        };
        let pipeline = family(layout, shader);
        let pipeline_bindless = bindless.map(|(s, l)| family(l, s));

        let t = &mut self.transmission;
        t.pipeline = Some(pipeline);
        t.pipeline_bindless = pipeline_bindless;
        t.copy = Some(copy);
        t.downsample = Some(downsample);
        t.copy_bgl = Some(copy_bgl);
        t.downsample_bgl = Some(downsample_bgl);
        t.scene_bgl = Some(scene_bgl);
        t.empty_bgl = Some(empty_bgl);
        t.empty_bind_group = Some(empty_bind_group);
        t.linear_sampler = Some(b::clamp_linear_mip_sampler(device, "transmission_sampler"));
        t.nearest_sampler = Some(b::clamp_nearest_sampler(
            device,
            "transmission_depth_sampler",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::transmission_mip_count;

    #[test]
    fn mip_count_halves_the_shorter_side_to_one_texel() {
        assert_eq!(transmission_mip_count(1, 1), 1);
        assert_eq!(transmission_mip_count(64, 64), 7);
        assert_eq!(transmission_mip_count(1920, 1080), 11);
        assert_eq!(transmission_mip_count(400, 300), 9);
    }
}
