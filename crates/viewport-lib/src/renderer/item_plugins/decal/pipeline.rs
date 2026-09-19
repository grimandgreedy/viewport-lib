//! Screen-space decal pipeline: the projection pass, its normal mapping, and
//! the stencil pass that marks surfaces the decals must not land on.

use crate::gpu::util::DeviceExt as _;
use crate::resources::DeviceResources;
use crate::resources::mesh::mesh_store::MeshId;

// ---------------------------------------------------------------------------
// GPU-internal types
// ---------------------------------------------------------------------------

/// Screen-space scissor for one decal's fullscreen quad.
pub(crate) enum DecalScissor {
    /// The decal projects entirely off screen: skip the draw.
    Skip,
    /// A box corner is at or behind the near plane (camera inside/straddling
    /// the decal), so the screen bound is unreliable: use the full framebuffer.
    Full,
    /// Tight scissor rect (x, y, w, h) in framebuffer pixels.
    Rect(u32, u32, u32, u32),
}

/// Project the decal's unit-cube corners and return a scissor rect bounding
/// their screen extent. The decal fragment shader still runs a fullscreen quad,
/// but the scissor confines rasterization to the decal's actual footprint,
/// removing the per-decal fullscreen overdraw. `vp_w`/`vp_h` are the decal-pass
/// target dimensions.
pub(crate) fn decal_scissor(
    model: &glam::Mat4,
    view_proj: &glam::Mat4,
    vp_w: u32,
    vp_h: u32,
) -> DecalScissor {
    let mvp = *view_proj * *model;
    let mut min = glam::Vec2::splat(f32::MAX);
    let mut max = glam::Vec2::splat(f32::MIN);
    for cz in [-0.5f32, 0.5] {
        for cy in [-0.5f32, 0.5] {
            for cx in [-0.5f32, 0.5] {
                let clip = mvp * glam::Vec4::new(cx, cy, cz, 1.0);
                if clip.w <= 1e-4 {
                    return DecalScissor::Full;
                }
                let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
                min = min.min(ndc);
                max = max.max(ndc);
            }
        }
    }
    // NDC (y up) -> framebuffer pixels (y down), clamped to the target.
    let (fw, fh) = (vp_w as f32, vp_h as f32);
    let x0 = ((min.x * 0.5 + 0.5) * fw).floor().clamp(0.0, fw);
    let x1 = ((max.x * 0.5 + 0.5) * fw).ceil().clamp(0.0, fw);
    let y0 = ((0.5 - max.y * 0.5) * fh).floor().clamp(0.0, fh);
    let y1 = ((0.5 - min.y * 0.5) * fh).ceil().clamp(0.0, fh);
    let w = (x1 - x0) as u32;
    let h = (y1 - y0) as u32;
    if w == 0 || h == 0 {
        DecalScissor::Skip
    } else {
        DecalScissor::Rect(x0 as u32, y0 as u32, w, h)
    }
}

/// Flat uniform buffer matching the WGSL `DecalUniform` struct (144 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DecalUniformRaw {
    pub inv_transform: [[f32; 4]; 4], // 64 bytes
    pub blend_mode: u32,              //  4
    pub alpha: f32,                   //  4
    pub normal_blend_strength: f32,   //  4
    pub has_normal: u32,              //  4
    // Surface response, driving the BRDF the lit blend mode evaluates.
    pub roughness: f32,         //  4
    pub metallic: f32,          //  4
    pub has_roughness_tex: u32, //  4
    pub has_metallic_tex: u32,  //  4
    // UV transform. vec2 pairs, 8-byte aligned.
    pub uv_offset: [f32; 2], //  8
    pub uv_scale: [f32; 2],  //  8
    // Emission, added on top of whatever the blend mode produces.
    pub emissive: f32,         //  4
    pub has_emissive_tex: u32, //  4
    // Edge fade and the ambient floor.
    pub edge_fade: f32, //  4
    pub ambient: f32,   //  4
    // Projection mode.
    pub projection: u32,          //  4  (0 = Planar, 1 = TriPlanar)
    pub tri_blend_sharpness: f32, //  4
    pub _pad2: u32,               //  4  (pad to 144-byte struct size)
    pub _pad3: u32,               //  4
                                  // total: 144 bytes
}

/// Per-draw GPU data for one [`DecalItem`](crate::renderer::DecalItem).
///
/// `Clone` is cheap: the buffer and bind group are reference-counted GPU
/// handles, so cloning bumps a refcount rather than reallocating. This lets the
/// per-frame decal cache hand the same GPU resources to the draw list without
/// rebuilding them.
#[derive(Clone)]
pub(crate) struct DecalGpuItem {
    pub blend_mode: crate::renderer::DecalBlendMode,
    pub _uniform_buf: crate::gpu::Buffer,
    pub bind_group: crate::gpu::BindGroup,
    /// Decal model matrix (local-space unit cube -> world). Used at draw time
    /// to compute a per-decal scissor rect so the fullscreen decal quad only
    /// shades the decal's screen footprint instead of the whole framebuffer.
    pub model: glam::Mat4,
    /// Whether this decal is selected. Selected decals contribute to the decal
    /// outline mask so a ring is traced around their footprint.
    pub selected: bool,
    /// Group 1 of the pick pass: this decal's projection-box transform and the
    /// id it answers with. `None` when the decal is not pickable.
    pub pick: Option<DecalPickBinding>,
}

/// The per-decal uniform and bind group the pick pass binds, kept together so
/// the buffer outlives the bind group that names it.
#[derive(Clone)]
pub(crate) struct DecalPickBinding {
    pub _uniform_buf: crate::gpu::Buffer,
    pub bind_group: crate::gpu::BindGroup,
}

/// GPU mirror of `decal_pick.wgsl`'s `ProxyUniform`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DecalProxyUniform {
    pub model: [[f32; 4]; 4],
    pub object_id: u32,
    pub _pad: [u32; 3],
}

/// Per-draw GPU data for one non-receiver surface in the decal exclude pass.
pub(crate) struct DecalExcludeGpuItem {
    pub mesh_id: MeshId,
    pub _uniform_buf: crate::gpu::Buffer,
    pub bind_group: crate::gpu::BindGroup,
}

/// Build the flat uniform for a decal. Pure: no GPU access, so it can also feed
/// the content hash used to cache GPU resources across frames.
pub(crate) fn decal_uniform_raw(
    item: &crate::renderer::DecalItem,
    texture_is_resident: &dyn Fn(crate::resources::TextureId) -> bool,
) -> DecalUniformRaw {
    let model = glam::Mat4::from_cols_array_2d(&item.transform);
    let inv_transform = model.inverse().to_cols_array_2d();

    let blend_mode_u32 = match item.blend_mode {
        crate::renderer::DecalBlendMode::Replace => 0u32,
        crate::renderer::DecalBlendMode::Multiply => 1u32,
        // Additive uses a separate pipeline; shader logic is identical to Replace.
        crate::renderer::DecalBlendMode::Additive => 0u32,
    };

    let (projection_u32, tri_blend_sharpness) = match item.projection {
        crate::renderer::DecalProjection::Planar => (0u32, 1.0f32),
        crate::renderer::DecalProjection::TriPlanar { blend_sharpness } => {
            (1u32, blend_sharpness.max(0.1))
        }
        crate::renderer::DecalProjection::Cylindrical { facing } => {
            let code = match facing {
                crate::renderer::CylindricalFacing::Outward => 2u32,
                crate::renderer::CylindricalFacing::Inward => 3u32,
            };
            (code, 0.0f32)
        }
    };

    // Each flag says "sample the texture bound in this slot", so it has to agree
    // with what the bind group actually bound. Resolving through the store is
    // what makes them agree: a handle whose slot has been freed does not
    // resolve, the slot binds its fallback, and the flag says to use the scalar
    // instead of sampling it.
    let live =
        |id: Option<crate::resources::TextureId>| id.is_some_and(&texture_is_resident) as u32;
    let has_normal = live(item.normal_texture_id);
    let has_roughness_tex = live(item.roughness_texture_id);
    let has_metallic_tex = live(item.metallic_texture_id);
    let has_emissive_tex = live(item.emissive_texture_id);

    DecalUniformRaw {
        inv_transform,
        blend_mode: blend_mode_u32,
        alpha: item.alpha,
        normal_blend_strength: if has_normal != 0 {
            item.normal_blend_strength
        } else {
            0.0
        },
        has_normal,
        roughness: item.roughness,
        metallic: item.metallic,
        has_roughness_tex,
        has_metallic_tex,
        uv_offset: item.uv_offset,
        uv_scale: item.uv_scale,
        emissive: item.emissive,
        has_emissive_tex,
        edge_fade: item.edge_fade.clamp(0.0, 0.5),
        ambient: item.ambient.max(0.0),
        projection: projection_u32,
        tri_blend_sharpness,
        _pad2: 0,
        _pad3: 0,
    }
}

/// Content hash for a decal, covering everything that affects its GPU
/// resources: the uniform bytes, the bound texture ids, and the blend mode
/// (which the uniform does not distinguish between Replace and Additive but
/// which selects a different pipeline at draw time).
///
/// Two decals with the same hash produce identical GPU resources, so the cache
/// can reuse one across frames instead of rebuilding a buffer and bind group.
pub(crate) fn hash_decal_item(
    item: &crate::renderer::DecalItem,
    texture_is_resident: &dyn Fn(crate::resources::TextureId) -> bool,
) -> u64 {
    use std::hash::Hasher as _;
    let raw = decal_uniform_raw(item, texture_is_resident);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytemuck::bytes_of(&raw));
    h.write_u8(item.blend_mode as u8);
    h.write_u64(item.texture_id.raw());
    // The pick id is baked into the cached entry's pick binding, so two decals
    // that look identical but answer different ids must not share an entry.
    h.write_u64(item.settings.pick_id.0);
    // The key is content identity only. Whether the textures a cached entry
    // names are still resident is the cache's job, answered per entry through
    // `ResourceDeps` when the free epoch moves, so a freed albedo drops the
    // entry and the rebuild binds the fallback under the same key.
    for id in [
        item.normal_texture_id,
        item.roughness_texture_id,
        item.metallic_texture_id,
        item.emissive_texture_id,
    ] {
        h.write_u8(id.is_some() as u8);
        h.write_u64(id.map(|t| t.raw()).unwrap_or(0));
    }
    h.finish()
}

// ---------------------------------------------------------------------------
// Cluster resources
// ---------------------------------------------------------------------------

/// Screen-space decal pipelines and their bind group layouts.
///
/// All fields are lazily built: the render pipelines and item BGL by
/// `ensure_decal_pipeline`, the exclude pipeline by `ensure_decal_exclude_pipeline`,
/// and `depth_bgl` / `sampler` by `ensure_hdr_shared`.
#[derive(Default)]
pub(crate) struct DecalGpu {
    /// Replace-blend decal pipeline (LDR + HDR). None until first decal is submitted.
    pub(crate) replace_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Multiply-blend decal pipeline (LDR + HDR). None until first decal is submitted.
    pub(crate) multiply_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Additive-blend decal pipeline (LDR + HDR). None until first decal is submitted.
    pub(crate) additive_pipeline: Option<crate::gpu::RenderPipeline>,
    /// BGL for group 1 of the decal pass: depth texture + stencil texture bindings.
    pub(crate) depth_bgl: Option<crate::gpu::BindGroupLayout>,
    /// BGL for group 2 of the decal pass: uniform buffer + albedo texture + sampler.
    pub(crate) item_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Linear-clamp sampler used by the decal fragment shader.
    pub(crate) sampler: Option<crate::gpu::Sampler>,
    /// Pipeline that writes stencil = 0 for non-receiver surfaces.
    pub(crate) exclude_pipeline: Option<crate::gpu::RenderPipeline>,
    /// BGL for group 1 of the decal exclude pass: one model matrix uniform buffer.
    pub(crate) exclude_obj_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Pipeline that stamps a selected decal's footprint into the R8 outline
    /// mask. Reuses the decal colour pass's bind groups (camera, depth+stencil,
    /// per-decal uniform). None until first selected decal is submitted.
    pub(crate) outline_mask_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Fullscreen edge-detect pipeline that traces the outline ring from the
    /// decal outline mask onto the HDR colour target with alpha blending.
    pub(crate) outline_edge_pipeline: Option<crate::gpu::RenderPipeline>,
    /// BGL for the edge-detect pass: mask texture + sampler + edge uniform.
    pub(crate) outline_edge_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Object-id pick pipeline: rasterises each decal's projection box.
    /// Built on the first frame a pickable decal is submitted.
    pub(crate) pick_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 1 of the pick pass: one `DecalProxyUniform` per decal.
    pub(crate) pick_bgl: Option<crate::gpu::BindGroupLayout>,
    /// The unit cube every decal's projection box is a transform of. Positions
    /// only: the pick pass needs no normals or uvs.
    pub(crate) pick_cube: Option<(crate::gpu::Buffer, crate::gpu::Buffer)>,
}

/// Persistent GPU resources for the decal outline pass, keyed by viewport size.
/// Only the edge uniform contents change per frame (refreshed with
/// `queue.write_buffer`); the texture, view, buffer, and bind group are reused.
pub(crate) struct DecalOutlineTargets {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// R8 mask the selected decals stamp their footprint into.
    pub(crate) mask_view: crate::gpu::TextureView,
    /// Retained so the view stays valid; not read directly.
    pub(crate) _mask_tex: crate::gpu::Texture,
    /// Edge-detect uniform (outline colour, width, viewport size).
    pub(crate) edge_uniform_buf: crate::gpu::Buffer,
    /// Edge-detect bind group: mask view + sampler + edge uniform.
    pub(crate) edge_bind_group: crate::gpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Pipeline init and upload (impl DeviceResources)
// ---------------------------------------------------------------------------

impl DecalGpu {
    /// Create the decal depth bind group layout and sampler if missing. These
    /// carry no target-size state, so they can be built at renderer creation.
    pub(crate) fn ensure_shared(&mut self, device: &crate::gpu::Device) {
        if self.depth_bgl.is_none() {
            let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("decal_depth_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Depth,
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Texture {
                            sample_type: crate::gpu::TextureSampleType::Uint,
                            view_dimension: crate::gpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
            self.depth_bgl = Some(bgl);
        }
        if self.sampler.is_none() {
            // Repeat address mode so UV scroll animation tiles correctly.
            let sampler = crate::resources::builders::repeat_linear_sampler(
                device,
                "decal_sampler",
                crate::gpu::FilterMode::Nearest,
            );
            self.sampler = Some(sampler);
        }
    }

    /// Create the decal render pipelines and item bind group layout.
    ///
    /// No-op if already created. Requires `decal_depth_bgl` to exist (created
    /// by [`ensure_decal_shared`](Self::ensure_decal_shared)). Called at
    /// renderer creation so the first decal in a scene does not pay a pipeline
    /// compile mid-session.
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        camera_bgl: &crate::gpu::BindGroupLayout,
    ) {
        if self.replace_pipeline.is_some() {
            return;
        }
        // Decals need a third bind group (group 2: per-item uniforms + textures).
        // On limited devices (max_bind_groups < 3, e.g. iced's WebGL2-portable
        // device) the pipeline cannot be created. Leave it None; every decal draw
        // path already guards on the Option, so decals simply do not render.
        if device.limits().max_bind_groups < 3 {
            return;
        }

        let shader = crate::resources::builders::wgsl_module(
            device,
            "decal_shader",
            crate::resources::builders::wgsl_source!("decal"),
        );

        let tex2d_entry = |binding: u32| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Texture {
                sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                view_dimension: crate::gpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        // Group 2: per-item uniforms + textures.
        //  0: DecalUniform buffer
        //  1: albedo texture
        //  2: sampler (shared by all texture slots)
        //  3: normal map    (fallback_texture when absent)
        //  4: roughness map (fallback_texture when absent)
        //  5: metallic map  (fallback_texture when absent)
        //  6: emissive map  (fallback_texture when absent)
        let item_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("decal_item_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex2d_entry(1),
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex2d_entry(3), // normal map
                tex2d_entry(4), // roughness map
                tex2d_entry(5), // metallic map
                tex2d_entry(6), // emissive map
            ],
        });

        let depth_bgl = self
            .depth_bgl
            .as_ref()
            .expect("decal_depth_bgl must exist before ensure_decal_pipeline");

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "decal_pipeline_layout",
            &[camera_bgl, depth_bgl, &item_bgl],
        );

        // No depth attachment: decals read depth as a texture, they do not write
        // to the depth buffer. Blend mode is the only thing that varies.
        let make = |fmt: crate::gpu::TextureFormat, blend: crate::gpu::BlendState| {
            crate::resources::builders::build_fullscreen_pipeline(
                device,
                "decal_pipeline",
                &layout,
                &shader,
                fmt,
                Some(blend),
            )
        };

        let replace_blend = crate::gpu::BlendState::ALPHA_BLENDING;
        // Multiply: result.rgb = dst.rgb * src.rgb; result.a = dst.a.
        let multiply_blend = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::Zero,
                dst_factor: crate::gpu::BlendFactor::Src,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::Zero,
                dst_factor: crate::gpu::BlendFactor::One,
                operation: crate::gpu::BlendOperation::Add,
            },
        };
        // Additive: result.rgb = dst.rgb + src.rgb * src.a; result.a = dst.a.
        // src.a modulates the additive contribution so alpha still controls intensity.
        let additive_blend = crate::gpu::BlendState {
            color: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::SrcAlpha,
                dst_factor: crate::gpu::BlendFactor::One,
                operation: crate::gpu::BlendOperation::Add,
            },
            alpha: crate::gpu::BlendComponent {
                src_factor: crate::gpu::BlendFactor::Zero,
                dst_factor: crate::gpu::BlendFactor::One,
                operation: crate::gpu::BlendOperation::Add,
            },
        };

        // HDR only: decals are composited between the opaque scene and the
        // transparency passes, which exist on the HDR path alone.
        self.item_bgl = Some(item_bgl);
        self.replace_pipeline = Some(make(crate::gpu::TextureFormat::Rgba16Float, replace_blend));
        self.multiply_pipeline = Some(make(crate::gpu::TextureFormat::Rgba16Float, multiply_blend));
        self.additive_pipeline = Some(make(crate::gpu::TextureFormat::Rgba16Float, additive_blend));
    }

    /// Lazily create the decal outline mask + edge-detect pipelines.
    ///
    /// No-op if already created. Requires `ensure_decal_pipeline` to have run
    /// first (it creates `item_bgl`) and `depth_bgl` to exist (created by
    /// `ensure_hdr_shared`). The mask pipeline reuses the decal colour pass's
    /// three bind groups, so no new per-decal resources are needed.
    pub(crate) fn ensure_outline_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        camera_bgl: &crate::gpu::BindGroupLayout,
    ) {
        if self.outline_mask_pipeline.is_some() {
            return;
        }

        if self.depth_bgl.is_none() || self.item_bgl.is_none() {
            return;
        }
        let (Some(depth_bgl), Some(item_bgl)) = (self.depth_bgl.as_ref(), self.item_bgl.as_ref())
        else {
            return;
        };

        // Mask pipeline: stamp the decal footprint into an R8 mask.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "decal_outline_mask_shader",
            crate::resources::builders::wgsl_source!("decal_outline_mask"),
        );
        let mask_layout = crate::resources::builders::pipeline_layout(
            device,
            "decal_outline_mask_layout",
            &[camera_bgl, depth_bgl, item_bgl],
        );
        let mask_pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "decal_outline_mask_pipeline",
            &mask_layout,
            &mask_shader,
            crate::resources::MASK_COLOR_FORMAT,
            None,
        );

        // Edge pipeline: ring edge-detect over the mask, blended onto HDR.
        let edge_shader = crate::resources::builders::wgsl_module(
            device,
            "decal_outline_edge_shader",
            crate::resources::builders::wgsl_source!("outline_edge"),
        );
        let edge_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("decal_outline_edge_bgl"),
            entries: &[
                crate::resources::builders::texture_entry(0, crate::gpu::ShaderStages::FRAGMENT),
                crate::resources::builders::sampler_entry(1, crate::gpu::ShaderStages::FRAGMENT),
                crate::resources::builders::uniform_entry(2, crate::gpu::ShaderStages::FRAGMENT),
            ],
        });
        let edge_layout = crate::resources::builders::pipeline_layout(
            device,
            "decal_outline_edge_layout",
            &[&edge_bgl],
        );
        let edge_pipeline = crate::resources::builders::build_fullscreen_pipeline(
            device,
            "decal_outline_edge_pipeline",
            &edge_layout,
            &edge_shader,
            crate::gpu::TextureFormat::Rgba16Float,
            Some(crate::gpu::BlendState::ALPHA_BLENDING),
        );

        self.outline_mask_pipeline = Some(mask_pipeline);
        self.outline_edge_pipeline = Some(edge_pipeline);
        self.outline_edge_bgl = Some(edge_bgl);
    }

    /// Ensure the persistent decal-outline mask target and edge bind group exist
    /// at `(w, h)`, rebuilding only when the size changes. This keeps the outline
    /// pass allocation-free per frame: only the edge uniform contents are
    /// refreshed (by the caller, via `queue.write_buffer`). Call after
    /// `ensure_decal_outline_pipelines`.
    pub(crate) fn ensure_outline_targets(
        &self,
        device: &crate::gpu::Device,
        slot: &mut Option<DecalOutlineTargets>,
        w: u32,
        h: u32,
    ) {
        if let Some(t) = slot.as_ref() {
            if t.width == w && t.height == h {
                return;
            }
        }
        let (Some(edge_bgl), Some(sampler)) =
            (self.outline_edge_bgl.as_ref(), self.sampler.as_ref())
        else {
            return;
        };

        let mask_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("decal_outline_mask_tex"),
            size: crate::gpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::resources::MASK_COLOR_FORMAT,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mask_view = mask_tex.create_view(&crate::gpu::TextureViewDescriptor::default());
        let edge_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("decal_outline_edge_uniform_buf"),
            size: std::mem::size_of::<crate::resources::OutlineEdgeUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let edge_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("decal_outline_edge_bg"),
            layout: edge_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&mask_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: edge_uniform_buf.as_entire_binding(),
                },
            ],
        });
        *slot = Some(DecalOutlineTargets {
            width: w,
            height: h,
            mask_view,
            _mask_tex: mask_tex,
            edge_uniform_buf,
            edge_bind_group,
        });
    }

    /// Create the per-viewport depth+stencil bind group used by the decal pass.
    ///
    /// Rebuilt every frame: the HDR attachments can be reallocated at the same
    /// size, which would leave a cached bind group pointing at a dead view.
    pub(crate) fn create_depth_bg(
        &self,
        device: &crate::gpu::Device,
        depth_only_view: &crate::gpu::TextureView,
        stencil_only_view: &crate::gpu::TextureView,
    ) -> crate::gpu::BindGroup {
        let bgl = self
            .depth_bgl
            .as_ref()
            .expect("decal_depth_bgl not created");
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("decal_depth_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(depth_only_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(stencil_only_view),
                },
            ],
        })
    }

    /// Upload one [`DecalItem`](crate::renderer::DecalItem) to GPU and return the per-draw data.
    ///
    /// Panics if called before `ensure_decal_pipeline`.
    pub(crate) fn upload_item(
        &self,
        device: &crate::gpu::Device,
        res: &DeviceResources,
        item: &crate::renderer::DecalItem,
    ) -> DecalGpuItem {
        let model = glam::Mat4::from_cols_array_2d(&item.transform);
        let raw = decal_uniform_raw(item, &|id| res.has_texture(id));

        let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("decal_uniform_buf"),
            contents: bytemuck::bytes_of(&raw),
            usage: crate::gpu::BufferUsages::UNIFORM,
        });

        let fallback = res.fallback_texture_view(crate::scene::material::TextureSlot::Albedo);
        let resolve_tex = |id: Option<crate::resources::TextureId>| -> &crate::gpu::TextureView {
            id.and_then(|i| res.texture_view(i)).unwrap_or(fallback)
        };

        let tex_view = res.texture_view(item.texture_id).unwrap_or(fallback);
        let normal_view = resolve_tex(item.normal_texture_id);
        let roughness_view = resolve_tex(item.roughness_texture_id);
        let metallic_view = resolve_tex(item.metallic_texture_id);
        let emissive_view = resolve_tex(item.emissive_texture_id);

        let bgl = self
            .item_bgl
            .as_ref()
            .expect("ensure_decal_pipeline not called");

        let sampler = self.sampler.as_ref().expect("decal_sampler not created");

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("decal_item_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(tex_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: crate::gpu::BindingResource::TextureView(normal_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: crate::gpu::BindingResource::TextureView(roughness_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: crate::gpu::BindingResource::TextureView(metallic_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: crate::gpu::BindingResource::TextureView(emissive_view),
                },
            ],
        });

        // Group 1 of the pick pass. Built only for a pickable decal: an
        // unpickable one contributes nothing to the id pass.
        let pick = (item.settings.pick_id != crate::PickId::NONE)
            .then(|| self.pick_binding(device, model, item.settings.pick_id))
            .flatten();

        DecalGpuItem {
            blend_mode: item.blend_mode,
            _uniform_buf: uniform_buf,
            bind_group,
            model,
            selected: item.settings.selected,
            pick,
        }
    }

    /// Build one decal's pick binding: its projection-box transform and pick
    /// id. `None` before [`ensure_pick`](Self::ensure_pick) has run.
    fn pick_binding(
        &self,
        device: &crate::gpu::Device,
        model: glam::Mat4,
        pick_id: crate::PickId,
    ) -> Option<DecalPickBinding> {
        use crate::gpu::util::DeviceExt as _;
        let bgl = self.pick_bgl.as_ref()?;
        let raw = DecalProxyUniform {
            model: model.to_cols_array_2d(),
            object_id: pick_id.0 as u32,
            _pad: [0; 3],
        };
        let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("decal_pick_uniform_buf"),
            contents: bytemuck::bytes_of(&raw),
            usage: crate::gpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("decal_pick_bg"),
            layout: bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        Some(DecalPickBinding {
            _uniform_buf: uniform_buf,
            bind_group,
        })
    }

    /// Lazily build the object-id pick pipeline, its group-1 layout, and the
    /// shared unit cube every decal's projection box is a transform of.
    pub(crate) fn ensure_pick(
        &mut self,
        device: &crate::gpu::Device,
        resources: &crate::resources::DeviceResources,
    ) {
        if self.pick_pipeline.is_some() {
            return;
        }
        use crate::gpu::util::DeviceExt as _;
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("decal_pick_bgl"),
            entries: &[crate::resources::builders::uniform_entry(
                0,
                crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            )],
        });
        let shader = crate::resources::builders::wgsl_module(
            device,
            "decal_pick_shader",
            crate::resources::builders::wgsl_source!("decal_pick"),
        );
        const POS_ATTRS: [crate::gpu::VertexAttribute; 1] = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }];
        let vertex_layout = crate::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &POS_ATTRS,
        };
        let mut opts = crate::resources::PluginPipelineOpts::new(
            Some("decal_pick_pipeline"),
            &shader,
            "vs_main",
            "fs_main",
            std::slice::from_ref(&vertex_layout),
        );
        // Two-sided: the camera can sit inside a decal's projection box, and a
        // click from in there still selects it.
        opts.primitive.cull_mode = None;
        let extra: [&crate::gpu::BindGroupLayout; 1] = [&bgl];
        opts.extra_bind_group_layouts = &extra;
        let pipeline = resources.build_pick_pipeline(device, &opts);

        let positions: [[f32; 3]; 8] = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ];
        let indices: [u32; 36] = [
            0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7,
            3, 0, 4, 5, 5, 1, 0,
        ];
        let vbuf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("decal_pick_cube_vbuf"),
            contents: bytemuck::cast_slice(&positions),
            usage: crate::gpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("decal_pick_cube_ibuf"),
            contents: bytemuck::cast_slice(&indices),
            usage: crate::gpu::BufferUsages::INDEX,
        });

        self.pick_bgl = Some(bgl);
        self.pick_pipeline = Some(pipeline);
        self.pick_cube = Some((vbuf, ibuf));
    }

    /// Lazily create the decal exclude pipeline and its object BGL.
    ///
    /// No-op if already created. Must be called after `camera_bind_group_layout` exists.
    pub(crate) fn ensure_exclude_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        camera_bgl: &crate::gpu::BindGroupLayout,
    ) {
        if self.exclude_pipeline.is_some() {
            return;
        }

        let shader = crate::resources::builders::wgsl_module(
            device,
            "decal_exclude_shader",
            crate::resources::builders::wgsl_source!("decal_exclude"),
        );

        let obj_bgl = crate::resources::builders::uniform_bgl(
            device,
            "decal_exclude_obj_bgl",
            crate::gpu::ShaderStages::VERTEX,
        );

        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "decal_exclude_pipeline_layout",
            camera_bgl,
            &obj_bgl,
        );

        // Vertex layout: position only (location 0, Float32x3).
        // Stride matches the full Vertex struct (64 bytes) but we only read pos.
        let vertex_layout = crate::gpu::VertexBufferLayout {
            array_stride: 64,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &[crate::gpu::VertexAttribute {
                format: crate::gpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        };

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "decal_exclude_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[vertex_layout],
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::gpu::DepthStencilState {
                    format: crate::gpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: crate::resources::builders::dwrite(false),
                    depth_compare: crate::resources::builders::dcompare(
                        crate::gpu::CompareFunction::LessEqual,
                    ),
                    stencil: crate::gpu::StencilState {
                        front: crate::gpu::StencilFaceState {
                            compare: crate::gpu::CompareFunction::Always,
                            fail_op: crate::gpu::StencilOperation::Keep,
                            depth_fail_op: crate::gpu::StencilOperation::Keep,
                            pass_op: crate::gpu::StencilOperation::Replace,
                        },
                        back: crate::gpu::StencilFaceState {
                            compare: crate::gpu::CompareFunction::Always,
                            fail_op: crate::gpu::StencilOperation::Keep,
                            depth_fail_op: crate::gpu::StencilOperation::Keep,
                            pass_op: crate::gpu::StencilOperation::Replace,
                        },
                        read_mask: 0xff,
                        write_mask: 0xff,
                    },
                    // Slight negative bias so the re-projected geometry reliably passes
                    // the LessEqual depth test against values written by the opaque pass.
                    // Without this, floating-point rounding differences between two
                    // separate render passes of the same geometry can cause the depth
                    // test to fail intermittently, leaving stencil un-written.
                    bias: crate::gpu::DepthBiasState {
                        constant: -2,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                }),
                multisample: crate::gpu::MultisampleState::default(),
                cache: None,
            },
        );

        self.exclude_obj_bgl = Some(obj_bgl);
        self.exclude_pipeline = Some(pipeline);
    }

    /// Upload one non-receiver surface for the decal exclude pass and return per-draw data.
    ///
    /// Panics if called before `ensure_decal_exclude_pipeline`.
    pub(crate) fn upload_exclude_item(
        &self,
        device: &crate::gpu::Device,
        mesh_id: MeshId,
        model: [[f32; 4]; 4],
    ) -> DecalExcludeGpuItem {
        let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
            label: Some("decal_exclude_uniform_buf"),
            contents: bytemuck::cast_slice(&model),
            usage: crate::gpu::BufferUsages::UNIFORM,
        });

        let bgl = self
            .exclude_obj_bgl
            .as_ref()
            .expect("ensure_decal_exclude_pipeline not called");

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("decal_exclude_obj_bg"),
            layout: bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        DecalExcludeGpuItem {
            mesh_id,
            _uniform_buf: uniform_buf,
            bind_group,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::hash_decal_item;
    use crate::renderer::{DecalBlendMode, DecalItem};

    /// Nothing is resident, so every id in these items is unresolvable: the
    /// state the hash sees for a decal whose textures were never uploaded.
    fn no_textures() -> impl Fn(crate::resources::TextureId) -> bool {
        |_| false
    }

    #[test]
    fn identical_decals_hash_equal() {
        let a = DecalItem {
            texture_id: crate::resources::TextureId::from_raw(3),
            ..DecalItem::default()
        };
        let b = a.clone();
        assert_eq!(
            hash_decal_item(&a, &no_textures()),
            hash_decal_item(&b, &no_textures())
        );
    }

    #[test]
    fn blend_mode_changes_hash() {
        // Replace and Additive encode the same uniform bytes but select
        // different pipelines, so they must not share a cache entry.
        let replace = DecalItem {
            blend_mode: DecalBlendMode::Replace,
            ..DecalItem::default()
        };
        let additive = DecalItem {
            blend_mode: DecalBlendMode::Additive,
            ..DecalItem::default()
        };
        assert_ne!(
            hash_decal_item(&replace, &no_textures()),
            hash_decal_item(&additive, &no_textures())
        );
    }

    #[test]
    fn texture_and_transform_change_hash() {
        let base = DecalItem::default();
        let tex = DecalItem {
            texture_id: crate::resources::TextureId::from_raw(7),
            ..DecalItem::default()
        };
        assert_ne!(
            hash_decal_item(&base, &no_textures()),
            hash_decal_item(&tex, &no_textures())
        );

        let mut moved = DecalItem::default();
        moved.transform[3][0] = 5.0;
        assert_ne!(
            hash_decal_item(&base, &no_textures()),
            hash_decal_item(&moved, &no_textures())
        );
    }
}

#[cfg(test)]
mod decal_scissor_tests {
    use super::{DecalScissor, decal_scissor};
    use glam::{Mat4, Vec3};

    fn view_proj() -> Mat4 {
        let proj = Mat4::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        // Z-up camera 10 units back on -Y, looking at the origin.
        let view = Mat4::look_at_rh(Vec3::new(0.0, -10.0, 2.0), Vec3::ZERO, Vec3::Z);
        proj * view
    }

    #[test]
    fn centered_box_yields_subrect() {
        let model = Mat4::from_scale(Vec3::splat(1.0));
        match decal_scissor(&model, &view_proj(), 1920, 1080) {
            DecalScissor::Rect(x, y, w, h) => {
                assert!(w > 0 && h > 0);
                assert!(
                    w < 1920 && h < 1080,
                    "small distant box should not fill the screen"
                );
                assert!(
                    x + w <= 1920 && y + h <= 1080,
                    "rect must stay within the target"
                );
            }
            _ => panic!("expected a sub-rect for a centered box"),
        }
    }

    #[test]
    fn far_offscreen_box_skips() {
        // Far off to the +X side but still in front of the camera.
        let model = Mat4::from_translation(Vec3::new(500.0, 100.0, 0.0));
        assert!(matches!(
            decal_scissor(&model, &view_proj(), 1920, 1080),
            DecalScissor::Skip
        ));
    }

    #[test]
    fn box_enclosing_camera_falls_back_to_full() {
        // A large box centred on the camera puts a corner behind the near plane.
        let model =
            Mat4::from_translation(Vec3::new(0.0, -10.0, 2.0)) * Mat4::from_scale(Vec3::splat(4.0));
        assert!(matches!(
            decal_scissor(&model, &view_proj(), 1920, 1080),
            DecalScissor::Full
        ));
    }
}
