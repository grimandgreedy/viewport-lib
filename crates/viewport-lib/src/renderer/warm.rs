//! Building pipelines ahead of the frames that need them.

use crate::renderer::ViewportRenderer;
use crate::renderer::types::{EffectsFrame, GroundPlaneMode};

/// What [`ViewportRenderer::warm_pipelines`] builds.
///
/// Start from [`for_effects`](Self::for_effects) with the effect settings the
/// application renders with, add what a frame's settings cannot say
/// (transparency, outlines, shadows, the `Direct` path, item types), or take
/// [`all`](Self::all). An empty set builds nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PipelineSet {
    hdr: bool,
    direct: bool,
    transparency: bool,
    outlines: bool,
    shadows: bool,
    bloom: bool,
    ssao: bool,
    contact_shadows: bool,
    dof: bool,
    fxaa: bool,
    ssaa: bool,
    auto_exposure: bool,
    ground_plane: bool,
    skybox: bool,
    material_plugins: bool,
    item_types: Vec<std::any::TypeId>,
    all_item_types: bool,
}

impl PipelineSet {
    /// The pipelines `effects` makes a frame bind: the colour family of its
    /// display mode, each post effect it enables, shadows when its lighting
    /// casts them, the ground plane and the skybox when shown.
    pub fn for_effects(effects: &EffectsFrame) -> Self {
        let pp = &effects.post_process;
        Self {
            hdr: effects.display.is_hdr(),
            direct: !effects.display.is_hdr(),
            shadows: effects.lighting.shadows.enabled,
            bloom: pp.bloom.enabled,
            ssao: pp.ssao,
            contact_shadows: pp.contact_shadows.enabled,
            dof: pp.dof.enabled,
            fxaa: pp.fxaa,
            ssaa: pp.ssaa_factor > 1,
            auto_exposure: effects.display.exposure.manual_multiplier().is_none(),
            ground_plane: !matches!(effects.ground_plane.mode, GroundPlaneMode::None),
            skybox: effects.environment.as_ref().is_some_and(|e| e.show_skybox),
            ..Self::default()
        }
    }

    /// Everything the renderer can draw with, every registered material
    /// plugin, and every registered item type.
    pub fn all() -> Self {
        Self {
            hdr: true,
            direct: true,
            transparency: true,
            outlines: true,
            shadows: true,
            bloom: true,
            ssao: true,
            contact_shadows: true,
            dof: true,
            fxaa: true,
            ssaa: true,
            auto_exposure: true,
            ground_plane: true,
            skybox: true,
            material_plugins: true,
            item_types: Vec::new(),
            all_item_types: true,
        }
    }

    /// The transparent and order-independent-transparency pipelines, for a
    /// scene that will show an item at an opacity below one.
    pub fn with_transparency(mut self) -> Self {
        self.transparency = true;
        self
    }

    /// The selection outline pipelines.
    pub fn with_outlines(mut self) -> Self {
        self.outlines = true;
        self
    }

    /// The cascade, point and instanced shadow pipelines.
    pub fn with_shadows(mut self) -> Self {
        self.shadows = true;
        self
    }

    /// The LDR families the `Direct` display mode and `paint_direct` draw with.
    pub fn with_direct_paint(mut self) -> Self {
        self.direct = true;
        self
    }

    /// The HDR families, effects infrastructure and tone map.
    pub fn with_hdr(mut self) -> Self {
        self.hdr = true;
        self
    }

    /// Everything every registered material plugin can draw with.
    pub fn with_material_plugins(mut self) -> Self {
        self.material_plugins = true;
        self
    }

    /// Everything the registered item type `T` can draw with, through its
    /// `ItemTypePlugin::warm`.
    pub fn with_item_type<T: 'static>(mut self) -> Self {
        self.item_types.push(std::any::TypeId::of::<T>());
        self
    }
}

impl ViewportRenderer {
    /// Ask for every pipeline in `set` now instead of at the frame that first
    /// binds each.
    ///
    /// Under [`PipelineCompilation::Background`](crate::PipelineCompilation::Background)
    /// the call hands the work to the workers and returns at once;
    /// [`pipelines_pending`](Self::pipelines_pending) says how much is left
    /// and [`wait_for_pipelines`](Self::wait_for_pipelines) blocks until it is
    /// done. Under `Blocking` the pipelines are built before this returns.
    ///
    /// The shader modules and layouts the families share are composed on the
    /// calling thread either way, and a pending rebuild after a deformer
    /// registration is flushed first. Render targets and shadow maps are not
    /// allocated here; the first frame sizes them.
    ///
    /// A pipeline whose key depends on something the set cannot name is not
    /// covered: the depth upscale of a frame rendered below native scale, the
    /// foreground depth stamp, and whatever an item type's `warm` leaves to
    /// its first frame.
    pub fn warm_pipelines(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        set: &PipelineSet,
    ) {
        let format = self.resources.target_format;
        let cull = self.instancing.gpu_culling_enabled;
        self.resources.flush_mesh_pipeline_rebuild(device);
        let r = &mut self.resources;

        if set.direct || set.hdr {
            r.ensure_instanced_pipelines(device);
            if cull {
                r.ensure_cull_instance_pipelines(device);
                // The compute side of GPU culling, which the first culled
                // frame would otherwise build.
                let dev = device.clone();
                self.instancing
                    .cull_resources
                    .get(&r.pipeline_compiler, move || {
                        crate::renderer::indirect::CullResources::new(&dev)
                    });
            }
        }
        if set.direct {
            r.ensure_ldr_mesh_pipelines(device);
            r.ensure_ldr_instanced_pipelines(device);
            if let Some(f) = &r.scene.ldr {
                f.request_all();
            }
            if let Some(f) = &r.instancing.ldr {
                f.request_all();
            }
        }
        if set.hdr {
            r.ensure_hdr_infra(device, queue);
            r.ensure_tone_map_pipeline(device, format);
            r.ensure_hdr_mesh_pipelines(device);
            r.ensure_hdr_instanced_pipelines(device);
            if let Some(f) = &r.scene.hdr {
                f.request_all();
            }
            if let Some(f) = &r.instancing.hdr {
                f.request_all();
            }
            if cull {
                r.ensure_hdr_cull_pipelines(device);
                if let Some(f) = &r.cull.hdr {
                    f.request_all();
                }
            }
            if set.transparency {
                r.ensure_oit_composite_pipeline(device);
                r.ensure_oit_mesh_pipelines(device);
                r.ensure_oit_instanced_pipeline(device);
                if let Some(f) = &r.oit.composite_pipeline {
                    f.request_all();
                }
                if let Some(f) = &r.oit.pipeline {
                    f.request_all();
                }
                if let Some(f) = &r.oit.instanced {
                    f.request_all();
                }
                if cull {
                    r.ensure_oit_cull_pipelines(device);
                    if let Some(f) = &r.cull.oit {
                        f.request_all();
                    }
                }
            }
            // The effects need the HDR infrastructure above.
            if set.bloom {
                r.ensure_bloom_pipelines(device);
                for f in [
                    &r.post.bloom.threshold_pipeline,
                    &r.post.bloom.blur_pipeline,
                ] {
                    if let Some(f) = f {
                        f.request_all();
                    }
                }
            }
            if set.ssao {
                r.ensure_ssao_pipelines(device);
                for f in [&r.post.ssao.pipeline, &r.post.ssao.blur_pipeline] {
                    if let Some(f) = f {
                        f.request_all();
                    }
                }
            }
            if set.contact_shadows {
                r.ensure_contact_shadow_pipeline(device);
                if let Some(f) = &r.post.contact_shadow.pipeline {
                    f.request_all();
                }
            }
            if set.dof {
                r.ensure_dof_pipeline(device);
                if let Some(f) = &r.post.dof.pipeline {
                    f.request_all();
                }
            }
            if set.fxaa {
                r.ensure_fxaa_pipeline(device, format);
                if let Some(f) = &r.post.fxaa.pipeline {
                    f.request_all();
                }
            }
            if set.ssaa {
                r.ensure_ssaa_resolve_pipelines(device);
            }
            if set.auto_exposure {
                r.exposure.ensure_pipelines(device);
            }
        }
        if set.shadows {
            r.ensure_cascade_shadow_pipelines(device);
            r.ensure_point_shadow_pipeline(device);
            if let Some(f) = &r.shadow.pipeline {
                f.request_all();
            }
            if let Some(f) = &r.shadow.point_pipeline {
                f.request_all();
            }
            if let Some(f) = &r.instancing.shadow {
                f.request_all();
            }
            if let Some(f) = &r.cull.shadow {
                f.request_all();
            }
        }
        if set.outlines {
            r.ensure_outline_pipelines(device);
            r.ensure_outline_composite_pipelines(device);
            if let Some(f) = &r.outline.composite {
                f.request_all();
            }
        }
        if set.ground_plane {
            r.ensure_ground_plane_pipeline(device);
        }
        if set.skybox {
            r.ensure_skybox_pipeline(device);
        }
        if set.material_plugins {
            r.warm_all_material_plugin_pipelines(device);
        }
        if set.all_item_types || !set.item_types.is_empty() {
            for plugin in self.item_type_plugins.values_mut() {
                let id = plugin.as_any_plugin().type_id();
                if set.all_item_types || set.item_types.contains(&id) {
                    plugin.warm(device, &self.resources);
                }
            }
        }
    }
}
