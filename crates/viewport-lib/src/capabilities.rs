//! What a device can do, read once, and the tier the renderer runs at on it.
//!
//! Every capability decision the renderer makes reads [`Capabilities`], never a
//! platform or driver name. Consumers and documentation name the [`Tier`].

use crate::gpu;

/// The device facts the renderer decides on, read once at construction.
///
/// Build it with [`from_device`](Self::from_device) for a device you hold, or
/// [`probe`](Self::probe) to ask an adapter before creating a device.
/// [`ViewportRenderer::capabilities`](crate::ViewportRenderer::capabilities)
/// returns the one the renderer was built with.
///
/// The downlevel flags (`compute`, the per-stage storage flags, cube arrays,
/// base vertex, indirect execution) come from the adapter. A `Device` cannot
/// report them, so a renderer built from a device alone assumes them present,
/// which holds on Vulkan, Metal, DX12 and WebGPU but not on every GL or older
/// Metal device. Pass the adapter with
/// [`RendererConfig::with_adapter`](crate::RendererConfig::with_adapter) on
/// those. [`downlevel_known`](Self::downlevel_known) says which case this is.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Capabilities {
    /// `max_storage_buffers_per_shader_stage`. On wgpu's GL backend this is the
    /// fragment count when the vertex count is zero, so read
    /// [`vertex_storage`](Self::vertex_storage) for the vertex stage.
    pub storage_buffers_per_stage: u32,
    /// `max_uniform_buffer_binding_size`, in bytes.
    pub uniform_binding_size: u64,
    /// `max_bind_groups`.
    pub bind_groups: u32,
    /// `max_sampled_textures_per_shader_stage`.
    pub sampled_textures_per_stage: u32,
    /// `max_uniform_buffers_per_shader_stage`.
    pub uniform_buffers_per_stage: u32,
    /// `max_texture_dimension_2d`.
    pub texture_dimension_2d: u32,
    /// `max_binding_array_elements_per_shader_stage`.
    pub binding_array_elements: u32,
    /// Compute shaders can be dispatched: the adapter has
    /// `DownlevelFlags::COMPUTE_SHADERS` and the device's compute workgroup
    /// limits are non-zero. The flag alone survives a device created with
    /// downlevel limits, whose zero workgroup limits fail every compute
    /// pipeline.
    pub compute: bool,
    /// Storage buffers can be bound in the vertex stage.
    pub vertex_storage: bool,
    /// Storage buffers can be bound in the fragment stage.
    pub fragment_storage: bool,
    /// Cube-array texture views exist. Without them point-light shadows are
    /// not sampled.
    pub cube_array_textures: bool,
    /// Shaders can `textureLoad` a depth texture. naga's GLSL writer cannot
    /// translate that, so this is false on wgpu's GL backend, where the PCSS
    /// shadow filters fall back to PCF. It follows from the backend:
    /// [`probe`](Self::probe) reads it from the adapter and
    /// [`from_device`](Self::from_device) from the device on wgpu 29 and 30.
    /// On wgpu 27 a device cannot report its backend, so it comes from
    /// [`RendererConfig::with_adapter`](crate::RendererConfig::with_adapter)
    /// and is otherwise assumed true.
    pub depth_texture_loads: bool,
    /// The scene depth can be sampled through its own depth-aspect view while
    /// the same texture is attached read-only
    /// (`DownlevelFlags::READ_ONLY_DEPTH_STENCIL`) or its stencil aspect is
    /// sampled too. Neither holds on wgpu's GL backend: a read-only attachment
    /// counts as written there, and a depth-stencil texture samples one aspect
    /// at a time. Without it the renderer hands plugins a copy of the scene
    /// depth. Set from the backend the same way as
    /// [`depth_texture_loads`](Self::depth_texture_loads).
    pub direct_depth_sampling: bool,
    /// Indexed draws can start at a non-zero base vertex.
    pub base_vertex: bool,
    /// Indirect draws and dispatches run.
    pub indirect_execution: bool,
    /// Whether the downlevel flags above came from the adapter, rather than
    /// being assumed from a device alone.
    pub downlevel_known: bool,
    /// `Features::INDIRECT_FIRST_INSTANCE`: the GPU-culled indirect draw path.
    pub indirect_first_instance: bool,
    /// `Features::MULTI_DRAW_INDIRECT_COUNT`: native multi-draw.
    pub multi_draw_indirect_count: bool,
    /// `Features::TIMESTAMP_QUERY`: GPU frame timings.
    pub timestamps: bool,
    /// `Features::PIPELINE_CACHE`.
    pub pipeline_cache: bool,
    /// `@builtin(primitive_index)` in fragment shaders: sub-object GPU picks.
    pub primitive_index: bool,
    /// `Features::FLOAT32_FILTERABLE`.
    pub float32_filterable: bool,
    /// `Features::EXPERIMENTAL_RAY_QUERY`: the hardware path-tracer backend.
    pub ray_query: bool,
    /// `Features::DEPTH_CLIP_CONTROL`: the directional shadow pass clamps
    /// casters in front of a cascade to its near plane instead of clipping
    /// them, so each cascade's depth range can stop just past what it
    /// covers. Without it the range is stretched to take in the casters.
    pub depth_clip_control: bool,
    /// The bindless material-texture path: the texture-array feature set and
    /// enough binding-array elements to build its layout.
    pub bindless: bool,
    /// `Features::DEPTH32FLOAT_STENCIL8`: the scene depth can be
    /// `Depth32FloatStencil8`.
    pub depth32float_stencil8: bool,
    /// The format of the scene depth-stencil attachment: `Depth24PlusStencil8`
    /// unless
    /// [`RendererConfig::scene_depth_format`](crate::RendererConfig::scene_depth_format)
    /// asks for `Depth32FloatStencil8` on a device that has it. Every scene
    /// pipeline and depth target the renderer makes uses it.
    pub scene_depth_format: gpu::TextureFormat,
}

impl Capabilities {
    /// The capabilities of `device`. Pass the adapter's downlevel capabilities
    /// when you have them; without them the downlevel flags are assumed
    /// present.
    pub fn from_device(
        device: &gpu::Device,
        downlevel: Option<&gpu::DownlevelCapabilities>,
    ) -> Self {
        let caps = Self::from_parts(
            &device.limits(),
            device.features(),
            downlevel.map(|d| d.flags),
        );
        #[cfg(any(wgpu29, wgpu30))]
        let caps = caps.with_backend(device.adapter_info().backend);
        caps
    }

    /// The capabilities an adapter offers, before a device exists.
    ///
    /// An upper bound: a device created from `adapter` reports at most these,
    /// and less when it is requested with lower limits or fewer features (wgpu's
    /// defaults, or [`recommended_device_limits`](crate::ViewportRenderer::recommended_device_limits),
    /// which caps a few). Use it to decide on a path, or whether to tell the
    /// user, before committing to a device.
    pub fn probe(adapter: &gpu::Adapter) -> Self {
        Self::from_parts(
            &adapter.limits(),
            adapter.features(),
            Some(adapter.get_downlevel_capabilities().flags),
        )
        .with_backend(adapter.get_info().backend)
    }

    /// These capabilities with what follows from running on `backend`: on
    /// [`Backend::Gl`](gpu::Backend::Gl), no depth-texture loads and no
    /// direct depth sampling.
    pub fn with_backend(mut self, backend: gpu::Backend) -> Self {
        self.depth_texture_loads = backend != gpu::Backend::Gl;
        self.direct_depth_sampling &= backend != gpu::Backend::Gl;
        self
    }

    /// These capabilities reduced to what a device at `tier` has, so a
    /// renderer built from them runs at that tier on a device that has more.
    /// Unchanged when the device is already at or below `tier`.
    ///
    /// Below the full tier the storage buffers and compute go, and with them
    /// the features that need them (bindless textures, GPU-driven culling,
    /// ray queries). For the overlay-only tier the bind groups drop to the
    /// overlay floor as well. Only what the renderer reads is reduced; the
    /// device keeps its own limits, so this shows what a lesser device would
    /// draw without checking that the pipelines would fit its limits.
    pub fn capped_at(mut self, tier: Tier) -> Self {
        if self.tier() <= tier || tier >= Tier::Full {
            return self;
        }
        self.storage_buffers_per_stage = 0;
        self.vertex_storage = false;
        self.fragment_storage = false;
        self.compute = false;
        self.bindless = false;
        self.indirect_first_instance = false;
        self.multi_draw_indirect_count = false;
        self.ray_query = false;
        if tier < Tier::Compatibility {
            self.bind_groups = self.bind_groups.min(Tier::BIND_GROUPS);
        }
        self
    }

    /// The capabilities described by limits, features and downlevel flags you
    /// already hold. `None` for the flags assumes them all present.
    pub fn from_parts(
        limits: &gpu::Limits,
        features: gpu::Features,
        downlevel: Option<gpu::DownlevelFlags>,
    ) -> Self {
        let flag = |f: gpu::DownlevelFlags| downlevel.is_none_or(|d| d.contains(f));
        let storage = limits.max_storage_buffers_per_shader_stage;
        let compute = flag(gpu::DownlevelFlags::COMPUTE_SHADERS)
            && limits.max_compute_workgroup_size_x > 0
            && limits.max_compute_invocations_per_workgroup > 0
            && limits.max_compute_workgroups_per_dimension > 0;
        let bindless = features.contains(gpu::BINDLESS_TEXTURE_FEATURES)
            && limits.max_binding_array_elements_per_shader_stage
                >= crate::resources::mesh::instanced_bindless::BINDLESS_TEXTURE_CAPACITY;
        Self {
            storage_buffers_per_stage: storage,
            uniform_binding_size: limits.max_uniform_buffer_binding_size.into(),
            bind_groups: limits.max_bind_groups,
            sampled_textures_per_stage: limits.max_sampled_textures_per_shader_stage,
            uniform_buffers_per_stage: limits.max_uniform_buffers_per_shader_stage,
            texture_dimension_2d: limits.max_texture_dimension_2d,
            binding_array_elements: limits.max_binding_array_elements_per_shader_stage,
            compute,
            vertex_storage: storage > 0 && flag(gpu::DownlevelFlags::VERTEX_STORAGE),
            fragment_storage: storage > 0 && flag(gpu::DownlevelFlags::FRAGMENT_STORAGE),
            cube_array_textures: flag(gpu::DownlevelFlags::CUBE_ARRAY_TEXTURES),
            depth_texture_loads: true,
            direct_depth_sampling: flag(gpu::DownlevelFlags::READ_ONLY_DEPTH_STENCIL),
            base_vertex: flag(gpu::DownlevelFlags::BASE_VERTEX),
            indirect_execution: flag(gpu::DownlevelFlags::INDIRECT_EXECUTION),
            downlevel_known: downlevel.is_some(),
            indirect_first_instance: features.contains(gpu::Features::INDIRECT_FIRST_INSTANCE),
            multi_draw_indirect_count: features.contains(gpu::Features::MULTI_DRAW_INDIRECT_COUNT),
            timestamps: features.contains(gpu::Features::TIMESTAMP_QUERY),
            pipeline_cache: features.contains(gpu::Features::PIPELINE_CACHE),
            primitive_index: features.contains(gpu::PRIMITIVE_INDEX_FEATURE),
            float32_filterable: features.contains(gpu::Features::FLOAT32_FILTERABLE),
            ray_query: features.contains(gpu::RAY_QUERY_FEATURE),
            depth_clip_control: features.contains(gpu::Features::DEPTH_CLIP_CONTROL),
            bindless,
            depth32float_stencil8: features.contains(gpu::Features::DEPTH32FLOAT_STENCIL8),
            scene_depth_format: gpu::TextureFormat::Depth24PlusStencil8,
        }
    }

    /// The tier this device lands on.
    pub fn tier(&self) -> Tier {
        if self.full_tier_missing().is_empty() {
            Tier::Full
        } else if self.compatibility_tier_missing().is_empty() {
            Tier::Compatibility
        } else if self.overlay_tier_missing().is_empty() {
            Tier::OverlayOnly
        } else {
            Tier::Unsupported
        }
    }

    /// What this device lacks for `tier`, one entry per missing capability.
    /// Empty when the device meets it.
    pub fn missing_for(&self, tier: Tier) -> Vec<String> {
        match tier {
            Tier::Unsupported => Vec::new(),
            Tier::OverlayOnly => self.overlay_tier_missing(),
            Tier::Compatibility => self.compatibility_tier_missing(),
            _ => self.full_tier_missing(),
        }
    }

    /// What this device lacks for `requirements`, one entry per missing
    /// capability. Empty when the device meets them.
    pub fn missing_requirements(&self, requirements: &Requirements) -> Vec<String> {
        let mut missing = Vec::new();
        if self.storage_buffers_per_stage < requirements.storage_buffers_per_stage {
            missing.push(format!(
                "{} storage buffers per shader stage (the device has {})",
                requirements.storage_buffers_per_stage, self.storage_buffers_per_stage
            ));
        }
        if requirements.vertex_storage && !self.vertex_storage {
            missing.push("storage buffers in the vertex stage".to_owned());
        }
        if requirements.fragment_storage && !self.fragment_storage {
            missing.push("storage buffers in the fragment stage".to_owned());
        }
        if requirements.compute && !self.compute {
            missing.push("compute shaders".to_owned());
        }
        if self.bind_groups < requirements.bind_groups {
            missing.push(format!(
                "{} bind groups (the device has {})",
                requirements.bind_groups, self.bind_groups
            ));
        }
        if self.sampled_textures_per_stage < requirements.sampled_textures_per_stage {
            missing.push(format!(
                "{} sampled textures per shader stage (the device has {})",
                requirements.sampled_textures_per_stage, self.sampled_textures_per_stage
            ));
        }
        missing
    }

    fn full_tier_missing(&self) -> Vec<String> {
        self.missing_requirements(&Requirements::full_tier())
    }

    fn compatibility_tier_missing(&self) -> Vec<String> {
        let mut missing = self.overlay_tier_missing();
        missing.extend(self.missing_requirements(&Requirements::compatibility_tier()));
        if self.uniform_buffers_per_stage < Tier::COMPATIBILITY_UNIFORM_BUFFERS_PER_STAGE {
            missing.push(format!(
                "{} uniform buffers per shader stage (the device has {})",
                Tier::COMPATIBILITY_UNIFORM_BUFFERS_PER_STAGE,
                self.uniform_buffers_per_stage
            ));
        }
        missing
    }

    /// The scene depth format for a requested
    /// [`RendererConfig::scene_depth_format`](crate::RendererConfig::scene_depth_format)
    /// on this device, and whether a request for `Depth32FloatStencil8` fell
    /// back to `Depth24PlusStencil8` because the device lacks
    /// `Features::DEPTH32FLOAT_STENCIL8`. No request takes `Depth24PlusStencil8`
    /// on every device, so a host that does not own its depth attachment gets
    /// the format every framework can allocate. Any other format is an error:
    /// the scene passes need a stencil aspect and a depth aspect they can
    /// sample.
    pub(crate) fn select_scene_depth_format(
        &self,
        requested: Option<gpu::TextureFormat>,
    ) -> Result<(gpu::TextureFormat, bool), crate::ViewportError> {
        match requested {
            None | Some(gpu::TextureFormat::Depth24PlusStencil8) => {
                Ok((gpu::TextureFormat::Depth24PlusStencil8, false))
            }
            Some(gpu::TextureFormat::Depth32FloatStencil8) if self.depth32float_stencil8 => {
                Ok((gpu::TextureFormat::Depth32FloatStencil8, false))
            }
            Some(gpu::TextureFormat::Depth32FloatStencil8) => {
                Ok((gpu::TextureFormat::Depth24PlusStencil8, true))
            }
            Some(other) => Err(crate::ViewportError::InvalidSceneDepthFormat {
                format: format!("{other:?}"),
            }),
        }
    }

    fn overlay_tier_missing(&self) -> Vec<String> {
        let mut missing = Vec::new();
        if self.bind_groups < Tier::BIND_GROUPS {
            missing.push(format!(
                "{} bind groups (the device has {})",
                Tier::BIND_GROUPS,
                self.bind_groups
            ));
        }
        if self.uniform_binding_size < Tier::UNIFORM_BINDING_SIZE {
            missing.push(format!(
                "{}-byte uniform bindings (the device has {})",
                Tier::UNIFORM_BINDING_SIZE,
                self.uniform_binding_size
            ));
        }
        missing
    }
}

pub use viewport_lib_types::tier::Tier;

/// What a plugin needs from the device to work.
///
/// Returned by `requires()` on the plugin traits and checked at registration
/// against the renderer's [`Capabilities`]: a plugin whose requirements the
/// device does not meet is declined with
/// [`ViewportError::PluginUnsupported`](crate::ViewportError::PluginUnsupported)
/// instead of failing at pipeline creation. The default is
/// [`full_tier`](Self::full_tier).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Requirements {
    /// Storage buffers per shader stage, summed across every group a pipeline
    /// layout binds.
    pub storage_buffers_per_stage: u32,
    /// Storage buffers are bound in the vertex stage.
    pub vertex_storage: bool,
    /// Storage buffers are bound in the fragment stage.
    pub fragment_storage: bool,
    /// Compute shaders are dispatched.
    pub compute: bool,
    /// Bind groups, counting group 0.
    pub bind_groups: u32,
    /// Sampled textures per shader stage, counting the renderer's own groups.
    pub sampled_textures_per_stage: u32,
}

impl Requirements {
    /// What the full tier guarantees: storage buffers in both stages, eight per
    /// stage, compute, and two bind groups.
    pub fn full_tier() -> Self {
        Self {
            storage_buffers_per_stage: Tier::FULL_STORAGE_BUFFERS_PER_STAGE,
            vertex_storage: true,
            fragment_storage: true,
            compute: true,
            bind_groups: Tier::BIND_GROUPS,
            sampled_textures_per_stage: 0,
        }
    }

    /// What the compatibility tier guarantees: no storage buffers, no compute, three
    /// bind groups and 16 sampled textures per stage. A plugin returning this
    /// registers on the compatibility tier as well as the full one, and must then
    /// draw without storage buffers or compute where the device has neither.
    pub fn compatibility_tier() -> Self {
        Self {
            storage_buffers_per_stage: 0,
            vertex_storage: false,
            fragment_storage: false,
            compute: false,
            bind_groups: Tier::COMPATIBILITY_BIND_GROUPS,
            sampled_textures_per_stage: Tier::COMPATIBILITY_SAMPLED_TEXTURES_PER_STAGE,
        }
    }

    /// Nothing beyond what every tier has: for a plugin that binds no storage
    /// buffers and dispatches no compute.
    pub fn none() -> Self {
        Self {
            storage_buffers_per_stage: 0,
            vertex_storage: false,
            fragment_storage: false,
            compute: false,
            bind_groups: Tier::BIND_GROUPS,
            sampled_textures_per_stage: 0,
        }
    }

    /// Set the storage-buffer count per stage.
    pub fn with_storage_buffers_per_stage(mut self, count: u32) -> Self {
        self.storage_buffers_per_stage = count;
        self
    }

    /// Set the bind-group count.
    pub fn with_bind_groups(mut self, count: u32) -> Self {
        self.bind_groups = count;
        self
    }

    /// Set the sampled-texture count per stage.
    pub fn with_sampled_textures_per_stage(mut self, count: u32) -> Self {
        self.sampled_textures_per_stage = count;
        self
    }
}

impl Default for Requirements {
    fn default() -> Self {
        Self::full_tier()
    }
}

#[cfg(test)]
mod tests {
    /// A capped device lands on the tier it was capped at, and a cap above
    /// the device changes nothing.
    #[test]
    fn capping_lands_on_the_tier_asked_for() {
        let full = Capabilities::from_parts(&gpu::Limits::default(), gpu::Features::empty(), None);
        assert_eq!(full.tier(), Tier::Full);
        assert_eq!(
            full.clone().capped_at(Tier::Compatibility).tier(),
            Tier::Compatibility
        );
        assert_eq!(
            full.clone().capped_at(Tier::OverlayOnly).tier(),
            Tier::OverlayOnly
        );
        assert_eq!(full.clone().capped_at(Tier::Full), full);

        let webgl2 = Capabilities::from_parts(
            &gpu::Limits::downlevel_webgl2_defaults(),
            gpu::Features::empty(),
            None,
        );
        assert_eq!(webgl2.tier(), Tier::Compatibility);
        assert_eq!(webgl2.clone().capped_at(Tier::Compatibility), webgl2);
        assert_eq!(
            webgl2.capped_at(Tier::OverlayOnly).tier(),
            Tier::OverlayOnly
        );
    }

    use super::*;

    /// No request is 32F with the feature and 24-bit without it; an explicit
    /// 24-bit request is kept; 32F is honoured with the feature and falls back
    /// without it; a format without stencil is refused.
    #[test]
    fn scene_depth_format_selection() {
        let without =
            Capabilities::from_parts(&gpu::Limits::default(), gpu::Features::empty(), None);
        let with = Capabilities::from_parts(
            &gpu::Limits::default(),
            gpu::Features::DEPTH32FLOAT_STENCIL8,
            None,
        );
        let f24 = gpu::TextureFormat::Depth24PlusStencil8;
        let f32s8 = gpu::TextureFormat::Depth32FloatStencil8;
        assert_eq!(without.scene_depth_format, f24);
        assert!(with.depth32float_stencil8 && !without.depth32float_stencil8);
        assert_eq!(
            without.select_scene_depth_format(None).unwrap(),
            (f24, false)
        );
        assert_eq!(with.select_scene_depth_format(None).unwrap(), (f24, false));
        for caps in [&without, &with] {
            assert_eq!(
                caps.select_scene_depth_format(Some(f24)).unwrap(),
                (f24, false)
            );
            assert!(matches!(
                caps.select_scene_depth_format(Some(gpu::TextureFormat::Depth32Float)),
                Err(crate::ViewportError::InvalidSceneDepthFormat { .. })
            ));
        }
        assert_eq!(
            with.select_scene_depth_format(Some(f32s8)).unwrap(),
            (f32s8, false)
        );
        assert_eq!(
            without.select_scene_depth_format(Some(f32s8)).unwrap(),
            (f24, true)
        );
    }

    fn tier_of(limits: gpu::Limits, flags: Option<gpu::DownlevelFlags>) -> Tier {
        Capabilities::from_parts(&limits, gpu::Features::empty(), flags).tier()
    }

    #[test]
    fn default_limits_are_full_tier() {
        assert_eq!(tier_of(gpu::Limits::default(), None), Tier::Full);
        assert_eq!(
            tier_of(gpu::Limits::default(), Some(gpu::DownlevelFlags::all())),
            Tier::Full
        );
    }

    #[test]
    fn iced_two_bind_groups_stay_full_tier() {
        let limits = gpu::Limits {
            max_bind_groups: 2,
            ..gpu::Limits::default()
        };
        assert_eq!(tier_of(limits, None), Tier::Full);
    }

    #[test]
    fn webgl2_limits_are_forward() {
        let limits = gpu::Limits::downlevel_webgl2_defaults();
        assert_eq!(tier_of(limits.clone(), None), Tier::Compatibility);
        // The adapter's compute flag survives a device created with WebGL2
        // limits; the zero workgroup limits are what rule compute out.
        let caps = Capabilities::from_parts(
            &limits,
            gpu::Features::empty(),
            Some(gpu::DownlevelFlags::all()),
        );
        assert!(!caps.compute);
        assert_eq!(caps.tier(), Tier::Compatibility);
    }

    #[test]
    fn two_bind_groups_below_full_are_overlay_only() {
        let limits = gpu::Limits {
            max_bind_groups: 2,
            ..gpu::Limits::downlevel_webgl2_defaults()
        };
        let caps = Capabilities::from_parts(&limits, gpu::Features::empty(), None);
        assert_eq!(caps.tier(), Tier::OverlayOnly);
        assert_eq!(
            caps.missing_for(Tier::Compatibility),
            vec!["3 bind groups (the device has 2)".to_owned()]
        );
    }

    #[test]
    fn fragment_only_storage_is_not_full_tier() {
        // A GLES 3.1 device with eight fragment storage blocks and none in the
        // vertex stage reports a count of eight.
        let flags = gpu::DownlevelFlags::all() - gpu::DownlevelFlags::VERTEX_STORAGE;
        let caps =
            Capabilities::from_parts(&gpu::Limits::default(), gpu::Features::empty(), Some(flags));
        assert!(!caps.vertex_storage);
        assert!(caps.fragment_storage);
        assert_eq!(caps.tier(), Tier::Compatibility);
        assert_eq!(
            caps.missing_for(Tier::Full),
            vec!["storage buffers in the vertex stage".to_owned()]
        );
    }

    #[test]
    fn one_bind_group_is_unsupported() {
        let limits = gpu::Limits {
            max_bind_groups: 1,
            ..gpu::Limits::downlevel_webgl2_defaults()
        };
        assert_eq!(tier_of(limits, None), Tier::Unsupported);
    }

    #[test]
    fn requirements_report_each_shortfall() {
        let caps = Capabilities::from_parts(
            &gpu::Limits::default(),
            gpu::Features::empty(),
            Some(gpu::DownlevelFlags::all()),
        );
        assert!(
            caps.missing_requirements(&Requirements::full_tier())
                .is_empty()
        );
        let wants = Requirements::full_tier()
            .with_storage_buffers_per_stage(10)
            .with_bind_groups(5);
        assert_eq!(caps.missing_requirements(&wants).len(), 2);
    }

    #[test]
    fn tiers_order_by_capability() {
        assert!(Tier::Unsupported < Tier::OverlayOnly);
        assert!(Tier::OverlayOnly < Tier::Compatibility);
        assert!(Tier::Compatibility < Tier::Full);
    }
}
