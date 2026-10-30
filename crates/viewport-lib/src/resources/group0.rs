//! Group 0, the camera, lights, shadow and clip bindings every lit pipeline
//! and item type binds, declared once for every tier.
//!
//! A [`Group0Profile`] says which form each binding takes on a device and
//! which bindings exist at all. The layout entries, the bind-group entries and
//! the shader adaptation (`builders::adapt_wgsl`) all read it, so a binding is
//! declared here and nowhere else, and a new device class is a new profile
//! rather than a new set of branches.

use crate::gpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};
use crate::resources::builders::{sampler_entry, texture_entry, uniform_entry};

/// Which form each group 0 binding takes on a device, and which exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Group0Profile {
    /// The light list at binding 13 as a uniform array of this many lights,
    /// or `None` for a storage array.
    pub(crate) uniform_light_list: Option<usize>,
    /// Clustered lighting: the grid uniform (14) and the cells and index list
    /// (15), built by compute. Without them the lighting loops walk every
    /// light.
    pub(crate) clusters: bool,
    /// The mesh family's scene data: indirect light (18), the material block
    /// (21) and the deform header (23).
    pub(crate) mesh_family: bool,
    /// The shadow atlas and punctual shadow array hold real shadow maps, rather
    /// than placeholders the shaders never sample.
    pub(crate) shadows: bool,
    /// The IBL textures can hold an environment, rather than placeholders.
    pub(crate) ibl: bool,
}

impl Group0Profile {
    /// The profile for `caps`, or `None` on the overlay-only tier, which has
    /// no lit group 0. This is the one place a tier maps to a group 0.
    pub(crate) fn for_caps(caps: &crate::capabilities::Capabilities) -> Option<Self> {
        match caps.tier() {
            crate::capabilities::Tier::Full => Some(Self {
                uniform_light_list: None,
                clusters: true,
                mesh_family: true,
                shadows: true,
                ibl: true,
            }),
            crate::capabilities::Tier::Compatibility => Some(Self {
                uniform_light_list: Some(crate::resources::COMPATIBILITY_MAX_LIGHTS),
                clusters: false,
                mesh_family: false,
                shadows: false,
                ibl: false,
            }),
            _ => None,
        }
    }

    /// The group 0 layout entries for this profile.
    pub(crate) fn layout_entries(&self) -> Vec<BindGroupLayoutEntry> {
        let fragment = ShaderStages::FRAGMENT;
        let both = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let storage = |binding| BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let float_array = |binding| BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: BindingType::Texture {
                sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                view_dimension: crate::gpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let depth = |binding, view_dimension| BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: BindingType::Texture {
                sample_type: crate::gpu::TextureSampleType::Depth,
                view_dimension,
                multisampled: false,
            },
            count: None,
        };

        let mut entries = vec![
            // Camera: view-projection, eye, lit clamp.
            uniform_entry(0, both),
            // Shadow atlas and its comparison sampler.
            depth(1, crate::gpu::TextureViewDimension::D2),
            BindGroupLayoutEntry {
                binding: 2,
                visibility: fragment,
                ty: BindingType::Sampler(crate::gpu::SamplerBindingType::Comparison),
                count: None,
            },
            // Lights header. Lights are read only in the fragment stage, so it
            // takes none of Metal's per-stage vertex buffer slots (the
            // per-object mesh pipeline binds many vertex-stage storage buffers
            // and is slot-sensitive).
            uniform_entry(3, fragment),
            // Clip planes (section-view clipping).
            uniform_entry(4, both),
            // Shadow atlas uniform: cascade matrices, splits, PCSS parameters.
            uniform_entry(5, fragment),
            // Clip volumes. Tested only in the fragment stage (discard), so
            // fragment-only frees a Metal vertex buffer slot.
            uniform_entry(6, fragment),
            // IBL irradiance and prefiltered specular, one layer per
            // environment; the BRDF LUT and its sampler; the skybox.
            float_array(7),
            float_array(8),
            texture_entry(9, fragment),
            sampler_entry(10, fragment),
            texture_entry(11, fragment),
        ];
        // The light list, indexed against the header's count.
        entries.push(match self.uniform_light_list {
            None => storage(13),
            Some(_) => uniform_entry(13, fragment),
        });
        if self.clusters {
            // Grid uniform (dimensions, near/far, screen size, fallback flag).
            // Fragment only: the cluster helpers are fragment-stage, and the
            // vertex-stage Metal buffer table runs out around fifteen buffers.
            entries.push(uniform_entry(14, fragment));
            // The cluster cells then the global light index list, one buffer
            // so the pair costs one fragment storage binding. 16 is free.
            entries.push(storage(15));
        }
        // Point and spot light shadows: a 2D array, a point light's cube
        // faces picked in the shader.
        entries.push(depth(17, crate::gpu::TextureViewDimension::D2Array));
        if self.mesh_family {
            // Indirect light: per-object light-probe SH, environment zones,
            // the adaptive probe volume and per-instance custom data, regions
            // of one buffer. 19, 20 and 22 are free.
            entries.push(storage(18));
            // Per-material blocks indexed by the draw's material id.
            entries.push(storage(21));
            // The deform header, read by the deform hooks in the vertex stage.
            entries.push(uniform_entry(23, ShaderStages::VERTEX));
        }
        entries
    }
}

/// The shadow pass's group 0: the cascade light matrix behind a dynamic offset
/// and the deform header. The same on every tier, so a plugin's shadow
/// pipeline builds against it wherever it registers.
pub(crate) fn shadow_layout_entries() -> [BindGroupLayoutEntry; 2] {
    [
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                // The cascade loop selects its matrix slot by offset rather
                // than writing the buffer inside the pass (wgpu batches every
                // write before the pass runs).
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        },
        uniform_entry(23, ShaderStages::VERTEX),
    ]
}

/// The group 0 buffers each viewport owns.
#[derive(Clone, Copy)]
pub(crate) struct PerViewport<'a> {
    pub(crate) camera: &'a crate::gpu::Buffer,
    pub(crate) clip_planes: &'a crate::gpu::Buffer,
    pub(crate) shadow_info: &'a crate::gpu::Buffer,
    pub(crate) clip_volume: &'a crate::gpu::Buffer,
}

/// What a group 0 bind group binds. The fields a profile leaves out are
/// `None` on that tier and ignored.
pub(crate) struct Group0Resources<'a> {
    pub(crate) per_viewport: PerViewport<'a>,
    pub(crate) shadow_atlas: &'a crate::gpu::TextureView,
    pub(crate) shadow_sampler: &'a crate::gpu::Sampler,
    pub(crate) lights_header: &'a crate::gpu::Buffer,
    pub(crate) ibl_irradiance: &'a crate::gpu::TextureView,
    pub(crate) ibl_specular: &'a crate::gpu::TextureView,
    pub(crate) brdf_lut: &'a crate::gpu::TextureView,
    pub(crate) ibl_sampler: &'a crate::gpu::Sampler,
    pub(crate) skybox: &'a crate::gpu::TextureView,
    pub(crate) light_list: &'a crate::gpu::Buffer,
    pub(crate) point_shadows: &'a crate::gpu::TextureView,
    /// Bindings 14 and 15, when the profile has clusters.
    pub(crate) clusters: Option<(&'a crate::gpu::Buffer, &'a crate::gpu::Buffer)>,
    /// Bindings 18, 21 and 23, when the profile has the mesh family.
    pub(crate) mesh_family: Option<(
        &'a crate::gpu::Buffer,
        &'a crate::gpu::Buffer,
        &'a crate::gpu::Buffer,
    )>,
}

impl Group0Profile {
    /// A group 0 bind group over `res`, against a layout built from
    /// [`layout_entries`](Self::layout_entries).
    pub(crate) fn bind_group(
        &self,
        device: &crate::gpu::Device,
        layout: &crate::gpu::BindGroupLayout,
        res: &Group0Resources<'_>,
        label: &str,
    ) -> crate::gpu::BindGroup {
        use crate::gpu::{BindGroupEntry, BindingResource};
        fn buffer(binding: u32, buf: &crate::gpu::Buffer) -> BindGroupEntry<'_> {
            BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            }
        }
        fn view(binding: u32, view: &crate::gpu::TextureView) -> BindGroupEntry<'_> {
            BindGroupEntry {
                binding,
                resource: BindingResource::TextureView(view),
            }
        }
        fn sampler(binding: u32, sampler: &crate::gpu::Sampler) -> BindGroupEntry<'_> {
            BindGroupEntry {
                binding,
                resource: BindingResource::Sampler(sampler),
            }
        }
        let vp = &res.per_viewport;
        let mut entries = vec![
            buffer(0, vp.camera),
            view(1, res.shadow_atlas),
            sampler(2, res.shadow_sampler),
            buffer(3, res.lights_header),
            buffer(4, vp.clip_planes),
            buffer(5, vp.shadow_info),
            buffer(6, vp.clip_volume),
            view(7, res.ibl_irradiance),
            view(8, res.ibl_specular),
            view(9, res.brdf_lut),
            sampler(10, res.ibl_sampler),
            view(11, res.skybox),
            buffer(13, res.light_list),
        ];
        if self.clusters {
            let (grid, cells) = res
                .clusters
                .expect("a profile with clusters binds the cluster buffers");
            entries.push(buffer(14, grid));
            entries.push(buffer(15, cells));
        }
        entries.push(view(17, res.point_shadows));
        if self.mesh_family {
            let (indirect, materials, deform) = res
                .mesh_family
                .expect("a profile with the mesh family binds its buffers");
            entries.push(buffer(18, indirect));
            entries.push(buffer(21, materials));
            entries.push(buffer(23, deform));
        }
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::{Capabilities, Tier};
    use crate::gpu;

    fn profile(limits: gpu::Limits, flags: Option<gpu::DownlevelFlags>) -> Option<Group0Profile> {
        Group0Profile::for_caps(&Capabilities::from_parts(
            &limits,
            gpu::Features::empty(),
            flags,
        ))
    }

    fn bindings(profile: &Group0Profile) -> Vec<u32> {
        profile.layout_entries().iter().map(|e| e.binding).collect()
    }

    /// Each device class gets the group 0 the tier table describes.
    #[test]
    fn each_device_class_gets_its_group0() {
        let full = profile(gpu::Limits::default(), None).unwrap();
        assert_eq!(full.uniform_light_list, None);
        assert!(full.clusters && full.mesh_family && full.shadows && full.ibl);
        assert_eq!(
            bindings(&full),
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 21, 23
            ]
        );

        let no_compute = gpu::DownlevelFlags::all() - gpu::DownlevelFlags::COMPUTE_SHADERS;
        let fragment_only = gpu::DownlevelFlags::all() - gpu::DownlevelFlags::VERTEX_STORAGE;
        for compat in [
            profile(gpu::Limits::downlevel_webgl2_defaults(), None),
            profile(gpu::Limits::default(), Some(no_compute)),
            profile(gpu::Limits::default(), Some(fragment_only)),
        ] {
            let compat = compat.unwrap();
            assert_eq!(
                compat.uniform_light_list,
                Some(crate::resources::COMPATIBILITY_MAX_LIGHTS)
            );
            assert!(!compat.clusters && !compat.mesh_family && !compat.shadows && !compat.ibl);
            assert_eq!(
                bindings(&compat),
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17]
            );
        }

        let two_groups = gpu::Limits {
            max_bind_groups: 2,
            ..gpu::Limits::downlevel_webgl2_defaults()
        };
        assert_eq!(profile(two_groups, None), None);
    }

    /// The compatibility group 0 fits the compatibility tier's guarantees: no storage
    /// buffers, 6 fragment textures and 6 fragment uniform blocks.
    #[test]
    fn the_compatibility_group0_fits_the_compatibility_tier() {
        let compat = profile(gpu::Limits::downlevel_webgl2_defaults(), None).unwrap();
        let entries = compat.layout_entries();
        let count = |pred: &dyn Fn(&BindingType) -> bool| {
            entries
                .iter()
                .filter(|e| e.visibility.contains(ShaderStages::FRAGMENT) && pred(&e.ty))
                .count() as u32
        };
        assert_eq!(
            count(&|t| matches!(
                t,
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { .. },
                    ..
                }
            )),
            0
        );
        let textures = count(&|t| matches!(t, BindingType::Texture { .. }));
        let uniforms = count(&|t| {
            matches!(
                t,
                BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    ..
                }
            )
        });
        assert_eq!((textures, uniforms), (6, 6));
        assert!(textures <= Tier::COMPATIBILITY_SAMPLED_TEXTURES_PER_STAGE);
        assert!(uniforms <= Tier::COMPATIBILITY_UNIFORM_BUFFERS_PER_STAGE);
    }
}
