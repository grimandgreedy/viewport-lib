//! Vertex displacement sidecar.
//!
//! A per-mesh storage buffer of one `f32` per vertex called the sway mask.
//! Presence of the sidecar marks the mesh as displaceable, which selects a
//! pipeline variant whose vertex stage calls
//! `viewport_apply_vertex_displacement(world_pos, sway_mask)`. The function
//! body is supplied by a host plugin: viewport-lib declares the signature,
//! the plugin defines it.
//!
//! Static meshes pay nothing. The sidecar is allocated only when
//! [`crate::ViewportGpuResources::set_vertex_displacement_weights`] is called
//! for a `MeshId`.
//!
//! The host plugin also supplies the displacement uniform buffer that the
//! helper function reads (for example: wind globals). It installs the buffer
//! via [`crate::ViewportGpuResources::set_displacement_uniform_buffer`],
//! which binds it alongside the sway-mask storage in the displacement bind
//! group.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::resources::ViewportGpuResources;
use crate::resources::mesh_store::MeshId;

/// Per-mesh displacement data: a storage buffer of one `f32` per vertex plus
/// the bind group that pairs it with the host uniform.
pub(crate) struct MeshDisplacement {
    pub sway_mask_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub vertex_count: u32,
    /// Bind group for the displaceable variant. `None` until a host uniform
    /// has been installed.
    pub bind_group: Option<wgpu::BindGroup>,
}

/// Renderer-side displacement state.
///
/// Owns the bind group layout shared by every displaceable pipeline variant,
/// the per-mesh sway-mask buffers, and the host-installed uniform buffer
/// holding global displacement parameters.
pub(crate) struct DisplacementState {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub meshes: HashMap<MeshId, MeshDisplacement>,
    /// Uniform buffer the host installs to drive the helper function.
    pub host_uniform: Option<wgpu::Buffer>,
}

impl DisplacementState {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("displacement_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        Self {
            bind_group_layout,
            meshes: HashMap::new(),
            host_uniform: None,
        }
    }
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sway_mask_buffer: &wgpu::Buffer,
    host_uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("displacement_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sway_mask_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: host_uniform.as_entire_binding(),
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl ViewportGpuResources {
    /// Attach per-vertex sway-mask weights to an uploaded mesh.
    ///
    /// Creates a sidecar storage buffer of one `f32` per vertex. The mesh's
    /// vertex buffer is not modified. Calling this on a `mesh_id` marks the
    /// mesh as displaceable: subsequent draws are routed through the
    /// displacement pipeline variant once a host displacement-uniform buffer
    /// has been installed via [`Self::set_displacement_uniform_buffer`].
    ///
    /// Each entry in `weights` scales the displacement at the corresponding
    /// vertex. `0.0` is "no movement", `1.0` is "full strength". Values in
    /// between scale linearly. Mesh authoring decides what "full strength"
    /// means; the helper function reads the value as-is.
    ///
    /// Calling this again on the same `mesh_id` replaces the buffer.
    pub fn set_vertex_displacement_weights(
        &mut self,
        device: &wgpu::Device,
        mesh_id: MeshId,
        weights: &[f32],
    ) {
        let sway_mask_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("displacement_sway_mask_buffer"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = self.displacement.host_uniform.as_ref().map(|u| {
            build_bind_group(device, &self.displacement.bind_group_layout, &sway_mask_buffer, u)
        });
        self.displacement.meshes.insert(
            mesh_id,
            MeshDisplacement {
                sway_mask_buffer,
                vertex_count: weights.len() as u32,
                bind_group,
            },
        );
    }

    /// Install the displacement uniform buffer the helper function reads.
    ///
    /// A host plugin owns this buffer (for wind: the `WindGlobals` UBO) and
    /// installs it once at startup. The buffer can be written each frame
    /// without reinstalling. The same uniform feeds every displaceable mesh
    /// in the scene.
    ///
    /// Existing per-mesh bind groups are rebuilt against the new uniform.
    pub fn set_displacement_uniform_buffer(&mut self, device: &wgpu::Device, buffer: wgpu::Buffer) {
        self.displacement.host_uniform = Some(buffer);
        let uniform = self.displacement.host_uniform.as_ref().unwrap();
        for mesh in self.displacement.meshes.values_mut() {
            mesh.bind_group = Some(build_bind_group(
                device,
                &self.displacement.bind_group_layout,
                &mesh.sway_mask_buffer,
                uniform,
            ));
        }
    }

    /// Whether `mesh_id` has a sway-mask sidecar attached.
    pub fn is_displaceable_mesh(&self, mesh_id: MeshId) -> bool {
        self.displacement.meshes.contains_key(&mesh_id)
    }

    /// Whether a host has installed a displacement uniform buffer.
    pub fn has_displacement_uniform(&self) -> bool {
        self.displacement.host_uniform.is_some()
    }

    /// Bind group for one displaceable mesh, ready for a pipeline that
    /// consumes the displacement sidecar. Returns `None` until both the
    /// sway mask (via `set_vertex_displacement_weights`) and the host
    /// uniform (via `set_displacement_uniform_buffer`) are present.
    ///
    /// Plugin pipelines that ship a displaceable variant call this from
    /// their `paint` to look up the bind group to bind alongside the
    /// per-item state.
    pub fn displacement_bind_group(&self, mesh_id: MeshId) -> Option<&wgpu::BindGroup> {
        self.displacement
            .meshes
            .get(&mesh_id)
            .and_then(|m| m.bind_group.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use crate::ViewportGpuResources;
    use crate::geometry::primitives;

    fn try_make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn make_resources(device: &wgpu::Device) -> ViewportGpuResources {
        ViewportGpuResources::new(device, wgpu::TextureFormat::Rgba8UnormSrgb, 1)
    }

    fn make_dummy_uniform(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_displacement_uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn fresh_resources_have_no_displaceable_meshes() {
        let Some((device, _)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let resources = make_resources(&device);
        assert!(!resources.has_displacement_uniform());
    }

    #[test]
    fn set_weights_marks_mesh_displaceable() {
        let Some((device, _)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let weights = vec![1.0_f32; plane.positions.len()];

        resources.set_vertex_displacement_weights(&device, mesh_id, &weights);

        assert!(resources.is_displaceable_mesh(mesh_id));
    }

    #[test]
    fn bind_group_is_none_without_host_uniform() {
        let Some((device, _)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let weights = vec![1.0_f32; plane.positions.len()];

        resources.set_vertex_displacement_weights(&device, mesh_id, &weights);

        assert!(resources.displacement_bind_group(mesh_id).is_none());
    }

    #[test]
    fn installing_uniform_builds_bind_groups_for_existing_meshes() {
        let Some((device, _)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let weights = vec![1.0_f32; plane.positions.len()];

        resources.set_vertex_displacement_weights(&device, mesh_id, &weights);
        resources.set_displacement_uniform_buffer(&device, make_dummy_uniform(&device));

        assert!(resources.has_displacement_uniform());
        assert!(resources.displacement_bind_group(mesh_id).is_some());
    }

    #[test]
    fn uniform_first_then_weights_also_builds_bind_group() {
        let Some((device, _)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = make_resources(&device);
        let plane = primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();
        let weights = vec![1.0_f32; plane.positions.len()];

        resources.set_displacement_uniform_buffer(&device, make_dummy_uniform(&device));
        resources.set_vertex_displacement_weights(&device, mesh_id, &weights);

        assert!(resources.displacement_bind_group(mesh_id).is_some());
    }
}

