use super::*;

impl ViewportGpuResources {
    /// Re-upload the gizmo mesh with updated hover highlight colors.
    ///
    /// Called each frame when the hovered axis changes to brighten the appropriate axis color.
    /// The gizmo mesh is small (~300 vertices), so re-uploading every frame is acceptable.
    pub fn update_gizmo_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mode: crate::interaction::gizmo::GizmoMode,
        hovered: crate::interaction::gizmo::GizmoAxis,
        space_orientation: glam::Quat,
    ) {
        let (verts, indices) =
            crate::interaction::gizmo::build_gizmo_mesh(mode, hovered, space_orientation);

        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        // Recreate buffers if the new mesh is larger than the current allocation.
        if vert_bytes.len() as u64 > self.gizmo_vertex_buffer.size() {
            self.gizmo_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gizmo_vertex_buf"),
                size: vert_bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if idx_bytes.len() as u64 > self.gizmo_index_buffer.size() {
            self.gizmo_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gizmo_index_buf"),
                size: idx_bytes.len() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(&self.gizmo_vertex_buffer, 0, vert_bytes);
        queue.write_buffer(&self.gizmo_index_buffer, 0, idx_bytes);
        self.gizmo_index_count = indices.len() as u32;
    }

    /// Update the gizmo model matrix uniform (translation to gizmo center + scale for screen size).
    pub fn update_gizmo_uniform(&self, queue: &wgpu::Queue, model: glam::Mat4) {
        let uniform = crate::interaction::gizmo::GizmoUniform {
            model: model.to_cols_array_2d(),
        };
        queue.write_buffer(&self.gizmo_uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Create a quad mesh (2 triangles, 4 vertices) for an overlay on a face.
    ///
    /// Create an overlay quad from pre-computed corner positions and color.
    ///
    /// Corners should be in CCW winding order when viewed from outside.
    /// Returns (vertex_buffer, index_buffer, uniform_buffer, bind_group).
    pub fn create_overlay_quad(
        &self,
        device: &wgpu::Device,
        corners: &[[f32; 3]; 4],
        color: [f32; 4],
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
        use bytemuck::cast_slice;
        use wgpu;

        let quad_verts = corners;

        // 2 triangles: [0,1,2] and [0,2,3]
        let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        let vertices: Vec<OverlayVertex> = quad_verts
            .iter()
            .map(|p| OverlayVertex { position: *p })
            .collect();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc_quad_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc_quad_ibuf"),
            size: (std::mem::size_of::<u32>() * quad_indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&quad_indices));
        index_buffer.unmap();

        // Uniform buffer: identity model matrix + given color.
        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bc_quad_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[uniform_data]));
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bc_quad_bg"),
            layout: &self.overlay_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        (vertex_buffer, index_buffer, uniform_buffer, bind_group)
    }

    /// Create a line-list overlay for an active transform constraint.
    pub fn create_constraint_overlay(
        &self,
        device: &wgpu::Device,
        overlay: &crate::interaction::snap::ConstraintOverlay,
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    ) {
        use bytemuck::cast_slice;

        let (vertices, color): (Vec<OverlayVertex>, [f32; 4]) = match overlay {
            crate::interaction::snap::ConstraintOverlay::AxisLine {
                origin,
                direction,
                color,
            } => (
                vec![
                    OverlayVertex {
                        position: (*origin - *direction).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin + *direction).to_array(),
                    },
                ],
                *color,
            ),
            crate::interaction::snap::ConstraintOverlay::Plane {
                origin,
                axis_a,
                axis_b,
                color,
            } => (
                vec![
                    OverlayVertex {
                        position: (*origin - *axis_a).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin + *axis_a).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin - *axis_b).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin + *axis_b).to_array(),
                    },
                ],
                *color,
            ),
        };
        let indices: Vec<u32> = (0..vertices.len() as u32).collect();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("constraint_overlay_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("constraint_overlay_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&indices));
        index_buffer.unmap();

        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("constraint_overlay_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[uniform_data]));
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constraint_overlay_bg"),
            layout: &self.overlay_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        (
            vertex_buffer,
            index_buffer,
            indices.len() as u32,
            uniform_buffer,
            bind_group,
        )
    }

    /// Upload cap geometry (cross-section fill) as transient overlay buffers.
    ///
    /// Uses the overlay pipeline (position-only vertices + flat color uniform).
    pub(crate) fn upload_cap_geometry(
        &self,
        device: &wgpu::Device,
        cap: &crate::geometry::cap_geometry::CapMesh,
        color: [f32; 4],
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    ) {
        use bytemuck::cast_slice;

        let vertices: Vec<OverlayVertex> = cap
            .positions
            .iter()
            .map(|p| OverlayVertex { position: *p })
            .collect();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cap_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cap_ibuf"),
            size: (std::mem::size_of::<u32>() * cap.indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&cap.indices));
        index_buffer.unmap();

        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cap_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        uniform_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(&[uniform_data]));
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cap_bg"),
            layout: &self.overlay_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let idx_count = cap.indices.len() as u32;
        (
            vertex_buffer,
            index_buffer,
            idx_count,
            uniform_buffer,
            bind_group,
        )
    }
}
