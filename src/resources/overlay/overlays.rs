use crate::resources::*;

impl DeviceResources {
    /// Re-upload the gizmo mesh with updated hover highlight colours.
    ///
    /// Called each frame when the hovered axis changes to brighten the appropriate axis colour.
    /// The gizmo mesh is small (~300 vertices), so re-uploading every frame is acceptable.
    pub fn update_gizmo_mesh(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        mode: crate::interaction::manipulation::gizmo::GizmoMode,
        hovered: crate::interaction::manipulation::gizmo::GizmoAxis,
        space_orientation: glam::Quat,
    ) {
        let (verts, indices) = crate::interaction::manipulation::gizmo::build_gizmo_mesh(
            mode,
            hovered,
            space_orientation,
        );

        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        // Recreate buffers if the new mesh is larger than the current allocation.
        if vert_bytes.len() as u64 > self.gizmo.vertex_buffer.size() {
            self.gizmo.vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("gizmo_vertex_buf"),
                size: vert_bytes.len() as u64,
                usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if idx_bytes.len() as u64 > self.gizmo.index_buffer.size() {
            self.gizmo.index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("gizmo_index_buf"),
                size: idx_bytes.len() as u64,
                usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(&self.gizmo.vertex_buffer, 0, vert_bytes);
        queue.write_buffer(&self.gizmo.index_buffer, 0, idx_bytes);
        self.gizmo.index_count = indices.len() as u32;
    }

    /// Update the gizmo model matrix uniform (translation to gizmo center + scale for screen size).
    pub fn update_gizmo_uniform(&self, queue: &crate::gpu::Queue, model: glam::Mat4) {
        let uniform = crate::interaction::manipulation::gizmo::GizmoUniform {
            model: model.to_cols_array_2d(),
        };
        queue.write_buffer(&self.gizmo.uniform_buf, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Create a line-list overlay for an active transform constraint.
    pub fn create_constraint_overlay(
        &self,
        device: &crate::gpu::Device,
        overlay: &crate::interaction::query::snap::ConstraintOverlay,
    ) -> (
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        u32,
        crate::gpu::Buffer,
        crate::gpu::BindGroup,
    ) {
        use bytemuck::cast_slice;

        let (vertices, colour): (Vec<OverlayVertex>, [f32; 4]) = match overlay {
            crate::interaction::query::snap::ConstraintOverlay::AxisLine {
                origin,
                direction,
                colour,
            } => (
                vec![
                    OverlayVertex {
                        position: (*origin - *direction).to_array(),
                    },
                    OverlayVertex {
                        position: (*origin + *direction).to_array(),
                    },
                ],
                *colour,
            ),
            crate::interaction::query::snap::ConstraintOverlay::Plane {
                origin,
                axis_a,
                axis_b,
                colour,
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
                *colour,
            ),
        };
        let indices: Vec<u32> = (0..vertices.len() as u32).collect();

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("constraint_overlay_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(vertex_buffer.slice(..), cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("constraint_overlay_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(index_buffer.slice(..), cast_slice(&indices));
        index_buffer.unmap();

        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            colour,
        };
        let uniform_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("constraint_overlay_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            uniform_buffer.slice(..),
            cast_slice(&[uniform_data]),
        );
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("constraint_overlay_bg"),
            layout: &self.guides.overlay_bgl,
            entries: &[crate::gpu::BindGroupEntry {
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

    /// Create a triangle-list fill overlay for a clip plane handle quad.
    ///
    /// Produces a semi-transparent filled quad at the plane's world position.
    pub(crate) fn create_clip_plane_fill_overlay(
        &self,
        device: &crate::gpu::Device,
        overlay: &crate::interaction::manipulation::clip_plane::ClipPlaneOverlay,
    ) -> (
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        u32,
        crate::gpu::Buffer,
        crate::gpu::BindGroup,
    ) {
        use crate::interaction::manipulation::clip_plane::plane_tangents;
        use bytemuck::cast_slice;

        let (t1, t2) = plane_tangents(overlay.normal);
        let e = overlay.extent;
        let c = overlay.center;

        // 4 corners of the quad.
        let corners = [
            c + e * t1 + e * t2,
            c - e * t1 + e * t2,
            c - e * t1 - e * t2,
            c + e * t1 - e * t2,
        ];

        let vertices: Vec<OverlayVertex> = corners
            .iter()
            .map(|p| OverlayVertex {
                position: p.to_array(),
            })
            .collect();
        // Two triangles: (0,1,2) and (0,2,3).
        let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("clip_plane_fill_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(vertex_buffer.slice(..), cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("clip_plane_fill_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(index_buffer.slice(..), cast_slice(&indices));
        index_buffer.unmap();

        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            colour: overlay.fill_colour,
        };
        let uniform_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("clip_plane_fill_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            uniform_buffer.slice(..),
            cast_slice(&[uniform_data]),
        );
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("clip_plane_fill_bg"),
            layout: &self.guides.overlay_bgl,
            entries: &[crate::gpu::BindGroupEntry {
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

    /// Create a line-list border + normal indicator overlay for a clip plane handle.
    ///
    /// Produces 4 border edges around the quad and a short line along the normal direction.
    pub(crate) fn create_clip_plane_line_overlay(
        &self,
        device: &crate::gpu::Device,
        overlay: &crate::interaction::manipulation::clip_plane::ClipPlaneOverlay,
    ) -> (
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        u32,
        crate::gpu::Buffer,
        crate::gpu::BindGroup,
    ) {
        use crate::interaction::manipulation::clip_plane::plane_tangents;
        use bytemuck::cast_slice;

        let (t1, t2) = plane_tangents(overlay.normal);
        let e = overlay.extent;
        let c = overlay.center;

        // 4 quad corners (shared between border edges).
        let c0 = c + e * t1 + e * t2;
        let c1 = c - e * t1 + e * t2;
        let c2 = c - e * t1 - e * t2;
        let c3 = c + e * t1 - e * t2;

        // Normal indicator: short line from center along the normal.
        let n_tip = c + overlay.normal * (e * 0.5);

        // LineList vertices: each pair is one segment.
        // 4 border edges + 1 normal indicator = 10 vertices.
        let vertices: Vec<OverlayVertex> = vec![
            // Edge 0->1
            OverlayVertex {
                position: c0.to_array(),
            },
            OverlayVertex {
                position: c1.to_array(),
            },
            // Edge 1->2
            OverlayVertex {
                position: c1.to_array(),
            },
            OverlayVertex {
                position: c2.to_array(),
            },
            // Edge 2->3
            OverlayVertex {
                position: c2.to_array(),
            },
            OverlayVertex {
                position: c3.to_array(),
            },
            // Edge 3->0
            OverlayVertex {
                position: c3.to_array(),
            },
            OverlayVertex {
                position: c0.to_array(),
            },
            // Normal indicator
            OverlayVertex {
                position: c.to_array(),
            },
            OverlayVertex {
                position: n_tip.to_array(),
            },
        ];
        let indices: Vec<u32> = (0..vertices.len() as u32).collect();

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("clip_plane_line_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(vertex_buffer.slice(..), cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("clip_plane_line_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(index_buffer.slice(..), cast_slice(&indices));
        index_buffer.unmap();

        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            colour: overlay.border_colour,
        };
        let uniform_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("clip_plane_line_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            uniform_buffer.slice(..),
            cast_slice(&[uniform_data]),
        );
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("clip_plane_line_bg"),
            layout: &self.guides.overlay_bgl,
            entries: &[crate::gpu::BindGroupEntry {
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
    /// Uses the overlay pipeline (position-only vertices + flat colour uniform).
    pub(crate) fn upload_cap_geometry(
        &self,
        device: &crate::gpu::Device,
        cap: &crate::geometry::cap_geometry::CapMesh,
        colour: [f32; 4],
    ) -> (
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        u32,
        crate::gpu::Buffer,
        crate::gpu::BindGroup,
    ) {
        use bytemuck::cast_slice;

        let vertices: Vec<OverlayVertex> = cap
            .positions
            .iter()
            .map(|p| OverlayVertex { position: *p })
            .collect();

        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cap_vbuf"),
            size: (std::mem::size_of::<OverlayVertex>() * vertices.len()) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(vertex_buffer.slice(..), cast_slice(&vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cap_ibuf"),
            size: (std::mem::size_of::<u32>() * cap.indices.len()) as u64,
            usage: crate::gpu::BufferUsages::INDEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(index_buffer.slice(..), cast_slice(&cap.indices));
        index_buffer.unmap();

        let uniform_data = OverlayUniform {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            colour,
        };
        let uniform_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("cap_ubuf"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        crate::resources::builders::write_mapped(
            uniform_buffer.slice(..),
            cast_slice(&[uniform_data]),
        );
        uniform_buffer.unmap();

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("cap_bg"),
            layout: &self.guides.overlay_bgl,
            entries: &[crate::gpu::BindGroupEntry {
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

/// Per-vertex data for overlay rendering: position only (no normal/colour in vertex).
///
/// Colour is provided via the OverlayUniform rather than per-vertex to keep
/// the buffer minimal : all vertices of a single overlay quad share the same colour.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayVertex {
    /// World-space XYZ position of this overlay vertex.
    pub position: [f32; 3],
}

impl OverlayVertex {
    /// wgpu vertex buffer layout matching shader location 0 (position vec3f).
    pub fn buffer_layout() -> crate::gpu::VertexBufferLayout<'static> {
        crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayVertex>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &[crate::gpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: crate::gpu::VertexFormat::Float32x3,
            }],
        }
    }
}

/// Per-overlay uniform: model matrix and RGBA colour with alpha for transparency.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayUniform {
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) colour: [f32; 4], // RGBA with alpha for transparency
}
/// Cached GPU textures for the backdrop blur (frosted glass) effect.
///
/// Stored on `ViewportRenderer` and recreated when the viewport size changes.
/// Contains a full-resolution intermediate (for rendering the scene when the
/// output surface lacks `TEXTURE_BINDING`), two half-resolution ping-pong
/// textures for the separable blur passes, and pre-built bind groups.
pub(crate) struct BackdropBlurState {
    /// Full-resolution intermediate render target. The scene is rendered here
    /// instead of directly to the surface so the result can be sampled. Kept
    /// alive so the matching view remains valid.
    #[allow(dead_code)]
    pub intermediate_texture: crate::gpu::Texture,
    pub intermediate_view: crate::gpu::TextureView,
    /// Half-resolution blur ping-pong texture A. Kept alive for its view.
    #[allow(dead_code)]
    pub blur_a_texture: crate::gpu::Texture,
    pub blur_a_view: crate::gpu::TextureView,
    /// Half-resolution blur ping-pong texture B. Kept alive for its view.
    #[allow(dead_code)]
    pub blur_b_texture: crate::gpu::Texture,
    pub blur_b_view: crate::gpu::TextureView,
    /// Viewport physical size the textures were created for.
    pub size: [u32; 2],
    /// Format the textures were created with.
    pub format: crate::gpu::TextureFormat,
}

/// Uniform buffer layout for the full-screen ground plane shader.
///
/// Matches `GroundPlaneUniform` in `ground_plane.wgsl` exactly (256 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GroundPlaneUniform {
    pub view_proj: [[f32; 4]; 4], // offset   0, 64 bytes
    pub cam_right: [f32; 4],      // offset  64, 16 bytes
    pub cam_up: [f32; 4],         // offset  80, 16 bytes
    pub cam_back: [f32; 4],       // offset  96, 16 bytes
    pub eye_pos: [f32; 3],        // offset 112, 12 bytes
    pub height: f32,              // offset 124,  4 bytes
    pub colour: [f32; 4],         // offset 128, 16 bytes
    pub shadow_colour: [f32; 4],  // offset 144, 16 bytes
    pub light_vp: [[f32; 4]; 4],  // offset 160, 64 bytes
    pub tan_half_fov: f32,        // offset 224,  4 bytes
    pub aspect: f32,              // offset 228,  4 bytes
    pub tile_size: f32,           // offset 232,  4 bytes
    pub shadow_bias: f32,         // offset 236,  4 bytes
    pub mode: u32,                // offset 240,  4 bytes
    pub shadow_opacity: f32,      // offset 244,  4 bytes
    pub _pad: [f32; 2],           // offset 248,  8 bytes
    pub colour2: [f32; 4],        // offset 256, 16 bytes : second tile colour
} // total  272 bytes

/// Uniform buffer layout for the full-screen analytical grid shader.
///
/// Contains all data needed by `grid.wgsl`: camera matrices for ray unprojection,
/// eye position, grid plane height, spacing for minor/major lines, and RGBA colours.
/// Total size: 192 bytes (fits in one 256-byte UBO slot).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GridUniform {
    /// Combined view-projection matrix for computing clip-space depth of grid hits.
    pub view_proj: [[f32; 4]; 4], // offset   0, 64 bytes
    /// Camera-to-world rotation matrix (3 columns as vec4 with w=0 padding, matching
    /// WGSL mat3x3<f32> layout). Col 0 = right, Col 1 = up, Col 2 = back (camera +Z).
    /// Used to rotate the analytical camera-space ray direction into world space,
    /// bypassing the ill-conditioned inv(view_proj) at large camera distances.
    pub cam_to_world: [[f32; 4]; 3], // offset  64, 48 bytes
    /// tan(fov_y / 2) : scales NDC x/y to camera-space ray direction.
    pub tan_half_fov: f32, // offset 112,  4 bytes
    /// Viewport aspect ratio (width / height).
    pub aspect: f32, // offset 116,  4 bytes
    /// Padding to keep snap_origin at offset 152 (8-byte aligned).
    pub _pad_ivp: [f32; 2], // offset 120,  8 bytes
    /// Eye (camera) position in world space.
    pub eye_pos: [f32; 3], // offset 128, 12 bytes
    /// Z-coordinate of the horizontal grid plane (Z-up, XY ground plane).
    pub grid_z: f32, // offset 140,  4 bytes
    /// Minor grid line spacing (world units).
    pub spacing_minor: f32, // offset 144,  4 bytes
    /// Major grid line spacing (world units, typically spacing_minor * 10).
    pub spacing_major: f32, // offset 148,  4 bytes
    /// XZ origin used to keep `hit.xz - snap_origin` small for f32 precision.
    /// Set to `floor(eye.xz / spacing_major) * spacing_major` each frame.
    pub snap_origin: [f32; 2], // offset 152,  8 bytes
    /// RGBA colour for minor grid lines.
    pub colour_minor: [f32; 4], // offset 160, 16 bytes
    /// RGBA colour for major grid lines.
    pub colour_major: [f32; 4], // offset 176, 16 bytes
                                // Total: 192 bytes
}
