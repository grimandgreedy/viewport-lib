//! GPU pick-pipeline construction for `DeviceResources`.
//!
//! Each `ensure_*_pick_pipeline` lazily builds the render pipeline and bind
//! group layouts for one pickable content kind (surfaces, glyphs, sprites,
//! polylines, volumes, and so on) the first time GPU picking touches it, so an
//! app that never picks a given kind pays nothing for it. All of these methods
//! are continuations of the `DeviceResources` impl that lives in
//! `device_resources.rs`.

use crate::resources::types::*;

impl DeviceResources {
    /// Lazily create the GPU pick pipeline and associated bind group layouts.
    ///
    /// No-op if already created. Called from `ViewportRenderer::pick_scene_gpu`
    /// on first invocation : zero overhead when GPU picking is never used.
    pub(crate) fn ensure_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // --- group 0: pick camera bind group layout ---
        // Includes binding 0 (CameraUniform) and binding 6 (ClipVolumesUniform).
        // The full camera_bind_group_layout has many more bindings; a separate
        // minimal layout is cleaner and avoids binding unused resources.
        let pick_camera_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("pick_camera_bgl"),
                entries: &[
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::gpu::ShaderStages::VERTEX,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    crate::gpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: crate::gpu::ShaderStages::FRAGMENT,
                        ty: crate::gpu::BindingType::Buffer {
                            ty: crate::gpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // --- group 1: PickInstance storage buffer ---
        // Visible to both stages: the object-id pipeline reads it in the vertex
        // stage, and the per-pixel VERTEX / NODE variants also read the model
        // matrix in the fragment stage to place the hit primitive's corners.
        let pick_instance_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("pick_instance_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // The default fragment writes a constant `0u` into the primitive-id
        // channel. When the device supports SHADER_PRIMITIVE_INDEX, rewrite it to
        // read `@builtin(primitive_index)` and write the hit triangle index, which
        // sub-object readback maps to a face / cell / tube segment. The builtin
        // requires the feature, so it can only appear in the module on a device
        // that has it; otherwise shader-module validation would reject it.
        let base_src = crate::resources::builders::wgsl_source!("pick_id");
        let shader = if device
            .features()
            .contains(crate::gpu::PRIMITIVE_INDEX_FEATURE)
        {
            let src = base_src
                .replace(
                    "fn fs_main(in: VertexOut) -> FragOut {",
                    "fn fs_main(in: VertexOut, @builtin(primitive_index) prim_index: u32) -> FragOut {",
                )
                .replace("out.primitive_id = 0u;", "out.primitive_id = prim_index;");
            crate::resources::builders::wgsl_module(
                device,
                "pick_id_shader",
                crate::resources::builders::with_primitive_index_enable(&src),
            )
        } else {
            crate::resources::builders::wgsl_module(device, "pick_id_shader", base_src)
        };

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "pick_pipeline_layout",
            &[&pick_camera_bgl, &pick_instance_bgl],
        );

        // Vertex layout: reuse the 64-byte Vertex stride but only declare position (location 0).
        let pick_vertex_layout = crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as crate::gpu::BufferAddress, // 64 bytes
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &[crate::gpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: crate::gpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "pick_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[pick_vertex_layout],
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        // location 0: R32Uint object ID
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None, // replace : no blending for integer targets
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        // location 1: R32Uint primitive ID (sub-object; written as 0 for now)
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        // location 2: R32Float depth
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Float,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None, // No culling: 3D meshes are often rendered two-sided; pick both faces.
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    true,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1, // pick pass is always 1x (no MSAA)
                    ..Default::default()
                },
                cache: None,
            },
        );

        self.pick.camera_bgl = Some(pick_camera_bgl);
        self.pick.bind_group_layout_1 = Some(pick_instance_bgl);
        self.pick.pipeline = Some(pipeline);
    }

    /// Build the surface VERTEX pick pipeline (writes the nearest corner's global
    /// vertex index into the primitive channel). No-op without
    /// SHADER_PRIMITIVE_INDEX or if already built. Reuses the group 0 / group 1
    /// layouts from [`ensure_pick_pipeline`], and adds group 2 for the hit mesh's
    /// vertex + index storage buffers.
    pub(crate) fn ensure_pick_vertex_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.vertex_pipeline.is_some() {
            return;
        }
        if !device
            .features()
            .contains(crate::gpu::PRIMITIVE_INDEX_FEATURE)
        {
            return;
        }
        self.ensure_pick_pipeline(device);
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let storage_entry = |binding: u32| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: crate::gpu::ShaderStages::FRAGMENT,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mesh_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("pick_vertex_mesh_bgl"),
            entries: &[storage_entry(0), storage_entry(1)],
        });

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "pick_vertex_pipeline_layout",
            &[
                self.pick.camera_bgl.as_ref().expect("pick camera bgl"),
                self.pick
                    .bind_group_layout_1
                    .as_ref()
                    .expect("pick instance bgl"),
                &mesh_bgl,
            ],
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "pick_vertex_shader",
            crate::resources::builders::with_primitive_index_enable(
                crate::resources::builders::wgsl_source!("pick_vertex"),
            ),
        );

        let pick_vertex_layout = crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &[crate::gpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: crate::gpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "pick_vertex_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[pick_vertex_layout],
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Float,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    true,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        self.pick.vertex_mesh_bgl = Some(mesh_bgl);
        self.pick.vertex_pipeline = Some(pipeline);
    }

    /// Build the surface EDGE pick pipeline (writes the nearest edge id
    /// `primitive_index * 3 + local_edge` into the primitive channel). No-op without
    /// SHADER_PRIMITIVE_INDEX or if already built. Reuses `vertex_mesh_bgl` (mesh
    /// vertex + index storage) for group 2, built by `ensure_pick_vertex_pipeline`.
    pub(crate) fn ensure_pick_edge_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.edge_pipeline.is_some() {
            return;
        }
        if !device
            .features()
            .contains(crate::gpu::PRIMITIVE_INDEX_FEATURE)
        {
            return;
        }
        // Reuse the vertex variant's group-0/1 layouts and the mesh storage layout.
        self.ensure_pick_vertex_pipeline(device);
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "pick_edge_pipeline_layout",
            &[
                self.pick.camera_bgl.as_ref().expect("pick camera bgl"),
                self.pick
                    .bind_group_layout_1
                    .as_ref()
                    .expect("pick instance bgl"),
                self.pick.vertex_mesh_bgl.as_ref().expect("pick vertex bgl"),
            ],
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "pick_edge_shader",
            crate::resources::builders::with_primitive_index_enable(
                crate::resources::builders::wgsl_source!("pick_edge"),
            ),
        );

        let pick_vertex_layout = crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &[crate::gpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: crate::gpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "pick_edge_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[pick_vertex_layout],
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Float,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    true,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        self.pick.edge_pipeline = Some(pipeline);
    }

    /// Lazily create the sprite pick pipeline. Reuses the sprite render vertex
    /// expansion (same position vertex buffer + sprite bind group) with a
    /// fragment that writes the item's object id. Group 0 is the full camera
    /// bind group (the sprite billboard expansion needs the viewport size that
    /// lives there); group 2 carries the per-draw pick id.
    pub(crate) fn ensure_sprite_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.sprite.pick_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        self.ensure_sprite_pipelines(device);

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("sprite_pick_id_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let sprite_bgl = self
            .sprite
            .bgl
            .as_ref()
            .expect("ensure_sprite_pipelines must build the sprite bind group layout");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_pick_shader",
            crate::resources::builders::wgsl_source!("sprite_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_pick_pipeline_layout",
            &[&self.binds.camera_bgl, sprite_bgl, &pick_id_bgl],
        );

        // Position vertex buffer: one vec3 per sprite, instance-stepped, exactly
        // as the sprite render pipeline binds it.
        let vert_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }];
        let vertex_buffers = [crate::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &vert_attrs,
        }];

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "sprite_pick_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &vertex_buffers,
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Float,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    true,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        self.sprite.pick_id_bgl = Some(pick_id_bgl);
        self.sprite.pick_pipeline = Some(pipeline);
    }
}
